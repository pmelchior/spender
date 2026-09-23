#!/usr/bin/env python

import argparse

import numpy as np
import torch

from spender import NeuralDensityEstimator, load_flow_model, load_model
from spender.data.desi import DESI
from spender.data.sdss import BOSS, SDSS
from spender.util import LossTracker

INSTRUMENTS = {"SDSS": SDSS, "BOSS": BOSS, "DESI": DESI}


def encode(model, loader, device):
    """Encode all spectra of a data loader into latents

    Parameters
    ----------
    model: :class:`spender.SpectrumAutoencoder`
        Trained spender model
    loader: :class:`torch.utils.data.DataLoader`
        Loader of the spectra, see the `get_data_loader` method of the instrument
    device: `torch.Device`
        Device to run the encoder on

    Returns
    -------
    s: `torch.tensor`, shape (N, n_latent)
        Latents of all spectra
    """
    model.eval()
    s = []
    with torch.no_grad():
        for spec, w, z in loader:
            s.append(model.encode(spec.to(device)))
    return torch.cat(s)


def train(nde, s, s_valid, n_epoch=100, batch_size=10000, lr=1e-2, outfile=None, tracker=None, verbose=False):
    """Train the normalizing flow on latents

    Parameters
    ----------
    nde: :class:`spender.NeuralDensityEstimator`
        Flow model
    s: `torch.tensor`, shape (N, n_latent)
        Latents for training
    s_valid: `torch.tensor`, shape (M, n_latent)
        Latents for validation
    n_epoch: int
        Number of epochs
    batch_size: int
        Number of latents in each batch
    lr: float
        Maximum learning rate
    outfile: string
        Path to save the flow model to
    tracker: :class:`spender.util.LossTracker`
        Tracker for the training and validation losses
    verbose: bool
        Whether to print the losses of every epoch

    Returns
    -------
    None
    """
    if outfile is None:
        outfile = "flow.pt"
    if tracker is None:
        tracker = LossTracker()

    optimizer = torch.optim.Adam(nde.parameters(), lr=lr)
    n_batch = max(1, int(np.ceil(len(s) / batch_size)))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, steps_per_epoch=n_batch, epochs=n_epoch)

    for epoch in range(n_epoch):
        nde.train()
        # batch composition changes in every epoch
        perm = torch.randperm(len(s), device=s.device)
        for k in range(n_batch):
            batch = s[perm[k * batch_size : (k + 1) * batch_size]]
            loss = -nde.log_prob(batch).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            tracker.update("train", {"log_prob": loss}, len(batch))

        with torch.no_grad():
            nde.eval()
            for k in range(0, len(s_valid), batch_size):
                batch = s_valid[k : k + batch_size]
                loss = -nde.log_prob(batch).mean()
                tracker.update("valid", {"log_prob": loss}, len(batch))

        tracker.end_epoch()
        if verbose:
            train_loss = tracker.history["train"]["log_prob"][-1]
            valid_loss = tracker.history["valid"]["log_prob"][-1]
            print(f"====> Epoch: {epoch} TRAINING Loss: {train_loss:.3f}  VALIDATION Loss: {valid_loss:.3f}")

        if epoch % 10 == 0 or epoch == n_epoch - 1:
            torch.save(nde.state_dict(), outfile)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="dataset directory or HuggingFace Hub repository")
    parser.add_argument("model", help="file name of the trained spender model")
    parser.add_argument("outfile", help="output file name of the flow model")
    parser.add_argument("-i", "--instrument", help="instrument that observed the spectra", choices=list(INSTRUMENTS), default="SDSS")
    parser.add_argument("-b", "--batch_size", help="batch size for the flow", type=int, default=10000)
    parser.add_argument("-B", "--encode_batch_size", help="batch size for the encoder", type=int, default=1024)
    parser.add_argument("-e", "--epochs", help="number of epochs", type=int, default=100)
    parser.add_argument("-r", "--rate", help="maximum learning rate", type=float, default=1e-2)
    parser.add_argument("-C", "--clobber", help="continue training of existing flow model", action="store_true")
    parser.add_argument("-v", "--verbose", help="verbose printing", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # encode all spectra once: the latents are small enough to stay in memory
    Instrument = INSTRUMENTS[args.instrument]
    instrument = Instrument()
    model = load_model(args.model, instrument, map_location=device)
    model.to(device)
    s, s_valid = tuple(
        encode(model, Instrument.get_data_loader(args.dir, which=which, batch_size=args.encode_batch_size), device)
        for which in ("train", "valid")
    )
    n_latent = s.shape[1]

    if args.verbose:
        print(f"Latents:\t{len(s)} (train), {len(s_valid)} (valid), {n_latent} dimensions")
        print(f"device:\t\t{device}")

    if args.clobber:
        nde = load_flow_model(args.outfile, n_latent, map_location=device)
    else:
        nde = NeuralDensityEstimator(
            dim=n_latent,
            initial_pos={"bounds": [[0, 0]] * n_latent, "std": [0.05] * n_latent},
        )
    nde.to(device)

    train(
        nde,
        s,
        s_valid,
        n_epoch=args.epochs,
        batch_size=args.batch_size,
        lr=args.rate,
        outfile=args.outfile,
        verbose=args.verbose,
    )
