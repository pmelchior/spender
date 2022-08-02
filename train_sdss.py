#!/usr/bin/env python

import argparse
import numpy as np
import torch
from torch import nn
from torch import optim
from accelerate import Accelerator

from batch_wrapper import get_data_loader
from instrument import get_instrument
from model import SpectrumAutoencoder


def train(model, instrument, trainloader, validloader, n_epoch=200, mask_skyline=True, label="", verbose=False, lr=3e-4):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=n_epoch)

    accelerator = Accelerator(mixed_precision='fp16')
    model, instrument, trainloader, validloader, optimizer = accelerator.prepare(model, instrument, trainloader, validloader, optimizer)

    losses = []
    for epoch in range(n_epoch):
        model.train()
        train_loss = 0.
        for batch in trainloader:
            spec, w, z = batch
            if mask_skyline:
                w[:, instrument.skyline_mask] = 0

            loss = model.loss(spec, w, instrument=instrument, z=z)
            accelerator.backward(loss)
            train_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
        train_loss /= len(trainloader.dataset)

        with torch.no_grad():
            model.eval()
            valid_loss = 0.
            for batch in validloader:
                spec, w, z = batch
                if mask_skyline:
                    w[:, instrument.skyline_mask] = 0
                loss = model.loss(spec, w, instrument=instrument, z=z)
                valid_loss += loss.item()
            valid_loss /= len(validloader.dataset)

        scheduler.step()
        losses.append((train_loss, valid_loss))

        if epoch % 20 == 0 or epoch == n_epoch - 1:
            if verbose:
                print('====> Epoch: %i TRAINING Loss: %.2e VALIDATION Loss: %.2e' % (epoch, train_loss, valid_loss))

            # checkpoints
            torch.save(model, f'{label}.pt')
            np.save(f'{label}.losses.npy', np.array(losses))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="data file directory")
    parser.add_argument("label", help="output file labels")
    parser.add_argument("-n", "--latents", help="latent dimensionality", type=int, default=2)
    parser.add_argument("-b", "--batches", help="batch size", type=int, default=1024)
    parser.add_argument("-e", "--epochs", help="number of epochs", type=int, default=200)
    parser.add_argument("-r", "--rate", help="learning rate", type=float, default=1e-3)
    parser.add_argument("-v", "--verbose", help="verbose printing", action="store_true")
    args = parser.parse_args()

    # define SDSS instrument
    sdss = get_instrument("SDSS")

    # restframe wavelength for reconstructed spectra
    z_max = 0.2
    lmbda_min = sdss.wave_obs.min()/(1+z_max)
    lmbda_max = sdss.wave_obs.max()
    bins = int(sdss.wave_obs.shape[0] * (1 + z_max))
    wave_rest = torch.linspace(lmbda_min, lmbda_max, bins, dtype=torch.float32)

    # data loaders
    trainloader = get_data_loader(args.dir, sdss.name, which="train", batch_size=args.batches)
    validloader = get_data_loader(args.dir, sdss.name, which="valid", batch_size=args.batches)

    if args.verbose:
        print ("Observed frame:\t{:.0f} .. {:.0f} A ({} bins)".format(sdss.wave_obs.min(), sdss.wave_obs.max(), len(sdss.wave_obs)))
        print ("Restframe:\t{:.0f} .. {:.0f} A ({} bins)".format(lmbda_min, lmbda_max, bins))

    # define and train the model
    model = SpectrumAutoencoder(
            wave_rest,
            n_latent=args.latents,
            normalize=True,
    )
    train(model, sdss, trainloader, validloader, n_epoch=args.epochs, label=args.label, lr=args.rate, verbose=args.verbose)
