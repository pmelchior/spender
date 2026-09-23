#!/usr/bin/env python

import argparse
import os

import numpy as np
import torch
from accelerate import Accelerator
from torch import nn, optim

from spender import SpectrumAutoencoder, SpeculatorActivation
from spender.data.sdss import SDSS
from spender.util import LossTracker


def load_model(filename, model, instrument):
    device = instrument.wave_obs.device
    model_struct = torch.load(filename, map_location=device)

    # backwards compat: encoder.mlp instead of encoder.mlp.mlp
    if 'encoder.mlp.mlp.0.weight' in model_struct['model'].keys():
        from collections import OrderedDict
        model_struct['model'] = OrderedDict([(k.replace('mlp.mlp', 'mlp'), v) for k, v in model_struct['model'].items()])

    # backwards compat: add instrument to encoder
    try:
        model.load_state_dict(model_struct['model'], strict=False)
    except RuntimeError:
        model_struct['model']['encoder.instrument.wave_obs']= instrument.wave_obs
        model_struct['model']['encoder.instrument.skyline_mask']= instrument.skyline_mask
        model.load_state_dict(model_struct['model'], strict=False)
    tracker = LossTracker()
    tracker.load_state_dict(model_struct['losses'])
    return model, tracker


def train(model, instrument, trainloader, validloader, n_epoch=200, n_batch=None, outfile=None, tracker=None, verbose=False, lr=3e-4):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=n_epoch)

    accelerator = Accelerator(mixed_precision='fp16')
    model, instrument, trainloader, validloader, optimizer = accelerator.prepare(model, instrument, trainloader, validloader, optimizer)

    if outfile is None:
        outfile = "checkpoint.pt"

    if tracker is None:
        tracker = LossTracker()
    epoch = tracker.epoch
    n_epoch += epoch
    if verbose and epoch > 0:
        train_loss = tracker.history["train"]["fidelity"][-1]
        valid_loss = tracker.history["valid"]["fidelity"][-1]
        print(f'====> Epoch: {epoch-1} TRAINING Loss: {train_loss:.3e}  VALIDATION Loss: {valid_loss:.3e}')
        if instrument.lsf is not None:
            print (f'LSF: {instrument.lsf.weight.data}')

    for epoch_ in range(epoch, n_epoch):
        model.train()
        for k, batch in enumerate(trainloader):
            batch_size = len(batch[0])
            spec, w, z = batch
            loss = model.loss(spec, w, instrument=instrument, z=z)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            tracker.update("train", {"fidelity": loss}, batch_size)

            # stop after n_batch
            if n_batch is not None and k == n_batch - 1:
                break

        with torch.no_grad():
            model.eval()
            for k, batch in enumerate(validloader):
                batch_size = len(batch[0])
                spec, w, z = batch
                loss = model.loss(spec, w, instrument=instrument, z=z)
                tracker.update("valid", {"fidelity": loss}, batch_size)
                # stop after n_batch
                if n_batch is not None and k == n_batch - 1:
                    break

        scheduler.step()
        tracker.end_epoch()
        train_loss = tracker.history["train"]["fidelity"][-1]
        valid_loss = tracker.history["valid"]["fidelity"][-1]

        if verbose:
            print(f'====> Epoch: {epoch_} TRAINING Loss: {train_loss:.3e}  VALIDATION Loss: {valid_loss:.3e}')
            if instrument.lsf is not None:
                print (f'LSF: {instrument.lsf.weight.data}')

        # checkpoints
        if epoch_ % 5 == 0 or epoch_ == n_epoch - 1:
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save({
                "model": unwrapped_model.state_dict(),
                "losses": tracker.state_dict(),
            }, outfile)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="dataset directory or HuggingFace Hub repository")
    parser.add_argument("outfile", help="output file name")
    parser.add_argument("-n", "--latents", help="latent dimensionality", type=int, default=2)
    parser.add_argument("-b", "--batch_size", help="batch size", type=int, default=1024)
    parser.add_argument("-l", "--batch_number", help="number of batches per epoch", type=int, default=None)
    parser.add_argument("-e", "--epochs", help="number of epochs", type=int, default=200)
    parser.add_argument("-r", "--rate", help="learning rate", type=float, default=1e-3)
    parser.add_argument("-S", "--superresolution", help="Superresolution factor", type=int, default=1)
    parser.add_argument("-L", "--lsf_size", help="LSF kernel size", type=int, default=0)
    parser.add_argument("-C", "--clobber", help="continue training of existing model", action="store_true")
    parser.add_argument("-v", "--verbose", help="verbose printing", action="store_true")
    args = parser.parse_args()

    # set LSF if requested
    if args.lsf_size > 0:
        lsf = torch.zeros(args.lsf_size)
        lsf[args.lsf_size // 2] = 1
    else:
        lsf = None

    # define SDSS instrument
    instrument = SDSS(lsf=lsf)

    # restframe wavelength for reconstructed spectra
    z_max = 0.5
    lmbda_min = instrument.wave_obs.min()/(1+z_max)
    lmbda_max = instrument.wave_obs.max()
    bins = args.superresolution * int(instrument.wave_obs.shape[0] * (1 + z_max))
    wave_rest = torch.linspace(lmbda_min, lmbda_max, bins, dtype=torch.float32)

    # data loaders
    trainloader = SDSS.get_data_loader(args.dir, which="train", batch_size=args.batch_size, shuffle=True)
    validloader = SDSS.get_data_loader(args.dir, which="valid", batch_size=args.batch_size)

    if args.verbose:
        print ("Observed frame:\t{:.0f} .. {:.0f} A ({} bins)".format(instrument.wave_obs.min(), instrument.wave_obs.max(), len(instrument.wave_obs)))
        print ("Restframe:\t{:.0f} .. {:.0f} A ({} bins)".format(lmbda_min, lmbda_max, bins))

    # define and train the model
    model = SpectrumAutoencoder(
            instrument,
            wave_rest,
            n_latent=args.latents,
            act=(SpeculatorActivation(64), SpeculatorActivation(256), SpeculatorActivation(1024), SpeculatorActivation(len(wave_rest), plus_one=True)),
    )

    # check if outfile already exists, continue only of -c is set
    if os.path.isfile(args.outfile) and not args.clobber:
        raise SystemExit("\nOutfile exists! Set option -C to continue training.")
    tracker = None
    if os.path.isfile(args.outfile):
        if args.verbose:
            print (f"\nLoading file {args.outfile}")
        model, tracker = load_model(args.outfile, model, instrument)

    train(model, instrument, trainloader, validloader, n_epoch=args.epochs, n_batch=args.batch_number, outfile=args.outfile, tracker=tracker, lr=args.rate, verbose=args.verbose)
