#!/usr/bin/env python

import argparse, os
import numpy as np
import torch
from torch import nn
from torch import optim
from accelerate import Accelerator
from spender import SpectrumAutoencoder
from spender.data.sdss import SDSS


def load_model(filename, model, instrument):
    device = instrument.wave_obs.device
    model_struct = torch.load(filename, map_location=device)
    # backwards compat: add instrument to encoder
    try:
        model.load_state_dict(model_struct['model'], strict=False)
    except RuntimeError:
        model_struct['model']['encoder.instrument.wave_obs']= instrument.wave_obs
        model_struct['model']['encoder.instrument.skyline_mask']= instrument.skyline_mask
        model.load_state_dict(model_struct['model'], strict=False)
    losses = model_struct['losses']
    return model, losses


def train(model, instrument, trainloader, validloader, n_epoch=200, n_batch=None, mask_skyline=True, outfile=None, losses=None, verbose=False, lr=3e-4):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=n_epoch)

    accelerator = Accelerator(mixed_precision='fp16')
    model, instrument, trainloader, validloader, optimizer = accelerator.prepare(model, instrument, trainloader, validloader, optimizer)

    if outfile is None:
        outfile = "checkpoint.pt"

    epoch = 0
    if losses is None:
        losses = []
    else:
        try:
            epoch = len(losses)
            n_epoch += epoch
            if verbose:
                train_loss, valid_loss = losses[-1]
                print(f'====> Epoch: {epoch-1} TRAINING Loss: {train_loss:.3e}  VALIDATION Loss: {valid_loss:.3e}')
                if instrument.lsf is not None:
                    print (f'LSF: {instrument.lsf.weight.data}')
        except: # OK if losses are empty
            pass

    for epoch_ in range(epoch, n_epoch):
        model.train()
        train_loss = 0.
        n_sample = 0
        for k, batch in enumerate(trainloader):
            batch_size = len(batch[0])
            spec, w, z = batch

            if mask_skyline:
                w[:, instrument.skyline_mask] = 0

            loss = model.loss(spec, w, instrument=instrument, z=z)
            accelerator.backward(loss)
            train_loss += loss.item()
            n_sample += batch_size
            optimizer.step()
            optimizer.zero_grad()

            # stop after n_batch
            if n_batch is not None and k == n_batch - 1:
                break
        train_loss /= n_sample

        with torch.no_grad():
            model.eval()
            valid_loss = 0.
            n_sample = 0
            for k, batch in enumerate(validloader):
                batch_size = len(batch[0])
                spec, w, z = batch
                if mask_skyline:
                    w[:, instrument.skyline_mask] = 0
                loss = model.loss(spec, w, instrument=instrument, z=z)
                valid_loss += loss.item()
                n_sample += batch_size
                # stop after n_batch
                if n_batch is not None and k == n_batch - 1:
                    break
            valid_loss /= n_sample

        scheduler.step()
        losses.append((train_loss, valid_loss))

        if verbose:
            print(f'====> Epoch: {epoch_} TRAINING Loss: {train_loss:.3e}  VALIDATION Loss: {valid_loss:.3e}')
            if instrument.lsf is not None:
                print (f'LSF: {instrument.lsf.weight.data}')

        # checkpoints
        if epoch_ % 5 == 0 or epoch_ == n_epoch - 1:
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save({
                "model": unwrapped_model.state_dict(),
                "losses": losses,
            }, outfile)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="data file directory")
    parser.add_argument("outfile", help="output file name")
    parser.add_argument("-n", "--latents", help="latent dimensionality", type=int, default=2)
    parser.add_argument("-b", "--batch_size", help="batch size", type=int, default=1024)
    parser.add_argument("-l", "--batch_number", help="number of batches per epoch", type=int, default=None)
    parser.add_argument("-e", "--epochs", help="number of epochs", type=int, default=200)
    parser.add_argument("-r", "--rate", help="learning rate", type=float, default=1e-3)
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

    # fit the LSF
    if args.lsf_size > 0:
        instrument.lsf.weight.requires_grad = True

    # restframe wavelength for reconstructed spectra
    z_max = 0.2
    lmbda_min = instrument.wave_obs.min()/(1+z_max)
    lmbda_max = instrument.wave_obs.max()
    bins = int(instrument.wave_obs.shape[0] * (1 + z_max))
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
            normalize=True,
    )

    # check if outfile already exists, continue only of -c is set
    if os.path.isfile(args.outfile) and not args.clobber:
        raise SystemExit("\nOutfile exists! Set option -C to continue training.")
    losses = None
    if os.path.isfile(args.outfile):
        if args.verbose:
            print (f"\nLoading file {args.outfile}")
        model, losses = load_model(args.outfile, model, instrument)

    train(model, instrument, trainloader, validloader, n_epoch=args.epochs, n_batch=args.batch_number, outfile=args.outfile, losses=losses, lr=args.rate, verbose=args.verbose)
