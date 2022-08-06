#!/usr/bin/env python

import argparse
import numpy as np
import torch
from torch import nn
from torch import optim
from accelerate import Accelerator
from spender import SpectrumAutoencoder, get_instrument, get_data_loader

def train(model, instrument, trainloader, validloader, n_epoch=200, n_batch=None, mask_skyline=True, label="", verbose=False, lr=3e-4):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=n_epoch)

    accelerator = Accelerator(mixed_precision='fp16')
    model, instrument, trainloader, validloader, optimizer = accelerator.prepare(model, instrument, trainloader, validloader, optimizer)

    losses = []
    for epoch in range(n_epoch):
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
            print('====> Epoch: %i TRAINING Loss: %.2e VALIDATION Loss: %.2e' % (epoch, train_loss, valid_loss))

        # checkpoints
        if epoch % 5 == 0 or epoch == n_epoch - 1:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_instrument = accelerator.unwrap_model(instrument)
            accelerator.save({
                "model": unwrapped_model.state_dict(),
                "instrument": unwrapped_instrument.state_dict(),
                "optimizer": optimizer.optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "losses": losses,
            }, f'{label}.pt')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="data file directory")
    parser.add_argument("label", help="output file label")
    parser.add_argument("-o", "--outdir", help="output file directory", default=".")
    parser.add_argument("-n", "--latents", help="latent dimensionality", type=int, default=2)
    parser.add_argument("-b", "--batch_size", help="batch size", type=int, default=1024)
    parser.add_argument("-l", "--batch_number", help="number of batches per epoch", type=int, default=None)
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
    trainloader = get_data_loader(args.dir, sdss.name, which="train", batch_size=args.batch_size, shuffle=True)
    validloader = get_data_loader(args.dir, sdss.name, which="valid", batch_size=args.batch_size)

    if args.verbose:
        print ("Observed frame:\t{:.0f} .. {:.0f} A ({} bins)".format(sdss.wave_obs.min(), sdss.wave_obs.max(), len(sdss.wave_obs)))
        print ("Restframe:\t{:.0f} .. {:.0f} A ({} bins)".format(lmbda_min, lmbda_max, bins))

    # define and train the model
    model = SpectrumAutoencoder(
            wave_rest,
            n_latent=args.latents,
            normalize=True,
    )
    label = f'{args.outdir}/{args.label}.{args.latents}'
    train(model, sdss, trainloader, validloader, n_epoch=args.epochs, n_batch=args.batch_number, label=label, lr=args.rate, verbose=args.verbose)
