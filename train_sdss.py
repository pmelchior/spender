#!/usr/bin/env python
# coding: utf-8

#!pip install git+https://github.com/aliutkus/torchinterp1d.git
#!pip install accelerate

import sys
assert len(sys.argv) == 2, "usage: train_sdss <path_to_spectra.npz>"
filename = sys.argv[1]

import numpy as np
import torch
from torch import nn
from torch import optim
from accelerate import Accelerator

from model import SpectrumAutoencoder, Instrument
from util import load_data, augment_batch

# load data, specify device to prevent copying later
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = load_data(filename, which="train", device=device)
valid_data = load_data(filename, which="valid", device=device)


batch_size=2048
trainloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(data['y'], data['w'], data['z']),
    batch_size=batch_size,
    shuffle=False)

validloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(valid_data['y'], valid_data['w'], valid_data['z']),
    batch_size=batch_size)

# define SDSS instrument
wave_obs = torch.tensor(data['wave'], dtype=torch.float32)
sdss = Instrument(wave_obs)

# restframe wavelength for reconstructed spectra
lmbda_min = data['wave'].min()/(1+data['z'].max())
lmbda_max = data['wave'].max()
bins = int(data['wave'].shape[0] * (1 + data['z'].max()))
wave_rest = torch.linspace(lmbda_min, lmbda_max, bins, dtype=torch.float32)

print ("Observed frame:\t{:.0f} .. {:.0f} A ({} bins)".format(wave_obs.min(), wave_obs.max(), len(wave_obs)))
print ("Restframe:\t{:.0f} .. {:.0f} A ({} bins)".format(lmbda_min, lmbda_max, bins))

def train(model, accelerator, instrument, trainloader, validloader, n_epoch=200, label="", silent=False, lr=3e-4, augmented=False):

    assert augmented in [True, False, "redshift", "mask"]
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=n_epoch)
    model, optimizer = accelerator.prepare(model, optimizer)

    losses = []
    for epoch in range(n_epoch):
        model.train()
        train_loss = 0.
        for batch in trainloader:
            spec, w, z = batch
            loss = model.loss(spec, w, instrument=instrument, z=z)
            accelerator.backward(loss)
            train_loss += loss.item()

            if augmented:
                options = ["redshift", "mask"]
                if augmented in options:
                    how = augmented
                else:
                    how = options[np.random.randint(2)] # select augmentation type at random
                spec_, w_, z_ = augment_batch(batch, how=how, wave_obs=instrument.wave_obs)
                
                # encode augmented data
                s = model.encode(spec_, w=w_, z=z_)
                # but compute loss of decoded result against original (including redshift)
                spectrum_restframe = model.decode(s)
                spectrum_observed = model.decoder.transform(spectrum_restframe, instrument=instrument, z=z)
                loss =  model._loss(spec, w, spectrum_observed)
                accelerator.backward(loss)
            
            optimizer.step()
            optimizer.zero_grad()
        train_loss /= len(trainloader.dataset)

        with torch.no_grad():
            model.eval()
            valid_loss = 0.
            for batch in validloader:
                spec, w, z = batch
                loss = model.loss(spec, w, instrument=instrument, z=z)
                valid_loss += loss.item()
            valid_loss /= len(validloader.dataset)

        scheduler.step()
        losses.append((train_loss, valid_loss))

        if epoch % 20 == 0 or epoch == n_epoch - 1:
            if not silent:
                print('====> Epoch: %i TRAINING Loss: %.2e VALIDATION Loss: %.2e' % (epoch, train_loss, valid_loss))

            # checkpoints
            torch.save(model, f'{label}.pt')
            np.save(f'{label}.losses.npy', np.array(losses))


n_latent = 10
n_model = 5
label = "model.series.test"
n_epoch = 400


accelerator = Accelerator(mixed_precision='fp16')
trainloader, validloader, sdss = accelerator.prepare(trainloader, validloader, sdss)

for i in range(n_model):
    model = SpectrumAutoencoder(
            wave_rest,
            n_latent=n_latent,
    )
    print (f"--- Model {i}/{n_model}")

    train(model, accelerator, sdss, trainloader, validloader, n_epoch=n_epoch, label=label+f".{i}", lr=1e-3, augmented=True)
