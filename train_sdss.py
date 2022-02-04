#!/usr/bin/env python
# coding: utf-8

#!pip install git+https://github.com/aliutkus/torchinterp1d.git
#!pip install accelerate


import os, sys 
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions.normal import Normal

from model import *
from accelerate import Accelerator

data_dir = "/scratch/gpfs/yanliang"
dataname = "cutted-sdss_spectra"
data_file = "%s/%s.npz"%(data_dir,dataname)

# load data, specify GPU to prevent copying later
# TODO: use torch.cuda.amp.autocast() for FP16/BF16 typcast
# TODO: need dynamic data loader if we use larger data sets
device = torch.device(type='cuda', index=0)
data = load_data(data_file, which="train", device=device)
valid_data = load_data(data_file, which="valid", device=device)

batch_size=2048
trainloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(data['y'], data['w'], data['z']),
    batch_size=batch_size,
    shuffle=False)

validloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(valid_data['y'], valid_data['w'], valid_data['z']),
    batch_size=batch_size)


# restframe wavelength for reconstructed spectra
wave_obs = torch.tensor(data['wave'], dtype=torch.float32)
lmbda_min = data['wave'].min()/(1+data['z'].max())
lmbda_max = data['wave'].max()
bins = int(data['wave'].shape[0] * (1 + data['z'].max()))
wave_rest = torch.linspace(lmbda_min, lmbda_max, bins, dtype=torch.float32)

print ("Observed frame:\t{:.0f} .. {:.0f} A ({} bins)".format(wave_obs.min(), wave_obs.max(), len(wave_obs)))
print ("Restframe:\t{:.0f} .. {:.0f} A ({} bins)".format(lmbda_min, lmbda_max, bins))

def train(model, trainloader, validloader, n_epoch=200, label="", silent=False, lr=3e-4):
    accelerator = Accelerator()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=n_epoch)
    model, optimizer, trainloader, validloader = accelerator.prepare(model, optimizer, trainloader, validloader)
    
    losses = []
    for epoch in range(n_epoch):
        model.train()
        train_loss = 0.
        for batch in trainloader:
            spec, w, z = batch
            optimizer.zero_grad()
            loss = model.loss(spec, w, z0=z)
            accelerator.backward(loss)
            train_loss += loss.item() 
            optimizer.step()        
        train_loss /= len(trainloader.dataset)

        with torch.no_grad():
            model.eval()
            valid_loss = 0.
            for batch in validloader:
                spec, w, z = batch
                loss = model.loss(spec, w, z0=z)
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
#label = "model.series.z-forward.10.sdss.38816"
label = "%s/%s"%(savemodel,dataname)
n_epoch = 300

for i in range(n_model):
    n_hidden = (1024, 256, 64)
    model = SeriesAutoencoder(
            wave_obs,
            wave_rest,
            n_latent=n_latent,
            n_hidden_dec=n_hidden,
    )
    print (f"--- Model {i}/{n_model}")
    print (f"Parameters:", model.n_parameters)
    train(model, trainloader, validloader, n_epoch=n_epoch, label=label+f".{i}", lr=1e-3)