#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from functools import partial

from spender.util import BatchedFilesDataset, load_batch
from spender.flow import NeuralDensityEstimator

def get_data_loader(dir, which=None, batch_size=10000, shuffle=False, shuffle_instance=True,tag=None):
    files = ["%s/%s"%(dir,item) for item in os.listdir(dir)]
    if tag is not None:files=[item for item in files if tag in item]
    NBATCH = len(files)
    train_batches = files[:int(0.85*NBATCH)]
    valid_batches = files[int(0.85*NBATCH):]

    if which == "valid":files = valid_batches
    elif which == "train": files = train_batches

    load_fct = partial(load_batch)
    data = BatchedFilesDataset(files, load_fct, shuffle=shuffle, shuffle_instance=shuffle_instance)
    return DataLoader(data, batch_size=batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

s_dir = "runtime"
tag = "zfree_c"
model_file = "flow.pkl"

data_loader = get_data_loader(s_dir,which="train",tag=tag)
valid_data_loader = get_data_loader(s_dir,which="valid",tag=tag)

for k,batch in enumerate(data_loader):
    sample = batch[0]
    break

print("sample to infer dimensionality",
      sample.shape,sample.device)
print("device:", device)

n_latent = 6

if os.path.isfile(model_file):
    print("loading from ",model_file)
    NDE_theta = torch.load(model_file,map_location=device)
else:
    NDE_theta = NeuralDensityEstimator(normalize=False,initial_pos={'bounds': [[0, 0]] * n_latent, 'std': [0.05] * n_latent}, method='maf')
    sample = torch.Tensor(sample).to(device)
    NDE_theta.build(sample)

n_epoch = 100
n_steps = 20

scheduler = torch.optim.lr_scheduler.OneCycleLR(NDE_theta.optimizer,max_lr=1e-2,steps_per_epoch=n_steps,epochs=n_epoch)
for i, epoch in enumerate(range(n_epoch)):
    print('    Epoch {0}'.format(epoch))
    print('    lr:', NDE_theta.optimizer.param_groups[0]['lr'])
    
    train_loss = []
    for k,batch in enumerate(data_loader):
        NDE_theta.optimizer.zero_grad()
        latent = batch[0]
        loss = -NDE_theta.net.log_prob(latent).mean()
        loss.backward()
        NDE_theta.optimizer.step()
        train_loss.append(loss.item())
        if k>=n_steps:continue
    train_loss = np.mean(train_loss)
    NDE_theta.train_loss_history.append(train_loss)

    valid_loss = []
    for k,batch in enumerate(valid_data_loader):
        latent = batch[0]
        loss = -NDE_theta.net.log_prob(latent).mean()
        valid_loss.append(loss.item())
    valid_loss = np.mean(valid_loss)
    NDE_theta.valid_loss_history.append(valid_loss)
    print(f'Loss = {train_loss:.3f} (train), {valid_loss:.3f} (valid)')
    scheduler.step()

    if epoch%10 ==0 or epoch==n_epoch-1:
        NDE_theta.save_model(model_file)
