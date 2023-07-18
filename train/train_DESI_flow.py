#!/usr/bin/env python
# coding: utf-8

import io, os, sys, random
import numpy as np
import pickle
import torch
from torch.utils.data import IterableDataset, DataLoader
from tqdm import trange
from itertools import chain

from spender.flow import NeuralDensityEstimator

def _train(self, n_epochs: int = 2000, suffix: str = "nde"):
    """
    Train the neural density estimator based on input data.
    Here we use the ``log(P)`` loss. This function is not used in the ``popsed`` project.
    Parameters
    ----------
    n_epochs: int.
        Number of epochs to train.
    display: bool.
        Whether to display the training loss.
    suffix: str.
        Suffix to add to the output file.
    """
    min_loss = -19
    patience = 5
    self.best_loss_epoch = 0
    self.net.train()

    for epoch in trange(n_epochs, desc='Training NDE', unit='epochs'):
        self.optimizer.zero_grad()
        loss = -self.net.log_prob(self.batch_x).mean()
        loss.backward()
        self.optimizer.step()
        self.train_loss_history.append(loss.item())

        if loss.item() < min_loss:
            min_loss = loss.item()
            if epoch - self.best_loss_epoch > patience:
                # Don't save model too frequently
                self.best_loss_epoch = epoch
                self.save_model(
                    f'best_loss_model_{suffix}_{self.method}.pkl')

    if min_loss == -18:
        raise Warning('The training might be failed, try more epochs')


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

class BatchedFilesDataset(IterableDataset):

    def __init__(self, file_list, load_fct, shuffle=False, shuffle_instance=False):
        assert len(file_list), "File list cannot be empty"
        self.file_list = file_list
        self.shuffle = shuffle
        self.shuffle_instance = shuffle_instance
        self.load_fct = load_fct

    def process_data(self, idx):
        if self.shuffle:
            idx = random.randint(0, len(self.file_list) -1)
        batch_name = self.file_list[idx]
        data = self.load_fct(batch_name)
        data = list(zip(*data))
        if self.shuffle_instance:
            random.shuffle(data)
        for x in data:
            yield x

    def get_stream(self):
        return chain.from_iterable(map(self.process_data, range(len(self.file_list))))

    def __iter__(self):
        return self.get_stream()

    def __len__(self):
        return len(self.file_list)
    
from functools import partial

def load_batch(batch_name):
    #print("batch_name:",batch_name)
    with open(batch_name, 'rb') as f:
        if torch.cuda.is_available():
            batch = pickle.load(f)
        else:
            batch = CPU_Unpickler(f).load()
    batch = [item.detach().to(device) for item in batch]
    return batch

def get_latent_data_loader(dir, which=None, batch_size=10000, shuffle=False, shuffle_instance=True,latent_tag=None):
    files = ["%s/%s"%(dir,item) for item in os.listdir(dir)]
    if latent_tag is not None:files=[item for item in files if latent_tag in item]
    NBATCH = len(files)
    train_batches = files[:int(0.85*NBATCH)]
    valid_batches = files[int(0.85*NBATCH):]

    if which == "valid":files = valid_batches
    elif which == "train": files = train_batches

    load_fct = partial(load_batch)
    data = BatchedFilesDataset(files, load_fct, shuffle=shuffle, shuffle_instance=shuffle_instance)
    return DataLoader(data, batch_size=batch_size)




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
s_dir = "runtime" # directory that saves latent vectors
model_tag = "star_k2" # spender model + data tag
data_tag = "DESIStars" # DESI data
latent_tag = "%s-%s"%(model_tag,data_tag)

flow_file = sys.argv[1]

print("flow_file:",flow_file)
print("latent data:",latent_tag)

data_loader = get_latent_data_loader(s_dir,which="train",latent_tag=latent_tag)
valid_data_loader = get_latent_data_loader(s_dir,which="valid",latent_tag=latent_tag)

for k,batch in enumerate(data_loader):
    sample = batch[0]
    break

print("sample to infer dimensionality",
      sample.shape,sample.device)
print("device:", device)
print("torch.cuda.device_count():",torch.cuda.device_count())
n_latent = 6

if "new" in sys.argv:
    NDE_theta = NeuralDensityEstimator(normalize=False,initial_pos={'bounds': [[0, 0]] * n_latent, 'std': [0.05] * n_latent}, method='maf')
    sample = torch.Tensor(sample).to(device)
    NDE_theta.build(sample)
else: NDE_theta = torch.load(flow_file,map_location=device)

n_epoch = 100
n_steps = 20

scheduler = torch.optim.lr_scheduler.OneCycleLR(NDE_theta.optimizer,max_lr=2e-3,
                                                steps_per_epoch=n_steps,
                                                epochs=n_epoch)
for i, epoch in enumerate(range(n_epoch)):
    print('    Epoch {0}'.format(epoch))
    print('    lr:', NDE_theta.optimizer.param_groups[0]['lr'])
    
    train_loss = []
    #t = trange(100,desc='Training NDE_theta',unit='epochs')
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
        NDE_theta.save_model(flow_file)
