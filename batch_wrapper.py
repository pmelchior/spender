#!/usr/bin/env python
# coding: utf-8
import os, glob, sys, time, io
import numpy as np
import astropy.io.fits as fits
import multiprocessing as mp
import pickle, random
import torch
from torch.utils.data import IterableDataset, DataLoader
from itertools import cycle, chain

from util import get_norm, mem_report


############ Functions for creating batched files ###############

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def load_batch(batch_name):
    with open(batch_name, 'rb') as f:
        if torch.cuda.is_available():
            batch_copy = pickle.load(f)
        else:
            batch_copy = CPU_Unpickler(f).load()

    if type(batch_copy)==list:
        spec,w,z = batch_copy[:3]
        w[w<=1e-6] = 0
        return spec,w,z
    else:
        return batch_copy


# based on https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd
class BatchedFilesDataset(IterableDataset):

    def __init__(self, file_list, shuffle=False, repeat=False):
        assert len(file_list), "File list cannot be empty"
        self.file_list = file_list
        self.shuffle = shuffle
        self.repeat = repeat

    def process_data(self, idx):
        if self.shuffle:
            idx = random.randint(0, len(self.file_list) -1)
        batch_name = self.file_list[idx]
        data = load_batch(batch_name)
        for x in zip(*data):
            yield x

    def get_stream(self):
        if self.repeat:
            return chain.from_iterable(map(self.process_data, cycle(range(len(self.file_list)))))
        return chain.from_iterable(map(self.process_data, range(len(self.file_list))))

    def __iter__(self):
        return self.get_stream()

    def __len__(self):
        return len(self.file_list)

def collect_batches(dir, name, tag="chunk1024", which=None):

    assert name in ["SDSS", "BOSS"]
    filename = f"{name}{tag}_*.pkl"
    batch_files = glob.glob(dir + "/" + filename)
    batches = [item for item in batch_files if not "copy" in item]

    NBATCH = len(batches)
    train_batches = batches[:int(0.7*NBATCH)]
    valid_batches = batches[int(0.7*NBATCH):int(0.85*NBATCH)]
    test_batches = batches[int(0.85*NBATCH):]

    if which == "test": return test_batches
    elif which == "valid": return valid_batches
    elif which == "train": return train_batches
    else: return all_batches

def get_data_loader(dir, name, tag="chunk1024", which=None, batch_size=1024, shuffle=False, repeat=False):
    files = collect_batches(dir, name, tag=tag, which=which)
    # load data on demand, random order
    data = BatchedFilesDataset(files, shuffle=shuffle, repeat=repeat)
    return DataLoader(data, batch_size=batch_size)


############ Functions for creating batched files ###############
def read_sdss_spectra(plate, mjd, fiberid):
    flocal = os.path.join(data_dir, 'sdss-spectra/spec-%s-%i-%s.fits' % (str(plate).zfill(4), mjd, str(fiberid).zfill(4)))
    if not os.path.isfile(flocal):
        flocal = os.path.join(data_dir, 'lite/%s/spec-%s-%i-%s.fits' % (str(plate).zfill(4),str(plate).zfill(4), mjd, str(fiberid).zfill(4)))
        if not os.path.isfile(flocal):raise ValueError
    hdulist = fits.open(flocal)

    header = hdulist[0].header
    data = hdulist[1].data
    logw = data['loglam']
    spec = data['flux']
    ivar = data['ivar']
    mask = data['and_mask'].astype(bool)
    ivar[mask] = 0.
    return logw, spec, ivar, mask

def save_batch(batch_copy,batch_name):
    with open("%s/%s"%(dynamic_dir,batch_name), 'wb') as f:
        pickle.dump(batch_copy,f)
    print("Saving to %s/%s.."%(dynamic_dir,batch_name))
    return

def prepare_batch(input_list):
    targets,wave,code,wh_start,Nspec = input_list
    # padded wavelength range
    max_len = len(wave)
    specs = np.zeros((Nspec, len(wave)), dtype=np.float32)
    weights = np.zeros((Nspec, len(wave)), dtype=np.float32)
    norms = np.zeros(Nspec, dtype=np.float32)
    zreds = np.zeros(Nspec, dtype=np.float32)
    zerrs = np.zeros(Nspec, dtype=np.float32)
    specid = []

    ta = time.time()

    interval = Nspec//5

    i = 0
    for j in range(len(targets[0])):
        plate, mjd, fiberid = [targets[k][j] for k in [0,1,2]]
        try:
            x = read_sdss_spectra(plate, mjd, fiberid)
        except:# (IndexError, OSError):
            print("Error!",plate, mjd, fiberid)
            continue

        Z = targets[3][j]
        Z_ERR = targets[4][j]

        #print("i:",j,"j:",j,plate, mjd, fiberid, Z, Z_ERR)
        if np.sum(x[1]) == 0:continue
        if Z <= 0:continue
        if Z > 0.5:continue
        if np.max(np.abs(x[1]))>1e5:continue

        iw = int(np.around((x[0][0] - wave[0])/0.0001))

        if iw < 0:
            print("x[0][0]:",x[0][0])
            continue

        endpoint = min(iw+len(x[1]),max_len)

        if iw+len(x[1])>max_len:
            print("wavemax:",max(x[0]))
            print("max_len:",max_len,"iw:",iw,
                  "len(x[1]):",len(x[1]))

        norm = get_norm(np.expand_dims(x[1], axis=0))
        w = x[2] * ~x[3] * (norm**2)[:,None]

        if norm<0:
            print("negative norm, continue...")
            continue

        if (w<=0).all():
            print("All zero weight, continue...",plate, mjd, fiberid)
            continue

        w[w<1e-6] = 1e-6
        specs[i,iw:endpoint] = x[1][:endpoint-iw]/norm
        weights[i,iw:endpoint] = w[0][:endpoint-iw]
        zreds[i] = Z
        zerrs[i] = Z_ERR
        norms[i] = norm
        specid.append("%d-%d-%d"%(plate,mjd,fiberid))

        i += 1

        # collected all samples
        if i==Nspec:break
        if debug and (i % interval) == 0:
            tb = time.time()

            t_seg = tb-ta
            remain_seg = (Nspec-i)/interval
            print("\n\n%d: remaining time = %.2f"%(i,remain_seg*t_seg))
            ta = tb
            if debug:mem_report()

    if i<0.9*Nspec:
        print("filling fraction too low... %.2f"%(i/Nspec))
        return
    # save
    rand_int = np.arange(len(specs))
    print("len(specs):",len(specs),#"specid:",specid,
          "rand_int",rand_int.max())

    specid = np.array(specid)
    batch_copy = [torch.tensor(specs[rand_int],device=device),
                  torch.tensor(weights[rand_int],device=device),
                  torch.tensor(zreds[rand_int],device=device),
                  specid[rand_int],norms[rand_int],
                  torch.tensor(zerrs[rand_int],device=device)]

    batch_name = "%s%s_%d.pkl"%("",code,wh_start)
    save_batch(batch_copy,batch_name)
    return

def wrap_batches(which,tag,k_range=[0,10],Nspec=1024):
    header_dir = "/scratch/gpfs/yanliang/headers"
    headers = {0:"truncated-specobj.pkl",
               1:"boss_headers.pkl"}

    header_name = headers[which]
    if "joint" in tag:header_name = "joint_headers.pkl"

    f = open("%s/%s"%(header_dir,header_name),"rb")
    targets = pickle.load(f)
    f.close()

    # all targets
    name = ["SDSS","BOSS"]
    id_names = ['PLATE','MJD','FIBERID',"Z","Z_ERR"]
    print("All targets:",len(targets))
    if "joint" in header_name:
        keys = ["%s_%s"%(k,name[which]) for k in id_names]
    else: keys = id_names

    #targets_slice = {}
    #for k in keys:targets_slice[k] = targets[k]
    targets_slice = []
    for k in keys:targets_slice.append(targets[k].data)

    lower,upper = LOGWAVE_RANGE[which]
    wave = np.arange(lower,upper, 0.0001).astype(np.float32)

    start,end = k_range[0]*Nspec,k_range[1]*Nspec
    start_index = np.arange(start,end, Nspec)
    nbatch = len(start_index)

    size = len(targets_slice[0])
    print("nbatch:",nbatch,"start_index:",start_index)

    code = "%s%s"%(name[which],tag)

    input_list = []
    for k,index in enumerate(start_index):
        loc = range(index,min(index+3*Nspec,size))
        print("loc:",len(loc),loc[:10])
        batch_targets=[item[loc] for item in targets_slice]
        args = [batch_targets,wave,code,index,Nspec]
        input_list.append(args)
        #print("prepared batch %d/%d, time=%.2f"%(k,nbatch,tb-ta))

    n_pool = min(25,len(input_list))
    print("%d processes..."%(n_pool))
    pool = mp.Pool(n_pool)
    result = pool.map(func=prepare_batch, iterable=input_list)
    pool.close()
    pool.join()
    return


if __name__ == "__main__":

    if "batch_wrapper" in sys.argv[0]:
        which = int(sys.argv[1])
        tag = sys.argv[2]
        wrap_batches(which,tag,k_range=[0,20],Nspec=256)
