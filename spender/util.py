import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from itertools import chain
import pickle, humanize, psutil, GPUtil, io, random
from torchinterp1d import Interp1d

############ Functions for creating batched files ###############
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def load_batch(batch_name, subset=None):
    with open(batch_name, 'rb') as f:
        if torch.cuda.is_available():
            batch = pickle.load(f)
        else:
            batch = CPU_Unpickler(f).load()

    if subset is not None:
        return batch[subset]
    return batch

# based on https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd
class BatchedFilesDataset(IterableDataset):

    def __init__(self, file_list, load_fct, shuffle=False):
        assert len(file_list), "File list cannot be empty"
        self.file_list = file_list
        self.shuffle = shuffle
        self.load_fct = load_fct

    def process_data(self, idx):
        if self.shuffle:
            idx = random.randint(0, len(self.file_list) -1)
        batch_name = self.file_list[idx]
        data = self.load_fct(batch_name)
        for x in zip(*data):
            yield x

    def get_stream(self):
        return chain.from_iterable(map(self.process_data, range(len(self.file_list))))

    def __iter__(self):
        return self.get_stream()

    def __len__(self):
        return len(self.file_list)


def mem_report():
    print("CPU RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ))

    if torch.cuda.device_count() ==0: return

    GPUs = GPUtil.getGPUs()
    for i, gpu in enumerate(GPUs):
        print('GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%'.format(i, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil*100))
    return


def resample_to_restframe(wave_obs,wave_rest,y,w,z):
    wave_z = (wave_rest.unsqueeze(1)*(1 + z)).T
    wave_obs = wave_obs.repeat(y.shape[0],1)
    # resample observed spectra to restframe
    yrest = Interp1d()(wave_obs, y, wave_z)
    wrest =  Interp1d()(wave_obs, w, wave_z)

    # interpolation = extrapolation outside of observed region, need to mask
    msk = (wave_z<=wave_obs.min())|(wave_z>=wave_obs.max())
    # yrest[msk]=0 # not needed because all spectral elements are weighted
    wrest[msk]=0
    return yrest,wrest
