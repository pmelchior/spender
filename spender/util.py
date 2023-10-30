import io
import pickle
import random
from itertools import chain

import GPUtil
import humanize
import psutil
import torch
from torch.utils.data import IterableDataset
from torchinterp1d import interp1d


############ Functions for creating batched files ###############
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def load_batch(batch_name, subset=None):
    with open(batch_name, "rb") as f:
        if torch.cuda.is_available():
            batch = pickle.load(f)
        else:
            batch = CPU_Unpickler(f).load()

    if subset is not None:
        return batch[subset]
    return batch


class BatchedFilesDataset(IterableDataset):
    """Creates a dataset from a list of batched files

    This class allows the use of batched files, whose size can be optimized for loading
    performance, as input for a :class:`torch.utils.data.DataLoader`, whose batch size
    can be chosen independently to optimize training.

    See https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd
    for details.

    The file list and the items in each loaded file can be shuffled if desired.

    Parameters
    ----------
    file_list: list(str)
        List of filenames to load batches from
    load_fct: callable
        Function to return batch when given filename
    shuffle: bool
        Whether to shuffle the order of the batch files
    shuffle_instance: bool
        Whether to shuffle spectra within each batch

    """

    def __init__(self, file_list, load_fct, shuffle=False, shuffle_instance=False):
        assert len(file_list), "File list cannot be empty"
        self.file_list = file_list
        self.shuffle = shuffle
        self.shuffle_instance = shuffle_instance
        self.load_fct = load_fct

    def process_data(self, idx):
        if self.shuffle:
            idx = random.randint(0, len(self.file_list) - 1)
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


def mem_report():
    print("CPU RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available))

    if torch.cuda.device_count() == 0:
        return

    GPUs = GPUtil.getGPUs()
    for i, gpu in enumerate(GPUs):
        print(
            "GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%".format(
                i, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil * 100
            )
        )
    return


def resample_to_restframe(wave_obs, wave_rest, y, w, z):
    wave_z = (wave_rest.unsqueeze(1) * (1 + z)).T
    wave_obs = wave_obs.repeat(y.shape[0], 1)
    # resample observed spectra to restframe
    yrest = interp1d(wave_obs, y, wave_z)
    wrest = interp1d(wave_obs, w, wave_z)

    # interpolation = extrapolation outside of observed region, need to mask
    msk = (wave_z <= wave_obs.min()) | (wave_z >= wave_obs.max())
    # yrest[msk]=0 # not needed because all spectral elements are weighted
    wrest[msk] = 0
    return yrest, wrest


def calc_normalization(x, y, ivar):
    return ((x * ivar) @ y) / ((x * ivar) @ x)
