import io
import pickle
import random
from itertools import chain

import GPUtil
import humanize
import psutil
import torch
from torch.utils.data import IterableDataset


@torch.jit.script
def interp1d_single(
    x: torch.Tensor, y: torch.Tensor, target: torch.Tensor, mask: bool = True
) -> torch.Tensor:
    m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    b = y[:-1] - (m * x[:-1])

    idx = torch.sum(torch.ge(target[:, None], x[None, :]), 1) - 1
    idx = torch.clamp(idx, 0, len(m) - 1)

    itp = m[idx] * target + b[idx]

    if mask:
        low_mask = torch.le(target, x[0])
        high_mask = torch.ge(target, x[-1])
        itp[low_mask] = y[0]
        itp[high_mask] = y[-1]

    return itp


@torch.jit.script
def interp1d(
    x: torch.Tensor, y: torch.Tensor, target: torch.Tensor, mask: bool = True
) -> torch.Tensor:
    """One-dimensional linear interpolation. If x is not sorted, this will sort x for you.

    Args:
        x: the x-coordinates of the data points, must be increasing.
        y: the y-coordinates of the data points, same length as `x`.
        target: the x-coordinates at which to evaluate the interpolated values.
        mask: whether to clamp target values outside of the range of x (i.e., don't extrapolate)

    Returns:
        the interpolated values, same size as `target`.
    """
    # check dimensions
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if y.dim() == 1:
        y = y.unsqueeze(0)
    if target.dim() == 1:
        target = target.unsqueeze(0)

    # check whether we need to broadcast x and y
    assert (
        x.shape[0] == y.shape[0] or x.shape[0] == 1 or y.shape[0] == 1
    ), f"x and y must have same length, or either x or y must have length 1, got {x.shape} and {y.shape}"

    if y.shape[0] == 1 and x.shape[0] > 1:
        y = y.expand(x.shape[0], -1)
        bs = x.shape[0]
    elif x.shape[0] == 1 and y.shape[0] > 1:
        x = x.expand(y.shape[0], -1)
        bs = y.shape[0]
    else:
        bs = x.shape[0]

    # check whether we need to broadcast target
    assert (
        target.shape[0] == bs or target.shape[0] == 1
    ), f"target must have same length as x and y, or length 1, got {target.shape} and {x.shape}"

    if target.shape[0] == 1:
        target = target.expand(bs, -1)

    # check for sorting
    if not torch.all(torch.diff(x, dim=-1) > 0):
        # if reverse-sorted, just flip
        if torch.all(torch.diff(x, dim=-1) < 0):
            x = x.flip(-1)
            y = y.flip(-1)
        else:
            # sort x and y if not already sorted
            x, idx = torch.sort(x, dim=-1)
            y = y[torch.arange(bs)[:, None], idx]

    # this is apparantly how parallelism works in pytorch?
    futures = [
        torch.jit.fork(interp1d_single, x[i], y[i], target[i], mask) for i in range(bs)
    ]
    itp = torch.stack([torch.jit.wait(f) for f in futures])

    return itp


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
