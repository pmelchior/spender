import io
import pickle
import random
from collections import defaultdict
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


class LossTracker:
    """Tracks an arbitrary number of named losses, split by train/validation

    Training scripts call :meth:`update` once per batch with a dict of that batch's
    losses (any set of names, which may differ between calls) and a normalization
    weight (typically the batch size); :meth:`end_epoch` folds those into a running,
    weighted per-epoch mean for each name. A name that is skipped in a given epoch
    (e.g. a loss term that is turned off for a stretch of training) is recorded as 0
    for that epoch so all histories stay aligned by epoch index.

    :meth:`state_dict` / :meth:`load_state_dict` let the tracker be saved and resumed
    alongside the model weights in the same checkpoint file.
    """

    def __init__(self):
        self.history = {"train": defaultdict(list), "valid": defaultdict(list)}
        self._epoch = 0
        self._reset_running()

    def _reset_running(self):
        self._sums = {"train": defaultdict(float), "valid": defaultdict(float)}
        self._counts = {"train": defaultdict(float), "valid": defaultdict(float)}

    def update(self, split, losses, weight=1):
        """Accumulate one batch's losses

        Parameters
        ----------
        split: "train" or "valid"
        losses: dict
            Mapping of loss name to value (`torch.Tensor` or float) for this batch
        weight: float
            Normalization weight for this batch, typically the batch size
        """
        sums, counts = self._sums[split], self._counts[split]
        for name, value in losses.items():
            sums[name] += (value.item() if hasattr(value, "item") else value) * weight
            counts[name] += weight

    def end_epoch(self):
        """Fold the accumulated batches into this epoch's per-name means"""
        for split in ("train", "valid"):
            names = set(self.history[split]) | set(self._sums[split])
            for name in names:
                count = self._counts[split][name]
                mean = self._sums[split][name] / count if count else 0.0
                self.history[split][name].append(mean)
        self._epoch += 1
        self._reset_running()

    @property
    def epoch(self):
        """Number of completed epochs"""
        return self._epoch

    def state_dict(self):
        return {
            "epoch": self._epoch,
            "train": dict(self.history["train"]),
            "valid": dict(self.history["valid"]),
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict) or "train" not in state or "valid" not in state:
            # backwards compat: pre-LossTracker checkpoints stored losses as a
            # plain array; that history can't be recovered, so start tracking fresh
            return
        self.history = {split: defaultdict(list, state[split]) for split in ("train", "valid")}
        self._epoch = state.get("epoch", 0)
        self._reset_running()

    def plot(self, names=None, log=True, ax=None):
        """Plot train (solid) and validation (dashed) loss curves by name"""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        if names is None:
            names = sorted(set(self.history["train"]) | set(self.history["valid"]))
        for name in names:
            if name in self.history["train"]:
                line, = ax.plot(self.history["train"][name], label=f"{name} (train)")
            if name in self.history["valid"]:
                color = line.get_color() if name in self.history["train"] else None
                ax.plot(self.history["valid"][name], "--", color=color, label=f"{name} (valid)")
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        return ax


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
