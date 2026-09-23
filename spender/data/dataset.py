"""Spectrum datasets as sharded Parquet files

The prepared spectra of an instrument are stored as Parquet shards in the layout of a
HuggingFace dataset::

    path/data/train-00000.parquet
    path/data/train-00001.parquet
    ...
    path/data/validation-00000.parquet
    path/data/test-00000.parquet

`path` can be a local directory or uploaded as-is to the HuggingFace Hub, and
:func:`get_data_loader` reads from either.
"""
import hashlib
import os
from functools import partial

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

SPLITS = {"train": 0.7, "validation": 0.15, "test": 0.15}


def get_features(L, id_columns):
    """Get the dataset features for prepared spectra

    Parameters
    ----------
    L: int
        Length of the observed wavelength vector of the instrument
    id_columns: list of string
        Names of the integer columns that identify each spectrum

    Returns
    -------
    :class:`datasets.Features`
    """
    from datasets import Features, List, Value

    features = {
        "spec": List(Value("float32"), length=L),
        "w": List(Value("float32"), length=L),
        "z": Value("float32"),
        "zerr": Value("float32"),
        "norm": Value("float32"),
    }
    features.update({c: Value("int64") for c in id_columns})
    return Features(features)


def assign_split(ids, fractions=SPLITS):
    """Assign spectra to splits by hashing their identifiers

    The assignment only depends on the identifiers, not on the order or the grouping
    of the spectra, so it is reproducible across machines and data releases.

    Parameters
    ----------
    ids: `numpy.ndarray`, shape (N, K)
        Integer identifiers of each spectrum
    fractions: dict
        Fraction of spectra for each split name

    Returns
    -------
    `numpy.ndarray` of split names, shape (N, )
    """
    ids = np.ascontiguousarray(ids, dtype=np.int64).reshape(len(ids), -1)
    h = [hashlib.blake2b(row.tobytes(), digest_size=8).digest() for row in ids]
    u = np.array([int.from_bytes(h_, "little") for h_ in h]) / 2.0**64
    edges = np.cumsum(list(fractions.values()))
    idx = np.minimum(np.searchsorted(edges, u, side="right"), len(edges) - 1)
    return np.array(list(fractions.keys()))[idx]


def write_dataset(
    path,
    batches,
    features,
    id_columns,
    fractions=SPLITS,
    shard_size=8192,
    row_group_size=1024,
    compression="zstd",
):
    """Write batches of prepared spectra into Parquet shards

    Parameters
    ----------
    path: string
        Root directory of the dataset
    batches: iterable of dict
        Each batch maps the column names of `features` to arrays with a leading
        dimension N
    features: :class:`datasets.Features`
        Features of the dataset, see :func:`get_features`
    id_columns: list of string
        Columns that identify each spectrum, used for :func:`assign_split`
    fractions: dict
        Fraction of spectra for each split name
    shard_size: int
        Number of spectra in each Parquet file
    row_group_size: int
        Number of spectra in each row group, the unit of reading when streaming
    compression: string
        Parquet compression codec

    Returns
    -------
    dict with the number of spectra in each split
    """
    os.makedirs(os.path.join(path, "data"), exist_ok=True)
    schema = features.arrow_schema
    writers = {split: _ShardWriter(path, split, schema, shard_size, row_group_size, compression) for split in fractions}

    for batch in batches:
        if not len(batch[id_columns[0]]):
            continue
        table = _to_table(batch, schema)
        ids = np.stack([np.asarray(batch[c]) for c in id_columns], axis=1)
        splits = assign_split(ids, fractions)
        for split, writer in writers.items():
            sel = np.flatnonzero(splits == split)
            if len(sel):
                writer.write(table.take(sel))

    for writer in writers.values():
        writer.close()
    return {split: writer.total for split, writer in writers.items()}


def _to_table(batch, schema):
    arrays = []
    for field in schema:
        x = batch[field.name]
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        is_list = pa.types.is_fixed_size_list(field.type)
        dtype = (field.type.value_type if is_list else field.type).to_pandas_dtype()
        x = np.asarray(x, dtype=dtype)
        if is_list:
            L = field.type.list_size
            assert x.ndim == 2 and x.shape[1] == L, f"{field.name} must have shape (N, {L})"
            arrays.append(pa.FixedSizeListArray.from_arrays(pa.array(x.ravel()), L))
        else:
            arrays.append(pa.array(x.ravel()))
    return pa.Table.from_arrays(arrays, schema=schema)


class _ShardWriter:
    """Writes one split into Parquet files of `shard_size` rows with full row groups"""

    def __init__(self, path, split, schema, shard_size, row_group_size, compression):
        assert shard_size % row_group_size == 0, "shard_size must be a multiple of row_group_size"
        self.path, self.split, self.schema = path, split, schema
        self.shard_size, self.row_group_size, self.compression = shard_size, row_group_size, compression
        self.buffer, self.buffered = [], 0
        self.writer, self.shard, self.in_shard, self.total = None, 0, 0, 0

    def write(self, table):
        self.buffer.append(table)
        self.buffered += len(table)
        if self.buffered >= self.row_group_size:
            table = pa.concat_tables(self.buffer)
            n = len(table) - len(table) % self.row_group_size
            for start in range(0, n, self.row_group_size):
                self._write_group(table.slice(start, self.row_group_size))
            self.buffer = [table.slice(n)]
            self.buffered = len(table) - n

    def close(self):
        if self.buffered:
            self._write_group(pa.concat_tables(self.buffer))
        self.buffer, self.buffered = [], 0
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def _write_group(self, table):
        if self.writer is None:
            filename = os.path.join(self.path, "data", f"{self.split}-{self.shard:05d}.parquet")
            self.writer = pq.ParquetWriter(filename, self.schema, compression=self.compression)
        self.writer.write_table(table, row_group_size=self.row_group_size)
        self.in_shard += len(table)
        self.total += len(table)
        if self.in_shard >= self.shard_size:
            self.writer.close()
            self.writer, self.shard, self.in_shard = None, self.shard + 1, 0


class ParquetStream(IterableDataset):
    """Stream spectra from Parquet files

    Each pass reads the files in random order, one row group at a time, and passes the
    spectra through a shuffle buffer. The buffer replaces randomly selected spectra with
    each new row group, so that batches mix spectra across row groups and files, and
    their composition changes in every epoch.

    With several :class:`~torch.utils.data.DataLoader` workers, each worker reads a
    different subset of the files.

    Parameters
    ----------
    files: list of string
        Local paths or URLs (e.g. `hf://datasets/...`) of the Parquet files
    columns: list of string
        Which columns to return, in this order, for each spectrum
    shuffle: bool
        Whether to shuffle the files and the spectra
    buffer_size: int
        Number of spectra in the shuffle buffer
    seed: int
        Random seed for shuffling
    """

    def __init__(self, files, columns, shuffle=False, buffer_size=8192, seed=None):
        assert len(files), "File list cannot be empty"
        self.files = list(files)
        self.columns = list(columns)
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        # common seed so that all workers agree on the file order
        self.seed = np.random.SeedSequence().entropy if seed is None else seed
        self.epoch = 0
        # counts passes over this copy, so that persistent workers reshuffle even
        # though they don't see calls to set_epoch
        self._passes = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        files = self.files
        rng = np.random.default_rng([self.seed, self.epoch, self._passes])
        self._passes += 1
        if self.shuffle:
            files = [files[i] for i in rng.permutation(len(files))]
        worker = get_worker_info()
        if worker is not None:
            files = files[worker.id :: worker.num_workers]
            rng = np.random.default_rng([self.seed, self.epoch, self._passes, worker.id])

        chunks = self._read(files)
        if self.shuffle:
            chunks = self._shuffle(chunks, rng)
        for chunk in chunks:
            tensors = [torch.from_numpy(chunk[c]) for c in self.columns]
            yield from zip(*tensors)

    def _read(self, files):
        for filename in files:
            with fsspec.open(filename, "rb") as f:
                pf = pq.ParquetFile(f)
                for i in range(pf.num_row_groups):
                    table = pf.read_row_group(i, columns=self.columns)
                    yield {c: _to_numpy(table.column(c).combine_chunks()) for c in self.columns}

    def _shuffle(self, chunks, rng):
        buffer, n = None, 0
        for chunk in chunks:
            if buffer is None:
                buffer = {c: np.empty((self.buffer_size,) + x.shape[1:], x.dtype) for c, x in chunk.items()}
            m = len(chunk[self.columns[0]])
            # fill buffer first
            k = min(self.buffer_size - n, m)
            for c in self.columns:
                buffer[c][n : n + k] = chunk[c][:k]
            n += k
            # then swap the remaining spectra with random ones from the buffer
            for start in range(k, m, self.buffer_size):
                stop = min(start + self.buffer_size, m)
                idx = rng.choice(self.buffer_size, size=stop - start, replace=False)
                out = {c: buffer[c][idx] for c in self.columns}
                for c in self.columns:
                    buffer[c][idx] = chunk[c][start:stop]
                yield out
        if n:
            idx = rng.permutation(n)
            yield {c: buffer[c][idx] for c in self.columns}


def _to_numpy(array):
    if pa.types.is_fixed_size_list(array.type):
        L = array.type.list_size
        return array.values.to_numpy(zero_copy_only=False, writable=True).reshape(len(array), L)
    return array.to_numpy(zero_copy_only=False, writable=True)


def _collate(rows, columns):
    return tuple(torch.stack([row[c] for row in rows]) for c in columns)


class _EpochDataLoader(DataLoader):
    """DataLoader that advances the epoch of its dataset on every pass

    When wrapped by `accelerate`, its own data loader calls `set_epoch` in the same way.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch = 0

    def __iter__(self):
        self.dataset.set_epoch(self.epoch)
        self.epoch += 1
        return super().__iter__()


def get_data_loader(
    path,
    which="train",
    columns=("spec", "w", "z"),
    batch_size=1024,
    shuffle=False,
    buffer_size=8192,
    streaming=True,
    seed=None,
    num_workers=0,
    **kwargs,
):
    """Get a dataloader for batches of spectra

    Parameters
    ----------
    path: string
        Local directory or HuggingFace Hub repository of the dataset
    which: ['train', 'valid', 'test']
        Which split of the spectra to return
    columns: list of string
        Which columns to return, in this order, for each batch
    batch_size: int
        Number of spectra in each batch
    shuffle: bool
        Whether to shuffle the spectra
    buffer_size: int
        Number of spectra in the shuffle buffer when `streaming`, see
        :class:`ParquetStream`. Each worker holds its own buffer.
    streaming: bool
        Whether to stream the Parquet files instead of converting them into a
        memory-mapped Arrow cache. The cache allows a global shuffle but needs
        additional disk space of about the size of the dataset.
    seed: int
        Random seed for shuffling a streaming dataset
    num_workers: int
        Number of worker processes. For streaming, each worker reads its own files,
        so the dataset needs at least `num_workers` files. Workers are persistent
        by default to avoid restarting them in every epoch.
    kwargs: dict
        Additional arguments for :class:`torch.utils.data.DataLoader`

    Returns
    -------
    :class:`torch.utils.data.DataLoader`, yields tuples of `columns`
    """
    import datasets

    split = {"valid": "validation"}.get(which, which)
    if num_workers > 0:
        kwargs.setdefault("persistent_workers", True)
    if streaming:
        files = datasets.load_dataset_builder(path).config.data_files[split]
        ds = ParquetStream(files, columns, shuffle=shuffle, buffer_size=buffer_size, seed=seed)
        return _EpochDataLoader(ds, batch_size=batch_size, num_workers=num_workers, **kwargs)

    ds = datasets.load_dataset(path, split=split)
    ds = ds.select_columns(list(columns)).with_format("torch")
    collate_fn = partial(_collate, columns=columns)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers, **kwargs)
