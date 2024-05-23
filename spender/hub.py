# thin wrapper on top of torch.hub to provide list() and help() for local code directories
import os
from torch.hub import _add_to_sys_path, _import_module, _load_entry_from_hubconf, _load_local, MODULE_HUBCONF

repo_dir = os.path.join(os.path.dirname(__file__), "..")


def list():
    # direct copy from pytorch/torch/hub.py
    with _add_to_sys_path(repo_dir):
        hubconf_path = os.path.join(repo_dir, MODULE_HUBCONF)
        hub_module = _import_module(MODULE_HUBCONF, hubconf_path)

    # We take functions starts with '_' as internal helper functions
    entrypoints = [f for f in dir(hub_module) if callable(getattr(hub_module, f)) and not f.startswith('_')]

    return entrypoints


def help(model):

    with _add_to_sys_path(repo_dir):
        hubconf_path = os.path.join(repo_dir, MODULE_HUBCONF)
        hub_module = _import_module(MODULE_HUBCONF, hubconf_path)

    entry = _load_entry_from_hubconf(hub_module, model)

    return entry.__doc__


def load(model, *args, **kwargs):
    model = _load_local(repo_dir, model, *args, **kwargs)
    return model
