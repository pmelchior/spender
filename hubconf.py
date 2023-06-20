dependencies = ['torch', 'os']

from spender import load_model as _load_model
from spender.data.sdss import SDSS as _SDSS

hub_server = "https://hub.pmelchior.net/"

def _sdss_model(url, **kwargs):
    """needs docstrings
    """
    instrument = _SDSS()
    model = _load_model(url, instrument, kwargs)
    return instrument, model

def sdss_I(**kwargs):
    """needs docstrings
    """
    url = hub_server + "spender.sdss.paperI-08798cbc.pt"
    return _sdss_model(url, kwargs)

def sdss_I_superres(**kwargs):
    """needs docstrings
    """
    url = hub_server + "spender.sdss.paperI.superres-0403266c.pt"
    return _sdss_model(url, kwargs)

def sdss_II(**kwargs):
    """needs docstrings
    """
    url = hub_server + "pender.sdss.paperII-c273bb69.pt"
    return _sdss_model(url, kwargs)
