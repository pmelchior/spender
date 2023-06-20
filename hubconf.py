dependencies = ['torch', 'os']

from spender import load_model as _load_model
from spender.data.sdss import SDSS as _SDSS

hub_server = "https://hub.pmelchior.net/"

def sdss_I(device=None, **kwargs):
    instrument = _SDSS()
    url = hub_server + "sdss.speculator+1.variable.lr_1e-3.latent_10.0.pt"
    model = _load_model(url, instrument, device=device)
    return instrument, model