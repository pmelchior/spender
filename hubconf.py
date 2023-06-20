dependencies = ['torch', 'os']

from spender import load_model
from spender.model import SpectrumAutoencoder
from spender.data.sdss import SDSS

hub_server = "https://hub.pmelchior.net/"

def sdss_I(device=None, **kwargs):
    instrument = SDSS()
    url = hub_server + "sdss.speculator+1.variable.lr_1e-3.latent_10.0.pt"
    model = load_model(url, instrument, device=device)
    return instrument, model