dependencies = ['torch', 'os']

from spender import load_model as _load_model
from spender.data.sdss import SDSS as _SDSS

hub_server = "https://hub.pmelchior.net/"

def _sdss_model(url, **kwargs):
    instrument = _SDSS()
    model = _load_model(url, instrument, **kwargs)
    return instrument, model

def sdss_I(**kwargs):
    """Spectrum Autoencoder model for the SDSS main galaxy sample.

    See Spender paper I for details:
        Melchior et al. (2022): arXiv:2211.07890

    """
    url = hub_server + "spender.sdss.paperI-08798cbc.pt"
    return _sdss_model(url, **kwargs)

def sdss_I_superres(**kwargs):
    """Spectrum Autoencoder model for the SDSS main galaxy sample.

    See Spender paper I (section 4.2) for details:
        Melchior et al. (2022): arXiv:2211.07890

    """
    url = hub_server + "spender.sdss.paperI.superres-0403266c.pt"
    return _sdss_model(url, **kwargs)

def sdss_II(**kwargs):
    """Spectrum Autoencoder model for the SDSS main galaxy sample.

    This model is trained with fidelity, similarity, and consistency losses
    to provide a redshift-invariant latent space.

    See spender papers for details:
        Liang at al. (2023): arXiv:2302.02496
        Melchior et al. (2022): arXiv:2211.07890
    """
    url = hub_server + "spender.sdss.paperII-c273bb69.pt"
    return _sdss_model(url, **kwargs)
