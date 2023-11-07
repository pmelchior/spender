dependencies = ['torch', 'os']

from spender import load_model as _load_model
from spender import load_flow_model as _load_flow_model

hub_server = "https://hub.pmelchior.net/"

def _sdss_model(url, **kwargs):
    from spender.data.sdss import SDSS
    instrument = SDSS()
    model = _load_model(url, instrument, **kwargs)
    return instrument, model

def _desi_model(url, **kwargs):
    from spender.data.desi import DESI
    instrument = DESI()
    model = _load_model(url, instrument, **kwargs)
    return instrument, model

def sdss_I(**kwargs):
    """Spectrum Autoencoder model for the SDSS main galaxy sample.

    This model has S=10 latents and a restframe resolution R=5881 for a maximum redshift of z_max=0.5.

    See Spender paper I for details:
        Melchior et al. (2022): arXiv:2211.07890

    """
    url = hub_server + "spender.sdss.paperI-08798cbc.pt"
    return _sdss_model(url, **kwargs)

def sdss_I_superres(**kwargs):
    """Spectrum Autoencoder super-resolution model for the SDSS main galaxy sample.

    This model has S=8 latents and a restframe resolution R=11762 for a maximum redshift of z_max=0.5.

    See Spender paper I (section 4.2) for details:
        Melchior et al. (2022): arXiv:2211.07890

    """
    url = hub_server + "spender.sdss.paperI.superres-0403266c.pt"
    return _sdss_model(url, **kwargs)

def sdss_II(**kwargs):
    """Spectrum Autoencoder model for the SDSS main galaxy sample.

    This model has S=6 latents and a restframe resolution R=7000 for a maximum redshift of z_max=0.5.
    It has been trained with fidelity, similarity, and consistency losses to provide a redshift-invariant latent space.

    See spender papers for details:
        Liang at al. (2023): arXiv:2302.02496
        Melchior et al. (2022): arXiv:2211.07890
    """
    url = hub_server + "spender.sdss.paperII-c273bb69.pt"
    return _sdss_model(url, **kwargs)

def desi_edr_galaxy(**kwargs):
    """Spectrum Autoencoder model for the DESI EDR Bright Galaxy Survey sample.

    This model has S=6 latents and a restframe resolution R=9780 for a maximum redshift of z_max=0.8.
    It has been trained with fidelity, similarity, and consistency losses to provide a redshift-invariant latent space.

    See spender papers for details:
        Liang at al. (2023b): arXiv:2307.07664
        Liang at al. (2023a): arXiv:2302.02496
        Melchior et al. (2022): arXiv:2211.07890
    """
    url = hub_server + "spender.desi-edr.galaxyae-b9bc8d12.pt"
    return _desi_model(url, **kwargs)

def desi_edr_star(**kwargs):
    """Spectrum Autoencoder model for the DESI EDR Milky Way Survey sample.

    This model has S=6 latents and a restframe resolution R=7864 for a maximum redshift of z_max=0.005.
    It has been trained with fidelity, and similarity but not consistency loss.

    See spender papers for details:
        Liang at al. (2023b): arXiv:2307.07664
        Liang at al. (2023a): arXiv:2302.02496
        Melchior et al. (2022): arXiv:2211.07890
    """
    url = hub_server + "spender.desi-edr.starae-2e33f4e5.pt"
    return _desi_model(url, **kwargs)

def desi_edr_galaxy_flow(**kwargs):
    """Normalizing flow model for the DESI EDR BGS spender latent space.

    See spender papers for details:
        Liang at al. (2023b): arXiv:2307.07664
    """
    url = hub_server + "spender.desi-edr.galaxyflow-b71f8966.pt"
    n_latent = 6
    return _load_flow_model(url, n_latent, **kwargs)

def desi_edr_star_flow(**kwargs):
    """Normalizing flow model for the DESI EDR MWS spender latent space.

    See spender papers for details:
        Liang at al. (2023b): arXiv:2307.07664
    """
    url = hub_server + "spender.desi-edr.starflow-a6ff6fcf.pt"
    n_latent = 6
    return _load_flow_model(url, n_latent, **kwargs)
