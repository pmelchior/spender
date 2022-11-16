import torch
from .model import MLP, SpeculatorActivation, SpectrumEncoder, SpectrumDecoder, SpectrumAutoencoder
from .instrument import Instrument, LSF


def load_model(filename, instrument, device=None):
    """Load models from file

    Parameter
    ---------
    filename: str
        Path to file which contains the torch state dictionary
    instrument: :class:`spender.Instrument`
        Instrument to generate spectrum for
    device: `torch.Device`
        Device to load model structure into

    Returns
    -------
    model: `torch.nn.Module`
        The default :class`SpectrumAutoencoder` model loaded from file
    loss: `torch.tensor`
        Traning and validation loss for this model
    """
    assert isinstance(instrument, Instrument)
    model_struct = torch.load(filename, map_location=device)

    # check if LSF is contained in model_struct
    try:
        kernel = model_struct['model']['encoder.instrument.lsf.weight'].flatten()
        lsf = LSF(kernel)
        instrument.lsf = lsf
    except KeyError:
        pass

    wave_rest = model_struct['model']['decoder.wave_rest']
    n_latent = model_struct['model']['decoder.mlp.0.weight'].shape[1]

    model = SpectrumAutoencoder(
                instrument,
                wave_rest,
                n_latent=n_latent,
    )

    # backwards compat: encoder.mlp instead of encoder.mlp.mlp
    if 'encoder.mlp.mlp.0.weight' in model_struct['model'].keys():
        from collections import OrderedDict
        model_struct['model'] = OrderedDict([(k.replace('mlp.mlp', 'mlp'), v) for k, v in model_struct['model'].items()])
    # backwards compat: add instrument to encoder
    try:
        model.load_state_dict(model_struct['model'], strict=False)
    except RuntimeError:
        model_struct['model']['encoder.instrument.wave_obs']= instrument.wave_obs
        model_struct['model']['encoder.instrument.skyline_mask']= instrument._skyline_mask
        model.load_state_dict(model_struct['model'], strict=False)

    loss = torch.tensor(model_struct['losses'])
    return model, loss
