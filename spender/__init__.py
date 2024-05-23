import torch
import torch.hub
from . import hub
from .flow import NeuralDensityEstimator
from .instrument import LSF, Instrument
from .model import (MLP, SpectrumAutoencoder, SpectrumDecoder, SpectrumEncoder,
                    SpeculatorActivation)


def load_model(filename, instrument, **kwargs):
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
        Training and validation loss for this model
    """
    assert isinstance(instrument, Instrument)

    # load model_struct from hub if url is given
    if filename[:4].lower() == "http":
        kwargs.pop("check_hash", True) # remove check_hash
        model_struct = torch.hub.load_state_dict_from_url(filename, check_hash=True, **kwargs)
    else:
        model_struct = torch.load(filename, **kwargs)

    # check if model_struct['model'] is a single model or list
    try:
        model_struct["model"]["decoder.wave_rest"]
    except TypeError:
        model_struct["model"] = model_struct["model"][0]


    # check if LSF is contained in model_struct
    try:
        kernel = model_struct["model"]["encoder.instrument.lsf.weight"].flatten()
        lsf = LSF(kernel)
        instrument.lsf = lsf
    except KeyError:
        pass

    wave_rest = model_struct["model"]["decoder.wave_rest"]
    n_latent = model_struct["model"]["decoder.mlp.0.weight"].shape[1]
    
    # activation function for decoder MLP: by default Speculator, but older models use LeakyReLU
    act = None if "decoder.mlp.1.beta" in model_struct["model"].keys() else [torch.nn.LeakyReLU(), ] * 4

    model = SpectrumAutoencoder(
        instrument,
        wave_rest,
        n_latent=n_latent,
        act=act,
    )

    # backwards compat: encoder.mlp instead of encoder.mlp.mlp
    if "encoder.mlp.mlp.0.weight" in model_struct["model"].keys():
        from collections import OrderedDict

        model_struct["model"] = OrderedDict(
            [(k.replace("mlp.mlp", "mlp"), v) for k, v in model_struct["model"].items()]
        )

    # backwards compat: remove z (=last) input from encoder mlp
    if model_struct['model']['encoder.mlp.0.weight'].shape[1] == 257:
        model_struct['model']['encoder.mlp.0.weight'] = model_struct['model']['encoder.mlp.0.weight'][:,:-1]

    # backwards compat: add instrument to encoder
    try:
        model.load_state_dict(model_struct["model"], strict=False)
    except RuntimeError:
        model_struct["model"]["encoder.instrument.wave_obs"] = instrument.wave_obs
        model_struct["model"][
            "encoder.instrument.skyline_mask"
        ] = instrument._skyline_mask
        model.load_state_dict(model_struct["model"], strict=False)

    return model

def load_flow_model(filename, n_latent, **kwargs):
    nde = NeuralDensityEstimator(
        dim=n_latent,
        initial_pos={"bounds": [[0, 0]] * n_latent, "std": [0.05] * n_latent},
    )

    # load model_struct from hub if url is given
    if filename[:4].lower() == "http":
        kwargs.pop("check_hash", True)  # remove check_hash
        model_struct = torch.hub.load_state_dict_from_url(filename, check_hash=True, **kwargs)
    else:
        model_struct = torch.load(filename, **kwargs)
    nde.load_state_dict(model_struct)
    return nde
