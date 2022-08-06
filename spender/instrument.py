import torch
from torch import nn

from .util import skylines_mask

class Instrument(nn.Module):
    def __init__(self,
                 wave_obs,
                 lsf=None,
                 calibration=None,
                 name=None,
                ):

        super(Instrument, self).__init__()

        self.lsf = lsf
        self.calibration = calibration
        self.name = name

        # register wavelength tensors on the same device as the entire model
        self.register_buffer('wave_obs', wave_obs)
        self.register_buffer('skyline_mask', skylines_mask(wave_obs))

    def set_lsf(self, lsf_kernel, wave_kernel, wave_rest, requires_grad=False):
        # resample in wave_rest pixels
        h = (wave_rest.max() - wave_rest.min()) / len(wave_rest)
        wave_kernel_rest = torch.arange(wave_kernel.min().floor(), wave_kernel.max().ceil(), h)
        # make sure kernel has odd length for padding 'same'
        if len(wave_kernel_rest) % 2 == 0:
            wave_kernel_rest = torch.cat((wave_kernel_rest, torch.tensor([wave_kernel_rest.max() + h,])), 0)
        lsf_kernel_rest = Interp1d()(wave_kernel, lsf_kernel, wave_kernel_rest)
        lsf_kernel_rest /= lsf_kernel_rest.sum()

        # construct conv1d layer
        self.lsf = nn.Conv1d(1, 1, len(lsf_kernel_rest), bias=False, padding='same')
        self.lsf.weight = nn.Parameter(lsf_kernel_rest.flip(0).reshape(1,1,-1), requires_grad=requires_grad)

def get_instrument(name, lsf=None, calibration=None):
    assert name in ["SDSS", "BOSS"]
    if name == "SDSS":
        lower, upper = 3.578, 3.97
    elif name == "BOSS":
        lower, upper = 3.549, 4.0175
    wave_obs = 10**torch.arange(lower, upper, 0.0001)
    return Instrument(wave_obs, lsf=lsf, calibration=calibration, name=name)
