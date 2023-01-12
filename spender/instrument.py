import os

import numpy as np
import torch
from torch import nn


class BaseInstrument(nn.Module):
    """Base class for instruments

    Container for wavelength vector, LSF and calibration functions.

    CAUTION:
    Don't base concrete implementations on this class, use :class:`Instrument` instead!

    Parameters
    ----------
    wave_obs: `torch.tensor`
        Observed wavelengths
    lsf: :class:`LSF`
        (optional) Line spread function model
    calibration: callable
        (optional) function to calibrate the observed spectrum
    """

    def __init__(
        self,
        wave_obs,
        lsf=None,
        calibration=None,
    ):

        super(BaseInstrument, self).__init__()

        self.calibration = calibration
        if lsf is not None:
            assert isinstance(lsf, (LSF, torch.Tensor))
            if isinstance(lsf, LSF):
                self.lsf = lsf
            else:
                self.lsf = LSF(lsf)
        else:
            self.lsf = None

        # register wavelength tensors on the same device as the entire model
        self.register_buffer("wave_obs", wave_obs)

    @property
    def name(self):
        return self.__class__.__name__


class LSF(nn.Conv1d):
    def __init__(self, kernel, requires_grad=True):
        super(LSF, self).__init__(1, 1, len(kernel), bias=False, padding="same")
        # if LSF should be fit, set `requires_grad=True`
        self.weight = nn.Parameter(
            kernel.flip(0).reshape(1, 1, -1), requires_grad=requires_grad
        )

    def forward(self, x):
        # convolution with flux preservation
        return super(LSF, self).forward(x) / self.weight.sum()


def get_skyline_mask(wave_obs, min_intensity=2, mask_size=5):
    """Return vector that masks the major skylines

    For ever line in the skyline list in the file `data/sky-lines.txt` that is brighter
    than a threshold, this method creates a mask whose size scales logarithmically with
    line brightness.

    Parameter
    ---------
    wave_obs: `torch.tensor`
        Observed wavelengths
    min_intensity: float
        Intensity threshold
    mask_size: float
        Number of spectral elements to mask on either side of the line. This number
        is the minmum size for lines with `min_intensity`.
    Returns
    -------
    mask, `torch.tensor` of dtype `bool` with same shape as `wave_obs`
    """
    this_dir, this_filename = os.path.split(__file__)
    filename = os.path.join(this_dir, "data", "sky-lines.txt")
    skylines = np.genfromtxt(
        filename,
        names=["wavelength", "intensity", "name", "status"],
        dtype=None,
        encoding=None,
    )
    # wavelength in nm, need A
    skylines["wavelength"] *= 10

    mask = torch.zeros(len(wave_obs), dtype=torch.bool)
    for line in skylines[skylines["intensity"] > min_intensity]:
        # increase masking area with intensity
        mask_size_ = mask_size * (1 + np.log10(line["intensity"] / min_intensity))
        mask |= (wave_obs - line["wavelength"]).abs() < mask_size_
    return mask


# allow registry of new instruments
# see https://effectivepython.com/2015/02/02/register-class-existence-with-metaclasses
instrument_register = {}


def register_class(target_class):
    instrument_register[target_class.__name__] = target_class


class Meta(type):
    """Meta class to enable registration of instruments"""

    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        # remove those that are directly derived from the base class
        if BaseInstrument not in bases:
            register_class(cls)
        return cls


class Instrument(BaseInstrument, metaclass=Meta):
    """Instrument class

    Container for wavelength vector, LSF and calibration functions.

    See `spender.instrument.instrument_register` for all known classes that derive from
    :class:`Instrument`.
    """

    pass
