import torch, os
import numpy as np
from torch import nn

class BaseInstrument(nn.Module):
    def __init__(self,
                 wave_obs,
                 lsf=None,
                 calibration=None,
                ):

        super(BaseInstrument, self).__init__()

        self.lsf = lsf
        self.calibration = calibration

        # register wavelength tensors on the same device as the entire model
        self.register_buffer('wave_obs', wave_obs)
        self.register_buffer('skyline_mask', skylines_mask(wave_obs))

    @property
    def name(self):
        return self.__class__.__name__

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


def skylines_mask(waves, intensity_limit=2, radii=5):
    this_dir, this_filename = os.path.split(__file__)
    filename = os.path.join(this_dir, "data", "sky-lines.txt")
    f=open(filename,"r")
    content = f.readlines()
    f.close()

    skylines = [[10*float(line.split()[0]),float(line.split()[1])] for line in content if not line[0]=="#" ]
    skylines = np.array(skylines)

    n_lines = 0
    mask = ~(waves>0)

    for line in skylines:
        line_pos, intensity = line
        if line_pos>waves.max():continue
        if intensity<intensity_limit:continue
        n_lines += 1
        mask[(waves<(line_pos+radii))*(waves>(line_pos-radii))] = True

    non_zero = torch.count_nonzero(mask)
    return mask

# allow registry of new instruments
# see https://effectivepython.com/2015/02/02/register-class-existence-with-metaclasses
instrument_register = {}

def register_class(target_class):
    instrument_register[target_class.__name__] = target_class

class Meta(type):
    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        # remove those that are directly derived from the base class
        if BaseInstrument not in bases:
            register_class(cls)
        return cls

class Instrument(BaseInstrument, metaclass=Meta):
    pass
