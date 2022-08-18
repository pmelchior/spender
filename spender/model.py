import numpy as np
import torch
from torch import nn
from torchinterp1d import Interp1d

#### Simple MLP ####
class MLP(nn.Module):
    def __init__(self,
                 n_in,
                 n_out,
                 n_hidden=(16, 16, 16),
                 dropout=0):
        super(MLP, self).__init__()

        layer = []
        n_ = [n_in, *n_hidden, n_out]
        for i in range(len(n_)-1):
                layer.append(nn.Linear(n_[i], n_[i+1]))
                layer.append(nn.LeakyReLU())
                layer.append(nn.Dropout(p=dropout))
        self.mlp = nn.Sequential(*layer)

    def forward(self, x):
        return self.mlp(x)


#### Spectrum encoder    ####
#### based on Serra 2018 ####
#### with robust feature combination from Geisler 2020 ####
class SpectrumEncoder(nn.Module):
    def __init__(self, instrument, n_latent, n_hidden=(128, 64, 32), n_aux=1, dropout=0):

        super(SpectrumEncoder, self).__init__()
        self.instrument = instrument
        self.n_latent = n_latent
        self.n_aux = n_aux

        filters = [128, 256, 512]
        sizes = [5, 11, 21]
        self.conv1, self.conv2, self.conv3 = self._conv_blocks(filters, sizes, dropout=dropout)
        self.n_feature = filters[-1] // 2

        # pools and softmax work for spectra and weights
        self.pool1, self.pool2 = tuple(nn.MaxPool1d(s, padding=s//2) for s in sizes[:2])
        self.softmax = nn.Softmax(dim=-1)

        # small MLP to go from CNN features + redshift to latents
        self.mlp = MLP(self.n_feature + n_aux, self.n_latent, n_hidden=n_hidden, dropout=dropout)


    def _conv_blocks(self, filters, sizes, dropout=0):
        convs = []
        for i in range(len(filters)):
            f_in = 1 if i == 0 else filters[i-1]
            f = filters[i]
            s = sizes[i]
            p = s // 2
            conv = nn.Conv1d(in_channels=f_in,
                             out_channels=f,
                             kernel_size=s,
                             padding=p,
                            )
            norm = nn.InstanceNorm1d(f)
            act = nn.PReLU(num_parameters=f)
            drop = nn.Dropout(p=dropout)
            convs.append(nn.Sequential(conv, norm, act, drop))
        return tuple(convs)

    def _downsample(self, x):
        # compression
        x = x.unsqueeze(1)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.conv3(x)
        C = x.shape[1] // 2
        # split half channels into attention value and key
        h, a = torch.split(x, [C, C], dim=1)

        return h, a

    def forward(self, x, aux=None):
        # run through CNNs
        h, a = self._downsample(x)
        # softmax attention
        a = self.softmax(a)
        # apply attention
        x = torch.sum(h * a, dim=2)

        # redshift depending feature combination to final latents
        if aux is not None and aux is not False:
            x = torch.cat((x, aux), dim=-1)
        x = self.mlp(x)
        return x

    @property
    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


#### Spectrum decoder ####
#### Simple MLP but with explicit redshift and instrument path ####
class SpectrumDecoder(MLP):
    def __init__(self,
                 wave_rest,
                 n_latent=5,
                 n_hidden=(64, 256, 1024),
                 dropout=0,
                ):

        super(SpectrumDecoder, self).__init__(
            n_latent,
            len(wave_rest),
            n_hidden=n_hidden,
            dropout=dropout,
            )

        self.n_latent = n_latent

        # register wavelength tensors on the same device as the entire model
        self.register_buffer('wave_rest', wave_rest)

    def decode(self, s):
        return super().forward(s)

    def forward(self, s, instrument=None, z=None):
        # restframe
        spectrum = self.decode(s)
        # observed frame
        if instrument is not None or z is not None:
            spectrum = self.transform(spectrum, instrument=instrument, z=z)
        return spectrum

    def transform(self, spectrum_restframe, instrument=None, z=0):
        wave_redshifted = (self.wave_rest.unsqueeze(1) * (1 + z)).T

        if instrument in [False, None]:
            wave_obs = self.wave_rest
        else:
            wave_obs = instrument.wave_obs

            # convolve with LSF
            if instrument.lsf is not None:
                spectrum_restframe = instrument.lsf(spectrum_restframe.unsqueeze(1)).squeeze(1)
        spectrum = Interp1d()(wave_redshifted, spectrum_restframe, wave_obs)

        # apply calibration function to observed spectrum
        if instrument is not None and instrument.calibration is not None:
            spectrum = instrument.calibration(wave_obs, spectrum)

        return spectrum

    @property
    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Combine spectrum encoder and decoder
class BaseAutoencoder(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 normalize=False,
                ):

        super(BaseAutoencoder, self).__init__()
        assert encoder.n_latent == decoder.n_latent
        self.encoder = encoder
        self.decoder = decoder
        self.normalize = normalize

    def encode(self, x, aux=None):
        return self.encoder(x, aux=aux)

    def decode(self, x):
        return self.decoder(x)

    def _forward(self, x, w=None, instrument=None, z=None, s=None, aux=None):
        if s is None:
            if aux is None and z is not None:
                aux = z.unsqueeze(1)
            s = self.encode(x, aux=aux)
        if instrument is None:
            instrument = self.encoder.instrument

        spectrum_restframe = self.decode(s)
        spectrum_observed = self.decoder.transform(spectrum_restframe, instrument=instrument, z=z)

        if self.normalize:
            c = self._normalization(x, spectrum_observed, w=w)
            spectrum_observed = spectrum_observed * c
            spectrum_restframe = spectrum_restframe * c

        return s, spectrum_restframe, spectrum_observed

    def forward(self, x, w=None, instrument=None, z=None, s=None, aux=None):
        s, spectrum_restframe, spectrum_observed = self._forward(x, w=w, instrument=instrument, z=z, s=s, aux=aux)
        return spectrum_observed

    def loss(self, x, w, instrument=None, z=None, s=None, aux=None, individual=False):
        spectrum_observed = self.forward(x, instrument=instrument, z=z, s=s, aux=aux)
        return self._loss(x, w, spectrum_observed, individual=individual)

    def _loss(self, x, w, spectrum_observed, individual=False):
        # loss = total squared deviation in units of variance
        # if the model is identical to observed spectrum (up to the noise),
        # then loss per object = D (number of non-zero bins)

        # to make it to order unity for comparing losses, divide out L (number of bins)
        # instead of D, so that spectra with more valid bins have larger impact
        loss_ind = torch.sum(0.5 * w * (x - spectrum_observed).pow(2), dim=1) / x.shape[1]

        if individual:
            return loss_ind

        return torch.sum(loss_ind)

    def _normalization(self, x, m, w=None):
        # apply constant factor c that minimizes (c*m - x)^2
        if w is None:
            w = 1
        mw = m*w
        c = (mw * x).sum(dim=-1) / (mw * m).sum(dim=-1)
        return c.unsqueeze(-1)

    @property
    def n_parameter(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def wave_obs(self):
        return self.encoder.instrument.wave_obs

    @property
    def wave_rest(self):
        return self.decoder.wave_rest

class SpectrumAutoencoder(BaseAutoencoder):
    def __init__(self,
                 instrument,
                 wave_rest,
                 n_latent=10,
                 n_aux=1,
                 n_hidden=(64, 256, 1024),
                 normalize=False,
                ):

        encoder = SpectrumEncoder(instrument, n_latent, n_aux=n_aux)

        decoder = SpectrumDecoder(
            wave_rest,
            n_latent,
            n_hidden=n_hidden,
        )

        super(SpectrumAutoencoder, self).__init__(
            encoder,
            decoder,
            normalize=normalize,
        )
