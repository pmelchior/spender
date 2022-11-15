import numpy as np
import torch
from torch import nn
from torchinterp1d import Interp1d

class MLP(nn.Sequential):
    """Multi-Layer Perceptron

    A simple implementation with a configurable number of hidden layers and
    activation functions.

    Parameters
    ----------
    n_in: int
        Input dimension
    n_out: int
        Output dimension
    n_hidden: list of int
        Dimensions for every hidden layer
    act: list of callables
        Activation functions after every layer. Needs to have len(n_hidden) + 1
        If `None`, will be set to `LeakyReLU` for every layer.
    dropout: float
        Dropout probability
    """
    def __init__(self,
                 n_in,
                 n_out,
                 n_hidden=(16, 16, 16),
                 act=None,
                 dropout=0):

        if act is None:
            act = [ nn.LeakyReLU(), ] * (len(n_hidden) + 1)
        assert len(act) == len(n_hidden) + 1

        layer = []
        n_ = [n_in, *n_hidden, n_out]
        for i in range(len(n_)-1):
                layer.append(nn.Linear(n_[i], n_[i+1]))
                layer.append(act[i])
                layer.append(nn.Dropout(p=dropout))

        super(MLP, self).__init__(*layer)


class SpeculatorActivation(nn.Module):
    """Activation function from the Speculator paper

    .. math:

        a(\mathbf{x}) = \left[\boldsymbol{\gamma} + (1+e^{-\boldsymbol\beta\odot\mathbf{x}})^{-1}(1-\boldsymbol{\gamma})\right]\odot\mathbf{x}

    Paper: Alsing et al., 2020, ApJS, 249, 5

    Parameters
    ----------
    n_parameter: int
        Number of parameters for the activation function to act on
    plus_one: bool
        Whether to add 1 to the output
    """
    def __init__(self, n_parameter, plus_one=False):
        super().__init__()
        self.plus_one = plus_one
        self.beta = nn.Parameter(torch.randn(n_parameter), requires_grad=True)
        self.gamma = nn.Parameter(torch.randn(n_parameter), requires_grad=True)

    def forward(self, x):
        """Forward method

        Parameters
        ----------
        x: `torch.tensor`

        Returns
        -------
        x': `torch.tensor`, same shape as `x`
        """
        # eq 8 in Alsing+2020
        x = (self.gamma + (1 - self.gamma) * torch.sigmoid(self.beta * x)) * x
        if self.plus_one:
            return x + 1
        return x

class SpectrumEncoder(nn.Module):
    """Spectrum encoder

    Modified version of the encoder by Serrà et al. (2018), which combines a 3 layer CNN
    with a dot-product attention module. This encoder adds a MLP to further compress the
    attended values into a low-dimensional latent space.

    Paper: Serrà et al., https://arxiv.org/abs/1805.03908

    Parameters
    ----------
    instrument: :class:`spender.Instrument`
        Instrument that observed the data
    n_latent: int
        Dimension of latent space
    n_hidden: list of int
        Dimensions for every hidden layer of the :class:`MLP`
    act: list of callables
        Activation functions after every layer. Needs to have len(n_hidden) + 1
        If `None`, will be set to `LeakyReLU` for every layer.
    n_aux: int
        Dimensions of auxiliary inputs for the :class:`MLP`
    dropout: float
        Dropout probability
    """
    def __init__(self,
                 instrument,
                 n_latent,
                 n_hidden=(128, 64, 32),
                 act=None,
                 n_aux=1,
                 dropout=0):

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

        # small MLP to go from CNN features + aux to latents
        if act is None:
            act = [ nn.PReLU(n) for n in n_hidden ]
            # last activation identity to have latents centered around 0
            act.append(nn.Identity())
        self.mlp = MLP(self.n_feature + n_aux, self.n_latent, n_hidden=n_hidden, act=act, dropout=dropout)


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
            act = nn.PReLU(f)
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

    def forward(self, y, aux=None):
        """Forward method

        Parameters
        ----------
        y: `torch.tensor`, shape (N, L)
            Batch of observed spectra
        aux: `torch.tensor`, shape (N, n_aux)
            (optional) Batch of auxiliary inputs to MLP

        Returns
        -------
        s: `torch.tensor`, shape (N, n_latent)
            Batch of latents that encode `spectra`
        """
        # run through CNNs
        h, a = self._downsample(y)
        # softmax attention
        a = self.softmax(a)

        # attach hook to extract backward gradient of a scalar prediction
        # for Grad-FAM (Feature Activation Map)
        if ~self.training and a.requires_grad == True:
            a.register_hook(self._attention_hook)

        # apply attention
        x = torch.sum(h * a, dim=2)

        # redshift depending feature combination to final latents
        if aux is not None and aux is not False:
            x = torch.cat((x, aux), dim=-1)
        x = self.mlp(x)
        return x

    @property
    def n_parameters(self):
        """Number of parameters in this model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _attention_hook(self, grad):
        self._attention_grad = grad

    @property
    def attention_grad(self):
        """Gradient of the attention weights

        Factor to compute the importance of attention for Grad-FAM method.

        Requires a previous `loss.backward` call for any scalar loss function based on
        outputs of this classes `forward` method. This functionality is switched off
        during training.
        """
        if hasattr(self, '_attention_grad'):
            return self._attention_grad
        else:
            return None


class SpectrumDecoder(nn.Module):
    """Spectrum decoder

    Simple :class:`MLP` to create a restframe spectrum from a latent vector,
    followed by explicit redshifting, resampling, and convolution transformations to
    match the observations from a given instrument.

    Parameter
    ---------
    wave_rest: `torch.tensor`
        Restframe wavelengths
    n_latent: int
        Dimension of latent space
    n_hidden: list of int
        Dimensions for every hidden layer of the :class:`MLP`
    act: list of callables
        Activation functions after every layer. Needs to have len(n_hidden) + 1
        If `None`, will be set to :class:`SpeculatorActivation` for every layer.
    dropout: float
        Dropout probability
    """
    def __init__(self,
                 wave_rest,
                 n_latent=5,
                 n_hidden=(64, 256, 1024),
                 act=None,
                 dropout=0,
                ):

        super(SpectrumDecoder, self).__init__()

        if act is None:
            act = [ SpeculatorActivation(n) for n in n_hidden ]
            act.append(SpeculatorActivation(len(wave_rest), plus_one=True))

        self.mlp = MLP(
            n_latent,
            len(wave_rest),
            n_hidden=n_hidden,
            act=act,
            dropout=dropout,
        )

        self.n_latent = n_latent

        # register wavelength tensors on the same device as the entire model
        self.register_buffer('wave_rest', wave_rest)

    def decode(self, s):
        """Decode latents into restframe spectrum

        Parameter
        ---------
        s: `torch.tensor`, shape (N, S)
            Batch of latents

        Returns
        -------
        x: `torch.tensor`, shape (N, L)
            Batch of restframe spectra
        """
        return self.mlp.forward(s)

    def forward(self, s, instrument=None, z=None):
        """Forward method

        Parameter
        ---------
        s: `torch.tensor`, shape (N, S)
            Batch of latents
        instrument: :class:`spender.Instrument`
            Instrument to generate spectrum for
        z: `torch.tensor`, shape (N, 1)
            Redshifts for each spectrum

        Returns
        -------
        y: `torch.tensor`, shape (N, L)
            Batch of spectra at redshift `z` as observed by `instrument`
        """
        # restframe
        spectrum = self.decode(s)
        # observed frame
        if instrument is not None or z is not None:
            spectrum = self.transform(spectrum, instrument=instrument, z=z)
        return spectrum

    def transform(self, x, instrument=None, z=0):
        """Transformations from restframe to observed frame

        Parameter
        ---------
        x: `torch.tensor`, shape (N, S)
            Batch of restframe spectra
        instrument: :class:`spender.Instrument`
            Instrument to generate spectrum for
        z: `torch.tensor`, shape (N, 1)
            Redshifts for each spectrum

        Returns
        -------
        y: `torch.tensor`, shape (N, L)
            Batch of spectra at redshift `z` as observed by `instrument`
        """
        wave_redshifted = (self.wave_rest.unsqueeze(1) * (1 + z)).T

        if instrument in [False, None]:
            wave_obs = self.wave_rest
        else:
            wave_obs = instrument.wave_obs

        spectrum = Interp1d()(wave_redshifted, x, wave_obs)

        # convolve with LSF
        if instrument.lsf is not None:
            spectrum = instrument.lsf(spectrum.unsqueeze(1)).squeeze(1)

        # apply calibration function to observed spectrum
        if instrument is not None and instrument.calibration is not None:
            spectrum = instrument.calibration(wave_obs, spectrum)

        return spectrum

    @property
    def n_parameters(self):
        """Number of parameters in this model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BaseAutoencoder(nn.Module):
    """Base class for spectrum autoencoder

    This class is agnostic about the encoder and decoder architectures. It simply calls
    them in order and computes the loss for the recontruction fidelity.

    The only requirements for the modules is that they have the same latent
    dimensionality, and for the `loss` method the length of the observed spectrum
    vectors need to agree.

    Parameter
    ---------
    encoder: `nn.Module`
        Encoder
    decoder: `nn.Module`
        Decoder
    """
    def __init__(self,
                 encoder,
                 decoder,
                ):

        super(BaseAutoencoder, self).__init__()
        assert encoder.n_latent == decoder.n_latent
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, y, aux=None):
        """Encode from observed spectrum to latents

        Parameters
        ----------
        y: `torch.tensor`, shape (N, L)
            Batch of observed spectra
        aux: `torch.tensor`, shape (N, n_aux)
            (optional) Batch of auxiliary inputs to MLP
        Returns
        -------
        s: `torch.tensor`, shape (N, n_latent)
            Batch of latents that encode `spectra`
        """
        return self.encoder(y, aux=aux)

    def decode(self, s):
        """Decode latents into restframe spectrum

        Parameter
        ---------
        s: `torch.tensor`, shape (N, S)
            Batch of latents

        Returns
        -------
        x: `torch.tensor`, shape (N, L)
            Batch of restframe spectra
        """
        return self.decoder(s)

    def _forward(self, y, instrument=None, z=None, s=None, aux=None):
        if s is None:
            if aux is None and z is not None:
                aux = z.unsqueeze(1)
            s = self.encode(y, aux=aux)
        if instrument is None:
            instrument = self.encoder.instrument

        x = self.decode(s)
        y = self.decoder.transform(x, instrument=instrument, z=z)

        return s, x, y

    def forward(self, y, instrument=None, z=None, s=None, aux=None):
        """Forward method

        Transforms observed spectra into their reconstruction for a given intrument
        and redshift.

        Parameter
        ---------
        y: `torch.tensor`, shape (N, L)
            Batch of observed spectra
        instrument: :class:`spender.Instrument`
            Instrument to generate spectrum for
        z: `torch.tensor`, shape (N, 1)
            Redshifts for each spectrum. When given, `aux` is ignored.
        s: `torch.tensor`, shape (N, S)
            (optional) Batch of latents. When given, encoding is omitted and these
            latents are used instead.
        aux: `torch.tensor`, shape (N, n_aux)
            (optional) Batch of auxiliary inputs to encoder MLP

        Returns
        -------
        y: `torch.tensor`, shape (N, L)
            Batch of spectra at redshift `z` as observed by `instrument`
        """
        s, x, y_ = self._forward(y, instrument=instrument, z=z, s=s, aux=aux)
        return y_

    def loss(self, y, w, instrument=None, z=None, s=None, aux=None, individual=False):
        """Weighted MSE loss

        Parameter
        --------
        y: `torch.tensor`, shape (N, L)
            Batch of observed spectra
        w: `torch.tensor`, shape (N, L)
            Batch of weights for observed spectra
        instrument: :class:`spender.Instrument`
            Instrument to generate spectrum for
        z: `torch.tensor`, shape (N, 1)
            Redshifts for each spectrum. When given, `aux` is ignored.
        s: `torch.tensor`, shape (N, S)
            (optional) Batch of latents. When given, encoding is omitted and these
            latents are used instead.
        aux: `torch.tensor`, shape (N, n_aux)
            (optional) Batch of auxiliary inputs to encoder MLP
        individual: bool
            Whether the loss is computed for each spectrum individually or aggregated

        Returns
        -------
        float or `torch.tensor`, shape (N,) of weighted MSE loss
        """
        y_ = self.forward(y, instrument=instrument, z=z, s=s, aux=aux)
        return self._loss(y, w, y_, individual=individual)

    def _loss(self, y, w, y_, individual=False):
        # loss = total squared deviation in units of variance
        # if the model is identical to observed spectrum (up to the noise),
        # then loss per object = D (number of non-zero bins)

        # to make it to order unity for comparing losses, divide out L (number of bins)
        # instead of D, so that spectra with more valid bins have larger impact
        loss_ind = torch.sum(0.5 * w * (y - y_).pow(2), dim=1) / y.shape[1]

        if individual:
            return loss_ind

        return torch.sum(loss_ind)

    @property
    def n_parameter(self):
        """Number of parameters in this model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def wave_obs(self):
        """Observed wavelengths used by the encoder"""
        return self.encoder.instrument.wave_obs

    @property
    def wave_rest(self):
        """Resframe wavelengths used by the decoder"""
        return self.decoder.wave_rest


class SpectrumAutoencoder(BaseAutoencoder):
    """Concrete implementation of spectrum encoder

    Constructs and uses :class:`SpectrumEncoder` as encoder and :class:`SpectrumDecoder`
    as decoder.

    Parameter
    ---------
    instrument: :class:`spender.Instrument`
        Observing instrument
    wave_rest: `torch.tensor`
        Restframe wavelengths
    n_latent: int
        Dimension of latent space
    n_aux: int
        Dimensions of auxiliary inputs for the encoder :class:`MLP`. Set to 1 to use the
        redshift as auxiliary.
    n_hidden: list of int
        Dimensions for every hidden layer of the decoder :class:`MLP`
    act: list of callables
        Activation functions for the decoder. Needs to have len(n_hidden) + 1
        If `None`, will be set to `LeakyReLU` for every layer.
    """
    def __init__(self,
                 instrument,
                 wave_rest,
                 n_latent=10,
                 n_aux=1,
                 n_hidden=(64, 256, 1024),
                 act=None,
                ):

        encoder = SpectrumEncoder(instrument, n_latent, n_aux=n_aux)

        decoder = SpectrumDecoder(
            wave_rest,
            n_latent,
            n_hidden=n_hidden,
            act=act,
        )

        super(SpectrumAutoencoder, self).__init__(
            encoder,
            decoder,
        )
