import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from torchinterp1d import Interp1d


def get_norm(x):
    # simple median as robust mean across the spectrum
    norm = np.median(x, axis=1)
    return norm

def permute_indices(length,n_redundant=1):
    wrap_indices = torch.arange(length).repeat(n_redundant)
    rand_permut = wrap_indices[torch.randperm(length*n_redundant)]
    return rand_permut

# adapted from https://github.com/sigeisler/reliable_gnn_via_robust_aggregation/
def _distance_matrix(x, eps_factor=1e2):
    """Naive dense distance matrix calculation.

    Parameters
    ----------
    x : torch.Tensor
        Dense [n, d] or [n, k, d] tensor containing the node attributes/embeddings.
    eps_factor : [type], optional
        Factor to be multiplied by `torch.finfo(x.dtype).eps` for "safe" sqrt, by default 1e2.
    Returns
    -------
    torch.Tensor
        [n, n] or [n, k, k] distance matrix.
    """
    x_norm = (x ** 2).sum(-1).unsqueeze(-1)
    x_norm_t = x_norm.transpose(-2,-1)
    squared = x_norm + x_norm_t - (2 * (x @ x.transpose(-2, -1)))
    # For "save" sqrt
    eps = eps_factor * torch.finfo(x.dtype).eps
    return torch.sqrt(torch.abs(squared) + eps)

def soft_weighted_medoid(x, temperature=1.0):
    """A weighted Medoid aggregation.

    Parameters
    ----------
    x : torch.Tensor
        Dense [n, d] tensor containing the node attributes/embeddings.
    temperature : float, optional
        Temperature for the argmin approximation by softmax, by default 1.0
    Returns
    -------
    torch.Tensor
        The new embeddings [n, d].
    """
    # Geisler 2020, eqs 2-3
    distances = _distance_matrix(x)
    s = F.softmax(-distances.sum(dim=-1) / temperature, dim=-1)
    return (s.unsqueeze(-1) * x).sum(dim=-2)


# read in SDSS theta and spectra
def load_data(path, which=None, device=None):

    assert which in [None, "train", "valid", "test"]

    data = np.load(path)

    # define subsets
    N = len(data['spectra'])
    n_train, n_valid = int(N*0.7), int(N*0.15)
    n_test = N - n_train - n_valid

    sl = {None: slice(None),
          "train": slice(0, n_train),
          "valid": slice(n_train, n_train+n_valid),
          "test": slice(n_train+n_valid, n_train+n_valid+n_test),
         }
    sl = sl[which]

    y = data['spectra'][sl]
    wave = 10**data['wave']
    z = data['z'][sl]
    zerr = np.maximum(data['zerr'][sl], 1e-6) # one case has zerr=0, but looks otherwise OK
    ivar = data['ivar'][sl]
    mask = data['mask'][sl]

    # SDSS IDs
    id = data['plate'][sl], data['mjd'][sl], data['fiber'][sl]
    id = [f"{plate}-{mjd}-{fiber}" for plate, mjd, fiber in np.array(id).T]

    # get normalization
    norm = get_norm(y)

    w = ivar * ~mask * (norm**2)[:,None]
    sel = np.any(w > 0, axis=1)   # remove all spectra that have all zero weights
    sel &= (norm > 0) & (z < 0.5)   # plus noisy ones and redshift outliers

    w = np.maximum(w, 1e-6)       # avoid zero weights for logL normalization
    w = w[sel]
    y = y[sel] / norm[sel, None]
    z = z[sel]
    zerr = zerr[sel]
    norm = norm[sel]
    id = np.array(id)[sel]


    print (f"Loading {len(y)} spectra (which = {which})")

    y = torch.tensor(y, dtype=torch.float32, device=device)
    w = torch.tensor(w, dtype=torch.float32, device=device)
    z = torch.tensor(z, dtype=torch.float32, device=device)
    zerr = torch.tensor(zerr, dtype=torch.float32, device=device)

    return {"wave": wave,
            "y": y,
            "w": w,
            "z": z,
            "zerr": zerr,
            "norm": norm,
            "id": id,
            "N": len(y),
           }

def load_model(fileroot):
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = None

    path = f'{fileroot}.pt'
    model = torch.load(path, map_location=device)
    if type(model)==list or type(model)==tuple:
        [m.eval() for m in model]
    else: model.eval()
    path = f'{fileroot}.losses.npy'
    loss = np.load(path)

    print (f"model {fileroot}: iterations {len(loss)}, final loss: {loss[-1]}")
    return model, loss

def load_models(label, n_config):
    models, losses = {}, {}
    best_model, best_loss = 0, np.inf
    for i in range(n_config):
        try:
            label_ = label + f".{i}"
            model, loss = load_model(label_)
            models[i] = model
            losses[i] = loss
            if loss[-1][1] < best_loss:
                best_loss = loss[-1][1]
                best_model = i
        except FileNotFoundError:
            pass

    return models, losses, best_model


# Redshift distribution from histogram
class RedshiftPrior(nn.Module):
    def __init__(self,
                 zbins,
                 pz,
                ):

        super(RedshiftPrior, self).__init__()

        # register tensors on the same dives as the entire model
        self.register_buffer('zbins', zbins)

        # extend counts to provide an "empty" bin for extreme values outside of the histogram
        pz_ = torch.empty(len(zbins))
        pz_[0] = 1e-16
        pz_[1:] = pz / pz.sum()
        self.register_buffer('pz', pz_)

    def forward(self, z):
        loc = torch.argmin((z.unsqueeze(1) > self.zbins).float(), axis=1)
        return self.pz[loc]

    def log_prob(self, z):
        return torch.log(self.forward(z))

    def sample(self, size=1):
        idx = self.pz.multinomial(num_samples=size, replacement=True) - 1
        u = torch.rand(size)
        z_ = self.zbins[idx] + u * (self.zbins[idx + 1] - self.zbins[idx])
        return z_


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
    def __init__(self, n_latent, n_hidden=(128, 64, 32), dropout=0):
        
        super(SpectrumEncoder, self).__init__()
        self.n_latent = n_latent
        
        # spectrum convolutions
        filters = [128, 256, 512]
        sizes = [5, 11, 21]
        self.conv1, self.conv2, self.conv3 = self._conv_blocks(filters, sizes, dropout=dropout)

        # weight convolutions: only need attention part, so 1/2 of all channels
        filters = [f // 2 for f in filters ]
        self.conv1w, self.conv2w, self.conv3w = self._conv_blocks(filters, sizes, dropout=dropout)
        self.n_feature = filters[-1]

        # pools and softmax work for spectra and weights
        self.pool1, self.pool2 = tuple(nn.MaxPool1d(s, padding=s//2) for s in sizes[:2])
        self.softmax = nn.Softmax(dim=-1)

        # small MLP to go from CNN features + redshift to latents
        self.mlp = MLP(self.n_feature + 1, self.n_latent, n_hidden=n_hidden, dropout=dropout)
        

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

    def forward(self, x, w=None, z=None):
        N, D = x.shape
        # spectrum compression
        x = x.unsqueeze(1)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.conv3(x)
        C = x.shape[1] // 2
        h, a = torch.split(x, [C, C], dim=1)

        # weight compression
        if w is None:
            aw = 1
        else:
            tiny = 1e-6
            w = torch.log(w.unsqueeze(1) + tiny) # reduce dynamic range of w
            w = self.pool1(self.conv1w(w))
            w = self.pool2(self.conv2w(w))
            aw = self.conv3w(w)

        # modulate signal attention with weight attention
        a = self.softmax(a * aw)
        # apply attention
        x = torch.sum(h * a, dim=2)
        
        # redshift depending feature combination to final latents
        if z is None:
            z = torch.zeros(len(x))
        assert len(x) == len(z)
        x = torch.concat((x, z.unsqueeze(-1)), dim=-1)
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
                 dropout=0):

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
        wave_obs = self.wave_rest

        if instrument is not None:
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
                ):

        super(BaseAutoencoder, self).__init__()
        assert encoder.n_latent == decoder.n_latent
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x, w=None, z=0):
        return self.encoder(x, w=w, z=z)

    def decode(self, x):
        return self.decoder(x)

    def _forward(self, x, w=None, instrument=None, z=None):
        s = self.encode(x, w=w, z=z)
        spectrum_restframe = self.decode(s)
        spectrum_observed = self.decoder.transform(spectrum_restframe, instrument=instrument, z=z)
        return s, spectrum_restframe, spectrum_observed

    def forward(self, x, w=None, instrument=None, z=None):
        s, spectrum_restframe, spectrum_observed = self._forward(x, w=w,  instrument=instrument, z=z)
        return spectrum_observed

    def loss(self, x, w, instrument=None, z=None, individual=False):
        spectrum_observed = self.forward(x, w=w, instrument=instrument, z=z)
        return self._loss(x, w, spectrum_observed, individual=individual)

    def _loss(self, x, w, spectrum_observed, individual=False):
        # loss = average squared deviation in units of variance
        # if the model is identical to observed spectrum (up to the noise)
        # in every unmasked bin, then loss = 1 per object
        D = (w > 0).sum(dim=1)
        loss_ind = torch.sum(0.5 * w * (x - spectrum_observed).pow(2), dim=1)

        if individual:
            return loss_ind / D

        return torch.sum(loss_ind / D)

    @property
    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SpectrumAutoencoder(BaseAutoencoder):
    def __init__(self,
                 wave_rest,
                 n_latent=10,
                 n_hidden=(64, 256, 1024),
                 K=16,
                 n_redundant=2,
                 T=0.5,
                 dropout=0,
                ):

        encoder = SpectrumEncoder(n_latent, dropout=dropout)

        decoder = SpectrumDecoder(
            wave_rest,
            n_latent,
            n_hidden=n_hidden,
            dropout=dropout,
        )

        super(SpectrumAutoencoder, self).__init__(
            encoder,
            decoder,
        )


class Instrument(nn.Module):
    def __init__(self,
                 wave_obs,
                 lsf=None,
                 calibration=None,
                ):

        super(Instrument, self).__init__()

        # register wavelength tensors on the same device as the entire model
        self.register_buffer('wave_obs', wave_obs)

        self.lsf = lsf
        self.calibration = calibration

    def set_lsf(self, lsf_kernel, wave_kernel, wave_rest, requires_grad=False):
        # resample in wave_rest pixels
        h = (wave_rest.max() - wave_rest.min()) / len(wave_rest)
        wave_kernel_rest = torch.arange(wave_kernel.min().floor(), wave_kernel.max().ceil(), h)
        # make sure kernel has odd length for padding 'same'
        if len(wave_kernel_rest) % 2 == 0:
            wave_kernel_rest = torch.concat((wave_kernel_rest, torch.tensor([wave_kernel_rest.max() + h,])), 0)
        lsf_kernel_rest = Interp1d()(wave_kernel, lsf_kernel, wave_kernel_rest)
        lsf_kernel_rest /= lsf_kernel_rest.sum()

        # construct conv1d layer
        self.lsf = nn.Conv1d(1, 1, len(lsf_kernel_rest), bias=False, padding='same')
        self.lsf.weight = nn.Parameter(lsf_kernel_rest.flip(0).reshape(1,1,-1), requires_grad=requires_grad)
