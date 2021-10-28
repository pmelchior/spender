import numpy as np
import torch
from torch import nn

from torchinterp1d import Interp1d


def get_norm(x):
    # simple median as robust mean across the spectrum
    norm = np.median(x, axis=1)
    return norm

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
    model.eval()
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


# simple MLP for encoder and decoder
class Encoder(nn.Module):
    def __init__(self,
                 n_feature,
                 n_latent,
                 n_hidden=[128, 64, 32],
                 dropout=0):
        super(Encoder, self).__init__()
        self.n_latent = n_latent
        self.n_feature = n_feature

        layer = []
        n_hidden = [n_feature, *n_hidden, n_latent]
        for i in range(len(n_hidden)-1):
            layer.append(nn.Linear(n_hidden[i], n_hidden[i+1]))
            layer.append(nn.LeakyReLU())
            layer.append(nn.Dropout(p=dropout))
        self.encoder = nn.Sequential(*layer)
    
    def forward(self, x):
        x = self.encoder(x)
        return x

    
class Decoder(nn.Module):
    def __init__(self,
                 n_feature,
                 n_latent,
                 n_hidden=[128, 64, 32],
                 dropout=0):
        super(Decoder, self).__init__()
        self.n_latent = n_latent
        self.n_feature = n_feature

        layer = []
        n_hidden = [n_latent, *(n_hidden[::-1]), n_feature]
        for i in range(len(n_hidden)-1):
                layer.append(nn.Linear(n_hidden[i], n_hidden[i+1]))
                layer.append(nn.LeakyReLU())
                layer.append(nn.Dropout(p=dropout))
        self.decoder = nn.Sequential(*layer)
        
    def forward(self, x):
        x = self.decoder(x)
        return x

                 
class AutoEncoder(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
    ):
        
        super(AutoEncoder, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
        
    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)



# Base class for all models with explicit redshifting transformation
class ZLatentBase(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 wave_obs, 
                 wave_rest,
                 loss='l2',
                ):
        
        
        super(ZLatentBase, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        # register wavelength tensors on the same dives as the entire model
        self.register_buffer('wave_obs', wave_obs)
        self.register_buffer('wave_rest', wave_rest)
        
        assert loss in ['l2', 'emd']
        if loss == 'emd':
            self._loss = self._emd_loss
        else:
            self._loss = self._l2_loss        
        
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    def _forward(self, x, z0=0):
        s = self.encode(x)
        spectrum_restframe = self.decode(s)
        spectrum_observed = self.transform(spectrum_restframe, z0)
        return s, spectrum_restframe, spectrum_observed

    def forward(self, x, z0=0):
        s, spectrum_restframe, spectrum_observed = self._forward(x, z0=z0)
        return spectrum_observed
        
    def transform(self, spectrum_restframe, z):
        wave_redshifted = (self.wave_rest.unsqueeze(1) * (1 + z)).T
        return Interp1d()(wave_redshifted, spectrum_restframe, self.wave_obs)

    def loss(self, x, w, z0=0, individual=False):
        spectrum_observed = self.forward(x, z0=z0)
        return self._loss(x, w, spectrum_observed, individual=individual)
                
    def _l2_loss(self, x, w, spectrum_observed, individual=False):
        # proper neg log likelihood, w = inverse variance
        mask = w > 0
        D = (mask).sum(dim=1)
        lognorm =  D / 2 * np.log(2 * np.pi)
        tiny = 1e-10
        lognorm -= torch.sum(torch.log(w + tiny), dim=1)
        if individual:
            return torch.sum(0.5 * w * (x - spectrum_observed).pow(2), dim=1) + lognorm
        return torch.sum(0.5 * w * (x - spectrum_observed).pow(2)) + lognorm.sum()
        
    def _emd_loss(self, x, w, spectrum_observed, individual=False):
        if individual:
            dim=1
        else:
            dim=None
        return torch.cumsum(x - spectrum_observed, dim=-1).abs().sum(dim=dim)

    @property
    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
#### Simple MLP encoder ####

class ZLatentAutoEncoder(ZLatentBase):
    def __init__(self,
                 wave_obs, 
                 wave_rest, 
                 n_latent=5, 
                 n_hidden_enc=[128, 64, 32],
                 n_hidden_dec=[128, 64, 32],   
                 dropout=0,
                 loss='l2',
        ):
        
        encoder = Encoder(
            len(wave_obs),
            n_latent,
            n_hidden=n_hidden_enc,
            dropout=dropout,
        )
        decoder = Decoder(
            len(wave_rest),
            n_latent,
            n_hidden=n_hidden_dec,
            dropout=dropout,
        )
        
        super(ZLatentAutoEncoder, self).__init__(
            encoder,
            decoder,
            wave_obs,
            wave_rest,
            loss=loss,
        )

#### Series encoder ####

class SeriesEncoder(nn.Module):
    def __init__(self, n_latent, dropout=0):
        super(SeriesEncoder, self).__init__()
        
        filters = [128, 256, 512]
        sizes = [5, 11, 21]
        
        convs = []
        for i in range(3):
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
        
        self.conv1, self.conv2, self.conv3 = tuple(convs)
        self.pool1, self.pool2 = tuple(nn.MaxPool1d(s, padding=s//2) for s in sizes[:2])
        self.softmax = nn.Softmax(dim=2)
        self.linear = nn.Linear(filters[-1] // 2, n_latent)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.conv3(x)
        C = x.shape[1]
        h, a = torch.split(x, [C//2,C//2], dim=1)
        a = self.softmax(a)
        x = torch.sum(h * a, dim=2)
        x = self.linear(x)
        return x


class SeriesAutoencoder(ZLatentBase):
    def __init__(self,
                 wave_obs, 
                 wave_rest,
                 n_latent=5, 
                 n_hidden_dec=[128, 64, 32],  
                 dropout=0,
                 loss='l2',
                ):
        
        encoder = SeriesEncoder(n_latent, dropout=dropout)

        decoder = Decoder(
            len(wave_rest),
            n_latent,
            n_hidden=n_hidden_dec,
            dropout=dropout,
        )
        

        super(SeriesAutoencoder, self).__init__(
            encoder,
            decoder,
            wave_obs,
            wave_rest,
            loss=loss,
        )