import numpy as np
import torch
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
    plates, mjds, fibers = data['plate'][sl], data['mjd'][sl], data['fiber'][sl]
    id = [f"{plate}-{mjd}-{fiber}" for plate, mjd, fiber in zip(plates, mjds, fibers)]

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

def augment_batch(batch, how="mask", wave_obs=None):
    assert how in ["mask", "redshift"]

    spec, w, z = batch
    N, L = spec.shape

    if how == "mask":
        # randomly mask ~1000 bins
        delta = np.random.randint(500, 1500, N)
        start = np.random.randint(0, L-delta)
        end = start + delta
        new_spec = spec.clone()
        new_w = w.clone()
        for i in range(N):
            new_spec[i,start[i]:end[i]] = 0
            new_w[i,start[i]:end[i]] = 0
        return new_spec, new_w, z

    if how == "redshift":
        assert wave_obs is not None
        z_max = 0.5
        z_new = torch.rand(N).to(z.device) * z_max

        # apply inverse of redshift correction
        zfactor = ((1+z_new)/(1+z)).unsqueeze(1)

        # redshift linear interpolation
        spec_new = Interp1d()(wave_obs*zfactor, spec, wave_obs)
        w_new = Interp1d()(wave_obs*zfactor, w, wave_obs)
        return spec_new, w_new, z_new
