import numpy as np
import torch
import torch.nn.functional as F
import humanize, psutil, GPUtil
from torchinterp1d import Interp1d
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


def load_model(fileroot, n_latent=10):
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = None

    path = f'{fileroot}.pt'
    model = torch.load(path, map_location=device)

    if type(model)==list or type(model)==tuple:
        [m.eval() for m in model]
    elif type(model)==dict:
        mdict = model
        print("states:",mdict.keys())

        models = mdict["model"]
        instruments = mdict["instrument"]
        model = []
        if "n_latent" in mdict:n_latent=mdict["n_latent"]
        for m in models:
            loadm = SpectrumAutoencoder(wave_rest,n_latent=n_latent,
                                        normalize=option_normalize)
            loadm.load_state_dict(m)
            loadm.eval()
            model.append(loadm)

        for ins in instruments:
            empty=Instrument(wave_obs=None, calibration=None)
            model.append(empty)
    else: model.eval()

    path = f'{fileroot}.losses.npy'
    loss = np.load(path)
    print (f"model {fileroot}: iterations {len(loss)}, final loss: {loss[-1]}")
    return model, loss

def load_models(label, n_config, n_latent=10):
    models, losses = {}, {}
    best_model, best_loss = 0, np.inf
    for i in range(n_config):
        try:
            label_ = label + f".{i}"
            model, loss = load_model(label_, n_latent=n_latent)
            models[i] = model
            losses[i] = loss
            if loss[-1][1] < best_loss:
                best_loss = loss[-1][1]
                best_model = i
        except FileNotFoundError:
            pass

    return models, losses, best_model


def skylines_mask(waves, intensity_limit=2, radii=5, debug=True):

    f=open("sky-lines.txt","r")
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
    if debug:print("number of lines: %d, fraction: %.2f"%(n_lines,non_zero/mask.shape[0]))
    return mask

def mem_report():
    print("CPU RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ))

    if torch.cuda.device_count() ==0: return

    GPUs = GPUtil.getGPUs()
    for i, gpu in enumerate(GPUs):
        print('GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%'.format(i, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil*100))
    return


class LogLinearDistribution():
    def __init__(self, a, bound):
        x0,xf = bound
        self.bound = bound
        self.a = a
        self.norm = -a*np.log(10)/(10**(a*x0)-10**(a*xf))

    def pdf(self,x):
        pdf = self.norm*10**(self.a*x)
        pdf[(x<self.bound[0])|(x>self.bound[1])] = 0
        return pdf

    def cdf(self,x):
        factor = self.norm/(-self.a*np.log(10))
        cdf = -factor*(10**(self.a*x)-10**(self.a*self.bound[0]))
        cdf[x<self.bound[0]] = 0
        cdf[x>self.bound[1]] = 1
        return cdf

    def inv_cdf(self,cdf):
        factor = self.norm/(-self.a*np.log(10))
        return (1/self.a)*torch.log10(10**(self.a*self.bound[0])-cdf/factor)


def insert_jitters(spec,number,slope=-1.32,bound=[0.0,2]):
    number = int(number)
    location = torch.randint(len(spec), device=device,
                             size=(1,number)).squeeze(0)

    loglinear = LogLinearDistribution(slope,bound)
    var = loglinear.inv_cdf(torch.rand(number,device=device))
    amp = var**0.5

    # half negative
    half = torch.rand(number)>0.5
    amp[half] = -amp[half]

    # avoid inserting jitter to padded regions
    mask = spec[location]>0
    return location[mask],amp[mask]

def jitter_redshift(batch, params, inst):
    # original batch
    spec, w, true_z = batch
    wave_obs = inst.wave_obs

    wave_mat = wave_obs*torch.ones_like(spec)
    ncopy = len(params)

    batch_out  = {}

    data = []

    begin = 0
    # number of copys
    for copy,param in enumerate(params):
        batch_size = len(true_z)

        #z_offset,n_lim = param
        z_lim, n_lim = param

        # uniform distribution
        z_offset = z_lim*(2*torch.rand(batch_size,device=device)-1)

        n_jit = np.random.randint(n_lim[0],n_lim[1],
                                  size=batch_size)

        z_new = true_z+z_offset

        z_new[z_new<0] = 0 # avoid negative redshift
        z_new[z_new>0.5] = 0.5 # avoid very high redshift

        zfactor = ((1+z_new)/(1+true_z)).unsqueeze(1)*torch.ones_like(spec)

        # redshift linear interpolation
        spec_new = Interp1d()(wave_mat*zfactor,spec,wave_mat)
        w_new = Interp1d()(wave_mat*zfactor,w,wave_mat)
        w_new[w_new<=1e-6] = 0

        record = []
        for i in range(len(spec_new)):
            loc,amp = insert_jitters(spec_new[i],n_jit[i])
            spec_new[i][loc] += amp
            w_new[i][loc] = 1/(amp**2+1/w_new[i][loc])
            record.append([z_offset[i].item(),n_jit[i]])

        med = spec_new.median(1,False).values[:,None]
        med[med<1e-1] = 1e-1

        spec_new /= med
        #print("median:",spec_new.median(1,False).min())

        if torch.isnan(spec_new).any():
            nan_ind = torch.isnan(spec_new)
            print("spec nan! z:", true_z[nan_ind],
                  "offset:",z_offset[nan_ind])

        end = begin+batch_size
        data.append([spec_new,w_new,z_new])
        batch_out[copy]={"param":record,"range":[begin,end]}
        begin=end

    new_batch = [torch.cat([d[i] for d in data]) for i in range(3)]
    batch_out['batch'] = new_batch
    return batch_out
