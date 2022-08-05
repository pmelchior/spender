import numpy as np
import torch
import torch.nn.functional as F
import humanize, psutil, GPUtil
from torchinterp1d import Interp1d

def get_norm(x):
    # simple median as robust mean across the spectrum
    norm = np.median(x, axis=1)
    return norm


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


def resample_to_restframe(wave_obs,wave_rest,y,w,z):
    wave_z = (wave_rest.unsqueeze(1)*(1 + z)).T
    wave_obs = wave_obs.repeat(y.shape[0],1)
    # resample observed spectra to restframe
    yrest = Interp1d()(wave_obs, y, wave_z)
    wrest =  Interp1d()(wave_obs, w, wave_z)

    # interpolation = extrapolation outside of observed region, need to mask
    msk = (wave_z<=wave_obs.min())|(wave_z>=wave_obs.max())
    # yrest[msk]=0 # not needed because all spectral elements are weighted
    wrest[msk]=0
    return yrest,wrest


def augment_spectra(batch, instrument, redshift=True, noise=True, mask=True):
    spec, w, z = batch
    batch_size, spec_size = spec.shape
    device = spec.device
    wave_obs = instrument.wave_obs

    if redshift:
        # uniform distribution of redshift offsets
        z_lim = 0.8 * torch.max(z)
        z_offset = z_lim*(torch.rand(batch_size, device=device)-0.5)
        # keep redshifts between 0 and 0.5
        z_new = z + z_offset
        z_new = torch.minimum(torch.nn.functional.relu(z_new), 0.5 * torch.ones(batch_size, device=device))
        zfactor = ((1 + z_new)/(1 + z))
        wave_redshifted = (wave_obs.unsqueeze(1) * zfactor).T

        # redshift linear interpolation
        spec_new = Interp1d()(wave_redshifted, spec, wave_obs)
        # ensure extrapolated values have zero weights
        w_new = torch.clone(w)
        w_new[:,0] = 0
        w_new[:,-1] = 0
        w_new = Interp1d()(wave_redshifted, w_new, wave_obs)
        w_new = torch.nn.functional.relu(w_new)
    else:
        spec_new, w_new, z_new = torch.clone(spec), torch.clone(w), z

    # add noise
    if noise:
        sigma = 0.2 * torch.max(spec, 1, keepdim=True)[0]
        noise = sigma * torch.distributions.Normal(0, 1).sample(spec.shape).to(device)
        spec_new += noise
        # add variance in quadrature, avoid division by 0
        w_new = 1/(1/(w_new + 1e-6) + (sigma**2))

    if mask:
        length = spec_size // 10
        start = torch.randint(0, spec_size-length, (1,)).item()
        spec_new[:, start:start+length] = 0
        w_new[:, start:start+length] = 0

    return spec_new, w_new, z_new
