import glob, os, urllib.request
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchinterp1d import Interp1d
import astropy.io.fits as fits
import astropy.table as aTable
from functools import partial

from ..instrument import Instrument
from ..util import BatchedFilesDataset, load_batch

class SDSS(Instrument):
    _wave_obs = 10**torch.arange(3.578, 3.97, 0.0001)
    _base_url = "https://data.sdss.org/sas/dr16/sdss/spectro/redux/26/spectra/lite/"

    def __init__(self, lsf=None, calibration=None):
        super().__init__(SDSS._wave_obs, lsf=lsf, calibration=calibration)

    @classmethod
    def get_data_loader(cls, dir, which=None, tag=None, batch_size=1024, shuffle=False):
        files = cls.list_batches(dir, which=which, tag=tag)
        if which in ["train", "valid"]:
            subset = slice(0,3)
        else:
            subset = None
        load_fct = partial(load_batch, subset=subset)
        data = BatchedFilesDataset(files, load_fct, shuffle=shuffle)
        return DataLoader(data, batch_size=batch_size)

    @classmethod
    def list_batches(cls, dir, which=None, tag=None):
        if tag is None:
            tag = "chunk1024"
        classname = cls.__mro__[0].__name__
        filename = f"{classname}{tag}_*.pkl"
        batch_files = glob.glob(dir + "/" + filename)
        batches = [item for item in batch_files if not "copy" in item]

        NBATCH = len(batches)
        train_batches = batches[:int(0.7*NBATCH)]
        valid_batches = batches[int(0.7*NBATCH):int(0.85*NBATCH)]
        test_batches = batches[int(0.85*NBATCH):]

        if which == "test": return test_batches
        elif which == "valid": return valid_batches
        elif which == "train": return train_batches
        else: return batches

    @classmethod
    def save_batch(cls, dir, batch, tag=None, counter=None):
        if tag is None:
            tag = f"chunk{len(batch)}"
        if counter is None:
            counter = ""
        classname = cls.__mro__[0].__name__
        filename = os.path.join(dir, f"{classname}{tag}_{counter}.pkl")

        with open(filename, 'wb') as f:
            pickle.dump(batch, f)

    @classmethod
    def save_in_batches(cls, dir, ids, tag=None, batch_size=1024):
        N = len(ids)
        idx = np.arange(0, N, batch_size)
        batches = np.array_split(ids, idx[1:])
        for counter, ids_ in zip(idx, batches):
            print (f"saving batch {counter} / {N}")
            batch = cls.make_batch(dir, ids_)
            cls.save_batch(dir, batch, tag=tag, counter=counter)

    @classmethod
    def get_spectrum(cls, dir, plate, mjd, fiberid, return_file=False):
        filename = "spec-%s-%i-%s.fits" % (str(plate).zfill(4), mjd, str(fiberid).zfill(4))
        classname = cls.__mro__[0].__name__.lower()
        dirname = os.path.join(dir, f"{classname}-spectra")
        flocal = os.path.join(dirname, filename)
        if not os.path.isfile(flocal):
            os.makedirs(dirname, exist_ok=True)
            url = "%s/%s/%s" % (cls._base_url, str(plate).zfill(4), filename)
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, flocal)

        if return_file:
            return flocal
        return cls.prepare_spectrum(flocal)

    @classmethod
    def get_image(cls, dir, plate, mjd, fiberid, return_file=False):
        filename = "im-%s-%i-%s.jpeg" % (str(plate).zfill(4), mjd, str(fiberid).zfill(4))
        dirname = os.path.join(dir, "sdss-images")
        flocal = os.path.join(dirname, filename)
        if not os.path.isfile(flocal):
            os.makedirs(dirname, exist_ok=True)
            # get RA/DEC from spectrum file
            specname = cls.get_spectrum(dir, plate, mjd, fiberid, return_file=True)
            hdulist = fits.open(specname)
            specinfo = hdulist[2].data[0]
            ra, dec = specinfo['PLUG_RA'], specinfo['PLUG_DEC']
            # query skyserver cutout service
            print(f"downloading image at {ra}/{dec}")
            image_url = "https://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg"
            scale, size, opt = 0.2, 256, "S" # "S" = outline of fiber
            params = {"ra":ra, "dec":dec, "scale": scale,
                      "height": size, "width": size, "opt": opt}
            url = image_url + "?" + "&".join('='.join((key,str(val))) for (key,val) in params.items())
            urllib.request.urlretrieve(url, flocal)

        if return_file:
            return flocal

        from IPython import display
        return display.Image(flocal)

    @classmethod
    def prepare_spectrum(cls, filename):
        hdulist = fits.open(filename)
        data = hdulist[1].data
        loglam = data['loglam']
        flux = data['flux']
        ivar = data['ivar']
        sky = data['sky']

        # apply bitmask, remove small values, and mask sky-dominated bins
        mask = data['and_mask'].astype(bool) | (ivar <= 1e-6) | (flux < sky)
        ivar[mask] = 0

        # loglam is subset of _wave_obs, need to insert into extended tensor
        L = len(cls._wave_obs)
        start = int(np.around((loglam[0] - torch.log10(cls._wave_obs[0]).item())/0.0001))
        end = min(start+len(loglam), L)
        spec = torch.zeros(L)
        w = torch.zeros(L)
         # explicit type conversion to float32 to get to little endian
        spec[start:end]  = torch.from_numpy(flux.astype(np.float32))
        w[start:end] = torch.from_numpy(ivar.astype(np.float32))

        # get plate, mjd, fiberid info
        specinfo = hdulist[2].data[0]
        id = torch.tensor((specinfo['PLATE'], specinfo['MJD'], specinfo['FIBERID']), dtype=torch.int)

        # normalize spectrum to make it easier for encoder
        norm = torch.median(spec[w>0])
        spec /= norm
        w *= norm**2

        # get redshift and error
        z = torch.tensor(specinfo['Z'])
        zerr = torch.tensor(specinfo['Z_ERR'])

        return spec, w, z, id, norm, zerr

    @classmethod
    def make_batch(cls, dir, ids):
        files = [ cls.get_spectrum(dir, plate, mjd, fiberid, return_file=True) for plate, mjd, fiberid in ids ]
        N = len(files)
        L = len(cls._wave_obs)
        spec = torch.empty((N, L))
        w = torch.empty((N, L))
        z = torch.empty(N)
        id = torch.empty((N, 3), dtype=torch.int)
        norm = torch.empty(N)
        zerr = torch.empty(N)
        for i, f in enumerate(files):
            spec[i], w[i], z[i], id[i], norm[i], zerr[i] = cls.prepare_spectrum(f)
        return spec, w, z, id, norm, zerr

    @classmethod
    def get_ids(cls, dir, selection_fct=None):
        main_file = os.path.join(dir, "specObj-dr16.fits")
        if not os.path.isfile(mainfile):
            url = "https://data.sdss.org/sas/dr16/sdss/spectro/redux/specObj-dr16.fits"
            print (f"downloading {url}, this will take a while...")
            urllib.request.urlretrieve(url, main_file)

        print(f"opening {main_file}")
        specobj = aTable.Table.read(spec_file)

        if selection_fct is None:
            # apply default selections
            classname = cls.__mro__[0].__name__.lower()
            sel = ((specobj['SURVEY'] == f'{classname}  ') & # SDSS survey
                    (specobj['PLATEQUALITY'] == 'good    ') &
                    (specobj['TARGETTYPE'] == 'SCIENCE ')
                   )
            sel &= ((specobj['Z'] > 0.) & (specobj['Z_ERR'] < 1e-4))
            sel &= ((specobj['SOURCETYPE'] == 'GALAXY                   ') &
                         (specobj['CLASS'] == 'GALAXY'))
        else:
            sel = selection_fct(specobj)

        plate = specobj['PLATE'][sel]
        mjd = specobj['MJD'][sel]
        fiberid = specobj['FIBERID'][sel]
        return tuple(zip(plate, mjd, fiberid))

    @classmethod
    def augment_spectra(cls, batch, redshift=True, noise=True, mask=True, ratio=0.05):
        spec, w, z = batch[:3]
        batch_size, spec_size = spec.shape
        device = spec.device
        wave_obs = cls._wave_obs.to(device)

        if redshift:
            # uniform distribution of redshift offsets, width = z_lim
            z_lim = 0.2
            z_base = torch.relu(z-z_lim)
            z_new = z_base+z_lim*(torch.rand(batch_size, device=device))
            # keep redshifts between 0 and 0.5
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
            noise_mask = torch.distributions.Uniform(0, 1).sample(spec.shape).to(device)>ratio
            noise[noise_mask]=0
            spec_new += noise
            # add variance in quadrature, avoid division by 0
            w_new = 1/(1/(w_new + 1e-6) + noise**2)

        if mask:
            length = int(spec_size * ratio)
            start = torch.randint(0, spec_size-length, (1,)).item()
            spec_new[:, start:start+length] = 0
            w_new[:, start:start+length] = 0

        return spec_new, w_new, z_new


class BOSS(SDSS):
    _wave_obs = 10**torch.arange(3.549, 4.0175, 0.0001)
    _base_url = "https://data.sdss.org/sas/dr16/eboss/spectro/redux/v5_13_0/spectra/lite/"

    def __init__(self, lsf=None, calibration=None):
        super(Instrument, self).__init__(BOSS._wave_obs, lsf=lsf, calibration=calibration)
