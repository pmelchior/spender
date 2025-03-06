import glob
import os
import urllib.request
from functools import partial

import astropy.io.fits as fits
import astropy.table as aTable
import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader
from ..instrument import Instrument, get_skyline_mask
from ..util import BatchedFilesDataset, load_batch, interp1d


class SDSS(Instrument):
    """SDSS instrument

    Implements basic parameterization of the SDSS-II spectrograph as well as functions
    to download and organize the spectra from the DR16 data archive.
    """

    _wave_obs = 10 ** torch.arange(3.578, 3.97, 0.0001)
    _skyline_mask = get_skyline_mask(_wave_obs)
    _base_url = "https://data.sdss.org/sas/dr16/sdss/spectro/redux/26/spectra/lite/"

    def __init__(self, lsf=None, calibration=None):
        """Create instrument

        Parameters
        ----------
        lsf: :class:`LSF`
            (optional) Line spread function model
        calibration: callable
            (optional) function to calibrate the observed spectrum
        """
        super().__init__(SDSS._wave_obs, lsf=lsf, calibration=calibration)

    @classmethod
    def get_data_loader(
        cls,
        dir,
        which=None,
        tag=None,
        batch_size=1024,
        shuffle=False,
        shuffle_instance=False,
    ):
        """Get a dataloader for batches of spectra

        Parameters
        ----------
        dir: string
            Root directory for data storage
        which: ['train', 'valid', 'test'] or None
            Which subset of the spectra to return. If `None`, returns all of them.
        tag: string
            Name to specify which batch files to load
        batch_size: int
            Number of spectra in each batch
        shuffle: bool
            Whether to shuffle the order of the batch files
        shuffle_instance: bool
            Whether to shuffle spectra within each batch

        Returns
        -------
        :class:`torch.utils.data.DataLoader`
        """
        files = cls.list_batches(dir, which=which, tag=tag)
        if which in ["train", "valid"]:
            subset = slice(0, 3)
        else:
            subset = None
        load_fct = partial(load_batch, subset=subset)
        data = BatchedFilesDataset(
            files, load_fct, shuffle=shuffle, shuffle_instance=shuffle_instance
        )
        return DataLoader(data, batch_size=batch_size)

    @classmethod
    def list_batches(cls, dir, which=None, tag=None):
        """List all batch files

        Parameters
        ----------
        dir: string
            Root directory for data storage
        which: ['train', 'valid', 'test'] or None
            Which subset of the spectra to return. If `None`, returns all of them.
        tag: string
            Name to specify which batch files to load

        Returns
        -------
        list of filepaths
        """
        if tag is None:
            tag = "variable"
        classname = cls.__mro__[0].__name__
        filename = f"{classname}{tag}_*.pkl"
        batch_files = glob.glob(dir + "/" + filename)
        batches = [item for item in batch_files if not "copy" in item]

        NBATCH = len(batches)
        train_batches = batches[: int(0.7 * NBATCH)]
        valid_batches = batches[int(0.7 * NBATCH) : int(0.85 * NBATCH)]
        test_batches = batches[int(0.85 * NBATCH) :]

        if which == "test":
            return test_batches
        elif which == "valid":
            return valid_batches
        elif which == "train":
            return train_batches
        else:
            return batches

    @classmethod
    def save_batch(cls, dir, batch, tag=None, counter=None):
        """Save batch into a pickled file

        Parameters
        ----------
        dir: string
            Root directory for data storage
        batch: `torch.tensor`, shape (N, L)
            Spectrum batch
        tag: string
            Name to specify which batch file name
        counter: int
            Set to add a batch counter to the filename

        Returns
        -------
        None

        """
        if tag is None:
            tag = f"chunk{len(batch)}"
        if counter is None:
            counter = ""
        classname = cls.__mro__[0].__name__
        filename = os.path.join(dir, f"{classname}{tag}_{counter}.pkl")

        with open(filename, "wb") as f:
            pickle.dump(batch, f)

    @classmethod
    def save_in_batches(cls, dir, ids, tag=None, batch_size=1024):
        """Save all spectra for given ids into batch files

        Parameters
        ----------
        dir: string
            Root directory for data storage
        ids: list of (plate, mjd, fiberid)
            Identifier of spectrum
        tag: string
            Name to specify the batch file name
        batch_size: int
            Number of spectra in each batch

        Returns
        -------
        None

        """
        N = len(ids)
        idx = np.arange(0, N, batch_size)
        batches = np.array_split(ids, idx[1:])
        for counter, ids_ in zip(idx, batches):
            print(f"saving batch {counter} / {N}")
            batch = cls.make_batch(dir, ids_)
            cls.save_batch(dir, batch, tag=tag, counter=counter)

    @classmethod
    def get_spectrum(cls, dir, plate, mjd, fiberid, return_file=False):
        """Download and prepare spectrum for analysis

        Parameters
        ----------
        dir: string
            Root directory for data storage
        plate: int
            SDSS plate number
        mjd: int
            SDSS Modified Julian Date of the night when the spectrum was observed
        fiberid: int
            Fiber number on SDSS plate
        return_file: bool
            Whether to return the local file name or the prepared spectrum

        Returns
        -------
        Either the local file name or the prepared spectrum
        """

        plate, mjd, fiberid = [int(i) for i in [plate, mjd, fiberid]]
        filename = "spec-%s-%i-%s.fits" % (
            str(plate).zfill(4),
            mjd,
            str(fiberid).zfill(4),
        )
        dirname = os.path.join(dir, str(plate).zfill(4))
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
        """Download image cutout for spectrum target

        Parameters
        ----------
        dir: string
            Root directory for data storage
        plate: int
            SDSS plate number
        mjd: int
            SDSS Modified Julian Date of the night when the spectrum was observed
        fiberid: int
            Fiber number on SDSS plate
        return_file: bool
            Whether to return the local file name or actual image

        Returns
        -------
        Either the local file name or :class:`IPython.display.Image`
        """
        filename = "im-%s-%i-%s.jpeg" % (
            str(plate).zfill(4),
            mjd,
            str(fiberid).zfill(4),
        )
        dirname = os.path.join(dir, "sdss-images")
        flocal = os.path.join(dirname, filename)
        if not os.path.isfile(flocal):
            os.makedirs(dirname, exist_ok=True)
            # get RA/DEC from spectrum file
            specname = cls.get_spectrum(dir, plate, mjd, fiberid, return_file=True)
            hdulist = fits.open(specname)
            specinfo = hdulist[2].data[0]
            ra, dec = specinfo["PLUG_RA"], specinfo["PLUG_DEC"]
            # query skyserver cutout service
            print(f"downloading image at {ra}/{dec}")
            image_url = "https://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg"
            scale, size, opt = 0.2, 256, "S"  # "S" = outline of fiber
            params = {
                "ra": ra,
                "dec": dec,
                "scale": scale,
                "height": size,
                "width": size,
                "opt": opt,
            }
            url = (
                image_url
                + "?"
                + "&".join("=".join((key, str(val))) for (key, val) in params.items())
            )
            urllib.request.urlretrieve(url, flocal)

        if return_file:
            return flocal

        from IPython import display

        return display.Image(flocal)

    @classmethod
    def prepare_spectrum(cls, filename, z=None):
        """Prepare spectrum for analysis

        This method creates an extended mask, using the original SDSS `and_mask` and
        the skyline mask of the instrument. The weights for all masked regions is set to
        0, but the spectrum itself is not altered.

        The spectrum and weights are then cast into a fixed-format data vector, which
        standardizes the variable wavelengths of each observation.

        A normalization is computed as the median flux in the relatively flat region
        between restframe 5300 and 5850 A. The spectrum is divided by this factor, the weights
        are muliplied with the square of this factor.

        Parameter
        ---------
        filename: string
            Path to local file containing the spectrum
        z: float or None
            Redshift of the spectrum
            If None, redshift and its error will be read from the file and returned

        Returns
        -------
        spec: `torch.tensor`, shape (L, )
            Normalized spectrum
        w: `torch.tensor`, shape (L, )
            Inverse variance weights of normalized spectrum
        norm: `torch.tensor`, shape (1, )
            Flux normalization factor
        z: `torch.tensor`, shape (1, )
            Redshift (only returned when argument z=None)
        zerr: `torch.tensor`, shape (1, )
            Redshift error (only returned when argument z=None)
        """
        hdulist = fits.open(filename)
        data = hdulist[1].data
        loglam = data["loglam"]
        flux = data["flux"]
        ivar = data["ivar"]

        # apply bitmask, remove small values
        mask = data["and_mask"].astype(bool) | (ivar <= 1e-6)
        ivar[mask] = 0

        # loglam is subset of _wave_obs, need to insert into extended tensor
        L = len(cls._wave_obs)
        start = int(np.around((loglam[0] - torch.log10(cls._wave_obs[0]).item())/0.0001))
        if start<0:
            flux = flux[-start:]
            ivar = ivar[-start:]
            end = min(start+len(loglam), L)
            start = 0
        else:
            end = min(start+len(loglam), L)
        spec = torch.zeros(L)
        w = torch.zeros(L)
        # explicit type conversion to float32 to get to little endian
        spec[start:end]  = torch.from_numpy(flux.astype(np.float32))
        w[start:end] = torch.from_numpy(ivar.astype(np.float32))

        # remove regions around skylines
        w[cls._skyline_mask] = 0

        extended_return = False
        if z is None:
            # get plate, mjd, fiberid info
            specinfo = hdulist[2].data[0]
            # get redshift and error
            z = torch.tensor(specinfo["Z"])
            zerr = torch.tensor(specinfo["Z_ERR"])
            extended_return = True

        # normalize spectrum:
        # for redshift invariant encoder: select norm window in restframe
        wave_rest = cls._wave_obs / (1 + z)
        # flatish region that is well observed out to z ~ 0.5
        sel = (w > 0) & (wave_rest > 5300) & (wave_rest < 5850)
        if sel.count_nonzero() == 0: norm = torch.tensor(0)
        else: norm = torch.median(spec[sel])
        # remove spectra (from training) for which no valid norm could be found
        if not torch.isfinite(norm):
            norm = 0
        else:
            spec /= norm
        w *= norm**2

        if extended_return:
            return spec, w, norm, z, zerr
        return spec, w, norm
    
    @classmethod
    def make_batch(cls, dir, fields):
        """Make a batch of spectra from their IDs

        Parameters
        ----------
        dir: string
            Root directory for data storage
        fields: list of (plate, mjd, fiberid, [z, z_err])
            List of object qualifiers from query()

        Returns
        -------
        spec: `torch.tensor`, shape (N, L)
            Normalized spectrum
        w: `torch.tensor`, shape (N, L)
            Inverse variance weights of normalized spectrum
        z: `torch.tensor`, shape (N, )
            Redshift from SDSS pipeline
        norm: `torch.tensor`, shape (N, )
            Normalization factor
        zerr: torch.tensor`, shape (N, )
            Redshift error from SDSS pipeline
        """

        N = len(fields)
        L = len(cls._wave_obs)
        spec = torch.empty((N, L))
        w = torch.empty((N, L))
        z = torch.empty(N)
        id = torch.empty((N, 3), dtype=torch.int)
        norm = torch.empty(N)
        zerr = torch.empty(N)
        for i in range(N):
            if len(fields[i]) == 5:
                plate, mjd, fiberid, z_, zerr_ = fields[i]
                f = cls.get_spectrum(dir, plate, mjd, fiberid, return_file=True)
                spec[i], w[i], norm[i] = cls.prepare_spectrum(f, z_)
                z[i], zerr[i] = z_, zerr_
            elif len(fields[i]) == 3:
                plate, mjd, fiberid = fields[i]
                f = cls.get_spectrum(dir, plate, mjd, fiberid, return_file=True)
                spec[i], w[i], norm[i], z[i], zerr[i] = cls.prepare_spectrum(f)
            else:
                raise AttributeError("fields must contain (plate, mjd, fiberid, z, z_err) or (plate, mjd, fiberid)")
            id[i] = torch.tensor((plate, mjd, fiberid), dtype=torch.int)
        return spec, w, z, norm, zerr

    @classmethod
    def query(cls, dir, fields=["PLATE", "MJD", "FIBERID", "Z", "Z_ERR"], selection_fct=None):
        """Select fields from main specrum table for objects that match `selection_fct`

        NOTE: This function will download the file `specObj-dr16.fits` from the data
        archive. This file is *not* small...

        Parameters
        ----------
        dir: string
            Root directory for data storage
        fields: list of string
            Catalog field names to return
        selection_fct: callable
            Function to select matches from all items in the main table

        Returns
        -------
        fields: `torch.tensor`, shape (N, F)
            Tensor of fields for the selected objects

        """
        main_file = os.path.join(dir, "specObj-dr16.fits")
        if not os.path.isfile(main_file):
            url = "https://data.sdss.org/sas/dr16/sdss/spectro/redux/specObj-dr16.fits"
            print(f"downloading {url}, this will take a while...")
            urllib.request.urlretrieve(url, main_file)

        print(f"opening {main_file}")
        specobj = aTable.Table.read(main_file)

        if selection_fct is None:
            # apply default selections
            classname = cls.__mro__[0].__name__.lower()
            sel = (
                (specobj["SURVEY"] == f"{classname}  ")
                & (specobj["PLATEQUALITY"] == "good    ")  # SDSS survey
                & (specobj["TARGETTYPE"] == "SCIENCE ")
            )
            sel &= (specobj["Z"] > 0.0) & (specobj["Z_ERR"] < 1e-4)
            sel &= (specobj["SOURCETYPE"] == "GALAXY                   ") & (
                specobj["CLASS"] == "GALAXY"
            )
        else:
            sel = selection_fct(specobj)

        return specobj[fields][sel]

    @classmethod
    def augment_spectra(cls, batch, redshift=True, noise=True, mask=True, ratio=0.05, z_new=None):
        """Augment spectra for greater diversity

        Parameters
        ----------
        batch: `torch.tensor`, shape (N, L)
            Spectrum batch
        redshift: bool
            Modify redshift by up to 0.2 (keeping it within 0...0.5)
        noise: bool
            Whether to add noise to the spectrum (up to 0.2*max(spectrum))
        mask: bool
            Whether to block out a fraction (given by `ratio`) of the spectrum
        ratio: float
            Fraction of the spectrum that will be masked
        z_new: float
            Adopt this redshift for all spectra in the batch

        Returns
        -------
        spec: `torch.tensor`, shape (N, L)
            Altered spectrum
        w: `torch.tensor`, shape (N, L)
            Altered inverse variance weights of spectrum
        z: `torch.tensor`, shape (N, )
            Altered redshift
        """

        spec, w, z = batch[:3]
        batch_size, spec_size = spec.shape
        device = spec.device
        wave_obs = cls._wave_obs.to(device)

        if redshift:
            if z_new == None:
                # uniform distribution of redshift offsets, width = z_lim
                z_lim = 0.5
                z_base = torch.relu(z-z_lim)
                z_new = z_base+z_lim*(torch.rand(batch_size, device=device))
            # keep redshifts between 0 and 0.5
            z_new = torch.minimum(
                torch.nn.functional.relu(z_new),
                0.5 * torch.ones(batch_size, device=device),
            )
            zfactor = (1 + z_new) / (1 + z)
            wave_redshifted = (wave_obs.unsqueeze(1) * zfactor).T

            # redshift linear interpolation
            spec_new = interp1d(wave_redshifted, spec, wave_obs)
            # ensure extrapolated values have zero weights
            w_new = torch.clone(w)
            w_new[:, 0] = 0
            w_new[:, -1] = 0
            w_new = interp1d(wave_redshifted, w_new, wave_obs)
            w_new = torch.nn.functional.relu(w_new)
        else:
            spec_new, w_new, z_new = torch.clone(spec), torch.clone(w), z

        # add noise
        if noise:
            sigma = 0.2 * torch.max(spec, 1, keepdim=True)[0]
            noise = sigma * torch.distributions.Normal(0, 1).sample(spec.shape).to(
                device
            )
            noise_mask = (
                torch.distributions.Uniform(0, 1).sample(spec.shape).to(device) > ratio
            )
            noise[noise_mask] = 0
            spec_new += noise
            # add variance in quadrature, avoid division by 0
            w_new = 1 / (1 / (w_new + 1e-6) + noise**2)

        if mask:
            length = int(spec_size * ratio)
            start = torch.randint(0, spec_size - length, (1,)).item()
            spec_new[:, start : start + length] = 0
            w_new[:, start : start + length] = 0

        return spec_new, w_new, z_new


class BOSS(SDSS):
    """BOSS instrument

    This is a variant of :class:`SDSS` with a different observed wavelength vector and
    data archive URL.
    """

    _wave_obs = 10 ** torch.arange(3.549, 4.0175, 0.0001)
    _skyline_mask = get_skyline_mask(_wave_obs)
    _base_url = (
        "https://data.sdss.org/sas/dr16/eboss/spectro/redux/v5_13_0/spectra/lite/"
    )

    def __init__(self, lsf=None, calibration=None):
        super(Instrument, self).__init__(
            BOSS._wave_obs, lsf=lsf, calibration=calibration
        )
