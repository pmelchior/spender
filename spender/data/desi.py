import glob
import os
from typing import Optional
import urllib.request
from functools import partial

import requests

import astropy.io.fits as fits
import astropy.table as aTable
import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader
import h5py

from ..instrument import Instrument, get_skyline_mask
from ..util import BatchedFilesDataset, interp1d, load_batch


class DESI(Instrument):
    """DESI instrument

    Implements basic parameterization of the DESI spectrograph as well as functions
    to download and organize the spectra from the EDR release.
    """

    _wave_obs = torch.linspace(3600.0, 9824.0, 7781, dtype=torch.float64)
    _skyline_mask = get_skyline_mask(_wave_obs)
    _base_url = "https://data.desi.lbl.gov/public/edr/spectro/redux/fuji/"  # this should be public on Tuesday

    def __init__(self, lsf=None, calibration=None):
        """Create instrument

        Parameters
        ----------
        lsf: :class:`LSF`
            (optional) Line spread function model
        calibration: callable
            (optional) function to calibrate the observed spectrum
        """
        super().__init__(DESI._wave_obs, lsf=lsf, calibration=calibration)

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
            tag = "Variable"
        classname = cls.__mro__[0].__name__
        filename = f"{classname}{tag}*_*.pkl"
        batches = glob.glob(dir + "/" + filename)

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
            tag = f"chunk{len(batch[0])}"
        if counter is None:
            counter = ""
        classname = cls.__mro__[0].__name__
        filename = os.path.join(dir, f"{classname}{tag}_{counter}.pkl")

        with open(filename, "wb") as f:
            pickle.dump(batch, f)
        return filename

    @classmethod
    def save_in_batches(cls, dir, ids, tag=None, batch_size=1024):
        """Save all spectra for given ids into batch files

        Parameters
        ----------
        dir: string
            Root directory for data storage
        ids: list of (survey, program, healpix, target)
            Identifier of healpix
        tag: string
            Name to specify the batch file name
        batch_size: int
            Number of spectra in each batch

        Returns
        -------
        None

        """
        counter, new_batch = 0, True
        for _id in ids:
            survey, prog, hpix, target = _id

            f = cls.get_spectra(dir, survey, prog, hpix, return_file=True)

            spec, w, z, target_id, norm, zerr = cls.prepare_spectra(f, target=target)
            if new_batch: 
                batches = [spec, w, z, target_id, norm, zerr]
            else: 
                batches[0] = torch.concatenate([batch[0], spec], axis=0)
                batches[1] = torch.concatenate([batch[1], w], axis=0)
                batches[2] = torch.concatenate([batch[2], z], axis=0)
                batches[3] = torch.concatenate([batch[3], target_id], axis=0)
                batches[4] = torch.concatenate([batch[4], norm], axis=0)
                batches[5] = torch.concatenate([batch[5], zerr], axis=0)
            
            N = batches[0].shape[0]
            while N > batch_size:
                batch = [_batch[:batch_size] for _batch in batches]

                print(f"saving batch {counter}")
                cls.save_batch(dir, batch, tag=tag, counter=counter)
                counter += 1
                N -= batch_size

                batches = [_batch[batch_size:] for _batch in batches]

    @classmethod
    def get_spectra(cls, dir, survey, prog, hpix, return_file=False):
        """Download and prepare spectrum for analysis

        Parameters
        ----------
        dir: string
            Root directory for data storage
        survey: string
            SV1, SV2, or SV3. Probably want to use SV3 for starters
        prog: string
            dark/bright
        hpix: int
            healipx number
        return_file: bool
            Whether to return the local file name or the prepared spectrum

        Returns
        -------
        Either the local file name or the prepared spectrum
        """

        for ftype in ["redrock", "rrdetails", "emline", "qso_mgii", "qso_qn", "coadd"]:
            filename = "%s-%s-%s-%i.fits" % (ftype, survey, prog, hpix)
            if ftype == 'rrdetails': filename = filename.replace('.fits', '.h5')
            dirname = os.path.join(dir, str(hpix))
            flocal = os.path.join(dirname, filename)
            # download spectra file
            if not os.path.isfile(flocal):
                os.makedirs(dirname, exist_ok=True)
                url = "%s/healpix/%s/%s/%s/%i/%s" % (
                    cls._base_url,
                    survey,
                    prog,
                    str(hpix)[:-2],
                    hpix,
                    filename,
                )
                print(f"downloading {url}")
                urllib.request.urlretrieve(url, flocal)

        if return_file:
            return flocal
        return cls.prepare_spectra(flocal)

    @classmethod
    def prepare_spectra(cls, filename, target=None, spectype='GALAXY'):
        """Prepare spectra in DESI healpix for analysis

        This method creates an extended mask, using the original DESI mask and
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
        target: string
            target class: BGS, LRG, ELG, QSO

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
        # read spectra file
        hdulist = fits.open(filename)
        survey = hdulist[0].header['SURVEY'].upper()
        meta = aTable.Table.read(filename) # meta data 
        target_id = hdulist[1].data['TARGETID'] # unique target ID

        # read redrock file
        rr = fits.open(filename.replace("coadd", "redrock"))
        # get redshift and error
        z = torch.tensor(rr[1].data['Z'].astype(np.float32))
        zerr = torch.tensor(rr[1].data['ZERR'].astype(np.float32))

        keep = ((meta['COADD_FIBERSTATUS'] == 0) &                  # good fiber 
                (meta['%s_DESI_TARGET' % survey] != 0) &            # is DESI target
                (rr[1].data['ZWARN'] == 0) &                        # no warning flags
                (rr[1].data['SPECTYPE'] == spectype) &              # is a galaxy according to redrock
                (rr[1].data['Z'] < 0.8))                            # Redshift < 0.8

        if target is not None: 
            target = target.upper()

            # see https://arxiv.org/pdf/2208.08518.pdf Section 2.2 for details
            # on the bitmask
            import desitarget # https://github.com/desihub/desitarget/

            if survey.lower() == "sv1":
                from desitarget.sv1.sv1_targetmask import desi_mask
            elif survey.lower() == "sv2":
                from desitarget.sv2.sv2_targetmask import desi_mask
            elif survey.lower() == "sv3":
                from desitarget.sv3.sv3_targetmask import desi_mask
            else: 
                raise ValueError("not included in EDR") 
            if target == 'BGS': 
                target = 'BGS_ANY'
            if target == 'MWS':
                target = 'MWS_ANY'

            keep = keep & (desi_mask[target] > 0)

            # redshift criteria for BGS and LRG. no additional criteria imposed
            # for ELG and QSO. ELG requires OII flux SNR; QSO has
            # afterburners...
            # see DESI Collaboration SV overview paper
            if target == "BGS":
                keep = keep & (
                    (rr[1].data["ZERR"] < 0.0005 * (1.0 + rr[1].data["Z"]))
                    & (rr[1].data["DELTACHI2"] > 40)  # low redshift error
                )  # chi2 difference with
            elif target == "LRG":
                keep = keep & (rr[1].data["DELTACHI2"] > 15)  # chi2 difference with

        # read in data
        _wave, _flux, _ivar, _mask, _res = {}, {}, {}, {}, {}
        for h in range(2, len(hdulist)):
            if "WAVELENGTH" in hdulist[h].header["EXTNAME"]:
                band = hdulist[h].header["EXTNAME"].split("_")[0].lower()
                _wave[band] = hdulist[h].data
            if "FLUX" in hdulist[h].header["EXTNAME"]:
                band = hdulist[h].header["EXTNAME"].split("_")[0].lower()
                _flux[band] = hdulist[h].data
            if "IVAR" in hdulist[h].header["EXTNAME"]:
                band = hdulist[h].header["EXTNAME"].split("_")[0].lower()
                _ivar[band] = hdulist[h].data
            if "MASK" in hdulist[h].header["EXTNAME"]:
                band = hdulist[h].header["EXTNAME"].split("_")[0].lower()
                _mask[band] = hdulist[h].data
            if "RESOLUTION" in hdulist[h].header["EXTNAME"]:
                band = hdulist[h].header["EXTNAME"].split("_")[0].lower()
                _res[band] = hdulist[h].data

        # coadd the b, r, z arm spectra (scraped from
        # https://github.com/desihub/desispec/blob/main/py/desispec/coaddition.py#L529)
        tolerance = 0.0001  # A , tolerance
        wave = _wave["b"]
        for b in ["b", "r", "z"]:
            wave = np.append(wave, _wave[b][_wave[b] > wave[-1] + tolerance])
        nwave = wave.size
        ntarget = _flux["b"].shape[0]
        check_agreement = torch.abs(torch.from_numpy(wave) - cls._wave_obs)
        if check_agreement.max() > tolerance:
            print(
                "Warning: input wavelength grids inconsistent with class variable wave_obs!"
            )
        # check alignment, caching band wavelength grid indices as we go
        windict = {}
        number_of_overlapping_cameras = np.zeros(nwave)
        for b in ["b", "r", "z"]:
            imin = np.argmin(np.abs(_wave[b][0] - wave))
            windices = np.arange(imin, imin + len(_wave[b]), dtype=int)
            dwave = _wave[b] - wave[windices]

            if np.any(np.abs(dwave) > tolerance):
                msg = "Input wavelength grids (band '{}') are not aligned. Use --lin-step or --log10-step to resample to a common grid.".format(
                    b
                )
                raise ValueError(msg)
            number_of_overlapping_cameras[windices] += 1
            windict[b] = windices

        # ndiag = max of all cameras
        ndiag = 0
        for b in ["b", "r", "z"]:
            ndiag = max(ndiag, _res[b].shape[1])

        flux = np.zeros((ntarget, nwave), dtype=_flux["b"].dtype)
        ivar = np.zeros((ntarget, nwave), dtype=_ivar["b"].dtype)
        ivar_unmasked = np.zeros((ntarget, nwave), dtype=_ivar["b"].dtype)
        mask = np.zeros((ntarget, nwave), dtype=_mask["b"].dtype)
        rdata = np.zeros((ntarget, ndiag, nwave), dtype=_res["b"].dtype)

        for b in ["b", "r", "z"]:
            # indices
            windices = windict[b]

            band_ndiag = _res[b].shape[1]

            for i in range(ntarget):
                ivar_unmasked[i, windices] += np.sum(_ivar[b][i], axis=0)
                ivar[i, windices] += _ivar[b][i] * (_mask[b][i] == 0)
                flux[i, windices] += _ivar[b][i] * (_mask[b][i] == 0) * _flux[b][i]
                for r in range(band_ndiag):
                    rdata[i, r + (ndiag - band_ndiag) // 2, windices] += (
                        _ivar[b][i] * _res[b][i, r]
                    )

                # directly copy mask where no overlap
                jj = number_of_overlapping_cameras[windices] == 1
                mask[i, windices[jj]] = _mask[b][i][jj]

                # 'and' in overlapping regions
                jj = number_of_overlapping_cameras[windices] > 1
                mask[i, windices[jj]] = mask[i, windices[jj]] & _mask[b][i][jj]

        for i in range(ntarget):
            ok = ivar[i] > 0
            if np.sum(ok) > 0:
                flux[i][ok] /= ivar[i][ok]
            ok = ivar_unmasked[i] > 0
            if np.sum(ok) > 0:
                rdata[i][:, ok] /= ivar_unmasked[i][ok]

        # apply bitmask, remove small values
        mask = mask.astype(bool) | (ivar <= 1e-6)
        ivar[mask] = 0

        # explicit type conversion to float32 to get to little endian
        spec = torch.from_numpy(flux.astype(np.float32))
        w = torch.from_numpy(ivar.astype(np.float32))
        target_id = torch.from_numpy(target_id.astype(np.int64))

        # remove regions around skylines
        w[:, cls._skyline_mask] = 0

        # normalize spectra:
        norm = torch.zeros(ntarget)
        for i in range(ntarget):
            # for redshift invariant encoder: select norm window in restframe
            wave_rest = cls._wave_obs / (1 + z[i])
            # flatish region that is well observed out to z ~ 0.5
            sel = (w[i] > 0) & (wave_rest > 5300) & (wave_rest < 5850)
            if sel.count_nonzero() > 0:
                norm[i] = torch.median(spec[i][sel])
            # remove spectra (from training) for which no valid norm could be found
            if not torch.isfinite(norm[i]):
                norm[i] = 0
            else:
                spec[i] /= norm[i]
            w[i] *= norm[i]**2

        # selects finite fluxes
        keep = keep & (spec.isfinite().sum(axis=-1) == nwave).numpy()
        print("keep: %d / %d"%(keep.sum(),len(keep)))
        return spec[keep], w[keep], z[keep], target_id[keep],  norm[keep], zerr[keep]

    @classmethod
    def query(
        cls, dir, target, fields=["SURVEY", "PROGRAM", "HEALPIX"], selection_fct=None
    ):
        """Select SURVEY, PROGRAM, HEALPIX from healpix look up table for healpix that
        match `selection_fct`. Look up table only includes information on
        TILEID, SURVEY, PROGRAM, PETAL_LOC, HEALPIX


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
            Tensor of fields for the selected healpix

        """
        main_file = os.path.join(dir, "tilepix.fits")
        if not os.path.isfile(main_file):
            url = "https://data.desi.lbl.gov/public/edr/spectro/redux/fuji/healpix/tilepix.fits"
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, main_file)

        print(f"opening {main_file}")
        thpix = aTable.Table.read(main_file)

        if target in ["LRG", "ELG", "QSO"]:
            program = "dark"
        elif target in ["BGS", "MWS"]:
            program = "bright"

        if selection_fct is None:
            # apply default selections
            sel = (thpix["SURVEY"] == "sv3") & (thpix["PROGRAM"] == program)
        else:
            sel = selection_fct(thpix)

        _, uind = np.unique(thpix[sel]["HEALPIX"], return_index=True)

        out_tab = thpix[fields][sel][uind]
        out_tab["TARGET"] = target
        return out_tab


    @classmethod
    def augment_spectra(cls, batch, redshift=True, noise=True, mask=False, ratio=0.05, z_new=None, z_max = 0.8):
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
                # uniform distribution of redshift offsets
                z_new = z_max*(torch.rand(batch_size, device=device))
            if z_max < 0.01:
                z_new = z_max*(torch.rand(batch_size, device=device)-0.5)

            zfactor = (1 + z) / (1 + z_new)
            # transform augments into the wavelengths of observed spectra
            wave_redshifted =  (wave_obs.unsqueeze(1) * zfactor).T

            # redshift interpolation
            spec_new = interp1d(wave_obs.repeat(batch_size,1), spec, wave_redshifted).float()
            # ensure extrapolated values have zero weights
            wmin = wave_obs.min()
            wmax = wave_obs.max()

            # ensure extrapolated values have zero weights
            w_new = torch.clone(w)
            w_new = interp1d(wave_obs.repeat(batch_size,1), w_new, wave_redshifted).float()

            out = (wave_redshifted<wmin)|(wave_redshifted>wmax)

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

        if redshift:
            spec_new[out] = 0
            w_new[out] = 0
        return spec_new, w_new, z_new


    @classmethod
    def get_image(
        cls,
        target_id: Optional[int] = None,
        dir: Optional[str] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        size: int = 256,
        pixscale: float = 0.262,
        bands: str = "griz",
    ) -> bytes:
        """
        Searches the Legacy Survey DR10 (which includes DECaLS, BASS and MzLS) for a JPEG cutout image.

        Parameters
        ==========
        one of:
            - target_id (DESI ID) and dir (directory containing the PROVABGS HDF5 file)
            - ra (deg) and dec (deg)
        size: image h/w in pixels, largest is 512
        pixscale: arcsec, 0.262" is native resolution
        bands: any combination of g,r,i,z

        Returns
        =======
        PIL image
        """
        if target_id is not None:
            assert dir is not None, "Must provide dir if target_id is not None"
            catalog_file = os.path.join(dir, "BGS_ANY_full.provabgs.sv3.v0.hdf5")
            with h5py.File(catalog_file, "r") as f:
                selection = f["__astropy_table__"]["TARGETID"][:] == target_id
                assert (
                    selection.sum() > 0
                ), f"target_id {target_id} not found in {catalog_file}"
                ra = f["__astropy_table__"]["RA"][selection][0]
                dec = f["__astropy_table__"]["DEC"][selection][0]
        else:
            assert (
                ra is not None and dec is not None
            ), "Must provide ra and dec if target_id and dir are None"

        url = f"https://www.legacysurvey.org/viewer/jpeg-cutout?ra={ra}&dec={dec}&layer=ls-dr10&size={size}&pixscale={pixscale}&bands={bands}"
        r = requests.get(url)
        return r.content
