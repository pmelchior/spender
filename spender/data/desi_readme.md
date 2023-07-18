# DESI EDR Primer for Astro Data Lab

[toc] 

## data directory

I believe the plan is to make https://data.desi.lbl.gov/public/edr public on Tuesday. Here's how the directory is organized. I've outlined some of the important subdirectories. 

```bash
├── /public/edr
│   ├── survey # fiber assignment info, guide field array details (probably not useful)
│   ├── target # target catalogs
│   ├── vac # value-added catalogs 
│   |   ├── edr
│   |   |   ├── vi # visual inspection catalogs for BGS, ELG, LRG, and QSO (may be useful for labeled data sets) 
│   ├── spectro
│   |   ├── redux
│   |   |   ├── fuji # spectral reduction version
│   |   |   │   ├── exposures-fuji.csv # deatils on exposure times
│   |   |   │   ├── exposures-fuji.fits
│   |   |   │   ├── tiles-fuji.csv # deatils on tiles (DESI pointings)
│   |   |   │   ├── tiles-fuji.fits
│   |   |   │   ├── exposures # directory includes all the data for given exposures.
															# includes psf, frame, sky, std stars, flux 
															# calibration, combined data. orangized by
															# date/exposure number/
│   |   |   │   ├── tiles # directory includes spectra organized by tile number
│   |   |   │   │   ├── cumulative # spectra cumulative across night
│   |   |   │   │   │   ├── TILE_NUMBER
│   |   |   │   │   │   │   ├── SOME_DATE
│   |   |   │   │   │   │   │   ├── spectra-* # spectra file 
│   |   |   │   │   │   │   │   ├── coadd-* # coadded spectra file
│   |   |   │   ├── healpix # directory includes spectra for a given healpix 
														# combined across all tiles
|   |   │   │   │   ├── tilepix.fits # table with tileid, healpix mapping
|   |   │   │   │   ├── sv1 # SV1 
|   |   │   │   │   ├── sv2
|   |   │   │   │   ├── sv3 # SV3 (One-Percent Survey) 
|   |   │   │   │   │   ├── dark # dark time programs (LRG, ELG, QSOs)
|   |   │   │   │   │   ├── bright # bright time programs (BGS, MWS) 
|   |   │   │   │   │   │   ├── FIRST_DIGITS_OF_HEALPIX
|   |   │   │   │   │   │   │   ├── HEALPIX_NUMBER
│   │   |   |   │   │   │   │   │   ├── spectra-* # spectra file 
│   │   |   |   │   │   │   │   │   ├── coadd-* # coadded spectra file
│   │   |   |   │   │   │   │   │   ├── redrock-* # redrock redshift fitter output
│   │   |   |   │   │   │   │   │   ├── emline-* 	# emission line fitter output
│   │   |   |   │   │   │   │   │   ├── qso_mgii-* # QSO Mg-II afterburner
│   │   |   |   │   │   │   │   │   ├── qso_qn-* # QSO Quasarnet afterburner 
```

references: 

- https://desidatamodel.readthedocs.io/en/latest/DESI_SPECTRO_REDUX/index.html#

## getting DESI data using `desi-spender`

First install two required DESI packages: 

install [desiutil](https://github.com/desihub/desiutil)

```bash
# in conda env
git clone https://github.com/desihub/desiutil.git
cd desiutil
pip install -e . 
```

install [desitarget](https://github.com/desihub/desitarget)

```bash
# in conda env
git clone https://github.com/desihub/desitarget.git
cd desitarget
pip install -e . 
```



Next, is a short snippet of how to download DESI data using the `spender.data.desi`. Rather than downloading spectra of galaxies individually, we'll download DESI spectra in chunks. Each chunk corresponds to all the spectra for a given survey, program, and healpix. 

- survey: different surveys conducted before the main survey — "sv1", "sv2", "sv3". You'll most likely want to use sv3, which is also referred to as the "One-Percent Survey" because it corresponds to roughly 1% of the full DESI data. A few things to note about sv3, exposure times were ~1.2x longer than the main survey so the spectra have slightly higher SNR. sv3 targeted a smaller area (180 deg^2) with many more passes than the main survey so the fiber assignment completeness is much higher than the main survey. 
- program: "dark" or "bright" time programs. Dark time includes LRG, ELG, and QSO targets. Bright time includes BGS and MWS targets. Note there are some repeats: e.g. LRGs in bright time. 
- healpix: healpix number. The `desi.query` script will automatically download the `tilepix.fits` table that includes info on which program and survey corresponds to which healpix.

```python
import numpy as np 
from spender.data import desi

desi = desi.DESI()

# get healpix info that corresponds to BGS spectra
bgs_hpixs = desi.query('.', 'BGS') 

# save the files to batches by
# 1. desi.get_spectra: download the coadd and redrock output files 
# 2. desi.prepare_spectra: keep only spectra observed with good fibers, specified target class, and 
#    has good redshifts
desi.save_in_batches('YOUR_DIR_HERE', bgs_hpixs, batch_size=1024)

```

A few notes

- in `desi.prepare_spectra` we only keep spectra with "good redshifts". By default this only keeps spectra with no redrock flags, and are classified as galaxies. For BGS, there's also an additional z uncertainty and a delta chi2 criteria. For LRG there's an additional delta chi2 criteria. 
  For ELG and QSO **proceed with caution.** There is no additional redshift cuts in `desi.prepare_spectra`, but the official DESI cuts includes additional steps. For ELG, this is a OII flux SNR + delta chi2 cut. For QSO, there are two QSO redshift afterburners. `get_spectra` will download the emission line data and the QSO afterburner outputs but they are not yet implemented. 
- Specifying BGS will download all BGS targets. This includes both BGS Bright and BGS Faint galaxies. BGS Bright is a r < 19.5 magnitude limited survey. BGS Faint is a sample of fainter galaxies selected based on a color proxy for emission lines. 