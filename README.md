# Spender

_Neural spectrum encoder and decoder_

* Paper I (SDSS): Peter Melchior et al. (2023) [AJ 166 74](https://doi.org/10.3847/1538-3881/ace0ff)
* Paper II (SDSS): Yan Liang et al (2023) [AJ 166 75](https://doi.org/10.3847/1538-3881/ace100)
* Paper III (DESI EDR): Yan Liang et al (2023) [ApJL 956 L6](https://doi.org/10.3847/2041-8213/acfa03)

From a data-driven side, galaxy spectra have two fundamental degrees of freedom: their instrinsic spectral properties (or type, if you believe in such a thing) and their redshift. The latter makes them awkward to ingest because it stretches everything, which means spectral features don't appear at the same places. This is why most analyses of the intrinsic properties are done by transforming the observed spectrum to restframe.

We decided to do the opposite. We build a custom architecture, which describes the restframe spectrum by an autoencoder and transforms the restframe model to the observed redshift. While we're at it we also match the spectral resolution and line spread function of the instrument:
![sketch](https://github.com/pmelchior/spender/assets/1463403/8e861c0b-358c-4b92-8862-e31325acae1b)

Doing so clearly separates the responsibilities in the architecture. Spender establishes a restframe that has higher resolution and larger wavelength range than the spectra from which it is trained. The model can be trained from spectra at different redshifts or even from different instruments without the need to standardize the observations. Spender also has an explicit, differentiable redshift dependence, which can be coupled with a redshift estimator for a fully data-driven spectrum analysis pipeline.

## Installation

The easiest way is `pip install spender`. When installing from a downloaded code repo, run `pip install -e .`.

## Pretrained models

We make the best-fitting models discussed in the paper available through the Astro Data Lab Hub. Here's the workflow:

```python
import os
import spender

# show list of pretrained models
spender.hub.list()

# print out details for SDSS model from paper II
print(spender.hub.help('sdss_II'))

# load instrument and spectrum model from the hub
sdss, model = spender.hub.load('sdss_II')

# if your machine does not have GPUs, specify the device
from accelerate import Accelerator
accelerator = Accelerator(mixed_precision='fp16')
sdss, model = spender.hub.load('sdss_II', map_location=accelerator.device)
```
 
## Outliers Catalogs

Catalogs of latent-space probabilities, stored as Parquet files on the [spender-catalogs](https://huggingface.co/datasets/pmelchior/spender-catalogs) dataset repo:
* SDSS-I main galaxy sample, keyed by `PLATE-MJD-FIBERID`; see Liang et al. (2023a) for details
* DESI EDR BGS sample, keyed by `target_id`; see Liang et al. (2023b) for details

```python
from huggingface_hub import hf_hub_download
import pandas as pd

path = hf_hub_download(repo_id="pmelchior/spender-catalogs", repo_type="dataset", filename="spender.sdss.paperII.logP.parquet")
catalog = pd.read_parquet(path)
```

## Use

Documentation and tutorials are forthcoming. In the meantime, check out `train/diagnostics.ipynb` for a worked through example that generates the figures from the paper.

In short, you can run spender like this:
```python
import os
import spender
import torch
from accelerate import Accelerator

# hardware optimization
accelerator = Accelerator(mixed_precision='fp16')

# get code, instrument, and pretrained spectrum model from the hub
sdss, model = spender.hub.load('sdss_II',  map_location=accelerator.device)

# get some SDSS spectra from the ids, store locally in data_path
data_path = "./DATA"
ids = ((412, 52254, 308), (412, 52250, 129))
spec, w, z, norm, zerr = sdss.make_batch(data_path, ids)

# run spender end-to-end
with torch.no_grad():
  spec_reco = model(spec, instrument=sdss, z=z)

# for more fine-grained control, run spender's internal _forward method
# which return the latents s, the model for the restframe, and the observed spectrum
with torch.no_grad():
  s, spec_rest, spec_reco = model._forward(spec, instrument=sdss, z=z)

# only encode into latents
with torch.no_grad():
  s = model.encode(spec)
```

Plotting the results of the above nicely shows what spender can do:

![examples_2](https://user-images.githubusercontent.com/1463403/202062952-4a27dacf-2733-47d9-a9ca-e5b3387961e2.png)

Noteworthy aspects: The restframe model has an extended wavelength range, e.g. predicting the [O II] doublet that was not observed in the first example, and being unaffected by glitches like the skyline residuals at about 5840 A in the second example.

In addition, the latents vectors `s` form a highly informative distribution, from which we can read off physical properties like star-formation rate (e.g. the H-alpha intensity) in a redshift-independent way:

![embedding](https://github.com/user-attachments/assets/8448f916-a933-47fb-92cc-aba199e38adf)


## Citation

If you make use of this code, please cite the following paper:

```
@ARTICLE{2023AJ....166...74M,
       author = {{Melchior}, Peter and {Liang}, Yan and {Hahn}, ChangHoon and {Goulding}, Andy},
        title = "{Autoencoding Galaxy Spectra. I. Architecture}",
      journal = {Astronomical Journal},
         year = 2023,
        month = aug,
       volume = {166},
       number = {2},
          eid = {74},
        pages = {74},
          doi = {10.3847/1538-3881/ace0ff},
archivePrefix = {arXiv},
       eprint = {2211.07890},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023AJ....166...74M},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

If you use specific models from the hub, please cite the papers listed by `spender.hub.help(model_name)`.
