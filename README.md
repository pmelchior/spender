# Spender

_Neural spectrum encoder and decoder_

* Paper I: https://arxiv.org/abs/2211.07890
* Paper II: https://arxiv.org/abs/2302.02496

From a data-driven side, galaxy spectra have two fundamental degrees of freedom: their instrinsic spectral properties (or type, if you believe in such a thing) and their redshift. The latter makes them awkward to ingest because it stretches everything, which means spectral features don't appear at the same places. This is why most analyses of the intrinsic properties are done by transforming the observed spectrum to restframe.

We decided to do the opposite. We build a custom architecture, which describes the restframe spectrum by an autoencoder and transforms the restframe model to the observed redshift. While we're at it we also match the spectral resolution and line spread function of the instrument:
![sketch](https://github.com/pmelchior/spender/assets/1463403/8e861c0b-358c-4b92-8862-e31325acae1b)

Doing so clearly separates the responsibilities in the architecture. Spender establishes a restframe that has higher resolution and larger wavelength range than the spectra from which it is trained. The model can be trained from spectra at different redshifts or even from different instruments without the need to standardize the observations. Spender also has an explicit, differentiable redshift dependence, which can be coupled with a redshift estimator for a fully data-driven spectrum analysis pipeline.

## Installation

The easiest way is `pip install spender`. When installing from the code repo, run `pip install -e .`.

For the time being, you will have to install one dependency manually: `torchinterp1d` is available [here](https://github.com/aliutkus/torchinterp1d).

## Pretrained models

We make the best-fitting models discussed in the paper available:
* [standard resolution](https://www.dropbox.com/s/6o5htaic8wimito/sdss.speculator%2B1.variable.lr_1e-3.latent_10.0.pt?dl=0) (S=10, R=5881)
* [super-resolution](https://www.dropbox.com/s/d14f1jryelxc5if/sdss.speculator%2B1.variable.superres.lsf_5.lr_1e-3.latent_8.0.pt?dl=0) (S=8, R=11762)
* [similarity+consistency training](https://www.dropbox.com/s/7ecvnbpc8do6pjy/sdss.similarity-consistency.latent_6.0.pt?dl=0) (S=6, R=7000; see Liang et al. 2023)

## SDSS Outliers Catalog

The [catalog of the latent-space probability](https://www.dropbox.com/s/2eo8r4mlsh7p15o/FULL_SDSSID_logP.txt.bz2?dl=0) for the SDSS-I main galaxy sample; see Liang et al. (2023) for details

## Use

Documentation and tutorials are forthcoming. In the meantime, check out `train/diagnostics.ipynb` for a worked through example that generates the figures from the paper.

In short, you can run spender like this:
```python
import spender
from spender.data.sdss import SDSS

# create the instrument
sdss = SDSS()

# load the model
model, loss = spender.load_model(model_file, sdss)

# get some SDSS spectra from the ids
data_path = "./DATA"
ids = ((412, 52254, 308), (412, 52250, 129))
spec, w, z, ids, norm, zerr = SDSS.make_batch(data_path, ids)

# run spender end-to-end
with torch.no_grad():
  spec_reco = model(spec, instrument=sdss, z=z)

# for more fine-grained control, run spender's internal _forward method
# which return the latents s, the model for the restframe and the observed spectrum
with torch.no_grad():
  s, spec_rest, spec_reco = model._forward(spec, instrument=sdss, z=z)

# to only encode into latents, using redshift as extra input
with torch.no_grad():
  s = model.encode(spec, aux=z.unsqueeze(1))
```

Plotting the results of the above nicely shows what spender can do:

![examples_2](https://user-images.githubusercontent.com/1463403/202062952-4a27dacf-2733-47d9-a9ca-e5b3387961e2.png)

Noteworthy aspects: The restframe model has an extended wavelength range, e.g. predicting the [O II] doublet that was not observed in the first example, and being unaffected by glitches like the skyline residuals at about 5840 A in the second example.
