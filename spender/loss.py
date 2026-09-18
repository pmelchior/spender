from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .util import resample_to_restframe


@dataclass
class LossConfig:
    """Configuration for the similarity and consistency losses

    Collects the hyperparameters of :func:`consistency_loss`, :func:`similarity_loss`,
    :func:`restframe_weight`, and :func:`similarity_restframe` in one place, so that
    training scripts can adjust them without editing the loss functions themselves.

    `consistency_amp` and `similarity_amp` are not applied inside :func:`consistency_loss`
    or :func:`similarity_restframe_loss` themselves: :func:`get_losses` returns the raw,
    unweighted losses, and it is up to the caller (typically the training script) to
    combine them with these amplitudes into the loss used for backpropagation.

    Parameters
    ----------
    consistency_amp: float
        Weight of the consistency loss relative to the fidelity loss, applied by the
        caller when combining losses for backpropagation
    consistency_tol: float
        Tolerance of :func:`consistency_loss` for latent drift under augmentation;
        smaller values penalize drift more strongly
    similarity_wid: float
        Width of the no-penalty region around equal (dis)similarity in
        :func:`similarity_loss`
    similarity_amp: float
        Weight of the similarity loss relative to the fidelity loss, applied by the
        caller when combining losses for backpropagation
    similarity_data_amp: float
        Weighting of the spectrum dissimilarity relative to the latent dissimilarity
        inside the similarity loss
    restframe_mu: float
        Center of the :func:`restframe_weight` Gaussian window, in restframe wavelength
    restframe_sigma: float
        Width of the :func:`restframe_weight` Gaussian window
    restframe_wid: float
        Width of the no-penalty region around equal (dis)similarity in
        :func:`similarity_restframe`
    restframe_bound: tuple of float, length 2
        Restframe wavelength range used by :func:`similarity_restframe` to normalize
        decoded spectra before comparison
    similarity_slope: float
        Steepness of the sigmoid that compares latent and spectral dissimilarity in
        :func:`similarity_restframe_loss`. Training scripts typically anneal this
        value over the course of training by updating this field directly.
    """
    consistency_amp: float = 1
    consistency_tol: float = 0.5
    similarity_wid: float = 5
    similarity_amp: float = 3
    similarity_data_amp: float = 30
    restframe_mu: float = 5000
    restframe_sigma: float = 2000
    restframe_wid: float = 5
    restframe_bound: Tuple[float, float] = (4000, 7000)
    similarity_slope: float = 1.0


def consistency_loss(s, s_aug, individual=False, config: Optional[LossConfig] = None):
    """Consistency loss between latents of original and augmented spectra

    Penalizes latents that change under data augmentation (e.g. added noise or
    redshifting) of the same underlying spectrum. The squared latent distance is
    passed through a sigmoid and centered so that a value of zero indicates perfect
    alignment between `s` and `s_aug`.

    Parameters
    ----------
    s: `torch.tensor`, shape (N, S)
        Batch of latents of the original spectra
    s_aug: `torch.tensor`, shape (N, S)
        Batch of latents of the augmented versions of the same spectra
    individual: bool
        Whether the per-sample discrepancy and loss are returned instead of the
        aggregated loss
    config: :class:`LossConfig`
        Loss hyperparameters. Uses `LossConfig.consistency_tol` as the alignment
        tolerance. If `None`, the defaults of :class:`LossConfig` are used.

    Returns
    -------
    x: `torch.tensor`, shape (N,)
        Per-sample squared latent distance, in units of the alignment tolerance
        (only returned if `individual`)
    sim_loss: `torch.tensor`, shape (N,), or float
        Consistency loss; zero indicates perfect alignment between `s` and `s_aug`.
        If `individual` is False, this is summed over the batch into a single float.
    """
    if config is None:
        config = LossConfig()
    batch_size, s_size = s.shape
    x = torch.sum((s_aug - s) ** 2 / config.consistency_tol ** 2, dim=1) / s_size
    sim_loss = torch.sigmoid(x) - 0.5  # zero = perfect alignment
    if individual:
        return x, sim_loss
    return sim_loss.sum()


def similarity_loss(instrument, model, spec, w, z, s, slope=0.5, individual=False, config: Optional[LossConfig] = None):
    """Similarity loss between observed spectra and their latents

    Compares the pairwise dissimilarity of observed spectra (resampled to the
    restframe and weighted by their combined inverse variance) to the pairwise
    dissimilarity of the corresponding latents. Pairs whose spectral and latent
    (dis)similarities disagree are penalized, which encourages the latent space to
    preserve the (dis)similarity structure of the data.

    Parameters
    ----------
    instrument: :class:`spender.Instrument`
        Instrument that observed `spec`
    model: :class:`spender.BaseAutoencoder`
        Autoencoder model providing the restframe wavelength grid used for
        resampling `spec`
    spec: `torch.tensor`, shape (N, L)
        Batch of observed spectra
    w: `torch.tensor`, shape (N, L)
        Batch of inverse-variance weights for `spec`
    z: `torch.tensor`, shape (N, 1)
        Redshifts for each spectrum, used to resample `spec` into the restframe
    s: `torch.tensor`, shape (N, S)
        Batch of latents that encode `spec`
    slope: float
        Steepness of the sigmoid that compares latent and spectral dissimilarity
    individual: bool
        Whether the pairwise dissimilarities are returned instead of the
        aggregated loss
    config: :class:`LossConfig`
        Loss hyperparameters. Uses `LossConfig.similarity_wid`. If `None`, the
        defaults of :class:`LossConfig` are used.

    Returns
    -------
    s_sim: `torch.tensor`, shape (N, N)
        Pairwise dissimilarity of latents (only returned if `individual`)
    spec_sim: `torch.tensor`, shape (N, N)
        Pairwise dissimilarity of restframe spectra, weighted by their combined
        variance (only returned if `individual`)
    sim_loss: `torch.tensor`, shape (N, N), or float
        Similarity loss (not weighted by `LossConfig.similarity_amp`) that penalizes
        latent and spectral (dis)similarities that disagree; the diagonal is masked
        out. If `individual` is False, this is summed and normalized into a single
        float.
    """
    if config is None:
        config = LossConfig()
    spec, w = resample_to_restframe(instrument.wave_obs, model.decoder.wave_rest, spec, w, z)

    batch_size, spec_size = spec.shape
    _, s_size = s.shape
    device = s.device

    # pairwise dissimilarity of spectra
    S = (spec[None, :, :] - spec[:, None, :]) ** 2

    # pairwise weights
    non_zero = w > 1e-6
    N = non_zero[None, :, :] * non_zero[:, None, :]
    W = (1 / w)[None, :, :] + (1 / w)[:, None, :]
    W = N / W

    N = N.sum(-1)
    N[N == 0] = 1
    # dissimilarity of spectra
    # of order unity, larger for spectrum pairs with more comparable bins
    spec_sim = (W * S).sum(-1) / N

    # dissimilarity of latents
    s_sim = ((s[None, :, :] - s[:, None, :]) ** 2).sum(-1) / s_size

    # only give large loss of (dis)similarities are different (either way)
    x = s_sim - spec_sim
    sim_loss = torch.sigmoid(slope * x - 0.5 * config.similarity_wid) + torch.sigmoid(
        -slope * x - 0.5 * config.similarity_wid
    )
    diag_mask = torch.diag(torch.ones(batch_size, device=device, dtype=bool))
    sim_loss[diag_mask] = 0

    if individual:
        return s_sim, spec_sim, sim_loss
    # total loss: sum over N^2 terms,
    # needs to have amplitude of N terms to compare to fidelity loss
    return sim_loss.sum() / batch_size


def restframe_weight(model, config: Optional[LossConfig] = None):
    """Gaussian weighting function over the restframe wavelength grid

    Used by :func:`similarity_restframe` to emphasize a particular restframe
    wavelength region when comparing decoded spectra.

    Parameters
    ----------
    model: :class:`spender.BaseAutoencoder`
        Autoencoder model providing the restframe wavelength grid
    config: :class:`LossConfig`
        Loss hyperparameters. Uses `LossConfig.restframe_mu`, `restframe_sigma`.
        If `None`, the defaults of :class:`LossConfig` are used.

    Returns
    -------
    `torch.tensor`, shape (L,)
        Weight for every restframe wavelength bin, largest around `config.restframe_mu`
    """
    if config is None:
        config = LossConfig()
    x = model.decoder.wave_rest
    return torch.exp(-(0.5 * (x - config.restframe_mu) / config.restframe_sigma) ** 2)


def similarity_restframe_loss(model, s=None, individual=False, config: Optional[LossConfig] = None):
    """Similarity loss between decoded restframe spectra and their latents

    Like :func:`similarity_loss`, but compares decoded restframe spectra, normalized
    by their median flux in `config.restframe_bound`, instead of observed spectra.
    This avoids the need to resample observations and lets the loss be weighted by
    :func:`restframe_weight` rather than by observational noise.

    Parameters
    ----------
    model: :class:`spender.BaseAutoencoder`
        Autoencoder model used to decode `s` into restframe spectra
    s: `torch.tensor`, shape (N, S)
        Batch of latents to decode and compare
    individual: bool
        Whether the pairwise dissimilarities are returned instead of the
        aggregated loss
    config: :class:`LossConfig`
        Loss hyperparameters. Uses `LossConfig.similarity_slope`,
        `LossConfig.similarity_data_amp`, `LossConfig.similarity_wid`, and
        `LossConfig.restframe_bound`, and is forwarded to :func:`restframe_weight`.
        If `None`, the defaults of :class:`LossConfig` are used.

    Returns
    -------
    s_sim: `torch.tensor`, shape (N, N)
        Pairwise dissimilarity of latents (only returned if `individual`)
    spec_sim: `torch.tensor`, shape (N, N)
        Pairwise dissimilarity of decoded restframe spectra, weighted by
        `restframe_weight` (only returned if `individual`)
    sim_loss: `torch.tensor`, shape (N, N), or float
        Similarity loss (not weighted by `LossConfig.similarity_amp`) that penalizes
        latent and spectral (dis)similarities that disagree; the diagonal is masked
        out. If `individual` is False, this is summed and normalized into a single
        float.
    """
    if config is None:
        config = LossConfig()
    _, s_size = s.shape
    device = s.device

    spec = model.decode(s)
    wave = model.decoder.wave_rest
    bound = config.restframe_bound
    mask = (wave > bound[0]) * (wave < bound[1])
    spec /= spec[:, mask].median(dim=1)[0][:, None]
    batch_size, spec_size = spec.shape
    # pairwise dissimilarity of spectra
    S = (spec[None, :, :] - spec[:, None, :]) ** 2
    # dissimilarity of spectra
    # of order unity, larger for spectrum pairs with more comparable bins
    W = restframe_weight(model, config=config)
    spec_sim = (W * S).sum(-1) / spec_size
    # dissimilarity of latents
    s_sim = ((s[None, :, :] - s[:, None, :]) ** 2).sum(-1) / s_size

    # only give large loss of (dis)similarities are different (either way)
    x = s_sim - config.similarity_data_amp * spec_sim
    slope = config.similarity_slope
    sim_loss = torch.sigmoid(slope * x - 0.5 * config.similarity_wid) + torch.sigmoid(
        -slope * x - 0.5 * config.similarity_wid
    )
    diag_mask = torch.diag(torch.ones(batch_size, device=device, dtype=bool))
    sim_loss[diag_mask] = 0

    if individual:
        return s_sim, spec_sim, sim_loss

    # total loss: sum over N^2 terms,
    # needs to have amplitude of N terms to compare to fidelity loss
    return sim_loss.sum() / batch_size

def get_losses(model,
               instrument,
               batch,
               aug_fct=None,
               similarity=True,
               consistency=True,
               loss_config=None,
):
    """Fidelity, similarity, and consistency losses for a batch

    Convenience wrapper that computes the three losses training scripts typically
    need for a batch: the model's fidelity loss, :func:`similarity_restframe_loss`,
    and :func:`consistency_loss`. The similarity and consistency losses are returned
    raw, without the `LossConfig.similarity_amp` and `LossConfig.consistency_amp`
    weights applied; the caller is responsible for combining the three returned
    losses (e.g. `loss + config.similarity_amp * sim_loss +
    config.consistency_amp * cons_loss`) into the value used for backpropagation.

    Parameters
    ----------
    model: :class:`spender.BaseAutoencoder`
        Autoencoder model to encode and evaluate `batch` with
    instrument: :class:`spender.Instrument`
        Instrument that observed `batch`
    batch: tuple of `torch.tensor`
        `(spec, w, z)` tuple of observed spectra, inverse-variance weights, and
        redshifts, as produced by the data loader
    aug_fct: callable
        (optional) Function that returns an augmented copy of `batch`, used to
        compute the consistency loss. If `None`, the consistency loss is not
        computed
    similarity: bool
        Whether to compute the similarity loss
    consistency: bool
        Whether to compute the consistency loss (requires `aug_fct`)
    loss_config: :class:`LossConfig`
        Loss hyperparameters, forwarded to :func:`similarity_restframe_loss` and
        :func:`consistency_loss`. If `None`, the defaults of :class:`LossConfig`
        are used.

    Returns
    -------
    loss: `torch.tensor`
        Fidelity loss between `batch` and its reconstruction
    sim_loss: `torch.tensor`, or float
        Similarity loss, not weighted by `LossConfig.similarity_amp`; 0 if
        `similarity` is False
    cons_loss: `torch.tensor`, or float
        Consistency loss, not weighted by `LossConfig.consistency_amp`; 0 if
        `consistency` is False or `aug_fct` is `None`
    """
    spec, w, z = batch
    s = model.encode(spec)
    loss = model.loss(spec, w, instrument, z=z, s=s)

    if similarity:
        sim_loss = similarity_restframe_loss(model, s, config=loss_config)
    else:
        sim_loss = 0

    if consistency and aug_fct is not None:
        spec_, w_, z_ = aug_fct(batch)
        s_ = model.encode(spec_)
        cons_loss = consistency_loss(s, s_, config=loss_config)
    else:
        cons_loss = 0

    return loss, sim_loss, cons_loss
