"""RecFlowTime: Path-signature and Rectified-flow with Implicit Spectral Matching.

A generative model for multivariate time series built from four pieces and
nothing else:

  flow.py            **Rectified flow** as the generative core -- the
                     centerpiece. The clean-sequence estimate is
                     `x_hat = x_t - t*v_theta(x_t, t)`: division-free at
                     every t, so the three mitigations an eps-parameterised
                     DDPM needs to differentiate through its own sampler
                     (low-noise timestep restriction, value clamping,
                     gradient-step truncation) are unnecessary by
                     construction rather than by tuning.
  distributional.py  **Signature-MMD** -- two-sample distributional
                     alignment in a depth-truncated path-signature feature
                     space, which is graded by order-sensitive structure
                     (net displacement, then signed area / lead-lag between
                     feature pairs, ...) that an RBF kernel on a flattened
                     (T*F,) vector cannot see.
  spectral.py        **A parameter-free spectral discrepancy** -- batch mean
                     and std of the log power spectrum, matched two-sided to
                     an EMA reference from real data rather than minimized.
  modules.py /       **Standard training machinery**: RoPE,
  denoiser.py        self-conditioning, min-SNR loss weighting. Borrowed
                     from RoFormer / Analog Bits / Hang et al.; makes the
                     model train better, not claimed as a contribution.

There are **no auxiliary critic networks** anywhere in the method: no
masked interpolation/extrapolation models, no discriminator, no critic
pretraining stage, no adversarial game. Both regularisers are non-parametric
statistics of a batch. The local temporal coherence that masked critics are
usually introduced to supply falls out of the spectral term instead -- by
Wiener-Khinchin the power spectrum is the Fourier transform of the
autocorrelation, so matching the log spectrum constrains second-order
temporal structure directly -- with the signature term covering higher-order
and cross-feature ordering structure on top.

`aux_nets.py`, `masking.py` and `diffusion.py` exist only to construct the
prior-method baseline (`RecFlowTimeConfig.tide_baseline(...)`) that RecFlowTime is
compared against on the same data, model family, and metrics.
"""
from .config import (RecFlowTimeConfig, DenoiserConfig, AuxConfig, SpectralConfig,
                     CoreConfig, DistConfig, TrainConfig)
from .denoiser import TransformerDenoiser
from .flow import RectifiedFlow
from .spectral import SpectralDiscrepancy
from .losses import RecFlowTimeLoss
from .distributional import (mmd2, raw_mmd2, signature_mmd2, signature_features,
                             path_signature, distributional_loss, mmd_sanity_check)
from .train import RecFlowTimeTrainer, build_and_train, pick_device, EMA
from .aux_nets import AuxCritics, MaskedSeqModel          # baseline only
from .diffusion import GaussianDiffusion                   # baseline only
from . import data, metrics, masking, viz

__version__ = "0.1.0"
__all__ = [
    "RecFlowTimeConfig", "DenoiserConfig", "AuxConfig", "SpectralConfig", "CoreConfig",
    "DistConfig", "TrainConfig", "TransformerDenoiser", "RectifiedFlow",
    "SpectralDiscrepancy", "RecFlowTimeLoss", "mmd2", "raw_mmd2", "signature_mmd2",
    "signature_features", "path_signature", "distributional_loss", "mmd_sanity_check",
    "RecFlowTimeTrainer", "build_and_train", "pick_device", "EMA",
    "AuxCritics", "MaskedSeqModel", "GaussianDiffusion",
    "data", "metrics", "masking", "viz",
]
