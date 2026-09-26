"""RecFlowTime: transport-coupled rectified flow for time-series generation.

A rectified-flow generator that differs from the standard recipe in one
respect: instead of pairing each data sequence with an independently drawn
noise sequence, it solves a linear assignment problem inside every minibatch
and trains on the resulting pairs. The coupling adds no parameters, leaves
both marginals exactly unchanged, and its cost is O(B^3) in the batch size and
independent of sequence length. It straightens the marginal velocity field,
which is what allows sampling in 20 function evaluations.

  flow.py       the rectified-flow core: the transport coupling, the floored
                min-SNR weight, Euler sampling, and the division-free clean
                estimate x_hat = x_t - t*v_theta(x_t, t)
  conditional.py  imputation and forecasting from the same trained model, by
                masked replacement inside the sampling loop
  denoiser.py   the Transformer velocity field
  modules.py    attention, RoPE, time embedding
  config.py     every configuration dataclass
  train.py      training loop, EMA, sampling
  metrics.py    the evaluation metrics
  ts2vec.py     the TS2Vec representation behind Context-FID
  data.py       the four benchmarks and their splits
  diffusion.py  a DDPM/DDIM core, used only by the `ddpm_core` ablation arm

`spectral.py`, `distributional.py`, `masking.py` and `aux_nets.py` implement
optional objective terms that are disabled in the reported configuration; see
`losses.py`.
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
from .aux_nets import AuxCritics, MaskedSeqModel          # optional, off by default
from .diffusion import GaussianDiffusion                   # the ddpm_core arm
from .conditional import (impute_mask, forecast_mask, complete, complete_k,
                          scores, linear_fill, last_value_fill)
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
