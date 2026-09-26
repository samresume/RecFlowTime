"""Configuration dataclasses for RecFlowTime.

RecFlowTime -- Path-signature and Rectified-flow with Implicit Spectral Matching.

The method is four things, and deliberately nothing else:

  1. **Rectified flow** as the generative core (`flow.py`). The centerpiece.
     It does not merely replace DDPM; it *removes* machinery. Recovering a
     clean sequence from a noisy one under the eps-parameterisation needs
     `(x_tau - sqrt(1-abar)*eps)/sqrt(abar)`, whose division blows up as
     `abar -> 0`, and every implementation that differentiates through it
     ends up carrying the same three mitigations: restricting the auxiliary
     timestep to a low-noise slice, clamping intermediate clean-sequence
     estimates, and truncating backprop to the last few low-noise sampler
     steps. Under the straight-line interpolation `x_t = (1-t)x + t*eps`,
     the same quantity is `x_hat = x_t - t*v_theta(x_t, t)` -- no division
     at any t, Jacobian bounded by t <= 1 -- so all three mitigations become
     unnecessary by construction rather than by tuning.

  2. **Signature-MMD** for global distributional alignment
     (`distributional.py`). An RBF kernel on a flattened (T*F,) vector
     treats the sequence as an unordered bag of coordinates; the
     depth-truncated path signature is graded precisely by order-sensitive
     information (level 1 = net displacement, level 2 = signed area /
     lead-lag between every feature pair, and so on). This is the piece
     with the highest conceptual novelty and the least confirmed empirical
     payoff -- and the one that needed a genuine bug fix during development
     (per-level rather than per-coordinate normalisation; see that module).

  3. **A parameter-free spectral discrepancy** (`spectral.py`): the batch
     mean and standard deviation of the log power spectrum, matched
     two-sided to an EMA reference from real data rather than minimized.
     No network, no pretraining stage, no adversarial game.

  4. **Standard modern training machinery**: RoPE, self-conditioning, and
     min-SNR loss weighting. Borrowed wholesale from RoFormer / Analog Bits
     / Hang et al. -- legitimate engineering that makes the model train
     better, explicitly *not* claimed as a contribution.

Note what is absent. RecFlowTime has **no auxiliary critic networks** -- no
masked interpolation/extrapolation models, no critic pretraining stage, no
learned discriminator of any kind. Both regularisers are non-parametric
statistical discrepancies computed directly from batches. The local
temporal coherence that masked critics were introduced to supply is
covered instead by the spectral term: by Wiener-Khinchin the power spectrum
is the Fourier transform of the autocorrelation, so matching the log
spectrum constrains second-order temporal structure directly, and the
signature term covers higher-order and cross-feature ordering structure on
top of that.

`RecFlowTimeConfig.tide_baseline(...)` builds the prior method (DDPM+DDIM, frozen
masked critics with a two-sided discrepancy, raw flattened-vector MMD) on
the same data pipeline, model family, and metrics -- as a *baseline to
compare against*, not as part of RecFlowTime.
"""
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional
import json


@dataclass
class DenoiserConfig:
    """Velocity field v_theta(x_t, t, x_self_cond) -- or eps_theta for the
    DDPM baseline. A bidirectional transformer over timesteps; standard
    architecture, not a claimed contribution."""
    n_features: int = 4
    seq_len: int = 24
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 4
    d_ff_mult: int = 4
    dropout: float = 0.0
    causal: bool = False
    cond_mode: Literal["add", "adaln"] = "add"
    use_rope: bool = True           # rotary position embedding (component 4)
    use_self_cond: bool = True      # self-conditioning (component 4)
    use_lag_bias: bool = False      # learned per-head bias over relative lags


@dataclass
class AuxConfig:
    """Masked critics. OFF in RecFlowTime -- present only to build the TIDE
    baseline for comparison."""
    enabled: bool = False
    n_features: int = 4
    seq_len: int = 24
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.0
    gamma_range: tuple = (0.20, 0.35)


@dataclass
class SpectralConfig:
    """Parameter-free frequency-domain discrepancy (component 3)."""
    enabled: bool = True
    n_bins: Optional[int] = None      # None -> all rFFT bins up to Nyquist
    eps: float = 1e-6                 # log(power + eps)
    ref_ema_decay: float = 0.99


@dataclass
class CoreConfig:
    """Generative core: rectified flow (RecFlowTime, component 1) or DDPM (baseline)."""
    kind: Literal["flow", "ddpm"] = "flow"
    n_steps: int = 500                # DDPM schedule length (ddpm only)
    schedule: Literal["cosine", "linear"] = "cosine"
    cosine_s: float = 0.008
    aux_tau_frac: float = 0.30        # low-noise slice for x_hat (ddpm only; see flow.py)
    min_snr_gamma: float = 5.0        # min-SNR loss weighting cap (component 4)
    min_snr_floor: float = 0.0        # lower bound on the min-SNR weight; 0 = unfloored
    use_min_snr: bool = True
    # Temporal preconditioner W = I + lambda*D on the flow-matching residual.
    # 0 disables it; lambda = 1 is excluded because W is then singular.
    precond_lambda: float = 0.0
    precond_lags: tuple = (1,)        # lags used by the difference metric
    colored_prior: bool = False       # prior noise matched to the data spectrum
    # Pair noise to data within the batch by optimal transport (flow.py: couple)
    ot_coupling: bool = False
    sampling_steps: int = 20          # ODE/DDIM steps at generation time


@dataclass
class DistConfig:
    """Global distributional alignment: signature-MMD (RecFlowTime, component 2)
    or raw flattened-vector MMD (baseline).

    `unbiased=False` (the V-statistic) is the default, and this is not a
    detail. The unbiased U-statistic is the textbook choice because it
    estimates MMD^2 without bias -- but it is not guaranteed non-negative,
    and its variance does not shrink just because the two distributions are
    close. Measured here at the training batch size of 64, between two
    *real* batches (true discrepancy ~0) it returns a negative value 67-75%
    of the time with a noise std of 0.025-0.040. Once the generator is good
    enough that the true discrepancy falls into that band, clamping the
    estimate at zero -- which is necessary, since a negative value is
    meaningless as a minimisation target -- zeroes the *gradient* too, and
    the term silently stops training the model. A first full-budget run of
    RecFlowTime did exactly this: `dist` logged as 0.00000 at nearly every step on
    all four datasets, so signature-MMD contributed nothing to those
    results.

    The V-statistic is biased upward by O(1/n) but is a sum of positive
    kernel terms, so it is non-negative by construction, never hits the
    clamp, and carries a smooth non-zero gradient throughout. The bias is a
    slowly-varying offset that does not change what the gradient points at,
    which is what matters for optimisation (the same reasoning is standard
    in MMD-GAN-style training). `unbiased=True` remains available for
    *evaluation*, where an unbiased point estimate is what you want and no
    gradient is involved.
    """
    kind: Literal["sigmmd", "mmd"] = "sigmmd"
    bandwidth_scales: tuple = (0.25, 0.5, 1.0, 2.0, 4.0)
    unbiased: bool = False
    clamp_min: float = 0.0
    sig_depth: int = 3                 # truncated signature depth (sigmmd only)
    sig_time_aug: bool = True          # append a clock channel before the signature


@dataclass
class TrainConfig:
    batch_size: int = 128
    lr: float = 2e-4
    aux_lr: float = 1e-3
    betas: tuple = (0.9, 0.96)
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    ema_decay: float = 0.999

    aux_steps: int = 2000             # critic pretraining; baseline only
    warmup_steps: int = 20000
    joint_steps: int = 6000

    lambda_ext: float = 1.0           # baseline only (no critics in RecFlowTime)
    lambda_int: float = 1.0           # baseline only
    lambda_spec: float = 1.0
    lambda_dist: float = 1.0

    ref_ema_decay: float = 0.99

    dist_every: int = 4
    dist_sample_steps: int = 16
    dist_grad_steps: int = 4
    dist_batch: int = 64
    sample_clamp: float = 1.5

    log_every: int = 250
    val_every: int = 500
    seed: int = 0
    device: Optional[str] = None


@dataclass
class RecFlowTimeConfig:
    denoiser: DenoiserConfig = field(default_factory=DenoiserConfig)
    aux: AuxConfig = field(default_factory=AuxConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    core: CoreConfig = field(default_factory=CoreConfig)
    dist: DistConfig = field(default_factory=DistConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def for_dataset(cls, seq_len: int, n_features: int, **overrides):
        """RecFlowTime itself: rectified flow + signature-MMD + spectral
        discrepancy + (RoPE, self-conditioning, min-SNR). No critics."""
        cfg = cls()
        cfg.denoiser.seq_len = seq_len
        cfg.denoiser.n_features = n_features
        cfg.aux.seq_len = seq_len
        cfg.aux.n_features = n_features
        for k, v in overrides.items():
            head, _, tail = k.partition(".")
            if tail:
                setattr(getattr(cfg, head), tail, v)
            else:
                setattr(cfg, k, v)
        return cfg

    @classmethod
    def tide_baseline(cls, seq_len: int, n_features: int, **overrides):
        """The prior method (TIDE) as a comparison baseline: DDPM+DDIM core,
        two frozen masked critics with a two-sided discrepancy loss, raw
        flattened-vector MMD, fixed sinusoidal positions, no
        self-conditioning, no spectral term."""
        cfg = cls.for_dataset(seq_len, n_features, **overrides)
        cfg.aux.enabled = True
        cfg.denoiser.use_rope = False
        cfg.denoiser.use_self_cond = False
        cfg.spectral.enabled = False
        cfg.core.kind = "ddpm"
        cfg.core.use_min_snr = False
        cfg.dist.kind = "mmd"
        return cfg

    @classmethod
    def ablation(cls, seq_len: int, n_features: int, *, signature=True,
                 spectral=True, flow=True, modern=True, **overrides):
        """Leave-one-out ablation arms over RecFlowTime's four components."""
        cfg = cls.for_dataset(seq_len, n_features, **overrides)
        if not signature:
            cfg.dist.kind = "mmd"
        if not spectral:
            cfg.spectral.enabled = False
        if not flow:
            cfg.core.kind = "ddpm"
        if not modern:
            cfg.denoiser.use_rope = False
            cfg.denoiser.use_self_cond = False
            cfg.core.use_min_snr = False
        return cfg

    def to_json(self, indent=2):
        return json.dumps(asdict(self), indent=indent, default=str)
