"""Configuration for RecFlowTime.

One dataclass per component, assembled by `RecFlowTimeConfig`. The defaults are
the reported configuration, so

    RecFlowTimeConfig.for_dataset(seq_len, n_features)

is the method as evaluated in the paper: rectified flow with the transport
coupling, RoPE, self-conditioning, the floored min-SNR weight, 20 sampling
steps, and no additional objective terms. Overrides are dotted paths, e.g.
`core.ot_coupling=False` for the `no_ot` ablation arm.

`AuxConfig`, `SpectralConfig` and `DistConfig` control optional objective terms
that are disabled by default (`aux.enabled=False`, `spectral.enabled=False`,
`train.joint_steps=0`) and contribute to no reported number; they are retained
so the development runs stay reproducible.
"""
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional
import json


@dataclass
class DenoiserConfig:
    """The velocity field v_theta(x_t, t, x_self_cond), or eps_theta under the
    DDPM core. A pre-norm bidirectional Transformer over timesteps."""
    n_features: int = 4
    seq_len: int = 24
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 4
    d_ff_mult: int = 4
    dropout: float = 0.0
    causal: bool = False
    cond_mode: Literal["add", "adaln"] = "add"
    use_rope: bool = True           # rotary position embedding
    use_self_cond: bool = True      # self-conditioning
    use_lag_bias: bool = False      # learned per-head bias over relative lags


@dataclass
class AuxConfig:
    """Optional masked critics (`aux_nets.py`). Off in every reported run."""
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
    """Optional log-power-spectrum discrepancy. Off in every reported run."""
    enabled: bool = False
    n_bins: Optional[int] = None      # None -> all rFFT bins up to Nyquist
    eps: float = 1e-6                 # log(power + eps)
    ref_ema_decay: float = 0.99


@dataclass
class CoreConfig:
    """The generative core: rectified flow, or DDPM for the `ddpm_core` arm."""
    kind: Literal["flow", "ddpm"] = "flow"
    n_steps: int = 500                # DDPM schedule length (ddpm only)
    schedule: Literal["cosine", "linear"] = "cosine"
    cosine_s: float = 0.008
    aux_tau_frac: float = 0.30        # low-noise slice for x_hat (ddpm only; see flow.py)
    min_snr_gamma: float = 5.0        # min-SNR loss weighting cap
    min_snr_floor: float = 0.1        # lower bound on the min-SNR weight; 0 = unfloored
    use_min_snr: bool = True
    # Temporal preconditioner W = I + lambda*D on the flow-matching residual.
    # 0 disables it; lambda = 1 is excluded because W is then singular.
    precond_lambda: float = 0.0
    precond_lags: tuple = (1,)        # lags used by the difference metric
    colored_prior: bool = False       # prior noise matched to the data spectrum
    # Pair noise to data within the batch by optimal transport (flow.py: couple)
    ot_coupling: bool = True          # transport coupling; False = independent pairing
    sampling_steps: int = 20          # ODE/DDIM steps at generation time


@dataclass
class DistConfig:
    """Optional two-sample discrepancy (`distributional.py`), evaluated only
    during the joint stage. Off in every reported run.

    `unbiased=False` (the V-statistic) is the default when a gradient is
    involved; see that module for why the unbiased estimator silently stops
    training once the generator is good.
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

    warmup_steps: int = 8000          # the reported budget
    joint_steps: int = 0              # optional regulariser stage; off

    aux_steps: int = 2000             # critic pretraining; only if aux.enabled
    lambda_ext: float = 1.0           # only if aux.enabled
    lambda_int: float = 1.0           # only if aux.enabled
    lambda_spec: float = 1.0          # only if spectral.enabled
    lambda_dist: float = 1.0          # only during the joint stage

    ref_ema_decay: float = 0.99

    dist_every: int = 4
    dist_sample_steps: int = 16
    dist_grad_steps: int = 4
    dist_batch: int = 64
    sample_clamp: float = 1.5

    log_every: int = 250
    val_every: int = 10 ** 9   # no reported run does model selection on the val split
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
        """The reported configuration: rectified flow with the transport
        coupling, RoPE, self-conditioning and the floored min-SNR weight.

        `overrides` are dotted paths, e.g. `core.ot_coupling=False` or
        `denoiser.d_model=128`.
        """
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

    def to_json(self, indent=2):
        return json.dumps(asdict(self), indent=indent, default=str)
