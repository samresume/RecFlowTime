"""Masking operators for the optional masked critics in `aux_nets.py`.

Used only when `cfg.aux.enabled` is True, which is not the case in any reported
run. The mask is carried as an explicit extra channel rather than by writing
zeros into the hidden entries, because the data is normalised to a symmetric
range in which 0 is an ordinary observed value.

The masks used for imputation and forecasting at sampling time are unrelated to
these and live in `conditional.py`.
"""
import torch


def sample_span(B, T, gamma_range, device, min_margin=0, generator=None):
    """Random (start, span) for a contiguous run of length `span` inside [0, T),
    leaving at least `min_margin` free positions on each side."""
    lo, hi = gamma_range
    frac = torch.rand(B, device=device, generator=generator) * (hi - lo) + lo
    span = (frac * T).round().clamp(1, max(1, T - 2 * min_margin))
    max_start = (T - span - min_margin).clamp(min=min_margin)
    u = torch.rand(B, device=device, generator=generator)
    start = (min_margin + u * (max_start - min_margin + 1e-6)).floor().clamp(min_margin)
    return start, span


def extrapolation_mask(B, T, gamma_range, device, generator=None):
    """(B, T) in {0,1}; 1 marks the held-out future suffix."""
    lo, hi = gamma_range
    frac = torch.rand(B, device=device, generator=generator) * (hi - lo) + lo
    span = (frac * T).round().clamp(1, T - 1)
    idx = torch.arange(T, device=device).unsqueeze(0)
    start = (T - span).unsqueeze(1)
    return (idx >= start).float()


def interpolation_mask(B, T, gamma_range, device, generator=None):
    """(B, T) in {0,1}; 1 marks a strictly-interior contiguous span (context
    survives on both sides, unlike extrapolation)."""
    start, span = sample_span(B, T, gamma_range, device, min_margin=1, generator=generator)
    idx = torch.arange(T, device=device).unsqueeze(0)
    return ((idx >= start.unsqueeze(1)) & (idx < (start + span).unsqueeze(1))).float()


def apply_mask(x, m):
    """x: (B, T, F), m: (B, T) with 1 = hidden -> (B, T, F+1) masked input + indicator."""
    visible = (1.0 - m).unsqueeze(-1)
    return torch.cat([x * visible, m.unsqueeze(-1)], dim=-1)


def masked_mse(pred, target, m, reduce="mean"):
    """MSE over hidden timesteps only, normalised per-sample by #hidden entries."""
    w = m.unsqueeze(-1)
    se = ((pred - target) ** 2 * w).sum(dim=(1, 2))
    n_obs = (w.sum(dim=(1, 2)) * pred.shape[-1]).clamp(min=1e-8)
    per_sample = se / n_obs
    if reduce == "mean":
        return per_sample.mean()
    if reduce == "none":
        return per_sample
    raise ValueError(reduce)
