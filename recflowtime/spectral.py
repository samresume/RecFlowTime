"""A parameter-free discrepancy on the log power spectrum. Optional and off.

The batch mean and standard deviation of the log power spectrum, matched
two-sided to an EMA reference computed from real data rather than minimised, so
it cannot be satisfied by driving spectral energy to zero. No network and no
pretraining stage; it is differentiable through the generated batch via
`torch.fft.rfft`.

This term is disabled in the reported configuration (`spectral.enabled = False`
and `train.joint_steps = 0`) and contributes to no reported number. The
spectral *distance* used for evaluation is a separate, non-differentiable
quantity and lives in `metrics.py`.
"""
import torch
import torch.nn as nn

from .config import SpectralConfig


class SpectralDiscrepancy(nn.Module):
    def __init__(self, cfg: SpectralConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("ref_mean", torch.tensor(float("nan")))
        self.register_buffer("ref_std", torch.tensor(float("nan")))

    def log_power(self, x):
        """x: (B, T, F) -> (B, n_bins, F) log power spectrum along time."""
        spec = torch.fft.rfft(x, dim=1)
        power = spec.real ** 2 + spec.imag ** 2
        logp = torch.log(power + self.cfg.eps)
        if self.cfg.n_bins is not None:
            logp = logp[:, : self.cfg.n_bins]
        return logp

    def _update_ref(self, mean, std):
        decay = self.cfg.ref_ema_decay
        if torch.isnan(self.ref_mean).any():
            self.ref_mean, self.ref_std = mean.detach(), std.detach()
        else:
            self.ref_mean = decay * self.ref_mean + (1 - decay) * mean.detach()
            self.ref_std = decay * self.ref_std + (1 - decay) * std.detach()

    def forward(self, x_fake, x_real):
        """Two-sided: match the batch mean/std of log-power to real data's own
        level (an EMA reference), never minimize it -- minimizing spectral
        energy outright is exactly the flat-line degeneracy the int/ext
        discrepancy loss also guards against."""
        logp_fake = self.log_power(x_fake)
        mean_f, std_f = logp_fake.mean(dim=0), logp_fake.std(dim=0)
        with torch.no_grad():
            logp_real = self.log_power(x_real)
            mean_r, std_r = logp_real.mean(dim=0), logp_real.std(dim=0)
        self._update_ref(mean_r, std_r)
        loss = (mean_f - self.ref_mean).abs().mean() + (std_f - self.ref_std).abs().mean()
        logs = {
            "spec_loss": loss.item(),
            "spec_mean_fake": mean_f.mean().item(),
            "spec_mean_real": float(self.ref_mean.mean()),
        }
        return loss, logs
