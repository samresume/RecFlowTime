"""Parameter-free frequency-domain critic -- new in RecFlowTime.

Motivation. TIDE's two masked critics supervise *time-domain* local
predictability, and its qualitative evaluation separately plots the power
spectrum and autocorrelation function to check periodic structure -- but
nothing in the TIDE training objective optimizes those directly. Its own
related work names STDiffusion (a learnable seasonal-trend decomposition
with wavelet distillation) as its strongest baseline and the one dataset
TIDE does not win outright (Energy, discriminative score 0.118 vs 0.110),
attributing the gap to architecture: "it builds temporal structure into the
architecture, whereas TIDE keeps one undecomposed backbone." Diffusion-TS
(ICLR 2024) reaches a similar conclusion from the objective side, adding a
Fourier-domain reconstruction loss alongside its time-domain one.

RecFlowTime follows the objective-side route, in TIDE's own idiom: rather than
adding a decomposition module to the backbone (more parameters, more
inductive bias, exactly the kind of architectural commitment TIDE's related
work contrasts itself against), it adds a *parameter-free* two-sided
discrepancy on the log power spectrum -- no network to train, no extra
critic pretraining stage, differentiable through the generated batch via
`torch.fft.rfft`, matched to a real-data EMA reference exactly as the
int/ext critics are (Section on losses in the paper) so it cannot be gamed
by minimizing spectral energy to zero.
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
