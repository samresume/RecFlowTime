"""Classic Gaussian DDPM + DDIM core -- kept only as the ablation baseline.

Exposes the same method surface as `flow.RectifiedFlow` (`flow_loss`,
`x1_hat`, `sample`, `sample_differentiable`) so `losses.py`/`train.py` can
swap generative cores via `RecFlowTimeConfig.core.kind` without touching the
training loop. Selecting `core.kind="ddpm"` (equivalently
`RecFlowTimeConfig.tide_baseline(...)`) reproduces TIDE's own generative
mechanics: eps-prediction on a cosine schedule, x0 recovered through the
division `(x_tau - sqrt(1-abar)*eps) / sqrt(abar)`, and DDIM sampling with
gradient-truncated backprop for the differentiable branch -- see `flow.py`'s
module docstring for why RecFlowTime replaces this.
"""
import math
import torch

from .config import CoreConfig


def cosine_betas(T, s=0.008):
    t = torch.linspace(0, T, T + 1, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    betas = 1 - ab[1:] / ab[:-1]
    return betas.clamp(1e-8, 0.999).float()


def linear_betas(T, b0=1e-4, b1=0.02):
    return torch.linspace(b0, b1, T, dtype=torch.float32)


class GaussianDiffusion:
    def __init__(self, cfg: CoreConfig, device="cpu"):
        self.cfg = cfg
        T = cfg.n_steps
        betas = cosine_betas(T, cfg.cosine_s) if cfg.schedule == "cosine" else linear_betas(T)
        self.betas = betas.to(device)
        self.alphas = 1.0 - self.betas
        self.abar = torch.cumprod(self.alphas, dim=0)
        self.abar_prev = torch.cat([torch.ones(1, device=device), self.abar[:-1]])
        self.sqrt_abar = self.abar.sqrt()
        self.sqrt_one_minus_abar = (1.0 - self.abar).sqrt()
        self.post_var = self.betas * (1.0 - self.abar_prev) / (1.0 - self.abar)
        self.device = device

    def to(self, device):
        for k, v in list(self.__dict__.items()):
            if torch.is_tensor(v):
                setattr(self, k, v.to(device))
        self.device = device
        return self

    def _tau_int(self, t_float):
        """Map a continuous t in [0,1] (used by the shared training loop's
        random draws) to an integer diffusion step, so this core can be
        driven identically to RectifiedFlow from outside."""
        return (t_float * (self.cfg.n_steps - 1)).round().long().clamp(0, self.cfg.n_steps - 1)

    def sample_t(self, B, device, low=0.0, high=1.0, generator=None):
        return torch.rand(B, device=device, generator=generator) * (high - low) + low

    def sample_aux_t(self, B, device, generator=None):
        high = max(1e-3, self.cfg.aux_tau_frac)
        return self.sample_t(B, device, 0.0, high, generator)


    def couple(self, x_data, noise):
        """Same minibatch optimal-transport pairing as RectifiedFlow.couple.

        Present so that an arm which swaps only the generative core is a
        genuine leave-one-out: without it, replacing the flow core would also
        silently remove the coupling, and a difference could not be attributed.
        """
        if not self.cfg.ot_coupling:
            return noise
        from scipy.optimize import linear_sum_assignment
        with torch.no_grad():
            cost = torch.cdist(x_data.flatten(1), noise.flatten(1)).pow(2).cpu().numpy()
            _, col = linear_sum_assignment(cost)
            return noise[torch.as_tensor(col, device=noise.device)]

    def corrupt(self, x0, t_float, noise=None):
        tau = self._tau_int(t_float)
        noise = torch.randn_like(x0) if noise is None else noise
        a = self.sqrt_abar[tau].view(-1, *([1] * (x0.dim() - 1)))
        b = self.sqrt_one_minus_abar[tau].view(-1, *([1] * (x0.dim() - 1)))
        return a * x0 + b * noise, noise

    def predict_clean(self, x_tau, t_float, eps_pred, clamp=None):
        tau = self._tau_int(t_float)
        a = self.sqrt_abar[tau].view(-1, *([1] * (x_tau.dim() - 1)))
        b = self.sqrt_one_minus_abar[tau].view(-1, *([1] * (x_tau.dim() - 1)))
        x0 = (x_tau - b * eps_pred) / a
        if clamp is not None:
            x0 = x0.clamp(-clamp, clamp)
        return x0

    def x1_hat(self, model, x_data, t=None, generator=None, clamp=None,
              self_cond_prob=0.5, use_self_cond=False):
        B = x_data.shape[0]
        if t is None:
            t = self.sample_aux_t(B, x_data.device, generator)
        x_tau, noise = self.corrupt(x_data, t)
        sc = torch.zeros_like(x_data) if use_self_cond else None
        if use_self_cond and torch.rand((), generator=generator).item() < self_cond_prob:
            with torch.no_grad():
                e0 = model(x_tau, t, sc)
                sc = self.predict_clean(x_tau, t, e0, clamp=clamp).detach()
        eps_pred = model(x_tau, t, sc) if use_self_cond else model(x_tau, t, None)
        return self.predict_clean(x_tau, t, eps_pred, clamp), eps_pred, t, noise, x_tau

    def flow_loss(self, model, x_data, generator=None, use_self_cond=False, self_cond_prob=0.5):
        B = x_data.shape[0]
        t = self.sample_t(B, x_data.device, generator=generator)
        noise = self.couple(x_data, torch.randn_like(x_data))
        x_tau, noise = self.corrupt(x_data, t, noise=noise)
        sc = torch.zeros_like(x_data) if use_self_cond else None
        if use_self_cond and torch.rand((), generator=generator).item() < self_cond_prob:
            with torch.no_grad():
                e0 = model(x_tau, t, sc)
                sc = self.predict_clean(x_tau, t, e0).detach()
        eps_pred = model(x_tau, t, sc) if use_self_cond else model(x_tau, t, None)
        se = (eps_pred - noise) ** 2
        if self.cfg.use_min_snr:
            tau = self._tau_int(t)
            snr = self.abar[tau] / (1 - self.abar[tau])
            w = torch.clamp(snr, max=self.cfg.min_snr_gamma) / snr.clamp_min(1e-8)
            floor = getattr(self.cfg, "min_snr_floor", 0.0)
            if floor:
                w = w.clamp_min(floor)
            se = se * w.view(-1, *([1] * (x_data.dim() - 1)))
        return se.mean()

    @torch.no_grad()
    def sample(self, model, shape, n_steps=None, device=None, clamp=1.0,
              use_self_cond=False, x_init=None):
        return self._ddim(model, shape, n_steps, device, clamp, use_self_cond,
                          x_init, differentiable=False, grad_steps=None)

    def sample_differentiable(self, model, shape, n_steps, device=None, clamp=1.0,
                              use_self_cond=False, grad_steps=None, x_init=None):
        return self._ddim(model, shape, n_steps, device, clamp, use_self_cond,
                          x_init, differentiable=True, grad_steps=grad_steps)

    def _ddim(self, model, shape, n_steps, device, clamp, use_self_cond, x_init,
             differentiable, grad_steps):
        device = device or self.device
        n_steps = n_steps or self.cfg.sampling_steps
        ts_int = torch.linspace(self.cfg.n_steps - 1, 0, n_steps).round().long().tolist()
        x = torch.randn(shape, device=device) if x_init is None else x_init
        sc = torch.zeros(shape, device=device) if use_self_cond else None
        grad_steps = n_steps if (grad_steps is None or not differentiable) else grad_steps
        cutover = max(0, n_steps - grad_steps)

        def _step(x, sc, i, tau_int):
            t_vec = torch.full((shape[0],), tau_int / (self.cfg.n_steps - 1), device=device)
            eps = model(x, t_vec, sc)
            x0 = self.predict_clean(x, t_vec, eps, clamp=clamp)
            t_prev = ts_int[i + 1] if i + 1 < len(ts_int) else -1
            ab_prev = self.abar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
            dir_xt = (1 - ab_prev).clamp_min(0).sqrt()
            x_next = ab_prev.sqrt() * x0 + dir_xt * eps
            return x_next, (x0.detach() if not differentiable else x0)

        if cutover > 0:
            with torch.no_grad():
                for i, tau in enumerate(ts_int[:cutover]):
                    x, sc_new = _step(x, sc, i, tau)
                    if use_self_cond:
                        sc = sc_new
        ctx = torch.enable_grad() if differentiable else torch.no_grad()
        with ctx:
            for i, tau in enumerate(ts_int[cutover:], start=cutover):
                x, sc_new = _step(x, sc, i, tau)
                if use_self_cond:
                    sc = sc_new if differentiable else sc_new.detach()
        return x
