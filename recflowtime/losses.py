"""The training objective.

    L = L_flow  [ + lambda_dist * L_dist + lambda_spec * L_spec ]

`L_flow` is the min-SNR-weighted conditional flow-matching loss from `flow.py`,
and it is the whole objective in the reported configuration. The two bracketed
terms are optional regularisers evaluated only during the joint stage, which is
disabled by default (`train.joint_steps = 0`, `spectral.enabled = False`); they
are retained so the code that produced the development runs stays runnable, and
they contribute to no reported number.

`DiscrepancyLoss` is likewise inert unless `cfg.aux.enabled`, which is False by
default.
"""
import torch
import torch.nn as nn

from .distributional import distributional_loss
from .spectral import SpectralDiscrepancy
from .config import DistConfig, SpectralConfig

KINDS = ("int", "ext")


class DiscrepancyLoss(nn.Module):
    """Two-sided masked-critic discrepancy. Inert unless `cfg.aux.enabled`,
    which is False in every reported run."""

    def __init__(self, critics, ref_ema_decay=0.99, one_sided=False):
        super().__init__()
        self.critics = critics
        self.decay = ref_ema_decay
        self.one_sided = one_sided
        for k in KINDS:
            self.register_buffer(f"ref_{k}", torch.tensor(float("nan")))

    def _update_ref(self, kind, value):
        name = f"ref_{kind}"
        buf = getattr(self, name)
        v = value.detach()
        new = v if torch.isnan(buf) else self.decay * buf + (1 - self.decay) * v
        setattr(self, name, new)
        return new

    def forward(self, x_fake, x_real, masks=None):
        if masks is None:
            masks = self.critics.sample_masks(x_fake.shape[0], x_fake.device)
        terms, logs = {}, {}
        for kind in KINDS:
            m = masks[kind]
            l_fake = self.critics.discrepancy(x_fake, kind, m)
            with torch.no_grad():
                l_real = self.critics.discrepancy(x_real, kind, m)
            ref = self._update_ref(kind, l_real)
            terms[kind] = l_fake if self.one_sided else (l_fake - ref).abs()
            logs[f"disc_{kind}"] = terms[kind].item()
        return terms, logs


class RecFlowTimeLoss(nn.Module):
    def __init__(self, core, dist_cfg: DistConfig, spec_cfg: SpectralConfig,
                 critics=None, lambda_spec=1.0, lambda_dist=1.0,
                 lambda_int=1.0, lambda_ext=1.0, ref_ema_decay=0.99,
                 use_self_cond=True, dist_sample_steps=16, dist_batch=64,
                 sample_clamp=1.5, dist_grad_steps=4):
        super().__init__()
        self.core = core
        self.spec = SpectralDiscrepancy(spec_cfg) if spec_cfg.enabled else None
        self.disc = DiscrepancyLoss(critics, ref_ema_decay) if critics is not None else None
        self.dist_cfg = dist_cfg
        self.lam_spec = lambda_spec
        self.lam_dist = lambda_dist
        self.lam = {"int": lambda_int, "ext": lambda_ext}
        self.use_self_cond = use_self_cond
        self.dist_sample_steps = dist_sample_steps
        self.dist_batch = dist_batch
        self.sample_clamp = sample_clamp
        self.dist_grad_steps = dist_grad_steps

    def sample_differentiable(self, model, n, seq_len, n_features, device):
        """Unconditional samples from the reverse ODE, graph retained.

        Under rectified flow a step is `x - dt*v_theta(x, t)`: no division,
        so unlike the DDIM route this carries no schedule-dependent gradient
        blow-up and `dist_grad_steps` is a memory knob, not a stability
        requirement."""
        return self.core.sample_differentiable(
            model, (n, seq_len, n_features), n_steps=self.dist_sample_steps,
            device=device, clamp=self.sample_clamp, use_self_cond=self.use_self_cond,
            grad_steps=self.dist_grad_steps,
        )

    def forward(self, model, x0, x0_ref=None, use_aux=True, use_dist=True):
        """x0      -- real batch driving the flow loss and the spectral reference.
        x0_ref     -- INDEPENDENT real batch for the two-sample distributional term.
        """
        logs = {}
        loss = self.core.flow_loss(model, x0, use_self_cond=self.use_self_cond)
        logs["core"] = loss.item()

        x_gen = None
        if use_dist:
            B, T, Fd = x0.shape
            x_gen = self.sample_differentiable(model, min(self.dist_batch, B), T, Fd, x0.device)
            logs["gen_std"] = x_gen.std().item()

        # Spectral discrepancy: on genuine samples when we have them (they are
        # what generation actually emits), else on the cheap one-step estimate.
        if self.spec is not None and use_aux:
            if x_gen is not None:
                spec_target, spec_ref = x_gen, x0[: x_gen.shape[0]]
            else:
                spec_target, *_ = self.core.x1_hat(model, x0, use_self_cond=self.use_self_cond)
                spec_ref = x0
            spec_loss, slogs = self.spec(spec_target, spec_ref)
            loss = loss + self.lam_spec * spec_loss
            logs.update(slogs)

        # Masked critics: baseline configuration only.
        if self.disc is not None and use_aux:
            x_hat, *_ = self.core.x1_hat(model, x0, use_self_cond=self.use_self_cond)
            terms, dlogs = self.disc(x_hat, x0)
            for kind, t in terms.items():
                loss = loss + self.lam[kind] * t
            logs.update(dlogs)

        if use_dist:
            if x0_ref is None:
                x0_ref = x0[x0.shape[0] // 2:]
            n = min(len(x0_ref), len(x_gen))
            d = distributional_loss(x0_ref[:n], x_gen[:n], self.dist_cfg)
            loss = loss + self.lam_dist * d
            logs["dist"] = d.item()

        logs["total"] = loss.item()
        return loss, logs
