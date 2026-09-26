"""v_theta / eps_theta(X_t, t, x_self_cond) -- the shared backbone.

Bidirectional self-attention over timesteps is kept unchanged from TIDE: its
own ablation shows removing attention for a convolutional U-Net costs more
than any other single change (+554% Context-FID, the largest effect in that
study), so this is the one architectural choice RecFlowTime does not touch.
Two additions layer on top of it:

  * RoPE (`use_rope`) instead of a fixed sinusoidal table added once at the
    input -- see `modules.py`.
  * Self-conditioning (`use_self_cond`, Chen et al. 2022): the network also
    reads its own previous clean-data estimate. At generation time this is
    free (the reverse ODE already computes that estimate at every step); at
    training time it costs one extra no-grad forward pass half the time
    (`self_cond_prob`).

Conditioning on the diffusion step / flow time is additive at the input
(`cond_mode='add'`), following the same finding used in TIDE: adaLN-Zero's
zero-initialised gates trade early convergence speed for stability this
model scale does not need. `cond_mode='adaln'` is kept for completeness.
"""
import torch
import torch.nn as nn

from .modules import DiTBlock, PlainBlock, timestep_embedding, sinusoidal_positions, rope_cache
from .config import DenoiserConfig


class TransformerDenoiser(nn.Module):
    def __init__(self, cfg: DenoiserConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.in_proj = nn.Linear(cfg.n_features, d)
        if cfg.use_self_cond:
            self.sc_proj = nn.Linear(cfg.n_features, d)

        if not cfg.use_rope:
            self.register_buffer("pos", sinusoidal_positions(cfg.seq_len, d))

        self.step_mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.SiLU(), nn.Linear(4 * d, d))

        Block = DiTBlock if cfg.cond_mode == "adaln" else PlainBlock
        self.blocks = nn.ModuleList([
            Block(d, cfg.n_heads, cfg.d_ff_mult, cfg.dropout, cfg.causal, cfg.use_rope,
                  cfg.use_lag_bias, cfg.seq_len)
            for _ in range(cfg.n_layers)
        ])

        self.out_norm = nn.LayerNorm(d, elementwise_affine=(cfg.cond_mode != "adaln"), eps=1e-6)
        if cfg.cond_mode == "adaln":
            self.out_ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.out_proj = nn.Linear(d, cfg.n_features)

    def _rope(self, device):
        head_dim = self.cfg.d_model // self.cfg.n_heads
        return rope_cache(self.cfg.seq_len, head_dim, device)

    def forward(self, x, t, x_self_cond=None):
        """x: (B, T, F) interpolated sequence. t: (B,) float in [0,1] (flow) or
        integer-step-as-float (ddpm core). x_self_cond: (B, T, F) or None."""
        c = self.step_mlp(timestep_embedding(t, self.cfg.d_model))
        h = self.in_proj(x)
        if self.cfg.use_self_cond:
            sc = x_self_cond if x_self_cond is not None else torch.zeros_like(x)
            h = h + self.sc_proj(sc)
        if not self.cfg.use_rope:
            h = h + self.pos
        if self.cfg.cond_mode == "add":
            h = h + c.unsqueeze(1)

        rope = self._rope(x.device) if self.cfg.use_rope else None
        for blk in self.blocks:
            h = blk(h, c, rope=rope)

        if self.cfg.cond_mode == "adaln":
            shift, scale = self.out_ada(c).chunk(2, dim=-1)
            h = self.out_norm(h) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        else:
            h = self.out_norm(h)
        return self.out_proj(h)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
