"""Frozen masked critics: bidirectional interpolator f_int, causal extrapolator
f_ext. Trained once on real data (Stage 1), then frozen; their masked
reconstruction error on generated samples is a differentiable realism signal
that is matched to the level real data itself exhibits (two-sided), never
minimized outright -- minimizing it rewards the most *predictable* sequence
(e.g. a flat line), not the most realistic one. See `losses.py`.
"""
import torch
import torch.nn as nn

from .modules import PlainBlock
from .masking import apply_mask, masked_mse, extrapolation_mask, interpolation_mask
from .config import AuxConfig


class MaskedSeqModel(nn.Module):
    def __init__(self, cfg: AuxConfig, causal: bool):
        super().__init__()
        d = cfg.d_model
        self.in_proj = nn.Linear(cfg.n_features + 1, d)
        self.pos = nn.Parameter(torch.zeros(1, cfg.seq_len, d))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([
            PlainBlock(d, cfg.n_heads, 4, cfg.dropout, causal, use_rope=False)
            for _ in range(cfg.n_layers)
        ])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, cfg.n_features)

    def forward(self, x_masked):
        h = self.in_proj(x_masked) + self.pos
        for blk in self.blocks:
            h = blk(h)
        return self.out(self.norm(h))


class AuxCritics(nn.Module):
    def __init__(self, cfg: AuxConfig):
        super().__init__()
        self.cfg = cfg
        self.f_int = MaskedSeqModel(cfg, causal=False)
        self.f_ext = MaskedSeqModel(cfg, causal=True)

    def sample_masks(self, B, device, generator=None):
        T = self.cfg.seq_len
        return {
            "ext": extrapolation_mask(B, T, self.cfg.gamma_range, device, generator),
            "int": interpolation_mask(B, T, self.cfg.gamma_range, device, generator),
        }

    def discrepancy(self, x, kind, m=None, generator=None):
        assert kind in ("int", "ext")
        net = self.f_int if kind == "int" else self.f_ext
        if m is None:
            m = self.sample_masks(x.shape[0], x.device, generator)[kind]
        pred = net(apply_mask(x, m))
        return masked_mse(pred, x, m)

    def aux_loss(self, x_real, generator=None):
        masks = self.sample_masks(x_real.shape[0], x_real.device, generator)
        l_int = self.discrepancy(x_real, "int", masks["int"])
        l_ext = self.discrepancy(x_real, "ext", masks["ext"])
        return l_int + l_ext, {"aux_int": l_int.item(), "aux_ext": l_ext.item()}

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        return self
