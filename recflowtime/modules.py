"""Shared building blocks: time embedding, RoPE, attention, Transformer block.

RoPE is applied inside attention rather than as a fixed sinusoidal table added
once at the input. It encodes relative offsets between timesteps directly in
the attention dot product, which matches signals whose dynamics depend on lag
-- periodic structure in Sines and Energy, PQRST spacing in ECG -- better than
an absolute position added before the first layer and then diffused away by
residual updates. `use_rope=False` recovers the fixed-sinusoidal behaviour and
is what the `no_rope` ablation arm uses.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0):
    """Sinusoidal embedding of a scalar (diffusion step or flow time). t: (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def sinusoidal_positions(seq_len, d_model):
    """Fixed (B=1, T, d) sin/cos positional table -- used only when use_rope=False."""
    pos = torch.arange(seq_len).unsqueeze(1).float()
    half = d_model // 2
    freq = torch.exp(-math.log(10_000.0) * torch.arange(half).float() / max(half - 1, 1))
    angles = pos * freq
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
    if d_model % 2:
        emb = F.pad(emb, (0, 1))
    return emb.unsqueeze(0)


def rope_cache(seq_len, head_dim, device, base=10_000.0):
    """cos/sin tables for RoPE, each (1, 1, T, head_dim/2)."""
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)                     # (T, half)
    return freqs.cos()[None, None], freqs.sin()[None, None]


def apply_rope(x, cos, sin):
    """x: (B, h, T, dh). Rotate pairs of channels by the per-position angle."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    xr1 = x1 * cos - x2 * sin
    xr2 = x1 * sin + x2 * cos
    out = torch.stack([xr1, xr2], dim=-1).flatten(-2)
    return out


class SelfAttention(nn.Module):
    """Self-attention with optional rotary positions and an optional learned
    lag bias.

    The lag bias adds a learned scalar to the attention logits for every
    relative offset t-s, separately per head. It lets the model favour
    specific lags directly, for example one heartbeat or one daily cycle,
    instead of having to express that preference through the query and key
    projections. The parameters are initialised to zero, so the module starts
    out identical to plain attention and can only add capacity.
    """

    def __init__(self, d_model, n_heads, dropout=0.0, causal=False, use_rope=True,
                 use_lag_bias=False, seq_len=None):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.h = n_heads
        self.dh = d_model // n_heads
        self.causal = causal
        self.use_rope = use_rope
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.use_lag_bias = use_lag_bias
        if use_lag_bias:
            assert seq_len is not None, "lag bias needs seq_len"
            assert not causal, "lag bias and causal masking are not combined here"
            # one weight per head and per relative offset in [-(T-1), T-1]
            self.lag_bias = nn.Parameter(torch.zeros(n_heads, 2 * seq_len - 1))
            idx = torch.arange(seq_len)
            self.register_buffer("lag_index", (idx[:, None] - idx[None, :]) + seq_len - 1,
                                 persistent=False)

    def forward(self, x, rope=None):                       # x: (B, T, d)
        B, T, d = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                    # (B, h, T, dh)
        if self.use_rope and rope is not None:
            cos, sin = rope
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        attn_mask = None
        if self.use_lag_bias:
            attn_mask = self.lag_bias[:, self.lag_index].unsqueeze(0)   # (1, h, T, T)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal,
        )
        out = out.transpose(1, 2).reshape(B, T, d)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, mult * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mult * d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)


class PlainBlock(nn.Module):
    """Pre-LN transformer block, step-conditioning injected once at the input."""

    def __init__(self, d_model, n_heads, mult=4, dropout=0.0, causal=False, use_rope=True,
                 use_lag_bias=False, seq_len=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, dropout, causal, use_rope,
                                  use_lag_bias, seq_len)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, mult, dropout)

    def forward(self, x, c=None, rope=None):
        x = x + self.attn(self.norm1(x), rope=rope)
        x = x + self.ff(self.norm2(x))
        return x


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Pre-LN block with adaLN-Zero conditioning (kept for completeness/ablation;
    RecFlowTime defaults to `cond_mode='add'` + PlainBlock, following the finding in
    at this scale that adaLN-Zero's zero-initialised gates cost convergence
    speed on small transformers with modest step budgets, for no stability
    benefit at this scale."""

    def __init__(self, d_model, n_heads, mult=4, dropout=0.0, causal=False, use_rope=True,
                 use_lag_bias=False, seq_len=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(d_model, n_heads, dropout, causal, use_rope,
                                  use_lag_bias, seq_len)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(d_model, mult, dropout)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c, rope=None):
        s1, sc1, g1, s2, sc2, g2 = self.ada(c).chunk(6, dim=-1)
        x = x + g1.unsqueeze(1) * self.attn(modulate(self.norm1(x), s1, sc1), rope=rope)
        x = x + g2.unsqueeze(1) * self.ff(modulate(self.norm2(x), s2, sc2))
        return x
