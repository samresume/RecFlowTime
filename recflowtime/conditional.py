r"""Conditional generation by masked replacement.

RecFlowTime is trained unconditionally, but the straight conditional path makes
imputation and forecasting available at sampling time without retraining and
without a second model. Because the forward process is the interpolation
x_t = (1-t) x + t eps, carrying the observed entries of a target to the noise
level of the current integration step costs one line and no model call. We
write them back after every Euler step; at t = 0 they are reproduced exactly.

Nothing about the model changes and no gradient is taken, so the number of
function evaluations is the unconditional one. The two tasks differ only in the
mask: hidden streaks for imputation, a hidden tail for forecasting.
"""
import numpy as np
import torch

__all__ = ["impute_mask", "forecast_mask", "complete", "complete_k", "scores",
           "linear_fill", "last_value_fill"]


# ---- masks ---------------------------------------------------------------
def geom_streaks(T, ratio, lm, rng):
    """A boolean run of length T that is False for about `ratio` of its entries.

    A Markov chain with mean masked-run length `lm`, so hidden entries arrive in
    contiguous stretches rather than as isolated points. Isolated points are
    recoverable by interpolation from their immediate neighbours and would
    flatter any method.
    """
    keep = np.ones(T, dtype=bool)
    p_m = 1.0 / lm                                   # leave a masked run
    p_u = p_m * ratio / (1.0 - ratio + 1e-12)        # enter a masked run
    masked = rng.random() < ratio
    for i in range(T):
        keep[i] = not masked
        masked = (rng.random() > p_m) if masked else (rng.random() < p_u)
    return keep


def impute_mask(shape, ratio, style="concurrent", lm=3, rng=None):
    """True where observed. `shape` is (N, T, F).

    `style` is "separate" (each feature hidden independently) or "concurrent"
    (all features hidden at the same timesteps).
    """
    rng = rng or np.random.default_rng(0)
    N, T, F = shape
    m = np.ones(shape, dtype=bool)
    for i in range(N):
        if style == "concurrent":
            m[i] = geom_streaks(T, ratio, lm, rng)[:, None]
        else:
            for f in range(F):
                m[i, :, f] = geom_streaks(T, ratio, lm, rng)
    return m


def forecast_mask(shape, horizon):
    """True where observed; the final `horizon` steps are hidden."""
    N, T, F = shape
    m = np.ones(shape, dtype=bool)
    m[:, T - horizon:, :] = False
    return m


# ---- sampling ------------------------------------------------------------
def complete(model, core, dcfg, target, keep, steps=20, batch=256, seed=0,
             after_selfcond=True):
    """Fill the hidden entries of `target`; `keep` is True where observed."""
    torch.manual_seed(seed)
    tgt = torch.as_tensor(np.asarray(target), dtype=torch.float32)
    msk = torch.as_tensor(np.asarray(keep))
    outs = []
    with torch.no_grad():
        for s in range(0, len(tgt), batch):
            e = min(s + batch, len(tgt))
            outs.append(core.sample(model, (e - s, dcfg.seq_len, dcfg.n_features),
                                    n_steps=steps, device="cpu", clamp=1.0,
                                    use_self_cond=dcfg.use_self_cond,
                                    cond_target=tgt[s:e], cond_mask=msk[s:e],
                                    cond_after_selfcond=after_selfcond))
    return torch.cat(outs).numpy()


def complete_k(model, core, dcfg, target, keep, steps=20, k=10, seed=0, **kw):
    """K independent completions of the same masked input, shape (K, N, T, F)."""
    return np.stack([complete(model, core, dcfg, target, keep, steps,
                              seed=seed + 1000 * i, **kw) for i in range(k)])


# ---- references ----------------------------------------------------------
def linear_fill(x, keep):
    """Linear interpolation through the observed entries, per feature.

    A scale rather than a competitor: on a near-random walk this is the
    Brownian-bridge posterior mean and therefore close to optimal.
    """
    out = np.asarray(x).copy()
    N, T, F = out.shape
    grid = np.arange(T)
    for i in range(N):
        for f in range(F):
            k = keep[i, :, f]
            if k.sum() == 0:
                out[i, :, f] = 0.0
            elif k.sum() < T:
                out[i, :, f] = np.interp(grid, grid[k], out[i, k, f])
    return out


def last_value_fill(x, horizon):
    """Repeat the last observed value across the horizon."""
    out = np.asarray(x).copy()
    out[:, out.shape[1] - horizon:, :] = out[:, out.shape[1] - horizon - 1, :][:, None, :]
    return out


# ---- scoring -------------------------------------------------------------
def scores(preds, real, keep):
    """Score K completions on the hidden entries. `preds` is (K, N, T, F).

    A generative model returns a sample from the conditional distribution,
    whereas squared error is minimised by the conditional mean, so scoring one
    draw penalises a calibrated model by construction. We therefore report the
    error of the K-sample mean, which estimates that conditional mean, and
    CRPS, a proper scoring rule, alongside the single-draw value.
    """
    preds = np.asarray(preds)
    hid = ~keep
    K = preds.shape[0]
    d1, dm = preds[0] - real, preds.mean(0) - real
    X, y = preds[:, hid], real[None][:, hid]
    term1 = np.abs(X - y).mean()
    if K > 1:                       # unbiased E|X - X'| without a K^2 array
        Xs = np.sort(X, axis=0)
        w = 2 * np.arange(1, K + 1) - K - 1
        term2 = (w[:, None] * Xs).sum(0).mean() * (2.0 / (K * (K - 1)))
    else:
        term2 = 0.0
    return {"mse": float((d1[hid] ** 2).mean()),
            "mae": float(np.abs(d1[hid]).mean()),
            "mse_mean": float((dm[hid] ** 2).mean()),
            "crps": float(term1 - 0.5 * term2),
            "n_samples": int(K),
            # observed entries are written back exactly, so this must be 0
            "obs_max_abs": float(np.abs(d1[keep]).max()) if keep.any() else 0.0}
