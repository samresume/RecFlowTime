"""Plotting helpers for a training report: samples, PCA/t-SNE, ACF, spectrum,
loss curves. Mirrors the diagnostics TIDE's own report checks qualitatively,
so a RecFlowTime run and a TIDE-baseline run (same trainer, `core.kind="ddpm"`)
produce directly comparable figures.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # avoid the torch+sklearn TSNE segfault

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .metrics import acf


def _flat(X, n=500):
    X = X.numpy() if torch.is_tensor(X) else np.asarray(X)
    return X[:n].reshape(X[:n].shape[0], -1)


def training_report(real, fake, history, out_path, feature=0, title="RecFlowTime",
                    summary=None):
    """2x5 panel report: samples, PCA, t-SNE, marginal density, ACF (top);
    cross-feature correlation, power spectrum, losses, grad norm, summary
    (bottom). `summary` is an optional {label: value} dict for the last panel.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    fig, axes = plt.subplots(2, 5, figsize=(24, 9))
    fig.suptitle(title, fontsize=14)

    # raw samples with a 5-95% band
    ax = axes[0, 0]
    R, Fk = real.numpy(), fake.numpy()
    lo, hi = np.percentile(R[..., feature], [5, 95], axis=0)
    ax.fill_between(range(R.shape[1]), lo, hi, alpha=0.25, color="C0", label="real 5-95%")
    ax.plot(R[0, :, feature], "C0", lw=1.2, label="real")
    ax.plot(Fk[0, :, feature], "C1--", lw=1.2, label="generated")
    ax.plot(Fk[1, :, feature], "C1--", lw=1.2, alpha=0.6)
    ax.set_title("Samples"); ax.legend(fontsize=7)

    # PCA
    ax = axes[0, 1]
    flat = np.concatenate([_flat(real), _flat(fake)])
    p = PCA(n_components=2).fit_transform(flat)
    n = min(500, len(real))
    ax.scatter(*p[:n].T, s=4, alpha=0.4, label="real")
    ax.scatter(*p[n:].T, s=4, alpha=0.4, label="generated")
    ax.set_title("PCA"); ax.legend(fontsize=7)

    # t-SNE
    ax = axes[0, 2]
    try:
        emb = TSNE(n_components=2, init="pca", perplexity=30).fit_transform(flat)
        ax.scatter(*emb[:n].T, s=4, alpha=0.4, label="real")
        ax.scatter(*emb[n:].T, s=4, alpha=0.4, label="generated")
    except Exception as e:
        ax.text(0.5, 0.5, f"t-SNE failed:\n{e}", ha="center", va="center")
    ax.set_title("t-SNE"); ax.legend(fontsize=7)

    # marginal density
    ax = axes[0, 3]
    ax.hist(R[..., feature].ravel(), bins=50, density=True, alpha=0.5, label="real")
    ax.hist(Fk[..., feature].ravel(), bins=50, density=True, alpha=0.5, label="generated")
    ax.set_title("Marginal density"); ax.legend(fontsize=7)

    # ACF -- mean over features, with a per-feature spread band
    ax = axes[0, 4]
    ar, af = acf(real), acf(fake)              # (n_lags, F)
    lags = range(1, ar.shape[0] + 1)
    ax.plot(lags, ar.mean(1), "C0", label="real (mean over features)")
    ax.plot(lags, af.mean(1), "C1--", label="generated")
    if ar.shape[1] > 1:
        ax.fill_between(lags, ar.min(1), ar.max(1), color="C0", alpha=0.15)
        ax.fill_between(lags, af.min(1), af.max(1), color="C1", alpha=0.15)
    ax.set_xlabel("lag")
    ax.set_title("Autocorrelation"); ax.legend(fontsize=7)

    # cross-feature correlation, real lower / generated upper triangle
    ax = axes[1, 0]
    cr = np.corrcoef(R.reshape(-1, R.shape[-1]).T)
    cf = np.corrcoef(Fk.reshape(-1, Fk.shape[-1]).T)
    mat = np.tril(cr, -1) + np.triu(cf, 1) + np.diag(np.full(cr.shape[0], np.nan))
    im = ax.imshow(mat, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Correlation (lower=real, upper=gen)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # power spectrum
    ax = axes[1, 1]
    pr = np.abs(np.fft.rfft(R[..., feature], axis=1)).mean(0)
    pf = np.abs(np.fft.rfft(Fk[..., feature], axis=1)).mean(0)
    ax.plot(pr, "C0", label="real"); ax.plot(pf, "C1--", label="generated")
    ax.set_title("Power spectrum"); ax.legend(fontsize=7)

    # loss curves
    ax = axes[1, 2]
    for key in ("core", "disc_int", "disc_ext", "spec_loss", "dist"):
        if key in history:
            steps = [s for s, st in zip(history["step"], history["stage"]) if st == "joint"] \
                if key != "core" else history["step"]
            vals = history[key]
            n = min(len(steps), len(vals))
            if n > 1:
                ax.plot(steps[-n:], vals[-n:], label=key)
    ax.set_yscale("log"); ax.set_title("Losses"); ax.legend(fontsize=6)

    # grad norm
    ax = axes[1, 3]
    if "grad_norm" in history:
        ax.plot(history["step"][-len(history["grad_norm"]):], history["grad_norm"])
    ax.set_title("Grad norm on theta")

    # summary panel
    ax = axes[1, 4]
    ax.axis("off")
    ax.text(0.0, 0.97, title, fontsize=13, fontweight="bold", va="top")
    if summary:
        lines = []
        for k, v in summary.items():
            if isinstance(v, float):
                v = f"{v:.4f}" if abs(v) < 1e4 else f"{v:.3e}"
            lines.append(f"{k}: {v}")
        ax.text(0.0, 0.88, "\n".join(lines), fontsize=9, va="top", family="monospace")

    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
