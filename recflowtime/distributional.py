"""Two-sample distributional discrepancies. Optional and off.

Maximum mean discrepancy between a real and a generated batch, either on the
flattened (T*F,) sequence vector or in a depth-truncated path-signature feature
space. The signature variant is graded by order-sensitive structure -- net
displacement at level 1, signed area and lead-lag between every feature pair at
level 2, and so on -- which an RBF kernel on a flattened vector cannot see.

These terms are evaluated only during the joint stage, which is disabled by
default (`train.joint_steps = 0`), so they contribute to no reported number.
`raw_mmd2` is also used as a diagnostic by `metrics.py`.

`unbiased=False` (the V-statistic) is the default when a gradient is involved.
The unbiased U-statistic is not guaranteed non-negative and its variance does
not shrink as the two distributions approach each other, so once the generator
is good the estimate frequently goes negative; clamping it at zero -- necessary,
since a negative value is meaningless as a minimisation target -- zeroes the
gradient with it and the term silently stops training the model. The
V-statistic is a sum of positive kernel terms, so it is non-negative by
construction, never hits the clamp, and carries a smooth gradient throughout.
Its O(1/n) upward bias is a slowly varying offset that does not change what the
gradient points at. `unbiased=True` remains available for evaluation, where an
unbiased point estimate is what is wanted and no gradient is involved.
"""
import math
import torch

from .config import DistConfig


# ------------------------------------------------------------- RBF-kernel MMD
def _pdist2(a, b):
    return torch.cdist(a, b, p=2).pow(2)


def median_bandwidth(x, y, eps=1e-8):
    z = torch.cat([x, y], dim=0)
    with torch.no_grad():
        d2 = _pdist2(z, z)
        n = z.shape[0]
        off = d2[~torch.eye(n, dtype=torch.bool, device=z.device)]
        med = off.median()
    return med.clamp_min(eps)


def mmd2(x, y, cfg: DistConfig, sigma2=None):
    """Unbiased multi-bandwidth RBF MMD^2 between two (n, d) feature batches."""
    n, m = x.shape[0], y.shape[0]
    if sigma2 is None:
        sigma2 = median_bandwidth(x, y)
    dxx, dyy, dxy = _pdist2(x, x), _pdist2(y, y), _pdist2(x, y)
    Kxx = Kyy = Kxy = 0.0
    for s in cfg.bandwidth_scales:
        denom = 2.0 * sigma2 * s
        Kxx = Kxx + torch.exp(-dxx / denom)
        Kyy = Kyy + torch.exp(-dyy / denom)
        Kxy = Kxy + torch.exp(-dxy / denom)
    if cfg.unbiased:
        ex = ~torch.eye(n, dtype=torch.bool, device=x.device)
        ey = ~torch.eye(m, dtype=torch.bool, device=y.device)
        term_xx = Kxx[ex].sum() / (n * (n - 1))
        term_yy = Kyy[ey].sum() / (m * (m - 1))
    else:
        term_xx, term_yy = Kxx.sum() / (n * n), Kyy.sum() / (m * m)
    term_xy = Kxy.sum() / (n * m)
    val = term_xx + term_yy - 2.0 * term_xy
    if cfg.clamp_min is not None:
        val = val.clamp_min(cfg.clamp_min)
    return val


def raw_mmd2(X, Y, cfg: DistConfig):
    """MMD^2 on flattened (T, F) -> (T*F,) sequence vectors."""
    return mmd2(X.flatten(1), Y.flatten(1), cfg)


# ---------------------------------------------------------- truncated signature
def _outer(a, b):
    """Batched outer product: a (B, *sa), b (B, *sb) -> (B, *sa, *sb)."""
    a2 = a.reshape(a.shape[0], *a.shape[1:], *([1] * (b.dim() - 1)))
    b2 = b.reshape(b.shape[0], *([1] * (a.dim() - 1)), *b.shape[1:])
    return a2 * b2


def path_signature(path, depth):
    """path: (B, T, d) -> list of tensors, levels 1..depth, each (B, d, d, ..., d).

    Iterated integrals of the piecewise-linear path through `path`, built by
    Chen's identity: S(concat of two paths) = S(path1) (x) S(path2) in the
    truncated tensor algebra, applied one linear segment at a time.
    """
    B, T, d = path.shape
    device, dtype = path.device, path.dtype
    dx = path[:, 1:] - path[:, :-1]                        # (B, T-1, d)
    sig = [torch.zeros(B, *([d] * k), device=device, dtype=dtype) for k in range(depth + 1)]
    sig[0] = torch.ones(B, device=device, dtype=dtype)

    for t in range(dx.shape[1]):
        d_t = dx[:, t]                                      # (B, d)
        incr = [torch.ones(B, device=device, dtype=dtype)]
        pow_k = None
        for k in range(1, depth + 1):
            pow_k = d_t if k == 1 else _outer(pow_k, d_t)
            incr.append(pow_k / math.factorial(k))
        new_sig = [torch.ones(B, device=device, dtype=dtype)]
        for k in range(1, depth + 1):
            term = incr[k] + sig[k]
            for j in range(1, k):
                term = term + _outer(sig[j], incr[k - j])
            new_sig.append(term)
        sig = new_sig
    return sig[1:]


def _signature_levels(x, depth, time_augment=True):
    """x: (B, T, F) -> list of raw (unflattened) signature-level tensors.

    Basepoint augmentation (prepend a zero) makes the level-1 term see the
    sequence's absolute starting level, not only its increments; time
    augmentation appends a monotone clock channel so the signature is
    sensitive to *when* things happen, not just their order (standard
    practice, Levin et al. 2013).
    """
    B, T, F = x.shape
    path = x
    if time_augment:
        t = torch.linspace(0, 1, T, device=x.device, dtype=x.dtype).view(1, T, 1).expand(B, T, 1)
        path = torch.cat([path, t], dim=-1)
    basepoint = torch.zeros(B, 1, path.shape[-1], device=x.device, dtype=x.dtype)
    path = torch.cat([basepoint, path], dim=1)
    return path_signature(path, depth)


def signature_features(x, depth, time_augment=True):
    """x: (B, T, F) -> (B, D) flattened truncated-signature feature vector,
    each level rescaled to unit RMS (see `signature_mmd2` for why)."""
    levels = _signature_levels(x, depth, time_augment)
    return torch.cat([
        (lvl / lvl.pow(2).mean().sqrt().clamp_min(1e-6)).flatten(1) for lvl in levels
    ], dim=1)


def signature_mmd2(X, Y, cfg: DistConfig):
    """Signature levels differ in natural magnitude by orders of magnitude
    (level k ~ path-variation^k / k!), so each level is rescaled by a single
    *per-level* RMS pooled over both batches before concatenation.

    This must be a per-level scalar, not a per-coordinate one: an earlier
    version divided each of the D individual signature coordinates by its
    own pooled std, and a handful of coordinates that are structurally
    near-constant on a given dataset (e.g. certain symmetric cross-terms on
    Sines' narrow phase range) have std ~1e-7, so dividing blew those
    coordinates up by ~1e6x and swamped the kernel distance -- the RBF
    kernel then saturated to exp(-huge) = 0 for every pair, silently zeroing
    the whole loss (`mmd_sanity_check` catches exactly this). Aggregating
    the RMS over an entire level's many coordinates instead of one at a time
    is not vulnerable to a single near-constant coordinate.
    """
    lx = _signature_levels(X, cfg.sig_depth, cfg.sig_time_aug)
    ly = _signature_levels(Y, cfg.sig_depth, cfg.sig_time_aug)
    fx_parts, fy_parts = [], []
    for a, b in zip(lx, ly):
        pooled = torch.cat([a, b], dim=0)
        scale = pooled.pow(2).mean().sqrt().clamp_min(1e-6)
        fx_parts.append((a / scale).flatten(1))
        fy_parts.append((b / scale).flatten(1))
    fx, fy = torch.cat(fx_parts, dim=1), torch.cat(fy_parts, dim=1)
    return mmd2(fx, fy, cfg)


def distributional_loss(X, Y, cfg: DistConfig):
    if cfg.kind == "sigmmd":
        return signature_mmd2(X, Y, cfg)
    if cfg.kind == "mmd":
        return raw_mmd2(X, Y, cfg)
    raise ValueError(cfg.kind)


def mmd_sanity_check(X_real, X_real_heldout, cfg: DistConfig, verbose=True):
    """Is the kernel discriminating at all on this data?

    The failure this exists to catch is a *saturated* kernel: if the
    bandwidth is far below or far above the typical pairwise distance, every
    kernel entry goes to 0 or to 1, all three MMD terms cancel, and the loss
    is a constant with zero gradient -- silently, with nothing in the loss
    value revealing the cause. (Exactly this bit during development here:
    per-coordinate signature normalisation blew up near-constant coordinates
    and drove every kernel entry to 0.)

    So the test is: does the kernel separate real data from *any* corruption
    of it? Two corruptions are tried, because their difficulty is
    dataset-dependent -- time-shuffling within a window is a severe
    corruption for oscillatory data (Sines, ECG) but a weak one for short,
    smooth, highly autocorrelated real-world windows (Stocks, Energy), where
    reordering barely moves the flattened or signature representation.
    Requiring the shuffle specifically to separate therefore reports a false
    alarm on those two datasets for *any* kernel, raw or signature (verified
    directly). Requiring separation from either corruption keeps the check
    sensitive to the real failure mode without the false alarm.
    """
    perm = torch.stack([x[torch.randperm(x.shape[0])] for x in X_real])
    same = distributional_loss(X_real, X_real_heldout, cfg).item()
    shuf = distributional_loss(X_real, perm, cfg).item()
    noise = distributional_loss(X_real, torch.randn_like(X_real), cfg).item()
    ok = max(shuf, noise) > max(10 * same, 1e-4)
    if verbose:
        print(f"  D({cfg.kind}; real, real_heldout) = {same:.6f}   <- should be ~0")
        print(f"  D({cfg.kind}; real, time-shuffled) = {shuf:.6f}")
        print(f"  D({cfg.kind}; real, gaussian)      = {noise:.6f}   <- at least one >> same")
        print(f"  kernel alive: {ok}")
    return {"same": same, "shuffled": shuf, "noise": noise, "alive": ok}
