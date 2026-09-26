"""Training loop, EMA, and sampling.

The reported configuration trains in one stage: the flow-matching loss alone,
for `warmup_steps` steps, with an exponential moving average of the weights at
decay 0.999 that is what sampling uses. A second `joint_steps` stage exists for
the optional regularisers in `losses.py` and is disabled by default
(`joint_steps = 0`), as is the optional critic stage (`aux.enabled = False`),
so neither contributes to any reported result.

`core.kind` selects the rectified-flow or the DDPM core; everything else -- data
pipeline, model family, optimiser, EMA, metrics -- is shared across arms, so
ablations differ only in the stated respect.
"""
import copy
import json
import math
import time
from collections import defaultdict

import numpy as np
import torch

from .config import RecFlowTimeConfig
from .data import InfiniteLoader, loader
from .denoiser import TransformerDenoiser
from .aux_nets import AuxCritics
from .flow import RectifiedFlow
from .diffusion import GaussianDiffusion
from .losses import RecFlowTimeLoss
from .masking import masked_mse


def pick_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_core(cfg: RecFlowTimeConfig, device):
    if cfg.core.kind == "flow":
        return RectifiedFlow(cfg.core, device)
    if cfg.core.kind == "ddpm":
        return GaussianDiffusion(cfg.core, device)
    raise ValueError(cfg.core.kind)


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(b)

    @property
    def model(self):
        return self.shadow


class RecFlowTimeTrainer:
    def __init__(self, cfg: RecFlowTimeConfig, X_train, X_val=None, device=None, verbose=True):
        self.cfg = cfg
        self.device = pick_device(device or cfg.train.device)
        self.verbose = verbose
        torch.manual_seed(cfg.train.seed)
        np.random.seed(cfg.train.seed)

        self.X_train, self.X_val = X_train, X_val
        self.dl = InfiniteLoader(loader(X_train, cfg.train.batch_size, seed=cfg.train.seed))

        self.model = TransformerDenoiser(cfg.denoiser).to(self.device)
        self.critics = AuxCritics(cfg.aux).to(self.device) if cfg.aux.enabled else None
        self.core = build_core(cfg, self.device)
        if getattr(cfg.core, "colored_prior", False) and cfg.core.kind == "flow":
            self.core.fit_noise_spectrum(X_train, seed=cfg.train.seed)
        self.ema = None
        self.history = defaultdict(list)
        extras = [n for n, on in (("spectral", cfg.spectral.enabled),
                                  (f"dist:{cfg.dist.kind}", cfg.train.joint_steps > 0),
                                  ("critics", cfg.aux.enabled)) if on]
        self._log(f"device={self.device}  core={cfg.core.kind}  denoiser params="
                  f"{self.model.n_params():,}  ot_coupling={cfg.core.ot_coupling}  "
                  f"rope={cfg.denoiser.use_rope}  "
                  f"self_cond={cfg.denoiser.use_self_cond}  "
                  f"min_snr={cfg.core.use_min_snr} (floor {cfg.core.min_snr_floor:g})  "
                  f"extra terms: {', '.join(extras) if extras else 'none'}")

    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def _record(self, stage, step, logs):
        self.history["stage"].append(stage)
        self.history["step"].append(step)
        for k, v in logs.items():
            self.history[k].append(v)

    # ============== optional pre-stage: critics (off unless aux.enabled) ======
    def train_critics(self, steps=None, early_stop_patience=6):
        """No-op unless `cfg.aux.enabled`; see `aux_nets.py`."""
        if self.critics is None:
            self._log("\n[no critic stage] RecFlowTime has no auxiliary critics -- "
                      "both regularisers are non-parametric batch statistics.")
            return self
        steps = steps or self.cfg.train.aux_steps
        self._log(f"\n[critic pretraining] auxiliary critics  ({steps} steps)")
        opt = torch.optim.Adam(self.critics.parameters(), lr=self.cfg.train.aux_lr,
                               betas=self.cfg.train.betas)
        best, bad, best_state = math.inf, 0, None
        t0 = time.time()
        for i in range(1, steps + 1):
            x = self.dl.next().to(self.device)
            loss, logs = self.critics.aux_loss(x)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics.parameters(), self.cfg.train.grad_clip)
            opt.step()
            if i % self.cfg.train.log_every == 0:
                self._record("aux", i, logs)
            if self.X_val is not None and i % self.cfg.train.val_every == 0:
                v = self.validate_critics()
                self._log(f"  step {i:>6}  train={loss.item():.5f}  "
                          f"val_int={v['int']:.5f}  val_ext={v['ext']:.5f}")
                tot = v["int"] + v["ext"]
                if tot < best - 1e-6:
                    best, bad, best_state = tot, 0, copy.deepcopy(self.critics.state_dict())
                else:
                    bad += 1
                    if bad >= early_stop_patience:
                        self._log(f"  early stop at step {i} (val plateau)")
                        break
        if best_state is not None:
            self.critics.load_state_dict(best_state)
            self._log(f"  restored best critics (val total={best:.5f})")
        self.critics.freeze()
        self._log(f"  critics frozen. {time.time()-t0:.1f}s")
        return self

    @torch.no_grad()
    def validate_critics(self, n=1024, n_draws=4):
        self.critics.eval()
        X = self.X_val[:n].to(self.device)
        out = {}
        for kind in ("int", "ext"):
            vals = [self.critics.discrepancy(X, kind).item() for _ in range(n_draws)]
            out[kind] = float(np.mean(vals))
        self.critics.train()
        return out

    # ============================= main training: warmup -> joint ============
    def train_diffusion(self, warmup_steps=None, joint_steps=None,
                        use_aux=True, use_dist=True, probe_every=0, probe_fn=None):
        """`probe_fn(trainer, global_step)` is called every `probe_every`
        optimiser steps, so sample quality can be tracked *during* training
        rather than only at the end -- which is what distinguishes "needs a
        longer budget" from "converged and limited by something else"."""
        tc = self.cfg.train
        warmup = tc.warmup_steps if warmup_steps is None else warmup_steps
        joint = tc.joint_steps if joint_steps is None else joint_steps

        self.lossfn = RecFlowTimeLoss(
            self.core, self.cfg.dist, self.cfg.spectral, critics=self.critics,
            lambda_ext=tc.lambda_ext, lambda_int=tc.lambda_int,
            lambda_spec=tc.lambda_spec, lambda_dist=tc.lambda_dist,
            ref_ema_decay=tc.ref_ema_decay,
            use_self_cond=self.cfg.denoiser.use_self_cond,
            dist_sample_steps=tc.dist_sample_steps, dist_batch=tc.dist_batch,
            sample_clamp=tc.sample_clamp, dist_grad_steps=tc.dist_grad_steps,
        ).to(self.device)

        opt = torch.optim.Adam(self.model.parameters(), lr=tc.lr, betas=tc.betas,
                               weight_decay=tc.weight_decay)
        total = warmup + joint
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / 500) * (0.5 * (1 + math.cos(math.pi * s / total)))
        )
        self.ema = EMA(self.model, tc.ema_decay)

        n_stages = sum(1 for n in (warmup, joint) if n > 0)
        global_step = 0
        for k, (stage, n, aux_on, dist_on) in enumerate(
                (("warmup", warmup, False, False),
                 ("joint", joint, use_aux, use_dist)), start=1):
            if n == 0:
                continue
            extra = "" if stage == "warmup" else f"  regularisers={aux_on} dist={dist_on}"
            label = f"[stage {k}/{n_stages}] " if n_stages > 1 else ""
            self._log(f"\n{label}{self.cfg.core.kind} {stage}  ({n} steps){extra}")
            t0 = time.time()
            for i in range(1, n + 1):
                x = self.dl.next().to(self.device)
                dist_now = dist_on and (i % tc.dist_every == 0)
                x_ref = self.dl.next().to(self.device) if dist_now else None
                loss, logs = self.lossfn(self.model, x, x_ref, use_aux=aux_on, use_dist=dist_now)
                opt.zero_grad(); loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.grad_clip)
                opt.step(); sched.step(); self.ema.update(self.model)
                global_step += 1
                if i % tc.log_every == 0:
                    logs["grad_norm"] = float(gn)
                    logs["lr"] = opt.param_groups[0]["lr"]
                    self._record(stage, i, logs)
                    self._log("  " + self._fmt(i, logs))
                if probe_every and probe_fn is not None and global_step % probe_every == 0:
                    probe_fn(self, global_step)
            self._log(f"  {stage} done in {time.time()-t0:.1f}s")
        return self

    @staticmethod
    def _fmt(i, logs):
        keys = ["core", "disc_int", "disc_ext", "spec_loss", "dist", "total", "grad_norm"]
        parts = [f"step {i:>6}"]
        for k in keys:
            if k in logs:
                parts.append(f"{k}={logs[k]:.5f}")
        return "  ".join(parts)

    # ======================================================= state save/load
    def save_state(self, path):
        """Persist model + EMA so the joint stage can be branched from a
        single shared warmup. The warmup stage optimises L_flow alone, so it
        is independent of lambda_spec/lambda_dist -- sweeping those by
        re-running warmup each time would repeat identical work."""
        torch.save({"model": self.model.state_dict(),
                    "ema": self.ema.model.state_dict() if self.ema else None,
                    "config": json.loads(self.cfg.to_json())}, path)
        return path

    def load_state(self, path):
        st = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(st["model"])
        if st.get("ema") is not None:
            if self.ema is None:
                self.ema = EMA(self.model, self.cfg.train.ema_decay)
            self.ema.shadow.load_state_dict(st["ema"])
        return self

    # ============================================================== generation
    def generate(self, n, n_steps=None, use_ema=True, batch=512, clamp=1.0):
        model = (self.ema.model if (use_ema and self.ema is not None) else self.model).eval()
        T, Fdim = self.cfg.denoiser.seq_len, self.cfg.denoiser.n_features
        n_steps = n_steps or self.cfg.core.sampling_steps
        outs = []
        with torch.no_grad():
            for s in range(0, n, batch):
                b = min(batch, n - s)
                x = self.core.sample(model, (b, T, Fdim), n_steps=n_steps, device=self.device,
                                     clamp=clamp, use_self_cond=self.cfg.denoiser.use_self_cond)
                outs.append(x.cpu())
        return torch.cat(outs)[:n]

    # ============================================================ diagnostics
    def health_check(self, n=512):
        """Check that each active term can actually act on the denoiser.

        Every one of these catches a failure that is otherwise silent: a
        regulariser whose gradient never reaches theta, or a kernel whose
        bandwidth has saturated so that its value is constant (and its
        gradient zero) regardless of how good the samples are.
        """
        print("\n=== health check ===")
        ok = True
        x = self.dl.next().to(self.device)

        # 1. spectral discrepancy -> gradient on theta
        if self.lossfn.spec is not None:
            x_hat, *_ = self.core.x1_hat(
                self.model, x, use_self_cond=self.cfg.denoiser.use_self_cond)
            spec_loss, _ = self.lossfn.spec(x_hat, x)
            self.model.zero_grad()
            spec_loss.backward()
            g = sum(p.grad.abs().sum().item()
                    for p in self.model.parameters() if p.grad is not None)
            print(f"  spectral grad reaching denoiser : {g:.4e}  {'OK' if g > 0 else 'DEAD'}")
            ok &= g > 0
            self.model.zero_grad()

        # 2. distributional kernel alive on this data
        from .distributional import mmd_sanity_check
        n2 = min(n, len(self.X_val) if self.X_val is not None else n)
        a = self.X_train[:n2]
        b = self.X_val[:n2] if self.X_val is not None else self.X_train[n2:2 * n2]
        chk = mmd_sanity_check(a, b, self.cfg.dist, verbose=False)
        print(f"  {self.cfg.dist.kind} kernel: real/real'={chk['same']:.6f}  "
              f"real/shuffled={chk['shuffled']:.6f}  real/noise={chk['noise']:.6f}  "
              f"alive={chk['alive']}")
        ok &= chk["alive"]

        # 2b. does the estimator survive its own noise floor at the batch size
        # actually used in training? An unbiased MMD estimate between two real
        # batches straddles zero; if it lands <= 0 often, clamping kills the
        # gradient and the term silently stops training the model -- which is
        # exactly what happened in the first full-budget RecFlowTime run.
        from .distributional import distributional_loss
        nb = self.cfg.train.dist_batch
        if len(self.X_train) >= 2 * nb:
            vals = []
            for i in range(8):
                p = self.X_train[i * nb:(i + 1) * nb].to(self.device)
                q = self.X_train[(i + 8) * nb:(i + 9) * nb].to(self.device)
                if len(p) < nb or len(q) < nb:
                    break
                with torch.no_grad():
                    vals.append(distributional_loss(p, q, self.cfg.dist).item())
            if vals:
                v = torch.tensor(vals)
                frac_dead = float((v <= 0).float().mean())
                good = frac_dead < 0.25
                print(f"  estimator @ batch {nb}: real-vs-real mean={v.mean():+.5f} "
                      f"std={v.std():.5f} frac<=0={frac_dead:.0%}  "
                      f"{'OK' if good else 'GRADIENT DIES (use unbiased=False)'}")
                ok &= good

        # 3. critics, when present (baseline configuration only)
        if self.critics is not None and self.X_val is not None:
            v = self.validate_critics()
            Xv = self.X_val[:1024].to(self.device)
            for kind in ("int", "ext"):
                m = self.critics.sample_masks(len(Xv), self.device)[kind]
                vis = (1 - m).unsqueeze(-1)
                mean = (Xv * vis).sum(1, keepdim=True) / vis.sum(1, keepdim=True).clamp(min=1)
                base = masked_mse(mean.expand_as(Xv), Xv, m).item()
                good = v[kind] < base
                print(f"  critic f_{kind}: val={v[kind]:.5f} vs mean-baseline={base:.5f}"
                      f"  {'OK' if good else 'NOT LEARNING'}")
                ok &= good
        print(f"=== {'all checks passed' if ok else 'SOME CHECKS FAILED'} ===\n")
        return ok


def build_and_train(X_train, X_val, cfg=None, seq_len=None, n_features=None,
                    use_aux=True, use_dist=True, device=None,
                    verbose=True, **overrides):
    if cfg is None:
        cfg = RecFlowTimeConfig.for_dataset(seq_len or X_train.shape[1],
                                      n_features or X_train.shape[2], **overrides)
    tr = RecFlowTimeTrainer(cfg, X_train, X_val, device=device, verbose=verbose)
    tr.train_critics()          # no-op unless cfg.aux.enabled (baseline)
    tr.train_diffusion(use_aux=use_aux, use_dist=use_dist)
    return tr
