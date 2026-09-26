"""Train RecFlowTime, or any single-component ablation of it.

Each entry in VARIANTS is a set of config overrides applied on top of the
defaults, so an arm differs from the full method in exactly the stated respect
and a difference in score is attributable to that component.

  ours          the full method: transport coupling, rotary embeddings,
                self-conditioning, floored min-SNR weighting
  no_ot         independent pairing instead of the transport coupling
  no_rope       fixed sinusoidal positions instead of rotary
  no_selfcond   self-conditioning branch removed
  no_minsnr     uniform loss weight (removes min-SNR and its floor together)
  floor000      min-SNR kept, floor removed -- isolates the floor
  ddpm_core     eps-prediction DDPM/DDIM core, coupling and floor retained

Results are written to <dir>/<dataset>_<variant>{.json,_data.npz,_ckpt.pt,
_history.json}, the layout scripts/evaluate_saved.py expects.

Example
-------
    python scripts/run_variants.py --datasets sines --variants ours \\
        --steps 8000 --sampling-steps 20 --min-snr-floor 0.1 --dir results/run
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from recflowtime.config import RecFlowTimeConfig
from recflowtime.data import make_dataset, split
from recflowtime.train import RecFlowTimeTrainer
from recflowtime.metrics import evaluate_all, fit_context_encoder, discriminative_score, predictive_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAN = {"sines": (64, 96, 4), "energy": (96, 112, 4),
        "stocks": (128, 128, 4), "ecg": (192, 160, 4)}
STEPS = 10000
SAMPLING_STEPS = None          # set from the CLI; falls back to 20
LOG_EVERY = 1000               # history sampling interval; smaller = smoother curves
PROBE_EVERY = 2000
MIN_SNR_FLOOR = None           # set from the CLI
PRECOND_LAMBDA = 0.5          # transition errors cost ~3x smooth errors; see flow.py
MULTI_LAGS = (1, 2, 4, 8, 16, 32)   # dyadic lags: adjacent steps up to the signal period

VARIANTS = {
    # ---- final leave-one-out design (round 6) -------------------------------
    # `ours` is the method; every `no_*` arm is `ours` minus exactly one
    # component, so a difference is attributable to that component alone.
    "ours":            {"core.ot_coupling": True},
    "no_ot":           {},
    "no_rope":         {"core.ot_coupling": True, "denoiser.use_rope": False},
    "no_selfcond":     {"core.ot_coupling": True, "denoiser.use_self_cond": False},
    "no_minsnr":       {"core.ot_coupling": True, "core.use_min_snr": False},
    # ---- min-SNR floor study -----------------------------------------------
    # The unfloored weight vanishes as t -> 0, leaving the low-noise end of the
    # path almost unsupervised and the velocity field rough near the data.
    "floor000":        {"core.ot_coupling": True, "core.min_snr_floor": 0.0},
    "floor005":        {"core.ot_coupling": True, "core.min_snr_floor": 0.05},
    "floor010":        {"core.ot_coupling": True, "core.min_snr_floor": 0.10},
    # ---- generative-core arm (round 7) --------------------------------------
    # Swaps rectified flow for eps-prediction DDPM/DDIM and changes nothing
    # else: diffusion.py carries the same coupling and the same floored
    # min-SNR weight, so a difference is attributable to the core alone.
    "ddpm_core":       {"core.ot_coupling": True, "core.kind": "ddpm"},
    # ---- exploratory arms kept for the record; all four were rejected -------
    "baseline":        {},
    "precond":         {"core.precond_lambda": PRECOND_LAMBDA},
    "lagbias":         {"denoiser.use_lag_bias": True},
    "precond_lagbias": {"core.precond_lambda": PRECOND_LAMBDA, "denoiser.use_lag_bias": True},
    "ot":              {"core.precond_lambda": PRECOND_LAMBDA, "denoiser.use_lag_bias": True,
                        "core.ot_coupling": True},
    # round 5: `ot` above bundles three features, so a win there is not
    # attributable. This is the coupling alone, against `baseline`.
    "ot_only":         {"core.ot_coupling": True},
    # round 4: does the difference metric help more across scales, and does a
    # prior that already has the data's spectrum help on top of it?
    "mlag":            {"core.precond_lambda": PRECOND_LAMBDA, "core.precond_lags": MULTI_LAGS},
    "cprior":          {"core.precond_lambda": PRECOND_LAMBDA, "core.colored_prior": True},
    "mlag_cprior":     {"core.precond_lambda": PRECOND_LAMBDA, "core.precond_lags": MULTI_LAGS,
                        "core.colored_prior": True},
}
SEQ_LEN_OVERRIDE = {}          # dataset -> sequence length, set from the CLI


def build_cfg(dataset, variant, probe_every):
    T, d_model, n_heads = PLAN[dataset]
    if dataset in SEQ_LEN_OVERRIDE:
        T, d_model = SEQ_LEN_OVERRIDE[dataset]
    X, _, meta = make_dataset(dataset, seed=0, seq_len=T)
    over = {"denoiser.d_model": d_model, "denoiser.n_heads": n_heads,
            "train.warmup_steps": STEPS, "train.joint_steps": 0,
            "train.log_every": LOG_EVERY, "train.val_every": 10 ** 9,
            "core.sampling_steps": SAMPLING_STEPS or 20}
    if MIN_SNR_FLOOR is not None:
        over["core.min_snr_floor"] = MIN_SNR_FLOOR
    over.update(VARIANTS[variant])
    cfg = RecFlowTimeConfig.for_dataset(T, meta["n_features"], **over)
    cfg.spectral.enabled = False          # no loss on generated samples in this round
    return cfg, X, meta


def run(dataset, variant, out_dir, seed=0, probe_every=None):
    # seed 0 keeps the original filenames, so existing results stay valid;
    # further seeds get their own tag instead of colliding with them.
    tag = f"{dataset}_{variant}" + ("" if seed == 0 else f"_s{seed}")
    if os.path.exists(os.path.join(out_dir, f"{tag}.json")):
        print(f"[skip] {tag}"); return
    print(f"\n{'#'*64}\n# {tag}\n{'#'*64}", flush=True)
    torch.manual_seed(seed)
    probe_every = probe_every or PROBE_EVERY
    cfg, X, meta = build_cfg(dataset, variant, probe_every)
    Xtr, Xva, Xte = split(X, seed=seed)

    probe_hist = []

    def probe(trainer, step):
        f = trainer.generate(600, n_steps=cfg.core.sampling_steps)
        d = discriminative_score(Xte[:600], f, steps=300, device="cpu", seed=seed)
        p = predictive_score(Xte[:600], f, steps=300, device="cpu", seed=seed)
        probe_hist.append({"step": step, "discriminative": d, "predictive": p, "stage": "warmup"})
        print(f"    [probe @ {step}] disc={d:.4f} pred={p:.4f}", flush=True)

    t0 = time.time()
    tr = RecFlowTimeTrainer(cfg, Xtr, Xva, device=None, verbose=True)
    tr.train_diffusion(warmup_steps=STEPS, joint_steps=0,
                       probe_every=probe_every, probe_fn=probe)
    train_time = time.time() - t0

    n_gen = max(len(Xte), 1000)
    t1 = time.time()
    fake = tr.generate(n_gen, n_steps=cfg.core.sampling_steps)
    gen_time = time.time() - t1

    enc = fit_context_encoder(Xtr[:4000], device="cpu", steps=800, seed=seed)
    metrics = evaluate_all(Xte, fake[:len(Xte)], device="cpu", context_encoder=enc,
                           seed=seed, repeat=1, quick=False)

    np.savez_compressed(os.path.join(out_dir, f"{tag}_data.npz"),
                        real_test=Xte.numpy(), real_train=Xtr[:2000].numpy(),
                        real_val=Xva.numpy(), generated=fake.numpy())
    torch.save({"ema_state": tr.ema.model.state_dict(), "model_state": tr.model.state_dict(),
                "config": json.loads(cfg.to_json()), "meta": meta},
               os.path.join(out_dir, f"{tag}_ckpt.pt"))
    json.dump({k: list(v) for k, v in tr.history.items()},
              open(os.path.join(out_dir, f"{tag}_history.json"), "w"), default=float)
    json.dump({"dataset": dataset, "variant": variant, "seed": seed, "meta": meta,
               "train_time_s": train_time, "gen_time_s": gen_time,
               "denoiser_params": tr.model.n_params(), "metrics": metrics,
               "probe": probe_hist, "budget": {"warmup_steps": STEPS, "joint_steps": 0},
               "config": json.loads(cfg.to_json())},
              open(os.path.join(out_dir, f"{tag}.json"), "w"), indent=2, default=float)
    print(f"  === {tag}: disc={metrics['discriminative']:.4f} pred={metrics['predictive']:.4f} "
          f"ctx-fid={metrics['context_fid']:.4f} acf={metrics['acf_dist']:.4f} "
          f"delta={metrics['delta_dist']:.4f} spec={metrics['spectral_dist']:.4f} "
          f"({train_time/60:.1f} min) ===", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="sines")
    ap.add_argument("--variants", default="baseline,precond,lagbias,precond_lagbias")
    ap.add_argument("--dir", default=os.path.join("results", "round2"))
    ap.add_argument("--seq-len", default="", help="e.g. ecg:64:96 -> dataset:T:d_model")
    ap.add_argument("--seeds", default="0", help="comma-separated training seeds")
    ap.add_argument("--steps", type=int, default=None,
                    help="training steps (default: module STEPS)")
    ap.add_argument("--sampling-steps", type=int, default=None,
                    help="Euler steps used at generation (default: config)")
    ap.add_argument("--log-every", type=int, default=1000,
                    help="how often loss and gradient norm are recorded; "
                         "smaller gives denser training curves")
    ap.add_argument("--probe-every", type=int, default=2000,
                    help="how often discriminative/predictive probes are run")
    ap.add_argument("--min-snr-floor", type=float, default=None,
                    help="lower bound on the min-SNR loss weight; keeps the "
                         "low-noise end of the path supervised")
    args = ap.parse_args()
    global STEPS, SAMPLING_STEPS, LOG_EVERY, PROBE_EVERY, MIN_SNR_FLOOR
    if args.steps:
        STEPS = args.steps
    SAMPLING_STEPS = args.sampling_steps
    LOG_EVERY = args.log_every
    PROBE_EVERY = args.probe_every
    MIN_SNR_FLOOR = args.min_snr_floor
    for spec in filter(None, args.seq_len.split(",")):
        ds, T, d = spec.split(":")
        SEQ_LEN_OVERRIDE[ds] = (int(T), int(d))
    out_dir = args.dir if os.path.isabs(args.dir) else os.path.join(ROOT, args.dir)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    for s in (int(x) for x in args.seeds.split(",")):
        for ds in args.datasets.split(","):
            for v in args.variants.split(","):
                run(ds, v, out_dir, seed=s)
    print(f"\nALL VARIANTS DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
