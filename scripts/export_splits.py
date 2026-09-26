"""Export the canonical train/validation/test splits.

Every method compared in the paper must be trained on exactly the same training
split and evaluated against exactly the same test split, or a difference in
scores is confounded with a difference in data. This writes the arrays that
`run_variants.py` itself constructs, through the same calls in the same order,
so there is one definition of the data and every method reads it from disk.

A competing method should write its samples as `<dataset>_<method>_data.npz`
with the same keys (real_test, real_train, real_val, generated), after which
`evaluate_saved.py --suffix <method>` scores it unchanged.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from recflowtime.data import make_dataset, split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAN = {"sines": 64, "energy": 96, "stocks": 128, "ecg": 192}
OUT = os.path.join(ROOT, "data", "splits")

os.makedirs(OUT, exist_ok=True)
manifest = {}
for ds, T in PLAN.items():
    X, _, meta = make_dataset(ds, seed=0, seq_len=T)
    Xtr, Xva, Xte = split(X, seed=0)
    f = os.path.join(OUT, f"{ds}_T{T}.npz")
    np.savez_compressed(f, train=Xtr.numpy(), val=Xva.numpy(), test=Xte.numpy())
    manifest[ds] = {"seq_len": T, "n_features": int(meta["n_features"]),
                    "train": list(Xtr.shape), "val": list(Xva.shape),
                    "test": list(Xte.shape), "file": os.path.relpath(f, ROOT),
                    "range": [float(X.min()), float(X.max())]}
    print(f"{ds:8} T={T:3}  train={tuple(Xtr.shape)}  val={tuple(Xva.shape)}  "
          f"test={tuple(Xte.shape)}  range=[{X.min():.3f},{X.max():.3f}]")

json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=2)
print(f"\nwrote {len(manifest)} splits + manifest.json to {os.path.relpath(OUT, ROOT)}")
