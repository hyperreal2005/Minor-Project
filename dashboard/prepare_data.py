"""Snapshot the result tables the dashboard reads into dashboard/data/.

    venv\\Scripts\\python.exe dashboard\\prepare_data.py

`results/` is gitignored (it comes off Kaggle), so the dashboard keeps its own small, committed
copy: it then runs on any laptop with the repo checked out, no Kaggle download needed. Re-run
this after unzipping a new results bundle. Optional inputs are picked up when present:

* results/calibration/candidates.parquet -- each retrain's and M0's own audit values (notebook
  06 at commit >= the candidates export). Enables the per-retrain views.
* results/curves/*.parquet -- relearning curves step by step (notebook 07).
* data/cifar-10-batches-py -- for the forget-set gallery thumbnails.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = Path(__file__).resolve().parent / "data"
RES = ROOT / "results"

CALIBRATION = ["flags", "bands", "validity", "canary", "canary_validity", "utility", "training",
               "relearning_protocol", "relearning_vs_accuracy", "twins", "provenance"]
ANALYSIS = ["correlations", "canary_continuous", "mixed_effects", "method_summary", "h3",
            "h3_controlled", "difficulty", "interaction", "correlation_matrix", "disagreement",
            "rank_tables", "rq4_by_method", "agreement_by_condition"]
GALLERY_PER_SET = 24


def copy_tables() -> dict:
    found = {}
    for sub, names in (("calibration", CALIBRATION + ["candidates"]), ("analysis", ANALYSIS)):
        for n in names:
            src = RES / sub / f"{n}.parquet"
            if src.is_file():
                shutil.copy2(src, OUT / f"{n}.parquet")
                found[n] = len(pd.read_parquet(src))
            elif n != "candidates":
                raise SystemExit(f"missing {src}: unzip the Stage 8 results bundle first")
    return found


def signed_gaps() -> int:
    """Signed G for every healthy model and analysed metric: 0 at the retrains, 1 at M0."""
    from forgetcheck.analysis.agreement import KEYS, signed_gaps as sg

    flags = pd.read_parquet(OUT / "flags.parquet")
    validity = pd.read_parquet(OUT / "validity.parquet")
    bands = pd.read_parquet(OUT / "bands.parquet")
    f = flags[~flags["damaged"].fillna(False).astype(bool)]
    S = sg(f, validity, bands, f.set_index(KEYS).index.unique()).reset_index()
    S.to_parquet(OUT / "signed_gaps.parquet", index=False)
    return len(S)


def curves() -> int:
    files = sorted((RES / "curves").glob("*.parquet")) if (RES / "curves").is_dir() else []
    if not files:
        (OUT / "curves.parquet").unlink(missing_ok=True)
        return 0
    c = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    c.to_parquet(OUT / "curves.parquet", index=False)
    return len(c)


def gallery() -> int:
    """A fixed sample of real images from every forget set, as one uint8 array + metadata."""
    from forgetcheck.data.cifar import canary_labels, load_cifar

    cifar = ROOT / "data"
    if not (cifar / "cifar-10-batches-py").is_dir():
        return 0
    bundle = load_cifar(cifar, download=False)
    mem = np.load(ROOT / "data" / "memorization" / "cifar10_memorization.npy")
    rng = np.random.default_rng(300)
    images, rows = [], []
    for npz in sorted((ROOT / "artifacts" / "forget_sets").glob("*.npz")):
        cond = npz.stem
        idx = np.load(npz)["indices"].astype(np.int64)
        pick = np.sort(rng.choice(idx, size=min(GALLERY_PER_SET, len(idx)), replace=False))
        planted = canary_labels(bundle.train_y, pick) if cond.startswith("canary") else None
        for k, i in enumerate(pick):
            images.append(bundle.train_x[i])
            rows.append({"forget_id": cond, "index": int(i), "label": int(bundle.train_y[i]),
                         "planted": int(planted[k]) if planted is not None else -1,
                         "memorization": float(mem[i])})
    if images:
        np.save(OUT / "gallery_images.npy", np.stack(images))
        pd.DataFrame(rows).to_parquet(OUT / "gallery.parquet", index=False)
    return len(images)


def forget_indices() -> int:
    """Every forget set's training-image indices (artifacts/ is gitignored)."""
    rows = []
    for npz in sorted((ROOT / "artifacts" / "forget_sets").glob("*.npz")):
        idx = np.load(npz)["indices"].astype(np.int32)
        rows.append(pd.DataFrame({"forget_id": npz.stem, "index": idx}))
    if rows:
        pd.concat(rows, ignore_index=True).to_parquet(OUT / "forget_indices.parquet", index=False)
    return sum(len(r) for r in rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    found = copy_tables()
    forget_indices()
    n_sg = signed_gaps()
    n_curves = curves()
    n_gallery = gallery()
    shutil.copy2(ROOT / "data" / "memorization" / "cifar10_memorization.npy",
                 OUT / "memorization.npy")
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"
    meta = {"built": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "commit": commit,
            "has_candidates": "candidates" in found, "curves_rows": n_curves,
            "gallery_images": n_gallery, "tables": found}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for k, v in found.items():
        print(f"  {k:24s} {v:6d} rows")
    print(f"  {'signed_gaps':24s} {n_sg:6d} rows")
    print(f"  {'curves':24s} {n_curves:6d} rows" + ("" if n_curves else "  (none yet: notebook 07)"))
    print(f"  {'gallery':24s} {n_gallery:6d} images" + ("" if n_gallery else "  (no CIFAR-10 here)"))
    if "candidates" not in found:
        print("  candidates: not in results/calibration yet (re-run notebook 06 at the new commit)")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
