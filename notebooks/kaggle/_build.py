"""Generate the Kaggle notebooks.

The four notebooks share almost all of their structure, so they are generated from one spec
rather than maintained as four drifting copies of the same JSON. Re-run after changing anything
here:

    python notebooks/kaggle/_build.py

Each notebook is deliberately thin: install the package at a pinned commit, run one CLI command,
inspect the results, push the artefacts. The implementations live in tested modules, which is
what makes the ``git_commit`` recorded in every row mean something, and what stops a dozen
accounts from each carrying a drifting copy of the training loop.

The inspection cells are *not* an afterthought — they are where you actually read what happened.
What is absent is a 300-line training loop pasted into a cell.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent

# --------------------------------------------------------------------------- shared cells

INSTALL = """\
# --- clone the repo at a PINNED commit ------------------------------------------------------
# Clone rather than `pip install git+...`. A wheel would contain only src/forgetcheck/, but the
# CLI also needs configs/ (the metric registry, seed streams, audit protocols) and
# data/memorization/ (the RUM scores -- 400 KB, committed precisely so a fresh session does not
# have to re-download 2 GB from Google Drive).
#
# Pinning is what makes provenance work: every record this session writes carries this commit,
# so any result can be traced back to the exact code that produced it.
REPO   = "https://github.com/hyperreal2005/Minor-Project.git"
COMMIT = "main"          # <-- pin to a sha for real runs, e.g. "a1b2c3d"

import os
from pathlib import Path

os.chdir("/kaggle/working")
if not Path("Minor-Project").exists():
    !git clone --quiet $REPO
%cd /kaggle/working/Minor-Project
!git fetch --quiet --all && git checkout --quiet $COMMIT
!git log -1 --format="pinned at %h  %s"
!pip install -q -e .
"""

SETUP = """\
import importlib
import sys
from pathlib import Path

REPO_DIR = Path("/kaggle/working/Minor-Project")

# Make the package importable *in this kernel*.
#
# `pip install -e .` writes a .pth file into site-packages, and .pth files are only processed at
# interpreter startup. The kernel was already running when the previous cell installed, so
# sys.path never picked it up and `import forgetcheck` fails with ModuleNotFoundError. The
# `!forgetcheck` CLI calls below are unaffected -- each spawns a fresh Python that does read the
# .pth -- which makes this failure look stranger than it is.
#
# Adding src/ directly is deterministic and avoids making anyone restart the kernel.
SRC = str(REPO_DIR / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
importlib.invalidate_caches()

# `!` cells run in a subshell, which inherits this. Belt and braces: the console script should
# already be on PATH after the editable install, but if it is not, PYTHONPATH keeps
# `python -m forgetcheck.cli` working as a fallback.
import os
os.environ["PYTHONPATH"] = SRC + os.pathsep + os.environ.get("PYTHONPATH", "")

import forgetcheck
print("forgetcheck imported from:", Path(forgetcheck.__file__).parent)

# CIFAR-10 downloads at roughly 130 kB/s on Kaggle -- about 20 minutes, repeated on every
# session and every account. Attaching it as a Dataset (Add Input -> Datasets) skips that
# entirely: torchvision checks md5s of the extracted folder and only downloads if it is missing
# or corrupt. See 00_verify_setup.ipynb for how to create the dataset once.
CIFAR_IN = Path("/kaggle/input/datasets/pranitdeepsingh/forgetcheck-cifar10")
if CIFAR_IN.exists():
    (REPO_DIR / "data").mkdir(parents=True, exist_ok=True)
    !cp -r $CIFAR_IN/cifar-10-batches-py $REPO_DIR/data/ 2>/dev/null || true
    print("CIFAR-10 restored from the attached dataset - no download needed")
else:
    print("no CIFAR-10 dataset attached - it will download (~20 min)")

# Attach a previous artefact dataset (Add Input -> Datasets) so completed runs are skipped
# rather than recomputed. Without it, every session starts from nothing.
ARTIFACTS_IN = Path("/kaggle/input/forgetcheck-artifacts")
if ARTIFACTS_IN.exists():
    !cp -r $ARTIFACTS_IN/artifacts $REPO_DIR/ 2>/dev/null || true
    !cp -r $ARTIFACTS_IN/results   $REPO_DIR/ 2>/dev/null || true
    print("restored artefacts from a previous session")
else:
    print("no previous artefacts attached - starting fresh")

import torch
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
else:
    print("!! running on CPU. Set Settings -> Accelerator -> GPU.")
    print("!! On CPU one 30-epoch training run takes ~4.8 hours instead of ~12 minutes.")

import shutil
CLI = "forgetcheck" if shutil.which("forgetcheck") else f"{sys.executable} -m forgetcheck.cli"
print("cli:", CLI)
"""

STATUS = """\
!{CLI} --root . status
"""

PUSH = """\
# --- push the artefacts back out -------------------------------------------------------------
# Kaggle sessions are disposable. Anything not saved to a Dataset version is gone, and the next
# session would recompute it. /kaggle/working persists per notebook (~20 GB); a Dataset is what
# shares it between notebooks and accounts.
import json
from pathlib import Path

OUT = Path("/kaggle/working/to_upload")
OUT.mkdir(exist_ok=True)
!cp -r /kaggle/working/Minor-Project/artifacts $OUT/ 2>/dev/null || true
!cp -r /kaggle/working/Minor-Project/results   $OUT/ 2>/dev/null || true

META = OUT / "dataset-metadata.json"
META.write_text(json.dumps({
    "title": "forgetcheck-artifacts",
    "id": "YOUR-KAGGLE-USERNAME/forgetcheck-artifacts",   # <-- your username
    "licenses": [{"name": "CC0-1.0"}],
}, indent=2))

# Needs an API token at ~/.kaggle/kaggle.json (Kaggle -> Account -> Create New API Token).
# First time:   !kaggle datasets create  -p $OUT --dir-mode zip
# Afterwards:   !kaggle datasets version -p $OUT -m "stage N account K" --dir-mode zip
#
# Simplest alternative: just "Save Version" the notebook. /kaggle/working persists per notebook
# (~20 GB), which is enough for one account to resume itself -- but a Dataset is what shares
# artefacts between accounts, and that is what the team needs.
print(f"staged in {OUT} - uncomment whichever line above applies")
"""


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# --------------------------------------------------------------------------- the notebooks


def nb_verify() -> dict:
    return notebook([
        md("""\
# 00 - Verify the setup

Run this **first**, on one account, before any training. It confirms that this machine produces
byte-identical data to everyone else's, which is the precondition for any result being
comparable across the team.

Nothing here trains a model. It takes a few minutes.

### Before running - two Kaggle settings

In the right-hand panel:

1. **Settings -> Accelerator -> GPU** (T4 x2 or P100). Without it everything runs on CPU, where
   one 30-epoch training run takes ~4.8 hours instead of ~12 minutes.
2. **Settings -> Internet -> On.** Required to clone the repo and download CIFAR-10.
"""),
        code(INSTALL),
        code(SETUP),
        md("## Data and memorization scores\n\nBoth are hash-checked. A mismatch is fatal by "
           "design: a different download would silently make this account's results "
           "incomparable with everyone else's."),
        code("""\
from forgetcheck.config import Context, find_configs
from pathlib import Path

ctx = Context(configs=find_configs(), root=Path("."))
print("CIFAR-10 sha :", ctx.bundle.sha, " (expected 6b3883dca6c867f1)")
print("train/test   :", ctx.bundle.train_x.shape, ctx.bundle.test_x.shape)

mem = ctx.memorization
print("memorization :", mem.shape, f"[{mem.min():.3f}, {mem.max():.3f}]")
"""),
        md("## The eight forget conditions\n\nExpect `mem-low-3000` at exactly 0.0000, "
           "`mem-med-3000` near 0.5, `mem-high-3000` near 0.97."),
        code("""\
from forgetcheck.data.memorization import check_scores

report = check_scores(mem)          # raises if the scores are inverted or degenerate
summary = report.pop("summary")
for stratum, row in report.items():
    print(f"  {stratum:7s} mean={row['mean']:.4f}  std={row['std']:.4f}")
print("\\n ", {k: v for k, v in summary.items() if not k.startswith("overlap_")})
"""),
        code("!{CLI} --root . forget-sets"),
        md("## What the full matrix will cost\n\n`--dry-run` lists the work without doing any "
           "of it."),
        code("""\
!{CLI} --root . --dry-run queue --stage 3 --account 1 --of 3 | tail -5
!{CLI} --root . --dry-run queue --stage 4 --account 1 --of 3 | tail -5
!{CLI} --root . --dry-run queue --stage 5 --account 1 --of 3 | tail -5
"""),
        md("""\
## Save CIFAR-10 as a Kaggle Dataset — do this once

CIFAR-10 downloads at ~130 kB/s here, about **20 minutes**, and it would repeat on every session
and every account. Publishing it once as a private Dataset removes that from every future run.

Why publish our own rather than use one of the public CIFAR-10 datasets already on Kaggle: the
hash pinned in `configs/base.yaml` is of *this exact* torchvision-format download. A public
dataset in a different layout (PNGs, a re-pickled archive) would fail the hash check — correctly,
because it would not be the same bytes everyone else trained on.

The cell below stages the files. Then either run the `kaggle datasets create` line (needs an API
token at `~/.kaggle/kaggle.json` — Kaggle → Account → Create New API Token), or use the UI:
**Save Version** this notebook, then **New Dataset → From your notebook output**.
"""),
        code("""\
from pathlib import Path

CIFAR_OUT = Path("/kaggle/working/cifar10_dataset")
CIFAR_OUT.mkdir(exist_ok=True)

# Only the extracted folder is needed -- torchvision verifies its md5s and skips the download.
# The .tar.gz would just double the size.
!cp -r /kaggle/working/Minor-Project/data/cifar-10-batches-py $CIFAR_OUT/ 2>/dev/null || true

import json
(CIFAR_OUT / "dataset-metadata.json").write_text(json.dumps({
    "title": "forgetcheck-cifar10",
    "id": "YOUR-KAGGLE-USERNAME/forgetcheck-cifar10",   # <-- your username
    "licenses": [{"name": "CC0-1.0"}],
}, indent=2))

!du -sh $CIFAR_OUT
print("\\nthen run:  !kaggle datasets create -p", CIFAR_OUT, "--dir-mode zip")
print("afterwards, attach it via Add Input -> Datasets in every notebook")
"""),
        md("""\
### If everything above passed

This account is ready. Move on to `01_train.ipynb`.

If the CIFAR hash or the memorization hash differed, **stop** — do not train. Compare
`configs/base.yaml` against the repo and re-download rather than proceeding: a different download
would silently make this account's results incomparable with everyone else's, and that is
precisely what the hash exists to catch.
"""),
    ])


def nb_stage(number: str, stage: int, title: str, blurb: str, extra_md: str = "") -> dict:
    return notebook([
        md(f"""\
# {number} - {title}

{blurb}

**Set `ACCOUNT` below** to this account's number. Every account runs the identical notebook with
a different `ACCOUNT`, and they compute disjoint halves of the work with no coordination -- run
identifiers are pure functions of the config, so all accounts derive the same work list and each
takes its own stripe of it.

Runs already present in the attached artefact dataset are skipped, so a session that dies costs
only its in-flight model.
"""),
        code(INSTALL),
        code(SETUP),
        code(f"""\
ACCOUNT = 1     # <-- this account's number, 1-based
OF      = 3     # <-- how many accounts are sharing this stage

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
STAGE  = {stage}
print(f"account {{ACCOUNT}} of {{OF}}, stage {{STAGE}}, device {{DEVICE}}")
"""),
        md("## What this account will do"),
        code("""\
!{CLI} --root . --dry-run queue --stage {STAGE} --account {ACCOUNT} --of {OF}
""".replace("{STAGE}", "{STAGE}")),
        md("## Run it\n\nThis is the long cell. It prints each run as it starts and finishes, so "
           "you can watch progress and estimate the remaining time."),
        code("""\
!{CLI} --root . --device {DEVICE} queue --stage {STAGE} --account {ACCOUNT} --of {OF}
"""),
        md("## Inspect what came out" + extra_md),
        code("""\
from forgetcheck.registry import read_records
import pandas as pd

df = read_records("results/records")
print(f"{len(df)} rows across {df['run_id'].nunique()} runs\\n")

wide = df.pivot_table(index=["run_id", "role"], columns="metric", values="value")
wide.round(4)
"""),
        code("""\
# Seed-to-seed spread. This is not a diagnostic to average away -- it is the raw material for
# every oracle band in the calibration stage, and stage 3's gate is that it stays within 0.5 pp.
acc = df[df.metric == "test_acc"]
if len(acc) > 1:
    spread = acc.groupby("forget_id")["value"].agg(["mean", "std", "count"])
    print(spread.round(4))
"""),
        code(STATUS),
        code(PUSH),
    ])


def main() -> None:
    notebooks = {
        "00_verify_setup.ipynb": nb_verify(),
        "01_train.ipynb": nb_stage(
            "01", 3, "Train base models and oracles",
            "Stage 3: the 10 original models (5 seeds x clean/canary) and the retrained oracles "
            "-- 5 paired per condition, plus the 12-model ensemble at the primary condition. "
            "62 trainings in total.",
            "\n\nThe oracle ensemble at `mem-high-3000` is what every later threshold is "
            "expressed against, so its spread matters more than its mean.",
        ),
        "02_shadows.ipynb": nb_stage(
            "02", 4, "Train RMIA reference models",
            "Stage 4: 32 shadow models, each on a random half of the training set, so every "
            "example is OUT for roughly 16 of them. They are condition-independent -- the same "
            "32 serve all eight forget conditions -- and nothing blocks on them until the "
            "privacy audit, so they can run early on a spare account.",
        ),
        "03_unlearn.ipynb": nb_stage(
            "03", 5, "Apply the unlearning methods",
            "Stage 5: six methods x eight conditions x five seeds = 240 runs. Requires stage 3 "
            "to have finished -- an unlearning run needs the original model it modifies, and "
            "will fail loudly rather than unlearn from a fresh initialisation.",
            "\n\nWatch `neggrad` specifically: it is the **destructive control** and is expected "
            "to wreck retain accuracy while driving forget accuracy down. That is the point of "
            "including it, not a bug.",
        ),
    }
    for name, nb in notebooks.items():
        path = HERE / name
        path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print(f"wrote {path.relative_to(HERE.parents[1])}  ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
