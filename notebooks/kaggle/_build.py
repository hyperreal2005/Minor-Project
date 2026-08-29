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
from pathlib import Path

REPO_DIR = Path("/kaggle/working/Minor-Project")

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
"""

STATUS = """\
!forgetcheck --root . status
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
        code("!forgetcheck --root . forget-sets"),
        md("## What the full matrix will cost\n\n`--dry-run` lists the work without doing any "
           "of it."),
        code("""\
!forgetcheck --root . --dry-run queue --stage 3 --account 1 --of 3 | tail -5
!forgetcheck --root . --dry-run queue --stage 4 --account 1 --of 3 | tail -5
!forgetcheck --root . --dry-run queue --stage 5 --account 1 --of 3 | tail -5
"""),
        md("""\
### If everything above passed

This account is ready. Move on to `01_train.ipynb`.

If the CIFAR hash or the memorization hash differed, **stop** — do not train. Compare
`configs/base.yaml` against the repo and re-download rather than proceeding.
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
!forgetcheck --root . --dry-run queue --stage {STAGE} --account {ACCOUNT} --of {OF}
""".replace("{STAGE}", "{STAGE}")),
        md("## Run it\n\nThis is the long cell. It prints each run as it starts and finishes, so "
           "you can watch progress and estimate the remaining time."),
        code("""\
!forgetcheck --root . --device {DEVICE} queue --stage {STAGE} --account {ACCOUNT} --of {OF}
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
