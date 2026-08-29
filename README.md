# ForgetCheck

**Audit disagreement, reversibility and audit validity in approximate machine unlearning.**

When different audits disagree about whether an unlearning method worked, which audit is right —
and does the answer depend on how hard the deletion request is?

CIFAR-10 / ResNet-18, six unlearning methods, four audit families, grounded against an ensemble
of independently retrained oracles.

---

## Start here

| If you want to… | Read |
|---|---|
| Understand what the project claims and why | [`docs/FORGETCHECK_REVIEW.md`](docs/FORGETCHECK_REVIEW.md) |
| Build something | [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) — §4 is **binding** |
| Know where the build is right now | [`docs/STATUS.md`](docs/STATUS.md) |
| See the evidence behind a decision | [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md) |
| Cite the science | `ForgetCheck_Master_Project_Reference.docx` (v2.0) |

## Local setup

```bash
python -m venv venv
venv/Scripts/activate          # Linux/macOS: source venv/bin/activate
pip install -e ".[dev]"
pytest                          # 265 tests (244 without CIFAR-10 downloaded)
pytest -m slow                  # + the end-to-end training test, minutes on CPU
```

CIFAR-10 downloads on first use. The memorization scores are committed (`data/memorization/`,
400 KB) so no Google Drive detour is needed.

## Running experiments

Training happens on Kaggle GPU, not locally — see [Compute](#compute). Notebooks are in
[`notebooks/kaggle/`](notebooks/kaggle/); run `00_verify_setup.ipynb` first.

```bash
forgetcheck status                                        # what exists, what each stage needs
forgetcheck forget-sets                                   # the eight conditions
forgetcheck --dry-run queue --stage 3 --account 1 --of 3  # list without running
forgetcheck --device cuda queue --stage 3 --account 1 --of 3
```

`queue` is what makes several Kaggle accounts usable at once. It enumerates a stage's work,
takes this account's stripe, and skips anything already in the store. Run identifiers are pure
functions of the config, so every account derives the identical list and no two are given the
same item — with no coordination between them.

| Stage | Work | Items |
|---|---|---:|
| 3 | base models + oracles | 62 |
| 4 | RMIA shadow models | 32 |
| 5 | unlearning runs | 240 |

### Two Kaggle Datasets

Create both once, then attach them to every notebook via **Add Input → Datasets**:

| Dataset | What | Why |
|---|---|---|
| `forgetcheck-cifar10` | the extracted `cifar-10-batches-py` | CIFAR downloads at ~130 kB/s on Kaggle (~20 min), on every session and account. Publish our own rather than a public one: the hash in `configs/base.yaml` is of this exact torchvision-format download |
| `forgetcheck-artifacts` | `artifacts/` + `results/` | Lets a session skip runs another account already finished. Re-version it after each stage |

`00_verify_setup.ipynb` has a cell that stages CIFAR for upload.

## Layout

```
src/forgetcheck/
  registry/     ids · metrics · records · store · provenance     BINDING (plan §4)
  data/         cifar · forget_sets · memorization
  models/       resnet                                           CIFAR ResNet-18 + GAP taps
  evaluation.py                                                  shared by training and audits
  train/        loop · tasks                                     the ONE training loop
  unlearn/      base · methods · runner                          six methods, one interface
  audits/ calibrate/ analysis/                                   next to be written
  config.py · cli.py
configs/        metrics.yaml (the direction commitment) · base.yaml · audits.yaml
notebooks/kaggle/
docs/
```

## Design commitments

These are not incidental; results depend on them.

**One training loop.** M₀, oracles and shadows all go through it unmodified. If the oracle were
trained under a different schedule than its M₀, the gap between them would contain a schedule
difference as well as a forget-set difference, and no audit could separate the two.

**Metric direction is fixed before results exist.** `configs/metrics.yaml` commits in advance to
what "better" means for all 30 metrics, and `validate()` rejects any unregistered name on write.
Deciding direction after seeing results is p-hacking. Note in particular that forget-set accuracy
is `closer_to_oracle`, not `lower_better` — a retrained model still classifies most forgotten
examples correctly, so driving it to zero is over-forgetting, not success.

**The oracle is a distribution, not a model.** Twelve independently retrained oracles at the
primary condition; three held out as validity probes. Thresholds are expressed in oracle standard
deviations. A threshold a genuine retrained model fails is a broken threshold, not a strict one.

**Three independent seed streams** — `train`, `oracle`, `audit` — never shared, because mixing
them silently couples effects the analysis assumes independent.

**Records are append-only Parquet shards, one per run.** Never a shared CSV: concurrent Kaggle
sessions writing one file is guaranteed corruption, discovered late.

## Compute

Training runs on **Kaggle GPU**. Local machines do audits, statistics and the dashboard.

This is not a preference. The team's fastest local machine (Ryzen AI 9 HX 370) has no
PyTorch-capable GPU — ROCm does not support its Radeon 890M iGPU — so local training is CPU-only
and **measured** at 4.76 hours per 30-epoch run, roughly 40–50× slower than a free Kaggle T4.
The full matrix is ~11–26 GPU-hours, comfortably inside one account's weekly quota.

## Status

Stages 0–5 are code-complete and tested; the audit modules (stage 6) are next.
See [`docs/STATUS.md`](docs/STATUS.md) for the current gate-by-gate state.
