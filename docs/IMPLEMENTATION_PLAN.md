# ForgetCheck — Implementation Plan

**Version:** 1.0 · 26 August 2026
**Governs:** all implementation work from week 3 onward
**Derived from:** [`FORGETCHECK_REVIEW.md`](FORGETCHECK_REVIEW.md) · master reference v2.0
**Team:** 4 · **Training compute:** Kaggle GPU (multiple accounts) · **Analysis:** local

---

## 0. How to use this document

This is the operative reference for building ForgetCheck. It fixes the things that are expensive
to change later — identifiers, schemas, directory layout, execution order, acceptance criteria —
and deliberately leaves the things that are cheap to change (hyperparameters, plot styling) to be
decided during the pilot.

**Rules of engagement:**

- **§4 (Core contracts) is binding.** Nothing in it may be changed by one person unilaterally. It
  is the interface between four people working in parallel, and the single most likely cause of a
  failed week 14.
- Everything else is a strong default. If you deviate, record it in the run's `notes` field and
  raise it, so the deviation is in the data rather than in someone's memory.
- If this document and the master `.docx` disagree, the `.docx` wins on *scientific* claims and
  this document wins on *engineering* detail.
- Acceptance criteria in §6 are gates, not aspirations. A stage is not done until its gate passes.

---

## 1. Scope: where the compute headroom goes

Compute is not scarce. Four members × 2–3 accounts × 30 GPU-h/week is on the order of
**240–360 GPU-hours per week**, against a full matrix that costs tens of hours. The binding
constraint is now **person-weeks and analysis surface**, not GPU time.

That changes what "spending the headroom" should mean. The contribution is **audit validity** —
a claim about statistical calibration. Calibration precision scales with the number of *reference
models*, not with the number of methods. So the headroom goes into **depth, not breadth**:

| Spend | v2.0 | **v2.1 (adopted)** | Why |
|---|---:|---:|---|
| Seeds | 3 | **5** | Every claim tightens; mixed-effects models estimate the seed random effect properly rather than from three points |
| Oracles at primary condition | 5 | **12** (3 held out as validity probes) | Directly determines the precision of every oracle-SD threshold and the entire §15.5 validity analysis |
| RMIA reference models | 8 | **32** | Attack B is a headline result; per-example attack power depends on how many references have each example OUT |
| SSD class-unlearning side condition | cited | **run** | Converts a citation into an experiment; demonstrates the mechanism/task mismatch rather than asserting it |
| Ranked methods | 6 | **6** | Unchanged — more methods add analysis surface without strengthening the validity claim |
| Datasets | CIFAR-10 (+CIFAR-100 optional) | **unchanged** | A second dataset doubles everything to answer a question we are not asking |

**Total: ~700 runs, ~40–70 GPU-hours.** Comfortably one week of one member's quota.

> **Nothing was ever cut for compute.** v1.0 was 147 runs; v2.0 was 353; v2.1 is ~700. The only
> removal was the LLM/TOFU extension, cut on scientific grounds — v1.0 already marked it "optional
> only after the core paper is complete," and it dilutes a paper whose contribution is audit
> validity in vision.

---

## 2. Repository layout

```
forgetcheck/
├── configs/
│   ├── base.yaml                  # dataset, arch, training schedule, seeds
│   ├── forget_sets.yaml           # the 8 forget conditions, declaratively
│   ├── methods/                   # one YAML per unlearning method
│   │   ├── finetune.yaml  neggrad.yaml  neggrad_plus.yaml
│   │   └── scrub.yaml  salun.yaml  l1sparse.yaml
│   ├── audits.yaml                # probe sets, layers, relearn protocol
│   └── metrics.yaml               # THE METRIC REGISTRY — see §4.4
├── src/forgetcheck/
│   ├── registry/                  # ← build first, everything depends on it
│   │   ├── ids.py                 # deterministic run_id / forget_id
│   │   ├── records.py             # RunRecord schema + writer
│   │   ├── store.py               # checkpoint & activation artefact store
│   │   └── provenance.py          # git commit, env hash, checkpoint sha
│   ├── data/
│   │   ├── cifar.py               # loaders, fixed splits, augmentation
│   │   ├── memorization.py        # RUM score loading + proxy fallback
│   │   └── forget_sets.py         # ForgetSpec -> index array
│   ├── models/resnet.py           # CIFAR ResNet-18 (3x3 stem, no maxpool)
│   ├── train/
│   │   ├── loop.py                # single shared training loop
│   │   ├── train_base.py          # M0
│   │   ├── retrain_oracle.py      # Mr (paired + ensemble)
│   │   └── shadows.py             # RMIA reference models
│   ├── unlearn/
│   │   ├── base.py                # Unlearner ABC — see §4.5
│   │   ├── finetune.py  neggrad.py  neggrad_plus.py
│   │   ├── scrub.py  salun.py  l1sparse.py
│   │   └── ssd.py                 # side condition only
│   ├── audits/
│   │   ├── behavioral.py          # Layers 1, 1B, 1C
│   │   ├── privacy_population.py  # Attack A
│   │   ├── privacy_rmia.py        # Attack B
│   │   ├── representation.py      # CKA + second measure
│   │   └── relearning.py          # Layer 4
│   ├── calibrate/
│   │   ├── oracle_bands.py        # null distributions, oracle-SD units
│   │   └── validity.py            # oracle FPR + canary scoring
│   ├── analysis/
│   │   ├── agreement.py           # instance-level tau/rho, disagreement rate
│   │   ├── mixed_effects.py       # confirmatory tests
│   │   └── figures.py
│   └── cli.py                     # single entry point — see §7.1
├── notebooks/kaggle/              # thin launchers, no logic
├── results/
│   ├── records/                   # run_records.parquet (append-only)
│   ├── curves/                    # relearning trajectories
│   └── figures/
├── docs/                          # this file, the review, the research log
├── tests/
├── pyproject.toml
└── README.md
```

**Rule: notebooks contain no logic.** A Kaggle notebook installs the package from the repo, calls
one CLI command, and pushes artefacts. Anything else is unreviewable and unreproducible.

---

## 3. Experimental design, concretely

### 3.1 The eight forget conditions

Two orthogonal axes plus a ground-truth condition. **Sizes are absolute counts, not fractions**, so
that difficulty strata are compared at matched size.

| id | Axis | Definition | n | Role |
|---|---|---|---:|---|
| `rand-500` | Size | Uniform random | 500 | Scaling check (1%) |
| `rand-2500` | Size | Uniform random | 2500 | Scaling check (5%) |
| `rand-5000` | Size | Uniform random | 5000 | Scaling check (10%) |
| `rand-3000` | Both | Uniform random | 3000 | **Bridge** — matches difficulty-axis size |
| `mem-low-3000` | Difficulty | Lowest memorization scores | 3000 | **Negative control** |
| `mem-med-3000` | Difficulty | Middle stratum | 3000 | Interpolation |
| `mem-high-3000` | Difficulty | Highest memorization scores | 3000 | **Primary condition** |
| `canary-500` | Ground truth | Injected mislabeled canaries | 500 | **Validity scoring** |

`mem-high-3000` is the **primary condition**: the oracle ensemble, the validity analysis and the
headline results are all anchored there. n=3000 matches published practice in the RUM and
vision-transformer benchmarks, keeping our strata comparable to theirs.

### 3.2 The canary condition — precise specification

This is the only condition where the correct audit verdict is known in advance, so it must be
built exactly.

1. Select 500 training indices with `selection_seed`. Call these the canaries.
2. Assign each canary a **deterministically wrong** label:
   `y_canary = (y_true + 1 + (idx % 9)) % 10`, re-drawn if it collides with `y_true`.
3. **M0 trains on the corrupted dataset** (all 50,000, canaries carrying wrong labels). A canary
   label cannot be predicted by generalisation — only by memorising that example.
4. **Oracles train on the 49,500 non-canary examples.** The canaries are *removed*, not corrected
   — the forget set is exactly the canaries, so the retain set is everything else.
5. **The ground-truth signal** is the canary-label probability:
   `s(x) = P_model(y_canary | x)`.
   - `s_M0(x)` is high — M0 memorised it.
   - `s_Mr(x)` is at the oracle-ensemble baseline — Mr provably never saw the association.
   - Any `s_Mu(x)` above the oracle band is **unambiguously residual influence**.

Every audit then produces a verdict on each `canary-500` model, and that verdict is scored against
this known answer as accuracy / sensitivity / specificity. This is what makes §15.5 possible.

> **Caveat to state in the paper:** canaries are maximally-memorised by construction, so they are
> an upper bound on detectability, not a typical deletion request. An audit that fails *here* is
> unusable; an audit that succeeds here is not thereby validated for ordinary forget sets. Report
> canary results alongside the memorization strata, never instead of them.

### 3.3 Seeds — three independent streams

Never share a seed across streams. Mixing them silently couples effects the analysis assumes are
independent.

| Stream | Values | Controls |
|---|---|---|
| `train_seed` | `{0,1,2,3,4}` | Model init, data order, augmentation. M0 and its **paired** oracle share this seed. |
| `selection_seed` | `{100}` fixed | Forget-set membership. Fixed so all methods/seeds attack an identical forget set. |
| `oracle_seed` | `{200..211}` | The 12-model oracle **ensemble** at the primary condition. Independent of `train_seed`. |
| `audit_seed` | `{300}` fixed | Probe subsampling, bootstrap resampling, shadow subset draws. |

**Two distinct roles for oracles**, easily confused:

- **Paired oracle** `Mr(condition, train_seed=s)` — matched to `M0(s)`, used for the primary
  comparison in every condition. 5 per condition.
- **Ensemble oracles** `Mr(mem-high-3000, oracle_seed=o)` for `o ∈ {200..211}` — used to estimate
  the null band and the oracle-vs-oracle baselines. **Seeds 209–211 are held out** and never
  contribute to a band; they are the validity probes fed through every audit in §6.8.

### 3.4 Relearning normalisation

The review called for anchor arms. The clean form reuses the Normalized Oracle Gap of master
reference §15.4 rather than inventing a second scale:

```
normalized_recovery(Mu) = (AUC(Mu) − AUC(Mr)) / (AUC(M0) − AUC(Mr))
```

- **0** → recovers exactly like a genuine retrain. Good.
- **1** → recovers like the model that never forgot. All structure retained.
- **>1** → recovers *faster* than the original — evidence the unlearning left a primed state.

M0 is the upper anchor (maximal retained structure, T80 = 0) and the paired oracle is the lower
anchor and the comparison target simultaneously. A **random-init arm is run once per condition as
a protocol sanity check** — confirming the reintroduction data alone cannot trivially produce
recovery — rather than in every cell. This removes ~120 runs at no scientific cost.

### 3.5 RMIA reference models

- **32 shadow models**, each trained on a random 50% subset of the training set drawn with
  `audit_seed`. Each example is therefore OUT for ~16 of them.
- Shadows are **condition-independent** — they model the data distribution, not any forget set —
  so the same 32 serve every condition. 32 trainings, not 32 × 8.
- Attack A (population) needs no shadows at all. That asymmetry in cost is part of the finding:
  the attack everyone reports is free, and it is the one that misleads.

---

## 4. Core contracts — binding

> Nothing in this section changes without all four members agreeing. Build and freeze it in
> **week 3**, before any audit module is written.

### 4.1 Identifiers

```python
# src/forgetcheck/registry/ids.py

def forget_id(spec: ForgetSpec) -> str:
    """'mem-high-3000' — human-readable, stable, collision-checked."""

def run_id(*, role: str, forget: str, method: str | None,
           seed: int, seed_kind: str = "train") -> str:
    """
    'c10r18__{role}__{forget}__{method}__{seed_kind}{seed}'

    role   : base | oracle | shadow | unlearn | relearn
    Examples:
      c10r18__base__full__none__train0
      c10r18__oracle__mem-high-3000__none__oracle205
      c10r18__unlearn__mem-high-3000__scrub__train2
    """
```

Run IDs are **deterministic and content-free** — derivable from the config alone, before the run
exists. That is what lets four people generate work queues independently without collisions.

### 4.2 ForgetSpec

```python
@dataclass(frozen=True)
class ForgetSpec:
    kind: Literal["random", "memstratum", "canary"]
    size: int                          # absolute count
    stratum: Literal["low","medium","high"] | None = None
    selection_seed: int = 100
    proxy: str | None = None           # "rum" | "confidence" | "holdout"

    def indices(self, dataset) -> np.ndarray:   # deterministic, sorted
    def sha(self) -> str:                        # hash of the resolved index array
```

The **resolved index array is cached and hashed**. The spec regenerates it; the hash proves every
run used the same one. Any mismatch is a hard failure, not a warning.

### 4.3 RunRecord — the result schema

**Long format, one row per (run, audit, metric, probe_set).** Long rather than wide because four
people write four audit modules producing heterogeneous metrics; long lets each append rows
without coordinating columns, and D pivots at analysis time.

| Field | Type | Notes |
|---|---|---|
| `run_id` | str | §4.1 |
| `role` | str | base / oracle / shadow / unlearn / relearn |
| `dataset`, `arch` | str | `cifar10`, `resnet18` |
| `forget_id` | str | §4.1 |
| `forget_kind`, `forget_size`, `forget_stratum` | str/int/str | denormalised for easy grouping |
| `method` | str | `none` for base/oracle/shadow |
| `hparams_sha` | str | hash of the resolved method config |
| `train_seed`, `selection_seed`, `oracle_seed`, `audit_seed` | int | nullable per role |
| `audit` | str | `behavior` / `privacy_pop` / `privacy_rmia` / `representation` / `relearning` / `meta` |
| `metric` | str | must exist in `configs/metrics.yaml` — **validated on write** |
| `probe_set` | str | `forget` / `retain` / `test` / `canary` / `layer3` … |
| `value` | float | |
| `n_probe` | int | how many examples the value was computed over |
| `runtime_s` | float | |
| `checkpoint_sha`, `git_commit`, `env_sha` | str | provenance |
| `timestamp` | str | ISO-8601 UTC |
| `notes` | str | **any deviation from the plan goes here** |

Written as **append-only Parquet shards**, one per run — never a shared mutable CSV. Concurrent
Kaggle sessions writing one file is a guaranteed corruption. `analysis/` concatenates shards.

Curves (relearning trajectories, layer-wise CKA vectors) go to `results/curves/{run_id}.parquet`
keyed by `run_id`, not into `value`.

```python
# records.py must expose exactly this and nothing looser
def write_records(rows: list[RunRecord], out_dir: Path) -> Path: ...
def validate(row: RunRecord) -> None:
    """Raises if `metric` is not in the registry, or a required field is null for the role."""
```

### 4.4 Metric registry — `configs/metrics.yaml`

The review requires metric direction to be fixed **before** analysis. This file is that
commitment, and `validate()` enforces it.

```yaml
retain_acc:      {family: behavior,        direction: higher_better,    oracle_ref: false}
test_acc:        {family: behavior,        direction: higher_better,    oracle_ref: false}
forget_acc:      {family: behavior,        direction: closer_to_oracle, oracle_ref: true}
js_to_oracle:    {family: behavior,        direction: lower_better,     oracle_ref: true}
pred_agreement:  {family: behavior,        direction: higher_better,    oracle_ref: true}
mia_auc_pop:     {family: privacy_weak,    direction: closer_to_oracle, oracle_ref: true}
mia_tpr_at_fpr:  {family: privacy_weak,    direction: closer_to_oracle, oracle_ref: true}
mia_auc_rmia:    {family: privacy_strong,  direction: closer_to_oracle, oracle_ref: true}
cka_linear:      {family: representation,  direction: closer_to_oracle, oracle_ref: true}
cka_rbf:         {family: representation,  direction: closer_to_oracle, oracle_ref: true}
procrustes:      {family: representation,  direction: closer_to_oracle, oracle_ref: true}
relearn_auc:     {family: reversibility,   direction: closer_to_oracle, oracle_ref: true}
relearn_t80:     {family: reversibility,   direction: closer_to_oracle, oracle_ref: true}
canary_prob:     {family: meta,            direction: lower_better,     oracle_ref: true}
```

**Every ranked comparison uses one common goodness scale**, computed centrally in `agreement.py`,
never per-audit:

```
G = |m(Mu) − m̄(Mr_ensemble)| / (|m(M0) − m̄(Mr_ensemble)| + ε)     # normalized oracle gap
score = −G                                                          # higher is always better
```

Low-discriminability flag fires when `|m(M0) − m̄(Mr)|` is within the oracle ensemble's own
noise band; that cell is marked, not scored (master reference §15.4).

### 4.5 Unlearner interface

```python
class Unlearner(ABC):
    name: str
    @abstractmethod
    def unlearn(self, model: nn.Module, *, forget_loader, retain_loader,
                cfg: dict, device) -> nn.Module: ...
```

Every method is a drop-in. The training loop, checkpointing, timing and record-writing live
outside and are shared — so wall-clock efficiency numbers are comparable across methods by
construction rather than by hope.

---

## 5. Storage budget

ResNet-18 (CIFAR, 10 classes) ≈ 11.17 M parameters.

| Artefact | Count | Each | Total |
|---|---:|---:|---:|
| Checkpoints, fp16 `state_dict` | ~330 | 22 MB | **7.3 GB** |
| Pooled activations (fp16) | ~300 | 6 MB | **1.8 GB** |
| Records + curves | — | — | < 0.5 GB |
| **Total** | | | **~10 GB** |

Comfortable against a 200 GB private quota. Two constraints shape the layout:

- **`/kaggle/working` is ~20 GB and persists per notebook**; `/kaggle/tmp` gives ~60 GB of
  non-persistent scratch. Stage outputs must be pushed to a Dataset before the session ends.
- **Kaggle datasets allow ~50 top-level files** — so use directory structure, one subdirectory per
  stage, and never dump 300 checkpoints at the root.

**Activations: pool before storing.** Raw `layer1` output for 3000 probes is 64×32×32×3000×4 B ≈
786 MB per model — untenable. Global-average-pool over spatial dims first: 3000 × (64+128+256+512)
= 3000 × 960, ≈ 6 MB in fp16. This is standard practice for CKA on convnets and must be applied
identically to every model. **State the pooling in the paper** — GAP-CKA discards spatial
structure, which is an accepted convention but a real limitation.

---

## 6. Stages, with acceptance gates

A stage is done when its gate passes. Not before.

### Stage 0 — Environment & data · *owner A · week 3*
- Package installs clean from a fresh Kaggle session; `pyproject.toml` pins versions.
- CIFAR-10 loads with **fixed, hashed** train/test splits.
- RUM CIFAR-10 memorization scores downloaded and hashed. If unavailable, the confidence proxy
  is implemented and the substitution recorded in `notes`.

> **Gate:** `pytest tests/` green on Kaggle and locally; data hash identical on both.

### Stage 1 — Registry · *owner A · week 3* — **blocks everyone**
- `ids.py`, `records.py`, `store.py`, `provenance.py` complete.
- `configs/metrics.yaml` populated and frozen.

> **Gate:** a dummy run writes a valid record shard; `validate()` rejects an unregistered metric
> name and a null required field. **All four members have reviewed and signed off on §4.**

### Stage 2 — Forget sets · *owner A · week 4*
- All 8 conditions materialise to index arrays; canary label corruption implemented per §3.2.

> **Gate:** every `ForgetSpec` regenerates a byte-identical index array across two machines;
> strata means match RUM's published values (low ≈ 0.08, medium ≈ 0.13, high ≈ 0.39).

### Stage 3 — Base models & oracles · *owner A · weeks 4–5*
- Shared training loop; M0 × 5 seeds; paired oracles for all 8 conditions × 5 seeds; ensemble
  oracles × 12 at `mem-high-3000`.

> **Gate:** test accuracy reproducible across seeds within **0.5 pp**, and the seed-to-seed spread
> is *reported*, not averaged away — it is the raw material for every oracle band. Pin the actual
> accuracy target during the pilot rather than assuming a number now.

### Stage 4 — Unlearning methods · *owner A + B · week 5*
- Six methods behind the `Unlearner` interface. SSD implemented but wired only to the class-
  unlearning side condition.

> **Gate:** each method reproduces its source repo's reported behaviour on one reference setting,
> to within a documented tolerance. **Deviations recorded in `notes`, per master reference §18.**

### Stage 5 — Full-pipeline pilot · *all four · week 6* — **protect this week**
One seed, `mem-high-3000` only, all six methods, **all four audits**, calibration, and the
disagreement analysis run through to an actual figure.

> **Gate:** a disagreement figure exists, generated end-to-end from logged records with no manual
> data handling. Every design fault this surfaces is one that would otherwise have surfaced in
> week 13.

### Stage 6 — Audits · *B and C · weeks 8–12*
| Module | Owner | Gate |
|---|---|---|
| `behavioral.py` | B | Oracle-vs-oracle JS divergence is near zero; M0-vs-oracle is measurably larger |
| `privacy_population.py` | B | Reproduces NeurIPS-starter-kit-style AUC on M0 |
| `privacy_rmia.py` | B | TPR at low FPR is stable when reference count is halved (16 vs 32) |
| `representation.py` | C | Linear CKA and the second measure agree in rank on the pilot; oracle-vs-oracle baseline computed |
| `relearning.py` | C | M0 arm gives `normalized_recovery ≈ 1`, oracle arm `≈ 0`, random-init arm well below 0 |

Those relearning anchor values are the module's own self-test: if M0 does not normalise to ≈1 and
the oracle to ≈0, the protocol is wrong, not the finding.

### Stage 7 — Calibration & validity · *owner D · week 7, revisited week 14*
- Oracle null bands per metric from the 9 band-forming ensemble oracles.
- **The three held-out oracles (seeds 209–211) pass through every audit as candidates.**
- Canary scoring per audit family.

> **Gate:** every audit reports an oracle false-positive rate and a canary accuracy. An audit that
> flags a genuine retrain is reported as such — that is a **finding, not a bug to hide**.

### Stage 8 — Analysis · *owner D · week 14*
- Instance-level Kendall τ / Spearman with bootstrap CIs over ~240 unlearned models.
- Mixed-effects: `metric ~ method + forget_condition + (1 | train_seed)`.
- Disagreement matrix; per-setting rank tables marked **descriptive**.

> **Gate:** every number in the paper traces to a record shard. No hand-edited values anywhere.

---

## 7. Execution

### 7.1 One CLI, called from thin notebooks

```bash
forgetcheck train-base      --seed 0
forgetcheck train-oracle    --forget mem-high-3000 --oracle-seed 205
forgetcheck train-shadows   --start 0 --count 8
forgetcheck unlearn         --forget mem-high-3000 --method scrub --seed 2
forgetcheck relearn         --run-id <id>
forgetcheck audit           --audit representation --forget mem-high-3000
forgetcheck calibrate       --forget mem-high-3000
forgetcheck analyse         --out results/figures
forgetcheck queue           --stage 3 --account 2 --of 3   # shard work across accounts
```

`queue` is what makes multiple accounts usable without collisions: it deterministically partitions
the run list, so account 2 of 3 computes its slice with no coordination and no duplicated work.

### 7.2 Dependency order

```
Stage 0 data ─┬─> Stage 2 forget sets ─┬─> Stage 3 M0 + oracles ─┬─> Stage 5 unlearning
Stage 1 registry ┘                     │                          ├─> Stage 6 relearning
                                       └─> Stage 4 shadows ───────┴─> Stage 7 audits
                                                                       └─> Stage 8 calibrate
                                                                            └─> Stage 9 analyse
```

Shadows are on an independent branch — start them early, in parallel, on a spare account. They are
32 trainings that block nothing until the privacy audit.

### 7.3 Kaggle session discipline

1. Notebook installs the package from GitHub at a **pinned commit** (recorded in every record).
2. Attaches the artefact Dataset(s) read-only.
3. Runs one CLI command; writes to `/kaggle/working/{stage}/`.
4. Pushes a new Dataset version before the session ends.

**Every run is independently resumable.** A run whose artefacts already exist in the store is
skipped, so a session that dies mid-queue costs only its in-flight run. Never write a notebook that
must complete in full to be useful.

---

## 8. Ownership and milestones

| Week | A — substrate | B — behaviour & privacy | C — representation & reversibility | D — calibration & delivery |
|---|---|---|---|---|
| 3 | Stages 0–1 · **freeze §4** | review §4 | review §4 | review §4 |
| 4 | Stage 2, start Stage 3 | shadow-model queue | probe-set + pooling spec | metrics registry + analysis skeleton |
| 5 | Finish Stage 3, Stage 4 | Stage 4 support | relearn protocol draft | oracle-band maths |
| **6** | **Full-pipeline pilot — all four, together** ||||
| 7 | ensemble oracles | Attack A | activation caching | **Stage 7 calibration** |
| 8–10 | matrix support | Attacks A & B, gap analysis | CKA + second measure | figure pipeline |
| 11–12 | — | — | relearning + anchors | validity tables |
| 13 | **full multi-seed matrix** ||||
| 14 | — | — | — | Stage 8 analysis |
| 15–16 | dashboard | paper | paper | paper assembly |

**A is alone on the critical path until week 7.** If the week-6 pilot slips at all, move B onto the
substrate immediately — B and C have nothing to audit until checkpoints exist, and that idleness is
invisible until it is fatal.

---

## 9. Failure playbook

| Symptom | Likely cause | Action |
|---|---|---|
| An audit gives near-identical values for M0 and the oracle | Low discriminability, expected on `rand-*` and `mem-low-3000` | Fire the §15.4 flag. **This is a result** — it is what the negative control is for |
| SSD does nothing in a random condition | Expected — finding B1 | Do not tune. Record it, cite [24], move on |
| An audit flags a held-out oracle as "not forgotten" | Audit false positive | **Report it.** Highest-value finding available; do not adjust the threshold to hide it |
| RMIA TPR unstable at low FPR | Too few references with the example OUT | Raise shadows 32 → 48. Compute is available |
| Two representation measures disagree | Possibly the Davari failure mode [26] | Report the disagreement; do not pick the flattering one |
| Records won't join at analysis time | Schema drift | Should be impossible — `validate()` is the guard. If it happens, §4 was bypassed |
| Kaggle session dies mid-queue | Normal | Re-run the same command; completed runs are skipped |

---

## 10. Open decisions — resolve during the pilot

Deliberately unresolved. Each needs one measurement, not one argument.

1. **Epoch budget.** Pin the schedule that reaches a stated accuracy, publish it, stop there.
   Applied identically to M0, oracles and shadows, a shorter schedule is scientifically fine and
   halves everything.
2. **Second representation measure** — RBF-CKA, orthogonal Procrustes, or distance correlation.
   Pick on pilot behaviour: whichever is most stable across oracle pairs.
3. **Relearning reintroduction set** — the whole forget set, or a fixed-size subset. A subset is
   fairer across conditions of different size; confirm on the pilot.
4. **Probe-set size for CKA** — 3000 assumed in §5. Check CKA stability against 1000 and 5000.
5. **Low-FPR operating point** for TPR reporting — 0.1% or 1%, chosen once and applied everywhere.

---

## 11. Before week 3

- [ ] `git init`, push to a private GitHub repo all four can access
- [ ] Resolve the outstanding novelty thread: OpenReview `9IzfArmoHq` (research log §7)
- [ ] Confirm RUM memorization scores download successfully
- [ ] All four members read and sign off on **§4**
- [ ] Create the shared Kaggle artefact Dataset and confirm write access from a notebook

---

*Companion documents: [`FORGETCHECK_REVIEW.md`](FORGETCHECK_REVIEW.md) for why these decisions were
made; [`RESEARCH_LOG.md`](RESEARCH_LOG.md) for the evidence behind them; the master `.docx` (v2.0)
for the scientific claims this implementation must support.*
