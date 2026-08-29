# ForgetCheck — implementation status

**Updated:** 27 August 2026 (second pass)
**Tracks:** [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) stages and gates

Read this first when resuming work. It records what exists, what its gate says, and what is
genuinely unresolved — as opposed to merely unwritten.

---

## Stage status

| Stage | Owner | State | Gate |
|---|---|---|---|
| 0 — Environment & data | A | **DONE** | ✅ Passes — data hash pinned, loader guard verified |
| 1 — Registry | A | **DONE** | ✅ Passes — see below |
| 2 — Forget sets | A | **DONE** | ✅ **Fully passes** — verified against the real RUM scores |
| 3 — Base models & oracles | A | **CODE DONE** | Shared loop + tasks + CLI; awaits GPU execution |
| 4 — Unlearning methods | A, B | **CODE DONE** | Six methods + SSD behind one interface |
| 5 — Full-pipeline pilot | all | not started | — |
| 6 — Audits | B, C | not started | — |
| 7 — Calibration & validity | D | not started | — |
| 8 — Analysis | D | not started | — |

**Test suite: 265 passing** (plus 1 `slow` end-to-end, run with `-m slow`). Run with `venv/Scripts/python.exe -m pytest tests/`.

---

## What exists

```
src/forgetcheck/
├── registry/            BINDING — plan §4. Built first because it blocks everyone.
│   ├── ids.py           deterministic run identifiers, five-segment grammar
│   ├── metrics.py       the metric registry loaded from configs/metrics.yaml
│   ├── records.py       RunRecord, Arrow schema, validate(), append-only shards
│   ├── store.py         checkpoints, activations, forget-set index cache
│   └── provenance.py    git commit, env hash, content hashes
├── data/
│   ├── cifar.py         in-memory bundle, deterministic loaders, canary corruption
│   ├── forget_sets.py   ForgetSpec, the eight conditions, stratum_summary()
│   └── memorization.py  RUM score loading, proxies, the inversion guard
└── models/resnet.py     CIFAR ResNet-18 + GAP feature taps

configs/
├── metrics.yaml         30 metrics, 6 families — the direction commitment
├── base.yaml            dataset, model, training schedule, seed streams
└── audits.yaml          audit protocols, calibration bands, agreement settings

tests/                   201 tests across 7 modules
```

### Stage 1 gate — passed

The gate reads: *"a dummy run writes a valid record shard; `validate()` rejects an unregistered
metric name and a null required field. All four members have reviewed and signed off on §4."*

Code half is done and tested. **The sign-off half is not** — §4 is binding and four people have
not yet reviewed it. Do that before Stage 3 starts, not after.

### Stage 0 gate — passed

CIFAR-10 downloaded, hashed to `6b3883dca6c867f1` (50,000/10,000, exactly 5,000 per class), and
pinned in `configs/base.yaml`. A mismatched download on another machine now fails loudly.

### Stage 2 gate — passed, on real data

Determinism verified: resolution is a pure function of (spec, labels), with an index tiebreak so
ties cannot reorder across numpy versions. Stratum definitions match RUM's exactly.

**Real RUM scores downloaded 27 Aug 2026** and pinned at sha `2875af6972b5af57`
(`data/memorization/cifar10_memorization.npy`, 400 KB). The Drive archive is
`estimates_results.npz`, ~2 GB, of which 2 GB is an influence matrix we do not use — only the
50,000-element memorization array is kept.

`check_scores()` passes:

| stratum | mean | std | range |
|---|---|---|---|
| low | **0.0000** | 0.0000 | [0.000, 0.000] |
| medium | **0.5004** | 0.0345 | [0.440, 0.560] |
| high | **0.9655** | 0.0270 | [0.915, 1.000] |

Stratum overlap 0. Median 0.1328; 26.6% of scores below 0.01; 24.9% above 0.5.

Two things worth carrying into the paper. **The negative control is exact, not approximate** —
the bottom 3000 all have memorization of precisely 0.0, so theory predicts M0 ≡ M_r there rather
than merely M0 ≈ M_r. And the fat tail above 0.5 (24.9%) is what keeps medium-mem from colliding
with high-mem under RUM's "nearest to 0.5" definition; on a thinner-tailed distribution it would
have, which is why that check exists.

---

## Decisions taken during implementation

Departures from the plan, with reasons. Each is a deliberate change, not drift.

**Checkpoints are fp32, not fp16.** The plan budgeted fp16 to halve storage. Wrong trade:
relearning continues training *from* these checkpoints, and relearning speed is a core
measurement, so fp16 rounding would perturb the very trajectory the reversibility audit
measures. At ~45 MB × ~330 checkpoints that is ~15 GB against a 200 GB quota — the saving buys
nothing and risks a headline result. Activations stay fp16: they are only ever read by
similarity measures, never trained from.

**`shadow_idx` added to the record schema.** The plan listed four seed fields; shadows genuinely
need an index as well, since 32 reference models are drawn with one `audit_seed`.

**Base models carry a dataset variant in the `forget` slot.** `full` is the clean training set,
`canary-500` the corrupted one. The canary condition needs its own M0 — canaries must be present
during training for there to be anything to forget. This means **10 base models, not 5**.

**`env_sha` excludes the accelerator.** Naming the device meant importing torch, costing ~9.4 s
on every record-writing process including pure-analysis ones. Device moved to
`runtime_context()`, which reads torch only if already imported. Also makes `env_sha` identical
whether or not torch happens to be loaded. Cost: 9.4 s → 0.36 s.

**The canary formula needs no collision re-draw.** `y = (y_true + 1 + idx % 9) % 10` has an
offset in [1, 9], never congruent to 0 mod 10, so it cannot coincide with the true label. The
plan hedged with "re-drawn if it collides"; that branch is unreachable and does not exist.

**RUM comparison is advisory, not a gate.** See open question 1.

**Random forget sets use `permutation()[:size]`, not `choice(size, replace=False)`.** Found by
running the real data: numpy's `choice` switches between Floyd's algorithm and a full
permutation depending on the size-to-population ratio, which made the size axis *inconsistently*
nested — `rand-2500 ⊂ rand-3000 ⊂ rand-5000` held but `rand-500 ⊄ rand-2500`. That is a result
that depends on an implementation detail and could change between numpy versions, which is
exactly what Stage 2's gate exists to prevent. `permutation()` has one code path, and the size
axis is now properly nested, so a difference between size conditions is attributable to size
alone rather than to which examples were drawn.

**Each condition kind draws from its own random stream.** Also found on real data: with a shared
seed and equal size, `rand-500` and `canary-500` selected *identical* examples — so the canary
condition would have corrupted exactly what `rand-500` forgets, confounding two conditions meant
to be independent. Streams are now separated per kind via `SeedSequence([selection_seed,
kind_stream])`, keeping nesting within the size axis while making the kinds independent. Overlap
is now 3/500, consistent with chance.

**`penultimate` dropped from the default feature taps.** In ResNet-18 `avgpool` is an
`AdaptiveAvgPool2d((1,1))` sitting directly after `layer4` — the same global average pool the
taps already apply — so `penultimate` is numerically identical to `layer4` (there is a test
asserting it). Storing both added 512 floats per probe for zero information: 1472 → 960 dims,
**35% less activation storage**, roughly 600 MB across the full matrix. It remains requestable
by name for architectures where the two would differ.

---

## Stage 3-5 notes

**The training loop is verified to learn**, not merely to run: 5 epochs on 6,000 CIFAR images
takes test accuracy from 0.161 to 0.500. An earlier smoke run reported a cross-entropy of 47.9,
which looked alarming but was the degenerate 1-epoch-on-2,000-images config, not a defect.

**Work sharding is striped, not hashed.** Hashing run ids was tried first and is stable when new
work is added later, but it distributed badly at small counts — 32 shadow models across 12
accounts left one account with nothing and another with four. Striping a canonically sorted list
is exactly balanced, and the instability is harmless because `_execute` skips anything already in
the store: a redistributed item is skipped, not recomputed. Verified disjoint and complete for
1, 2, 3, 5 and 12 accounts.

**Stage sizes**: 62 (stage 3: 10 base + 40 paired oracles + 12 ensemble), 32 (stage 4 shadows),
240 (stage 5: 6 methods x 8 conditions x 5 seeds).

### Two bugs the tests caught

**SalUn was updating masked-out weights.** Zeroing the gradient outside the saliency mask is not
sufficient: SGD's momentum carries velocity accumulated on earlier steps, and weight decay adds
`wd * p` to the update *inside* the optimiser, after any masking applied to `p.grad`. Both move
weights whose gradient is exactly zero. Left uncorrected, SalUn drifts toward unmasked
random-label fine-tuning — plain label-noise injection — while still appearing masked. Fixed by
snapshotting and restoring the masked-out weights around each step, which is correct regardless
of what the optimiser does internally.

**`ce_loss` and `forget_loss` both claimed the forget set.** The record validator refused a write
and forced the question. The same number was rankable two contradictory ways: `ce_loss` is
`lower_better`, which is wrong on the forget set — a low forget-set loss means the model still
fits the forgotten data. `ce_loss` is now restricted to retain/test; the forget set uses
`forget_loss`, which is `closer_to_oracle`. This is the metric registry doing exactly the job it
was built for.

## Kaggle notebook gotcha, fixed

`pip install -e .` writes a `.pth` file into site-packages, and **`.pth` files are only processed
at interpreter startup**. A Kaggle kernel is already running when the install cell executes, so
`import forgetcheck` failed with `ModuleNotFoundError` even though the install had succeeded.

Confusingly, the `!forgetcheck` CLI calls were unaffected — each spawns a fresh Python that does
read the `.pth` — so the failure looked stranger than it was.

The setup cell now inserts `src/` into `sys.path` directly, calls `importlib.invalidate_caches()`,
and exports `PYTHONPATH` for the `!` subshells. It also resolves the CLI to
`python -m forgetcheck.cli` if the console script is not on PATH.

## Open questions

### 1. RUM strata — RESOLVED (27 Aug 2026)

Settled by reading the paper directly. RUM §3, verbatim:

> "we sort all examples according to their scores. We then use that sorted list to create three
> different forget sets, corresponding to the lowest N scores ('low-mem'), the highest N
> ('high-mem'), and the **N that are nearest to 0.5, i.e. the midpoint of the range of
> memorization scores** ('medium-mem'), where N = 3000."

Two corrections came out of this, both now applied:

**Medium was defined wrongly.** We took the middle of the *rank* ordering; RUM takes the N
nearest to 0.5 on the *score* scale. On a bottom-weighted memorization distribution these are
very different — the rank-middle still sits near zero, so a rank-defined medium collapses onto
low and stops interpolating anything. Now matches RUM.

**The numbers we were checking against were the wrong quantity.** The figures 0.084 ± 0.203,
0.134 ± 0.235, 0.390 ± 0.326 are from RUM's Figure 8, and they describe memorization *within
low/medium/high embedding-space entanglement (ES) partitions* — their other difficulty factor —
not within memorization partitions. ES partitions are not sorted by memorization, which is
exactly why those standard deviations are so large; that anomaly is what prompted the check.
RUM publishes no summary statistics for its memorization strata, so **there is no external
number to validate ours against**. `check_against_published()` has been replaced by
`check_scores()`, which is self-contained.

### 1b. Strata are not guaranteed disjoint — new, guarded

A consequence of RUM's medium definition that they do not discuss. Because medium is selected by
distance to 0.5 on the score scale while high is a rank extreme, a distribution with too thin a
tail above 0.5 makes the medium selection reach up into the top of the ranking and **collide
with high**. Two conditions sharing examples would be confounded rather than independent.

`check_scores()` now computes pairwise stratum overlap and fails if it exceeds 5% of the stratum
size. On a realistic CIFAR-10-shaped distribution the overlap is 0 and medium centres on 0.499,
which is exactly what RUM intends — but this must be re-verified on the real scores, and the
error message names the two ways out (smaller strata, or a percentile-defined medium recorded as
a departure).

### 2. Ordering checks on strata are tautological — resolved, recorded

Worth writing down because it is a natural thing to reach for. Strata are *defined* by rank on
the array passed in, so `low ≤ medium ≤ high` holds for **any** input including a fully inverted
one. Ordering therefore cannot detect an un-inverted proxy. Magnitude can, and does.

### 3. Shadow models for the canary condition

The 32 shadows train on the clean dataset, which is right for the seven ordinary conditions.
Strictly, RMIA on the canary condition wants canary-corrupted shadows. Not blocking: the canary
condition's job is validity scoring via `canary_prob`, which needs no shadows. **State the
limitation** rather than quietly running the mismatched attack.

### 4. Nesting on the size axis is now a design commitment

`rand-500 ⊂ rand-2500 ⊂ rand-3000 ⊂ rand-5000` is deliberate and worth stating in the paper: it
means a difference between size conditions is attributable to size rather than to sample
composition. The cost is that the size conditions are not independent draws, so they cannot be
treated as independent observations in the statistics. The mixed-effects model must account for
this — do not pool size conditions as if they were separate samples.

---

## Next actions

1. **Push the repo**, then set `REPO` and `COMMIT` in the notebooks. They install the package at
   a pinned commit, so provenance depends on it. A pre-push fallback (install from an uploaded
   Kaggle Dataset) is noted in each notebook.
2. **Run `00_verify_setup.ipynb`** on one Kaggle account. It confirms the data and memorization
   hashes match this machine's, which is the precondition for any cross-account comparison.
3. **Stage 3 on Kaggle**: `forgetcheck queue --stage 3 --account K --of N`. 62 trainings.
4. **Stage 4** (shadows, 32) can run in parallel on a spare account — nothing blocks on them
   until the privacy audit.
5. **Stage 5** (240 unlearning runs) after stage 3 completes.
6. Then **Stage 6**: the audit modules, which are the next thing to write.

Environment notes: Python 3.13.5, torch 2.13.0+cpu locally. pandas needed a
`--force-reinstall --no-cache-dir` on this machine — its first install left broken C extensions.
Local training is CPU-only and measured at 4.76 h per 30-epoch run, so **Stage 3 runs on
Kaggle**, not here.
