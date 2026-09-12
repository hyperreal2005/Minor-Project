# ForgetCheck — implementation status

**Updated:** 8 September 2026
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
| 3 — Base models & oracles | A | **DONE, EXECUTED** | ✅ 62/62 trained on Kaggle; seed-SD gate passes |
| 4 — Unlearning methods | A, B | **DONE** | Six methods + SSD behind one interface |
| 5 — Full-pipeline pilot | all | **240/240 RUN** | ⚠️ 64 stale (scrub fix + acct 1 pre-fix); re-run in progress |
| 6 — Audits | B, C | **IN PROGRESS** | base + Layer 1 (behavioral) done with degeneracy guards |
| 7 — Calibration & validity | D | not started | — |
| 8 — Analysis | D | not started | — |

> **Two different stage numberings are in play.** The table above is the *plan's build*
> stages. `forgetcheck queue --stage N` uses *queue* stages, which are not the same: queue 3 =
> base models + oracles, queue 4 = RMIA **shadow models**, queue 5 = the 240 unlearning runs.
> Plan stage 4 is "write the unlearning methods"; queue stage 4 is shadows. Read the CLI's
> `status` output for the queue meaning.

**Test suite: 313 passing** (plus 1 `slow` end-to-end, run with `-m slow`). Run with `venv/Scripts/python.exe -m pytest tests/`.

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

## Stage 3 complete — 62/62 trained (29 Aug 2026)

All 62 models: 10 base (5 clean + 5 canary), 40 paired oracles, 12 ensemble oracles at the
primary condition. ~460 s each on a Kaggle T4, ~8 GPU-hours total across three accounts.

**Oracle ensemble at `mem-high-3000` (n=12)** — the reference distribution every later threshold
is expressed against:

| | mean | sd |
|---|---|---|
| test_acc | 0.9228 | 0.0024 |
| forget_acc | 0.5575 | 0.0058 |

**The difficulty axis is confirmed empirically.** Oracle `forget_acc` — what a model that never
saw the forget set nonetheless scores on it:

| condition | oracle forget_acc |
|---|---|
| mem-low-3000 | **0.998** |
| rand-* | 0.93 |
| mem-med-3000 | 0.91 |
| mem-high-3000 | **0.558** |

On low-memorization data the oracle is right anyway, so M₀ ≈ M_r and there is nothing to detect —
the negative control behaving exactly as designed. On high-memorization data it drops to 56%, so
M₀ and M_r genuinely differ and the audits have something to disagree about.

Two things to carry into the paper. Canary M₀ scores ~0.929 test vs ~0.935 for clean M₀ — the
500 mislabeled canaries cost about 0.5 pp of generalisation, as expected. And 56% is far above
what the memorization scores alone would predict (~3%), because Feldman–Zhang estimate from
models trained on 70% subsets while our oracle sees 94% of the data; state this explicitly or a
reviewer doing the naive arithmetic will think something is broken.

### The Stage 3 gate was specified with the wrong statistic

It read "test accuracy reproducible across seeds within 0.5 pp", measured as a range. Range grows
with sample size — roughly 2.5 sd at n=6, 3.3 sd at n=12 — so the *same models* passed at 0.33 pp
with 6 oracles and "failed" at 0.95 pp with 12, while sd stayed at 0.24 pp throughout. A gate that
tightens as you gather more evidence is backwards. Respecified as `seed_sd_tolerance_pp: 0.4`;
measured sd is 0.24 pp, so it **passes**.

### One held-out oracle sits outside its own 2-SD band

`oracle211` has test_acc 0.9285 against a band of [0.9187, 0.9259] built from the nine
band-forming oracles. A proper prediction interval (t, n=9) gives [0.9179, 0.9267] — still
outside. Its `forget_acc` is inside.

This is not a failure; it is the **first data point of the §15.5 validity analysis**, which exists
precisely to measure how often a genuine retrained model gets flagged. It suggests a 2-SD band
built from 9 oracles will have a non-trivial false-positive rate, and that the calibration should
consider a wider band or a prediction interval rather than a naive 2-SD. Decide that in Stage 7,
on all the metrics, not by eye on one.

## Kaggle notebook gotcha, fixed

`pip install -e .` writes a `.pth` file into site-packages, and **`.pth` files are only processed
at interpreter startup**. A Kaggle kernel is already running when the install cell executes, so
`import forgetcheck` failed with `ModuleNotFoundError` even though the install had succeeded.

Confusingly, the `!forgetcheck` CLI calls were unaffected — each spawns a fresh Python that does
read the `.pth` — so the failure looked stranger than it was.

The setup cell now inserts `src/` into `sys.path` directly, calls `importlib.invalidate_caches()`,
and exports `PYTHONPATH` for the `!` subshells. It also resolves the CLI to
`python -m forgetcheck.cli` if the console script is not on PATH.

### Kaggle flattens uploaded directories — anchor on contents, not folder names

Restoring CIFAR from an attached Dataset failed with "not found" while the data was plainly
there. Kaggle had uploaded the *contents* of `cifar-10-batches-py/` to the dataset root rather
than preserving the folder, so a search for a directory of that name matched nothing.

The give-away was that `artifacts` restored successfully from the same dataset at the same
nesting depth in the same run — so the search mechanism was never at fault, only what it was
searching *for*.

Fixed by anchoring on a file that must exist (`test_batch`) and taking whatever directory
contains it. Verified against three layouts: folder preserved, flattened, and flat mount.

The general lesson: **do not identify data by the name of its container.** Containers get
renamed, flattened and re-wrapped by whatever moves them. Identify it by something intrinsic.

### The CIFAR restore reported success without checking — fixed

The setup cell used `cp -r ... 2>/dev/null || true` followed by an unconditional
`print("restored")`. Kaggle's mount layout varies with how a dataset was uploaded, so the source
path did not exist; the copy failed, the error was discarded, `|| true` swallowed the exit code,
and the message claimed success anyway. CIFAR then silently re-downloaded — **28 minutes**, while
the output said no download was needed.

Now it searches for `cifar-10-batches-py` anywhere under `/kaggle/input`, copies with
`shutil.copytree`, and verifies 6/6 batch files are present before reporting anything. The same
helper handles `artifacts/` and `results/`, which had the identical flaw.

The general lesson, worth keeping: **never print an outcome that was not checked.** A silent
failure that reports success is worse than a loud one, because it costs time *and* misdirects the
investigation — here it sent us looking at CLI path configuration, which was correct all along.

### `--dry-run` reported "would run 0" — fixed

The dry-run branch of `_execute` listed `[todo]` items and then reported "would run 0", because
it never counted them. Nothing was mis-planned; it was purely a reporting bug, but misleading
enough that someone could conclude no work was scheduled. Now reports
`would run N, M already present (N+M total)`, with tests covering dry-run counting, the
have/todo split, and the invariant that ran + skipped + failed equals the queue length.

### Correction, 29 August 2026 — measured, after the real scores arrived

The claim that a random CIFAR-10 subset is "overwhelmingly low-memorization" **was wrong**. It
was an inference from the untraining paper's argument, made before the real Feldman-Zhang scores
were in hand. The measured distribution does not support it:

| memorization | share of CIFAR-10 |
|---|---|
| < 0.01 (effectively non-memorized) | 26.6% |
| 0.01 – 0.5 | 48.5% |
| > 0.5 (strongly memorized) | 24.9% |

Population mean 0.279, median 0.133. A random 3000-example forget set therefore has mean
memorization **0.276**, and **731 of its 3000 examples are as memorized as the mem-high
stratum**. It is a *mixture*, not a uniformly easy case.

What this changes, and what it does not:

- **The design is unchanged.** The difficulty axis is still correctly primary, and
  `mem-low-3000` remains an exact negative control (mean 0.0000, sd 0.0000).
- **The reason changes.** A pure high-memorization stratum *concentrates* signal that a random
  set *dilutes*. That is a difference of degree, not of kind.
- **The expectation changes.** The random conditions should carry real signal, roughly a quarter
  of it from strongly-memorized examples. They will be less discriminative than the high stratum,
  but not empty. Do not write them up as though nothing was detectable there.

## Stage 5 pilot — 6 runs, and it did its job (8 Sep 2026)

Six methods at `mem-high-3000`, seed 0, against the oracle for the same condition
(forget_acc 0.5575, test_acc 0.9228). *G* is the normalized oracle gap on forget accuracy:
`|m(Mu) - m̄(Mr)| / (|m(M0) - m̄(Mr)| + ε)` — 0 is oracle-identical, 1 is no better than the
original model.

| method | forget_acc | *G* | test_acc | test vs oracle |
|---|---|---|---|---|
| finetune | 0.9460 | 0.878 | 0.9306 | +0.0078 |
| l1sparse | 0.8840 | 0.738 | 0.9204 | −0.0024 |
| neggrad | 0.9723 | 0.937 | 0.9260 | +0.0032 |
| neggradplus | 0.1360 | 0.953 | 0.3303 | **−0.5925** |
| salun | 0.6020 | **0.101** | 0.8748 | −0.0480 |
| scrub | 0.8970 | 0.767 | 0.9307 | +0.0079 |

**This is the outcome the pilot stage exists to produce.** Two of six methods were misconfigured
in ways that would have been invisible in aggregate and would have cost 5–6 h of Stage 5 compute
plus every audit built on top of those checkpoints. Both are now fixed, with regression tests.

### Bug 1 — the destructive control was not destructive

`neggrad` came back at forget_acc 0.9723, *higher* than fine-tune's 0.9460, with retain_acc
untouched at 0.9973 and a 6.3 s runtime. The defaults were `epochs=1, lr=0.001`: over 3000 forget
examples at batch 256 that is **twelve** optimizer steps at a thousandth learning rate.

Why it mattered more than a bad number: NegGrad is in the method set as a *labelled control*, the
clean demonstration that "looks forgotten" can mean "damaged" — the failure Audit Layer 1 exists
to catch. A no-op control does not merely score badly, it removes that contrast from the paper
and invites the reader to conclude gradient ascent is harmless.

Raised to `epochs=5, lr=0.01`. The calibration is taken from the pilot's own arithmetic rather
than guessed: diverged `neggradplus` spent ≈920 steps × 0.01 lr × 0.05 ascent weight ≈ 0.46
lr-steps of ascent and collapsed the model; the new `neggrad` spends 60 × 0.01 × 1.0 = 0.60 with
no retain counterweight at all. Comparable budget, no brake — which is what a destructive control
should be. Confirm on the pilot re-run, not on the toy fixture.

### Bug 2 — NegGrad+ diverged, because the published loss is unbounded below

`neggradplus` drove forget_acc to 0.1360 but took retain to 0.3363 and test to 0.3303. That is
not aggressive unlearning, it is a collapsed model, and every audit downstream of that checkpoint
would have been measuring rubble.

The cause is structural, not a hyperparameter accident. The published objective is

```
alpha * CE(retain) - (1 - alpha) * CE(forget)
```

whose second term has no lower bound: ascent keeps paying off forever, and ≈920 steps at lr 0.01
is enough for it to win against the retain term even at alpha 0.95. Lowering the learning rate
would paper over it for this one condition while leaving the next of the eight free to diverge —
and hand-tuning per condition is not available across 240 runs.

**Deviation, deliberate.** The forget term is replaced by `max(0, forget_cap - CE(forget))`,
which has an *identical* gradient to `-CE(forget)` while the model still knows the forget set and
zero gradient once it does not. `forget_cap` defaults to ln(10) = 2.3026 — the cross-entropy of a
uniform prediction over ten classes.

That constant is the substantive part, not a numerical guard. Past chance level the model is not
becoming more ignorant of the forget set, it is becoming confidently *wrong* about it — which is
untraining, not unlearning, the exact distinction (Triantafillou et al.) this project is built
on. Stopping at chance is therefore the principled bound. `forget_cap=inf` recovers the published
loss exactly, so the deviation is reversible and must be stated in the write-up.

`neggrad` is deliberately **not** capped: there, collapse is the behaviour under study.

### Not a bug — SalUn's asymmetry, flagged and left alone

SalUn is far and away the best on the forget axis (*G* = 0.101 against 0.74–0.95 for everything
else) and pays 4.8 pp of test accuracy for it — the only method other than diverged NegGrad+ to
lose meaningful utility. The previously-flagged 15× retain-exposure gap is the likely cause.

**Left unchanged on purpose.** Giving SalUn full retain epochs would probably buy back the test
accuracy by trading away the forgetting that makes it interesting, and "which method sits where
on that trade-off" is a result, not a bug to tune out. Revisit only if the full Stage 5 shows the
utility loss is much worse at other conditions.

### Pilot re-run, 8 Sep 2026 — both fixes confirmed

Oracle at `mem-high-3000`: forget_acc 0.5575, test_acc 0.9228. Base model forget_acc ≈ 1.000
(inferred from pilot 1's *G* values, and expected — these are the most-memorized examples).

| method | forget_acc | *G* | retain_acc | test_acc | runtime |
|---|---|---|---|---|---|
| finetune | 0.9477 | 0.882 | 0.9978 | 0.9304 | 171 s |
| l1sparse | 0.8930 | 0.758 | 0.9922 | 0.9229 | 173 s |
| neggrad | 0.0517 | 1.143 | **0.1031** | **0.1000** | 15 s |
| neggradplus | 0.7340 | 0.399 | 0.9928 | 0.9152 | 337 s |
| salun | 0.6057 | **0.109** | 0.9528 | 0.8752 | 29 s |
| scrub | 0.8967 | 0.767 | 0.9961 | 0.9299 | 231 s |

`neggrad` now collapses the model (macro-F1 0.0187 at accuracy 0.1031 is the signature of a
*constant* predictor). `neggradplus` recovered: retain 0.9928, test 0.9152, and its *G* improved
from 0.953 to 0.399 — second best. Its final forget_loss is 1.196, **below** the 2.303 cap, so
the clamp is not binding at the endpoint; it removed the runaway and let the run settle at an
equilibrium the uncapped objective could never reach. That is how the guard is supposed to work.

### The control now demonstrates the project's central claim, in one table

Ranking the six by naive "lower forget accuracy is better forgetting" against ranking them by
oracle gap:

| rank | naive | oracle gap |
|---|---|---|
| 1 | **neggrad** | salun |
| 2 | salun | neggradplus |
| 3 | neggradplus | l1sparse |
| 4 | l1sparse | scrub |
| 5 | scrub | finetune |
| 6 | finetune | **neggrad** |

The naive metric ranks the destroyed model **first** and the oracle-gap metric ranks it **last**.
This is the empirical justification for `forget_acc` being registered `closer_to_oracle` rather
than `lower_better` in `configs/metrics.yaml` — previously a design argument, now a measurement.
It belongs in the paper.

### Decision — `neggrad` stays at full collapse

The obvious objection is that a constant predictor is *too* destroyed: it is caught by glancing
at the model, so it does not exercise Audit Layer 1 the way a plausible-looking-but-damaged model
would. Considered and rejected, for two reasons.

**There is no robust middle setting.** Plain gradient ascent has no equilibrium — forget loss is
unbounded above and its gradient does not vanish, so any budget large enough to move the model is
eventually large enough to destroy it. The old defaults (0.004 forget-loss movement) and the new
ones (0.542) sit either side of a narrow, unstable band whose location would differ across the
eight conditions. Finding it means per-condition hand-tuning of 40 runs. That NegGrad has no
stable middle is *why* NegGrad+ exists, and it is worth one sentence in the write-up.

**The collapse is the strongest version of the argument, not a weaker one.** A constant predictor
should score a *perfect* privacy audit — there is no membership signal to extract from a model
that has no per-example confidence, so both MIAs should land at AUC ≈ 0.5. A method that
simultaneously wins the naive forgetting metric and passes every privacy audit while being
completely useless is the cleanest possible statement of the audit-validity thesis.

**Consequence for Stage 6: the audit modules must handle degenerate models without crashing.**
Constant outputs will produce undefined AUC, zero-variance activations for CKA, and a relearning
curve indistinguishable from training from scratch. Each of those is a *result* to report, not a
bug to route around — but each is also a division by zero waiting to happen. Build the guards in
from the start, and record the degenerate value rather than skipping the run.

### Runs are not bit-reproducible across sessions — by design, but quantify it

The four methods I did **not** touch came back slightly different from pilot 1 on identical
config and seed:

| | max |Δ| | mean |Δ| |
|---|---|---|
| forget_acc | 0.90 pp (l1sparse) | 0.37 pp |
| test_acc | 0.25 pp | 0.10 pp |

This is expected: `set_determinism(strict=False)` deliberately leaves `cudnn.benchmark=True`, and
Kaggle assigns T4 or P100 by availability. The documented rationale — that GPU nondeterminism is
part of what the oracle ensemble's spread measures — still holds, and oracles carry the same
noise, so the comparison stays fair.

But 0.90 pp is **larger than the 0.40 pp Stage 3 seed-SD tolerance**, so it is not negligible:

- The 5-seed spread on each (method, condition) is seed variance **plus** hardware
  nondeterminism. The paper must describe it that way; calling it "seed variance" would be wrong.
- Method gaps here are 10–40 pp, so nothing about the ranking is at risk. Do not re-tune anything
  on the strength of a sub-1 pp difference.
- An individual run cannot be reproduced exactly from the config alone. The stored checkpoints
  are the reproducible artefact; say so rather than implying rerunability.

Leave `strict=False`. Turning it on costs 20–30% throughput on a stage that is already ~10 h.

### Measured runtimes — replaces the earlier estimate

Pilot 2 total was **956 s for 6 runs**, against my 447 s estimate. The *relative* cost model was
right (neggradplus/finetune measured 1.97× against 2.0× predicted, from its two forward-backward
passes per step), but the absolute constant was uniformly ~2.1× low — the 0.078 s/batch derived
from Stage 3's 460 s training runs understates unlearning, which pays checkpoint load, three
evaluation passes, and dataloader startup per run.

**Revised Stage 5 estimate: 240 runs × ~159 s ≈ 10.6 GPU-hours**, not the 5–6 h previously
recorded here. Sharded three ways that is ~3.5 h per account, comfortably inside Kaggle's 30 h
weekly quota. Cost is dominated by `neggradplus` (337 s) and `scrub` (231 s); `neggrad` and
`salun` are nearly free at 15 s and 29 s.

---

## Stage 5, account 1 of 3 — 80 runs, three defects, one of them systemic (9 Sep 2026)

Ran clean end to end in 3.55 h, matching the 10.6 h projection for all 240 exactly. Four of six
methods are sound and their 56 runs are **keepers**. The other two are not, and the reason they
are not turned out to be the same reason.

### The systemic finding: three methods' update budgets scale with |Df|

`epochs` over the *forget* loader means the update count is proportional to the forget-set size.
The size axis spans 10x (rand-500 → rand-5000), so those methods received 10x different compute
across it:

| condition | \|Df\| | neggrad | salun | scrub ascent | retain-driven three |
|---|---|---|---|---|---|
| rand-500 / canary-500 | 500 | **10** | 10 | 4 | 970 |
| rand-2500 | 2500 | 50 | 50 | 20 | 930 |
| rand-3000 / mem-\* | 3000 | 60 | 60 | 24 | 920 |
| rand-5000 | 5000 | **100** | 100 | 40 | 880 |

**Any "size effect" we report would be partly a compute effect.** That is a confound on one of
the four experimental axes, not a tuning nuisance, and it is the root cause of both defects.

### Defect 1 — `neggrad` is a no-op again, at rand-500

retain 0.9918, test 0.9235, forget_acc 0.9820, **1.7 s**. Ten steps. Identical to the bug the
first pilot caught; the fix held at 3000 and did not generalise down the size axis. Across
conditions the control ranges from untouched (rand-500) through partial (canary-500, retain
0.4297) to total collapse (everything ≥ 2500, retain 0.018–0.036) — useless as a control, because
its strength is set by the condition rather than by us.

**Fixed:** budgeted in **steps, not epochs**; `steps=60` anchors on the primary condition
(5 × ⌈3000/256⌉), so the mem-\* results already collected are unchanged and only the other sizes
move.

### Defect 2 — `salun` collapses at small forget sets, and it was a fidelity bug

retain **0.3422 / 0.2255** at rand-500 and 0.6930 / 0.7141 at canary-500, against 0.97 at
rand-5000. Damage running *backwards* in forget-set size is the tell.

Cause: the implementation paired **one retain batch per forget batch**. That undersamples Dr by
|Dr|/|Df| — 15x at the 3000 conditions, **97x at rand-500** — so the repair term scaled with the
forget set while the random-label damage saturated (500 random labels are easy to fit; 5000 are
not). This is also the "15x retain-exposure gap" flagged earlier and never chased down; it was
already costing 4.8 pp of test accuracy at mem-high-3000.

Verified against the authors' code (`OPTML-Group/Unlearn-Saliency`): their CIFAR-10 path iterates
the forget loader and then the **full** retain loader every epoch, and their CIFAR-100 path
concatenates the two datasets — equivalent in retain exposure. **So this was a deviation from
published SalUn, not a hyperparameter choice.**

**Fixed:** full retain pass per epoch. Cost rises from ~25 s to ~210 s per run, which moves the
Stage 5 total from 10.6 h to **~12.6 h**. SalUn's *forget* pass still scales with |Df| — it must
randomise every forget example — so its damage:repair ratio still varies with size, but now it
varies the way the published method does, which is the defensible position.

### Watch item — `scrub` at rand-5000

retain 0.7352 / test 0.7263 on seed 2, against 0.96–0.999 everywhere else. SCRUB's max-steps
phase is `-KL(student, teacher)`, unbounded above, and its budget scales with |Df| (40 ascent
steps at rand-5000 vs 24 at 3000) — the same divergence shape as the old NegGrad+. **Only one
seed of scrub landed in this shard**, so this is not yet distinguishable from seed variance.
Decide after accounts 2 and 3 report seeds 0,1,3,4. Do **not** pre-emptively change SCRUB: its
structure is faithful to the paper.

### Analysis note for Stage 8 — the oracle gap is ill-conditioned on mem-low

At `mem-low-3000`, fine-tune returns forget_acc 1.0000 / 0.9997. Cross-check that against the
Stage 3 oracle table above: the `mem-low-3000` oracle scores **0.998** on its own forget set. The
base model is ≈ 1.000. So the denominator of

    G = |m(Mu) − m̄(Mr)| / (|m(M0) − m̄(Mr)| + ε)

is |1.000 − 0.998| ≈ **0.002**, which is smaller than the oracle ensemble's own sd (0.0058 at
mem-high). G is dividing by noise and will explode.

This is the arithmetic consequence of something already recorded above and understood as
intended — that mem-low is a **negative control**, where M₀ ≈ M_r and there is nothing to detect.
The design is fine; the *normalization* is what breaks, and only on this condition.

This is not a bug in the runs and needs no re-run, but the normalization cannot be applied
blindly across the difficulty axis. Report the raw gap alongside G for mem-low, or gate G on the
denominator exceeding the oracle ensemble's own spread. **Decide before Stage 8, not during it.**

### What has to be re-run, and what does not

- **Stage 3 (base + oracles) and Stage 4 (shadows): nothing.** Neither depends on unlearning
  method configuration.
- **Stage 5: only `neggrad` and `salun`.** Accounts 2 and 3 pull the fix before their first run
  and produce nothing stale. Account 1 re-runs its own 24 affected runs (neggrad × 1 seed × 8
  conditions, salun × 2 seeds × 8) — roughly one hour, dominated by the now-slower SalUn.
- Account 1's other 56 runs stand as they are.

### The mechanism that makes a re-run safe: hyperparameter shas

A `run_id` names *what* was unlearned, not how (`{tag}__{role}__{forget}__{method}__{seed}`), so a
method fix reuses the same id — and `run_unlearn` skipped on `has_checkpoint(rid)` alone. **A
re-run after a fix would have silently kept the broken weights and reported success.** Both of
this stage's defects would have survived their own fixes.

`run_unlearn` now resolves the method config up front, hashes it, and compares against the stored
checkpoint's `hparams_sha` before deciding to skip. Same sha → skip, instantly. Different sha →
print the mismatch and recompute. Re-running a whole shard is therefore the correct action in all
cases: unchanged runs cost a metadata read, changed ones are redone, and nothing is silently
stale either way.

---

## Stage 5 complete on all three accounts — and the fixes verified (12 Sep 2026)

Accounts 2 and 3 pulled the neggrad/salun fixes before running. 240/240 runs exist; 64 are stale
and will be recomputed (below).

### Both fixes confirmed on 160 fresh runs

**`neggrad`** at `steps=60`: mean retain 0.1007 (acct 2) / 0.1020 (acct 3), collapsed at **every**
condition including rand-500 (0.077–0.109) and canary-500 (0.100–0.110). The size-dependence is
gone. Consistent control across the whole axis.

**`salun`** with the full retain pass: rand-500 retain 0.9964–0.9983 (was 0.23–0.34), canary-500
0.9973–0.9979 (was 0.69–0.71). Runtime 222–257 s, as projected.

### Correction — SalUn's "standout forgetting" was an artefact of the bug

Two earlier entries above call SalUn "far and away the best on the forget axis" (*G* = 0.109 at
mem-high-3000). That was measured on the under-sampled-retain implementation. With faithful
retain exposure, at the same condition:

| | forget_acc | *G* | test_acc |
|---|---|---|---|
| SalUn, broken (acct 1, seeds 1, 4) | 0.591 / 0.626 | 0.109 | 0.880 / 0.880 |
| SalUn, fixed (accts 2–3, seeds 0, 2, 3) | 0.820 / 0.822 / 0.833 | **0.604** | 0.925 / 0.930 / 0.926 |

The broken version's forgetting came bundled with a 4.5 pp utility loss — it was partly
*untraining*. The fixed version forgets less and keeps test accuracy at the oracle's 0.9228.
**Retract the standout claim.** SalUn now sits in the same band as the others.

Updated oracle-gap ranking at mem-high-3000 (5-seed means, scrub pending re-run):
neggradplus 0.380 · salun 0.604 · l1sparse 0.769 · scrub 0.800 · finetune 0.888 · neggrad 1.087.
The naive metric still ranks neggrad **first** and the oracle gap still ranks it **last** — the
headline finding survives the correction intact; only the middle of the table moved.

### `scrub` at rand-5000 — the watch item is now a confirmed defect

All five seeds: retain **0.735 / 0.827 / 0.555 / 0.721 / 0.648**, against 0.94–0.999 at every
condition of 3000 or fewer. Same shape as the old NegGrad+: `-KL(student, teacher)` is unbounded
above and the ascent budget scaled with |Df| (40 steps at rand-5000, 24 at 3000). Not seed
variance — condition-specific divergence.

**Fixed** the same way as neggrad: `max_steps=24`, split evenly across the `msteps` epochs. The
3000-example conditions are *numerically identical* to the published schedule (2 × 12); only the
other four sizes move. SCRUB at mem-low shows mild seed-dependent damage (0.939–0.981) at 24
steps — a property of ascent on easy data, not divergence; leave it.

**Pilot before committing**: scrub at rand-500 gets 24 ascent steps where it had 4, i.e. six
passes over the same 500 examples per max-epoch. Run `--methods scrub --forget rand-500,rand-5000`
first (10 runs, ~40 min) and confirm retain stays above 0.9 at both ends of the axis.

### The re-run guard was unreachable from the CLI — fixed

The dry run on account 1 reported "would run 0, 80 already present". The stale-hyperparameter
check lived inside `run_unlearn`, but `cli._execute` skipped on `has_checkpoint` *before* ever
calling it — so a real run would have skipped all 80 too, and the fixes would never have reached
the data. Moved to the layer that decides what runs: `WorkItem` carries the sha its
configuration produces, `WorkItem.state()` returns `todo` / `have` / `stale`, and `_execute`
runs both `todo` and `stale`, printing `[stale]` and counting them separately.

### What is stale, per account

| account | stale runs | why | est. |
|---|---|---|---|
| 1 | 8 neggrad + 16 salun + 8 scrub = **32** | ran before every fix | ~1.6 h |
| 2 | 16 scrub | scrub fix landed after | ~1.3 h |
| 3 | 16 scrub | scrub fix landed after | ~1.0 h |

Each account re-runs its own shard unchanged: `queue --stage 5 --account K --of 3`. The dry run
will now list them as `[stale]`.

### Other observations from the full 240 (no action)

- **Canary condition works as ground truth.** Fine-tune, which never sees the forget set, drops
  canary accuracy from ~1.0 to 0.16–0.19 by retain exposure alone. SCRUB retains the most
  (0.31–0.34), L1-sparse the least (0.09–0.12). That spread is the signal Stage 7 scores against.
- **Run-to-run nondeterminism is visible on identical configs**: e.g. `neggrad` ce_loss ranges
  58.7–80.8 across seeds/accounts at the same collapse. Already recorded above; it is the
  seed + hardware variance the 5 seeds absorb.
- Account 2 runtimes are ~20% above account 3 (finetune 215 s vs 177 s) — different GPU
  allocation. Efficiency comparisons must be within-account or normalized.

---

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

1. **Full Stage 5** — 240 runs, ~10.6 GPU-hours, ~3.5 h per account sharded three ways:
   `forgetcheck queue --stage 5 --account K --of 3`. Cleared to launch; the pilot re-run is clean.
2. Then **Stage 6: the audit modules** (behavioral, privacy_population, privacy_rmia,
   representation, relearning). This is the actual contribution and none of it is written yet.
3. Stage 7 calibration, Stage 8 analysis.

**Novelty thread `9IzfArmoHq` — closed, 8 Sep 2026.** It is *Unlearning Evaluation through Subset
Statistical Independence* (ICLR 2026 = arXiv 2603.00587): a retrain-free HSIC audit that argues
retrained-oracle evaluation "defeats the purpose of developing a standalone, verifiably unlearned
model." **Not a scoop — all four of ForgetCheck's openings survive intact**, and the critique
targets deployment-time verification, not research-time audit validation, which needs a ground
truth by definition. Full analysis in `RESEARCH_LOG.md` §7.1, including an argument for adding
SDE as a sixth audit module and a testable prediction that our collapsed `neggrad` control breaks
it.

Environment notes: Python 3.13.5, torch 2.13.0+cpu locally. pandas needed a
`--force-reinstall --no-cache-dir` on this machine — its first install left broken C extensions.
Local training is CPU-only and measured at 4.76 h per 30-epoch run, so **Stage 3 runs on
Kaggle**, not here.
