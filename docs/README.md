# ForgetCheck — project documentation

Reference material for the ForgetCheck project. Read these before writing implementation code;
they carry decisions the code is expected to follow.

## Contents

| File | What it is | Read it when |
|---|---|---|
| [`STATUS.md`](STATUS.md) | **Where the build is right now.** Stage-by-stage state against the plan's gates, decisions taken during implementation and why, open questions that are genuinely unresolved, and next actions. | Resuming work after any break; before starting a stage |
| [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | **The build spec.** Binding data contracts (run IDs, result schema, metric registry), the eight forget conditions, nine stages each with an acceptance gate, ownership across four members, Kaggle execution workflow, storage budget, failure playbook. | Writing any code; before every stage; when a run needs an ID or a metric needs a name |
| [`FORGETCHECK_REVIEW.md`](FORGETCHECK_REVIEW.md) | Independent red-team review of the frozen v1.0 position against the 2026 literature. Novelty assessment, four blocking design flaws, three major ones, revised experimental design, compute plan, team split, revised schedule. | Starting any work stream; deciding what to build; defending a design choice |
| [`RESEARCH_LOG.md`](RESEARCH_LOG.md) | The evidence behind the review. Source-by-source findings, reasoning chains for every recommendation, the computations performed, the hardware investigation, what was checked and found sound, and what remains unverified. | A recommendation looks wrong; a reviewer challenges a claim; re-running the novelty check before submission |
| `../ForgetCheck_Master_Project_Reference.docx` | The master reference, **now at v2.0**. Canonical project description for the synopsis, reports, paper and presentation. | Always — it is the single source of truth |
| [`ForgetCheck_Master_Project_Reference_v1.0_ORIGINAL.docx`](ForgetCheck_Master_Project_Reference_v1.0_ORIGINAL.docx) | Untouched v1.0, kept for provenance so any change can be diffed against the frozen position. | Showing the mentor exactly what moved between versions |

Web versions (same content, easier to share):
- Build spec — https://claude.ai/code/artifact/60c4b354-899a-4b00-b6b9-1ed6db81e34d
- Review — https://claude.ai/code/artifact/6ec6a133-7537-4015-85de-1cae219f52c3

## The short version

The project is sound and worth building. Three things changed after review:

1. **The novelty claim was narrowed.** Cross-audit disagreement has been published since the v1.0
   freeze, including on CIFAR-10 with ResNet-18. What remains open is reversibility as a ranked
   audit family, disagreement *within* the privacy family, forget-set difficulty as a factor, and
   **audit validity** — scoring audits for correctness rather than only comparing them.
2. **Four design decisions were fixed** before they could produce uninterpretable results: SSD
   removed (no-op on random instance forget sets), the primary statistic replaced (Kendall's τ over
   four methods cannot reach p < 0.05 under any data), the forget-set axis split in two, and the
   oracle promoted from a single model to an ensemble.
3. **The compute plan was inverted.** Training runs on Kaggle, not locally — the Ryzen AI 9 HX 370
   cannot train PyTorch models on its Radeon 890M iGPU, and CPU training measures at 4.76 hours per
   30-epoch run against roughly 12 minutes on a free cloud GPU. Local machines do audits and
   analysis, where 32 GB of RAM is a real advantage.

## Scale, so nobody wonders

The matrix grew at every revision. Nothing was ever cut for compute.

| | Methods | Conditions | Seeds | Total runs |
|---|---:|---:|---:|---:|
| v1.0 as frozen | 4 | 4 | 3 | 147 |
| v2.0 after review | 6 | 7 | 3 | 353 |
| **v2.1 adopted** | **6** | **8** | **5** | **~700** |

~700 runs is roughly 40–70 GPU-hours — one week of one member's Kaggle quota. The headroom goes
into *depth* (5 seeds, 12 oracles, 32 privacy reference models), not breadth, because the
contribution is a calibration claim and calibration precision scales with reference models rather
than with method count.

## Before the synopsis is approved

- Resolve the one unchecked novelty thread: ICLR 2026 paper, OpenReview `9IzfArmoHq`. It sat behind
  a bot check during the review. Ten minutes in a browser.
- Freeze the result-record and checkpoint schema (week 3). Under an ephemeral-session training
  workflow this is load-bearing infrastructure, and it is the single thing most likely to cost
  weeks 13–14 if deferred.
- Protect week 6 — the full-pipeline pilot, run through to a disagreement figure.

## Conventions

- The master `.docx` is the canonical description. If these markdown files and the `.docx` ever
  disagree, the `.docx` wins and the markdown gets corrected.
- Reference numbers `[1]`–`[18]` are v1.0's bibliography; `[19]`–`[34]` were added in v2.0.
- Terminology follows the *Untraining* / *Unlearning* distinction of reference `[19]`. ForgetCheck
  is an **Untraining** study throughout.
