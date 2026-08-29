# ForgetCheck — Research Log & Reasoning Trail

**Companion to:** [`FORGETCHECK_REVIEW.md`](FORGETCHECK_REVIEW.md)
**Review conducted:** 25 August 2026
**Subject:** ForgetCheck Master Project Reference v1.0 (research position frozen 7 August 2026)

This document records *how* the review was produced: what was searched, what was read, what
was verified, what could not be verified, the reasoning behind every recommendation, and the
things that were checked and turned out to be fine. The review document states conclusions;
this one states the evidence and the argument, so that any conclusion can be re-examined or
overturned later without redoing the search.

**Read this when:** a recommendation in the review looks wrong, a reviewer challenges a
claim, the team wants to know why a decision was made, or new literature appears and the
positioning needs re-checking.

---

## Table of contents

1. [Method and coverage](#1-method-and-coverage)
2. [Source-by-source findings](#2-source-by-source-findings)
3. [Reasoning chains for each finding](#3-reasoning-chains-for-each-finding)
4. [Computations performed](#4-computations-performed)
5. [Hardware investigation](#5-hardware-investigation)
6. [Checked and found sound](#6-checked-and-found-sound--no-action-needed)
7. [Open uncertainties](#7-open-uncertainties-and-things-i-could-not-verify)
8. [Changes applied to the master document](#8-changes-applied-to-the-master-document-v10--v20)
9. [How to re-run this review](#9-how-to-re-run-this-review)

---

## 1. Method and coverage

### 1.1 What was read in full

- The complete master reference `.docx` (51,715 characters, 750 paragraphs), extracted by
  unzipping the OOXML package and parsing `word/document.xml` directly. All 25 sections plus
  both appendices.
- Full PDF text of five papers, extracted locally with `pypdf` where the fetch tool returned
  raw binary:
  - arXiv 2605.02206 — *Metric Unreliability in Multimodal Machine Unlearning* (16 pages)
  - arXiv 2604.07962 — *Is your algorithm unlearning or untraining?* (15 pages)
  - arXiv 2605.27569 — *RULER: Representation-Level Verification of Machine Unlearning*
  - arXiv 2407.17710 — *Revisiting Machine Unlearning with Dimensional Alignment*
  - arXiv 2404.11577 — *A Reliable Cryptographic Framework for Empirical MU Evaluation*
- Abstracts, key sections, or structured summaries of ~20 further papers (listed in §2).

### 1.2 What could not be accessed

| Resource | Why | Mitigation |
|---|---|---|
| The ChatGPT share link in the brief | Renders client-side; the fetch returns only the page title | Review is based on the `.docx` alone. **If that conversation contains constraints not in the document, some recommendations may change — worth pasting the key parts.** |
| OpenReview PDFs (2 attempts) | Bot-verification interstitial | Routed around via arXiv mirrors and search summaries |
| CVPR 2026 workshop survey PDF | HTTP 403 from `openaccess.thecvf.com` | Non-critical; used only as background context |
| ICLR 2026 paper `9IzfArmoHq` | OpenReview interstitial | **Unresolved — see §7.** Titled something like "Unlearning Evaluation"; could be a further novelty threat |

### 1.3 Search strategy

Four passes, deliberately ordered:

1. **Citation verification** — check every suspicious reference in the document, especially the
   four dated 2026, before trusting anything built on them.
2. **Direct novelty threat** — search for the exact claim ("do unlearning audits agree?") rather
   than the topic, because a topic search returns the field and a claim search returns competitors.
3. **Method soundness** — for each of the four audit layers, search for published critiques of the
   specific instrument chosen (MIA protocol, CKA, relearning protocol, oracle design).
4. **Feasibility** — code availability, dataset artefacts, compute cost, hardware.

Pass 2 is what surfaced the scoop. Pass 3 is what surfaced B1 and M1. Neither would have come
out of a general "machine unlearning 2026" search, which returns surveys.

---

## 2. Source-by-source findings

### 2.1 The direct novelty threat

#### arXiv 2605.02206 — *Metric Unreliability in Multimodal Machine Unlearning: A Systematic Analysis and Principled Unified Score*
Khan, Laga & Sohel (Murdoch University). Preprint; the code repo is under a `neurips26` org,
so presumably a NeurIPS 2026 submission.

Read in full. Verbatim from the abstract:

> "Five standard metrics, Forget Accuracy (FA), Retain Accuracy (RA), Membership Inference
> Attack (MIA), Activation Distance (AD), and JS divergence (JS), yield conflicting method
> rankings across three VQA benchmarks... Kendall's τ ... over 36 unlearned LLaVA-1.5-7B models
> reveals two opposing clusters, {FA, RA, MIA} and {AD, JS}, with τ_FA,AD = −0.26"

And from §5.2:

> "mean pairwise Kendall's τ is lower in multimodal VQA (τ̄ = 0.086) than in unimodal
> CIFAR-10 (τ̄ = 0.158, ∆ = 0.072)"

**This is ForgetCheck's Section 6 contribution, already executed.** Same question, same
statistic, same oracle grounding, and it includes a CIFAR-10 / ResNet-18 arm.

Critically, from their §4 (experimental setup):

> "CIFAR-10 [30]: ResNet-18, **forget class 0**; unimodal baseline."

and their methods are GA, RL (random labels), FT, SalUn.

**What they leave open — this is where ForgetCheck's surviving contribution lives:**

| Gap | Evidence from their own text |
|---|---|
| Reversibility / relearning | "All five metrics share a key limitation: none assess whether removed knowledge is erased or merely suppressed. We define this as knowledge recoverability" — and "KR does not scale directly", so it was excluded from the ranking analysis; only a pilot (96% leakage) was run |
| Layer-wise representation analysis | They use a single scalar Activation Distance, not layer-wise CKA |
| Forget-set difficulty as a factor | Their CIFAR arm is class-0 forgetting; no memorization stratification |
| A true from-scratch oracle | Their §4 says the reference "M* is **fine-tuned on retain only**" — which is a weaker oracle than a from-scratch retrain, and arguably a methodological weakness ForgetCheck can beat them on |
| Instance-level deletion | Class unlearning throughout the vision arm |

**Judgement:** this kills the novelty claim as written but does *not* kill the project. Four
distinct openings remain and three of them were already in ForgetCheck's plan.

#### arXiv 2605.27569 — *RULER: Representation-Level Verification of Machine Unlearning*
Cosma & Finke (Loughborough / Newcastle). Accepted at WIPE-OUT 2026, ECML PKDD; to appear in
Springer CCIS. Read in full.

From the abstract:

> "Four approximate unlearning methods all pass output-level evaluation, yet under a linear
> mixed-effects model M2 detects significant residuals in 10 of 12 conditions (p<0.05)"

**This is ForgetCheck's H1, already established.** Two consequences:

1. H1 must be demoted from hypothesis to replication target. Presenting it as a novel
   prediction in 2026 would be a visible error.
2. Their statistical machinery — **linear mixed-effects models** over conditions — is a much
   better fit for ForgetCheck's design than Kendall's τ over four methods, and is directly
   borrowable. This is where the B2 fix came from.

Note: grep found only **one** mention of CKA in the whole paper, so RULER does *not* occupy the
layer-wise CKA ground. That space is still open.

#### arXiv 2503.06991 — *Are We Truly Forgetting? A Critical Re-examination of MU Evaluation Protocols*
Also establishes H1, from a different direction: existing methods either (a) completely degrade
representation quality, or (b) modify only the classifier while staying representationally close
to the original. Large-scale, representation-based. Third independent confirmation that H1 is
settled science.

### 2.2 The reframing paper — most important single read

#### arXiv 2604.07962 — *Is your algorithm unlearning or untraining?*
Triantafillou, Humayun, Ribero, Turner, Mozer (Google) & Kaissis (HPI), April 2026. Read in full.
Authors include the people who ran the NeurIPS 2023 unlearning competition.

Draws a distinction the field had been conflating:

- **Untraining** — remove the influence *of these specific forget-set examples*. Retrain-from-
  scratch is the correct oracle. MIA is explicitly named as "an established metric for Untraining".
- **Unlearning** — remove the *underlying distribution/concept* the forget set was sampled from.
  Retraining is **not** the right reference here, and MIA is explicitly called inappropriate.

**ForgetCheck is unambiguously an Untraining study.** Saying so costs one paragraph and
pre-empts the obvious challenge ("why is retraining your reference?").

The paper also supplies the mechanism behind B3, quoted from their §6:

> "the fact that the examples in S are not memorized means that the retrained-from-scratch model
> A(D\S) also predicts the examples of S correctly. This means that the classic definition of
> 'unlearning' ... wants the 'unlearned' model to still predict the examples of S correctly,
> **effectively necessitating no change over the original model**."

This is the formal statement of why random low-memorization forget sets are degenerate. The
document's §15.4 safeguard anticipated the symptom; this explains the cause and shows it affects
most of the planned matrix, not the occasional cell.

### 2.3 Audit-instrument critiques

#### arXiv 2403.01218 — Hayes et al., *Inexact Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy*
Splits unlearning MIAs into:
- **population U-MIA** — one attacker for all examples (what ForgetCheck §14.4 specifies, and
  what the NeurIPS starter kit ships)
- **per-example U-MIA** (U-LiRA) — a dedicated attacker per example; substantially stronger

Findings: population attacks "overestimate the privacy protection afforded by existing unlearning
techniques", and privacy risk actually *increases* post-unlearning for a significant number of
examples under some algorithms.

**Why this matters more than the document's existing caveat:** §14.4 already warns that a failed
MIA doesn't prove erasure. True, but insufficient. The real problem is that a weak attack produces
a *misleading ranking* — and ranking is ForgetCheck's whole output. A wrong rank propagates into
every τ, every disagreement count, every conclusion.

#### arXiv 2312.03262 — Zarifzadeh, Liu & Shokri, *Low-Cost High-Power Membership Inference Attacks* (RMIA, ICML 2024)
The solution to the compute problem M1 creates. RMIA works "under computational constraints where
only a limited number of pre-trained reference models (as few as 1) are available... unlike prior
attacks that approach random guessing." Full U-LiRA needs hundreds of shadow models; RMIA gets
strong per-example attack power from a handful. This is what makes a defensible privacy audit
affordable on a student budget.

#### arXiv 2210.16156 — Davari et al., *Reliability of CKA as a Similarity Measure in Deep Learning* (ICLR 2023)
CKA values "can be easily manipulated without substantial changes to the functional behaviour of
models"; formally characterises sensitivity to outliers and to transformations that preserve
linear separability.

**Why this is dangerous specifically for ForgetCheck:** Audit Layer 3's entire purpose is to make
a claim about internal state that output-level metrics miss. A metric that moves independently of
function is exactly the wrong sole instrument for that job. A reviewer who knows this paper will
ask, and the answer needs to already be in the manuscript.

#### arXiv 2602.16400 — *Easy Data Unlearning Bench* (EasyDUB)
Ships **200 pretrained models, 10 forget sets, and 200 oracle models per forget set** for CIFAR-10
(ResNet-9), on HuggingFace. Primary metric KLoM (KL of margins) compares against the oracle
*ensemble*. Reports N=100 models suffice for reliable evaluation.

Two uses: (a) it's the evidence for B4 — the field now treats the oracle as a distribution, not a
point; (b) the released checkpoints are a free sample of oracle-to-oracle variation, usable to
sanity-check ForgetCheck's own null bands before spending compute.

Also critiques U-LiRA as gameable by degenerate constant-margin strategies — hence the utility-guard
caveat attached to M1.

### 2.4 The SSD problem

#### arXiv 2412.19583 — *A Comparative Study of MU Techniques for Image and Text Classification Models*
The decisive evidence for B1:

> "Selective Synaptic Dampening fails to forget across all percentage values in random forgetting
> scenarios, likely due to similarities in Fisher Information Matrix values."

versus, for class unlearning: SSD "consistently demonstrates performance comparable to or exceeding
state-of-the-art retraining-based methods across... single-class and sub-class unlearning."

**Corroborating evidence:**
- The SSD paper itself (arXiv 2308.07707, AAAI 2024) evaluates class and sub-class forgetting; the
  official repo (`if-loops/selective-synaptic-dampening`) has class-unlearning-oriented scripts.
- arXiv 2606.16110 (Ye et al. auditing survey) independently finds **Fisher/Hessian-based methods
  failed** their audit despite formal certification, while retraining- and fine-tuning-based
  methods passed.

**Mechanistic reasoning (why this is predictable, not just empirical):** SSD selects parameters
whose Fisher importance is disproportionately high *for the forget set relative to the retain set*.
A random i.i.d. subset of CIFAR-10 has, by construction, the same distribution as the retain set.
The importance ratio is therefore ≈1 everywhere, no parameters exceed the selection threshold, and
the dampening step is a no-op. This is not a tuning problem — it is a mechanism/task mismatch, and
no hyperparameter sweep fixes it.

**Consequence for the paper:** SSD would appear as "perfect utility, zero forgetting" in every
random condition, occupying a rank slot while carrying no information. Worse, a careless reading
could present that as a finding about SSD's quality rather than about task fit.

### 2.5 Forget-set difficulty

#### arXiv 2406.09073 / Wichert & Sikdar (EMNLP 2024 Findings, ref [7] — verified real)
Verified at ACL Anthology `2024.findings-emnlp.271`. Key finding beyond what the document records:

> "forget accuracies on influential data are significantly lower compared to random sampling, but
> **the gap becomes smaller for larger forget sets**"

**Implication the document misses:** the random-vs-hard contrast is *most* visible at **small**
forget sets and washes out as size grows. This inverts the intuition behind a 10% condition, and
is a second, independent argument for separating the size axis from the difficulty axis (B3).

#### RUM (ref [8], NeurIPS 2024) and arXiv 2410.16516 (*Scalability of memorization-based MU*)
- CIFAR-10 memorization scores **are** released via the RUM repo. The document's assumption holds.
- Worth knowing: Feldman & Zhang's own release covers **CIFAR-100 and ImageNet, not CIFAR-10** —
  so the RUM repo is the only source, with no canonical fallback.
- Fallback if it breaks: proxies. Reported Spearman correlations against true memorization —
  confidence −0.80 to −0.91, binary accuracy −0.71 to −0.89, loss curvature 0.69–0.70, holdout
  retraining 0.62–0.67 — at roughly 0.001–0.002% of the cost of exact scores.
- RUM's reported CIFAR-10 memorization strata means: low 0.084±0.203, medium 0.134±0.235,
  high 0.390±0.326.

#### arXiv 2602.20114 — *Benchmarking Unlearning for Vision Transformers*
- Uses RUM-style stratification: 3,000 examples split into low/medium/high memorization thirds.
  A sensible fixed-size design to copy.
- Documents ranking instability directly: "Method rankings from CNNs do not transfer to VTs; they
  depend strongly on architectures and proxies."
- Notes SalUn scores well on ToW but poorly on ToW-MIA — another instance of metric disagreement
  appearing as a side finding in 2026 work.
- **NegGrad+ is the strong baseline**, reported as the only method improving ToW-MIA under both
  memorization proxies on CIFAR-10/ResNet-18. Source of finding m4.

### 2.6 Relearning / reversibility — the surviving contribution

- **arXiv 2505.16831** (ref [13], ICML 2026) — verified real, poster at icml.cc/virtual/2026/poster/65395.
  LLM setting. Representation-level framework (PCA similarity/shift, CKA, Fisher information) plus
  reversibility. Four forgetting regimes by reversibility and catastrophicity.
- **Re-learn time already exists as a vision metric** — published results show NegGrad+ recovering
  far faster than retraining, implying implicit retention. So ForgetCheck should present relearning
  as an existing instrument *elevated to a ranked audit family*, not as a new metric. Overclaiming
  here would be an easy, avoidable error.
- **arXiv 2506.01318** (*Prototypical Relearning Attack*) — vision, CIFAR-10/100, ResNet-18, but
  **class-level** unlearning. Recovers forgotten classes from a handful of samples via feature
  prototypes. Adjacent, not overlapping: ForgetCheck's setting is instance-level.
- **arXiv 2607.19442** (*Unlearning as Distribution Restoration*) — LLM setting, but supplies the
  strongest argument for the calibration recommendation. Their proposed absolute thresholds fail
  so badly that **the retraining reference itself certifies in only 1 of 45 cells**. That is the
  clearest published demonstration that thresholds must be calibrated against the oracle
  distribution rather than set absolutely — and that an audit which fails a genuine retrain is
  broken, not strict.

### 2.7 Adjacent work worth knowing

| Paper | Relevance |
|---|---|
| arXiv 2510.26714 — *On the importance of multiple training seeds* | Exists solely to argue single-seed unlearning evaluation is unsound. Direct support for the document's existing ≥3-seed rule |
| arXiv 2404.11577 — *A Reliable Cryptographic Framework* (Tu, Hu & Ma) | Models MIA-based evaluation as a cryptographic game between unlearning algorithm and adversary; provable guarantees existing metrics lack |
| arXiv 2606.16110 — Ye et al., *Auditing Machine Unlearning* | Preprint, no venue. "Proof of ignorance" framing; avoids retraining baselines and shadow models. Ten methods, six datasets |
| arXiv 2508.12730 — *Unlearning Comparator* | A visual analytics system for comparative unlearning evaluation — i.e. the dashboard deliverable, already published. Cite it |
| arXiv 2601.19755 — Ribero, Schrab & Gretton, *Regularized f-Divergence Kernel Tests* | Refs [14]/[15] verified. Relative distance test: is the unlearned model distributionally closer to the retrained or the original model? Validated on synthetic and high-energy-physics data, **not LLMs** — a caveat worth stating if cited |

---

## 3. Reasoning chains for each finding

Each finding below is stated as: observation → why it matters → what breaks → chosen fix →
alternatives rejected.

### B1 — SSD is a no-op on random instance forget sets

- **Observation:** SSD selects parameters by forget/retain Fisher-importance ratio.
- **Why it matters:** a random i.i.d. forget set is distributionally identical to the retain set,
  so the ratio is ≈1 everywhere and nothing is selected.
- **What breaks:** SSD ≈ M₀ in every random condition. A constant row in the ranking analysis,
  removing 25% of the signal — and a false "perfect utility, zero forgetting" result.
- **Evidence:** arXiv 2412.19583 (direct); SSD's own evaluation scope (class/sub-class);
  arXiv 2606.16110 (Fisher/Hessian methods fail auditing).
- **Fix chosen:** replace with SalUn + L1-sparse — both instance-level standards, both already in
  the RUM codebase, and adding two methods simultaneously fixes B2's power problem.
- **Rejected:** (a) *tune SSD harder* — mechanism mismatch, not a hyperparameter issue;
  (b) *keep SSD as a documented negative case* — defensible but wastes a method slot the ranking
  analysis needs; demoted to appendix instead;
  (c) *add a class-unlearning condition so SSD works* — scope creep into a different problem
  (Unlearning, not Untraining, per arXiv 2604.07962).

### B2 — Kendall's τ over 4 methods is statistically inert

- **Observation:** §16.1 names τ the primary agreement statistic; §12 fixes 4 methods.
- **What breaks:** computed exhaustively (see §4.1) — with n=4 there are 24 permutations, τ takes
  7 distinct values, and the minimum attainable two-sided p is **0.0833**. Perfect agreement
  cannot reach α=0.05. The primary statistic can never produce a significant result.
- **Fix chosen (three parts):**
  1. **Change the unit of analysis** — correlate metric *values* across all (method × condition ×
     seed) model instances, n ≈ 75+, not ranks over 4 methods. Precedent: arXiv 2605.02206 did
     exactly this with 36 models and reported p = 0.003.
  2. **Raise method count to ≥6** (B1's fix already does this) so per-setting rank tables at least
     *can* reach significance: min p at n=6 is 0.0028.
  3. **Linear mixed-effects models** for confirmatory tests, following RULER: metric ~ method +
     condition + (1|seed). Handles the repeated-measures structure that τ ignores entirely.
- **Rejected:** (a) *Spearman instead* — same n, same problem;
  (b) *permutation test on τ* — the exhaustive computation in §4.1 **is** the permutation test;
  the limit is the discreteness of the space, not the approximation;
  (c) *report τ without p-values* — acceptable as description, but then the project has no
  inferential backbone at all.

### B3 — Random forget sets are near-degenerate

- **Observation:** §15.4 anticipates low discriminability but treats it as an occasional flag.
- **Why it's worse than assumed:** arXiv 2604.07962 shows that for low-memorization examples the
  ideal untraining solution is *no change from the original model*. A random CIFAR-10 subset is
  overwhelmingly low-memorization. So M₀ ≈ M_r across most cells, and every normalized oracle-gap
  denominator collapses — not occasionally, but structurally.
- **Second argument:** Wichert & Sikdar — the influential-vs-random gap *shrinks* with forget-set
  size, so a 10% condition is the *least* discriminative, not the most.
- **Fix chosen:** split into two orthogonal factors — size (random 1/5/10%, demoted to a scaling
  check) and difficulty (low/med/high memorization strata at fixed size, promoted to primary).
- **The move I'm most confident about:** make the **low-memorization stratum a designed negative
  control**. Theory predicts no signal there. If the audits correctly report "nothing to detect"
  at low memorization and a clear signal at high, that *validates the audit battery itself* —
  turning the weakest condition into evidence of instrument quality rather than an excuse.
- **Rejected:** *drop random conditions entirely* — RQ on deletion-size scalability is legitimate
  and cheap to keep; just don't build conclusions on it.

### B4 — One oracle per cell is not a reference distribution

- **Observation:** §10 treats M_r as *the* counterfactual; §16.1 provisions one oracle per (setting, seed).
- **What breaks:** every "distance from oracle" number silently contains training randomness with
  no estimate of its magnitude. Thresholds against a single draw are arbitrary. §16.2 says the
  right thing ("treat stochastic retraining variation as part of the reference distribution") but
  the matrix doesn't provision for it.
- **Evidence the field has moved:** EasyDUB uses 200 oracles per forget set; the Google/AISTATS
  work is built entirely on relative distribution tests; arXiv 2510.26714 exists to argue this point.
- **Fix chosen:** ≥5 independent oracles at the primary setting; thresholds in oracle-SD units;
  ≥1 oracle held out as an audit probe.
- **The high-value add-on — "oracle-passes-its-own-audit":** feed a held-out genuine retrain into
  every audit as if it were a candidate. Any audit that flags it as "not forgotten" has a
  false-positive problem, reportable as a number. arXiv 2607.19442 found precisely this failure
  (reference certified in 1/45 cells), so the check catches real breakage.
- **Why this matters strategically:** it converts the paper's question from *"do audits disagree?"*
  (descriptive, and now scooped) to *"which audits are trustworthy?"* (normative, still open).

### M1 — The specified MIA is the weak one

Covered in §2.3. The strategic point worth restating: this is the finding that most improves the
paper. Running both a population attack and RMIA and reporting the gap produces **intra-family
disagreement** — the same model private under one attack, leaky under another. No cross-audit
study currently does this, and it directly extends 2605.02206, which used a single MIA.

### M2 — CKA cannot be the sole representation instrument

Covered in §2.3. Fix: linear CKA + one mechanistically different measure (RBF-CKA, orthogonal
Procrustes, or distance correlation), requiring agreement before any representation claim, plus
the oracle-vs-oracle CKA baseline for scale — which B4 supplies for free.

Note: §14.6's argument for keeping linear probes optional (a network legitimately retains the
"dog" concept after one dog image is deleted) is **correct and well-reasoned**. It should stay.

### M3 — Relearning protocol needs anchors

- **Gap 1:** models entering relearning at different utility levels have different headroom, so a
  raw recovery curve conflates "retained structure" with "started closer to the target".
- **Gap 2:** with only M_u and M_r there is no scale — fast relative to what?
- **Fix:** add M₀ (upper anchor — never forgot) and a random-init or from-scratch arm (lower
  anchor — no prior exposure); normalise recovery between them; match starting utility explicitly
  or carry it as a covariate in the B2 mixed-effects model.

---

## 4. Computations performed

### 4.1 Kendall's τ null distribution for small n

The exhaustive permutation computation behind B2. Reproducible:

```python
from itertools import permutations
from scipy.stats import kendalltau
import numpy as np

for n in range(3, 9):
    base = list(range(n))
    taus = np.array([round(kendalltau(base, p).correlation, 6)
                     for p in permutations(base)])
    p_perfect = float((np.abs(taus) >= 1.0 - 1e-9).mean())
    print(n, len(taus), p_perfect, len(set(taus)))
```

| n (methods ranked) | Permutations | Min. attainable 2-sided p | Distinct τ values |
|---:|---:|---:|---:|
| 3 | 6 | 0.3333 | 4 |
| **4** (as planned) | **24** | **0.0833** | **7** |
| 5 | 120 | 0.0167 | 11 |
| **6** (recommended floor) | **720** | **0.0028** | **16** |
| 7 | 5,040 | 0.0004 | 22 |
| 8 | 40,320 | <0.0001 | 29 |

**Reading:** at n=4, even *identical* rankings from two audits give p = 0.0833. The statistic
cannot reject the null at α=0.05 under any possible data. This is a property of the design, not
of the results, and would have been discovered only after the full matrix had been run.

### 4.2 CPU training benchmark

See §5.3 — measured on the actual target machine rather than estimated.

---

## 5. Hardware investigation

Triggered by the answer "I have something better, which is this PC itself" (Asus Zenbook S16,
Ryzen AI 9 HX 370, 32 GB). This turned out to be the single most consequential factual correction
in the whole review, so it is documented in full.

### 5.1 Machine as configured (measured, not assumed)

```
AMD Ryzen AI 9 HX 370 w/ Radeon 890M
Cores: 12   Threads: 24
RAM: 31.1 GB
Display adapters: AMD Radeon(TM) 890M Graphics    <- only GPU present
```

No discrete GPU. The Radeon 890M is an RDNA 3.5 integrated GPU; the XDNA 2 NPU is an
inference accelerator, not a training device, and is not exposed to PyTorch.

### 5.2 Can PyTorch use the 890M for training?

**No.** Verified against AMD's own documentation and support channels:

- AMD *does* now ship "PyTorch on Windows" with ROCm 7.2.1, and the marketing lists Ryzen AI 9
  HX 370 among compatible processors — which is how this looks workable at first glance.
- But **ROCm does not support the Radeon 890M (`gfx1150`)**. RDNA 3.5 integrated GPUs are excluded,
  and the Windows HIP SDK does not support it either. AMD's position on this is explicit.
- `torch-directml` exists as a theoretical fallback but lags PyTorch releases badly and is not a
  credible base for a semester of reproducible training runs.

**Conclusion:** on this laptop, PyTorch training is **CPU-only**.

### 5.3 So how slow is CPU-only, actually?

Rather than estimate, this was **measured on the target machine** (`torch 2.13.0+cpu`) with
[`tooling/bench.py`](tooling/bench.py).
ResNet-18 with the standard CIFAR stem (3×3 conv, no maxpool), synthetic 32×32 batches,
forward + backward + SGD step, `torch.set_num_threads(12)` (physical cores; SMT threads hurt
on this workload). Synthetic batches isolate compute from dataloading, so these are
*optimistic* figures — real runs with augmentation will be slower.

```
threads=12  torch=2.13.0+cpu
 batch    s/step      img/s   s/epoch  min/epoch
   128     1.479       86.6     577.6       9.63
   256     2.922       87.6     570.7       9.51
```

| Quantity | Measured / derived |
|---|---:|
| Throughput | **~87 images/second** |
| Time per CIFAR-10 epoch (50k images) | **~9.5 minutes** |
| One 30-epoch training run | **4.76 hours** |
| One 50-epoch training run | **7.93 hours** |
| 36 full training runs @ 30 epochs | **171 hours ≈ 7.1 days continuous** |
| 36 full training runs @ 50 epochs | **285 hours ≈ 11.9 days continuous** |

And 171 hours covers **only the full trainings** — it excludes ~108 unlearning runs, ~150
relearning runs, and every audit pass.

**Comparison (estimated, not measured):** a free Kaggle T4 or P100 runs this workload at
roughly 3,000–5,000 img/s with mixed precision, i.e. **~40–50× faster**. A 30-epoch run is
~10–15 minutes there against 4.76 hours here. The entire 36-run training matrix is ~6–9 GPU-hours
— comfortably inside *one* account's 30 h weekly quota.

**Conclusion: the laptop is roughly 40–50× slower than the free GPU that was being treated as
the fallback.** The intuition that "the PC is better than Kaggle/Colab" is exactly inverted for
this workload — it is true for almost everything else the project needs, and false for training.

### 5.4 The recommendation this produces

The laptop is a **development machine, not a training machine**. Correct division of labour:

| Work | Where | Why |
|---|---|---|
| Model training (M₀, oracles, RMIA references) | **Kaggle** — 30 GPU-h/week, P100 or T4×2, predictable quota | An order of magnitude faster than 12 Zen 5 cores; the quota is published and reliable, unlike Colab's demand-dependent 15–30 h |
| Overflow training | Colab free, and teammates' Kaggle accounts | 4 members × 30 h/week = **120 GPU-h/week**, against a total budget of 17–80 h. Comfortable |
| Unlearning + relearning runs | Kaggle (short) | Minutes each; batch many per session |
| CKA, statistics, plots, dashboard, dev | **The Zenbook** | 12 cores and 32 GB is genuinely excellent for this. Activation analysis is memory-bound, and 32 GB beats a 16 GB T4 here |

**The 32 GB of RAM is a real asset** — just for the audit stage, not the training stage. Caching
activations for layer-wise CKA across many models is exactly the workload it suits.

**Operational consequence:** the project must be structured around **checkpoint artefacts**, not
long live sessions. Kaggle sessions are time-limited and ephemeral. Every training run must write
its checkpoint plus a structured result record to persistent storage (Kaggle Datasets or Drive),
and every audit must run from checkpoints rather than from an in-memory model. The document's §18
logging discipline already requires `checkpoint_path` per experiment — that requirement is now
load-bearing infrastructure, not just good hygiene.

---

## 6. Checked and found sound — no action needed

Recording these so nobody re-litigates settled points.

| Checked | Verdict |
|---|---|
| All 18 existing references | **Real and correctly attributed.** Every one verified against a primary source. Only the two PMLR volume numbers are unconfirmed (m1) |
| Ref [7] Wichert & Sikdar venue | Verified: ACL Anthology `2024.findings-emnlp.271`, EMNLP 2024 Findings, pp. 4727–4739 |
| Ref [13] *Unlearning Isn't Deletion* | Verified: arXiv 2505.16831, ICML 2026 poster |
| Refs [14]/[15] f-divergence + Google blog | Verified: arXiv 2601.19755, Ribero/Schrab/Gretton, AISTATS 2026; blog post live |
| Ref [8] RUM CIFAR-10 memorization scores | Verified released via the RUM repo |
| CIFAR-10 + ResNet-18 as benchmark choice | **Still standard** in 2026 vision unlearning work. §11.1's reasoning holds |
| §3.2 "low forget accuracy is not the goal" | Correct, and now formally backed by arXiv 2604.07962 |
| §14.6 linear probes kept optional | Correct and well-argued. Keep as written |
| §16.2 ≥3 seeds, mean and SD | Correct; arXiv 2510.26714 exists to make this exact argument |
| §18 reproducibility/logging discipline | Strong. Becomes *more* important under the Kaggle workflow (§5.4) |
| §22.3 "claims we must not make" | Genuinely excellent. Do not weaken it. Add one row for the new positioning |
| Dashboard scoped as visualization, not contribution | Correct. Just cite *Unlearning Comparator* as related work |
| Out-of-scope boundaries (§9.2) | Well drawn. No changes |

---

## 7. Open uncertainties and things I could not verify

Stated plainly so they are not mistaken for settled.

1. **ICLR 2026 paper `9IzfArmoHq`** — surfaced in search as an ICLR 2026 paper on unlearning
   evaluation; the OpenReview PDF was behind a bot check. **Could be a further novelty threat.**
   Worth 10 minutes with a browser before the synopsis is finalised.
2. **PMLR volume numbers** for ICML 2026 (doc says vol. 306) and AISTATS 2026 (vol. 300) are not
   publicly confirmed. ICML 2025 was PMLR 267; AISTATS 2025 was PMLR 258. Cite arXiv IDs plus
   "to appear" until confirmed.
3. **arXiv 2605.02206's peer-review status** — preprint under a `neurips26` GitHub org. If it is
   rejected it remains citable, but its weight as a novelty threat changes. Re-check before
   submission.
4. **The exact oracle protocol in 2605.02206** — their §4 says the reference is "fine-tuned on
   retain only", which reads like a *weaker* oracle than a from-scratch retrain. If confirmed,
   that is a legitimate methodological criticism ForgetCheck can make. It should be verified from
   their code before being asserted in print.
5. **The ChatGPT conversation** could not be retrieved. If it contains constraints not in the
   `.docx`, some recommendations may need revisiting.
6. **The compute table's per-run figures** for Kaggle GPUs are estimates from published training
   schedules, not measurements on Kaggle. Only the CPU numbers (§5.3) are measured. Pin the GPU
   figure down with the week-6 pilot.

---

## 8. Changes applied to the master document (v1.0 → v2.0)

The `.docx` was revised in place on 25 August 2026. The frozen v1.0 is preserved at
`docs/ForgetCheck_Master_Project_Reference_v1.0_ORIGINAL.docx` so any change can be diffed against
the original position.

**Method.** Edits were applied programmatically with `python-docx` in four staged passes, each
guarded and verified. Text was replaced at the run level and new rows/paragraphs were cloned from
existing ones, so paragraph styles, table borders, shading and both embedded figures are preserved
(verified: ZIP integrity OK, 21 parts, 2 media files intact). Body text grew from ~51,700 to
~87,400 characters.

| Stage | Sections touched | Edits |
|---|---|---:|
| 1 | Metadata, changelog, §1, §5, §6, §8, §12, §13 | 13 |
| 2 | §14.4, §14.5, §14.7, §15.2, new §15.5, §16.1–16.3 | 9 |
| 3 | §17, §19, §20, §21, §22, §24, §25, Appendices A & B | 13 |
| 4 | Consistency sweep — §2, §5, §9.1, §11.3, §14.8, §16.3, §20.2, §23, §25 | 10 |

**What changed substantively:**

- **New Changelog section** after Document Contents, stating what moved and why, so the mentor can
  see the delta without reading the review.
- **§1, §3, §11.3** — project reframed explicitly as an **Untraining** study per [19]. Title and
  research question updated to lead with audit *validity*, not just disagreement.
- **§5, §23, §24** — literature table and map extended with seven post-freeze works;
  bibliography grown from 18 to 34 references.
- **§6** — novelty statement replaced; contributions explicitly ordered, with replication of [20]
  stated *as* replication rather than buried.
- **§8** — H1 demoted to a replication target with an effect-size prediction; H2 split into H2a
  (between families) and H2b (within the privacy family); H4 extended to cover the negative
  control; **H5 added** for audit validity. RQ2 extended, **RQ6 added**.
- **§12** — SSD withdrawn from the core set with the mechanism/task-mismatch reasoning stated in
  full; NegGrad+ promoted; plain NegGrad relabelled as a destructive control; SalUn and L1-sparse
  added. Six ranked methods.
- **§13** — forget-set design split into size and difficulty axes, low-memorization stratum named
  as a designed negative control, canary condition added, slip-order stated.
- **§14.4** — two attack strengths with the ranking-distortion argument; limitation callout
  rewritten to cover both under- and over-statement.
- **§14.5** — second non-CKA measure and oracle-vs-oracle baseline made *required*.
- **§14.7** — upper and lower anchor arms added; normalised recovery required.
- **§15.2** — rank correlation reworked around instance-level units, with the p = 0.0833 argument
  stated in the document itself so the choice is defensible to a reviewer.
- **§15.5 (new)** — audit calibration and validity: oracle false-positive rate, canary accuracy,
  and a reporting rule for what disagreement with an unreliable audit means.
- **§16** — matrix, statistical reporting and metrics table updated; oracle-SD thresholds mandated.
- **§17, §19, §21** — compute plan corrected to Kaggle-first with the ROCm finding and measured CPU
  figure recorded; two new risk rows (underpowered statistics, schema drift); Gantt rebalanced and
  four-person ownership stated.
- **§22.1** — abstract rewritten around validity. **§22.3** — three rows added to "claims we must
  not make", including one guarding against mischaracterising SSD.
- **Appendices A and B** — decision sheet and cheat sheet updated; three validity rows added.

**What deliberately did not change:** the scoping rules (§9.2), the discipline rule, §14.6's
argument for keeping linear probes optional, the ≥3-seed requirement, §18's logging discipline, and
the structure and tone of the document. Those were assessed and found sound (§6 above).

**Re-running the edits.** The scripts are kept in [`tooling/`](tooling/) — `docx_helpers.py`
(formatting-preserving edit primitives) and `revise_docx.py` … `revise_docx4.py` (the four stages,
in order). Run them from the project root with the venv Python. Each stage is guarded on the
document version, so re-running stage 1 against a v2.0 document fails fast rather than duplicating
content. [`tooling/bench.py`](tooling/bench.py) reproduces the CPU throughput measurement in §5.3
on any machine.

---

## 9. How to re-run this review

The literature in this area is moving fast enough that this review has a half-life of roughly a
semester. To re-check before submission:

1. **Novelty re-check** — search the *claim*, not the topic. Query patterns that worked:
   `"machine unlearning" metrics disagree correlation rank different conclusions`,
   `cross-metric agreement unlearning audit families retraining oracle Kendall`.
   Watch specifically for follow-ups to arXiv 2605.02206.
2. **Instrument re-check** — for any audit added later, search for published critiques of that
   *specific* instrument before adopting it. This is what caught M1 and M2, and both would have
   been invisible in a topic-level search.
3. **Resolve item 1 in §7** (the ICLR paper) — highest-value open thread.
4. **Verify PMLR volumes** once the 2026 proceedings publish.

### Search queries that were productive

- `"Unlearning Isn't Deletion" reversibility machine unlearning LLMs ICML`
- `machine unlearning evaluation metrics disagree correlation between metrics rank different conclusions`
- `Hayes "Inexact Unlearning Needs More Careful Evaluations" U-LiRA per-example membership inference`
- `SSD selective synaptic dampening random subset unlearning fails class unlearning only Fisher importance`
- `CKA centered kernel alignment unreliable similarity measure critique Davari ICLR`
- `RMIA robust membership inference attack few shadow models efficient`
- `relearn time metric machine unlearning vision CIFAR reversibility residual knowledge recovery`
- `PyTorch training AMD Ryzen AI 9 HX 370 Radeon 890M iGPU ROCm Windows support`

### Queries that wasted effort

- Generic `machine unlearning 2026` / `machine unlearning survey` — returns surveys, not competitors.
- Fetching arXiv PDFs directly — often returns raw binary. Prefer `arxiv.org/abs/…` or
  `arxiv.org/html/…`; when only a PDF exists, download and extract locally with `pypdf`.
- OpenReview PDFs — bot-verification interstitial; route via arXiv.

---

*Maintained alongside `FORGETCHECK_REVIEW.md`. Update both when the positioning changes.*
