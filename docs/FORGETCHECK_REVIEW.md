# ForgetCheck, red-teamed against the 2026 literature

**Independent review** · ForgetCheck v1.0 (research position frozen 7 Aug 2026) · Reviewed 25 Aug 2026
**Team of 4** · Compute: Kaggle + Colab free
**Companion:** [`RESEARCH_LOG.md`](RESEARCH_LOG.md) — evidence trail, source-by-source findings, reasoning behind every recommendation
**Web version:** https://claude.ai/code/artifact/6ec6a133-7537-4015-85de-1cae219f52c3
**Status:** Acted upon — the master `.docx` has been revised to **v2.0** implementing every change
below. See `RESEARCH_LOG.md` §8 for the edit record, and
`ForgetCheck_Master_Project_Reference_v1.0_ORIGINAL.docx` for the frozen original.

> The project is **viable, feasible and worth building** — but the headline novelty claim has been
> overtaken by work published since the document was frozen, and four specific design decisions
> will produce degenerate or uninterpretable results if implemented exactly as written. This
> document states what survives, what has to change, and what to claim instead.

---

## Verdict

**Proceed — with a repositioned claim and five structural changes.**

The scientific instincts in the master document are good and unusually well-disciplined for an
undergraduate project. The scoping rules, the "claims we must not make" table, the
low-discriminability safeguard and the refusal to treat MIA as proof of deletion are all things
the published literature has had to learn the hard way. **That discipline is the project's real
asset and none of it should change.**

What has changed is the ground underneath the novelty claim. Between the freeze date and now, at
least three papers have published the specific finding ForgetCheck proposed to discover, and one
of them uses the same statistic, the same oracle grounding and the same CIFAR-10 / ResNet-18 arm.
Separately, four implementation decisions will not do what the document expects them to do.

None of this is fatal. The fixes are known, cheap, and each one makes the paper stronger than the
original plan. Two of them convert a weakness into a result.

| Dimension | Assessment |
|---|---|
| Problem choice | **Strong** |
| Novelty as written | **Overtaken** |
| Novelty, repositioned | **Defensible** |
| Technical feasibility | **Good** |
| Statistical design | **Needs rework** |
| Compute realism | **Not a constraint** |
| Hardware plan | **Was inverted** |
| Scope for 4 × 16 weeks | **Affordable** |
| Scientific integrity | **Exemplary** |

---

## Assessment 01 — The novelty claim has been overtaken

Section 6 states the contribution as "a retraining-grounded empirical study of cross-audit
disagreement." That exact contribution now exists in print.

| Work | Risk | What it already establishes | What it leaves open |
|---|---|---|---|
| **Metric Unreliability in Multimodal Machine Unlearning**<br>Khan, Laga & Sohel · arXiv 2605.02206 · preprint | 🔴 **Direct** | The whole thesis. Five metric families (FA, RA, MIA, activation distance, JS), Kendall's τ across 36 unlearned models, oracle = model retrained on retain only, two opposing metric clusters {FA,RA,MIA} vs {AD,JS}, τ<sub>FA,AD</sub> = −0.26. **Includes a CIFAR-10 / ResNet-18 unimodal arm** (mean pairwise τ = 0.158). | No relearning axis — they name knowledge recoverability as the missing aspect and only run a pilot. No layer-wise representation analysis. No forget-set difficulty axis. Their CIFAR arm is *class-0* forgetting, not instance-level. Their oracle is described as fine-tuned on retain, not trained from scratch. |
| **RULER: Representation-Level Verification**<br>Cosma & Finke · arXiv 2605.27569 · WIPE-OUT @ ECML PKDD 2026 | 🔴 **Direct** | Hypothesis H1, already demonstrated: four approximate methods *all pass output-level evaluation* yet show significant representation residuals in 10 of 12 conditions under a linear mixed-effects model. | Not a disagreement study — it argues representation metrics are *better*, not that audits conflict. No relearning. No memorization stratification. Barely uses CKA. |
| **Are We Truly Forgetting?**<br>arXiv 2503.06991 | 🔴 **Direct** | Also H1: methods either destroy representation quality outright or modify only the classifier while staying representationally close to the original model. | Focused on class-level and large-scale settings; no cross-audit agreement statistics. |
| **Is your algorithm unlearning or untraining?**<br>Triantafillou et al. (Google) · arXiv 2604.07962 | 🟡 **Reframing** | Splits the field into *Untraining* (remove the influence of these specific examples; retrain-from-scratch is the correct oracle) and *Unlearning* (remove the underlying concept; retraining is **not** the right reference). Confirms MIA is appropriate for Untraining specifically. | A position note, not an experiment. ForgetCheck is squarely an **Untraining** study and should say so in that vocabulary. |
| **Benchmarking Unlearning for Vision Transformers**<br>arXiv 2602.20114 | 🟡 **Partial** | Documents ranking instability directly: "method rankings from CNNs do not transfer to VTs; they depend strongly on architectures and proxies," and SalUn scores well on ToW but poorly on ToW-MIA. Uses RUM memorization strata. | Disagreement is an observation, not the object of study. No relearning, no representation layer, no agreement statistics. |
| **Unlearning Comparator**<br>arXiv 2508.12730 | 🟢 **Deliverable** | A visual analytics system for comparative evaluation of unlearning methods — i.e. the Streamlit dashboard's job, already built and published. | Nothing scientific. The document already scopes the dashboard as a visualization layer, which remains correct — just cite this. |
| **Auditing Machine Unlearning**<br>Ye et al. · arXiv 2606.16110 · preprint | 🟢 **Context** | General-purpose auditing framework, ten methods, six datasets. Notably finds Fisher/Hessian-based methods *failed* despite formal certification — relevant to B1. | Preprint only, no accepted venue as of this review. Cite as such. |

> **The uncomfortable sentence.** If a reviewer reads ForgetCheck's Section 6 and then reads
> arXiv 2605.02206, the paper looks like a unimodal replication of a multimodal study — including
> replication of the one CIFAR-10 arm that study already ran. The current novelty statement cannot
> survive that comparison. It has to be narrowed to the part nobody has done.

---

## Assessment 02 — Where genuine headroom remains

Four openings survive the sweep. Two are gaps competitors explicitly named and could not close;
two are moves nobody in this literature has made.

### 1. Reversibility as a ranked audit family, in vision, at instance level

arXiv 2605.02206 says outright that none of its five metrics measure whether knowledge is erased
or merely suppressed, calls this *knowledge recoverability*, and reports that it "does not scale
directly" — so they excluded it from the ranking analysis and ran a pilot instead. Reversibility
exists in the vision literature as a metric (re-learn time; NegGrad+ recovers much faster than
retraining) and as an attack (prototypical relearning, arXiv 2506.01318 — but that is *class*
unlearning).

**Nobody has put relearning in as a full audit family and asked whether it agrees with the other
three, at instance level, in vision.** That is ForgetCheck's Audit Layer 4, and it is the
strongest card in the deck.

### 2. Disagreement crossed with forget-set difficulty

RUM established that difficulty varies by memorization; Wichert & Sikdar established that
influential forget sets are more discriminative than random ones and that the gap *shrinks as
forget-set size grows*. Neither asked whether **audits start contradicting each other** as
difficulty rises. RQ5 / H4 is untouched ground and should be promoted from a secondary
contribution to a co-primary one.

### 3. Disagreement *inside* a family, not just between families

This is the move I would most want the paper to make. Hayes et al. show that population-level
U-MIAs systematically overestimate the privacy protection of unlearning, while per-example attacks
(U-LiRA) are far stronger and reveal that privacy risk actually *increases* after unlearning for a
substantial number of examples. So the same model is "private" under the attack the NeurIPS
starter kit ships and "leaky" under a stronger one.

Running both and reporting the gap turns finding M1 from a methodological embarrassment into a
headline result — and no cross-audit study currently does it.

### 4. Scoring audits instead of only comparing them

"Audits disagree" is descriptive and, as of 2605.02206, no longer new. "**Which audit is right?**"
is normative and is still open. Two cheap additions make it answerable:

- **Oracle-passes-its-own-audit.** Feed a held-out, independently retrained oracle into every
  audit as if it were a candidate. Any audit that flags a genuine from-scratch retrain as "not
  forgotten" has a false-positive problem, and you can report each audit's FPR as a number. The
  recent LLM work at arXiv 2607.19442 found exactly this failure — their retraining reference
  certified in only 1 of 45 cells — which shows the check catches real breakage rather than being
  a formality.
- **A ground-truth-by-construction condition.** Inject a small canary set of deliberately
  mislabeled examples into training. These can only be fitted by memorization, so the retrained
  oracle provably has zero knowledge of the canary → label mapping, and any residual canary
  behaviour in an unlearned model is unambiguously residual influence. Now audits can be scored
  for *accuracy against a known answer*, not just for mutual agreement. This is the vision
  analogue of what TOFU does with fictitious authors and what arXiv 2607.19442 does with injected
  nonce facts.

Together these turn the paper from "we observed disagreement" into "we calibrated four audit
families against a counterfactual reference and a known ground truth, and here is which ones you
can trust and when." That is a materially harder claim to scoop.

---

## Findings · Tier 1 — Blocking (fix before writing code)

Each of these will produce results that cannot be interpreted, not merely results that are weaker
than hoped.

### 🔴 B1 — SSD will act as a no-op on random instance forget sets

SSD identifies parameters disproportionately important to the forget set via Fisher information,
then dampens them. That mechanism needs the forget set to have a *distinct* importance signature.
A random 1–10% subset of CIFAR-10 is drawn from the same distribution as the retain set, so it has
no such signature.

This is not speculation. A 2024 comparative study (arXiv 2412.19583) reports that SSD "fails to
forget across all percentage values in random forgetting scenarios, likely due to similarities in
Fisher Information Matrix values," while performing at or above state of the art on single-class
and sub-class unlearning. The SSD paper itself evaluates class and sub-class forgetting, and the
official repo's scripts are oriented to class unlearning. Independently, arXiv 2606.16110 finds
Fisher/Hessian-based methods failing despite formal certification.

- **Impact:** SSD ≈ M₀ in every audit, in every random condition. It occupies a rank slot without
  carrying information, and 25% of the ranking signal evaporates. Worse, it will look like
  "perfect utility retention, zero forgetting" — which a careless reading could mistake for a
  finding.
- **Fix:** Replace SSD in the core method set with **SalUn** and **L1-sparse** — both are standard
  instance-level baselines, both are already implemented in the RUM codebase for exactly
  CIFAR-10 / ResNet-18. Retain SSD only if you add a class-unlearning condition (where it is the
  right tool), or report it in an appendix as a documented mechanism–task mismatch. **This also
  solves B2.**

### 🔴 B2 — Kendall's τ over four methods cannot produce a significant result

Section 16.1 names Kendall's τ the primary agreement statistic, and Section 12 fixes the method
count at four. With *n* = 4 items there are only 24 permutations, so τ takes just 7 distinct values
and the smallest attainable two-sided p-value is **0.0833**. If two audits rank your four methods
*identically*, you still cannot reject the null at α = 0.05. **The primary statistic is inert by
construction.**

| Methods ranked (*n*) | Permutations | Min. attainable two-sided p | Distinct τ values |
|---:|---:|---:|---:|
| **4** ← as planned | 24 | **0.0833** | 7 |
| 5 | 120 | 0.0167 | 11 |
| **6** ← recommended floor | 720 | **0.0028** | 16 |
| 7 | 5,040 | 0.0004 | 22 |

**Fix — change the unit of analysis.** Do not correlate ranks over 4 methods; correlate metric
*values* over all model instances — every (method × forget-setting × seed) cell is one observation,
giving *n* ≈ 75 instead of 4. This is precisely what arXiv 2605.02206 did with its 36 models, and
it is why they could report p = 0.003. Then:

1. Raise the method count to ≥6 anyway, which B1's fix already does.
2. Use **linear mixed-effects models** — `metric ~ method + forget_condition + (1 | seed)` — for
   the confirmatory tests, following RULER.
3. Keep per-setting rank tables as *descriptive* displays with bootstrap CIs, never as the
   inferential backbone.

### 🔴 B3 — Random 1% (and likely 5%) forget sets are near-degenerate

The document already anticipates this in Section 15.4's low-discriminability safeguard, which is
to its credit. The Google position note makes the mechanism explicit: for low-memorization
examples, the retrained-from-scratch model still predicts them correctly, so the ideal untraining
solution is **effectively no change from the original model at all**. Where that holds, M₀ ≈ M_r,
the normalized oracle-gap denominator collapses, and the safeguard fires.

> **Corrected 29 Aug 2026, after measuring the real scores.** This finding originally claimed a
> random CIFAR-10 subset is *"overwhelmingly low-memorization"* and therefore near-degenerate.
> That was an inference, and it was wrong. Measured: 26.6% of CIFAR-10 falls below 0.01, but
> **24.9% sits above 0.5**, and a random 3000-example forget set has mean memorization 0.276 with
> 731 examples as memorized as the high stratum. Random conditions are a *mixture* — they dilute
> the signal rather than lacking it. The recommendation below is unchanged, because a pure
> high-memorization stratum still concentrates what random dilutes, but the effect is one of
> degree and the random conditions should not be written up as empty.

Wichert & Sikdar sharpen it further: the influential-vs-random gap *shrinks as forget-set size
grows*. So the random-vs-hard contrast is most visible at *small* forget sets, which is the
opposite of the intuition that motivates a 10% condition.

- **Fix:** Split the single forget-set axis into two orthogonal factors. **Size** (random 1 / 5 /
  10%) stays, demoted to a scaling check. **Difficulty** becomes the primary factor at *fixed*
  size, using RUM's low / medium / high memorization strata — CIFAR-10 memorization scores are
  released in the RUM repo, so this is a download, not a computation.
- **Bonus:** Add the **low-memorization stratum as a designed negative control**. Theory predicts
  it carries no signal. If your audits correctly report "nothing to detect" there and a clear
  signal at high memorization, you have validated the audit battery itself rather than merely
  excusing an empty cell. This reframes your weakest condition as evidence of instrument quality.

### 🔴 B4 — One retrained oracle per cell is a sample, not a reference

Section 10 treats M_r as the counterfactual, and Section 16.2 correctly says to treat retraining
variation as part of the reference distribution. But the matrix in 16.1 provisions one oracle per
(setting, seed), which means every "distance from oracle" number silently contains training
randomness you have no estimate of. Any threshold defined against a single draw is arbitrary.

The field has moved decisively here: EasyDUB (arXiv 2602.16400) ships **200 oracle models per
forget set** for CIFAR-10 and evaluates against the oracle *ensemble* via KL-of-margins; the
Google/AISTATS f-divergence work is built entirely on relative distribution tests rather than point
comparisons; and arXiv 2510.26714 exists solely to argue that single-seed unlearning evaluation is
unsound.

- **Fix:** Train **≥5 independent oracles at the primary setting** (more if compute allows) to
  estimate the oracle-vs-oracle null band for every metric. Express all thresholds in **oracle-SD
  units**, never absolute values. Reserve at least one oracle as a held-out probe for the
  "oracle-passes-its-own-audit" check. Consider reusing EasyDUB's released checkpoints for
  calibration — note they are ResNet-9, so treat them as a calibration reference rather than a
  substitute for your own ResNet-18 runs.

---

## Findings · Tier 2 — Major (fix before the results are written up)

### 🟡 M1 — The specified MIA is the weak attack the literature says overstates privacy

Section 14.4 specifies ROC-AUC and attack accuracy from loss/confidence/entropy — a
population-level attack, one attacker for all examples. Hayes et al. categorise exactly this as a
"population U-MIA" and show it systematically overestimates the privacy protection of unlearning
methods; per-example attacks such as U-LiRA are far stronger and reveal that privacy risk
*increases* post-unlearning for a substantial fraction of examples.

The document's own caution about MIA is right in spirit but does not go far enough: the issue is
not that a weak attack proves nothing, it is that **a weak attack produces an actively misleading
ranking** — and ranking is ForgetCheck's whole output.

- **Fix:** Run **both**, and treat the gap as a result. For the strong attack use **RMIA**
  (Zarifzadeh, Liu & Shokri, ICML 2024) rather than full U-LiRA — RMIA was designed for exactly
  this constraint and performs well with as few as one to a handful of reference models, where
  earlier attacks collapse to random guessing.
- **Caveat:** EasyDUB argues U-LiRA-style scores are gameable by degenerate strategies (constant
  margins) that destroy utility. Always pair the privacy number with the retained-utility guard in
  Layer 1 — the audit ladder already handles this correctly, so just say so explicitly.

### 🟡 M2 — CKA alone cannot carry the representation audit

Davari et al. (ICLR 2023, arXiv 2210.16156) show that CKA values can be changed substantially
*without meaningful change to a model's functional behaviour*, and formally characterise CKA's
sensitivity to outliers and to transformations that preserve linear separability. Since the whole
point of Audit Layer 3 is to make a claim about internal state that output-level metrics miss, a
metric that moves independently of function is a dangerous sole instrument.

- **Fix:** Report linear CKA *plus* at least one mechanistically different measure — RBF-CKA,
  orthogonal Procrustes distance, or distance correlation — and require them to agree before any
  representation-level claim. Critically, always report the **oracle-vs-oracle CKA baseline** so
  "how similar is similar" has a scale. Given B4 you will have the extra oracles anyway.
- **Note:** Section 14.6's argument for keeping linear probes optional is correct and
  well-reasoned. It should stay.

### 🟡 M3 — The relearning protocol needs anchors and a matched starting point

Section 14.7 correctly requires identical optimizer and data schedules for M_u and M_r, and the
risk table names optimization confounds. Two gaps remain. First, models entering relearning at
different utility levels have different amounts of headroom, so a raw recovery curve conflates
"retained structure" with "started closer." Second, with only M_u and M_r there is no scale — fast
relative to what?

- **Fix:** Add two anchor arms: **M₀** (upper bound — the model that never forgot) and a
  **randomly initialised model** or from-scratch retrain (lower bound — no prior exposure). Report
  recovery on a scale normalised between those anchors. Match starting utility explicitly, or
  report it as a covariate in the mixed-effects model from B2. Log the full trajectory, report both
  T80 and AUC, and confirm retained-set utility does not collapse during relearning.
- **Prior art:** Re-learn time already exists as a vision unlearning metric — published results
  show NegGrad+ recovering far faster than retraining. Cite it as an existing instrument you are
  *elevating to a ranked audit family*, not as a new metric.

---

## Findings · Tier 3 — Minor

### 🔵 m1 — Two bibliography entries cite unconfirmed PMLR volumes

Reference [13] gives ICML 2026 as PMLR vol. 306 and [14] gives AISTATS 2026 as PMLR vol. 300. Both
papers are real and correctly attributed — [13] is arXiv 2505.16831, accepted to ICML 2026; [14] is
arXiv 2601.19755 by Ribero, Schrab & Gretton, AISTATS 2026 — but neither volume number could be
verified. ICML 2025 was PMLR 267 and AISTATS 2025 was PMLR 258. Cite the arXiv IDs plus "to appear"
until the volumes publish.

### 🔵 m2 — Reference [16] is a preprint with no accepted venue

arXiv 2606.16110 (Ye et al., submitted 15 June 2026) shows no venue. The document already labels it
a preprint, which is correct — flagged only so nobody upgrades it during editing. Its finding is
useful: retraining- and fine-tuning-based methods passed their audit while de-optimization and
Fisher/Hessian methods failed, which independently corroborates B1.

### 🔵 m3 — The memorization-score dependency is satisfied, with a fallback worth knowing

Section 13 assumes released CIFAR-10 memorization scores. Those exist in the RUM repo. Worth
knowing that Feldman & Zhang's own release covers **CIFAR-100 and ImageNet, not CIFAR-10**, so if
the RUM download breaks you are not falling back on the canonical source. The escape hatch is
cheap: Zhao et al. (arXiv 2410.16516) report Spearman correlations of **−0.80 to −0.91** for a
simple confidence proxy, at roughly **0.002%** of the cost of exact scores. Pre-register which
proxy you would use.

### 🔵 m4 — "Negative Gradient" should be NegGrad+, with plain NegGrad as a control

Section 12 lists gradient ascent "optionally with retained-data regularization." Make that
non-optional and name it: **NegGrad+** is the standard baseline in RUM and the ViT benchmark, and
is reported as the only method improving ToW-MIA under both memorization proxies on
CIFAR-10/ResNet-18. Keep *plain* NegGrad as a separate arm, explicitly labelled the **destructive
control** — it is the cleanest demonstration that "looks forgotten" can mean "damaged," which is
the argument Audit Layer 1 exists to make.

### 🔵 m5 — Adopt the Untraining / Unlearning vocabulary explicitly

Sections 3.2 and 3.3 make the right distinction in your own words. Since April 2026 there is
standard terminology for it, from the group that ran the NeurIPS competition. Stating in Section 1
that ForgetCheck studies **Untraining** costs one paragraph, immunises against "why is retraining
your reference?", and signals currency with the literature. Cheapest credibility gain available.

---

## Revised plan 01 — The claim to make instead

> **Proposed replacement for Section 6**
>
> Cross-audit disagreement in machine unlearning has been documented, but existing studies compare
> output-, privacy- and alignment-based metrics only, quantify agreement without adjudicating it,
> and do not vary the difficulty of the deletion request. ForgetCheck extends the analysis in three
> directions in the *Untraining* setting: it adds **controlled relearning** as a fourth ranked audit
> family — the axis prior work identified as missing but did not scale; it measures disagreement
> **within** the privacy family as well as between families, contrasting a population
> membership-inference attack with a per-example one; and it treats **forget-set memorization
> difficulty** as an experimental factor rather than a fixed condition. All audits are calibrated
> against an ensemble of independently retrained oracles, and validated on a canary condition where
> residual influence is known by construction, allowing audits to be scored for accuracy rather
> than only compared for agreement.

Note what this concedes and what it keeps. It concedes that disagreement exists and was found by
others — which is honest, is what the discipline rule in the document demands, and costs nothing,
because *replicating* a multimodal preprint's finding in a clean unimodal instance-level setting
with a true from-scratch oracle is a legitimate secondary contribution. What it keeps is the part
that is actually yours: the reversibility axis, the difficulty interaction, the intra-family
contrast, and calibration.

**Hypothesis edits:**

- **H1 is no longer a hypothesis** — RULER and arXiv 2503.06991 have established it. Restate as a
  replication target with a prediction about effect size.
- **H2 becomes stronger** if split into inter-family and intra-family versions.
- **H3 and H4 are untouched** and become the paper's core.

---

## Revised plan 02 — Revised experimental design

Changes marked against Appendix A's canonical decision sheet. Everything unmarked stays as written.

| Decision | | Revised choice | Because |
|---|---|---|---|
| Dataset | KEEP | CIFAR-10 primary; CIFAR-100 optional | Still the standard instance-unlearning benchmark; retraining-feasible; memorization scores released |
| Architecture | KEEP | ResNet-18 | Rank-transfer across architectures is a separate paper (arXiv 2602.20114) |
| Unlearning methods | **CHANGE** | Fine-tune · **NegGrad+** · NegGrad (destructive control) · SCRUB · **SalUn** · **L1-sparse** → **6** | B1 (SSD degenerate at instance level) and B2 (τ needs ≥6 ranked units). All six are in the RUM codebase |
| SSD | **DEMOTE** | Appendix only, or a class-unlearning side condition | Mechanism–task mismatch; would contribute a constant row |
| Forget sets — size axis | **DEMOTE** | Random 1 / 5 / 10% as a scaling check | B3: low discriminability. Keep for the scalability RQ, don't build conclusions on it |
| Forget sets — difficulty axis | **PROMOTE** | Low / medium / high memorization strata at fixed size (RUM scores) | B3 and headroom 2. Low stratum doubles as a negative control |
| Canary condition | **ADD** | Small mislabeled-canary forget set | Headroom 4: ground truth by construction, so audits can be scored |
| Reference models | **CHANGE** | ≥5 independent oracles at primary setting; ≥1 held out as audit probe | B4: the oracle is a distribution. Thresholds in oracle-SD units |
| Privacy audit | **CHANGE** | Population loss/confidence attack **and** RMIA per-example attack | M1: the gap between them is a headline result |
| Representation audit | **CHANGE** | Linear CKA + one non-CKA measure + oracle-vs-oracle baseline | M2: CKA moves independently of function |
| Relearning audit | **EXTEND** | Add M₀ and random-init anchor arms; normalise between them | M3: raw curves conflate retained structure with starting headroom |
| Primary statistic | **CHANGE** | Metric-value correlation across ~75+ model instances; mixed-effects for confirmatory tests | B2: τ over 4 methods cannot reach significance |
| Seeds | KEEP | ≥3, mean and SD reported | Correct as written; arXiv 2510.26714 makes this exact point |
| Dashboard | KEEP | Streamlit, scoped as a visualization layer | Correctly scoped. Cite Unlearning Comparator as related work |
| LLM / TOFU extension | **CUT** | Remove from the plan entirely | The added canary and calibration work consumes the slack this was drawing on |

> **On the scope trade.** Six methods instead of four, and two forget-set axes instead of one, is
> more work than the frozen plan — but adopting the RUM codebase returns more time than these cost,
> and cutting the LLM extension returns the rest. Net: roughly neutral on effort, substantially
> positive on what the results can support.

---

## Revised plan 03 — Compute: the Zenbook cannot train this

**This is the largest factual correction in the review.** The laptop is an excellent machine for
the project — but not for the part it was going to be used for.

> **Measured on the target machine, 25 Aug 2026**
>
> **~87 images/second. ~9.5 minutes per CIFAR-10 epoch. 4.76 hours for one 30-epoch training run.**
> The 36 full training runs the revised matrix needs would take **171 hours — 7.1 days of
> continuous compute** — excluding ~108 unlearning runs, ~150 relearning runs and every audit pass.
>
> A free Kaggle T4 or P100 does the same 30-epoch run in roughly 10–15 minutes.
> **The laptop is about 40–50× slower than the free GPU that was being treated as the fallback.**

### Why: there is no usable GPU for PyTorch

The Ryzen AI 9 HX 370 has no discrete GPU. Its Radeon 890M is an RDNA 3.5 integrated GPU, and
**ROCm does not support the 890M (`gfx1150`)** — nor does the Windows HIP SDK. This is easy to get
wrong, because AMD *does* now ship "PyTorch on Windows" with ROCm 7.2.1 and lists the HX 370 among
compatible processors; the **processor** is supported, the **integrated GPU** is not. The XDNA 2
NPU is an inference accelerator, not exposed to PyTorch training. `torch-directml` exists but lags
PyTorch releases badly and is not a credible base for a semester of reproducible runs.

Measured benchmark (`torch 2.13.0+cpu`, ResNet-18 with CIFAR stem, `set_num_threads(12)`):

```
 batch    s/step      img/s   s/epoch  min/epoch
   128     1.479       86.6     577.6       9.63
   256     2.922       87.6     570.7       9.51
```

### Division of labour

| Work | Where | Why |
|---|---|---|
| All model training — M₀, oracles, RMIA references | **Kaggle** (primary) | 30 GPU-h/week, P100 or T4×2, **published and predictable** quota — unlike Colab free, which is demand-dependent at 15–30 h |
| Overflow and parallel sweeps | Teammates' Kaggle accounts, Colab free | **4 × 30 = 120 GPU-h/week** against a budget of ~11–26 h. Compute constraint disappears |
| Unlearning + relearning runs | Kaggle, batched | Minutes each — pack many per session |
| CKA, statistics, figures, dashboard, all development | **The Zenbook** (ideal) | Activation analysis is memory-bound. **32 GB beats a 16 GB T4 here.** Genuinely the best machine in the project for the audit stage |

### Revised budget, on GPU

| Workload | Runs | Est. each | Subtotal |
|---|---:|---:|---:|
| Original models M₀ | 3 | 10–15 min | ~1 h |
| Retrained oracles M_r | ~21 | 10–15 min | 4–5 h |
| RMIA reference models | 8–16 | 10–15 min | 2–4 h |
| Unlearning runs | ~108 | 1–5 min | 2–9 h |
| Relearning runs | ~150 | <1–3 min | 2–7 h |
| Audit passes | — | — | 4–8 h (on the Zenbook, off-quota) |
| **Total GPU** | | | **~11–26 h** — one week of one member's quota |

### The real constraint: ephemeral sessions

Kaggle sessions are time-limited and disposable, which changes the architecture. The project must
be built around **checkpoint artefacts, not long live sessions**: every training run writes its
checkpoint *and* a structured result record to persistent storage (Kaggle Dataset or Drive), and
every audit runs from checkpoints rather than from a model in memory. Section 18 already mandates a
`checkpoint_path` per experiment — that requirement is now **load-bearing infrastructure**, and it
should be built in week 3, not retrofitted in week 10.

Two further levers: fix a **shorter epoch schedule** that reaches a stated accuracy and publish it
(a short schedule applied identically to M₀, oracles and references is scientifically fine, and
halves everything), and enable **mixed precision plus channels-last** on the GPU path from the start.

---

## Revised plan 04 — Splitting this across four people

The revised design parallelises unusually well, because the four audit layers are genuinely
independent once the model artefacts exist. The risk is not overlap — it is the shared substrate
becoming a bottleneck.

| Owner | Owns | Weeks | Critical path? |
|---|---|---|---|
| **A — Substrate** | RUM codebase adoption, M₀/oracle training, forget-set generator (size, memorization strata, canaries), checkpoint + logging infrastructure, Kaggle orchestration | 3–7, then support | 🔴 **Yes — blocks everyone** |
| **B — Behaviour & privacy** | Audit Layers 1, 1B, 1C; both MIAs (population + RMIA); intra-family gap analysis | 8–10 | 🟡 Partly |
| **C — Representation & reversibility** | Audit Layers 3 and 4; CKA + second measure; relearning with anchor arms | 10–12 | 🟡 Partly |
| **D — Analysis & delivery** | Oracle-ensemble calibration, mixed-effects models, disagreement analysis, figures, dashboard, paper assembly | 7–16 | 🟢 No, but owns the output |

Three rules make this work:

- **Freeze the result-record schema in week 3, before any audit is written.** Every audit writes
  one row per (model, condition, seed) with the same keys. If B and C invent their own formats, D
  spends weeks 13–14 writing joins instead of statistics — the classic way this kind of project
  fails at the end.
- **A is on the critical path alone until week 7.** Give A a second person during weeks 3–5 if the
  pilot slips at all — B and C have nothing to audit until checkpoints exist, so idle time there is
  invisible until it is fatal.
- **D starts in week 7, not week 13.** Calibration is a prerequisite for interpreting *any* audit,
  so it cannot be back-loaded with the writing.

Every member gets a Kaggle account contributing to the shared pool, but **A should own the training
queue** — parallel uncoordinated runs across four accounts is how checkpoint provenance gets lost.

---

## Revised plan 05 — What to build on

The single highest-leverage change in this review: weeks 3–7 of the frozen plan are largely already
written by someone else, under a matching setup.

| Resource | Use | Why it matters |
|---|---|---|
| **[kairanzhao/RUM](https://github.com/kairanzhao/RUM)** | **Primary base** | CIFAR-10 + ResNet-18 + instance-level forget sets + released memorization scores, with Retrain, Fine-tune, L1-sparse, NegGrad variants, SCRUB, Influence, SalUn and Random-label already implemented, plus MIA analysis scripts. **This is your exact setting.** Collapses weeks 3–7 into roughly two |
| **[torchunlearn](https://github.com/Harry24k/machine-unlearning-pytorch)** (MIT, pip) | Cross-check | 17+ methods behind a unified PyTorch API, CIFAR-10/ResNet-18 as documented benchmark. Independent implementation to validate that SCRUB and NegGrad+ results aren't codebase artefacts |
| **[meghdadk/SCRUB](https://github.com/meghdadk/SCRUB)** | Reference | Official SCRUB. Validate your port against a reported baseline before trusting it |
| **EasyDUB** (arXiv 2602.16400, HuggingFace) | Calibration | 200 pretrained models, 10 forget sets, 200 oracles per forget set on CIFAR-10 — a free sample of the oracle distribution. ResNet-9, so not a drop-in substitute, but invaluable for sanity-checking oracle-to-oracle variation |
| **NeurIPS 2023 starter kit** | Keep | Still the right source for the *weak* population MIA in M1 — you want the attack the community actually ships |
| **[SSD official repo](https://github.com/if-loops/selective-synaptic-dampening)** | Only if | Class-unlearning oriented. Needed only if you keep SSD as a side condition per B1 |

Section 25's rule — every imported component must be reproduced, validated and documented under
your own protocol — stays as written, and matters **more** under this recommendation, not less.
Adopting a codebase is not the same as adopting its results.

---

## Revised plan 06 — Revised schedule

Same 16 weeks. Method implementation shrinks; calibration and the two audit layers that carry the
contribution expand.

| Week | Revised work | Change |
|---:|---|---|
| 1–2 | Reposition claim and hypotheses; rewrite §6 and §8; adopt Untraining vocabulary; synopsis | **REWRITE** — repositioning must precede synopsis approval |
| 3 | Stand up RUM codebase; reproduce its reported baseline; M₀ pipeline; download memorization scores; **freeze result-record schema** | **COMPRESSED** — adopt rather than build |
| 4 | Forget-set generator: size axis, memorization strata, canary condition. First oracles | **EXPANDED** — canary condition is new |
| 5 | All six unlearning methods running end to end at one seed | **COMPRESSED** — was 3 weeks for 4 methods |
| 6 | **Full-pipeline pilot**: one seed, one condition, all four audits, disagreement analysis run to a figure | **NEW** — the most valuable week in the plan |
| 7 | Oracle ensemble; oracle-vs-oracle null bands; oracle-passes-its-own-audit calibration | **NEW** — B4 and headroom 4 |
| 8 | Behavioural and output-similarity audits, thresholds in oracle-SD units | as planned |
| 9–10 | Both MIAs: population attack and RMIA reference models; intra-family gap analysis | **EXPANDED** — now a result, not a checkbox |
| 11 | Representation audit: CKA + second measure + oracle baseline | as planned |
| 12 | Relearning audit with anchor arms and normalised recovery | **EXPANDED** — anchors are new |
| 13 | Full multi-seed matrix | as planned |
| 14 | Mixed-effects models, correlation across model instances, disagreement matrix, canary scoring | **REWORK** — new statistical backbone per B2 |
| 15 | Dashboard + report draft | as planned |
| 16 | Paper + presentation | as planned |

**If the schedule slips**, the order to sacrifice is: CIFAR-100 secondary validation first, then the
size axis (1% and 10%), then the canary condition. **Never** the oracle ensemble or the relearning
layer — those are the contribution.

---

## Revised plan 07 — Bibliography: what to add

All eighteen existing references check out as real and correctly attributed — only the two PMLR
volume numbers in m1 are unverified. These are the additions a 2026 reviewer will expect.

| Work | arXiv / venue | Why it must be cited |
|---|---|---|
| Is your algorithm unlearning or untraining? | 2604.07962 | Defines the Untraining/Unlearning split you work inside. Non-negotiable |
| Metric Unreliability in Multimodal MU | 2605.02206 | Nearest neighbour to your claim. Cite prominently, state your delta explicitly |
| Inexact Unlearning Needs More Careful Evaluations | 2403.01218 | Population vs per-example U-MIA. Basis of M1 |
| RULER: Representation-Level Verification | 2605.27569 | Establishes your H1; supplies the mixed-effects statistical template |
| Are We Truly Forgetting? | 2503.06991 | Also establishes H1, from the representation-quality direction |
| Low-Cost High-Power MIA (RMIA) | 2312.03262 · ICML 2024 | The attack that makes a strong per-example MIA affordable |
| Reliability of CKA as a Similarity Measure | 2210.16156 · ICLR 2023 | Basis of M2. A reviewer will ask |
| Benchmarking Unlearning for Vision Transformers | 2602.20114 | Rank instability across architectures and proxies; ToW / ToW-MIA |
| Easy Data Unlearning Bench | 2602.16400 | Oracle-ensemble evaluation and KLoM; basis of B4 |
| Are we making progress in unlearning? | 2406.09073 | Competition findings — companion to existing ref [17] |
| On multiple training seeds for evaluating unlearning | 2510.26714 | Direct support for your ≥3-seed requirement |
| Scalability of memorization-based unlearning | 2410.16516 | Memorization proxies and their cost — the m3 fallback |
| Unlearning Comparator | 2508.12730 | Related work for the dashboard deliverable |
| A Reliable Cryptographic Framework for MU Evaluation | 2404.11577 | Formal treatment of what makes an MIA-based metric reliable |

---

## Decisions now locked

| Question | Answer | Consequence |
|---|---|---|
| Hardware | Kaggle + Colab free; Asus Zenbook S16 (Ryzen AI 9 HX 370, 32 GB); teammates' PCs; time no object | **Largest correction in the review.** Zenbook cannot GPU-train — 4.76 h/run vs ~12 min on free Kaggle. Kaggle becomes primary compute; laptop becomes the audit machine, where 32 GB is a real advantage |
| Team size | Four | Six-method, two-axis design comfortably affordable. Binding constraint is the shared result-record schema, not person-hours |
| Submission intent | Treat as a real submission | Repositioning is **mandatory, not optional**. Canary condition moves to high priority |
| Freedom to edit frozen document | Complete | §6, §8, §12, §13, §16 rewritten directly. Version bumped to 2.0 with a changelog |

### The three things to do first

1. **Resolve the one unchecked novelty thread.** An ICLR 2026 paper on unlearning evaluation
   (OpenReview `9IzfArmoHq`) sat behind a bot check and could not be read. Ten minutes in a browser.
   Do it before the synopsis is approved.
2. **Stand up the checkpoint and result-record schema in week 3.** Under an ephemeral-session
   workflow this is load-bearing, and it is the single thing most likely to cost weeks 13–14 if
   deferred.
3. **Protect week 6 — the full-pipeline pilot.** One seed, one condition, all four audits, run
   through to a disagreement figure. The cheapest possible place to discover a wrong decision.

---

## Sources consulted

Primary sources read for this review, beyond the eighteen already in the project document. Full
findings and reasoning: [`RESEARCH_LOG.md`](RESEARCH_LOG.md).

- Triantafillou, Humayun, Ribero, Turner, Mozer & Kaissis — [Is your algorithm unlearning or untraining?](https://arxiv.org/abs/2604.07962) (Google, Apr 2026)
- Khan, Laga & Sohel — [Metric Unreliability in Multimodal Machine Unlearning](https://arxiv.org/abs/2605.02206)
- Hayes, Shumailov et al. — [Inexact Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy](https://arxiv.org/abs/2403.01218)
- Cosma & Finke — [RULER: Representation-Level Verification of Machine Unlearning](https://arxiv.org/abs/2605.27569) (WIPE-OUT @ ECML PKDD 2026)
- [Are We Truly Forgetting? A Critical Re-examination of MU Evaluation Protocols](https://arxiv.org/abs/2503.06991)
- Zarifzadeh, Liu & Shokri — [Low-Cost High-Power Membership Inference Attacks](https://arxiv.org/abs/2312.03262) (ICML 2024)
- Davari et al. — [Reliability of CKA as a Similarity Measure in Deep Learning](https://arxiv.org/abs/2210.16156) (ICLR 2023)
- [Benchmarking Unlearning for Vision Transformers](https://arxiv.org/abs/2602.20114)
- [Easy Data Unlearning Bench](https://arxiv.org/abs/2602.16400)
- Triantafillou et al. — [Are we making progress in unlearning?](https://arxiv.org/abs/2406.09073)
- Zhao et al. — [Scalability of memorization-based machine unlearning](https://arxiv.org/abs/2410.16516)
- [A Comparative Study of MU Techniques for Image and Text Classification Models](https://arxiv.org/abs/2412.19583)
- [On the importance of multiple training seeds for evaluating machine unlearning](https://arxiv.org/abs/2510.26714)
- Ye et al. — [Auditing Machine Unlearning](https://arxiv.org/abs/2606.16110) (preprint)
- [Unlearning as Distribution Restoration](https://arxiv.org/abs/2607.19442)
- [Unlearning's Blind Spots: Over-Unlearning and Prototypical Relearning Attack](https://arxiv.org/abs/2506.01318)
- Tu, Hu & Ma — [A Reliable Cryptographic Framework for Empirical MU Evaluation](https://arxiv.org/abs/2404.11577)
- Ribero, Schrab & Gretton — [Regularized f-Divergence Kernel Tests](https://arxiv.org/abs/2601.19755) and the [Google Research blog post](https://research.google/blog/new-framework-for-auditing-machine-unlearning/)
- Code: [kairanzhao/RUM](https://github.com/kairanzhao/RUM) · [torchunlearn](https://github.com/Harry24k/machine-unlearning-pytorch) · [meghdadk/SCRUB](https://github.com/meghdadk/SCRUB) · [SSD](https://github.com/if-loops/selective-synaptic-dampening)

---

*Review conducted 25 August 2026 against ForgetCheck Master Reference v1.0, revised the same day
once hardware and team details were supplied. CPU training figures are **measured on the target
machine**; GPU figures are estimates from published throughput and should be pinned down in the
week-6 pilot. Two bibliography items (m1) and one possible novelty threat (OpenReview `9IzfArmoHq`)
could not be verified and are flagged rather than resolved. The ChatGPT conversation linked in the
brief renders client-side and could not be retrieved, so this review is based on the master document
alone — if it contains constraints not in the document, they may change some recommendations.*
