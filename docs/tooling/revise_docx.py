"""Revise the ForgetCheck master reference from v1.0 to v2.0.

Applies the changes recommended in docs/FORGETCHECK_REVIEW.md. Formatting-preserving:
edits run text in place, clones existing rows/paragraphs for new content.

Guarded: refuses to run if the document is not at Version 1.0.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))  # docx_helpers lives beside this file
import docx
from docx_helpers import (blocks, set_text, cell_text, row_values, append_row,
                          insert_para_after, insert_paras_after, find_para,
                          find_table_by_header, find_callout, set_callout)

SRC = 'ForgetCheck_Master_Project_Reference.docx'
d = docx.Document(SRC)

T = lambda h: find_table_by_header(d, h)
P = lambda needle: blocks(d)[find_para(d, needle)][1]

# ---------------------------------------------------------------- guard
meta = T('Document Field')
assert meta.rows[2].cells[1].text.strip() == '1.0', \
    f"expected Version 1.0, found {meta.rows[2].cells[1].text!r} — already revised?"
log = []
def done(what): log.append(what); print('  ok  ' + what)

# ================================================================ 1. METADATA
row_values(meta, 2, [None, '2.0'])
row_values(meta, 3, [None, '7 August 2026 (v1.0); repositioned 25 August 2026 (v2.0) — see Changelog'])
row_values(meta, 6, [None, 'Fine-tuning, NegGrad+, NegGrad (destructive control), SCRUB, SalUn, L1-sparse'])
row_values(meta, 7, [None, 'Behavioral, Privacy (two attack strengths), Representation, Relearning / Reversibility'])
row_values(meta, 8, [None,
    'Reversibility-inclusive, difficulty-stratified, oracle-calibrated cross-audit analysis'])
append_row(meta, ['Problem formulation',
    'Untraining (instance-level influence removal), per Triantafillou et al., 2026 [19]'])
done('metadata table -> v2.0')

# ================================================================ 2. CHANGELOG
# anchor on the "Document Contents" heading; the discipline rule above it lives in a table
anchor = P('Document Contents')
insert_paras_after(anchor, [
 ('Changelog: v1.0 \u2192 v2.0 (25 August 2026)', 'Heading 1'),
 ('Version 2.0 follows an independent review of the frozen v1.0 position against literature '
  'published between February and August 2026. The review is archived in the project repository '
  'at docs/FORGETCHECK_REVIEW.md, with the full evidence trail and reasoning in '
  'docs/RESEARCH_LOG.md. Nothing in the project\u2019s scientific discipline changed; the research '
  'claim, four experimental decisions and the statistical backbone did.', None),
 ('Why the claim moved. Three papers published after the freeze date establish parts of what '
  'v1.0 proposed to discover. Metric Unreliability in Multimodal Machine Unlearning [20] performs '
  'cross-metric Kendall-tau disagreement analysis against a retrained oracle across five metric '
  'families, including a CIFAR-10 / ResNet-18 arm. RULER [22] and Are We Truly Forgetting? [23] '
  'independently establish former hypothesis H1. Section 6 is therefore narrowed to the parts '
  'that remain open: reversibility as a ranked audit family, disagreement within the privacy '
  'family, forget-set difficulty as a factor, and calibration of audits against an oracle '
  'ensemble and a ground-truth canary condition.', None),
 ('Substantive changes in v2.0:', None),
 ('Section 1 and 3 \u2014 the project is now explicitly framed as an Untraining study in the sense '
  'of [19], which fixes retraining-from-scratch as the correct reference and makes membership '
  'inference an appropriate metric for this setting.', 'List Paragraph'),
 ('Section 6 \u2014 novelty statement replaced. Section 8 \u2014 H1 demoted to a replication target, '
  'H2 split into inter-family and intra-family forms, H5 added for audit validity.', 'List Paragraph'),
 ('Section 12 \u2014 Selective Synaptic Dampening removed from the core method set. Its Fisher-'
  'importance mechanism cannot separate a random i.i.d. forget set from the retain set, and it is '
  'reported to fail at all forget fractions under random forgetting [24]. Replaced by SalUn and '
  'L1-sparse; NegGrad+ replaces plain NegGrad as the primary gradient-ascent method, with plain '
  'NegGrad retained as a labelled destructive control. Six methods total.', 'List Paragraph'),
 ('Section 13 \u2014 forget-set design split into two orthogonal factors: a deletion-size axis '
  '(demoted to a scaling check) and a memorization-difficulty axis (promoted to primary), plus a '
  'mislabeled-canary condition providing ground truth by construction.', 'List Paragraph'),
 ('Section 14.4 \u2014 two membership-inference attacks of different strength, not one. '
  'Section 14.5 \u2014 a second, non-CKA representation measure and an oracle-vs-oracle baseline. '
  'Section 14.7 \u2014 anchor arms added to the relearning protocol.', 'List Paragraph'),
 ('Section 15 \u2014 new subsection 15.5 on audit calibration and validity. Section 16 \u2014 the '
  'primary agreement statistic changes: correlation is computed over model instances rather than '
  'over four method ranks, because Kendall\u2019s tau over four items cannot attain p < 0.05 under '
  'any possible data (minimum two-sided p = 0.0833).', 'List Paragraph'),
 ('Section 17, 19, 21 \u2014 compute plan corrected. Training moves to Kaggle GPU; the local '
  'Ryzen AI 9 HX 370 machine cannot train PyTorch models on its Radeon 890M iGPU (unsupported by '
  'ROCm) and is measured at 4.76 hours per 30-epoch CIFAR-10 run on CPU. Gantt rebalanced.', 'List Paragraph'),
 ('The optional LLM / TOFU extension is withdrawn to fund the calibration and canary work.', 'List Paragraph'),
])
done('changelog inserted')

# ================================================================ 3. SECTION 1
s1 = T('Field')
row_values(s1, 1, [None, 'ForgetCheck: A Retraining-Grounded Study of Audit Disagreement, '
                         'Reversibility and Audit Validity in Approximate Machine Unlearning'])
row_values(s1, 2, [None, 'ForgetCheck: Do Behavioral, Privacy, Representation and Relearning '
                         'Audits Agree \u2014 and Which of Them Can Be Trusted?'])
row_values(s1, 5, [None, 'When audit families disagree about whether approximate unlearning '
                         'succeeded, which audit is right \u2014 and does the answer depend on how '
                         'hard the deletion request is?'])
row_values(s1, 6, [None, 'An ensemble of models retrained from scratch without the forget set, '
                         'treated as a reference distribution rather than a single model'])
append_row(s1, ['Problem formulation',
                'Untraining [19] \u2014 removal of the influence of specific forget-set examples, '
                'not removal of an underlying concept'])
done('section 1 identity table')

set_callout(d, 'Project in one sentence',
 'We will apply six existing machine-unlearning algorithms to forget sets of controlled '
 'difficulty, audit each resulting model behaviorally, for membership privacy under two attack '
 'strengths, at the representation level and under controlled relearning, and determine not only '
 'whether these audits agree but which of them can be trusted — calibrating each against an '
 'ensemble of independently retrained models and against a canary condition where the correct '
 'answer is known by construction.')
done('one-sentence definition')

# ================================================================ 4. SECTION 5 (lit table)
lit = T('Existing Research')
for r in [
 ['Is your algorithm unlearning or untraining? [19]',
  'Separates Untraining (remove the influence of specific examples; retraining is the correct '
  'oracle) from Unlearning (remove an underlying concept; retraining is not).',
  'Supplies the vocabulary this project works in. ForgetCheck is an Untraining study, which is '
  'what licenses both the retrained oracle and the use of membership inference.'],
 ['Metric Unreliability in Multimodal Unlearning [20]',
  'Cross-metric Kendall-tau disagreement over 36 unlearned models against a retained-only oracle; '
  'reports two opposing metric clusters and a CIFAR-10 / ResNet-18 arm.',
  'The nearest neighbour to this project and the reason Section 6 is narrowed. It explicitly '
  'excludes knowledge recoverability as unscalable \u2014 which is the axis ForgetCheck supplies.'],
 ['Inexact Unlearning Needs More Careful Evaluations [21]',
  'Population membership-inference attacks overestimate privacy protection; per-example attacks '
  'are far stronger and can show risk increasing after unlearning.',
  'Motivates running two attacks of different strength and treating the gap between them as an '
  'intra-family disagreement result rather than a caveat.'],
 ['RULER [22]; Are We Truly Forgetting? [23]',
  'Methods that pass every output-level check still show significant representation-level '
  'residuals; representation quality can collapse or only the classifier can change.',
  'Establishes former hypothesis H1. ForgetCheck treats it as a replication target and moves its '
  'novelty to reversibility, difficulty and audit validity.'],
 ['Comparative study of unlearning techniques [24]',
  'Reports that Selective Synaptic Dampening fails to forget at all forget fractions under random '
  'forgetting, attributed to similar Fisher information values.',
  'Directly motivates removing SSD from the core method set for instance-level deletion.'],
 ['Easy Data Unlearning Bench [25]',
  'Evaluates against an ensemble of oracle models rather than a single retrained model; releases '
  '200 oracles per forget set for CIFAR-10.',
  'Motivates treating the oracle as a reference distribution and expressing thresholds in '
  'oracle-standard-deviation units.'],
 ['Reliability of CKA [26]',
  'CKA values can change substantially without corresponding change in functional behavior; '
  'sensitive to outliers and to separability-preserving transformations.',
  'Motivates pairing CKA with a mechanistically different similarity measure before any '
  'representation-level claim is made.'],
]:
    append_row(lit, r)
done(f'literature table +{7} rows')

# ================================================================ 5. SECTION 6
set_text(P('The novelty claim must be narrow and accurate'),
 'The novelty claim must be narrow and accurate, and it was narrowed in v2.0. This project is not '
 'the first unlearning benchmark, not the first to use membership inference, not the first to '
 'compare against retraining, not the first to study representations, not the first to test '
 'relearning, and \u2014 since [20] \u2014 not the first to quantify disagreement between audit '
 'families. What remains open is narrower and sharper: existing disagreement studies compare '
 'output-, privacy- and alignment-based metrics only, quantify disagreement without adjudicating '
 'it, and hold the difficulty of the deletion request fixed.')

set_callout(d, 'Final novelty statement',
 'ForgetCheck extends cross-audit analysis of approximate Untraining in three directions and '
 'adds a validity layer. It introduces controlled relearning as a fourth ranked audit family — '
 'the axis prior work identified as missing but did not scale; it measures disagreement within '
 'the privacy family as well as between families, by contrasting a population membership-'
 'inference attack with a per-example one; and it treats forget-set memorization difficulty as an '
 'experimental factor rather than a fixed condition. Every audit is calibrated against an '
 'ensemble of independently retrained oracles and validated on a canary condition where residual '
 'influence is known by construction, so that audits can be scored for accuracy rather than only '
 'compared for agreement.')
done('section 6 novelty statement')

set_text(P('A second contribution is the explicit coupling'),
 'The contributions are therefore ordered as follows. First and primary: audit validity \u2014 '
 'determining which audit families give trustworthy verdicts, measured as false-positive rate '
 'against independently retrained oracles and as accuracy against the canary condition. Second: '
 'the reversibility axis, absent from existing disagreement studies. Third: the interaction '
 'between audit disagreement and forget-set difficulty, motivated by evidence that random '
 'deletion sets may be insufficiently discriminative [7] and that unlearning difficulty depends '
 'on the characteristics of the deleted samples [8]. Fourth and explicitly secondary: replication '
 'of the cross-audit disagreement finding of [20] in a clean unimodal, instance-level setting '
 'with a true from-scratch oracle ensemble. Replication is stated as replication.')
done('section 6 contribution ordering')

# ================================================================ 6. SECTION 8
rq = T('ID')  # first ID table is research questions
row_values(rq, 2, [None, 'Do behavioral metrics and membership-inference metrics reach the same '
                         'conclusions regarding successful unlearning \u2014 and do two membership-'
                         'inference attacks of different strength reach the same conclusion as '
                         'each other?'])
append_row(rq, ['RQ6', 'Which audit families produce correct verdicts, judged by their '
                       'false-positive rate against independently retrained oracles and by their '
                       'accuracy on a canary condition where residual influence is known by '
                       'construction?'])
done('research questions')

hyp = None
for k, o in blocks(d):
    if k == 'T' and o.rows[0].cells[0].text.strip() == 'ID' and 'Hypothesis' in o.rows[0].cells[1].text:
        hyp = o
assert hyp is not None
row_values(hyp, 1, [None,
 'Replication target, not a novel prediction. Approximate unlearning algorithms that perform well '
 'under output-level evaluation will remain distinguishable from full retraining under '
 'representation-level analysis. This has been established independently by [22] and [23]; '
 'ForgetCheck predicts the effect will be larger for high-memorization forget sets than for '
 'random ones.'])
row_values(hyp, 2, [None,
 'H2a (between families): membership-inference success and behavioral similarity will not always '
 'rank unlearning algorithms identically. H2b (within the privacy family): a population '
 'membership-inference attack and a per-example attack will not always rank them identically '
 'either, and the population attack will systematically rank methods as more private [21].'])
row_values(hyp, 4, [None,
 'High-memorization forget sets will reveal larger differences among unlearning algorithms, and '
 'greater disagreement among audit families, than equally sized low-memorization or randomly '
 'selected forget sets. Low-memorization forget sets are expected to show little of either, and '
 'serve as a designed negative control.'])
append_row(hyp, ['H5',
 'Audit families will differ in validity, not merely in verdict. At least one audit family is '
 'expected to flag an independently retrained oracle as insufficiently forgotten, which would '
 'indicate a false-positive problem in that audit rather than a failure of retraining.'])
done('hypotheses H1/H2/H4 revised, H5 added')

# ================================================================ 7. SECTION 12 METHODS
m = T('Method')
row_values(m, 2, ['NegGrad+ (gradient ascent with retain regularization)',
 'Ascend the loss on the forget set while simultaneously descending it on the retain set.',
 'The standard strong gradient-ascent baseline in current vision unlearning work, and a '
 'consistent top performer on CIFAR-10 / ResNet-18. Primary representative of its family.'])
row_values(m, 4, ['NegGrad (plain gradient ascent) \u2014 destructive control',
 'Ascend the loss on the forget set only, with no retain-set regularization.',
 'Deliberately included as a control, not a competitor. It is expected to degrade forget-set '
 'performance while damaging the model generally, and is the cleanest demonstration that "looks '
 'forgotten" can mean "damaged" \u2014 the failure mode Audit Layer 1 exists to detect.'])
append_row(m, ['SalUn (saliency-based unlearning)',
 'Compute a weight-saliency mask from forget-set gradients and apply the unlearning update only '
 'through the masked parameters.',
 'Mechanistically distinct from both fine-tuning and gradient ascent, works at instance level, '
 'and is a standard baseline in the RUM and vision-transformer benchmarks.'])
append_row(m, ['L1-sparse unlearning',
 'Fine-tune on the retain set under an L1 penalty that sparsifies the network during unlearning.',
 'Represents the model-sparsity route to unlearning [5]. Cheap, well-established at instance '
 'level, and brings the ranked method count to six \u2014 which the agreement statistics require.'])
done('methods table: 6 methods, SSD removed')

set_text(P('Optional fifth method: the Incompetent Teacher'),
 'Selective Synaptic Dampening [6] was a core method in v1.0 and has been withdrawn from the core '
 'set. Its mechanism selects parameters whose Fisher importance is disproportionately high for '
 'the forget set relative to the retain set. A randomly drawn forget set is, by construction, '
 'distributed identically to the retain set, so that importance ratio is approximately uniform, '
 'no parameters are selected, and the dampening step approximates a no-op. This is a mismatch '
 'between mechanism and task rather than a tuning problem, and it is reported empirically: SSD '
 'fails to forget at all tested forget fractions under random forgetting while performing at or '
 'above the state of the art on class and sub-class unlearning [24]. SSD may be reported in an '
 'appendix as a documented mechanism-task mismatch, or evaluated in a separate class-unlearning '
 'condition where it is the appropriate tool. The Incompetent Teacher approach [3] remains an '
 'optional seventh method, to be added only after all six core methods and all audit modules are '
 'stable.')
done('SSD withdrawal note')

# ================================================================ 8. SECTION 13 FORGET SETS
f = T('Condition')
row_values(f, 1, ['Size axis \u2014 Random 1%',
 'Randomly select 1% of training examples for F.',
 'Deletion-size scaling check only. Expected to be low-discriminability: a random CIFAR-10 subset '
 'is overwhelmingly low-memorization, and for such examples the retrained model behaves almost '
 'identically to the original [19].'])
row_values(f, 2, ['Size axis \u2014 Random 5%',
 'Randomly select 5% of training examples for F.',
 'Scaling check and the point of comparison with the difficulty axis at matched size.'])
row_values(f, 3, ['Size axis \u2014 Random 10%',
 'Randomly select 10% of training examples for F.',
 'Scaling check. Note that the gap between influential and random forget sets narrows as forget-'
 'set size grows [7], so this is expected to be the least discriminative size, not the most.'])
row_values(f, 4, ['Difficulty axis \u2014 High memorization (fixed size)',
 'Select the highest-memorization examples at a fixed forget-set size, using the released '
 'CIFAR-10 memorization scores [8].',
 'Primary condition. Hard deletion requests are where original and retrained models genuinely '
 'differ, and therefore where audits have something to disagree about.'])
append_row(f, ['Difficulty axis \u2014 Medium memorization (fixed size)',
 'Select middle-stratum examples at the same fixed size.',
 'Interpolates the difficulty axis so that any trend in disagreement can be observed rather than '
 'inferred from two endpoints.'])
append_row(f, ['Difficulty axis \u2014 Low memorization (fixed size) \u2014 negative control',
 'Select the lowest-memorization examples at the same fixed size.',
 'Designed negative control. Theory predicts almost no difference between the original and '
 'retrained models here [19], so a correctly functioning audit battery should report no detectable '
 'signal. An audit that reports strong forgetting evidence in this condition is misbehaving.'])
append_row(f, ['Canary condition \u2014 ground truth by construction',
 'Inject a small set of deliberately mislabeled canary examples into training; the forget set is '
 'exactly those canaries.',
 'Canaries can only be fitted by memorization, so a model retrained without them provably carries '
 'no canary-label association. Any residual canary behavior in an unlearned model is therefore '
 'unambiguously residual influence, which allows each audit to be scored for accuracy against a '
 'known answer rather than only compared with other audits.'])
done('forget-set design: two axes + canary')

set_text(P('Optional structured/user-level deletion can be added later'),
 'The size axis and the difficulty axis are separate experimental factors and must not be '
 'collapsed back into a single list of conditions. The difficulty axis is primary because it is '
 'tied directly to the research gap; the size axis answers the scalability question and should '
 'not carry primary conclusions. Optional structured or user-level deletion may be added later, '
 'but must not displace either axis. If the schedule slips, the order of sacrifice is: secondary '
 'dataset first, then the 1% and 10% size conditions, then the canary condition \u2014 never the '
 'difficulty axis and never the oracle ensemble.')
done('forget-set axis note')

d.save(SRC)
print('\nSTAGE 1 COMPLETE:', len(log), 'edits')
