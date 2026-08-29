"""Stage 3: compute plan, risks, deliverables, Gantt, abstract, references, appendices."""
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
log = []
def done(w): log.append(w); print('  ok  ' + w)

# ============================================== 17.1 TECHNOLOGY STACK
st = T('Area')
row_values(st, 4, [None, 'Structured CSV/JSON records written per run to persistent storage; '
                         'optional MLflow or Weights & Biases'])
row_values(st, 5, [None, 'RUM codebase as the primary experimental substrate (CIFAR-10, '
                         'ResNet-18, instance-level forget sets, released memorization scores, '
                         'six of the six core methods, MIA scripts); SCRUB and torchunlearn as '
                         'independent cross-checks; NeurIPS starter kit for the population attack'])
append_row(st, ['Training compute', 'Kaggle notebooks (P100 / T4, 30 GPU-hours per week per '
                                    'account, four accounts) as primary; Colab free tier as '
                                    'overflow'])
append_row(st, ['Local compute', 'Audits, layer-wise CKA, statistics, figures and the dashboard. '
                                 'Note: the team\u2019s Ryzen AI 9 HX 370 machine cannot train '
                                 'PyTorch models on its Radeon 890M integrated GPU, which is '
                                 'unsupported by ROCm; local training is CPU-only and measured at '
                                 '4.76 hours per 30-epoch CIFAR-10 run.'])
done('17.1 technology stack')

insert_paras_after(P('17.1 Technology Stack'), [
 ('Compute is not a constraint on this project, but session persistence is. Training runs on '
  'ephemeral cloud notebook sessions, so the pipeline must be built around checkpoint artefacts '
  'rather than long-lived sessions: every training run writes its checkpoint and a structured '
  'result record to persistent storage, and every audit reads from checkpoints rather than from a '
  'model held in memory. The logging requirements of Section 18 are therefore load-bearing '
  'infrastructure and must be built in week 3, not retrofitted later.', None),
])
done('17.1 session-persistence note')

# ============================================== 19 RISKS
rk = T('Risk')
row_values(rk, 1, ['Compute platform mismatch',
 'The team\u2019s most powerful local machine has no PyTorch-capable GPU: ROCm does not support the '
 'Radeon 890M integrated GPU, and the NPU is not exposed for training. Measured CPU throughput is '
 '4.76 hours per 30-epoch CIFAR-10 run, roughly forty to fifty times slower than a free cloud GPU.',
 'Train exclusively on Kaggle and Colab GPU sessions; use local machines for audits, statistics '
 'and the dashboard, where 32 GB of RAM is a genuine advantage. Four Kaggle accounts supply about '
 '120 GPU-hours per week against a total budget of roughly 11 to 26 hours.'])
row_values(rk, 3, ['MIA choice changes the ranking',
 'Population and per-example attacks disagree, and population attacks systematically overestimate '
 'privacy protection [21]. Reporting only one would produce a misleading method ranking, not '
 'merely an understated leakage number.',
 'Run both attack strengths and report the gap as an intra-family disagreement result. Read every '
 'privacy verdict together with the retained-utility guard, since per-example scores can be gamed '
 'by utility-destroying strategies [25].'])
row_values(rk, 4, ['Representation metrics are ambiguous',
 'High CKA similarity does not equal retained private data, and CKA values can change without a '
 'corresponding change in functional behavior [26].',
 'Require agreement between linear CKA and a second, mechanistically different measure before any '
 'representation claim; interpret only against the oracle-vs-oracle baseline.'])
row_values(rk, 5, ['Forget set is too easy',
 'For low-memorization examples the ideal untraining solution is close to no change from the '
 'original model [19], so much of a randomly drawn forget set carries no detectable signal and '
 'normalized oracle-gap denominators collapse.',
 'Make memorization difficulty an explicit experimental axis; keep the low-memorization stratum '
 'as a designed negative control rather than a discarded cell; retain the low-discriminability '
 'flag of Section 15.4.'])
row_values(rk, 8, ['Novelty overclaim',
 'Work published after the v1.0 freeze already performs cross-audit disagreement analysis against '
 'a retrained oracle, including on CIFAR-10 with ResNet-18 [20], and independently establishes '
 'former hypothesis H1 [22], [23].',
 'Use the narrowed novelty statement of Section 6, cite the overlapping work prominently, and '
 'state the replication component as replication. Re-run the novelty check before submission; the '
 'procedure is recorded in docs/RESEARCH_LOG.md.'])
append_row(rk, ['Underpowered agreement statistics',
 'Rank correlation over a small set of methods cannot reach significance under any possible data; '
 'at four methods the minimum attainable two-sided p-value is 0.0833.',
 'Correlate metric values across model instances rather than ranks across methods; carry six '
 'ranked methods; use mixed-effects models for confirmatory tests and report bootstrap intervals.'])
append_row(rk, ['Schema drift across the team',
 'Four members writing four audit modules can produce four incompatible result formats, which '
 'surfaces only at the analysis stage when there is no time to fix it.',
 'Freeze the result-record schema in week 3, before any audit module is written. One row per '
 '(model, condition, seed) with identical keys across all audits.'])
done('19 risks updated and extended')

# ============================================== 21 GANTT
g = T('Week')
plan = [
 ('1', 'Reposition claim and hypotheses against post-freeze literature; rewrite Sections 6 and 8',
       'Revised master reference v2.0; changelog'),
 ('2', 'Synopsis and experiment specification; freeze RQs, metrics and metric directions',
       'Approved synopsis; frozen specification'),
 ('3', 'Adopt RUM codebase; reproduce its reported baseline; M0 pipeline; download memorization '
       'scores; freeze result-record and checkpoint schema',
       'Reproducible M0 pipeline; frozen schema'),
 ('4', 'Forget-set generator: size axis, memorization strata, canary condition; first oracles',
       'All forget conditions; first Mr'),
 ('5', 'All six unlearning methods running end to end at one seed',
       'Six validated unlearning pipelines'),
 ('6', 'Full-pipeline pilot: one seed, one condition, all four audits, calibration checks and '
       'disagreement analysis run through to a figure',
       'Pilot result figure; design faults surfaced'),
 ('7', 'Oracle ensemble; oracle-vs-oracle null bands; oracle false-positive-rate calibration',
       'Calibrated thresholds in oracle-SD units'),
 ('8', 'Behavioral, forget-set and output-similarity audits',
       'Utility and output-similarity results'),
 ('9', 'Membership inference, population attack',
       'Attack A results and checks'),
 ('10', 'Membership inference, per-example attack (RMIA); intra-family gap analysis',
        'Attack B results; A-vs-B disagreement'),
 ('11', 'Representation extraction; layer-wise CKA plus second measure; oracle baseline',
        'Representation audit results'),
 ('12', 'Relearning experiment with anchor arms and normalised recovery',
        'Recovery curves; T80 and AUC'),
 ('13', 'Full multi-seed matrix across both forget-set axes',
        'Primary experiment completion'),
 ('14', 'Mixed-effects models; instance-level correlations; disagreement matrix; canary scoring',
        'Agreement statistics; audit validity table'),
 ('15', 'Dashboard and report draft', 'Demo and report draft'),
 ('16', 'IEEE paper and presentation', 'Final paper and defense materials'),
]
for i, (wk, work, out) in enumerate(plan, start=1):
    row_values(g, i, [wk, work, out])
done('21 gantt rebalanced')

insert_paras_after(P('21. Semester Plan / Gantt'), [
 ('The schedule assumes four contributors. Ownership: member A holds the experimental substrate '
  '(weeks 3 to 7) and is on the critical path alone until the oracle ensemble exists; member B '
  'holds the behavioral and privacy audits; member C holds the representation and relearning '
  'audits; member D holds calibration, statistics, figures, the dashboard and paper assembly, '
  'starting in week 7 rather than at the end. Member A owns the training queue even though all '
  'four accounts contribute compute, so that checkpoint provenance is never ambiguous.', None),
 ('If the schedule slips, the order of sacrifice is: secondary dataset, then the 1% and 10% size '
  'conditions, then the canary condition. The oracle ensemble and the relearning audit are never '
  'sacrificed, because they carry the contribution.', None),
])
done('21 ownership and slip order')

# ============================================== 20.1 DELIVERABLES
insert_paras_after(P('Cross-audit agreement/disagreement analyzer'), [
 ('Audit calibration module: oracle false-positive rates and canary-condition scoring per audit '
  'family.', 'List Paragraph'),
])
done('20.1 deliverables')

# ============================================== 22.1 ABSTRACT
set_text(P('Machine unlearning aims to remove the influence of selected training data from an '
           'already trained machine-learning model without incurring'),
 'Machine unlearning aims to remove the influence of selected training data from an already '
 'trained model without incurring the computational cost of complete retraining. Although many '
 'approximate techniques have been proposed, determining whether a model has genuinely forgotten '
 'the requested data remains difficult, and recent work has shown that different evaluation '
 'metrics can rank the same methods differently. Establishing that such disagreement exists, '
 'however, does not establish which evaluation to believe. This project presents ForgetCheck, a '
 'retraining-grounded study of audit disagreement, reversibility and audit validity in '
 'approximate machine unlearning, in the Untraining setting where the influence of specific '
 'training examples is to be removed. Six unlearning methods are evaluated on CIFAR-10 image '
 'classifiers under four audit families: behavioral, membership privacy at two attack strengths, '
 'representation-level, and controlled relearning. Forget-set memorization difficulty is treated '
 'as an experimental factor rather than held fixed, and an ensemble of independently retrained '
 'models provides a reference distribution rather than a single reference model. Beyond measuring '
 'agreement, each audit is calibrated for validity: its false-positive rate is measured against '
 'held-out retrained oracles, and its accuracy is measured on a canary condition in which '
 'residual influence is known by construction. The study aims to determine not only when '
 'conventional measures of forgetting disagree, but which of them can be trusted, and whether '
 'that answer depends on how hard the deletion request is.')
done('22.1 abstract rewritten')

# ============================================== 22.3 CLAIMS TABLE
cl = T('Do Not Say')
append_row(cl, ['"We are the first to study whether unlearning audits agree."',
                '"Cross-audit disagreement has been reported in recent work [20]; we extend the '
                'analysis with a reversibility axis, an intra-family privacy comparison, a '
                'difficulty factor, and a calibration procedure that scores audits for validity '
                'rather than only comparing them."'])
append_row(cl, ['"Audit X is better than audit Y."',
                '"Under this protocol, audit X showed a lower false-positive rate against '
                'held-out retrained oracles and higher accuracy on the canary condition than '
                'audit Y. This is evidence about these audits in this setting, not a general '
                'ordering."'])
append_row(cl, ['"SSD performs poorly at unlearning."',
                '"SSD is designed for class and sub-class forgetting; its parameter-importance '
                'mechanism is not applicable to randomly drawn instance-level forget sets, which '
                'is a mismatch between mechanism and task rather than a quality judgement."'])
done('22.3 claims table extended')

# ============================================== 24 REFERENCES
last_ref = P('[18] T. T. Nguyen')
refs = [
 '[19] E. Triantafillou, A. I. Humayun, M. Ribero, A. M. Turner, M. C. Mozer, and G. Kaissis, '
 '\u201cIs your algorithm unlearning or untraining?,\u201d arXiv:2604.07962, 2026. Preprint.',
 '[20] A. A. Khan, H. Laga, and F. Sohel, \u201cMetric Unreliability in Multimodal Machine '
 'Unlearning: A Systematic Analysis and Principled Unified Score,\u201d arXiv:2605.02206, 2026. '
 'Preprint.',
 '[21] J. Hayes, I. Shumailov, E. Triantafillou, A. Khalifa, and N. Papernot, \u201cInexact '
 'Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy,\u201d '
 'arXiv:2403.01218, 2024.',
 '[22] G. Cosma and A. Finke, \u201cRULER: Representation-Level Verification of Machine '
 'Unlearning,\u201d in 2nd Workshop on Machine Unlearning and Privacy Preservation (WIPE-OUT), '
 'ECML PKDD, 2026. To appear in Springer CCIS.',
 '[23] \u201cAre We Truly Forgetting? A Critical Re-examination of Machine Unlearning Evaluation '
 'Protocols,\u201d arXiv:2503.06991, 2025.',
 '[24] \u201cA Comparative Study of Machine Unlearning Techniques for Image and Text '
 'Classification Models,\u201d arXiv:2412.19583, 2024.',
 '[25] \u201cEasy Data Unlearning Bench,\u201d arXiv:2602.16400, 2026. Preprint.',
 '[26] M. R. Davari, S. Horoi, A. Natik, G. Lajoie, G. Wolf, and E. Belilovsky, \u201cReliability '
 'of CKA as a Similarity Measure in Deep Learning,\u201d in International Conference on Learning '
 'Representations (ICLR), 2023.',
 '[27] S. Zarifzadeh, P. Liu, and R. Shokri, \u201cLow-Cost High-Power Membership Inference '
 'Attacks,\u201d in Proc. 41st Int. Conf. Machine Learning (ICML), PMLR, vol. 235, pp. '
 '58244\u201358282, 2024.',
 '[28] \u201cUnlearning as Distribution Restoration: A Controlled Counterfactual Study, a '
 'Validated Selective Screen, and the Limits of Oracle-Free Certification,\u201d arXiv:2607.19442, '
 '2026. Preprint.',
 '[29] \u201cBenchmarking Unlearning for Vision Transformers,\u201d arXiv:2602.20114, 2026. '
 'Preprint.',
 '[30] E. Triantafillou et al., \u201cAre we making progress in unlearning? Findings from the '
 'first NeurIPS unlearning competition,\u201d arXiv:2406.09073, 2024.',
 '[31] K. Zhao, M. Kurmanji, G.-O. Barbulescu, E. Triantafillou, and P. Triantafillou, '
 '\u201cScalability of memorization-based machine unlearning,\u201d in NeurIPS 2024 FITML '
 'Workshop, 2024.',
 '[32] \u201cOn the importance of multiple training seeds for evaluating machine unlearning,\u201d '
 'arXiv:2510.26714, 2025.',
 '[33] Y. Tu, P. Hu, and J. W. Ma, \u201cA Reliable Cryptographic Framework for Empirical Machine '
 'Unlearning Evaluation,\u201d arXiv:2404.11577, 2024.',
 '[34] \u201cUnlearning Comparator: A Visual Analytics System for Comparative Evaluation of '
 'Machine Unlearning Methods,\u201d arXiv:2508.12730, 2025.',
]
insert_paras_after(last_ref, [(r, None) for r in refs])
done(f'24 references +{len(refs)}')

# ============================================== 25 CODE RESOURCES
cr = T('Resource')
row_values(cr, 3, ['RUM implementation (primary substrate)',
 'Adopted as the experimental base: CIFAR-10 with ResNet-18, instance-level forget sets, released '
 'memorization scores, and implementations of retrain, fine-tune, L1-sparse, NegGrad variants, '
 'SCRUB, influence and SalUn, plus MIA analysis scripts.',
 'github.com/kairanzhao/RUM'])
row_values(cr, 5, ['torchunlearn (cross-check)',
 'Independent MIT-licensed implementation of many of the same methods on CIFAR-10 with ResNet-18. '
 'Used to confirm that method results are not artefacts of a single codebase.',
 'github.com/Harry24k/machine-unlearning-pytorch'])
append_row(cr, ['EasyDUB checkpoints [25]',
 'Released ensembles of oracle models for CIFAR-10, used to sanity-check the expected magnitude '
 'of oracle-to-oracle variation before generating our own ensemble. Different architecture, so a '
 'calibration reference rather than a substitute.',
 'Hugging Face'])
append_row(cr, ['Selective Synaptic Dampening (optional)',
 'Official SSD implementation. Required only if the optional class-unlearning side condition of '
 'Section 12 is run.', 'github.com/if-loops/selective-synaptic-dampening'])
done('25 code resources')

# ============================================== APPENDIX A
ap = T('Decision')
row_values(ap, 1, [None, 'ForgetCheck: A Retraining-Grounded Study of Audit Disagreement, '
                         'Reversibility and Audit Validity in Approximate Machine Unlearning'])
row_values(ap, 2, [None, 'Audit validity and cross-audit disagreement under oracle-ensemble '
                         'grounding, with reversibility and forget-set difficulty as factors'])
row_values(ap, 6, [None, 'Fine-tune, NegGrad+, NegGrad (destructive control), SCRUB, SalUn, '
                         'L1-sparse'])
row_values(ap, 7, [None, 'Incompetent Teacher \u2014 optional seventh. SSD withdrawn from the core '
                         'set (mechanism-task mismatch at instance level); optional class-'
                         'unlearning side condition only'])
row_values(ap, 8, [None, 'Size axis: random 1/5/10%. Difficulty axis at fixed size: low '
                         '(negative control), medium, high memorization. Canary condition.'])
row_values(ap, 9, [None, 'Utility/behavior, MIA privacy at two attack strengths, CKA plus a '
                         'second representation measure, relearning with anchor arms, and audit '
                         'validity calibration'])
row_values(ap, 10, [None, 'Mean/SD across at least 3 seeds; instance-level Kendall tau and '
                          'Spearman with bootstrap CIs; linear mixed-effects models; disagreement '
                          'rate. Rank correlation over methods is descriptive only.'])
row_values(ap, 11, [None, 'Original model plus an ensemble of at least five independently '
                          'retrained oracles, with at least one held out as a validity probe'])
row_values(ap, 13, [None, 'Withdrawn \u2014 compute and effort reallocated to calibration and the '
                          'canary condition'])
append_row(ap, ['Problem formulation', 'Untraining [19], not concept-level Unlearning'])
append_row(ap, ['Training compute', 'Kaggle GPU (four accounts); local machines are CPU-only and '
                                    'are used for audits and analysis, not training'])
append_row(ap, ['Team ownership', 'A substrate; B behavior and privacy; C representation and '
                                  'relearning; D calibration, statistics and delivery'])
done('appendix A decision sheet')

# ============================================== APPENDIX B
apb = T('Question')
append_row(apb, ['Does this audit ever flag a genuine retrained model?',
                 'Oracle false-positive rate over held-out retrained oracles',
                 'An audit that fails a true retrain is broken, not strict.'])
append_row(apb, ['Is this audit right, not just different?',
                 'Canary-condition accuracy, sensitivity and specificity',
                 'Only valid where residual influence is known by construction.'])
append_row(apb, ['Do two privacy attacks of different strength agree?',
                 'Population attack vs per-example attack (RMIA)',
                 'A population attack systematically overstates privacy [21].'])
done('appendix B cheat sheet')

set_callout(d, 'Project-wide final reference sentence',
 'ForgetCheck does not attempt to prove absolute erasure. It asks whether different empirical '
 'audits provide consistent evidence that an approximate unlearned model behaves like a model '
 'retrained without the forgotten data, which of those audits can be trusted when they disagree, '
 'and whether both answers remain stable as deletion requests become harder.')
done('closing statement')

d.save(SRC)
print('\nSTAGE 3 COMPLETE:', len(log), 'edits')
