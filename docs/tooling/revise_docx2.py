"""Stage 2: audit layers, statistics, compute plan, schedule, references, appendices."""
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

meta = T('Document Field')
assert meta.rows[2].cells[1].text.strip() == '2.0', 'run stage 1 first'
log = []
def done(w): log.append(w); print('  ok  ' + w)

EM = '\u2014'

# ============================================== 14.4 MEMBERSHIP INFERENCE
set_text(P('Membership inference asks whether an attacker can distinguish'),
 'Membership inference asks whether an attacker can distinguish former training members from '
 'non-members using model behavior. For forgotten samples, successful Untraining should make '
 'their membership characteristics resemble those of a model that never trained on them. '
 'Membership inference is a baseline audit rather than a novel contribution [4], [17]; what is '
 'not baseline is the choice of attack. Attacks divide into population attacks, which instantiate '
 'one attacker for all examples, and per-example attacks, which tailor a decision rule to each '
 'example. Population attacks systematically overestimate the privacy protection offered by '
 'unlearning, and per-example attacks can show privacy risk increasing after unlearning for a '
 'substantial fraction of examples [21]. Because this project ranks methods, a weak attack does '
 'not merely understate leakage \u2014 it produces a misleading ranking that propagates into every '
 'agreement statistic. Both attack strengths are therefore run, and the gap between them is '
 'reported as a result in its own right.')
insert_paras_after(P('Difference from retrained-oracle attack performance'), [
 ('Attack A (population): loss-, confidence- or entropy-thresholded attack of the kind shipped '
  'with the NeurIPS starter kit [17]. Included precisely because it is what the community '
  'commonly reports.', 'List Paragraph'),
 ('Attack B (per-example): RMIA [27], chosen over full U-LiRA because it retains strong '
  'per-example power with only a handful of reference models, where earlier per-example attacks '
  'degrade toward random guessing. This is what makes a defensible strong attack affordable.',
  'List Paragraph'),
 ('Intra-family disagreement: rank correlation and verdict disagreement between Attack A and '
  'Attack B, reported alongside the between-family comparisons of Section 15.', 'List Paragraph'),
])
set_callout(d, 'Important limitation',
 'A weak or failed membership-inference attack does not prove that the model erased the sample. '
 'It only shows that this attack, under this protocol, could not reliably exploit a membership '
 'signal \u2014 and a stronger attack may still succeed, which is why two attack strengths are run. '
 'Conversely, per-example attack scores can be gamed by degenerate strategies that flatten model '
 'outputs and destroy utility [25], so every privacy verdict must be read together with the '
 'retained-utility result from Audit Layer 1. Neither attack, alone or together, constitutes '
 'proof of deletion.')
done('14.4 membership inference: two attack strengths')

# ============================================== 14.5 REPRESENTATION
set_text(P('Internal activations are collected from selected ResNet stages'),
 'Internal activations are collected from selected ResNet stages for the same probe examples '
 'across M0, Mu and Mr. The purpose is not to assert that representation similarity equals '
 'privacy leakage. Rather, it asks whether an unlearned model that looks retraining-like at the '
 'output level still remains internally much closer to the original model than to retraining. '
 'This layer requires two safeguards. First, CKA cannot be the sole instrument: CKA values can '
 'change substantially without a corresponding change in a model\u2019s functional behavior, and '
 'CKA is demonstrably sensitive to outliers and to transformations that preserve linear '
 'separability [26]. A metric that moves independently of function is the wrong single instrument '
 'for a layer whose entire purpose is to detect what output metrics miss. Second, no '
 'representation number is interpretable without a scale, which the oracle ensemble of Section '
 '16.1 supplies directly.')
insert_paras_after(P('Probe-set separation'), [
 ('Second similarity measure (required): report at least one mechanistically different measure '
  'alongside linear CKA \u2014 RBF-CKA, orthogonal Procrustes distance, or distance correlation. A '
  'representation-level claim is made only where the two measures agree; where they disagree, '
  'that disagreement is itself reported.', 'List Paragraph'),
 ('Oracle-vs-oracle baseline (required): compute the same measures between pairs of '
  'independently retrained oracles. This establishes what "as similar as two retrainings" looks '
  'like numerically, so that a similarity value can be read as near-oracle or far-from-oracle '
  'rather than merely high or low.', 'List Paragraph'),
])
done('14.5 representation: second measure + oracle baseline')

# ============================================== 14.7 RELEARNING
set_text(P('Relearning tests whether apparently forgotten behavior can be reacquired'),
 'Relearning tests whether apparently forgotten behavior can be reacquired unusually quickly. '
 'Time-to-relearn is an established instrument in vision unlearning rather than a new metric; '
 'what this project contributes is elevating it to a fully ranked audit family and asking whether '
 'it agrees with the other three. Starting separately from Mu and Mr, models are fine-tuned under '
 'an identical controlled protocol using a small, specified reintroduction of forgotten '
 'information, with performance recorded at fixed steps, for example 0, 1, 5, 10, 25 and 50 '
 'updates. Two anchor arms are required, because a recovery curve is uninterpretable without a '
 'scale and because models entering relearning at different utility levels have different amounts '
 'of headroom \u2014 which would otherwise let "started closer" masquerade as "retained more '
 'structure".')
insert_paras_after(P('Utility check: Confirm that the relearning procedure'), [
 ('Upper anchor \u2014 M0: the model that never forgot, relearned under the identical protocol. '
  'Defines the fastest recovery physically available.', 'List Paragraph'),
 ('Lower anchor \u2014 a randomly initialised or from-scratch model with no prior exposure to the '
  'forget set. Defines recovery attributable to the relearning data alone.', 'List Paragraph'),
 ('Normalised recovery: report each model\u2019s recovery on a scale between the two anchors, not '
  'in raw units, and report starting utility either matched across arms or carried as a covariate '
  'in the mixed-effects models of Section 16.2.', 'List Paragraph'),
])
done('14.7 relearning: anchor arms + normalisation')

# ============================================== 15.2 RANK CORRELATION
set_text(P('Kendall\u2019s tau is recommended as the primary rank-agreement statistic'),
 'The unit of analysis is the model instance, not the method. Ranking six methods and correlating '
 'those ranks would be statistically inert: with n items there are n! orderings, and for small n '
 'the attainable p-values are severely restricted. With the four methods of v1.0 the minimum '
 'attainable two-sided p-value for Kendall\u2019s tau is 0.0833, meaning that two audits ranking '
 'all four methods identically could still not be declared to agree at the 0.05 level; at six '
 'methods it is 0.0028. Rank tables per setting are therefore retained as descriptive displays '
 'only. The inferential analysis instead correlates metric values across every (method x forget-'
 'condition x seed) model instance, giving on the order of a hundred observations rather than '
 'six, which is the approach taken by comparable published work [20]. Kendall\u2019s tau and '
 'Spearman correlation are both reported over that instance-level population, with bootstrap '
 'confidence intervals. Pairwise comparisons include behavior vs privacy, behavior vs '
 'representation, privacy vs relearning, representation vs relearning, and \u2014 within the '
 'privacy family \u2014 population attack vs per-example attack.')
done('15.2 rank correlation reworked')

# ============================================== 15.5 NEW: AUDIT CALIBRATION
tail = P('If the original and retrained models already have nearly identical') if False else None
anchor = find_callout(d, 'Low-discriminability safeguard')
# insert new subsection after the 15.4 callout table by anchoring on the next heading
h16 = P('16. Experimental Matrix and Statistical Methodology')
new = insert_paras_after(h16, [
 ('15.5 Audit Calibration and Validity', 'Heading 2'),
 ('Sections 15.1 to 15.4 measure whether audits agree. This section measures whether they are '
  'right, which is a different and stronger question. Two checks make it answerable, and both are '
  'run before any disagreement result is interpreted.', None),
 ('Oracle false-positive rate. Each independently retrained oracle held out from the reference '
  'set is passed through every audit as though it were a candidate unlearned model. A genuine '
  'from-scratch retrain is, by definition, perfectly untrained on the forget set, so any audit '
  'that returns a "not forgotten" verdict for it has a false-positive problem. The proportion of '
  'held-out oracles that each audit wrongly flags is reported per audit family as that audit\u2019s '
  'false-positive rate. This is not a formality: recent work found a proposed certification '
  'criterion that its own retraining reference passed in only one case out of forty-five [28].',
  None),
 ('Canary accuracy. In the canary condition of Section 13, residual influence is known by '
  'construction: a model retrained without the mislabeled canaries provably carries no canary-'
  'label association, so any residual canary behavior is unambiguously residual influence. Each '
  'audit\u2019s verdict can therefore be scored against a known answer, and reported as accuracy, '
  'sensitivity and specificity rather than merely as agreement with other audits.', None),
 ('Reporting rule. An audit family that shows a high oracle false-positive rate or poor canary '
  'accuracy is reported as unreliable in that setting, and its contribution to the disagreement '
  'analysis is interpreted accordingly. Disagreement between a validated audit and an unreliable '
  'one is not evidence that the question is hard; it is evidence about the unreliable audit.',
  None),
])
done('15.5 audit calibration and validity added')

# ============================================== 16.1 CORE MATRIX
f = T('Factor')
row_values(f, 3, [None, 'Fine-tuning; NegGrad+; NegGrad (destructive control); SCRUB; SalUn; '
                        'L1-sparse (six ranked methods)'])
row_values(f, 4, [None, 'Size axis: Random 1%, 5%, 10%. Difficulty axis at fixed size: low, '
                        'medium, high memorization. Canary condition.'])
row_values(f, 6, [None, 'Original model; an ensemble of at least five independently retrained '
                        'oracles at the primary setting, with at least one held out as an audit '
                        'validity probe'])
row_values(f, 7, [None, 'Behavior, MIA privacy at two attack strengths, representation, '
                        'relearning \u2014 plus audit validity (Section 15.5)'])
row_values(f, 8, [None, 'Instance-level metric correlation (Kendall tau and Spearman with '
                        'bootstrap CIs) plus disagreement rate; linear mixed-effects models for '
                        'confirmatory tests'])
append_row(f, ['Compute platform', 'Kaggle GPU (P100 / T4) for all training; local CPU machines '
                                   'for audits, statistics and the dashboard'])
done('16.1 core matrix')

set_text(P('The four methods \u00d7 four forget settings \u00d7 three seeds'),
 'Six methods across the size and difficulty axes and three seeds produce on the order of one '
 'hundred approximate-unlearning runs, in addition to original training, the retrained oracle '
 'ensemble, membership-inference reference models, relearning runs and their anchor arms. This is '
 'a larger matrix than v1.0, but a cheaper one in wall-clock terms, because the experimental '
 'substrate is adopted from an existing public codebase rather than rebuilt. A single-seed pilot '
 'must be run through the entire pipeline \u2014 including all four audits, the calibration checks '
 'of Section 15.5 and the disagreement analysis, not merely training \u2014 before the full matrix '
 'is launched, so that a design fault surfaces while it is still cheap to fix.')
done('16.1 matrix note')

# ============================================== 16.2 STATISTICAL REPORTING
insert_paras_after(P('For paired comparisons, ensure the same forget set and seed logic'), [
 ('Use linear mixed-effects models for confirmatory tests, of the form metric ~ method + '
  'forget-condition + (1 | seed), so that the repeated-measures structure of the design is '
  'modelled rather than ignored. This follows established practice in comparable '
  'representation-level unlearning work [22].', 'List Paragraph'),
 ('Define all pass/fail thresholds in units of the oracle ensemble\u2019s standard deviation for '
  'the metric in question, never as absolute values. A threshold that a genuine retrained model '
  'fails is a broken threshold, not a strict one.', 'List Paragraph'),
 ('Report the number of model instances entering each correlation, and never report a rank '
  'correlation over the six methods as though it carried inferential weight.', 'List Paragraph'),
])
done('16.2 statistical reporting')

# ============================================== 16.3 METRICS TABLE
mt = T('Dimension')
row_values(mt, 4, [None, 'MIA ROC-AUC and attack accuracy under both a population attack and a '
                         'per-example attack (RMIA); TPR at fixed low FPR; gap between the two '
                         'attack strengths'])
row_values(mt, 5, [None, 'Layer-wise linear CKA plus a second, non-CKA similarity measure; '
                         'activation distance; oracle-vs-oracle baseline; PCA visualization'])
row_values(mt, 6, [None, 'Recovery curve normalised between M0 and no-exposure anchors; T80; '
                         'area under the relearning curve'])
append_row(mt, ['Audit validity', 'Oracle false-positive rate per audit family; canary-condition '
                                  'accuracy, sensitivity and specificity'])
done('16.3 metrics table')

d.save(SRC)
print('\nSTAGE 2 COMPLETE:', len(log), 'edits')
