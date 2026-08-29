"""Stage 4: sweep out remaining v1.0 statements that now contradict v2.0."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))  # docx_helpers lives beside this file
import docx
from docx_helpers import (blocks, set_text, cell_text, cell_text_clean, row_values,
                          append_row, insert_paras_after, find_para,
                          find_table_by_header, set_callout)

SRC = 'ForgetCheck_Master_Project_Reference.docx'
d = docx.Document(SRC)
T = lambda h: find_table_by_header(d, h)
P = lambda needle: blocks(d)[find_para(d, needle)][1]
log = []
def done(w): log.append(w); print('  ok  ' + w)

# ---------- 2. EXECUTIVE SUMMARY
set_text(P('The core experiment uses CIFAR-10 and ResNet-18'),
 'The core experiment uses CIFAR-10 and ResNet-18. The original model is trained on the full '
 'training dataset. Forget sets are then selected along two axes: deletion size, and memorization '
 'difficulty at fixed size, together with a canary condition in which residual influence is known '
 'by construction. An ensemble of oracle models is retrained from scratch on the retained data '
 'only, providing a reference distribution rather than a single reference model, while six '
 'approximate unlearning methods are applied to the original model. Each resulting model is '
 'evaluated under a layered audit comprising retained utility, forget-set behavior, output-'
 'distribution similarity, membership privacy under two attack strengths, internal representation '
 'similarity and controlled relearning. Audits are then compared with one another, and separately '
 'calibrated for validity against held-out oracles and the canary condition, so that the study '
 'can report not only where audits disagree but which of them to believe.')
done('executive summary')

set_text(P('ForgetCheck therefore does not claim to invent machine unlearning'),
 'ForgetCheck does not claim to invent machine unlearning, membership inference, representation '
 'analysis or relearning diagnostics, and it does not claim to be the first to observe that '
 'evaluation metrics disagree \u2014 that has been reported in recent work [20]. Its contribution '
 'is to extend cross-audit analysis with a reversibility axis that prior studies identified as '
 'missing but could not scale, to compare attacks of different strength within the privacy family '
 'rather than treating privacy as a single verdict, to treat forget-set difficulty as an '
 'experimental factor, and to calibrate each audit for validity so that disagreement can be '
 'adjudicated instead of merely counted.')
done('executive summary positioning')

# ---------- 5. LITERATURE TABLE: SSD ROW
lit = T('Existing Research')
for i, r in enumerate(lit.rows):
    if r.cells[0].text.strip().startswith('SSD ['):
        row_values(lit, i, [None, None,
         'Provides a mechanism qualitatively different from SCRUB, but one that depends on the '
         'forget set having a distinct Fisher-importance signature. That condition fails for '
         'randomly drawn instance-level forget sets, so SSD is withdrawn from the core method set '
         'in v2.0 and retained only as an optional class-unlearning comparator. See Section 12.'])
        break
done('literature table SSD row')

# ---------- 9.1 IN SCOPE
set_text(P('Random deletion fractions and a high-memorization/high-influence deletion condition'),
 'Two orthogonal forget-set factors: a deletion-size axis and a memorization-difficulty axis at '
 'fixed size, plus a canary condition providing ground truth by construction.')
insert_paras_after(P('Multi-seed experiments, uncertainty reporting and rank/disagreement analysis'), [
 ('Calibration of each audit family for validity, using held-out retrained oracles and the canary '
  'condition, in addition to comparing audits with one another.', 'List Paragraph'),
 ('An ensemble of independently retrained models treated as a reference distribution, rather than '
  'a single retrained model treated as ground truth.', 'List Paragraph'),
])
done('9.1 in-scope')

# ---------- 14.8 STRESS CONDITIONS
set_text(P('Forget-difficulty stress: compare random 5% with high-memorization 5%'),
 'Forget-difficulty stress: compare low, medium and high memorization strata at fixed forget-set '
 'size, with the low stratum serving as a designed negative control.')
insert_paras_after(P('Forget-difficulty stress: compare low, medium and high memorization strata'), [
 ('Audit-validity stress: pass held-out retrained oracles and the canary condition through every '
  'audit, per Section 15.5.', 'List Paragraph'),
])
done('14.8 stress conditions')

# ---------- 16.3 METRICS TABLE leftovers
mt = T('Dimension')
for i, r in enumerate(mt.rows):
    k = r.cells[0].text.strip()
    if k == 'Audit agreement':
        row_values(mt, i, [None, 'Instance-level Kendall tau and Spearman with bootstrap '
                                 'confidence intervals; pairwise disagreement rate; '
                                 'between-family and within-family (attack A vs attack B)'])
    elif k == 'Difficulty':
        row_values(mt, i, [None, 'Low vs medium vs high memorization at fixed forget-set size; '
                                 'random forget set as an additional comparison point'])
done('16.3 leftover rows')

# ---------- 20.2 DASHBOARD
dash = T('Dashboard Component')
row_values(dash, 1, [None, 'Original / Retrained oracle / Fine-tune / NegGrad+ / NegGrad control '
                           '/ SCRUB / SalUn / L1-sparse'])
row_values(dash, 2, [None, 'Size axis (random 1 / 5 / 10%); difficulty axis (low / medium / high '
                           'memorization); canary condition'])
row_values(dash, 4, [None, 'MIA AUC and attack accuracy under both attack strengths, with the gap '
                           'between them'])
append_row(dash, ['Audit validity', 'Oracle false-positive rate and canary accuracy per audit '
                                    'family'])
done('20.2 dashboard')

# ---------- 23 LITERATURE MAP
lm = T('Research Stage')
for i, r in enumerate(lm.rows):
    if r.cells[0].text.strip() == 'Deep unlearning algorithms':
        row_values(lm, i, [None, 'Incompetent Teacher [3], SCRUB [4], sparsity [5], SSD [6], '
                                 'SalUn and NegGrad+ via RUM [8]',
                                 'Provides the six core methods and their mechanisms. SSD is '
                                 'retained as a documented mechanism-task mismatch rather than a '
                                 'core method [24].'])
for r in [
 ['Problem formulation', 'Triantafillou et al. [19]',
  'Fixes the Untraining framing, which licenses both the retrained oracle and membership '
  'inference as appropriate for this setting.'],
 ['Cross-audit disagreement', 'Khan et al. [20]',
  'The nearest prior work. Defines what is already known and therefore what Section 6 may claim.'],
 ['Attack strength within privacy', 'Hayes et al. [21], RMIA [27]',
  'Motivates two attack strengths and makes the stronger one affordable.'],
 ['Representation-level residuals', 'RULER [22], [23]',
  'Establishes former hypothesis H1, now treated as a replication target.'],
 ['Similarity-metric reliability', 'Davari et al. [26]',
  'Requires a second, non-CKA measure before any representation claim.'],
 ['Oracle as a distribution', 'EasyDUB [25], [28], [32]',
  'Motivates the oracle ensemble, oracle-SD thresholds and the validity calibration of 15.5.'],
]:
    append_row(lm, r)
done('23 literature map')

# ---------- 25 CODE RESOURCES: fix stale SSD row and hyperlinks
cr = T('Resource')
row_values(cr, 4, ['CIFAR-10 memorization scores',
 'Released with the RUM codebase and used to construct the difficulty axis. Fallback if '
 'unavailable: cheap proxies such as confidence or holdout retraining, which correlate strongly '
 'with exact memorization at a small fraction of the cost [31].', None])
cell_text_clean(cr.rows[4].cells[2], 'github.com/kairanzhao/RUM')
for i in (1, 2, 3, 5, 6, 7):
    txt = cr.rows[i].cells[2].text.replace('Open resource', '').strip()
    cell_text_clean(cr.rows[i].cells[2], txt if txt else 'Open resource')
done('25 code resources cleaned')

# ---------- 11.3 SYNTHETIC DATA POLICY -> canary
set_text(P('Synthetic data is academically acceptable when it serves a controlled research purpose'),
 'Synthetic and constructed data is academically acceptable when it serves a controlled research '
 'purpose. TOFU is an important precedent, deliberately constructing fictitious author profiles to '
 'make LLM unlearning ground truth precise [9]. v2.0 adopts the same principle in a form suited to '
 'vision: a small canary set of deliberately mislabeled training examples. Canaries cannot be '
 'fitted by generalisation, only by memorisation, so a model retrained without them provably '
 'carries no canary-label association. This makes the canary condition the only setting in the '
 'study where the correct audit verdict is known in advance, which is what allows audits to be '
 'scored for accuracy in Section 15.5 rather than only compared with one another. The optional '
 'user-level simulation described in v1.0 remains optional and secondary; the LLM extension is '
 'withdrawn.')
done('11.3 synthetic data / canary policy')

d.save(SRC)
print('\nSTAGE 4 COMPLETE:', len(log), 'edits')
