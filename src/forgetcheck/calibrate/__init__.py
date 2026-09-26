"""Stage 7 — calibration and audit validity.

Records in, verdicts and validity rates out. No GPU: every number here is computed from the
audit records Stage 6 and the Stage 7 audit pass wrote.
"""

from .bands import Band, clopper_pearson, coverage_for, flag, loo_flags, nominal_fpr, prediction_band
from .canary import canary_scores
from .native import NATIVE_RULES, native_flag, native_rule_for
from .report import CalibrationConfig, calibrate, load_audit_records, write_tables

__all__ = [
    "Band", "clopper_pearson", "coverage_for", "flag", "loo_flags", "nominal_fpr",
    "prediction_band", "canary_scores", "NATIVE_RULES", "native_flag", "native_rule_for",
    "CalibrationConfig", "calibrate", "load_audit_records", "write_tables",
]
