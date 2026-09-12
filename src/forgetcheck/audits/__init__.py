"""Stage 6 — the audit modules.

Six audits, each answering "does this model still carry the forget set's influence?" by a
different mechanism. The contribution is the pattern of agreement and disagreement between them,
so importing this package registers all six: a missing import would silently shrink the
comparison rather than fail.
"""

from .base import (
    UNDEFINED,
    Audit,
    AuditContext,
    Degeneracy,
    REGISTRY,
    audit_names,
    describe_degeneracy,
    get_audit,
    register,
)
from .behavioral import Behavioral
from .privacy_population import PopulationMIA

__all__ = [
    "Audit",
    "AuditContext",
    "Degeneracy",
    "REGISTRY",
    "UNDEFINED",
    "audit_names",
    "describe_degeneracy",
    "get_audit",
    "register",
    "Behavioral",
    "PopulationMIA",
]
