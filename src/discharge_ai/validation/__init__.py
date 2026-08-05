"""
Clinical & business validation engine.

    completeness.py      specification Table 3 — mandatory-field validation
    cross_validation.py  specification Table 4 — EHR / care-plan / lab checks
    risk.py              rules.yaml risk matrix → score, level, recommendation
"""

from .completeness import check_completeness, elicitation_schema_for  # noqa: F401
from .cross_validation import cross_validate  # noqa: F401
from .risk import score_case  # noqa: F401
