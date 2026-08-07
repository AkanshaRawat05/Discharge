"""
dashboard/i18n.py
=================

Thin re-export of the shared English display layer.

The lookup tables live in `discharge_ai.common.display_terms` because the
exported summary HTML/PDF needs exactly the same wording as the on-screen
tables — if the two diverge, a reviewer signs off on one thing and the patient
goes home with another.  This module exists so dashboard code can keep saying
`from i18n import english_route` without reaching into the package layout.
"""

from __future__ import annotations

from discharge_ai.common.display_terms import (  # noqa: F401
    DURATION_UNITS,
    FLAG_TERMS,
    LAB_TEST_TERMS,
    REMARK_TERMS,
    ROUTE_TERMS,
    english_duration,
    english_flag,
    english_lab_test,
    english_remark,
    english_route,
)

__all__ = [
    "DURATION_UNITS",
    "FLAG_TERMS",
    "LAB_TEST_TERMS",
    "REMARK_TERMS",
    "ROUTE_TERMS",
    "english_duration",
    "english_flag",
    "english_lab_test",
    "english_remark",
    "english_route",
]
