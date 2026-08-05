"""Audit / risk / discharge-summary report generation (JSON · HTML · PDF)."""

from .report_builder import (  # noqa: F401
    build_reports,
    build_summary_reports,
    render_audit_html,
    render_summary_html,
    write_pdf,
)
