"""
reporting/report_builder.py
===========================

Renders the two clinician-facing artefacts the specification demands (§2.5):

* **JSON** for system consumption — the full `ValidationReport` / `DischargeSummary`
* **HTML** for clinicians — Jinja2 templates, also served over MCP as
  `resource://report-template/html`
* **PDF**  — optional export used by the HITL dashboard's download buttons

Outputs land in `Data/reports/` (configurable via `paths.reports_dir`):

    P1019_audit.json      P1019_audit.html      P1019_audit.pdf
    P1019_summary.json    P1019_summary.html    P1019_summary.pdf
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..common.rules import risk_thresholds, rules_version
from ..common.schemas import DischargeSummary, ValidationReport
from ..observability import tracing
from ..settings import settings

log = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"

_env = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
    trim_blocks=True,
    lstrip_blocks=True,
)


def _hospital() -> str:
    return settings.cfg.get("project", {}).get(
        "hospital", "St. Marian Regional Medical Center"
    )


def _reports_dir() -> Path:
    return settings.path("reports_dir")


# --------------------------------------------------------------------------- #
#  Audit / risk report
# --------------------------------------------------------------------------- #
def render_audit_html(report: ValidationReport) -> str:
    template = _env.get_template("audit_report.html")
    return template.render(
        report=report,
        hospital=_hospital(),
        thresholds=risk_thresholds(),
        trace_url=tracing.trace_url(report.trace_id),
    )


def build_reports(
    report: ValidationReport, *, write_files: bool = True, pdf: bool = False
) -> dict[str, str]:
    """Write `<patient>_audit.json` + `.html` (+ `.pdf`) and return the paths."""
    if not report.rules_version:
        report.rules_version = rules_version()

    artefacts: dict[str, str] = {}
    payload = report.model_dump(mode="json")
    payload["trace_url"] = tracing.trace_url(report.trace_id)
    html = render_audit_html(report)

    if not write_files:
        return {"json_inline": json.dumps(payload, ensure_ascii=False), "html_inline": html}

    directory = _reports_dir()
    json_path = directory / f"{report.patient_id}_audit.json"
    html_path = directory / f"{report.patient_id}_audit.html"

    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    html_path.write_text(html, encoding="utf-8")
    artefacts["json"] = str(json_path)
    artefacts["html"] = str(html_path)

    if pdf:
        pdf_path = directory / f"{report.patient_id}_audit.pdf"
        if write_pdf(_audit_pdf_lines(report), pdf_path):
            artefacts["pdf"] = str(pdf_path)

    log.info("Audit report written for %s → %s", report.patient_id, html_path.name)
    return artefacts


# --------------------------------------------------------------------------- #
#  Patient-friendly discharge summary
# --------------------------------------------------------------------------- #
def render_summary_html(summary: DischargeSummary) -> str:
    template = _env.get_template("discharge_summary.html")
    return template.render(
        summary=summary,
        hospital=_hospital(),
        trace_url=tracing.trace_url(summary.trace_id),
    )


def build_summary_reports(
    summary: DischargeSummary, *, write_files: bool = True, pdf: bool = False
) -> dict[str, str]:
    """Write `<patient>_summary.json` + `.html` (+ `.pdf`)."""
    artefacts: dict[str, str] = {}
    payload = summary.model_dump(mode="json")
    payload["markdown"] = summary.as_markdown()
    html = render_summary_html(summary)

    if not write_files:
        return {"json_inline": json.dumps(payload, ensure_ascii=False), "html_inline": html}

    directory = _reports_dir()
    json_path = directory / f"{summary.patient_id}_summary.json"
    html_path = directory / f"{summary.patient_id}_summary.html"

    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    html_path.write_text(html, encoding="utf-8")
    artefacts["json"] = str(json_path)
    artefacts["html"] = str(html_path)

    if pdf:
        pdf_path = directory / f"{summary.patient_id}_summary.pdf"
        if write_pdf(_summary_pdf_lines(summary), pdf_path):
            artefacts["pdf"] = str(pdf_path)

    log.info("Discharge summary written for %s → %s", summary.patient_id, html_path.name)
    return artefacts


# --------------------------------------------------------------------------- #
#  PDF export
# --------------------------------------------------------------------------- #
def write_pdf(blocks: list[tuple[str, str]], destination: Path) -> bool:
    """Render `(style, text)` blocks to PDF via reportlab.

    Styles: "title" | "heading" | "body" | "bullet" | "spacer".
    Returns False (with a warning) when reportlab is unavailable, so the caller
    can offer HTML/JSON only rather than failing the export.
    """
    try:
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError:
        log.warning("reportlab is not installed — PDF export unavailable.")
        return False

    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "DocTitle", parent=base["Title"], fontSize=16, spaceAfter=10, alignment=TA_LEFT
        ),
        "heading": ParagraphStyle(
            "DocHeading", parent=base["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=5
        ),
        "body": ParagraphStyle("DocBody", parent=base["BodyText"], fontSize=9.5, leading=13),
        "bullet": ParagraphStyle(
            "DocBullet", parent=base["BodyText"], fontSize=9.5, leading=13, leftIndent=12
        ),
    }

    story: list[Any] = []
    for style, text in blocks:
        if style == "spacer":
            story.append(Spacer(1, 6))
            continue
        story.append(Paragraph(_pdf_escape(text), styles.get(style, styles["body"])))

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        SimpleDocTemplate(
            str(destination), pagesize=A4,
            leftMargin=18 * mm, rightMargin=18 * mm,
            topMargin=16 * mm, bottomMargin=16 * mm,
        ).build(story)
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("PDF export failed for %s: %s", destination.name, exc)
        return False


def _pdf_escape(text: str) -> str:
    """Escape XML and swap glyphs the built-in PDF fonts cannot render."""
    replacements = {
        "&": "&amp;", "<": "&lt;", ">": "&gt;",
        "—": "-", "–": "-", "≤": "&lt;=", "≥": "&gt;=", "→": "-&gt;",
        "•": "-", "’": "'", "‘": "'", "“": '"', "”": '"', "…": "...",
    }
    out = str(text)
    for old, new in replacements.items():
        out = out.replace(old, new)
    #  reportlab's Helvetica has no glyphs for non-Latin scripts (Devanagari,
    #  etc.) and renders them as black boxes — transliterate to a marker.
    return "".join(ch if ord(ch) < 0x2000 else "?" for ch in out)


def _audit_pdf_lines(report: ValidationReport) -> list[tuple[str, str]]:
    risk = report.risk
    blocks: list[tuple[str, str]] = [
        ("title", f"Discharge Audit &amp; Risk Report - {report.patient_id}"),
        ("body", f"{_hospital()} | generated {report.generated_at}"),
        ("body", f"Patient: {report.patient_name or report.patient_id}"),
        ("spacer", ""),
        ("heading", "Decision"),
        ("body", f"Risk level: {risk.level.value} (score {risk.score})"),
        ("body", f"Recommendation: {risk.recommendation.value} - {risk.recommendation_text}"),
        ("body", f"Discharge blocked: {'YES' if risk.discharge_blocked else 'no'}"),
        ("body", f"HITL required: {'YES' if risk.hitl_required else 'no'}"),
        ("body", f"Completeness: {report.completeness.score}%"),
        ("body", f"Translation confidence: {report.translation_confidence:.2f} "
                 f"({report.detected_language})"),
        ("body", f"Bill: {report.bill_total_amount} {report.bill_currency or ''} "
                 f"[{report.bill_payment_status or 'UNKNOWN'}]"),
    ]

    if risk.hard_guardrails_hit:
        blocks.append(("body", "Hard guardrails: " + ", ".join(risk.hard_guardrails_hit)))

    blocks += [("spacer", ""), ("heading", f"Missing fields ({len(report.completeness.missing_fields)})")]
    if report.completeness.missing_fields:
        for field in report.completeness.missing_fields:
            marker = "BLOCKING" if field.blocking else "non-blocking"
            row = f" row {field.row}" if field.row else ""
            blocks.append(("bullet", f"- {field.document}.{field.field}{row} [{marker}]"))
    else:
        blocks.append(("body", "All mandatory fields documented."))

    blocks += [("spacer", ""), ("heading", f"Cross-validation findings ({len(report.findings)})")]
    if report.findings:
        for finding in report.findings:
            blocks.append((
                "bullet",
                f"- [{finding.severity.value}] {finding.rule_id}: {finding.message}",
            ))
            if finding.clinical_impact:
                blocks.append(("bullet", f"    Impact: {finding.clinical_impact}"))
    else:
        blocks.append(("body", "No discrepancies against the EHR, care plan or labs."))

    if report.guardrail_events:
        blocks += [("spacer", ""), ("heading", "Responsible-AI guardrails")]
        for event in report.guardrail_events:
            state = "TRIGGERED" if event.triggered else "pass"
            blocks.append(("bullet", f"- {event.guardrail} [{state}]: {event.action}"))

    blocks += [("spacer", ""), ("heading", f"Audit trail ({len(report.audit_trail)} steps)")]
    for index, entry in enumerate(report.audit_trail, start=1):
        duration = f"{entry.duration_ms} ms" if entry.duration_ms is not None else "-"
        blocks.append((
            "bullet",
            f"{index}. {entry.step} | {entry.actor} "
            f"({entry.framework or 'n/a'}) | {entry.status} | {duration}",
        ))

    blocks += [
        ("spacer", ""),
        ("body", f"Rules version: {report.rules_version}"),
        ("body", f"LangFuse trace: {report.trace_id or 'n/a'}"),
    ]
    return blocks


def _summary_pdf_lines(summary: DischargeSummary) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = [
        ("title", f"Discharge Summary - {summary.patient_name or summary.patient_id}"),
        ("body", f"{_hospital()} | Patient ID {summary.patient_id} | "
                 f"prepared {summary.generated_at}"),
        ("body", f"Risk level: {summary.risk_level.value} | audience: {summary.audience}"),
        ("spacer", ""),
    ]

    for section in summary.sections:
        blocks.append(("heading", section.title))
        for line in (section.content or "").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            blocks.append(("bullet" if stripped.startswith(("-", "*", "•")) else "body", stripped))

    if summary.prescription_table:
        blocks += [("spacer", ""), ("heading", "Your medicines")]
        for index, row in enumerate(summary.prescription_table, start=1):
            blocks.append((
                "bullet",
                f"{row.get('sl_no') or index}. {row.get('medicine_name') or '-'} "
                f"{row.get('strength') or ''} - {row.get('dosage') or ''} "
                f"{row.get('frequency_plain') or row.get('frequency') or ''} "
                f"{row.get('route_plain') or row.get('route') or ''} "
                f"for {row.get('period') or '-'}",
            ))

    if summary.lab_table:
        blocks += [("spacer", ""), ("heading", "Your test results")]
        for row in summary.lab_table:
            blocks.append((
                "bullet",
                f"- {row.get('test')}: {row.get('value') or '-'} {row.get('unit') or ''} "
                f"(normal {row.get('reference_range') or '-'}) "
                f"[{row.get('flag') or 'NORMAL'}]",
            ))

    if summary.bill_snapshot:
        blocks += [
            ("spacer", ""),
            ("heading", "Your bill"),
            ("body", f"Total: {summary.bill_snapshot.get('total_amount')} "
                     f"{summary.bill_snapshot.get('currency') or ''} "
                     f"[{summary.bill_snapshot.get('payment_status') or 'unknown'}]"),
        ]

    blocks += [("spacer", ""), ("body", f"LangFuse trace: {summary.trace_id or 'n/a'}")]
    return blocks
