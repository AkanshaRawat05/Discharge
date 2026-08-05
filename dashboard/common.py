"""
dashboard/common.py
===================

Shared helpers for the five Streamlit HITL pages: bootstrapping `sys.path`,
session-state management, async bridging, and the small UI atoms (risk badges,
findings tables, trace links) the pages reuse.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Callable

import streamlit as st

#  The dashboard lives outside the package, so put `src/` on the path first.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (str(SRC), str(ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from discharge_ai.common.doc_loader import (  # noqa: E402
    load_patient_documents,
    scan_incoming,
)
from discharge_ai.common.parsing import build_extracted_case  # noqa: E402
from discharge_ai.common.schemas import (  # noqa: E402
    DischargeSummary,
    ExtractedCase,
    ValidationReport,
)
from discharge_ai.llm.provider import provider_banner  # noqa: E402
from discharge_ai.observability import tracing  # noqa: E402
from discharge_ai.pipeline import default_mode  # noqa: E402
from discharge_ai.settings import settings  # noqa: E402

PAGE_ICON = "🏥"

RISK_COLOURS = {
    "Low": ("#1c6b3f", "#e9f6ee"),
    "Medium": ("#8a5a00", "#fff6e2"),
    "High": ("#b8202e", "#fdecee"),
}
SEVERITY_COLOURS = {
    "Critical": ("#b8202e", "#fdecee"),
    "Warning": ("#8a5a00", "#fff6e2"),
    "Info": ("#1b5e8a", "#eaf4fb"),
}


# --------------------------------------------------------------------------- #
#  Page setup
# --------------------------------------------------------------------------- #
def page_setup(title: str, subtitle: str = "") -> None:
    st.set_page_config(page_title=f"{title} — Discharge HITL", page_icon=PAGE_ICON,
                       layout="wide")
    st.title(f"{PAGE_ICON} {title}")
    if subtitle:
        st.caption(subtitle)
    _init_state()


def _init_state() -> None:
    defaults: dict[str, Any] = {
        "patient_id": None,
        "cases": {},              # patient_id -> ExtractedCase
        "reports": {},            # patient_id -> ValidationReport
        "summaries": {},          # patient_id -> DischargeSummary
        "artefacts": {},          # patient_id -> {kind: path}
        "pipeline_runs": {},      # patient_id -> result dict
        "feedback": {},           # patient_id -> reviewer decisions
        "rag_history": [],
        "execution_mode": default_mode(),
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def available_patients() -> list[str]:
    return sorted(scan_incoming().keys())


def patient_selector() -> str | None:
    """Global patient picker shared across every page.

    Uses a single fixed widget key so changing the patient on one page
    is reflected immediately when the user navigates to another.
    """
    patients = available_patients()
    if not patients:
        st.error(
            f"No patient documents found under `{settings.path('input_root')}`. "
            "Add discharge/lab/bill files named with a patient id (e.g. `P1019_…`)."
        )
        return None

    current = st.session_state.get("patient_id")
    index = patients.index(current) if current in patients else 0
    chosen = st.sidebar.selectbox(
        "Patient", patients, index=index, key="global_patient_selector"
    )
    st.session_state["patient_id"] = chosen
    return chosen


def sidebar_status() -> None:
    """System status panel shown on every page."""
    with st.sidebar:
        st.divider()
        st.subheader("System")
        st.session_state["execution_mode"] = st.radio(
            "Execution mode",
            ["local", "a2a"],
            index=["local", "a2a"].index(st.session_state.get("execution_mode", "local")),
            help=(
                "local = agent handlers run inside this process. "
                "a2a = each agent is called over the A2A protocol "
                "(start them with `python run_services.py --agents`)."
            ),
        )
        st.caption(provider_banner())
        st.caption(
            f"LangFuse: {'connected' if tracing.langfuse_available() else 'disabled'}"
        )
        # st.caption(f"MCP: {settings.mcp_primary_url}")
        # st.caption(f"MCP analytics: {settings.mcp_analytics_url}")
        # st.caption(f"Mock EHR: {settings.ehr_base_url}")


# --------------------------------------------------------------------------- #
#  Async bridge
# --------------------------------------------------------------------------- #
def run_async(coroutine: Any) -> Any:
    """Run a coroutine from Streamlit's synchronous script thread."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    #  A loop is already running (rare in Streamlit): use a private one.
    import threading

    box: dict[str, Any] = {}

    def _worker() -> None:
        loop = asyncio.new_event_loop()
        try:
            box["result"] = loop.run_until_complete(coroutine)
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc
        finally:
            loop.close()

    thread = threading.Thread(target=_worker)
    thread.start()
    thread.join()

    if "error" in box:
        raise box["error"]
    return box.get("result")


def iterate_async(async_generator_factory: Callable[[], Any]):
    """Consume an async generator into a list (Streamlit renders after the run)."""

    async def _collect() -> list[Any]:
        return [item async for item in async_generator_factory()]

    return run_async(_collect())


# --------------------------------------------------------------------------- #
#  Case / report access
# --------------------------------------------------------------------------- #
def get_case(patient_id: str, *, refresh: bool = False) -> ExtractedCase:
    """The extracted case for a patient (parsed on demand, then cached)."""
    cache: dict[str, Any] = st.session_state["cases"]
    if refresh or patient_id not in cache:
        documents = load_patient_documents(patient_id)
        cache[patient_id] = build_extracted_case(patient_id, documents)
    value = cache[patient_id]
    return (
        value if isinstance(value, ExtractedCase)
        else ExtractedCase.model_validate(value)
    )


def get_report(patient_id: str) -> ValidationReport | None:
    """The validation report from this session, or from disk if one was written."""
    cached = st.session_state["reports"].get(patient_id)
    if cached is not None:
        return (
            cached if isinstance(cached, ValidationReport)
            else ValidationReport.model_validate(cached)
        )

    path = settings.path("reports_dir") / f"{patient_id}_audit.json"
    if path.exists():
        try:
            report = ValidationReport.model_validate(
                json.loads(path.read_text(encoding="utf-8"))
            )
            st.session_state["reports"][patient_id] = report
            return report
        except Exception:  # noqa: BLE001
            return None
    return None


def get_summary(patient_id: str) -> DischargeSummary | None:
    cached = st.session_state["summaries"].get(patient_id)
    if cached is not None:
        return (
            cached if isinstance(cached, DischargeSummary)
            else DischargeSummary.model_validate(cached)
        )

    path = settings.path("reports_dir") / f"{patient_id}_summary.json"
    if path.exists():
        try:
            summary = DischargeSummary.model_validate(
                json.loads(path.read_text(encoding="utf-8"))
            )
            st.session_state["summaries"][patient_id] = summary
            return summary
        except Exception:  # noqa: BLE001
            return None
    return None


def store_pipeline_result(result: Any) -> None:
    patient_id = result.patient_id
    st.session_state["pipeline_runs"][patient_id] = result.as_dict()
    if result.case is not None:
        st.session_state["cases"][patient_id] = result.case
    if result.report is not None:
        st.session_state["reports"][patient_id] = result.report
    if result.summary is not None:
        st.session_state["summaries"][patient_id] = result.summary
    if result.artefacts:
        st.session_state["artefacts"].setdefault(patient_id, {}).update(result.artefacts)


# --------------------------------------------------------------------------- #
#  UI atoms
# --------------------------------------------------------------------------- #
def badge(text: str, palette: tuple[str, str]) -> str:
    fore, back = palette
    return (
        f"<span style='background:{back};color:{fore};padding:3px 12px;"
        f"border-radius:12px;font-weight:700;font-size:0.78rem;"
        f"letter-spacing:.4px;text-transform:uppercase'>{text}</span>"
    )


def risk_badge(level: str) -> str:
    return badge(level, RISK_COLOURS.get(level, ("#333", "#eee")))


def severity_badge(severity: str) -> str:
    return badge(severity, SEVERITY_COLOURS.get(severity, ("#333", "#eee")))


def language_badge(code: str) -> str:
    from discharge_ai.common.terminology import language_name

    palette = ("#1b5e8a", "#eaf4fb") if code == "en" else ("#5b3a8a", "#f1ebfb")
    return badge(f"{language_name(code)} ({code})", palette)


def trace_link(trace_id: str | None) -> None:
    if not trace_id:
        return
    url = tracing.trace_url(trace_id)
    if url:
        st.markdown(f"🔗 [Open the LangFuse trace]({url}) · `{trace_id}`")
    else:
        st.caption(f"LangFuse trace id: `{trace_id}` (tracing disabled)")


def findings_table(report: ValidationReport) -> None:
    import pandas as pd

    if not report.findings:
        st.success("No discrepancies found against the EHR, care plan or labs.")
        return

    rows = [
        {
            "Severity": finding.severity.value,
            "Rule": finding.rule_id,
            "Finding": finding.message,
            "Clinical impact": finding.clinical_impact or "—",
            "Action": finding.suggested_action or finding.action,
            "Blocks discharge": "YES" if finding.blocks_discharge else "no",
        }
        for finding in report.findings
    ]
    frame = pd.DataFrame(rows)

    def _highlight(row: Any) -> list[str]:
        colour = {
            "Critical": "background-color:#fdecee",
            "Warning": "background-color:#fff6e2",
            "Info": "background-color:#eaf4fb",
        }.get(row["Severity"], "")
        return [colour] * len(row)

    st.dataframe(
        frame.style.apply(_highlight, axis=1),
        width="stretch",
        hide_index=True,
    )


def guardrail_table(events: list[Any]) -> None:
    import pandas as pd

    if not events:
        st.caption("No guardrail activity recorded for this case.")
        return

    rows = []
    for event in events:
        payload = event if isinstance(event, dict) else event.model_dump(mode="json")
        rows.append(
            {
                "Guardrail": payload.get("guardrail"),
                "Triggered": "yes" if payload.get("triggered") else "no",
                "Action": payload.get("action"),
                "Detail": payload.get("detail"),
                "Score": payload.get("score"),
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def audit_trail_table(report: ValidationReport) -> None:
    import pandas as pd

    if not report.audit_trail:
        st.caption("No audit trail recorded.")
        return

    rows = [
        {
            "#": index,
            "Step": entry.step,
            "Actor": entry.actor,
            "Framework": entry.framework or "—",
            "Status": entry.status,
            "ms": entry.duration_ms,
            "Trace": (entry.langfuse_trace_id or "—")[:16],
        }
        for index, entry in enumerate(report.audit_trail, start=1)
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def artefact_downloads(patient_id: str) -> None:
    """Download buttons for every generated artefact of this patient."""
    directory = settings.path("reports_dir")
    candidates = [
        ("Audit report (JSON)", f"{patient_id}_audit.json", "application/json"),
        ("Audit report (HTML)", f"{patient_id}_audit.html", "text/html"),
        ("Audit report (PDF)", f"{patient_id}_audit.pdf", "application/pdf"),
        ("Discharge summary (JSON)", f"{patient_id}_summary.json", "application/json"),
        ("Discharge summary (HTML)", f"{patient_id}_summary.html", "text/html"),
        ("Discharge summary (PDF)", f"{patient_id}_summary.pdf", "application/pdf"),
    ]

    existing = [(label, directory / name, mime) for label, name, mime in candidates
                if (directory / name).exists()]
    if not existing:
        st.caption("No artefacts have been generated yet — run the pipeline first.")
        return

    columns = st.columns(min(3, len(existing)))
    for index, (label, path, mime) in enumerate(existing):
        with columns[index % len(columns)]:
            st.download_button(
                label,
                data=path.read_bytes(),
                file_name=path.name,
                mime=mime,
                width="stretch",
                key=f"download-{path.name}",
            )


def no_report_warning(patient_id: str) -> None:
    st.info(
        f"No validation report for **{patient_id}** yet. Open "
        "**1 · Document Viewer** and press *Process this patient*, or run "
        f"`python -m discharge_ai.cli run {patient_id}` from a terminal."
    )
