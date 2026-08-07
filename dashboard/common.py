"""
dashboard/common.py
===================

Shared helpers for the five Streamlit HITL views: bootstrapping `sys.path`,
session-state management, async bridging, and the small UI atoms (risk badges,
findings tables, trace links) the views reuse.

The console is a **gated flow**, not five independent tabs.  `st.session_state`
carries the active patient, the pipeline stage reached for that patient, and
whether the discharge is blocked; the navigation is rebuilt from that state on
every run, so a view is only reachable once the pipeline has actually put the
case there.  See `flow_state()`, `page_unlocked()` and `render_nav()`.
"""

from __future__ import annotations

import asyncio
import json
import queue
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Iterator

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
from discharge_ai.observability import tracing  # noqa: E402
from discharge_ai.pipeline import default_mode  # noqa: E402
from discharge_ai.settings import settings  # noqa: E402

PAGE_ICON = "🏥"

# --------------------------------------------------------------------------- #
#  Flow definition
# --------------------------------------------------------------------------- #
#  Pipeline stages a patient can be in. The navigation is derived from these.
STAGE_NEW = "new"                # documents on disk, nothing run yet
STAGE_VALIDATED = "validated"    # extraction → normalisation → validation done
STAGE_APPROVED = "approved"      # cleared for release (not blocked, or signed off)

#  Navigation keys → the labels/icons used by `render_nav()` and `goto()`.
NAV_ITEMS: list[dict[str, str]] = [
    {"key": "documents", "label": "1 · Document Viewer", "icon": ":material/description:"},
    {"key": "validation", "label": "2 · Validation Report", "icon": ":material/fact_check:"},
    {"key": "corrections", "label": "3 · HITL Corrections", "icon": ":material/edit_note:"},
    {"key": "rag", "label": "4 · RAG Q&A", "icon": ":material/forum:"},
    {"key": "summary", "label": "5 · Discharge Summary", "icon": ":material/assignment_turned_in:"},
]

# --------------------------------------------------------------------------- #
#  Palette — clinical, low-saturation.  White / light blue / teal / soft green
#  / light grey.  Status colours are muted enough to sit calmly on a screen a
#  clinician reads all day, but still clear a WCAG AA contrast check against
#  their tinted backgrounds.
# --------------------------------------------------------------------------- #
PALETTE = {
    "ink": "#1f2933",          # primary text
    "muted": "#5c6b7a",        # secondary text
    "line": "#e3e9ef",         # hairline borders
    "surface": "#ffffff",      # card background
    "canvas": "#f5f8fa",       # page background
    "accent": "#0f6e8c",       # teal — primary brand accent
    "accent_soft": "#e6f2f6",  # teal tint
    "ok_fore": "#20724f",      # soft green
    "ok_back": "#e8f4ee",
    "warn_fore": "#8a6100",    # muted amber
    "warn_back": "#fdf4e3",
    "crit_fore": "#a32d38",    # desaturated clinical red
    "crit_back": "#fbecee",
    "info_fore": "#1b5e8a",    # light blue
    "info_back": "#e9f2f9",
}

RISK_COLOURS = {
    "Low": (PALETTE["ok_fore"], PALETTE["ok_back"]),
    "Medium": (PALETTE["warn_fore"], PALETTE["warn_back"]),
    "High": (PALETTE["crit_fore"], PALETTE["crit_back"]),
}
SEVERITY_COLOURS = {
    "Critical": (PALETTE["crit_fore"], PALETTE["crit_back"]),
    "Warning": (PALETTE["warn_fore"], PALETTE["warn_back"]),
    "Info": (PALETTE["info_fore"], PALETTE["info_back"]),
}


# --------------------------------------------------------------------------- #
#  Page setup
# --------------------------------------------------------------------------- #
def configure_page() -> None:
    """`st.set_page_config` — called exactly once, from the entrypoint."""
    st.set_page_config(
        page_title="Discharge Review Console",
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme()


def inject_theme() -> None:
    """Clinical visual language, applied once per session.

    Styling only — no component behaviour is touched.  The goal is a calm,
    high-legibility surface a clinician reads all shift: generous white space,
    a single teal accent, hairline borders instead of heavy shadows, and status
    colour used sparingly so that when something *is* red it actually reads as
    urgent.
    """
    st.markdown(
        f"""
<style>
  /* ---- Type scale ------------------------------------------------------ */
  html, body, [class*="st-"] {{
      font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
      color: {PALETTE['ink']};
  }}
  .stApp {{ background: {PALETTE['canvas']}; }}

  /* Page title: smaller and calmer than Streamlit's default slab. */
  h1 {{
      font-size: 1.65rem !important;
      font-weight: 650 !important;
      letter-spacing: -.015em;
      color: {PALETTE['ink']};
      padding-bottom: .15rem !important;
  }}
  h2 {{ font-size: 1.2rem !important;  font-weight: 640 !important;
        letter-spacing: -.01em; margin-top: .4rem !important; }}
  h3 {{ font-size: 1.02rem !important; font-weight: 640 !important; }}
  h4 {{ font-size: .95rem !important;  font-weight: 640 !important; }}

  /* ---- Main container spacing ----------------------------------------- */
  .block-container {{
      padding-top: 2.4rem;
      padding-bottom: 3.5rem;
      max-width: 1500px;
  }}

  /* ---- Cards ----------------------------------------------------------- */
  /* Bordered containers are the primary grouping device across every page. */
  div[data-testid="stVerticalBlockBorderWrapper"] {{
      background: {PALETTE['surface']};
      border: 1px solid {PALETTE['line']} !important;
      border-radius: 10px;
      padding: 2px 4px;
      box-shadow: 0 1px 2px rgba(31,41,51,.04);
  }}

  /* ---- Metrics --------------------------------------------------------- */
  div[data-testid="stMetric"] {{
      background: {PALETTE['surface']};
      border: 1px solid {PALETTE['line']};
      border-radius: 10px;
      padding: 14px 16px;
  }}
  div[data-testid="stMetricLabel"] p {{
      font-size: .74rem !important;
      font-weight: 600;
      letter-spacing: .04em;
      text-transform: uppercase;
      color: {PALETTE['muted']};
  }}
  div[data-testid="stMetricValue"] {{
      font-size: 1.5rem !important;
      font-weight: 650;
      color: {PALETTE['ink']};
  }}

  /* ---- Tabs ------------------------------------------------------------ */
  button[data-baseweb="tab"] {{
      font-weight: 560;
      color: {PALETTE['muted']};
  }}
  button[data-baseweb="tab"][aria-selected="true"] {{
      color: {PALETTE['accent']};
  }}
  div[data-baseweb="tab-highlight"] {{ background-color: {PALETTE['accent']}; }}
  div[data-baseweb="tab-border"]    {{ background-color: {PALETTE['line']}; }}

  /* ---- Buttons --------------------------------------------------------- */
  .stButton > button {{
      border-radius: 8px;
      font-weight: 560;
      border: 1px solid {PALETTE['line']};
      transition: background .12s ease, border-color .12s ease;
  }}
  .stButton > button[kind="primary"] {{
      background: {PALETTE['accent']};
      border-color: {PALETTE['accent']};
      color: #ffffff;
  }}
  .stButton > button[kind="primary"]:hover {{
      background: #0c5a73;
      border-color: #0c5a73;
  }}
  .stButton > button[kind="secondary"]:hover {{
      border-color: {PALETTE['accent']};
      color: {PALETTE['accent']};
  }}

  /* ---- Sidebar --------------------------------------------------------- */
  section[data-testid="stSidebar"] {{
      background: {PALETTE['surface']};
      border-right: 1px solid {PALETTE['line']};
  }}
  section[data-testid="stSidebar"] h3 {{
      font-size: .74rem !important;
      font-weight: 640 !important;
      letter-spacing: .07em;
      text-transform: uppercase;
      color: {PALETTE['muted']};
      margin-bottom: .35rem;
  }}
  /* Nav links read as a list, with the teal accent reserved for the active one. */
  section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] {{
      border-radius: 7px;
      padding: 5px 9px;
      margin-bottom: 1px;
  }}
  section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]:hover {{
      background: {PALETTE['accent_soft']};
  }}

  /* ---- Tables ---------------------------------------------------------- */
  div[data-testid="stDataFrame"] {{
      border: 1px solid {PALETTE['line']};
      border-radius: 8px;
  }}

  /* ---- Alerts: flat tints, no gradients ------------------------------- */
  div[data-testid="stAlert"] {{
      border-radius: 8px;
      border-left-width: 3px;
  }}

  /* ---- Inputs ---------------------------------------------------------- */
  div[data-baseweb="input"], div[data-baseweb="select"] > div,
  div[data-baseweb="textarea"] {{
      border-radius: 7px !important;
  }}

  /* ---- Expanders ------------------------------------------------------- */
  details[data-testid="stExpander"] {{
      border: 1px solid {PALETTE['line']};
      border-radius: 9px;
      background: {PALETTE['surface']};
  }}

  /* ---- Dividers: hairline, not a slab --------------------------------- */
  hr {{ border-color: {PALETTE['line']} !important; margin: 1.4rem 0 !important; }}

  /* ---- Captions -------------------------------------------------------- */
  div[data-testid="stCaptionContainer"] {{ color: {PALETTE['muted']}; }}

  /* ---- Progress -------------------------------------------------------- */
  div[data-testid="stProgress"] > div > div > div {{
      background-color: {PALETTE['accent']};
  }}
</style>
""",
        unsafe_allow_html=True,
    )


def page_setup(title: str, subtitle: str = "") -> None:
    """Header for one view. Page config lives in the entrypoint (`app.py`)."""
    init_state()
    #  Re-injected per view: Streamlit runs each page as its own script, so the
    #  <style> block from `configure_page()` does not survive a page switch.
    inject_theme()
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def init_state() -> None:
    defaults: dict[str, Any] = {
        "patient_id": None,
        "cases": {},              # patient_id -> ExtractedCase
        "reports": {},            # patient_id -> ValidationReport
        "summaries": {},          # patient_id -> DischargeSummary
        "artefacts": {},          # patient_id -> {kind: path}
        "pipeline_runs": {},      # patient_id -> result dict
        "feedback": {},           # patient_id -> reviewer decisions
        "flow": {},               # patient_id -> flow state (see flow_state)
        "rag_history": [],
        "execution_mode": default_mode(),
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


#  Kept as an alias: earlier revisions of the views imported the private name.
_init_state = init_state


# --------------------------------------------------------------------------- #
#  Flow state — the single source of truth for navigation gating
# --------------------------------------------------------------------------- #
def _default_flow() -> dict[str, Any]:
    return {
        "stage": STAGE_NEW,
        "blocked": None,          # None = unknown (never validated)
        "hitl_required": None,
        "approved": False,        # explicit reviewer sign-off on view 3
        "signed_off_by": None,
        "risk_level": None,
        "risk_score": None,
        "findings": 0,
        "summary_ready": False,
    }


def flow_state(patient_id: str | None = None) -> dict[str, Any]:
    """Mutable flow record for a patient (the active one by default)."""
    init_state()
    patient_id = patient_id or st.session_state.get("patient_id")
    if not patient_id:
        return _default_flow()
    flows: dict[str, Any] = st.session_state["flow"]
    if patient_id not in flows:
        flows[patient_id] = _default_flow()
    return flows[patient_id]


def sync_flow_from_report(
    patient_id: str, report: Any, *, reset_signoff: bool = False
) -> dict[str, Any]:
    """Fold a `ValidationReport` into the flow state for a patient.

    This is what moves a case from `new` to `validated`, and on to `approved`
    when validation itself clears the discharge.

    `reset_signoff` must be set **only** when a genuinely new report arrives
    from a pipeline run (see `store_pipeline_result`): a reviewer's sign-off is
    made against a specific set of findings, so re-validating a still-blocked
    case revokes it.  Plain re-renders of an existing report pass False —
    otherwise simply revisiting view 2 would silently re-lock view 5 on a case
    the reviewer had already signed off.
    """
    state = flow_state(patient_id)
    if report is None:
        return state

    risk = report.risk
    state["blocked"] = bool(risk.discharge_blocked)
    state["hitl_required"] = bool(risk.hitl_required)
    state["risk_level"] = risk.level.value
    state["risk_score"] = risk.score
    state["findings"] = len(report.findings)

    if not risk.discharge_blocked:
        #  Cleared by the pipeline itself — no reviewer sign-off needed.
        state["stage"] = STAGE_APPROVED
    else:
        state["stage"] = STAGE_APPROVED if state["approved"] else STAGE_VALIDATED
        if reset_signoff:
            state["stage"] = STAGE_VALIDATED
            state["approved"] = False
            state["signed_off_by"] = None
    return state


def mark_signed_off(patient_id: str, reviewer: str, *, approved: bool = True) -> None:
    """Record an explicit HITL approval (view 3) that releases a blocked case."""
    state = flow_state(patient_id)
    state["approved"] = bool(approved)
    state["signed_off_by"] = reviewer or "unnamed reviewer" if approved else None
    if approved:
        state["stage"] = STAGE_APPROVED


def hydrate_flow(patient_id: str) -> dict[str, Any]:
    """Populate flow state from a report already on disk or in session."""
    state = flow_state(patient_id)
    if state["blocked"] is None:
        report = get_report(patient_id)
        if report is not None:
            state = sync_flow_from_report(patient_id, report)
    state["summary_ready"] = get_summary(patient_id) is not None
    return state


# --------------------------------------------------------------------------- #
#  Gating
# --------------------------------------------------------------------------- #
def page_unlocked(page_key: str, patient_id: str | None = None) -> tuple[bool, str]:
    """`(unlocked, reason_when_locked)` for one navigation entry.

    View 4 (RAG Q&A) is deliberately never gated — it queries the FAISS index,
    which is independent of any single patient's pipeline stage.
    """
    patient_id = patient_id or st.session_state.get("patient_id")

    if page_key in {"documents", "rag"}:
        return True, ""

    if not patient_id:
        return False, "Select a patient first."

    state = hydrate_flow(patient_id)

    if page_key in {"validation", "corrections"}:
        if state["stage"] == STAGE_NEW:
            return False, (
                f"{patient_id} has not been processed yet — run the pipeline on "
                "**1 · Document Viewer**."
            )
        return True, ""

    if page_key == "summary":
        if state["stage"] == STAGE_NEW:
            return False, (
                f"{patient_id} has not been processed yet — run the pipeline on "
                "**1 · Document Viewer**."
            )
        if state["blocked"] and not state["approved"]:
            return False, (
                f"Discharge is **BLOCKED** for {patient_id}. Resolve the Critical "
                "findings on **3 · HITL Corrections** and sign the case off before "
                "a summary can be released."
            )
        return True, ""

    return True, ""


def require_page(page_key: str) -> str:
    """Guard at the top of a gated view. Stops the script when locked.

    Belt-and-braces: the navigation already hides locked views, but a stale
    browser tab or a deep link must never render a summary for a blocked case.
    """
    patient_id = st.session_state.get("patient_id")
    unlocked, reason = page_unlocked(page_key, patient_id)
    if unlocked:
        return patient_id  # type: ignore[return-value]

    st.warning(reason or "This step is not available yet.")
    targets = {
        "validation": ("documents", "Go to 1 · Document Viewer"),
        "corrections": ("documents", "Go to 1 · Document Viewer"),
        "summary": ("corrections", "Go to 3 · HITL Corrections"),
    }
    target_key, label = targets.get(page_key, ("documents", "Go to 1 · Document Viewer"))
    if not patient_id:
        target_key, label = "documents", "Go to 1 · Document Viewer"
    if st.button(label, type="primary"):
        goto(target_key)
    st.stop()
    return ""  # unreachable — keeps type checkers happy


# --------------------------------------------------------------------------- #
#  Navigation
# --------------------------------------------------------------------------- #
def register_pages(registry: dict[str, Any]) -> None:
    """Entry point hands the built `st.Page` objects over for `goto()`."""
    st.session_state["_page_registry"] = registry


def goto(page_key: str) -> None:
    """Programmatic route — the button-driven equivalent of clicking the nav."""
    registry = st.session_state.get("_page_registry") or {}
    target = registry.get(page_key)
    if target is None:
        st.rerun()
    st.switch_page(target)


def render_nav() -> None:
    """Gated sidebar navigation, rebuilt from flow state on every run."""
    registry = st.session_state.get("_page_registry") or {}
    patient_id = st.session_state.get("patient_id")

    with st.sidebar:
        st.markdown("### Review flow")
        for item in NAV_ITEMS:
            key = item["key"]
            page = registry.get(key)
            if page is None:
                continue
            unlocked, reason = page_unlocked(key, patient_id)
            if unlocked:
                st.page_link(page, label=item["label"], icon=item["icon"])
            else:
                st.page_link(
                    page, label=f"{item['label']} 🔒", icon=item["icon"],
                    disabled=True, help=_plain(reason),
                )

        if patient_id:
            state = hydrate_flow(patient_id)
            st.caption(_stage_caption(patient_id, state))


def _plain(markdown_text: str) -> str:
    return markdown_text.replace("**", "")


def _stage_caption(patient_id: str, state: dict[str, Any]) -> str:
    stage = state["stage"]
    if stage == STAGE_NEW:
        return f"{patient_id}: not processed yet"
    if state["blocked"]:
        if state["approved"]:
            return f"{patient_id}: blocked · signed off by {state['signed_off_by']}"
        return f"{patient_id}: 🛑 discharge BLOCKED"
    if state["hitl_required"]:
        return f"{patient_id}: ⚠️ human review required"
    return f"{patient_id}: ✅ cleared for release"


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

    #  Switching patients must not leave another patient's widget state on the
    #  page.  Per-patient widget keys are namespaced `<prefix>-<patient_id>-…`,
    #  so on a switch we drop every namespaced key that belongs to somebody
    #  else.  Without this, Streamlit re-uses the stale session_state value for
    #  a key and one patient's document text renders under another's tab.
    if current and chosen != current:
        stale = [
            key for key in list(st.session_state.keys())
            if isinstance(key, str) and f"-{current}-" in key
        ]
        for key in stale:
            del st.session_state[key]

    st.session_state["patient_id"] = chosen
    return chosen


#  The former `sidebar_status()` "System" panel (execution-mode radio, LLM
#  provider banner, LangFuse connection state) has been removed: it exposed
#  deployment internals to clinical staff who cannot act on them.  Execution
#  mode is resolved automatically by `default_mode()`, which probes the agent
#  ports and uses the real A2A path when the agents are listening.


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


_STREAM_DONE = object()


def stream_async(async_generator_factory: Callable[[], Any]) -> Iterator[Any]:
    """Bridge an async generator into a **synchronous** generator, live.

    `iterate_async` buffers the whole run before Streamlit draws anything, which
    is fine for the static views but defeats the point on the two real-time
    touchpoints (the HITL re-run and RAG Q&A).  Here the coroutine runs on a
    worker thread with its own event loop and hands each item to the Streamlit
    script thread through a queue, so the caller can repaint a placeholder — or
    feed `st.write_stream` — as tokens arrive.

    Only the *consumer* touches Streamlit, so this never writes to the UI from
    a thread without a script context.
    """
    channel: queue.Queue = queue.Queue()

    def _worker() -> None:
        async def _pump() -> None:
            async for item in async_generator_factory():
                channel.put(("item", item))

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_pump())
        except BaseException as exc:  # noqa: BLE001 — re-raised on the consumer
            channel.put(("error", exc))
        finally:
            try:
                loop.close()
            finally:
                channel.put(("done", _STREAM_DONE))

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    try:
        while True:
            kind, payload = channel.get()
            if kind == "done":
                break
            if kind == "error":
                raise payload
            yield payload
    finally:
        thread.join(timeout=5.0)


def stream_pipeline(patient_id: str, **kwargs: Any) -> Iterator[tuple[str, Any, Any]]:
    """Run `run_pipeline` while yielding its progress callbacks live.

    Yields `("progress", stage, message)` for every pipeline step and finally
    `("result", result, None)`.  The pipeline's own `progress=` callback is a
    plain synchronous function, so it simply drops onto the queue that
    `stream_async` is already draining.
    """
    from discharge_ai.pipeline import run_pipeline

    channel: queue.Queue = queue.Queue()

    def _progress(stage: str, message: str) -> None:
        channel.put(("progress", stage, message))

    def _worker() -> None:
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                run_pipeline(patient_id, progress=_progress, **kwargs)
            )
            channel.put(("result", result, None))
        except BaseException as exc:  # noqa: BLE001 — re-raised on the consumer
            channel.put(("error", exc, None))
        finally:
            try:
                loop.close()
            finally:
                channel.put(("done", None, None))

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    try:
        while True:
            item = channel.get()
            kind = item[0]
            if kind == "done":
                break
            if kind == "error":
                raise item[1]
            yield item  # ("progress", stage, message) | ("result", result, None)
    finally:
        thread.join(timeout=5.0)


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

    #  Advance the flow so the navigation reflects what actually happened. This
    #  is the one place a sign-off is revoked: the findings just changed under
    #  it, so a blocked case must be signed off again against the new report.
    if result.report is not None:
        state = sync_flow_from_report(patient_id, result.report, reset_signoff=True)
        state["summary_ready"] = result.summary is not None


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


def decision_hero(report: Any) -> str:
    """The release decision, rendered as the visually dominant element.

    Returns the decision key: "blocked" | "hitl" | "cleared".  A blocked
    discharge is the single most important fact on the validation view, so it
    is drawn far larger and at higher contrast than any metric around it —
    a reviewer scanning the page must not be able to miss it.
    """
    risk = report.risk
    if risk.discharge_blocked:
        decision, fore, back, icon, headline = (
            "blocked", PALETTE["crit_fore"], PALETTE["crit_back"],
            "🛑", "Discharge blocked",
        )
    elif risk.hitl_required:
        decision, fore, back, icon, headline = (
            "hitl", PALETTE["warn_fore"], PALETTE["warn_back"],
            "⚠️", "Human review required",
        )
    else:
        decision, fore, back, icon, headline = (
            "cleared", PALETTE["ok_fore"], PALETTE["ok_back"],
            "✅", "Cleared for release",
        )

    detail = risk.recommendation_text or ""
    guardrails = (
        "Hard guardrails: " + ", ".join(risk.hard_guardrails_hit)
        if getattr(risk, "hard_guardrails_hit", None) else ""
    )

    #  Still the visually dominant element on the page — a reviewer must not be
    #  able to miss a block — but carried by a heavy accent rule and scale
    #  rather than a saturated full-bleed fill, which reads as alarming chrome
    #  on a screen someone looks at all day.
    #
    #  Emitted as a single line: a wrapped `style="…"` attribute is indented
    #  far enough that Streamlit's markdown pass can treat it as a code block.
    guardrail_chip = (
        f'<div style="display:inline-block;margin-top:12px;background:{back};'
        f'color:{fore};border-radius:6px;padding:5px 11px;font-size:.83rem;'
        f'font-weight:640;">{guardrails}</div>'
        if guardrails else ""
    )
    ink, line, muted = PALETTE["ink"], PALETTE["line"], PALETTE["muted"]
    st.markdown(
        f'<div style="background:{PALETTE["surface"]};border:1px solid {line};'
        f'border-left:6px solid {fore};border-radius:10px;padding:22px 26px;'
        f'margin:4px 0 20px 0;">'
        f'<div style="font-size:1.9rem;line-height:1.15;font-weight:680;'
        f'letter-spacing:-.02em;color:{fore};">{icon}&nbsp;{headline}</div>'
        f'<div style="font-size:1rem;margin-top:10px;color:{ink};">{detail}</div>'
        f"{guardrail_chip}"
        f'<div style="font-size:.85rem;margin-top:14px;padding-top:12px;'
        f'border-top:1px solid {line};color:{muted};">'
        f'Risk <strong style="color:{ink};">{risk.level.value}</strong>'
        f'&nbsp;·&nbsp; Score <strong style="color:{ink};">{risk.score}</strong>'
        f'&nbsp;·&nbsp; Recommendation '
        f'<strong style="color:{ink};">{risk.recommendation.value}</strong>'
        f"</div></div>",
        unsafe_allow_html=True,
    )
    return decision


# --------------------------------------------------------------------------- #
#  Schema-driven elicitation form
# --------------------------------------------------------------------------- #
def _field_type(spec: dict[str, Any]) -> str:
    """Resolve a JSON-Schema property to one primitive type name.

    Optional fields arrive from Pydantic as `anyOf: [{type: X}, {type: null}]`,
    so the null branch is discarded.
    """
    candidates = [
        entry.get("type")
        for entry in spec.get("anyOf", []) or []
        if isinstance(entry, dict)
    ] or [spec.get("type", "string")]
    return next((t for t in candidates if t and t != "null"), "string")


def _field_enum(spec: dict[str, Any]) -> list[Any] | None:
    if spec.get("enum"):
        return list(spec["enum"])
    for entry in spec.get("anyOf", []) or []:
        if isinstance(entry, dict) and entry.get("enum"):
            return list(entry["enum"])
    return None


def render_elicitation_form(schema: dict[str, Any], *, key_prefix: str) -> dict[str, Any]:
    """Render an MCP elicitation schema as a form and collect the answers.

    Nothing here is hardcoded to a particular field: the widget for each
    property is chosen from that property's own `type` / `enum` / `format`, and
    the label comes from its `description` (falling back to `title`, then the
    property name).  Whatever `ctx.elicit()` asks for is what gets drawn.
    """
    properties: dict[str, Any] = schema.get("properties", {}) or {}
    required: set[str] = set(schema.get("required", []) or [])
    answers: dict[str, Any] = {}

    if not properties:
        st.caption("The schema carries no properties — nothing to collect.")
        return answers

    columns = st.columns(2)
    for index, (name, spec) in enumerate(properties.items()):
        spec = spec if isinstance(spec, dict) else {}
        label = spec.get("description") or spec.get("title") or name
        if name in required:
            label = f"{label} *"
        field_type = _field_type(spec)
        choices = _field_enum(spec)
        widget_key = f"{key_prefix}-{name}"
        help_text = f"`{name}` · {field_type}" + (" · required" if name in required else "")

        with columns[index % 2]:
            if choices:
                picked = st.selectbox(
                    label, ["(unknown)"] + [str(c) for c in choices],
                    key=widget_key, help=help_text,
                )
                value = None if picked == "(unknown)" else picked
            elif field_type == "boolean":
                picked = st.selectbox(
                    label, ["(unknown)", "Yes", "No"], key=widget_key, help=help_text
                )
                value = None if picked == "(unknown)" else (picked == "Yes")
            elif field_type == "integer":
                value = st.number_input(
                    label, value=None, step=1,
                    min_value=_as_number(spec.get("minimum")),
                    max_value=_as_number(spec.get("maximum")),
                    key=widget_key, help=help_text,
                    placeholder="leave blank if unknown",
                )
                value = int(value) if value is not None else None
            elif field_type == "number":
                value = st.number_input(
                    label, value=None,
                    min_value=_as_number(spec.get("minimum"), float_type=True),
                    max_value=_as_number(spec.get("maximum"), float_type=True),
                    key=widget_key, help=help_text,
                    placeholder="leave blank if unknown",
                )
            elif field_type == "array":
                raw = st.text_input(
                    label, value="", key=widget_key,
                    help=help_text + " · comma-separated",
                    placeholder="separate multiple entries with commas",
                )
                value = [part.strip() for part in raw.split(",") if part.strip()] or None
            elif spec.get("format") == "date":
                value = st.date_input(
                    label, value=None, key=widget_key, help=help_text, format="YYYY-MM-DD"
                )
                value = value.isoformat() if value else None
            elif _is_long_text(spec):
                value = st.text_area(
                    label, value="", key=widget_key, help=help_text, height=90,
                    placeholder="leave blank if unknown",
                )
            else:
                value = st.text_input(
                    label, value="", key=widget_key, help=help_text,
                    placeholder="leave blank if unknown",
                )

        if value not in (None, "", []):
            answers[name] = value

    return answers


def _as_number(value: Any, *, float_type: bool = False) -> Any:
    if value is None:
        return None
    try:
        return float(value) if float_type else int(value)
    except (TypeError, ValueError):
        return None


def _is_long_text(spec: dict[str, Any]) -> bool:
    max_length = spec.get("maxLength")
    return isinstance(max_length, int) and max_length > 120


# --------------------------------------------------------------------------- #
#  English-only display layer
# --------------------------------------------------------------------------- #
#  The console is used by English-speaking hospital staff, but source documents
#  arrive in Dutch, Spanish and Hindi.  Everything below converts a value to its
#  English display form at render time only — the underlying `ExtractedCase`
#  keeps the original text, and the Clinical Normalizer / MCP Sampling
#  translation path is untouched.  These helpers never mutate the case.
#  `canonical_drug()` is a *comparison* key, not a display name, and a few of
#  its keys are US-market forms.  This hospital's records use the international
#  (INN/BAN) name, so override those few for display only.  Comparison and
#  validation continue to use the canonical key untouched.
_DRUG_DISPLAY_OVERRIDES = {
    "acetaminophen": "Paracetamol",
    "amoxicillin-clavulanate": "Amoxicillin-Clavulanate",
}


def english_drug_name(name: str | None) -> str:
    """Localised drug spelling → English INN for display.

    `Amoxicilline` (nl) / `Amoxicilina` (es) / `पैरासिटामोल` (hi) → the English
    name.  Falls back to the original string whenever the canonicaliser does not
    recognise it, so an unknown drug is shown as written rather than blanked.
    """
    from discharge_ai.common.terminology import canonical_drug

    original = str(name).strip()
    if not original:
        return ""
    canonical = canonical_drug(original)
    if not canonical:
        return original

    override = _DRUG_DISPLAY_OVERRIDES.get(canonical)
    if override:
        return override

    #  No synonym actually fired — the canonicaliser only lower-cased what we
    #  gave it.  Keep the reviewer's own spelling and capitalisation rather than
    #  title-casing an unrecognised drug into nonsense ("SomeDrug" → "Somedrug").
    if canonical == original.lower():
        return original

    #  Title-case each hyphen-separated part: "amoxicillin-clavulanate" reads
    #  as "Amoxicillin-Clavulanate", not "Amoxicillin-clavulanate".
    return "-".join(part.title() for part in canonical.split("-"))


def english_text(case: Any, field: str, fallback: Any = None) -> Any:
    """Prefer the Normalizer's English translation of a case field.

    `case.translated_text` is populated by the Clinical Normalizer Agent from
    the MCP Sampling `medical_lang_bridge` round-trip.  When a block was
    translated we show that; otherwise we show what was extracted.
    """
    translated = getattr(case, "translated_text", None) or {}
    value = translated.get(field)
    if value:
        return value
    return fallback if fallback is not None else getattr(case, field, None)


def english_medication_rows(medications: list[Any]) -> list[dict[str, Any]]:
    """Medication rows fully in English, for display tables.

    Translates every cell the Clinical Normalizer leaves in the source
    language — drug name, route, duration and remarks — via the display
    lookups in `dashboard.i18n`.  Frequency codes (`TID`, `q6h PRN`) are
    clinical abbreviations the Normalizer already expands, so they are left
    alone here.  Nothing is mutated on the case.
    """
    from i18n import english_duration, english_remark, english_route

    rows: list[dict[str, Any]] = []
    for medication in medications:
        payload = (
            medication if isinstance(medication, dict)
            else medication.model_dump(mode="json")
        )
        row = dict(payload)

        source_name = payload.get("medicine_name") or ""
        display_name = english_drug_name(source_name)
        row["medicine_name"] = display_name
        if display_name.strip().lower() != str(source_name).strip().lower():
            row["as_written"] = source_name

        if payload.get("route"):
            row["route"] = english_route(payload["route"])
        if payload.get("period"):
            row["period"] = english_duration(payload["period"])
        if payload.get("remarks"):
            row["remarks"] = english_remark(payload["remarks"])

        rows.append(row)
    return rows


def english_lab_rows(lab_tests: list[Any]) -> list[dict[str, Any]]:
    """Lab result rows with English test names and flags, for display tables.

    Values, units and reference ranges are numeric/universal and pass through
    untouched — only the test name and the result flag are language-dependent.
    """
    from i18n import english_flag, english_lab_test

    rows: list[dict[str, Any]] = []
    for test in lab_tests:
        payload = test if isinstance(test, dict) else test.model_dump(mode="json")
        row = dict(payload)
        #  The schema field is `test`; some payloads use `test_name`.
        for key in ("test", "test_name"):
            if payload.get(key):
                row[key] = english_lab_test(payload[key])
        if payload.get("flag"):
            row["flag"] = english_flag(payload["flag"])
        rows.append(row)
    return rows


#  Heatmap intensity (0 clean → 3 severe) as returned by the analytics MCP
#  tool, mapped to the clinical palette. Index = `intensity`.
_HEAT_SCALE = [
    ("#eef4f7", PALETTE["muted"]),      # 0 clean    — near-white
    ("#d6e9de", PALETTE["ok_fore"]),    # 1 minor    — soft green
    ("#fbe6c0", PALETTE["warn_fore"]),  # 2 moderate — muted amber
    ("#f2c2c7", PALETTE["crit_fore"]),  # 3 severe   — desaturated red
]


def risk_domain_panel(heatmap: dict[str, Any]) -> None:
    """Per-domain risk heatmap from the `generate_risk_heatmap` MCP payload.

    Rendered with `st.dataframe` + a pandas Styler rather than hand-written
    HTML.  An earlier revision emitted a styled `<div>` grid through
    `st.markdown(unsafe_allow_html=True)` and did not appear on the page at
    all; the Styler path is the same one `findings_table` uses, which is known
    to render here, so the heatmap no longer depends on raw HTML surviving
    Streamlit's markdown pass.

    Falls back to the tool's own `markdown` table, then to a plain table, so
    the panel degrades to *something* visible rather than to nothing.
    """
    cells = heatmap.get("cells") or []

    if not cells:
        #  Say *why* nothing is here rather than rendering an empty panel, and
        #  use whatever the tool did send.
        if heatmap.get("markdown"):
            st.markdown(heatmap["markdown"])
            return
        keys = ", ".join(sorted(heatmap)) if heatmap else "none"
        st.info(
            "The analytics server returned no domain breakdown for this case.  \n"
            f"Payload keys received: `{keys}`"
        )
        return

    import pandas as pd

    worst = heatmap.get("worst_domain")
    rows = []
    for cell in cells:
        label = cell.get("label") or str(cell.get("domain", "")).replace("_", " ").title()
        rows.append(
            {
                "Clinical domain": label,
                "Severity": str(cell.get("severity") or "clean").title(),
                "Score": int(cell.get("score") or 0),
                #  Kept for the row tint, dropped from the rendered columns.
                "_intensity": max(0, min(int(cell.get("intensity") or 0),
                                         len(_HEAT_SCALE) - 1)),
                "_worst": cell.get("domain") == worst and int(cell.get("score") or 0) > 0,
            }
        )

    frame = pd.DataFrame(rows)

    def _tint(row: Any) -> list[str]:
        back, fore = _HEAT_SCALE[int(row["_intensity"])]
        weight = "font-weight:700;" if row["_worst"] else ""
        return [f"background-color:{back};color:{fore};{weight}"] * len(row)

    st.dataframe(
        frame.style.apply(_tint, axis=1),
        width="stretch",
        hide_index=True,
        column_order=("Clinical domain", "Severity", "Score"),
    )

    clean = heatmap.get("clean_domains") or []
    total = heatmap.get("total_score")
    caption = []
    if total is not None:
        caption.append(f"Total {total}")
    if worst:
        caption.append(f"driven by **{str(worst).replace('_', ' ')}**")
    if clean:
        caption.append(f"{len(clean)} of {len(cells)} domains clean")
    if caption:
        st.caption(" · ".join(caption))


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
            "Critical": f"background-color:{PALETTE['crit_back']}",
            "Warning": f"background-color:{PALETTE['warn_back']}",
            "Info": f"background-color:{PALETTE['info_back']}",
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
        f"No validation report for **{patient_id}** yet. Process the case on "
        "**1 · Document Viewer**, or run "
        f"`python -m discharge_ai.cli run {patient_id}` from a terminal."
    )
    if st.button("← Go to 1 · Document Viewer", type="primary", key="no-report-route"):
        goto("documents")
