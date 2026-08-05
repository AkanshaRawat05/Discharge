"""
Overview — the console landing view.

Shows where every patient currently sits in the gated flow, plus service status.
Not a pipeline step: it is always reachable and never gates anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (  # noqa: E402
    STAGE_NEW,
    available_patients,
    goto,
    hydrate_flow,
    page_setup,
    settings,
    tracing,
)

page_setup(
    "Discharge Review Console",
    "Human-in-the-loop review for the agentic discharge pipeline.",
)

patients = available_patients()

st.markdown(
    f"""
Welcome to the **{settings.cfg['project']['hospital']}** discharge review console.

This dashboard is the human-in-the-loop surface for a multi-agent pipeline that
ingests discharge documentation in several formats and languages, validates it
against the hospital EHR and `configs/rules.yaml`, scores discharge risk, and
generates a patient-friendly summary for cases that clear review.

Each patient moves through the flow one step at a time — the steps a case has
not reached yet stay locked in the sidebar.
"""
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Patients with documents", len(patients))
reports_dir = settings.path("reports_dir")
col2.metric("Audit reports generated", len(list(reports_dir.glob("*_audit.json"))))
col3.metric("Summaries generated", len(list(reports_dir.glob("*_summary.json"))))
col4.metric("LangFuse tracing", "on" if tracing.langfuse_available() else "off")

st.divider()

# --------------------------------------------------------------------------- #
#  Worklist — where every patient sits in the flow
# --------------------------------------------------------------------------- #
st.subheader("Worklist")

if not patients:
    st.error(
        f"No patient documents found under `{settings.path('input_root')}`. "
        "Add discharge/lab/bill files named with a patient id (e.g. `P1019_…`)."
    )
    st.stop()

STATUS = {
    "new": ("⚪", "Not processed", "documents", "Process"),
    "blocked": ("🛑", "Discharge BLOCKED", "corrections", "Resolve in HITL"),
    "signed_off": ("🔏", "Blocked · signed off", "summary", "View summary"),
    "hitl": ("⚠️", "Human review required", "validation", "Review"),
    "cleared": ("✅", "Cleared for release", "summary", "View summary"),
}


def _status_key(state: dict) -> str:
    if state["stage"] == STAGE_NEW:
        return "new"
    if state["blocked"]:
        return "signed_off" if state["approved"] else "blocked"
    return "hitl" if state["hitl_required"] else "cleared"


header = st.columns([1, 2, 1, 1, 1.4])
for column, label in zip(
    header, ["Patient", "Status", "Risk", "Findings", ""]
):
    column.markdown(f"**{label}**")

for patient in patients:
    state = hydrate_flow(patient)
    key = _status_key(state)
    icon, label, target, action = STATUS[key]

    row = st.columns([1, 2, 1, 1, 1.4])
    row[0].markdown(f"**{patient}**")
    row[1].markdown(f"{icon} {label}")
    row[2].markdown(state["risk_level"] or "—")
    row[3].markdown(str(state["findings"]) if state["stage"] != STAGE_NEW else "—")
    if row[4].button(action, key=f"worklist-{patient}", width="stretch"):
        st.session_state["patient_id"] = patient
        goto(target)

st.divider()

# --------------------------------------------------------------------------- #
#  Flow guide + service status
# --------------------------------------------------------------------------- #
left, right = st.columns([3, 2])

with left:
    st.subheader("The review flow")
    st.markdown(
        """
| Step | What you do there | Unlocked when |
| --- | --- | --- |
| **1 · Document Viewer** | Inspect the discharge report, lab report and bill; see the detected language; run the pipeline. | Always — this is the entry point. |
| **2 · Validation Report** | Completeness, cross-validation issues, risk, and the release decision. | The pipeline has validated the case. |
| **3 · HITL Corrections** | Edit medications, answer the MCP elicitation form, override the risk label, re-run validation. | The pipeline has validated the case. |
| **4 · RAG Q&A** | Ask questions about the indexed records with streaming answers and citations. | Always — it queries the index, not a case. |
| **5 · Discharge Summary** | The patient letter, prescription table, labs, and JSON/HTML/PDF export. | The discharge is **not blocked**, or a reviewer signed it off. |
"""
    )

    st.subheader("Suggested walkthrough")
    st.markdown(
        """
1. **P1019** or **P1023** — fully reconciled records. Low risk, auto-approved, a
   summary is generated straight away and step 5 unlocks immediately.
2. **P1020** — Spanish PDF whose only gap is a missing address (a soft
   demographic weight), so it still auto-approves.
3. **P1021** — Hindi record with an unpaid bill, no follow-up and no address;
   escalates to human review.
4. **P1022** / **P1024** — Dutch records that prescribe *Amoxicilline* to a
   patient with a **penicillin allergy** on file. Critical: discharge blocked,
   step 5 stays locked until a reviewer intervenes on step 3.
"""
    )

with right:
    st.subheader("Service status")
    import socket

    def _listening(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.25)
            return sock.connect_ex(("127.0.0.1", port)) == 0

    services = [
        ("Mock EHR", 8050),
        ("MCP Clinical Tools", 8200),
        ("MCP Analytics", 8201),
        ("Extractor (A2A)", 8100),
        ("Validator (A2A)", 8101),
        ("Normalizer (A2A)", 8102),
        ("Monitor (A2A)", 8103),
        ("Summary Generator (A2A)", 8104),
        ("RAG Q&A (A2A)", 8105),
        ("Host Orchestrator", 8083),
    ]
    for label, port in services:
        up = _listening(port)
        st.markdown(
            f"{'🟢' if up else '⚪'} **{label}** · `:{port}`"
            + ("" if up else " — not running")
        )

    st.caption(
        "The dashboard works with only the core services running: it falls back "
        "to in-process agent execution. Start everything with "
        "`python run_services.py`."
    )

st.divider()
st.caption(
    "This system is clinical decision support. Every High-risk or blocked "
    "discharge requires human review before release — it is never auto-approved."
)
