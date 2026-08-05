"""
dashboard/app.py
================

Streamlit HITL Dashboard — landing page (:8501).

The five review pages live in `dashboard/pages/` and Streamlit puts them in the
sidebar automatically:

    1 · Document Viewer      per-patient documents, language badges, process trigger
    2 · Validation Report    completeness, cross-validation issues, risk, trace link
    3 · HITL Corrections     editable meds, elicitation form, override, re-run
    4 · RAG Q&A              streaming answers, sources, RAG Triad metrics
    5 · Discharge Summary     patient-friendly summary + JSON/HTML/PDF export

Run:
    streamlit run dashboard/app.py --server.port 8501
"""

from __future__ import annotations

import streamlit as st

from common import (
    available_patients,
    page_setup,
    settings,
    sidebar_status,
    tracing,
)

page_setup(
    "Discharge Review Console",
    "Human-in-the-loop review for the agentic discharge pipeline.",
)
sidebar_status()

patients = available_patients()

# --------------------------------------------------------------------------- #
#  Overview
# --------------------------------------------------------------------------- #
st.markdown(
    f"""
Welcome to the **{settings.cfg['project']['hospital']}** discharge review console.

This dashboard is the human-in-the-loop surface for a multi-agent pipeline that
ingests discharge documentation in several formats and languages, validates it
against the hospital EHR and `configs/rules.yaml`, scores discharge risk, and
generates a patient-friendly summary for cases that clear review.
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
#  Page guide
# --------------------------------------------------------------------------- #
left, right = st.columns([3, 2])

with left:
    st.subheader("The five review pages")
    st.markdown(
        """
| Page | What you do there |
| --- | --- |
| **1 · Document Viewer** | Inspect the discharge report, lab report and bill for a patient; see the detected language; trigger processing. |
| **2 · Validation Report** | Completeness score, cross-validation issues, risk level, recommendation, discharge-blocked indicator, LangFuse trace. |
| **3 · HITL Corrections** | Edit the medication table, fill the MCP elicitation form, override the risk label, record your approval decision, re-run validation. |
| **4 · RAG Q&A** | Ask questions about the indexed records with streaming answers, source documents and RAG Triad quality metrics. |
| **5 · Discharge Summary** | Read the patient-friendly summary, the plain-English prescription table and colour-coded labs; export JSON / HTML / PDF. |
"""
    )

    st.subheader("Suggested walkthrough")
    st.markdown(
        """
1. **P1019** or **P1023** — fully reconciled records. Low risk, auto-approved, a
   summary is generated straight away.
2. **P1020** — Spanish PDF whose only gap is a missing address (a soft
   demographic weight), so it still auto-approves.
3. **P1021** — Hindi record with an unpaid bill, no follow-up and no address;
   escalates to human review.
4. **P1022** / **P1024** — Dutch records that prescribe *Amoxicilline* to a
   patient with a **penicillin allergy** on file. Critical: discharge blocked,
   summary generation refused until a reviewer intervenes.
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
