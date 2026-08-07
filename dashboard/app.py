"""
dashboard/app.py
================

Streamlit HITL Dashboard — entry point (:8501).

This is a **gated flow**, not five freely clickable tabs.  Navigation is built
with `st.navigation` from the pipeline state held in `st.session_state`, so a
view is only reachable once the pipeline has actually put the case there:

    1 · Document Viewer      always — the landing page; runs the pipeline
    2 · Validation Report    once extraction → normalisation → validation ran
    3 · HITL Corrections     once validated (the route when a discharge blocks)
    4 · RAG Q&A              always — queries the FAISS index, not a case
    5 · Discharge Summary    only when blocked=false, or after HITL sign-off

The gating rules live in `common.page_unlocked()`; every gated view also calls
`common.require_page()` so a stale tab can never render a summary for a blocked
discharge.

Run:
    streamlit run dashboard/app.py --server.port 8501
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    configure_page,
    init_state,
    patient_selector,
    register_pages,
    render_nav,
)

configure_page()
init_state()

VIEWS = Path(__file__).resolve().parent / "views"

pages = {
    "documents": st.Page(str(VIEWS / "1_Document_Viewer.py"),
                         title="1 · Document Viewer", icon=":material/description:",
                         default=True),
    "validation": st.Page(str(VIEWS / "2_Validation_Report.py"),
                          title="2 · Validation Report", icon=":material/fact_check:"),
    "corrections": st.Page(str(VIEWS / "3_HITL_Corrections.py"),
                           title="3 · HITL Corrections", icon=":material/edit_note:"),
    "rag": st.Page(str(VIEWS / "4_RAG_QA.py"),
                   title="4 · RAG Q&A", icon=":material/forum:"),
    "summary": st.Page(str(VIEWS / "5_Discharge_Summary.py"),
                       title="5 · Discharge Summary",
                       icon=":material/assignment_turned_in:"),
}
register_pages(pages)

#  `position="hidden"` suppresses Streamlit's own page list: the sidebar shows
#  our gated `render_nav()` instead, which greys out the steps the case has not
#  reached yet.
current = st.navigation(list(pages.values()), position="hidden")

#  Sidebar order: who we are looking at → where we may go.
patient_selector()
render_nav()

current.run()
