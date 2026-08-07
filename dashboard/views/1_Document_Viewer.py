"""
View 1 — Document Viewer  (flow entry point).

Patient selector · tab view (discharge / lab / bill) · language detection badge ·
structured data preview · process trigger.

Pressing *Process* runs extraction → normalisation → validation, folds the
result into the flow state (which unlocks the later steps) and routes straight
to view 2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (  # noqa: E402
    get_case,
    goto,
    language_badge,
    page_setup,
    store_pipeline_result,
    stream_pipeline,
)
from discharge_ai.common.doc_loader import load_patient_documents  # noqa: E402

page_setup(
    "1 · Document Viewer",
    "Inspect what the hospital sent us, then hand the case to the agents.",
)

patient_id = st.session_state.get("patient_id")
if not patient_id:
    st.info("Select a patient in the sidebar to begin.")
    st.stop()

documents = load_patient_documents(patient_id)
case = get_case(patient_id, refresh=st.session_state.pop("force_case_refresh", False))

# --------------------------------------------------------------------------- #
#  Header
# --------------------------------------------------------------------------- #
header = st.columns([2, 1, 1, 1])
header[0].markdown(f"### {case.patient_name or patient_id}")
header[0].markdown(
    "Detected language: " + language_badge(case.detected_language),
    unsafe_allow_html=True,
)
header[1].metric("Documents", len(documents))
header[2].metric("Medications", len(case.medications))
header[3].metric("Lab results", len(case.lab_tests))

if case.extraction_notes:
    with st.expander(f"Extraction notes ({len(case.extraction_notes)})"):
        for note in case.extraction_notes:
            st.markdown(f"- {note}")

st.divider()

# --------------------------------------------------------------------------- #
#  Document tabs
# --------------------------------------------------------------------------- #
TAB_LABELS = {
    "discharge_report": "Discharge report",
    "lab_report": "Lab report",
    "bill": "Hospital bill",
}

present = [key for key in TAB_LABELS if key in documents]
missing = [key for key in TAB_LABELS if key not in documents]

if missing:
    st.warning(
        "Missing document type(s): "
        + ", ".join(TAB_LABELS[key] for key in missing)
    )

tabs = st.tabs([TAB_LABELS[key] for key in present] + ["Structured data"])

for index, doc_type in enumerate(present):
    document = documents[doc_type]
    with tabs[index]:
        info = st.columns(4)
        info[0].caption(f"**File** `{document.name}`")
        info[1].caption(f"**Format** {document.fmt}")
        info[2].caption(f"**OCR used** {'yes' if document.ocr_used else 'no'}")
        info[3].caption(f"**Characters** {len(document.text):,}")

        if document.error:
            st.error(f"Load issue: {document.error}")
        if document.ocr_used:
            st.info(
                "This document came from a scan or a handwritten form. Values were "
                "recovered by OCR and should be spot-checked against the paper."
            )

        if document.data is not None:
            view = st.radio(
                "View", ["Structured JSON", "Rendered text"],
                horizontal=True, key=f"view-{doc_type}",
            )
            if view == "Structured JSON":
                st.json(document.data, expanded=False)
            else:
                st.text_area("Document text", document.text, height=440,
                             key=f"text-{doc_type}")
        else:
            st.text_area("Document text", document.text or "(no extractable text)",
                         height=460, key=f"text-{doc_type}")

# --------------------------------------------------------------------------- #
#  Structured preview
# --------------------------------------------------------------------------- #
with tabs[-1]:
    st.subheader("What the extractor read")

    demographics = {
        "Patient id": case.patient_id,
        "Name": case.patient_name,
        "Age": case.age,
        "Gender": case.gender,
        "Address": case.address or "— NOT RECORDED —",
        "Admission date": case.admission_date,
        "Discharge date": case.discharge_date,
        "Ward / bed": f"{case.ward or '?'} / {case.bed_no or '?'}",
        "Service line": case.service_line,
        "Attending physician": case.attending_physician or "— NOT RECORDED —",
        "Consulting doctors": ", ".join(case.consulting_doctors) or "— NOT RECORDED —",
        "Discharge approved": case.discharge_approved,
        "Approved by": case.discharge_approved_by or "— NOT RECORDED —",
    }

    left, right = st.columns(2)
    with left:
        st.markdown("**Demographics and encounter**")
        st.table({"Value": demographics})
    with right:
        st.markdown("**Diagnosis and allergies**")
        st.write("Discharge diagnosis:")
        for diagnosis in case.discharge_diagnosis or ["— none recorded —"]:
            st.markdown(f"- {diagnosis}")
        st.write("Allergies:")
        for allergy in case.allergies or ["— none documented —"]:
            st.markdown(f"- {allergy}")
        st.write("Follow-up appointment:")
        st.markdown(f"- {case.follow_up_appointment or '— NONE SCHEDULED —'}")

    st.markdown("**Discharge prescriptions**")
    if case.medications:
        st.dataframe(
            [medication.model_dump(mode="json") for medication in case.medications],
            width="stretch", hide_index=True,
        )
    else:
        st.warning("No discharge medications were extracted.")

    st.markdown("**Laboratory results**")
    if case.lab_tests:
        st.dataframe(
            [test.model_dump(mode="json") for test in case.lab_tests],
            width="stretch", hide_index=True,
        )
    else:
        st.caption("No laboratory results were extracted.")

    st.markdown("**Hospital bill**")
    bill = case.bill
    bill_cols = st.columns(4)
    bill_cols[0].metric(
        "Total", f"{bill.total_amount:,.2f}" if bill.total_amount is not None else "—",
        bill.currency or "",
    )
    bill_cols[1].metric("Payment status", bill.payment_status or "unknown")
    bill_cols[2].metric("Settled", "yes" if bill.is_settled else "NO")
    bill_cols[3].metric("Line items", len(bill.line_items))
    if bill.line_items:
        st.dataframe(
            [item.model_dump(mode="json") for item in bill.line_items],
            width="stretch", hide_index=True,
        )

st.divider()

# --------------------------------------------------------------------------- #
#  Process trigger  →  routes to view 2
# --------------------------------------------------------------------------- #
st.subheader("Process this patient")
st.caption(
    "Runs Clinical Extractor → Clinical Normalizer → Clinical Validation. "
    "In **a2a** mode each agent is called over the A2A protocol; in **local** "
    "mode the handlers run in this process. You are taken to **2 · Validation "
    "Report** when it finishes. The discharge summary is a separate step: it "
    "is only generated when a reviewer clicks **Generate summary** on "
    "**5 · Discharge Summary**."
)

controls = st.columns([1, 3])
use_llm = controls[0].checkbox(
    "LLM gap-filling", value=True,
    help="Let the LLM fill only the fields the deterministic parser could not read.",
)

if controls[1].button("▶ Process this patient", type="primary", width="stretch"):
    mode = st.session_state["execution_mode"]
    status_box = st.status(f"Processing {patient_id} in {mode} mode…", expanded=True)
    result = None

    with status_box:
        for kind, payload, message in stream_pipeline(
            patient_id,
            mode=mode,
            generate_summary=False,
            use_llm_extraction=use_llm,
        ):
            if kind == "progress":
                st.markdown(f"**{payload}** — {message}")
            elif kind == "result":
                result = payload

    if result is None or result.report is None:
        errors = result.errors if result is not None else ["the pipeline produced no result"]
        status_box.update(label=f"Processing {patient_id} failed", state="error")
        st.error("Processing failed:\n\n" + "\n".join(f"- {e}" for e in errors))
    else:
        store_pipeline_result(result)
        st.session_state["force_case_refresh"] = False
        status_box.update(
            label=f"{patient_id} processed — opening the validation report",
            state="complete",
        )
        goto("validation")
