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
    english_lab_rows,
    english_medication_rows,
    english_text,
    get_case,
    goto,
    language_badge,
    page_setup,
    store_pipeline_result,
    stream_pipeline,
)
from uploads import render_upload_panel  # noqa: E402

from discharge_ai.common.doc_loader import load_patient_documents  # noqa: E402
from discharge_ai.common.terminology import language_name  # noqa: E402

page_setup(
    "1 · Document Viewer",
    "Inspect what the hospital sent us, then hand the case to the agents.",
)

#  Above the patient check on purpose: uploading is how the first patient gets
#  into an empty incoming folder.
render_upload_panel()

patient_id = st.session_state.get("patient_id")
if not patient_id:
    st.info("Select a patient in the sidebar, or upload documents above to begin.")
    st.stop()

documents = load_patient_documents(patient_id)
case = get_case(patient_id, refresh=st.session_state.pop("force_case_refresh", False))

# --------------------------------------------------------------------------- #
#  Header
# --------------------------------------------------------------------------- #
with st.container(border=True):
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

#  Document type → the key the Clinical Normalizer stores its English
#  translation under on `case.translated_text`. Only the discharge narrative is
#  translated; lab reports and bills are numeric/structured.
TRANSLATED_BLOCK = {"discharge_report": "narrative"}

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

        #  English-first preview. The Clinical Normalizer translates the
        #  discharge narrative through MCP Sampling and stores it on the case;
        #  when that exists for a non-English record we show it by default and
        #  keep the source text one click away, because the original document
        #  is still the evidence a reviewer reconciles against.
        english_narrative = (
            (case.translated_text or {}).get(TRANSLATED_BLOCK.get(doc_type, ""))
            if case.detected_language != "en" else None
        )

        options: list[str] = []
        if english_narrative:
            options.append("English translation")
        if document.data is not None:
            options.append("Structured JSON")
        options.append("Source text" if english_narrative else "Document text")

        #  Widget keys MUST carry the patient id. A key that is stable across
        #  patients keeps its previous session_state value on rerun, which is
        #  how one patient's document text used to bleed into another's tab.
        view = (
            st.radio("View", options, horizontal=True,
                     key=f"view-{patient_id}-{doc_type}")
            if len(options) > 1 else options[0]
        )

        if view == "English translation":
            st.text_area("Discharge record (translated to English)",
                         english_narrative, height=440,
                         key=f"en-{patient_id}-{doc_type}", disabled=True)
            st.caption(
                f"Translated from {language_name(case.detected_language)} by the "
                "Clinical Normalizer. Switch to **Source text** to see the "
                "original wording."
            )
        elif view == "Structured JSON":
            st.json(document.data, expanded=False)
        else:
            st.text_area(view, document.text or "(no extractable text)",
                         height=440 if document.data is not None else 460,
                         key=f"text-{patient_id}-{doc_type}", disabled=True)

# --------------------------------------------------------------------------- #
#  Structured preview
# --------------------------------------------------------------------------- #
with tabs[-1]:
    st.subheader("What the extractor read")

    NOT_RECORDED = "— NOT RECORDED —"

    def _display(value: object) -> str:
        """One column, one type.

        `st.table` hands the column straight to PyArrow, which needs a single
        type per column. Mixing `int` (age), `bool` (discharge approved) and
        `str` in the same column raises ArrowTypeError and forces Streamlit
        into its slow "automatic fixes" fallback, so everything is rendered as
        text here.
        """
        if value is None or value == "":
            return NOT_RECORDED
        if isinstance(value, bool):
            return "Yes" if value else "No"
        return str(value)

    demographics = {
        "Patient id": _display(case.patient_id),
        "Name": _display(case.patient_name),
        "Age": _display(case.age),
        "Gender": _display(case.gender),
        "Address": _display(case.address),
        "Admission date": _display(case.admission_date),
        "Discharge date": _display(case.discharge_date),
        "Ward / bed": f"{case.ward or '?'} / {case.bed_no or '?'}",
        "Service line": _display(case.service_line),
        "Attending physician": _display(case.attending_physician),
        "Consulting doctors": _display(", ".join(case.consulting_doctors)),
        "Discharge approved": _display(case.discharge_approved),
        "Approved by": _display(case.discharge_approved_by),
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
        st.markdown(
            f"- {english_text(case, 'follow_up_appointment') or '— NONE SCHEDULED —'}"
        )

    st.markdown("**Discharge prescriptions**")
    if case.medications:
        #  Drug names are shown in English; `as_written` preserves the source
        #  spelling so a reviewer can still reconcile against the paper.
        st.dataframe(
            english_medication_rows(case.medications),
            width="stretch", hide_index=True,
        )
    else:
        st.warning("No discharge medications were extracted.")

    st.markdown("**Laboratory results**")
    if case.lab_tests:
        #  Test names and flags shown in English; values/units are universal.
        st.dataframe(
            english_lab_rows(case.lab_tests),
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
    "Extracts the record, normalises it to English, and validates it against "
    "the hospital EHR and care plan. You are taken to **2 · Validation Report** "
    "when it finishes. The discharge summary is a separate step — it is only "
    "generated when a reviewer clicks **Generate summary** on "
    "**5 · Discharge Summary**."
)

if st.button("▶ Process this patient", type="primary"):
    mode = st.session_state["execution_mode"]
    status_box = st.status(f"Processing {patient_id}…", expanded=True)
    result = None

    with status_box:
        #  LLM gap-filling is always on: it only fills fields the deterministic
        #  parser could not read and can never overwrite a value read verbatim
        #  from the source document, so there is nothing for a reviewer to
        #  decide here.
        for kind, payload, message in stream_pipeline(
            patient_id,
            mode=mode,
            generate_summary=False,
            use_llm_extraction=True,
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
