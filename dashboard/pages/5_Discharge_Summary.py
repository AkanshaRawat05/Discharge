"""
Page 5 — Discharge Summary.

Patient-friendly summary for auto-approved cases · plain-English prescription
table · colour-coded lab results · export JSON / HTML / PDF · LangFuse trace link.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (  # noqa: E402
    artefact_downloads,
    get_case,
    get_report,
    get_summary,
    guardrail_table,
    no_report_warning,
    page_setup,
    patient_selector,
    risk_badge,
    run_async,
    settings,
    sidebar_status,
    trace_link,
)
from discharge_ai.pipeline import stream_summary  # noqa: E402
from discharge_ai.reporting import render_summary_html  # noqa: E402

page_setup(
    "5 · Discharge Summary",
    "The letter the patient goes home with — plus the exports.",
)
sidebar_status()

patient_id = patient_selector()
if not patient_id:
    st.stop()

case = get_case(patient_id)
report = get_report(patient_id)
summary = get_summary(patient_id)

if report is None:
    no_report_warning(patient_id)
    st.stop()

risk = report.risk

# --------------------------------------------------------------------------- #
#  Gate banner
# --------------------------------------------------------------------------- #
gate = st.columns([2, 1, 1, 1])
gate[0].markdown(f"### {case.patient_name or patient_id}")
with gate[1]:
    st.markdown("**Risk**")
    st.markdown(risk_badge(risk.level.value), unsafe_allow_html=True)
gate[2].metric("Recommendation", risk.recommendation.value)
gate[3].metric("Blocked", "YES" if risk.discharge_blocked else "no")

if risk.discharge_blocked:
    st.error(
        "🛑 **Automatic summary generation is blocked for this discharge.** "
        "Resolve the Critical findings on **2 · Validation Report**, or record an "
        "explicit approval on **3 · HITL Corrections**, before releasing a summary "
        "to the patient."
    )
elif risk.hitl_required:
    st.warning(
        "⚠️ This case requires human review. A summary can be generated, but a "
        "clinician must sign it off before it goes to the patient."
    )
else:
    st.success("✅ Auto-approved — this summary is cleared for release.")

st.divider()

# --------------------------------------------------------------------------- #
#  Generate / regenerate
# --------------------------------------------------------------------------- #
controls = st.columns([1, 1, 1, 2])
audience = controls[0].selectbox("Audience", ["patient", "clinician"], key="audience")
force = controls[1].checkbox(
    "Reviewer override", value=False,
    help="Generate the summary even though validation blocked the discharge.",
)
generate = controls[2].button(
    "✍️ Generate summary", type="primary", width="stretch",
    disabled=risk.discharge_blocked and not force,
)
controls[3].caption(
    "The Summary Generator streams section by section over A2A: "
    "patient → medicines → labs → bill → instructions."
)

if generate:
    stream_box = st.empty()
    section_log: list[str] = []
    final_payload: dict | None = None

    async def _run() -> None:
        global final_payload
        async for section, text, payload in stream_summary(
            patient_id, case, report,
            mode=st.session_state["execution_mode"],
            force=force,
            trace_id=report.trace_id,
        ):
            if payload is not None:
                final_payload = payload
            elif section:
                section_log.append(section)

    with st.spinner("Streaming the discharge summary…"):
        run_async(_run())
        stream_box.markdown(
            "Streamed sections: " + " → ".join(f"`{s}`" for s in section_log)
        )

    if final_payload and final_payload.get("generated"):
        from discharge_ai.common.schemas import DischargeSummary

        summary = DischargeSummary.model_validate(final_payload["summary"])
        st.session_state["summaries"][patient_id] = summary
        st.session_state["artefacts"].setdefault(patient_id, {}).update(
            final_payload.get("artefacts", {})
        )
        st.success(
            f"Generated {len(summary.sections)} section(s)."
            + (" (reviewer override)" if final_payload.get("forced") else "")
        )
    elif final_payload:
        st.error(
            "Summary generation was refused:\n\n"
            + "\n".join(f"- {r}" for r in final_payload.get("blocking_reasons", []))
        )

if summary is None:
    st.info("No summary has been generated for this patient yet.")
    st.stop()

st.divider()

# --------------------------------------------------------------------------- #
#  The summary itself
# --------------------------------------------------------------------------- #
view = st.radio(
    "View", ["Readable", "Rendered HTML", "Raw JSON"], horizontal=True, key="summary_view"
)

if view == "Rendered HTML":
    components.html(render_summary_html(summary), height=1100, scrolling=True)

elif view == "Raw JSON":
    st.json(summary.model_dump(mode="json"), expanded=False)

else:
    for section in summary.sections:
        if section.key == "warning_signs":
            st.markdown(f"#### 🚨 {section.title}")
            st.error(section.content)
        elif section.key == "follow_up":
            st.markdown(f"#### {section.title}")
            st.info(section.content)
        else:
            st.markdown(f"#### {section.title}")
            st.markdown(section.content)

    # ---- plain-English prescription table ------------------------------
    if summary.prescription_table:
        st.markdown("#### 💊 Your medicines")
        import pandas as pd

        frame = pd.DataFrame(
            [
                {
                    "#": row.get("sl_no"),
                    "Medicine": row.get("medicine_name"),
                    "Strength": row.get("strength"),
                    "How much": row.get("dosage"),
                    "How often": row.get("frequency_plain") or row.get("frequency"),
                    "How to take": row.get("route_plain") or row.get("route"),
                    "For how long": row.get("period"),
                    "Notes": row.get("remarks"),
                }
                for row in summary.prescription_table
            ]
        )
        st.dataframe(frame, width="stretch", hide_index=True)

    # ---- colour-coded labs ---------------------------------------------
    if summary.lab_table:
        st.markdown("#### 🧪 Your test results")
        import pandas as pd

        frame = pd.DataFrame(
            [
                {
                    "Test": row.get("test"),
                    "Result": row.get("value"),
                    "Units": row.get("unit"),
                    "Normal range": row.get("reference_range"),
                    "Status": row.get("flag") or "NORMAL",
                }
                for row in summary.lab_table
            ]
        )

        def _colour(row: object) -> list[str]:
            status = str(row["Status"]).upper()  # type: ignore[index]
            if status in {"NORMAL", "NORMAAL", ""}:
                return ["background-color:#e9f6ee"] * len(row)  # type: ignore[arg-type]
            return ["background-color:#fdecee"] * len(row)  # type: ignore[arg-type]

        st.dataframe(
            frame.style.apply(_colour, axis=1), width="stretch", hide_index=True
        )
        abnormal = [
            row for row in summary.lab_table
            if str(row.get("flag", "")).upper() not in {"NORMAL", "NORMAAL", ""}
        ]
        if abnormal:
            st.caption(
                f"🔴 {len(abnormal)} result(s) outside the normal range — discuss "
                "these at your follow-up appointment."
            )

    # ---- bill -----------------------------------------------------------
    if summary.bill_snapshot:
        st.markdown("#### 🧾 Your hospital bill")
        bill = summary.bill_snapshot
        bill_columns = st.columns(4)
        total = bill.get("total_amount")
        bill_columns[0].metric(
            "Total", f"{total:,.2f}" if isinstance(total, (int, float)) else "—",
            bill.get("currency") or "",
        )
        bill_columns[1].metric("Payment status", bill.get("payment_status") or "unknown")
        bill_columns[2].metric("Settled", "yes" if bill.get("settled") else "NO")
        bill_columns[3].metric("Items", len(bill.get("line_items") or []))

        if bill.get("line_items"):
            st.dataframe(bill["line_items"], width="stretch", hide_index=True)
        if not bill.get("settled"):
            st.warning(
                "This bill is not settled. Hospital policy requires settlement or "
                "an insurance guarantee letter before release."
            )

st.divider()

# --------------------------------------------------------------------------- #
#  Guardrails + export
# --------------------------------------------------------------------------- #
st.subheader("Guardrails applied to this summary")
guardrail_table(summary.guardrail_events)
st.caption(
    "Every generated section passes the toxicity filter, PII redaction and a "
    "grounding check against the validated case data before it is shown."
)

st.subheader("Export")
artefact_downloads(patient_id)

extra = st.columns(2)
with extra[0]:
    st.download_button(
        "Markdown",
        data=summary.as_markdown(),
        file_name=f"{patient_id}_summary.md",
        mime="text/markdown",
        width="stretch",
    )
with extra[1]:
    st.download_button(
        "Summary JSON (in-memory)",
        data=json.dumps(summary.model_dump(mode="json"), indent=2, default=str),
        file_name=f"{patient_id}_summary_session.json",
        mime="application/json",
        width="stretch",
    )

trace_link(summary.trace_id or report.trace_id)
st.caption(f"Artefacts live in `{settings.path('reports_dir')}`.")
