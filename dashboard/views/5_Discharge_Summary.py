"""
View 5 — Discharge Summary.

Reachable only once the case is approved: the discharge is not blocked, or a
reviewer signed it off on view 3.  `require_page("summary")` enforces that even
if the view is reached from a stale tab.

Patient-friendly summary · plain-English prescription table · colour-coded lab
results · export JSON / HTML / PDF · LangFuse trace link.  Rendering here is
deliberately static — the streaming touchpoints are views 3 and 4.
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
    english_drug_name,
    flow_state,
    get_case,
    get_report,
    get_summary,
    goto,
    guardrail_table,
    no_report_warning,
    page_setup,
    require_page,
    risk_badge,
    run_async,
    settings,
    trace_link,
)
from discharge_ai.pipeline import stream_summary  # noqa: E402
from discharge_ai.reporting import render_summary_html  # noqa: E402

page_setup(
    "5 · Discharge Summary",
    "The letter the patient goes home with — plus the exports.",
)

patient_id = require_page("summary")

case = get_case(patient_id)
report = get_report(patient_id)
summary = get_summary(patient_id)

if report is None:
    no_report_warning(patient_id)
    st.stop()

risk = report.risk
state = flow_state(patient_id)

#  Reaching this view means the case is releasable. A blocked case can only be
#  here on a recorded reviewer sign-off, which is what authorises the override
#  the pipeline needs to generate at all.
released_under_override = bool(risk.discharge_blocked and state["approved"])

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

if released_under_override:
    st.error(
        f"🔏 **Released under a reviewer override.** This discharge is still "
        f"blocked by a Critical finding; **{state['signed_off_by']}** signed it "
        "off on **3 · HITL Corrections**. The override is recorded in the audit "
        "trail and the summary must not go to the patient without that "
        "clinician's confirmation."
    )
elif risk.hitl_required:
    st.warning(
        "⚠️ This case requires human review. The summary can be generated, but a "
        "clinician must sign it off before it goes to the patient."
    )
else:
    st.success("✅ Auto-approved — this summary is cleared for release.")

st.divider()

# --------------------------------------------------------------------------- #
#  Generate / regenerate
# --------------------------------------------------------------------------- #
controls = st.columns([1, 3])
generate = controls[0].button("✍️ Generate summary", type="primary", width="stretch")
controls[1].caption(
    "The Summary Generator streams section by section over A2A: "
    "patient → medicines → labs → bill → instructions."
    + (
        "  \nGeneration runs with the reviewer override recorded on view 3."
        if released_under_override else ""
    )
)

if generate:
    section_log: list[str] = []
    box: dict = {"payload": None}

    async def _run() -> None:
        async for section, _text, payload in stream_summary(
            patient_id, case, report,
            mode=st.session_state["execution_mode"],
            force=released_under_override,
            trace_id=report.trace_id,
        ):
            if payload is not None:
                box["payload"] = payload
            elif section:
                section_log.append(section)

    with st.spinner("Generating the discharge summary…"):
        run_async(_run())

    if section_log:
        st.caption("Sections written: " + " → ".join(f"`{s}`" for s in section_log))

    final_payload = box["payload"]
    if final_payload and final_payload.get("generated"):
        from discharge_ai.common.schemas import DischargeSummary

        summary = DischargeSummary.model_validate(final_payload["summary"])
        st.session_state["summaries"][patient_id] = summary
        st.session_state["artefacts"].setdefault(patient_id, {}).update(
            final_payload.get("artefacts", {})
        )
        state["summary_ready"] = True
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
    if st.button("← Back to 2 · Validation Report"):
        goto("validation")
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
                    "Medicine": english_drug_name(row.get("medicine_name")),
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
