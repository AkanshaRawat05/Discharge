"""
Page 3 — HITL Corrections.

Editable medication table (`st.data_editor`) · **Elicitation Response Form**
(dynamic, schema-driven — this page is the MCP `elicitation_callback`) · risk
label override · approval decision · save feedback · re-run validation.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (  # noqa: E402
    findings_table,
    get_case,
    get_report,
    no_report_warning,
    page_setup,
    patient_selector,
    risk_badge,
    run_async,
    settings,
    sidebar_status,
    store_pipeline_result,
    trace_link,
)
from discharge_ai.common.schemas import Medication  # noqa: E402
from discharge_ai.pipeline import run_pipeline  # noqa: E402
from discharge_ai.validation import elicitation_schema_for  # noqa: E402

page_setup(
    "3 · HITL Corrections",
    "Correct the record, answer the elicitation request, and record your decision.",
)
sidebar_status()

patient_id = patient_selector()
if not patient_id:
    st.stop()

case = get_case(patient_id)
report = get_report(patient_id)
if report is None:
    no_report_warning(patient_id)
    st.stop()

risk = report.risk

# --------------------------------------------------------------------------- #
#  Context banner
# --------------------------------------------------------------------------- #
banner = st.columns([2, 1, 1, 1])
banner[0].markdown(f"### {case.patient_name or patient_id}")
with banner[1]:
    st.markdown("**Current risk**")
    st.markdown(risk_badge(risk.level.value), unsafe_allow_html=True)
banner[2].metric("Score", risk.score)
banner[3].metric("Blocked", "YES" if risk.discharge_blocked else "no")

if risk.hard_guardrails_hit:
    st.error(
        "Hard guardrails triggered — this case can never auto-approve: "
        + ", ".join(risk.hard_guardrails_hit)
    )

with st.expander(f"Outstanding findings ({len(report.findings)})", expanded=True):
    findings_table(report)

st.divider()

# --------------------------------------------------------------------------- #
#  1. Editable medication table
# --------------------------------------------------------------------------- #
st.subheader("1 · Medication table")
st.caption(
    "Correct any dose, frequency, route or duration that the extractor misread. "
    "Edits are applied to the case before validation is re-run."
)

MED_COLUMNS = [
    "sl_no", "medicine_name", "strength", "dosage", "frequency", "route",
    "period", "remarks", "total_quantity",
]

med_rows = [
    {column: getattr(medication, column, None) for column in MED_COLUMNS}
    for medication in case.medications
] or [{column: None for column in MED_COLUMNS}]

edited_meds = st.data_editor(
    med_rows,
    column_config={
        "sl_no": st.column_config.NumberColumn("#", width="small"),
        "medicine_name": st.column_config.TextColumn("Medicine", required=False),
        "strength": st.column_config.TextColumn("Strength"),
        "dosage": st.column_config.TextColumn("Dose"),
        "frequency": st.column_config.TextColumn("Frequency"),
        "route": st.column_config.TextColumn("Route"),
        "period": st.column_config.TextColumn("Duration"),
        "remarks": st.column_config.TextColumn("Remarks", width="medium"),
        "total_quantity": st.column_config.TextColumn("Qty"),
    },
    num_rows="dynamic",
    width="stretch",
    hide_index=True,
    key=f"med_editor_{patient_id}",
)

st.divider()

# --------------------------------------------------------------------------- #
#  2. MCP Elicitation Response Form  (dynamic, schema-driven)
# --------------------------------------------------------------------------- #
st.subheader("2 · Elicitation Response Form")

non_blocking = [field for field in report.completeness.missing_fields if not field.blocking]

if not non_blocking:
    st.success("No non-blocking fields are missing — nothing to elicit.")
    elicitation_answers: dict[str, object] = {}
    elicitation_action = "not_requested"
else:
    #  The schema the Rules Engine tool sends over MCP is what we render.
    schema = report.elicitation.schema_sent
    if not schema:
        _model, schema = elicitation_schema_for(non_blocking)

    properties = schema.get("properties", {}) or {}
    st.caption(
        f"The Clinical Rules Engine tool requested **{len(properties)}** value(s) "
        "through the MCP Elicitation primitive. This form is the "
        "`elicitation_callback`: your response is returned to the MCP server as "
        "an `ElicitResult` with action `accept`, `decline` or `cancel`."
    )

    with st.expander("Schema received from the MCP server"):
        st.json(schema, expanded=False)

    elicitation_answers = {}
    columns = st.columns(2)
    for index, (key, spec) in enumerate(properties.items()):
        description = spec.get("description", key)
        #  Optional fields come through as anyOf[type, null].
        types = [
            entry.get("type") for entry in spec.get("anyOf", [])
            if isinstance(entry, dict)
        ] or [spec.get("type", "string")]
        field_type = next((t for t in types if t and t != "null"), "string")

        with columns[index % 2]:
            if field_type == "integer":
                value = st.number_input(
                    description, min_value=0, max_value=130, step=1, value=None,
                    key=f"elicit-{patient_id}-{key}", placeholder="leave blank if unknown",
                )
            elif field_type == "number":
                value = st.number_input(
                    description, value=None, key=f"elicit-{patient_id}-{key}",
                    placeholder="leave blank if unknown",
                )
            elif field_type == "boolean":
                choice = st.selectbox(
                    description, ["(unknown)", "Yes", "No"],
                    key=f"elicit-{patient_id}-{key}",
                )
                value = None if choice == "(unknown)" else (choice == "Yes")
            else:
                value = st.text_input(
                    description, value="", key=f"elicit-{patient_id}-{key}",
                    placeholder="leave blank if unknown",
                )
            if value not in (None, ""):
                elicitation_answers[key] = value

    elicitation_action = st.radio(
        "Your response to the elicitation request",
        ["accept", "decline", "cancel"],
        horizontal=True,
        key=f"elicit-action-{patient_id}",
        help=(
            "accept — send the values above and continue validation · "
            "decline — leave the fields unresolved and flag for HITL · "
            "cancel — abort validation and escalate to a senior clinician"
        ),
    )
    if elicitation_action == "accept" and not elicitation_answers:
        st.warning("Accepting with no values behaves like a decline.")

st.divider()

# --------------------------------------------------------------------------- #
#  3. Risk override + approval decision
# --------------------------------------------------------------------------- #
st.subheader("3 · Reviewer decision")

decision_columns = st.columns([1, 1, 2])

with decision_columns[0]:
    risk_override = st.selectbox(
        "Risk label override",
        ["(keep computed)", "Low", "Medium", "High"],
        key=f"risk-override-{patient_id}",
        help="Overrides the computed tier. Recorded in the audit trail with your note.",
    )

with decision_columns[1]:
    approval = st.radio(
        "Approval decision",
        ["Pending", "Approve", "Approve with edits", "Reject"],
        key=f"approval-{patient_id}",
    )

with decision_columns[2]:
    reviewer = st.text_input(
        "Reviewer name / id", key=f"reviewer-{patient_id}",
        placeholder="e.g. Dr. R. Greene (staff #4471)",
    )
    note = st.text_area(
        "Clinical note", key=f"note-{patient_id}", height=90,
        placeholder="Why you are approving, editing or rejecting this discharge.",
    )

if approval == "Approve" and risk.discharge_blocked:
    st.error(
        "This discharge is **blocked** by a Critical finding. Approving it "
        "overrides a hard safety guardrail — document your justification in the "
        "clinical note. The override is recorded in the audit trail."
    )

st.divider()

# --------------------------------------------------------------------------- #
#  4. Save feedback / re-run validation
# --------------------------------------------------------------------------- #
st.subheader("4 · Save and re-run")

def _feedback_payload() -> dict:
    return {
        "patient_id": patient_id,
        "reviewer": reviewer or "unnamed reviewer",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "approval_decision": approval,
        "risk_override": None if risk_override == "(keep computed)" else risk_override,
        "computed_risk_level": risk.level.value,
        "computed_risk_score": risk.score,
        "discharge_blocked": risk.discharge_blocked,
        "overrode_block": bool(approval == "Approve" and risk.discharge_blocked),
        "clinical_note": note,
        "elicitation_action": elicitation_action,
        "elicitation_answers": elicitation_answers,
        "medication_edits": edited_meds,
        "trace_id": report.trace_id,
    }


def _apply_medication_edits() -> int:
    """Write the edited table back onto the cached case."""
    updated: list[Medication] = []
    for index, row in enumerate(edited_meds, start=1):
        if not str(row.get("medicine_name") or "").strip():
            continue
        payload = {key: row.get(key) for key in MED_COLUMNS}
        try:
            payload["sl_no"] = int(payload.get("sl_no") or index)
        except (TypeError, ValueError):
            payload["sl_no"] = index
        payload = {
            key: (str(value).strip() if isinstance(value, str) else value)
            for key, value in payload.items()
        }
        updated.append(Medication(**payload))

    if updated:
        case.medications = updated
        st.session_state["cases"][patient_id] = case
    return len(updated)


buttons = st.columns([1, 1, 2])

if buttons[0].button("💾 Save feedback", width="stretch"):
    payload = _feedback_payload()
    count = _apply_medication_edits()
    payload["medication_rows_applied"] = count

    directory = settings.path("feedback_dir")
    path = directory / f"{patient_id}_feedback.json"
    history = []
    if path.exists():
        try:
            history = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(history, dict):
                history = [history]
        except Exception:  # noqa: BLE001
            history = []
    history.append(payload)
    path.write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")

    st.session_state["feedback"][patient_id] = payload
    st.success(
        f"Feedback saved to `{path.name}` ({count} medication row(s) applied to "
        "the working case)."
    )

if buttons[1].button("🔁 Re-run validation", type="primary", width="stretch"):
    _apply_medication_edits()

    kwargs: dict = {}
    if elicitation_action == "accept" and elicitation_answers:
        kwargs["elicitation_answers"] = elicitation_answers
    elif elicitation_action == "decline":
        kwargs["elicitation_decline"] = True
    elif elicitation_action == "cancel":
        kwargs["elicitation_cancel"] = True

    force = approval in {"Approve", "Approve with edits"}

    progress_box = st.empty()
    lines: list[str] = []

    def _progress(stage: str, message: str) -> None:
        lines.append(f"**{stage}** — {message}")
        progress_box.markdown("\n\n".join(lines[-8:]))

    with st.spinner("Re-running validation with your corrections…"):
        result = run_async(
            run_pipeline(
                patient_id,
                mode=st.session_state["execution_mode"],
                generate_summary=True,
                force_summary=force,
                use_llm_extraction=False,     # keep the reviewer's values verbatim
                trace_id=report.trace_id,
                progress=_progress,
                **kwargs,
            )
        )

    store_pipeline_result(result)

    if result.report is None:
        st.error("Re-run failed:\n\n" + "\n".join(f"- {e}" for e in result.errors))
    else:
        new_risk = result.report.risk
        st.success(
            f"Re-validated: risk **{new_risk.level.value}** (score {new_risk.score}), "
            f"{len(result.report.findings)} finding(s), "
            f"elicitation outcome **{result.report.elicitation.outcome.value}**."
        )
        delta = new_risk.score - risk.score
        if delta:
            st.metric("Risk score change", new_risk.score, delta=delta, delta_color="inverse")
        if result.summary is not None:
            st.info("A discharge summary was generated — see **5 · Discharge Summary**.")
        trace_link(result.report.trace_id)

with buttons[2]:
    st.caption(
        "**Re-run** replays the MCP Elicitation round-trip with your chosen "
        "action, so `accept` / `decline` / `cancel` are all exercised for real "
        "against the Clinical Rules Engine tool on the primary MCP server."
    )

# --------------------------------------------------------------------------- #
#  Feedback history
# --------------------------------------------------------------------------- #
history_path = settings.path("feedback_dir") / f"{patient_id}_feedback.json"
if history_path.exists():
    st.divider()
    st.subheader("Reviewer history")
    try:
        entries = json.loads(history_path.read_text(encoding="utf-8"))
        entries = entries if isinstance(entries, list) else [entries]
        for entry in reversed(entries[-5:]):
            st.markdown(
                f"- **{entry.get('timestamp')}** — {entry.get('reviewer')}: "
                f"{entry.get('approval_decision')}"
                + (f" (override → {entry['risk_override']})" if entry.get("risk_override") else "")
                + (f" · elicitation: {entry.get('elicitation_action')}")
                + (f"\n\n  > {entry['clinical_note']}" if entry.get("clinical_note") else "")
            )
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Could not read the feedback history: {exc}")
