"""
View 3 — HITL Corrections.

Editable medication table (`st.data_editor`) · **Elicitation Response Form**
(dynamic, schema-driven — this view is the MCP `elicitation_callback`) · risk
label override · approval decision · save feedback · re-run validation.

The re-run is one of the two real-time touchpoints: it drives a live progress
bar and a stage log fed by the pipeline's own progress callbacks as they happen
(see `common.stream_pipeline`), rather than hiding the run behind a spinner.
When it finishes, the reviewer is routed back to view 2 with the new result.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (  # noqa: E402
    english_drug_name,
    findings_table,
    flow_state,
    get_case,
    get_report,
    goto,
    mark_signed_off,
    no_report_warning,
    page_setup,
    render_elicitation_form,
    require_page,
    risk_badge,
    settings,
    store_pipeline_result,
    stream_pipeline,
)
from i18n import (  # noqa: E402
    english_duration,
    english_remark,
    english_route,
)
from resolutions import (  # noqa: E402
    apply_resolutions,
    render_resolution_panel,
)
from discharge_ai.common.schemas import Medication  # noqa: E402
from discharge_ai.validation import elicitation_schema_for  # noqa: E402

page_setup(
    "3 · HITL Corrections",
    "Correct the record, answer the elicitation request, and record your decision.",
)

patient_id = require_page("corrections")

case = get_case(patient_id)
report = get_report(patient_id)
if report is None:
    no_report_warning(patient_id)
    st.stop()

risk = report.risk
state = flow_state(patient_id)

# --------------------------------------------------------------------------- #
#  Context banner
# --------------------------------------------------------------------------- #
with st.container(border=True):
    banner = st.columns([2, 1, 1, 1])
    banner[0].markdown(f"### {case.patient_name or patient_id}")
    with banner[1]:
        st.markdown("**Current risk**")
        st.markdown(risk_badge(risk.level.value), unsafe_allow_html=True)
    banner[2].metric("Score", risk.score)
    banner[3].metric("Blocked", "Yes" if risk.discharge_blocked else "No")

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
    "Drug names are shown in English. Correct any dose, frequency, route or "
    "duration that the extractor misread, or delete a row to remove a "
    "medicine. Edits are applied to the case before validation is re-run."
)

MED_COLUMNS = [
    "sl_no", "medicine_name", "strength", "dosage", "frequency", "route",
    "period", "remarks", "total_quantity",
]

#  Drug names are rendered in English (Amoxicilline → Amoxicillin) so a
#  reviewer is never asked to reason about a Dutch or Spanish spelling. The
#  English name is what gets written back to the case on save: it is the
#  normalised value, it keeps the generated summary in English, and validation
#  is unaffected because the rules compare on `canonical_drug()`, which maps
#  both spellings to the same key. The original wording stays visible in the
#  "Source spellings" expander below for reconciliation against the paper.
_source_names: dict[int, str] = {}
med_rows = []
for row_index, medication in enumerate(case.medications):
    row = {column: getattr(medication, column, None) for column in MED_COLUMNS}
    source_name = row.get("medicine_name") or ""
    english_name = english_drug_name(source_name)
    if english_name and english_name != source_name:
        _source_names[row_index] = source_name
    row["medicine_name"] = english_name or source_name
    #  Route, duration and remarks are the other cells the Normalizer leaves in
    #  the source language — translate them for display and for the value the
    #  reviewer edits, so a correction never re-introduces Dutch or Spanish.
    if row.get("route"):
        row["route"] = english_route(row["route"])
    if row.get("period"):
        row["period"] = english_duration(row["period"])
    if row.get("remarks"):
        row["remarks"] = english_remark(row["remarks"])
    med_rows.append(row)

med_rows = med_rows or [{column: None for column in MED_COLUMNS}]

if _source_names:
    with st.expander("Source spellings (as written on the original document)"):
        for index, original in sorted(_source_names.items()):
            st.markdown(
                f"- Row {index + 1}: `{original}` → "
                f"**{english_drug_name(original)}**"
            )

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
#  2. Resolve blocking findings
# --------------------------------------------------------------------------- #
#  Table 4 rules are read-only on view 2. Here each *blocking* finding gets a
#  control that fixes the underlying record — remove the conflicting medicine,
#  settle the bill, record the physician's approval, schedule the follow-up.
#  The controls are matched to findings generically off the finding's own
#  `details` payload (see dashboard/resolutions.py), and the corrections feed
#  the same "Re-run validation" pipeline the elicitation form uses.
st.subheader("2 · Resolve blocking findings")

pending_resolutions = render_resolution_panel(report, case, patient_id)

st.divider()

# --------------------------------------------------------------------------- #
#  3. MCP Elicitation Response Form  (dynamic, schema-driven)
# --------------------------------------------------------------------------- #
st.subheader("3 · Elicitation Response Form")

non_blocking = [field for field in report.completeness.missing_fields if not field.blocking]

if not non_blocking:
    st.success("No non-blocking fields are missing — nothing to elicit.")
    elicitation_answers: dict[str, object] = {}
    elicitation_action = "not_requested"
else:
    #  Whatever schema the Rules Engine tool sent over MCP is what we render —
    #  no field is hardcoded here. Fall back to rebuilding it locally only if
    #  the report did not capture the schema.
    schema = report.elicitation.schema_sent
    if not schema:
        _model, schema = elicitation_schema_for(non_blocking)

    properties = schema.get("properties", {}) or {}
    st.caption(
        f"The Clinical Rules Engine tool requested **{len(properties)}** value(s) "
        "through the MCP Elicitation primitive. This form is the "
        "`elicitation_callback`: every input below is generated from the "
        "schema's own field name, type and description, and your response goes "
        "back to the MCP server as an `ElicitResult` with action `accept`, "
        "`decline` or `cancel`."
    )

    with st.expander("Schema received from the MCP server"):
        st.json(schema, expanded=False)

    elicitation_answers = render_elicitation_form(
        schema, key_prefix=f"elicit-{patient_id}"
    )

    if elicitation_answers:
        st.caption(f"{len(elicitation_answers)} of {len(properties)} value(s) supplied.")

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
#  4. Risk override + approval decision
# --------------------------------------------------------------------------- #
st.subheader("4 · Reviewer decision")

decision_columns = st.columns([1, 1, 2])

with decision_columns[0]:
    risk_override = st.selectbox(
        "Risk label override",
        ["(keep computed)", "Low", "Medium", "High"],
        key=f"risk-override-{patient_id}",
        help="Overrides the computed tier. Recorded in the audit trail.",
    )

with decision_columns[1]:
    approval = st.radio(
        "Approval decision",
        ["Pending", "Approve", "Reject"],
        key=f"approval-{patient_id}",
    )

with decision_columns[2]:
    reviewer = st.text_input(
        "Reviewer name / id", key=f"reviewer-{patient_id}",
        placeholder="e.g. Dr. R. Greene (staff #4471)",
    )

if approval == "Approve" and risk.discharge_blocked:
    st.error(
        "This discharge is **blocked** by a Critical finding. Approving it "
        "overrides a hard safety guardrail. The override is recorded in the "
        "audit trail against your reviewer id."
    )

st.divider()

# --------------------------------------------------------------------------- #
#  5. Save feedback / re-run validation
# --------------------------------------------------------------------------- #
st.subheader("5 · Save and re-run")

APPROVING = {"Approve"}


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
        "elicitation_action": elicitation_action,
        "elicitation_answers": elicitation_answers,
        "medication_edits": edited_meds,
        "blocking_resolutions": [
            entry for entry in
            (st.session_state.get("resolutions", {}).get(patient_id, {}) or {}).values()
        ],
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

    #  An explicit approval is the HITL sign-off that unlocks view 5 for a case
    #  the pipeline itself refused to release.
    if approval in APPROVING:
        mark_signed_off(patient_id, reviewer, approved=True)
    elif approval == "Reject":
        mark_signed_off(patient_id, reviewer, approved=False)

    st.success(
        f"Feedback saved to `{path.name}` ({count} medication row(s) applied to "
        "the working case)."
    )
    if approval in APPROVING and risk.discharge_blocked:
        st.info(
            "Sign-off recorded — **5 · Discharge Summary** is now unlocked for "
            "this case despite the block."
        )

# --------------------------------------------------------------------------- #
#  Re-run — live streaming progress, then route back to view 2
# --------------------------------------------------------------------------- #
#  Fractions the progress bar walks through as each pipeline stage reports in.
STAGE_PROGRESS = {
    "extract": 0.25,
    "normalise": 0.50,
    "validate": 0.85,
    "index": 0.95,
}

if buttons[1].button("🔁 Re-run validation", type="primary", width="stretch"):
    _apply_medication_edits()

    #  Fold the blocking-finding resolutions into the same working case the
    #  medication edits just landed on. Order matters: the medication table is
    #  applied first so a "remove this medicine" resolution operates on the
    #  reviewer's current rows, not the originally extracted ones.
    applied_resolutions = apply_resolutions(case, patient_id)
    if applied_resolutions:
        st.session_state["cases"][patient_id] = case
        st.info(
            "Applying "
            f"{len(applied_resolutions)} resolution(s):\n\n"
            + "\n".join(f"- {note}" for note in applied_resolutions)
        )

    kwargs: dict = {}
    if elicitation_action == "accept" and elicitation_answers:
        kwargs["elicitation_answers"] = elicitation_answers
    elif elicitation_action == "decline":
        kwargs["elicitation_decline"] = True
    elif elicitation_action == "cancel":
        kwargs["elicitation_cancel"] = True

    #  Live indicator: a real progress bar plus a stage log that grows as the
    #  agents report in, not a spinner that hides the whole run.
    #
    #  Two behaviours the reviewer relies on:
    #   1. `initial_case=case`  — feed the reviewer-edited case straight into
    #      validation so extract does NOT re-parse the PDFs and overwrite the
    #      corrections (removed medicines stay removed, edited doses stay edited).
    #   2. `generate_summary=False` — per PDF §2.5, a re-validation must not
    #      silently generate a summary; the reviewer requests it on view 5.
    st.markdown("**Re-running the pipeline with your corrections**")
    progress_bar = st.progress(0.0, text="starting…")
    log_box = st.empty()
    stage_log: list[str] = []
    result = None

    for kind, payload, message in stream_pipeline(
        patient_id,
        mode=st.session_state["execution_mode"],
        generate_summary=False,
        use_llm_extraction=False,     # keep the reviewer's values verbatim
        initial_case=case,            # the reviewer-edited case — do not re-parse
        trace_id=report.trace_id,
        **kwargs,
    ):
        if kind == "progress":
            stage, text = payload, message
            progress_bar.progress(
                STAGE_PROGRESS.get(stage, 0.5), text=f"{stage} — {text}"
            )
            stage_log.append(f"- **{stage}** — {text}")
            log_box.markdown("\n".join(stage_log[-10:]))
        elif kind == "result":
            result = payload

    progress_bar.progress(1.0, text="done")

    if result is None or result.report is None:
        errors = result.errors if result is not None else ["the pipeline produced no result"]
        st.error("Re-run failed:\n\n" + "\n".join(f"- {e}" for e in errors))
    else:
        store_pipeline_result(result)
        if approval in APPROVING:
            #  Sign-off survives the re-validation the reviewer just authorised.
            mark_signed_off(patient_id, reviewer, approved=True)

        new_risk = result.report.risk
        delta = new_risk.score - risk.score
        st.success(
            f"Re-validated: risk **{new_risk.level.value}** (score {new_risk.score}, "
            f"{delta:+d}), {len(result.report.findings)} finding(s), "
            f"elicitation outcome **{result.report.elicitation.outcome.value}**."
        )
        goto("validation")

with buttons[2]:
    if pending_resolutions:
        st.caption(
            f"**{pending_resolutions} resolution(s) ready.** Re-run applies them "
            "to the case, replays the MCP Elicitation round-trip with your "
            "chosen action, and re-evaluates every rule. You are returned to "
            "**2 · Validation Report** with the new result."
        )
    else:
        st.caption(
            "**Re-run** replays the MCP Elicitation round-trip with your chosen "
            "action, so `accept` / `decline` / `cancel` are all exercised for real "
            "against the Clinical Rules Engine tool on the primary MCP server. "
            "You are returned to **2 · Validation Report** with the new result."
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
            )
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Could not read the feedback history: {exc}")
