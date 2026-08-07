"""
View 2 — Validation Report.

The release decision is the headline: `decision_hero()` renders it far larger
and at higher contrast than anything else on the page, because "is this
discharge blocked?" is the one thing a reviewer must not miss.

Below it: completeness score (colour-coded), cross-validation issues, risk tier,
recommendation, missing fields, analytics, guardrails and the audit trail.

The single call-to-action routes on state — blocked cases go to view 3 to be
resolved, cleared cases go to view 5 for the patient letter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (  # noqa: E402
    audit_trail_table,
    decision_hero,
    findings_table,
    flow_state,
    get_report,
    goto,
    guardrail_table,
    language_badge,
    no_report_warning,
    page_setup,
    require_page,
    risk_badge,
    risk_domain_panel,
    sync_flow_from_report,
    trace_link,
)
from discharge_ai.common.rules import risk_thresholds  # noqa: E402

page_setup(
    "2 · Validation Report",
    "Completeness, EHR cross-validation, risk tier and the release decision.",
)

patient_id = require_page("validation")

report = get_report(patient_id)
if report is None:
    no_report_warning(patient_id)
    st.stop()

#  Keep the flow in step with whatever report we are actually showing.
sync_flow_from_report(patient_id, report)
state = flow_state(patient_id)

risk = report.risk
thresholds = risk_thresholds()

# --------------------------------------------------------------------------- #
#  Decision — the dominant element on the page
# --------------------------------------------------------------------------- #
decision = decision_hero(report)

# --------------------------------------------------------------------------- #
#  Next step  (state-gated: the summary route only exists when not blocked)
# --------------------------------------------------------------------------- #
action = st.columns([2, 3])

with action[0]:
    if decision == "blocked":
        if st.button("🛠 Resolve in HITL Corrections", type="primary",
                     width="stretch", key="route-corrections"):
            goto("corrections")
    else:
        if st.button("📄 View Discharge Summary", type="primary",
                     width="stretch", key="route-summary"):
            goto("summary")

with action[1]:
    if decision == "blocked":
        st.caption(
            "A blocked discharge cannot produce a patient summary. Correct the "
            "record, answer the elicitation request and re-run validation on "
            "**3 · HITL Corrections** — or sign the case off explicitly there if "
            "you are overriding the guardrail."
        )
        if state["approved"]:
            st.info(
                f"Signed off by **{state['signed_off_by']}** — the summary step "
                "has been unlocked despite the block."
            )
    elif decision == "hitl":
        st.caption(
            "This case is not blocked, so the summary is available — but it "
            "needs a clinician's sign-off before it goes to the patient. Record "
            "that on **3 · HITL Corrections**."
        )
    else:
        st.caption("Cleared for auto-release — the patient letter is ready to review.")

    if st.button("Open 3 · HITL Corrections", key="route-corrections-secondary"):
        goto("corrections")

st.divider()

# --------------------------------------------------------------------------- #
#  Metric band
# --------------------------------------------------------------------------- #
#  Five tiles, not six: translation confidence is Normalizer metadata for QA,
#  not a number a discharge reviewer acts on. It is still produced, still
#  written to the audit report and still attached to the LangFuse spans — it is
#  simply not surfaced here.
band = st.columns(5)

with band[0]:
    st.markdown("**Risk level**")
    st.markdown(risk_badge(risk.level.value), unsafe_allow_html=True)
    st.caption(f"score {risk.score} · low ≤ {thresholds['low_max']} · "
               f"medium ≤ {thresholds['medium_max']}")

with band[1]:
    completeness = report.completeness.score
    st.metric("Completeness", f"{completeness:.1f}%")
    st.progress(min(1.0, completeness / 100.0))
    if completeness < 85:
        st.caption("🔴 significant gaps")
    elif completeness < 100:
        st.caption("🟠 minor gaps")
    else:
        st.caption("🟢 complete")

band[2].metric("Recommendation", risk.recommendation.value)
band[3].metric(
    "Bill",
    f"{report.bill_total_amount:,.2f}" if report.bill_total_amount is not None else "—",
    report.bill_currency or "",
)
band[4].metric("Payment", report.bill_payment_status or "unknown")

st.markdown(
    "Record language: " + language_badge(report.detected_language)
    + f" &nbsp;&nbsp; Rules version: <code>{report.rules_version}</code>",
    unsafe_allow_html=True,
)
trace_link(report.trace_id)

st.divider()

# --------------------------------------------------------------------------- #
#  Cross-validation findings
# --------------------------------------------------------------------------- #
st.subheader(f"Cross-validation issues ({len(report.findings)})")

counts = st.columns(4)
counts[0].metric("Critical", sum(1 for f in report.findings if f.severity.value == "Critical"))
counts[1].metric("Warning", sum(1 for f in report.findings if f.severity.value == "Warning"))
counts[2].metric("Info", sum(1 for f in report.findings if f.severity.value == "Info"))
counts[3].metric("Blocking", sum(1 for f in report.findings if f.blocks_discharge))

findings_table(report)

if report.ehr_verdict:
    st.info(f"**Clinical reviewer verdict:** {report.ehr_verdict}")

with st.expander("Finding details (raw)"):
    for finding in report.findings:
        st.markdown(f"**`{finding.rule_id}`** — {finding.severity.value}")
        st.json(finding.details, expanded=False)

st.divider()

# --------------------------------------------------------------------------- #
#  Missing fields
# --------------------------------------------------------------------------- #
st.subheader(f"Missing fields ({len(report.completeness.missing_fields)})")

if report.completeness.missing_fields:
    import pandas as pd

    rows = [
        {
            "Document": field.document,
            "Field": field.field,
            "Row": field.row or "—",
            "Blocking": "BLOCKING" if field.blocking else "non-blocking",
            "Reviewer prompt": field.prompt,
        }
        for field in report.completeness.missing_fields
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if report.completeness.blocking_missing:
        st.error(
            "Blocking gaps stop automatic summary generation: "
            + ", ".join(report.completeness.blocking_missing)
        )
    if report.completeness.non_blocking_missing:
        st.warning(
            "Non-blocking gaps are collected through MCP Elicitation on "
            "**3 · HITL Corrections**: "
            + ", ".join(report.completeness.non_blocking_missing)
        )
else:
    st.success("Every mandatory field is documented.")

if report.completeness.prescription_gaps:
    st.markdown("**Prescription rows with gaps**")
    st.dataframe(report.completeness.prescription_gaps, width="stretch", hide_index=True)

st.divider()

# --------------------------------------------------------------------------- #
#  Risk score breakdown
# --------------------------------------------------------------------------- #
#  One section, not two side-by-side panels. The per-domain heatmap and the
#  line-by-line contributions are two views of the same number — the heatmap
#  says *where* the risk sits, the table says *what* produced it — so they are
#  read together, top to bottom, rather than competing for attention in
#  parallel columns.
st.subheader("Risk score breakdown")

analytics = report.analytics or {}

#  Where the risk sits, by clinical domain (generate_risk_heatmap, MCP :8201).
if analytics and not analytics.get("error"):
    risk_domain_panel(analytics.get("heatmap") or {})
    if analytics.get("benchmark_interpretation"):
        st.caption(analytics["benchmark_interpretation"])
elif analytics.get("error"):
    st.caption(f"Domain breakdown unavailable — {analytics['error']}")

#  What produced it, rule by rule.
if risk.contributions:
    import pandas as pd

    rows = [
        {
            "Source": item.get("source"),
            "Item": item.get("rule_id") or item.get("field"),
            "Weight key": item.get("risk_key"),
            "Weight": item.get("weight"),
        }
        for item in risk.contributions
    ]
    with st.expander(
        f"What produced this score — {len(rows)} contribution(s)", expanded=False
    ):
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(f"Total: **{risk.score}** (weights from `configs/rules.yaml`)")
else:
    st.success("Nothing contributed to the risk score.")

benchmarks = (analytics.get("benchmarks") or {}).get("benchmarks") or []
if benchmarks:
    import pandas as pd

    with st.expander("Population benchmarks", expanded=False):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Diagnosis": b.get("diagnosis"),
                        "30-day readmission": f"{b.get('readmission_30d_rate', 0):.1%}",
                        "Network median": f"{b.get('network_median', 0):.1%}",
                        "Follow-up attendance": f"{b.get('followup_attendance_rate', 0):.0%}",
                    }
                    for b in benchmarks
                ]
            ),
            width="stretch", hide_index=True,
        )

st.divider()

# --------------------------------------------------------------------------- #
#  Elicitation, guardrails, audit trail
# --------------------------------------------------------------------------- #
st.subheader("MCP Elicitation")
elicitation = report.elicitation
st.markdown(
    f"Outcome: **{elicitation.outcome.value}**"
    + (f" — {elicitation.note}" if elicitation.note else "")
)
if elicitation.reviewer_values:
    st.json(elicitation.reviewer_values)
if elicitation.schema_sent:
    with st.expander("Schema sent to the reviewer"):
        st.json(elicitation.schema_sent, expanded=False)

st.subheader("Responsible-AI guardrails")
guardrail_table(report.guardrail_events)

st.subheader(f"Audit trail ({len(report.audit_trail)} steps)")
audit_trail_table(report)

st.divider()
st.subheader("Raw Validation JSON")
with st.expander("Full validation report (machine-readable)", expanded=False):
    st.json(report.model_dump(mode="json"), expanded=False)
