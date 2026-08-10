"""
agents/validator_agent.py
=========================

**Clinical Validation Agent** — LangGraph — A2A :8101 — non-streaming
MCP primitives: Tools + Elicitation + Resources

A `StateGraph` with six nodes:

    load_rules     MCP Resource  resource://clinical-rules/cross-validation
    completeness   MCP Tool      clinical_rules_engine  (→ MCP **Elicitation**)
    cross_validate MCP Tool      ehr_validation         (Mock EHR :8050)
    explain        MCP Prompt    ehr-cross-validation-prompt → clinical impact
    risk           MCP Tool      calculate_risk_score + generate_risk_heatmap
                                 + get_population_benchmarks  (Analytics :8201)
    report         MCP Tool      clinical_insight_reporter → JSON + HTML

The reviewer's elicitation answers are injected by the caller
(`elicitation_answers` in the A2A payload), which is how the Streamlit HITL page
exercises the accept / decline / cancel outcomes for real.

Run:
    python -m discharge_ai.agents.validator_agent
"""

from __future__ import annotations

import json
import logging
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from ..common.doc_loader import load_patient_documents
from ..common.parsing import build_extracted_case
from ..common.rules import rules_version
from ..common.schemas import (
    CompletenessResult,
    ElicitationOutcome,
    ElicitationRecord,
    ExtractedCase,
    MissingField,
    RiskAssessment,
    ValidationFinding,
    ValidationReport,
)
from ..ehr import ehr_client
from ..guardrails import guardrails
from ..observability import tracing
from ..settings import settings
from ..validation import check_completeness, cross_validate, score_case
from .base import AuditTrail, agent_mcp, case_from_payload, require_patient_id, trace_for

log = logging.getLogger(__name__)

AGENT_KEY = "validator"
AGENT_NAME = settings.agent(AGENT_KEY)["name"]
FRAMEWORK = "langgraph"


class ValidatorState(TypedDict, total=False):
    patient_id: str
    trace_id: str | None
    case: dict[str, Any]
    cross_validation_rules: dict[str, Any]
    completeness: dict[str, Any]
    elicitation: dict[str, Any]
    findings: list[dict[str, Any]]
    ehr_snapshot: dict[str, Any]
    explanations: dict[str, Any]
    risk: dict[str, Any]
    analytics: dict[str, Any]
    artifacts: dict[str, str]
    notes: list[str]


def _build_graph(mcp: Any, audit: AuditTrail, manager: Any):
    async def load_rules(state: ValidatorState) -> ValidatorState:
        """MCP Resource: the cross-validation rule catalogue + risk matrix."""
        async with audit.step("read_cross_validation_resource", "MCP Resources",
                              framework=FRAMEWORK):
            rules = (
                await mcp.read_resource_json(
                    "resource://clinical-rules/cross-validation", default={}
                )
                if mcp.is_connected else {}
            )

        return {
            "cross_validation_rules": rules,
            "notes": [
                "read resource://clinical-rules/cross-validation "
                f"({len(rules.get('rules', []))} rules, v={rules.get('rules_version', '?')})"
                if rules else "cross-validation rules loaded locally"
            ],
        }

    async def completeness(state: ValidatorState) -> ValidatorState:
        """MCP Tool + Elicitation: clinical_rules_engine."""
        case = ExtractedCase.model_validate(state["case"])
        patient_id = state["patient_id"]

        result: dict[str, Any] = {}
        async with audit.step("validate_completeness",
                              "clinical_rules_engine tool (MCP Elicitation)",
                              framework=FRAMEWORK):
            if mcp.is_connected:
                result = await mcp.call_tool(
                    "clinical_rules_engine",
                    {
                        "patient_id": patient_id,
                        "case": case.model_dump(mode="json"),
                        "elicit_missing": True,
                    },
                )

        #  Always compute the structured result locally too: the MCP tool returns
        #  a flattened summary, while the report needs the full field list.
        local = check_completeness(case)

        elicitation = ElicitationRecord(
            requested_fields=[f for f in local.missing_fields if not f.blocking],
            outcome=ElicitationOutcome(
                result.get("elicitation_outcome", ElicitationOutcome.NOT_REQUESTED.value)
            ) if isinstance(result, dict) else ElicitationOutcome.NOT_REQUESTED,
            reviewer_values=(result or {}).get("reviewer_values", {}),
            schema_sent=(result or {}).get("elicitation_schema", {}),
            note=(result or {}).get("note", ""),
        )

        notes: list[str] = [
            f"completeness {local.score}% "
            f"({local.present}/{local.total_required} fields present)"
        ]

        #  Apply whatever the reviewer supplied, then re-score.
        if elicitation.reviewer_values:
            applied = _apply_reviewer_values(case, elicitation.reviewer_values,
                                             local.missing_fields)
            local = check_completeness(case)
            notes.append(
                f"reviewer supplied {applied} value(s) via MCP Elicitation → "
                f"completeness now {local.score}%"
            )

        if elicitation.outcome == ElicitationOutcome.CANCEL:
            notes.append("elicitation cancelled — escalating to a senior clinician")

        return {
            "case": case.model_dump(mode="json"),
            "completeness": local.model_dump(mode="json"),
            "elicitation": elicitation.model_dump(mode="json"),
            "notes": notes,
        }

    async def cross_validate_node(state: ValidatorState) -> ValidatorState:
        """MCP Tool: ehr_validation against the Mock EHR."""
        case = ExtractedCase.model_validate(state["case"])
        patient_id = state["patient_id"]

        findings: list[dict[str, Any]] = []
        snapshot: dict[str, Any] = {}
        source = ""

        async with audit.step("cross_validate_vs_ehr", "ehr_validation tool",
                              framework=FRAMEWORK):
            if mcp.is_connected:
                result = await mcp.call_tool(
                    "ehr_validation",
                    {
                        "patient_id": patient_id,
                        "case": case.model_dump(mode="json"),
                        "translation_confidence": case.translation_confidence,
                    },
                )
                if isinstance(result, dict) and not result.get("error"):
                    findings = result.get("findings", [])
                    snapshot = result.get("ehr_snapshot", {})
                    source = str(result.get("ehr_source", ""))

            if not findings and not snapshot:
                bundle = ehr_client.bundle(patient_id)
                findings = [
                    f.model_dump(mode="json") for f in cross_validate(case, bundle)
                ]
                snapshot = {
                    "medications": bundle.get("medications", []),
                    "allergies": bundle.get("allergies", []),
                    "abnormal_labs": bundle.get("abnormal_labs", []),
                    "care_plan": bundle.get("care_plan", {}),
                }
                source = str(bundle.get("source", ""))

        critical = sum(1 for f in findings if f.get("severity") == "Critical")
        return {
            "findings": findings,
            "ehr_snapshot": snapshot,
            "notes": [
                f"cross-validated against {source or 'the EHR'}: "
                f"{len(findings)} finding(s), {critical} critical"
            ],
        }

    async def explain(state: ValidatorState) -> ValidatorState:
        """MCP Prompt + LLM: clinical impact for each finding."""
        findings = state.get("findings") or []
        if not findings or settings.offline_mode:
            return {"explanations": {}, "notes": []}

        prompt = ""
        async with audit.step("fetch_cross_validation_prompt", "MCP Prompts",
                              framework=FRAMEWORK):
            prompt = await mcp.get_prompt(
                "ehr-cross-validation-prompt", {"patient_id": state["patient_id"]}
            )

        payload_for_llm = [
            {
                "rule_id": f.get("rule_id"),
                "severity": f.get("severity"),
                "message": f.get("message"),
                "action": f.get("action"),
            }
            for f in findings
        ]

        from ..llm import provider

        explanations: dict[str, Any] = {}
        async with audit.step("explain_findings", f"LLM ({provider.resolve_model_id()})",
                              framework=FRAMEWORK):
            with tracing.llm_generation(
                "validation.explain_findings",
                model=provider.resolve_model_id("reasoning"),
                prompt="[prompt from MCP] ehr-cross-validation-prompt",
                trace_id=state.get("trace_id"),
            ) as span:
                result = await provider.acomplete_json(
                    f"{prompt}\n\nDETERMINISTIC FINDINGS:\n"
                    f"{json.dumps(payload_for_llm, indent=2, default=str)}",
                    purpose="reasoning",
                )
                if isinstance(result, dict):
                    explanations = result
                span.set_output(explanations)

        #  Attach the explanations to their findings; never let the model add or
        #  remove a finding.
        by_rule = {
            str(item.get("rule_id")): item
            for item in (explanations.get("findings_explained") or [])
            if isinstance(item, dict)
        }
        enriched: list[dict[str, Any]] = []
        for finding in findings:
            explanation = by_rule.get(str(finding.get("rule_id")))
            if explanation:
                finding = {
                    **finding,
                    "clinical_impact": explanation.get("clinical_impact"),
                    "suggested_action": explanation.get("suggested_action")
                    or finding.get("action"),
                }
            enriched.append(finding)

        return {
            "findings": enriched,
            "explanations": explanations,
            "notes": [f"LLM explained {len(by_rule)} of {len(findings)} finding(s)"],
        }

    async def risk(state: ValidatorState) -> ValidatorState:
        """Analytics MCP Server (:8201): score, heatmap and benchmarks."""
        completeness_result = CompletenessResult.model_validate(state["completeness"])
        findings = [ValidationFinding.model_validate(f) for f in state.get("findings", [])]

        #  The authoritative score comes from the local rules-matrix engine; the
        #  Analytics server independently recomputes it (multi-server MCP proof)
        #  and adds the heatmap + population benchmarks.
        assessment = score_case(findings, completeness_result)

        analytics: dict[str, Any] = {}
        risk_keys = sorted({f.risk_key for f in findings if f.risk_key})

        async with audit.step("analytics_risk_scoring",
                              "calculate_risk_score tool (Analytics MCP :8201)",
                              framework=FRAMEWORK):
            if "analytics" in mcp.connected_servers:
                try:
                    case = ExtractedCase.model_validate(state["case"])
                    remote = await mcp.call_tool(
                        "calculate_risk_score",
                        {
                            "patient_id": state["patient_id"],
                            "risk_keys": risk_keys,
                            "missing_field_count": len(
                                completeness_result.non_blocking_missing
                            ),
                            "translation_confidence": case.translation_confidence,
                            "discharge_blocked": assessment.discharge_blocked,
                        },
                        server="analytics",
                    )
                    heatmap = await mcp.call_tool(
                        "generate_risk_heatmap",
                        {"patient_id": state["patient_id"], "risk_keys": risk_keys},
                        server="analytics",
                    )
                    benchmarks = await mcp.call_tool(
                        "get_population_benchmarks",
                        {"patient_id": state["patient_id"]},
                        server="analytics",
                    )
                    analytics = {
                        "analytics_risk_score": remote.get("risk_score"),
                        "analytics_risk_level": remote.get("risk_level"),
                        "domain_totals": remote.get("domain_totals", {}),
                        "heatmap": heatmap,
                        "benchmarks": benchmarks,
                        "cohort_readmission_30d_rate": benchmarks.get(
                            "cohort_readmission_30d_rate"
                        ),
                        "benchmark_interpretation": benchmarks.get("interpretation"),
                        "computed_by": "analytics-server:8201",
                    }
                except Exception as exc:  # noqa: BLE001
                    log.warning("Analytics MCP server call failed: %s", exc)
                    analytics = {"error": f"{type(exc).__name__}: {exc}"}

        #  RAI guardrail: High risk or a blocked discharge forces human review.
        decision = manager.evaluate_escalation(
            risk_level=assessment.level,
            discharge_blocked=assessment.discharge_blocked,
            hard_guardrails_hit=assessment.hard_guardrails_hit,
        )
        assessment.hitl_required = assessment.hitl_required or decision.hitl_required

        return {
            "risk": assessment.model_dump(mode="json"),
            "analytics": analytics,
            "notes": [
                f"risk score {assessment.score} → {assessment.level.value} "
                f"({assessment.recommendation.value}); {decision.detail}"
            ],
        }

    async def report(state: ValidatorState) -> ValidatorState:
        """MCP Tool: clinical_insight_reporter → JSON + HTML artifacts."""
        validation_report = _assemble_report(state, audit, manager)

        artifacts: dict[str, str] = {}
        async with audit.step("generate_audit_report",
                              "clinical_insight_reporter tool", framework=FRAMEWORK):
            if mcp.is_connected:
                try:
                    result = await mcp.call_tool(
                        "clinical_insight_reporter",
                        {
                            "patient_id": state["patient_id"],
                            "report": validation_report.model_dump(mode="json"),
                            "write_files": True,
                        },
                    )
                    artifacts = (result or {}).get("artifacts", {})
                except Exception as exc:  # noqa: BLE001
                    log.warning("Reporter tool failed: %s", exc)

            if not artifacts:
                from ..reporting import build_reports

                artifacts = build_reports(validation_report, write_files=True)

        return {
            "artifacts": artifacts,
            "notes": [f"audit report written: {', '.join(sorted(artifacts))}"],
        }

    graph = StateGraph(ValidatorState)
    graph.add_node("load_rules", load_rules)
    graph.add_node("completeness", completeness)
    graph.add_node("cross_validate", cross_validate_node)
    graph.add_node("explain", explain)
    graph.add_node("risk", risk)
    graph.add_node("report", report)

    graph.add_edge(START, "load_rules")
    graph.add_edge("load_rules", "completeness")
    graph.add_edge("completeness", "cross_validate")
    graph.add_edge("cross_validate", "explain")
    graph.add_edge("explain", "risk")
    graph.add_edge("risk", "report")
    graph.add_edge("report", END)

    return graph.compile(checkpointer=MemorySaver())


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
_REVIEWER_TO_ATTR = {
    "doctors": "attending_physician",
    "adr_allergy_info": "allergies",
    "follow_up_appointments": "follow_up_appointment",
    "vendor_name": "lab_vendor_name",
    "report_date": "lab_report_date",
    "hospital_name": "bill.hospital_name",
    "billing_date": "bill.billing_date",
}


def _apply_reviewer_values(
    case: ExtractedCase, values: dict[str, Any], missing: list[MissingField]
) -> int:
    """Write reviewer-supplied elicitation values onto the case."""
    by_key = {
        (f"{f.field}_row{f.row}" if f.document == "prescription" and f.row else f.field): f
        for f in missing
    }
    applied = 0

    for key, value in values.items():
        if value in (None, ""):
            continue
        field = by_key.get(key)

        #  Prescription columns are per-row.
        if field is not None and field.document == "prescription" and field.row:
            for medication in case.medications:
                if medication.sl_no == field.row:
                    setattr(medication, field.field, str(value))
                    applied += 1
            continue

        attribute = _REVIEWER_TO_ATTR.get(key, key)
        if attribute.startswith("bill."):
            setattr(case.bill, attribute.split(".", 1)[1], value)
            applied += 1
            continue

        if not hasattr(case, attribute):
            continue

        current = getattr(case, attribute)
        try:
            if isinstance(current, list):
                items = value if isinstance(value, list) else str(value).split(",")
                setattr(case, attribute, [str(i).strip() for i in items if str(i).strip()])
            elif attribute == "age":
                setattr(case, attribute, int(float(value)))
            elif isinstance(current, bool) or attribute == "discharge_approved":
                from ..common.terminology import parse_bool

                parsed = parse_bool(value)
                if parsed is None:
                    continue
                setattr(case, attribute, parsed)
            else:
                setattr(case, attribute, str(value).strip())
        except (TypeError, ValueError):
            continue

        applied += 1
        case.extraction_notes.append(
            f"{attribute} supplied by the human reviewer via MCP Elicitation"
        )

    return applied


def _assemble_report(
    state: ValidatorState, audit: AuditTrail, manager: Any
) -> ValidationReport:
    case = ExtractedCase.model_validate(state["case"])
    explanations = state.get("explanations") or {}

    return ValidationReport(
        patient_id=state["patient_id"],
        patient_name=case.patient_name,
        rules_version=rules_version(),
        trace_id=state.get("trace_id"),
        completeness=CompletenessResult.model_validate(state["completeness"]),
        findings=[ValidationFinding.model_validate(f) for f in state.get("findings", [])],
        risk=RiskAssessment.model_validate(state["risk"]),
        elicitation=ElicitationRecord.model_validate(state["elicitation"]),
        guardrail_events=list(manager.events),
        audit_trail=audit.entries,
        translation_confidence=case.translation_confidence,
        detected_language=case.detected_language,
        bill_total_amount=case.bill.total_amount,
        bill_currency=case.bill.currency,
        bill_payment_status=case.bill.payment_status,
        ehr_verdict=str(explanations.get("overall_verdict", "")),
        analytics=state.get("analytics") or {},
    )


# --------------------------------------------------------------------------- #
async def handle(payload: dict[str, Any], ctx: Any) -> dict[str, Any]:
    """A2A entry point: `{"patient_id": …, "case": {…}}` → validation report."""
    patient_id = require_patient_id(payload)
    trace_id = trace_for(payload, getattr(ctx, "trace_id", None), patient_id)
    audit = AuditTrail(trace_id)

    case = case_from_payload(payload)
    if case is None:
        documents = load_patient_documents(patient_id)
        if not documents:
            raise FileNotFoundError(f"No documents found for {patient_id}")
        case = build_extracted_case(patient_id, documents)

    manager = guardrails(trace_id, known_names=[case.patient_name or ""])

    #  A reviewer's dashboard answers arrive here and drive the real MCP
    #  elicitation round-trip (accept / decline / cancel).
    from ..mcp_client import dashboard_elicitation_responder

    responder = None
    answers = payload.get("elicitation_answers")
    if payload.get("elicitation_cancel"):
        responder = dashboard_elicitation_responder(None, cancel=True)
    elif payload.get("elicitation_decline"):
        responder = dashboard_elicitation_responder(None, decline=True)
    elif isinstance(answers, dict) and answers:
        responder = dashboard_elicitation_responder(answers)

    async with agent_mcp(
        AGENT_NAME, trace_id=trace_id, elicitation_responder=responder
    ) as mcp:
        graph = _build_graph(mcp, audit, manager)
        final_state = await graph.ainvoke(
            {
                "patient_id": patient_id,
                "trace_id": trace_id,
                "case": case.model_dump(mode="json"),
                "notes": [],
            },
            config={"configurable": {"thread_id": f"validate:{patient_id}"}},
        )

    report = _assemble_report(final_state, audit, manager)
    validated_case = ExtractedCase.model_validate(final_state["case"])

    log.info(
        "Validated %s: score=%d level=%s blocked=%s hitl=%s",
        patient_id, report.risk.score, report.risk.level.value,
        report.risk.discharge_blocked, report.risk.hitl_required,
    )

    return {
        "agent": AGENT_NAME,
        "framework": FRAMEWORK,
        "patient_id": patient_id,
        "trace_id": trace_id,
        "trace_url": tracing.trace_url(trace_id),
        "report": report.model_dump(mode="json"),
        "case": validated_case.model_dump(mode="json"),
        "artifacts": final_state.get("artifacts", {}),
        "risk_level": report.risk.level.value,
        "risk_score": report.risk.score,
        "recommendation": report.risk.recommendation.value,
        "discharge_blocked": report.risk.discharge_blocked,
        "hitl_required": report.risk.hitl_required,
        "elicitation_outcome": report.elicitation.outcome.value,
        "notes": list(final_state.get("notes", [])),
        "audit_trail": audit.dump(),
        "mcp_primitives_used": ["tools", "elicitation", "resources", "prompts"],
    }


def main() -> None:
    from ..a2a_layer import run_agent_server

    run_agent_server(AGENT_KEY, handle, artifact_name="validation_report")


if __name__ == "__main__":
    main()
