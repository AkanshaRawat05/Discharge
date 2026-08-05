"""
agents/extractor_agent.py
=========================

**Clinical Extractor Agent** — LangGraph — A2A :8100 — non-streaming
MCP primitives: Tools + Resources + Prompts

A `StateGraph` with a `MemorySaver` checkpointer (so a re-run of the same
`patient_id` thread resumes rather than restarting) and four nodes:

    harvest        MCP Tool     clinical_data_harvester → deterministic parse
    load_rules     MCP Resource resource://clinical-rules/completeness
                                resource://medical-abbreviations
    llm_enrich     MCP Prompt   discharge-extraction-prompt → LLM fills the gaps
                                the deterministic parser could not read
    finalise                    merge, PII-redact for logging, emit the case

LLM values are only accepted for fields the deterministic parser left empty, so
a model can never overwrite a patient id, dose or payment status that was read
verbatim from the source document.

Run:
    python -m discharge_ai.agents.extractor_agent
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from ..common.doc_loader import load_patient_documents
from ..common.parsing import build_extracted_case
from ..common.schemas import ExtractedCase, Medication
from ..common.terminology import language_name, normalize_gender
from ..guardrails import guardrails
from ..llm import provider
from ..observability import tracing
from ..settings import settings
from .base import AuditTrail, agent_mcp, require_patient_id, trace_for

log = logging.getLogger(__name__)

AGENT_KEY = "extractor"
AGENT_NAME = settings.agent(AGENT_KEY)["name"]
FRAMEWORK = "langgraph"


# --------------------------------------------------------------------------- #
#  Graph state
# --------------------------------------------------------------------------- #
def _merge_notes(left: list[str], right: list[str]) -> list[str]:
    return list(dict.fromkeys((left or []) + (right or [])))


class ExtractorState(TypedDict, total=False):
    patient_id: str
    trace_id: str | None
    use_llm: bool
    case: dict[str, Any]
    completeness_rules: dict[str, Any]
    abbreviations: dict[str, str]
    extraction_prompt: str
    llm_fields: dict[str, Any]
    notes: Annotated[list[str], _merge_notes]
    audit: list[dict[str, Any]]


# --------------------------------------------------------------------------- #
#  Nodes
# --------------------------------------------------------------------------- #
def _build_graph(mcp: Any, audit: AuditTrail):
    """Compile the extraction StateGraph, closing over the MCP session."""

    async def harvest(state: ExtractorState) -> ExtractorState:
        """MCP Tool: clinical_data_harvester."""
        patient_id = state["patient_id"]

        async with audit.step("harvest_documents", "clinical_data_harvester tool",
                              framework=FRAMEWORK):
            if mcp.is_connected:
                payload = await mcp.call_tool(
                    "clinical_data_harvester",
                    {"patient_id": patient_id, "include_raw_text": True},
                )
                if isinstance(payload, dict) and payload.get("error"):
                    raise FileNotFoundError(
                        payload.get("detail", f"No documents for {patient_id}")
                    )
                case = ExtractedCase.model_validate(payload)
                note = "documents harvested via MCP tool"
            else:
                documents = load_patient_documents(patient_id)
                if not documents:
                    raise FileNotFoundError(f"No documents found for {patient_id}")
                case = build_extracted_case(patient_id, documents)
                note = "documents harvested in-process (MCP server unreachable)"

        return {
            "case": case.model_dump(mode="json"),
            "notes": [f"{note}: {', '.join(case.doc_types_present)}"],
        }

    async def load_rules(state: ExtractorState) -> ExtractorState:
        """MCP Resources: completeness rules + abbreviation dictionary."""
        rules: dict[str, Any] = {}
        abbreviations: dict[str, str] = {}
        notes: list[str] = []

        async with audit.step("read_mcp_resources", "MCP Resources", framework=FRAMEWORK):
            if mcp.is_connected:
                rules = await mcp.read_resource_json(
                    "resource://clinical-rules/completeness", default={}
                )
                abbreviation_payload = await mcp.read_resource_json(
                    "resource://medical-abbreviations", default={}
                )
                abbreviations = abbreviation_payload.get("abbreviations", {})
                notes.append(
                    "read resource://clinical-rules/completeness "
                    f"(v={rules.get('rules_version', '?')}) and "
                    f"resource://medical-abbreviations ({len(abbreviations)} entries)"
                )
            else:
                from ..common.rules import COMPLETENESS_RULES
                from ..common.terminology import full_abbreviation_map

                rules = {"documents": COMPLETENESS_RULES}
                abbreviations = full_abbreviation_map()
                notes.append("resources loaded from the local rules copy")

        return {"completeness_rules": rules, "abbreviations": abbreviations, "notes": notes}

    async def llm_enrich(state: ExtractorState) -> ExtractorState:
        """MCP Prompt + LLM: fill only the fields the parser could not read."""
        case = ExtractedCase.model_validate(state["case"])
        gaps = _gap_fields(case, state.get("completeness_rules") or {})

        if not state.get("use_llm", True):
            return {"notes": ["LLM enrichment skipped (use_llm=false)"]}
        if not gaps:
            return {"notes": ["no gaps for the LLM to fill"]}
        if settings.offline_mode:
            return {"notes": ["LLM enrichment skipped (OFFLINE_MODE=1)"]}

        prompt_template = ""
        async with audit.step("fetch_extraction_prompt", "MCP Prompts",
                              framework=FRAMEWORK):
            prompt_template = await mcp.get_prompt(
                "discharge-extraction-prompt",
                {
                    "language": language_name(case.detected_language),
                    "doc_types": ", ".join(case.doc_types_present),
                },
            )

        source_text = "\n\n".join(
            f"===== {doc_type.upper()} =====\n{text}"
            for doc_type, text in case.raw_text.items()
            if text
        )[:24000]

        instruction = (
            f"{prompt_template}\n\n"
            f"The deterministic parser could NOT read these fields: "
            f"{', '.join(gaps)}.\n"
            "Return JSON containing ONLY those keys, using null where the "
            "documents genuinely do not state the value.\n\n"
            f"SOURCE DOCUMENTS:\n{source_text}"
        )

        llm_fields: dict[str, Any] = {}
        async with audit.step("llm_extraction", f"LLM ({provider.resolve_model_id()})",
                              framework=FRAMEWORK):
            with tracing.llm_generation(
                "extraction.fill_gaps",
                model=provider.resolve_model_id("reasoning"),
                prompt=f"[prompt from MCP] gaps={gaps}",
                trace_id=state.get("trace_id"),
            ) as span:
                #  Off the event loop: this call can sleep on a rate limit.
                payload = await provider.acomplete_json(
                    instruction, purpose="reasoning"
                )
                if isinstance(payload, dict):
                    llm_fields = {
                        key: value for key, value in payload.items()
                        if key in gaps and value not in (None, "", [], {})
                    }
                span.set_output(llm_fields)

        return {
            "llm_fields": llm_fields,
            "notes": [
                f"LLM supplied {len(llm_fields)} of {len(gaps)} gap field(s): "
                f"{', '.join(sorted(llm_fields)) or 'none'}"
            ],
        }

    async def finalise(state: ExtractorState) -> ExtractorState:
        """Merge LLM values into empty slots only, then finalise the case."""
        case = ExtractedCase.model_validate(state["case"])
        applied = _apply_llm_fields(case, state.get("llm_fields") or {})

        case.extraction_method = "hybrid" if applied else "deterministic"
        case.extraction_notes = list(
            dict.fromkeys(case.extraction_notes + list(state.get("notes", [])))
        )

        return {
            "case": case.model_dump(mode="json"),
            "notes": [f"LLM values applied to {applied} empty field(s)"] if applied else [],
        }

    graph = StateGraph(ExtractorState)
    graph.add_node("harvest", harvest)
    graph.add_node("load_rules", load_rules)
    graph.add_node("llm_enrich", llm_enrich)
    graph.add_node("finalise", finalise)

    graph.add_edge(START, "harvest")
    graph.add_edge("harvest", "load_rules")
    graph.add_edge("load_rules", "llm_enrich")
    graph.add_edge("llm_enrich", "finalise")
    graph.add_edge("finalise", END)

    #  MemorySaver keeps per-patient threads, so re-running a case resumes.
    return graph.compile(checkpointer=MemorySaver())


# --------------------------------------------------------------------------- #
#  Gap detection / merge
# --------------------------------------------------------------------------- #
#  Rule field name → ExtractedCase attribute (only where they differ).
_RULE_TO_ATTR = {
    "doctors": "attending_physician",
    "adr_allergy_info": "allergies",
    "follow_up_appointments": "follow_up_appointment",
    "tests": "lab_tests",
    "vendor_name": "lab_vendor_name",
    "report_date": "lab_report_date",
}

_LLM_SETTABLE = {
    "patient_name", "age", "gender", "address", "admission_date", "discharge_date",
    "ward", "bed_no", "service_line", "attending_physician", "consulting_doctors",
    "discharge_diagnosis", "allergies", "follow_up_appointment",
    "discharge_instructions", "discharge_approved", "discharge_approved_by",
    "lab_name", "lab_vendor_name", "lab_report_date", "medications",
}


def _gap_fields(case: ExtractedCase, rules: dict[str, Any]) -> list[str]:
    """Fields the completeness rules require that the parser left empty."""
    documents = rules.get("documents") or {}
    if not documents:
        from ..common.rules import COMPLETENESS_RULES

        documents = COMPLETENESS_RULES

    gaps: list[str] = []
    for doc_type, spec in documents.items():
        if doc_type in {"prescription", "bill"}:
            continue          # bill fields are numeric/verbatim — never LLM-filled
        if case.doc_types_present and doc_type not in case.doc_types_present:
            continue
        for rule in spec.get("fields", []):
            field = rule["field"] if isinstance(rule, dict) else str(rule)
            attribute = _RULE_TO_ATTR.get(field, field)
            if attribute not in _LLM_SETTABLE:
                continue
            value = getattr(case, attribute, None)
            if value in (None, "", [], {}):
                gaps.append(attribute)
    return sorted(set(gaps))


def _apply_llm_fields(case: ExtractedCase, fields: dict[str, Any]) -> int:
    """Write LLM values into *empty* attributes only. Returns how many landed."""
    applied = 0

    for key, value in fields.items():
        if key not in _LLM_SETTABLE:
            continue
        current = getattr(case, key, None)
        if current not in (None, "", [], {}):
            continue          # never overwrite a verbatim parse

        try:
            if key == "medications":
                from ..common.parsing import parse_medications_from_json

                parsed = parse_medications_from_json(value)
                if not parsed:
                    continue
                case.medications = parsed
            elif key in {"consulting_doctors", "discharge_diagnosis", "allergies"}:
                items = value if isinstance(value, list) else [value]
                setattr(case, key, [str(item).strip() for item in items if item])
            elif key == "age":
                setattr(case, key, int(float(value)))
            elif key == "gender":
                setattr(case, key, normalize_gender(value))
            elif key == "discharge_approved":
                from ..common.terminology import parse_bool

                parsed_bool = parse_bool(value)
                if parsed_bool is None:
                    continue
                setattr(case, key, parsed_bool)
            else:
                setattr(case, key, str(value).strip())
        except (TypeError, ValueError) as exc:
            log.debug("Ignoring unusable LLM value for %s (%r): %s", key, value, exc)
            continue

        applied += 1
        case.extraction_notes.append(f"{key} supplied by LLM extraction (was empty)")

    return applied


# --------------------------------------------------------------------------- #
#  A2A handler
# --------------------------------------------------------------------------- #
async def handle(payload: dict[str, Any], ctx: Any) -> dict[str, Any]:
    """A2A entry point: `{"patient_id": "P1019"}` → extracted case."""
    patient_id = require_patient_id(payload)
    trace_id = trace_for(payload, getattr(ctx, "trace_id", None), patient_id)
    audit = AuditTrail(trace_id)

    async with agent_mcp(AGENT_NAME, trace_id=trace_id, servers=("primary",)) as mcp:
        graph = _build_graph(mcp, audit)
        final_state = await graph.ainvoke(
            {
                "patient_id": patient_id,
                "trace_id": trace_id,
                "use_llm": bool(payload.get("use_llm", True)),
                "notes": [],
            },
            config={"configurable": {"thread_id": f"extract:{patient_id}"}},
        )

    case = ExtractedCase.model_validate(final_state["case"])

    #  PII must be masked before it reaches the log or the trace backend.
    manager = guardrails(trace_id, known_names=[case.patient_name or ""])
    log.info(
        "Extracted %s: %s",
        patient_id,
        manager.safe_for_logging(
            {
                "language": case.detected_language,
                "medications": len(case.medications),
                "lab_tests": len(case.lab_tests),
                "bill_total": case.bill.total_amount,
                "method": case.extraction_method,
            }
        ),
    )

    return {
        "agent": AGENT_NAME,
        "framework": FRAMEWORK,
        "patient_id": patient_id,
        "trace_id": trace_id,
        "case": case.model_dump(mode="json"),
        "documents_present": case.doc_types_present,
        "detected_language": case.detected_language,
        "extraction_method": case.extraction_method,
        "notes": list(final_state.get("notes", [])),
        "audit_trail": audit.dump(),
        "guardrail_events": [e.model_dump(mode="json") for e in manager.events],
        "mcp_primitives_used": ["tools", "resources", "prompts"],
    }


def main() -> None:
    from ..a2a_layer import run_agent_server

    run_agent_server(AGENT_KEY, handle, artifact_name="extracted_case")


if __name__ == "__main__":
    main()
