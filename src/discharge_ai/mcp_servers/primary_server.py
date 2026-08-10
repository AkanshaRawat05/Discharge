"""
mcp_servers/primary_server.py
=============================

**Primary MCP Server — Clinical Tools Server**
port 8200, streamable-HTTP, path `/clinicaltools`

Demonstrates all six MCP primitives (specification Table 9):

┌─────────────┬──────────────────────────────────────────────────────────────┐
│ Tools       │ 6 tools via `@mcp.tool()`                                    │
│ Resources   │ 6 resources via `@mcp.resource()` (2 templated)              │
│ Prompts     │ 5 prompts via `@mcp.prompt()`                                │
│ Sampling    │ `medical_lang_bridge` → `ctx.session.create_message()` with   │
│             │ `ModelPreferences` model hints; the *client* runs inference   │
│ Elicitation │ `clinical_rules_engine` → `ctx.elicit()` with a Pydantic      │
│             │ schema; handles accept / decline / cancel                     │
│ Roots       │ `clinical_watcher` → `ctx.session.list_roots()`; no raw paths │
│             │ as tool parameters; `Path.relative_to()` traversal guard      │
└─────────────┴──────────────────────────────────────────────────────────────┘

Run:
    python -m discharge_ai.mcp_servers.primary_server
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ModelHint, ModelPreferences, SamplingMessage, TextContent
from pydantic import BaseModel, Field

from ..common import prompt_store
from ..common.doc_loader import (
    RootAccessError,
    load_document,
    load_patient_documents,
    resolve_within_root,
    scan_incoming,
)
from ..common.parsing import build_extracted_case
from ..common.rules import (
    completeness_resource_text,
    cross_validation_resource_text,
    rules_version,
)
from ..common.schemas import (
    ElicitationOutcome,
    ExtractedCase,
    MissingField,
)
from ..common.terminology import (
    detect_language,
    expand_abbreviations,
    full_abbreviation_map,
    language_name,
)
from ..ehr import ehr_client
from ..observability import tracing
from ..settings import configure_logging, settings
from ..validation import check_completeness, cross_validate, elicitation_schema_for

log = logging.getLogger(__name__)

mcp = FastMCP(
    name="clinical-tools-server",
    instructions=(
        "Primary Clinical Tools MCP Server for St. Marian Regional Medical "
        "Center. Exposes discharge-document discovery (Roots-scoped), "
        "extraction, translation (via Sampling), completeness validation (via "
        "Elicitation), EHR cross-validation and audit reporting. Clinical rules "
        "and document text are available as Resources; agent prompts as Prompts."
    ),
    host=settings.service("mcp_primary")["host"],
    port=int(settings.service("mcp_primary")["port"]),
    streamable_http_path=settings.service("mcp_primary")["path"],
)


# =========================================================================== #
#  RESOURCES  (MCP primitive: Resources)
# =========================================================================== #
@mcp.resource(
    "resource://clinical-rules/completeness",
    name="clinical-rules-completeness",
    description="Completeness rules from rules.yaml (specification Table 3).",
    mime_type="application/json",
)
def resource_completeness_rules() -> str:
    return completeness_resource_text()


@mcp.resource(
    "resource://clinical-rules/cross-validation",
    name="clinical-rules-cross-validation",
    description="Cross-validation rules + risk matrix from rules.yaml (Table 4).",
    mime_type="application/json",
)
def resource_cross_validation_rules() -> str:
    return cross_validation_resource_text()


@mcp.resource(
    "resource://discharge-report/{patient_id}",
    name="discharge-report",
    description="Raw discharge document text for a patient.",
    mime_type="text/plain",
)
def resource_discharge_report(patient_id: str) -> str:
    documents = load_patient_documents(patient_id)
    document = documents.get("discharge_report")
    if document is None:
        return f"No discharge report found for patient {patient_id}."
    header = f"# source: {document.name} (format={document.fmt}, ocr={document.ocr_used})\n\n"
    return header + (document.text or "[no extractable text]")


@mcp.resource(
    "resource://lab-report/{patient_id}",
    name="lab-report",
    description="Raw lab report text for a patient.",
    mime_type="text/plain",
)
def resource_lab_report(patient_id: str) -> str:
    documents = load_patient_documents(patient_id)
    document = documents.get("lab_report")
    if document is None:
        return f"No lab report found for patient {patient_id}."
    header = f"# source: {document.name} (format={document.fmt}, ocr={document.ocr_used})\n\n"
    return header + (document.text or "[no extractable text]")


@mcp.resource(
    "resource://report-template/html",
    name="report-template-html",
    description="Jinja2 HTML template for the clinician-facing discharge summary.",
    mime_type="text/html",
)
def resource_report_template() -> str:
    template_path = settings.path("html_template", create=False)
    if not template_path.exists():
        return "<!-- discharge_summary.html template not found -->"
    return template_path.read_text(encoding="utf-8")


@mcp.resource(
    "resource://medical-abbreviations",
    name="medical-abbreviations",
    description="Medical abbreviation → expansion dictionary (rules.yaml + clinical set).",
    mime_type="application/json",
)
def resource_medical_abbreviations() -> str:
    mapping = full_abbreviation_map()
    return json.dumps(
        {
            "rules_version": rules_version(),
            "count": len(mapping),
            "abbreviations": dict(sorted(mapping.items())),
        },
        indent=2,
        ensure_ascii=False,
    )


# =========================================================================== #
#  PROMPTS  (MCP primitive: Prompts)
# =========================================================================== #
@mcp.prompt(
    name="discharge-extraction-prompt",
    description="Extract structured clinical fields from discharge/lab/bill documents.",
)
def prompt_discharge_extraction(language: str = "English", doc_types: str = "") -> str:
    return prompt_store.render_mcp_prompt(
        "discharge-extraction-prompt", language=language, doc_types=doc_types
    )


@mcp.prompt(
    name="ehr-cross-validation-prompt",
    description="Explain deterministic EHR cross-validation findings to a clinician.",
)
def prompt_ehr_cross_validation(patient_id: str) -> str:
    return prompt_store.render_mcp_prompt(
        "ehr-cross-validation-prompt", patient_id=patient_id
    )


@mcp.prompt(
    name="abbreviation-normalization-prompt",
    description="Translate clinical text to English and expand medical abbreviations.",
)
def prompt_abbreviation_normalization(source_language: str = "English") -> str:
    return prompt_store.render_mcp_prompt(
        "abbreviation-normalization-prompt", source_language=source_language
    )


@mcp.prompt(
    name="summary-generation-prompt",
    description="Write a patient-friendly discharge summary section.",
)
def prompt_summary_generation(risk_level: str = "Low", audience: str = "patient") -> str:
    return prompt_store.render_mcp_prompt(
        "summary-generation-prompt", risk_level=risk_level, audience=audience
    )


@mcp.prompt(
    name="rag-answer-prompt",
    description="Answer an administrator question strictly from retrieved patient records.",
)
def prompt_rag_answer(context_length: str = "0") -> str:
    return prompt_store.render_mcp_prompt(
        "rag-answer-prompt", context_length=context_length
    )


# =========================================================================== #
#  TOOL 1 — Clinical Watcher   (Tools + ROOTS)
# =========================================================================== #
async def _authorised_roots(ctx: Context) -> list[Path]:
    """Discover authorised folders through `ctx.list_roots()`.

    This is the Roots primitive doing its job: the agent declares
    `file:///…/Data/incoming` as a Root when it opens the MCP connection, and
    the server *asks* for that list instead of accepting a path parameter.
    """
    roots: list[Path] = []

    try:
        result = await ctx.session.list_roots()
        for root in result.roots:
            uri = str(root.uri)
            try:
                from urllib.parse import unquote, urlparse

                parsed = urlparse(uri)
                if parsed.scheme and parsed.scheme != "file":
                    log.warning("Ignoring non-file Root URI: %s", uri)
                    continue
                raw = unquote(parsed.path or "")
                #  Windows file URIs look like file:///C:/… → strip the leading /
                if len(raw) > 2 and raw[0] == "/" and raw[2] == ":":
                    raw = raw[1:]
                candidate = Path(raw).resolve()
                if candidate.exists():
                    roots.append(candidate)
                else:
                    log.warning("Declared Root does not exist on this host: %s", uri)
            except Exception as exc:  # noqa: BLE001
                log.warning("Unparsable Root URI %s: %s", uri, exc)
    except Exception as exc:  # noqa: BLE001
        #  The client did not advertise the roots capability.  Refusing to fall
        #  back to a wide-open filesystem is the whole point of Roots, so we
        #  fall back only to the single configured workspace.
        log.info(
            "ctx.list_roots() unavailable (%s) — restricting to the configured "
            "input workspace.", type(exc).__name__,
        )

    if not roots:
        roots = [settings.path("input_root")]
    return roots


@mcp.tool(
    name="clinical_watcher",
    description=(
        "Detect new discharge/lab/bill documents inside the Roots-scoped "
        "workspace. Folder discovery uses ctx.list_roots(); no raw filesystem "
        "path is ever accepted as a parameter."
    ),
)
async def clinical_watcher(
    ctx: Context,
    patient_id: str | None = None,
    subpath: str | None = None,
) -> dict[str, Any]:
    """Scan for patient documents within the authorised MCP Roots.

    Args:
        patient_id: optional filter, e.g. "P1019".
        subpath:    optional *relative* folder inside a Root (e.g.
                    "doctor_reports"). Absolute paths and any `..` escape are
                    rejected by the `Path.relative_to()` guard.
    """
    with tracing.tool_span("clinical_watcher", params={"patient_id": patient_id,
                                                      "subpath": subpath}) as span:
        roots = await _authorised_roots(ctx)
        root_uris = [f"file:///{str(root).replace(chr(92), '/').lstrip('/')}" for root in roots]

        #  Path-traversal prevention: resolve the requested subpath *inside* a
        #  declared root and prove containment before touching the filesystem.
        scan_roots: list[Path] = []
        if subpath:
            for root in roots:
                try:
                    scan_roots.append(resolve_within_root(subpath, root))
                except RootAccessError as exc:
                    span.event("roots.access_denied", root=str(root), subpath=subpath)
                    return {
                        "error": "access_denied",
                        "detail": str(exc),
                        "authorised_roots": root_uris,
                        "hint": (
                            "Only relative paths inside a declared Root are "
                            "permitted; '..' and absolute paths are rejected."
                        ),
                    }
        else:
            scan_roots = roots

        catalogue: dict[str, dict[str, list[str]]] = {}
        for root in scan_roots:
            discovered = scan_incoming(root if root in roots else root.parent)
            for pid, documents in discovered.items():
                if patient_id and pid.upper() != patient_id.upper():
                    continue
                catalogue.setdefault(pid, {}).update(documents)

        payload = {
            "authorised_roots": root_uris,
            "roots_discovered_via": "ctx.list_roots()",
            "scanned": [str(path) for path in scan_roots],
            "patient_count": len(catalogue),
            "patients": catalogue,
            "new_cases": sorted(catalogue.keys()),
            "rules_version": rules_version(),
        }
        span.set_output({"patient_count": len(catalogue), "patients": sorted(catalogue)})
        return payload


# =========================================================================== #
#  TOOL 2 — Clinical Data Harvester   (Tools)
# =========================================================================== #
@mcp.tool(
    name="clinical_data_harvester",
    description=(
        "Extract text, tables and structured fields from a patient's discharge "
        "report, lab report and hospital bill (txt / json / pdf / docx / scanned "
        "image with OCR sidecar)."
    ),
)
async def clinical_data_harvester(
    ctx: Context,
    patient_id: str,
    include_raw_text: bool = False,
) -> dict[str, Any]:
    """Harvest all documents for one patient into a structured case payload."""
    with tracing.tool_span(
        "clinical_data_harvester", params={"patient_id": patient_id}
    ) as span:
        documents = load_patient_documents(patient_id)
        if not documents:
            span.set_output({"error": "no_documents"})
            return {
                "patient_id": patient_id,
                "error": "no_documents_found",
                "detail": f"No documents found for patient {patient_id}",
            }

        case = build_extracted_case(patient_id, documents)
        payload = case.model_dump(mode="json")

        if not include_raw_text:
            payload["raw_text"] = {
                key: f"[{len(value)} chars — request include_raw_text=true]"
                for key, value in case.raw_text.items()
            }

        payload["documents"] = [doc.to_payload() for doc in documents.values()]
        payload["tables_found"] = sum(len(doc.tables) for doc in documents.values())

        span.set_output(
            {
                "documents": len(documents),
                "medications": len(case.medications),
                "lab_tests": len(case.lab_tests),
                "language": case.detected_language,
            }
        )
        return payload


# =========================================================================== #
#  TOOL 3 — Medical Lang Bridge   (Tools + SAMPLING)
# =========================================================================== #
@mcp.tool(
    name="medical_lang_bridge",
    description=(
        "Translate clinical text to English and normalise medical abbreviations. "
        "Uses MCP Sampling: the server requests inference from the CALLING "
        "agent's LLM via ctx.session.create_message() with ModelPreferences "
        "hints, so LLM resource management stays a client responsibility."
    ),
)
async def medical_lang_bridge(
    ctx: Context,
    text: str,
    source_language: str | None = None,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    """Translate + normalise clinical text through MCP Sampling.

    The server never calls an LLM itself.  It builds the request (text, system
    prompt fetched from its own Prompts primitive, and ModelPreferences hinting
    `nova-lite` for multilingual content or `command-r-plus` for English) and
    asks the client to run it.
    """
    with tracing.tool_span(
        "medical_lang_bridge",
        params={"source_language": source_language, "chars": len(text or "")},
    ) as span:
        language_code = detect_language(text or "", source_language)
        is_multilingual = language_code != "en"

        sampling_cfg = settings.sampling_cfg
        hints_cfg = sampling_cfg.get("model_hints", {})
        primary_hint = (
            hints_cfg.get("multilingual", "nova-lite")
            if is_multilingual
            else hints_cfg.get("english", "command-r-plus")
        )
        secondary_hint = (
            hints_cfg.get("english", "command-r-plus")
            if is_multilingual
            else hints_cfg.get("multilingual", "nova-lite")
        )

        preferences = ModelPreferences(
            hints=[ModelHint(name=primary_hint), ModelHint(name=secondary_hint)],
            intelligencePriority=float(sampling_cfg.get("intelligence_priority", 0.6)),
            speedPriority=float(sampling_cfg.get("speed_priority", 0.4)),
            costPriority=0.3,
        )

        #  The system prompt comes from this server's own Prompts primitive.
        system_prompt = prompt_store.render_mcp_prompt(
            "abbreviation-normalization-prompt",
            source_language=language_name(language_code),
        )

        #  Deterministic abbreviation expansion always runs, so the tool returns
        #  something useful even if sampling is unsupported by the client.
        expanded_text, applied = expand_abbreviations(text or "")

        result: dict[str, Any] = {
            "source_language": language_code,
            "source_language_name": language_name(language_code),
            "sampling_requested": True,
            "model_preferences": preferences.model_dump(exclude_none=True),
            "expanded_abbreviations": applied,
            "translated_text": expanded_text,
            "translation_confidence": None,
            "sampling_model": None,
            "sampling_status": "pending",
        }

        try:
            sampling_result = await ctx.session.create_message(
                messages=[
                    SamplingMessage(
                        role="user",
                        content=TextContent(type="text", text=(text or "")[:12000]),
                    )
                ],
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                temperature=0.0,
                model_preferences=preferences,
                related_request_id=ctx.request_id,
            )

            content = sampling_result.content
            raw = content.text if isinstance(content, TextContent) else str(content)
            result["sampling_model"] = sampling_result.model
            result["sampling_status"] = "completed"
            result["raw_sampling_response"] = raw[:4000]

            from ..llm.provider import extract_json

            payload = extract_json(raw)
            if isinstance(payload, dict) and payload.get("translated_text"):
                translated = str(payload["translated_text"])
                #  Expand abbreviations in the translated text too.
                translated, extra = expand_abbreviations(translated)
                result["translated_text"] = translated
                result["expanded_abbreviations"] = (
                    (payload.get("expanded_abbreviations") or []) + applied + extra
                )
                confidence = payload.get("translation_confidence")
                if confidence is not None:
                    try:
                        result["translation_confidence"] = max(
                            0.0, min(1.0, float(confidence))
                        )
                    except (TypeError, ValueError):
                        pass
                result["confidence_reason"] = payload.get("confidence_reason", "")
            elif raw.strip():
                #  Model answered in prose rather than JSON — still usable.
                translated, extra = expand_abbreviations(raw.strip())
                result["translated_text"] = translated
                result["expanded_abbreviations"] = applied + extra
                result["sampling_status"] = "completed_non_json"

        except Exception as exc:  # noqa: BLE001
            log.warning(
                "MCP sampling unavailable (%s: %s) — returning deterministic "
                "abbreviation normalisation only.", type(exc).__name__, exc,
            )
            result["sampling_status"] = "unavailable"
            result["sampling_error"] = f"{type(exc).__name__}: {exc}"
            tracing.record_error(
                "mcp.sampling", exc,
                fallback_action="deterministic abbreviation expansion only",
            )

        tracing.record_sampling_event(
            trace_id=None,
            server_preferences=result["model_preferences"],
            client_model=result.get("sampling_model") or "unavailable",
            source_language=language_code,
            result_preview=str(result["translated_text"])[:300],
            confidence=result.get("translation_confidence"),
        )
        span.set_output(
            {
                "status": result["sampling_status"],
                "model": result.get("sampling_model"),
                "confidence": result.get("translation_confidence"),
                "abbreviations_expanded": len(result["expanded_abbreviations"]),
            }
        )
        return result


# =========================================================================== #
#  TOOL 4 — Clinical Rules Engine   (Tools + ELICITATION)
# =========================================================================== #
class RulesEngineResult(BaseModel):
    """Structured result of a completeness validation run."""

    patient_id: str
    completeness_score: float
    total_required: int
    present: int
    blocking_missing: list[str] = Field(default_factory=list)
    non_blocking_missing: list[str] = Field(default_factory=list)
    prescription_gaps: list[dict[str, Any]] = Field(default_factory=list)
    auto_generation_blocked: bool = False
    elicitation_outcome: str = ElicitationOutcome.NOT_REQUESTED.value
    elicitation_schema: dict[str, Any] = Field(default_factory=dict)
    reviewer_values: dict[str, Any] = Field(default_factory=dict)
    unresolved_fields: list[str] = Field(default_factory=list)
    note: str = ""
    rules_version: str = ""


@mcp.tool(
    name="clinical_rules_engine",
    description=(
        "Validate discharge/lab/bill completeness against rules.yaml. When "
        "NON-BLOCKING fields are missing, uses MCP Elicitation (ctx.elicit()) "
        "with a Pydantic schema to collect them from the human reviewer, and "
        "handles all three outcomes: accept / decline / cancel."
    ),
)
async def clinical_rules_engine(
    ctx: Context,
    patient_id: str,
    case: dict | None = None,
    elicit_missing: bool = True,
) -> dict[str, Any]:
    """Completeness validation with MCP Elicitation for missing values.

    Args:
        patient_id:     the patient to validate.
        case:           optional pre-extracted `ExtractedCase` payload. When
                        omitted the tool harvests the documents itself.
        elicit_missing: set false to validate without prompting a reviewer.
    """
    with tracing.tool_span(
        "clinical_rules_engine", params={"patient_id": patient_id}
    ) as span:
        extracted = _load_case(patient_id, case)
        if extracted is None:
            return {
                "patient_id": patient_id,
                "error": "no_documents_found",
                "detail": f"No documents found for patient {patient_id}",
            }

        completeness = check_completeness(extracted)
        result = RulesEngineResult(
            patient_id=extracted.patient_id or patient_id,
            completeness_score=completeness.score,
            total_required=completeness.total_required,
            present=completeness.present,
            blocking_missing=completeness.blocking_missing,
            non_blocking_missing=completeness.non_blocking_missing,
            prescription_gaps=completeness.prescription_gaps,
            #  Blocking gaps stop auto-generation; HITL must intervene.
            auto_generation_blocked=completeness.has_blocking,
            rules_version=rules_version(),
        )

        non_blocking = [f for f in completeness.missing_fields if not f.blocking]

        if not non_blocking:
            result.note = "No non-blocking gaps — elicitation not required."
            span.set_output(result.model_dump())
            return result.model_dump()

        if not elicit_missing:
            result.elicitation_outcome = ElicitationOutcome.NOT_REQUESTED.value
            result.unresolved_fields = [f.field for f in non_blocking]
            result.note = "Elicitation disabled by caller; gaps left unresolved."
            span.set_output(result.model_dump())
            return result.model_dump()

        # ---- MCP ELICITATION ------------------------------------------------
        schema_model, json_schema = elicitation_schema_for(non_blocking)
        result.elicitation_schema = json_schema

        message = (
            f"Patient {result.patient_id}: {len(non_blocking)} non-blocking "
            "field(s) are missing from the discharge documentation. Please "
            "supply the values you can confirm from the chart — leave a field "
            "blank if it is genuinely unknown."
        )

        try:
            elicit_result = await ctx.elicit(message=message, schema=schema_model)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "MCP elicitation unavailable (%s: %s) — flagging fields for HITL.",
                type(exc).__name__, exc,
            )
            result.elicitation_outcome = ElicitationOutcome.DECLINE.value
            result.unresolved_fields = [f.field for f in non_blocking]
            result.note = (
                "Client does not support elicitation "
                f"({type(exc).__name__}); fields flagged for the HITL dashboard."
            )
            tracing.record_error(
                "mcp.elicitation", exc, fallback_action="flag for HITL"
            )
            span.set_output(result.model_dump())
            return result.model_dump()

        action = getattr(elicit_result, "action", "decline")

        if action == "accept":
            data = getattr(elicit_result, "data", None)
            values = (
                data.model_dump(exclude_none=True) if hasattr(data, "model_dump")
                else dict(data or {})
            )
            values = {k: v for k, v in values.items() if v not in (None, "")}

            result.elicitation_outcome = ElicitationOutcome.ACCEPT.value
            result.reviewer_values = values
            result.unresolved_fields = [
                f.field for f in non_blocking
                if _elicit_key(f) not in values
            ]
            result.note = (
                f"Reviewer supplied {len(values)} value(s); continuing validation."
            )

        elif action == "decline":
            result.elicitation_outcome = ElicitationOutcome.DECLINE.value
            result.unresolved_fields = [f.field for f in non_blocking]
            result.note = (
                "Reviewer declined to provide the missing values — marked "
                "unresolved and flagged for HITL review."
            )

        else:  # "cancel"
            result.elicitation_outcome = ElicitationOutcome.CANCEL.value
            result.unresolved_fields = [f.field for f in non_blocking]
            result.auto_generation_blocked = True
            result.note = (
                "Reviewer cancelled the elicitation — validation aborted and "
                "escalated to a senior clinician."
            )

        tracing.record_elicitation_event(
            trace_id=None,
            schema_sent=json_schema,
            reviewer_response=result.reviewer_values,
            action=result.elicitation_outcome,
        )
        span.set_output(result.model_dump())
        return result.model_dump()


def _elicit_key(field: MissingField) -> str:
    return (
        f"{field.field}_row{field.row}"
        if field.document == "prescription" and field.row
        else field.field
    )


# =========================================================================== #
#  TOOL 5 — EHR Validation   (Tools)
# =========================================================================== #
@mcp.tool(
    name="ehr_validation",
    description=(
        "Cross-check extracted discharge data against the Mock EHR REST API "
        "(medication orders, allergy registry, care plan, labs) and apply the "
        "Table 4 cross-validation rules."
    ),
)
async def ehr_validation(
    ctx: Context,
    patient_id: str,
    case: dict | None = None,
    translation_confidence: float | None = None,
) -> dict[str, Any]:
    """Run every cross-validation rule for one patient."""
    with tracing.tool_span("ehr_validation", params={"patient_id": patient_id}) as span:
        extracted = _load_case(patient_id, case)
        if extracted is None:
            return {
                "patient_id": patient_id,
                "error": "no_documents_found",
                "detail": f"No documents found for patient {patient_id}",
            }

        if translation_confidence is not None:
            extracted.translation_confidence = float(translation_confidence)

        bundle = ehr_client.bundle(patient_id)
        findings = cross_validate(extracted, bundle)

        payload = {
            "patient_id": extracted.patient_id or patient_id,
            "ehr_source": bundle.get("source"),
            "ehr_degraded": bool(bundle.get("degraded_reason")),
            "findings": [finding.model_dump(mode="json") for finding in findings],
            "critical_count": sum(1 for f in findings if f.severity.value == "Critical"),
            "warning_count": sum(1 for f in findings if f.severity.value == "Warning"),
            "blocks_discharge": any(f.blocks_discharge for f in findings),
            "ehr_snapshot": {
                "medications": bundle.get("medications", []),
                "allergies": bundle.get("allergies", []),
                "abnormal_labs": bundle.get("abnormal_labs", []),
                "care_plan": bundle.get("care_plan", {}),
                "primary_dx": (bundle.get("demographics") or {}).get("primary_dx", []),
            },
            "rules_version": rules_version(),
        }
        span.set_output(
            {
                "findings": len(findings),
                "critical": payload["critical_count"],
                "blocks_discharge": payload["blocks_discharge"],
            }
        )
        return payload


# =========================================================================== #
#  TOOL 6 — Clinical Insight Reporter   (Tools + RESOURCES)
# =========================================================================== #
@mcp.tool(
    name="clinical_insight_reporter",
    description=(
        "Generate the discharge audit/risk report as JSON (system consumption) "
        "and HTML (clinician-friendly), using the HTML template served from "
        "resource://report-template/html. Writes both to Data/reports/."
    ),
)
async def clinical_insight_reporter(
    ctx: Context,
    patient_id: str,
    report: dict,
    write_files: bool = True,
) -> dict[str, Any]:
    """Render an audit report from a `ValidationReport` payload."""
    with tracing.tool_span(
        "clinical_insight_reporter", params={"patient_id": patient_id}
    ) as span:
        from ..common.schemas import ValidationReport
        from ..reporting.report_builder import build_reports

        try:
            payload = json.loads(report) if isinstance(report, str) else report
            validation_report = ValidationReport.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            span.fail(exc)
            return {
                "error": "invalid_report_payload",
                "detail": f"{type(exc).__name__}: {exc}",
            }

        #  The HTML renderer reads its template through the same content the
        #  Resources primitive serves, keeping one source of truth.
        artifacts = build_reports(validation_report, write_files=write_files)
        span.set_output({key: str(value) for key, value in artifacts.items()})
        return {
            "patient_id": validation_report.patient_id,
            "risk_level": validation_report.risk.level.value,
            "recommendation": validation_report.risk.recommendation.value,
            "artifacts": {key: str(value) for key, value in artifacts.items()},
            "template_resource": "resource://report-template/html",
            "rules_version": validation_report.rules_version or rules_version(),
        }


# =========================================================================== #
#  Helpers
# =========================================================================== #
def _load_case(patient_id: str, case: dict | str | None) -> ExtractedCase | None:
    """Rehydrate an `ExtractedCase` from a payload, or harvest it from disk."""
    if case:
        try:
            payload = json.loads(case) if isinstance(case, str) else case
            return ExtractedCase.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "The supplied case payload for %s could not be parsed (%s) — "
                "re-harvesting from disk.", patient_id, exc,
            )

    documents = load_patient_documents(patient_id)
    if not documents:
        return None
    return build_extracted_case(patient_id, documents)


# --------------------------------------------------------------------------- #
def main() -> None:
    configure_logging("mcp-primary")
    service = settings.service("mcp_primary")
    log.info(
        "Primary MCP Clinical Tools Server → http://%s:%s%s "
        "(6 tools · 6 resources · 5 prompts · sampling · elicitation · roots)",
        service["host"], service["port"], service["path"],
    )
    log.info("MCP Root boundary: %s", settings.path("input_root"))
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
