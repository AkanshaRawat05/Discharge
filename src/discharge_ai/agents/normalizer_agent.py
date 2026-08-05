"""
agents/normalizer_agent.py
==========================

**Clinical Normalizer Agent** — LangGraph — A2A :8102 — non-streaming
MCP primitives: Tools + Sampling + Prompts

A `StateGraph` with four nodes:

    detect        identify the source language of the record
    translate     MCP Tool `medical_lang_bridge`, which issues
                  `ctx.session.create_message()` back to THIS agent's LLM client
                  (MCP **Sampling**) with `nova-lite` / `command-r-plus` hints
    normalise     expand medical abbreviations (BID → twice daily, PO → by mouth)
                  across diagnoses, prescriptions and instructions
    confidence    compute the translation confidence the risk engine gates on

Confidence is the conservative minimum of a deterministic per-language heuristic
(`configs/agent_config.yaml → normalization`) and whatever the model reported, so
an over-confident model can never suppress the `translation_confidence_min`
guardrail in rules.yaml.

Run:
    python -m discharge_ai.agents.normalizer_agent
"""

from __future__ import annotations

import logging
import re
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from ..common.doc_loader import load_patient_documents
from ..common.parsing import build_extracted_case
from ..common.rules import quality_thresholds
from ..common.schemas import ExtractedCase
from ..common.terminology import (
    detect_language,
    expand_abbreviations,
    language_name,
)
from ..observability import tracing
from ..settings import settings
from .base import AuditTrail, agent_mcp, case_from_payload, require_patient_id, trace_for

log = logging.getLogger(__name__)

AGENT_KEY = "normalizer"
AGENT_NAME = settings.agent(AGENT_KEY)["name"]
FRAMEWORK = "langgraph"

#  Characters outside Latin-1 that indicate untranslated residue.
_NON_LATIN_RE = re.compile(r"[^\x00-\x7FÀ-ɏ -⁯\s]")


class NormalizerState(TypedDict, total=False):
    patient_id: str
    trace_id: str | None
    case: dict[str, Any]
    source_language: str
    translated: dict[str, str]
    sampling: dict[str, Any]
    model_confidence: float | None
    heuristic_confidence: float
    final_confidence: float
    abbreviations: list[dict[str, str]]
    notes: list[str]


def _build_graph(mcp: Any, audit: AuditTrail):
    async def detect(state: NormalizerState) -> NormalizerState:
        case = ExtractedCase.model_validate(state["case"])
        combined = "\n".join(case.raw_text.values())
        language = detect_language(combined, case.detected_language)

        async with audit.step("detect_language", "Clinical Normalizer",
                              framework=FRAMEWORK):
            pass

        return {
            "source_language": language,
            "notes": [f"source language detected: {language} ({language_name(language)})"],
        }

    async def translate(state: NormalizerState) -> NormalizerState:
        """MCP Tool + Sampling: `medical_lang_bridge`."""
        case = ExtractedCase.model_validate(state["case"])
        language = state["source_language"]

        #  Translate the free-text blocks that clinicians actually read.
        blocks: dict[str, str] = {}
        if case.discharge_instructions:
            blocks["discharge_instructions"] = case.discharge_instructions
        if case.follow_up_appointment:
            blocks["follow_up_appointment"] = case.follow_up_appointment
        if case.discharge_diagnosis:
            blocks["discharge_diagnosis"] = "\n".join(case.discharge_diagnosis)
        narrative = case.raw_text.get("narrative") or case.raw_text.get("discharge_report")
        if narrative:
            blocks["narrative"] = narrative[:6000]

        translated: dict[str, str] = {}
        sampling_meta: dict[str, Any] = {}
        confidences: list[float] = []
        abbreviations: list[dict[str, str]] = []
        notes: list[str] = []

        if language == "en":
            #  Still normalise abbreviations for English records.
            for key, text in blocks.items():
                expanded, applied = expand_abbreviations(text)
                translated[key] = expanded
                abbreviations += applied
            notes.append("record already in English — translation skipped")
            return {
                "translated": translated,
                "sampling": {"sampling_status": "not_required", "source_language": "en"},
                "model_confidence": 1.0,
                "abbreviations": abbreviations,
                "notes": notes,
            }

        async with audit.step("translate_via_mcp_sampling",
                              "medical_lang_bridge tool (MCP Sampling)",
                              framework=FRAMEWORK):
            if mcp.is_connected:
                for key, text in blocks.items():
                    result = await mcp.call_tool(
                        "medical_lang_bridge",
                        {"text": text, "source_language": language},
                    )
                    if not isinstance(result, dict):
                        continue
                    translated[key] = str(result.get("translated_text") or text)
                    abbreviations += result.get("expanded_abbreviations") or []
                    if result.get("translation_confidence") is not None:
                        confidences.append(float(result["translation_confidence"]))
                    if not sampling_meta:
                        sampling_meta = {
                            "sampling_status": result.get("sampling_status"),
                            "sampling_model": result.get("sampling_model"),
                            "model_preferences": result.get("model_preferences"),
                            "source_language": result.get("source_language"),
                        }
                notes.append(
                    f"translated {len(translated)} block(s) via MCP Sampling "
                    f"(status={sampling_meta.get('sampling_status')}, "
                    f"model={sampling_meta.get('sampling_model')})"
                )
            else:
                for key, text in blocks.items():
                    expanded, applied = expand_abbreviations(text)
                    translated[key] = expanded
                    abbreviations += applied
                sampling_meta = {"sampling_status": "mcp_unavailable"}
                notes.append(
                    "MCP server unreachable — abbreviation expansion only, no translation"
                )

        return {
            "translated": translated,
            "sampling": sampling_meta,
            "model_confidence": (
                round(sum(confidences) / len(confidences), 3) if confidences else None
            ),
            "abbreviations": abbreviations,
            "notes": notes,
        }

    async def normalise(state: NormalizerState) -> NormalizerState:
        """Expand abbreviations across the structured clinical fields."""
        case = ExtractedCase.model_validate(state["case"])
        translated = dict(state.get("translated") or {})
        applied = list(state.get("abbreviations") or [])

        async with audit.step("normalise_terminology", "Clinical Normalizer",
                              framework=FRAMEWORK):
            #  Medication frequency/route codes are what patients misread most.
            for medication in case.medications:
                for field in ("frequency", "route", "dosage", "remarks", "period"):
                    value = getattr(medication, field, None)
                    if not value:
                        continue
                    expanded, extra = expand_abbreviations(str(value))
                    setattr(medication, field, expanded)
                    applied += extra

            if translated.get("discharge_diagnosis"):
                case.discharge_diagnosis = [
                    line.strip()
                    for line in translated["discharge_diagnosis"].splitlines()
                    if line.strip()
                ] or case.discharge_diagnosis
            if translated.get("discharge_instructions"):
                case.discharge_instructions = translated["discharge_instructions"]
            if translated.get("follow_up_appointment"):
                case.follow_up_appointment = translated["follow_up_appointment"]

            for abnormal in case.abnormal_labs:
                if abnormal.action:
                    abnormal.action, extra = expand_abbreviations(abnormal.action)
                    applied += extra

        case.translated_text = translated
        case.expanded_abbreviations = list(
            {(item.get("abbreviation"), item.get("expansion")): item for item in applied}.values()
        )

        return {
            "case": case.model_dump(mode="json"),
            "abbreviations": case.expanded_abbreviations,
            "notes": [
                f"expanded {len(case.expanded_abbreviations)} distinct abbreviation(s)"
            ],
        }

    async def confidence(state: NormalizerState) -> NormalizerState:
        """Conservative translation-confidence score."""
        case = ExtractedCase.model_validate(state["case"])
        language = state["source_language"]
        config = settings.cfg.get("normalization", {})
        baselines = config.get("language_confidence", {})
        penalties = config.get("penalties", {})

        score = float(baselines.get(language, baselines.get("unknown", 0.6)))
        reasons = [f"baseline for '{language}' = {score:.2f}"]

        #  OCR / handwritten origin.
        if any("OCR" in note or "ocr" in note for note in case.extraction_notes):
            penalty = float(penalties.get("ocr_source", 0.10))
            score -= penalty
            reasons.append(f"-{penalty:.2f} scanned/handwritten source")

        sampling_status = (state.get("sampling") or {}).get("sampling_status")
        if language != "en" and sampling_status not in {"completed", "completed_non_json",
                                                        "not_required"}:
            penalty = float(penalties.get("missing_translation", 0.15))
            score -= penalty
            reasons.append(f"-{penalty:.2f} translation unavailable ({sampling_status})")

        #  Non-Latin characters left in the translated output.
        translated_blob = " ".join((state.get("translated") or {}).values())
        if translated_blob and _NON_LATIN_RE.search(translated_blob):
            penalty = float(penalties.get("untranslated_residue", 0.10))
            score -= penalty
            reasons.append(f"-{penalty:.2f} untranslated characters remain")

        heuristic = max(0.0, min(1.0, round(score, 3)))
        model_confidence = state.get("model_confidence")

        #  Conservative minimum — see agent_config.yaml `blend_strategy`.
        final = (
            round(min(heuristic, float(model_confidence)), 3)
            if model_confidence is not None else heuristic
        )

        case.translation_confidence = final
        case.detected_language = language
        case.translation_notes = reasons + [
            f"model-reported confidence: "
            f"{model_confidence if model_confidence is not None else 'n/a'}",
            f"final (conservative minimum): {final:.2f}",
        ]

        minimum = float(quality_thresholds().get("translation_confidence_min", 0.70))
        async with audit.step("score_translation_confidence", "Clinical Normalizer",
                              framework=FRAMEWORK):
            tracing.record_event(
                "normalization.confidence",
                trace_id=state.get("trace_id"),
                source_language=language,
                heuristic=heuristic,
                model_reported=model_confidence,
                final=final,
                minimum=minimum,
                below_threshold=final < minimum,
            )

        return {
            "case": case.model_dump(mode="json"),
            "heuristic_confidence": heuristic,
            "final_confidence": final,
            "notes": [
                f"translation confidence {final:.2f} "
                f"({'BELOW' if final < minimum else 'meets'} minimum {minimum:.2f})"
            ],
        }

    graph = StateGraph(NormalizerState)
    graph.add_node("detect", detect)
    graph.add_node("translate", translate)
    graph.add_node("normalise", normalise)
    graph.add_node("confidence", confidence)

    graph.add_edge(START, "detect")
    graph.add_edge("detect", "translate")
    graph.add_edge("translate", "normalise")
    graph.add_edge("normalise", "confidence")
    graph.add_edge("confidence", END)

    return graph.compile(checkpointer=MemorySaver())


# --------------------------------------------------------------------------- #
async def handle(payload: dict[str, Any], ctx: Any) -> dict[str, Any]:
    """A2A entry point: `{"patient_id": …, "case": {…}}` → normalised case."""
    patient_id = require_patient_id(payload)
    trace_id = trace_for(payload, getattr(ctx, "trace_id", None), patient_id)
    audit = AuditTrail(trace_id)

    case = case_from_payload(payload)
    if case is None:
        documents = load_patient_documents(patient_id)
        if not documents:
            raise FileNotFoundError(f"No documents found for {patient_id}")
        case = build_extracted_case(patient_id, documents)

    async with agent_mcp(AGENT_NAME, trace_id=trace_id, servers=("primary",)) as mcp:
        graph = _build_graph(mcp, audit)
        final_state = await graph.ainvoke(
            {
                "patient_id": patient_id,
                "trace_id": trace_id,
                "case": case.model_dump(mode="json"),
                "notes": [],
            },
            config={"configurable": {"thread_id": f"normalize:{patient_id}"}},
        )

    normalised = ExtractedCase.model_validate(final_state["case"])
    minimum = float(quality_thresholds().get("translation_confidence_min", 0.70))

    return {
        "agent": AGENT_NAME,
        "framework": FRAMEWORK,
        "patient_id": patient_id,
        "trace_id": trace_id,
        "case": normalised.model_dump(mode="json"),
        "source_language": final_state.get("source_language"),
        "translation_confidence": normalised.translation_confidence,
        "translation_confidence_minimum": minimum,
        "below_threshold": normalised.translation_confidence < minimum,
        "heuristic_confidence": final_state.get("heuristic_confidence"),
        "model_reported_confidence": final_state.get("model_confidence"),
        "sampling": final_state.get("sampling", {}),
        "expanded_abbreviations": normalised.expanded_abbreviations,
        "translation_notes": normalised.translation_notes,
        "notes": list(final_state.get("notes", [])),
        "audit_trail": audit.dump(),
        "mcp_primitives_used": ["tools", "sampling", "prompts"],
    }


def main() -> None:
    from ..a2a_layer import run_agent_server

    run_agent_server(AGENT_KEY, handle, artifact_name="normalised_case")


if __name__ == "__main__":
    main()
