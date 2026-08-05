"""
agents/rag_agent.py
===================

**Clinical RAG Q&A Agent** — Agno — A2A :8105 — **STREAMING**
MCP primitives: MultiMCPTools (both servers) + Prompts

Orchestrates the five Agno RAG roles of specification Table 5:

    1. Indexing Agent      parses + indexes documents into FAISS
    2. Retrieval Agent     question → embedding → top-k chunks
    3. Augmentation Agent  re-ranks the chunks by keyword relevance
    4. Generation Agent    grounded answer; prompt fetched via MCP Prompts,
                           `agno.Agent` + `MultiMCPTools` + `SqliteDb` (last 3
                           turns) + async `arun()`
    5. Reflection Agent    RAG Triad — faithfulness / answer relevance /
                           context relevance

Responsible-AI guardrails on the Q&A path:

* `PromptInjectionGuard` screens the question (reject / sanitise) before anything
  else happens, and neutralises instructions embedded in retrieved chunks;
* `HallucinationChecker` gates the finished answer at faithfulness < 0.70 and
  requests one regeneration before refusing;
* `PIIRedactor` masks identifiers in the answer that leaves the agent.

Run:
    python -m discharge_ai.agents.rag_agent
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from ..common.schemas import RagAnswer, RagTriad
from ..guardrails import guardrails
from ..llm import provider
from ..observability import tracing
from ..rag import augmentation, generation, indexing, reflection, retrieval
from ..settings import settings
from .base import AuditTrail, agent_mcp, trace_for

log = logging.getLogger(__name__)

AGENT_KEY = "rag"
AGENT_NAME = settings.agent(AGENT_KEY)["name"]
FRAMEWORK = "agno"

OUT_OF_CONTEXT_ANSWER = generation.OUT_OF_CONTEXT_ANSWER

INJECTION_REFUSAL = (
    "That request was blocked by the prompt-injection guard. Please ask a "
    "question about the patient records instead — for example: "
    "\"What medications was P1019 discharged on?\""
)


# --------------------------------------------------------------------------- #
#  Pipeline shared by the streaming and non-streaming paths
# --------------------------------------------------------------------------- #
async def _prepare(
    question: str, patient_id: str | None, top_k: int | None,
    trace_id: str | None, audit: AuditTrail, manager: Any,
) -> dict[str, Any]:
    """Guardrails → retrieval → augmentation → MCP prompt."""
    # ---- 1. prompt-injection guard on the incoming question -----------------
    async with audit.step("guard_question", "PromptInjectionGuard", framework=FRAMEWORK):
        injection = await provider.arun_blocking(manager.check_query, question)

    if injection.blocked:
        return {
            "blocked": True,
            "reason": "prompt_injection",
            "patterns": injection.patterns_matched,
        }

    safe_question = injection.sanitised_text or question

    # ---- 2. Retrieval Agent -------------------------------------------------
    async with audit.step("retrieval_agent", "Agno Retrieval Agent", framework=FRAMEWORK):
        chunks = retrieval.retrieve(
            safe_question, patient_id=patient_id, top_k=top_k, trace_id=trace_id
        )

    # ---- 3. Augmentation Agent ---------------------------------------------
    async with audit.step("augmentation_agent", "Agno Augmentation Agent",
                          framework=FRAMEWORK):
        ranked = augmentation.rerank_chunks(safe_question, chunks, trace_id=trace_id)
        #  Retrieved text is data: strip anything that reads like an instruction.
        for chunk in ranked:
            chunk.text = manager.sanitise_context(chunk.text)
        context = augmentation.build_context(ranked)

    # ---- 4. MCP Prompts: fetch the answer prompt ---------------------------
    system_prompt = ""
    async with audit.step("fetch_rag_prompt", "MCP Prompts", framework=FRAMEWORK):
        async with agent_mcp(AGENT_NAME, trace_id=trace_id, servers=("primary",)) as mcp:
            system_prompt = await mcp.get_prompt(
                "rag-answer-prompt", {"context_length": str(len(context))}
            )

    return {
        "blocked": False,
        "question": safe_question,
        "original_question": question,
        "sanitised": injection.is_injection,
        "chunks": ranked,
        "context": context,
        "system_prompt": system_prompt,
    }


def _finalise(
    question: str, answer: str, context: str, chunks: list[Any],
    trace_id: str | None, manager: Any, engine: str,
) -> RagAnswer:
    """Grounding gate + triad scoring + PII redaction."""
    grounding = manager.check_grounding(answer, context)
    out_of_context = OUT_OF_CONTEXT_ANSWER.lower()[:20] in answer.lower()

    if grounding.blocked and not out_of_context:
        log.warning(
            "RAG answer failed the grounding gate (%.2f) — refusing.",
            grounding.faithfulness,
        )
        answer = OUT_OF_CONTEXT_ANSWER
        out_of_context = True

    triad: RagTriad = reflection.score_triad(
        question, context, answer, chunks=chunks, trace_id=trace_id
    )
    answer = manager.redact(answer, purpose="rag_answer")

    return RagAnswer(
        question=question,
        answer=answer,
        chunks=chunks,
        triad=triad,
        guardrail_events=list(manager.events),
        trace_id=trace_id,
        out_of_context=out_of_context,
    )


# --------------------------------------------------------------------------- #
#  Streaming A2A handler
# --------------------------------------------------------------------------- #
async def handle(payload: dict[str, Any], ctx: Any) -> AsyncIterator[Any]:
    """Streaming A2A entry point — token-by-token answer, then the final payload."""
    from ..a2a_layer.executor import StreamEvent

    question = str(
        payload.get("question") or payload.get("query") or payload.get("text") or ""
    ).strip()
    patient_id = (str(payload.get("patient_id") or "").strip().upper() or None)
    top_k = payload.get("top_k")
    session_id = payload.get("session_id")
    trace_id = trace_for(payload, getattr(ctx, "trace_id", None), patient_id or "rag")
    audit = AuditTrail(trace_id)
    manager = guardrails(trace_id)

    # ---- reindex on request (Indexing Agent) -------------------------------
    if payload.get("reindex"):
        async with audit.step("indexing_agent", "Agno Indexing Agent",
                              framework=FRAMEWORK):
            stats = indexing.build_index(trace_id=trace_id)
        yield StreamEvent(
            text=f"Index rebuilt: {stats['chunks']} chunks across "
                 f"{stats['patient_count']} patient(s).\n",
            section="indexing",
        )

    if not question:
        stats = retrieval.store_stats()
        yield StreamEvent(
            data={
                "agent": AGENT_NAME,
                "framework": FRAMEWORK,
                "trace_id": trace_id,
                "answer": "",
                "note": "no question supplied",
                "index_stats": stats,
                "audit_trail": audit.dump(),
            },
            final=True,
        )
        return

    prepared = await _prepare(question, patient_id, top_k, trace_id, audit, manager)

    # ---- injection rejection ------------------------------------------------
    if prepared["blocked"]:
        yield StreamEvent(text=INJECTION_REFUSAL, section="guardrail")
        yield StreamEvent(
            data={
                "agent": AGENT_NAME,
                "framework": FRAMEWORK,
                "trace_id": trace_id,
                "question": question,
                "answer": INJECTION_REFUSAL,
                "blocked_by_guardrail": True,
                "guardrail": "PromptInjectionGuard",
                "patterns_matched": prepared["patterns"],
                "guardrail_events": [e.model_dump(mode="json") for e in manager.events],
                "audit_trail": audit.dump(),
            },
            final=True,
        )
        return

    safe_question = prepared["question"]
    context = prepared["context"]
    chunks = prepared["chunks"]

    yield StreamEvent(
        text="", section="retrieval",
        metadata={
            "chunks_retrieved": len(chunks),
            "sources": [chunk.source for chunk in chunks],
        },
    )

    # ---- 4. Generation Agent (streamed) ------------------------------------
    collected: list[str] = []
    async with audit.step("generation_agent", "Agno Generation Agent",
                          framework=FRAMEWORK):
        async for piece in generation.generate_streaming(
            safe_question, context, prepared["system_prompt"],
            patient_id=patient_id, session_id=session_id, trace_id=trace_id,
        ):
            collected.append(piece)
            yield StreamEvent(text=piece, section="answer")

    raw_answer = "".join(collected).strip()

    # ---- 5. Reflection Agent + guardrails ---------------------------------
    async with audit.step("reflection_agent", "Agno Reflection Agent",
                          framework=FRAMEWORK):
        #  `_finalise` runs the grounding judge and the triad scorer, both
        #  synchronous LLM calls — keep them off the event loop.
        result = await provider.arun_blocking(
            _finalise, safe_question, raw_answer, context, chunks,
            trace_id, manager, "agno",
        )

    #  If a guardrail replaced the streamed text, tell the client explicitly —
    #  the UI has already rendered the original tokens.
    if result.answer.strip() != raw_answer:
        yield StreamEvent(
            text=f"\n\n[guardrail] The streamed answer was replaced: "
                 f"{result.answer}\n",
            section="guardrail",
        )

    yield StreamEvent(
        data={
            "agent": AGENT_NAME,
            "framework": FRAMEWORK,
            "trace_id": trace_id,
            "trace_url": tracing.trace_url(trace_id),
            "question": question,
            "sanitised_question": safe_question if prepared["sanitised"] else None,
            "answer": result.answer,
            "out_of_context": result.out_of_context,
            "prompt_source": result.prompt_source,
            "patient_scope": patient_id,
            "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
            "context_chars": len(context),
            "triad": result.triad.model_dump(mode="json"),
            "triad_verdict": reflection.triad_verdict(result.triad),
            "guardrail_events": [e.model_dump(mode="json") for e in manager.events],
            "index_stats": retrieval.store_stats(),
            "audit_trail": audit.dump(),
            "mcp_primitives_used": ["multi_mcp_tools", "prompts"],
        },
        final=True,
    )


# --------------------------------------------------------------------------- #
#  Non-streaming helper (used by the dashboard and tests)
# --------------------------------------------------------------------------- #
async def ask(
    question: str,
    *,
    patient_id: str | None = None,
    top_k: int | None = None,
    session_id: str | None = None,
    trace_id: str | None = None,
) -> RagAnswer:
    """Answer a question in one shot (no A2A, no streaming)."""
    trace_id = trace_id or tracing.start_case_trace(patient_id or "rag")
    audit = AuditTrail(trace_id)
    manager = guardrails(trace_id)

    prepared = await _prepare(question, patient_id, top_k, trace_id, audit, manager)
    if prepared["blocked"]:
        return RagAnswer(
            question=question,
            answer=INJECTION_REFUSAL,
            guardrail_events=list(manager.events),
            trace_id=trace_id,
            out_of_context=True,
        )

    generated = await generation.generate(
        prepared["question"], prepared["context"], prepared["system_prompt"],
        patient_id=patient_id, session_id=session_id, trace_id=trace_id,
    )
    return await provider.arun_blocking(
        _finalise, prepared["question"], generated["answer"],
        prepared["context"], prepared["chunks"], trace_id, manager,
        generated["engine"],
    )


def main() -> None:
    from ..a2a_layer import run_agent_server

    #  Warm the index at start-up so the first question is not slowed by it.
    try:
        stats = indexing.get_store().stats()
        log.info(
            "FAISS index ready: %d chunks, %d patient(s), provider=%s",
            stats["chunks"], stats["patient_count"], stats["embedding_provider"],
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not warm the FAISS index: %s", exc)

    run_agent_server(AGENT_KEY, handle, artifact_name="rag_answer")


if __name__ == "__main__":
    main()
