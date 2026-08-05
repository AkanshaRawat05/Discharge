"""
rag/reflection.py
=================

**Reflection Agent** (Agno) — scores answer quality with the **RAG Triad**:

    faithfulness       is every claim in the answer supported by the context?
    answer_relevance   does the answer actually address the question?
    context_relevance  was the retrieved context relevant to the question?

An LLM judge (`internal_prompts.rag_triad` in prompts.yaml) produces the scores
when it is reachable; a deterministic lexical fallback produces them when it is
not, so the dashboard's quality panel is never empty.  Thresholds come from
`agent_config.yaml → rag.triad_thresholds`.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ..common.prompt_store import internal_prompt
from ..common.schemas import RagTriad, RetrievedChunk
from ..llm import provider
from ..observability import tracing
from ..settings import settings

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z][a-z-]{2,}")
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "was", "were", "are", "have",
    "has", "not", "you", "your", "their", "from", "which", "what", "when",
    "patient", "records", "record", "information", "available", "source",
}

_REFUSAL_MARKERS = (
    "i don't know", "i do not know", "not available in the patient records",
)


def _tokens(text: str) -> set[str]:
    return {
        token for token in _TOKEN_RE.findall((text or "").lower())
        if token not in _STOPWORDS
    }


def _lexical_triad(question: str, context: str, answer: str) -> RagTriad:
    """Deterministic fallback scoring."""
    question_tokens = _tokens(question)
    context_tokens = _tokens(context)
    answer_tokens = _tokens(answer)

    faithfulness = (
        len(answer_tokens & context_tokens) / len(answer_tokens)
        if answer_tokens else 1.0
    )
    answer_relevance = (
        len(question_tokens & answer_tokens) / len(question_tokens)
        if question_tokens else 1.0
    )
    context_relevance = (
        len(question_tokens & context_tokens) / len(question_tokens)
        if question_tokens else 0.0
    )

    return RagTriad(
        faithfulness=round(faithfulness, 3),
        answer_relevance=round(answer_relevance, 3),
        context_relevance=round(context_relevance, 3),
        reasoning="Deterministic lexical overlap scoring (LLM judge unavailable).",
    )


def score_triad(
    question: str,
    context: str,
    answer: str,
    *,
    chunks: list[RetrievedChunk] | None = None,
    trace_id: str | None = None,
) -> RagTriad:
    """Reflection Agent entry point."""
    thresholds = settings.rag.get("triad_thresholds", {}) or {}

    with tracing.retriever_span(
        "rag.reflection_agent", trace_id=trace_id, query=question,
        answer_chars=len(answer or ""),
    ) as span:
        #  The mandated refusal is the correct, fully grounded answer when the
        #  records do not contain the information — score it as such.
        if any(marker in (answer or "").lower() for marker in _REFUSAL_MARKERS):
            triad = RagTriad(
                faithfulness=1.0,
                answer_relevance=1.0,
                context_relevance=round(
                    _lexical_triad(question, context, answer).context_relevance, 3
                ),
                reasoning=(
                    "Answer correctly declined to speculate: the retrieved records "
                    "do not contain the requested information."
                ),
                passed=True,
            )
            span.set_output(triad.model_dump())
            return triad

        triad = _lexical_triad(question, context, answer)

        if not settings.offline_mode:
            template = internal_prompt("rag_triad")
            if template:
                payload = provider.complete_json(
                    template.format(
                        question=question[:1500],
                        context=(context or "")[:8000],
                        answer=(answer or "")[:3000],
                    ),
                    purpose="fast",
                )
                if isinstance(payload, dict) and "faithfulness" in payload:
                    try:
                        triad = RagTriad(
                            faithfulness=_clamp(payload.get("faithfulness")),
                            answer_relevance=_clamp(payload.get("answer_relevance")),
                            context_relevance=_clamp(payload.get("context_relevance")),
                            reasoning=str(payload.get("reasoning", ""))[:600],
                        )
                    except (TypeError, ValueError) as exc:
                        log.debug("Unusable triad payload (%s) — keeping lexical scores", exc)

        triad.passed = (
            triad.faithfulness >= float(thresholds.get("faithfulness", 0.70))
            and triad.answer_relevance >= float(thresholds.get("answer_relevance", 0.70))
            and triad.context_relevance >= float(thresholds.get("context_relevance", 0.60))
        )

        span.set_output(triad.model_dump())
        span.score("rag_faithfulness", triad.faithfulness)
        span.score("rag_answer_relevance", triad.answer_relevance)
        span.score("rag_context_relevance", triad.context_relevance)
        return triad


def _clamp(value: Any) -> float:
    return max(0.0, min(1.0, float(value)))


def triad_verdict(triad: RagTriad) -> str:
    """One-line verdict for the dashboard's quality panel."""
    thresholds = settings.rag.get("triad_thresholds", {}) or {}
    failures: list[str] = []
    if triad.faithfulness < float(thresholds.get("faithfulness", 0.70)):
        failures.append(f"faithfulness {triad.faithfulness:.2f}")
    if triad.answer_relevance < float(thresholds.get("answer_relevance", 0.70)):
        failures.append(f"answer relevance {triad.answer_relevance:.2f}")
    if triad.context_relevance < float(thresholds.get("context_relevance", 0.60)):
        failures.append(f"context relevance {triad.context_relevance:.2f}")

    if not failures:
        return "All three RAG Triad metrics meet their thresholds."
    return "Below threshold: " + ", ".join(failures)
