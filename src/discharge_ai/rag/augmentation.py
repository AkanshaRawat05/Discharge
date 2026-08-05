"""
rag/augmentation.py
===================

**Augmentation Agent** (Agno) — re-ranks the retrieved chunks by keyword
relevance before they are handed to the Generation Agent.

The embedding score alone is a weak signal on clinical text: two chunks about the
same patient look nearly identical in vector space even when only one contains
the number the question asks for.  The re-ranker therefore blends:

    0.55 × embedding similarity
  + 0.30 × keyword overlap with the question (content words only)
  + 0.15 × structural priority (structured fact cards and validation reports
           state values explicitly, so they answer factual questions best)

and gives a small bonus when the question asks about a number/date and the chunk
actually contains digits.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ..common.schemas import RetrievedChunk
from ..observability import tracing
from ..settings import settings

log = logging.getLogger(__name__)

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "at",
    "by", "from", "as", "is", "are", "was", "were", "be", "what", "which", "who",
    "whom", "whose", "when", "where", "why", "how", "did", "does", "do", "has",
    "have", "had", "can", "could", "will", "would", "should", "many", "much",
    "any", "all", "there", "this", "that", "these", "those", "it", "its", "his",
    "her", "their", "please", "tell", "me", "show", "list", "give",
}

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9-]{1,}")

#  Chunk types that state facts explicitly.
_PRIORITY_BY_DOC_TYPE = {
    "structured_facts": 1.0,
    "validation_report": 0.9,
    "translation": 0.6,
    "discharge_report": 0.55,
    "lab_report": 0.5,
    "bill": 0.5,
}

_NUMERIC_QUESTION_RE = re.compile(
    r"\b(how much|how many|what dose|dosage|strength|amount|total|cost|price|"
    r"value|level|score|date|when|quantity|mg|percent)\b",
    re.IGNORECASE,
)


def _keywords(question: str) -> set[str]:
    return {
        token for token in _TOKEN_RE.findall((question or "").lower())
        if token not in _STOPWORDS
    }


def rerank_chunks(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    top_k: int | None = None,
    trace_id: str | None = None,
) -> list[RetrievedChunk]:
    """Augmentation Agent entry point."""
    if not chunks:
        return []

    keep = int(top_k or settings.rag.get("rerank_top_k", 3))
    keywords = _keywords(question)
    numeric_question = bool(_NUMERIC_QUESTION_RE.search(question or ""))

    with tracing.retriever_span(
        "rag.augmentation_agent",
        trace_id=trace_id,
        query=question,
        candidates=len(chunks),
        rerank_top_k=keep,
    ) as span:
        #  Normalise embedding scores across the candidate set so the blend is
        #  comparable regardless of the embedding provider's score range.
        scores = [chunk.score for chunk in chunks]
        low, high = min(scores), max(scores)
        span_width = (high - low) or 1.0

        for chunk in chunks:
            similarity = (chunk.score - low) / span_width
            lowered = chunk.text.lower()

            overlap = (
                sum(1 for keyword in keywords if keyword in lowered) / len(keywords)
                if keywords else 0.0
            )
            priority = _PRIORITY_BY_DOC_TYPE.get(chunk.doc_type or "", 0.5)

            blended = 0.55 * similarity + 0.30 * overlap + 0.15 * priority
            if numeric_question and re.search(r"\d", chunk.text):
                blended += 0.05

            chunk.rerank_score = round(min(1.0, blended), 4)

        ranked = sorted(chunks, key=lambda c: -(c.rerank_score or 0.0))[:keep]

        span.set_output(
            {
                "kept": len(ranked),
                "keywords": sorted(keywords)[:12],
                "numeric_question": numeric_question,
                "ranking": [
                    {
                        "source": chunk.source,
                        "doc_type": chunk.doc_type,
                        "embedding_score": chunk.score,
                        "rerank_score": chunk.rerank_score,
                    }
                    for chunk in ranked
                ],
            }
        )
        return ranked


def build_context(chunks: list[RetrievedChunk], *, max_chars: int = 9000) -> str:
    """Assemble the final prompt context, labelled for citation."""
    blocks: list[str] = []
    used = 0

    for index, chunk in enumerate(chunks, start=1):
        header = (
            f"--- CONTEXT {index} "
            f"[source: {chunk.source}] "
            f"[patient: {chunk.patient_id or 'n/a'}] "
            f"[type: {chunk.doc_type}] ---"
        )
        block = f"{header}\n{chunk.text}"
        if used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining > 400:
                blocks.append(block[:remaining] + "\n…[truncated]")
            break
        blocks.append(block)
        used += len(block)

    return "\n\n".join(blocks)


def context_metadata(chunks: list[RetrievedChunk]) -> dict[str, Any]:
    return {
        "chunk_count": len(chunks),
        "patients": sorted({c.patient_id for c in chunks if c.patient_id}),
        "sources": [c.source for c in chunks],
        "doc_types": sorted({c.doc_type or "" for c in chunks}),
    }
