"""
rag/retrieval.py
================

**Retrieval Agent** (Agno) — converts the question to an embedding and retrieves
the top-k chunks from the FAISS store.

Two details that matter for clinical Q&A:

* **patient scoping** — a question that names a patient (or a UI filter that
  supplies one) restricts retrieval to that patient's chunks, so P1019's dose
  never answers a question about P1022;
* **query expansion** — clinical questions use everyday words ("what pills",
  "how much did it cost") while the records use clinical/billing vocabulary, so a
  small synonym map is appended to the embedding query.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ..common.schemas import RetrievedChunk
from ..observability import tracing
from ..settings import settings
from .indexing import get_store

log = logging.getLogger(__name__)

PATIENT_ID_RE = re.compile(r"(?<![A-Za-z0-9])(P\d{4})(?![0-9])", re.IGNORECASE)

#  Everyday phrasing → the vocabulary the records actually use.
QUERY_EXPANSIONS: dict[str, str] = {
    "pill": "medication tablet prescription",
    "pills": "medications tablets prescriptions",
    "medicine": "medication prescription drug",
    "medicines": "medications prescriptions drugs",
    "drug": "medication prescription",
    "dose": "strength dosage frequency",
    "dosage": "strength dose frequency",
    "cost": "bill total amount charges payment",
    "price": "bill total amount charges",
    "paid": "payment status bill settled",
    "unpaid": "payment status UNPAID outstanding bill",
    "bill": "hospital bill total amount payment status line items",
    "allergy": "allergies adverse drug reaction",
    "allergies": "allergy adverse drug reaction",
    "appointment": "follow-up appointment clinic visit",
    "followup": "follow-up appointment clinic",
    "follow-up": "follow-up appointment clinic",
    "test": "laboratory results lab test value reference range",
    "tests": "laboratory results lab tests values",
    "blood": "laboratory results haemoglobin glucose HbA1c",
    "sugar": "glucose HbA1c diabetes",
    "diagnosis": "discharge diagnosis ICD-10",
    "doctor": "attending physician consulting doctors",
    "risk": "risk level risk score recommendation",
    "blocked": "discharge blocked critical finding",
    "discharged": "discharge date discharge summary",
    "admitted": "admission date",
    "instructions": "discharge instructions",
}


def extract_patient_id(text: str) -> str | None:
    match = PATIENT_ID_RE.search(text or "")
    return match.group(1).upper() if match else None


def expand_query(question: str) -> str:
    """Append record-vocabulary synonyms for the everyday words in a question."""
    lowered = (question or "").lower()
    additions = [
        expansion for term, expansion in QUERY_EXPANSIONS.items()
        if re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", lowered)
    ]
    if not additions:
        return question
    return f"{question}\n[related terms: {' '.join(dict.fromkeys(additions))}]"


def retrieve(
    question: str,
    *,
    patient_id: str | None = None,
    top_k: int | None = None,
    trace_id: str | None = None,
) -> list[RetrievedChunk]:
    """Retrieval Agent entry point."""
    k = int(top_k or settings.rag.get("top_k", 5))
    #  An explicit filter wins; otherwise honour a patient id inside the question.
    scope = (patient_id or extract_patient_id(question) or "") or None

    with tracing.retriever_span(
        "rag.retrieval_agent",
        trace_id=trace_id,
        query=question,
        top_k=k,
        patient_scope=scope or "all patients",
    ) as span:
        store = get_store()
        expanded = expand_query(question)
        hits = store.search(expanded, top_k=k, patient_id=scope)

        #  A patient filter that returns nothing usually means the patient is not
        #  indexed; retry unscoped so the Generation Agent can say so honestly
        #  rather than answering from an empty context.
        if scope and not hits:
            log.info("No chunks for patient %s — retrying without the filter.", scope)
            hits = store.search(expanded, top_k=k, patient_id=None)

        chunks = [
            RetrievedChunk(
                text=chunk.text,
                source=chunk.source,
                patient_id=chunk.patient_id,
                doc_type=chunk.doc_type,
                score=round(score, 4),
                chunk_index=chunk.chunk_index,
            )
            for chunk, score in hits
        ]

        span.set_output(
            {
                "chunks": len(chunks),
                "patient_scope": scope,
                "index_size": store.size,
                "top_scores": [chunk.score for chunk in chunks[:5]],
                "sources": [chunk.source for chunk in chunks],
            }
        )
        return chunks


def store_stats() -> dict[str, Any]:
    return get_store().stats()
