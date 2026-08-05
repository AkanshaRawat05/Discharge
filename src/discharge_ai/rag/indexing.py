"""
rag/indexing.py
===============

**Indexing Agent** (Agno) — parses and indexes discharge documents into a FAISS
vector store.

What gets indexed, per patient:

* the raw discharge report, lab report and bill text (all languages);
* the English translation produced by the Normalizer, when present;
* a synthesised "structured facts" card (demographics, prescriptions, labs,
  bill) so a question like *"what dose of Metformin?"* retrieves a chunk that
  states the dose explicitly rather than a paragraph that merely mentions it;
* the validation report summary, so administrators can ask about risk levels,
  blocking findings and payment status.

Chunks carry `patient_id` / `doc_type` metadata, which the Retrieval Agent uses
to scope a question to one patient.

The index is persisted under `Data/vector_db/` (`.faiss` + `.json` sidecar) and
is rebuilt automatically when the embedding provider or dimension changes.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..common.doc_loader import load_patient_documents, scan_incoming
from ..common.parsing import build_extracted_case
from ..common.schemas import ExtractedCase
from ..llm import embeddings
from ..observability import tracing
from ..settings import settings

log = logging.getLogger(__name__)


@dataclass
class Chunk:
    text: str
    patient_id: str
    doc_type: str
    source: str
    chunk_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "patient_id": self.patient_id,
            "doc_type": self.doc_type,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Chunk":
        return cls(
            text=payload["text"],
            patient_id=payload.get("patient_id", ""),
            doc_type=payload.get("doc_type", ""),
            source=payload.get("source", ""),
            chunk_index=int(payload.get("chunk_index", 0)),
            metadata=payload.get("metadata", {}) or {},
        )


# --------------------------------------------------------------------------- #
#  Chunking
# --------------------------------------------------------------------------- #
def split_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    """Paragraph-aware splitter that keeps clinical sections intact."""
    if not text or not text.strip():
        return []

    paragraphs = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= chunk_size:
            current = f"{current}\n\n{paragraph}" if current else paragraph
            continue

        if current:
            chunks.append(current)

        if len(paragraph) <= chunk_size:
            #  Carry a little overlap so a fact split across a boundary is still
            #  retrievable from either side.
            tail = current[-overlap:] if current and overlap else ""
            current = f"{tail}\n\n{paragraph}".strip() if tail else paragraph
        else:
            #  A single oversized block (e.g. a long prescription table).
            for start in range(0, len(paragraph), max(1, chunk_size - overlap)):
                chunks.append(paragraph[start : start + chunk_size])
            current = ""

    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk.strip()]


def structured_facts_card(case: ExtractedCase) -> str:
    """A dense, explicit fact sheet — the highest-value chunk for retrieval."""
    lines = [
        f"STRUCTURED CLINICAL FACTS FOR PATIENT {case.patient_id}",
        f"Patient name: {case.patient_name or 'not recorded'}",
        f"Age: {case.age if case.age is not None else 'not recorded'}",
        f"Gender: {case.gender or 'not recorded'}",
        f"Address: {'recorded' if case.address else 'NOT RECORDED'}",
        f"Admission date: {case.admission_date or 'not recorded'}",
        f"Discharge date: {case.discharge_date or 'not recorded'}",
        f"Ward: {case.ward or 'not recorded'}; Bed: {case.bed_no or 'not recorded'}",
        f"Service line: {case.service_line or 'not recorded'}",
        f"Attending physician: {case.attending_physician or 'NOT RECORDED'}",
        f"Consulting doctors: {'; '.join(case.consulting_doctors) or 'not recorded'}",
        f"Record language: {case.detected_language}",
        f"Translation confidence: {case.translation_confidence:.2f}",
        f"Discharge approved: {case.discharge_approved}",
        f"Discharge approved by: {case.discharge_approved_by or 'NOT RECORDED'}",
        "Discharge diagnosis: " + ("; ".join(case.discharge_diagnosis) or "not recorded"),
        "Allergies: " + ("; ".join(case.allergies) or "none documented"),
        "",
        "DISCHARGE MEDICATIONS:",
    ]

    if case.medications:
        for medication in case.medications:
            lines.append(
                f"  {medication.sl_no}. {medication.medicine_name or '?'} — "
                f"strength {medication.strength or '?'}, dose {medication.dosage or '?'}, "
                f"frequency {medication.frequency or '?'}, route {medication.route or '?'}, "
                f"period {medication.period or '?'}, quantity {medication.total_quantity or '?'}"
                + (f", remarks: {medication.remarks}" if medication.remarks else "")
            )
    else:
        lines.append("  none recorded")

    lines += ["", "LABORATORY RESULTS:"]
    if case.lab_tests:
        for test in case.lab_tests:
            lines.append(
                f"  {test.test}: {test.value or '?'} {test.unit or ''} "
                f"(reference {test.reference_range or '?'}) flag={test.flag or 'NORMAL'}"
            )
    else:
        lines.append("  none recorded")

    if case.abnormal_labs:
        lines += ["", "ABNORMAL LAB ACTIONS:"]
        for abnormal in case.abnormal_labs:
            lines.append(
                f"  {abnormal.test}: {abnormal.value or ''} — "
                f"action: {abnormal.action or 'NO ACTION DOCUMENTED'}"
            )

    bill = case.bill
    lines += [
        "",
        "HOSPITAL BILL:",
        f"  Bill id: {bill.bill_id or 'not recorded'}",
        f"  Billed by: {bill.hospital_name or 'not recorded'}",
        f"  Bill date: {bill.billing_date or 'not recorded'}",
        f"  Total amount: {bill.total_amount if bill.total_amount is not None else '?'} "
        f"{bill.currency or ''}".rstrip(),
        f"  Payment status: {bill.payment_status or 'unknown'}",
        f"  Settled: {bill.is_settled}",
        "",
        "FOLLOW-UP APPOINTMENT: " + (case.follow_up_appointment or "NONE SCHEDULED"),
        "",
        "DISCHARGE INSTRUCTIONS: " + (case.discharge_instructions or "none recorded"),
    ]
    return "\n".join(lines)


def validation_facts_card(patient_id: str) -> str | None:
    """Index the audit report so admins can ask about risk and blocking issues."""
    path = settings.path("reports_dir") / f"{patient_id}_audit.json"
    if not path.exists():
        return None

    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.debug("Could not read %s: %s", path.name, exc)
        return None

    risk = report.get("risk", {}) or {}
    completeness = report.get("completeness", {}) or {}
    lines = [
        f"VALIDATION AND RISK REPORT FOR PATIENT {patient_id}",
        f"Risk score: {risk.get('score')}",
        f"Risk level: {risk.get('level')}",
        f"Recommendation: {risk.get('recommendation')} — {risk.get('recommendation_text')}",
        f"Discharge blocked: {risk.get('discharge_blocked')}",
        f"Human review (HITL) required: {risk.get('hitl_required')}",
        f"Hard guardrails triggered: {', '.join(risk.get('hard_guardrails_hit') or []) or 'none'}",
        f"Completeness score: {completeness.get('score')}%",
        f"Missing blocking fields: {', '.join(completeness.get('blocking_missing') or []) or 'none'}",
        f"Missing non-blocking fields: {', '.join(completeness.get('non_blocking_missing') or []) or 'none'}",
        f"Bill payment status: {report.get('bill_payment_status')}",
        f"Bill total: {report.get('bill_total_amount')} {report.get('bill_currency') or ''}",
        f"Translation confidence: {report.get('translation_confidence')}",
        f"Record language: {report.get('detected_language')}",
        "",
        "CROSS-VALIDATION FINDINGS:",
    ]
    findings = report.get("findings") or []
    if findings:
        for finding in findings:
            lines.append(
                f"  [{finding.get('severity')}] {finding.get('rule_id')}: "
                f"{finding.get('message')}"
                + (f" | Clinical impact: {finding['clinical_impact']}"
                   if finding.get("clinical_impact") else "")
            )
    else:
        lines.append("  none — no discrepancies found")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  FAISS store
# --------------------------------------------------------------------------- #
class ClinicalVectorStore:
    """FAISS inner-product index over L2-normalised embeddings (= cosine)."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.index: Any = None
        self.dimension: int = 0
        self.provider: str = ""
        self._lock = threading.Lock()

        directory = settings.path("vector_db_dir")
        name = settings.rag.get("faiss_index_name", "clinical_docs")
        self.index_path = directory / f"{name}.faiss"
        self.meta_path = directory / f"{name}.json"

    # ------------------------------------------------------------------ #
    @property
    def size(self) -> int:
        return len(self.chunks)

    @property
    def patient_ids(self) -> list[str]:
        return sorted({chunk.patient_id for chunk in self.chunks if chunk.patient_id})

    def stats(self) -> dict[str, Any]:
        by_type: dict[str, int] = {}
        for chunk in self.chunks:
            by_type[chunk.doc_type] = by_type.get(chunk.doc_type, 0) + 1
        return {
            "chunks": self.size,
            "patients": self.patient_ids,
            "patient_count": len(self.patient_ids),
            "chunks_by_doc_type": by_type,
            "embedding_provider": self.provider or embeddings.active_provider(),
            "dimension": self.dimension,
            "index_file": str(self.index_path),
            "persisted": self.index_path.exists(),
        }

    # ------------------------------------------------------------------ #
    def build(self, chunks: list[Chunk]) -> None:
        """Embed and index a fresh set of chunks."""
        import faiss
        import numpy as np

        if not chunks:
            log.warning("Nothing to index — no chunks were produced.")
            return

        with self._lock:
            vectors = embeddings.embed_texts([chunk.text for chunk in chunks])
            matrix = np.asarray(vectors, dtype="float32")
            self.dimension = matrix.shape[1]
            self.provider = embeddings.active_provider()

            index = faiss.IndexFlatIP(self.dimension)
            index.add(matrix)

            self.index = index
            self.chunks = list(chunks)
            self._persist(matrix)

        log.info(
            "FAISS index built: %d chunks, %d patients, dim=%d, provider=%s",
            self.size, len(self.patient_ids), self.dimension, self.provider,
        )

    def upsert_patient(self, patient_id: str, chunks: list[Chunk]) -> None:
        """Replace one patient's chunks, keeping every other patient indexed.

        The pipeline re-indexes a single case after each run. A plain `build()`
        there would wipe the store down to that one patient, so incremental
        updates must merge instead of replace.
        """
        patient_id = patient_id.upper()
        with self._lock:
            kept = [
                chunk for chunk in self.chunks
                if chunk.patient_id.upper() != patient_id
            ]
        merged = kept + list(chunks)

        if not merged:
            log.warning("Upsert for %s produced no chunks — index left unchanged.",
                        patient_id)
            return

        #  Re-embedding the whole store keeps every vector in one provider's
        #  space; the corpus here is small enough that this is cheap and safe.
        self.build(merged)
        log.info(
            "Upserted %d chunk(s) for %s (index now %d chunks, %d patients)",
            len(chunks), patient_id, self.size, len(self.patient_ids),
        )

    def _persist(self, matrix: Any) -> None:
        try:
            import faiss

            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(self.index_path))
            self.meta_path.write_text(
                json.dumps(
                    {
                        "embedding_provider": self.provider,
                        "dimension": self.dimension,
                        "chunk_count": len(self.chunks),
                        "chunks": [chunk.to_payload() for chunk in self.chunks],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not persist the FAISS index: %s", exc)

    def load(self) -> bool:
        """Load a persisted index, rejecting it if the embedder changed."""
        if not (self.index_path.exists() and self.meta_path.exists()):
            return False
        try:
            import faiss

            metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))
            stored_provider = metadata.get("embedding_provider")
            current_provider = settings.embedding_provider

            if stored_provider and stored_provider != current_provider:
                log.info(
                    "Persisted index was built with %r but EMBEDDING_PROVIDER is "
                    "%r — rebuilding.", stored_provider, current_provider,
                )
                return False

            self.index = faiss.read_index(str(self.index_path))
            self.chunks = [Chunk.from_payload(c) for c in metadata.get("chunks", [])]
            self.dimension = int(metadata.get("dimension", 0))
            self.provider = stored_provider or current_provider

            if self.index.ntotal != len(self.chunks):
                log.warning("Index/metadata mismatch — rebuilding.")
                return False

            log.info("Loaded FAISS index: %d chunks from %s",
                     self.size, self.index_path.name)
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not load the persisted FAISS index: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    def search(
        self, query: str, *, top_k: int = 5, patient_id: str | None = None
    ) -> list[tuple[Chunk, float]]:
        """Top-k chunks for a query, optionally scoped to one patient."""
        if self.index is None or not self.chunks:
            return []

        import numpy as np

        vector = np.asarray([embeddings.embed_query(query)], dtype="float32")

        if vector.shape[1] != self.dimension:
            #  The embedder changed under us — usually the configured provider
            #  became unavailable mid-session (rate limit) and fell back to the
            #  offline hashing embedder. Re-embed the stored chunks in the new
            #  space rather than returning nothing.
            log.warning(
                "Query dimension %d != index dimension %d (embedding provider "
                "changed to %r) — re-embedding %d stored chunk(s).",
                vector.shape[1], self.dimension,
                embeddings.active_provider(), len(self.chunks),
            )
            stored = list(self.chunks)
            self.build(stored)
            vector = np.asarray([embeddings.embed_query(query)], dtype="float32")
            if vector.shape[1] != self.dimension:
                log.error("Index rebuild did not resolve the dimension mismatch.")
                return []

        #  Over-fetch when filtering so a patient filter cannot starve the result.
        fetch = min(len(self.chunks), top_k * 8 if patient_id else top_k * 2)
        scores, indices = self.index.search(vector, fetch)

        results: list[tuple[Chunk, float]] = []
        for position, score in zip(indices[0], scores[0]):
            if position < 0 or position >= len(self.chunks):
                continue
            chunk = self.chunks[position]
            if patient_id and chunk.patient_id.upper() != patient_id.upper():
                continue
            results.append((chunk, float(score)))
            if len(results) >= top_k:
                break
        return results


# --------------------------------------------------------------------------- #
#  Chunk production
# --------------------------------------------------------------------------- #
def chunks_for_patient(patient_id: str, case: ExtractedCase | None = None) -> list[Chunk]:
    """Every chunk that represents one patient."""
    chunk_size = int(settings.rag.get("chunk_size", 900))
    overlap = int(settings.rag.get("chunk_overlap", 150))

    if case is None:
        documents = load_patient_documents(patient_id)
        if not documents:
            return []
        case = build_extracted_case(patient_id, documents)

    chunks: list[Chunk] = []

    #  1. the structured fact card (single chunk — never split)
    chunks.append(
        Chunk(
            text=structured_facts_card(case),
            patient_id=patient_id,
            doc_type="structured_facts",
            source=f"{patient_id} structured clinical facts",
            metadata={"patient_name": case.patient_name, "priority": "high"},
        )
    )

    #  2. raw source documents
    for doc_type, text in case.raw_text.items():
        if not text or not text.strip():
            continue
        source = case.source_files.get(doc_type, f"{patient_id} {doc_type}")
        for index, piece in enumerate(
            split_text(text, chunk_size=chunk_size, overlap=overlap)
        ):
            chunks.append(
                Chunk(
                    text=f"[{doc_type} — {source}]\n{piece}",
                    patient_id=patient_id,
                    doc_type=doc_type,
                    source=source,
                    chunk_index=index,
                    metadata={"language": case.detected_language},
                )
            )

    #  3. English translations produced by the Normalizer
    for key, text in (case.translated_text or {}).items():
        if not text or not text.strip():
            continue
        for index, piece in enumerate(
            split_text(text, chunk_size=chunk_size, overlap=overlap)
        ):
            chunks.append(
                Chunk(
                    text=f"[english translation — {key}]\n{piece}",
                    patient_id=patient_id,
                    doc_type="translation",
                    source=f"{patient_id} {key} (translated to English)",
                    chunk_index=index,
                    metadata={"source_language": case.detected_language},
                )
            )

    #  4. the validation / risk report, when one has been generated
    validation = validation_facts_card(patient_id)
    if validation:
        chunks.append(
            Chunk(
                text=validation,
                patient_id=patient_id,
                doc_type="validation_report",
                source=f"{patient_id}_audit.json",
                metadata={"priority": "high"},
            )
        )

    return chunks


def build_index(
    patient_ids: list[str] | None = None, *, trace_id: str | None = None
) -> dict[str, Any]:
    """Indexing Agent entry point.

    * `patient_ids=None` → full rebuild across every patient with documents.
    * `patient_ids=[…]`  → **upsert** just those patients, leaving the rest of
      the index intact (this is what the pipeline calls after each run).
    """
    full_rebuild = patient_ids is None
    targets = list(patient_ids) if patient_ids else list(scan_incoming().keys())

    with tracing.retriever_span(
        "rag.indexing_agent",
        trace_id=trace_id,
        query=f"{'rebuild' if full_rebuild else 'upsert'} {len(targets)} patient(s)",
    ) as span:
        store = get_store(auto_build=full_rebuild)
        indexed: list[str] = []

        if full_rebuild:
            chunks: list[Chunk] = []
            for patient_id in targets:
                patient_chunks = chunks_for_patient(patient_id)
                if patient_chunks:
                    chunks += patient_chunks
                    indexed.append(patient_id)
            store.build(chunks)
        else:
            for patient_id in targets:
                patient_chunks = chunks_for_patient(patient_id)
                if patient_chunks:
                    store.upsert_patient(patient_id, patient_chunks)
                    indexed.append(patient_id)

        stats = store.stats()
        stats["indexed_patients"] = indexed
        stats["mode"] = "rebuild" if full_rebuild else "upsert"
        span.set_output(stats)
        return stats


# --------------------------------------------------------------------------- #
#  Module-level store
# --------------------------------------------------------------------------- #
_store: ClinicalVectorStore | None = None
_store_lock = threading.Lock()


def get_store(*, auto_build: bool = True) -> ClinicalVectorStore:
    """The shared vector store, loading or building it on first use."""
    global _store

    with _store_lock:
        if _store is None:
            _store = ClinicalVectorStore()
            if not _store.load() and auto_build:
                log.info("No usable FAISS index found — building one now…")
                targets = list(scan_incoming().keys())
                chunks: list[Chunk] = []
                for patient_id in targets:
                    chunks += chunks_for_patient(patient_id)
                _store.build(chunks)
        return _store


def reset_store() -> None:
    """Drop the in-memory store so the next call rebuilds (used after re-runs)."""
    global _store
    with _store_lock:
        _store = None
