"""
Page 4 — RAG Q&A.

Global patient selector · question input with patient filter · example queries ·
prompt-injection indicator · streaming response · source documents · RAG Triad
quality metrics · guardrail activity · session history.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (  # noqa: E402
    available_patients,
    guardrail_table,
    page_setup,
    patient_selector,
    run_async,
    settings,
    sidebar_status,
    trace_link,
)
from discharge_ai.agents import rag_agent  # noqa: E402
from discharge_ai.rag import indexing, reflection, retrieval  # noqa: E402

log = logging.getLogger(__name__)

page_setup(
    "4 · RAG Q&A",
    "Ask the indexed patient records. Answers are grounded, cited and scored.",
)
sidebar_status()

patient_id = patient_selector()
patients = available_patients()

EXAMPLE_QUERIES = [
    "What medications was {pid} discharged on and at what doses?",
    "Which patients have an unpaid hospital bill?",
    "Does {pid} have any drug allergy conflicts?",
    "What follow-up appointment is documented for {pid}?",
    "Which discharges are currently blocked and why?",
    "What were {pid}'s abnormal lab results and what action was documented?",
]

# --------------------------------------------------------------------------- #
#  Index status — auto-build if empty
# --------------------------------------------------------------------------- #
try:
    stats = retrieval.store_stats()
except Exception as exc:  # noqa: BLE001
    stats = {"error": str(exc), "chunks": 0, "patient_count": 0}

if stats.get("chunks", 0) == 0 and not stats.get("error"):
    with st.spinner("Building the FAISS vector index for the first time…"):
        try:
            indexing.reset_store()
            stats = indexing.build_index()
            st.success(
                f"Indexed {stats['chunks']} chunks across "
                f"{stats['patient_count']} patient(s)."
            )
        except Exception as exc:  # noqa: BLE001
            stats = {"error": str(exc), "chunks": 0, "patient_count": 0}
            st.error(f"Index build failed: {exc}")

status = st.columns([1, 1, 1, 1, 2])
status[0].metric("Indexed chunks", stats.get("chunks", 0))
status[1].metric("Patients indexed", stats.get("patient_count", 0))
status[2].metric("Embeddings", str(stats.get("embedding_provider", "—")))
status[3].metric("Dimension", stats.get("dimension", 0))
with status[4]:
    if st.button("🔄 Rebuild the FAISS index", width="stretch"):
        with st.spinner("Re-indexing every patient…"):
            indexing.reset_store()
            fresh = indexing.build_index()
        st.success(
            f"Indexed {fresh['chunks']} chunks across {fresh['patient_count']} "
            f"patient(s) using {fresh['embedding_provider']} embeddings."
        )
        st.rerun()

if stats.get("error"):
    st.error(f"Vector store problem: {stats['error']}")

st.divider()

# --------------------------------------------------------------------------- #
#  Question input
# --------------------------------------------------------------------------- #
st.subheader("Ask a question")

pid = patient_id or "P1019"
example_queries = [q.format(pid=pid) for q in EXAMPLE_QUERIES]

controls = st.columns([3, 1, 1])
question = controls[0].text_area(
    "Question",
    value=st.session_state.get("rag_pending_question", ""),
    height=90,
    placeholder=f"What medications was {pid} discharged on?",
    key="rag_question_input",
)

filter_options = ["(all patients)"] + patients
default_filter_index = (
    filter_options.index(patient_id) if patient_id in filter_options else 0
)
patient_filter = controls[1].selectbox(
    "Restrict to patient",
    filter_options,
    index=default_filter_index,
    key="rag_patient_filter",
)
top_k = controls[2].slider(
    "Chunks to retrieve", min_value=2, max_value=10,
    value=int(settings.rag.get("top_k", 5)), key="rag_top_k",
)

st.caption("Example questions")
example_columns = st.columns(3)
for idx, example in enumerate(example_queries):
    if example_columns[idx % 3].button(example, key=f"example-{idx}", width="stretch"):
        st.session_state["rag_pending_question"] = example
        st.rerun()

ask = st.button("🔎 Ask", type="primary")

# --------------------------------------------------------------------------- #
#  Answer
# --------------------------------------------------------------------------- #
if ask and question.strip():
    scope = None if patient_filter == "(all patients)" else patient_filter

    collected: list[str] = []
    result_box: dict[str, object] = {"payload": None, "error": None}

    async def _stream() -> None:
        from types import SimpleNamespace

        ctx = SimpleNamespace(trace_id=None)
        payload = {"question": question, "top_k": top_k}
        if scope:
            payload["patient_id"] = scope

        async for event in rag_agent.handle(payload, ctx):
            if event.final:
                result_box["payload"] = event.data
            elif event.text:
                collected.append(event.text)

    with st.spinner("Retrieving, generating and scoring…"):
        try:
            run_async(_stream())
        except Exception as exc:  # noqa: BLE001
            result_box["error"] = exc
            log.warning("RAG Q&A pipeline error: %s", exc, exc_info=True)

    if result_box["error"]:
        st.error(
            f"The RAG pipeline encountered an error: **{result_box['error']}**\n\n"
            "This usually happens when the vector index hasn't been built yet or "
            "the LLM API quota is exhausted. Try pressing **Rebuild the FAISS index** "
            "above, or wait for the API quota to reset."
        )
    else:
        streamed = "".join(collected).strip()
        payload = result_box["payload"] or {}
        answer = payload.get("answer") or streamed or "(no answer produced)"

        # ---- prompt-injection indicator --------------------------------
        if payload.get("blocked_by_guardrail"):
            st.error(
                "🛡️ **Prompt-injection guard: REJECTED.** Patterns matched: "
                + ", ".join(payload.get("patterns_matched", []))
            )
        elif payload.get("sanitised_question"):
            st.warning(
                "🛡️ **Prompt-injection guard: SANITISED.** The question was "
                "neutralised before it reached the retrieval pipeline.\n\n"
                f"Sanitised form: `{payload['sanitised_question']}`"
            )
        else:
            st.caption("🛡️ Prompt-injection guard: no injection pattern matched.")

        # ---- answer ----------------------------------------------------
        st.subheader("Answer")
        if payload.get("out_of_context"):
            st.info(answer)
        else:
            st.markdown(answer)

        if streamed and streamed != answer and not payload.get("blocked_by_guardrail"):
            with st.expander("Streamed tokens before guardrail post-processing"):
                st.text(streamed)

        meta = st.columns(4)
        meta[0].metric("Chunks used", len(payload.get("chunks", [])))
        meta[1].metric("Context chars", payload.get("context_chars", 0))
        meta[2].metric("Scope", payload.get("patient_scope") or "all")
        meta[3].metric(
            "Out of context", "yes" if payload.get("out_of_context") else "no"
        )
        st.caption(
            f"Prompt source: `{payload.get('prompt_source', 'n/a')}` (MCP Prompts)"
        )
        trace_link(payload.get("trace_id"))

        # ---- RAG Triad -------------------------------------------------
        st.subheader("RAG Triad quality metrics")
        triad = payload.get("triad") or {}
        thresholds = settings.rag.get("triad_thresholds", {}) or {}

        triad_columns = st.columns(3)
        for column, (label, tkey, threshold_key) in zip(
            triad_columns,
            [
                ("Faithfulness", "faithfulness", "faithfulness"),
                ("Answer relevance", "answer_relevance", "answer_relevance"),
                ("Context relevance", "context_relevance", "context_relevance"),
            ],
        ):
            value = float(triad.get(tkey, 0.0))
            minimum = float(thresholds.get(threshold_key, 0.7))
            with column:
                st.metric(
                    label, f"{value:.2f}",
                    delta=f"{value - minimum:+.2f} vs min {minimum:.2f}",
                    delta_color="normal" if value >= minimum else "inverse",
                )
                st.progress(min(1.0, value))

        if payload.get("triad_verdict"):
            (st.success if triad.get("passed") else st.warning)(
                payload["triad_verdict"]
            )
        if triad.get("reasoning"):
            st.caption(f"Judge reasoning: {triad['reasoning']}")

        # ---- sources ---------------------------------------------------
        st.subheader("Source documents")
        chunks = payload.get("chunks") or []
        if not chunks:
            st.caption("No chunks were retrieved.")
        else:
            for cidx, chunk in enumerate(chunks, start=1):
                label = (
                    f"{cidx}. {chunk.get('source')} "
                    f"[{chunk.get('doc_type')}] · "
                    f"patient {chunk.get('patient_id') or 'n/a'} · "
                    f"similarity {chunk.get('score', 0):.3f}"
                    + (
                        f" · rerank {chunk['rerank_score']:.3f}"
                        if chunk.get("rerank_score") is not None else ""
                    )
                )
                with st.expander(label):
                    st.text(chunk.get("text", "")[:4000])

        # ---- guardrails ------------------------------------------------
        st.subheader("Guardrail activity")
        guardrail_table(payload.get("guardrail_events", []))

        # ---- history ---------------------------------------------------
        st.session_state["rag_history"].append(
            {
                "question": question,
                "answer": answer[:400],
                "out_of_context": payload.get("out_of_context"),
                "faithfulness": triad.get("faithfulness"),
            }
        )
        st.session_state["rag_pending_question"] = ""

elif ask:
    st.warning("Type a question first.")

# --------------------------------------------------------------------------- #
#  Session history
# --------------------------------------------------------------------------- #
if st.session_state["rag_history"]:
    st.divider()
    st.subheader("This session's questions")
    for entry in reversed(st.session_state["rag_history"][-8:]):
        marker = "🟡" if entry.get("out_of_context") else "🟢"
        faith = entry.get("faithfulness")
        st.markdown(
            f"{marker} **{entry['question']}**  \n"
            f"{entry['answer']}"
            + (f"  \n*faithfulness {faith:.2f}*" if isinstance(faith, (int, float)) else "")
        )

st.divider()
st.caption(
    "Out-of-context questions must return exactly: "
    f"*\"{settings.rag.get('out_of_context_answer')}\"* — the Generation Agent's "
    "prompt is fetched from the MCP server, and the Hallucination guardrail "
    "blocks any answer whose faithfulness falls below "
    f"{settings.guardrail_cfg.get('hallucination_faithfulness_min', 0.7)}."
)
