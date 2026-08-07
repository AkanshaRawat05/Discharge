"""
View 4 — RAG Q&A.

Never gated by the pipeline: this view queries the FAISS index, so it only needs
documents to be indexed, not a particular patient to have reached a stage.

Clickable suggested questions · question input with a patient filter ·
prompt-injection indicator · **token-by-token streaming answer** · source
documents · RAG Triad metrics tucked into a small expander · guardrail
activity · session history.

Index mechanics — chunk counts, embedding provider, vector dimension, rebuild —
are deliberately not shown: they are operator concerns. The index builds itself
on first use and only a genuine failure is surfaced. Retrieval depth comes from
`configs/agent_config.yaml` rather than a control on the page.

This is the second real-time touchpoint (matching the streaming RAG agent on
:8105): tokens are painted into a placeholder as they arrive off the wire
rather than being buffered until the run finishes.
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
    settings,
    stream_async,
    trace_link,
)
from discharge_ai.agents import rag_agent  # noqa: E402
from discharge_ai.rag import indexing, retrieval  # noqa: E402

log = logging.getLogger(__name__)

page_setup(
    "4 · RAG Q&A",
    "Ask the indexed patient records. Answers are grounded, cited and scored.",
)

#  Used only to make the suggested questions concrete ("What medications was
#  P1019 discharged on?"). Retrieval itself is never scoped to this patient.
patient_id = st.session_state.get("patient_id")

st.caption(
    "🔓 Available at any point in the review — it reads the vector index, not "
    "the pipeline state."
)

EXAMPLE_QUERIES = [
    "What medications was {pid} discharged on and at what doses?",
    "Which patients have an unpaid hospital bill?",
    "Does {pid} have any drug allergy conflicts?",
    "What follow-up appointment is documented for {pid}?",
    "Which discharges are currently blocked and why?",
    "What were {pid}'s abnormal lab results and what action was documented?",
]

# --------------------------------------------------------------------------- #
#  Index — built silently on first use
# --------------------------------------------------------------------------- #
#  Chunk counts, embedding provider, vector dimension and a rebuild button are
#  operator concerns, not clinical ones, so none of them are surfaced. The
#  index still builds itself the first time this page is opened; only a genuine
#  failure is reported, because that is the one case a reviewer must act on.
try:
    stats = retrieval.store_stats()
except Exception as exc:  # noqa: BLE001
    stats = {"error": str(exc), "chunks": 0, "patient_count": 0}

if stats.get("chunks", 0) == 0 and not stats.get("error"):
    with st.spinner("Preparing the patient record index…"):
        try:
            indexing.reset_store()
            stats = indexing.build_index()
        except Exception as exc:  # noqa: BLE001
            stats = {"error": str(exc), "chunks": 0, "patient_count": 0}

if stats.get("error"):
    st.error(
        f"The patient record index is unavailable: {stats['error']}"
    )

# --------------------------------------------------------------------------- #
#  Question input
# --------------------------------------------------------------------------- #
st.subheader("Ask a question")

pid = patient_id or "P1019"
example_queries = [q.format(pid=pid) for q in EXAMPLE_QUERIES]

#  Suggested questions come first: clicking one writes it straight into the
#  question box below.
#
#  The click MUST assign to `rag_question_input` — the text area's own widget
#  key. Streamlit reads a keyed widget's value from session_state and ignores
#  the `value=` argument on rerun, so the previous approach (staging the text
#  under a separate `rag_pending_question` key and re-passing it as `value=`)
#  could never update the box.
st.caption("Suggested questions — click one to use it")
example_columns = st.columns(3)
for idx, example in enumerate(example_queries):
    if example_columns[idx % 3].button(
        example, key=f"example-{idx}", width="stretch"
    ):
        st.session_state["rag_question_input"] = example
        st.rerun()

controls = st.columns([3, 1])
question = controls[0].text_area(
    "Question",
    height=90,
    placeholder=f"What medications was {pid} discharged on?",
    key="rag_question_input",
)

filter_options = ["All patients"] + available_patients()
patient_filter = controls[1].selectbox(
    "Restrict to patient",
    filter_options,
    index=(
        filter_options.index(patient_id) if patient_id in filter_options else 0
    ),
    key="rag_patient_filter",
    help="Narrow retrieval to one patient's records, or search every record.",
)

#  Retrieval depth is a tuning parameter, not a clinical decision — it comes
#  from `configs/agent_config.yaml` rather than a control on the page.
top_k = int(settings.rag.get("top_k", 5))

ask = st.button("🔎 Ask", type="primary")

# --------------------------------------------------------------------------- #
#  Answer — streamed token by token
# --------------------------------------------------------------------------- #
if ask and question.strip():
    scope = None if patient_filter == "All patients" else patient_filter

    def _events():
        from types import SimpleNamespace

        ctx = SimpleNamespace(trace_id=None)
        payload = {"question": question, "top_k": top_k}
        if scope:
            payload["patient_id"] = scope
        return rag_agent.handle(payload, ctx)

    st.subheader("Answer")
    answer_box = st.empty()
    answer_box.caption("Retrieving…")

    collected: list[str] = []
    payload: dict = {}
    error: Exception | None = None

    try:
        for event in stream_async(_events):
            if getattr(event, "final", False):
                payload = event.data or {}
            elif getattr(event, "text", ""):
                collected.append(event.text)
                #  Repaint on every chunk — this is the token-by-token display.
                answer_box.markdown("".join(collected) + " ▌")
    except Exception as exc:  # noqa: BLE001
        error = exc
        log.warning("RAG Q&A pipeline error: %s", exc, exc_info=True)

    if error is not None:
        answer_box.empty()
        st.error(
            f"The question could not be answered: **{error}**\n\n"
            "This usually means the language model is temporarily unavailable "
            "or its quota is exhausted. Try again shortly."
        )
    else:
        streamed = "".join(collected).strip()
        answer = payload.get("answer") or streamed or "(no answer produced)"

        # ---- prompt-injection indicator --------------------------------
        if payload.get("blocked_by_guardrail"):
            answer_box.empty()
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

        # ---- settle the streamed text on the final, post-guardrail answer
        if not payload.get("blocked_by_guardrail"):
            if payload.get("out_of_context"):
                answer_box.info(answer)
            else:
                answer_box.markdown(answer)

            if streamed and streamed != answer:
                with st.expander("Streamed tokens before guardrail post-processing"):
                    st.text(streamed)

        # ---- quality, kept deliberately quiet --------------------------
        triad = payload.get("triad") or {}
        thresholds = settings.rag.get("triad_thresholds", {}) or {}
        passed = bool(triad.get("passed"))
        faith = float(triad.get("faithfulness", 0.0))

        st.caption(
            f"{'🟢' if passed else '🟠'} grounding {faith:.2f} · "
            f"{len(payload.get('chunks', []))} source(s) · "
            f"scope {payload.get('patient_scope') or 'all'} · "
            f"prompt `{payload.get('prompt_source', 'n/a')}` (MCP Prompts)"
        )

        with st.expander("Answer quality — RAG Triad", expanded=False):
            triad_columns = st.columns(3)
            for column, (label, tkey) in zip(
                triad_columns,
                [
                    ("Faithfulness", "faithfulness"),
                    ("Answer relevance", "answer_relevance"),
                    ("Context relevance", "context_relevance"),
                ],
            ):
                value = float(triad.get(tkey, 0.0))
                minimum = float(thresholds.get(tkey, 0.7))
                with column:
                    st.caption(
                        f"**{label}** {value:.2f} "
                        f"({'≥' if value >= minimum else '<'} min {minimum:.2f})"
                    )
                    st.progress(min(1.0, value))

            if payload.get("triad_verdict"):
                st.caption(payload["triad_verdict"])
            if triad.get("reasoning"):
                st.caption(f"Judge reasoning: {triad['reasoning']}")
            st.caption(
                f"Context chars: {payload.get('context_chars', 0)} · "
                f"out of context: {'yes' if payload.get('out_of_context') else 'no'}"
            )
            guardrail_table(payload.get("guardrail_events", []))

        trace_link(payload.get("trace_id"))

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

        # ---- history ---------------------------------------------------
        st.session_state["rag_history"].append(
            {
                "question": question,
                "answer": answer[:400],
                "out_of_context": payload.get("out_of_context"),
                "faithfulness": triad.get("faithfulness"),
            }
        )

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
