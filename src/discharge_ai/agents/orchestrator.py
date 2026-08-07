"""
agents/orchestrator.py
======================

**Host Orchestrator** — Google ADK — Gradio UI on :8083 — A2A **client**

Coordinates all six agents over the A2A Protocol (streaming-capable) and exposes
a Gradio console with four tabs:

    Process a discharge   run the full pipeline for one patient, live progress
    Ask the records       streaming RAG Q&A against the indexed documents
    Agent network         AgentCard discovery for every peer (/.well-known/agent.json)
    System                the active LLM provider, MCP endpoints, LangFuse status

The orchestrator itself holds no clinical logic: it is a pure A2A client, exactly
as specified in Table 6.  A Google ADK `LlmAgent` provides the natural-language
front door — it routes a typed instruction ("process P1019", "who has an unpaid
bill?") to the right A2A skill through two FunctionTools.

Run:
    python -m discharge_ai.agents.orchestrator
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

from ..a2a_layer import A2AAgentClient
from ..llm.provider import provider_banner
from ..observability import tracing
from ..pipeline import default_mode, run_pipeline
from ..settings import configure_logging, settings

log = logging.getLogger(__name__)

AGENT_KEY = "orchestrator"
AGENT_NAME = settings.agent(AGENT_KEY)["name"]
FRAMEWORK = "google-adk"

ORCHESTRATOR_INSTRUCTION = """
You are the Host Orchestrator for St. Marian Regional Medical Center's agentic
discharge system. You do not perform clinical work yourself — you route requests
to specialist agents over the A2A protocol using your two tools:

* `process_discharge_case(patient_id)` — run the full discharge pipeline
  (extract → normalise → validate → summarise) for one patient.
* `ask_patient_records(question, patient_id)` — put a question to the Clinical
  RAG Q&A agent.

Rules:
- Pick exactly one tool per request; never guess a result without calling a tool.
- Patient ids look like P1019. If the user does not give one for a pipeline
  request, ask for it.
- Report what the tool returned — risk level, recommendation, whether discharge
  is blocked — and never soften or override a blocking finding.
- You give no clinical advice of your own.
""".strip()


# --------------------------------------------------------------------------- #
#  A2A orchestration
# --------------------------------------------------------------------------- #
async def process_case(
    patient_id: str, *, mode: str | None = None, generate_summary: bool = False,
    force_summary: bool = False,
) -> dict[str, Any]:
    """Run extract → normalise → validate for one patient over A2A (or locally).

    Per PDF §2.5, the discharge summary is a separate, explicitly-requested step
    — the caller must pass `generate_summary=True` to trigger it.
    """
    resolved_mode = mode or default_mode()
    trace_id = tracing.start_case_trace(patient_id)

    log.info("Orchestrating %s in %s mode (trace %s)", patient_id, resolved_mode, trace_id)
    result = await run_pipeline(
        patient_id,
        mode=resolved_mode,
        generate_summary=generate_summary,
        force_summary=force_summary,
        trace_id=trace_id,
    )
    return result.as_dict()


async def ask_records(
    question: str, patient_id: str | None = None
) -> AsyncIterator[tuple[str, dict[str, Any] | None]]:
    """Stream an answer from the RAG agent (A2A streaming, local fallback)."""
    payload: dict[str, Any] = {"question": question}
    if patient_id:
        payload["patient_id"] = patient_id

    client = A2AAgentClient(trace_id=tracing.start_case_trace(patient_id or "rag"))
    card = await client.discover("rag")

    if card is not None:
        async for chunk in client.stream("rag", payload):
            yield chunk.text, chunk.data
        return

    #  The RAG agent is not listening — answer in-process so the console still works.
    log.info("RAG agent :8105 is offline — answering in-process.")
    from . import rag_agent

    answer = await rag_agent.ask(question, patient_id=patient_id)
    yield answer.answer, {
        "answer": answer.answer,
        "triad": answer.triad.model_dump(mode="json"),
        "chunks": [chunk.model_dump(mode="json") for chunk in answer.chunks],
        "out_of_context": answer.out_of_context,
        "guardrail_events": [e.model_dump(mode="json") for e in answer.guardrail_events],
        "offline_fallback": True,
    }


async def discover_network() -> dict[str, dict[str, Any]]:
    return await A2AAgentClient().discover_all()


# --------------------------------------------------------------------------- #
#  ADK natural-language front door
# --------------------------------------------------------------------------- #
async def route_instruction(instruction: str) -> str:
    """Let an ADK LlmAgent choose which A2A skill an instruction needs."""
    try:
        from google.adk.agents import LlmAgent
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.adk.tools import FunctionTool
        from google.genai import types

        from ..llm.provider import get_adk_model
    except ImportError as exc:  # pragma: no cover
        return f"Google ADK is unavailable ({exc}); use the tabs above instead."

    def process_discharge_case(patient_id: str) -> dict:
        """Run the full discharge pipeline for one patient over A2A.

        Args:
            patient_id: hospital patient id, e.g. "P1019".
        """
        return asyncio.run(process_case(patient_id))

    def ask_patient_records(question: str, patient_id: str = "") -> dict:
        """Ask the Clinical RAG Q&A agent a question about patient records.

        Args:
            question:   the administrator's question.
            patient_id: optional patient id to scope the search.
        """

        async def _collect() -> dict:
            final: dict[str, Any] | None = None
            parts: list[str] = []
            async for text, data in ask_records(question, patient_id or None):
                if text:
                    parts.append(text)
                if data:
                    final = data
            return final or {"answer": "".join(parts)}

        return asyncio.run(_collect())

    agent = LlmAgent(
        name="host_orchestrator",
        model=get_adk_model("fast"),
        description="Routes discharge requests to specialist A2A agents.",
        instruction=ORCHESTRATOR_INSTRUCTION,
        tools=[
            FunctionTool(func=process_discharge_case),
            FunctionTool(func=ask_patient_records),
        ],
    )

    session_service = InMemorySessionService()
    runner = Runner(
        app_name="host-orchestrator", agent=agent, session_service=session_service
    )
    try:
        await session_service.create_session(
            app_name="host-orchestrator", user_id="admin", session_id="console"
        )
    except Exception:  # noqa: BLE001 — already exists
        pass

    parts: list[str] = []
    try:
        async for event in runner.run_async(
            user_id="admin",
            session_id="console",
            new_message=types.Content(role="user", parts=[types.Part(text=instruction)]),
        ):
            content = getattr(event, "content", None)
            if content and getattr(content, "parts", None):
                for part in content.parts:
                    if getattr(part, "text", None):
                        parts.append(part.text)
    except Exception as exc:  # noqa: BLE001
        log.warning("ADK routing failed: %s", exc)
        return f"Routing failed: {type(exc).__name__}: {exc}"

    return "\n".join(part.strip() for part in parts if part.strip()) or "(no response)"


# --------------------------------------------------------------------------- #
#  Gradio UI
# --------------------------------------------------------------------------- #
def build_ui():  # noqa: ANN201
    import gradio as gr

    from ..common.doc_loader import scan_incoming

    patient_ids = sorted(scan_incoming().keys())

    def _run_pipeline(patient_id: str, mode: str):  # noqa: ANN202
        if not patient_id:
            return "Select a patient first.", "{}"
        lines: list[str] = []

        def progress(stage: str, message: str) -> None:
            lines.append(f"[{stage}] {message}")

        #  Per PDF §2.5, the summary is a separate reviewer-requested step —
        #  this button only runs extract → normalise → validate.
        result = asyncio.run(
            run_pipeline(
                patient_id,
                mode=mode,
                generate_summary=False,
                progress=progress,
            )
        )
        payload = result.as_dict()

        verdict = [
            f"### {patient_id} — {payload['risk_level']} risk "
            f"(score {payload['risk_score']})",
            f"**Recommendation:** {payload['recommendation']}",
            f"**Discharge blocked:** {payload['discharge_blocked']}",
            f"**Human review required:** {payload['hitl_required']}",
            f"**Mode:** {payload['mode']} | **Trace:** `{payload['trace_id']}`",
        ]
        if payload.get("trace_url"):
            verdict.append(f"[Open the LangFuse trace]({payload['trace_url']})")
        if payload["artefacts"]:
            verdict.append("**Artefacts:** " + ", ".join(sorted(payload["artefacts"])))
        if payload["errors"]:
            verdict.append("**Issues:** " + "; ".join(payload["errors"]))

        verdict.append("\n#### Progress\n```\n" + "\n".join(lines) + "\n```")
        return "\n\n".join(verdict), json.dumps(payload, indent=2, default=str)

    def _ask(question: str, patient_id: str):  # noqa: ANN202
        if not question.strip():
            return "Type a question first.", "{}"

        async def _collect() -> tuple[str, dict[str, Any]]:
            parts: list[str] = []
            final: dict[str, Any] = {}
            async for text, data in ask_records(question, patient_id or None):
                if text:
                    parts.append(text)
                if data:
                    final = data
            return "".join(parts), final

        answer_text, payload = asyncio.run(_collect())
        answer = payload.get("answer") or answer_text or "(no answer)"

        triad = payload.get("triad") or {}
        blocks = [f"### Answer\n{answer}"]
        if triad:
            blocks.append(
                "**RAG Triad** — faithfulness "
                f"{triad.get('faithfulness', 0):.2f} · answer relevance "
                f"{triad.get('answer_relevance', 0):.2f} · context relevance "
                f"{triad.get('context_relevance', 0):.2f}"
            )
        sources = [chunk.get("source") for chunk in payload.get("chunks", [])]
        if sources:
            blocks.append("**Sources:** " + ", ".join(dict.fromkeys(map(str, sources))))
        triggered = [
            event["guardrail"] for event in payload.get("guardrail_events", [])
            if event.get("triggered")
        ]
        if triggered:
            blocks.append("**Guardrails triggered:** " + ", ".join(triggered))

        return "\n\n".join(blocks), json.dumps(payload, indent=2, default=str)

    def _network():  # noqa: ANN202
        summary = asyncio.run(discover_network())
        rows = [
            "| Agent | Framework | URL | Online | Streaming | Push | Skills |",
            "| --- | --- | --- | :-: | :-: | :-: | --- |",
        ]
        for key, info in summary.items():
            rows.append(
                f"| {info['name']} | {info['framework']} | `{info['url']}` | "
                f"{'yes' if info['online'] else 'NO'} | "
                f"{'yes' if info.get('card_streaming') else 'no'} | "
                f"{'yes' if info.get('card_push_notifications') else 'no'} | "
                f"{', '.join(info['skills']) or '—'} |"
            )
        offline = [info["name"] for info in summary.values() if not info["online"]]
        note = (
            "\n\nAll agents are reachable."
            if not offline
            else "\n\nOffline: " + ", ".join(offline)
            + "\n\nStart them with `python run_services.py --agents`."
        )
        return "\n".join(rows) + note

    def _system():  # noqa: ANN202
        from ..rag import retrieval

        try:
            index_stats = retrieval.store_stats()
        except Exception as exc:  # noqa: BLE001
            index_stats = {"error": str(exc)}

        info = {
            "llm": provider_banner(),
            "mcp_primary": settings.mcp_primary_url,
            "mcp_analytics": settings.mcp_analytics_url,
            "mock_ehr": settings.ehr_base_url,
            "langfuse_enabled": tracing.langfuse_available(),
            "langfuse_host": settings.langfuse_base_url,
            "a2a_auth_header": settings.a2a_auth_header,
            "pipeline_mode_detected": default_mode(),
            "vector_index": index_stats,
            "patients_available": patient_ids,
        }
        return json.dumps(info, indent=2, default=str)

    with gr.Blocks(title="Host Orchestrator — Agentic Discharge System") as ui:
        gr.Markdown(
            "# Host Orchestrator (Google ADK)\n"
            "A2A client for the six discharge agents. "
            f"Provider: `{settings.llm_provider}` · "
            f"MCP: `{settings.mcp_primary_url}` + `{settings.mcp_analytics_url}`"
        )

        with gr.Tab("Process a discharge"):
            gr.Markdown(
                "Runs extract → normalise → validate. The discharge summary is "
                "generated separately by a reviewer on the HITL dashboard "
                "(view 5), per PDF §2.5."
            )
            with gr.Row():
                patient_input = gr.Dropdown(
                    choices=patient_ids,
                    value=patient_ids[0] if patient_ids else None,
                    label="Patient",
                )
                mode_input = gr.Radio(
                    choices=["a2a", "local"], value=default_mode(),
                    label="Execution mode",
                    info="a2a = call each agent over the A2A protocol",
                )
            run_button = gr.Button("Run the pipeline", variant="primary")
            pipeline_output = gr.Markdown()
            pipeline_json = gr.Code(language="json", label="Full result")
            run_button.click(
                _run_pipeline,
                inputs=[patient_input, mode_input],
                outputs=[pipeline_output, pipeline_json],
            )

        with gr.Tab("Ask the records"):
            with gr.Row():
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="What medications was P1019 discharged on?",
                    lines=2, scale=3,
                )
                scope_input = gr.Dropdown(
                    choices=[""] + patient_ids, value="",
                    label="Restrict to patient (optional)", scale=1,
                )
            ask_button = gr.Button("Ask the RAG agent", variant="primary")
            answer_output = gr.Markdown()
            answer_json = gr.Code(language="json", label="Full result")
            ask_button.click(
                _ask,
                inputs=[question_input, scope_input],
                outputs=[answer_output, answer_json],
            )

        with gr.Tab("Agent network"):
            discover_button = gr.Button("Discover AgentCards", variant="primary")
            network_output = gr.Markdown()
            discover_button.click(_network, outputs=network_output)

        with gr.Tab("System"):
            system_button = gr.Button("Refresh", variant="primary")
            system_output = gr.Code(language="json", label="Configuration")
            system_button.click(_system, outputs=system_output)

    return ui


def main() -> None:
    configure_logging("orchestrator")
    port = int(settings.service("orchestrator_ui")["port"])
    log.info("Host Orchestrator (ADK + Gradio) → http://127.0.0.1:%d", port)
    log.info("  %s", provider_banner())

    ui = build_ui()
    #  `launch()` kwargs move between Gradio majors (4/5 accept `show_api`,
    #  6 removed it), so only pass what this installation actually supports.
    import inspect

    supported = set(inspect.signature(ui.launch).parameters)
    options = {
        key: value
        for key, value in {
            "server_name": "127.0.0.1",
            "server_port": port,
            "quiet": True,
            "inbrowser": False,
            "show_api": False,
        }.items()
        if key in supported
    }

    try:
        ui.launch(**options)
    finally:
        tracing.flush()


if __name__ == "__main__":
    main()
