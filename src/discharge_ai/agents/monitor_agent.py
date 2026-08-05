"""
agents/monitor_agent.py
=======================

**Discharge Monitor Agent** — Google ADK — A2A :8103 — non-streaming
MCP primitives: Tools + **Roots**

Scans the incoming-document folder for new patient discharge files.  No live EHR
integration; file-system simulation, as the specification allows.

Roots integration (mandatory, §2.1):

* the agent **declares** `file:///…/Data/incoming` as a Root URI when it opens
  the MCP connection (`ClinicalMCPClient(roots=[…])` → `list_roots_callback`);
* the Clinical Watcher MCP tool calls `ctx.list_roots()` to discover authorised
  folders at runtime — **no raw path is ever passed as a tool parameter**;
* the server rejects anything outside the declared root with a
  `Path.relative_to()` containment check.

The ADK `LlmAgent` sits on top as the reasoning layer: it is given a
`FunctionTool` wrapper around the Roots-scoped watcher and asked to report which
patients have a complete document set.  If the LLM is unavailable (quota, no
key), the deterministic scan result is still returned — monitoring must not
depend on a model being reachable.

Run:
    python -m discharge_ai.agents.monitor_agent
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..common.doc_loader import scan_incoming
from ..observability import tracing
from ..settings import settings
from .base import AuditTrail, agent_mcp, trace_for

log = logging.getLogger(__name__)

AGENT_KEY = "monitor"
AGENT_NAME = settings.agent(AGENT_KEY)["name"]
FRAMEWORK = "google-adk"

REQUIRED_DOC_TYPES = ("discharge_report", "lab_report", "bill")

ADK_INSTRUCTION = """
You are the Discharge Monitor Agent at St. Marian Regional Medical Center.

You have one tool, `scan_discharge_workspace`, which lists the patient documents
found inside the hospital's authorised document workspace. Call it exactly once,
then report:

1. how many patients were detected;
2. which patients have a COMPLETE document set (discharge report + lab report +
   bill) and are ready for processing;
3. which patients are INCOMPLETE, naming the missing document type(s).

Be terse and factual. Never invent a patient id or a document that the tool did
not return. Do not give clinical opinions — you are a file-system monitor.
""".strip()


# --------------------------------------------------------------------------- #
#  Roots-scoped scan  (the ADK tool body)
# --------------------------------------------------------------------------- #
async def _scan_via_mcp(mcp: Any, patient_id: str | None) -> dict[str, Any]:
    """Call the Clinical Watcher tool, which discovers folders via Roots."""
    if mcp.is_connected:
        arguments: dict[str, Any] = {}
        if patient_id:
            arguments["patient_id"] = patient_id
        #  Deliberately no path argument — folder discovery is the server's job
        #  via ctx.list_roots().
        result = await mcp.call_tool("clinical_watcher", arguments)
        if isinstance(result, dict) and not result.get("error"):
            return result

    catalogue = scan_incoming()
    if patient_id:
        catalogue = {
            pid: docs for pid, docs in catalogue.items()
            if pid.upper() == patient_id.upper()
        }
    root = settings.path("input_root")
    return {
        "authorised_roots": [root.as_uri()],
        "roots_discovered_via": "local fallback (MCP server unreachable)",
        "patient_count": len(catalogue),
        "patients": catalogue,
        "new_cases": sorted(catalogue.keys()),
    }


def _classify(catalogue: dict[str, dict[str, list[str]]]) -> dict[str, Any]:
    """Split the catalogue into complete vs incomplete document sets."""
    complete: list[str] = []
    incomplete: list[dict[str, Any]] = []

    for patient_id, documents in sorted(catalogue.items()):
        missing = [
            doc_type for doc_type in REQUIRED_DOC_TYPES if not documents.get(doc_type)
        ]
        if missing:
            incomplete.append({"patient_id": patient_id, "missing": missing})
        else:
            complete.append(patient_id)

    return {
        "ready_for_processing": complete,
        "incomplete_cases": incomplete,
        "ready_count": len(complete),
        "incomplete_count": len(incomplete),
    }


# --------------------------------------------------------------------------- #
#  ADK agent
# --------------------------------------------------------------------------- #
async def _run_adk_summary(scan_payload: dict[str, Any], trace_id: str | None) -> str:
    """Let the ADK LlmAgent narrate the scan through a FunctionTool."""
    try:
        from google.adk.agents import LlmAgent
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.adk.tools import FunctionTool
        from google.genai import types

        from ..llm import provider
        from ..llm.provider import get_adk_model
    except ImportError as exc:  # pragma: no cover
        log.warning("Google ADK unavailable (%s) — skipping the narrative step.", exc)
        return ""

    if provider.is_quota_exhausted():
        log.info("LLM breaker is open — returning the scan without narration.")
        return ""

    def scan_discharge_workspace() -> dict[str, Any]:
        """List patient documents in the authorised discharge workspace.

        Returns the patients discovered, the document types held for each, and
        which cases are ready for processing.
        """
        return scan_payload

    agent = LlmAgent(
        name="discharge_monitor",
        model=get_adk_model("fast"),
        description="Monitors the hospital discharge document workspace.",
        instruction=ADK_INSTRUCTION,
        tools=[FunctionTool(func=scan_discharge_workspace)],
    )

    session_service = InMemorySessionService()
    runner = Runner(
        app_name="discharge-monitor", agent=agent, session_service=session_service
    )
    await session_service.create_session(
        app_name="discharge-monitor", user_id="orchestrator", session_id="monitor-scan"
    )

    with tracing.llm_generation(
        "monitor.adk_summary",
        model=str(get_adk_model("fast")),
        prompt="Scan the workspace and report readiness.",
        trace_id=trace_id,
    ) as span:
        parts: list[str] = []
        try:
            async for event in runner.run_async(
                user_id="orchestrator",
                session_id="monitor-scan",
                new_message=types.Content(
                    role="user",
                    parts=[types.Part(
                        text="Scan the discharge workspace and report which "
                             "patients are ready for processing."
                    )],
                ),
            ):
                content = getattr(event, "content", None)
                if content and getattr(content, "parts", None):
                    for part in content.parts:
                        if getattr(part, "text", None):
                            parts.append(part.text)
        except Exception as exc:  # noqa: BLE001
            log.warning("ADK narration failed (%s: %s)", type(exc).__name__, exc)
            span.fail(exc)
            tracing.record_error(
                "adk.monitor", exc, trace_id=trace_id,
                fallback_action="return the deterministic scan without narration",
            )
            return ""

        summary = "\n".join(part.strip() for part in parts if part.strip())
        span.set_output(summary[:2000])
        return summary


# --------------------------------------------------------------------------- #
async def handle(payload: dict[str, Any], ctx: Any) -> dict[str, Any]:
    """A2A entry point: `{}` or `{"patient_id": "P1019"}` → workspace scan."""
    patient_id = str(payload.get("patient_id") or "").strip().upper() or None
    trace_id = trace_for(payload, getattr(ctx, "trace_id", None), patient_id or "workspace")
    audit = AuditTrail(trace_id)

    async with agent_mcp(AGENT_NAME, trace_id=trace_id, servers=("primary",)) as mcp:
        async with audit.step("scan_workspace_via_roots",
                              "clinical_watcher tool (MCP Roots)", framework=FRAMEWORK):
            scan = await _scan_via_mcp(mcp, patient_id)

    catalogue = scan.get("patients", {}) or {}
    classification = _classify(catalogue)

    narrative = ""
    if payload.get("narrate", True):
        async with audit.step("adk_narrate_scan", "ADK LlmAgent", framework=FRAMEWORK):
            narrative = await _run_adk_summary({**scan, **classification}, trace_id)

    log.info(
        "Monitor scan: %d patient(s), %d ready, %d incomplete (roots=%s)",
        scan.get("patient_count", 0),
        classification["ready_count"],
        classification["incomplete_count"],
        scan.get("authorised_roots"),
    )

    return {
        "agent": AGENT_NAME,
        "framework": FRAMEWORK,
        "trace_id": trace_id,
        "authorised_roots": scan.get("authorised_roots", []),
        "roots_discovered_via": scan.get("roots_discovered_via"),
        "patient_count": scan.get("patient_count", 0),
        "patients": catalogue,
        "new_cases": scan.get("new_cases", []),
        **classification,
        "narrative": narrative,
        "audit_trail": audit.dump(),
        "mcp_primitives_used": ["tools", "roots"],
    }


def main() -> None:
    from ..a2a_layer import run_agent_server

    run_agent_server(AGENT_KEY, handle, artifact_name="workspace_scan")


if __name__ == "__main__":
    main()
