"""
agents/base.py
==============

Shared plumbing for the six agents: audit-trail accumulation, MCP session
helpers, and the ExtractedCase (de)serialisation used to move a case between
agents over A2A.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from ..common.schemas import AuditTrailEntry, ExtractedCase
from ..mcp_client import ClinicalMCPClient
from ..observability import tracing
from ..settings import settings

log = logging.getLogger(__name__)


class AuditTrail:
    """Accumulates the per-step audit entries stamped onto every report."""

    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id = trace_id
        self.entries: list[AuditTrailEntry] = []

    def add(
        self,
        step: str,
        actor: str,
        *,
        framework: str = "",
        status: str = "ok",
        detail: str = "",
        duration_ms: int | None = None,
    ) -> AuditTrailEntry:
        entry = AuditTrailEntry(
            step=step,
            actor=actor,
            framework=framework,
            status=status,
            detail=detail,
            duration_ms=duration_ms,
            langfuse_trace_id=self.trace_id,
        )
        self.entries.append(entry)
        return entry

    @asynccontextmanager
    async def step(self, step: str, actor: str, *, framework: str = ""):
        """Time a step and record it, capturing failures as error entries."""
        started = time.perf_counter()
        try:
            yield
        except Exception as exc:
            self.add(
                step, actor, framework=framework, status="error",
                detail=f"{type(exc).__name__}: {exc}",
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
            raise
        else:
            self.add(
                step, actor, framework=framework,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

    def extend(self, entries: list[dict[str, Any]] | list[AuditTrailEntry]) -> None:
        for entry in entries or []:
            if isinstance(entry, AuditTrailEntry):
                self.entries.append(entry)
            elif isinstance(entry, dict):
                try:
                    self.entries.append(AuditTrailEntry.model_validate(entry))
                except Exception:  # noqa: BLE001
                    log.debug("Skipping malformed audit entry: %r", entry)

    def dump(self) -> list[dict[str, Any]]:
        return [entry.model_dump(mode="json") for entry in self.entries]


@asynccontextmanager
async def agent_mcp(
    agent_name: str,
    *,
    trace_id: str | None = None,
    servers: tuple[str, ...] = ("primary", "analytics"),
    elicitation_responder: Any = None,
):
    """Open MCP sessions for an agent, declaring the input Root.

    Declaring `Data/incoming` as a Root here is what lets the Clinical Watcher
    tool discover the workspace with `ctx.list_roots()` instead of being handed
    a raw path.
    """
    client = ClinicalMCPClient(
        trace_id=trace_id,
        servers=servers,
        roots=[settings.path("input_root")],
        agent_name=agent_name,
        elicitation_responder=elicitation_responder,
    )
    async with client:
        if not client.is_connected:
            log.warning(
                "%s could not reach any MCP server — falling back to in-process "
                "tool logic. Start them with: python -m discharge_ai.mcp_servers."
                "primary_server / analytics_server", agent_name,
            )
        yield client


def case_from_payload(payload: dict[str, Any]) -> ExtractedCase | None:
    """Rehydrate an `ExtractedCase` that arrived over A2A."""
    raw = payload.get("case") or payload.get("extracted")
    if not raw:
        return None
    try:
        return ExtractedCase.model_validate(raw)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not parse the incoming case payload: %s", exc)
        return None


def require_patient_id(payload: dict[str, Any]) -> str:
    patient_id = str(
        payload.get("patient_id") or payload.get("patientId") or ""
    ).strip().upper()
    if not patient_id:
        raise ValueError(
            "patient_id is required. Send {\"patient_id\": \"P1019\"} as the A2A "
            "message payload."
        )
    return patient_id


def trace_for(payload: dict[str, Any], call_trace_id: str | None,
              patient_id: str) -> str:
    """Resolve the end-to-end trace id for this case."""
    return (
        call_trace_id
        or payload.get("trace_id")
        or tracing.start_case_trace(patient_id)
    )
