"""
a2a_layer/executor.py
=====================

`HandlerAgentExecutor` — the bridge between an ordinary async Python handler and
the A2A `AgentExecutor` interface, so each agent module only has to write its
clinical logic.

Two handler shapes are supported:

* **non-streaming** — `async def handler(payload, ctx) -> dict | str`
  One final artifact is returned when the work completes.

* **streaming** — `async def handler(payload, ctx) -> AsyncIterator[StreamEvent]`
  Each yielded `StreamEvent` becomes an incremental A2A status update, and the
  event marked `final=True` (or the last one) carries the completed artifact.
  This is what the Summary Generator (:8104) and RAG agent (:8105) use for
  progressive delivery.

The end-to-end **trace id** travels in A2A message metadata (`trace_id`), which
is what makes spans from six separate processes land on one LangFuse trace.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Part, TaskState, TextPart
from a2a.utils import new_agent_text_message

from ..observability import tracing

log = logging.getLogger(__name__)


@dataclass
class StreamEvent:
    """One progressive update emitted by a streaming agent handler."""

    text: str = ""
    section: str = ""
    data: dict[str, Any] | None = None
    final: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentCallContext:
    """What a handler is told about the A2A call it is serving."""

    payload: dict[str, Any]
    raw_text: str
    trace_id: str | None
    task_id: str
    context_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


NonStreamingHandler = Callable[[dict[str, Any], AgentCallContext], Awaitable[Any]]
StreamingHandler = Callable[
    [dict[str, Any], AgentCallContext], AsyncIterator[StreamEvent]
]


class HandlerAgentExecutor(AgentExecutor):
    """Adapt a plain handler to the A2A execution protocol."""

    def __init__(
        self,
        *,
        agent_name: str,
        framework: str,
        handler: NonStreamingHandler | StreamingHandler,
        streaming: bool = False,
        artifact_name: str = "result",
    ) -> None:
        self.agent_name = agent_name
        self.framework = framework
        self.handler = handler
        self.streaming = streaming
        self.artifact_name = artifact_name

    # ------------------------------------------------------------------ #
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        raw_text = context.get_user_input() or ""
        payload, parse_note = _parse_payload(raw_text, context)
        metadata = dict(context.message.metadata or {}) if context.message else {}

        #  The caller propagates the case trace id in message metadata so all
        #  agents' spans join the same LangFuse trace.
        trace_id = metadata.get("trace_id") or payload.get("trace_id")

        call_context = AgentCallContext(
            payload=payload,
            raw_text=raw_text,
            trace_id=trace_id,
            task_id=context.task_id,
            context_id=context.context_id,
            metadata=metadata,
        )

        await updater.submit()
        await updater.start_work(
            new_agent_text_message(
                f"{self.agent_name} started.", context.context_id, context.task_id
            )
        )
        if parse_note:
            log.debug("%s: %s", self.agent_name, parse_note)

        with tracing.agent_span(
            self.agent_name,
            self.framework,
            trace_id=trace_id,
            input=payload,
            a2a_task_id=context.task_id,
            a2a_streaming=self.streaming,
        ) as span:
            try:
                if self.streaming:
                    result = await self._run_streaming(call_context, updater, context)
                else:
                    result = await self._run_once(call_context, updater, context)

                span.set_output(result)
                span.set_metadata(duration_ms=span.duration_ms)

            except Exception as exc:  # noqa: BLE001 — must surface as an A2A failure
                log.exception("%s failed", self.agent_name)
                span.fail(exc)
                tracing.record_error(
                    f"a2a.{self.agent_name}", exc, trace_id=trace_id,
                    fallback_action="A2A task marked failed",
                )
                await updater.failed(
                    new_agent_text_message(
                        f"{self.agent_name} failed: {type(exc).__name__}: {exc}",
                        context.context_id, context.task_id,
                    )
                )

    # ------------------------------------------------------------------ #
    async def _run_once(
        self, call: AgentCallContext, updater: TaskUpdater, context: RequestContext
    ) -> Any:
        result = await self.handler(call.payload, call)  # type: ignore[misc]
        parts = _result_to_parts(result)

        await updater.add_artifact(parts, name=self.artifact_name)
        await updater.complete(
            new_agent_text_message(
                f"{self.agent_name} completed.", context.context_id, context.task_id
            )
        )
        return result

    async def _run_streaming(
        self, call: AgentCallContext, updater: TaskUpdater, context: RequestContext
    ) -> Any:
        collected: list[str] = []
        final_data: dict[str, Any] | None = None
        section_count = 0

        async for event in self.handler(call.payload, call):  # type: ignore[misc]
            if event.text:
                collected.append(event.text)
            if event.data:
                final_data = event.data
            if event.section:
                section_count += 1

            if event.final:
                break

            #  Progressive delivery: each chunk is a non-final status update the
            #  A2A client consumes with `async for event in stream`.
            message = new_agent_text_message(
                event.text or f"[{event.section}]",
                context.context_id,
                context.task_id,
            )
            if event.section or event.metadata:
                message.metadata = {"section": event.section, **event.metadata}

            await updater.update_status(TaskState.working, message=message)

        full_text = "".join(collected)
        parts: list[Part] = []
        if full_text.strip():
            parts.append(Part(root=TextPart(text=full_text)))
        if final_data is not None:
            parts.append(Part(root=DataPart(data=final_data)))
        if not parts:
            parts.append(Part(root=TextPart(text="")))

        await updater.add_artifact(
            parts, name=self.artifact_name, metadata={"sections": section_count}
        )
        await updater.complete(
            new_agent_text_message(
                f"{self.agent_name} completed ({section_count} sections).",
                context.context_id, context.task_id,
            )
        )
        return final_data if final_data is not None else full_text

    # ------------------------------------------------------------------ #
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        log.info("%s: task %s cancelled by the client", self.agent_name, context.task_id)
        await updater.cancel(
            new_agent_text_message(
                f"{self.agent_name} task cancelled.",
                context.context_id, context.task_id,
            )
        )


# --------------------------------------------------------------------------- #
#  Payload helpers
# --------------------------------------------------------------------------- #
def _parse_payload(raw_text: str, context: RequestContext) -> tuple[dict[str, Any], str]:
    """Recover the structured request from a JSON text part or DataParts."""
    #  1. DataPart is the cleanest carrier — check it first.
    if context.message and context.message.parts:
        for part in context.message.parts:
            root = getattr(part, "root", part)
            data = getattr(root, "data", None)
            if isinstance(data, dict) and data:
                return data, "payload read from DataPart"

    # 2. JSON in the text part.
    text = (raw_text or "").strip()
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed, "payload read from JSON text part"
        except json.JSONDecodeError:
            pass

    # 3. Plain prose — hand it to the agent as `question`/`text` and try to spot
    #    a patient id so single-line requests still work from a chat UI.
    import re

    payload: dict[str, Any] = {"text": text, "question": text}
    match = re.search(r"(?<![A-Za-z0-9])(P\d{4})(?![0-9])", text, re.IGNORECASE)
    if match:
        payload["patient_id"] = match.group(1).upper()
    return payload, "payload treated as free text"


def _result_to_parts(result: Any) -> list[Part]:
    """Wrap a handler result as A2A parts (DataPart for dicts, TextPart else)."""
    if result is None:
        return [Part(root=TextPart(text=""))]

    if isinstance(result, str):
        return [Part(root=TextPart(text=result))]

    if hasattr(result, "model_dump"):
        result = result.model_dump(mode="json")

    if isinstance(result, dict):
        #  Send both: DataPart for machines, JSON TextPart for clients that only
        #  read text (Gradio chat, curl).
        return [
            Part(root=DataPart(data=result)),
            Part(root=TextPart(text=json.dumps(result, ensure_ascii=False, default=str))),
        ]

    return [Part(root=TextPart(text=json.dumps(result, ensure_ascii=False, default=str)))]
