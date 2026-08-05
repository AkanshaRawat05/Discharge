"""
a2a_layer/client.py
===================

A2A client used by the Host Orchestrator and the HITL dashboard.

Two modes, exactly as the specification's Table 10 requires:

* **non-streaming** — `await client.send(agent_key, payload)` → one final artifact
* **streaming**     — `async for event in client.stream(agent_key, payload)`
  → progressive `StreamChunk`s from the Summary Generator (:8104) and the
  Clinical RAG Q&A agent (:8105)

Both attach the shared secret `X-Agent-Auth-Token` header and propagate the case
`trace_id` in A2A message metadata, so every hop lands on the same LangFuse
trace.  Agent discovery goes through the AgentCard at
`/.well-known/agent.json`, and the card's `capabilities.streaming` flag decides
which mode is legal for a given agent.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    DataPart,
    Message,
    MessageSendConfiguration,
    MessageSendParams,
    Part,
    Role,
    SendMessageRequest,
    SendStreamingMessageRequest,
    TextPart,
)

from ..observability import tracing
from ..settings import settings
from .auth import auth_headers

log = logging.getLogger(__name__)


@dataclass
class StreamChunk:
    """One progressive event from a streaming agent."""

    text: str = ""
    section: str = ""
    data: dict[str, Any] | None = None
    final: bool = False
    state: str = "working"

    @property
    def is_data(self) -> bool:
        return self.data is not None


@dataclass
class AgentResult:
    """Final result of a non-streaming (or completed streaming) A2A call."""

    agent: str
    ok: bool = True
    data: dict[str, Any] = field(default_factory=dict)
    text: str = ""
    task_id: str | None = None
    error: str | None = None

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


class A2AAgentClient:
    """Streaming-capable A2A client for the six agents."""

    def __init__(self, *, trace_id: str | None = None, timeout: float | None = None) -> None:
        self.trace_id = trace_id
        self.timeout = timeout or float(settings.a2a_timeout)
        self._cards: dict[str, AgentCard] = {}

    # ------------------------------------------------------------------ #
    #  Discovery
    # ------------------------------------------------------------------ #
    async def discover(self, agent_key: str) -> AgentCard | None:
        """Fetch and cache an AgentCard from `/.well-known/agent.json`."""
        if agent_key in self._cards:
            return self._cards[agent_key]

        base_url = settings.agent_url(agent_key)
        try:
            async with httpx.AsyncClient(timeout=15.0, headers=auth_headers()) as http:
                resolver = A2ACardResolver(httpx_client=http, base_url=base_url)
                card = await resolver.get_agent_card()
                self._cards[agent_key] = card
                return card
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "AgentCard discovery failed for %s at %s (%s: %s)",
                agent_key, base_url, type(exc).__name__, exc,
            )
            return None

    async def discover_all(self) -> dict[str, dict[str, Any]]:
        """Card summary for every configured agent (for the dashboard/UI)."""
        summary: dict[str, dict[str, Any]] = {}
        for agent_key, config in settings.cfg["a2a"]["agents"].items():
            if config.get("role") == "a2a-client":
                continue
            card = await self.discover(agent_key)
            summary[agent_key] = {
                "name": config["name"],
                "url": config["url"],
                "framework": config.get("framework"),
                "configured_streaming": bool(config.get("streaming")),
                "online": card is not None,
                "card_streaming": bool(card.capabilities.streaming) if card else None,
                "card_push_notifications": (
                    bool(card.capabilities.push_notifications) if card else None
                ),
                "skills": [skill.id for skill in card.skills] if card else [],
            }
        return summary

    # ------------------------------------------------------------------ #
    #  Non-streaming
    # ------------------------------------------------------------------ #
    async def send(self, agent_key: str, payload: dict[str, Any]) -> AgentResult:
        """`send_message()` — await a single final artifact."""
        config = settings.agent(agent_key)
        message = self._build_message(payload)

        with tracing.chain_span(
            f"a2a.send → {config['name']}",
            trace_id=self.trace_id,
            input=payload,
            agent=agent_key,
            streaming=False,
        ) as span:
            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout, headers=auth_headers()
                ) as http:
                    card = await self.discover(agent_key)
                    client = (
                        A2AClient(httpx_client=http, agent_card=card)
                        if card
                        else A2AClient(httpx_client=http, url=config["url"])
                    )

                    request = SendMessageRequest(
                        id=uuid.uuid4().hex,
                        params=MessageSendParams(
                            message=message,
                            configuration=MessageSendConfiguration(
                                accepted_output_modes=["text", "application/json"],
                                blocking=True,
                            ),
                        ),
                    )
                    response = await client.send_message(request)

                result = _parse_response(config["name"], response)
                span.set_output(result.data or result.text)
                return result

            except Exception as exc:  # noqa: BLE001
                span.fail(exc)
                log.warning("A2A call to %s failed: %s", config["name"], exc)
                tracing.record_error(
                    f"a2a.client.{agent_key}", exc, trace_id=self.trace_id,
                    fallback_action="orchestrator records the agent as unavailable",
                )
                return AgentResult(
                    agent=config["name"], ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                )

    # ------------------------------------------------------------------ #
    #  Streaming
    # ------------------------------------------------------------------ #
    async def stream(
        self, agent_key: str, payload: dict[str, Any]
    ) -> AsyncIterator[StreamChunk]:
        """`send_message_streaming()` — yield progressive chunks."""
        config = settings.agent(agent_key)
        message = self._build_message(payload)

        if not config.get("streaming"):
            #  Never silently downgrade: surface the mismatch, then deliver the
            #  non-streaming result as a single final chunk so callers still work.
            log.warning(
                "%s is configured as non-streaming; returning one final chunk.",
                config["name"],
            )
            result = await self.send(agent_key, payload)
            yield StreamChunk(
                text=result.text, data=result.data or None, final=True,
                state="completed" if result.ok else "failed",
            )
            return

        with tracing.chain_span(
            f"a2a.stream → {config['name']}",
            trace_id=self.trace_id,
            input=payload,
            agent=agent_key,
            streaming=True,
        ) as span:
            chunks = 0
            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout, headers=auth_headers()
                ) as http:
                    card = await self.discover(agent_key)
                    client = (
                        A2AClient(httpx_client=http, agent_card=card)
                        if card
                        else A2AClient(httpx_client=http, url=config["url"])
                    )

                    request = SendStreamingMessageRequest(
                        id=uuid.uuid4().hex,
                        params=MessageSendParams(
                            message=message,
                            configuration=MessageSendConfiguration(
                                accepted_output_modes=["text", "application/json"],
                            ),
                        ),
                    )

                    async for event in client.send_message_streaming(request):
                        for chunk in _parse_stream_event(event):
                            chunks += 1
                            yield chunk

                span.set_metadata(chunks=chunks)

            except Exception as exc:  # noqa: BLE001
                span.fail(exc)
                log.warning("A2A stream from %s failed: %s", config["name"], exc)
                tracing.record_error(
                    f"a2a.stream.{agent_key}", exc, trace_id=self.trace_id,
                    fallback_action="stream terminated with a failure chunk",
                )
                yield StreamChunk(
                    text=f"[stream error: {type(exc).__name__}: {exc}]",
                    final=True, state="failed",
                )

    # ------------------------------------------------------------------ #
    #  Push notifications
    # ------------------------------------------------------------------ #
    async def register_push_notification(
        self, agent_key: str, task_id: str, webhook_url: str,
        token: str | None = None,
    ) -> bool:
        """Register a webhook so the agent POSTs task updates to `webhook_url`."""
        from a2a.types import (
            PushNotificationConfig,
            SetTaskPushNotificationConfigRequest,
            TaskPushNotificationConfig,
        )

        config = settings.agent(agent_key)
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, headers=auth_headers()
            ) as http:
                client = A2AClient(httpx_client=http, url=config["url"])
                request = SetTaskPushNotificationConfigRequest(
                    id=uuid.uuid4().hex,
                    params=TaskPushNotificationConfig(
                        task_id=task_id,
                        push_notification_config=PushNotificationConfig(
                            url=webhook_url,
                            token=token or settings.a2a_auth_token,
                        ),
                    ),
                )
                await client.set_task_callback(request)
            log.info(
                "Push notifications registered for task %s → %s", task_id, webhook_url
            )
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("Push-notification registration failed: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    def _build_message(self, payload: dict[str, Any]) -> Message:
        """Build the A2A message, carrying the trace id in metadata."""
        body = dict(payload)
        if self.trace_id:
            body.setdefault("trace_id", self.trace_id)

        return Message(
            message_id=uuid.uuid4().hex,
            role=Role.user,
            parts=[
                #  DataPart for the structured request, plus a JSON TextPart so
                #  simple/CLI peers can read it too.
                Part(root=DataPart(data=body)),
                Part(root=TextPart(text=json.dumps(body, ensure_ascii=False, default=str))),
            ],
            metadata={"trace_id": self.trace_id} if self.trace_id else None,
        )


# --------------------------------------------------------------------------- #
#  Response parsing
# --------------------------------------------------------------------------- #
def _parts_of(container: Any) -> list[Any]:
    parts = getattr(container, "parts", None) or []
    return [getattr(part, "root", part) for part in parts]


def _collect(parts: list[Any]) -> tuple[dict[str, Any], str]:
    data: dict[str, Any] = {}
    texts: list[str] = []

    for part in parts:
        part_data = getattr(part, "data", None)
        if isinstance(part_data, dict):
            data.update(part_data)
            continue
        text = getattr(part, "text", None)
        if text:
            texts.append(text)

    joined = "\n".join(texts)
    if not data and joined.strip().startswith("{"):
        try:
            parsed = json.loads(joined)
            if isinstance(parsed, dict):
                data = parsed
        except json.JSONDecodeError:
            pass
    return data, joined


def _parse_response(agent_name: str, response: Any) -> AgentResult:
    root = getattr(response, "root", response)

    error = getattr(root, "error", None)
    if error is not None:
        return AgentResult(
            agent=agent_name, ok=False,
            error=f"{getattr(error, 'code', '')} {getattr(error, 'message', error)}".strip(),
        )

    result = getattr(root, "result", None)
    if result is None:
        return AgentResult(agent=agent_name, ok=False, error="empty A2A response")

    #  A Task carries artifacts; a bare Message carries parts directly.
    artifacts = getattr(result, "artifacts", None)
    if artifacts:
        parts: list[Any] = []
        for artifact in artifacts:
            parts += _parts_of(artifact)
        data, text = _collect(parts)
        state = getattr(getattr(result, "status", None), "state", None)
        return AgentResult(
            agent=agent_name,
            ok=str(state) != "TaskState.failed",
            data=data,
            text=text,
            task_id=getattr(result, "id", None),
        )

    data, text = _collect(_parts_of(result))
    return AgentResult(
        agent=agent_name, ok=True, data=data, text=text,
        task_id=getattr(result, "id", None),
    )


def _parse_stream_event(event: Any) -> list[StreamChunk]:
    """Turn one SSE event into zero or more `StreamChunk`s."""
    root = getattr(event, "root", event)
    result = getattr(root, "result", None)
    if result is None:
        return []

    chunks: list[StreamChunk] = []

    #  1. Incremental status updates carry the progressive text.
    status = getattr(result, "status", None)
    if status is not None and getattr(status, "message", None) is not None:
        data, text = _collect(_parts_of(status.message))
        metadata = getattr(status.message, "metadata", None) or {}
        state = str(getattr(status, "state", "working")).split(".")[-1]
        if text or data:
            chunks.append(
                StreamChunk(
                    text=text,
                    section=str(metadata.get("section", "")),
                    data=data or None,
                    final=bool(getattr(result, "final", False)),
                    state=state,
                )
            )

    #  2. Artifact updates carry the completed payload.
    artifact = getattr(result, "artifact", None)
    if artifact is not None:
        data, text = _collect(_parts_of(artifact))
        chunks.append(StreamChunk(text=text, data=data or None, final=False, state="working"))

    #  3. Terminal event.
    if getattr(result, "final", False) and not chunks:
        chunks.append(StreamChunk(final=True, state="completed"))

    return chunks


# --------------------------------------------------------------------------- #
#  One-shot helpers
# --------------------------------------------------------------------------- #
async def call_agent(
    agent_key: str, payload: dict[str, Any], *, trace_id: str | None = None
) -> AgentResult:
    return await A2AAgentClient(trace_id=trace_id).send(agent_key, payload)


async def stream_agent(
    agent_key: str, payload: dict[str, Any], *, trace_id: str | None = None
) -> AsyncIterator[StreamChunk]:
    client = A2AAgentClient(trace_id=trace_id)
    async for chunk in client.stream(agent_key, payload):
        yield chunk
