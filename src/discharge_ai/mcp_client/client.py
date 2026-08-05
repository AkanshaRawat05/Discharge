"""
mcp_client/client.py
====================

`ClinicalMCPClient` — connects to **both** MCP servers simultaneously
(:8200 `/clinicaltools` and :8201 `/analyticstools`) and implements the three
client-side halves of the MCP primitives:

* **sampling_callback**   — the server asks *us* to run inference
  (`ctx.session.create_message()`).  We read the server's `ModelPreferences`
  hints (`nova-lite` for multilingual, `command-r-plus` for English), route them
  onto the ACTIVE provider via `configs/agent_config.yaml → sampling.hint_routing`,
  perform the completion and return a `CreateMessageResult`.  This is the point
  of the Sampling primitive: LLM resource management is the client's job, tool
  logic is the server's.

* **elicitation_callback** — the server asks a human for missing field values
  (`ctx.elicit()`).  The default responder is non-interactive (declines, so the
  fields are flagged for HITL); the Streamlit dashboard injects a responder that
  renders a dynamic form and returns the reviewer's `ElicitResult`.

* **list_roots_callback** — we declare the authorised workspace
  (`file:///…/Data/incoming`) so the Clinical Watcher tool can discover it with
  `ctx.list_roots()` instead of receiving a raw path.

Usage
-----
    async with ClinicalMCPClient(trace_id=trace_id) as client:
        docs = await client.call_tool("clinical_watcher", {"patient_id": "P1019"})
        rules = await client.read_resource("resource://clinical-rules/completeness")
        prompt = await client.get_prompt("rag-answer-prompt", {"context_length": "800"})
        risk = await client.call_tool(
            "calculate_risk_score", {...}, server="analytics"
        )
"""

from __future__ import annotations

import json
import logging
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.context import RequestContext
from mcp.types import (
    CreateMessageRequestParams,
    CreateMessageResult,
    ElicitRequestParams,
    ElicitResult,
    ErrorData,
    Implementation,
    Root,
    TextContent,
)
from pydantic import AnyUrl, FileUrl

from ..llm import provider
from ..observability import tracing
from ..settings import settings

log = logging.getLogger(__name__)

#  An elicitation responder receives (message, requested_schema) and returns
#  either a dict of values (accept), or None (decline), or raises
#  `ElicitationCancelled` (cancel).
ElicitationResponder = Callable[
    [str, dict[str, Any]], Awaitable[dict[str, Any] | None]
]


class ElicitationCancelled(Exception):
    """Raised by a responder to signal the reviewer cancelled the request."""


async def default_elicitation_responder(
    message: str, requested_schema: dict[str, Any]
) -> dict[str, Any] | None:
    """Non-interactive default: decline so gaps are flagged for HITL.

    Agents run headless, so there is no human at the other end of an agent's MCP
    session.  Declining is the honest answer — it marks the fields unresolved and
    routes them to the Streamlit dashboard, which *does* have a reviewer.
    """
    fields = list((requested_schema or {}).get("properties", {}).keys())
    log.info(
        "Elicitation declined by the headless agent client (%d field(s): %s) — "
        "routed to the HITL dashboard.", len(fields), ", ".join(fields[:8]),
    )
    return None


def dashboard_elicitation_responder(
    answers: dict[str, Any] | None, *, cancel: bool = False, decline: bool = False
) -> ElicitationResponder:
    """Build a responder that replays a reviewer's dashboard form submission.

    The Streamlit HITL page renders the JSON Schema it receives, collects the
    reviewer's input, then re-runs validation with this responder installed so
    the MCP round-trip is genuine (`accept` / `decline` / `cancel` all exercised).
    """

    async def _responder(
        message: str, requested_schema: dict[str, Any]
    ) -> dict[str, Any] | None:
        if cancel:
            raise ElicitationCancelled(message)
        if decline or not answers:
            return None
        allowed = set((requested_schema or {}).get("properties", {}).keys())
        return {
            key: value
            for key, value in answers.items()
            if key in allowed and value not in (None, "")
        }

    return _responder


class ClinicalMCPClient:
    """Simultaneous sessions to the Primary and Analytics MCP servers."""

    PRIMARY = "primary"
    ANALYTICS = "analytics"

    def __init__(
        self,
        *,
        trace_id: str | None = None,
        servers: Iterable[str] = (PRIMARY, ANALYTICS),
        roots: list[Path] | None = None,
        elicitation_responder: ElicitationResponder | None = None,
        agent_name: str = "discharge-ai-agent",
        timeout_seconds: float = 120.0,
    ) -> None:
        self.trace_id = trace_id
        self.agent_name = agent_name
        self.timeout_seconds = timeout_seconds
        self.elicitation_responder = elicitation_responder or default_elicitation_responder

        self._server_urls = {
            self.PRIMARY: settings.mcp_primary_url,
            self.ANALYTICS: settings.mcp_analytics_url,
        }
        self._wanted = [name for name in servers if name in self._server_urls]

        #  Declared MCP Roots: the authorised document workspace.
        self.roots: list[Path] = roots or [settings.path("input_root")]

        self.sessions: dict[str, ClientSession] = {}
        self._stack: AsyncExitStack | None = None
        self._tool_index: dict[str, str] = {}       # tool name -> server key
        self.sampling_calls: list[dict[str, Any]] = []
        self.elicitation_calls: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #
    async def __aenter__(self) -> "ClinicalMCPClient":
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()

        for name in self._wanted:
            url = self._server_urls[name]
            try:
                read_stream, write_stream, _ = await self._stack.enter_async_context(
                    streamablehttp_client(url)
                )
                session = await self._stack.enter_async_context(
                    ClientSession(
                        read_stream,
                        write_stream,
                        sampling_callback=self._sampling_callback,
                        elicitation_callback=self._elicitation_callback,
                        list_roots_callback=self._list_roots_callback,
                        client_info=Implementation(
                            name=self.agent_name,
                            version=settings.cfg.get("project", {}).get("version", "1.0.0"),
                        ),
                    )
                )
                await session.initialize()
                self.sessions[name] = session
                log.info("MCP session established: %s → %s", name, url)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "Could not connect to the %s MCP server at %s (%s: %s)",
                    name, url, type(exc).__name__, exc,
                )
                tracing.record_error(
                    f"mcp.connect.{name}", exc, trace_id=self.trace_id,
                    fallback_action="agent continues without this MCP server",
                )

        await self._build_tool_index()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        if self._stack is not None:
            await self._stack.__aexit__(*exc_info)
            self._stack = None
        self.sessions.clear()

    @property
    def connected_servers(self) -> list[str]:
        return sorted(self.sessions.keys())

    @property
    def is_connected(self) -> bool:
        return bool(self.sessions)

    # ------------------------------------------------------------------ #
    #  MCP primitive: ROOTS  (client side)
    # ------------------------------------------------------------------ #
    async def _list_roots_callback(
        self, context: RequestContext[ClientSession, Any]
    ) -> Any:
        """Declare the authorised workspace to the server."""
        from mcp.types import ListRootsResult

        declared: list[Root] = []
        for path in self.roots:
            try:
                declared.append(
                    Root(
                        uri=FileUrl(path.resolve().as_uri()),
                        name=f"clinical-input:{path.name}",
                    )
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not express %s as a Root URI: %s", path, exc)

        tracing.record_event(
            "mcp.roots.list_roots",
            trace_id=self.trace_id,
            roots=[str(root.uri) for root in declared],
        )
        return ListRootsResult(roots=declared)

    # ------------------------------------------------------------------ #
    #  MCP primitive: SAMPLING  (client side)
    # ------------------------------------------------------------------ #
    async def _sampling_callback(
        self,
        context: RequestContext[ClientSession, Any],
        params: CreateMessageRequestParams,
    ) -> CreateMessageResult | ErrorData:
        """Run inference on the server's behalf, honouring its model hints."""
        prompt_text = "\n\n".join(
            block.text
            for message in params.messages
            for block in [message.content]
            if isinstance(block, TextContent)
        )

        #  Read the server's ModelPreferences and route onto the ACTIVE provider.
        hints = [
            hint.name for hint in (
                params.modelPreferences.hints if params.modelPreferences else []
            ) or [] if hint.name
        ]
        chosen_model = (
            provider.resolve_sampling_hint(hints[0]) if hints
            else provider.resolve_model_id("reasoning")
        )

        log.info(
            "MCP sampling request: server hints=%s → %s model %r",
            hints or ["<none>"], settings.llm_provider, chosen_model,
        )

        with tracing.llm_generation(
            "mcp.sampling.create_message",
            model=chosen_model,
            prompt=prompt_text[:4000],
            trace_id=self.trace_id,
            server_hints=hints,
            max_tokens=params.maxTokens,
        ) as span:
            try:
                #  `acomplete` runs the blocking SDK call in a worker thread:
                #  this callback executes on the agent's event loop, and blocking
                #  it would stall the agent's own A2A endpoint and MCP transport.
                text = await provider.acomplete(
                    prompt_text,
                    model=chosen_model,
                    system=params.systemPrompt,
                    temperature=params.temperature if params.temperature is not None else 0.0,
                    max_tokens=params.maxTokens or settings.llm_max_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                span.fail(exc)
                tracing.record_error(
                    "mcp.sampling.inference", exc, trace_id=self.trace_id,
                    fallback_action="return ErrorData to the MCP server",
                )
                return ErrorData(code=-32603, message=f"Sampling failed: {exc}")

            span.set_output(text[:2000])

        self.sampling_calls.append(
            {
                "server_hints": hints,
                "client_model_selected": chosen_model,
                "provider": settings.llm_provider,
                "prompt_chars": len(prompt_text),
                "response_chars": len(text or ""),
            }
        )

        if not text:
            return ErrorData(
                code=-32603,
                message="Sampling produced no output (LLM unavailable or rate-limited)",
            )

        return CreateMessageResult(
            role="assistant",
            content=TextContent(type="text", text=text),
            model=chosen_model,
            stopReason="endTurn",
        )

    # ------------------------------------------------------------------ #
    #  MCP primitive: ELICITATION  (client side)
    # ------------------------------------------------------------------ #
    async def _elicitation_callback(
        self,
        context: RequestContext[ClientSession, Any],
        params: ElicitRequestParams,
    ) -> ElicitResult | ErrorData:
        """Collect missing field values from the human reviewer."""
        requested_schema = params.requestedSchema or {}
        message = params.message or "Additional information is required."

        record: dict[str, Any] = {
            "message": message,
            "schema": requested_schema,
            "action": "decline",
            "content": None,
        }

        try:
            answers = await self.elicitation_responder(message, requested_schema)
        except ElicitationCancelled:
            record["action"] = "cancel"
            self.elicitation_calls.append(record)
            tracing.record_elicitation_event(
                trace_id=self.trace_id,
                schema_sent=requested_schema,
                reviewer_response=None,
                action="cancel",
            )
            return ElicitResult(action="cancel")
        except Exception as exc:  # noqa: BLE001
            log.warning("Elicitation responder failed: %s", exc)
            tracing.record_error(
                "mcp.elicitation.responder", exc, trace_id=self.trace_id,
                fallback_action="decline the elicitation",
            )
            self.elicitation_calls.append(record)
            return ElicitResult(action="decline")

        if answers is None:
            self.elicitation_calls.append(record)
            tracing.record_elicitation_event(
                trace_id=self.trace_id,
                schema_sent=requested_schema,
                reviewer_response=None,
                action="decline",
            )
            return ElicitResult(action="decline")

        record["action"] = "accept"
        record["content"] = answers
        self.elicitation_calls.append(record)
        tracing.record_elicitation_event(
            trace_id=self.trace_id,
            schema_sent=requested_schema,
            reviewer_response=answers,
            action="accept",
        )
        return ElicitResult(action="accept", content=answers)

    # ------------------------------------------------------------------ #
    #  Tools
    # ------------------------------------------------------------------ #
    async def _build_tool_index(self) -> None:
        for name, session in self.sessions.items():
            try:
                listing = await session.list_tools()
                for tool in listing.tools:
                    self._tool_index[tool.name] = name
            except Exception as exc:  # noqa: BLE001
                log.warning("list_tools failed on the %s server: %s", name, exc)

    async def list_tools(self) -> dict[str, list[dict[str, Any]]]:
        catalogue: dict[str, list[dict[str, Any]]] = {}
        for name, session in self.sessions.items():
            try:
                listing = await session.list_tools()
                catalogue[name] = [
                    {"name": tool.name, "description": (tool.description or "").strip()}
                    for tool in listing.tools
                ]
            except Exception as exc:  # noqa: BLE001
                catalogue[name] = [{"error": str(exc)}]
        return catalogue

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None,
        *, server: str | None = None,
    ) -> Any:
        """Invoke an MCP tool on whichever server provides it."""
        target = server or self._tool_index.get(tool_name) or self.PRIMARY
        session = self.sessions.get(target)
        if session is None:
            raise RuntimeError(
                f"Tool {tool_name!r} needs the {target!r} MCP server, which is not "
                f"connected (connected: {self.connected_servers or 'none'}). "
                f"Start it with: python -m discharge_ai.mcp_servers."
                f"{'primary_server' if target == self.PRIMARY else 'analytics_server'}"
            )

        with tracing.tool_span(
            tool_name, trace_id=self.trace_id, params=arguments or {}, mcp_server=target
        ) as span:
            result = await session.call_tool(tool_name, arguments or {})

            if getattr(result, "isError", False):
                detail = _content_to_text(result.content)
                span.update(level="ERROR", status_message=detail[:500])
                raise RuntimeError(f"MCP tool {tool_name!r} failed: {detail}")

            payload = _extract_result(result)
            span.set_output(payload)
            return payload

    # ------------------------------------------------------------------ #
    #  Resources
    # ------------------------------------------------------------------ #
    async def list_resources(self) -> list[dict[str, Any]]:
        session = self.sessions.get(self.PRIMARY)
        if session is None:
            return []
        out: list[dict[str, Any]] = []
        try:
            listing = await session.list_resources()
            out += [
                {"uri": str(resource.uri), "name": resource.name,
                 "mime_type": resource.mimeType, "templated": False}
                for resource in listing.resources
            ]
        except Exception as exc:  # noqa: BLE001
            log.warning("list_resources failed: %s", exc)
        try:
            templates = await session.list_resource_templates()
            out += [
                {"uri": template.uriTemplate, "name": template.name,
                 "mime_type": template.mimeType, "templated": True}
                for template in templates.resourceTemplates
            ]
        except Exception as exc:  # noqa: BLE001
            log.debug("list_resource_templates failed: %s", exc)
        return out

    async def read_resource(self, uri: str) -> str:
        """Read an MCP resource and return its text content."""
        session = self.sessions.get(self.PRIMARY)
        if session is None:
            raise RuntimeError("The primary MCP server is not connected")

        with tracing.tool_span(
            f"resource:{uri}", trace_id=self.trace_id, params={"uri": uri}
        ) as span:
            result = await session.read_resource(AnyUrl(uri))
            chunks = [
                content.text for content in result.contents
                if getattr(content, "text", None)
            ]
            text = "\n".join(chunks)
            span.set_output({"chars": len(text)})
            return text

    async def read_resource_json(self, uri: str, default: Any = None) -> Any:
        try:
            return json.loads(await self.read_resource(uri))
        except Exception as exc:  # noqa: BLE001
            log.warning("Resource %s is not valid JSON: %s", uri, exc)
            return default if default is not None else {}

    # ------------------------------------------------------------------ #
    #  Prompts
    # ------------------------------------------------------------------ #
    async def list_prompts(self) -> list[dict[str, Any]]:
        session = self.sessions.get(self.PRIMARY)
        if session is None:
            return []
        try:
            listing = await session.list_prompts()
            return [
                {
                    "name": prompt.name,
                    "description": prompt.description,
                    "arguments": [arg.name for arg in (prompt.arguments or [])],
                }
                for prompt in listing.prompts
            ]
        except Exception as exc:  # noqa: BLE001
            log.warning("list_prompts failed: %s", exc)
            return []

    async def get_prompt(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> str:
        """Fetch a prompt template through the MCP Prompts primitive.

        Agents must use this rather than hardcoding prompt strings.  Falls back
        to the local prompts.yaml only if the MCP server is unreachable, and
        says so in the log so the degraded path is visible.
        """
        session = self.sessions.get(self.PRIMARY)
        arguments = {k: str(v) for k, v in (arguments or {}).items()}

        if session is not None:
            try:
                result = await session.get_prompt(name, arguments)
                text = "\n\n".join(
                    message.content.text
                    for message in result.messages
                    if isinstance(message.content, TextContent)
                )
                if text.strip():
                    tracing.record_event(
                        "mcp.prompts.get_prompt",
                        trace_id=self.trace_id, prompt=name, arguments=arguments,
                    )
                    return text
            except Exception as exc:  # noqa: BLE001
                log.warning("get_prompt(%s) failed over MCP: %s", name, exc)

        from ..common.prompt_store import render_mcp_prompt

        log.warning(
            "Falling back to the local prompts.yaml copy of %r (MCP Prompts "
            "unavailable) — start the primary MCP server for the real path.", name,
        )
        return render_mcp_prompt(name, **arguments)


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _content_to_text(content: Any) -> str:
    if not content:
        return ""
    parts = [
        block.text for block in content if getattr(block, "text", None) is not None
    ]
    return "\n".join(parts)


def _extract_result(result: Any) -> Any:
    """Prefer MCP structured content; fall back to parsing the text blocks."""
    structured = getattr(result, "structuredContent", None)
    if structured:
        #  FastMCP wraps non-dict returns as {"result": ...}
        if isinstance(structured, dict) and set(structured.keys()) == {"result"}:
            return structured["result"]
        return structured

    text = _content_to_text(getattr(result, "content", None))
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


@asynccontextmanager
async def mcp_session(**kwargs: Any):
    """`async with mcp_session(trace_id=…) as client:` convenience wrapper."""
    client = ClinicalMCPClient(**kwargs)
    async with client:
        yield client
