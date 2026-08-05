"""
observability/tracing.py
========================

Thin, dependency-tolerant wrapper over LangFuse (v3/v4 OTEL client) that gives
the whole system the observability the specification requires:

* one end-to-end **trace id per discharge case**, propagated to every agent
  through A2A message metadata (`trace_id`) so spans from six different
  processes land on the same LangFuse trace;
* **per-agent spans** with latency + input/output payloads;
* **per-tool-call spans** for every MCP tool invocation;
* **LLM generation events** with model, prompt, response and token usage;
* dedicated helpers for **sampling**, **elicitation**, **guardrail** and
  **error** events.

Every helper is a no-op context manager when LangFuse is disabled or its keys
are missing, so nothing in the pipeline ever fails because tracing is off.

Usage
-----
    trace_id = start_case_trace("P1019")

    with agent_span("Clinical Extractor Agent", "langgraph",
                    trace_id=trace_id, input={"patient_id": "P1019"}) as span:
        with tool_span("clinical_data_harvester", trace_id=trace_id,
                       params={"patient_id": "P1019"}) as tool:
            ...
            tool.set_output({"fields": 22})
        span.set_output(case.model_dump())
"""

from __future__ import annotations

import contextlib
import logging
import time
import uuid
from typing import Any, Iterator

from ..settings import settings

log = logging.getLogger(__name__)

_client: Any | None = None
_client_initialised = False


# --------------------------------------------------------------------------- #
#  Client
# --------------------------------------------------------------------------- #
def _get_client() -> Any | None:
    """Lazily build the LangFuse client (None when tracing is disabled)."""
    global _client, _client_initialised

    if _client_initialised:
        return _client
    _client_initialised = True

    if not settings.langfuse_enabled:
        log.info("LangFuse tracing disabled (no keys or LANGFUSE_ENABLED=false).")
        return None

    try:
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_base_url,
            environment=settings.cfg.get("observability", {}).get(
                "environment", "development"
            ),
            release=settings.cfg.get("project", {}).get("version", "1.0.0"),
        )
        log.info("LangFuse tracing enabled → %s", settings.langfuse_base_url)
    except Exception as exc:  # noqa: BLE001
        log.warning("LangFuse unavailable (%s: %s) — tracing disabled.",
                    type(exc).__name__, exc)
        _client = None

    return _client


def langfuse_available() -> bool:
    return _get_client() is not None


def new_trace_id(seed: str | None = None) -> str:
    """A LangFuse-compatible (32-hex) trace id for one discharge case."""
    client = _get_client()
    if client is not None:
        try:
            return client.create_trace_id(seed=seed)
        except Exception:  # noqa: BLE001
            pass
    return uuid.uuid4().hex


def start_case_trace(patient_id: str, *, trace_id: str | None = None) -> str:
    """Deterministic trace id for a patient case, seeded so re-runs group."""
    return trace_id or new_trace_id(seed=f"discharge:{patient_id}")


def trace_url(trace_id: str | None) -> str | None:
    """Deep link into the LangFuse UI for a trace (None when disabled)."""
    client = _get_client()
    if client is None or not trace_id:
        return None
    try:
        return client.get_trace_url(trace_id=trace_id)
    except Exception:  # noqa: BLE001
        base = settings.langfuse_base_url.rstrip("/")
        return f"{base}/trace/{trace_id}"


def flush() -> None:
    """Push buffered spans; call before a short-lived process exits."""
    client = _get_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as exc:  # noqa: BLE001
        log.debug("LangFuse flush failed: %s", exc)


# --------------------------------------------------------------------------- #
#  Span handle
# --------------------------------------------------------------------------- #
class SpanHandle:
    """Uniform handle over a LangFuse observation (or nothing at all)."""

    __slots__ = ("_observation", "_trace_id", "_name", "_start")

    def __init__(self, observation: Any | None, trace_id: str | None, name: str) -> None:
        self._observation = observation
        self._trace_id = trace_id
        self._name = name
        self._start = time.perf_counter()

    # ---- properties -----------------------------------------------------
    @property
    def trace_id(self) -> str | None:
        return self._trace_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def duration_ms(self) -> int:
        return int((time.perf_counter() - self._start) * 1000)

    # ---- mutation -------------------------------------------------------
    def set_output(self, output: Any) -> None:
        self.update(output=_truncate(output))

    def set_metadata(self, **metadata: Any) -> None:
        self.update(metadata=metadata)

    def update(self, **kwargs: Any) -> None:
        if self._observation is None:
            return
        try:
            self._observation.update(**kwargs)
        except Exception as exc:  # noqa: BLE001
            log.debug("Span update failed for %s: %s", self._name, exc)

    def event(self, name: str, **metadata: Any) -> None:
        if self._observation is None:
            log.debug("[trace %s] event %s %s", self._trace_id, name, metadata)
            return
        try:
            self._observation.create_event(name=name, metadata=_truncate(metadata))
        except Exception as exc:  # noqa: BLE001
            log.debug("Span event failed for %s: %s", name, exc)

    def score(self, name: str, value: float, comment: str = "") -> None:
        if self._observation is None:
            return
        try:
            self._observation.score(name=name, value=value, comment=comment or None)
        except Exception as exc:  # noqa: BLE001
            log.debug("Span score failed for %s: %s", name, exc)

    def fail(self, exc: BaseException) -> None:
        """Record an error span: exception type, message, and level=ERROR."""
        self.update(
            level="ERROR",
            status_message=f"{type(exc).__name__}: {exc}",
            metadata={"exception_type": type(exc).__name__},
        )


# --------------------------------------------------------------------------- #
#  Span factories
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _observation(
    name: str,
    as_type: str,
    *,
    trace_id: str | None = None,
    input: Any = None,          # noqa: A002 — mirrors the LangFuse kwarg
    metadata: dict[str, Any] | None = None,
) -> Iterator[SpanHandle]:
    client = _get_client()
    observation_cm: Any = None

    if client is not None:
        kwargs: dict[str, Any] = {
            "name": name,
            "as_type": as_type,
            "input": _truncate(input),
            "metadata": _truncate(metadata or {}),
        }
        if trace_id:
            kwargs["trace_context"] = {"trace_id": trace_id}
        try:
            observation_cm = client.start_as_current_observation(**kwargs)
        except Exception as exc:  # noqa: BLE001 — tracing must never break a run
            log.debug("LangFuse span %r failed to start: %s", name, exc)
            observation_cm = None

    #  No backend (or it refused to start the span): still time the block, but
    #  yield exactly once so an exception inside the body propagates cleanly.
    if observation_cm is None:
        handle = SpanHandle(None, trace_id, name)
        started = time.perf_counter()
        try:
            yield handle
        finally:
            log.debug(
                "[%s] %s (%s) took %d ms",
                trace_id or "-", name, as_type,
                int((time.perf_counter() - started) * 1000),
            )
        return

    with observation_cm as observation:
        handle = SpanHandle(observation, trace_id, name)
        try:
            yield handle
        except Exception as exc:
            handle.fail(exc)
            raise


def agent_span(
    agent_name: str,
    framework: str = "",
    *,
    trace_id: str | None = None,
    input: Any = None,          # noqa: A002
    **metadata: Any,
):
    """Per-agent span (latency + input/output payload)."""
    return _observation(
        agent_name,
        "agent",
        trace_id=trace_id,
        input=input,
        metadata={"framework": framework, **metadata},
    )


def tool_span(
    tool_name: str,
    *,
    trace_id: str | None = None,
    params: Any = None,
    mcp_server: str = "primary",
    **metadata: Any,
):
    """Per-tool-call span for every MCP tool invocation."""
    return _observation(
        f"tool:{tool_name}",
        "tool",
        trace_id=trace_id,
        input=params,
        metadata={"mcp_server": mcp_server, "tool": tool_name, **metadata},
    )


def llm_generation(
    name: str,
    *,
    model: str,
    prompt: Any = None,
    trace_id: str | None = None,
    **metadata: Any,
):
    """LLM generation event (model, prompt, response, tokens, cost)."""
    return _observation(
        name,
        "generation",
        trace_id=trace_id,
        input=prompt,
        metadata={"model": model, "provider": settings.llm_provider, **metadata},
    )


def retriever_span(name: str, *, trace_id: str | None = None, query: Any = None,
                   **metadata: Any):
    """RAG retrieval span (query, top-k, chunk scores)."""
    return _observation(
        name, "retriever", trace_id=trace_id, input=query, metadata=metadata
    )


def guardrail_span(guardrail_name: str, *, trace_id: str | None = None,
                   content: Any = None, **metadata: Any):
    """Guardrail intervention span (check result, content blocked/allowed)."""
    return _observation(
        f"guardrail:{guardrail_name}",
        "guardrail",
        trace_id=trace_id,
        input=_redact_for_trace(content),
        metadata={"guardrail": guardrail_name, **metadata},
    )


def chain_span(name: str, *, trace_id: str | None = None, input: Any = None,  # noqa: A002
               **metadata: Any):
    """Generic pipeline-stage span."""
    return _observation(name, "chain", trace_id=trace_id, input=input, metadata=metadata)


# --------------------------------------------------------------------------- #
#  Fire-and-forget events
# --------------------------------------------------------------------------- #
def record_event(name: str, *, trace_id: str | None = None, **metadata: Any) -> None:
    """A point-in-time event (sampling request, elicitation outcome, …)."""
    client = _get_client()
    if client is None:
        log.debug("[%s] event %s %s", trace_id or "-", name, metadata)
        return
    try:
        kwargs: dict[str, Any] = {
            "name": name,
            "as_type": "span",
            "metadata": _truncate(metadata),
        }
        if trace_id:
            kwargs["trace_context"] = {"trace_id": trace_id}
        observation = client.start_observation(**kwargs)
        observation.end()
    except Exception as exc:  # noqa: BLE001
        log.debug("LangFuse event %r failed: %s", name, exc)


def record_error(
    where: str, exc: BaseException, *, trace_id: str | None = None,
    fallback_action: str = "", **metadata: Any,
) -> None:
    """Error span: exception type, message, and the fallback action taken."""
    import traceback

    record_event(
        f"error:{where}",
        trace_id=trace_id,
        exception_type=type(exc).__name__,
        exception_message=str(exc)[:1000],
        stack_trace="".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )[-3000:],
        fallback_action=fallback_action,
        level="ERROR",
        **metadata,
    )


def record_sampling_event(
    *, trace_id: str | None, server_preferences: Any, client_model: str,
    source_language: str, result_preview: str, confidence: float | None = None,
) -> None:
    """MCP Sampling event: server hints → client model → translation result."""
    record_event(
        "mcp.sampling.create_message",
        trace_id=trace_id,
        server_model_preferences=server_preferences,
        client_model_selected=client_model,
        provider=settings.llm_provider,
        source_language=source_language,
        translation_preview=str(result_preview)[:500],
        translation_confidence=confidence,
    )


def record_elicitation_event(
    *, trace_id: str | None, schema_sent: Any, reviewer_response: Any, action: str,
) -> None:
    """MCP Elicitation event: schema sent, reviewer response, action taken."""
    record_event(
        "mcp.elicitation.elicit",
        trace_id=trace_id,
        schema_sent=schema_sent,
        reviewer_response=_redact_for_trace(reviewer_response),
        action=action,
    )


# --------------------------------------------------------------------------- #
#  Payload hygiene
# --------------------------------------------------------------------------- #
_MAX_TRACE_CHARS = 12_000


def _truncate(value: Any) -> Any:
    """Keep trace payloads small enough to stay useful in the UI."""
    if value is None:
        return None
    if isinstance(value, str):
        return value if len(value) <= _MAX_TRACE_CHARS else value[:_MAX_TRACE_CHARS] + " …[truncated]"
    if isinstance(value, dict):
        return {k: _truncate(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_truncate(v) for v in value[:50]]
        if len(value) > 50:
            items.append(f"…[{len(value) - 50} more]")
        return items
    if hasattr(value, "model_dump"):
        try:
            return _truncate(value.model_dump())
        except Exception:  # noqa: BLE001
            return str(value)[:_MAX_TRACE_CHARS]
    return value


def _redact_for_trace(value: Any) -> Any:
    """PII/PHI must be masked before it reaches an external trace backend."""
    if value is None:
        return None
    try:
        from ..guardrails.pii import PIIRedactor

        return _truncate(PIIRedactor().redact_any(value))
    except Exception:  # noqa: BLE001
        return _truncate(value)
