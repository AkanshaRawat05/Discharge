
"""
llm/provider.py
===============

THE ONE FILE that knows which LLM vendor is in use.

The system runs entirely on **Amazon Bedrock** (Converse API), with Amazon Nova
Lite as the default model.  Credentials are never read from the source tree:
`boto3` resolves them through the standard AWS chain (environment variables ->
`aws configure` / shared credentials file -> AWS_PROFILE -> IAM role).

Changing model or region is a **configuration change only**:

    BEDROCK_MODEL_ID=amazon.nova-lite-v1:0
    BEDROCK_FALLBACK_MODEL_ID=cohere.command-r-plus-v1:0
    AWS_REGION=us-east-1

No agent, tool, guardrail or UI module needs to change: they all call
`get_chat_model()`, `complete()`, `get_adk_model()` or `get_agno_model()`.

Public API
----------
    get_chat_model(purpose=...)      -> LangChain BaseChatModel (LangGraph)
    complete(prompt, ...)            -> str                      (any caller)
    complete_json(prompt, ...)       -> dict                     (any caller)
    stream_complete(prompt, ...)     -> Iterator[str]             (streaming)
    get_adk_model(...)               -> LiteLlm model for the ADK agents
    get_agno_model(...)              -> Agno model object
    resolve_sampling_hint(hint)      -> concrete model id for an MCP hint
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from functools import lru_cache
from typing import Any, Iterator

from ..settings import settings

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Rate limiting  (Bedrock applies per-account requests-per-minute quotas)
# --------------------------------------------------------------------------- #
_rate_lock = threading.Lock()
_last_call_at = 0.0

_RATE_LIMIT_MARKERS = (
    "429", "resource_exhausted", "rate limit", "quota", "too many requests",
    "throttling", "throttled", "slow down",
)
_RETRY_DELAY_RE = re.compile(r"retry(?:_?delay|\s+in)\D{0,12}(\d+(?:\.\d+)?)\s*s")


def _throttle() -> None:
    """Space out calls when LLM_MIN_INTERVAL_SECONDS is configured."""
    interval = settings.llm_min_interval_seconds
    if interval <= 0:
        return
    global _last_call_at
    with _rate_lock:
        wait = interval - (time.monotonic() - _last_call_at)
        if wait > 0:
            log.debug("Throttling the LLM call for %.1fs", wait)
            time.sleep(wait)
        _last_call_at = time.monotonic()


#  A *daily* quota cannot be waited out inside one request, so retrying just
#  stalls the caller. Per-minute limits are worth retrying.
_DAILY_QUOTA_MARKERS = ("perday", "per day", "requestsperday")


def _is_rate_limited(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _RATE_LIMIT_MARKERS)


def _is_daily_quota(exc: BaseException) -> bool:
    text = f"{exc}".lower().replace("-", "").replace("_", "")
    return any(marker.replace(" ", "") in text for marker in _DAILY_QUOTA_MARKERS)


# --------------------------------------------------------------------------- #
#  Quota circuit breaker
# --------------------------------------------------------------------------- #
#  Once the provider reports a *daily* quota exhaustion there is no point in any
#  further LLM attempt in this process for a while — and the framework SDKs
#  (ADK, Agno) run their own internal retry loops we cannot configure, so each
#  further attempt costs ~30 s of pointless backoff. Tripping a breaker keeps the
#  pipeline responsive on its deterministic path.
_breaker: dict[str, float] = {"tripped_at": 0.0}


def mark_quota_exhausted(reason: str = "daily quota") -> None:
    if _breaker["tripped_at"] == 0.0:
        log.warning(
            "LLM circuit breaker OPEN (%s): skipping LLM calls for %.0f minutes. "
            "Deterministic paths continue to run.",
            reason, settings.llm_breaker_cooldown_seconds / 60.0,
        )
    _breaker["tripped_at"] = time.monotonic()


def is_quota_exhausted() -> bool:
    """True while the breaker is open (callers should skip the LLM)."""
    tripped_at = _breaker["tripped_at"]
    if tripped_at == 0.0:
        return False
    if time.monotonic() - tripped_at >= settings.llm_breaker_cooldown_seconds:
        log.info("LLM circuit breaker CLOSED — retrying LLM calls.")
        _breaker["tripped_at"] = 0.0
        return False
    return True


def reset_quota_breaker() -> None:
    _breaker["tripped_at"] = 0.0


def _suggested_delay(exc: BaseException, attempt: int) -> float:
    """Honour the server's `retryDelay` when it gives one, else back off."""
    match = _RETRY_DELAY_RE.search(str(exc).lower())
    if match:
        try:
            return min(float(match.group(1)) + 1.0, settings.llm_max_retry_wait)
        except ValueError:
            pass
    return min(4.0 * (2 ** attempt), settings.llm_max_retry_wait)


def _with_retries(operation: str, call: Any) -> Any:
    """Run `call()`, retrying rate-limit failures with backoff."""
    attempts = max(1, settings.llm_max_retries)
    last_exc: BaseException | None = None

    for attempt in range(attempts):
        _throttle()
        try:
            return call()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if not _is_rate_limited(exc) or attempt == attempts - 1:
                raise
            if _is_daily_quota(exc):
                log.warning(
                    "%s hit the provider's DAILY quota — not retrying. The run "
                    "continues on its deterministic path; raise the quota or set "
                    "OFFLINE_MODE=1 to silence this.", operation,
                )
                mark_quota_exhausted()
                raise
            delay = _suggested_delay(exc, attempt)
            log.warning(
                "%s hit a provider rate limit (attempt %d/%d) — retrying in %.0fs",
                operation, attempt + 1, attempts, delay,
            )
            time.sleep(delay)

    if last_exc is not None:
        raise last_exc
    return None


# --------------------------------------------------------------------------- #
#  Model purposes
# --------------------------------------------------------------------------- #
# Logical purposes -> which of the two configured models to prefer.
#   "reasoning" : extraction / validation / judging  (the stronger model)
#   "fast"      : translation, short rewrites, scoring
_PURPOSE_ALIASES = {"default": "reasoning", "": "reasoning"}


# --------------------------------------------------------------------------- #
#  Model id resolution
# --------------------------------------------------------------------------- #
def resolve_model_id(purpose: str = "reasoning") -> str:
    """Concrete provider-native model id for a logical purpose."""
    purpose = _PURPOSE_ALIASES.get(purpose, purpose)
    return (
        settings.bedrock_fallback_model_id
        if purpose == "fast"
        else settings.bedrock_model_id
    )


def resolve_sampling_hint(hint: str) -> str:
    """Map an MCP Sampling `ModelPreferences` hint onto a real model.

    The Primary MCP Server publishes vendor-neutral hints ("nova-lite" for
    multilingual work, "command-r-plus" for English).  The *client* owns LLM
    resource management, so it is the client that decides what those hints mean
    for the provider currently in use — see `configs/agent_config.yaml`
    (`sampling.hint_routing`).
    """
    routing = settings.sampling_cfg.get("hint_routing", {})
    provider_routing = routing.get(settings.llm_provider, {})
    if hint in provider_routing:
        return provider_routing[hint]
    # Unknown hint: multilingual-ish hints get the stronger model.
    return resolve_model_id("reasoning" if "nova" in hint.lower() else "fast")


# --------------------------------------------------------------------------- #
#  LangChain chat models  (used by every LangGraph agent)
# --------------------------------------------------------------------------- #
_chat_model_cache: dict[str, Any] = {}


def get_chat_model(purpose: str = "reasoning", *, model: str | None = None,
                   temperature: float | None = None, max_tokens: int | None = None):
    """Return a LangChain `BaseChatModel` for the ACTIVE provider."""
    model_id = model or resolve_model_id(purpose)
    temp = settings.llm_temperature if temperature is None else temperature
    tokens = settings.llm_max_tokens if max_tokens is None else max_tokens

    cache_key = f"{settings.llm_provider}:{model_id}:{temp}:{tokens}"
    if cache_key in _chat_model_cache:
        return _chat_model_cache[cache_key]

    client = _build_bedrock_chat_model(model_id, temp, tokens)

    _chat_model_cache[cache_key] = client
    return client


def _build_bedrock_chat_model(model_id: str, temperature: float, max_tokens: int):
    """ACTIVE provider — Amazon Bedrock via the unified Converse API.

    `ChatBedrockConverse` speaks the Converse API, so the same class serves
    Amazon Nova, Cohere Command R+, Anthropic, Llama, ...

    Credentials are resolved by boto3's default chain — environment variables,
    `aws configure` / the shared credentials file, AWS_PROFILE, or an IAM role.
    Nothing is read from, or hard-coded into, this repository.

    Model-access note: request access to "Amazon Nova Lite" and "Cohere
    Command R+" in the Bedrock console (Model access) first, and make sure the
    inference-profile prefix matches your region when you use one — e.g. `us.`
    for us-east-1, `eu.` for eu-central-1.
    """
    from langchain_aws import ChatBedrockConverse

    return ChatBedrockConverse(
        model_id=model_id,                       # e.g. "amazon.nova-lite-v1:0"
        region_name=settings.aws_region,
        temperature=temperature,
        max_tokens=max_tokens,
        # One retry policy, not two. The SDK's own backoff would nest inside
        # `_with_retries()` and stall a caller for minutes on an exhausted
        # quota, so botocore does a single attempt and this module owns retries.
        config=_boto_config(),
    )


@lru_cache(maxsize=1)
def _boto_config():
    """botocore Config shared by every Bedrock client built here."""
    from botocore.config import Config

    return Config(
        retries={"max_attempts": max(1, settings.llm_sdk_retries), "mode": "standard"},
    )


# --------------------------------------------------------------------------- #
#  ADK models  (Discharge Monitor :8103, Summary Generator :8104, Host)
# --------------------------------------------------------------------------- #
def get_adk_model(purpose: str = "reasoning"):
    """Model argument for `google.adk.agents.LlmAgent(model=...)`.

    ADK reaches Bedrock through LiteLLM, whose Bedrock route expects the
    "bedrock/<model-id>" prefix. LiteLLM/boto3 pick the credentials up from the
    standard AWS chain, so nothing has to be passed here.
    """
    model_id = resolve_model_id(purpose)

    from google.adk.models.lite_llm import LiteLlm

    return LiteLlm(
        model=f"bedrock/{model_id}",             # e.g. bedrock/amazon.nova-lite-v1:0
        aws_region_name=settings.aws_region,
    )


# --------------------------------------------------------------------------- #
#  Agno models  (Clinical RAG Q&A agent :8105)
# --------------------------------------------------------------------------- #
def get_agno_model(purpose: str = "reasoning"):
    """Model object for `agno.agent.Agent(model=...)`.

    Agno ships a first-class Bedrock integration; `AwsBedrock` is the generic
    Converse client and covers Nova Lite and Cohere Command R+ alike.
    """
    model_id = resolve_model_id(purpose)

    from agno.models.aws import AwsBedrock          # generic Converse client

    return AwsBedrock(id=model_id, region_name=settings.aws_region)


# --------------------------------------------------------------------------- #
#  Provider-agnostic convenience helpers
# --------------------------------------------------------------------------- #
def complete(prompt: str, *, purpose: str = "reasoning", system: str | None = None,
             temperature: float | None = None, max_tokens: int | None = None,
             model: str | None = None) -> str:
    """Single-turn completion. Returns plain text ("" on failure).

    `model` pins a specific provider-native model id — used by the MCP
    sampling callback, which must honour the server's `ModelPreferences` hints
    rather than the caller's logical purpose.
    """
    if settings.offline_mode:
        return _offline_stub(prompt)
    if is_quota_exhausted():
        return ""

    messages: list[tuple[str, str]] = []
    if system:
        messages.append(("system", system))
    messages.append(("human", prompt))

    try:
        client = get_chat_model(
            purpose, model=model, temperature=temperature, max_tokens=max_tokens
        )
        response = _with_retries("completion", lambda: client.invoke(messages))
        return _as_text(response)
    except Exception as exc:  # noqa: BLE001 — never let an LLM outage kill a run
        if _is_daily_quota(exc):
            mark_quota_exhausted()
        log.warning("LLM completion failed (%s): %s", type(exc).__name__, exc)
        return ""


def stream_complete(prompt: str, *, purpose: str = "reasoning",
                    system: str | None = None, model: str | None = None) -> Iterator[str]:
    """Yield text chunks. Falls back to a single chunk when streaming fails."""
    if settings.offline_mode:
        yield _offline_stub(prompt)
        return
    if is_quota_exhausted():
        return

    messages: list[tuple[str, str]] = []
    if system:
        messages.append(("system", system))
    messages.append(("human", prompt))

    try:
        client = get_chat_model(purpose, model=model)
        #  Retries wrap opening the stream; once tokens flow we let them flow.
        stream = _with_retries("streaming", lambda: iter(client.stream(messages)))
        for chunk in stream:
            text = _as_text(chunk)
            if text:
                yield text
    except Exception as exc:  # noqa: BLE001
        if _is_daily_quota(exc):
            mark_quota_exhausted()
        log.warning("LLM streaming failed (%s): %s", type(exc).__name__, exc)
        fallback = complete(prompt, purpose=purpose, system=system)
        if fallback:
            yield fallback


def complete_json(prompt: str, *, purpose: str = "reasoning",
                  system: str | None = None, default: Any = None) -> Any:
    """Completion parsed as JSON, tolerant of ```json fences and prose."""
    raw = complete(prompt, purpose=purpose, system=system)
    parsed = extract_json(raw)
    return parsed if parsed is not None else (default if default is not None else {})


# --------------------------------------------------------------------------- #
#  Async wrappers
# --------------------------------------------------------------------------- #
#  `complete()` is synchronous (LangChain's `invoke`, plus a rate-limit sleep of
#  up to a minute).  Calling it directly from an async agent handler would block
#  that agent's event loop — its A2A endpoint and /health probe would stop
#  answering for the duration.  Async callers must use these wrappers.
async def acomplete(prompt: str, **kwargs: Any) -> str:
    import asyncio

    return await asyncio.to_thread(lambda: complete(prompt, **kwargs))


async def acomplete_json(prompt: str, **kwargs: Any) -> Any:
    import asyncio

    return await asyncio.to_thread(lambda: complete_json(prompt, **kwargs))


async def arun_blocking(function: Any, *args: Any, **kwargs: Any) -> Any:
    """Run any blocking callable (guardrails, judges) off the event loop."""
    import asyncio

    return await asyncio.to_thread(lambda: function(*args, **kwargs))


def extract_json(text: str) -> Any | None:
    """Best-effort JSON extraction from an LLM response."""
    if not text:
        return None
    cleaned = text.strip()

    fenced = re.search(r"```(?:json)?\s*(.+?)```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fall back to the outermost {...} / [...] span.
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = cleaned.find(opener), cleaned.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


def _as_text(response: Any) -> str:
    """Normalise LangChain message content (str | list of parts) to text."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "".join(parts)
    return str(content or "")


def _offline_stub(prompt: str) -> str:
    """Deterministic stand-in used when OFFLINE_MODE=1.

    Keeps demos runnable with no network/quota: JSON-shaped prompts get an
    empty object back so callers fall through to their deterministic paths.
    """
    if "STRICT JSON" in prompt or "Return STRICT JSON" in prompt:
        return "{}"
    return "[OFFLINE_MODE] LLM generation disabled; deterministic output used."


def provider_banner() -> str:
    return (
        f"LLM provider = {settings.llm_provider} | "
        f"reasoning = {resolve_model_id('reasoning')} | "
        f"fast = {resolve_model_id('fast')} | "
        f"embeddings = {settings.embedding_provider} | "
        f"offline = {settings.offline_mode}"
    )
