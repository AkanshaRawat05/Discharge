"""
a2a_layer/server.py
===================

`run_agent_server()` — turn an agent handler into a running A2A server.

Provides, for every agent:

* the AgentCard at `GET /.well-known/agent.json`
* the JSON-RPC endpoint at `POST /` (`message/send`, `message/stream`,
  `tasks/get`, `tasks/cancel`, `tasks/pushNotificationConfig/set`)
* shared-secret `X-Agent-Auth-Token` enforcement on every non-discovery route
* **push notifications** — an `InMemoryPushNotificationConfigStore` plus a
  `BasePushNotificationSender`, so a client can register a webhook and be told
  when a long-running discharge task finishes
* a plain `GET /health` probe used by the launcher and the dashboard
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import httpx
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import (
    BasePushNotificationSender,
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
)
from starlette.responses import JSONResponse
from starlette.routing import Route

from ..llm.provider import provider_banner
from ..observability import tracing
from ..settings import configure_logging, settings
from .auth import AgentAuthMiddleware
from .cards import build_agent_card
from .executor import HandlerAgentExecutor

log = logging.getLogger(__name__)


def build_app(
    agent_key: str,
    handler: Callable[..., Any],
    *,
    artifact_name: str = "result",
    extra_routes: list[Route] | None = None,
):
    """Build the Starlette app for one A2A agent (without running it)."""
    config = settings.agent(agent_key)
    card = build_agent_card(agent_key)
    streaming = bool(config.get("streaming"))

    executor = HandlerAgentExecutor(
        agent_name=config["name"],
        framework=config.get("framework", ""),
        handler=handler,
        streaming=streaming,
        artifact_name=artifact_name,
    )

    #  Push notifications: the client registers a webhook via
    #  tasks/pushNotificationConfig/set and the sender POSTs task updates to it.
    httpx_client = httpx.AsyncClient(timeout=30.0)
    push_config_store = InMemoryPushNotificationConfigStore()
    push_sender = BasePushNotificationSender(
        httpx_client=httpx_client, config_store=push_config_store
    )

    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        push_config_store=push_config_store,
        push_sender=push_sender,
    )

    def health(_request):  # noqa: ANN001, ANN202
        return JSONResponse(
            {
                "status": "ok",
                "agent": config["name"],
                "agent_key": agent_key,
                "framework": config.get("framework"),
                "port": config["port"],
                "streaming": streaming,
                "mcp_primitives": config.get("mcp_primitives", []),
                "llm": provider_banner(),
                "langfuse": tracing.langfuse_available(),
            }
        )

    routes: list[Route] = [Route("/health", health, methods=["GET"])]
    routes += extra_routes or []

    application = A2AStarletteApplication(agent_card=card, http_handler=request_handler)
    app = application.build(routes=routes)
    app.add_middleware(AgentAuthMiddleware, agent_name=config["name"])
    return app


def run_agent_server(
    agent_key: str,
    handler: Callable[..., Any],
    *,
    artifact_name: str = "result",
    extra_routes: list[Route] | None = None,
) -> None:
    """Serve one agent over A2A until interrupted."""
    config = settings.agent(agent_key)
    configure_logging(f"a2a-{agent_key}")

    app = build_app(
        agent_key, handler, artifact_name=artifact_name, extra_routes=extra_routes
    )

    host = "127.0.0.1"
    port = int(config["port"])
    log.info(
        "%s (%s) → http://%s:%d  [%s]  card: /.well-known/agent.json",
        config["name"], config.get("framework"), host, port,
        "STREAMING" if config.get("streaming") else "non-streaming",
    )
    log.info("  %s", provider_banner())
    log.info(
        "  auth header: %s | MCP primitives: %s",
        settings.a2a_auth_header, ", ".join(config.get("mcp_primitives", []) or ["-"]),
    )

    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    finally:
        tracing.flush()
