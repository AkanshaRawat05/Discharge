"""
a2a_layer/auth.py
=================

Shared-secret authentication for the A2A surface.

The specification requires `X-Agent-Auth-Token` on **all** A2A servers.  This
module supplies both halves:

* `AgentAuthMiddleware`  — Starlette middleware that rejects unauthenticated
  JSON-RPC calls with 401.  The AgentCard endpoint stays public, because
  discovery must work before a caller can know what token to send.
* `auth_headers()`       — the header dict every A2A client sends.

The token comes from `A2A_AUTH_TOKEN` in the environment; if it is left at the
placeholder value the middleware logs a loud warning but still enforces it, so a
misconfigured deployment fails closed rather than silently open.
"""

from __future__ import annotations

import logging
import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from ..settings import settings

log = logging.getLogger(__name__)

#  Paths that must stay reachable without a token.
PUBLIC_PATHS = frozenset(
    {
        "/.well-known/agent.json",
        "/.well-known/agent-card.json",
        "/health",
        "/healthz",
    }
)

PLACEHOLDER_TOKENS = frozenset({"", "change-me-shared-secret", "changeme"})


def auth_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Headers for an outbound A2A request."""
    headers = {settings.a2a_auth_header: settings.a2a_auth_token}
    if extra:
        headers.update(extra)
    return headers


class AgentAuthMiddleware(BaseHTTPMiddleware):
    """Enforce the shared-secret header on every non-public A2A request."""

    def __init__(self, app, agent_name: str = "agent") -> None:  # noqa: ANN001
        super().__init__(app)
        self.agent_name = agent_name
        self.header_name = settings.a2a_auth_header
        self.expected = settings.a2a_auth_token

        if self.expected in PLACEHOLDER_TOKENS:
            log.warning(
                "%s is using the placeholder A2A_AUTH_TOKEN — set a real secret "
                "in .env before running outside a local demo.", agent_name,
            )

    async def dispatch(self, request: Request, call_next):  # noqa: ANN001, ANN201
        path = request.url.path.rstrip("/") or "/"

        if path in PUBLIC_PATHS or request.method == "OPTIONS":
            return await call_next(request)

        presented = request.headers.get(self.header_name, "")

        #  Constant-time comparison: this is a bearer secret.
        if not presented or not secrets.compare_digest(presented, self.expected):
            log.warning(
                "Rejected unauthenticated A2A request to %s %s (client=%s)",
                request.method, path,
                request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=401,
                content={
                    "error": "unauthorized",
                    "detail": (
                        f"Missing or invalid {self.header_name} header. "
                        "A2A calls require the shared secret from A2A_AUTH_TOKEN."
                    ),
                    "agent": self.agent_name,
                },
            )

        return await call_next(request)
