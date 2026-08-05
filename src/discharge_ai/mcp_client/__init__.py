"""Multi-server MCP client with sampling / elicitation / roots callbacks."""

from .client import (  # noqa: F401
    ClinicalMCPClient,
    ElicitationResponder,
    dashboard_elicitation_responder,
    default_elicitation_responder,
    mcp_session,
)
