"""
common/prompt_store.py
======================

Loader for `configs/prompts.yaml`.

The five templates under `mcp_prompts:` are what the Primary MCP Server exposes
through the MCP **Prompts** primitive.  Agents are expected to fetch them at
runtime with `get_prompt(...)` — this module is the *server-side* source, plus a
local fallback for the rare case where an agent cannot reach the MCP server
(logged as a degraded path so it is visible in the audit trail).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import yaml

from ..settings import settings

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_prompts() -> dict[str, Any]:
    with settings.prompts_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def mcp_prompt_names() -> list[str]:
    return list(load_prompts().get("mcp_prompts", {}).keys())


def mcp_prompt_spec(name: str) -> dict[str, Any]:
    """`{description, arguments, template}` for one MCP prompt."""
    spec = load_prompts().get("mcp_prompts", {}).get(name)
    if not spec:
        raise KeyError(
            f"Unknown MCP prompt {name!r}. Available: {', '.join(mcp_prompt_names())}"
        )
    return spec


def render_mcp_prompt(name: str, **arguments: Any) -> str:
    """Render an MCP prompt template with its declared arguments.

    Missing arguments render as "not specified" rather than raising, so a tool
    call never fails because an optional hint was omitted.
    """
    spec = mcp_prompt_spec(name)
    template: str = spec["template"]
    declared: list[str] = spec.get("arguments", []) or []

    values = {key: arguments.get(key, "not specified") for key in declared}
    values.update({k: v for k, v in arguments.items() if k not in values})

    try:
        return template.format(**values)
    except (KeyError, IndexError) as exc:
        log.warning("Prompt %s could not be fully rendered (%s)", name, exc)
        return template


def internal_prompt(name: str) -> str:
    """Template for a non-MCP internal prompt (judge / triad / toxicity)."""
    spec = load_prompts().get("internal_prompts", {}).get(name)
    if not spec:
        log.warning("Internal prompt %r not found in prompts.yaml", name)
        return ""
    return spec.get("template", "")
