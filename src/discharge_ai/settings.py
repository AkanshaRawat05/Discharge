"""
settings.py
===========

The single place where configuration is resolved.  Everything else in the
codebase imports `settings` from here and never touches `os.environ` or reads
YAML directly.

    from discharge_ai.settings import settings

    settings.llm_provider          # "bedrock"
    settings.cfg["services"]       # parsed configs/agent_config.yaml
    settings.agent("extractor")    # A2A agent descriptor dict
    settings.path("reports_dir")   # absolute Path, created on demand
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

try:  # python-dotenv is in requirements, but never hard-fail on it
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_a, **_kw):  # type: ignore[misc]
        return False


# --------------------------------------------------------------------------- #
#  Repository root discovery
# --------------------------------------------------------------------------- #
#  This file lives at <root>/src/discharge_ai/settings.py
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


def _env(key: str, default: str = "") -> str:
    value = os.getenv(key, default)
    return value.strip() if isinstance(value, str) else value


def _env_bool(key: str, default: bool = False) -> bool:
    raw = _env(key, str(default)).lower()
    return raw in {"1", "true", "yes", "on"}


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(float(_env(key, str(default))))
    except ValueError:
        return default


class Settings:
    """Immutable-ish view over `.env` + `configs/agent_config.yaml`."""

    def __init__(self) -> None:
        self.root: Path = PROJECT_ROOT
        self.config_file: Path = PROJECT_ROOT / "configs" / "agent_config.yaml"
        self.cfg: dict[str, Any] = self._load_yaml(self.config_file)

        # ---------------- LLM ------------------------------------------------
        self.llm_provider: str = _env("LLM_PROVIDER", "bedrock").lower()
        self.offline_mode: bool = _env_bool("OFFLINE_MODE", False)

        #  AWS credentials are deliberately NOT read here: boto3 resolves them
        #  through its default chain (env vars -> shared credentials/config ->
        #  AWS_PROFILE -> IAM role), so nothing sensitive lives in this repo.
        self.aws_region: str = _env("AWS_REGION", "us-east-1")
        self.bedrock_model_id: str = _env("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")
        self.bedrock_fallback_model_id: str = _env(
            "BEDROCK_FALLBACK_MODEL_ID", "cohere.command-r-plus-v1:0"
        )
        self.bedrock_embedding_model_id: str = _env(
            "BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"
        )

        self.llm_temperature: float = _env_float("LLM_TEMPERATURE", 0.1)
        self.llm_max_tokens: int = _env_int("LLM_MAX_TOKENS", 2048)

        #  Bedrock enforces per-account requests-per-minute quotas, so the
        #  provider layer retries throttling errors and can optionally space
        #  calls out (set LLM_MIN_INTERVAL_SECONDS for a strict RPM budget).
        self.llm_max_retries: int = _env_int("LLM_MAX_RETRIES", 3)
        #  Retries inside the vendor SDK, which nest inside our own retry loop.
        #  Keep at 1 (a single attempt) so this project owns the retry policy.
        self.llm_sdk_retries: int = _env_int("LLM_SDK_RETRIES", 1)
        self.llm_max_retry_wait: float = _env_float("LLM_MAX_RETRY_WAIT", 62.0)
        self.llm_min_interval_seconds: float = _env_float("LLM_MIN_INTERVAL_SECONDS", 0.0)
        #  How long to stop attempting LLM calls after a daily-quota error.
        self.llm_breaker_cooldown_seconds: float = _env_float(
            "LLM_BREAKER_COOLDOWN_SECONDS", 900.0
        )

        # ---------------- embeddings -----------------------------------------
        #  The specification names sentence-transformers/all-MiniLM-L6-v2 as the
        #  embedding model, so that is the default; `bedrock` and `hashing`
        #  remain available for machines that cannot carry the PyTorch download.
        self.embedding_provider: str = _env(
            "EMBEDDING_PROVIDER", "sentence_transformers"
        ).lower()
        self.sentence_transformer_model: str = _env(
            "SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # ---------------- observability --------------------------------------
        self.langfuse_secret_key: str = _env("LANGFUSE_SECRET_KEY")
        self.langfuse_public_key: str = _env("LANGFUSE_PUBLIC_KEY")
        self.langfuse_base_url: str = _env("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
        self.langfuse_enabled: bool = (
            _env_bool("LANGFUSE_ENABLED", True)
            and bool(self.langfuse_secret_key)
            and bool(self.langfuse_public_key)
        )

        # ---------------- A2A -------------------------------------------------
        self.a2a_auth_token: str = _env("A2A_AUTH_TOKEN", "change-me-shared-secret")
        self.a2a_auth_header: str = self.cfg.get("a2a", {}).get(
            "auth_header", "X-Agent-Auth-Token"
        )
        self.a2a_timeout: int = int(
            self.cfg.get("a2a", {}).get("request_timeout_seconds", 180)
        )

        # ---------------- misc ------------------------------------------------
        self.log_level: str = _env("LOG_LEVEL", "INFO").upper()

    # ------------------------------------------------------------------ #
    #  helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Required config file is missing: {path}")
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    def path(self, key: str, *, create: bool = True) -> Path:
        """Absolute path for a `paths:` entry in agent_config.yaml.

        Directory-ish keys are created on first use so no service ever crashes
        on a missing output folder.
        """
        rel = self.cfg["paths"][key]
        resolved = (self.root / rel).resolve()
        if create and (key.endswith("_dir") or key.endswith("_root")):
            resolved.mkdir(parents=True, exist_ok=True)
        elif create and resolved.suffix and not resolved.exists():
            resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    def service(self, key: str) -> dict[str, Any]:
        return self.cfg["services"][key]

    def agent(self, key: str) -> dict[str, Any]:
        return self.cfg["a2a"]["agents"][key]

    def agent_url(self, key: str) -> str:
        return self.agent(key)["url"]

    @property
    def rules_path(self) -> Path:
        return self.path("rules_file", create=False)

    @property
    def prompts_path(self) -> Path:
        return self.path("prompts_file", create=False)

    @property
    def rag(self) -> dict[str, Any]:
        return self.cfg.get("rag", {})

    @property
    def sampling_cfg(self) -> dict[str, Any]:
        return self.cfg.get("sampling", {})

    @property
    def guardrail_cfg(self) -> dict[str, Any]:
        return self.cfg.get("guardrails", {})

    @property
    def ehr_base_url(self) -> str:
        return self.service("mock_ehr")["base_url"]

    @property
    def mcp_primary_url(self) -> str:
        return self.service("mcp_primary")["url"]

    @property
    def mcp_analytics_url(self) -> str:
        return self.service("mcp_analytics")["url"]

    def summary(self) -> dict[str, Any]:
        """Small dict used by the dashboard / CLI banners."""
        return {
            "llm_provider": self.llm_provider,
            "chat_model": self.bedrock_model_id,
            "embedding_provider": self.embedding_provider,
            "langfuse_enabled": self.langfuse_enabled,
            "offline_mode": self.offline_mode,
            "mcp_primary": self.mcp_primary_url,
            "mcp_analytics": self.mcp_analytics_url,
            "ehr": self.ehr_base_url,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings: Settings = get_settings()


# --------------------------------------------------------------------------- #
#  Logging — every service calls this once at start-up
# --------------------------------------------------------------------------- #
def configure_logging(service_name: str) -> None:
    import logging

    logging.basicConfig(
        level=getattr(logging, settings.log_level, 20),
        format=f"%(asctime)s | %(levelname)-7s | {service_name} | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # third-party chatter that drowns out our own logs
    for noisy in ("httpx", "httpcore", "urllib3", "mcp", "asyncio", "watchfiles"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
