"""LLM provider abstraction (Amazon Bedrock — Nova Lite by default)."""

from .provider import (  # noqa: F401
    complete,
    complete_json,
    get_adk_model,
    get_agno_model,
    get_chat_model,
    resolve_sampling_hint,
    stream_complete,
)
from .embeddings import embed_texts, embedding_dimension  # noqa: F401
