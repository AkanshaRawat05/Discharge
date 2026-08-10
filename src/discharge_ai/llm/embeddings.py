"""
llm/embeddings.py
=================

Pluggable text embeddings for the FAISS clinical vector store.

Three providers, selected with `EMBEDDING_PROVIDER` in `.env`:

    sentence_transformers  the spec model all-MiniLM-L6-v2 (default; local, torch)
    bedrock                Amazon Titan Text Embeddings (no heavy deps)
    hashing                offline deterministic hashing embeddings, zero deps

Any provider that fails at runtime degrades to `hashing` so indexing and
retrieval keep working (a warning is logged once).  All vectors are
L2-normalised, so FAISS inner-product search == cosine similarity.
"""

from __future__ import annotations
import hashlib
import logging
import math
import re
import json
from typing import Sequence

from ..settings import settings

log = logging.getLogger(__name__)

_HASHING_DIM = 384          # same width as all-MiniLM-L6-v2, keeps indexes swappable
_BEDROCK_DIM = 1024         # amazon.titan-embed-text-v2:0 native width

_state: dict[str, object] = {"provider": None, "model": None, "dim": None, "warned": False}


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #
def embedding_dimension() -> int:
    """Vector width of the ACTIVE embedding provider."""
    if _state["dim"] is None:
        embed_texts(["dimension probe"])
    return int(_state["dim"] or _HASHING_DIM)


def active_provider() -> str:
    if _state["provider"] is None:
        embed_texts(["provider probe"])
    return str(_state["provider"])


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Embed a batch of texts with the configured provider (with fallback)."""
    if not texts:
        return []

    provider = settings.embedding_provider
    try:
        if provider == "sentence_transformers":
            vectors = _embed_sentence_transformers(texts)
        elif provider == "bedrock":
            vectors = _embed_bedrock(texts)
        elif provider == "hashing":
            vectors = _embed_hashing(texts)
        else:
            log.warning("Unknown EMBEDDING_PROVIDER=%r — using hashing", provider)
            vectors = _embed_hashing(texts)
    except Exception as exc:  # noqa: BLE001
        if not _state["warned"]:
            log.warning(
                "Embedding provider %r unavailable (%s: %s) — falling back to "
                "offline hashing embeddings.", provider, type(exc).__name__, exc,
            )
            _state["warned"] = True
        vectors = _embed_hashing(texts)

    return [_normalise(v) for v in vectors]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]


# --------------------------------------------------------------------------- #
#  Provider 1 — sentence-transformers (spec model)
# --------------------------------------------------------------------------- #
def _embed_sentence_transformers(texts: Sequence[str]) -> list[list[float]]:
    if _state["model"] is None or _state["provider"] != "sentence_transformers":
        from sentence_transformers import SentenceTransformer  # heavy import

        log.info("Loading %s …", settings.sentence_transformer_model)
        model = SentenceTransformer(settings.sentence_transformer_model)
        _state.update(
            provider="sentence_transformers",
            model=model,
            dim=model.get_sentence_embedding_dimension(),
        )

    model = _state["model"]
    return [list(map(float, v)) for v in model.encode(list(texts))]  # type: ignore[union-attr]


# --------------------------------------------------------------------------- #
#  Provider 2 — Amazon Bedrock Titan Embeddings
# --------------------------------------------------------------------------- #
def _embed_bedrock(texts: Sequence[str]) -> list[list[float]]:
    """
    Embed texts using Amazon Titan Text Embeddings V2.

    Credentials come from the boto3 default chain (env vars -> shared
    credentials/config -> AWS_PROFILE -> IAM role); none are passed here.
    """
    import boto3

    client = boto3.client(
        "bedrock-runtime",
        region_name=settings.aws_region,
    )

    model_id = settings.bedrock_embedding_model_id

    vectors: list[list[float]] = []

    # Titan embeds one document per call; batches here are small (a few hundred
    # chunks per index build) so serial calls are fine.
    for text in texts:
        body = json.dumps({
            "inputText": text[:8000]
        })

        response = client.invoke_model(
            modelId=model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )

        result = json.loads(response["body"].read())

        vectors.append(
            [float(x) for x in result["embedding"]]
        )

    _state.update(
        provider="bedrock",
        dim=len(vectors[0]) if vectors else _BEDROCK_DIM,
    )

    return vectors

# --------------------------------------------------------------------------- #
#  Provider 3 — offline hashing embeddings (always available)
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _embed_hashing(texts: Sequence[str]) -> list[list[float]]:
    """Deterministic bag-of-words hashing trick with sublinear TF weighting.

    Not semantic, but it retrieves keyword-overlapping clinical chunks well
    enough for the demo and needs no model download and no network.
    """
    _state.update(provider="hashing", dim=_HASHING_DIM)

    vectors: list[list[float]] = []
    for text in texts:
        vector = [0.0] * _HASHING_DIM
        tokens = _TOKEN_RE.findall(text.lower())
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % _HASHING_DIM
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        # sublinear scaling keeps long chunks from dominating
        vectors.append([math.copysign(math.log1p(abs(v)), v) for v in vector])
    return vectors


# --------------------------------------------------------------------------- #
def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]
