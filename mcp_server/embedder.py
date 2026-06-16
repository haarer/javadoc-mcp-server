from __future__ import annotations
import logging
import struct
import httpx
from .config import EMBED_API_URL, EMBED_API_KEY, EMBED_MODEL, EMBED_BATCH_SIZE, EMBED_DIM

log = logging.getLogger("javadoc-mcp.embedder")

_headers: dict[str, str] | None = None


def _get_headers() -> dict[str, str]:
    global _headers
    if _headers is None:
        _headers = {"Content-Type": "application/json"}
        if EMBED_API_KEY:
            _headers["Authorization"] = f"Bearer {EMBED_API_KEY}"
    return _headers


def _embed_batch_api(texts: list[str]) -> list[list[float]]:
    payload = {"model": EMBED_MODEL, "input": texts}
    resp = httpx.post(EMBED_API_URL, json=payload, headers=_get_headers(), timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return [e["embedding"] for e in data["data"]]


def embed_batch(texts: list[str]) -> list[bytes]:
    """Embed a list of texts via OpenAPI endpoint, return list of packed float32 bytes."""
    if not texts:
        return []
    all_embeddings: list[bytes] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        embs = _embed_batch_api(batch)
        for emb in embs:
            packed = struct.pack(f"{EMBED_DIM}f", *emb)
            all_embeddings.append(packed)
    return all_embeddings


def embed_single(text: str) -> bytes:
    """Embed one text string."""
    return embed_batch([text])[0]


def cosine_similarity(query_blob: bytes, db_blobs: list[bytes | None]) -> list[float]:
    """Compute cosine similarity between query and a list of DB embeddings."""
    q = struct.unpack(f"{EMBED_DIM}f", query_blob)
    scores = []
    for b in db_blobs:
        if not b or len(b) == 0:
            scores.append(0.0)
            continue
        if len(b) != EMBED_DIM * 4:
            scores.append(0.0)
            continue
        d = struct.unpack(f"{EMBED_DIM}f", b)
        scores.append(sum(qi * di for qi, di in zip(q, d)))
    return scores
