from __future__ import annotations
import logging
import numpy as np
import requests
from .config import EMBED_API_URL, EMBED_API_KEY, EMBED_MODEL

log = logging.getLogger("javadoc-mcp.embedder")


def embed_batch(texts: list[str]) -> list[bytes]:
    if not texts:
        return []

    try:
        res = requests.post(EMBED_API_URL, json={
            "model": EMBED_MODEL,
            "input": texts,
        }, headers=_headers(), timeout=(10, 60))
        if res.status_code == 200:
            return [np.array(d["embedding"], dtype=np.float32).tobytes() for d in res.json()["data"]]
        log.warning(f"[embedder] Batch embed failed ({res.status_code}): {res.text[:200]}")
    except Exception as e:
        log.warning(f"[embedder] Batch embed error: {e}")

    return [b""] * len(texts)


def embed_single(text: str) -> bytes:
    try:
        res = requests.post(EMBED_API_URL, json={
            "model": EMBED_MODEL,
            "input": text,
        }, headers=_headers(), timeout=(10, 60))
        res.raise_for_status()
        return np.array(res.json()["data"][0]["embedding"], dtype=np.float32).tobytes()
    except Exception as e:
        log.warning(f"[embedder] Single embed failed: {e}")
        return b""


def _headers():
    h = {"Content-Type": "application/json"}
    if EMBED_API_KEY:
        h["Authorization"] = f"Bearer {EMBED_API_KEY}"
    return h


def cosine_similarity(query: bytes, documents: list[bytes], dim: int = 1024) -> list[float]:
    q = np.frombuffer(query, dtype=np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return [0.0] * len(documents)
    scores = []
    for d_bytes in documents:
        d = np.frombuffer(d_bytes, dtype=np.float32)
        d_norm = np.linalg.norm(d)
        if d_norm == 0:
            scores.append(0.0)
        else:
            scores.append(float(np.dot(q, d) / (q_norm * d_norm)))
    return scores
