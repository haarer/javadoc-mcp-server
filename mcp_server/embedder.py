from __future__ import annotations
import struct
import numpy as np
from .config import EMBED_MODEL, EMBED_BATCH_SIZE, EMBED_DIM

_model = None
_tokenizer = None


def _get_model():
    global _model, _tokenizer
    if _model is None:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer
        _model = ORTModelForFeatureExtraction.from_pretrained(EMBED_MODEL)
        _model.to("cpu")
        _model.eval()
        _tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    return _model, _tokenizer


def embed_batch(texts: list[str]) -> list[bytes]:
    """Embed a list of texts, return list of packed float32 bytes."""
    if not texts:
        return []
    model, tokenizer = _get_model()
    all_embeddings: list[bytes] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        import torch
        with torch.no_grad():
            out = model(**tokens)
        emb = out.last_hidden_state.mean(dim=1)
        norm = emb.norm(dim=1, keepdim=True).clamp(min=1e-9)
        emb = emb / norm
        for row in emb.cpu().numpy():
            packed = struct.pack(f"{EMBED_DIM}f", *row.astype(np.float32))
            all_embeddings.append(packed)
    return all_embeddings


def embed_single(text: str) -> bytes:
    """Embed one text string."""
    return embed_batch([text])[0]


def cosine_similarity(query_blob: bytes, db_blobs: list[bytes | None]) -> list[float]:
    """Compute cosine similarity between query and a list of DB embeddings."""
    q = np.frombuffer(query_blob, dtype=np.float32)
    scores = []
    for b in db_blobs:
        if not b or len(b) == 0:
            scores.append(0.0)
            continue
        if len(b) != EMBED_DIM * 4:
            scores.append(0.0)
            continue
        d = np.frombuffer(b, dtype=np.float32)
        scores.append(float(np.dot(q, d)))
    return scores
