from __future__ import annotations
import logging
import struct
import torch
import numpy as np
from .config import EMBED_MODEL, EMBED_BATCH_SIZE, EMBED_DIM

log = logging.getLogger("javadoc-mcp.embedder")

_model = None
_tokenizer = None
_device = None


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_model():
    global _model, _tokenizer, _device
    if _model is None:
        from transformers import AutoModel, AutoTokenizer

        _device = _detect_device()
        _model = AutoModel.from_pretrained(EMBED_MODEL)
        _model = _model.to(_device)
        _model.eval()

        if _device == "cuda":
            log.info(f"PyTorch CUDA detected — embeddings on GPU ({torch.cuda.get_device_name(0)})")
        elif _device == "mps":
            log.info("PyTorch MPS detected — embeddings on GPU")
        else:
            log.info("No GPU — embeddings on CPU")

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
        tokens = {k: v.to(_device) for k, v in tokens.items()}
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
