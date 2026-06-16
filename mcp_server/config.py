from __future__ import annotations
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
INDEX_DIR = os.environ.get("JAVADOC_INDEX_DIR", str(PROJECT_DIR.parent / "data"))
INDEX_PATH = os.environ.get("JAVADOC_INDEX_PATH", os.path.join(INDEX_DIR, "javadoc.db"))
JARS_DIR = os.environ.get("JAVADOC_JARS_DIR", os.path.join(INDEX_DIR, "jars"))
HOST = os.environ.get("JAVADOC_MCP_HOST", "0.0.0.0")
PORT = int(os.environ.get("JAVADOC_MCP_PORT", "8600"))
EMBED_API_URL = os.environ.get("JAVADOC_EMBED_API_URL", "http://localhost:8000/v1/embeddings")
EMBED_API_KEY = os.environ.get("JAVADOC_EMBED_API_KEY", "")
EMBED_MODEL = os.environ.get("JAVADOC_EMBED_MODEL", "bge-large-en-v1.5")
EMBED_BATCH_SIZE = int(os.environ.get("JAVADOC_EMBED_BATCH", "8"))
EMBED_DIM = int(os.environ.get("JAVADOC_EMBED_DIM", "1024"))
RRF_K = 60
