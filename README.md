# Javadoc MCP Server

MCP server that indexes Javadoc JARs into SQLite with hybrid BM25 + vector search.

## Quick start

```bash
cd /workspace/javadoc-mcp-server
uv sync
uv run javadoc-mcp
```

Server starts on `0.0.0.0:8600` (override with env vars below).

Connect your MCP client to `http://host.containers.internal:8600`.

## Embedding backend

The server uses an OpenAI-compatible `/v1/embeddings` API endpoint. No local model loading required — it calls the embedding service over HTTP.

Supported backends:
- **llama.cpp** — self-hosted, CPU/GPU inference (see below)
- **OpenAI** — `https://api.openai.com/v1/embeddings`
- **vLLM**, **Ollama**, **LiteLLM**, etc.

### Running llama.cpp server

```bash
/build/bin/llama-server \
  --hf-repo sabafallah/bge-large-en-v1.5-Q4_K_M-GGUF \
  --hf-file bge-large-en-v1.5-q4_k_m.gguf \
  --port 8000 \
  --embedding \
  -c 8192
```

The MCP server defaults to `bge-large-en-v1.5` (1024 dim) on `localhost:8000`, so you can start it directly:

```bash
uv run javadoc-mcp
```

Or override settings explicitly:

```bash
JAVADOC_EMBED_API_URL=http://localhost:8000/v1/embeddings \
JAVADOC_EMBED_MODEL=bge-large-en-v1.5 \
JAVADOC_EMBED_DIM=1024 \
uv run javadoc-mcp
```

### Using OpenAI API

```bash
JAVADOC_EMBED_API_URL=https://api.openai.com/v1/embeddings \
JAVADOC_EMBED_API_KEY=sk-... \
JAVADOC_EMBED_MODEL=text-embedding-3-small \
JAVADOC_EMBED_DIM=1536 \
uv run javadoc-mcp
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `JAVADOC_MCP_HOST` | `0.0.0.0` | Bind address |
| `JAVADOC_MCP_PORT` | `8600` | Port |
| `JAVADOC_INDEX_DIR` | `./data` | Index storage directory |
| `JAVADOC_INDEX_PATH` | `data/javadoc.db` | SQLite DB path |
| `JAVADOC_JARS_DIR` | `data/jars` | Directory for uploaded JAR files |
| `JAVADOC_EMBED_API_URL` | `http://localhost:8000/v1/embeddings` | Embedding API endpoint |
| `JAVADOC_EMBED_API_KEY` | *(none)* | Bearer token for API auth |
| `JAVADOC_EMBED_MODEL` | `bge-large-en-v1.5` | Model name sent to API |
| `JAVADOC_EMBED_DIM` | `1024` | Embedding vector dimension |
| `JAVADOC_EMBED_BATCH` | `64` | Embedding batch size |

## MCP tools

| Tool | Description |
|---|---|
| `lookup_symbol(fqn)` | Look up documentation by fully qualified name (e.g. com.example.MyClass). Returns name, kind, signature, summary, description. If not found, suggests similar FQNs. |
| `search_docs(query, limit, jar_filter)` | Search indexed Javadoc by keyword or natural language query. |
| `list_packages(jar_filter)` | List all packages. Optionally filter by jar name. |
| `list_classes(package, jar_filter)` | List classes, interfaces, and enums in a package. |
| `add_jar(name, content)` | Upload and index a Javadoc JAR. Content must be base64-encoded. File stored by SHA-256 hash; duplicate detection prevents re-indexing. |
| `remove_jar(name)` | Remove a previously indexed JAR and all its symbols. |
| `list_jars()` | List all indexed JARs with name, hash, and symbol count. |

## Hybrid search

1. BM25 via SQLite FTS5 on `fqn`, `name`, `package`, `summary`, `description`
2. Vector cosine similarity via dot product
3. Reciprocal rank fusion: `score = 1/(k+bm25_rank) + 1/(k+vec_rank)`, k=60
4. Returns top-N merged results

## Indexing

Each indexed symbol stores: FQN, kind, name, package, signature, summary, full description, HTML path, source JAR, and an embedding blob (dimension matches `JAVADOC_EMBED_DIM`).

JAR files are stored by SHA-256 hash of their content. The user-facing name is tracked in the `jars` table. This prevents duplicate uploads of the same file under different names.

## Architecture

```
mcp_server/
  server.py       # MCP app, 7 tool handlers, uvicorn HTTP
  indexer.py      # JAR extraction, HTML parsing, embedding pipeline
  embedder.py     # OpenAPI HTTP client for embeddings
  database.py     # SQLite schema, FTS5, vector queries
  parser.py       # BeautifulSoup Javadoc HTML -> structured data
  config.py       # Settings (port, index path, API URL, model, RRF k)
```

## Dependencies

- `mcp[streamable-http]` — MCP framework + HTTP transport
- `beautifulsoup4` — Javadoc HTML parsing
- `httpx` — HTTP client for embedding API calls
- `sqlite3` — stdlib (FTS5 virtual table)
