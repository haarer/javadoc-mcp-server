# Javadoc MCP Server

MCP server that indexes Javadoc JARs into SQLite with hybrid BM25 + vector search.

## Quick start

```bash
# On host machine (needs torch wheels)
cd /workspace/javadoc-mcp-server
uv sync
uv run javadoc-mcp
```

Server starts on `0.0.0.0:8600` (override with env vars below).

Connect your MCP client to `http://host.containers.internal:8600`.

### GPU acceleration

Uses PyTorch — if CUDA is available, it auto-detects and runs embeddings on GPU.

## Environment variables

| Variable | Default | Description |
|---|---|---|---|
| `JAVADOC_MCP_HOST` | `0.0.0.0` | Bind address |
| `JAVADOC_MCP_PORT` | `8600` | Port |
| `JAVADOC_INDEX_DIR` | `./data` | Index storage directory |
| `JAVADOC_INDEX_PATH` | `data/javadoc.db` | SQLite DB path |
| `JAVADOC_JARS_DIR` | `data/jars` | Directory for uploaded JAR files |
| `JAVADOC_EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `JAVADOC_EMBED_BATCH` | `64` | Embedding batch size |

## MCP tools

| Tool | Description |
|---|---|---|
| `lookup_symbol(fqn)` | Exact FQN lookup. Returns name, kind, summary, description. |
| `search_docs(query, limit, jar_filter)` | Hybrid BM25 + vector search with reciprocal rank fusion. |
| `list_packages(jar_filter)` | List all indexed packages. |
| `list_classes(package, jar_filter)` | List classes/interfaces/enums in a package. |
| `add_jar(name, content)` | Upload and index a Javadoc JAR (content is base64-encoded). |
| `remove_jar(name)` | Remove a JAR and all its symbols by name. |
| `list_jars()` | List all indexed JARs with name, path, symbol count. |

## Hybrid search

1. BM25 via SQLite FTS5 on `fqn`, `name`, `package`, `summary`, `description`
2. Vector cosine similarity on 384-dim embeddings (MiniLM-L6)
3. Reciprocal rank fusion: `score = 1/(k+bm25_rank) + 1/(k+vec_rank)`, k=60
4. Returns top-N merged results

## Indexing

Each indexed symbol stores: FQN, kind, name, package, signature, summary, full description, HTML path, source JAR, and a 384-dim embedding blob.

## Architecture

```
mcp_server/
  server.py       # MCP app, 7 tool handlers, uvicorn HTTP
  indexer.py      # JAR extraction, HTML parsing, embedding pipeline
  embedder.py     # PyTorch embedding (transformers + torch)
  database.py     # SQLite schema, FTS5, vector queries
  parser.py       # BeautifulSoup Javadoc HTML -> structured data
  config.py       # Settings (port, index path, model, RRF k)
```

## Dependencies

- `mcp[streamable-http]` — MCP framework + HTTP transport
- `beautifulsoup4` — Javadoc HTML parsing
- `torch` + `transformers` — Model + tokenizer for embedding inference
- `numpy` — Vector math
- `sqlite3` — stdlib (FTS5 virtual table)
