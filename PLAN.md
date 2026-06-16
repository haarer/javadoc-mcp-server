# Javadoc MCP Server — Plan

## What it does

Python MCP server (Streamable HTTP transport) that indexes Javadoc JARs into SQLite. Clients query via MCP tools: exact FQN lookup, hybrid BM25+vector search, package/class browsing, dynamic JAR management.

## Decisions

| Decision | Choice |
|---|---|
| Javadoc source | Javadoc JARs (pre-built HTML) |
| Multi-JAR | Dynamic add/remove at runtime |
| Transport | Streamable HTTP |
| Embeddings | OpenAPI-compatible `/v1/embeddings` endpoint (llama.cpp, OpenAI, vLLM, ...) |
| Storage | SQLite + FTS5 |
| JAR storage | SHA-256 hash as filename, user-facing name in DB |
| MCP tools | lookup_symbol, search_docs, list_packages, list_classes, add_jar, remove_jar, list_jars, jar_status |

## Architecture

```
mcp_server/
  server.py          # MCP app, tool handlers, HTTP transport
  indexer.py         # JAR extraction, HTML parsing, embedding pipeline
  embedder.py        # HTTP client for OpenAPI embedding endpoint
  database.py        # SQLite schema, FTS5, vector queries
  parser.py          # BeautifulSoup Javadoc HTML -> structured data
  config.py          # Settings (port, index path, API URL, model, dim, RRF k)
```

## Database schema

```sql
CREATE TABLE jars (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  path TEXT NOT NULL,
  file_hash TEXT UNIQUE,
  status TEXT DEFAULT 'indexed' CHECK(status IN ('indexing', 'indexed', 'failed')),
  symbols_indexed INTEGER DEFAULT 0,
  error_message TEXT,
  added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX idx_jars_name ON jars(name);

CREATE TABLE symbols (
  id INTEGER PRIMARY KEY,
  jar_id INTEGER REFERENCES jars(id),
  fqn TEXT NOT NULL,
  kind TEXT NOT NULL,
  name TEXT NOT NULL,
  package TEXT NOT NULL,
  signature TEXT,
  summary TEXT,
  description TEXT,
  html_path TEXT,
  embedding BLOB
);

CREATE VIRTUAL TABLE symbols_fts USING fts5(
  fqn, name, package, summary, description,
  content=symbols, content_rowid=id
);

CREATE UNIQUE INDEX idx_symbols_fqn ON symbols(fqn);
```

## Indexing flow

1. `add_jar(name, content)` — base64-decoded, SHA-256 hash computed
2. JAR saved to `JARS_DIR/{hash}.jar` (duplicate detection by hash)
3. DB row inserted with `status='indexing'`, `symbols_indexed=0`
4. `add_jar` returns immediately with `status: "indexing"`
5. Background task runs indexing:
   - Extract JAR to temp dir
   - For each `*.html` file:
     - Parse with BeautifulSoup
     - Extract: class/interface name, package, members (fields, constructors, methods)
     - Build FQN: `package.ClassName` for types, `package.ClassName#method(params)` for members
     - Strip HTML tags -> plain text summary + description (truncated to 500 chars max)
     - Generate embedding for `kind + fqn + summary + description` via OpenAPI endpoint
     - Insert into `symbols`, trigger FTS5 update
     - Every 100 files: `update_progress(jar_id, count)`
   - On success: `finish_indexing(jar_id, total_count)`
   - On failure: `finish_indexing(jar_id, count, error_msg)`
   - Cleanup temp dir

### Embedding pipeline notes
- Texts are truncated before embedding (summary≤100, desc≤200, total≤500 chars) to avoid API token limits
- Batch size: 8 (reduced from 64 to avoid 500 errors from embedding server)
- On 500 error: recursive split-half retry until batch size = 1
- Progress reported to DB every 100 files; final status on completion/failure

## Parser details

Javadoc 21 HTML structure:
- Class page: `class-declaration-page` body class
- Class name: `<div class="block">` after class signature
- Summary: `<div class="block">` in `<section class="class-description">`
- Members: `#method-summary` table (`<div class="summary-table three-column-summary">`) with 3-column grid rows
- Method anchors: `#method-name(params)` pattern in `<section class="detail">`
- Detail sections: `<section class="method-details" id="method-detail">` (same for field, constructor)

Parse per-symbol, not per-file. Each method/field/constructor gets own row.

## MCP Tools

### `lookup_symbol(fqn: str)`
Exact match on `symbols.fqn`. Returns name, kind, package, signature, summary, description. If not found, suggests similar FQNs via FTS5.

### `search_docs(query: str, limit: int = 10, jar_filter: str = None)`
Hybrid search:
1. BM25 via FTS5 on `fqn, name, package, summary, description`
2. Vector similarity: cosine distance on embedding column
3. Reciprocal rank fusion: `score = 1/(k+bm25_rank) + 1/(k+vector_rank)`, k=60
4. Return top-N with kind, FQN, summary

### `list_packages(jar_filter: str = None)`
Distinct packages. Optional JAR filter by name.

### `list_classes(package: str, jar_filter: str = None)`
Classes/interfaces/enums in package. Optional JAR filter.

### `add_jar(name: str, content: str)`
Upload a JAR by base64-encoded content. File stored as `{sha256}.jar`. Duplicate detection by content hash. Returns immediately with `status: "indexing"`. Same-name re-upload replaces old JAR.

### `remove_jar(name: str)`
Remove a JAR and all its symbols by name. Deletes the stored file on disk. Cannot remove JARs currently indexing.

### `list_jars()`
List all indexed JARs with name, hash, added_at, symbol count, status, and error message.

### `jar_status(name: str)`
Check indexing status of a JAR. Returns `status` (`indexing`, `indexed`, `failed`), `symbols_indexed`, and `error_message`.

## Embedding pipeline

- External API: OpenAI-compatible `/v1/embeddings` endpoint
- Default model: `bge-large-en-v1.5` via llama.cpp (1024 dim)
- Embed at index time only, store as `struct.pack` floats -> BLOB
- Query time: embed user query via same API, compute cosine similarity in Python
- Batch embed up to 8 symbols per API call (reduced from 64 to avoid token limit)
- Cosine similarity: pure Python dot product (no numpy dependency)
- Recursive split-half retry on 500 errors

## Dependencies

```
mcp[streamable-http]    # MCP framework
beautifulsoup4          # HTML parsing
httpx                   # HTTP client for embedding API
sqlite3                 # stdlib, no install needed
```

## Performance targets

- Index 4000-file JAR: < 120s (depends on embedding API throughput)
- Lookup symbol: < 50ms
- Search query: < 500ms (includes embedding API call)
- Memory: < 200MB during indexing

## Next steps

1. ~~Scaffold project, install deps~~ **DONE**
2. ~~Implement SQLite schema + database module~~ **DONE**
3. ~~Build Javadoc HTML parser~~ **DONE**
4. ~~Wire up OpenAPI embedder~~ **DONE**
5. ~~Implement indexer pipeline~~ **DONE**
6. ~~Create MCP tool handlers~~ **DONE**
7. ~~Test with MagicDraw JAR~~ **DONE**
8. Load test, tune

## Implemented features

- Background indexing: `add_jar` returns immediately, indexing runs in background
- Progress tracking: `jar_status` tool shows current status, symbol count, errors
- Graceful query responses: search/lookup show "indexing in progress" note during indexing
- Duplicate detection: same hash rejected; same-name re-upload replaces old JAR
- Embedding resilience: text truncation, reduced batch size (8), recursive split on 500
- Schema migration: safe ALTER TABLE adds `status`, `symbols_indexed`, `error_message` columns
- Silent embedder: no per-call logging, only errors/warnings
