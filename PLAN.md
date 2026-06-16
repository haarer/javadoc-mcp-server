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
| Indexing | Async two-worker pipeline (parser + embedder) with bounded queue |
| Embed failure | Fail-fast, zero-byte fallback, no retry loop |
| MCP tools | lookup_symbol, search_docs, list_packages, list_classes, add_jar, remove_jar, list_jars, jar_status |

## Architecture

```
mcp_server/
  server.py          # MCP app, tool handlers, HTTP transport, /upload-jar endpoint
  indexer.py         # Async producer-consumer: parser worker + embedder worker
  embedder.py        # HTTP client for OpenAPI embedding (fail-fast, short timeouts)
  database.py        # SQLite schema, FTS5, vector queries
  parser.py          # BeautifulSoup Javadoc HTML -> structured data (all members)
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
  source_jar TEXT,
  embedding BLOB
);

CREATE VIRTUAL TABLE symbols_fts USING fts5(
  fqn, name, package, summary, description,
  content=symbols, content_rowid=id
);

CREATE UNIQUE INDEX idx_symbols_fqn ON symbols(fqn);
```

## Indexing flow

### Upload

1. `POST /upload-jar` — multipart form with `jar` (binary) and optional `name`
2. SHA-256 hash computed, JAR stored at `JARS_DIR/{hash}.jar`
3. Duplicate detection by hash (rejects same content) and name (replaces old JAR on re-upload)
4. DB row inserted with `status='indexing'`, `symbols_indexed=0`
5. Returns immediately with `status: "indexing"`

### Background pipeline

Two async workers run concurrently, connected by an `asyncio.Queue` (maxsize=8):

**Parser worker:**
1. Opens JAR via `zipfile.ZipFile`
2. Iterates over all HTML files (except `_`-prefixed)
3. For each: `parse_class_page(html, path, jar)` returns all symbols (class + methods + fields + constructors)
4. Accumulates batches of `EMBED_BATCH_SIZE`, pushes to queue
5. Sends empty-list sentinel when done

**Embedder worker:**
1. Pulls batches from queue
2. Builds embedding text: `"{kind} {name} {package} {fqn} {signature} {summary} {description}"`
3. Calls `embed_batch` via `asyncio.to_thread` — HTTP POST to embedding API
4. On success: stores embeddings; on failure: stores zero bytes (fail-fast)
5. Inserts all symbols + embeddings into `symbols` table in a single SQLite transaction
6. Updates progress in `jars` table after each batch
7. Yields to event loop with `asyncio.sleep(0)` after each batch

Both workers run in the same event loop. CPU-bound parsing runs in `asyncio.to_thread`. The event loop remains responsive for serving MCP requests throughout indexing.

### Completion

- On success: `finish_indexing(jar_id, total_count)`
- On failure (exception): `finish_indexing(jar_id, count, error_msg)`, JAR file removed

## Parser details

Javadoc 21 HTML structure:
- Class page: `class-declaration-page` body class
- Enum page: `enum-declaration-page` body class
- Interface page: `interface-declaration-page` body class
- Class name: `<h1>` text with prefix stripped
- Class signature: `<div class="type-signature">`
- Description: `<section class="class-description">` (or `<div class="description">`)
- Members: `#method-summary`, `#field-summary`, `#constructor-summary` sections
- Summary table: `<div class="summary-table">` with `div.col-second > a.member-name-link` rows
- Member names in FQN: `ParentClass#memberName` (methods/fields), `ParentClass#<init>` (constructors)
- Detail sections: `#method-detail`, `#field-detail`, `#constructor-detail` sections

Parse per-class-page, returning all members as individual symbols.

## MCP Tools

### `lookup_symbol(fqn: str)`
Exact match on `symbols.fqn`. Lookup supports both class FQN (`pkg.ClassName`) and member FQN (`pkg.ClassName#memberName`). Returns name, kind, package, signature, summary, description. If not found, suggests similar FQNs via FTS5.

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

### `add_jar(name: str)`
Upload a JAR via the `/upload-jar` HTTP endpoint. Returns `status: "indexing"`. The MCP tool itself returns instructions to use the HTTP endpoint (MCP cannot transport binary data inline). Same-name re-upload replaces old JAR. Duplicate hash rejected.

### `remove_jar(name: str)`
Remove a JAR and all its symbols by name. Deletes the stored file on disk. Cannot remove JARs currently indexing.

### `list_jars()`
List all indexed JARs with name, hash, added_at, symbol count, status, and error message.

### `jar_status(name: str)`
Check indexing status of a JAR. Returns `status` (`indexing`, `indexed`, `failed`), `symbols_indexed`, and `error_message`.

## Embedding pipeline

- External API: OpenAI-compatible `/v1/embeddings` endpoint
- Default model: `bge-large-en-v1.5` via llama.cpp (1024 dim)
- Embed at index time only, store as float32 BLOB
- Query time: embed user query via `embed_single`, compute cosine similarity
- Batch embed: up to `EMBED_BATCH_SIZE` (default 256) symbols per API call
- Timeout: 10s connect + 60s read per batch
- Fail-fast: any error returns zero-byte embeddings for the batch (no retries)
- Cosine similarity: numpy dot product

## Dependencies

```
mcp[streamable-http]    # MCP framework
beautifulsoup4          # HTML parsing
requests                # HTTP client for embedding API
numpy                   # Vector math for embeddings
sqlite3                 # stdlib, no install needed
```

## Performance targets

- Index 4000-file JAR (~80k symbols): < 120s (depends on embedding API throughput)
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

See README.md for current tool list and usage. Key implementation notes:

- **Full symbol indexing**: all methods, fields, constructors indexed (not just classes)
- **Background indexing**: async two-worker pipeline, server stays responsive
- **Progress tracking**: `jar_status` tool shows status and symbol count
- **Duplicate detection**: same hash rejected; same-name re-upload replaces old JAR
- **Embedding resilience**: fail-fast with zero-byte fallback, 60s timeouts
- **Pipeline backpressure**: bounded queue (maxsize=8) between parser and embedder
- **HTTP upload**: binary JAR upload via multipart form POST to `/upload-jar`
