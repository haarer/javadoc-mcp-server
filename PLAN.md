# Javadoc MCP Server — Plan

## What it does

Python MCP server (Streamable HTTP transport) that indexes Javadoc JARs into SQLite. Clients query via MCP tools: exact FQN lookup, hybrid BM25+vector search, package/class browsing, dynamic JAR management.

## Decisions

| Decision | Choice |
|---|---|---|
| Javadoc source | Javadoc JARs (pre-built HTML) |
| Multi-JAR | Dynamic add/remove at runtime |
| Transport | Streamable HTTP |
| Embeddings | PyTorch (`all-MiniLM-L6-v2`) |
| Storage | SQLite + FTS5 |
| MCP tools | lookup_symbol, search_docs, list_packages, list_classes, add_jar, remove_jar, list_jars |

## Architecture

```
mcp_server/
  server.py          # MCP app, tool handlers, HTTP transport
  indexer.py         # JAR extraction, HTML parsing, FQN index
  embedder.py        # PyTorch embedding wrapper
  database.py        # SQLite schema, FTS5, query helpers
  parser.py          # BeautifulSoup Javadoc HTML -> structured JSON
  config.py          # Settings (port, index path, jars dir, model, RRF k)
  requirements.txt
```

## Database schema

```sql
CREATE TABLE jars (
  id INTEGER PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,
  path TEXT NOT NULL,
  added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE symbols (
  id INTEGER PRIMARY KEY,
  jar_id INTEGER REFERENCES jars(id),
  fqn TEXT NOT NULL,           -- e.g. com.nomagic.uml2.finder.AbstractByQualifiedNameFinder
  kind TEXT NOT NULL,          -- class, interface, enum, method, field, constructor
  name TEXT NOT NULL,          -- short name
  package TEXT NOT NULL,
  signature TEXT,              -- method signature for disambiguation
  summary TEXT,                -- one-line description
  description TEXT,            -- full javadoc body (text)
  html_path TEXT,              -- path inside JAR
  embedding BLOB               -- 384-dim float array
);

CREATE VIRTUAL TABLE symbols_fts USING fts5(
  fqn, name, package, summary, description,
  content=symbols, content_rowid=id
);

CREATE UNIQUE INDEX idx_symbols_fqn ON symbols(fqn);
```

## Indexing flow

1. `add_jar(name, content)` — base64-decoded JAR saved to `JARS_DIR/{name}.jar`
2. Extract JAR to temp dir
3. Read `element-list` for package list
4. For each `*.html` file:
   - Parse with BeautifulSoup
   - Extract: class/interface name, package, members (fields, constructors, methods)
   - Build FQN: `package.ClassName` for types, `package.ClassName#method(params)` for members
   - Strip HTML tags -> plain text summary + description
   - Generate embedding for `summary + description`
   - Insert into `symbols`, trigger FTS5 update
5. Register JAR with name in `jars` table
6. Cleanup temp dir

## Parser details

Javadoc 21 HTML structure:
- Class page: `class-declaration-page` body class
- Class name: `<div class="block">` after class signature
- Summary: `<div class="block">` in `<section class="class-description">`
- Members: `#method-summary` table (`<div class="summary-table three-column-summary">`) with 3-column grid rows (col-first, col-second, col-last)
- Method anchors: `#method-name(params)` pattern in `<section class="detail">`
- Detail sections: `<section class="method-details" id="method-detail">` (same for field, constructor)

Parse per-symbol, not per-file. Each method/field/constructor gets own row.

## MCP Tools

### `lookup_symbol(fqn: str)`
Exact match on `symbols.fqn`. Returns structured JSON: name, kind, package, summary, description, source JAR.

### `search_docs(query: str, limit: int = 10, jar_filter: str = None)`
Hybrid search:
1. BM25 via FTS5 on `fqn, name, package, summary, description`
2. Vector similarity: cosine distance on embedding column
3. Reciprocal rank fusion: `score = 1/(k+bm25_rank) + 1/(k+vector_rank)`, k=60
4. Return top-N with kind, FQN, summary, snippet

### `list_packages(jar_filter: str = None)`
Distinct packages. Optional JAR filter.

### `list_classes(package: str, jar_filter: str = None)`
Classes/interfaces/enums in package.

### `add_jar(name: str, content: str)`
Upload a JAR by base64-encoded content, index it. Returns symbol count.

### `remove_jar(name: str)`
Remove a JAR and all its symbols by name. Deletes the stored file.

### `list_jars()`
List all indexed JARs with name, path, added_at, symbol count.

## Embedding pipeline

- Model: `sentence-transformers/all-MiniLM-L6-v2` via PyTorch
- Embed at index time only
- Store as `struct.pack` 384 floats -> BLOB
- Query time: embed user query, compute cosine similarity in SQL using dot product
- Batch embed 64 symbols per call for speed
- GPU: auto-detects CUDA or MPS via PyTorch, falls back to CPU

## Dependencies

```
mcp[streamable-http]    # MCP framework
beautifulsoup4          # HTML parsing
torch + transformers     # Local embeddings
sqlite3                 # stdlib, no install needed
```

## Performance targets

- Index 4000-file JAR: < 60s
- Lookup symbol: < 50ms
- Search query: < 200ms
- Memory: < 500MB during indexing

## Next steps

1. Scaffold project, install deps
2. Implement SQLite schema + database module
3. Build Javadoc HTML parser
4. Wire up embedder
5. Implement indexer pipeline
6. Create MCP tool handlers
7. Test with MagicDraw JAR
8. Load test, tune
