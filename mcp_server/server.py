from __future__ import annotations
import asyncio
import base64
import hashlib
import logging
import os
import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.routing import Route
from starlette.responses import JSONResponse
from .config import HOST, PORT, RRF_K, JARS_DIR, EMBED_API_URL, EMBED_MODEL, EMBED_DIM, EMBED_BATCH_SIZE, INDEX_PATH, INDEX_DIR
from .database import Database
from .embedder import embed_single, cosine_similarity
from .indexer import Indexer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("javadoc-mcp")


def build_app() -> FastMCP:
    log.info(f"[build_app] Starting javadoc-mcp-server build")
    log.info(f"[build_app] Config: HOST={HOST}, PORT={PORT}, EMBED_API_URL={EMBED_API_URL}")
    log.info(f"[build_app] Config: EMBED_MODEL={EMBED_MODEL}, EMBED_DIM={EMBED_DIM}, EMBED_BATCH_SIZE={EMBED_BATCH_SIZE}")
    log.info(f"[build_app] Config: INDEX_PATH={INDEX_PATH}, JARS_DIR={JARS_DIR}, INDEX_DIR={INDEX_DIR}")
    log.info(f"[build_app] JARS_DIR exists: {os.path.isdir(JARS_DIR)}, contents: {os.listdir(JARS_DIR) if os.path.isdir(JARS_DIR) else 'N/A'}")

    mcp = FastMCP(
        "javadoc-mcp-server",
        host=HOST,
        port=PORT,
        streamable_http_path="/",
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=False
        ),
    )
    db = Database()
    indexer = Indexer(db)

    mcp._db = db
    mcp._indexer = indexer

    @mcp.tool()
    def lookup_symbol(fqn: str) -> str:
        """Look up documentation for a specific symbol by fully qualified name (e.g. com.example.MyClass)."""
        # Check if any jar is still indexing
        indexing_jars = db.get_indexing_jars()
        result = db.lookup_symbol(fqn)
        if not result:
            similar = db.fts_search(fqn.split(".")[-1], limit=3)
            if similar:
                suggestions = "\n".join(f"  - {s['fqn']} ({s['kind']})" for s in similar[:3])
                msg = f"Symbol '{fqn}' not found.\nDid you mean?\n{suggestions}"
            else:
                msg = f"Symbol '{fqn}' not found."
            if indexing_jars:
                msg += f"\n\nNote: {', '.join(indexing_jars)} is still indexing. Results may be incomplete."
            return msg

        lines = [
            f"Name: {result['name']}",
            f"Kind: {result['kind']}",
            f"FQN:  {result['fqn']}",
            f"Pkg:  {result['package']}",
        ]
        if result.get("signature"):
            lines.append(f"Sig:  {result['signature']}")
        if result.get("summary"):
            lines.append(f"\nSummary:\n{result['summary']}")
        if result.get("description"):
            desc = result["description"]
            if len(desc) > 2000:
                desc = desc[:2000] + "..."
            lines.append(f"\nDescription:\n{desc}")
        lines.append(f"\nSource: {result.get('jar_path', 'unknown')}")
        return "\n".join(lines)

    @mcp.tool()
    def search_docs(query: str, limit: int = 10, jar_filter: str | None = None) -> str:
        """Search indexed Javadoc by keyword or natural language query."""
        fts_results = db.fts_search(query, limit=limit * 3)
        fts_map = {r["fqn"]: (i, r) for i, r in enumerate(fts_results)}

        jar_id = None
        if jar_filter:
            jar_id = db.get_jar_id(jar_filter)

        q_emb = embed_single(query)
        vec_rows = db.vector_search(q_emb, limit=limit * 3, jar_id=jar_id)

        if vec_rows:
            emb_blobs = [r[8] for r in vec_rows]
            sims = cosine_similarity(q_emb, emb_blobs)
            vec_results = []
            for r, sim in zip(vec_rows, sims):
                vec_results.append({
                    "id": r[0], "fqn": r[1], "kind": r[2], "name": r[3],
                    "package": r[4], "summary": r[5], "description": r[6],
                    "jar_path": r[7], "similarity": sim
                })
            vec_sorted = sorted(vec_results, key=lambda x: -x["similarity"])
            vec_map = {r["fqn"]: (i, r) for i, r in enumerate(vec_sorted)}
        else:
            vec_map = {}

        all_fqns = set(fts_map.keys()) | set(vec_map.keys())
        fused = []
        for fqn in all_fqns:
            fts_rank = fts_map.get(fqn, (9999, None))[0]
            vec_rank = vec_map.get(fqn, (9999, None))[0]
            score = 1.0 / (RRF_K + fts_rank) + 1.0 / (RRF_K + vec_rank)
            merged = fts_map.get(fqn, (None, None))[1] or vec_map.get(fqn, (None, None))[1]
            if merged:
                fused.append((score, merged))

        fused.sort(key=lambda x: -x[0])
        top = fused[:limit]

        if not top:
            msg = f"No results for query: '{query}'"
            indexing_jars = db.get_indexing_jars()
            if indexing_jars:
                msg += f"\n\nNote: {', '.join(indexing_jars)} is still indexing. Results may be incomplete."
            return msg

        lines = [f"Top {len(top)} results for: '{query}'\n"]
        for i, (score, r) in enumerate(top, 1):
            kind_tag = r.get("kind", "?").upper()
            lines.append(f"{i}. [{kind_tag}] {r.get('fqn', 'unknown')}")
            summary = r.get("summary", "")
            if summary:
                summary = summary[:200].replace("\n", " ")
                lines.append(f"   {summary}")
            lines.append("")

        return "\n".join(lines)

    @mcp.tool()
    def list_packages(jar_filter: str | None = None) -> str:
        """List all packages in the indexed Javadoc. Optionally filter by jar name."""
        jar_id = None
        if jar_filter:
            jar_id = db.get_jar_id(jar_filter)
            if not jar_id:
                return f"Jar not found: {jar_filter}"
        packages = db.list_packages(jar_id)
        if not packages:
            msg = "No packages indexed. Use add_jar to index a Javadoc JAR."
            indexing_jars = db.get_indexing_jars()
            if indexing_jars:
                msg += f" Note: {', '.join(indexing_jars)} is still indexing."
            return msg
        return "\n".join(packages)

    @mcp.tool()
    def list_classes(package: str, jar_filter: str | None = None) -> str:
        """List all classes, interfaces, and enums in a package."""
        jar_id = None
        if jar_filter:
            jar_id = db.get_jar_id(jar_filter)
        classes = db.list_classes(package, jar_id)
        if not classes:
            msg = f"No classes found in package: {package}"
            indexing_jars = db.get_indexing_jars()
            if indexing_jars:
                msg += f" Note: {', '.join(indexing_jars)} is still indexing."
            return msg
        lines = [f"Classes in {package}:"]
        for c in classes:
            kind = c["kind"].upper()
            lines.append(f"  [{kind}] {c['name']}")
            if c.get("summary"):
                s = c["summary"][:120].replace("\n", " ")
                lines.append(f"    {s}")
        return "\n".join(lines)

    async def do_index_async(raw: bytes, name: str) -> str:
        log.info(f"[do_index] Starting indexing for name='{name}', size={len(raw)} bytes")
        os.makedirs(JARS_DIR, exist_ok=True)
        file_hash = hashlib.sha256(raw).hexdigest()
        jar_path = os.path.join(JARS_DIR, f"{file_hash}.jar")
        log.info(f"[do_index] file_hash={file_hash[:16]}..., jar_path={jar_path}")

        # Check if file already exists on disk
        if os.path.exists(jar_path):
            log.warning(f"[do_index] Jar file already exists on disk: {jar_path}")
            existing = db.get_jar_by_hash(file_hash)
            if existing:
                log.info(f"[do_index] Jar already indexed as '{existing['name']}' (id={existing['id']})")
                return f"Jar already indexed as '{existing['name']}' (hash: {file_hash[:12]}...)"
            # Stale file: exists on disk but not in DB -> auto-reindex
            log.warning(f"[do_index] Stale jar detected (file exists, no DB entry). Auto-reindexing as '{name}'.")
        else:
            log.info(f"[do_index] Writing {len(raw)} bytes to {jar_path}")
            with open(jar_path, "wb") as f:
                f.write(raw)

        log.info(f"[do_index] Starting background indexer.index_jar for {jar_path}")
        count, error = await asyncio.to_thread(indexer.index_jar, jar_path, jar_name=name, file_hash=file_hash)
        if error:
            log.error(f"[do_index] Indexing failed for '{name}': {error}. Cleaning up jar file and DB entry.")
            # Clean up: remove DB entry if it was created
            old_entry = db.get_jar_by_name(name)
            if old_entry:
                sym_count, _ = db.remove_jar(name)
                log.info(f"[do_index] Cleaned up {sym_count} orphan symbols from DB for '{name}'")
            # Clean up: remove jar file from disk
            if os.path.exists(jar_path):
                os.remove(jar_path)
                log.info(f"[do_index] Removed jar file from disk: {jar_path}")
            return f"Error indexing '{name}': {error}"
        log.info(f"[do_index] Successfully indexed {count} symbols from '{name}'")
        return f"Indexed {count} symbols from '{name}'"

    @mcp.tool()
    def add_jar(name: str) -> str:
        """Upload and index a Javadoc JAR file. For binary JAR upload, POST multipart/form-data to the server's /upload-jar endpoint (field: 'jar', optional field: 'name'). Example: curl -F jar=@file.jar -F name=myjar http://host:port/upload-jar. Returns symbol count. Verify success by calling list_jars afterward."""
        return "Use the /upload-jar HTTP endpoint to upload a JAR file. POST multipart/form-data with field 'jar' (binary file) and optional field 'name'."

    @mcp.tool()
    def remove_jar(name: str) -> str:
        """Remove a previously indexed JAR and all its symbols. Cannot remove jars that are currently indexing."""
        count, result = db.remove_jar(name)
        if count == 0:
            return result if result else f"Jar not found: {name}"
        if isinstance(result, str) and os.path.exists(result):
            os.remove(result)
        return f"Removed {count} symbols from '{name}'"

    @mcp.tool()
    def jar_status(name: str) -> str:
        """Get the indexing status of a JAR by name. Returns status (indexing/indexed/failed), symbol count, and error message if failed."""
        status = db.get_jar_status(name)
        if not status:
            return f"Jar not found: {name}"
        lines = [
            f"Name:      {status['name']}",
            f"Status:    {status['status']}",
            f"Symbols:   {status['symbols_indexed']}",
            f"Hash:      {status['file_hash'][:16]}...",
            f"Added:     {status['added_at']}",
        ]
        if status['status'] == 'indexing':
            lines.append("Indexing in progress...")
        elif status['status'] == 'failed' and status.get('error_message'):
            lines.append(f"Error:     {status['error_message']}")
        elif status['status'] == 'indexed':
            lines.append("Indexing complete.")
        return "\n".join(lines)

    @mcp.tool()
    def list_jars() -> str:
        """List all indexed JAR files with name, hash, status, and symbol count."""
        jars = db.list_jars()
        if not jars:
            return "No JARs indexed."
        lines = [f"Indexed JARs ({len(jars)}):\n"]
        for j in jars:
            lines.append(f"  {j['name']}")
            lines.append(f"    status:  {j.get('status', 'unknown')}")
            lines.append(f"    hash:    {j.get('file_hash', '')[:16]}")
            lines.append(f"    added:   {j['added_at']}")
            lines.append(f"    symbols: {j['symbol_count']}")
            if j.get('error_message'):
                lines.append(f"    error:   {j['error_message'][:100]}")
            lines.append("")
        return "\n".join(lines)

    async def upload_jar(request: Request) -> JSONResponse:
        log.info(f"[upload_jar] Received upload request")
        form = await request.form()
        jar_file = form.get("jar")
        name = form.get("name", "uploaded-jar")
        log.info(f"[upload_jar] Form fields: name={name}, jar_file={jar_file.filename if jar_file else None}")

        if not jar_file:
            log.error(f"[upload_jar] No 'jar' field in form data")
            return JSONResponse({"error": "No 'jar' field in form data. Use: curl -F jar=@file.jar http://host:port/upload-jar"}, status_code=400)

        raw = await jar_file.read()
        log.info(f"[upload_jar] Read {len(raw)} bytes from uploaded file")
        if not raw:
            log.error(f"[upload_jar] Empty file")
            return JSONResponse({"error": "Empty file"}, status_code=400)

        # Parse name: prefer form field, fallback to filename
        if not name or name.strip() == "":
            name = jar_file.filename or "uploaded-jar"
            name = os.path.splitext(name)[0]

        # Check if already indexed or already indexing
        existing = db.get_jar_by_name(name)
        if existing and existing.get("status") == "indexing":
            log.warning(f"[upload_jar] Jar '{name}' is already being indexed")
            return JSONResponse({"error": f"Jar '{name}' is already being indexed. Use jar_status to check progress."}, status_code=409)

        # Check for duplicate hash
        file_hash = hashlib.sha256(raw).hexdigest()
        jar_path = os.path.join(JARS_DIR, f"{file_hash}.jar")
        existing_by_hash = db.get_jar_by_hash(file_hash)
        if existing_by_hash and existing_by_hash.get("status") == "indexed":
            log.info(f"[upload_jar] Jar already indexed as '{existing_by_hash['name']}' (hash: {file_hash[:12]}...)")
            return JSONResponse({"error": f"Jar already indexed as '{existing_by_hash['name']}' (hash: {file_hash[:12]}...)"}, status_code=400)

        # Start background indexing
        asyncio.create_task(do_index_async(raw, name))
        log.info(f"[upload_jar] Started background indexing for '{name}'")

        return JSONResponse({
            "status": "indexing",
            "message": f"Indexing started for '{name}'. Use jar_status('{name}') to check progress."
        })

    async def upload_redirect(request: Request) -> JSONResponse:
        return JSONResponse({
            "message": "Upload a JAR file to POST /upload-jar",
            "example": "curl -F jar=@file.jar -F name=myjar http://host:port/upload-jar"
        })

    mcp_app = mcp.streamable_http_app()

    # Add upload routes directly to the MCP app's router
    mcp_app.routes.append(Route("/upload-jar", upload_jar, methods=["POST"]))
    mcp_app.routes.append(Route("/upload-jar", upload_redirect, methods=["GET"]))

    return mcp_app


def main():
    app = build_app()
    log.info(f"Javadoc MCP server on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info", proxy_headers=True, forwarded_allow_ips=["*"])


if __name__ == "__main__":
    main()
