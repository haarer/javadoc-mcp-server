from __future__ import annotations
import logging
import sqlite3
import struct
import os
import re
from typing import Any
from .config import INDEX_PATH, INDEX_DIR, JARS_DIR, EMBED_DIM

log = logging.getLogger("javadoc-mcp.database")


def _load_vec_extension(conn):
    try:
        import sqlite_vec
        sqlite_vec.load(conn)
        return True
    except Exception as e:
        log.warning(f"[database] sqlite-vec extension not available: {e}")
        return False


class Database:
    def __init__(self, db_path: str = INDEX_PATH):
        self.path = db_path
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        log.info(f"[database] Opening SQLite database at {db_path}")
        self.conn = sqlite3.connect(db_path, check_same_thread=False)

        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA cache_size=-64000")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute("PRAGMA mmap_size=268435456")
        self.conn.execute("PRAGMA busy_timeout=30000")

        self.has_vec = _load_vec_extension(self.conn)
        self._init_schema()
        db_size = os.path.getsize(db_path) if os.path.exists(db_path) else 0
        log.info(f"[database] Schema initialized, DB size={db_size} bytes, sqlite_vec={self.has_vec}")

    def _init_schema(self):
        self.conn.executescript(f"""
            CREATE TABLE IF NOT EXISTS jars (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                file_hash TEXT UNIQUE,
                status TEXT DEFAULT 'indexing',
                symbols_indexed INTEGER DEFAULT 0,
                error_message TEXT,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_jars_name ON jars(name);

            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                jar_id INTEGER REFERENCES jars(id) ON DELETE CASCADE,
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

            CREATE UNIQUE INDEX IF NOT EXISTS idx_symbols_fqn ON symbols(fqn);

            CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
                fqn, name, package, summary, description,
                content=symbols, content_rowid=id
            );

            CREATE TRIGGER IF NOT EXISTS symbols_ai AFTER INSERT ON symbols BEGIN
                INSERT INTO symbols_fts(rowid, fqn, name, package, summary, description)
                VALUES (new.id, new.fqn, new.name, new.package, new.summary, new.description);
            END;

            CREATE TRIGGER IF NOT EXISTS symbols_ad AFTER DELETE ON symbols BEGIN
                INSERT INTO symbols_fts(symbols_fts, rowid, fqn, name, package, summary, description)
                VALUES('delete', old.id, old.fqn, old.name, old.package, old.summary, old.description);
            END;

            CREATE TRIGGER IF NOT EXISTS symbols_au AFTER UPDATE ON symbols BEGIN
                INSERT INTO symbols_fts(symbols_fts, rowid, fqn, name, package, summary, description)
                VALUES('delete', old.id, old.fqn, old.name, old.package, old.summary, old.description);
                INSERT INTO symbols_fts(rowid, fqn, name, package, summary, description)
                VALUES (new.id, new.fqn, new.name, new.package, new.summary, new.description);
            END;
        """)
        for col in ["status", "symbols_indexed", "error_message"]:
            try:
                self.conn.execute(f"ALTER TABLE jars ADD COLUMN {col} TEXT")
            except sqlite3.OperationalError:
                pass
        # Ensure embedding column exists
        try:
            self.conn.execute("ALTER TABLE symbols ADD COLUMN embedding BLOB")
        except sqlite3.OperationalError:
            pass
        self.conn.execute("""
            UPDATE jars SET status = CASE WHEN symbols_indexed > 0 THEN 'indexed'
                                          WHEN status IS NULL THEN 'indexed'
                                          ELSE status END
            WHERE id IN (SELECT id FROM jars WHERE symbols_indexed > 0 OR (status IS NULL AND id IN (SELECT DISTINCT jar_id FROM symbols WHERE jar_id IS NOT NULL)))
        """)
        self.conn.commit()

    def begin_indexing(self, jar_id: int):
        log.info(f"[database] begin_indexing: jar_id={jar_id}")
        self.conn.execute(
            "UPDATE jars SET status = 'indexing', symbols_indexed = 0, error_message = NULL WHERE id = ?", (jar_id,)
        )
        self.conn.commit()

    def update_progress(self, jar_id: int, symbols_count: int, commit: bool = True):
        self.conn.execute(
            "UPDATE jars SET symbols_indexed = ? WHERE id = ?", (symbols_count, jar_id)
        )
        if commit:
            self.conn.commit()

    def finish_indexing(self, jar_id: int, total_symbols: int, error: str | None = None):
        status = "failed" if error else "indexed"
        log.info(f"[database] finish_indexing: jar_id={jar_id}, status={status}, symbols={total_symbols}")
        self.conn.execute(
            "UPDATE jars SET status = ?, symbols_indexed = ?, error_message = ? WHERE id = ?",
            (status, total_symbols, error, jar_id)
        )
        self.conn.commit()

    def get_jar_status(self, name: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            """SELECT j.id, j.name, j.path, j.file_hash, j.status, j.symbols_indexed, j.error_message, j.added_at,
                      (SELECT count(*) FROM symbols s WHERE s.jar_id = j.id) AS actual_symbol_count
               FROM jars j WHERE j.name = ?""",
            (name,)
        ).fetchone()
        if not row:
            return None
        cols = ["id", "name", "path", "file_hash", "status", "symbols_indexed", "error_message", "added_at", "actual_symbol_count"]
        return dict(zip(cols, row))

    def get_indexing_jars(self) -> list[str]:
        rows = self.conn.execute("SELECT name FROM jars WHERE status = 'indexing'").fetchall()
        return [r[0] for r in rows]

    def add_jar(self, name: str, path: str, file_hash: str) -> int:
        log.info(f"[database] add_jar: name={name}, path={path}, hash={file_hash}")
        existing = self.conn.execute(
            "SELECT id, file_hash FROM jars WHERE name = ?", (name,)
        ).fetchone()
        if existing:
            existing_id, existing_hash = existing
            if existing_hash == file_hash:
                log.info(f"[database] Jar '{name}' already exists with same hash, returning existing id={existing_id}")
                return existing_id
            log.warning(f"[database] Jar '{name}' exists with different hash. Replacing.")
            self._remove_jar_data(existing_id)
            self.conn.commit()
        self.conn.execute(
            "INSERT INTO jars (name, path, file_hash, status, symbols_indexed) VALUES (?, ?, ?, 'indexing', 0)", (name, path, file_hash)
        )
        self.conn.commit()
        row = self.conn.execute("SELECT id FROM jars WHERE name = ?", (name,)).fetchone()
        jar_id = row[0] if row else 0
        log.info(f"[database] add_jar returned jar_id={jar_id}")
        return jar_id

    def _remove_jar_data(self, jar_id: int):
        self.conn.execute("DELETE FROM symbols WHERE jar_id = ?", (jar_id,))
        self.conn.execute("DELETE FROM jars WHERE id = ?", (jar_id,))

    def get_jar_id(self, name: str) -> int | None:
        row = self.conn.execute("SELECT id FROM jars WHERE name = ?", (name,)).fetchone()
        return row[0] if row else None

    def get_jar_by_hash(self, file_hash: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT id, name, path, file_hash, status, added_at FROM jars WHERE file_hash = ?", (file_hash,)
        ).fetchone()
        if not row:
            return None
        cols = ["id", "name", "path", "file_hash", "status", "added_at"]
        return dict(zip(cols, row))

    def get_jar_by_name(self, name: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT id, name, path, file_hash, status, added_at FROM jars WHERE name = ?", (name,)
        ).fetchone()
        if not row:
            return None
        cols = ["id", "name", "path", "file_hash", "status", "added_at"]
        return dict(zip(cols, row))

    def insert_symbols_batch(self, jar_id: int, rows: list[tuple]):
        log.info(f"[database] insert_symbols_batch: jar_id={jar_id}, batch_size={len(rows)}")
        self.conn.executemany(
            """INSERT OR REPLACE INTO symbols
               (jar_id, fqn, kind, name, package, signature, summary, description, html_path, source_jar, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows
        )
        log.info(f"[database] insert_symbols_batch: inserted {len(rows)} symbols")

    def lookup_symbol(self, fqn: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            """SELECT s.fqn, s.kind, s.name, s.package, s.signature,
                      s.summary, s.description, s.html_path, j.path as jar_path
               FROM symbols s
               JOIN jars j ON s.jar_id = j.id
               WHERE s.fqn = ?""",
            (fqn,)
        ).fetchone()
        if not row:
            return None
        cols = ["fqn", "kind", "name", "package", "signature", "summary", "description", "html_path", "jar_path"]
        return dict(zip(cols, row))

    def fts_search(self, query: str, limit: int = 100) -> list[dict[str, Any]]:
        terms = re.findall(r'[\w]+', query.lower())
        if len(terms) > 1:
            fts_query = " OR ".join(terms)
        else:
            fts_query = query
        rows = self.conn.execute(
            """SELECT s.fqn, s.kind, s.name, s.package, s.summary, s.description,
                      j.path as jar_path, rank
               FROM symbols_fts
               JOIN symbols s ON symbols_fts.rowid = s.id
               JOIN jars j ON s.jar_id = j.id
               WHERE symbols_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (fts_query, limit)
        ).fetchall()
        cols = ["fqn", "kind", "name", "package", "summary", "description", "jar_path", "rank"]
        return [dict(zip(cols, r)) for r in rows]

    def vector_search(self, query_embedding: bytes, limit: int = 100, jar_id: int | None = None) -> list[dict[str, Any]]:
        """Search using sqlite-vec KNN if available, else fallback to Python cosine."""
        if self.has_vec:
            return self._vector_search_vec(query_embedding, limit, jar_id)
        return self._vector_search_python(query_embedding, limit, jar_id)

    def _vector_search_vec(self, query_embedding: bytes, limit: int, jar_id: int | None) -> list[dict[str, Any]]:
        emb = struct.unpack(f"{EMBED_DIM}f", query_embedding)
        emb_blob = struct.pack(f"{EMBED_DIM}f", *emb)

        rows = self.conn.execute(
            """SELECT s.id, s.fqn, s.kind, s.name, s.package, s.summary, s.description,
                      j.path as jar_path, v.distance
               FROM symbol_embeddings v
               JOIN symbols s ON v.rowid = s.id
               JOIN jars j ON s.jar_id = j.id
               WHERE v.embedding MATCH ?
               ORDER BY v.distance
               LIMIT ?""",
            (emb_blob, limit)
        ).fetchall()
        cols = ["id", "fqn", "kind", "name", "package", "summary", "description", "jar_path", "distance"]
        results = []
        for r in rows:
            d = dict(zip(cols, r))
            d["similarity"] = 1.0 - d["distance"]
            results.append(d)
        return results

    def _vector_search_python(self, query_embedding: bytes, limit: int, jar_id: int | None) -> list[dict[str, Any]]:
        import numpy as np
        q = np.frombuffer(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []

        base_sql = """SELECT s.id, s.fqn, s.kind, s.name, s.package, s.summary, s.description,
                      j.path as jar_path, s.embedding
               FROM symbols s
               JOIN jars j ON s.jar_id = j.id
               WHERE s.embedding IS NOT NULL AND length(s.embedding) > 0"""
        params: list[Any] = []
        if jar_id is not None:
            base_sql += " AND s.jar_id = ?"
            params.append(jar_id)

        all_rows = self.conn.execute(base_sql, params).fetchall()
        cols = ["id", "fqn", "kind", "name", "package", "summary", "description", "jar_path", "embedding"]

        scored = []
        for r in all_rows:
            d = dict(zip(cols, r))
            emb_bytes = d.pop("embedding")
            doc = np.frombuffer(emb_bytes, dtype=np.float32)
            d_norm = np.linalg.norm(doc)
            if d_norm == 0:
                continue
            sim = float(np.dot(q, doc) / (q_norm * d_norm))
            d["similarity"] = sim
            scored.append(d)

        scored.sort(key=lambda x: -x["similarity"])
        return scored[:limit]

    def list_packages(self, jar_id: int | None = None) -> list[str]:
        base_sql = """SELECT DISTINCT s.package
               FROM symbols s
               JOIN jars j ON s.jar_id = j.id
               WHERE j.status = 'indexed'"""
        params: list[Any] = []
        if jar_id is not None:
            base_sql += " AND s.jar_id = ?"
            params.append(jar_id)
        base_sql += " ORDER BY s.package"
        return [r[0] for r in self.conn.execute(base_sql, params).fetchall()]

    def list_classes(self, package: str, jar_id: int | None = None) -> list[dict[str, Any]]:
        base_sql = """SELECT s.name, s.kind, s.summary, s.fqn
               FROM symbols s
               JOIN jars j ON s.jar_id = j.id
               WHERE s.package = ? AND j.status = 'indexed' AND s.kind IN ('class', 'interface', 'enum')"""
        params: list[Any] = [package]
        if jar_id is not None:
            base_sql += " AND s.jar_id = ?"
            params.append(jar_id)
        base_sql += " ORDER BY s.name"
        rows = self.conn.execute(base_sql, params).fetchall()
        cols = ["name", "kind", "summary", "fqn"]
        return [dict(zip(cols, r)) for r in rows]

    def remove_jar(self, name: str) -> tuple[int, str]:
        jar = self.get_jar_by_name(name)
        if not jar:
            return 0, ""
        if jar.get("status") == "indexing":
            return 0, f"Jar '{name}' is currently being indexed. Wait for completion before removing."
        count = self._get_symbol_count(jar["id"])
        self._remove_jar_data(jar["id"])
        self.conn.commit()
        return count, jar.get("path", "")

    def _get_symbol_count(self, jar_id: int) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM symbols WHERE jar_id = ?", (jar_id,)).fetchone()
        return row[0] if row else 0

    def list_jars(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT j.id, j.name, j.path, j.file_hash, j.status, j.symbols_indexed,
                      j.error_message, j.added_at,
                      (SELECT count(*) FROM symbols s WHERE s.jar_id = j.id) AS symbol_count
               FROM jars j
               ORDER BY j.added_at DESC"""
        ).fetchall()
        cols = ["id", "name", "path", "file_hash", "status", "symbols_indexed", "error_message", "added_at", "symbol_count"]
        return [dict(zip(cols, r)) for r in rows]
