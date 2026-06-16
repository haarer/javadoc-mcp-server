from __future__ import annotations
import logging
import sqlite3
import os
import re
from typing import Any
from .config import INDEX_PATH, INDEX_DIR, JARS_DIR

log = logging.getLogger("javadoc-mcp.database")


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
        self._init_schema()
        log.info(f"[database] Schema initialized, DB size={os.path.getsize(db_path) if os.path.exists(db_path) else 0} bytes")

    def _init_schema(self):
        self.conn.executescript("""
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
        # Migrate existing jars that don't have new columns
        try:
            self.conn.execute("ALTER TABLE jars ADD COLUMN status TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            self.conn.execute("ALTER TABLE jars ADD COLUMN symbols_indexed INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        try:
            self.conn.execute("ALTER TABLE jars ADD COLUMN error_message TEXT")
        except sqlite3.OperationalError:
            pass
        # Set status for existing jars based on symbol count
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

    def update_progress(self, jar_id: int, symbols_count: int):
        self.conn.execute(
            "UPDATE jars SET symbols_indexed = ? WHERE id = ?", (symbols_count, jar_id)
        )
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
            # Same name, different hash -> clean up old entry and symbols
            log.warning(f"[database] Jar '{name}' exists with different hash ({existing_hash[:16]} vs {file_hash[:16]}). Replacing.")
            self.conn.execute("DELETE FROM symbols WHERE jar_id = ?", (existing_id,))
            self.conn.execute("DELETE FROM jars WHERE id = ?", (existing_id,))
            self.conn.commit()
        self.conn.execute(
            "INSERT INTO jars (name, path, file_hash, status, symbols_indexed) VALUES (?, ?, ?, 'indexing', 0)", (name, path, file_hash)
        )
        self.conn.commit()
        row = self.conn.execute("SELECT id FROM jars WHERE name = ?", (name,)).fetchone()
        jar_id = row[0] if row else 0
        log.info(f"[database] add_jar returned jar_id={jar_id}")
        return jar_id

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
        rows = self.conn.execute(
            """SELECT s.id, s.fqn, s.kind, s.name, s.package, s.summary, s.description,
                      j.path as jar_path, s.embedding
               FROM symbols s
               JOIN jars j ON s.jar_id = j.id
               WHERE s.embedding IS NOT NULL AND length(s.embedding) > 0""",
        ).fetchall()
        if jar_id is not None:
            rows = [r for r in rows
                    if self.conn.execute(
                        "SELECT jar_id FROM symbols WHERE id=?", (r[0],)
                    ).fetchone()[0] == jar_id]
        return rows

    def list_packages(self, jar_id: int | None = None) -> list[str]:
        sql = "SELECT DISTINCT package FROM symbols WHERE kind IN ('class','interface','enum')"
        params: list = []
        if jar_id is not None:
            sql += " AND jar_id = ?"
            params.append(jar_id)
        sql += " ORDER BY package"
        rows = self.conn.execute(sql, params).fetchall()
        return [r[0] for r in rows]

    def list_classes(self, package: str, jar_id: int | None = None) -> list[dict[str, Any]]:
        sql = """SELECT fqn, kind, name, summary FROM symbols
                 WHERE package = ? AND kind IN ('class','interface','enum')"""
        params = [package]
        if jar_id is not None:
            sql += " AND jar_id = ?"
            params.append(jar_id)
        sql += " ORDER BY name"
        rows = self.conn.execute(sql, params).fetchall()
        return [{"fqn": r[0], "kind": r[1], "name": r[2], "summary": r[3]} for r in rows]

    def remove_jar(self, name: str) -> tuple[int, str | None]:
        jar = self.get_jar_by_name(name)
        if not jar:
            return 0, f"Jar not found: {name}"
        jar_id = jar["id"]
        if jar["status"] == "indexing":
            return 0, f"Cannot remove '{name}' while indexing is in progress"
        count = self.conn.execute("SELECT count(*) FROM symbols WHERE jar_id = ?", (jar_id,)).fetchone()[0]
        self.conn.execute("DELETE FROM symbols WHERE jar_id = ?", (jar_id,))
        self.conn.execute("DELETE FROM jars WHERE id = ?", (jar_id,))
        self.conn.commit()
        return count, jar["path"]

    def list_jars(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT j.id, j.name, j.path, j.file_hash, j.status, j.symbols_indexed, j.error_message, j.added_at,
                      (SELECT count(*) FROM symbols s WHERE s.jar_id = j.id) AS symbol_count
               FROM jars j
               ORDER BY j.added_at DESC"""
        ).fetchall()
        return [
            {"id": r[0], "name": r[1], "path": r[2], "file_hash": r[3], "status": r[4], "symbols_indexed": r[5], "error_message": r[6], "added_at": r[7], "symbol_count": r[8]}
            for r in rows
        ]

    def close(self):
        self.conn.close()
