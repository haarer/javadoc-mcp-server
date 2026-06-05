from __future__ import annotations
import sqlite3
import os
import re
from typing import Any
from .config import INDEX_PATH, INDEX_DIR, JARS_DIR


class Database:
    def __init__(self, db_path: str = INDEX_PATH):
        self.path = db_path
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS jars (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                path TEXT NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

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
        self.conn.commit()

    def add_jar(self, name: str, path: str) -> int:
        self.conn.execute(
            "INSERT OR IGNORE INTO jars (name, path) VALUES (?, ?)", (name, path)
        )
        self.conn.commit()
        row = self.conn.execute("SELECT id FROM jars WHERE name = ?", (name,)).fetchone()
        return row[0] if row else 0

    def get_jar_id(self, name: str) -> int | None:
        row = self.conn.execute("SELECT id FROM jars WHERE name = ?", (name,)).fetchone()
        return row[0] if row else None

    def get_jar_by_name(self, name: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT id, name, path, added_at FROM jars WHERE name = ?", (name,)
        ).fetchone()
        if not row:
            return None
        cols = ["id", "name", "path", "added_at"]
        return dict(zip(cols, row))

    def insert_symbols_batch(self, jar_id: int, rows: list[tuple]):
        self.conn.executemany(
            """INSERT OR REPLACE INTO symbols
               (jar_id, fqn, kind, name, package, signature, summary, description, html_path, source_jar, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows
        )

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
        count = self.conn.execute("SELECT count(*) FROM symbols WHERE jar_id = ?", (jar_id,)).fetchone()[0]
        self.conn.execute("DELETE FROM symbols WHERE jar_id = ?", (jar_id,))
        self.conn.execute("DELETE FROM jars WHERE id = ?", (jar_id,))
        self.conn.commit()
        return count, jar["path"]

    def list_jars(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT j.id, j.name, j.path, j.added_at,
                      (SELECT count(*) FROM symbols s WHERE s.jar_id = j.id) AS symbol_count
               FROM jars j
               ORDER BY j.added_at DESC"""
        ).fetchall()
        return [
            {"id": r[0], "name": r[1], "path": r[2], "added_at": r[3], "symbol_count": r[4]}
            for r in rows
        ]

    def close(self):
        self.conn.close()
