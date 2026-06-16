from __future__ import annotations
import zipfile
import tempfile
import os
import logging
import shutil
from pathlib import Path
from .parser import parse_class_page
from .embedder import embed_batch
from .database import Database

log = logging.getLogger(__name__)


class Indexer:
    def __init__(self, db: Database):
        self.db = db

    def index_jar(self, jar_path: str, jar_name: str | None = None, file_hash: str | None = None) -> tuple[int, str | None]:
        jar_path = os.path.abspath(jar_path)
        log.info(f"[indexer] index_jar called: path={jar_path}, name={jar_name}, hash={file_hash}")
        if not zipfile.is_zipfile(jar_path):
            log.error(f"[indexer] Not a valid zip/jar: {jar_path}")
            return 0, f"Not a valid zip/jar: {jar_path}"

        if jar_name is None:
            jar_name = os.path.splitext(os.path.basename(jar_path))[0]

        log.info(f"[indexer] Adding jar to DB: name={jar_name}, hash={file_hash}")
        jar_id = self.db.add_jar(jar_name, jar_path, file_hash or "")
        if not jar_id:
            log.error(f"[indexer] Failed to register jar in DB")
            return 0, "Failed to register jar"
        log.info(f"[indexer] Jar registered with jar_id={jar_id}")

        tmpdir = tempfile.mkdtemp(prefix="javadoc_index_")
        log.info(f"[indexer] Extracting jar to tmpdir={tmpdir}")
        try:
            with zipfile.ZipFile(jar_path, "r") as zf:
                zf.extractall(tmpdir)

            html_files = sorted(Path(tmpdir).rglob("*.html"))
            skip_names = {
                "package-summary.html", "package-tree.html", "package-use.html",
                "overview-summary.html", "overview-tree.html", "index.html",
                "help-doc.html", "deprecated-list.html", "constant-values.html",
                "allclasses-index.html", "allclasses-frame.html", "allclasses-noframe.html",
                "module-graph.html", "index-all.html", "search.html",
                "module-list", "element-list", "member-search-index.js",
                "type-search-index.js", "tag-search-index.js",
            }
            class_files = [
                f for f in html_files
                if f.name not in skip_names
                and "/script-dir/" not in str(f)
                and "/resources/" not in str(f)
                and "/legal/" not in str(f)
            ]

            log.info(f"[indexer] Found {len(class_files)} class pages in {os.path.basename(jar_path)} (total html={len(html_files)}, skipped={len(html_files)-len(class_files)})")

            count = 0
            batch: list[tuple] = []
            batch_size = 256
            total_files = len(class_files)
            self.db.begin_indexing(jar_id)

            for idx, html_file in enumerate(class_files):
                if (idx + 1) % 100 == 0 or idx == 0:
                    pct = (idx + 1) / total_files * 100
                    log.info(f"[indexer] Progress: {idx + 1}/{total_files} files ({pct:.0f}%), {count} symbols indexed so far")
                    self.db.update_progress(jar_id, count)

                rel = str(html_file).replace(tmpdir, "").lstrip("/")
                try:
                    content = html_file.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    log.warning(f"[indexer] Failed to read {rel}: {e}")
                    continue

                symbols = parse_class_page(content, rel, jar_path)
                if not symbols:
                    continue

                log.debug(f"[indexer] Parsed {len(list(symbols))} symbols from {rel}")

                symbols_list = list(symbols)
                texts = []
                for sym in symbols_list:
                    text = f"{sym.kind} {sym.fqn}"
                    if sym.summary:
                        text += f" {sym.summary[:100]}"
                    if sym.description:
                        text += f" {sym.description[:200]}"
                    texts.append(text[:500])

                log.info(f"[indexer] Embedding {len(texts)} texts for {rel}")
                embeddings = embed_batch(texts)

                for sym, emb in zip(symbols_list, embeddings):
                    batch.append((
                        jar_id, sym.fqn, sym.kind, sym.name, sym.package,
                        sym.signature, sym.summary, sym.description,
                        sym.html_path, sym.source_jar, emb
                    ))

                if len(batch) >= batch_size:
                    log.info(f"[indexer] Flushing batch of {len(batch)} symbols to DB")
                    self.db.insert_symbols_batch(jar_id, batch)
                    count += len(batch)
                    batch = []

            if batch:
                log.info(f"[indexer] Flushing final batch of {len(batch)} symbols to DB")
                self.db.insert_symbols_batch(jar_id, batch)
                count += len(batch)

            self.db.conn.commit()
            self.db.finish_indexing(jar_id, count)
            log.info(f"[indexer] Done indexing {jar_path}: {count} symbols total")
            return count, None

        except Exception as e:
            self.db.conn.rollback()
            log.error(f"[indexer] Index error for {jar_path}: {e}", exc_info=True)
            self.db.finish_indexing(jar_id, count, str(e))
            # Clean up partial DB state: remove jar entry and any symbols already flushed
            try:
                sym_count = self.db.conn.execute("SELECT count(*) FROM symbols WHERE jar_id = ?", (jar_id,)).fetchone()[0]
                if sym_count > 0:
                    log.warning(f"[indexer] Removing {sym_count} orphan symbols for jar_id={jar_id}")
                    self.db.conn.execute("DELETE FROM symbols WHERE jar_id = ?", (jar_id,))
                self.db.conn.execute("DELETE FROM jars WHERE id = ?", (jar_id,))
                self.db.conn.commit()
                log.info(f"[indexer] Cleaned up partial DB state for jar_id={jar_id}")
            except Exception as cleanup_err:
                log.error(f"[indexer] Cleanup failed for jar_id={jar_id}: {cleanup_err}")
            return 0, str(e)
        finally:
            log.info(f"[indexer] Cleaning up tmpdir={tmpdir}")
            shutil.rmtree(tmpdir, ignore_errors=True)
