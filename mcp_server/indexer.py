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

    def index_jar(self, jar_path: str) -> tuple[int, str | None]:
        jar_path = os.path.abspath(jar_path)
        if not zipfile.is_zipfile(jar_path):
            return 0, f"Not a valid zip/jar: {jar_path}"

        jar_id = self.db.add_jar(jar_path)
        if not jar_id:
            return 0, "Failed to register jar"

        tmpdir = tempfile.mkdtemp(prefix="javadoc_index_")
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

            log.info(f"Found {len(class_files)} class pages in {os.path.basename(jar_path)}")

            count = 0
            batch: list[tuple] = []
            batch_size = 256

            for html_file in class_files:
                rel = str(html_file).replace(tmpdir, "").lstrip("/")
                try:
                    content = html_file.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    log.warning(f"Failed to read {rel}: {e}")
                    continue

                symbols = parse_class_page(content, rel, jar_path)
                if not symbols:
                    continue

                texts = []
                sym_list = list(symbols)
                for sym in sym_list:
                    text = f"{sym.kind} {sym.fqn}"
                    if sym.summary:
                        text += f" {sym.summary}"
                    if sym.description:
                        text += f" {sym.description[:1000]}"
                    texts.append(text)

                embeddings = embed_batch(texts)

                for sym, emb in zip(sym_list, embeddings):
                    batch.append((
                        jar_id, sym.fqn, sym.kind, sym.name, sym.package,
                        sym.signature, sym.summary, sym.description,
                        sym.html_path, sym.source_jar, emb
                    ))

                if len(batch) >= batch_size:
                    self.db.insert_symbols_batch(jar_id, batch)
                    count += len(batch)
                    batch = []
                    if count % 500 == 0:
                        self.db.conn.commit()
                        log.info(f"Indexed {count} symbols from {os.path.basename(jar_path)}")

            if batch:
                self.db.insert_symbols_batch(jar_id, batch)
                count += len(batch)

            self.db.conn.commit()
            log.info(f"Done indexing {jar_path}: {count} symbols")
            return count, None

        except Exception as e:
            self.db.conn.rollback()
            log.error(f"Index error for {jar_path}: {e}")
            return 0, str(e)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
