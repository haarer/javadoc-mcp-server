from __future__ import annotations
import asyncio
import logging
import os
import zipfile
from .config import EMBED_BATCH_SIZE, JARS_DIR
from .database import Database
from .embedder import embed_batch
from .parser import parse_class_page

log = logging.getLogger("javadoc-mcp.indexer")


class Indexer:
    def __init__(self, db: Database):
        self.db = db
        self.jar_paths = os.listdir(JARS_DIR) if os.path.isdir(JARS_DIR) else []

    async def index_jar(self, jar_path: str, jar_name: str = "uploaded-jar", file_hash: str = "") -> tuple[int, str | None]:
        log.info(f"[indexer] Starting index_jar: jar_path={jar_path}, name={jar_name}")
        jar_id = 0
        try:
            if not zipfile.is_zipfile(jar_path):
                return 0, "File is not a valid ZIP file"

            jar_id = self.db.add_jar(jar_name, jar_path, file_hash)
            self.db.begin_indexing(jar_id)
            log.info(f"[indexer] Added jar to DB with jar_id={jar_id}, starting indexing")

            queue: asyncio.Queue[list[dict]] = asyncio.Queue(maxsize=8)

            # Launch two concurrent workers
            parser_task = asyncio.create_task(self._parser_worker(jar_path, queue))
            embedder_task = asyncio.create_task(self._embedder_worker(jar_id, queue))

            try:
                await parser_task
            except BaseException:
                embedder_task.cancel()
                raise
            count, failed = await embedder_task

            log.info(f"[indexer] Inserted {count} symbols, {failed} failed")
            self.db.finish_indexing(jar_id, count)
            log.info(f"[indexer] Successfully indexed {count} symbols from '{jar_name}'")
            return count, None

        except Exception as e:
            log.error(f"[indexer] Failed to index jar: {e}")
            self.db.finish_indexing(jar_id, 0, str(e))
            return 0, str(e)

    async def _parser_worker(self, jar_path: str, queue: asyncio.Queue) -> None:
        """Scan JAR and parse HTML -> push symbol batches to queue."""
        skipped = 0
        parse_errors = 0
        parsed_symbols = 0
        with zipfile.ZipFile(jar_path, 'r') as zf:
            html_files = [n for n in zf.namelist() if n.endswith('.html') and not n.startswith('_')]
            total = len(html_files)
            batch: list[dict] = []
            for idx, name in enumerate(html_files, 1):
                try:
                    content = zf.read(name).decode('utf-8')
                    page_symbols = await asyncio.to_thread(parse_class_page, content, name, jar_path)
                    if not page_symbols:
                        skipped += 1
                    else:
                        parsed_symbols += len(page_symbols)
                        for sym in page_symbols:
                            batch.append({
                                'kind': sym.kind,
                                'fqn': sym.fqn,
                                'name': sym.name,
                                'package': sym.package,
                                'signature': sym.signature or '',
                                'summary': sym.summary or '',
                                'description': sym.description or '',
                                'html_path': sym.html_path or name,
                                'source_jar': sym.source_jar or jar_path,
                            })
                        if len(batch) >= EMBED_BATCH_SIZE:
                            await queue.put(batch)
                            batch = []
                except Exception as e:
                    parse_errors += 1
                    log.error(f"[parser] Failed to parse {name}: {e}")

                if idx % 100 == 0:
                    log.info(f"[parser] Parsed: {idx}/{total} ({parsed_symbols} symbols, {skipped} skipped)")
                    await asyncio.sleep(0)

            # Push remaining batch (empty list signals end)
            if batch:
                await queue.put(batch)
            await queue.put([])  # Sentinel: empty list means done
            log.info(f"[parser] Done: {total} html, {skipped} skipped, {parse_errors} errors")

    async def _embedder_worker(self, jar_id: int, queue: asyncio.Queue) -> tuple[int, int]:
        """Pull symbol batches from queue, embed, insert into DB."""
        count = 0
        failed = 0
        batch_num = 0
        while True:
            batch = await queue.get()
            if not batch:  # Sentinel
                break
            batch_num += 1
            pending_texts = []
            for sym in batch:
                text = f"{sym.get('kind', '')} {sym.get('name', '')} " \
                       f"{sym.get('package', '')} {sym.get('fqn', '')} " \
                       f"{sym.get('signature', '')} {sym.get('summary', '')} " \
                       f"{sym.get('description', '')}"
                pending_texts.append(text.strip())

            try:
                embeddings = await asyncio.wait_for(
                    asyncio.to_thread(embed_batch, pending_texts), timeout=90)
            except Exception as e:
                log.error(f"[embedder] Batch {batch_num} embed failed: {e}")
                failed += len(batch)
                embeddings = [b""] * len(batch)

            self.db.conn.execute("BEGIN")
            try:
                for sym, emb in zip(batch, embeddings):
                    embedding_bytes = emb if emb else b""
                    self.db.conn.execute(
                        """INSERT OR REPLACE INTO symbols
                           (jar_id, fqn, kind, name, package, signature, summary, description, html_path, source_jar, embedding)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (jar_id, sym.get('fqn', ''), sym.get('kind', ''), sym.get('name', ''),
                         sym.get('package', ''), sym.get('signature', ''), sym.get('summary', ''),
                         sym.get('description', ''), sym.get('html_path', ''), sym.get('source_jar', ''),
                         embedding_bytes)
                    )
                    count += 1
                self.db.conn.commit()
            except Exception as e:
                self.db.conn.rollback()
                log.error(f"[embedder] Batch {batch_num} DB insert failed, rolled back: {e}")
                failed += len(batch)

            self.db.update_progress(jar_id, count, commit=True)
            log.info(f"[embedder] Batch {batch_num}: {count} total symbols in DB")
            await asyncio.sleep(0)

        return count, failed
