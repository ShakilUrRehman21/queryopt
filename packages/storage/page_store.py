"""
storage.page_store — File-based storage for table pages.

Each table's pages are stored as individual JSON files in a subdirectory:
  <db_dir>/pages/<table_name>/<page_id>.json

Design notes:
- One file per page for simplicity in Phase 1.
- This makes individual page loads cheap and avoids loading all data at once.
- The format is intentionally isolated here so it can be replaced with a
  binary format in a future phase without changing the rest of the engine.
- PageStore is intentionally dumb — it does not know about schemas.
  Type conversion is handled by the Table layer.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from common.errors import StorageError
from common.logging import get_logger
from common.types import Row
from storage.page import Page, PAGE_SIZE_ROWS

logger = get_logger(__name__)


class PageStore:
    """
    File-based persistent storage for pages.

    All page I/O goes through this class. The rest of the engine
    uses PageStore via the Table abstraction.
    """

    def __init__(self, db_dir: str | Path) -> None:
        self._db_dir = Path(db_dir)
        self._pages_dir = self._db_dir / "pages"
        self._pages_dir.mkdir(parents=True, exist_ok=True)

    def _table_dir(self, table_name: str) -> Path:
        d = self._pages_dir / table_name.lower()
        d.mkdir(exist_ok=True)
        return d

    def _page_path(self, table_name: str, page_id: str) -> Path:
        return self._table_dir(table_name) / f"{page_id}.json"

    # ── Read ─────────────────────────────────────────────────────────────────

    def load_page(self, table_name: str, page_id: str) -> Page:
        """Load a single page from disk by its ID."""
        path = self._page_path(table_name, page_id)
        if not path.exists():
            raise StorageError(
                message=f"Page '{page_id}' not found for table '{table_name}'.",
                context={"table": table_name, "page_id": page_id},
            )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return Page.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            raise StorageError(
                message=f"Page file is corrupt: {path}",
                context={"path": str(path)},
            ) from e

    def iter_pages(self, table_name: str, page_ids: list[str]) -> Iterator[Page]:
        """
        Iterate over all pages for a table in order.

        This is the primary access path for sequential scans (SeqScan).
        Pages are loaded one at a time — no full table is held in memory.
        """
        for page_id in page_ids:
            yield self.load_page(table_name, page_id)

    # ── Write ─────────────────────────────────────────────────────────────────

    def save_page(self, page: Page) -> None:
        """Persist a page to disk."""
        path = self._page_path(page.table_name, page.page_id)
        tmp_path = path.with_suffix(".tmp")
        try:
            tmp_path.write_text(json.dumps(page.to_dict()), encoding="utf-8")
            tmp_path.replace(path)
        except OSError as e:
            raise StorageError(
                message=f"Failed to write page {page.page_id}: {e}",
                context={"page_id": page.page_id, "table": page.table_name},
            ) from e

    def new_page(self, table_name: str) -> Page:
        """Allocate and persist a new empty page for a table."""
        page = Page.new(table_name)
        self.save_page(page)
        logger.debug(f"Allocated new page {page.page_id} for table '{table_name}'")
        return page

    def delete_table_pages(self, table_name: str, page_ids: list[str]) -> None:
        """Delete all page files for a table (used when dropping a table)."""
        for page_id in page_ids:
            path = self._page_path(table_name, page_id)
            if path.exists():
                path.unlink()
        # Remove directory if empty
        table_dir = self._table_dir(table_name)
        try:
            table_dir.rmdir()  # only succeeds if empty
        except OSError:
            pass  # not empty, or doesn't exist — ignore

    # ── Statistics Helpers ────────────────────────────────────────────────────

    def page_count(self, table_name: str, page_ids: list[str]) -> int:
        """Return the number of pages for a table."""
        return len(page_ids)
