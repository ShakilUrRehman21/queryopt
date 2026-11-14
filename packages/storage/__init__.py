"""
storage — Page-based storage layer for QueryOpt Lab.

Provides:
- Catalog: registry of databases, tables, and schemas.
- PageStore: file-based persistent storage of table data.
- TableHandle: operations on a specific table (insert, scan, statistics).

Design constraints:
- The storage layer has NO knowledge of SQL or query planning.
- It exposes a simple, typed Python API only.
- Persistence uses JSON-encoded pages on disk (Phase 1; can be replaced).
- Statistics collection is done lazily on demand, not automatically.
"""
