"""
optimizer.feedback — Persistent cardinality feedback (Phase 6, RQ4).

After every successful SELECT the engine hands the executed physical plan to
`CardinalityFeedbackStore.record_execution`. For each trustworthy operator it
stores the *observed* row count under the operator's order-independent
signature (see `optimizer.signature`). The adaptive optimizer later consults the
store before falling back to modelled estimates.

The store is a single JSON file next to the database catalog so feedback
survives API restarts, is inspectable by hand, and can be wiped to reset an
experiment (`reset()`).
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from common.logging import get_logger
from optimizer.signature import RelSignature, physical_observations

if TYPE_CHECKING:
    from executor.base import PhysicalOperator

logger = get_logger(__name__)

FEEDBACK_FILENAME = "cardinality_feedback.json"
_LOCK = threading.Lock()


@dataclass
class FeedbackEntry:
    tables: list[str]
    predicates: list[str]
    actual_rows: int
    last_estimate: float
    observations: int = 1
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    @property
    def q_error(self) -> float:
        est = max(self.last_estimate, 1.0)
        act = max(float(self.actual_rows), 1.0)
        return max(est / act, act / est)


class CardinalityFeedbackStore:
    """Signature → observed cardinality map, persisted as JSON."""

    def __init__(self, db_dir: Path | str) -> None:
        self.path = Path(db_dir) / FEEDBACK_FILENAME
        self._entries: dict[str, FeedbackEntry] = {}
        self._mtime: float | None = None
        self._load()

    # ── persistence ───────────────────────────────────────────────────────
    def _load(self) -> None:
        if not self.path.exists():
            self._entries = {}
            self._mtime = None
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._entries = {k: FeedbackEntry(**v) for k, v in raw.get("entries", {}).items()}
            self._mtime = self.path.stat().st_mtime
        except Exception as exc:  # corrupt file → start fresh, never crash a query
            logger.warning(f"Ignoring unreadable feedback store {self.path}: {exc}")
            self._entries = {}

    def refresh(self) -> None:
        """Reload if another process/request has written the file since."""
        try:
            mtime = self.path.stat().st_mtime if self.path.exists() else None
        except OSError:
            mtime = None
        if mtime != self._mtime:
            self._load()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "entries": {k: asdict(v) for k, v in self._entries.items()},
        }
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".feedback-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)
        os.replace(tmp, self.path)
        self._mtime = self.path.stat().st_mtime

    # ── API ───────────────────────────────────────────────────────────────
    def lookup(self, sig: RelSignature | None) -> FeedbackEntry | None:
        if sig is None:
            return None
        return self._entries.get(sig.key)

    def record(self, sig: RelSignature, estimated: float, actual: int) -> None:
        now = time.time()
        entry = self._entries.get(sig.key)
        if entry is None:
            self._entries[sig.key] = FeedbackEntry(
                tables=sorted(sig.tables),
                predicates=sorted(sig.predicates),
                actual_rows=int(actual),
                last_estimate=float(estimated),
            )
        else:
            entry.actual_rows = int(actual)
            entry.last_estimate = float(estimated)
            entry.observations += 1
            entry.last_seen = now

    def record_execution(self, root: PhysicalOperator) -> int:
        """Record every trustworthy operator of an executed plan. Returns count."""
        obs = physical_observations(root)
        if not obs:
            return 0
        with _LOCK:
            self.refresh()
            for sig, est, act in obs:
                # Single-table, predicate-free scans carry no information the
                # catalog statistics don't already have.
                if len(sig.tables) == 1 and not sig.predicates:
                    continue
                self.record(sig, est, act)
            self._save()
        return len(obs)

    def reset(self) -> None:
        with _LOCK:
            self._entries = {}
            if self.path.exists():
                self.path.unlink()
            self._mtime = None

    def __len__(self) -> int:
        return len(self._entries)

    def summary(self, limit: int = 50) -> dict[str, Any]:
        entries = sorted(self._entries.values(), key=lambda e: e.q_error, reverse=True)
        return {
            "entries": len(self._entries),
            "worst_misestimates": [
                {
                    "tables": e.tables,
                    "predicates": e.predicates,
                    "actual_rows": e.actual_rows,
                    "last_estimate": round(e.last_estimate, 2),
                    "q_error": round(e.q_error, 3),
                    "observations": e.observations,
                }
                for e in entries[:limit]
            ],
        }
