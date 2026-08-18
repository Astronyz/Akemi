import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator, Optional, List
import sqlite_utils

from akemi.akemi.core.config import get_settings
from akemi.akemi.storage.models import (
    Event, EventType, AudioEvent, VisionEvent, BrainEvent,
    TTSEvent, SystemEvent, ErrorEvent, create_table_schema
)

import structlog

logger = structlog.get_logger()


class Database:
    """SQLite database wrapper using sqlite-utils."""

    def __init__(self, db_path: str = "data/akemi.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db: Optional[sqlite_utils.Database] = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database connection and create tables."""
        self._db = sqlite_utils.Database(self.db_path)
        # Execute schema
        for statement in create_table_schema().strip().split(";"):
            stmt = statement.strip()
            if stmt:
                self._db.conn.execute(stmt)
        self._db.conn.commit()
        logger.info("Database initialized", path=str(self.db_path))

    @property
    def db(self) -> sqlite_utils.Database:
        """Get the sqlite-utils database instance."""
        if self._db is None:
            self._init_db()
        return self._db

    @contextmanager
    def transaction(self) -> Iterator[sqlite_utils.Database]:
        """Context manager for database transactions."""
        try:
            yield self.db
            self.db.conn.commit()
        except Exception:
            self.db.conn.rollback()
            raise

    def insert_event(self, event: Event) -> str:
        """Insert an event into the database."""
        with self.transaction() as db:
            db["events"].insert(event.to_dict(), pk="id", replace=True)
        return event.id

    def insert_events(self, events: List[Event]) -> int:
        """Bulk insert events."""
        if not events:
            return 0
        with self.transaction() as db:
            db["events"].insert_all([e.to_dict() for e in events], pk="id", replace=True)
        return len(events)

    def get_events(
        self,
        event_type: Optional[EventType] = None,
        session_id: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[dict]:
        """Query events with filters."""
        query = "SELECT * FROM events WHERE 1=1"
        params = []

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type.value)

        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)

        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())

        if until:
            query += " AND timestamp <= ?"
            params.append(until.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self.db.conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_recent_transcriptions(
        self,
        session_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[dict]:
        """Get recent transcription events."""
        return self.get_events(
            event_type=EventType.AUDIO_TRANSCRIPTION,
            session_id=session_id,
            limit=limit,
        )

    def get_recent_vision_events(
        self,
        session_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[dict]:
        """Get recent vision events (screenshots, OCR, changes)."""
        return self.get_events(
            event_type=EventType.VISION_SCREENSHOT,
            session_id=session_id,
            limit=limit,
        )

    def get_error_events(
        self,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[dict]:
        """Get error events for self-improvement analysis."""
        return self.get_events(
            event_type=EventType.ERROR,
            since=since,
            limit=limit,
        )

    def cleanup_old_events(self, retention_days: int = 30) -> int:
        """Delete events older than retention_days."""
        cutoff = datetime.utcnow() - timedelta(days=retention_days)
        with self.transaction() as db:
            cursor = db.conn.execute(
                "DELETE FROM events WHERE timestamp < ?",
                (cutoff.isoformat(),)
            )
            deleted = cursor.rowcount
        logger.info("Cleaned up old events", deleted=deleted, cutoff=cutoff.isoformat())
        return deleted

    def get_stats(self) -> dict:
        """Get database statistics."""
        stats = {}
        stats["total_events"] = self.db.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

        # Count by type
        type_counts = self.db.conn.execute(
            "SELECT event_type, COUNT(*) as count FROM events GROUP BY event_type"
        ).fetchall()
        stats["by_type"] = {row[0]: row[1] for row in type_counts}

        # Recent activity
        recent = self.db.conn.execute(
            "SELECT COUNT(*) FROM events WHERE timestamp > datetime('now', '-1 hour')"
        ).fetchone()[0]
        stats["last_hour"] = recent

        # DB size
        stats["db_size_bytes"] = self.db_path.stat().st_size if self.db_path.exists() else 0

        return stats

    def vacuum(self) -> None:
        """Vacuum database to reclaim space."""
        self.db.conn.execute("VACUUM")
        logger.info("Database vacuumed")

    def close(self) -> None:
        """Close database connection."""
        if self._db:
            self._db.conn.close()
            self._db = None


# Global database instance
_db: Optional[Database] = None


def get_db() -> Database:
    """Get the global database instance."""
    global _db
    if _db is None:
        settings = get_settings()
        _db = Database(settings.storage.db_path)
    return _db


def close_db() -> None:
    """Close the global database."""
    global _db
    if _db:
        _db.close()
        _db = None