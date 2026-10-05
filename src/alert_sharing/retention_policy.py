from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional


class RetentionPolicyError(RuntimeError):
    """Raised when a retention operation cannot be completed safely."""


@dataclass(frozen=True)
class RetentionResult:
    retention_days: int
    cutoff_utc: str
    dry_run: bool
    matching_rows: int
    deleted_rows: int
    remaining_rows: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retention_days":
                self.retention_days,

            "cutoff_utc":
                self.cutoff_utc,

            "dry_run":
                self.dry_run,

            "matching_rows":
                self.matching_rows,

            "deleted_rows":
                self.deleted_rows,

            "remaining_rows":
                self.remaining_rows,
        }


def _parse_iso_utc(value: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise RetentionPolicyError(
            "Timestamp must be a non-empty ISO-8601 string."
        )

    candidate = value.strip()

    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(
            candidate
        )
    except ValueError as exc:
        raise RetentionPolicyError(
            f"Invalid ISO-8601 timestamp: {value!r}"
        ) from exc

    if parsed.tzinfo is None:
        raise RetentionPolicyError(
            "Retention timestamps must include a timezone."
        )

    return parsed.astimezone(
        timezone.utc
    )


def retention_cutoff(
    *,
    now_utc: str,
    retention_days: int,
) -> str:
    if int(retention_days) <= 0:
        raise RetentionPolicyError(
            "retention_days must be greater than zero."
        )

    now = _parse_iso_utc(
        now_utc
    )

    cutoff = now - timedelta(
        days=int(
            retention_days
        )
    )

    return cutoff.isoformat()


def _connect(
    database_path: Path,
):
    database_path = Path(
        database_path
    )

    if not database_path.exists():
        raise RetentionPolicyError(
            f"Central alert database does not exist: {database_path}"
        )

    connection = sqlite3.connect(
        database_path,
        timeout=10.0,
    )

    connection.row_factory = sqlite3.Row

    return connection


def retention_preview(
    database_path: Path,
    *,
    now_utc: str,
    retention_days: int,
) -> RetentionResult:
    cutoff = retention_cutoff(
        now_utc=
            now_utc,

        retention_days=
            retention_days,
    )

    connection = _connect(
        database_path
    )

    try:
        matching = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM alerts
            WHERE event_time < ?
            """,
            (
                cutoff,
            ),
        ).fetchone()[
            "count"
        ]

        remaining = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM alerts
            """
        ).fetchone()[
            "count"
        ]

    finally:
        connection.close()

    return RetentionResult(
        retention_days=
            int(
                retention_days
            ),

        cutoff_utc=
            cutoff,

        dry_run=
            True,

        matching_rows=
            int(
                matching
            ),

        deleted_rows=
            0,

        remaining_rows=
            int(
                remaining
            ),
    )


def apply_retention_policy(
    database_path: Path,
    *,
    now_utc: str,
    retention_days: int,
) -> RetentionResult:
    """
    Delete alerts older than the UTC retention cutoff.

    This must only be used on a deliberate retention target.
    Phase15G tests it on a dedicated synthetic benchmark database,
    not the user's earlier Phase15F research database.
    """

    cutoff = retention_cutoff(
        now_utc=
            now_utc,

        retention_days=
            retention_days,
    )

    connection = _connect(
        database_path
    )

    try:
        connection.execute(
            "BEGIN IMMEDIATE"
        )

        matching = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM alerts
            WHERE event_time < ?
            """,
            (
                cutoff,
            ),
        ).fetchone()[
            "count"
        ]

        cursor = connection.execute(
            """
            DELETE FROM alerts
            WHERE event_time < ?
            """,
            (
                cutoff,
            ),
        )

        deleted = int(
            cursor.rowcount
        )

        remaining = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM alerts
            """
        ).fetchone()[
            "count"
        ]

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()

    return RetentionResult(
        retention_days=
            int(
                retention_days
            ),

        cutoff_utc=
            cutoff,

        dry_run=
            False,

        matching_rows=
            int(
                matching
            ),

        deleted_rows=
            deleted,

        remaining_rows=
            int(
                remaining
            ),
    )


def oldest_newest_event_time(
    database_path: Path,
) -> Dict[str, Optional[str]]:
    connection = _connect(
        database_path
    )

    try:
        row = connection.execute(
            """
            SELECT
                MIN(event_time) AS oldest,
                MAX(event_time) AS newest
            FROM alerts
            """
        ).fetchone()

    finally:
        connection.close()

    return {
        "oldest_event_time":
            row[
                "oldest"
            ],

        "newest_event_time":
            row[
                "newest"
            ],
    }
