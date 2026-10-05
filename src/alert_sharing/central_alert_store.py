from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .alert_schema import PrivacyAwareAlert, validate_alert
from .alert_serializer import serialize_alert, serialized_sha256
from .privacy_validator import validate_no_privacy_leakage


STORE_SCHEMA_VERSION = "phase15f_v1"


class AlertStoreError(RuntimeError):
    """Raised when the central privacy-safe alert store fails."""


class DuplicateAlertError(AlertStoreError):
    """Raised when an alert_id already exists in the central store."""


@dataclass(frozen=True)
class StoreInsertResult:
    alert_id: str
    row_id: int
    message_sha256: str
    received_at_utc: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CentralAlertStore:
    """
    SQLite-backed central storage for privacy-safe Phase15 alerts.

    Storage policy
    --------------
    - Only Phase15A-valid privacy-safe alert fields are stored.
    - Raw IPs, payloads, raw features, CNN sequences and model state
      are never accepted into this storage layer.
    - alert_id is unique.
    - Stable pseudonymous tokens can be indexed for later correlation.
    - WAL is enabled for simple multi-reader / single-writer operation.
    """

    def __init__(
        self,
        database_path: Path,
    ) -> None:
        self.database_path = Path(
            database_path
        )

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(
            self.database_path,
            timeout=10.0,
        )

        connection.row_factory = sqlite3.Row

        try:
            connection.execute(
                "PRAGMA foreign_keys = ON"
            )

            connection.execute(
                "PRAGMA journal_mode = WAL"
            )

            connection.execute(
                "PRAGMA synchronous = NORMAL"
            )

            yield connection

            connection.commit()

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS alert_store_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    alert_id TEXT NOT NULL UNIQUE,
                    client_id TEXT NOT NULL,
                    event_time TEXT NOT NULL,

                    classification TEXT NOT NULL,
                    severity TEXT NOT NULL,

                    confidence REAL NOT NULL,
                    risk_score REAL NULL,
                    evidence_score REAL NOT NULL,
                    risk_mode TEXT NOT NULL,

                    decision_source TEXT NOT NULL,
                    processing_path TEXT NOT NULL,

                    source_token TEXT NOT NULL,
                    destination_token TEXT NOT NULL,

                    protocol TEXT NULL,
                    destination_port INTEGER NULL,

                    model_version TEXT NOT NULL,
                    policy_version TEXT NOT NULL,
                    privacy_version TEXT NOT NULL,
                    schema_version TEXT NOT NULL,

                    message_sha256 TEXT NOT NULL,
                    received_at_utc TEXT NOT NULL,

                    canonical_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_alerts_event_time
                    ON alerts(event_time);

                CREATE INDEX IF NOT EXISTS idx_alerts_client_id
                    ON alerts(client_id);

                CREATE INDEX IF NOT EXISTS idx_alerts_classification
                    ON alerts(classification);

                CREATE INDEX IF NOT EXISTS idx_alerts_severity
                    ON alerts(severity);

                CREATE INDEX IF NOT EXISTS idx_alerts_source_token
                    ON alerts(source_token);

                CREATE INDEX IF NOT EXISTS idx_alerts_destination_token
                    ON alerts(destination_token);

                CREATE INDEX IF NOT EXISTS idx_alerts_client_event_time
                    ON alerts(client_id, event_time);

                CREATE INDEX IF NOT EXISTS idx_alerts_source_event_time
                    ON alerts(source_token, event_time);

                CREATE INDEX IF NOT EXISTS idx_alerts_classification_severity
                    ON alerts(classification, severity);
                """
            )

            connection.execute(
                """
                INSERT INTO alert_store_metadata(key, value)
                VALUES('store_schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (
                    STORE_SCHEMA_VERSION,
                ),
            )

    def clear(self) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM alerts"
            )

    def insert_alert(
        self,
        alert: Mapping[str, Any] | PrivacyAwareAlert,
        *,
        received_at_utc: Optional[str] = None,
    ) -> StoreInsertResult:
        if isinstance(
            alert,
            PrivacyAwareAlert,
        ):
            validated = alert.validate()
        else:
            validated = validate_alert(
                alert
            )

        payload = validated.to_dict()

        # This is the storage privacy boundary.
        validate_no_privacy_leakage(
            payload,
            require_link_tokens=True,
        )

        canonical_bytes = serialize_alert(
            validated
        )

        canonical_text = canonical_bytes.decode(
            "utf-8"
        )

        digest = serialized_sha256(
            canonical_bytes
        )

        received = (
            received_at_utc
            if received_at_utc is not None
            else utc_now_iso()
        )

        values = (
            validated.alert_id,
            validated.client_id,
            validated.event_time,
            validated.classification,
            validated.severity,
            validated.confidence,
            validated.risk_score,
            validated.evidence_score,
            validated.risk_mode,
            validated.decision_source,
            validated.processing_path,
            validated.source_token,
            validated.destination_token,
            validated.protocol,
            validated.destination_port,
            validated.model_version,
            validated.policy_version,
            validated.privacy_version,
            validated.schema_version,
            digest,
            received,
            canonical_text,
        )

        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO alerts(
                        alert_id,
                        client_id,
                        event_time,
                        classification,
                        severity,
                        confidence,
                        risk_score,
                        evidence_score,
                        risk_mode,
                        decision_source,
                        processing_path,
                        source_token,
                        destination_token,
                        protocol,
                        destination_port,
                        model_version,
                        policy_version,
                        privacy_version,
                        schema_version,
                        message_sha256,
                        received_at_utc,
                        canonical_json
                    )
                    VALUES(
                        ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?
                    )
                    """,
                    values,
                )

                row_id = int(
                    cursor.lastrowid
                )

        except sqlite3.IntegrityError as exc:
            message = str(
                exc
            ).lower()

            if (
                "alert_id"
                in message
                or
                "unique"
                in message
            ):
                raise DuplicateAlertError(
                    f"Duplicate alert_id rejected: {validated.alert_id}"
                ) from exc

            raise AlertStoreError(
                f"SQLite integrity error: {exc}"
            ) from exc

        return StoreInsertResult(
            alert_id=
                validated.alert_id,

            row_id=
                row_id,

            message_sha256=
                digest,

            received_at_utc=
                received,
        )

    def get_alert(
        self,
        alert_id: str,
    ) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT canonical_json
                FROM alerts
                WHERE alert_id = ?
                """,
                (
                    alert_id,
                ),
            ).fetchone()

        if row is None:
            return None

        payload = json.loads(
            row[
                "canonical_json"
            ]
        )

        # Revalidate data on read.
        validated = validate_alert(
            payload
        )

        validate_no_privacy_leakage(
            validated.to_dict(),
            require_link_tokens=True,
        )

        return validated.to_dict()

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM alerts"
            ).fetchone()

        return int(
            row[
                "count"
            ]
        )

    def count_by_client(self) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    client_id,
                    COUNT(*) AS alert_count
                FROM alerts
                GROUP BY client_id
                ORDER BY client_id
                """
            ).fetchall()

        return [
            {
                "client_id":
                    row[
                        "client_id"
                    ],

                "alert_count":
                    int(
                        row[
                            "alert_count"
                        ]
                    ),
            }
            for row
            in rows
        ]

    def count_by_classification(self) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    classification,
                    COUNT(*) AS alert_count
                FROM alerts
                GROUP BY classification
                ORDER BY classification
                """
            ).fetchall()

        return [
            {
                "classification":
                    row[
                        "classification"
                    ],

                "alert_count":
                    int(
                        row[
                            "alert_count"
                        ]
                    ),
            }
            for row
            in rows
        ]

    def count_by_severity(self) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    severity,
                    COUNT(*) AS alert_count
                FROM alerts
                GROUP BY severity
                ORDER BY severity
                """
            ).fetchall()

        return [
            {
                "severity":
                    row[
                        "severity"
                    ],

                "alert_count":
                    int(
                        row[
                            "alert_count"
                        ]
                    ),
            }
            for row
            in rows
        ]

    def find_by_source_token(
        self,
        source_token: str,
        *,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT canonical_json
                FROM alerts
                WHERE source_token = ?
                ORDER BY event_time
                LIMIT ?
                """,
                (
                    source_token,
                    int(
                        limit
                    ),
                ),
            ).fetchall()

        output = []

        for row in rows:
            payload = json.loads(
                row[
                    "canonical_json"
                ]
            )

            validated = validate_alert(
                payload
            )

            output.append(
                validated.to_dict()
            )

        return output

    def recent_alerts(
        self,
        *,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT canonical_json
                FROM alerts
                ORDER BY event_time DESC
                LIMIT ?
                """,
                (
                    int(
                        limit
                    ),
                ),
            ).fetchall()

        return [
            validate_alert(
                json.loads(
                    row[
                        "canonical_json"
                    ]
                )
            ).to_dict()
            for row
            in rows
        ]

    def list_indexes(self) -> List[str]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type='index'
                  AND tbl_name='alerts'
                ORDER BY name
                """
            ).fetchall()

        return [
            str(
                row[
                    "name"
                ]
            )
            for row
            in rows
            if row[
                "name"
            ]
        ]

    def privacy_at_rest_audit(
        self,
        *,
        forbidden_literals: Sequence[str],
    ) -> Dict[str, Any]:
        """
        Search persisted canonical JSON for explicit raw/sensitive test literals.

        The HMAC secret itself must NOT be supplied to this method or persisted.
        """

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    alert_id,
                    canonical_json
                FROM alerts
                ORDER BY id
                """
            ).fetchall()

        findings = []

        for row in rows:
            text = str(
                row[
                    "canonical_json"
                ]
            )

            for literal in forbidden_literals:
                if (
                    literal
                    and
                    str(
                        literal
                    )
                    in text
                ):
                    findings.append(
                        {
                            "alert_id":
                                row[
                                    "alert_id"
                                ],

                            "matched_literal":
                                str(
                                    literal
                                ),
                        }
                    )

        return {
            "passed":
                len(
                    findings
                )
                ==
                0,

            "rows_scanned":
                len(
                    rows
                ),

            "findings":
                findings,
        }
