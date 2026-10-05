from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time
from typing import Any, Dict, Optional, Set

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .alert_serializer import (
    AlertSerializationError,
    deserialize_alert,
    serialize_alert,
    serialized_sha256,
)
from .privacy_validator import (
    PrivacyLeakageError,
    validate_no_privacy_leakage,
)


COLLECTOR_VERSION = "phase15e1_v1"


@dataclass
class CollectorState:
    storage_path: Path
    seen_alert_ids: Set[str]
    accepted_count: int = 0
    duplicate_count: int = 0
    rejected_count: int = 0


def _load_existing_alert_ids(
    storage_path: Path,
) -> Set[str]:
    seen: Set[str] = set()

    if not storage_path.exists():
        return seen

    with open(
        storage_path,
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                payload = json.loads(
                    line
                )

            except json.JSONDecodeError:
                # Do not trust malformed prior state.
                continue

            alert_id = payload.get(
                "alert_id"
            )

            if alert_id:
                seen.add(
                    str(
                        alert_id
                    )
                )

    return seen


def create_collector_app(
    *,
    storage_path: Path,
    reset_storage: bool = False,
) -> FastAPI:
    """
    Local privacy-safe central alert collector.

    Phase15E1 is a functional localhost smoke test.
    TLS is intentionally deferred to Phase15E2.
    """

    storage_path = Path(
        storage_path
    )

    storage_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        reset_storage
        and
        storage_path.exists()
    ):
        storage_path.unlink()

    state = CollectorState(
        storage_path=
            storage_path,

        seen_alert_ids=
            _load_existing_alert_ids(
                storage_path
            ),
    )

    lock = threading.Lock()

    app = FastAPI(
        title=
            "Phase15E1 Privacy-Safe Alert Collector",

        version=
            COLLECTOR_VERSION,
    )

    app.state.collector_state = state

    @app.get(
        "/health"
    )
    def health() -> Dict[str, Any]:
        return {
            "status":
                "ok",

            "collector_version":
                COLLECTOR_VERSION,

            "accepted_count":
                state.accepted_count,

            "duplicate_count":
                state.duplicate_count,

            "rejected_count":
                state.rejected_count,

            "stored_unique_alert_ids":
                len(
                    state.seen_alert_ids
                ),

            "transport_note":
                (
                    "Phase15E1 localhost HTTP smoke test only; "
                    "TLS belongs to Phase15E2."
                ),
        }

    @app.post(
        "/api/alerts"
    )
    async def receive_alert(
        request: Request,
    ):
        start = time.perf_counter()

        raw_body = await request.body()

        if not raw_body:
            state.rejected_count += 1

            raise HTTPException(
                status_code=400,
                detail=
                    "Request body is empty.",
            )

        supplied_digest = request.headers.get(
            "X-Message-SHA256"
        )

        actual_digest = serialized_sha256(
            raw_body
        )

        if supplied_digest:
            if supplied_digest.lower() != actual_digest.lower():
                state.rejected_count += 1

                raise HTTPException(
                    status_code=400,
                    detail=
                        "X-Message-SHA256 does not match request body.",
                )

        try:
            alert = deserialize_alert(
                raw_body
            )

        except (
            AlertSerializationError,
            ValueError,
        ) as exc:
            state.rejected_count += 1

            raise HTTPException(
                status_code=400,
                detail=
                    f"Invalid privacy-safe alert: {exc}",
            ) from exc

        alert_mapping = alert.to_dict()

        try:
            leakage_report = validate_no_privacy_leakage(
                alert_mapping,
                require_link_tokens=True,
            )

        except PrivacyLeakageError as exc:
            state.rejected_count += 1

            raise HTTPException(
                status_code=400,
                detail=
                    f"Privacy leakage audit failed: {exc}",
            ) from exc

        alert_id = alert.alert_id

        # Re-serialize using the collector's canonical serializer before storage.
        canonical_bytes = serialize_alert(
            alert
        )

        canonical_digest = serialized_sha256(
            canonical_bytes
        )

        with lock:
            if alert_id in state.seen_alert_ids:
                state.duplicate_count += 1

                raise HTTPException(
                    status_code=409,
                    detail=
                        f"Duplicate alert_id rejected: {alert_id}",
                )

            # Store only the privacy-safe canonical alert.
            with open(
                state.storage_path,
                "ab",
            ) as file:
                file.write(
                    canonical_bytes
                    +
                    b"\n"
                )

            state.seen_alert_ids.add(
                alert_id
            )

            state.accepted_count += 1

        collector_processing_ms = (
            time.perf_counter()
            -
            start
        ) * 1000.0

        return JSONResponse(
            status_code=201,
            content={
                "status":
                    "accepted",

                "alert_id":
                    alert.alert_id,

                "classification":
                    alert.classification,

                "severity":
                    alert.severity,

                "collector_status":
                    "stored",

                "message_sha256":
                    canonical_digest,

                "schema_valid":
                    True,

                "privacy_leakage_audit":
                    "PASS"
                    if leakage_report.passed
                    else
                    "FAIL",

                "collector_processing_ms":
                    collector_processing_ms,

                "transport_security":
                    (
                        "LOCALHOST_HTTP_SMOKE_TEST_ONLY"
                    ),

                "collector_version":
                    COLLECTOR_VERSION,
            },
        )

    return app
