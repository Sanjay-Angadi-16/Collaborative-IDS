from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .alert_serializer import (
    AlertSerializationError,
    deserialize_alert,
    serialized_sha256,
)
from .central_alert_store import (
    AlertStoreError,
    CentralAlertStore,
    DuplicateAlertError,
)
from .privacy_validator import (
    PrivacyLeakageError,
    validate_no_privacy_leakage,
)


PERSISTENT_COLLECTOR_VERSION = "phase15f_v1"


def create_persistent_collector_app(
    *,
    database_path: Path,
    clear_database: bool = False,
) -> FastAPI:
    store = CentralAlertStore(
        database_path
    )

    if clear_database:
        store.clear()

    app = FastAPI(
        title=
            "Phase15F Persistent Privacy-Safe Alert Collector",

        version=
            PERSISTENT_COLLECTOR_VERSION,
    )

    app.state.alert_store = store

    @app.get(
        "/health"
    )
    def health() -> Dict[str, Any]:
        return {
            "status":
                "ok",

            "collector_version":
                PERSISTENT_COLLECTOR_VERSION,

            "stored_alerts":
                store.count(),

            "database_backend":
                "SQLite",

            "privacy_policy":
                "Phase15A schema + Phase15C leakage audit",
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
            raise HTTPException(
                status_code=400,
                detail="Request body is empty.",
            )

        supplied_digest = request.headers.get(
            "X-Message-SHA256"
        )

        actual_digest = serialized_sha256(
            raw_body
        )

        if (
            supplied_digest
            and
            supplied_digest.lower()
            !=
            actual_digest.lower()
        ):
            raise HTTPException(
                status_code=400,
                detail=
                    "X-Message-SHA256 does not match request body.",
            )

        try:
            alert = deserialize_alert(
                raw_body
            )

            validate_no_privacy_leakage(
                alert.to_dict(),
                require_link_tokens=True,
            )

            result = store.insert_alert(
                alert
            )

        except DuplicateAlertError as exc:
            raise HTTPException(
                status_code=409,
                detail=str(
                    exc
                ),
            ) from exc

        except (
            AlertSerializationError,
            PrivacyLeakageError,
            AlertStoreError,
            ValueError,
        ) as exc:
            raise HTTPException(
                status_code=400,
                detail=str(
                    exc
                ),
            ) from exc

        processing_ms = (
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
                    result.alert_id,

                "row_id":
                    result.row_id,

                "classification":
                    alert.classification,

                "severity":
                    alert.severity,

                "collector_status":
                    "persisted",

                "message_sha256":
                    result.message_sha256,

                "collector_processing_ms":
                    processing_ms,

                "collector_version":
                    PERSISTENT_COLLECTOR_VERSION,
            },
        )

    @app.get(
        "/api/alerts/count"
    )
    def alert_count():
        return {
            "count":
                store.count()
        }

    @app.get(
        "/api/alerts/by-client"
    )
    def by_client():
        return {
            "items":
                store.count_by_client()
        }

    @app.get(
        "/api/alerts/by-classification"
    )
    def by_classification():
        return {
            "items":
                store.count_by_classification()
        }

    @app.get(
        "/api/alerts/recent"
    )
    def recent(
        limit: int = 20,
    ):
        safe_limit = max(
            1,
            min(
                int(
                    limit
                ),
                500,
            ),
        )

        return {
            "items":
                store.recent_alerts(
                    limit=
                        safe_limit
                )
        }

    return app
