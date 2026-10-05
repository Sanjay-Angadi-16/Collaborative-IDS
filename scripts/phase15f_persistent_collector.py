from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(
            PROJECT_ROOT
        ),
    )


from src.alert_sharing.persistent_alert_collector import (  # noqa: E402
    create_persistent_collector_app,
)


DEFAULT_DATABASE = (
    PROJECT_ROOT
    / "results"
    / "phase15"
    / "phase15f_central_alerts.sqlite3"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=
            "Phase15F Persistent Central Privacy-Safe Alert Collector"
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8770,
    )

    parser.add_argument(
        "--database",
        type=Path,
        default=
            DEFAULT_DATABASE,
    )

    parser.add_argument(
        "--clear-database",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    app = create_persistent_collector_app(
        database_path=
            args.database,

        clear_database=
            args.clear_database,
    )

    print()
    print(
        "PHASE 15F — PERSISTENT CENTRAL ALERT COLLECTOR"
    )

    print(
        f"Endpoint : http://{args.host}:{args.port}/api/alerts"
    )

    print(
        f"Database : {args.database}"
    )

    print(
        "NOTE     : For remote deployment, use the Phase15E2 "
        "mTLS transport configuration."
    )

    uvicorn.run(
        app,
        host=
            args.host,
        port=
            args.port,
        log_level=
            "warning",
    )


if __name__ == "__main__":
    main()
