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


from src.alert_sharing.alert_collector import create_collector_app  # noqa: E402


DEFAULT_STORAGE = (
    PROJECT_ROOT
    / "results"
    / "phase15"
    / "phase15e1_collected_alerts.jsonl"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=
            "Phase15E1 Local Central Privacy-Safe Alert Collector"
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8765,
    )

    parser.add_argument(
        "--storage",
        type=Path,
        default=DEFAULT_STORAGE,
    )

    parser.add_argument(
        "--reset-storage",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.host not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise RuntimeError(
            "Phase15E1 is restricted to localhost. "
            "Remote/TLS transport belongs to Phase15E2."
        )

    app = create_collector_app(
        storage_path=
            args.storage,

        reset_storage=
            args.reset_storage,
    )

    print()
    print(
        "PHASE 15E1 — LOCAL CENTRAL ALERT COLLECTOR"
    )

    print(
        f"Collector endpoint : http://{args.host}:{args.port}/api/alerts"
    )

    print(
        f"Health endpoint    : http://{args.host}:{args.port}/health"
    )

    print(
        f"Storage            : {args.storage}"
    )

    print(
        "Transport          : LOCALHOST HTTP SMOKE TEST ONLY"
    )

    print(
        "TLS                : NOT YET — Phase15E2"
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
