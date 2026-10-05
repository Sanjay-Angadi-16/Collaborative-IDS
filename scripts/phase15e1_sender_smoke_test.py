from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(
            PROJECT_ROOT
        ),
    )


from src.alert_sharing.alert_schema import validate_alert  # noqa: E402
from src.alert_sharing.alert_serializer import serialize_alert  # noqa: E402
from src.alert_sharing.secure_sender import send_privacy_safe_alert  # noqa: E402


DEFAULT_INPUT = (
    PROJECT_ROOT
    / "results"
    / "phase15"
    / "phase15d_shared_alert_pretty.json"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "phase15"
)

AUDIT_FILE = (
    RESULT_ROOT
    / "phase15e1_sender_smoke_audit.json"
)

METRICS_FILE = (
    RESULT_ROOT
    / "phase15e1_sender_smoke_metrics.csv"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=
            "Phase15E1 Sender Smoke Test"
    )

    parser.add_argument(
        "--collector-url",
        default=
            "http://127.0.0.1:8765/api/alerts",
    )

    parser.add_argument(
        "--input-alert",
        type=Path,
        default=
            DEFAULT_INPUT,
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help=
            "Number of unique accepted alerts for simple smoke latency.",
    )

    return parser.parse_args()


def load_template(
    path: Path,
) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Phase15D shared alert: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(
            file
        )

    validate_alert(
        payload
    )

    return payload


def make_unique_alert(
    template: Dict[str, Any],
) -> Dict[str, Any]:
    payload = dict(
        template
    )

    payload[
        "alert_id"
    ] = str(
        uuid.uuid4()
    )

    return validate_alert(
        payload
    ).to_dict()


def main():
    args = parse_args()

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    template = load_template(
        args.input_alert
    )

    print()
    print(
        "="
        *
        120
    )

    print(
        "PHASE 15E1 — LOCAL SENDER SMOKE TEST"
    )

    print(
        "="
        *
        120
    )

    print(
        f"Collector URL       : {args.collector_url}"
    )

    print(
        f"Input               : {args.input_alert}"
    )

    print(
        f"Unique sends        : {args.repeat}"
    )

    print(
        "Transport           : LOCALHOST HTTP SMOKE TEST ONLY"
    )

    print(
        "TLS                 : NO — Phase15E2"
    )

    # Health check
    health_url = args.collector_url.rsplit(
        "/api/alerts",
        1,
    )[0] + "/health"

    health = httpx.get(
        health_url,
        timeout=5.0,
    )

    if health.status_code != 200:
        raise RuntimeError(
            f"Collector health check failed: {health.status_code}"
        )

    print(
        "Collector health    : PASS"
    )

    rows: List[Dict[str, Any]] = []
    accepted_alerts: List[Dict[str, Any]] = []

    for index in range(
        args.repeat
    ):
        alert = make_unique_alert(
            template
        )

        result = send_privacy_safe_alert(
            args.collector_url,
            alert,
            allow_plain_http_localhost=True,
        )

        accepted = (
            result.status_code
            ==
            201
        )

        print(
            f"[{'PASS' if accepted else 'FAIL'}] "
            f"unique_send_{index + 1}: "
            f"HTTP {result.status_code}, "
            f"{result.round_trip_ms:.3f} ms"
        )

        rows.append(
            {
                "test":
                    f"unique_send_{index + 1}",

                "status_code":
                    result.status_code,

                "round_trip_ms":
                    result.round_trip_ms,

                "bytes_sent":
                    result.bytes_sent,

                "accepted":
                    accepted,
            }
        )

        if not accepted:
            raise RuntimeError(
                f"Unique alert was not accepted: {result.response_json}"
            )

        accepted_alerts.append(
            alert
        )

    # Duplicate rejection using the last accepted alert.
    duplicate_result = send_privacy_safe_alert(
        args.collector_url,
        accepted_alerts[
            -1
        ],
        allow_plain_http_localhost=True,
    )

    duplicate_pass = (
        duplicate_result.status_code
        ==
        409
    )

    print(
        f"[{'PASS' if duplicate_pass else 'FAIL'}] "
        f"duplicate_alert_rejected: HTTP "
        f"{duplicate_result.status_code}"
    )

    # Digest mismatch
    tamper_alert = make_unique_alert(
        template
    )

    body = serialize_alert(
        tamper_alert
    )

    digest_mismatch_response = httpx.post(
        args.collector_url,
        content=
            body,
        headers={
            "Content-Type":
                "application/json",

            "X-Message-SHA256":
                "0" * 64,
        },
        timeout=5.0,
    )

    digest_pass = (
        digest_mismatch_response.status_code
        ==
        400
    )

    print(
        f"[{'PASS' if digest_pass else 'FAIL'}] "
        f"digest_mismatch_rejected: HTTP "
        f"{digest_mismatch_response.status_code}"
    )

    # Privacy leakage attempt
    leaked_alert = make_unique_alert(
        template
    )

    leaked_alert[
        "source_ip"
    ] = "192.168.1.25"

    leak_response = httpx.post(
        args.collector_url,
        json=
            leaked_alert,
        timeout=5.0,
    )

    leak_pass = (
        leak_response.status_code
        ==
        400
    )

    print(
        f"[{'PASS' if leak_pass else 'FAIL'}] "
        f"raw_source_ip_rejected: HTTP "
        f"{leak_response.status_code}"
    )

    # Malformed JSON
    malformed_response = httpx.post(
        args.collector_url,
        content=
            b'{"alert_id":',
        headers={
            "Content-Type":
                "application/json"
        },
        timeout=5.0,
    )

    malformed_pass = (
        malformed_response.status_code
        ==
        400
    )

    print(
        f"[{'PASS' if malformed_pass else 'FAIL'}] "
        f"malformed_json_rejected: HTTP "
        f"{malformed_response.status_code}"
    )

    latency_values = [
        row[
            "round_trip_ms"
        ]
        for row
        in rows
    ]

    summary = {
        "phase":
            "15E1",

        "collector_url":
            args.collector_url,

        "transport":
            "LOCALHOST_HTTP_SMOKE_TEST_ONLY",

        "tls_enabled":
            False,

        "unique_sends_requested":
            args.repeat,

        "unique_sends_accepted":
            sum(
                1
                for row
                in rows
                if row[
                    "accepted"
                ]
            ),

        "duplicate_rejection":
            duplicate_pass,

        "digest_mismatch_rejection":
            digest_pass,

        "privacy_leak_rejection":
            leak_pass,

        "malformed_json_rejection":
            malformed_pass,

        "mean_round_trip_ms":
            statistics.mean(
                latency_values
            ),

        "median_round_trip_ms":
            statistics.median(
                latency_values
            ),

        "min_round_trip_ms":
            min(
                latency_values
            ),

        "max_round_trip_ms":
            max(
                latency_values
            ),

        "all_tests_passed":
            (
                all(
                    row[
                        "accepted"
                    ]
                    for row
                    in rows
                )
                and
                duplicate_pass
                and
                digest_pass
                and
                leak_pass
                and
                malformed_pass
            ),

        "locked_test_used":
            False,
    }

    with open(
        AUDIT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    with open(
        METRICS_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "test",
                "status_code",
                "round_trip_ms",
                "bytes_sent",
                "accepted",
            ],
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    print()
    print(
        "="
        *
        120
    )

    print(
        "PHASE 15E1 SENDER SMOKE TEST COMPLETE"
    )

    print(
        "="
        *
        120
    )

    print(
        f"Accepted unique alerts : "
        f"{summary['unique_sends_accepted']}/{args.repeat}"
    )

    print(
        f"Duplicate rejection    : "
        f"{'PASS' if duplicate_pass else 'FAIL'}"
    )

    print(
        f"Digest mismatch        : "
        f"{'PASS' if digest_pass else 'FAIL'}"
    )

    print(
        f"Privacy leak rejection : "
        f"{'PASS' if leak_pass else 'FAIL'}"
    )

    print(
        f"Malformed JSON         : "
        f"{'PASS' if malformed_pass else 'FAIL'}"
    )

    print(
        f"Mean round-trip        : "
        f"{summary['mean_round_trip_ms']:.3f} ms"
    )

    print(
        f"Audit                  : {AUDIT_FILE}"
    )

    print(
        f"Metrics                : {METRICS_FILE}"
    )

    print()

    print(
        f"STATUS                 : "
        f"{'PASS' if summary['all_tests_passed'] else 'FAIL'}"
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Phase 15E2 — HTTPS/TLS transport and authenticated "
        "collector deployment."
    )

    if not summary[
        "all_tests_passed"
    ]:
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()
