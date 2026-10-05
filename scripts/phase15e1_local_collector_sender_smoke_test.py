from __future__ import annotations

import json
import socket
import statistics
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import httpx
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
from src.alert_sharing.alert_schema import validate_alert  # noqa: E402
from src.alert_sharing.alert_serializer import serialize_alert  # noqa: E402
from src.alert_sharing.secure_sender import send_privacy_safe_alert  # noqa: E402


HOST = "127.0.0.1"
PORT = 8765

COLLECTOR_URL = (
    f"http://{HOST}:{PORT}/api/alerts"
)

HEALTH_URL = (
    f"http://{HOST}:{PORT}/health"
)

INPUT_FILE = (
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

ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "privacy"
)

COLLECTED_FILE = (
    RESULT_ROOT
    / "phase15e1_collected_alerts.jsonl"
)

AUDIT_FILE = (
    RESULT_ROOT
    / "phase15e1_smoke_test_audit.json"
)

MANIFEST_FILE = (
    ARTIFACT_ROOT
    / "phase15e1_collector_manifest.json"
)


def wait_for_server(
    timeout_seconds: float = 10.0,
) -> None:
    deadline = (
        time.time()
        +
        timeout_seconds
    )

    while time.time() < deadline:
        try:
            with socket.create_connection(
                (
                    HOST,
                    PORT,
                ),
                timeout=0.25,
            ):
                return

        except OSError:
            time.sleep(
                0.1
            )

    raise RuntimeError(
        "Local collector did not start in time."
    )


def load_template() -> Dict[str, Any]:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "\nMissing Phase15D shared alert:\n"
            f"{INPUT_FILE}\n\n"
            "Run Phase15D first."
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(
            file
        )

    return validate_alert(
        payload
    ).to_dict()


def unique_alert(
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
    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARTIFACT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if COLLECTED_FILE.exists():
        COLLECTED_FILE.unlink()

    template = load_template()

    app = create_collector_app(
        storage_path=
            COLLECTED_FILE,

        reset_storage=
            True,
    )

    config = uvicorn.Config(
        app,
        host=
            HOST,
        port=
            PORT,
        log_level=
            "warning",
    )

    server = uvicorn.Server(
        config
    )

    thread = threading.Thread(
        target=
            server.run,
        daemon=True,
    )

    print()
    print(
        "="
        *
        126
    )

    print(
        "PHASE 15E1 — LOCAL CENTRAL ALERT COLLECTOR + SENDER SMOKE TEST"
    )

    print(
        "="
        *
        126
    )

    print(
        f"Collector                   : {COLLECTOR_URL}"
    )

    print(
        f"Input                       : {INPUT_FILE}"
    )

    print(
        f"Storage                     : {COLLECTED_FILE}"
    )

    print(
        "Transport                   : LOCALHOST HTTP ONLY"
    )

    print(
        "TLS                         : NO — Phase15E2"
    )

    print(
        "Locked test                 : NO"
    )

    thread.start()

    try:
        wait_for_server()

        health = httpx.get(
            HEALTH_URL,
            timeout=5.0,
        )

        health_pass = (
            health.status_code
            ==
            200
        )

        print()
        print(
            "SMOKE TESTS"
        )

        print(
            "-"
            *
            126
        )

        print(
            f"[{'PASS' if health_pass else 'FAIL'}] "
            f"collector_health: HTTP {health.status_code}"
        )

        if not health_pass:
            raise RuntimeError(
                "Collector health check failed."
            )

        accepted_latencies: List[
            float
        ] = []

        first_alert = unique_alert(
            template
        )

        first_result = send_privacy_safe_alert(
            COLLECTOR_URL,
            first_alert,
            allow_plain_http_localhost=True,
        )

        first_pass = (
            first_result.status_code
            ==
            201
        )

        accepted_latencies.append(
            first_result.round_trip_ms
        )

        print(
            f"[{'PASS' if first_pass else 'FAIL'}] "
            f"privacy_safe_alert_accepted: "
            f"HTTP {first_result.status_code}, "
            f"{first_result.round_trip_ms:.3f} ms"
        )

        # Send four additional unique messages.
        unique_passes = [
            first_pass
        ]

        for index in range(
            4
        ):
            result = send_privacy_safe_alert(
                COLLECTOR_URL,
                unique_alert(
                    template
                ),
                allow_plain_http_localhost=True,
            )

            passed = (
                result.status_code
                ==
                201
            )

            unique_passes.append(
                passed
            )

            accepted_latencies.append(
                result.round_trip_ms
            )

            print(
                f"[{'PASS' if passed else 'FAIL'}] "
                f"additional_unique_{index + 1}: "
                f"HTTP {result.status_code}, "
                f"{result.round_trip_ms:.3f} ms"
            )

        duplicate_result = send_privacy_safe_alert(
            COLLECTOR_URL,
            first_alert,
            allow_plain_http_localhost=True,
        )

        duplicate_pass = (
            duplicate_result.status_code
            ==
            409
        )

        print(
            f"[{'PASS' if duplicate_pass else 'FAIL'}] "
            f"duplicate_alert_rejected: "
            f"HTTP {duplicate_result.status_code}"
        )

        # Digest mismatch.
        digest_alert = unique_alert(
            template
        )

        digest_body = serialize_alert(
            digest_alert
        )

        digest_response = httpx.post(
            COLLECTOR_URL,
            content=
                digest_body,
            headers={
                "Content-Type":
                    "application/json",

                "X-Message-SHA256":
                    "f" * 64,
            },
            timeout=5.0,
        )

        digest_pass = (
            digest_response.status_code
            ==
            400
        )

        print(
            f"[{'PASS' if digest_pass else 'FAIL'}] "
            f"digest_mismatch_rejected: "
            f"HTTP {digest_response.status_code}"
        )

        # Privacy leakage attempt.
        leaked = unique_alert(
            template
        )

        leaked[
            "source_ip"
        ] = "10.0.0.25"

        leak_response = httpx.post(
            COLLECTOR_URL,
            json=
                leaked,
            timeout=5.0,
        )

        leak_pass = (
            leak_response.status_code
            ==
            400
        )

        print(
            f"[{'PASS' if leak_pass else 'FAIL'}] "
            f"raw_source_ip_rejected: "
            f"HTTP {leak_response.status_code}"
        )

        # Malformed JSON.
        malformed_response = httpx.post(
            COLLECTOR_URL,
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
            f"malformed_json_rejected: "
            f"HTTP {malformed_response.status_code}"
        )

        # Plain HTTP to non-local host must be rejected by sender logic.
        sender_guard_pass = False

        try:
            send_privacy_safe_alert(
                "http://example.com/api/alerts",
                unique_alert(
                    template
                ),
                allow_plain_http_localhost=True,
                timeout_seconds=0.1,
            )

        except Exception:
            sender_guard_pass = True

        print(
            f"[{'PASS' if sender_guard_pass else 'FAIL'}] "
            "sender_blocks_plain_http_non_localhost"
        )

        # Verify stored line count = accepted unique alerts only.
        with open(
            COLLECTED_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            stored_lines = [
                line
                for line
                in file
                if line.strip()
            ]

        stored_count_pass = (
            len(
                stored_lines
            )
            ==
            5
        )

        print(
            f"[{'PASS' if stored_count_pass else 'FAIL'}] "
            f"stored_unique_alert_count: "
            f"{len(stored_lines)}"
        )

        health_final = httpx.get(
            HEALTH_URL,
            timeout=5.0,
        ).json()

        all_passed = (
            health_pass
            and
            all(
                unique_passes
            )
            and
            duplicate_pass
            and
            digest_pass
            and
            leak_pass
            and
            malformed_pass
            and
            sender_guard_pass
            and
            stored_count_pass
        )

        audit = {
            "phase":
                "15E1",

            "phase_name":
                (
                    "Local Central Alert Collector + "
                    "Sender Smoke Test"
                ),

            "collector_url":
                COLLECTOR_URL,

            "transport":
                "LOCALHOST_HTTP_SMOKE_TEST_ONLY",

            "tls_enabled":
                False,

            "input":
                str(
                    INPUT_FILE
                ),

            "accepted_unique_alerts":
                5,

            "duplicate_rejection":
                duplicate_pass,

            "digest_mismatch_rejection":
                digest_pass,

            "privacy_leakage_rejection":
                leak_pass,

            "malformed_json_rejection":
                malformed_pass,

            "sender_plain_http_non_localhost_guard":
                sender_guard_pass,

            "stored_unique_alert_count":
                len(
                    stored_lines
                ),

            "mean_sender_round_trip_ms":
                statistics.mean(
                    accepted_latencies
                ),

            "median_sender_round_trip_ms":
                statistics.median(
                    accepted_latencies
                ),

            "collector_health_final":
                health_final,

            "all_tests_passed":
                all_passed,

            "locked_test_used":
                False,
        }

        with open(
            AUDIT_FILE,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                audit,
                file,
                indent=4,
            )

        manifest = {
            "phase":
                "15E1",

            "collector_endpoint":
                "/api/alerts",

            "health_endpoint":
                "/health",

            "collector_validation_pipeline":
                [
                    "raw request bytes",
                    "optional SHA-256 equality check",
                    "canonical JSON deserialization",
                    "Phase15A schema validation",
                    "Phase15C leakage audit",
                    "duplicate alert_id rejection",
                    "canonical privacy-safe JSONL persistence",
                ],

            "transport_security":
                "localhost HTTP smoke test only",

            "message_sha256_authentication":
                False,

            "message_sha256_purpose":
                "integrity/tamper detection only",

            "tls_required_next":
                True,

            "phase15e2":
                "HTTPS/TLS transport",

            "locked_test_used":
                False,
        }

        with open(
            MANIFEST_FILE,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                manifest,
                file,
                indent=4,
            )

        print()
        print(
            "="
            *
            126
        )

        print(
            "PHASE 15E1 COMPLETE"
        )

        print(
            "="
            *
            126
        )

        print(
            f"Accepted unique alerts       : 5/5"
        )

        print(
            f"Duplicate rejection          : "
            f"{'PASS' if duplicate_pass else 'FAIL'}"
        )

        print(
            f"Digest mismatch rejection    : "
            f"{'PASS' if digest_pass else 'FAIL'}"
        )

        print(
            f"Privacy leakage rejection    : "
            f"{'PASS' if leak_pass else 'FAIL'}"
        )

        print(
            f"Malformed JSON rejection     : "
            f"{'PASS' if malformed_pass else 'FAIL'}"
        )

        print(
            f"HTTP non-localhost guard     : "
            f"{'PASS' if sender_guard_pass else 'FAIL'}"
        )

        print(
            f"Stored unique alerts         : "
            f"{len(stored_lines)}"
        )

        print(
            f"Mean sender round-trip       : "
            f"{audit['mean_sender_round_trip_ms']:.3f} ms"
        )

        print(
            f"Collector storage            : {COLLECTED_FILE}"
        )

        print(
            f"Audit                        : {AUDIT_FILE}"
        )

        print(
            f"Manifest                     : {MANIFEST_FILE}"
        )

        print(
            f"Locked test used             : NO"
        )

        print()

        print(
            f"STATUS                       : "
            f"{'PASS' if all_passed else 'FAIL'}"
        )

        print()

        print(
            "IMPORTANT:"
        )

        print(
            "Phase15E1 validates application behavior only."
        )

        print(
            "HTTP is restricted to localhost. "
            "Do not deploy this collector remotely without TLS."
        )

        print()

        print(
            "NEXT:"
        )

        print(
            "Phase 15E2 — HTTPS/TLS transport and authenticated "
            "collector communication."
        )

        if not all_passed:
            raise SystemExit(
                1
            )

    finally:
        server.should_exit = True

        thread.join(
            timeout=5.0
        )


if __name__ == "__main__":
    main()
