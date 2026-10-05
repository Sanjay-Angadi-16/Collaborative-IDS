from __future__ import annotations

import json
import socket
import ssl
import statistics
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import httpx
import uvicorn


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(
    PROJECT_ROOT
) not in sys.path:
    sys.path.insert(
        0,
        str(
            PROJECT_ROOT
        ),
    )


from src.alert_sharing.alert_collector import create_collector_app  # noqa: E402
from src.alert_sharing.alert_schema import validate_alert  # noqa: E402
from src.alert_sharing.alert_serializer import serialize_alert  # noqa: E402
from src.alert_sharing.secure_sender import (  # noqa: E402
    build_mtls_client_context,
    send_privacy_safe_alert,
)
from src.alert_sharing.tls_utils import generate_local_mtls_material  # noqa: E402


HOST = "127.0.0.1"
PORT = 8766

BASE_URL = (
    f"https://localhost:{PORT}"
)

COLLECTOR_URL = (
    BASE_URL
    +
    "/api/alerts"
)

HEALTH_URL = (
    BASE_URL
    +
    "/health"
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
    / "phase15e2_collected_alerts.jsonl"
)

AUDIT_FILE = (
    RESULT_ROOT
    / "phase15e2_mtls_smoke_test_audit.json"
)

MANIFEST_FILE = (
    ARTIFACT_ROOT
    / "phase15e2_mtls_manifest.json"
)


def load_template() -> Dict[str, Any]:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "\nMissing Phase15D privacy-safe alert:\n"
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


def wait_for_https_health(
    *,
    client_ssl_context: ssl.SSLContext,
    timeout_seconds: float = 12.0,
) -> Dict[str, Any]:
    deadline = (
        time.time()
        +
        timeout_seconds
    )

    last_error = None

    while time.time() < deadline:
        try:
            with httpx.Client(
                verify=
                    client_ssl_context,

                timeout=
                    1.0,

                trust_env=
                    False,
            ) as client:
                response = client.get(
                    HEALTH_URL
                )

            if response.status_code == 200:
                return response.json()

        except Exception as exc:
            last_error = exc

        time.sleep(
            0.15
        )

    raise RuntimeError(
        "mTLS collector did not become healthy in time. "
        f"Last error: {last_error}"
    )


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

    print()
    print(
        "="
        *
        128
    )

    print(
        "PHASE 15E2 — HTTPS/TLS + MUTUAL TLS AUTHENTICATION SMOKE TEST"
    )

    print(
        "="
        *
        128
    )

    print(
        f"Collector                   : {COLLECTOR_URL}"
    )

    print(
        f"Input                       : {INPUT_FILE}"
    )

    print(
        "Transport                   : HTTPS"
    )

    print(
        "Authentication              : MUTUAL TLS (mTLS)"
    )

    print(
        "Server certificate trust    : PRIVATE TEST CA"
    )

    print(
        "Client certificate required : YES"
    )

    print(
        "TLS private keys persisted  : NO (temporary test directory only)"
    )

    print(
        "Locked test                 : NO"
    )

    with tempfile.TemporaryDirectory(
        prefix=
            "phase15e2_mtls_"
    ) as temp_dir:
        tls_dir = Path(
            temp_dir
        )

        material = generate_local_mtls_material(
            tls_dir,
            server_dns_name=
                "localhost",
            server_ip=
                HOST,
            client_common_name=
                "phase15_client_3",
            valid_days=
                2,
        )

        client_ssl_context = build_mtls_client_context(
            ca_cert_file=
                str(
                    material.ca_cert
                ),

            client_cert_file=
                str(
                    material.client_cert
                ),

            client_key_file=
                str(
                    material.client_key
                ),
        )

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

            ssl_keyfile=
                str(
                    material.server_key
                ),

            ssl_certfile=
                str(
                    material.server_cert
                ),

            ssl_ca_certs=
                str(
                    material.ca_cert
                ),

            ssl_cert_reqs=
                ssl.CERT_REQUIRED,
        )

        server = uvicorn.Server(
            config
        )

        thread = threading.Thread(
            target=
                server.run,
            daemon=True,
        )

        thread.start()

        try:
            health_payload = wait_for_https_health(
                client_ssl_context=
                    client_ssl_context
            )

            print()
            print(
                "MTLS TESTS"
            )

            print(
                "-"
                *
                128
            )

            print(
                "[PASS] mtls_collector_health: HTTPS 200"
            )

            latencies: List[
                float
            ] = []

            accepted_alerts: List[
                Dict[
                    str,
                    Any,
                ]
            ] = []

            unique_passes = []

            for index in range(
                3
            ):
                alert = unique_alert(
                    template
                )

                result = send_privacy_safe_alert(
                    COLLECTOR_URL,
                    alert,
                    ssl_context=
                        client_ssl_context,
                )

                passed = (
                    result.status_code
                    ==
                    201
                )

                unique_passes.append(
                    passed
                )

                latencies.append(
                    result.round_trip_ms
                )

                accepted_alerts.append(
                    alert
                )

                print(
                    f"[{'PASS' if passed else 'FAIL'}] "
                    f"mtls_unique_send_{index + 1}: "
                    f"HTTP {result.status_code}, "
                    f"{result.round_trip_ms:.3f} ms"
                )

            # Duplicate rejection over mTLS.
            duplicate_result = send_privacy_safe_alert(
                COLLECTOR_URL,
                accepted_alerts[
                    0
                ],
                ssl_context=
                    client_ssl_context,
            )

            duplicate_pass = (
                duplicate_result.status_code
                ==
                409
            )

            print(
                f"[{'PASS' if duplicate_pass else 'FAIL'}] "
                f"duplicate_rejected_over_mtls: "
                f"HTTP {duplicate_result.status_code}"
            )

            # Digest mismatch over mTLS.
            digest_alert = unique_alert(
                template
            )

            digest_body = serialize_alert(
                digest_alert
            )

            with httpx.Client(
                verify=
                    client_ssl_context,

                timeout=
                    5.0,

                trust_env=
                    False,
            ) as client:
                digest_response = client.post(
                    COLLECTOR_URL,
                    content=
                        digest_body,
                    headers={
                        "Content-Type":
                            "application/json",

                        "X-Message-SHA256":
                            "0" * 64,
                    },
                )

            digest_pass = (
                digest_response.status_code
                ==
                400
            )

            print(
                f"[{'PASS' if digest_pass else 'FAIL'}] "
                f"digest_mismatch_rejected_over_mtls: "
                f"HTTP {digest_response.status_code}"
            )

            # ------------------------------------------------
            # Authentication test: trust server CA but DO NOT
            # present client certificate.
            # Expected: TLS handshake / request failure.
            # ------------------------------------------------

            no_client_context = ssl.create_default_context(
                ssl.Purpose.SERVER_AUTH,
                cafile=
                    str(
                        material.ca_cert
                    ),
            )

            no_client_context.minimum_version = (
                ssl.TLSVersion.TLSv1_2
            )

            no_client_cert_pass = False
            no_client_error = None

            try:
                with httpx.Client(
                    verify=
                        no_client_context,

                    timeout=
                        2.0,

                    trust_env=
                        False,
                ) as client:
                    response = client.get(
                        HEALTH_URL
                    )

                no_client_error = (
                    f"Unexpected HTTP {response.status_code}"
                )

            except Exception as exc:
                no_client_cert_pass = True
                no_client_error = (
                    type(
                        exc
                    ).__name__
                )

            print(
                f"[{'PASS' if no_client_cert_pass else 'FAIL'}] "
                "client_certificate_required"
            )

            # ------------------------------------------------
            # Server authentication test:
            # use system trust only; self-signed/private CA is
            # intentionally not trusted.
            # ------------------------------------------------

            untrusted_server_pass = False
            untrusted_server_error = None

            try:
                with httpx.Client(
                    verify=True,
                    timeout=2.0,
                    trust_env=False,
                ) as client:
                    response = client.get(
                        HEALTH_URL
                    )

                untrusted_server_error = (
                    f"Unexpected HTTP {response.status_code}"
                )

            except Exception as exc:
                untrusted_server_pass = True
                untrusted_server_error = (
                    type(
                        exc
                    ).__name__
                )

            print(
                f"[{'PASS' if untrusted_server_pass else 'FAIL'}] "
                "untrusted_server_certificate_rejected"
            )

            # Plain HTTP downgrade should fail because the TLS port
            # is not speaking HTTP and the sender policy does not
            # allow non-HTTPS except explicit E1 localhost mode.
            plain_http_sender_guard_pass = False

            try:
                send_privacy_safe_alert(
                    f"http://localhost:{PORT}/api/alerts",
                    unique_alert(
                        template
                    ),
                )

            except Exception:
                plain_http_sender_guard_pass = True

            print(
                f"[{'PASS' if plain_http_sender_guard_pass else 'FAIL'}] "
                "plain_http_downgrade_blocked"
            )

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
                3
            )

            print(
                f"[{'PASS' if stored_count_pass else 'FAIL'}] "
                f"stored_unique_alert_count: "
                f"{len(stored_lines)}"
            )

            all_passed = (
                all(
                    unique_passes
                )
                and
                duplicate_pass
                and
                digest_pass
                and
                no_client_cert_pass
                and
                untrusted_server_pass
                and
                plain_http_sender_guard_pass
                and
                stored_count_pass
            )

            audit = {
                "phase":
                    "15E2",

                "phase_name":
                    (
                        "HTTPS/TLS + Mutual TLS Authentication "
                        "Smoke Test"
                    ),

                "collector_url":
                    COLLECTOR_URL,

                "transport":
                    "HTTPS",

                "tls_enabled":
                    True,

                "mutual_tls":
                    True,

                "tls_minimum_version":
                    "TLSv1.2",

                "client_certificate_required":
                    True,

                "test_ca":
                    "ephemeral local private CA",

                "tls_private_keys_persisted":
                    False,

                "accepted_unique_alerts":
                    len(
                        accepted_alerts
                    ),

                "duplicate_rejection":
                    duplicate_pass,

                "digest_mismatch_rejection":
                    digest_pass,

                "missing_client_certificate_rejection":
                    no_client_cert_pass,

                "missing_client_certificate_error":
                    no_client_error,

                "untrusted_server_certificate_rejection":
                    untrusted_server_pass,

                "untrusted_server_error":
                    untrusted_server_error,

                "plain_http_downgrade_blocked":
                    plain_http_sender_guard_pass,

                "stored_unique_alert_count":
                    len(
                        stored_lines
                    ),

                "mean_sender_round_trip_ms":
                    statistics.mean(
                        latencies
                    ),

                "median_sender_round_trip_ms":
                    statistics.median(
                        latencies
                    ),

                "collector_health":
                    health_payload,

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
                    "15E2",

                "transport":
                    "HTTPS/TLS",

                "authentication":
                    "mutual TLS",

                "server_authentication":
                    (
                        "server certificate verified against "
                        "trusted CA"
                    ),

                "client_authentication":
                    (
                        "collector requires client certificate "
                        "signed by trusted CA"
                    ),

                "tls_minimum_version":
                    "TLSv1.2",

                "test_certificate_lifecycle":
                    (
                        "generated in temporary directory and "
                        "deleted after smoke test"
                    ),

                "private_keys_written_to_results_or_artifacts":
                    False,

                "message_sha256_authentication":
                    False,

                "message_sha256_purpose":
                    "additional payload integrity check only",

                "production_note":
                    (
                        "Replace ephemeral local CA/certs with "
                        "organization-managed PKI, certificate "
                        "rotation, and protected private keys."
                    ),

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
                128
            )

            print(
                "PHASE 15E2 COMPLETE"
            )

            print(
                "="
                *
                128
            )

            print(
                f"HTTPS transport                 : PASS"
            )

            print(
                f"Server certificate verification : "
                f"{'PASS' if untrusted_server_pass else 'FAIL'}"
            )

            print(
                f"Client certificate required     : "
                f"{'PASS' if no_client_cert_pass else 'FAIL'}"
            )

            print(
                f"Accepted unique alerts          : "
                f"{len(accepted_alerts)}/3"
            )

            print(
                f"Duplicate rejection             : "
                f"{'PASS' if duplicate_pass else 'FAIL'}"
            )

            print(
                f"Digest mismatch rejection       : "
                f"{'PASS' if digest_pass else 'FAIL'}"
            )

            print(
                f"HTTP downgrade blocked          : "
                f"{'PASS' if plain_http_sender_guard_pass else 'FAIL'}"
            )

            print(
                f"Stored unique alerts            : "
                f"{len(stored_lines)}"
            )

            print(
                f"Mean mTLS sender round-trip     : "
                f"{audit['mean_sender_round_trip_ms']:.3f} ms"
            )

            print(
                f"TLS private keys persisted      : NO"
            )

            print(
                f"Audit                           : {AUDIT_FILE}"
            )

            print(
                f"Manifest                        : {MANIFEST_FILE}"
            )

            print(
                f"Locked test used                : NO"
            )

            print()

            print(
                f"STATUS                          : "
                f"{'PASS' if all_passed else 'FAIL'}"
            )

            print()

            print(
                "IMPORTANT:"
            )

            print(
                "This proves local mutual-TLS behavior using "
                "short-lived test certificates."
            )

            print(
                "It is not a production PKI deployment."
            )

            print()

            print(
                "NEXT:"
            )

            print(
                "Phase 15F — central alert persistence/indexing "
                "and multi-client privacy-aware alert ingestion."
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
