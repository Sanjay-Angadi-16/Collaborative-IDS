from __future__ import annotations

import csv
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(
            PROJECT_ROOT
        ),
    )


from src.alert_sharing.alert_schema import (  # noqa: E402
    ALLOWED_FIELDS,
)
from src.alert_sharing.pseudonymizer import (  # noqa: E402
    DEFAULT_DIGEST_BYTES,
    DEFAULT_KEY_ID,
    PHASE15_HMAC_ENV,
    HMACPseudonymizer,
    encode_hmac_key_b64,
    generate_hmac_key_bytes,
)
from src.alert_sharing.privacy_filter import (  # noqa: E402
    SHARED_ALERT_ALLOWLIST,
    filter_local_alert_for_sharing,
)
from src.alert_sharing.privacy_validator import (  # noqa: E402
    PrivacyLeakageError,
    audit_shared_alert,
    validate_no_privacy_leakage,
)


PHASE_NAME = (
    "PHASE 15C — ALLOWLIST PRIVACY FILTER + LEAKAGE AUDIT"
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

AUDIT_FILE = (
    RESULT_ROOT
    / "phase15c_privacy_filter_audit.json"
)

FILTERED_ALERTS_FILE = (
    RESULT_ROOT
    / "phase15c_filtered_alerts.jsonl"
)

SIZE_METRICS_FILE = (
    RESULT_ROOT
    / "phase15c_alert_size_metrics.csv"
)

MANIFEST_FILE = (
    ARTIFACT_ROOT
    / "phase15c_privacy_filter_manifest.json"
)


RAW_SOURCE_IP = "192.168.10.25"
RAW_DESTINATION_IP = "10.20.30.40"
RAW_SOURCE_MAC = "00:11:22:33:44:55"
RAW_PAYLOAD = "GET /private/internal/resource HTTP/1.1"
RAW_HOSTNAME = "finance-workstation-07"


def save_json(
    path: Path,
    payload: Dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=4,
        )


def create_pseudonymizer():
    """
    Same Phase15B rule:
    - use PHASE15_HMAC_KEY_B64 if configured;
    - otherwise use an ephemeral test-only in-memory key.
    """
    env_value = os.getenv(
        PHASE15_HMAC_ENV
    )

    if env_value:
        return (
            HMACPseudonymizer.from_environment(
                env_name=
                    PHASE15_HMAC_ENV,

                key_id=
                    DEFAULT_KEY_ID,

                digest_bytes=
                    DEFAULT_DIGEST_BYTES,
            ),
            "ENVIRONMENT_STABLE_KEY",
            False,
        )

    key = generate_hmac_key_bytes(
        32
    )

    return (
        HMACPseudonymizer(
            key=
                key,

            key_id=
                DEFAULT_KEY_ID,

            digest_bytes=
                DEFAULT_DIGEST_BYTES,
        ),
        "EPHEMERAL_TEST_KEY",
        True,
    )


def build_sensitive_fast_local_alert() -> Dict[str, Any]:
    """
    Representative Phase14G2 FAST alert plus intentionally sensitive
    local-only fields. Phase15C must NOT copy those fields outbound.
    """
    return {
        "timestamp_utc":
            "2026-09-14T08:05:39.999435+00:00",

        "client_id":
            3,

        "classification":
            "PortScan",

        "severity":
            "HIGH",

        "risk_score":
            None,

        "confidence":
            0.9998209468981613,

        "evidence_score":
            0.9998209468981613,

        "decision_source":
            "Random Forest",

        "processing_path":
            "FAST_ATTACK_PATH",

        "risk_mode":
            "FAST_PATH_RF_CONFIDENCE",

        # Local-only details from Phase14 / network context:
        "source_ip":
            RAW_SOURCE_IP,

        "destination_ip":
            RAW_DESTINATION_IP,

        "source_mac":
            RAW_SOURCE_MAC,

        "hostname":
            RAW_HOSTNAME,

        "payload":
            RAW_PAYLOAD,

        "flow_features":
            [0.12, -0.42, 0.77, 0.01],

        "sequence":
            [
                [0.1, 0.2],
                [0.3, 0.4],
            ],

        "rf":
            {
                "model":
                    "Random Forest",

                "predicted_class":
                    "PortScan",

                "confidence":
                    0.9998209468981613,
            },

        "autoencoder":
            None,

        "fedprox70":
            None,

        "cnn_bilstm":
            None,

        "internal_debug":
            {
                "raw_feature_count":
                    70,

                "note":
                    "local only",
            },
    }


def build_sensitive_deep_local_alert() -> Dict[str, Any]:
    return {
        "timestamp_utc":
            "2026-09-14T08:10:00+00:00",

        "client_id":
            2,

        "classification":
            "DDoS",

        "severity":
            "CRITICAL",

        "risk_score":
            0.87,

        "confidence":
            0.94,

        "evidence_score":
            0.87,

        "decision_source":
            "CNN-BiLSTM",

        "processing_path":
            "DEEP_ANALYSIS",

        "risk_mode":
            "FULL_FOUR_MODEL_FUSION",

        # Deliberately sensitive fields:
        "source_ip":
            RAW_SOURCE_IP,

        "destination_ip":
            RAW_DESTINATION_IP,

        "packet_payload":
            RAW_PAYLOAD,

        "feature_vector":
            [0.2, 0.3, 0.4],

        "cnn_sequence":
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],

        "model_state_dict":
            {
                "example":
                    [1, 2, 3]
            },

        "rf":
            {
                "predicted_class":
                    "BENIGN",

                "confidence":
                    0.87,
            },

        "autoencoder":
            {
                "anomaly_score":
                    0.91,
            },

        "fedprox70":
            {
                "predicted_class":
                    "DDoS",

                "confidence":
                    0.90,
            },

        "cnn_bilstm":
            {
                "predicted_class":
                    "DDoS",

                "confidence":
                    0.94,
            },
    }


def main() -> None:
    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARTIFACT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        pseudonymizer,
        key_mode,
        ephemeral,
    ) = create_pseudonymizer()

    print()
    print(
        "="
        *
        124
    )

    print(
        PHASE_NAME
    )

    print(
        "="
        *
        124
    )

    print(
        f"Filter policy                  : ALLOWLIST-ONLY"
    )

    print(
        f"Shared fields                  : {len(SHARED_ALERT_ALLOWLIST)}"
    )

    print(
        f"Pseudonymization               : HMAC-SHA256"
    )

    print(
        f"Key mode                       : {key_mode}"
    )

    print(
        f"Require src/dst tokens         : YES"
    )

    print(
        f"Raw payload/features/sequence  : DROP"
    )

    print(
        f"Locked test                    : NO"
    )

    if ephemeral:
        print()
        print(
            "NOTE: ephemeral in-memory HMAC key is used for this test."
        )

    tests: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    def record(
        name: str,
        passed: bool,
        detail: str,
    ) -> None:
        tests.append(
            {
                "test":
                    name,

                "passed":
                    bool(
                        passed
                    ),

                "detail":
                    detail,
            }
        )

        print(
            f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}"
        )

    fast_local = build_sensitive_fast_local_alert()
    deep_local = build_sensitive_deep_local_alert()

    fast_original_copy = deepcopy(
        fast_local
    )

    deep_original_copy = deepcopy(
        deep_local
    )

    key_b64_for_audit = encode_hmac_key_b64(
        pseudonymizer.key
    )

    known_raw_identifiers = [
        RAW_SOURCE_IP,
        RAW_DESTINATION_IP,
        RAW_SOURCE_MAC,
        RAW_PAYLOAD,
        RAW_HOSTNAME,
    ]

    print()
    print(
        "FILTER + LEAKAGE TESTS"
    )

    print(
        "-"
        *
        124
    )

    # --------------------------------------------------------
    # Filter FAST
    # --------------------------------------------------------

    fast_start = time.perf_counter()

    fast_shared = filter_local_alert_for_sharing(
        fast_local,

        pseudonymizer=
            pseudonymizer,

        source_ip=
            RAW_SOURCE_IP,

        destination_ip=
            RAW_DESTINATION_IP,

        protocol=
            "TCP",

        destination_port=
            443,
    )

    fast_filter_ms = (
        time.perf_counter()
        -
        fast_start
    ) * 1000.0

    fast_payload = fast_shared.to_dict()

    fast_report = audit_shared_alert(
        fast_payload,

        known_raw_identifiers=
            known_raw_identifiers,

        known_secret_literals=
            [
                key_b64_for_audit
            ],

        require_link_tokens=
            True,
    )

    record(
        "fast_alert_leakage_audit",
        fast_report.passed,
        "Filtered FAST alert contains no detected sensitive material.",
    )

    record(
        "fast_alert_phase15a_schema",
        fast_report.schema_valid,
        "Filtered FAST alert preserves the frozen Phase15A schema.",
    )

    record(
        "fast_raw_fields_removed",
        (
            "source_ip"
            not in
            fast_payload
            and
            "destination_ip"
            not in
            fast_payload
            and
            "payload"
            not in
            fast_payload
            and
            "flow_features"
            not in
            fast_payload
            and
            "sequence"
            not in
            fast_payload
        ),
        "Raw IPs, payload, flow features and sequence are absent.",
    )

    record(
        "fast_nested_model_details_removed",
        all(
            field
            not in
            fast_payload
            for field
            in (
                "rf",
                "autoencoder",
                "fedprox70",
                "cnn_bilstm",
                "internal_debug",
            )
        ),
        "Nested model/debug objects remain local.",
    )

    record(
        "fast_phase14g2_semantics",
        (
            fast_payload[
                "risk_score"
            ]
            is None
            and
            fast_payload[
                "severity"
            ]
            ==
            "HIGH"
            and
            fast_payload[
                "evidence_score"
            ]
            ==
            fast_payload[
                "confidence"
            ]
            and
            fast_payload[
                "risk_mode"
            ]
            ==
            "FAST_PATH_RF_CONFIDENCE"
        ),
        "FAST risk/evidence/severity semantics are unchanged.",
    )

    # --------------------------------------------------------
    # Filter DEEP
    # --------------------------------------------------------

    deep_start = time.perf_counter()

    deep_shared = filter_local_alert_for_sharing(
        deep_local,

        pseudonymizer=
            pseudonymizer,

        source_ip=
            RAW_SOURCE_IP,

        destination_ip=
            RAW_DESTINATION_IP,

        protocol=
            "TCP",

        destination_port=
            443,
    )

    deep_filter_ms = (
        time.perf_counter()
        -
        deep_start
    ) * 1000.0

    deep_payload = deep_shared.to_dict()

    deep_report = audit_shared_alert(
        deep_payload,

        known_raw_identifiers=
            known_raw_identifiers,

        known_secret_literals=
            [
                key_b64_for_audit
            ],

        require_link_tokens=
            True,
    )

    record(
        "deep_alert_leakage_audit",
        deep_report.passed,
        "Filtered DEEP alert contains no detected sensitive material.",
    )

    record(
        "deep_alert_phase15a_schema",
        deep_report.schema_valid,
        "Filtered DEEP alert preserves the frozen Phase15A schema.",
    )

    record(
        "deep_raw_ml_material_removed",
        all(
            field
            not in
            deep_payload
            for field
            in (
                "packet_payload",
                "feature_vector",
                "cnn_sequence",
                "model_state_dict",
            )
        ),
        "Raw ML/network material remains local.",
    )

    record(
        "deep_phase14g2_semantics",
        (
            deep_payload[
                "risk_score"
            ]
            ==
            0.87
            and
            deep_payload[
                "evidence_score"
            ]
            ==
            0.87
            and
            deep_payload[
                "severity"
            ]
            ==
            "CRITICAL"
            and
            deep_payload[
                "risk_mode"
            ]
            ==
            "FULL_FOUR_MODEL_FUSION"
        ),
        "DEEP fused-risk semantics are unchanged.",
    )

    # --------------------------------------------------------
    # Linkability without raw identifiers
    # --------------------------------------------------------

    record(
        "same_source_cross_alert_linkability",
        (
            fast_payload[
                "source_token"
            ]
            ==
            deep_payload[
                "source_token"
            ]
        ),
        "Same raw source links across alerts through the same pseudonym.",
    )

    record(
        "same_destination_cross_alert_linkability",
        (
            fast_payload[
                "destination_token"
            ]
            ==
            deep_payload[
                "destination_token"
            ]
        ),
        "Same raw destination links across alerts through the same pseudonym.",
    )

    # --------------------------------------------------------
    # Input mutation check
    # --------------------------------------------------------

    record(
        "local_fast_alert_not_mutated",
        fast_local
        ==
        fast_original_copy,
        "Privacy filtering does not mutate the original local FAST alert.",
    )

    record(
        "local_deep_alert_not_mutated",
        deep_local
        ==
        deep_original_copy,
        "Privacy filtering does not mutate the original local DEEP alert.",
    )

    # --------------------------------------------------------
    # Explicit allowlist check
    # --------------------------------------------------------

    record(
        "outbound_keys_are_allowlisted",
        (
            set(
                fast_payload.keys()
            )
            <=
            SHARED_ALERT_ALLOWLIST
            and
            set(
                deep_payload.keys()
            )
            <=
            SHARED_ALERT_ALLOWLIST
        ),
        "Every outbound top-level key is explicitly allowlisted.",
    )

    # --------------------------------------------------------
    # Tamper tests against leakage validator
    # --------------------------------------------------------

    tampered_raw_ip = dict(
        fast_payload
    )
    tampered_raw_ip[
        "source_ip"
    ] = RAW_SOURCE_IP

    tampered_report = audit_shared_alert(
        tampered_raw_ip,

        known_raw_identifiers=
            known_raw_identifiers,

        require_link_tokens=
            True,
    )

    record(
        "validator_rejects_raw_ip_field",
        not tampered_report.passed,
        "Leakage validator rejects a reintroduced raw source_ip field.",
    )

    hidden_ip = dict(
        fast_payload
    )
    hidden_ip[
        "classification"
    ] = (
        "PortScan "
        +
        RAW_SOURCE_IP
    )

    hidden_ip_report = audit_shared_alert(
        hidden_ip,

        known_raw_identifiers=
            known_raw_identifiers,

        require_link_tokens=
            True,
    )

    record(
        "validator_detects_ip_hidden_in_allowed_field",
        (
            not hidden_ip_report.passed
            and
            RAW_SOURCE_IP
            in
            hidden_ip_report.raw_ip_literals
        ),
        "Leakage validator scans values, not only field names.",
    )

    missing_token = dict(
        fast_payload
    )
    missing_token[
        "source_token"
    ] = None

    missing_token_report = audit_shared_alert(
        missing_token,

        known_raw_identifiers=
            known_raw_identifiers,

        require_link_tokens=
            True,
    )

    record(
        "validator_requires_link_tokens",
        not missing_token_report.passed,
        "Phase15C shared-alert policy requires pseudonymous link tokens.",
    )

    # --------------------------------------------------------
    # Alert size audit
    # --------------------------------------------------------

    fast_local_json = json.dumps(
        fast_local,
        separators=(
            ",",
            ":",
        ),
    ).encode(
        "utf-8"
    )

    fast_shared_json = json.dumps(
        fast_payload,
        separators=(
            ",",
            ":",
        ),
    ).encode(
        "utf-8"
    )

    deep_local_json = json.dumps(
        deep_local,
        separators=(
            ",",
            ":",
        ),
    ).encode(
        "utf-8"
    )

    deep_shared_json = json.dumps(
        deep_payload,
        separators=(
            ",",
            ":",
        ),
    ).encode(
        "utf-8"
    )

    size_rows = []

    for alert_type, local_bytes, shared_bytes, filter_ms in (
        (
            "FAST",
            len(
                fast_local_json
            ),
            len(
                fast_shared_json
            ),
            fast_filter_ms,
        ),
        (
            "DEEP",
            len(
                deep_local_json
            ),
            len(
                deep_shared_json
            ),
            deep_filter_ms,
        ),
    ):
        reduction = (
            (
                local_bytes
                -
                shared_bytes
            )
            /
            local_bytes
            *
            100.0
        )

        size_rows.append(
            {
                "alert_type":
                    alert_type,

                "local_json_bytes":
                    local_bytes,

                "shared_json_bytes":
                    shared_bytes,

                "size_reduction_percent":
                    reduction,

                "filter_plus_schema_ms":
                    filter_ms,
            }
        )

    with open(
        SIZE_METRICS_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                size_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            size_rows
        )

    # --------------------------------------------------------
    # Save only filtered safe alerts
    # --------------------------------------------------------

    with open(
        FILTERED_ALERTS_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                fast_payload,
                separators=(
                    ",",
                    ":",
                ),
            )
            +
            "\n"
        )

        file.write(
            json.dumps(
                deep_payload,
                separators=(
                    ",",
                    ":",
                ),
            )
            +
            "\n"
        )

    passed = sum(
        1
        for row
        in tests
        if row[
            "passed"
        ]
    )

    failed = len(
        tests
    ) - passed

    audit_payload = {
        "phase":
            "15C",

        "phase_name":
            PHASE_NAME,

        "filter_policy":
            "ALLOWLIST_ONLY",

        "shared_field_count":
            len(
                SHARED_ALERT_ALLOWLIST
            ),

        "pseudonymization":
            "HMAC-SHA256",

        "key_mode":
            key_mode,

        "raw_identifiers_written_to_results":
            False,

        "raw_packet_or_payload_written":
            False,

        "raw_features_written":
            False,

        "cnn_sequences_written":
            False,

        "model_state_written":
            False,

        "require_source_destination_tokens":
            True,

        "tests_total":
            len(
                tests
            ),

        "tests_passed":
            passed,

        "tests_failed":
            failed,

        "all_tests_passed":
            failed
            ==
            0,

        "fast_leakage_report":
            fast_report.to_dict(),

        "deep_leakage_report":
            deep_report.to_dict(),

        "size_metrics":
            size_rows,

        "tests":
            tests,

        "locked_test_used":
            False,
    }

    save_json(
        AUDIT_FILE,
        audit_payload,
    )

    manifest = {
        "phase":
            "15C",

        "policy":
            "explicit allowlist construction",

        "shared_fields":
            sorted(
                SHARED_ALERT_ALLOWLIST
            ),

        "unknown_source_fields":
            "DROP",

        "nested_model_outputs":
            "DROP",

        "raw_identifiers":
            "PSEUDONYMIZE, THEN DROP",

        "raw_payload":
            "DROP",

        "raw_packets":
            "DROP",

        "raw_flow_features":
            "DROP",

        "cnn_sequence":
            "DROP",

        "model_parameters":
            "DROP",

        "privacy_validator":
            {
                "top_level_allowlist":
                    True,

                "forbidden_field_scan":
                    True,

                "raw_ipv4_value_scan":
                    True,

                "mac_literal_scan":
                    True,

                "known_raw_identifier_scan":
                    True,

                "known_secret_literal_scan":
                    True,

                "phase15a_schema_validation":
                    True,

                "pseudonymous_link_tokens_required":
                    True,
            },

        "phase14g2_semantics_preserved":
            True,

        "locked_test_used":
            False,
    }

    save_json(
        MANIFEST_FILE,
        manifest,
    )

    print()
    print(
        "="
        *
        124
    )

    print(
        "PHASE 15C COMPLETE"
    )

    print(
        "="
        *
        124
    )

    print(
        f"Tests                          : "
        f"{passed}/{len(tests)} passed"
    )

    print(
        f"FAST leakage audit             : "
        f"{'PASS' if fast_report.passed else 'FAIL'}"
    )

    print(
        f"DEEP leakage audit             : "
        f"{'PASS' if deep_report.passed else 'FAIL'}"
    )

    print(
        f"FAST filter + schema latency   : "
        f"{fast_filter_ms:.4f} ms"
    )

    print(
        f"DEEP filter + schema latency   : "
        f"{deep_filter_ms:.4f} ms"
    )

    print()

    for row in size_rows:
        print(
            f"{row['alert_type']:4s} JSON size                 : "
            f"{row['local_json_bytes']} -> "
            f"{row['shared_json_bytes']} bytes "
            f"({row['size_reduction_percent']:.2f}% reduction)"
        )

    print()

    print(
        f"Audit                          : {AUDIT_FILE}"
    )

    print(
        f"Filtered alerts JSONL          : {FILTERED_ALERTS_FILE}"
    )

    print(
        f"Size metrics                   : {SIZE_METRICS_FILE}"
    )

    print(
        f"Manifest                       : {MANIFEST_FILE}"
    )

    print()

    print(
        f"STATUS                         : "
        f"{'PASS' if failed == 0 else 'FAIL'}"
    )

    print()

    print(
        "PRIVACY GUARANTEE TESTED:"
    )

    print(
        "Raw identifiers and local ML/network details stay local."
    )

    print(
        "Only schema-valid allowlisted metadata plus HMAC tokens "
        "are emitted for sharing."
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Phase 15D — real Phase14G2 alert -> privacy-safe serializer "
        "and sharing integration."
    )

    if failed != 0:
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()
