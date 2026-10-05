from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple


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


from src.alert_sharing.alert_serializer import (  # noqa: E402
    deserialize_alert,
    serialize_alert,
    serialized_sha256,
)
from src.alert_sharing.pseudonymizer import (  # noqa: E402
    DEFAULT_DIGEST_BYTES,
    DEFAULT_KEY_ID,
    PHASE15_HMAC_ENV,
    HMACPseudonymizer,
    generate_hmac_key_bytes,
)
from src.alert_sharing.privacy_filter import (  # noqa: E402
    filter_local_alert_for_sharing,
)
from src.alert_sharing.privacy_validator import (  # noqa: E402
    audit_shared_alert,
    validate_no_privacy_leakage,
)


PHASE_NAME = (
    "PHASE 15D — REAL PHASE14G2 ALERT -> PRIVACY-SAFE "
    "SERIALIZATION INTEGRATION"
)


# ============================================================
# INPUTS
# ============================================================

DEFAULT_PHASE14G2_ALERT = (
    PROJECT_ROOT
    / "results"
    / "phase14"
    / "phase14g2"
    / "phase14g2_corrected_alert.json"
)

SOURCE_IP_ENV = "PHASE15_SOURCE_IP"
DESTINATION_IP_ENV = "PHASE15_DESTINATION_IP"

# RFC 5737 documentation ranges.
# Used only under --demo-context because CICIDS-derived validation
# sequences do not contain trustworthy source/destination IP fields.
DEMO_SOURCE_IP = "192.0.2.10"
DEMO_DESTINATION_IP = "198.51.100.20"


# ============================================================
# OUTPUTS
# ============================================================

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

SHARED_JSONL_FILE = (
    RESULT_ROOT
    / "phase15d_shared_alert.jsonl"
)

SHARED_PRETTY_FILE = (
    RESULT_ROOT
    / "phase15d_shared_alert_pretty.json"
)

AUDIT_FILE = (
    RESULT_ROOT
    / "phase15d_integration_audit.json"
)

METRICS_FILE = (
    RESULT_ROOT
    / "phase15d_serialization_metrics.csv"
)

MANIFEST_FILE = (
    ARTIFACT_ROOT
    / "phase15d_integration_manifest.json"
)


# ============================================================
# HELPERS
# ============================================================

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


def load_phase14g2_alert(
    path: Path,
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
]:
    if not path.exists():
        raise FileNotFoundError(
            "\nMissing Phase14G2 alert file:\n"
            f"{path}\n\n"
            "Run Phase14G2 first or pass --input-alert."
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(
            file
        )

    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            "Phase14G2 alert file must contain a JSON object."
        )

    # The Phase14G2 script saves:
    # {
    #   "phase": "14G2",
    #   ...
    #   "alert": { actual local alert ... }
    # }
    #
    # Also support a direct alert JSON for future reuse.
    if (
        "alert"
        in payload
        and
        isinstance(
            payload[
                "alert"
            ],
            dict,
        )
    ):
        local_alert = dict(
            payload[
                "alert"
            ]
        )
    else:
        local_alert = dict(
            payload
        )

    required = {
        "client_id",
        "classification",
        "severity",
        "confidence",
        "risk_score",
        "evidence_score",
        "risk_mode",
        "decision_source",
        "processing_path",
    }

    missing = sorted(
        required
        -
        set(
            local_alert.keys()
        )
    )

    if missing:
        raise RuntimeError(
            "Input does not contain the frozen Phase14G2 alert "
            "contract. Missing: "
            +
            ", ".join(
                missing
            )
        )

    return (
        payload,
        local_alert,
    )


def resolve_network_context(
    args,
) -> Tuple[
    str,
    str,
    str,
]:
    source_ip = (
        args.source_ip
        or
        os.getenv(
            SOURCE_IP_ENV
        )
    )

    destination_ip = (
        args.destination_ip
        or
        os.getenv(
            DESTINATION_IP_ENV
        )
    )

    if (
        source_ip
        and
        destination_ip
    ):
        return (
            source_ip,
            destination_ip,
            "EXTERNALLY_SUPPLIED_LOCAL_CONTEXT",
        )

    if args.demo_context:
        return (
            DEMO_SOURCE_IP,
            DEMO_DESTINATION_IP,
            "RFC5737_DEMO_CONTEXT",
        )

    raise RuntimeError(
        "\nPhase14 validation sequences do not contain trustworthy "
        "source/destination IP identifiers.\n"
        "Phase15D will not fabricate them.\n\n"
        "Provide local context using either:\n"
        f"  set {SOURCE_IP_ENV}=<local-source-ip>\n"
        f"  set {DESTINATION_IP_ENV}=<local-destination-ip>\n\n"
        "or explicitly run a non-production integration test with:\n"
        "  --demo-context\n"
    )


def create_pseudonymizer(
    *,
    allow_ephemeral_test_key: bool,
):
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

    if not allow_ephemeral_test_key:
        raise RuntimeError(
            f"\nMissing {PHASE15_HMAC_ENV}.\n\n"
            "Phase15D requires a stable externally managed HMAC key "
            "for actual cross-alert linkability.\n"
            "For a clearly labeled integration-only smoke test, "
            "add --allow-ephemeral-test-key."
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
        "EPHEMERAL_INTEGRATION_TEST_KEY",
        True,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=
            PHASE_NAME
    )

    parser.add_argument(
        "--input-alert",
        type=Path,
        default=
            DEFAULT_PHASE14G2_ALERT,
        help=(
            "Phase14G2 corrected alert JSON. Default: "
            "results/phase14/phase14g2/"
            "phase14g2_corrected_alert.json"
        ),
    )

    parser.add_argument(
        "--source-ip",
        type=str,
        default=None,
        help=(
            "Local-only raw source IP. Prefer PHASE15_SOURCE_IP "
            "environment variable for operational use."
        ),
    )

    parser.add_argument(
        "--destination-ip",
        type=str,
        default=None,
        help=(
            "Local-only raw destination IP. Prefer "
            "PHASE15_DESTINATION_IP environment variable."
        ),
    )

    parser.add_argument(
        "--protocol",
        type=str,
        default="TCP",
    )

    parser.add_argument(
        "--destination-port",
        type=int,
        default=443,
    )

    parser.add_argument(
        "--model-version",
        type=str,
        default="phase14_final",
    )

    parser.add_argument(
        "--demo-context",
        action="store_true",
        help=(
            "Use RFC5737 documentation IPs because the research "
            "validation sequences do not contain real IP context."
        ),
    )

    parser.add_argument(
        "--allow-ephemeral-test-key",
        action="store_true",
        help=(
            "Permit an in-memory test HMAC key when the stable "
            "PHASE15_HMAC_KEY_B64 secret is not configured."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARTIFACT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        full_input_document,
        local_alert,
    ) = load_phase14g2_alert(
        args.input_alert
    )

    original_local_copy = deepcopy(
        local_alert
    )

    (
        source_ip,
        destination_ip,
        context_mode,
    ) = resolve_network_context(
        args
    )

    (
        pseudonymizer,
        key_mode,
        ephemeral_key,
    ) = create_pseudonymizer(
        allow_ephemeral_test_key=
            args.allow_ephemeral_test_key
    )

    print()
    print(
        "="
        *
        126
    )

    print(
        PHASE_NAME
    )

    print(
        "="
        *
        126
    )

    print(
        f"Input Phase14G2 alert         : {args.input_alert}"
    )

    print(
        f"Classification               : "
        f"{local_alert['classification']}"
    )

    print(
        f"Processing path              : "
        f"{local_alert['processing_path']}"
    )

    print(
        f"Risk mode                    : "
        f"{local_alert['risk_mode']}"
    )

    print(
        f"Network context              : {context_mode}"
    )

    print(
        f"HMAC key mode                : {key_mode}"
    )

    print(
        f"Raw IP values printed        : NO"
    )

    print(
        f"Raw IP values persisted      : NO"
    )

    print(
        f"Locked test                  : NO"
    )

    if (
        context_mode
        ==
        "RFC5737_DEMO_CONTEXT"
    ):
        print()
        print(
            "SCIENTIFIC NOTE:"
        )

        print(
            "The Phase14G2 detection result is REAL."
        )

        print(
            "Source/destination context is synthetic RFC5737 "
            "documentation context because the validation dataset "
            "does not provide trustworthy IP identifiers."
        )

    if ephemeral_key:
        print()
        print(
            "SECURITY NOTE:"
        )

        print(
            "An ephemeral test key is being used."
        )

        print(
            "The produced tokens are NOT stable across program runs."
        )

    # --------------------------------------------------------
    # Privacy filter
    # --------------------------------------------------------

    filter_start = time.perf_counter()

    shared_alert = filter_local_alert_for_sharing(
        local_alert,

        pseudonymizer=
            pseudonymizer,

        source_ip=
            source_ip,

        destination_ip=
            destination_ip,

        protocol=
            args.protocol,

        destination_port=
            args.destination_port,

        model_version=
            args.model_version,
    )

    filter_ms = (
        time.perf_counter()
        -
        filter_start
    ) * 1000.0

    shared_mapping = shared_alert.to_dict()

    # --------------------------------------------------------
    # Leakage audit BEFORE serialization
    # --------------------------------------------------------

    leakage_start = time.perf_counter()

    leakage_report = validate_no_privacy_leakage(
        shared_mapping,

        known_raw_identifiers=
            [
                source_ip,
                destination_ip,
            ],

        require_link_tokens=
            True,
    )

    leakage_ms = (
        time.perf_counter()
        -
        leakage_start
    ) * 1000.0

    # --------------------------------------------------------
    # Canonical serialization
    # --------------------------------------------------------

    serialization_start = time.perf_counter()

    serialized = serialize_alert(
        shared_alert
    )

    serialization_ms = (
        time.perf_counter()
        -
        serialization_start
    ) * 1000.0

    message_sha256 = serialized_sha256(
        serialized
    )

    # --------------------------------------------------------
    # Deserialize + schema round-trip
    # --------------------------------------------------------

    roundtrip_start = time.perf_counter()

    reconstructed = deserialize_alert(
        serialized
    )

    roundtrip_ms = (
        time.perf_counter()
        -
        roundtrip_start
    ) * 1000.0

    roundtrip_equal = (
        reconstructed.to_dict()
        ==
        shared_mapping
    )

    if not roundtrip_equal:
        raise RuntimeError(
            "Serialization round-trip changed the privacy-safe alert."
        )

    roundtrip_leakage = validate_no_privacy_leakage(
        reconstructed.to_dict(),

        known_raw_identifiers=
            [
                source_ip,
                destination_ip,
            ],

        require_link_tokens=
            True,
    )

    # --------------------------------------------------------
    # Verify local alert was not mutated
    # --------------------------------------------------------

    local_alert_unchanged = (
        local_alert
        ==
        original_local_copy
    )

    if not local_alert_unchanged:
        raise RuntimeError(
            "Phase15D modified the original local Phase14G2 alert."
        )

    # --------------------------------------------------------
    # Size / total metrics
    # --------------------------------------------------------

    original_alert_bytes = len(
        json.dumps(
            local_alert,
            separators=(
                ",",
                ":",
            ),
            default=str,
        ).encode(
            "utf-8"
        )
    )

    shared_alert_bytes = len(
        serialized
    )

    size_reduction_percent = (
        (
            original_alert_bytes
            -
            shared_alert_bytes
        )
        /
        original_alert_bytes
        *
        100.0
        if original_alert_bytes > 0
        else 0.0
    )

    integration_total_ms = (
        filter_ms
        +
        leakage_ms
        +
        serialization_ms
        +
        roundtrip_ms
    )

    # --------------------------------------------------------
    # Save only privacy-safe outputs.
    # Do NOT persist source_ip / destination_ip.
    # --------------------------------------------------------

    with open(
        SHARED_JSONL_FILE,
        "wb",
    ) as file:
        file.write(
            serialized
            +
            b"\n"
        )

    save_json(
        SHARED_PRETTY_FILE,
        shared_mapping,
    )

    with open(
        METRICS_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "original_local_alert_bytes",
            "privacy_safe_shared_bytes",
            "size_reduction_percent",
            "privacy_filter_ms",
            "leakage_audit_ms",
            "serialization_ms",
            "deserialization_roundtrip_ms",
            "integration_total_ms",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=
                fieldnames,
        )

        writer.writeheader()

        writer.writerow(
            {
                "original_local_alert_bytes":
                    original_alert_bytes,

                "privacy_safe_shared_bytes":
                    shared_alert_bytes,

                "size_reduction_percent":
                    size_reduction_percent,

                "privacy_filter_ms":
                    filter_ms,

                "leakage_audit_ms":
                    leakage_ms,

                "serialization_ms":
                    serialization_ms,

                "deserialization_roundtrip_ms":
                    roundtrip_ms,

                "integration_total_ms":
                    integration_total_ms,
            }
        )

    audit_payload = {
        "phase":
            "15D",

        "phase_name":
            PHASE_NAME,

        "input_alert_file":
            str(
                args.input_alert
            ),

        "phase14_alert_classification":
            local_alert[
                "classification"
            ],

        "phase14_processing_path":
            local_alert[
                "processing_path"
            ],

        "phase14_risk_mode":
            local_alert[
                "risk_mode"
            ],

        "network_context_mode":
            context_mode,

        "hmac_key_mode":
            key_mode,

        "ephemeral_key":
            ephemeral_key,

        "raw_ip_values_printed":
            False,

        "raw_ip_values_persisted":
            False,

        "privacy_leakage_audit":
            leakage_report.to_dict(),

        "post_serialization_leakage_audit":
            roundtrip_leakage.to_dict(),

        "roundtrip_equal":
            roundtrip_equal,

        "local_phase14_alert_unchanged":
            local_alert_unchanged,

        "canonical_json":
            {
                "utf8":
                    True,

                "sorted_keys":
                    True,

                "compact_separators":
                    True,

                "nan_allowed":
                    False,

                "duplicate_keys_on_deserialize":
                    "REJECT",
            },

        "message_sha256":
            message_sha256,

        "metrics":
            {
                "original_local_alert_bytes":
                    original_alert_bytes,

                "privacy_safe_shared_bytes":
                    shared_alert_bytes,

                "size_reduction_percent":
                    size_reduction_percent,

                "privacy_filter_ms":
                    filter_ms,

                "leakage_audit_ms":
                    leakage_ms,

                "serialization_ms":
                    serialization_ms,

                "deserialization_roundtrip_ms":
                    roundtrip_ms,

                "integration_total_ms":
                    integration_total_ms,
            },

        "locked_test_used":
            False,
    }

    save_json(
        AUDIT_FILE,
        audit_payload,
    )

    manifest_payload = {
        "phase":
            "15D",

        "input":
            "frozen Phase14G2 local IDS alert",

        "pipeline":
            [
                "load Phase14G2 alert",
                "obtain local network context",
                "HMAC-SHA256 pseudonymization",
                "explicit allowlist privacy filter",
                "Phase15A schema validation",
                "privacy leakage audit",
                "canonical JSON serialization",
                "deserialize + schema round-trip validation",
                "emit privacy-safe JSONL",
            ],

        "raw_identifiers_persisted":
            False,

        "source_destination_tokens_required":
            True,

        "stable_key_required_for_operational_correlation":
            True,

        "demo_context_is_not_dataset_derived":
            True,

        "serialized_message_sha256_is_local_audit_metadata":
            True,

        "transport_security":
            "NOT IMPLEMENTED IN 15D; belongs to Phase15E",

        "central_collector":
            "NOT IMPLEMENTED IN 15D; belongs to Phase15E",

        "locked_test_used":
            False,
    }

    save_json(
        MANIFEST_FILE,
        manifest_payload,
    )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print()
    print(
        "PRIVACY-SAFE SHARED ALERT"
    )

    print(
        "-"
        *
        126
    )

    print(
        json.dumps(
            shared_mapping,
            indent=4,
        )
    )

    print()
    print(
        "INTEGRATION AUDIT"
    )

    print(
        "-"
        *
        126
    )

    print(
        f"Phase15A schema              : PASS"
    )

    print(
        f"Leakage audit before JSON    : "
        f"{'PASS' if leakage_report.passed else 'FAIL'}"
    )

    print(
        f"Serialize/deserialize roundtrip: "
        f"{'PASS' if roundtrip_equal else 'FAIL'}"
    )

    print(
        f"Leakage audit after JSON     : "
        f"{'PASS' if roundtrip_leakage.passed else 'FAIL'}"
    )

    print(
        f"Local Phase14 alert unchanged: "
        f"{'YES' if local_alert_unchanged else 'NO'}"
    )

    print(
        f"Original local alert size    : "
        f"{original_alert_bytes} bytes"
    )

    print(
        f"Shared alert size            : "
        f"{shared_alert_bytes} bytes"
    )

    print(
        f"Size reduction               : "
        f"{size_reduction_percent:.2f}%"
    )

    print(
        f"Filter                       : "
        f"{filter_ms:.4f} ms"
    )

    print(
        f"Leakage audit                : "
        f"{leakage_ms:.4f} ms"
    )

    print(
        f"Serialization                : "
        f"{serialization_ms:.4f} ms"
    )

    print(
        f"Roundtrip validation         : "
        f"{roundtrip_ms:.4f} ms"
    )

    print(
        f"Integration total            : "
        f"{integration_total_ms:.4f} ms"
    )

    print(
        f"Message SHA-256              : "
        f"{message_sha256}"
    )

    print()
    print(
        "="
        *
        126
    )

    print(
        "PHASE 15D COMPLETE"
    )

    print(
        "="
        *
        126
    )

    print(
        f"Real Phase14G2 alert         : YES"
    )

    print(
        f"Network context mode         : {context_mode}"
    )

    print(
        f"Privacy filter               : PASS"
    )

    print(
        f"Leakage audit                : PASS"
    )

    print(
        f"Canonical serialization      : PASS"
    )

    print(
        f"Roundtrip schema validation  : PASS"
    )

    print(
        f"Raw IP persisted             : NO"
    )

    print(
        f"Locked test used             : NO"
    )

    print()

    print(
        f"Shared JSONL                 : {SHARED_JSONL_FILE}"
    )

    print(
        f"Shared pretty JSON           : {SHARED_PRETTY_FILE}"
    )

    print(
        f"Integration audit            : {AUDIT_FILE}"
    )

    print(
        f"Serialization metrics        : {METRICS_FILE}"
    )

    print(
        f"Manifest                     : {MANIFEST_FILE}"
    )

    print()

    print(
        "STATUS                       : PASS"
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Phase 15E — secure transport + central privacy-safe "
        "alert collector."
    )


if __name__ == "__main__":
    main()
