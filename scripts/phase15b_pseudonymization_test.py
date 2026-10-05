from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


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


from src.alert_sharing.alert_schema import (  # noqa: E402
    PHASE14_POLICY_VERSION,
    PRIVACY_VERSION,
    SCHEMA_VERSION,
    new_alert_id,
    utc_now_iso,
    validate_alert,
)

from src.alert_sharing.pseudonymizer import (  # noqa: E402
    DEFAULT_DIGEST_BYTES,
    DEFAULT_KEY_ID,
    PHASE15_HMAC_ENV,
    HMACPseudonymizer,
    PseudonymizationError,
    encode_hmac_key_b64,
    generate_hmac_key_bytes,
    token_looks_valid,
)


PHASE_NAME = (
    "PHASE 15B — KEYED HMAC-SHA256 PSEUDONYMIZATION"
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
    / "phase15b_pseudonymization_audit.json"
)

SAMPLE_FILE = (
    RESULT_ROOT
    / "phase15b_pseudonymized_alerts.jsonl"
)

MANIFEST_FILE = (
    ARTIFACT_ROOT
    / "phase15b_pseudonymization_manifest.json"
)


# Raw identifiers are used only inside this local test process.
# They MUST NOT be written to results or artifacts.
RAW_SOURCE_IP = "192.168.10.25"
RAW_DESTINATION_IP = "10.20.30.40"
RAW_SECOND_SOURCE_IP = "192.168.10.26"


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
    Operational rule:
    - If PHASE15_HMAC_KEY_B64 exists, use that stable key.
    - If not, Phase15B TEST generates an ephemeral in-memory key.
      It is never written to disk.

    Phase15D integration should require a stable externally managed key.
    """

    env_value = os.getenv(
        PHASE15_HMAC_ENV
    )

    if env_value:
        pseudonymizer = HMACPseudonymizer.from_environment(
            env_name=
                PHASE15_HMAC_ENV,

            key_id=
                DEFAULT_KEY_ID,

            digest_bytes=
                DEFAULT_DIGEST_BYTES,
        )

        return (
            pseudonymizer,
            "ENVIRONMENT_STABLE_KEY",
            False,
        )

    ephemeral_key = generate_hmac_key_bytes(
        32
    )

    pseudonymizer = HMACPseudonymizer(
        key=
            ephemeral_key,

        key_id=
            DEFAULT_KEY_ID,

        digest_bytes=
            DEFAULT_DIGEST_BYTES,
    )

    return (
        pseudonymizer,
        "EPHEMERAL_TEST_KEY",
        True,
    )


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
        122
    )

    print(
        PHASE_NAME
    )

    print(
        "="
        *
        122
    )

    print(
        f"Algorithm                     : HMAC-SHA256"
    )

    print(
        f"Key ID                        : {pseudonymizer.key_id}"
    )

    print(
        f"Exposed digest bytes          : {pseudonymizer.digest_bytes}"
    )

    print(
        f"Token hex characters          : {pseudonymizer.digest_bytes * 2}"
    )

    print(
        f"Key mode                      : {key_mode}"
    )

    print(
        f"Secret written to disk        : NO"
    )

    print(
        f"Raw identifiers written       : NO"
    )

    print(
        f"Locked test                   : NO"
    )

    if ephemeral:
        print()
        print(
            "NOTE:"
        )

        print(
            "No PHASE15_HMAC_KEY_B64 environment variable was found."
        )

        print(
            "This test uses an ephemeral cryptographic key in memory only."
        )

        print(
            "Tokens will change on the next run."
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

        status = (
            "PASS"
            if passed
            else "FAIL"
        )

        print(
            f"[{status}] {name}: {detail}"
        )

    print()
    print(
        "PSEUDONYMIZATION TESTS"
    )

    print(
        "-"
        *
        122
    )

    start = time.perf_counter()

    source_1 = pseudonymizer.source_ip(
        RAW_SOURCE_IP
    )

    source_1_repeat = pseudonymizer.source_ip(
        RAW_SOURCE_IP
    )

    destination_same_raw = pseudonymizer.destination_ip(
        RAW_SOURCE_IP
    )

    destination_1 = pseudonymizer.destination_ip(
        RAW_DESTINATION_IP
    )

    source_2 = pseudonymizer.source_ip(
        RAW_SECOND_SOURCE_IP
    )

    pseudonymization_ms = (
        time.perf_counter()
        -
        start
    ) * 1000.0

    record(
        "same_input_same_token",
        source_1
        ==
        source_1_repeat,
        "Repeated source identifier is deterministic for the same key.",
    )

    record(
        "different_input_different_token",
        source_1
        !=
        source_2,
        "Different source identifiers produce different tokens.",
    )

    record(
        "domain_separation",
        source_1
        !=
        destination_same_raw,
        "The same raw identifier produces different source/destination tokens.",
    )

    record(
        "source_token_format",
        token_looks_valid(
            source_1
        )
        and
        source_1.startswith(
            "src_"
        ),
        "Source token has the frozen src_<32 hex> format.",
    )

    record(
        "destination_token_format",
        token_looks_valid(
            destination_1
        )
        and
        destination_1.startswith(
            "dst_"
        ),
        "Destination token has the frozen dst_<32 hex> format.",
    )

    record(
        "raw_source_not_exposed",
        RAW_SOURCE_IP
        not in
        source_1,
        "Raw source IP is not present in the token.",
    )

    record(
        "raw_destination_not_exposed",
        RAW_DESTINATION_IP
        not in
        destination_1,
        "Raw destination IP is not present in the token.",
    )

    second_instance = HMACPseudonymizer(
        key=
            pseudonymizer.key,

        key_id=
            pseudonymizer.key_id,

        digest_bytes=
            pseudonymizer.digest_bytes,
    )

    record(
        "same_key_cross_instance_stability",
        second_instance.source_ip(
            RAW_SOURCE_IP
        )
        ==
        source_1,
        "Same key and namespace preserve cross-instance linkability.",
    )

    different_key = generate_hmac_key_bytes(
        32
    )

    third_instance = HMACPseudonymizer(
        key=
            different_key,

        key_id=
            pseudonymizer.key_id,

        digest_bytes=
            pseudonymizer.digest_bytes,
    )

    record(
        "different_key_unlinkability",
        third_instance.source_ip(
            RAW_SOURCE_IP
        )
        !=
        source_1,
        "Changing the HMAC key changes the pseudonym.",
    )

    invalid_ip_rejected = False

    try:
        pseudonymizer.source_ip(
            "not-an-ip"
        )
    except PseudonymizationError:
        invalid_ip_rejected = True

    record(
        "invalid_ip_rejected",
        invalid_ip_rejected,
        "Invalid IP addresses are rejected rather than silently hashed.",
    )

    # --------------------------------------------------------
    # Integration with Phase 15A schema
    # --------------------------------------------------------

    fast_alert = {
        "alert_id":
            new_alert_id(),

        "client_id":
            "client_3",

        "event_time":
            utc_now_iso(),

        "classification":
            "PortScan",

        "severity":
            "HIGH",

        "confidence":
            0.999821,

        "risk_score":
            None,

        "evidence_score":
            0.999821,

        "risk_mode":
            "FAST_PATH_RF_CONFIDENCE",

        "decision_source":
            "Random Forest",

        "processing_path":
            "FAST_ATTACK_PATH",

        "source_token":
            source_1,

        "destination_token":
            destination_1,

        "protocol":
            "TCP",

        "destination_port":
            443,

        "model_version":
            "phase14_final",

        "policy_version":
            PHASE14_POLICY_VERSION,

        "privacy_version":
            PRIVACY_VERSION,

        "schema_version":
            SCHEMA_VERSION,
    }

    schema_integration_passed = False
    validated_alert = None

    try:
        validated_alert = validate_alert(
            fast_alert
        )

        schema_integration_passed = True

    except Exception as exc:
        schema_error = str(
            exc
        )
    else:
        schema_error = None

    record(
        "phase15a_schema_integration",
        schema_integration_passed,
        (
            "Pseudonymized source/destination tokens are accepted by "
            "the frozen Phase15A alert schema."
            if schema_integration_passed
            else
            f"Schema integration failed: {schema_error}"
        ),
    )

    # --------------------------------------------------------
    # Secret-leak audit
    # --------------------------------------------------------

    sample_payload = (
        validated_alert.to_dict()
        if validated_alert is not None
        else fast_alert
    )

    serialized_sample = json.dumps(
        sample_payload,
        sort_keys=True,
    )

    key_b64_for_check = encode_hmac_key_b64(
        pseudonymizer.key
    )

    secret_not_in_alert = (
        key_b64_for_check
        not in
        serialized_sample
    )

    record(
        "hmac_secret_not_in_alert",
        secret_not_in_alert,
        "HMAC secret is absent from the serialized shared alert.",
    )

    # --------------------------------------------------------
    # Save only privacy-safe outputs
    # --------------------------------------------------------

    with open(
        SAMPLE_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                sample_payload,
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
        for item
        in tests
        if item[
            "passed"
        ]
    )

    failed = len(
        tests
    ) - passed

    audit = {
        "phase":
            "15B",

        "phase_name":
            PHASE_NAME,

        "algorithm":
            "HMAC-SHA256",

        "key_id":
            pseudonymizer.key_id,

        "key_mode":
            key_mode,

        "ephemeral_test_key":
            ephemeral,

        "secret_written_to_disk":
            False,

        "raw_identifiers_written_to_disk":
            False,

        "digest_bytes_exposed":
            pseudonymizer.digest_bytes,

        "token_format":
            "src_<32 hex> / dst_<32 hex>",

        "domain_separation":
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

        "pseudonymization_test_batch_ms":
            pseudonymization_ms,

        "tests":
            tests,

        "locked_test_used":
            False,
    }

    save_json(
        AUDIT_FILE,
        audit,
    )

    manifest = {
        "phase":
            "15B",

        "algorithm":
            "HMAC-SHA256",

        "key_environment_variable":
            PHASE15_HMAC_ENV,

        "key_id":
            DEFAULT_KEY_ID,

        "minimum_key_bytes":
            32,

        "digest_bytes_exposed":
            DEFAULT_DIGEST_BYTES,

        "source_namespace":
            "source_ip",

        "destination_namespace":
            "destination_ip",

        "source_prefix":
            "src_",

        "destination_prefix":
            "dst_",

        "raw_ip_normalization":
            "canonical ipaddress compressed representation",

        "key_material_in_manifest":
            False,

        "plain_sha256_used":
            False,

        "stable_cross_alert_linkability":
            (
                "YES only when the same secret key and key_id are reused."
            ),

        "phase15a_schema_compatible":
            schema_integration_passed,

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
        122
    )

    print(
        "PHASE 15B COMPLETE"
    )

    print(
        "="
        *
        122
    )

    print(
        f"Tests                         : "
        f"{passed}/{len(tests)} passed"
    )

    print(
        f"Source token                  : "
        f"{source_1}"
    )

    print(
        f"Destination token             : "
        f"{destination_1}"
    )

    print(
        f"Key written                   : NO"
    )

    print(
        f"Raw source/destination saved  : NO"
    )

    print(
        f"Schema integration            : "
        f"{'PASS' if schema_integration_passed else 'FAIL'}"
    )

    print(
        f"Audit                         : "
        f"{AUDIT_FILE}"
    )

    print(
        f"Privacy-safe sample JSONL     : "
        f"{SAMPLE_FILE}"
    )

    print(
        f"Manifest                      : "
        f"{MANIFEST_FILE}"
    )

    print()

    print(
        f"STATUS                        : "
        f"{'PASS' if failed == 0 else 'FAIL'}"
    )

    print()

    print(
        "SECURITY RULE:"
    )

    print(
        "Use a stable externally managed HMAC key for actual "
        "cross-alert correlation."
    )

    print(
        "Do not hard-code, print, commit, or write the secret key "
        "into results/artifacts."
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Phase 15C — allowlist privacy filter + leakage audit."
    )

    if failed != 0:
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()
