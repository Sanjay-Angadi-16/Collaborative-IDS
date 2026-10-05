from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.alert_sharing.alert_schema import (  # noqa: E402
    ALLOWED_FIELDS,
    PHASE14_POLICY_VERSION,
    PRIVACY_VERSION,
    PROHIBITED_FIELDS,
    SCHEMA_VERSION,
    AlertSchemaError,
    new_alert_id,
    utc_now_iso,
    validate_alert,
)


PHASE_NAME = "PHASE 15A — PRIVACY-AWARE ALERT SCHEMA"

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
    / "phase15a_schema_audit.json"
)

VALID_ALERTS_FILE = (
    RESULT_ROOT
    / "phase15a_valid_alerts.jsonl"
)

MANIFEST_FILE = (
    ARTIFACT_ROOT
    / "phase15a_schema_manifest.json"
)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
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


def expect_valid(
    name: str,
    payload: Dict[str, Any],
    audit_rows: List[Dict[str, Any]],
    valid_alerts: List[Dict[str, Any]],
) -> None:
    try:
        alert = validate_alert(
            payload
        )

        normalized = alert.to_dict()

        audit_rows.append(
            {
                "test":
                    name,

                "expected":
                    "VALID",

                "actual":
                    "VALID",

                "passed":
                    True,

                "error":
                    None,
            }
        )

        valid_alerts.append(
            normalized
        )

        print(
            f"[PASS] {name}: VALID"
        )

    except Exception as exc:
        audit_rows.append(
            {
                "test":
                    name,

                "expected":
                    "VALID",

                "actual":
                    "REJECTED",

                "passed":
                    False,

                "error":
                    str(
                        exc
                    ),
            }
        )

        print(
            f"[FAIL] {name}: {exc}"
        )


def expect_rejected(
    name: str,
    payload: Dict[str, Any],
    audit_rows: List[Dict[str, Any]],
) -> None:
    try:
        validate_alert(
            payload
        )

        audit_rows.append(
            {
                "test":
                    name,

                "expected":
                    "REJECTED",

                "actual":
                    "VALID",

                "passed":
                    False,

                "error":
                    "Payload was unexpectedly accepted.",
            }
        )

        print(
            f"[FAIL] {name}: unexpectedly accepted"
        )

    except AlertSchemaError as exc:
        audit_rows.append(
            {
                "test":
                    name,

                "expected":
                    "REJECTED",

                "actual":
                    "REJECTED",

                "passed":
                    True,

                "error":
                    str(
                        exc
                    ),
            }
        )

        print(
            f"[PASS] {name}: rejected -> {exc}"
        )


def fast_alert() -> Dict[str, Any]:
    return {
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

        # Tokens are optional in 15A.
        # Phase 15B will create them using keyed HMAC.
        "source_token":
            None,

        "destination_token":
            None,

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


def deep_alert() -> Dict[str, Any]:
    return {
        "alert_id":
            new_alert_id(),

        "client_id":
            "client_2",

        "event_time":
            utc_now_iso(),

        "classification":
            "DDoS",

        "severity":
            "CRITICAL",

        "confidence":
            0.940000,

        "risk_score":
            0.870000,

        "evidence_score":
            0.870000,

        "risk_mode":
            "FULL_FOUR_MODEL_FUSION",

        "decision_source":
            "CNN-BiLSTM",

        "processing_path":
            "DEEP_ANALYSIS",

        "source_token":
            None,

        "destination_token":
            None,

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


def main() -> None:
    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARTIFACT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "="
        *
        118
    )

    print(
        PHASE_NAME
    )

    print(
        "="
        *
        118
    )

    print(
        f"Schema version              : {SCHEMA_VERSION}"
    )

    print(
        f"Privacy version             : {PRIVACY_VERSION}"
    )

    print(
        f"Frozen Phase14 policy       : {PHASE14_POLICY_VERSION}"
    )

    print(
        "Schema policy               : ALLOWLIST-ONLY"
    )

    print(
        "Raw/sensitive fields        : REJECT"
    )

    print(
        "Locked test                 : NO"
    )

    print()
    print(
        "ALLOWED FIELD COUNT"
    )

    print(
        "-"
        *
        118
    )

    print(
        len(
            ALLOWED_FIELDS
        )
    )

    print()
    print(
        "PRIVACY-SENSITIVE FIELD EXAMPLES"
    )

    print(
        "-"
        *
        118
    )

    for field in sorted(
        PROHIBITED_FIELDS
    ):
        print(
            field
        )

    audit_rows: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    valid_alerts: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    print()
    print(
        "SCHEMA TESTS"
    )

    print(
        "-"
        *
        118
    )

    valid_fast = fast_alert()

    valid_deep = deep_alert()

    expect_valid(
        "valid_fast_path_alert",
        valid_fast,
        audit_rows,
        valid_alerts,
    )

    expect_valid(
        "valid_deep_fusion_alert",
        valid_deep,
        audit_rows,
        valid_alerts,
    )

    invalid_raw_ip = dict(
        valid_fast
    )

    invalid_raw_ip[
        "source_ip"
    ] = "192.168.10.25"

    expect_rejected(
        "reject_raw_source_ip",
        invalid_raw_ip,
        audit_rows,
    )

    invalid_payload = dict(
        valid_deep
    )

    invalid_payload[
        "payload"
    ] = "raw network payload"

    expect_rejected(
        "reject_raw_payload",
        invalid_payload,
        audit_rows,
    )

    invalid_unknown = dict(
        valid_fast
    )

    invalid_unknown[
        "new_internal_debug_field"
    ] = "must not leave client"

    expect_rejected(
        "reject_unknown_non_allowlisted_field",
        invalid_unknown,
        audit_rows,
    )

    invalid_fast_risk = dict(
        valid_fast
    )

    invalid_fast_risk[
        "risk_score"
    ] = 0.999821

    expect_rejected(
        "reject_fast_path_ensemble_risk",
        invalid_fast_risk,
        audit_rows,
    )

    invalid_fast_severity = dict(
        valid_fast
    )

    invalid_fast_severity[
        "severity"
    ] = "CRITICAL"

    expect_rejected(
        "reject_fast_path_critical",
        invalid_fast_severity,
        audit_rows,
    )

    invalid_deep_null_risk = dict(
        valid_deep
    )

    invalid_deep_null_risk[
        "risk_score"
    ] = None

    expect_rejected(
        "reject_deep_path_null_risk",
        invalid_deep_null_risk,
        audit_rows,
    )

    invalid_deep_severity = dict(
        valid_deep
    )

    invalid_deep_severity[
        "severity"
    ] = "LOW"

    expect_rejected(
        "reject_deep_severity_risk_mismatch",
        invalid_deep_severity,
        audit_rows,
    )

    invalid_deep_evidence = dict(
        valid_deep
    )

    invalid_deep_evidence[
        "evidence_score"
    ] = 0.50

    expect_rejected(
        "reject_deep_evidence_risk_mismatch",
        invalid_deep_evidence,
        audit_rows,
    )

    passed = sum(
        1
        for row
        in audit_rows
        if row[
            "passed"
        ]
    )

    failed = len(
        audit_rows
    ) - passed

    audit_payload = {
        "phase":
            "15A",

        "phase_name":
            PHASE_NAME,

        "schema_version":
            SCHEMA_VERSION,

        "privacy_version":
            PRIVACY_VERSION,

        "phase14_policy_version":
            PHASE14_POLICY_VERSION,

        "allowlist_only":
            True,

        "allowed_fields":
            sorted(
                ALLOWED_FIELDS
            ),

        "prohibited_field_examples":
            sorted(
                PROHIBITED_FIELDS
            ),

        "tests_total":
            len(
                audit_rows
            ),

        "tests_passed":
            passed,

        "tests_failed":
            failed,

        "all_tests_passed":
            failed
            ==
            0,

        "tests":
            audit_rows,

        "locked_test_used":
            False,
    }

    save_json(
        AUDIT_FILE,
        audit_payload,
    )

    with open(
        VALID_ALERTS_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        for alert in valid_alerts:
            file.write(
                json.dumps(
                    alert,
                    separators=(
                        ",",
                        ":",
                    ),
                )
                +
                "\n"
            )

    manifest = {
        "phase":
            "15A",

        "schema_version":
            SCHEMA_VERSION,

        "privacy_version":
            PRIVACY_VERSION,

        "policy_version":
            PHASE14_POLICY_VERSION,

        "schema_policy":
            "allowlist_only",

        "fast_path_contract":
            {
                "processing_path":
                    "FAST_ATTACK_PATH",

                "risk_mode":
                    "FAST_PATH_RF_CONFIDENCE",

                "risk_score":
                    None,

                "evidence_score":
                    "must equal RF confidence",

                "severity":
                    "HIGH",

                "decision_source":
                    "Random Forest",
            },

        "deep_path_contract":
            {
                "processing_path":
                    "DEEP_ANALYSIS",

                "risk_mode":
                    "FULL_FOUR_MODEL_FUSION",

                "risk_score":
                    "required 0..1",

                "evidence_score":
                    "must equal risk_score",

                "severity":
                    "derived from frozen Phase14G2 risk bands",
            },

        "raw_identifiers_allowed":
            False,

        "raw_packets_allowed":
            False,

        "raw_payload_allowed":
            False,

        "raw_features_allowed":
            False,

        "cnn_sequence_allowed":
            False,

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
        118
    )

    print(
        "PHASE 15A COMPLETE"
    )

    print(
        "="
        *
        118
    )

    print(
        f"Tests                       : {passed}/{len(audit_rows)} passed"
    )

    print(
        f"Valid sample alerts         : {len(valid_alerts)}"
    )

    print(
        f"Schema audit                : {AUDIT_FILE}"
    )

    print(
        f"Valid JSONL                 : {VALID_ALERTS_FILE}"
    )

    print(
        f"Schema manifest             : {MANIFEST_FILE}"
    )

    print()

    if failed == 0:
        print(
            "STATUS                      : PASS"
        )
    else:
        print(
            "STATUS                      : FAIL"
        )

    print()

    print(
        "PRIVACY RULE:"
    )

    print(
        "Only allowlisted metadata may leave the client."
    )

    print(
        "Raw IPs, raw packets, payloads, feature vectors and "
        "CNN sequences are rejected."
    )

    print()

    print(
        "PHASE14G2 SEMANTICS PRESERVED:"
    )

    print(
        "FAST -> risk_score=NULL, evidence_score=RF confidence, severity=HIGH."
    )

    print(
        "DEEP -> risk_score=fused risk, evidence_score=fused risk, severity from risk bands."
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Phase 15B — keyed HMAC-SHA256 pseudonymization."
    )

    if failed != 0:
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()
