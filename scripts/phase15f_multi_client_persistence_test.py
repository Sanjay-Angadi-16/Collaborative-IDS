from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
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
    validate_alert,
)
from src.alert_sharing.central_alert_store import (  # noqa: E402
    CentralAlertStore,
    DuplicateAlertError,
)
from src.alert_sharing.pseudonymizer import (  # noqa: E402
    HMACPseudonymizer,
    generate_hmac_key_bytes,
)


PHASE_NAME = (
    "PHASE 15F — CENTRAL ALERT PERSISTENCE / INDEXING + "
    "MULTI-CLIENT PRIVACY-AWARE INGESTION"
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

DATABASE_FILE = (
    RESULT_ROOT
    / "phase15f_central_alerts.sqlite3"
)

AUDIT_FILE = (
    RESULT_ROOT
    / "phase15f_persistence_audit.json"
)

QUERY_RESULTS_FILE = (
    RESULT_ROOT
    / "phase15f_query_results.json"
)

METRICS_FILE = (
    RESULT_ROOT
    / "phase15f_ingestion_metrics.json"
)

MANIFEST_FILE = (
    ARTIFACT_ROOT
    / "phase15f_store_manifest.json"
)


CLIENT_ATTACKS = {
    1:
        "DoS Hulk",

    2:
        "DDoS",

    3:
        "PortScan",

    4:
        "DoS GoldenEye",

    5:
        "FTP-Patator",
}


def stable_hex(
    text: str,
    n: int = 32,
) -> str:
    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()[
        :n
    ]


def build_fast_alert(
    *,
    client_id: int,
    classification: str,
    source_token: str,
    destination_token: str,
    event_time: str,
    confidence: float,
) -> Dict[str, Any]:
    return validate_alert(
        {
            "alert_id":
                str(
                    uuid.uuid4()
                ),

            "client_id":
                f"client_{client_id}",

            "event_time":
                event_time,

            "classification":
                classification,

            "severity":
                "HIGH",

            "confidence":
                confidence,

            "risk_score":
                None,

            "evidence_score":
                confidence,

            "risk_mode":
                "FAST_PATH_RF_CONFIDENCE",

            "decision_source":
                "Random Forest",

            "processing_path":
                "FAST_ATTACK_PATH",

            "source_token":
                source_token,

            "destination_token":
                destination_token,

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
    ).to_dict()


def severity_for_risk(
    value: float,
) -> str:
    if value < 0.20:
        return "NORMAL"
    if value < 0.40:
        return "LOW"
    if value < 0.60:
        return "MEDIUM"
    if value < 0.80:
        return "HIGH"
    return "CRITICAL"


def build_deep_alert(
    *,
    client_id: int,
    classification: str,
    source_token: str,
    destination_token: str,
    event_time: str,
    confidence: float,
    risk_score: float,
) -> Dict[str, Any]:
    return validate_alert(
        {
            "alert_id":
                str(
                    uuid.uuid4()
                ),

            "client_id":
                f"client_{client_id}",

            "event_time":
                event_time,

            "classification":
                classification,

            "severity":
                severity_for_risk(
                    risk_score
                ),

            "confidence":
                confidence,

            "risk_score":
                risk_score,

            "evidence_score":
                risk_score,

            "risk_mode":
                "FULL_FOUR_MODEL_FUSION",

            "decision_source":
                "CNN-BiLSTM",

            "processing_path":
                "DEEP_ANALYSIS",

            "source_token":
                source_token,

            "destination_token":
                destination_token,

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
    ).to_dict()


def save_json(
    path: Path,
    payload: Dict[str, Any],
) -> None:
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


def main():
    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARTIFACT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if DATABASE_FILE.exists():
        DATABASE_FILE.unlink()

    wal_file = Path(
        str(
            DATABASE_FILE
        )
        +
        "-wal"
    )

    shm_file = Path(
        str(
            DATABASE_FILE
        )
        +
        "-shm"
    )

    for path in (
        wal_file,
        shm_file,
    ):
        if path.exists():
            path.unlink()

    store = CentralAlertStore(
        DATABASE_FILE
    )

    # Test-only HMAC key. It is never persisted.
    pseudonymizer = HMACPseudonymizer(
        key=
            generate_hmac_key_bytes(
                32
            )
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
        f"Database backend              : SQLite"
    )

    print(
        f"Database                      : {DATABASE_FILE}"
    )

    print(
        f"Clients                       : 5"
    )

    print(
        f"Alerts/client                 : 4"
    )

    print(
        f"Total planned                 : 20"
    )

    print(
        f"Privacy-safe schema           : Phase15A"
    )

    print(
        f"Leakage validation            : Phase15C"
    )

    print(
        f"Raw identifiers stored        : NO"
    )

    print(
        f"HMAC secret stored            : NO"
    )

    print(
        f"Locked test                   : NO"
    )

    print()
    print(
        "SCIENTIFIC NOTE:"
    )

    print(
        "The 20 alerts generated below are synthetic privacy-safe "
        "INGESTION TEST messages."
    )

    print(
        "They test persistence/indexing only and are NOT IDS "
        "detection-performance evidence."
    )

    now = datetime.now(
        timezone.utc
    ).replace(
        microsecond=0
    )

    common_raw_source = (
        "203.0.113.200"
    )

    common_source_token = pseudonymizer.source_ip(
        common_raw_source
    )

    destination_token = pseudonymizer.destination_ip(
        "198.51.100.10"
    )

    alerts: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    # Each client contributes:
    # - two local attack FAST alerts
    # - one local attack DEEP alert
    # - one shared-source cross-client DEEP alert
    for client_id in range(
        1,
        6,
    ):
        attack = CLIENT_ATTACKS[
            client_id
        ]

        local_source = pseudonymizer.source_ip(
            f"10.10.{client_id}.25"
        )

        base_time = (
            now
            +
            timedelta(
                seconds=
                    client_id
                    *
                    10
            )
        )

        alerts.append(
            build_fast_alert(
                client_id=
                    client_id,

                classification=
                    attack,

                source_token=
                    local_source,

                destination_token=
                    destination_token,

                event_time=
                    base_time.isoformat(),

                confidence=
                    0.995,
            )
        )

        alerts.append(
            build_fast_alert(
                client_id=
                    client_id,

                classification=
                    attack,

                source_token=
                    local_source,

                destination_token=
                    destination_token,

                event_time=
                    (
                        base_time
                        +
                        timedelta(
                            seconds=1
                        )
                    ).isoformat(),

                confidence=
                    0.985,
            )
        )

        alerts.append(
            build_deep_alert(
                client_id=
                    client_id,

                classification=
                    attack,

                source_token=
                    local_source,

                destination_token=
                    destination_token,

                event_time=
                    (
                        base_time
                        +
                        timedelta(
                            seconds=2
                        )
                    ).isoformat(),

                confidence=
                    0.91,

                risk_score=
                    0.83,
            )
        )

        alerts.append(
            build_deep_alert(
                client_id=
                    client_id,

                classification=
                    attack,

                source_token=
                    common_source_token,

                destination_token=
                    destination_token,

                event_time=
                    (
                        base_time
                        +
                        timedelta(
                            seconds=3
                        )
                    ).isoformat(),

                confidence=
                    0.89,

                risk_score=
                    0.76,
            )
        )

    insertion_latencies = []

    print()
    print(
        "MULTI-CLIENT INGESTION"
    )

    print(
        "-"
        *
        126
    )

    for index, alert in enumerate(
        alerts,
        start=1,
    ):
        start = time.perf_counter()

        result = store.insert_alert(
            alert
        )

        latency_ms = (
            time.perf_counter()
            -
            start
        ) * 1000.0

        insertion_latencies.append(
            latency_ms
        )

        print(
            f"[PASS] insert_{index:02d}: "
            f"{alert['client_id']:8s} "
            f"{alert['classification']:16s} "
            f"{latency_ms:.3f} ms"
        )

    count_pass = (
        store.count()
        ==
        20
    )

    print()
    print(
        f"[{'PASS' if count_pass else 'FAIL'}] "
        f"stored_alert_count: {store.count()}"
    )

    duplicate_pass = False

    try:
        store.insert_alert(
            alerts[
                0
            ]
        )

    except DuplicateAlertError:
        duplicate_pass = True

    print(
        f"[{'PASS' if duplicate_pass else 'FAIL'}] "
        "duplicate_alert_id_rejected"
    )

    by_client = store.count_by_client()

    expected_client_counts = {
        f"client_{client_id}":
            4
        for client_id
        in range(
            1,
            6,
        )
    }

    actual_client_counts = {
        row[
            "client_id"
        ]:
            row[
                "alert_count"
            ]
        for row
        in by_client
    }

    client_counts_pass = (
        actual_client_counts
        ==
        expected_client_counts
    )

    print(
        f"[{'PASS' if client_counts_pass else 'FAIL'}] "
        "per_client_counts_4_each"
    )

    by_classification = store.count_by_classification()

    classification_counts_pass = (
        all(
            row[
                "alert_count"
            ]
            ==
            4
            for row
            in by_classification
        )
        and
        len(
            by_classification
        )
        ==
        5
    )

    print(
        f"[{'PASS' if classification_counts_pass else 'FAIL'}] "
        "per_attack_class_counts_4_each"
    )

    cross_client = store.find_by_source_token(
        common_source_token
    )

    cross_client_ids = sorted(
        {
            item[
                "client_id"
            ]
            for item
            in cross_client
        }
    )

    cross_client_pass = (
        len(
            cross_client
        )
        ==
        5
        and
        cross_client_ids
        ==
        [
            "client_1",
            "client_2",
            "client_3",
            "client_4",
            "client_5",
        ]
    )

    print(
        f"[{'PASS' if cross_client_pass else 'FAIL'}] "
        "same_pseudonymous_source_correlates_across_5_clients"
    )

    # Read-back schema validation is built into get_alert().
    readback = store.get_alert(
        alerts[
            0
        ][
            "alert_id"
        ]
    )

    readback_pass = (
        readback
        ==
        alerts[
            0
        ]
    )

    print(
        f"[{'PASS' if readback_pass else 'FAIL'}] "
        "canonical_readback_matches_original"
    )

    indexes = store.list_indexes()

    expected_indexes = {
        "idx_alerts_event_time",
        "idx_alerts_client_id",
        "idx_alerts_classification",
        "idx_alerts_severity",
        "idx_alerts_source_token",
        "idx_alerts_destination_token",
        "idx_alerts_client_event_time",
        "idx_alerts_source_event_time",
        "idx_alerts_classification_severity",
    }

    index_pass = (
        expected_indexes
        <=
        set(
            indexes
        )
    )

    print(
        f"[{'PASS' if index_pass else 'FAIL'}] "
        f"required_indexes_present: "
        f"{len(expected_indexes)}/{len(expected_indexes)}"
    )

    at_rest_audit = store.privacy_at_rest_audit(
        forbidden_literals=[
            common_raw_source,
            "198.51.100.10",
            "10.10.1.25",
            "10.10.2.25",
            "10.10.3.25",
            "10.10.4.25",
            "10.10.5.25",
            "payload",
            "flow_features",
            "cnn_sequence",
            "model_state_dict",
        ]
    )

    privacy_at_rest_pass = bool(
        at_rest_audit[
            "passed"
        ]
    )

    print(
        f"[{'PASS' if privacy_at_rest_pass else 'FAIL'}] "
        "privacy_at_rest_literal_audit"
    )

    recent = store.recent_alerts(
        limit=5
    )

    recent_pass = (
        len(
            recent
        )
        ==
        5
    )

    print(
        f"[{'PASS' if recent_pass else 'FAIL'}] "
        "recent_alert_query"
    )

    all_passed = all(
        [
            count_pass,
            duplicate_pass,
            client_counts_pass,
            classification_counts_pass,
            cross_client_pass,
            readback_pass,
            index_pass,
            privacy_at_rest_pass,
            recent_pass,
        ]
    )

    query_results = {
        "count_by_client":
            by_client,

        "count_by_classification":
            by_classification,

        "count_by_severity":
            store.count_by_severity(),

        "shared_source_token_match_count":
            len(
                cross_client
            ),

        "shared_source_clients":
            cross_client_ids,

        "recent_alerts":
            recent,
    }

    save_json(
        QUERY_RESULTS_FILE,
        query_results,
    )

    metrics = {
        "alerts_inserted":
            len(
                alerts
            ),

        "mean_insert_ms":
            statistics.mean(
                insertion_latencies
            ),

        "median_insert_ms":
            statistics.median(
                insertion_latencies
            ),

        "min_insert_ms":
            min(
                insertion_latencies
            ),

        "max_insert_ms":
            max(
                insertion_latencies
            ),

        "database_bytes":
            DATABASE_FILE.stat().st_size
            if DATABASE_FILE.exists()
            else
            0,
    }

    save_json(
        METRICS_FILE,
        metrics,
    )

    audit = {
        "phase":
            "15F",

        "phase_name":
            PHASE_NAME,

        "database_backend":
            "SQLite",

        "database_file":
            str(
                DATABASE_FILE
            ),

        "synthetic_ingestion_test_alerts":
            True,

        "detection_performance_claim":
            False,

        "clients":
            5,

        "alerts_per_client":
            4,

        "total_alerts":
            20,

        "duplicate_rejection":
            duplicate_pass,

        "per_client_count_check":
            client_counts_pass,

        "per_classification_count_check":
            classification_counts_pass,

        "cross_client_source_token_correlation":
            cross_client_pass,

        "canonical_readback_check":
            readback_pass,

        "required_indexes_present":
            index_pass,

        "privacy_at_rest_audit":
            at_rest_audit,

        "recent_query_check":
            recent_pass,

        "all_tests_passed":
            all_passed,

        "locked_test_used":
            False,
    }

    save_json(
        AUDIT_FILE,
        audit,
    )

    manifest = {
        "phase":
            "15F",

        "storage_backend":
            "SQLite",

        "schema":
            "Phase15A privacy-aware shared alert only",

        "privacy_validation":
            "Phase15C leakage audit before persistence",

        "primary_identity":
            "alert_id UNIQUE",

        "indexed_fields":
            [
                "event_time",
                "client_id",
                "classification",
                "severity",
                "source_token",
                "destination_token",
                "client_id + event_time",
                "source_token + event_time",
                "classification + severity",
            ],

        "cross_cloud_correlation_ready":
            True,

        "raw_source_ip_stored":
            False,

        "raw_destination_ip_stored":
            False,

        "raw_payload_stored":
            False,

        "raw_features_stored":
            False,

        "cnn_sequence_stored":
            False,

        "model_state_stored":
            False,

        "hmac_secret_stored":
            False,

        "multi_client_ingestion_test":
            (
                "20 synthetic privacy-safe alerts across five clients; "
                "storage/indexing test only"
            ),

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
        126
    )

    print(
        "PHASE 15F COMPLETE"
    )

    print(
        "="
        *
        126
    )

    print(
        f"Stored alerts                : "
        f"{store.count()}/20"
    )

    print(
        f"Clients represented          : "
        f"{len(by_client)}/5"
    )

    print(
        f"Attack classes represented   : "
        f"{len(by_classification)}/5"
    )

    print(
        f"Duplicate rejection          : "
        f"{'PASS' if duplicate_pass else 'FAIL'}"
    )

    print(
        f"Cross-client token query     : "
        f"{'PASS' if cross_client_pass else 'FAIL'} "
        f"({len(cross_client)} matching alerts)"
    )

    print(
        f"Required indexes             : "
        f"{'PASS' if index_pass else 'FAIL'}"
    )

    print(
        f"Privacy-at-rest audit        : "
        f"{'PASS' if privacy_at_rest_pass else 'FAIL'}"
    )

    print(
        f"Mean insert latency          : "
        f"{metrics['mean_insert_ms']:.3f} ms"
    )

    print(
        f"Database                     : {DATABASE_FILE}"
    )

    print(
        f"Audit                        : {AUDIT_FILE}"
    )

    print(
        f"Query results                : {QUERY_RESULTS_FILE}"
    )

    print(
        f"Metrics                      : {METRICS_FILE}"
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
        "Phase15F tests central persistence, indexing and "
        "privacy-safe multi-client ingestion."
    )

    print(
        "Synthetic ingestion messages are not IDS accuracy evidence."
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Phase 15G — privacy/retention audit and central-store "
        "performance/throughput evaluation."
    )

    if not all_passed:
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()
