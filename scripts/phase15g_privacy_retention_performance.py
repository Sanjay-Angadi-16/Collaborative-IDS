from __future__ import annotations

import argparse
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
from src.alert_sharing.central_alert_store import CentralAlertStore  # noqa: E402
from src.alert_sharing.pseudonymizer import (  # noqa: E402
    HMACPseudonymizer,
    generate_hmac_key_bytes,
)
from src.alert_sharing.retention_policy import (  # noqa: E402
    apply_retention_policy,
    oldest_newest_event_time,
    retention_preview,
)
from src.alert_sharing.store_benchmark import (  # noqa: E402
    summarize_latencies,
    timed_sql_query,
)


PHASE_NAME = (
    "PHASE 15G — PRIVACY / RETENTION AUDIT + "
    "CENTRAL-STORE PERFORMANCE EVALUATION"
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
    / "phase15g_benchmark_alerts.sqlite3"
)

AUDIT_FILE = (
    RESULT_ROOT
    / "phase15g_privacy_retention_audit.json"
)

PERFORMANCE_FILE = (
    RESULT_ROOT
    / "phase15g_store_performance.json"
)

MANIFEST_FILE = (
    ARTIFACT_ROOT
    / "phase15g_evaluation_manifest.json"
)


ATTACKS = [
    "DoS Hulk",
    "DDoS",
    "PortScan",
    "DoS GoldenEye",
    "FTP-Patator",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=
            PHASE_NAME
    )

    parser.add_argument(
        "--alerts",
        type=int,
        default=1000,
        help=
            "Synthetic privacy-safe benchmark alerts. Default 1000.",
    )

    parser.add_argument(
        "--retention-days",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--query-repeats",
        type=int,
        default=100,
    )

    return parser.parse_args()


def clean_database_files(
    database_path: Path,
) -> None:
    for path in (
        database_path,
        Path(
            str(
                database_path
            )
            +
            "-wal"
        ),
        Path(
            str(
                database_path
            )
            +
            "-shm"
        ),
    ):
        if path.exists():
            path.unlink()


def severity_for_risk(
    risk_score: float,
) -> str:
    if risk_score < 0.20:
        return "NORMAL"
    if risk_score < 0.40:
        return "LOW"
    if risk_score < 0.60:
        return "MEDIUM"
    if risk_score < 0.80:
        return "HIGH"
    return "CRITICAL"


def make_alert(
    *,
    index: int,
    event_time: datetime,
    pseudonymizer: HMACPseudonymizer,
) -> Dict[str, Any]:
    client_id = (
        index % 5
    ) + 1

    classification = ATTACKS[
        client_id
        -
        1
    ]

    # Every tenth message uses one shared source across all clients.
    if index % 10 == 0:
        source_raw = (
            "203.0.113.250"
        )
    else:
        source_raw = (
            f"10.{client_id}."
            f"{(index // 254) % 250 + 1}."
            f"{index % 254 + 1}"
        )

    destination_raw = (
        "198.51.100.10"
    )

    source_token = pseudonymizer.source_ip(
        source_raw
    )

    destination_token = pseudonymizer.destination_ip(
        destination_raw
    )

    # Alternate FAST and DEEP so both frozen Phase14G2 contracts are tested.
    if index % 2 == 0:
        confidence = (
            0.95
            +
            (
                index % 40
            )
            /
            1000.0
        )

        payload = {
            "alert_id":
                str(
                    uuid.uuid4()
                ),

            "client_id":
                f"client_{client_id}",

            "event_time":
                event_time.isoformat(),

            "classification":
                classification,

            "severity":
                "HIGH",

            "confidence":
                min(
                    confidence,
                    0.999,
                ),

            "risk_score":
                None,

            "evidence_score":
                min(
                    confidence,
                    0.999,
                ),

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

    else:
        risk_score = (
            0.60
            +
            (
                index % 35
            )
            /
            100.0
        )

        risk_score = min(
            risk_score,
            0.94,
        )

        payload = {
            "alert_id":
                str(
                    uuid.uuid4()
                ),

            "client_id":
                f"client_{client_id}",

            "event_time":
                event_time.isoformat(),

            "classification":
                classification,

            "severity":
                severity_for_risk(
                    risk_score
                ),

            "confidence":
                0.90,

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

    return validate_alert(
        payload
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
    args = parse_args()

    if args.alerts < 100:
        raise ValueError(
            "--alerts must be at least 100 for this benchmark."
        )

    if args.retention_days <= 0:
        raise ValueError(
            "--retention-days must be greater than zero."
        )

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARTIFACT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean_database_files(
        DATABASE_FILE
    )

    store = CentralAlertStore(
        DATABASE_FILE
    )

    pseudonymizer = HMACPseudonymizer(
        key=
            generate_hmac_key_bytes(
                32
            )
    )

    benchmark_now = datetime.now(
        timezone.utc
    ).replace(
        microsecond=0
    )

    # Spread synthetic alerts uniformly over 40 days so a 30-day
    # retention policy has a deterministic non-empty deletion set.
    total_span_seconds = (
        40
        *
        24
        *
        60
        *
        60
    )

    print()
    print(
        "="
        *
        128
    )

    print(
        PHASE_NAME
    )

    print(
        "="
        *
        128
    )

    print(
        f"Benchmark alerts              : {args.alerts}"
    )

    print(
        f"Clients                       : 5"
    )

    print(
        f"Attack classes                : 5"
    )

    print(
        f"Retention policy              : {args.retention_days} days"
    )

    print(
        f"Query repeats                 : {args.query_repeats}"
    )

    print(
        f"Database                      : {DATABASE_FILE}"
    )

    print(
        f"Locked test                   : NO"
    )

    print()
    print(
        "SCIENTIFIC NOTE:"
    )

    print(
        "This is a local SQLite implementation benchmark using "
        "synthetic privacy-safe alert messages."
    )

    print(
        "It is NOT a network-scale throughput claim and NOT IDS "
        "detection-performance evidence."
    )

    insert_latencies = []

    shared_source_token = pseudonymizer.source_ip(
        "203.0.113.250"
    )

    raw_test_literals = {
        "203.0.113.250",
        "198.51.100.10",
    }

    print()
    print(
        "INGESTION BENCHMARK"
    )

    print(
        "-"
        *
        128
    )

    start_all = time.perf_counter()

    for index in range(
        args.alerts
    ):
        fraction = (
            index
            /
            max(
                args.alerts
                -
                1,
                1,
            )
        )

        age_seconds = int(
            total_span_seconds
            *
            (
                1.0
                -
                fraction
            )
        )

        event_time = (
            benchmark_now
            -
            timedelta(
                seconds=
                    age_seconds
            )
        )

        alert = make_alert(
            index=
                index,

            event_time=
                event_time,

            pseudonymizer=
                pseudonymizer,
        )

        start = time.perf_counter()

        store.insert_alert(
            alert
        )

        elapsed_ms = (
            time.perf_counter()
            -
            start
        ) * 1000.0

        insert_latencies.append(
            elapsed_ms
        )

    total_ingest_seconds = (
        time.perf_counter()
        -
        start_all
    )

    throughput = (
        args.alerts
        /
        total_ingest_seconds
    )

    insert_summary = summarize_latencies(
        insert_latencies
    )

    count_before_retention = store.count()

    print(
        f"Inserted                      : "
        f"{count_before_retention}/{args.alerts}"
    )

    print(
        f"Mean insert latency           : "
        f"{insert_summary['mean_ms']:.3f} ms"
    )

    print(
        f"P95 insert latency            : "
        f"{insert_summary['p95_ms']:.3f} ms"
    )

    print(
        f"Sequential ingest throughput  : "
        f"{throughput:.2f} alerts/sec"
    )

    # ========================================================
    # Privacy-at-rest audit
    # ========================================================

    privacy_audit = store.privacy_at_rest_audit(
        forbidden_literals=sorted(
            raw_test_literals
            |
            {
                "payload",
                "flow_features",
                "cnn_sequence",
                "model_state_dict",
            }
        )
    )

    privacy_pass = bool(
        privacy_audit[
            "passed"
        ]
    )

    print()
    print(
        f"[{'PASS' if privacy_pass else 'FAIL'}] "
        "privacy_at_rest_audit"
    )

    # ========================================================
    # Query benchmarks
    # ========================================================

    print()
    print(
        "INDEXED QUERY BENCHMARK"
    )

    print(
        "-"
        *
        128
    )

    client_query = timed_sql_query(
        DATABASE_FILE,

        sql="""
            SELECT alert_id
            FROM alerts
            WHERE client_id = ?
            ORDER BY event_time DESC
            LIMIT 100
        """,

        params=(
            "client_3",
        ),

        repeats=
            args.query_repeats,
    )

    source_query = timed_sql_query(
        DATABASE_FILE,

        sql="""
            SELECT alert_id
            FROM alerts
            WHERE source_token = ?
            ORDER BY event_time
            LIMIT 500
        """,

        params=(
            shared_source_token,
        ),

        repeats=
            args.query_repeats,
    )

    recent_query = timed_sql_query(
        DATABASE_FILE,

        sql="""
            SELECT alert_id
            FROM alerts
            ORDER BY event_time DESC
            LIMIT 100
        """,

        repeats=
            args.query_repeats,
    )

    classification_query = timed_sql_query(
        DATABASE_FILE,

        sql="""
            SELECT alert_id
            FROM alerts
            WHERE classification = ?
              AND severity IN ('HIGH', 'CRITICAL')
            ORDER BY event_time DESC
            LIMIT 100
        """,

        params=(
            "PortScan",
        ),

        repeats=
            args.query_repeats,
    )

    print(
        f"By client mean                : "
        f"{client_query['mean_ms']:.4f} ms"
    )

    print(
        f"Shared source mean            : "
        f"{source_query['mean_ms']:.4f} ms"
    )

    print(
        f"Recent alerts mean            : "
        f"{recent_query['mean_ms']:.4f} ms"
    )

    print(
        f"Class + severity mean         : "
        f"{classification_query['mean_ms']:.4f} ms"
    )

    # ========================================================
    # Retention audit
    # ========================================================

    now_iso = benchmark_now.isoformat()

    before_range = oldest_newest_event_time(
        DATABASE_FILE
    )

    preview = retention_preview(
        DATABASE_FILE,

        now_utc=
            now_iso,

        retention_days=
            args.retention_days,
    )

    preview_pass = (
        preview.matching_rows
        >
        0
        and
        preview.deleted_rows
        ==
        0
        and
        store.count()
        ==
        count_before_retention
    )

    print()
    print(
        "RETENTION POLICY AUDIT"
    )

    print(
        "-"
        *
        128
    )

    print(
        f"[{'PASS' if preview_pass else 'FAIL'}] "
        f"dry_run_preview: {preview.matching_rows} rows eligible"
    )

    apply_result = apply_retention_policy(
        DATABASE_FILE,

        now_utc=
            now_iso,

        retention_days=
            args.retention_days,
    )

    expected_remaining = (
        count_before_retention
        -
        preview.matching_rows
    )

    retention_apply_pass = (
        apply_result.deleted_rows
        ==
        preview.matching_rows
        and
        apply_result.remaining_rows
        ==
        expected_remaining
        and
        store.count()
        ==
        expected_remaining
    )

    print(
        f"[{'PASS' if retention_apply_pass else 'FAIL'}] "
        f"retention_apply: deleted={apply_result.deleted_rows}, "
        f"remaining={apply_result.remaining_rows}"
    )

    after_range = oldest_newest_event_time(
        DATABASE_FILE
    )

    post_retention_privacy = store.privacy_at_rest_audit(
        forbidden_literals=sorted(
            raw_test_literals
            |
            {
                "payload",
                "flow_features",
                "cnn_sequence",
                "model_state_dict",
            }
        )
    )

    post_privacy_pass = bool(
        post_retention_privacy[
            "passed"
        ]
    )

    print(
        f"[{'PASS' if post_privacy_pass else 'FAIL'}] "
        "post_retention_privacy_audit"
    )

    # ========================================================
    # Final
    # ========================================================

    all_passed = (
        count_before_retention
        ==
        args.alerts
        and
        privacy_pass
        and
        preview_pass
        and
        retention_apply_pass
        and
        post_privacy_pass
    )

    performance = {
        "phase":
            "15G",

        "benchmark_alerts":
            args.alerts,

        "insert_latency":
            insert_summary,

        "sequential_ingest_seconds":
            total_ingest_seconds,

        "sequential_ingest_throughput_alerts_per_second":
            throughput,

        "indexed_queries":
            {
                "client_id":
                    client_query,

                "source_token":
                    source_query,

                "recent_alerts":
                    recent_query,

                "classification_plus_severity":
                    classification_query,
            },

        "database_bytes_before_process_end":
            (
                DATABASE_FILE.stat().st_size
                if DATABASE_FILE.exists()
                else
                0
            ),

        "benchmark_scope":
            (
                "single-process local SQLite implementation benchmark"
            ),

        "network_throughput_claim":
            False,

        "locked_test_used":
            False,
    }

    save_json(
        PERFORMANCE_FILE,
        performance,
    )

    audit = {
        "phase":
            "15G",

        "phase_name":
            PHASE_NAME,

        "privacy_at_rest_before_retention":
            privacy_audit,

        "retention_preview":
            preview.to_dict(),

        "retention_apply":
            apply_result.to_dict(),

        "event_time_range_before":
            before_range,

        "event_time_range_after":
            after_range,

        "post_retention_privacy_audit":
            post_retention_privacy,

        "all_tests_passed":
            all_passed,

        "synthetic_privacy_safe_benchmark_alerts":
            True,

        "locked_test_used":
            False,
    }

    save_json(
        AUDIT_FILE,
        audit,
    )

    manifest = {
        "phase":
            "15G",

        "privacy_audit":
            "explicit raw/sensitive literal scan over persisted canonical JSON",

        "retention_policy":
            {
                "basis":
                    "event_time",

                "retention_days":
                    args.retention_days,

                "dry_run_before_delete":
                    True,

                "test_database_only":
                    True,
            },

        "performance_evaluation":
            {
                "backend":
                    "SQLite",

                "mode":
                    "single-process sequential insertion",

                "query_repeats":
                    args.query_repeats,

                "indexed_queries":
                    [
                        "client_id",
                        "source_token",
                        "recent event_time",
                        "classification + severity",
                    ],
            },

        "important_limitations":
            [
                "No network transport is included in throughput.",
                "No concurrent multi-writer benchmark is claimed.",
                "Results are hardware and implementation specific.",
                "Synthetic privacy-safe messages are not detection evidence.",
            ],

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
        128
    )

    print(
        "PHASE 15G COMPLETE"
    )

    print(
        "="
        *
        128
    )

    print(
        f"Benchmark alerts inserted     : "
        f"{count_before_retention}/{args.alerts}"
    )

    print(
        f"Privacy-at-rest audit         : "
        f"{'PASS' if privacy_pass else 'FAIL'}"
    )

    print(
        f"Retention dry-run             : "
        f"{'PASS' if preview_pass else 'FAIL'}"
    )

    print(
        f"Retention application         : "
        f"{'PASS' if retention_apply_pass else 'FAIL'}"
    )

    print(
        f"Rows deleted                  : "
        f"{apply_result.deleted_rows}"
    )

    print(
        f"Rows retained                 : "
        f"{apply_result.remaining_rows}"
    )

    print(
        f"Post-retention privacy audit  : "
        f"{'PASS' if post_privacy_pass else 'FAIL'}"
    )

    print(
        f"Mean insert latency           : "
        f"{insert_summary['mean_ms']:.3f} ms"
    )

    print(
        f"P95 insert latency            : "
        f"{insert_summary['p95_ms']:.3f} ms"
    )

    print(
        f"Sequential store throughput   : "
        f"{throughput:.2f} alerts/sec"
    )

    print(
        f"Audit                         : {AUDIT_FILE}"
    )

    print(
        f"Performance                   : {PERFORMANCE_FILE}"
    )

    print(
        f"Manifest                      : {MANIFEST_FILE}"
    )

    print(
        f"Locked test used              : NO"
    )

    print()

    print(
        f"STATUS                        : "
        f"{'PASS' if all_passed else 'FAIL'}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Do not report this local SQLite throughput as cloud/network "
        "ingestion throughput."
    )

    print(
        "The benchmark isolates the Phase15 central-store implementation."
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Phase 15H — final Phase15 end-to-end audit / report freeze, "
        "then Phase16 cross-cloud correlation."
    )

    if not all_passed:
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()
