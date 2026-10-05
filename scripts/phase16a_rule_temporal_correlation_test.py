from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
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


from src.alert_sharing.alert_schema import validate_alert  # noqa: E402
from src.correlation.rule_temporal_correlator import (  # noqa: E402
    RULE_DESTINATION_MULTI_SOURCE,
    RULE_MULTI_CLASS_SOURCE,
    RULE_SHARED_SOURCE,
    CorrelationConfig,
    RuleTemporalCorrelator,
    privacy_audit_correlation_events,
)


PHASE_NAME = (
    "PHASE 16A — PRIVACY-SAFE RULE / TEMPORAL CROSS-CLOUD CORRELATION"
)


RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "phase16"
)

ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "correlation"
)

PHASE15F_DB = (
    PROJECT_ROOT
    / "results"
    / "phase15"
    / "phase15f_central_alerts.sqlite3"
)

CONFIG_FILE = (
    PROJECT_ROOT
    / "configs"
    / "phase16a_correlation_policy.json"
)

CORRELATIONS_FILE = (
    RESULT_ROOT
    / "phase16a_correlations.json"
)

AUDIT_FILE = (
    RESULT_ROOT
    / "phase16a_correlation_audit.json"
)

MANIFEST_FILE = (
    ARTIFACT_ROOT
    / "phase16a_correlation_manifest.json"
)


def save_json(
    path: Path,
    payload: Dict[str, Any] | List[Dict[str, Any]],
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


def load_config() -> CorrelationConfig:
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Missing Phase16A config: {CONFIG_FILE}"
        )

    with open(
        CONFIG_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(
            file
        )

    return CorrelationConfig(
        window_seconds=
            int(
                payload[
                    "window_seconds"
                ]
            ),

        min_cross_client_clients=
            int(
                payload[
                    "min_cross_client_clients"
                ]
            ),

        min_multi_class_clients=
            int(
                payload[
                    "min_multi_class_clients"
                ]
            ),

        min_multi_class_classes=
            int(
                payload[
                    "min_multi_class_classes"
                ]
            ),

        min_destination_clients=
            int(
                payload[
                    "min_destination_clients"
                ]
            ),

        min_destination_sources=
            int(
                payload[
                    "min_destination_sources"
                ]
            ),
    ).validate()


def load_phase15f_alerts() -> List[Dict[str, Any]]:
    if not PHASE15F_DB.exists():
        raise FileNotFoundError(
            "\nMissing Phase15F central alert database:\n"
            f"{PHASE15F_DB}\n\n"
            "Run Phase15F first."
        )

    connection = sqlite3.connect(
        PHASE15F_DB
    )

    connection.row_factory = sqlite3.Row

    try:
        rows = connection.execute(
            """
            SELECT canonical_json
            FROM alerts
            ORDER BY event_time, id
            """
        ).fetchall()

    finally:
        connection.close()

    alerts = []

    for row in rows:
        payload = json.loads(
            row[
                "canonical_json"
            ]
        )

        # Reuse the frozen Phase15A validator.
        alerts.append(
            validate_alert(
                payload
            ).to_dict()
        )

    return alerts


def main():
    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARTIFACT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = load_config()

    alerts = load_phase15f_alerts()

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
        f"Input database                : {PHASE15F_DB}"
    )

    print(
        f"Input privacy-safe alerts     : {len(alerts)}"
    )

    print(
        f"Temporal window               : {config.window_seconds} seconds"
    )

    print(
        f"Rules                         : 3"
    )

    print(
        f"Raw identifiers required      : NO"
    )

    print(
        f"Locked test                   : NO"
    )

    print()
    print(
        "SCIENTIFIC NOTE:"
    )

    print(
        "Phase16A uses the synthetic privacy-safe Phase15F ingestion dataset."
    )

    print(
        "Therefore this validates correlation logic/functionality only; "
        "it is not detection accuracy evidence."
    )

    correlator = RuleTemporalCorrelator(
        config
    )

    events = correlator.correlate(
        alerts
    )

    event_payloads = [
        event.to_dict()
        for event
        in events
    ]

    rule_counts = Counter(
        event[
            "rule_id"
        ]
        for event
        in event_payloads
    )

    print()
    print(
        "CORRELATION EVENTS"
    )

    print(
        "-"
        *
        128
    )

    for index, event in enumerate(
        event_payloads,
        start=1,
    ):
        print(
            f"[EVENT {index:02d}] "
            f"{event['rule_id']} | "
            f"priority={event['correlation_priority']} | "
            f"alerts={event['alert_count']} | "
            f"clients={event['distinct_clients']} | "
            f"classes={event['distinct_classifications']} | "
            f"sources={event['distinct_sources']}"
        )

    # --------------------------------------------------------
    # Controlled expectations from the frozen Phase15F
    # synthetic ingestion construction.
    # --------------------------------------------------------

    shared_source_events = [
        event
        for event
        in event_payloads
        if event[
            "rule_id"
        ]
        ==
        RULE_SHARED_SOURCE
    ]

    multi_class_events = [
        event
        for event
        in event_payloads
        if event[
            "rule_id"
        ]
        ==
        RULE_MULTI_CLASS_SOURCE
    ]

    destination_events = [
        event
        for event
        in event_payloads
        if event[
            "rule_id"
        ]
        ==
        RULE_DESTINATION_MULTI_SOURCE
    ]

    shared_source_pass = (
        len(
            shared_source_events
        )
        ==
        1
        and
        shared_source_events[
            0
        ][
            "distinct_clients"
        ]
        ==
        5
        and
        shared_source_events[
            0
        ][
            "alert_count"
        ]
        ==
        5
    )

    multi_class_pass = (
        len(
            multi_class_events
        )
        ==
        1
        and
        multi_class_events[
            0
        ][
            "distinct_clients"
        ]
        ==
        5
        and
        multi_class_events[
            0
        ][
            "distinct_classifications"
        ]
        ==
        5
    )

    destination_pass = (
        len(
            destination_events
        )
        ==
        1
        and
        destination_events[
            0
        ][
            "distinct_clients"
        ]
        ==
        5
        and
        destination_events[
            0
        ][
            "distinct_sources"
        ]
        ==
        6
        and
        destination_events[
            0
        ][
            "alert_count"
        ]
        ==
        20
    )

    total_events_pass = (
        len(
            event_payloads
        )
        ==
        3
    )

    privacy_audit = privacy_audit_correlation_events(
        event_payloads
    )

    privacy_pass = bool(
        privacy_audit[
            "passed"
        ]
    )

    no_risk_recalculation_pass = all(
        "risk_score"
        not in
        event
        and
        "confidence"
        not in
        event
        and
        "evidence_score"
        not in
        event
        for event
        in event_payloads
    )

    print()
    print(
        "CONTROLLED AUDIT"
    )

    print(
        "-"
        *
        128
    )

    print(
        f"[{'PASS' if shared_source_pass else 'FAIL'}] "
        "R1 shared source across 5 clients"
    )

    print(
        f"[{'PASS' if multi_class_pass else 'FAIL'}] "
        "R2 shared source across 5 classifications"
    )

    print(
        f"[{'PASS' if destination_pass else 'FAIL'}] "
        "R3 shared destination/port from 6 sources across 5 clients"
    )

    print(
        f"[{'PASS' if total_events_pass else 'FAIL'}] "
        f"expected correlation event count: {len(event_payloads)}/3"
    )

    print(
        f"[{'PASS' if privacy_pass else 'FAIL'}] "
        "correlation-output privacy audit"
    )

    print(
        f"[{'PASS' if no_risk_recalculation_pass else 'FAIL'}] "
        "no Phase14 risk/confidence reinterpretation"
    )

    all_passed = all(
        [
            shared_source_pass,
            multi_class_pass,
            destination_pass,
            total_events_pass,
            privacy_pass,
            no_risk_recalculation_pass,
        ]
    )

    save_json(
        CORRELATIONS_FILE,
        event_payloads,
    )

    audit = {
        "phase":
            "16A",

        "phase_name":
            PHASE_NAME,

        "input_source":
            "Phase15F synthetic privacy-safe central alert store",

        "input_alert_count":
            len(
                alerts
            ),

        "window_seconds":
            config.window_seconds,

        "rules":
            {
                RULE_SHARED_SOURCE:
                    {
                        "events":
                            rule_counts.get(
                                RULE_SHARED_SOURCE,
                                0,
                            ),

                        "passed":
                            shared_source_pass,
                    },

                RULE_MULTI_CLASS_SOURCE:
                    {
                        "events":
                            rule_counts.get(
                                RULE_MULTI_CLASS_SOURCE,
                                0,
                            ),

                        "passed":
                            multi_class_pass,
                    },

                RULE_DESTINATION_MULTI_SOURCE:
                    {
                        "events":
                            rule_counts.get(
                                RULE_DESTINATION_MULTI_SOURCE,
                                0,
                            ),

                        "passed":
                            destination_pass,
                    },
            },

        "total_correlation_events":
            len(
                event_payloads
            ),

        "expected_total_events":
            3,

        "privacy_audit":
            privacy_audit,

        "phase14_risk_recalculated":
            False,

        "correlation_priority_is_probability":
            False,

        "correlation_priority_is_phase14_risk":
            False,

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
            "16A",

        "version":
            "phase16a_v1",

        "correlation_type":
            "deterministic rule + event-time temporal correlation",

        "input_contract":
            "Phase15 frozen privacy-safe alerts only",

        "rules":
            [
                {
                    "rule_id":
                        RULE_SHARED_SOURCE,

                    "meaning":
                        "same source pseudonym across multiple clients",
                },
                {
                    "rule_id":
                        RULE_MULTI_CLASS_SOURCE,

                    "meaning":
                        "same source pseudonym across multiple attack classifications and clients",
                },
                {
                    "rule_id":
                        RULE_DESTINATION_MULTI_SOURCE,

                    "meaning":
                        "same destination pseudonym/port targeted by multiple pseudonymous sources across clients",
                },
            ],

        "window_strategy":
            (
                "deterministic non-overlapping windows anchored at first event"
            ),

        "important_limitations":
            [
                "Boundary-sensitive baseline windowing.",
                "Rule thresholds are engineering baseline values, not statistically optimized.",
                "Correlation priority is deterministic rule priority, not a probability.",
                "Phase15F input messages are synthetic infrastructure-test alerts.",
                "No claim of correlation precision/recall is made in Phase16A.",
                "No locked test is used.",
            ],

        "raw_identifiers_used":
            False,

        "phase14_risk_semantics_modified":
            False,

        "ready_for_next_stage":
            all_passed,
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
        "PHASE 16A COMPLETE"
    )

    print(
        "="
        *
        128
    )

    print(
        f"Input alerts                  : {len(alerts)}"
    )

    print(
        f"Correlation events            : {len(event_payloads)}"
    )

    print(
        f"R1 shared-source events       : "
        f"{rule_counts.get(RULE_SHARED_SOURCE, 0)}"
    )

    print(
        f"R2 multi-class-source events  : "
        f"{rule_counts.get(RULE_MULTI_CLASS_SOURCE, 0)}"
    )

    print(
        f"R3 shared-destination events  : "
        f"{rule_counts.get(RULE_DESTINATION_MULTI_SOURCE, 0)}"
    )

    print(
        f"Privacy audit                 : "
        f"{'PASS' if privacy_pass else 'FAIL'}"
    )

    print(
        f"Locked test used              : NO"
    )

    print(
        f"Correlations                  : {CORRELATIONS_FILE}"
    )

    print(
        f"Audit                         : {AUDIT_FILE}"
    )

    print(
        f"Manifest                      : {MANIFEST_FILE}"
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
        "Correlation priority is a deterministic rule-engine priority only."
    )

    print(
        "It must not be interpreted as Phase14 confidence, fused risk, "
        "or a calibrated probability."
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Phase 16B — temporal-rule validation dataset + controlled "
        "precision/recall and threshold/window sensitivity."
    )

    if not all_passed:
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()
