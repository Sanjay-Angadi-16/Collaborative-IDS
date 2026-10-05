from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import itertools
import json
import random
from typing import Any, Dict, List, Mapping, Sequence, Tuple
import uuid

from .rule_temporal_correlator import (
    RULE_DESTINATION_MULTI_SOURCE,
    RULE_MULTI_CLASS_SOURCE,
    RULE_SHARED_SOURCE,
    CorrelationConfig,
)
from .phase16b_validation import (
    LabeledScenario,
    evaluate_scenarios,
    run_correlator_defensively,
)


PHASE16C_VERSION = "phase16c_v1"


@dataclass(frozen=True)
class SelectionPolicy:
    """
    Predeclared lexicographic selection rule.

    1. Baseline must remain perfect on Phase16B controlled validation.
    2. Baseline must pass all Phase16B exceptional scenarios.
    3. An alternative may replace the baseline only if it provides a material
       improvement in F1 or false-correlation rate without reducing recall.
    4. If no material improvement exists, preserve the existing Phase16A/16B
       baseline to avoid post-hoc threshold switching.
    """
    material_f1_improvement: float = 0.01
    material_false_corr_improvement: float = 0.01
    minimum_recall: float = 1.0


def choose_configuration(
    *,
    phase16b_audit: Mapping[str, Any],
    sensitivity_rows: Sequence[Mapping[str, Any]],
    policy: SelectionPolicy,
) -> Dict[str, Any]:
    baseline = dict(
        phase16b_audit["baseline_grid_result"]
    )

    best = dict(
        phase16b_audit["best_validation_grid_result"]
    )

    exceptional_pass = bool(
        phase16b_audit[
            "exceptional_case_validation_passed"
        ]
    )

    baseline_validation_pass = bool(
        phase16b_audit[
            "baseline_ground_truth_validation_passed"
        ]
    )

    baseline_eligible = (
        baseline_validation_pass
        and
        exceptional_pass
        and
        float(
            baseline["recall"]
        )
        >=
        policy.minimum_recall
    )

    if not baseline_eligible:
        raise RuntimeError(
            "The existing Phase16A baseline is not eligible for freezing."
        )

    baseline_f1 = float(
        baseline["f1"]
    )

    baseline_false_corr = float(
        baseline["false_correlation_rate"]
    )

    best_f1 = float(
        best["f1"]
    )

    best_false_corr = float(
        best["false_correlation_rate"]
    )

    best_recall = float(
        best["recall"]
    )

    f1_gain = (
        best_f1
        -
        baseline_f1
    )

    false_corr_gain = (
        baseline_false_corr
        -
        best_false_corr
    )

    alternative_materially_better = (
        best_recall
        >=
        policy.minimum_recall
        and
        (
            f1_gain
            >=
            policy.material_f1_improvement
            or
            false_corr_gain
            >=
            policy.material_false_corr_improvement
        )
    )

    if alternative_materially_better:
        selected = best
        reason = (
            "Alternative selected because it materially improved the "
            "predeclared validation objective without reducing recall."
        )
        changed_from_baseline = True

    else:
        selected = baseline
        reason = (
            "Existing 60-second baseline retained because no candidate "
            "materially improved F1 or false-correlation rate. Equal-scoring "
            "alternatives are not adopted post hoc."
        )
        changed_from_baseline = False

    return {
        "selected": selected,
        "baseline": baseline,
        "best_ranked_validation_row": best,
        "changed_from_phase16a_baseline": changed_from_baseline,
        "selection_reason": reason,
        "observed_f1_gain_of_best_row": f1_gain,
        "observed_false_correlation_gain_of_best_row": false_corr_gain,
        "policy": {
            "material_f1_improvement": policy.material_f1_improvement,
            "material_false_corr_improvement": policy.material_false_corr_improvement,
            "minimum_recall": policy.minimum_recall,
        },
    }


def config_from_row(
    row: Mapping[str, Any],
) -> CorrelationConfig:
    return CorrelationConfig(
        window_seconds=int(
            row[
                "window_seconds"
            ]
        ),
        min_cross_client_clients=int(
            row[
                "r1_min_clients"
            ]
        ),
        min_multi_class_clients=int(
            row[
                "r2_min_clients"
            ]
        ),
        min_multi_class_classes=int(
            row[
                "r2_min_classes"
            ]
        ),
        min_destination_clients=int(
            row[
                "r3_min_clients"
            ]
        ),
        min_destination_sources=int(
            row[
                "r3_min_sources"
            ]
        ),
    ).validate()


def _iso(
    value: datetime,
) -> str:
    return value.astimezone(
        timezone.utc
    ).replace(
        microsecond=0
    ).isoformat()


def _token(
    secret: bytes,
    namespace: str,
    value: str,
    prefix: str,
) -> str:
    message = (
        "phase16c|"
        +
        namespace
        +
        "|"
        +
        value
    ).encode(
        "utf-8"
    )

    return (
        prefix
        +
        "_"
        +
        hmac.new(
            secret,
            message,
            hashlib.sha256,
        ).hexdigest()[
            :32
        ]
    )


def _alert(
    *,
    client: int,
    event_time: datetime,
    classification: str,
    source_token: str,
    destination_token: str,
    destination_port: int = 443,
    alert_id: str | None = None,
    arrival_time: datetime | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "alert_id":
            alert_id
            or
            str(
                uuid.uuid4()
            ),

        "client_id":
            f"client_{client}",

        "event_time":
            _iso(
                event_time
            ),

        "classification":
            classification,

        "severity":
            "HIGH",

        "confidence":
            0.98,

        "risk_score":
            None,

        "evidence_score":
            0.98,

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
            int(
                destination_port
            ),

        "model_version":
            "phase14_final",

        "policy_version":
            "phase14g2_v1",

        "privacy_version":
            "phase15_v1",

        "schema_version":
            "phase15a_v1",
    }

    if arrival_time is not None:
        payload[
            "arrival_time"
        ] = _iso(
            arrival_time
        )

    return payload


def build_boundary_scenarios(
    *,
    window_seconds: int,
    seed: int = 42,
) -> List[LabeledScenario]:
    """
    Characterize the exact temporal boundary around the selected window.

    These are characterization scenarios, not a new parameter search.
    """
    secret = hashlib.sha256(
        f"phase16c-boundary-{seed}".encode(
            "utf-8"
        )
    ).digest()

    base = datetime(
        2026,
        9,
        14,
        13,
        0,
        0,
        tzinfo=timezone.utc,
    )

    def src(name: str) -> str:
        return _token(
            secret,
            "source",
            name,
            "src",
        )

    def dst(name: str) -> str:
        return _token(
            secret,
            "destination",
            name,
            "dst",
        )

    scenarios: List[
        LabeledScenario
    ] = []

    # R1/R2 exact boundary: same source across two clients,
    # different classes. <= window must correlate, window+1 must not.
    for separation in (
        max(
            1,
            window_seconds
            -
            1,
        ),
        window_seconds,
        window_seconds
        +
        1,
    ):
        expected = (
            separation
            <=
            window_seconds
        )

        expected_rules = (
            (
                RULE_SHARED_SOURCE,
                RULE_MULTI_CLASS_SOURCE,
            )
            if expected
            else
            ()
        )

        source = src(
            f"r12-{separation}"
        )

        alerts = [
            _alert(
                client=1,
                event_time=base,
                classification="DoS Hulk",
                source_token=source,
                destination_token=dst(
                    f"r12-a-{separation}"
                ),
            ),
            _alert(
                client=2,
                event_time=base
                +
                timedelta(
                    seconds=separation
                ),
                classification="DDoS",
                source_token=source,
                destination_token=dst(
                    f"r12-b-{separation}"
                ),
            ),
        ]

        scenarios.append(
            LabeledScenario(
                scenario_id=
                    f"B_R12_SEPARATION_{separation}s",

                description=
                    (
                        "Two-client shared-source boundary "
                        f"at {separation} seconds."
                    ),

                alerts=
                    alerts,

                expected_correlated=
                    expected,

                expected_rules=
                    expected_rules,

                exceptional_case=
                    "TEMPORAL_BOUNDARY",
            )
        )

    # R3 exact boundary: 3 clients/sources to same destination.
    for separation in (
        window_seconds,
        window_seconds
        +
        1,
    ):
        shared_destination = dst(
            f"r3-target-{separation}"
        )

        alerts = [
            _alert(
                client=1,
                event_time=base,
                classification="DoS Hulk",
                source_token=src(
                    f"r3-a-{separation}"
                ),
                destination_token=
                    shared_destination,
            ),
            _alert(
                client=2,
                event_time=base
                +
                timedelta(
                    seconds=
                        separation
                        //
                        2
                ),
                classification="DDoS",
                source_token=src(
                    f"r3-b-{separation}"
                ),
                destination_token=
                    shared_destination,
            ),
            _alert(
                client=3,
                event_time=base
                +
                timedelta(
                    seconds=
                        separation
                ),
                classification="PortScan",
                source_token=src(
                    f"r3-c-{separation}"
                ),
                destination_token=
                    shared_destination,
            ),
        ]

        expected = (
            separation
            <=
            window_seconds
        )

        scenarios.append(
            LabeledScenario(
                scenario_id=
                    f"B_R3_SPAN_{separation}s",

                description=
                    (
                        "Three-source shared-destination boundary "
                        f"at total span {separation} seconds."
                    ),

                alerts=
                    alerts,

                expected_correlated=
                    expected,

                expected_rules=
                    (
                        (
                            RULE_DESTINATION_MULTI_SOURCE,
                        )
                        if expected
                        else
                        ()
                    ),

                exceptional_case=
                    "TEMPORAL_BOUNDARY",
            )
        )

    return scenarios


def permutation_invariance_test(
    *,
    config: CorrelationConfig,
    permutations_to_test: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    secret = hashlib.sha256(
        b"phase16c-permutation"
    ).digest()

    base = datetime(
        2026,
        9,
        14,
        14,
        0,
        0,
        tzinfo=timezone.utc,
    )

    source = _token(
        secret,
        "source",
        "same",
        "src",
    )

    alerts = [
        _alert(
            client=1,
            event_time=base,
            classification="DoS Hulk",
            source_token=source,
            destination_token=_token(
                secret,
                "destination",
                "a",
                "dst",
            ),
        ),
        _alert(
            client=2,
            event_time=base
            +
            timedelta(
                seconds=10
            ),
            classification="DDoS",
            source_token=source,
            destination_token=_token(
                secret,
                "destination",
                "b",
                "dst",
            ),
        ),
        _alert(
            client=3,
            event_time=base
            +
            timedelta(
                seconds=20
            ),
            classification="PortScan",
            source_token=source,
            destination_token=_token(
                secret,
                "destination",
                "c",
                "dst",
            ),
        ),
    ]

    reference = run_correlator_defensively(
        alerts,
        config,
    )

    reference_rules = sorted(
        {
            event[
                "rule_id"
            ]
            for event
            in reference[
                "events"
            ]
        }
    )

    rng = random.Random(
        seed
    )

    passed = True
    tested = 0

    for _ in range(
        permutations_to_test
    ):
        candidate = [
            dict(
                alert
            )
            for alert
            in alerts
        ]

        rng.shuffle(
            candidate
        )

        outcome = run_correlator_defensively(
            candidate,
            config,
        )

        rules = sorted(
            {
                event[
                    "rule_id"
                ]
                for event
                in outcome[
                    "events"
                ]
            }
        )

        tested += 1

        if rules != reference_rules:
            passed = False
            break

    return {
        "passed":
            passed,

        "permutations_tested":
            tested,

        "reference_rules":
            reference_rules,
    }


def delayed_arrival_invariance_test(
    *,
    config: CorrelationConfig,
) -> Dict[str, Any]:
    secret = hashlib.sha256(
        b"phase16c-delay"
    ).digest()

    base = datetime(
        2026,
        9,
        14,
        15,
        0,
        0,
        tzinfo=timezone.utc,
    )

    source = _token(
        secret,
        "source",
        "same",
        "src",
    )

    delays = (
        1,
        5,
        120,
    )

    alerts = []

    for client, delay, offset, attack in (
        (
            1,
            delays[
                0
            ],
            0,
            "DoS Hulk",
        ),
        (
            2,
            delays[
                1
            ],
            10,
            "DDoS",
        ),
        (
            3,
            delays[
                2
            ],
            20,
            "PortScan",
        ),
    ):
        event_time = (
            base
            +
            timedelta(
                seconds=offset
            )
        )

        alerts.append(
            _alert(
                client=client,
                event_time=event_time,
                classification=attack,
                source_token=source,
                destination_token=_token(
                    secret,
                    "destination",
                    f"d{client}",
                    "dst",
                ),
                arrival_time=event_time
                +
                timedelta(
                    seconds=delay
                ),
            )
        )

    alerts.sort(
        key=lambda item: item[
            "arrival_time"
        ]
    )

    outcome = run_correlator_defensively(
        alerts,
        config,
    )

    rules = sorted(
        {
            event[
                "rule_id"
            ]
            for event
            in outcome[
                "events"
            ]
        }
    )

    expected = sorted(
        [
            RULE_SHARED_SOURCE,
            RULE_MULTI_CLASS_SOURCE,
        ]
    )

    return {
        "passed":
            rules
            ==
            expected,

        "arrival_delays_seconds":
            list(
                delays
            ),

        "predicted_rules":
            rules,

        "expected_rules":
            expected,

        "uses_event_time_for_correlation":
            True,
    }


def missing_evidence_boundary_test(
    *,
    config: CorrelationConfig,
) -> Dict[str, Any]:
    secret = hashlib.sha256(
        b"phase16c-missing"
    ).digest()

    base = datetime(
        2026,
        9,
        14,
        16,
        0,
        0,
        tzinfo=timezone.utc,
    )

    source = _token(
        secret,
        "source",
        "same",
        "src",
    )

    full = [
        _alert(
            client=i,
            event_time=base
            +
            timedelta(
                seconds=
                    (i - 1)
                    *
                    10
            ),
            classification=attack,
            source_token=source,
            destination_token=_token(
                secret,
                "destination",
                f"m{i}",
                "dst",
            ),
        )
        for i, attack
        in [
            (
                1,
                "DoS Hulk",
            ),
            (
                2,
                "DDoS",
            ),
            (
                3,
                "PortScan",
            ),
        ]
    ]

    results = []

    for keep_count in (
        3,
        2,
        1,
    ):
        subset = full[
            :keep_count
        ]

        outcome = run_correlator_defensively(
            subset,
            config,
        )

        correlated = (
            len(
                outcome[
                    "events"
                ]
            )
            >
            0
        )

        expected = (
            keep_count
            >=
            config.min_cross_client_clients
        )

        results.append(
            {
                "alerts_retained":
                    keep_count,

                "predicted_correlated":
                    correlated,

                "expected_correlated":
                    expected,

                "passed":
                    correlated
                    ==
                    expected,
            }
        )

    return {
        "passed":
            all(
                row[
                    "passed"
                ]
                for row
                in results
            ),

        "cases":
            results,
    }


def replay_characterization_test(
    *,
    config: CorrelationConfig,
) -> Dict[str, Any]:
    """
    Characterize two replay cases:
    A. exact alert_id replay -> removed by current defensive dedup
    B. same semantic alert with a NEW alert_id -> not removed

    Case B is recorded as a known limitation rather than silently "fixed",
    because collapsing semantically identical alerts can also remove legitimate
    repeated events.
    """
    secret = hashlib.sha256(
        b"phase16c-replay"
    ).digest()

    base = datetime(
        2026,
        9,
        14,
        17,
        0,
        0,
        tzinfo=timezone.utc,
    )

    source = _token(
        secret,
        "source",
        "same",
        "src",
    )

    a = _alert(
        client=1,
        event_time=base,
        classification="DoS Hulk",
        source_token=source,
        destination_token=_token(
            secret,
            "destination",
            "a",
            "dst",
        ),
    )

    b = _alert(
        client=2,
        event_time=base
        +
        timedelta(
            seconds=10
        ),
        classification="DDoS",
        source_token=source,
        destination_token=_token(
            secret,
            "destination",
            "b",
            "dst",
        ),
    )

    exact_replay = [
        dict(
            a
        ),
        dict(
            b
        ),
        dict(
            b
        ),
    ]

    exact_result = run_correlator_defensively(
        exact_replay,
        config,
    )

    new_id_replay_item = dict(
        b
    )

    new_id_replay_item[
        "alert_id"
    ] = str(
        uuid.uuid4()
    )

    new_id_replay = [
        dict(
            a
        ),
        dict(
            b
        ),
        new_id_replay_item,
    ]

    new_id_result = run_correlator_defensively(
        new_id_replay,
        config,
    )

    exact_removed = (
        exact_result[
            "duplicates_removed"
        ]
        ==
        1
    )

    new_id_removed = (
        new_id_result[
            "duplicates_removed"
        ]
        >
        0
    )

    return {
        "exact_alert_id_replay_removed":
            exact_removed,

        "new_alert_id_semantic_replay_removed":
            new_id_removed,

        "known_limitation":
            (
                "A semantically repeated alert carrying a fresh alert_id is "
                "not deduplicated by the current exact-id defense. This is "
                "documented for later replay/authentication hardening."
            ),

        "passed_for_current_contract":
            exact_removed,
    }
