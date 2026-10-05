from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import random
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .phase16d_gnn import (
    ATTACK_CLASSES,
    CLIENTS,
    GraphScenario,
    generate_graph_scenarios,
    make_alert,
    rule_baseline_prediction,
)
from .rule_temporal_correlator import CorrelationConfig


PHASE16F_VERSION = "phase16f_v1"

CHAIN_A = (
    "PortScan",
    "FTP-Patator",
    "DoS GoldenEye",
    "DDoS",
)

CHAIN_B = (
    "PortScan",
    "DoS GoldenEye",
    "DoS Hulk",
)

CHAIN_C = (
    "FTP-Patator",
    "PortScan",
    "DDoS",
)


def _token(
    secret: bytes,
    namespace: str,
    value: str,
    prefix: str,
) -> str:
    msg = f"phase16f|{namespace}|{value}".encode(
        "utf-8"
    )

    digest = hmac.new(
        secret,
        msg,
        hashlib.sha256,
    ).hexdigest()[:32]

    return f"{prefix}_{digest}"


def _base_time(
    index: int,
    rng: random.Random,
) -> datetime:
    return datetime(
        2026,
        2,
        1,
        tzinfo=timezone.utc,
    ) + timedelta(
        minutes=index * 7 + rng.randint(0, 4)
    )


def _force_deep(
    alert: Dict[str, Any],
    *,
    risk_score: float,
) -> Dict[str, Any]:
    alert = dict(
        alert
    )

    alert[
        "processing_path"
    ] = "DEEP_ANALYSIS_PATH"

    alert[
        "risk_mode"
    ] = "FULL_FOUR_MODEL_FUSION"

    alert[
        "decision_source"
    ] = "Full Fusion"

    alert[
        "risk_score"
    ] = float(
        risk_score
    )

    alert[
        "evidence_score"
    ] = float(
        risk_score
    )

    if risk_score >= 0.90:
        severity = "CRITICAL"

    elif risk_score >= 0.75:
        severity = "HIGH"

    elif risk_score >= 0.50:
        severity = "MEDIUM"

    else:
        severity = "LOW"

    alert[
        "severity"
    ] = severity

    return alert


def _unique_alert(
    *,
    rng: random.Random,
    secret: bytes,
    scenario_key: str,
    index: int,
    client_id: str,
    event_time: datetime,
    classification: str,
    risk_score: float,
) -> Dict[str, Any]:
    alert = make_alert(
        client_id=client_id,
        event_time=event_time,
        classification=classification,
        source_token=_token(
            secret,
            "source",
            f"{scenario_key}-src-{index}",
            "src",
        ),
        destination_token=_token(
            secret,
            "destination",
            f"{scenario_key}-dst-{index}",
            "dst",
        ),
        destination_port=[
            443,
            22,
            21,
            8080,
            8443,
        ][
            index
            %
            5
        ],
        rng=rng,
    )

    return _force_deep(
        alert,
        risk_score=risk_score,
    )


def build_rule_gap_positive(
    *,
    rng: random.Random,
    secret: bytes,
    index: int,
    variant: str,
) -> GraphScenario:
    """
    Positive campaign labels are defined by a campaign grammar, NOT by R1/R2/R3.

    All source/destination pseudonyms are unique across nodes, so the frozen
    shared-source/shared-destination rules should remain silent.

    The GNN can only use privacy-safe node attributes, temporal order and weak
    temporal graph structure.
    """
    base = _base_time(
        index,
        rng,
    )

    if variant == "CHAIN_A":
        chain = CHAIN_A
        gaps = (
            0,
            10,
            23,
            38,
        )
        clients = (
            "client_1",
            "client_3",
            "client_4",
            "client_2",
        )

    elif variant == "CHAIN_B":
        chain = CHAIN_B
        gaps = (
            0,
            17,
            41,
        )
        clients = (
            "client_3",
            "client_5",
            "client_2",
        )

    elif variant == "CHAIN_C":
        chain = CHAIN_C
        gaps = (
            0,
            21,
            49,
        )
        clients = (
            "client_5",
            "client_3",
            "client_2",
        )

    else:
        raise ValueError(
            f"Unknown rule-gap positive variant: {variant}"
        )

    alerts: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    for position, (
        attack,
        gap,
        client,
    ) in enumerate(
        zip(
            chain,
            gaps,
            clients,
        )
    ):
        risk = min(
            0.56
            +
            position
            *
            0.11,
            0.93,
        )

        alerts.append(
            _unique_alert(
                rng=rng,
                secret=secret,
                scenario_key=f"gap-pos-{variant}-{index}",
                index=position,
                client_id=client,
                event_time=base + timedelta(
                    seconds=gap
                ),
                classification=attack,
                risk_score=risk,
            )
        )

    return GraphScenario(
        graph_id=f"gap_pos_{variant}_{index}",
        label=1,
        subtype=f"RULE_GAP_{variant}",
        alerts=alerts,
    )


def build_hard_negative(
    *,
    rng: random.Random,
    secret: bytes,
    index: int,
    variant: str,
) -> GraphScenario:
    """
    Hard negatives resemble the positive campaigns in size and timing but break
    a defining campaign property such as stage order, client progression or
    temporal compactness.
    """
    base = _base_time(
        100000 + index,
        rng,
    )

    if variant == "SHUFFLED_A":
        chain = list(
            CHAIN_A
        )

        chain = [
            chain[2],
            chain[0],
            chain[3],
            chain[1],
        ]

        gaps = (
            0,
            10,
            23,
            38,
        )

        clients = (
            "client_1",
            "client_3",
            "client_4",
            "client_2",
        )

    elif variant == "WRONG_CLIENT_ORDER":
        chain = CHAIN_A

        gaps = (
            0,
            10,
            23,
            38,
        )

        clients = (
            "client_4",
            "client_1",
            "client_2",
            "client_5",
        )

    elif variant == "TOO_SLOW":
        chain = CHAIN_A

        gaps = (
            0,
            95,
            210,
            360,
        )

        clients = (
            "client_1",
            "client_3",
            "client_4",
            "client_2",
        )

    elif variant == "RANDOM_BURST":
        chain = tuple(
            rng.choice(
                list(
                    ATTACK_CLASSES
                )
            )
            for _
            in range(
                4
            )
        )

        if chain == CHAIN_A:
            chain = (
                "DDoS",
                "PortScan",
                "DoS Hulk",
                "FTP-Patator",
            )

        gaps = (
            0,
            12,
            24,
            39,
        )

        clients = tuple(
            rng.sample(
                list(
                    CLIENTS
                ),
                k=4,
            )
        )

    else:
        raise ValueError(
            f"Unknown hard-negative variant: {variant}"
        )

    alerts: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    for position, (
        attack,
        gap,
        client,
    ) in enumerate(
        zip(
            chain,
            gaps,
            clients,
        )
    ):
        # Keep magnitude similar to positives so the label cannot be solved
        # simply by risk-score scale.
        risk = min(
            0.56
            +
            position
            *
            0.11,
            0.93,
        )

        alerts.append(
            _unique_alert(
                rng=rng,
                secret=secret,
                scenario_key=f"gap-neg-{variant}-{index}",
                index=position,
                client_id=client,
                event_time=base + timedelta(
                    seconds=gap
                ),
                classification=attack,
                risk_score=risk,
            )
        )

    return GraphScenario(
        graph_id=f"gap_neg_{variant}_{index}",
        label=0,
        subtype=f"HARD_NEGATIVE_{variant}",
        alerts=alerts,
    )


def _rule_positive_pool(
    *,
    count: int,
    seed: int,
    window_seconds: int,
) -> List[GraphScenario]:
    """
    Reuse the existing Phase16 synthetic generator only to provide a subset of
    ordinary rule-detectable positive campaigns. Rule-gap labels are generated
    independently above.
    """
    pool: List[
        GraphScenario
    ] = []

    attempt = 0

    while len(
        pool
    ) < count:
        batch = generate_graph_scenarios(
            count=max(
                100,
                count
                *
                4,
            ),
            seed=seed + attempt,
            window_seconds=window_seconds,
        )

        for scenario in batch:
            if scenario.label != 1:
                continue

            pool.append(
                GraphScenario(
                    graph_id=f"rule_pos_{len(pool)}_{scenario.graph_id}",
                    label=1,
                    subtype=f"RULE_POSITIVE_{scenario.subtype}",
                    alerts=[
                        dict(
                            alert
                        )
                        for alert
                        in scenario.alerts
                    ],
                )
            )

            if len(
                pool
            ) >= count:
                break

        attempt += 1

        if attempt > 100:
            raise RuntimeError(
                "Unable to create enough rule-positive scenarios."
            )

    return pool


def generate_rule_gap_benchmark(
    *,
    total_graphs: int,
    seed: int,
    window_seconds: int,
    rule_positive_fraction_of_positives: float = 0.35,
) -> List[GraphScenario]:
    if total_graphs < 20:
        raise ValueError(
            "total_graphs must be at least 20"
        )

    rng = random.Random(
        seed
    )

    secret = hashlib.sha256(
        f"phase16f-{seed}".encode(
            "utf-8"
        )
    ).digest()

    positive_count = total_graphs // 2
    negative_count = total_graphs - positive_count

    rule_positive_count = int(
        round(
            positive_count
            *
            rule_positive_fraction_of_positives
        )
    )

    gap_positive_count = (
        positive_count
        -
        rule_positive_count
    )

    scenarios: List[
        GraphScenario
    ] = []

    positive_variants = (
        "CHAIN_A",
        "CHAIN_B",
        "CHAIN_C",
    )

    for index in range(
        gap_positive_count
    ):
        scenarios.append(
            build_rule_gap_positive(
                rng=rng,
                secret=secret,
                index=index,
                variant=positive_variants[
                    index
                    %
                    len(
                        positive_variants
                    )
                ],
            )
        )

    scenarios.extend(
        _rule_positive_pool(
            count=rule_positive_count,
            seed=seed + 50000,
            window_seconds=window_seconds,
        )
    )

    negative_variants = (
        "SHUFFLED_A",
        "WRONG_CLIENT_ORDER",
        "TOO_SLOW",
        "RANDOM_BURST",
    )

    for index in range(
        negative_count
    ):
        scenarios.append(
            build_hard_negative(
                rng=rng,
                secret=secret,
                index=index,
                variant=negative_variants[
                    index
                    %
                    len(
                        negative_variants
                    )
                ],
            )
        )

    rng.shuffle(
        scenarios
    )

    return scenarios


def audit_rule_gap(
    scenarios: Sequence[GraphScenario],
    rule_config: CorrelationConfig,
) -> Dict[str, Any]:
    gap_positives = [
        scenario
        for scenario
        in scenarios
        if (
            scenario.label == 1
            and
            scenario.subtype.startswith(
                "RULE_GAP_"
            )
        )
    ]

    hard_negatives = [
        scenario
        for scenario
        in scenarios
        if scenario.label == 0
    ]

    gap_rule_predictions = [
        rule_baseline_prediction(
            scenario.alerts,
            rule_config,
        )
        for scenario
        in gap_positives
    ]

    negative_rule_predictions = [
        rule_baseline_prediction(
            scenario.alerts,
            rule_config,
        )
        for scenario
        in hard_negatives
    ]

    silent_gap_count = sum(
        1
        for value
        in gap_rule_predictions
        if value == 0
    )

    negative_silent_count = sum(
        1
        for value
        in negative_rule_predictions
        if value == 0
    )

    return {
        "rule_gap_positive_count": len(
            gap_positives
        ),
        "frozen_rule_silent_on_rule_gap_count": silent_gap_count,
        "frozen_rule_silent_on_rule_gap_rate": (
            silent_gap_count
            /
            max(
                len(
                    gap_positives
                ),
                1,
            )
        ),
        "hard_negative_count": len(
            hard_negatives
        ),
        "frozen_rule_silent_on_hard_negative_count": negative_silent_count,
        "frozen_rule_silent_on_hard_negative_rate": (
            negative_silent_count
            /
            max(
                len(
                    hard_negatives
                ),
                1,
            )
        ),
        "passed": (
            silent_gap_count
            ==
            len(
                gap_positives
            )
            and
            negative_silent_count
            ==
            len(
                hard_negatives
            )
        ),
    }


def subtype_recall(
    *,
    subtypes: Sequence[str],
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> Dict[str, Any]:
    output: Dict[
        str,
        Any,
    ] = {}

    unique = sorted(
        set(
            subtypes
        )
    )

    for subtype in unique:
        indices = [
            i
            for i, value
            in enumerate(
                subtypes
            )
            if value == subtype
        ]

        truth = [
            y_true[
                i
            ]
            for i
            in indices
        ]

        pred = [
            y_pred[
                i
            ]
            for i
            in indices
        ]

        positive_truth = sum(
            truth
        )

        if positive_truth > 0:
            tp = sum(
                1
                for t, p
                in zip(
                    truth,
                    pred,
                )
                if t == 1
                and p == 1
            )

            recall = (
                tp
                /
                positive_truth
            )

        else:
            recall = None

        accuracy = float(
            np.mean(
                np.asarray(
                    truth
                )
                ==
                np.asarray(
                    pred
                )
            )
        )

        output[
            subtype
        ] = {
            "count": len(
                indices
            ),
            "positive_count": int(
                positive_truth
            ),
            "recall": (
                None
                if recall is None
                else
                float(
                    recall
                )
            ),
            "accuracy": accuracy,
        }

    return output
