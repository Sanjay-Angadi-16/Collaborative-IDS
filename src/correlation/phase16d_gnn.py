from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import math
import random
from typing import Any, Dict, List, Mapping, Sequence, Tuple
import uuid

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .phase16b_validation import run_correlator_defensively
from .rule_temporal_correlator import CorrelationConfig


PHASE16D_VERSION = "phase16d_v1"

ATTACK_CLASSES = (
    "DoS Hulk",
    "DDoS",
    "PortScan",
    "DoS GoldenEye",
    "FTP-Patator",
)

CLIENTS = (
    "client_1",
    "client_2",
    "client_3",
    "client_4",
    "client_5",
)

SEVERITIES = (
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL",
)

PROTOCOLS = (
    "TCP",
    "UDP",
)

PATHS = (
    "FAST_ATTACK_PATH",
    "DEEP_ANALYSIS_PATH",
)

RISK_MODES = (
    "FAST_PATH_RF_CONFIDENCE",
    "FULL_FOUR_MODEL_FUSION",
)


def set_all_seeds(
    seed: int,
) -> None:
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    try:
        torch.use_deterministic_algorithms(
            True
        )
    except Exception:
        pass


def _iso(
    dt: datetime,
) -> str:
    return dt.astimezone(
        timezone.utc
    ).replace(
        microsecond=0
    ).isoformat()


def _stable_token(
    secret: bytes,
    namespace: str,
    value: str,
    prefix: str,
) -> str:
    message = (
        f"phase16d|{namespace}|{value}"
    ).encode(
        "utf-8"
    )

    digest = hmac.new(
        secret,
        message,
        hashlib.sha256,
    ).hexdigest()[
        :32
    ]

    return (
        f"{prefix}_{digest}"
    )


def _severity_from_score(
    score: float,
) -> str:
    if score >= 0.90:
        return "CRITICAL"

    if score >= 0.75:
        return "HIGH"

    if score >= 0.50:
        return "MEDIUM"

    return "LOW"


def make_alert(
    *,
    client_id: str,
    event_time: datetime,
    classification: str,
    source_token: str,
    destination_token: str,
    destination_port: int,
    rng: random.Random,
) -> Dict[str, Any]:
    """
    Create a synthetic privacy-safe alert that preserves the Phase14G2
    FAST/DEEP semantic distinction.

    These alerts are synthetic correlation-validation records, not IDS
    detection outputs and not locked-test evidence.
    """

    use_fast = (
        rng.random()
        <
        0.55
    )

    protocol = (
        "TCP"
        if rng.random()
        <
        0.85
        else
        "UDP"
    )

    if use_fast:
        confidence = rng.uniform(
            0.95,
            0.999,
        )

        evidence_score = confidence
        risk_score = None
        risk_mode = (
            "FAST_PATH_RF_CONFIDENCE"
        )
        processing_path = (
            "FAST_ATTACK_PATH"
        )
        severity = "HIGH"

    else:
        risk_score = rng.uniform(
            0.35,
            0.95,
        )

        evidence_score = risk_score
        confidence = rng.uniform(
            0.70,
            0.98,
        )

        risk_mode = (
            "FULL_FOUR_MODEL_FUSION"
        )
        processing_path = (
            "DEEP_ANALYSIS_PATH"
        )
        severity = _severity_from_score(
            risk_score
        )

    return {
        "alert_id":
            str(
                uuid.uuid4()
            ),

        "client_id":
            client_id,

        "event_time":
            _iso(
                event_time
            ),

        "classification":
            classification,

        "severity":
            severity,

        "confidence":
            float(
                confidence
            ),

        "risk_score":
            (
                None
                if risk_score
                is None
                else
                float(
                    risk_score
                )
            ),

        "evidence_score":
            float(
                evidence_score
            ),

        "risk_mode":
            risk_mode,

        "decision_source":
            (
                "Random Forest"
                if use_fast
                else
                "Full Fusion"
            ),

        "processing_path":
            processing_path,

        "source_token":
            source_token,

        "destination_token":
            destination_token,

        "protocol":
            protocol,

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


@dataclass(frozen=True)
class GraphScenario:
    graph_id: str
    label: int
    subtype: str
    alerts: List[Dict[str, Any]]


def _new_base_time(
    rng: random.Random,
    index: int,
) -> datetime:
    return datetime(
        2026,
        1,
        1,
        tzinfo=timezone.utc,
    ) + timedelta(
        minutes=
            index
            *
            5
            +
            rng.randint(
                0,
                3,
            )
    )


def _positive_shared_source(
    *,
    rng: random.Random,
    secret: bytes,
    index: int,
    window_seconds: int,
) -> GraphScenario:
    base = _new_base_time(
        rng,
        index,
    )

    n = rng.randint(
        3,
        6,
    )

    clients = rng.sample(
        list(
            CLIENTS
        ),
        k=min(
            n,
            len(
                CLIENTS
            ),
        ),
    )

    while len(
        clients
    ) < n:
        clients.append(
            rng.choice(
                list(
                    CLIENTS
                )
            )
        )

    attack_classes = rng.sample(
        list(
            ATTACK_CLASSES
        ),
        k=min(
            n,
            len(
                ATTACK_CLASSES
            ),
        ),
    )

    while len(
        attack_classes
    ) < n:
        attack_classes.append(
            rng.choice(
                list(
                    ATTACK_CLASSES
                )
            )
        )

    common_source = _stable_token(
        secret,
        "source",
        f"pos-source-{index}",
        "src",
    )

    alerts: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    latest_allowed = max(
        1,
        window_seconds
        -
        5,
    )

    offsets = sorted(
        rng.sample(
            range(
                0,
                latest_allowed
                +
                1,
            ),
            k=n,
        )
    )

    for i in range(
        n
    ):
        alerts.append(
            make_alert(
                client_id=
                    clients[
                        i
                    ],

                event_time=
                    base
                    +
                    timedelta(
                        seconds=
                            offsets[
                                i
                            ]
                    ),

                classification=
                    attack_classes[
                        i
                    ],

                source_token=
                    common_source,

                destination_token=
                    _stable_token(
                        secret,
                        "destination",
                        f"pos-source-dst-{index}-{i}",
                        "dst",
                    ),

                destination_port=
                    rng.choice(
                        [
                            80,
                            443,
                            22,
                            21,
                            8080,
                        ]
                    ),

                rng=
                    rng,
            )
        )

    return GraphScenario(
        graph_id=
            f"pos_src_{index}",

        label=
            1,

        subtype=
            "SHARED_SOURCE",

        alerts=
            alerts,
    )


def _positive_shared_destination(
    *,
    rng: random.Random,
    secret: bytes,
    index: int,
    window_seconds: int,
) -> GraphScenario:
    base = _new_base_time(
        rng,
        index,
    )

    n = rng.randint(
        4,
        7,
    )

    shared_destination = _stable_token(
        secret,
        "destination",
        f"pos-destination-{index}",
        "dst",
    )

    destination_port = rng.choice(
        [
            80,
            443,
            22,
            21,
            3389,
        ]
    )

    clients = [
        rng.choice(
            list(
                CLIENTS
            )
        )
        for _
        in range(
            n
        )
    ]

    # Guarantee at least two clients.
    clients[
        0
    ] = "client_1"

    clients[
        1
    ] = "client_2"

    latest_allowed = max(
        1,
        window_seconds
        -
        5,
    )

    offsets = sorted(
        rng.sample(
            range(
                0,
                latest_allowed
                +
                1,
            ),
            k=n,
        )
    )

    alerts = []

    for i in range(
        n
    ):
        alerts.append(
            make_alert(
                client_id=
                    clients[
                        i
                    ],

                event_time=
                    base
                    +
                    timedelta(
                        seconds=
                            offsets[
                                i
                            ]
                    ),

                classification=
                    rng.choice(
                        list(
                            ATTACK_CLASSES
                        )
                    ),

                source_token=
                    _stable_token(
                        secret,
                        "source",
                        f"pos-destination-src-{index}-{i}",
                        "src",
                    ),

                destination_token=
                    shared_destination,

                destination_port=
                    destination_port,

                rng=
                    rng,
            )
        )

    return GraphScenario(
        graph_id=
            f"pos_dst_{index}",

        label=
            1,

        subtype=
            "SHARED_DESTINATION",

        alerts=
            alerts,
    )


def _positive_mixed(
    *,
    rng: random.Random,
    secret: bytes,
    index: int,
    window_seconds: int,
) -> GraphScenario:
    """
    Mixed positive graph containing one cross-client shared source plus
    additional unrelated decoy alerts.
    """
    scenario = _positive_shared_source(
        rng=rng,
        secret=secret,
        index=index,
        window_seconds=window_seconds,
    )

    base = datetime.fromisoformat(
        scenario.alerts[
            0
        ][
            "event_time"
        ]
    )

    alerts = [
        dict(
            alert
        )
        for alert
        in scenario.alerts
    ]

    for decoy_index in range(
        rng.randint(
            1,
            3,
        )
    ):
        alerts.append(
            make_alert(
                client_id=
                    rng.choice(
                        list(
                            CLIENTS
                        )
                    ),

                event_time=
                    base
                    +
                    timedelta(
                        seconds=
                            rng.randint(
                                0,
                                max(
                                    1,
                                    window_seconds
                                    -
                                    1,
                                ),
                            )
                    ),

                classification=
                    rng.choice(
                        list(
                            ATTACK_CLASSES
                        )
                    ),

                source_token=
                    _stable_token(
                        secret,
                        "source",
                        f"mixed-decoy-src-{index}-{decoy_index}",
                        "src",
                    ),

                destination_token=
                    _stable_token(
                        secret,
                        "destination",
                        f"mixed-decoy-dst-{index}-{decoy_index}",
                        "dst",
                    ),

                destination_port=
                    rng.choice(
                        [
                            80,
                            443,
                            22,
                            21,
                        ]
                    ),

                rng=
                    rng,
            )
        )

    rng.shuffle(
        alerts
    )

    return GraphScenario(
        graph_id=
            f"pos_mix_{index}",

        label=
            1,

        subtype=
            "MIXED_WITH_DECOYS",

        alerts=
            alerts,
    )


def _negative_independent(
    *,
    rng: random.Random,
    secret: bytes,
    index: int,
    window_seconds: int,
) -> GraphScenario:
    base = _new_base_time(
        rng,
        index,
    )

    n = rng.randint(
        3,
        7,
    )

    alerts = []

    for i in range(
        n
    ):
        alerts.append(
            make_alert(
                client_id=
                    rng.choice(
                        list(
                            CLIENTS
                        )
                    ),

                event_time=
                    base
                    +
                    timedelta(
                        seconds=
                            rng.randint(
                                0,
                                max(
                                    1,
                                    window_seconds
                                    -
                                    1,
                                ),
                            )
                    ),

                classification=
                    rng.choice(
                        list(
                            ATTACK_CLASSES
                        )
                    ),

                source_token=
                    _stable_token(
                        secret,
                        "source",
                        f"neg-independent-src-{index}-{i}",
                        "src",
                    ),

                destination_token=
                    _stable_token(
                        secret,
                        "destination",
                        f"neg-independent-dst-{index}-{i}",
                        "dst",
                    ),

                destination_port=
                    rng.choice(
                        [
                            80,
                            443,
                            22,
                            21,
                            8080,
                        ]
                    ),

                rng=
                    rng,
            )
        )

    return GraphScenario(
        graph_id=
            f"neg_ind_{index}",

        label=
            0,

        subtype=
            "INDEPENDENT",

        alerts=
            alerts,
    )


def _negative_same_source_single_client(
    *,
    rng: random.Random,
    secret: bytes,
    index: int,
    window_seconds: int,
) -> GraphScenario:
    base = _new_base_time(
        rng,
        index,
    )

    n = rng.randint(
        3,
        6,
    )

    client = rng.choice(
        list(
            CLIENTS
        )
    )

    common_source = _stable_token(
        secret,
        "source",
        f"neg-single-client-{index}",
        "src",
    )

    alerts = []

    for i in range(
        n
    ):
        alerts.append(
            make_alert(
                client_id=
                    client,

                event_time=
                    base
                    +
                    timedelta(
                        seconds=
                            rng.randint(
                                0,
                                max(
                                    1,
                                    window_seconds
                                    -
                                    1,
                                ),
                            )
                    ),

                classification=
                    rng.choice(
                        list(
                            ATTACK_CLASSES
                        )
                    ),

                source_token=
                    common_source,

                destination_token=
                    _stable_token(
                        secret,
                        "destination",
                        f"neg-single-client-dst-{index}-{i}",
                        "dst",
                    ),

                destination_port=
                    rng.choice(
                        [
                            80,
                            443,
                            22,
                        ]
                    ),

                rng=
                    rng,
            )
        )

    return GraphScenario(
        graph_id=
            f"neg_one_client_{index}",

        label=
            0,

        subtype=
            "SAME_SOURCE_SINGLE_CLIENT",

        alerts=
            alerts,
    )


def _negative_shared_source_outside_window(
    *,
    rng: random.Random,
    secret: bytes,
    index: int,
    window_seconds: int,
) -> GraphScenario:
    base = _new_base_time(
        rng,
        index,
    )

    common_source = _stable_token(
        secret,
        "source",
        f"neg-outside-{index}",
        "src",
    )

    separation = (
        window_seconds
        +
        rng.randint(
            5,
            90,
        )
    )

    alerts = [
        make_alert(
            client_id=
                "client_1",

            event_time=
                base,

            classification=
                "DoS Hulk",

            source_token=
                common_source,

            destination_token=
                _stable_token(
                    secret,
                    "destination",
                    f"neg-outside-a-{index}",
                    "dst",
                ),

            destination_port=
                443,

            rng=
                rng,
        ),
        make_alert(
            client_id=
                "client_2",

            event_time=
                base
                +
                timedelta(
                    seconds=
                        separation
                ),

            classification=
                "DDoS",

            source_token=
                common_source,

            destination_token=
                _stable_token(
                    secret,
                    "destination",
                    f"neg-outside-b-{index}",
                    "dst",
                ),

            destination_port=
                443,

            rng=
                rng,
        ),
    ]

    return GraphScenario(
        graph_id=
            f"neg_outside_{index}",

        label=
            0,

        subtype=
            "SHARED_SOURCE_OUTSIDE_WINDOW",

        alerts=
            alerts,
    )


def _negative_insufficient_destination_sources(
    *,
    rng: random.Random,
    secret: bytes,
    index: int,
    window_seconds: int,
) -> GraphScenario:
    base = _new_base_time(
        rng,
        index,
    )

    destination = _stable_token(
        secret,
        "destination",
        f"neg-insufficient-{index}",
        "dst",
    )

    port = rng.choice(
        [
            80,
            443,
            22,
        ]
    )

    alerts = [
        make_alert(
            client_id=
                "client_1",

            event_time=
                base,

            classification=
                "DoS Hulk",

            source_token=
                _stable_token(
                    secret,
                    "source",
                    f"neg-insufficient-src-a-{index}",
                    "src",
                ),

            destination_token=
                destination,

            destination_port=
                port,

            rng=
                rng,
        ),
        make_alert(
            client_id=
                "client_2",

            event_time=
                base
                +
                timedelta(
                    seconds=
                        min(
                            20,
                            max(
                                1,
                                window_seconds
                                -
                                1,
                            ),
                        )
                ),

            classification=
                "DDoS",

            source_token=
                _stable_token(
                    secret,
                    "source",
                    f"neg-insufficient-src-b-{index}",
                    "src",
                ),

            destination_token=
                destination,

            destination_port=
                port,

            rng=
                rng,
        ),
    ]

    return GraphScenario(
        graph_id=
            f"neg_insufficient_{index}",

        label=
            0,

        subtype=
            "INSUFFICIENT_DESTINATION_SOURCES",

        alerts=
            alerts,
    )


def generate_graph_scenarios(
    *,
    count: int,
    seed: int,
    window_seconds: int,
) -> List[GraphScenario]:
    if count < 2:
        raise ValueError(
            "count must be at least 2"
        )

    rng = random.Random(
        seed
    )

    secret = hashlib.sha256(
        f"phase16d-secret-{seed}".encode(
            "utf-8"
        )
    ).digest()

    positive_builders = (
        _positive_shared_source,
        _positive_shared_destination,
        _positive_mixed,
    )

    negative_builders = (
        _negative_independent,
        _negative_same_source_single_client,
        _negative_shared_source_outside_window,
        _negative_insufficient_destination_sources,
    )

    scenarios = []

    for index in range(
        count
    ):
        is_positive = (
            index
            %
            2
            ==
            0
        )

        if is_positive:
            builder = positive_builders[
                index
                %
                len(
                    positive_builders
                )
            ]

        else:
            builder = negative_builders[
                index
                %
                len(
                    negative_builders
                )
            ]

        scenario = builder(
            rng=rng,
            secret=secret,
            index=index,
            window_seconds=window_seconds,
        )

        scenarios.append(
            scenario
        )

    rng.shuffle(
        scenarios
    )

    return scenarios


def privacy_audit_scenarios(
    scenarios: Sequence[GraphScenario],
) -> Dict[str, Any]:
    forbidden_keys = {
        "source_ip",
        "destination_ip",
        "src_ip",
        "dst_ip",
        "payload",
        "packet_payload",
        "raw_packet",
        "flow_features",
        "cnn_sequence",
        "model_state_dict",
    }

    forbidden_found = []

    token_format_ok = True

    for scenario in scenarios:
        for alert in scenario.alerts:
            for key in alert.keys():
                if key.lower() in forbidden_keys:
                    forbidden_found.append(
                        f"{scenario.graph_id}.{key}"
                    )

            source_token = str(
                alert.get(
                    "source_token",
                    "",
                )
            )

            destination_token = str(
                alert.get(
                    "destination_token",
                    "",
                )
            )

            token_format_ok = (
                token_format_ok
                and
                source_token.startswith(
                    "src_"
                )
                and
                destination_token.startswith(
                    "dst_"
                )
            )

    return {
        "passed":
            not forbidden_found
            and
            token_format_ok,

        "forbidden_fields_found":
            forbidden_found,

        "token_format_ok":
            token_format_ok,

        "scenario_count":
            len(
                scenarios
            ),
    }


def _one_hot(
    value: str,
    vocabulary: Sequence[str],
) -> List[float]:
    return [
        1.0
        if value
        ==
        item
        else
        0.0
        for item
        in vocabulary
    ]


def node_features(
    alerts: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    if not alerts:
        raise ValueError(
            "Graph must contain at least one alert."
        )

    event_times = [
        datetime.fromisoformat(
            str(
                alert[
                    "event_time"
                ]
            )
        )
        for alert
        in alerts
    ]

    start = min(
        event_times
    )

    features = []

    for alert, event_time in zip(
        alerts,
        event_times,
    ):
        row: List[
            float
        ] = []

        row.extend(
            _one_hot(
                str(
                    alert[
                        "classification"
                    ]
                ),
                ATTACK_CLASSES,
            )
        )

        row.extend(
            _one_hot(
                str(
                    alert[
                        "client_id"
                    ]
                ),
                CLIENTS,
            )
        )

        row.extend(
            _one_hot(
                str(
                    alert[
                        "severity"
                    ]
                ),
                SEVERITIES,
            )
        )

        row.extend(
            _one_hot(
                str(
                    alert[
                        "protocol"
                    ]
                ),
                PROTOCOLS,
            )
        )

        row.extend(
            _one_hot(
                str(
                    alert[
                        "processing_path"
                    ]
                ),
                PATHS,
            )
        )

        row.extend(
            _one_hot(
                str(
                    alert[
                        "risk_mode"
                    ]
                ),
                RISK_MODES,
            )
        )

        row.append(
            min(
                max(
                    float(
                        alert.get(
                            "destination_port",
                            0,
                        )
                    )
                    /
                    65535.0,
                    0.0,
                ),
                1.0,
            )
        )

        row.append(
            float(
                alert.get(
                    "confidence",
                    0.0,
                )
            )
        )

        row.append(
            float(
                alert.get(
                    "evidence_score",
                    0.0,
                )
            )
        )

        risk_score = alert.get(
            "risk_score"
        )

        row.append(
            0.0
            if risk_score
            is None
            else
            float(
                risk_score
            )
        )

        elapsed = (
            event_time
            -
            start
        ).total_seconds()

        row.append(
            min(
                max(
                    elapsed
                    /
                    300.0,
                    0.0,
                ),
                1.0,
            )
        )

        features.append(
            row
        )

    return np.asarray(
        features,
        dtype=np.float32,
    )


def relation_adjacency(
    alerts: Sequence[Mapping[str, Any]],
    *,
    temporal_scale_seconds: float = 60.0,
) -> np.ndarray:
    n = len(
        alerts
    )

    adjacency = np.zeros(
        (
            n,
            n,
        ),
        dtype=np.float32,
    )

    event_times = [
        datetime.fromisoformat(
            str(
                alert[
                    "event_time"
                ]
            )
        )
        for alert
        in alerts
    ]

    for i in range(
        n
    ):
        adjacency[
            i,
            i
        ] = 1.0

        for j in range(
            i
            +
            1,
            n
        ):
            weight = 0.0

            if (
                alerts[
                    i
                ][
                    "source_token"
                ]
                ==
                alerts[
                    j
                ][
                    "source_token"
                ]
            ):
                weight += 1.0

            if (
                alerts[
                    i
                ][
                    "destination_token"
                ]
                ==
                alerts[
                    j
                ][
                    "destination_token"
                ]
                and
                int(
                    alerts[
                        i
                    ][
                        "destination_port"
                    ]
                )
                ==
                int(
                    alerts[
                        j
                    ][
                        "destination_port"
                    ]
                )
            ):
                weight += 1.0

            if (
                alerts[
                    i
                ][
                    "client_id"
                ]
                ==
                alerts[
                    j
                ][
                    "client_id"
                ]
            ):
                weight += 0.20

            delta = abs(
                (
                    event_times[
                        i
                    ]
                    -
                    event_times[
                        j
                    ]
                ).total_seconds()
            )

            temporal_weight = math.exp(
                -
                delta
                /
                max(
                    temporal_scale_seconds,
                    1.0,
                )
            )

            weight += (
                0.35
                *
                temporal_weight
            )

            if weight > 0.0:
                adjacency[
                    i,
                    j
                ] = weight

                adjacency[
                    j,
                    i
                ] = weight

    degree = adjacency.sum(
        axis=1
    )

    degree = np.maximum(
        degree,
        1e-8,
    )

    inverse_sqrt = 1.0 / np.sqrt(
        degree
    )

    normalized = (
        inverse_sqrt[
            :,
            None
        ]
        *
        adjacency
        *
        inverse_sqrt[
            None,
            :
        ]
    )

    return normalized.astype(
        np.float32
    )


@dataclass
class TensorGraph:
    graph_id: str
    subtype: str
    label: int
    x: torch.Tensor
    adjacency: torch.Tensor
    node_count: int
    alerts: List[Dict[str, Any]]


def tensorize_scenario(
    scenario: GraphScenario,
) -> TensorGraph:
    x_np = node_features(
        scenario.alerts
    )

    adjacency_np = relation_adjacency(
        scenario.alerts
    )

    return TensorGraph(
        graph_id=
            scenario.graph_id,

        subtype=
            scenario.subtype,

        label=
            int(
                scenario.label
            ),

        x=
            torch.from_numpy(
                x_np
            ),

        adjacency=
            torch.from_numpy(
                adjacency_np
            ),

        node_count=
            len(
                scenario.alerts
            ),

        alerts=
            scenario.alerts,
    )


class CorrelationGraphDataset(
    Dataset
):
    def __init__(
        self,
        scenarios: Sequence[GraphScenario],
    ) -> None:
        self.graphs = [
            tensorize_scenario(
                scenario
            )
            for scenario
            in scenarios
        ]

    def __len__(
        self,
    ) -> int:
        return len(
            self.graphs
        )

    def __getitem__(
        self,
        index: int,
    ) -> TensorGraph:
        return self.graphs[
            index
        ]


def collate_graphs(
    graphs: Sequence[TensorGraph],
) -> Dict[str, Any]:
    if not graphs:
        raise ValueError(
            "Empty graph batch."
        )

    batch_size = len(
        graphs
    )

    max_nodes = max(
        graph.node_count
        for graph
        in graphs
    )

    feature_dim = int(
        graphs[
            0
        ].x.shape[
            1
        ]
    )

    x = torch.zeros(
        (
            batch_size,
            max_nodes,
            feature_dim,
        ),
        dtype=torch.float32,
    )

    adjacency = torch.zeros(
        (
            batch_size,
            max_nodes,
            max_nodes,
        ),
        dtype=torch.float32,
    )

    mask = torch.zeros(
        (
            batch_size,
            max_nodes,
        ),
        dtype=torch.float32,
    )

    labels = torch.zeros(
        (
            batch_size,
        ),
        dtype=torch.long,
    )

    graph_ids = []
    subtypes = []
    alerts = []

    for batch_index, graph in enumerate(
        graphs
    ):
        n = graph.node_count

        x[
            batch_index,
            :n,
            :
        ] = graph.x

        adjacency[
            batch_index,
            :n,
            :n
        ] = graph.adjacency

        mask[
            batch_index,
            :n
        ] = 1.0

        labels[
            batch_index
        ] = graph.label

        graph_ids.append(
            graph.graph_id
        )

        subtypes.append(
            graph.subtype
        )

        alerts.append(
            graph.alerts
        )

    return {
        "x":
            x,

        "adjacency":
            adjacency,

        "mask":
            mask,

        "labels":
            labels,

        "graph_ids":
            graph_ids,

        "subtypes":
            subtypes,

        "alerts":
            alerts,
    }


class DenseGCNLayer(
    nn.Module
):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()

        self.linear = nn.Linear(
            input_dim,
            output_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        aggregated = torch.bmm(
            adjacency,
            x,
        )

        return self.linear(
            aggregated
        )


class AlertCorrelationGCN(
    nn.Module
):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()

        self.gcn1 = DenseGCNLayer(
            input_dim,
            hidden_dim,
        )

        self.gcn2 = DenseGCNLayer(
            hidden_dim,
            hidden_dim,
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_dim
                *
                2,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                hidden_dim,
                2,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.gcn1(
            x,
            adjacency,
        )

        h = F.relu(
            h
        )

        h = self.dropout(
            h
        )

        h = self.gcn2(
            h,
            adjacency,
        )

        h = F.relu(
            h
        )

        mask_expanded = mask.unsqueeze(
            -1
        )

        masked_h = (
            h
            *
            mask_expanded
        )

        denominator = mask_expanded.sum(
            dim=1
        ).clamp_min(
            1.0
        )

        mean_pool = masked_h.sum(
            dim=1
        ) / denominator

        negative_large = torch.full_like(
            h,
            -1e9,
        )

        max_input = torch.where(
            mask_expanded.bool(),
            h,
            negative_large,
        )

        max_pool = max_input.max(
            dim=1
        ).values

        graph_embedding = torch.cat(
            [
                mean_pool,
                max_pool,
            ],
            dim=1,
        )

        return self.classifier(
            graph_embedding
        )


def rule_baseline_prediction(
    alerts: Sequence[Mapping[str, Any]],
    config: CorrelationConfig,
) -> int:
    outcome = run_correlator_defensively(
        alerts,
        config,
    )

    return (
        1
        if outcome[
            "events"
        ]
        else
        0
    )
