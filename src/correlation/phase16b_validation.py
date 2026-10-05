from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import math
import random
import uuid
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .rule_temporal_correlator import (
    RULE_DESTINATION_MULTI_SOURCE,
    RULE_MULTI_CLASS_SOURCE,
    RULE_SHARED_SOURCE,
    CorrelationConfig,
    RuleTemporalCorrelator,
)


PHASE16B_VERSION = "phase16b_v1"


@dataclass(frozen=True)
class LabeledScenario:
    scenario_id: str
    description: str
    alerts: List[Dict[str, Any]]
    expected_correlated: bool
    expected_rules: Tuple[str, ...]
    exceptional_case: str = "NORMAL"

    def to_manifest(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "description": self.description,
            "expected_correlated": self.expected_correlated,
            "expected_rules": list(self.expected_rules),
            "exceptional_case": self.exceptional_case,
            "alert_count": len(self.alerts),
        }


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _hmac_token(
    secret: bytes,
    namespace: str,
    value: str,
    prefix: str,
) -> str:
    message = (
        "phase16b|"
        + namespace
        + "|"
        + value
    ).encode("utf-8")

    digest = hmac.new(
        secret,
        message,
        hashlib.sha256,
    ).hexdigest()[:32]

    return f"{prefix}_{digest}"


def _base_alert(
    *,
    client_id: int,
    event_time: datetime,
    classification: str,
    source_token: str,
    destination_token: str,
    destination_port: int = 443,
    alert_id: Optional[str] = None,
    arrival_time: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Synthetic privacy-safe Phase16B validation alert.

    It intentionally contains only privacy-safe identifiers and the frozen
    Phase15/Phase14G2-compatible alert semantics needed for correlation.
    """
    payload: Dict[str, Any] = {
        "alert_id": alert_id or str(uuid.uuid4()),
        "client_id": f"client_{client_id}",
        "event_time": _iso(event_time),
        "classification": classification,
        "severity": "HIGH",
        "confidence": 0.98,
        "risk_score": None,
        "evidence_score": 0.98,
        "risk_mode": "FAST_PATH_RF_CONFIDENCE",
        "decision_source": "Random Forest",
        "processing_path": "FAST_ATTACK_PATH",
        "source_token": source_token,
        "destination_token": destination_token,
        "protocol": "TCP",
        "destination_port": int(destination_port),
        "model_version": "phase14_final",
        "policy_version": "phase14g2_v1",
        "privacy_version": "phase15_v1",
        "schema_version": "phase15a_v1",
    }

    # arrival_time is Phase16B validation metadata only.
    # The correlator itself MUST use event_time.
    if arrival_time is not None:
        payload["arrival_time"] = _iso(arrival_time)

    return payload


def deduplicate_by_alert_id(
    alerts: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Deduplicate exact alert_id replays before correlation.

    This does not replace Phase15 collector-level duplicate rejection. It is a
    second defensive boundary for Phase16 validation and replay experiments.
    """
    seen: Set[str] = set()
    output: List[Dict[str, Any]] = []
    removed = 0

    for alert in alerts:
        alert_id = str(alert.get("alert_id", "")).strip()

        if not alert_id:
            # Let the underlying correlator reject malformed alerts.
            output.append(dict(alert))
            continue

        if alert_id in seen:
            removed += 1
            continue

        seen.add(alert_id)
        output.append(dict(alert))

    return output, removed


def strip_validation_metadata(
    alerts: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Remove Phase16B-only fields before passing alerts into the frozen correlator.
    """
    clean = []

    for alert in alerts:
        item = dict(alert)
        item.pop("arrival_time", None)
        item.pop("ground_truth_campaign", None)
        clean.append(item)

    return clean


def run_correlator_defensively(
    alerts: Sequence[Mapping[str, Any]],
    config: CorrelationConfig,
) -> Dict[str, Any]:
    deduped, duplicates_removed = deduplicate_by_alert_id(alerts)
    clean = strip_validation_metadata(deduped)

    correlator = RuleTemporalCorrelator(config)
    events = correlator.correlate(clean)
    payloads = [event.to_dict() for event in events]

    return {
        "events": payloads,
        "duplicates_removed": duplicates_removed,
        "input_alerts": len(alerts),
        "deduplicated_alerts": len(clean),
    }


def _classification_metrics(
    y_true: Sequence[bool],
    y_pred: Sequence[bool],
) -> Dict[str, Any]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if (not t) and p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if (not t) and (not p))
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and (not p))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    false_correlation_rate = fp / max(tp + fp, 1)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "false_positive_rate": false_positive_rate,
        "false_correlation_rate": false_correlation_rate,
    }


def evaluate_scenarios(
    scenarios: Sequence[LabeledScenario],
    config: CorrelationConfig,
) -> Dict[str, Any]:
    y_true: List[bool] = []
    y_pred: List[bool] = []
    scenario_results: List[Dict[str, Any]] = []

    all_rules = (
        RULE_SHARED_SOURCE,
        RULE_MULTI_CLASS_SOURCE,
        RULE_DESTINATION_MULTI_SOURCE,
    )

    rule_truth: Dict[str, List[bool]] = {rule: [] for rule in all_rules}
    rule_pred: Dict[str, List[bool]] = {rule: [] for rule in all_rules}

    for scenario in scenarios:
        outcome = run_correlator_defensively(
            scenario.alerts,
            config,
        )

        events = outcome["events"]
        predicted_correlated = len(events) > 0
        predicted_rules = sorted({event["rule_id"] for event in events})
        expected_rules = sorted(set(scenario.expected_rules))

        y_true.append(bool(scenario.expected_correlated))
        y_pred.append(bool(predicted_correlated))

        for rule in all_rules:
            rule_truth[rule].append(rule in expected_rules)
            rule_pred[rule].append(rule in predicted_rules)

        scenario_results.append(
            {
                "scenario_id": scenario.scenario_id,
                "description": scenario.description,
                "exceptional_case": scenario.exceptional_case,
                "expected_correlated": scenario.expected_correlated,
                "predicted_correlated": predicted_correlated,
                "expected_rules": expected_rules,
                "predicted_rules": predicted_rules,
                "event_count": len(events),
                "duplicates_removed": outcome["duplicates_removed"],
                "passed": (
                    predicted_correlated == scenario.expected_correlated
                    and set(predicted_rules) == set(expected_rules)
                ),
            }
        )

    overall = _classification_metrics(y_true, y_pred)

    per_rule = {
        rule: _classification_metrics(
            rule_truth[rule],
            rule_pred[rule],
        )
        for rule in all_rules
    }

    return {
        "overall": overall,
        "per_rule": per_rule,
        "scenarios": scenario_results,
    }


def build_controlled_validation_scenarios(
    *,
    seed: int = 42,
) -> List[LabeledScenario]:
    """
    Create labeled privacy-safe scenarios with known ground truth.

    Each scenario is evaluated independently, preventing accidental
    cross-scenario correlations.
    """
    random.seed(seed)

    secret = hashlib.sha256(
        f"phase16b-validation-{seed}".encode("utf-8")
    ).digest()

    base = datetime(
        2026,
        9,
        14,
        10,
        0,
        0,
        tzinfo=timezone.utc,
    )

    def src(name: str) -> str:
        return _hmac_token(secret, "source", name, "src")

    def dst(name: str) -> str:
        return _hmac_token(secret, "destination", name, "dst")

    scenarios: List[LabeledScenario] = []

    # P1: Same source across 3 clients, 3 classes -> R1 + R2
    alerts = [
        _base_alert(
            client_id=1,
            event_time=base + timedelta(seconds=0),
            classification="DoS Hulk",
            source_token=src("p1-attacker"),
            destination_token=dst("p1-target-1"),
        ),
        _base_alert(
            client_id=2,
            event_time=base + timedelta(seconds=12),
            classification="DDoS",
            source_token=src("p1-attacker"),
            destination_token=dst("p1-target-2"),
        ),
        _base_alert(
            client_id=3,
            event_time=base + timedelta(seconds=22),
            classification="PortScan",
            source_token=src("p1-attacker"),
            destination_token=dst("p1-target-3"),
        ),
    ]
    scenarios.append(
        LabeledScenario(
            scenario_id="P1_SHARED_SOURCE_MULTI_CLASS",
            description="One pseudonymous source appears across three clients and three attack classes.",
            alerts=alerts,
            expected_correlated=True,
            expected_rules=(RULE_SHARED_SOURCE, RULE_MULTI_CLASS_SOURCE),
        )
    )

    # P2: Multiple sources hit the same destination/port -> R3 only
    alerts = [
        _base_alert(
            client_id=i,
            event_time=base + timedelta(seconds=i * 8),
            classification=attack,
            source_token=src(f"p2-source-{i}"),
            destination_token=dst("p2-shared-target"),
            destination_port=443,
        )
        for i, attack in [
            (1, "DoS Hulk"),
            (2, "DDoS"),
            (3, "PortScan"),
            (4, "DoS GoldenEye"),
        ]
    ]
    scenarios.append(
        LabeledScenario(
            scenario_id="P2_SHARED_DESTINATION_MULTI_SOURCE",
            description="Four distinct pseudonymous sources across four clients target one destination/port.",
            alerts=alerts,
            expected_correlated=True,
            expected_rules=(RULE_DESTINATION_MULTI_SOURCE,),
        )
    )

    # P3: Same source across two clients, same class -> R1 only
    alerts = [
        _base_alert(
            client_id=1,
            event_time=base + timedelta(seconds=2),
            classification="DoS Hulk",
            source_token=src("p3-shared"),
            destination_token=dst("p3-target-a"),
        ),
        _base_alert(
            client_id=2,
            event_time=base + timedelta(seconds=18),
            classification="DoS Hulk",
            source_token=src("p3-shared"),
            destination_token=dst("p3-target-b"),
        ),
    ]
    scenarios.append(
        LabeledScenario(
            scenario_id="P3_SHARED_SOURCE_SAME_CLASS",
            description="One source appears across two clients but only one attack class.",
            alerts=alerts,
            expected_correlated=True,
            expected_rules=(RULE_SHARED_SOURCE,),
        )
    )

    # P4: Same source across four clients/classes -> R1 + R2
    alerts = [
        _base_alert(
            client_id=i,
            event_time=base + timedelta(seconds=(i - 1) * 10),
            classification=attack,
            source_token=src("p4-coordinated"),
            destination_token=dst(f"p4-target-{i}"),
        )
        for i, attack in [
            (1, "DoS Hulk"),
            (2, "DDoS"),
            (3, "PortScan"),
            (4, "DoS GoldenEye"),
        ]
    ]
    scenarios.append(
        LabeledScenario(
            scenario_id="P4_FOUR_CLIENT_MULTI_CLASS",
            description="One source appears across four clients and four classes within 30 seconds.",
            alerts=alerts,
            expected_correlated=True,
            expected_rules=(RULE_SHARED_SOURCE, RULE_MULTI_CLASS_SOURCE),
        )
    )

    # N1: Same source but outside 60-second temporal window
    alerts = [
        _base_alert(
            client_id=1,
            event_time=base,
            classification="DoS Hulk",
            source_token=src("n1-source"),
            destination_token=dst("n1-a"),
        ),
        _base_alert(
            client_id=2,
            event_time=base + timedelta(seconds=95),
            classification="DDoS",
            source_token=src("n1-source"),
            destination_token=dst("n1-b"),
        ),
    ]
    scenarios.append(
        LabeledScenario(
            scenario_id="N1_SHARED_SOURCE_OUTSIDE_WINDOW",
            description="Same source appears on two clients but 95 seconds apart.",
            alerts=alerts,
            expected_correlated=False,
            expected_rules=(),
        )
    )

    # N2: Same destination but only two sources; baseline R3 requires 3
    alerts = [
        _base_alert(
            client_id=1,
            event_time=base + timedelta(seconds=1),
            classification="DoS Hulk",
            source_token=src("n2-a"),
            destination_token=dst("n2-target"),
        ),
        _base_alert(
            client_id=2,
            event_time=base + timedelta(seconds=16),
            classification="DDoS",
            source_token=src("n2-b"),
            destination_token=dst("n2-target"),
        ),
    ]
    scenarios.append(
        LabeledScenario(
            scenario_id="N2_INSUFFICIENT_SOURCES",
            description="Only two sources target the same destination/port.",
            alerts=alerts,
            expected_correlated=False,
            expected_rules=(),
        )
    )

    # N3: Independent alerts
    alerts = [
        _base_alert(
            client_id=i,
            event_time=base + timedelta(seconds=i * 6),
            classification=attack,
            source_token=src(f"n3-source-{i}"),
            destination_token=dst(f"n3-target-{i}"),
        )
        for i, attack in [
            (1, "DoS Hulk"),
            (2, "DDoS"),
            (3, "PortScan"),
        ]
    ]
    scenarios.append(
        LabeledScenario(
            scenario_id="N3_INDEPENDENT_ALERTS",
            description="Independent clients, sources and destinations should not correlate.",
            alerts=alerts,
            expected_correlated=False,
            expected_rules=(),
        )
    )

    # N4: Repeated source but only on one client
    alerts = [
        _base_alert(
            client_id=1,
            event_time=base + timedelta(seconds=i * 9),
            classification="DoS Hulk",
            source_token=src("n4-one-client"),
            destination_token=dst(f"n4-target-{i}"),
        )
        for i in range(3)
    ]
    scenarios.append(
        LabeledScenario(
            scenario_id="N4_SINGLE_CLIENT_REPETITION",
            description="Repeated source on one client must not become a cross-client incident.",
            alerts=alerts,
            expected_correlated=False,
            expected_rules=(),
        )
    )

    return scenarios


def build_exceptional_scenarios(
    *,
    seed: int = 42,
) -> List[LabeledScenario]:
    """
    Build perturbations from one positive four-client campaign.
    """
    secret = hashlib.sha256(
        f"phase16b-exception-{seed}".encode("utf-8")
    ).digest()

    base = datetime(
        2026,
        9,
        14,
        12,
        0,
        0,
        tzinfo=timezone.utc,
    )

    def src(name: str) -> str:
        return _hmac_token(secret, "source", name, "src")

    def dst(name: str) -> str:
        return _hmac_token(secret, "destination", name, "dst")

    canonical = [
        _base_alert(
            client_id=i,
            event_time=base + timedelta(seconds=(i - 1) * 10),
            classification=attack,
            source_token=src("coordinated"),
            destination_token=dst(f"target-{i}"),
        )
        for i, attack in [
            (1, "DoS Hulk"),
            (2, "DDoS"),
            (3, "PortScan"),
            (4, "DoS GoldenEye"),
        ]
    ]

    scenarios: List[LabeledScenario] = []

    scenarios.append(
        LabeledScenario(
            scenario_id="E0_NORMAL_REFERENCE",
            description="Unperturbed four-client coordinated campaign.",
            alerts=[dict(a) for a in canonical],
            expected_correlated=True,
            expected_rules=(RULE_SHARED_SOURCE, RULE_MULTI_CLASS_SOURCE),
            exceptional_case="NORMAL_REFERENCE",
        )
    )

    # Delayed delivery: arrival order differs, event_time stays correct.
    delayed = []
    arrival_delays = [2, 95, 8, 4]

    for alert, delay in zip(canonical, arrival_delays):
        item = dict(alert)
        event_dt = datetime.fromisoformat(item["event_time"])
        item["arrival_time"] = _iso(event_dt + timedelta(seconds=delay))
        delayed.append(item)

    delayed = sorted(
        delayed,
        key=lambda item: item["arrival_time"],
    )

    scenarios.append(
        LabeledScenario(
            scenario_id="E1_DELAYED_ARRIVAL",
            description="One client's alert arrives much later, but event_time remains correct.",
            alerts=delayed,
            expected_correlated=True,
            expected_rules=(RULE_SHARED_SOURCE, RULE_MULTI_CLASS_SOURCE),
            exceptional_case="DELAYED_ALERTS",
        )
    )

    # Out of order input.
    out_of_order = [
        dict(canonical[2]),
        dict(canonical[0]),
        dict(canonical[3]),
        dict(canonical[1]),
    ]

    scenarios.append(
        LabeledScenario(
            scenario_id="E2_OUT_OF_ORDER",
            description="Alerts are supplied in non-chronological order.",
            alerts=out_of_order,
            expected_correlated=True,
            expected_rules=(RULE_SHARED_SOURCE, RULE_MULTI_CLASS_SOURCE),
            exceptional_case="OUT_OF_ORDER",
        )
    )

    # Moderate clock skew: last client shifted +35 seconds.
    skewed = [dict(a) for a in canonical]
    last_dt = datetime.fromisoformat(skewed[-1]["event_time"])
    skewed[-1]["event_time"] = _iso(
        last_dt + timedelta(seconds=35)
    )

    scenarios.append(
        LabeledScenario(
            scenario_id="E3_CLOCK_SKEW_35S",
            description="Client 4 clock is 35 seconds ahead; campaign should still be detectable.",
            alerts=skewed,
            expected_correlated=True,
            expected_rules=(RULE_SHARED_SOURCE, RULE_MULTI_CLASS_SOURCE),
            exceptional_case="CLOCK_SKEW",
        )
    )

    # Missing one alert; three clients remain.
    missing = [
        dict(canonical[0]),
        dict(canonical[1]),
        dict(canonical[3]),
    ]

    scenarios.append(
        LabeledScenario(
            scenario_id="E4_MISSING_ONE_ALERT",
            description="One of four client alerts is missing; three-client evidence remains.",
            alerts=missing,
            expected_correlated=True,
            expected_rules=(RULE_SHARED_SOURCE, RULE_MULTI_CLASS_SOURCE),
            exceptional_case="MISSING_ALERT",
        )
    )

    # Exact duplicate replay; defensive dedup should remove one.
    duplicated = [dict(a) for a in canonical]
    duplicated.append(dict(canonical[1]))

    scenarios.append(
        LabeledScenario(
            scenario_id="E5_EXACT_DUPLICATE_REPLAY",
            description="Exact duplicate alert_id is replayed; deduplication must avoid evidence inflation.",
            alerts=duplicated,
            expected_correlated=True,
            expected_rules=(RULE_SHARED_SOURCE, RULE_MULTI_CLASS_SOURCE),
            exceptional_case="DUPLICATE_REPLAY",
        )
    )

    return scenarios


def sensitivity_grid(
    scenarios: Sequence[LabeledScenario],
    *,
    windows: Sequence[int],
    r1_clients: Sequence[int],
    r2_clients: Sequence[int],
    r2_classes: Sequence[int],
    r3_clients: Sequence[int],
    r3_sources: Sequence[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for window in windows:
        for a in r1_clients:
            for b in r2_clients:
                for c in r2_classes:
                    for d in r3_clients:
                        for e in r3_sources:
                            config = CorrelationConfig(
                                window_seconds=int(window),
                                min_cross_client_clients=int(a),
                                min_multi_class_clients=int(b),
                                min_multi_class_classes=int(c),
                                min_destination_clients=int(d),
                                min_destination_sources=int(e),
                            ).validate()

                            result = evaluate_scenarios(
                                scenarios,
                                config,
                            )

                            metrics = result["overall"]

                            rows.append(
                                {
                                    "window_seconds": window,
                                    "r1_min_clients": a,
                                    "r2_min_clients": b,
                                    "r2_min_classes": c,
                                    "r3_min_clients": d,
                                    "r3_min_sources": e,
                                    **metrics,
                                }
                            )

    rows.sort(
        key=lambda row: (
            -row["f1"],
            row["false_correlation_rate"],
            -row["precision"],
            -row["recall"],
            row["window_seconds"],
            row["r1_min_clients"],
            row["r2_min_clients"],
            row["r2_min_classes"],
            row["r3_min_clients"],
            row["r3_min_sources"],
        )
    )

    return rows
