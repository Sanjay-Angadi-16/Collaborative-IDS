from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import ipaddress
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import uuid


PHASE16A_VERSION = "phase16a_v1"

RULE_SHARED_SOURCE = "R1_CROSS_CLIENT_SHARED_SOURCE"
RULE_MULTI_CLASS_SOURCE = "R2_MULTI_CLASS_SHARED_SOURCE"
RULE_DESTINATION_MULTI_SOURCE = "R3_SHARED_DESTINATION_MULTI_SOURCE"


class CorrelationError(ValueError):
    """Raised when privacy-safe alert correlation cannot be performed safely."""


@dataclass(frozen=True)
class CorrelationConfig:
    window_seconds: int = 60
    min_cross_client_clients: int = 2
    min_multi_class_clients: int = 2
    min_multi_class_classes: int = 2
    min_destination_clients: int = 2
    min_destination_sources: int = 3

    def validate(self) -> "CorrelationConfig":
        if self.window_seconds <= 0:
            raise CorrelationError(
                "window_seconds must be greater than zero."
            )

        for name, value in (
            (
                "min_cross_client_clients",
                self.min_cross_client_clients,
            ),
            (
                "min_multi_class_clients",
                self.min_multi_class_clients,
            ),
            (
                "min_multi_class_classes",
                self.min_multi_class_classes,
            ),
            (
                "min_destination_clients",
                self.min_destination_clients,
            ),
            (
                "min_destination_sources",
                self.min_destination_sources,
            ),
        ):
            if int(value) < 2:
                raise CorrelationError(
                    f"{name} must be at least 2."
                )

        return self


@dataclass(frozen=True)
class CorrelationEvent:
    correlation_id: str
    phase_version: str
    rule_id: str
    correlation_priority: str

    event_start: str
    event_end: str
    window_seconds: int

    alert_count: int
    distinct_clients: int
    distinct_classifications: int
    distinct_sources: int
    distinct_destinations: int

    client_ids: List[str]
    classifications: List[str]
    source_tokens: List[str]
    destination_tokens: List[str]
    destination_ports: List[int]

    evidence_alert_ids: List[str]
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_REQUIRED_ALERT_FIELDS = frozenset(
    {
        "alert_id",
        "client_id",
        "event_time",
        "classification",
        "source_token",
        "destination_token",
        "destination_port",
        "processing_path",
        "risk_mode",
    }
)


def _parse_event_time(
    value: str,
) -> datetime:
    if not isinstance(
        value,
        str,
    ) or not value.strip():
        raise CorrelationError(
            "event_time must be a non-empty ISO-8601 string."
        )

    candidate = value.strip()

    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(
            candidate
        )

    except ValueError as exc:
        raise CorrelationError(
            f"Invalid event_time: {value!r}"
        ) from exc

    if parsed.tzinfo is None:
        raise CorrelationError(
            "event_time must include timezone information."
        )

    return parsed.astimezone(
        timezone.utc
    )


def _validate_alert_minimum(
    alert: Mapping[str, Any],
) -> Dict[str, Any]:
    if not isinstance(
        alert,
        Mapping,
    ):
        raise CorrelationError(
            "Each alert must be a mapping."
        )

    missing = sorted(
        _REQUIRED_ALERT_FIELDS
        -
        set(
            alert.keys()
        )
    )

    if missing:
        raise CorrelationError(
            "Privacy-safe alert is missing required correlation field(s): "
            +
            ", ".join(
                missing
            )
        )

    source_token = alert.get(
        "source_token"
    )

    destination_token = alert.get(
        "destination_token"
    )

    if not (
        isinstance(
            source_token,
            str,
        )
        and
        source_token.startswith(
            "src_"
        )
    ):
        raise CorrelationError(
            "Phase16A requires privacy-safe src_ pseudonyms."
        )

    if not (
        isinstance(
            destination_token,
            str,
        )
        and
        destination_token.startswith(
            "dst_"
        )
    ):
        raise CorrelationError(
            "Phase16A requires privacy-safe dst_ pseudonyms."
        )

    normalized = dict(
        alert
    )

    normalized[
        "_event_dt"
    ] = _parse_event_time(
        str(
            alert[
                "event_time"
            ]
        )
    )

    return normalized


def _priority_from_client_count(
    distinct_clients: int,
) -> str:
    """
    Rule-engine priority only.

    This is NOT Phase14 classification confidence and NOT
    Phase14G2 fused risk.
    """
    if distinct_clients >= 5:
        return "CRITICAL"

    if distinct_clients >= 3:
        return "HIGH"

    return "MEDIUM"


def _greedy_windows(
    alerts: Sequence[Dict[str, Any]],
    *,
    window_seconds: int,
) -> List[List[Dict[str, Any]]]:
    """
    Deterministic non-overlapping event-time windows anchored at the
    first alert in each window.

    This baseline intentionally avoids overlapping duplicate correlation
    events. Boundary sensitivity is a documented Phase16A limitation.
    """

    ordered = sorted(
        alerts,
        key=lambda item: (
            item[
                "_event_dt"
            ],
            str(
                item[
                    "alert_id"
                ]
            ),
        ),
    )

    windows: List[
        List[
            Dict[
                str,
                Any,
            ]
        ]
    ] = []

    index = 0

    while index < len(
        ordered
    ):
        start_time = ordered[
            index
        ][
            "_event_dt"
        ]

        end_index = index

        while (
            end_index
            +
            1
            <
            len(
                ordered
            )
        ):
            next_time = ordered[
                end_index
                +
                1
            ][
                "_event_dt"
            ]

            elapsed = (
                next_time
                -
                start_time
            ).total_seconds()

            if elapsed > window_seconds:
                break

            end_index += 1

        windows.append(
            ordered[
                index:
                end_index
                +
                1
            ]
        )

        index = (
            end_index
            +
            1
        )

    return windows


def _event_from_window(
    *,
    rule_id: str,
    window: Sequence[Dict[str, Any]],
    window_seconds: int,
    explanation: str,
) -> CorrelationEvent:
    if not window:
        raise CorrelationError(
            "Cannot create correlation event from an empty window."
        )

    ordered = sorted(
        window,
        key=lambda item: item[
            "_event_dt"
        ],
    )

    client_ids = sorted(
        {
            str(
                item[
                    "client_id"
                ]
            )
            for item
            in ordered
        }
    )

    classifications = sorted(
        {
            str(
                item[
                    "classification"
                ]
            )
            for item
            in ordered
        }
    )

    source_tokens = sorted(
        {
            str(
                item[
                    "source_token"
                ]
            )
            for item
            in ordered
        }
    )

    destination_tokens = sorted(
        {
            str(
                item[
                    "destination_token"
                ]
            )
            for item
            in ordered
        }
    )

    destination_ports = sorted(
        {
            int(
                item[
                    "destination_port"
                ]
            )
            for item
            in ordered
            if item.get(
                "destination_port"
            )
            is not None
        }
    )

    evidence_alert_ids = [
        str(
            item[
                "alert_id"
            ]
        )
        for item
        in ordered
    ]

    return CorrelationEvent(
        correlation_id=
            str(
                uuid.uuid4()
            ),

        phase_version=
            PHASE16A_VERSION,

        rule_id=
            rule_id,

        correlation_priority=
            _priority_from_client_count(
                len(
                    client_ids
                )
            ),

        event_start=
            ordered[
                0
            ][
                "_event_dt"
            ].isoformat(),

        event_end=
            ordered[
                -1
            ][
                "_event_dt"
            ].isoformat(),

        window_seconds=
            int(
                window_seconds
            ),

        alert_count=
            len(
                ordered
            ),

        distinct_clients=
            len(
                client_ids
            ),

        distinct_classifications=
            len(
                classifications
            ),

        distinct_sources=
            len(
                source_tokens
            ),

        distinct_destinations=
            len(
                destination_tokens
            ),

        client_ids=
            client_ids,

        classifications=
            classifications,

        source_tokens=
            source_tokens,

        destination_tokens=
            destination_tokens,

        destination_ports=
            destination_ports,

        evidence_alert_ids=
            evidence_alert_ids,

        explanation=
            explanation,
    )


class RuleTemporalCorrelator:
    """
    Phase16A deterministic privacy-safe rule / temporal correlator.

    It operates only on Phase15 privacy-safe metadata and HMAC tokens.
    It does not consume raw packets, IPs, feature vectors, CNN sequences,
    model parameters, or locked-test samples.
    """

    def __init__(
        self,
        config: CorrelationConfig | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else CorrelationConfig()
        ).validate()

    def correlate(
        self,
        alerts: Sequence[Mapping[str, Any]],
    ) -> List[CorrelationEvent]:
        normalized = [
            _validate_alert_minimum(
                alert
            )
            for alert
            in alerts
        ]

        events: List[
            CorrelationEvent
        ] = []

        events.extend(
            self._rule_cross_client_shared_source(
                normalized
            )
        )

        events.extend(
            self._rule_multi_class_shared_source(
                normalized
            )
        )

        events.extend(
            self._rule_shared_destination_multi_source(
                normalized
            )
        )

        return sorted(
            events,
            key=lambda event: (
                event.event_start,
                event.rule_id,
                event.correlation_id,
            ),
        )

    def _rule_cross_client_shared_source(
        self,
        alerts: Sequence[Dict[str, Any]],
    ) -> List[CorrelationEvent]:
        grouped: Dict[
            str,
            List[
                Dict[
                    str,
                    Any,
                ]
            ],
        ] = {}

        for alert in alerts:
            grouped.setdefault(
                str(
                    alert[
                        "source_token"
                    ]
                ),
                [],
            ).append(
                alert
            )

        events = []

        for source_token, source_alerts in grouped.items():
            for window in _greedy_windows(
                source_alerts,
                window_seconds=
                    self.config.window_seconds,
            ):
                clients = {
                    str(
                        item[
                            "client_id"
                        ]
                    )
                    for item
                    in window
                }

                if (
                    len(
                        clients
                    )
                    <
                    self.config.min_cross_client_clients
                ):
                    continue

                events.append(
                    _event_from_window(
                        rule_id=
                            RULE_SHARED_SOURCE,

                        window=
                            window,

                        window_seconds=
                            self.config.window_seconds,

                        explanation=(
                            "The same privacy-safe source pseudonym "
                            f"appeared across {len(clients)} clients "
                            f"within {self.config.window_seconds} seconds."
                        ),
                    )
                )

        return events

    def _rule_multi_class_shared_source(
        self,
        alerts: Sequence[Dict[str, Any]],
    ) -> List[CorrelationEvent]:
        grouped: Dict[
            str,
            List[
                Dict[
                    str,
                    Any,
                ]
            ],
        ] = {}

        for alert in alerts:
            grouped.setdefault(
                str(
                    alert[
                        "source_token"
                    ]
                ),
                [],
            ).append(
                alert
            )

        events = []

        for source_token, source_alerts in grouped.items():
            for window in _greedy_windows(
                source_alerts,
                window_seconds=
                    self.config.window_seconds,
            ):
                clients = {
                    str(
                        item[
                            "client_id"
                        ]
                    )
                    for item
                    in window
                }

                classifications = {
                    str(
                        item[
                            "classification"
                        ]
                    )
                    for item
                    in window
                }

                if (
                    len(
                        clients
                    )
                    <
                    self.config.min_multi_class_clients
                    or
                    len(
                        classifications
                    )
                    <
                    self.config.min_multi_class_classes
                ):
                    continue

                events.append(
                    _event_from_window(
                        rule_id=
                            RULE_MULTI_CLASS_SOURCE,

                        window=
                            window,

                        window_seconds=
                            self.config.window_seconds,

                        explanation=(
                            "One privacy-safe source pseudonym was "
                            f"associated with {len(classifications)} "
                            f"attack classifications across "
                            f"{len(clients)} clients within "
                            f"{self.config.window_seconds} seconds."
                        ),
                    )
                )

        return events

    def _rule_shared_destination_multi_source(
        self,
        alerts: Sequence[Dict[str, Any]],
    ) -> List[CorrelationEvent]:
        grouped: Dict[
            Tuple[
                str,
                int,
            ],
            List[
                Dict[
                    str,
                    Any,
                ]
            ],
        ] = {}

        for alert in alerts:
            destination_port = alert.get(
                "destination_port"
            )

            if destination_port is None:
                continue

            key = (
                str(
                    alert[
                        "destination_token"
                    ]
                ),
                int(
                    destination_port
                ),
            )

            grouped.setdefault(
                key,
                [],
            ).append(
                alert
            )

        events = []

        for key, destination_alerts in grouped.items():
            for window in _greedy_windows(
                destination_alerts,
                window_seconds=
                    self.config.window_seconds,
            ):
                clients = {
                    str(
                        item[
                            "client_id"
                        ]
                    )
                    for item
                    in window
                }

                sources = {
                    str(
                        item[
                            "source_token"
                        ]
                    )
                    for item
                    in window
                }

                if (
                    len(
                        clients
                    )
                    <
                    self.config.min_destination_clients
                    or
                    len(
                        sources
                    )
                    <
                    self.config.min_destination_sources
                ):
                    continue

                events.append(
                    _event_from_window(
                        rule_id=
                            RULE_DESTINATION_MULTI_SOURCE,

                        window=
                            window,

                        window_seconds=
                            self.config.window_seconds,

                        explanation=(
                            "One privacy-safe destination/port received "
                            f"alerts from {len(sources)} pseudonymous "
                            f"sources across {len(clients)} clients "
                            f"within {self.config.window_seconds} seconds."
                        ),
                    )
                )

        return events


_IPV4_CANDIDATE_RE = re.compile(
    r"(?<![\w])(?:\d{1,3}\.){3}\d{1,3}(?![\w])"
)

_FORBIDDEN_FIELD_NAMES = frozenset(
    {
        "source_ip",
        "destination_ip",
        "src_ip",
        "dst_ip",
        "payload",
        "packet_payload",
        "raw_packet",
        "raw_packets",
        "flow_features",
        "raw_flow_features",
        "feature_vector",
        "features",
        "cnn_sequence",
        "sequence",
        "raw_sequence",
        "model_state_dict",
        "model_weights",
        "reconstruction_vector",
        "embedding",
        "embeddings",
    }
)


def privacy_audit_correlation_events(
    events: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Verify Phase16A correlation output remains privacy-safe.
    """

    forbidden_fields: List[
        str
    ] = []

    raw_ipv4_literals: List[
        str
    ] = []

    serialized = str(
        list(
            events
        )
    )

    for event_index, event in enumerate(
        events
    ):
        for key in event.keys():
            if str(
                key
            ).strip().lower() in _FORBIDDEN_FIELD_NAMES:
                forbidden_fields.append(
                    f"event[{event_index}].{key}"
                )

    for candidate in _IPV4_CANDIDATE_RE.findall(
        serialized
    ):
        try:
            parsed = ipaddress.ip_address(
                candidate
            )
        except ValueError:
            continue

        if parsed.version == 4:
            raw_ipv4_literals.append(
                parsed.compressed
            )

    source_token_ok = all(
        all(
            str(
                token
            ).startswith(
                "src_"
            )
            for token
            in event.get(
                "source_tokens",
                [],
            )
        )
        for event
        in events
    )

    destination_token_ok = all(
        all(
            str(
                token
            ).startswith(
                "dst_"
            )
            for token
            in event.get(
                "destination_tokens",
                [],
            )
        )
        for event
        in events
    )

    passed = (
        not forbidden_fields
        and
        not raw_ipv4_literals
        and
        source_token_ok
        and
        destination_token_ok
    )

    return {
        "passed":
            passed,

        "event_count":
            len(
                events
            ),

        "forbidden_fields":
            forbidden_fields,

        "raw_ipv4_literals":
            sorted(
                set(
                    raw_ipv4_literals
                )
            ),

        "source_token_format_ok":
            source_token_ok,

        "destination_token_format_ok":
            destination_token_ok,
    }
