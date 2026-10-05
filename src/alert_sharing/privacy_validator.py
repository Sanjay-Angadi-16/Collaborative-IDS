from __future__ import annotations

from dataclasses import asdict, dataclass
import ipaddress
import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from .alert_schema import (
    ALLOWED_FIELDS,
    AlertSchemaError,
    validate_alert,
)


class PrivacyLeakageError(ValueError):
    """Raised when an outbound alert contains privacy-sensitive material."""


# Exact names and common aliases that must never occur in the outbound object.
FORBIDDEN_FIELD_NAMES = frozenset(
    {
        "source_ip",
        "destination_ip",
        "src_ip",
        "dst_ip",
        "source_mac",
        "destination_mac",
        "src_mac",
        "dst_mac",
        "hostname",
        "username",
        "user_id",
        "payload",
        "packet_payload",
        "raw_packet",
        "raw_packets",
        "packet_bytes",
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


# IPv4 candidates are validated with ipaddress before reporting.
_IPV4_CANDIDATE_RE = re.compile(
    r"(?<![\w])(?:\d{1,3}\.){3}\d{1,3}(?![\w])"
)

# Common MAC formats: 00:11:22:33:44:55 or 00-11-22-33-44-55.
_MAC_RE = re.compile(
    r"(?i)(?<![0-9a-f])(?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2}(?![0-9a-f])"
)


@dataclass(frozen=True)
class PrivacyLeakageReport:
    passed: bool
    schema_valid: bool
    unknown_fields: List[str]
    forbidden_fields: List[str]
    raw_ip_literals: List[str]
    mac_literals: List[str]
    known_secret_literals_found: List[str]
    known_raw_identifiers_found: List[str]
    source_token_present: bool
    destination_token_present: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _walk(
    value: Any,
    *,
    path: str = "$",
):
    """
    Yield (path, key, scalar_value) recursively.
    key is None for list/scalar positions.
    """
    if isinstance(value, Mapping):
        for key, item in value.items():
            child_path = f"{path}.{key}"
            yield (
                child_path,
                str(key),
                item,
            )
            yield from _walk(
                item,
                path=child_path,
            )

    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child_path = f"{path}[{index}]"
            yield (
                child_path,
                None,
                item,
            )
            yield from _walk(
                item,
                path=child_path,
            )


def _extract_ipv4_literals(text: str) -> Set[str]:
    found: Set[str] = set()

    for match in _IPV4_CANDIDATE_RE.findall(text):
        try:
            parsed = ipaddress.ip_address(
                match
            )
        except ValueError:
            continue

        if parsed.version == 4:
            found.add(
                parsed.compressed
            )

    return found


def audit_shared_alert(
    payload: Mapping[str, Any],
    *,
    known_raw_identifiers: Optional[Sequence[str]] = None,
    known_secret_literals: Optional[Sequence[str]] = None,
    require_link_tokens: bool = True,
) -> PrivacyLeakageReport:
    if not isinstance(payload, Mapping):
        raise PrivacyLeakageError(
            "Shared alert must be a dictionary/mapping."
        )

    top_keys = set(
        str(key)
        for key
        in payload.keys()
    )

    unknown_fields = sorted(
        top_keys
        -
        set(
            ALLOWED_FIELDS
        )
    )

    forbidden_fields: Set[str] = set()

    scalar_strings: List[str] = []

    for path, key, value in _walk(
        payload
    ):
        if key is not None:
            normalized_key = key.strip().lower()

            if normalized_key in FORBIDDEN_FIELD_NAMES:
                forbidden_fields.add(
                    f"{path}"
                )

        if isinstance(
            value,
            str,
        ):
            scalar_strings.append(
                value
            )

    serialized = json.dumps(
        payload,
        sort_keys=True,
        default=str,
    )

    raw_ip_literals = sorted(
        _extract_ipv4_literals(
            serialized
        )
    )

    mac_literals = sorted(
        set(
            _MAC_RE.findall(
                serialized
            )
        )
    )

    known_raw_identifiers_found: List[str] = []

    for raw_identifier in (
        known_raw_identifiers
        or
        []
    ):
        if (
            raw_identifier
            and
            str(raw_identifier)
            in
            serialized
        ):
            known_raw_identifiers_found.append(
                str(raw_identifier)
            )

    known_secret_literals_found: List[str] = []

    for secret_literal in (
        known_secret_literals
        or
        []
    ):
        if (
            secret_literal
            and
            str(secret_literal)
            in
            serialized
        ):
            known_secret_literals_found.append(
                "<secret literal detected>"
            )

    schema_valid = True

    try:
        validate_alert(
            payload
        )
    except AlertSchemaError:
        schema_valid = False

    source_token_present = bool(
        payload.get(
            "source_token"
        )
    )

    destination_token_present = bool(
        payload.get(
            "destination_token"
        )
    )

    passed = (
        schema_valid
        and
        not unknown_fields
        and
        not forbidden_fields
        and
        not raw_ip_literals
        and
        not mac_literals
        and
        not known_raw_identifiers_found
        and
        not known_secret_literals_found
        and
        (
            not require_link_tokens
            or
            (
                source_token_present
                and
                destination_token_present
            )
        )
    )

    return PrivacyLeakageReport(
        passed=
            passed,

        schema_valid=
            schema_valid,

        unknown_fields=
            unknown_fields,

        forbidden_fields=
            sorted(
                forbidden_fields
            ),

        raw_ip_literals=
            raw_ip_literals,

        mac_literals=
            mac_literals,

        known_secret_literals_found=
            known_secret_literals_found,

        known_raw_identifiers_found=
            known_raw_identifiers_found,

        source_token_present=
            source_token_present,

        destination_token_present=
            destination_token_present,
    )


def validate_no_privacy_leakage(
    payload: Mapping[str, Any],
    *,
    known_raw_identifiers: Optional[Sequence[str]] = None,
    known_secret_literals: Optional[Sequence[str]] = None,
    require_link_tokens: bool = True,
) -> PrivacyLeakageReport:
    report = audit_shared_alert(
        payload,
        known_raw_identifiers=
            known_raw_identifiers,

        known_secret_literals=
            known_secret_literals,

        require_link_tokens=
            require_link_tokens,
    )

    if not report.passed:
        raise PrivacyLeakageError(
            "Privacy leakage audit failed: "
            +
            json.dumps(
                report.to_dict(),
                sort_keys=True,
            )
        )

    return report
