from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, Optional

from .alert_schema import (
    PHASE14_POLICY_VERSION,
    PRIVACY_VERSION,
    SCHEMA_VERSION,
    PrivacyAwareAlert,
    new_alert_id,
    validate_alert,
)
from .pseudonymizer import HMACPseudonymizer


class PrivacyFilterError(ValueError):
    """Raised when a local alert cannot be converted into a safe shared alert."""


# Phase 15C is explicitly ALLOWLIST-ONLY.
# Only these fields may leave the local IDS after transformation.
SHARED_ALERT_ALLOWLIST = frozenset(
    {
        "alert_id",
        "client_id",
        "event_time",
        "classification",
        "severity",
        "confidence",
        "risk_score",
        "evidence_score",
        "risk_mode",
        "decision_source",
        "processing_path",
        "source_token",
        "destination_token",
        "protocol",
        "destination_port",
        "model_version",
        "policy_version",
        "privacy_version",
        "schema_version",
    }
)


def normalize_client_id(value: Any) -> str:
    """
    Convert the Phase 14 local client identifier to the Phase 15A
    privacy-safe logical client identifier.

    Examples:
        3          -> client_3
        "3"        -> client_3
        "client_3" -> client_3
    """
    if isinstance(value, bool):
        raise PrivacyFilterError("client_id cannot be boolean.")

    if isinstance(value, int):
        if value < 0:
            raise PrivacyFilterError("client_id must be non-negative.")
        return f"client_{value}"

    if isinstance(value, str):
        candidate = value.strip()

        if not candidate:
            raise PrivacyFilterError("client_id cannot be empty.")

        if candidate.startswith("client_"):
            return candidate

        if candidate.isdigit():
            return f"client_{candidate}"

    raise PrivacyFilterError(
        "client_id must be an integer, numeric string, or 'client_<id>'."
    )


def _phase14_event_time(local_alert: Mapping[str, Any]) -> str:
    """
    Phase 14G2 uses timestamp_utc. Phase 15 uses event_time.
    """
    if "event_time" in local_alert:
        value = local_alert["event_time"]
    elif "timestamp_utc" in local_alert:
        value = local_alert["timestamp_utc"]
    else:
        raise PrivacyFilterError(
            "Local alert must contain timestamp_utc or event_time."
        )

    if value is None:
        raise PrivacyFilterError("Alert timestamp cannot be null.")

    return str(value)


def _required(local_alert: Mapping[str, Any], key: str) -> Any:
    if key not in local_alert:
        raise PrivacyFilterError(
            f"Local Phase 14 alert is missing required field: {key}"
        )
    return local_alert[key]


def filter_local_alert_for_sharing(
    local_alert: Mapping[str, Any],
    *,
    pseudonymizer: HMACPseudonymizer,
    source_ip: str,
    destination_ip: str,
    protocol: Optional[str] = None,
    destination_port: Optional[int] = None,
    model_version: str = "phase14_final",
    alert_id: Optional[str] = None,
) -> PrivacyAwareAlert:
    """
    Convert a potentially sensitive local Phase 14 alert into the
    minimal Phase 15 privacy-aware shared alert.

    SECURITY MODEL
    --------------
    1. The function never copies arbitrary source fields.
    2. It constructs a NEW object only from explicitly allowlisted data.
    3. Raw source/destination IPs are used only as local HMAC inputs.
    4. Raw IPs are never returned.
    5. Nested model details, payloads, feature vectors, sequences,
       reconstruction vectors and debug fields are silently discarded.
    6. The final output must pass the frozen Phase 15A schema.
    """

    if not isinstance(local_alert, Mapping):
        raise PrivacyFilterError(
            "local_alert must be a dictionary/mapping."
        )

    # Make a defensive copy only for mutation-safety checking;
    # output is constructed independently and never derived via dict filtering.
    _ = deepcopy(local_alert)

    source_token = pseudonymizer.source_ip(
        source_ip
    )

    destination_token = pseudonymizer.destination_ip(
        destination_ip
    )

    filtered: Dict[str, Any] = {
        "alert_id":
            alert_id
            if alert_id is not None
            else new_alert_id(),

        "client_id":
            normalize_client_id(
                _required(
                    local_alert,
                    "client_id",
                )
            ),

        "event_time":
            _phase14_event_time(
                local_alert
            ),

        "classification":
            _required(
                local_alert,
                "classification",
            ),

        "severity":
            _required(
                local_alert,
                "severity",
            ),

        "confidence":
            _required(
                local_alert,
                "confidence",
            ),

        "risk_score":
            _required(
                local_alert,
                "risk_score",
            ),

        "evidence_score":
            _required(
                local_alert,
                "evidence_score",
            ),

        "risk_mode":
            _required(
                local_alert,
                "risk_mode",
            ),

        "decision_source":
            _required(
                local_alert,
                "decision_source",
            ),

        "processing_path":
            _required(
                local_alert,
                "processing_path",
            ),

        "source_token":
            source_token,

        "destination_token":
            destination_token,

        "protocol":
            (
                None
                if protocol is None
                else str(protocol).upper()
            ),

        "destination_port":
            destination_port,

        "model_version":
            model_version,

        "policy_version":
            PHASE14_POLICY_VERSION,

        "privacy_version":
            PRIVACY_VERSION,

        "schema_version":
            SCHEMA_VERSION,
    }

    # Defensive assertion: code changes cannot accidentally add a new
    # outbound field without updating the explicit Phase 15C allowlist.
    unexpected = set(filtered.keys()) - SHARED_ALERT_ALLOWLIST

    if unexpected:
        raise PrivacyFilterError(
            "Privacy filter attempted to emit non-allowlisted field(s): "
            + ", ".join(sorted(unexpected))
        )

    # Phase 15A remains the authoritative semantic/schema validator.
    return validate_alert(
        filtered
    )
