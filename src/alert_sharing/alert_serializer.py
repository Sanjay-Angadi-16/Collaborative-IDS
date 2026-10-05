from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, Union

from .alert_schema import PrivacyAwareAlert, validate_alert


class AlertSerializationError(ValueError):
    """Raised when a privacy-aware alert cannot be safely serialized."""


def _reject_duplicate_keys(pairs):
    result = {}

    for key, value in pairs:
        if key in result:
            raise AlertSerializationError(
                f"Duplicate JSON key rejected: {key}"
            )

        result[key] = value

    return result


def serialize_alert(
    alert: Union[PrivacyAwareAlert, Mapping[str, Any]],
) -> bytes:
    """
    Serialize only a Phase15A-valid privacy-aware alert.

    Canonical transport representation:
    - UTF-8 JSON
    - sorted keys
    - compact separators
    - no NaN / Infinity
    """
    if isinstance(
        alert,
        PrivacyAwareAlert,
    ):
        validated = alert.validate()

    elif isinstance(
        alert,
        Mapping,
    ):
        validated = validate_alert(
            alert
        )

    else:
        raise AlertSerializationError(
            "alert must be PrivacyAwareAlert or mapping."
        )

    try:
        text = json.dumps(
            validated.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
            allow_nan=False,
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise AlertSerializationError(
            "Failed to serialize privacy-aware alert."
        ) from exc

    return text.encode(
        "utf-8"
    )


def deserialize_alert(
    payload: Union[
        bytes,
        bytearray,
        str,
    ],
) -> PrivacyAwareAlert:
    if isinstance(
        payload,
        (bytes, bytearray),
    ):
        try:
            text = bytes(
                payload
            ).decode(
                "utf-8"
            )
        except UnicodeDecodeError as exc:
            raise AlertSerializationError(
                "Serialized alert must be UTF-8."
            ) from exc

    elif isinstance(
        payload,
        str,
    ):
        text = payload

    else:
        raise AlertSerializationError(
            "payload must be bytes or string."
        )

    try:
        parsed = json.loads(
            text,
            object_pairs_hook=
                _reject_duplicate_keys,
        )
    except AlertSerializationError:
        raise
    except json.JSONDecodeError as exc:
        raise AlertSerializationError(
            "Serialized alert is not valid JSON."
        ) from exc

    if not isinstance(
        parsed,
        dict,
    ):
        raise AlertSerializationError(
            "Serialized alert must contain one JSON object."
        )

    return validate_alert(
        parsed
    )


def serialized_sha256(
    payload: Union[
        bytes,
        bytearray,
        str,
    ],
) -> str:
    if isinstance(
        payload,
        str,
    ):
        data = payload.encode(
            "utf-8"
        )
    elif isinstance(
        payload,
        (bytes, bytearray),
    ):
        data = bytes(
            payload
        )
    else:
        raise AlertSerializationError(
            "payload must be bytes or string."
        )

    return hashlib.sha256(
        data
    ).hexdigest()
