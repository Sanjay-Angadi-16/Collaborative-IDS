from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple


class Phase15FinalAuditError(RuntimeError):
    """Raised when Phase 15 cannot be frozen because required evidence is missing or failed."""


@dataclass(frozen=True)
class EvidenceItem:
    phase: str
    name: str
    path: Path
    required: bool = True


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise Phase15FinalAuditError(
            f"Missing required Phase15 evidence file: {path}"
        )

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            payload = json.load(
                file
            )
    except json.JSONDecodeError as exc:
        raise Phase15FinalAuditError(
            f"Invalid JSON evidence file: {path}"
        ) from exc

    if not isinstance(
        payload,
        dict,
    ):
        raise Phase15FinalAuditError(
            f"Evidence file must contain one JSON object: {path}"
        )

    return payload


def bool_from_any(
    payload: Dict[str, Any],
    keys: Iterable[str],
) -> Optional[bool]:
    for key in keys:
        if key in payload:
            value = payload[
                key
            ]

            if isinstance(
                value,
                bool,
            ):
                return value

            if isinstance(
                value,
                str,
            ):
                normalized = value.strip().lower()

                if normalized in {
                    "pass",
                    "passed",
                    "true",
                    "yes",
                    "ok",
                }:
                    return True

                if normalized in {
                    "fail",
                    "failed",
                    "false",
                    "no",
                }:
                    return False

    return None


def summarize_phase_status(
    *,
    phase: str,
    payload: Dict[str, Any],
) -> Tuple[
    bool,
    str,
]:
    """
    Normalize the heterogeneous audit files from Phase15A-G.
    """

    direct = bool_from_any(
        payload,
        (
            "all_tests_passed",
            "status",
            "passed",
        ),
    )

    if direct is not None:
        return (
            direct,
            "direct status flag",
        )

    # Phase15D integration audit stores successful component checks
    # rather than a single all_tests_passed field.
    if phase == "15D":
        required_true = [
            payload.get(
                "roundtrip_equal"
            )
            is True,

            payload.get(
                "local_phase14_alert_unchanged"
            )
            is True,

            payload.get(
                "raw_ip_values_persisted"
            )
            is False,
        ]

        leakage = payload.get(
            "privacy_leakage_audit",
            {},
        )

        post = payload.get(
            "post_serialization_leakage_audit",
            {},
        )

        required_true.extend(
            [
                leakage.get(
                    "passed"
                )
                is True,

                post.get(
                    "passed"
                )
                is True,
            ]
        )

        return (
            all(
                required_true
            ),
            "derived from Phase15D integration checks",
        )

    if phase == "15F":
        checks = [
            payload.get(
                "duplicate_rejection"
            )
            is True,

            payload.get(
                "per_client_count_check"
            )
            is True,

            payload.get(
                "per_classification_count_check"
            )
            is True,

            payload.get(
                "cross_client_source_token_correlation"
            )
            is True,

            payload.get(
                "canonical_readback_check"
            )
            is True,

            payload.get(
                "required_indexes_present"
            )
            is True,

            payload.get(
                "recent_query_check"
            )
            is True,

            payload.get(
                "privacy_at_rest_audit",
                {},
            ).get(
                "passed"
            )
            is True,
        ]

        return (
            all(
                checks
            ),
            "derived from Phase15F persistence checks",
        )

    if phase == "15G":
        checks = [
            payload.get(
                "privacy_at_rest_before_retention",
                {},
            ).get(
                "passed"
            )
            is True,

            payload.get(
                "post_retention_privacy_audit",
                {},
            ).get(
                "passed"
            )
            is True,

            payload.get(
                "retention_preview",
                {},
            ).get(
                "dry_run"
            )
            is True,

            payload.get(
                "retention_apply",
                {},
            ).get(
                "dry_run"
            )
            is False,
        ]

        return (
            all(
                checks
            ),
            "derived from Phase15G privacy/retention checks",
        )

    return (
        False,
        "no recognized success evidence",
    )


def extract_locked_test_flag(
    payload: Dict[str, Any],
) -> Optional[bool]:
    if "locked_test_used" in payload:
        value = payload[
            "locked_test_used"
        ]

        if isinstance(
            value,
            bool,
        ):
            return value

    return None
