from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import math
import re
import uuid
from typing import Any, Dict, Mapping, Optional


SCHEMA_VERSION = "phase15a_v1"
PRIVACY_VERSION = "phase15_v1"
PHASE14_POLICY_VERSION = "phase14g2_v1"


class AlertSchemaError(ValueError):
    """Raised when an alert violates the Phase 15A privacy-aware schema."""


class Severity(str, Enum):
    NORMAL = "NORMAL"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ProcessingPath(str, Enum):
    FAST_ATTACK_PATH = "FAST_ATTACK_PATH"
    DEEP_ANALYSIS = "DEEP_ANALYSIS"


class RiskMode(str, Enum):
    FAST_PATH_RF_CONFIDENCE = "FAST_PATH_RF_CONFIDENCE"
    FULL_FOUR_MODEL_FUSION = "FULL_FOUR_MODEL_FUSION"


ALLOWED_FIELDS = frozenset(
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


# Explicitly listed only to provide a clearer privacy error.
# Unknown fields are rejected anyway, so the schema behaves as an allowlist.
PROHIBITED_FIELDS = frozenset(
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


_TOKEN_RE = re.compile(r"^(src|dst|client)_[A-Za-z0-9_-]{8,128}$")
_CLIENT_RE = re.compile(r"^client_[A-Za-z0-9_-]{1,64}$")
_PROTOCOL_RE = re.compile(r"^[A-Z0-9_-]{1,16}$")


def new_alert_id() -> str:
    return str(uuid.uuid4())


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_finite_probability(name: str, value: Optional[float], allow_none: bool = False) -> Optional[float]:
    if value is None:
        if allow_none:
            return None
        raise AlertSchemaError(f"{name} is required.")

    if isinstance(value, bool):
        raise AlertSchemaError(f"{name} must be numeric, not boolean.")

    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise AlertSchemaError(f"{name} must be numeric.") from exc

    if not math.isfinite(numeric):
        raise AlertSchemaError(f"{name} must be finite.")

    if not (0.0 <= numeric <= 1.0):
        raise AlertSchemaError(f"{name} must be between 0 and 1.")

    return numeric


def _parse_utc_time(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AlertSchemaError("event_time must be a non-empty ISO-8601 timestamp.")

    candidate = value.strip()

    if candidate.endswith("Z"):
        candidate_for_parse = candidate[:-1] + "+00:00"
    else:
        candidate_for_parse = candidate

    try:
        parsed = datetime.fromisoformat(candidate_for_parse)
    except ValueError as exc:
        raise AlertSchemaError("event_time must be valid ISO-8601.") from exc

    if parsed.tzinfo is None:
        raise AlertSchemaError("event_time must include a timezone.")

    offset = parsed.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise AlertSchemaError("event_time must be UTC.")

    return parsed.astimezone(timezone.utc).isoformat()


def severity_from_full_fusion_risk(risk_score: float) -> Severity:
    if risk_score < 0.20:
        return Severity.NORMAL
    if risk_score < 0.40:
        return Severity.LOW
    if risk_score < 0.60:
        return Severity.MEDIUM
    if risk_score < 0.80:
        return Severity.HIGH
    return Severity.CRITICAL


@dataclass(frozen=True)
class PrivacyAwareAlert:
    alert_id: str
    client_id: str
    event_time: str

    classification: str
    severity: str

    confidence: float
    risk_score: Optional[float]
    evidence_score: float
    risk_mode: str

    decision_source: str
    processing_path: str

    source_token: Optional[str] = None
    destination_token: Optional[str] = None

    protocol: Optional[str] = None
    destination_port: Optional[int] = None

    model_version: str = "phase14_final"
    policy_version: str = PHASE14_POLICY_VERSION
    privacy_version: str = PRIVACY_VERSION
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        reject_unknown: bool = True,
    ) -> "PrivacyAwareAlert":
        if not isinstance(payload, Mapping):
            raise AlertSchemaError("Alert payload must be a mapping/dictionary.")

        keys = set(payload.keys())

        privacy_violations = sorted(keys.intersection(PROHIBITED_FIELDS))
        if privacy_violations:
            raise AlertSchemaError(
                "Privacy violation: prohibited raw/sensitive field(s): "
                + ", ".join(privacy_violations)
            )

        unknown = sorted(keys - ALLOWED_FIELDS)
        if reject_unknown and unknown:
            raise AlertSchemaError(
                "Schema is allowlist-only. Unknown field(s) rejected: "
                + ", ".join(unknown)
            )

        required = {
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
            "model_version",
            "policy_version",
            "privacy_version",
        }

        missing = sorted(required - keys)
        if missing:
            raise AlertSchemaError(
                "Missing required field(s): " + ", ".join(missing)
            )

        alert = cls(
            alert_id=str(payload["alert_id"]).strip(),
            client_id=str(payload["client_id"]).strip(),
            event_time=str(payload["event_time"]).strip(),
            classification=str(payload["classification"]).strip(),
            severity=str(payload["severity"]).strip().upper(),
            confidence=payload["confidence"],
            risk_score=payload["risk_score"],
            evidence_score=payload["evidence_score"],
            risk_mode=str(payload["risk_mode"]).strip(),
            decision_source=str(payload["decision_source"]).strip(),
            processing_path=str(payload["processing_path"]).strip(),
            source_token=(
                None
                if payload.get("source_token") is None
                else str(payload.get("source_token")).strip()
            ),
            destination_token=(
                None
                if payload.get("destination_token") is None
                else str(payload.get("destination_token")).strip()
            ),
            protocol=(
                None
                if payload.get("protocol") is None
                else str(payload.get("protocol")).strip().upper()
            ),
            destination_port=payload.get("destination_port"),
            model_version=str(payload["model_version"]).strip(),
            policy_version=str(payload["policy_version"]).strip(),
            privacy_version=str(payload["privacy_version"]).strip(),
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)).strip(),
        )

        return alert.validate()

    def validate(self) -> "PrivacyAwareAlert":
        # -----------------------------
        # Basic identifiers
        # -----------------------------
        try:
            uuid.UUID(self.alert_id)
        except (ValueError, AttributeError) as exc:
            raise AlertSchemaError("alert_id must be a valid UUID.") from exc

        if not _CLIENT_RE.fullmatch(self.client_id):
            raise AlertSchemaError(
                "client_id must use privacy-safe form such as 'client_3'."
            )

        event_time = _parse_utc_time(self.event_time)

        if not self.classification:
            raise AlertSchemaError("classification cannot be empty.")

        if len(self.classification) > 64:
            raise AlertSchemaError("classification is too long.")

        try:
            severity = Severity(self.severity)
        except ValueError as exc:
            raise AlertSchemaError(
                f"severity must be one of {[item.value for item in Severity]}."
            ) from exc

        try:
            path = ProcessingPath(self.processing_path)
        except ValueError as exc:
            raise AlertSchemaError(
                f"processing_path must be one of {[item.value for item in ProcessingPath]}."
            ) from exc

        try:
            risk_mode = RiskMode(self.risk_mode)
        except ValueError as exc:
            raise AlertSchemaError(
                f"risk_mode must be one of {[item.value for item in RiskMode]}."
            ) from exc

        confidence = _ensure_finite_probability("confidence", self.confidence)
        evidence_score = _ensure_finite_probability("evidence_score", self.evidence_score)
        risk_score = _ensure_finite_probability(
            "risk_score",
            self.risk_score,
            allow_none=True,
        )

        if not self.decision_source:
            raise AlertSchemaError("decision_source cannot be empty.")

        if not self.model_version:
            raise AlertSchemaError("model_version cannot be empty.")

        if self.policy_version != PHASE14_POLICY_VERSION:
            raise AlertSchemaError(
                f"policy_version must remain '{PHASE14_POLICY_VERSION}' for the frozen Phase 14 policy."
            )

        if self.privacy_version != PRIVACY_VERSION:
            raise AlertSchemaError(
                f"privacy_version must be '{PRIVACY_VERSION}'."
            )

        if self.schema_version != SCHEMA_VERSION:
            raise AlertSchemaError(
                f"schema_version must be '{SCHEMA_VERSION}'."
            )

        # -----------------------------
        # Optional privacy-safe tokens
        # Phase 15B will produce these.
        # -----------------------------
        if self.source_token is not None:
            if not _TOKEN_RE.fullmatch(self.source_token):
                raise AlertSchemaError(
                    "source_token must be a pseudonymous token such as 'src_<token>'."
                )
            if not self.source_token.startswith("src_"):
                raise AlertSchemaError("source_token must start with 'src_'.")

        if self.destination_token is not None:
            if not _TOKEN_RE.fullmatch(self.destination_token):
                raise AlertSchemaError(
                    "destination_token must be a pseudonymous token such as 'dst_<token>'."
                )
            if not self.destination_token.startswith("dst_"):
                raise AlertSchemaError("destination_token must start with 'dst_'.")

        # -----------------------------
        # Optional network metadata
        # -----------------------------
        if self.protocol is not None:
            if not _PROTOCOL_RE.fullmatch(self.protocol):
                raise AlertSchemaError(
                    "protocol must contain only uppercase letters, numbers, '_' or '-'."
                )

        if self.destination_port is not None:
            if isinstance(self.destination_port, bool):
                raise AlertSchemaError("destination_port must be an integer.")
            try:
                port = int(self.destination_port)
            except (TypeError, ValueError) as exc:
                raise AlertSchemaError("destination_port must be an integer.") from exc

            if not (0 <= port <= 65535):
                raise AlertSchemaError(
                    "destination_port must be between 0 and 65535."
                )
        else:
            port = None

        # =====================================================
        # Frozen Phase 14G2 semantic contract
        # =====================================================

        if path is ProcessingPath.FAST_ATTACK_PATH:
            if risk_mode is not RiskMode.FAST_PATH_RF_CONFIDENCE:
                raise AlertSchemaError(
                    "FAST_ATTACK_PATH requires risk_mode=FAST_PATH_RF_CONFIDENCE."
                )

            if risk_score is not None:
                raise AlertSchemaError(
                    "FAST_ATTACK_PATH requires risk_score=None. "
                    "RF confidence is evidence, not ensemble risk."
                )

            if severity is not Severity.HIGH:
                raise AlertSchemaError(
                    "FAST_ATTACK_PATH severity is frozen to HIGH."
                )

            if self.decision_source != "Random Forest":
                raise AlertSchemaError(
                    "FAST_ATTACK_PATH decision_source must be 'Random Forest'."
                )

            if abs(evidence_score - confidence) > 1e-6:
                raise AlertSchemaError(
                    "FAST_ATTACK_PATH requires evidence_score == confidence."
                )

        elif path is ProcessingPath.DEEP_ANALYSIS:
            if risk_mode is not RiskMode.FULL_FOUR_MODEL_FUSION:
                raise AlertSchemaError(
                    "DEEP_ANALYSIS requires risk_mode=FULL_FOUR_MODEL_FUSION."
                )

            if risk_score is None:
                raise AlertSchemaError(
                    "DEEP_ANALYSIS requires a four-model fused risk_score."
                )

            expected_severity = severity_from_full_fusion_risk(risk_score)

            if severity is not expected_severity:
                raise AlertSchemaError(
                    "DEEP_ANALYSIS severity does not match the frozen "
                    f"risk bands. risk_score={risk_score:.6f} "
                    f"requires severity={expected_severity.value}."
                )

            if abs(evidence_score - risk_score) > 1e-6:
                raise AlertSchemaError(
                    "DEEP_ANALYSIS requires evidence_score == risk_score."
                )

        # Normalize immutable dataclass values by recreating the object.
        return PrivacyAwareAlert(
            alert_id=self.alert_id,
            client_id=self.client_id,
            event_time=event_time,
            classification=self.classification,
            severity=severity.value,
            confidence=float(confidence),
            risk_score=(None if risk_score is None else float(risk_score)),
            evidence_score=float(evidence_score),
            risk_mode=risk_mode.value,
            decision_source=self.decision_source,
            processing_path=path.value,
            source_token=self.source_token,
            destination_token=self.destination_token,
            protocol=self.protocol,
            destination_port=port,
            model_version=self.model_version,
            policy_version=self.policy_version,
            privacy_version=self.privacy_version,
            schema_version=self.schema_version,
        )


def validate_alert(payload: Mapping[str, Any]) -> PrivacyAwareAlert:
    """Convenience entry point for the Phase 15A schema."""
    return PrivacyAwareAlert.from_mapping(payload)
