from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


# ============================================================
# PROJECT ROOT / EXISTING PHASES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import phase14g_full_detection_integration as phase14g
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "\nCould not import scripts\\phase14g_full_detection_integration.py"
    ) from exc

try:
    import phase14g1_real_model_wiring as phase14g1
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "\nCould not import scripts\\phase14g1_real_model_wiring.py\n"
        "Keep Phase14G2 inside the scripts folder and retain the "
        "working Phase14G1 real-model-wiring script."
    ) from exc


# ============================================================
# PHASE
# ============================================================

PHASE_NAME = (
    "PHASE 14G2 — CONSISTENT RISK / SEVERITY POLICY"
)


# ============================================================
# FROZEN POLICY
# ============================================================

RF_FAST_ATTACK_THRESHOLD = 0.95

# IMPORTANT SEMANTIC FIX:
#
# FAST PATH:
#   RF confidence is CLASSIFIER CONFIDENCE.
#   It is NOT a four-model ensemble risk score.
#   Therefore risk_score is intentionally None.
#   Operational severity is HIGH because the gate already requires
#   a high-confidence known-attack prediction.
#
# DEEP PATH:
#   risk_score is the full weighted ensemble risk produced using
#   RF + AE + FedProx70 + CNN-BiLSTM.
#   Severity is derived from that ensemble risk:
#       < 0.20  NORMAL
#       < 0.40  LOW
#       < 0.60  MEDIUM
#       < 0.80  HIGH
#       >=0.80  CRITICAL
#
# CRITICAL is therefore reserved for the FULL-FUSION path.
#
# This prevents the previous error:
#
#   RF weight = 0.20
#   only RF available
#   renormalize by 0.20
#   => RF confidence incorrectly becomes "ensemble risk".
#
FAST_PATH_OPERATIONAL_SEVERITY = "HIGH"
CRITICAL_REQUIRES_FULL_FUSION = True

POLICY_VERSION = "phase14g2_v1"

RISK_WEIGHTS = {
    "rf": 0.20,
    "fedprox70": 0.30,
    "cnn_bilstm": 0.35,
    "autoencoder": 0.15,
}

SEVERITY_THRESHOLDS = {
    "normal_max": 0.20,
    "low_max": 0.40,
    "medium_max": 0.60,
    "high_max": 0.80,
}


# ============================================================
# OUTPUTS
# ============================================================

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "phase14"
    / "phase14g2"
)

ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "detection"
    / "phase14g2"
)

CORRECTED_ALERT_FILE = (
    RESULT_ROOT
    / "phase14g2_corrected_alert.json"
)

POLICY_AUDIT_FILE = (
    RESULT_ROOT
    / "phase14g2_policy_audit.json"
)

POLICY_CONFIG_FILE = (
    ARTIFACT_ROOT
    / "phase14g2_risk_severity_policy.json"
)


# ============================================================
# OUTPUT CONTRACT
# ============================================================

@dataclass
class Phase14G2Alert:
    timestamp_utc: str
    client_id: int

    classification: str
    confidence: float
    decision_source: str
    processing_path: str

    severity: str
    severity_basis: str

    # Full-fusion ensemble risk only.
    # None for the RF-only fast path.
    risk_score: Optional[float]

    # Always present:
    # - FAST path = RF confidence
    # - DEEP path = full-fusion risk
    evidence_score: float

    risk_mode: str
    risk_basis: str
    risk_score_comparable_to_full_fusion: bool

    rf: Optional[Dict[str, Any]]
    autoencoder: Optional[Dict[str, Any]]
    fedprox70: Optional[Dict[str, Any]]
    cnn_bilstm: Optional[Dict[str, Any]]


# ============================================================
# HELPERS
# ============================================================

def separator(width: int = 122) -> None:
    print()
    print("=" * width)


def json_safe(value):
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    return value


def save_json(
    path: Path,
    payload,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_safe(payload),
            file,
            indent=4,
        )


def is_non_benign(
    result,
) -> bool:
    return (
        result.predicted_class
        .strip()
        .upper()
        !=
        "BENIGN"
    )


# ============================================================
# CORRECTED PHASE14G2 DETECTION POLICY
# ============================================================

def detect_with_phase14g2_policy(
    *,
    client_id: int,
    current_flow_70: np.ndarray,
    sequence_20x70: np.ndarray,
    rf,
    ae,
    fedprox70,
    cnn,
) -> Phase14G2Alert:

    # --------------------------------------------------------
    # 1. RANDOM FOREST — FAST FIRST-LINE CLASSIFIER
    # --------------------------------------------------------

    rf_result = rf.predict(
        current_flow_70
    )

    # --------------------------------------------------------
    # 2. ADAPTIVE GATE
    # --------------------------------------------------------

    take_fast_path = (
        is_non_benign(
            rf_result
        )
        and
        rf_result.confidence
        >=
        RF_FAST_ATTACK_THRESHOLD
    )

    if take_fast_path:

        # ====================================================
        # FAST PATH POLICY
        # ====================================================
        #
        # Do NOT call phase14g.fuse_risk() here.
        #
        # There is only one model output. Renormalizing the RF
        # weight would make:
        #
        #     weighted RF contribution / RF weight
        #     = RF confidence
        #
        # and falsely label it as an ensemble risk score.
        #
        # Instead:
        #   - classification  = RF class
        #   - confidence      = RF confidence
        #   - risk_score      = None
        #   - evidence_score  = RF confidence
        #   - severity        = HIGH operational severity
        #
        # CRITICAL remains reserved for full-fusion evidence.
        # ====================================================

        return Phase14G2Alert(
            timestamp_utc=
                datetime.now(
                    timezone.utc
                ).isoformat(),

            client_id=
                int(
                    client_id
                ),

            classification=
                rf_result.predicted_class,

            confidence=
                float(
                    rf_result.confidence
                ),

            decision_source=
                "Random Forest",

            processing_path=
                "FAST_ATTACK_PATH",

            severity=
                FAST_PATH_OPERATIONAL_SEVERITY,

            severity_basis=
                (
                    "High-confidence known-attack RF gate. "
                    "Severity is an operational fast-path category; "
                    "it is not derived from ensemble risk."
                ),

            risk_score=
                None,

            evidence_score=
                float(
                    rf_result.confidence
                ),

            risk_mode=
                "FAST_PATH_RF_CONFIDENCE",

            risk_basis=
                (
                    "Single local Random Forest confidence. "
                    "No ensemble risk was computed because deeper "
                    "models were intentionally skipped."
                ),

            risk_score_comparable_to_full_fusion=
                False,

            rf=
                asdict(
                    rf_result
                ),

            autoencoder=
                None,

            fedprox70=
                None,

            cnn_bilstm=
                None,
        )

    # --------------------------------------------------------
    # 3. DEEP ANALYSIS
    # --------------------------------------------------------

    ae_details = ae.details(
        current_flow_70
    )

    ae_result = (
        phase14g.AutoencoderResult(
            anomaly_score=
                float(
                    ae_details[
                        "normalized_anomaly_score"
                    ]
                ),

            is_anomaly=
                bool(
                    ae_details[
                        "raw_is_anomaly"
                    ]
                ),
        )
    )

    fedprox_result = fedprox70.predict(
        current_flow_70
    )

    cnn_result = cnn.predict(
        sequence_20x70
    )

    # --------------------------------------------------------
    # 4. FULL FOUR-MODEL RISK FUSION
    # --------------------------------------------------------

    fused = phase14g.fuse_risk(
        rf_result=
            rf_result,

        fedprox_result=
            fedprox_result,

        cnn_result=
            cnn_result,

        ae_result=
            ae_result,
    )

    # --------------------------------------------------------
    # 5. DEEP-PATH ALERT
    # --------------------------------------------------------

    return Phase14G2Alert(
        timestamp_utc=
            datetime.now(
                timezone.utc
            ).isoformat(),

        client_id=
            int(
                client_id
            ),

        classification=
            fused.final_class,

        confidence=
            float(
                fused.confidence
            ),

        decision_source=
            fused.decision_source,

        processing_path=
            "DEEP_ANALYSIS",

        severity=
            fused.severity,

        severity_basis=
            (
                "Severity derived from full four-model weighted "
                "ensemble risk using the frozen Phase14G thresholds."
            ),

        risk_score=
            float(
                fused.risk_score
            ),

        evidence_score=
            float(
                fused.risk_score
            ),

        risk_mode=
            "FULL_FOUR_MODEL_FUSION",

        risk_basis=
            (
                "Weighted RF + Personalized AE + FedProx70 MLP + "
                "frozen Round-20 CNN-BiLSTM risk."
            ),

        risk_score_comparable_to_full_fusion=
            True,

        rf=
            asdict(
                rf_result
            ),

        autoencoder=
            {
                **asdict(
                    ae_result
                ),
                "raw_reconstruction_error":
                    float(
                        ae_details[
                            "raw_reconstruction_error"
                        ]
                    ),
                "raw_threshold":
                    float(
                        ae_details[
                            "raw_threshold"
                        ]
                    ),
            },

        fedprox70=
            asdict(
                fedprox_result
            ),

        cnn_bilstm=
            asdict(
                cnn_result
            ),
    )


# ============================================================
# VALIDATION / POLICY AUDIT
# ============================================================

def run_always_run_all_audit(
    *,
    current_flow_70,
    sequence_20x70,
    rf,
    ae,
    fedprox70,
    cnn,
):
    """
    Research-only audit.

    All four real models are deliberately run on the same input so
    that we can compare what the full ensemble WOULD have produced.

    This does not alter the adaptive operational decision.
    """

    return phase14g1.run_all_models_once(
        flow=
            current_flow_70,

        sequence=
            sequence_20x70,

        rf=
            rf,

        ae=
            ae,

        fedprox70=
            fedprox70,

        cnn=
            cnn,
    )


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=PHASE_NAME
    )

    parser.add_argument(
        "--client-id",
        type=int,
        default=3,
        choices=[
            1,
            2,
            3,
            4,
            5,
        ],
    )

    parser.add_argument(
        "--target-class",
        type=str,
        default="auto",
        help=(
            "'auto' selects the client's locally seen attack."
        ),
    )

    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--ae-model",
        type=str,
        default=None,
        help=(
            "Optional explicit personalized AE artifact path."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARTIFACT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Project schema / real artifacts
    # --------------------------------------------------------

    (
        label_mapping,
        inverse_mapping,
        active_features,
    ) = phase14g1.load_project_schema()

    import torch

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    (
        rf,
        ae,
        fedprox70,
        cnn,
    ) = phase14g1.build_real_models(
        client_id=
            args.client_id,

        label_mapping=
            label_mapping,

        inverse_mapping=
            inverse_mapping,

        device=
            device,

        ae_model_path=
            args.ae_model,
    )

    sample = (
        phase14g1.load_real_validation_sample(
            client_id=
                args.client_id,

            label_mapping=
                label_mapping,

            inverse_mapping=
                inverse_mapping,

            sample_index=
                args.sample_index,

            target_class=
                args.target_class,
        )
    )

    current_flow = sample[
        "current_flow"
    ]

    sequence = sample[
        "sequence"
    ]

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    separator()

    print(
        PHASE_NAME
    )

    separator()

    print(
        f"Policy version            : {POLICY_VERSION}"
    )

    print(
        f"Device                    : {device}"
    )

    print(
        f"Client                    : {args.client_id}"
    )

    print(
        f"True class                : "
        f"{sample['true_class']}"
    )

    print(
        f"Validation sequence index : "
        f"{sample['sample_index']}"
    )

    print(
        f"RF fast threshold         : "
        f"{RF_FAST_ATTACK_THRESHOLD}"
    )

    print(
        f"Fast-path severity        : "
        f"{FAST_PATH_OPERATIONAL_SEVERITY}"
    )

    print(
        "Critical policy           : full-fusion only"
    )

    print(
        "Locked test               : NO"
    )

    # --------------------------------------------------------
    # Real model artifact verification
    # --------------------------------------------------------

    print()
    print("REAL ARTIFACTS")
    print("-" * 122)

    print(
        f"Random Forest             : {rf.path}"
    )

    print(
        f"Personalized AE           : {ae.path}"
    )

    print(
        f"FedProx70                 : {fedprox70.path}"
    )

    print(
        f"CNN-BiLSTM                : {phase14g1.CNN_CHECKPOINT}"
    )

    print(
        f"CNN frozen round          : "
        f"{cnn.checkpoint_round}"
    )

    # --------------------------------------------------------
    # Always-run-all research audit
    # --------------------------------------------------------

    audit_start = time.perf_counter()

    all_models = run_always_run_all_audit(
        current_flow_70=
            current_flow,

        sequence_20x70=
            sequence,

        rf=
            rf,

        ae=
            ae,

        fedprox70=
            fedprox70,

        cnn=
            cnn,
    )

    audit_ms = (
        time.perf_counter()
        -
        audit_start
    ) * 1000.0

    rf_audit = all_models[
        "rf_result"
    ]

    ae_audit = all_models[
        "ae_details"
    ]

    fedprox_audit = all_models[
        "fedprox_result"
    ]

    cnn_audit = all_models[
        "cnn_result"
    ]

    fused_audit = all_models[
        "risk_result"
    ]

    print()
    print("ALWAYS-RUN-ALL AUDIT — RESEARCH ONLY")
    print("-" * 122)

    print(
        f"RF                        : "
        f"{rf_audit.predicted_class} "
        f"({rf_audit.confidence:.6f})"
    )

    print(
        f"AE raw error              : "
        f"{ae_audit['raw_reconstruction_error']:.6f}"
    )

    print(
        f"AE anomaly                : "
        f"{ae_audit['raw_is_anomaly']}"
    )

    print(
        f"FedProx70                 : "
        f"{fedprox_audit.predicted_class} "
        f"({fedprox_audit.confidence:.6f})"
    )

    print(
        f"CNN-BiLSTM                : "
        f"{cnn_audit.predicted_class} "
        f"({cnn_audit.confidence:.6f})"
    )

    print(
        f"Full-fusion class         : "
        f"{fused_audit.final_class}"
    )

    print(
        f"Full-fusion risk          : "
        f"{fused_audit.risk_score:.6f}"
    )

    print(
        f"Full-fusion severity      : "
        f"{fused_audit.severity}"
    )

    print(
        f"Audit latency             : "
        f"{audit_ms:.3f} ms"
    )

    # --------------------------------------------------------
    # Corrected operational adaptive policy
    # --------------------------------------------------------

    adaptive_start = time.perf_counter()

    alert = detect_with_phase14g2_policy(
        client_id=
            args.client_id,

        current_flow_70=
            current_flow,

        sequence_20x70=
            sequence,

        rf=
            rf,

        ae=
            ae,

        fedprox70=
            fedprox70,

        cnn=
            cnn,
    )

    adaptive_ms = (
        time.perf_counter()
        -
        adaptive_start
    ) * 1000.0

    print()
    print("PHASE14G2 OPERATIONAL ALERT")
    print("-" * 122)

    print(
        json.dumps(
            json_safe(
                asdict(
                    alert
                )
            ),
            indent=4,
        )
    )

    print()

    print(
        f"Operational latency       : "
        f"{adaptive_ms:.3f} ms"
    )

    # --------------------------------------------------------
    # Semantic comparison
    # --------------------------------------------------------

    print()
    print("SEMANTIC CHECK")
    print("-" * 122)

    if (
        alert.processing_path
        ==
        "FAST_ATTACK_PATH"
    ):
        print(
            "Fast path used            : YES"
        )

        print(
            "Ensemble risk computed    : NO"
        )

        print(
            "risk_score                : NULL"
        )

        print(
            f"RF evidence/confidence    : "
            f"{alert.evidence_score:.6f}"
        )

        print(
            f"Operational severity      : "
            f"{alert.severity}"
        )

        print(
            "Comparable to full fusion : NO"
        )

    else:
        print(
            "Fast path used            : NO"
        )

        print(
            "Ensemble risk computed    : YES"
        )

        print(
            f"risk_score                : "
            f"{alert.risk_score:.6f}"
        )

        print(
            f"Operational severity      : "
            f"{alert.severity}"
        )

        print(
            "Comparable to full fusion : YES"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    alert_payload = {
        "phase":
            "14G2",

        "policy_version":
            POLICY_VERSION,

        "true_class":
            sample[
                "true_class"
            ],

        "validation_sequence_index":
            sample[
                "sample_index"
            ],

        "alert":
            asdict(
                alert
            ),

        "operational_latency_ms":
            float(
                adaptive_ms
            ),

        "locked_test_used":
            False,
    }

    save_json(
        CORRECTED_ALERT_FILE,
        alert_payload,
    )

    audit_payload = {
        "phase":
            "14G2",

        "policy_version":
            POLICY_VERSION,

        "purpose":
            (
                "Remove semantic conflation between RF-only "
                "fast-path confidence and full four-model "
                "ensemble risk."
            ),

        "true_class":
            sample[
                "true_class"
            ],

        "validation_sequence_index":
            sample[
                "sample_index"
            ],

        "always_run_all_research_audit":
            {
                "rf":
                    asdict(
                        rf_audit
                    ),

                "ae":
                    ae_audit,

                "fedprox70":
                    asdict(
                        fedprox_audit
                    ),

                "cnn_bilstm":
                    asdict(
                        cnn_audit
                    ),

                "full_fusion":
                    asdict(
                        fused_audit
                    ),

                "latency_ms":
                    float(
                        audit_ms
                    ),
            },

        "operational_adaptive_alert":
            asdict(
                alert
            ),

        "scientific_interpretation":
            (
                "RF confidence and full-fusion ensemble risk are "
                "different quantities. Phase14G2 no longer reports "
                "an RF-only confidence as if it were an ensemble "
                "risk score."
            ),

        "locked_test_used":
            False,
    }

    save_json(
        POLICY_AUDIT_FILE,
        audit_payload,
    )

    policy_config = {
        "phase":
            "14G2",

        "policy_version":
            POLICY_VERSION,

        "risk_weights":
            RISK_WEIGHTS,

        "severity_thresholds_for_full_fusion":
            SEVERITY_THRESHOLDS,

        "rf_fast_attack_threshold":
            RF_FAST_ATTACK_THRESHOLD,

        "fast_path":
            {
                "condition":
                    (
                        "RF predicts non-BENIGN and "
                        "confidence >= 0.95"
                    ),

                "classification":
                    "Random Forest predicted class",

                "confidence":
                    "Random Forest confidence",

                "risk_score":
                    None,

                "evidence_score":
                    "Random Forest confidence",

                "severity":
                    FAST_PATH_OPERATIONAL_SEVERITY,

                "severity_basis":
                    (
                        "Operational high-confidence known-attack "
                        "policy, not ensemble-risk thresholding."
                    ),

                "critical_allowed":
                    False,

                "risk_score_comparable_to_full_fusion":
                    False,
            },

        "deep_path":
            {
                "models":
                    [
                        "Random Forest",
                        "Personalized Autoencoder",
                        "FedProx70 MLP",
                        "Frozen Round-20 CNN-BiLSTM",
                    ],

                "risk_score":
                    "weighted four-model ensemble risk",

                "severity":
                    (
                        "NORMAL/LOW/MEDIUM/HIGH/CRITICAL "
                        "from frozen risk thresholds"
                    ),

                "critical_allowed":
                    True,

                "risk_score_comparable_to_full_fusion":
                    True,
            },

        "critical_requires_full_fusion":
            CRITICAL_REQUIRES_FULL_FUSION,

        "locked_test_used":
            False,

        "scientific_label":
            (
                "SOURCE-ORDERED CNN-BiLSTM sequences; "
                "NOT timestamp-confirmed temporal sequences."
            ),
    }

    save_json(
        POLICY_CONFIG_FILE,
        policy_config,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    separator()

    print(
        "PHASE 14G2 COMPLETE"
    )

    separator()

    print(
        f"True class                : "
        f"{sample['true_class']}"
    )

    print(
        f"Classification            : "
        f"{alert.classification}"
    )

    print(
        f"Processing path           : "
        f"{alert.processing_path}"
    )

    print(
        f"Confidence                : "
        f"{alert.confidence:.6f}"
    )

    print(
        f"Risk mode                 : "
        f"{alert.risk_mode}"
    )

    print(
        f"Risk score                : "
        f"{alert.risk_score}"
    )

    print(
        f"Evidence score            : "
        f"{alert.evidence_score:.6f}"
    )

    print(
        f"Severity                  : "
        f"{alert.severity}"
    )

    print(
        f"Decision source           : "
        f"{alert.decision_source}"
    )

    print()

    print(
        f"Corrected alert           : "
        f"{CORRECTED_ALERT_FILE}"
    )

    print(
        f"Policy audit              : "
        f"{POLICY_AUDIT_FILE}"
    )

    print(
        f"Frozen policy config      : "
        f"{POLICY_CONFIG_FILE}"
    )

    print()

    print(
        "LOCKED TEST USED : NO"
    )

    print()

    print(
        "SCIENTIFIC RULE:"
    )

    print(
        "RF confidence != ensemble risk."
    )

    print(
        "FAST path reports RF confidence as evidence only."
    )

    print(
        "DEEP path reports the four-model weighted risk score."
    )

    print(
        "CRITICAL severity is reserved for full-fusion evidence."
    )


if __name__ == "__main__":
    main()
