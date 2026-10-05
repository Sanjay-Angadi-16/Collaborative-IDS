from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from hashlib import sha256
import hmac
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import phase14g1_real_model_wiring as phase14g1
    import phase14h_full_validation_adaptive_gate as phase14h
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "\nThis script must be placed in your project's scripts folder.\n"
        "Required existing files:\n"
        "  scripts\\phase14g1_real_model_wiring.py\n"
        "  scripts\\phase14h_full_validation_adaptive_gate.py\n"
    ) from exc


# ============================================================
# CONFIG
# ============================================================

PHASE_NAME = "PHASE 17H — VALIDATION REPLAY -> DASHBOARD INTEGRATION"

SEED = 42
NUM_CLASSES = 6
NUM_CLIENTS = 5
EXPECTED_FEATURES = 70
SEQUENCE_LENGTH = 20
DEFAULT_SAMPLES = 1000
DEFAULT_BATCH_SIZE = 2048

ANOMALOUS_ID = 6
ANOMALOUS_NAME = "ANOMALOUS_TRAFFIC"

RESULT_ROOT = PROJECT_ROOT / "results" / "dashboard"
ALERTS_JSON = RESULT_ROOT / "validation_replay_alerts.json"
ALERTS_CSV = RESULT_ROOT / "validation_replay_alerts.csv"
SUMMARY_JSON = RESULT_ROOT / "validation_replay_summary.json"

# Only used for pseudonymous validation source IDs.
# No raw IP exists in validation_sequences.npz.
PSEUDONYM_KEY = b"phase17h-validation-replay-v1"


# ============================================================
# HELPERS
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def pseudonymous_validation_source(original_index: int) -> str:
    message = f"validation-index:{original_index}".encode("utf-8")
    digest = hmac.new(
        PSEUDONYM_KEY,
        message,
        sha256,
    ).hexdigest()
    return f"VAL-{digest[:12].upper()}"


def severity_name(severity_id: int) -> str:
    return str(
        phase14h.SEVERITY_NAMES[
            int(severity_id)
        ]
    )


def class_name(
    class_id: int,
    inverse_mapping: dict[int, str],
) -> str:
    class_id = int(class_id)
    if class_id == ANOMALOUS_ID:
        return ANOMALOUS_NAME
    return str(
        inverse_mapping.get(
            class_id,
            f"CLASS_{class_id}",
        )
    )


# ============================================================
# VALIDATION SAMPLE SELECTION
# ============================================================

def load_validation_subset(
    requested_samples: int,
):
    """
    Loads ONLY the existing validation sequence artifact.
    The locked test is never opened.

    Returns:
        sequences: sampled (N,20,70)
        flows:     last flow from each sequence, (N,70)
        y_true:    sampled labels
        indices:   original validation-sequence indices
    """
    validation_file = phase14h.VALIDATION_SEQUENCE_FILE

    if not validation_file.exists():
        raise FileNotFoundError(
            "\nMissing validation sequence file:\n"
            f"{validation_file}\n\n"
            "Run the existing Phase 14 sequence-preparation pipeline first."
        )

    with np.load(
        validation_file,
        allow_pickle=False,
    ) as data:
        X_all = data["X"].astype(
            np.float32,
            copy=False,
        )
        y_all = data["y"].astype(
            np.int64,
            copy=False,
        )

    if X_all.ndim != 3:
        raise RuntimeError(
            f"Expected X to be 3D; found {X_all.shape}."
        )

    if tuple(X_all.shape[1:]) != (
        SEQUENCE_LENGTH,
        EXPECTED_FEATURES,
    ):
        raise RuntimeError(
            "Validation sequence shape mismatch: "
            f"expected (*,{SEQUENCE_LENGTH},{EXPECTED_FEATURES}), "
            f"found {X_all.shape}."
        )

    if len(X_all) != len(y_all):
        raise RuntimeError("Validation X/y count mismatch.")

    if requested_samples <= 0:
        raise ValueError("--samples must be > 0.")

    requested_samples = min(
        int(requested_samples),
        len(y_all),
    )

    # Deterministic class-stratified subset, based on Phase14H.
    rng = np.random.default_rng(SEED)
    chosen: list[int] = []

    per_class = max(
        1,
        requested_samples // NUM_CLASSES,
    )

    for class_id in range(NUM_CLASSES):
        class_indices = np.flatnonzero(
            y_all == class_id
        )

        take = min(
            per_class,
            len(class_indices),
        )

        if take > 0:
            selected = rng.choice(
                class_indices,
                size=take,
                replace=False,
            )
            chosen.extend(
                selected.tolist()
            )

    if len(chosen) < requested_samples:
        already = set(chosen)
        remaining = np.asarray(
            [
                i
                for i in range(len(y_all))
                if i not in already
            ],
            dtype=np.int64,
        )

        extra_count = min(
            requested_samples - len(chosen),
            len(remaining),
        )

        if extra_count > 0:
            extra = rng.choice(
                remaining,
                size=extra_count,
                replace=False,
            )
            chosen.extend(
                extra.tolist()
            )

    chosen = np.asarray(
        sorted(
            chosen[:requested_samples]
        ),
        dtype=np.int64,
    )

    sequences = X_all[chosen]
    y_true = y_all[chosen]

    flows = np.asarray(
        sequences[:, -1, :],
        dtype=np.float32,
    )

    return (
        sequences,
        flows,
        y_true,
        chosen,
        validation_file,
    )


# ============================================================
# MAIN INFERENCE
# ============================================================

def run_replay(
    samples: int,
    batch_size: int,
):
    set_seed(SEED)
    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    (
        label_mapping,
        inverse_mapping,
        active_features,
    ) = phase14g1.load_project_schema()

    if len(active_features) != EXPECTED_FEATURES:
        raise RuntimeError(
            f"Expected {EXPECTED_FEATURES} active features; "
            f"found {len(active_features)}."
        )

    (
        sequences,
        flows,
        y_true,
        original_indices,
        validation_file,
    ) = load_validation_subset(
        requested_samples=samples
    )

    print("=" * 120)
    print(PHASE_NAME)
    print("=" * 120)
    print(f"Device                 : {device}")
    print(f"Validation source      : {validation_file}")
    print(f"Requested samples      : {samples}")
    print(f"Loaded samples         : {len(y_true)}")
    print(f"Sequence shape         : {sequences.shape}")
    print(f"Flow shape             : {flows.shape}")
    print("Client assignment      : deterministic round-robin; true label NOT used")
    print("Raw IP available       : NO (validation sequence artifact contains model features only)")
    print("Locked test used       : NO")
    print()

    # --------------------------------------------------------
    # Global models once
    # --------------------------------------------------------
    print("Loading global models...")

    fedprox70 = phase14g1.RealFedProx70Predictor(
        checkpoint_path=phase14g1.FEDPROX70_CHECKPOINT,
        inverse_mapping=inverse_mapping,
        device=device,
    )

    cnn = phase14g1.build_real_cnn_predictor(
        inverse_mapping=inverse_mapping,
        device=device,
    )

    start = time.perf_counter()

    fed_pred, fed_conf = phase14h.predict_fedprox70_batch(
        predictor=fedprox70,
        flows=flows,
        batch_size=batch_size,
        device=device,
    )

    cnn_pred, cnn_conf = phase14h.predict_cnn_batch(
        predictor=cnn,
        sequences=sequences,
        batch_size=batch_size,
        device=device,
    )

    global_seconds = (
        time.perf_counter() - start
    )

    # --------------------------------------------------------
    # Assign each validation sample to one client without label
    # leakage. This is an integration replay, not a federated
    # scientific evaluation.
    # --------------------------------------------------------
    assigned_clients = (
        np.arange(len(y_true)) % NUM_CLIENTS
    ) + 1

    adaptive_pred = np.empty(
        len(y_true),
        dtype=np.int64,
    )
    adaptive_conf = np.empty(
        len(y_true),
        dtype=np.float32,
    )
    adaptive_risk = np.empty(
        len(y_true),
        dtype=np.float32,
    )
    adaptive_severity = np.empty(
        len(y_true),
        dtype=np.int8,
    )
    processing_path = np.empty(
        len(y_true),
        dtype=object,
    )
    decision_source = np.empty(
        len(y_true),
        dtype=object,
    )

    rf_pred_all = np.empty(
        len(y_true),
        dtype=np.int64,
    )
    rf_conf_all = np.empty(
        len(y_true),
        dtype=np.float32,
    )
    ae_error_all = np.empty(
        len(y_true),
        dtype=np.float32,
    )
    ae_score_all = np.empty(
        len(y_true),
        dtype=np.float32,
    )
    ae_anomaly_all = np.empty(
        len(y_true),
        dtype=bool,
    )

    local_seconds_total = 0.0

    # --------------------------------------------------------
    # Local client models
    # --------------------------------------------------------
    for client_id in range(1, NUM_CLIENTS + 1):
        positions = np.flatnonzero(
            assigned_clients == client_id
        )

        if len(positions) == 0:
            continue

        print(
            f"Client {client_id}: "
            f"{len(positions)} replay samples"
        )

        rf = phase14g1.RealRandomForestPredictor(
            client_id=client_id,
            label_mapping=label_mapping,
            inverse_mapping=inverse_mapping,
        )

        ae = phase14g1.RealPersonalizedAutoencoderPredictor(
            client_id=client_id,
            device=device,
            explicit_model_path=None,
        )

        client_flows = flows[positions]

        local_start = time.perf_counter()

        rf_pred, rf_conf = phase14h.predict_rf_batch(
            rf=rf,
            flows=client_flows,
        )

        (
            ae_error,
            ae_score,
            ae_anomaly,
        ) = phase14h.predict_ae_batch(
            ae=ae,
            flows=client_flows,
            batch_size=batch_size,
            device=device,
        )

        local_seconds_total += (
            time.perf_counter() - local_start
        )

        full_result = phase14h.full_fusion_vectorized(
            rf_pred=rf_pred,
            rf_conf=rf_conf,
            ae_score=ae_score,
            fed_pred=fed_pred[positions],
            fed_conf=fed_conf[positions],
            cnn_pred=cnn_pred[positions],
            cnn_conf=cnn_conf[positions],
        )

        adaptive_result = phase14h.adaptive_policy_vectorized(
            rf_pred=rf_pred,
            rf_conf=rf_conf,
            full_result=full_result,
        )

        rf_pred_all[positions] = rf_pred
        rf_conf_all[positions] = rf_conf
        ae_error_all[positions] = ae_error
        ae_score_all[positions] = ae_score
        ae_anomaly_all[positions] = ae_anomaly

        adaptive_pred[positions] = adaptive_result[
            "prediction"
        ]
        adaptive_conf[positions] = adaptive_result[
            "confidence"
        ]
        adaptive_risk[positions] = adaptive_result[
            "risk"
        ]
        adaptive_severity[positions] = adaptive_result[
            "severity_id"
        ]

        fast_mask = adaptive_result[
            "fast_mask"
        ]

        processing_path[positions] = np.where(
            fast_mask,
            "FAST_ATTACK_PATH",
            "DEEP_ANALYSIS",
        )

        # For deep path, use Phase14H full-fusion decision source.
        decision_source[positions] = np.where(
            fast_mask,
            "Random Forest",
            full_result["decision_source"],
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    # Known project classes are 0..5.
    # Class 6 is the Phase-14 anomaly/rejection output and is
    # reported separately instead of being mixed into known-class
    # macro metrics.
    known_labels = [0, 1, 2, 3, 4, 5]

    anomalous_count = int(
        np.sum(
            adaptive_pred == ANOMALOUS_ID
        )
    )

    anomalous_rate = (
        anomalous_count
        /
        len(adaptive_pred)
    )

    # Overall exact prediction accuracy.
    # A class-6 prediction against a known class 0..5 is counted
    # as incorrect, which is appropriate for this validation replay.
    accuracy = float(
        accuracy_score(
            y_true,
            adaptive_pred,
        )
    )

    # Evaluate macro metrics only across the six known classes.
    # The ANOMALOUS_TRAFFIC output is reported separately below.
    macro_precision = float(
        precision_score(
            y_true,
            adaptive_pred,
            labels=known_labels,
            average="macro",
            zero_division=0,
        )
    )

    macro_recall = float(
        recall_score(
            y_true,
            adaptive_pred,
            labels=known_labels,
            average="macro",
            zero_division=0,
        )
    )

    macro_f1 = float(
        f1_score(
            y_true,
            adaptive_pred,
            labels=known_labels,
            average="macro",
            zero_division=0,
        )
    )

    # For this six-known-class validation replay, balanced accuracy
    # is the mean per-class recall across known labels.
    balanced_accuracy = macro_recall

    weighted_f1 = float(
        f1_score(
            y_true,
            adaptive_pred,
            labels=known_labels,
            average="weighted",
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Dashboard alerts
    #
    # IMPORTANT:
    # validation_sequences.npz does NOT preserve raw source /
    # destination IPs. We therefore expose a privacy-safe
    # validation source ID and explicitly set source_ip=None.
    # --------------------------------------------------------
    now = datetime.now(
        timezone.utc
    )

    alerts = []

    for i in range(len(y_true)):
        pred_id = int(
            adaptive_pred[i]
        )

        true_id = int(
            y_true[i]
        )

        risk_value = adaptive_risk[i]
        risk_score = (
            None
            if np.isnan(risk_value)
            else float(risk_value)
        )

        alert = {
            "id": int(i + 1),
            "replay_type": "VALIDATION_SAMPLE",
            "validation_index": int(
                original_indices[i]
            ),
            "event_time": now.isoformat(),
            "client_id": int(
                assigned_clients[i]
            ),
            "source_id": (
                pseudonymous_validation_source(
                    int(original_indices[i])
                )
            ),
            "source_ip": None,
            "destination_ip": None,
            "ip_note": (
                "Raw IP is not present in validation_sequences.npz. "
                "Live deployment must retain raw IP only at the local node."
            ),
            "true_class_id": true_id,
            "true_class": class_name(
                true_id,
                inverse_mapping,
            ),
            "prediction_id": pred_id,
            "attack_type": class_name(
                pred_id,
                inverse_mapping,
            ),
            "confidence": float(
                adaptive_conf[i]
            ),
            "risk_score": risk_score,
            "anomaly_score": float(
                ae_score_all[i]
            ),
            "raw_reconstruction_error": float(
                ae_error_all[i]
            ),
            "ae_anomaly": bool(
                ae_anomaly_all[i]
            ),
            "severity": severity_name(
                int(adaptive_severity[i])
            ),
            "processing_path": str(
                processing_path[i]
            ),
            "decision_source": str(
                decision_source[i]
            ),
            "rf_prediction": class_name(
                int(rf_pred_all[i]),
                inverse_mapping,
            ),
            "rf_confidence": float(
                rf_conf_all[i]
            ),
            "fedprox70_prediction": class_name(
                int(fed_pred[i]),
                inverse_mapping,
            ),
            "fedprox70_confidence": float(
                fed_conf[i]
            ),
            "cnn_bilstm_prediction": class_name(
                int(cnn_pred[i]),
                inverse_mapping,
            ),
            "cnn_bilstm_confidence": float(
                cnn_conf[i]
            ),
            "correct": bool(
                pred_id == true_id
            ),
            "correlated": False,
            "status": "ACTIVE",
            "blockable": False,
            "locked_test_used": False,
        }

        alerts.append(alert)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    with open(
        ALERTS_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            json_safe(alerts),
            f,
            indent=2,
        )

    pd.DataFrame(
        alerts
    ).to_csv(
        ALERTS_CSV,
        index=False,
    )

    class_distribution = {
        class_name(
            class_id,
            inverse_mapping,
        ): int(
            np.sum(
                y_true == class_id
            )
        )
        for class_id
        in range(NUM_CLASSES)
    }

    prediction_distribution = {}
    for pred_id in np.unique(
        adaptive_pred
    ):
        prediction_distribution[
            class_name(
                int(pred_id),
                inverse_mapping,
            )
        ] = int(
            np.sum(
                adaptive_pred == pred_id
            )
        )

    summary = {
        "phase": "17H",
        "purpose": (
            "Validation-only dashboard integration replay "
            "using the existing frozen Phase14 models/policy."
        ),
        "sample_count": int(
            len(y_true)
        ),
        "anomalous_predictions": anomalous_count,
        "anomalous_rate": float(
            anomalous_rate
        ),
        "validation_source": str(
            validation_file
        ),
        "seed": SEED,
        "assignment_policy": (
            "deterministic round-robin across 5 clients; "
            "true label not used"
        ),
        "class_distribution": class_distribution,
        "prediction_distribution": (
            prediction_distribution
        ),
        "metrics": {
            "accuracy": accuracy,
            "balanced_accuracy": (
                balanced_accuracy
            ),
            "macro_precision": (
                macro_precision
            ),
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        },
        "fast_path_count": int(
            np.sum(
                processing_path
                == "FAST_ATTACK_PATH"
            )
        ),
        "deep_path_count": int(
            np.sum(
                processing_path
                == "DEEP_ANALYSIS"
            )
        ),
        "global_inference_seconds": float(
            global_seconds
        ),
        "local_inference_seconds_total": float(
            local_seconds_total
        ),
        "raw_ip_available": False,
        "locked_test_used": False,
        "alerts_json": str(
            ALERTS_JSON
        ),
        "alerts_csv": str(
            ALERTS_CSV
        ),
    }

    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            json_safe(summary),
            f,
            indent=2,
        )

    print()
    print("=" * 120)
    print("VALIDATION REPLAY COMPLETE")
    print("=" * 120)
    print(f"Accuracy              : {accuracy:.6f}")
    print(f"Balanced Accuracy     : {balanced_accuracy:.6f}")
    print(f"Macro Precision       : {macro_precision:.6f}")
    print(f"Macro Recall          : {macro_recall:.6f}")
    print(f"Macro F1              : {macro_f1:.6f}")
    print(f"Weighted F1           : {weighted_f1:.6f}")
    print(
        f"Anomalous predictions : "
        f"{anomalous_count}"
    )
    print(
        f"Anomalous rate        : "
        f"{anomalous_rate * 100:.2f}%"
    )
    print(
        "Fast / Deep           : "
        f"{summary['fast_path_count']} / "
        f"{summary['deep_path_count']}"
    )
    print(f"Alerts JSON            : {ALERTS_JSON}")
    print(f"Alerts CSV             : {ALERTS_CSV}")
    print(f"Summary                : {SUMMARY_JSON}")
    print("Raw IPs                : NOT AVAILABLE in validation sequence artifact")
    print("LOCKED TEST USED       : NO")
    print()
    print(
        "Next: start ids_dashboard_connected.py and open "
        "http://127.0.0.1:9999"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=PHASE_NAME
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help=(
            "Validation samples to replay. "
            "Use 100 for smoke test or 1000 for integration test."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_replay(
        samples=args.samples,
        batch_size=args.batch_size,
    )
