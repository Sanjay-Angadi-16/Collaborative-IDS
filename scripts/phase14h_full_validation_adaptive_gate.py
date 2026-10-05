from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import confusion_matrix


# ============================================================
# PROJECT ROOT / FROZEN PHASE IMPORTS
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
        "\nCould not import scripts\\phase14g1_real_model_wiring.py"
    ) from exc

try:
    import phase14g2_consistent_risk_policy as phase14g2
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "\nCould not import scripts\\phase14g2_consistent_risk_policy.py"
    ) from exc


# ============================================================
# PHASE
# ============================================================

PHASE_NAME = (
    "PHASE 14H — FULL VALIDATION + ADAPTIVE-GATE EVALUATION"
)

POLICY_VERSION = "phase14g2_v1"

SEED = 42

EXPECTED_FEATURES = 70
SEQUENCE_LENGTH = 20
NUM_CLASSES = 6

ANOMALOUS_ID = 6
ANOMALOUS_NAME = "ANOMALOUS_TRAFFIC"

RF_FAST_ATTACK_THRESHOLD = 0.95
AE_DECISION_SCORE = 0.70

RISK_WEIGHTS = {
    "rf": 0.20,
    "fedprox70": 0.30,
    "cnn_bilstm": 0.35,
    "autoencoder": 0.15,
}

SEVERITY_NAMES = np.asarray(
    [
        "NORMAL",
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL",
    ],
    dtype=object,
)

SEVERITY_NORMAL = 0
SEVERITY_LOW = 1
SEVERITY_MEDIUM = 2
SEVERITY_HIGH = 3
SEVERITY_CRITICAL = 4

DEFAULT_BATCH_SIZE = 2048

# Metrics use the FULL validation set by default.
# Per-event latency quantiles are measured on a deterministic,
# class-stratified profiling subset to avoid turning the full
# scientific evaluation into hours of Python-call overhead.
DEFAULT_LATENCY_SAMPLES_PER_CLIENT = 300


# ============================================================
# INPUTS
# ============================================================

VALIDATION_SEQUENCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "sequences"
    / "validation_sequences.npz"
)


# ============================================================
# OUTPUTS
# ============================================================

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "phase14"
    / "phase14h"
)

ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "detection"
    / "phase14h"
)

CLIENT_SUMMARY_FILE = (
    RESULT_ROOT
    / "phase14h_client_summary.csv"
)

MEAN_CLIENT_METRICS_FILE = (
    RESULT_ROOT
    / "phase14h_mean_client_metrics.csv"
)

POOLED_EXPOSURE_METRICS_FILE = (
    RESULT_ROOT
    / "phase14h_pooled_client_exposure_metrics.json"
)

PER_CLASS_FILE = (
    RESULT_ROOT
    / "phase14h_per_class_metrics.csv"
)

PATH_AUDIT_FILE = (
    RESULT_ROOT
    / "phase14h_adaptive_gate_audit.csv"
)

SEVERITY_FILE = (
    RESULT_ROOT
    / "phase14h_severity_distribution.csv"
)

DISAGREEMENT_FILE = (
    RESULT_ROOT
    / "phase14h_model_disagreement.csv"
)

LATENCY_RAW_FILE = (
    RESULT_ROOT
    / "phase14h_latency_samples.csv"
)

LATENCY_SUMMARY_FILE = (
    RESULT_ROOT
    / "phase14h_latency_summary.csv"
)

FINAL_SUMMARY_FILE = (
    RESULT_ROOT
    / "phase14h_summary.json"
)

CONFIG_FILE = (
    ARTIFACT_ROOT
    / "phase14h_evaluation_config.json"
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def separator(width: int = 126) -> None:
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


def percentile(
    values: np.ndarray,
    q: float,
) -> float:
    if len(values) == 0:
        return float("nan")

    return float(
        np.percentile(
            values,
            q,
        )
    )


# ============================================================
# VALIDATION DATA
# ============================================================

def load_validation_sequences(
    max_samples: Optional[int],
):
    if not VALIDATION_SEQUENCE_FILE.exists():
        raise FileNotFoundError(
            "\nMissing validation sequences:\n"
            f"{VALIDATION_SEQUENCE_FILE}"
        )

    with np.load(
        VALIDATION_SEQUENCE_FILE,
        allow_pickle=False,
    ) as data:
        X = data[
            "X"
        ].astype(
            np.float32,
            copy=False,
        )

        y = data[
            "y"
        ].astype(
            np.int64,
            copy=False,
        )

    if X.ndim != 3:
        raise RuntimeError(
            f"\nExpected validation X to be 3D; found {X.shape}."
        )

    if tuple(
        X.shape[
            1:
        ]
    ) != (
        SEQUENCE_LENGTH,
        EXPECTED_FEATURES,
    ):
        raise RuntimeError(
            "\nExpected validation sequence shape "
            f"(*,{SEQUENCE_LENGTH},{EXPECTED_FEATURES}); "
            f"found {X.shape}."
        )

    if y.ndim != 1:
        raise RuntimeError(
            "\nValidation y must be one-dimensional."
        )

    if len(X) != len(y):
        raise RuntimeError(
            "\nValidation X/y count mismatch."
        )

    if max_samples is not None:
        max_samples = int(
            max_samples
        )

        if max_samples <= 0:
            raise ValueError(
                "--max-validation-samples must be > 0."
            )

        if max_samples < len(
            y
        ):
            # Deterministic class-stratified reduction for debugging only.
            rng = np.random.default_rng(
                SEED
            )

            chosen = []

            per_class = max(
                1,
                max_samples
                //
                NUM_CLASSES,
            )

            for class_id in range(
                NUM_CLASSES
            ):
                class_indices = np.flatnonzero(
                    y
                    ==
                    class_id
                )

                take = min(
                    per_class,
                    len(
                        class_indices
                    ),
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

            if len(
                chosen
            ) < max_samples:
                already = set(
                    chosen
                )

                remaining = np.asarray(
                    [
                        index
                        for index
                        in range(
                            len(
                                y
                            )
                        )
                        if index not in already
                    ],
                    dtype=np.int64,
                )

                extra_count = min(
                    max_samples
                    -
                    len(
                        chosen
                    ),
                    len(
                        remaining
                    ),
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
                    chosen[
                        :max_samples
                    ]
                ),
                dtype=np.int64,
            )

            X = X[
                chosen
            ]

            y = y[
                chosen
            ]

    current_flows = np.asarray(
        X[
            :,
            -1,
            :,
        ],
        dtype=np.float32,
    )

    return (
        X,
        current_flows,
        y,
    )


# ============================================================
# BATCHED GLOBAL MODEL INFERENCE
# ============================================================

@torch.no_grad()
def predict_fedprox70_batch(
    predictor,
    flows: np.ndarray,
    batch_size: int,
    device: torch.device,
):
    model = predictor.model

    model.eval()

    pred_ids = np.empty(
        len(
            flows
        ),
        dtype=np.int64,
    )

    confidences = np.empty(
        len(
            flows
        ),
        dtype=np.float32,
    )

    for start in range(
        0,
        len(
            flows
        ),
        batch_size,
    ):
        end = min(
            start
            +
            batch_size,
            len(
                flows
            ),
        )

        tensor = torch.from_numpy(
            np.asarray(
                flows[
                    start:end
                ],
                dtype=np.float32,
            )
        ).to(
            device
        )

        logits = model(
            tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        confidence_tensor, prediction_tensor = torch.max(
            probabilities,
            dim=1,
        )

        pred_ids[
            start:end
        ] = (
            prediction_tensor
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.int64,
                copy=False,
            )
        )

        confidences[
            start:end
        ] = (
            confidence_tensor
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

    return (
        pred_ids,
        confidences,
    )


@torch.no_grad()
def predict_cnn_batch(
    predictor,
    sequences: np.ndarray,
    batch_size: int,
    device: torch.device,
):
    model = predictor.model

    model.eval()

    pred_ids = np.empty(
        len(
            sequences
        ),
        dtype=np.int64,
    )

    confidences = np.empty(
        len(
            sequences
        ),
        dtype=np.float32,
    )

    for start in range(
        0,
        len(
            sequences
        ),
        batch_size,
    ):
        end = min(
            start
            +
            batch_size,
            len(
                sequences
            ),
        )

        tensor = torch.from_numpy(
            np.asarray(
                sequences[
                    start:end
                ],
                dtype=np.float32,
            )
        ).to(
            device
        )

        logits = model(
            tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        confidence_tensor, prediction_tensor = torch.max(
            probabilities,
            dim=1,
        )

        pred_ids[
            start:end
        ] = (
            prediction_tensor
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.int64,
                copy=False,
            )
        )

        confidences[
            start:end
        ] = (
            confidence_tensor
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

    return (
        pred_ids,
        confidences,
    )


# ============================================================
# BATCHED CLIENT RF / AE
# ============================================================

def predict_rf_batch(
    rf,
    flows: np.ndarray,
):
    probabilities = np.asarray(
        rf.model.predict_proba(
            flows
        )
    )

    model_classes = np.asarray(
        rf.model.classes_
    )

    mapped_classes = np.asarray(
        [
            rf._map_model_class_to_global_id(
                model_class,
                model_classes,
            )
            for model_class
            in model_classes
        ],
        dtype=np.int64,
    )

    best_positions = np.argmax(
        probabilities,
        axis=1,
    )

    pred_ids = mapped_classes[
        best_positions
    ]

    confidences = probabilities[
        np.arange(
            len(
                flows
            )
        ),
        best_positions,
    ].astype(
        np.float32,
        copy=False,
    )

    return (
        pred_ids.astype(
            np.int64,
            copy=False,
        ),
        confidences,
    )


def _extract_autoencoder_tensor_output(
    output,
):
    if isinstance(
        output,
        (tuple, list),
    ):
        output = output[
            0
        ]

    if isinstance(
        output,
        dict,
    ):
        for key in (
            "reconstruction",
            "output",
            "decoded",
        ):
            if key in output:
                output = output[
                    key
                ]
                break

    if not isinstance(
        output,
        torch.Tensor,
    ):
        raise TypeError(
            "\nAutoencoder forward() did not return "
            "a reconstruction tensor."
        )

    return output


@torch.no_grad()
def predict_ae_batch(
    ae,
    flows: np.ndarray,
    batch_size: int,
    device: torch.device,
):
    reconstructed = np.empty_like(
        flows,
        dtype=np.float32,
    )

    if ae.backend.startswith(
        "joblib"
    ):
        reconstructed[
            :
        ] = np.asarray(
            ae.model.predict(
                flows
            ),
            dtype=np.float32,
        )

    else:
        ae.model.eval()

        for start in range(
            0,
            len(
                flows
            ),
            batch_size,
        ):
            end = min(
                start
                +
                batch_size,
                len(
                    flows
                ),
            )

            tensor = torch.from_numpy(
                np.asarray(
                    flows[
                        start:end
                    ],
                    dtype=np.float32,
                )
            ).to(
                device
            )

            output = _extract_autoencoder_tensor_output(
                ae.model(
                    tensor
                )
            )

            reconstructed[
                start:end
            ] = (
                output
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32,
                    copy=False,
                )
            )

    if reconstructed.shape != flows.shape:
        raise RuntimeError(
            "\nAutoencoder reconstruction shape mismatch:\n"
            f"flow={flows.shape}, reconstruction={reconstructed.shape}"
        )

    raw_error = np.mean(
        (
            flows
            -
            reconstructed
        )
        ** 2,
        axis=1,
        dtype=np.float64,
    ).astype(
        np.float32
    )

    scores = np.clip(
        AE_DECISION_SCORE
        *
        raw_error
        /
        float(
            ae.threshold
        ),
        0.0,
        1.0,
    ).astype(
        np.float32,
        copy=False,
    )

    anomaly = (
        raw_error
        >=
        float(
            ae.threshold
        )
    )

    return (
        raw_error,
        scores,
        anomaly,
    )


# ============================================================
# VECTORISED FULL FUSION
# EXACTLY MATCHES FROZEN PHASE14G DECISION LOGIC
# ============================================================

def classification_risk_vector(
    pred_ids: np.ndarray,
    confidences: np.ndarray,
) -> np.ndarray:
    return np.where(
        pred_ids
        ==
        0,
        1.0
        -
        confidences,
        confidences,
    ).astype(
        np.float32,
        copy=False,
    )


def severity_ids_from_risk(
    risk_score: np.ndarray,
) -> np.ndarray:
    result = np.empty(
        len(
            risk_score
        ),
        dtype=np.int8,
    )

    result[
        risk_score
        <
        0.20
    ] = SEVERITY_NORMAL

    result[
        (
            risk_score
            >=
            0.20
        )
        &
        (
            risk_score
            <
            0.40
        )
    ] = SEVERITY_LOW

    result[
        (
            risk_score
            >=
            0.40
        )
        &
        (
            risk_score
            <
            0.60
        )
    ] = SEVERITY_MEDIUM

    result[
        (
            risk_score
            >=
            0.60
        )
        &
        (
            risk_score
            <
            0.80
        )
    ] = SEVERITY_HIGH

    result[
        risk_score
        >=
        0.80
    ] = SEVERITY_CRITICAL

    return result


def full_fusion_vectorized(
    *,
    rf_pred,
    rf_conf,
    ae_score,
    fed_pred,
    fed_conf,
    cnn_pred,
    cnn_conf,
):
    n = len(
        rf_pred
    )

    rf_risk = classification_risk_vector(
        rf_pred,
        rf_conf,
    )

    fed_risk = classification_risk_vector(
        fed_pred,
        fed_conf,
    )

    cnn_risk = classification_risk_vector(
        cnn_pred,
        cnn_conf,
    )

    full_risk = (
        RISK_WEIGHTS[
            "rf"
        ]
        *
        rf_risk
        +
        RISK_WEIGHTS[
            "fedprox70"
        ]
        *
        fed_risk
        +
        RISK_WEIGHTS[
            "cnn_bilstm"
        ]
        *
        cnn_risk
        +
        RISK_WEIGHTS[
            "autoencoder"
        ]
        *
        ae_score
    ).astype(
        np.float32
    )

    classifier_predictions = np.stack(
        [
            rf_pred,
            fed_pred,
            cnn_pred,
        ],
        axis=1,
    )

    classifier_confidences = np.stack(
        [
            rf_conf,
            fed_conf,
            cnn_conf,
        ],
        axis=1,
    ).astype(
        np.float32
    )

    non_benign_conf = np.where(
        classifier_predictions
        !=
        0,
        classifier_confidences,
        -1.0,
    )

    winner_position = np.argmax(
        non_benign_conf,
        axis=1,
    )

    has_non_benign = np.max(
        non_benign_conf,
        axis=1,
    ) >= 0.0

    rows = np.arange(
        n
    )

    full_pred = np.zeros(
        n,
        dtype=np.int64,
    )

    full_conf = np.max(
        classifier_confidences,
        axis=1,
    ).astype(
        np.float32
    )

    decision_source = np.full(
        n,
        "Risk Fusion",
        dtype=object,
    )

    source_names = np.asarray(
        [
            "Random Forest",
            "FedProx70 MLP",
            "CNN-BiLSTM",
        ],
        dtype=object,
    )

    if np.any(
        has_non_benign
    ):
        chosen_rows = rows[
            has_non_benign
        ]

        chosen_positions = winner_position[
            has_non_benign
        ]

        full_pred[
            has_non_benign
        ] = classifier_predictions[
            chosen_rows,
            chosen_positions,
        ]

        full_conf[
            has_non_benign
        ] = classifier_confidences[
            chosen_rows,
            chosen_positions,
        ]

        decision_source[
            has_non_benign
        ] = source_names[
            chosen_positions
        ]

    no_attack_classifier = (
        ~has_non_benign
    )

    anomaly_only = (
        no_attack_classifier
        &
        (
            ae_score
            >=
            AE_DECISION_SCORE
        )
    )

    full_pred[
        anomaly_only
    ] = ANOMALOUS_ID

    full_conf[
        anomaly_only
    ] = ae_score[
        anomaly_only
    ]

    decision_source[
        anomaly_only
    ] = "Personalized Autoencoder"

    # Remaining rows are BENIGN. Confidence already equals the
    # highest benign classifier confidence, matching Phase14G.
    benign_rows = (
        no_attack_classifier
        &
        ~anomaly_only
    )

    if np.any(
        benign_rows
    ):
        benign_matrix = classifier_confidences[
            benign_rows
        ]

        benign_winner = np.argmax(
            benign_matrix,
            axis=1,
        )

        decision_source[
            benign_rows
        ] = source_names[
            benign_winner
        ]

    severity = severity_ids_from_risk(
        full_risk
    )

    return {
        "prediction":
            full_pred,

        "confidence":
            full_conf,

        "risk":
            full_risk,

        "severity_id":
            severity,

        "decision_source":
            decision_source,
    }


# ============================================================
# VECTORISED PHASE14G2 ADAPTIVE POLICY
# ============================================================

def adaptive_policy_vectorized(
    *,
    rf_pred,
    rf_conf,
    full_result,
):
    fast_mask = (
        (
            rf_pred
            !=
            0
        )
        &
        (
            rf_conf
            >=
            RF_FAST_ATTACK_THRESHOLD
        )
    )

    adaptive_pred = np.where(
        fast_mask,
        rf_pred,
        full_result[
            "prediction"
        ],
    ).astype(
        np.int64,
        copy=False,
    )

    adaptive_conf = np.where(
        fast_mask,
        rf_conf,
        full_result[
            "confidence"
        ],
    ).astype(
        np.float32,
        copy=False,
    )

    # FAST path has no ensemble risk. NaN is the numeric storage
    # representation for the JSON-level semantic NULL.
    adaptive_risk = np.where(
        fast_mask,
        np.nan,
        full_result[
            "risk"
        ],
    ).astype(
        np.float32,
        copy=False,
    )

    adaptive_severity = np.where(
        fast_mask,
        SEVERITY_HIGH,
        full_result[
            "severity_id"
        ],
    ).astype(
        np.int8,
        copy=False,
    )

    return {
        "fast_mask":
            fast_mask,

        "prediction":
            adaptive_pred,

        "confidence":
            adaptive_conf,

        "risk":
            adaptive_risk,

        "severity_id":
            adaptive_severity,
    }


# ============================================================
# METRICS FROM CONFUSION MATRIX
# ============================================================

def classification_metrics_from_confusion(
    cm: np.ndarray,
    class_names: List[str],
):
    # Rows = true classes 0..5 plus optional anomalous row.
    # Columns = predictions 0..6.
    true_class_count = NUM_CLASSES

    total = float(
        np.sum(
            cm[
                :true_class_count,
                :
            ]
        )
    )

    correct = float(
        np.trace(
            cm[
                :true_class_count,
                :true_class_count
            ]
        )
    )

    accuracy = (
        correct
        /
        total
        if total > 0
        else 0.0
    )

    precisions = []
    recalls = []
    f1s = []
    supports = []

    per_class_rows = []

    for class_id in range(
        true_class_count
    ):
        tp = float(
            cm[
                class_id,
                class_id,
            ]
        )

        support = float(
            np.sum(
                cm[
                    class_id,
                    :
                ]
            )
        )

        predicted_count = float(
            np.sum(
                cm[
                    :true_class_count,
                    class_id,
                ]
            )
        )

        recall = (
            tp
            /
            support
            if support > 0
            else 0.0
        )

        precision = (
            tp
            /
            predicted_count
            if predicted_count > 0
            else 0.0
        )

        f1 = (
            2.0
            *
            precision
            *
            recall
            /
            (
                precision
                +
                recall
            )
            if (
                precision
                +
                recall
            ) > 0
            else 0.0
        )

        precisions.append(
            precision
        )

        recalls.append(
            recall
        )

        f1s.append(
            f1
        )

        supports.append(
            support
        )

        per_class_rows.append(
            {
                "class_id":
                    class_id,

                "class_name":
                    class_names[
                        class_id
                    ],

                "precision":
                    precision,

                "recall":
                    recall,

                "f1":
                    f1,

                "support":
                    int(
                        support
                    ),
            }
        )

    macro_precision = float(
        np.mean(
            precisions
        )
    )

    macro_recall = float(
        np.mean(
            recalls
        )
    )

    macro_f1 = float(
        np.mean(
            f1s
        )
    )

    balanced_accuracy = (
        macro_recall
    )

    supports_array = np.asarray(
        supports,
        dtype=np.float64,
    )

    f1_array = np.asarray(
        f1s,
        dtype=np.float64,
    )

    weighted_f1 = float(
        np.sum(
            supports_array
            *
            f1_array
        )
        /
        np.sum(
            supports_array
        )
    )

    anomalous_predictions = int(
        np.sum(
            cm[
                :true_class_count,
                ANOMALOUS_ID,
            ]
        )
    )

    metrics = {
        "accuracy":
            accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "macro_precision":
            macro_precision,

        "macro_recall":
            macro_recall,

        "macro_f1":
            macro_f1,

        "weighted_f1":
            weighted_f1,

        "anomalous_traffic_predictions":
            anomalous_predictions,
    }

    return (
        metrics,
        per_class_rows,
    )


def build_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
):
    labels = np.arange(
        NUM_CLASSES
        +
        1
    )

    return confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )


def save_confusion_csv(
    cm: np.ndarray,
    class_names: List[str],
    path: Path,
):
    names = (
        class_names
        +
        [
            ANOMALOUS_NAME
        ]
    )

    dataframe = pd.DataFrame(
        cm,
        index=[
            f"true_{name}"
            for name
            in names
        ],
        columns=[
            f"pred_{name}"
            for name
            in names
        ],
    )

    dataframe.to_csv(
        path
    )


# ============================================================
# PATH / SEVERITY / DISAGREEMENT AUDITS
# ============================================================

def gate_audit(
    *,
    y_true,
    rf_pred,
    rf_conf,
    adaptive_result,
):
    fast_mask = adaptive_result[
        "fast_mask"
    ]

    deep_mask = ~fast_mask

    n = len(
        y_true
    )

    fast_count = int(
        np.sum(
            fast_mask
        )
    )

    deep_count = int(
        np.sum(
            deep_mask
        )
    )

    fast_correct_mask = (
        fast_mask
        &
        (
            rf_pred
            ==
            y_true
        )
    )

    fast_wrong_mask = (
        fast_mask
        &
        (
            rf_pred
            !=
            y_true
        )
    )

    fast_correct = int(
        np.sum(
            fast_correct_mask
        )
    )

    fast_wrong = int(
        np.sum(
            fast_wrong_mask
        )
    )

    benign_false_fast = int(
        np.sum(
            fast_mask
            &
            (
                y_true
                ==
                0
            )
        )
    )

    wrong_attack_class_fast = int(
        np.sum(
            fast_mask
            &
            (
                y_true
                !=
                0
            )
            &
            (
                rf_pred
                !=
                y_true
            )
        )
    )

    deep_correct = int(
        np.sum(
            deep_mask
            &
            (
                adaptive_result[
                    "prediction"
                ]
                ==
                y_true
            )
        )
    )

    fast_precision = (
        fast_correct
        /
        fast_count
        if fast_count > 0
        else float("nan")
    )

    # "RF false-fast-path rate" is explicitly defined as the
    # proportion of gate-triggered fast decisions that are wrong.
    false_fast_path_rate = (
        fast_wrong
        /
        fast_count
        if fast_count > 0
        else float("nan")
    )

    benign_false_fast_rate = (
        benign_false_fast
        /
        fast_count
        if fast_count > 0
        else float("nan")
    )

    return {
        "validation_exposures":
            n,

        "fast_path_count":
            fast_count,

        "fast_path_rate":
            fast_count
            /
            n,

        "deep_path_count":
            deep_count,

        "deep_path_rate":
            deep_count
            /
            n,

        "fast_path_correct":
            fast_correct,

        "fast_path_wrong":
            fast_wrong,

        "fast_path_precision":
            fast_precision,

        "rf_false_fast_path_rate":
            false_fast_path_rate,

        "benign_false_fast_path_count":
            benign_false_fast,

        "benign_false_fast_path_rate":
            benign_false_fast_rate,

        "wrong_attack_class_fast_path_count":
            wrong_attack_class_fast,

        "deep_path_correct":
            deep_correct,

        "deep_path_accuracy":
            (
                deep_correct
                /
                deep_count
                if deep_count > 0
                else float("nan")
            ),

        "rf_fast_threshold":
            RF_FAST_ATTACK_THRESHOLD,
    }


def disagreement_audit(
    *,
    y_true,
    rf_pred,
    fed_pred,
    cnn_pred,
):
    attack_mask = (
        y_true
        !=
        0
    )

    def disagreement(
        left,
        right,
        mask=None,
    ):
        if mask is None:
            return float(
                np.mean(
                    left
                    !=
                    right
                )
            )

        if not np.any(
            mask
        ):
            return float(
                "nan"
            )

        return float(
            np.mean(
                left[
                    mask
                ]
                !=
                right[
                    mask
                ]
            )
        )

    all_three_agree = (
        (
            rf_pred
            ==
            fed_pred
        )
        &
        (
            fed_pred
            ==
            cnn_pred
        )
    )

    return {
        "rf_vs_fedprox70_disagreement":
            disagreement(
                rf_pred,
                fed_pred,
            ),

        "rf_vs_cnn_disagreement":
            disagreement(
                rf_pred,
                cnn_pred,
            ),

        "fedprox70_vs_cnn_disagreement":
            disagreement(
                fed_pred,
                cnn_pred,
            ),

        "rf_vs_fedprox70_attack_only_disagreement":
            disagreement(
                rf_pred,
                fed_pred,
                attack_mask,
            ),

        "rf_vs_cnn_attack_only_disagreement":
            disagreement(
                rf_pred,
                cnn_pred,
                attack_mask,
            ),

        "fedprox70_vs_cnn_attack_only_disagreement":
            disagreement(
                fed_pred,
                cnn_pred,
                attack_mask,
            ),

        "all_three_agreement_rate":
            float(
                np.mean(
                    all_three_agree
                )
            ),

        "all_three_attack_only_agreement_rate":
            (
                float(
                    np.mean(
                        all_three_agree[
                            attack_mask
                        ]
                    )
                )
                if np.any(
                    attack_mask
                )
                else float(
                    "nan"
                )
            ),
    }


def severity_distribution_rows(
    client_id: int,
    adaptive_severity: np.ndarray,
    full_severity: np.ndarray,
):
    rows = []

    for policy_name, severity_ids in (
        (
            "adaptive_gate",
            adaptive_severity,
        ),
        (
            "always_run_all",
            full_severity,
        ),
    ):
        total = len(
            severity_ids
        )

        for severity_id, severity_name in enumerate(
            SEVERITY_NAMES
        ):
            count = int(
                np.sum(
                    severity_ids
                    ==
                    severity_id
                )
            )

            rows.append(
                {
                    "client_id":
                        client_id,

                    "policy":
                        policy_name,

                    "severity":
                        severity_name,

                    "count":
                        count,

                    "rate":
                        count
                        /
                        total,
                }
            )

    return rows


# ============================================================
# LATENCY PROFILING
# ============================================================

def stratified_latency_indices(
    y: np.ndarray,
    requested: int,
    seed: int,
):
    requested = int(
        requested
    )

    if requested <= 0:
        return np.asarray(
            [],
            dtype=np.int64,
        )

    requested = min(
        requested,
        len(
            y
        ),
    )

    rng = np.random.default_rng(
        seed
    )

    per_class = max(
        1,
        requested
        //
        NUM_CLASSES,
    )

    selected = []

    for class_id in range(
        NUM_CLASSES
    ):
        class_indices = np.flatnonzero(
            y
            ==
            class_id
        )

        take = min(
            per_class,
            len(
                class_indices
            ),
        )

        if take > 0:
            sampled = rng.choice(
                class_indices,
                size=take,
                replace=False,
            )

            selected.extend(
                sampled.tolist()
            )

    if len(
        selected
    ) < requested:
        existing = set(
            selected
        )

        remaining = np.asarray(
            [
                index
                for index
                in range(
                    len(
                        y
                    )
                )
                if index not in existing
            ],
            dtype=np.int64,
        )

        extra_count = min(
            requested
            -
            len(
                selected
            ),
            len(
                remaining
            ),
        )

        if extra_count > 0:
            extra = rng.choice(
                remaining,
                size=extra_count,
                replace=False,
            )

            selected.extend(
                extra.tolist()
            )

    return np.asarray(
        selected[
            :requested
        ],
        dtype=np.int64,
    )


def profile_latency(
    *,
    client_id: int,
    indices: np.ndarray,
    sequences: np.ndarray,
    flows: np.ndarray,
    y_true: np.ndarray,
    rf,
    ae,
    fedprox70,
    cnn,
):
    rows = []

    if len(
        indices
    ) == 0:
        return rows

    # Warm-up each model path once before timing.
    warm_index = int(
        indices[
            0
        ]
    )

    phase14g1.run_all_models_once(
        flow=
            flows[
                warm_index
            ],

        sequence=
            sequences[
                warm_index
            ],

        rf=
            rf,

        ae=
            ae,

        fedprox70=
            fedprox70,

        cnn=
            cnn,
    )

    phase14g2.detect_with_phase14g2_policy(
        client_id=
            client_id,

        current_flow_70=
            flows[
                warm_index
            ],

        sequence_20x70=
            sequences[
                warm_index
            ],

        rf=
            rf,

        ae=
            ae,

        fedprox70=
            fedprox70,

        cnn=
            cnn,
    )

    for ordinal, index in enumerate(
        indices
    ):
        index = int(
            index
        )

        flow = flows[
            index
        ]

        sequence = sequences[
            index
        ]

        # Always-run-all path.
        full_start = time.perf_counter()

        full_result = phase14g1.run_all_models_once(
            flow=
                flow,

            sequence=
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

        full_total_ms = (
            time.perf_counter()
            -
            full_start
        ) * 1000.0

        # Frozen adaptive Phase14G2 path.
        adaptive_start = time.perf_counter()

        adaptive_alert = phase14g2.detect_with_phase14g2_policy(
            client_id=
                client_id,

            current_flow_70=
                flow,

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

        savings_ms = (
            full_total_ms
            -
            adaptive_ms
        )

        savings_percent = (
            (
                savings_ms
                /
                full_total_ms
            )
            *
            100.0
            if full_total_ms > 0
            else float(
                "nan"
            )
        )

        rows.append(
            {
                "client_id":
                    client_id,

                "profile_ordinal":
                    ordinal,

                "validation_index":
                    index,

                "true_class_id":
                    int(
                        y_true[
                            index
                        ]
                    ),

                "processing_path":
                    adaptive_alert.processing_path,

                "always_run_all_ms":
                    float(
                        full_total_ms
                    ),

                "adaptive_ms":
                    float(
                        adaptive_ms
                    ),

                "latency_savings_ms":
                    float(
                        savings_ms
                    ),

                "latency_savings_percent":
                    float(
                        savings_percent
                    ),

                "rf_ms":
                    float(
                        full_result[
                            "timings_ms"
                        ][
                            "random_forest"
                        ]
                    ),

                "ae_ms":
                    float(
                        full_result[
                            "timings_ms"
                        ][
                            "autoencoder"
                        ]
                    ),

                "fedprox70_ms":
                    float(
                        full_result[
                            "timings_ms"
                        ][
                            "fedprox70"
                        ]
                    ),

                "cnn_bilstm_ms":
                    float(
                        full_result[
                            "timings_ms"
                        ][
                            "cnn_bilstm"
                        ]
                    ),

                "risk_fusion_ms":
                    float(
                        full_result[
                            "timings_ms"
                        ][
                            "risk_fusion"
                        ]
                    ),
            }
        )

    return rows


def latency_summary_rows(
    raw_latency: pd.DataFrame,
):
    rows = []

    if raw_latency.empty:
        return rows

    for client_id in sorted(
        raw_latency[
            "client_id"
        ].unique()
    ):
        client_df = raw_latency[
            raw_latency[
                "client_id"
            ]
            ==
            client_id
        ]

        for path_name in (
            "ALL",
            "FAST_ATTACK_PATH",
            "DEEP_ANALYSIS",
        ):
            if path_name == "ALL":
                subset = (
                    client_df
                )
            else:
                subset = client_df[
                    client_df[
                        "processing_path"
                    ]
                    ==
                    path_name
                ]

            if subset.empty:
                continue

            always_values = subset[
                "always_run_all_ms"
            ].to_numpy(
                dtype=np.float64
            )

            adaptive_values = subset[
                "adaptive_ms"
            ].to_numpy(
                dtype=np.float64
            )

            savings_values = subset[
                "latency_savings_percent"
            ].to_numpy(
                dtype=np.float64
            )

            rows.append(
                {
                    "client_id":
                        int(
                            client_id
                        ),

                    "path":
                        path_name,

                    "samples":
                        len(
                            subset
                        ),

                    "always_mean_ms":
                        float(
                            np.mean(
                                always_values
                            )
                        ),

                    "always_median_ms":
                        float(
                            np.median(
                                always_values
                            )
                        ),

                    "always_p95_ms":
                        percentile(
                            always_values,
                            95,
                        ),

                    "always_p99_ms":
                        percentile(
                            always_values,
                            99,
                        ),

                    "adaptive_mean_ms":
                        float(
                            np.mean(
                                adaptive_values
                            )
                        ),

                    "adaptive_median_ms":
                        float(
                            np.median(
                                adaptive_values
                            )
                        ),

                    "adaptive_p95_ms":
                        percentile(
                            adaptive_values,
                            95,
                        ),

                    "adaptive_p99_ms":
                        percentile(
                            adaptive_values,
                            99,
                        ),

                    "mean_latency_savings_percent":
                        float(
                            np.mean(
                                savings_values
                            )
                        ),

                    "median_latency_savings_percent":
                        float(
                            np.median(
                                savings_values
                            )
                        ),
                }
            )

    return rows


# ============================================================
# CLI
# ============================================================

def parse_clients(
    value: str,
):
    value = value.strip()

    if value.lower() == "all":
        return [
            1,
            2,
            3,
            4,
            5,
        ]

    clients = []

    for item in value.split(
        ","
    ):
        client_id = int(
            item.strip()
        )

        if client_id not in (
            1,
            2,
            3,
            4,
            5,
        ):
            raise ValueError(
                "--clients must contain only 1..5."
            )

        if client_id not in clients:
            clients.append(
                client_id
            )

    if not clients:
        raise ValueError(
            "--clients cannot be empty."
        )

    return clients


def parse_args():
    parser = argparse.ArgumentParser(
        description=PHASE_NAME
    )

    parser.add_argument(
        "--clients",
        type=str,
        default="all",
        help=(
            "all or comma-separated client ids, e.g. 1,3,5"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=
            DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--latency-samples",
        type=int,
        default=
            DEFAULT_LATENCY_SAMPLES_PER_CLIENT,
        help=(
            "Per-client class-stratified per-event latency profiling "
            "samples. Classification metrics still use the full "
            "validation set. Set 0 to skip latency profiling."
        ),
    )

    parser.add_argument(
        "--max-validation-samples",
        type=int,
        default=None,
        help=(
            "DEBUG ONLY. If omitted, the full validation set is used."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    set_seed(
        SEED
    )

    clients = parse_clients(
        args.clients
    )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be > 0."
        )

    if args.latency_samples < 0:
        raise ValueError(
            "--latency-samples must be >= 0."
        )

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARTIFACT_ROOT.mkdir(
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

    class_names = [
        inverse_mapping[
            class_id
        ]
        for class_id
        in range(
            NUM_CLASSES
        )
    ]

    (
        sequences,
        flows,
        y_true,
    ) = load_validation_sequences(
        max_samples=
            args.max_validation_samples
    )

    full_validation = (
        args.max_validation_samples
        is None
    )

    separator()

    print(
        PHASE_NAME
    )

    separator()

    print(
        f"Device                        : {device}"
    )

    print(
        f"Seed                          : {SEED}"
    )

    print(
        f"Clients                       : {clients}"
    )

    print(
        f"Validation sequences          : {len(y_true):,}"
    )

    print(
        f"Sequence shape                : {sequences.shape}"
    )

    print(
        f"Current flow shape            : {flows.shape}"
    )

    print(
        f"Full validation               : "
        f"{'YES' if full_validation else 'NO — DEBUG SUBSET'}"
    )

    print(
        f"Batch size                    : {args.batch_size}"
    )

    print(
        f"Latency samples/client        : {args.latency_samples}"
    )

    print(
        f"RF fast threshold             : "
        f"{RF_FAST_ATTACK_THRESHOLD}"
    )

    print(
        f"Risk policy                   : {POLICY_VERSION}"
    )

    print(
        "Fast-path severity            : HIGH"
    )

    print(
        "CRITICAL policy               : full-fusion only"
    )

    print(
        "Locked test                    : NO"
    )

    print()
    print("VALIDATION CLASS DISTRIBUTION")
    print("-" * 126)

    for class_id in range(
        NUM_CLASSES
    ):
        count = int(
            np.sum(
                y_true
                ==
                class_id
            )
        )

        print(
            f"{class_id} | "
            f"{class_names[class_id]:18s} | "
            f"{count:,}"
        )

    print()
    print("EVALUATION DESIGN")
    print("-" * 126)

    print(
        "The same validation set is replayed independently through "
        "each selected client's LOCAL RF + LOCAL AE stack."
    )

    print(
        "Therefore pooled results are CLIENT-EXPOSURE metrics, not "
        "a larger set of independent flows."
    )

    print(
        "No sample is assigned to a client using its true label."
    )

    # ========================================================
    # LOAD GLOBAL FEDPROX70 + CNN ONCE
    # ========================================================

    print()
    print("LOADING GLOBAL MODELS")
    print("-" * 126)

    fedprox70 = phase14g1.RealFedProx70Predictor(
        checkpoint_path=
            phase14g1.FEDPROX70_CHECKPOINT,

        inverse_mapping=
            inverse_mapping,

        device=
            device,
    )

    cnn = phase14g1.build_real_cnn_predictor(
        inverse_mapping=
            inverse_mapping,

        device=
            device,
    )

    print(
        f"FedProx70                     : "
        f"{fedprox70.path}"
    )

    print(
        f"CNN-BiLSTM                    : "
        f"{phase14g1.CNN_CHECKPOINT}"
    )

    print(
        f"CNN frozen round              : "
        f"{cnn.checkpoint_round}"
    )

    global_start = time.perf_counter()

    (
        fed_pred,
        fed_conf,
    ) = predict_fedprox70_batch(
        predictor=
            fedprox70,

        flows=
            flows,

        batch_size=
            args.batch_size,

        device=
            device,
    )

    (
        cnn_pred,
        cnn_conf,
    ) = predict_cnn_batch(
        predictor=
            cnn,

        sequences=
            sequences,

        batch_size=
            args.batch_size,

        device=
            device,
    )

    global_inference_seconds = (
        time.perf_counter()
        -
        global_start
    )

    print(
        f"Global batched inference       : "
        f"{global_inference_seconds:.2f}s"
    )

    # ========================================================
    # ACCUMULATORS
    # ========================================================

    client_summary_rows = []
    per_class_rows = []
    gate_rows = []
    severity_rows = []
    disagreement_rows = []
    latency_rows = []

    pooled_adaptive_cm = np.zeros(
        (
            NUM_CLASSES
            +
            1,
            NUM_CLASSES
            +
            1,
        ),
        dtype=np.int64,
    )

    pooled_full_cm = np.zeros_like(
        pooled_adaptive_cm
    )

    total_start = time.perf_counter()

    # ========================================================
    # EACH CLIENT
    # ========================================================

    for client_id in clients:
        separator()

        print(
            f"CLIENT {client_id} — "
            f"{phase14g1.CLIENT_ATTACKS[client_id]}"
        )

        print("-" * 126)

        client_start = time.perf_counter()

        rf = phase14g1.RealRandomForestPredictor(
            client_id=
                client_id,

            label_mapping=
                label_mapping,

            inverse_mapping=
                inverse_mapping,
        )

        ae = phase14g1.RealPersonalizedAutoencoderPredictor(
            client_id=
                client_id,

            device=
                device,

            explicit_model_path=
                None,
        )

        print(
            f"RF                            : {rf.path}"
        )

        print(
            f"AE                            : {ae.path}"
        )

        print(
            f"AE threshold                  : {ae.threshold:.6f}"
        )

        # ----------------------------------------------------
        # BATCHED LOCAL MODELS
        # ----------------------------------------------------

        local_inference_start = time.perf_counter()

        (
            rf_pred,
            rf_conf,
        ) = predict_rf_batch(
            rf=
                rf,

            flows=
                flows,
        )

        (
            ae_error,
            ae_score,
            ae_anomaly,
        ) = predict_ae_batch(
            ae=
                ae,

            flows=
                flows,

            batch_size=
                args.batch_size,

            device=
                device,
        )

        local_batched_seconds = (
            time.perf_counter()
            -
            local_inference_start
        )

        # ----------------------------------------------------
        # ALWAYS-RUN-ALL FULL FUSION
        # ----------------------------------------------------

        full_result = full_fusion_vectorized(
            rf_pred=
                rf_pred,

            rf_conf=
                rf_conf,

            ae_score=
                ae_score,

            fed_pred=
                fed_pred,

            fed_conf=
                fed_conf,

            cnn_pred=
                cnn_pred,

            cnn_conf=
                cnn_conf,
        )

        # ----------------------------------------------------
        # ADAPTIVE PHASE14G2 POLICY
        # ----------------------------------------------------

        adaptive_result = adaptive_policy_vectorized(
            rf_pred=
                rf_pred,

            rf_conf=
                rf_conf,

            full_result=
                full_result,
        )

        # ----------------------------------------------------
        # CLASSIFICATION METRICS
        # ----------------------------------------------------

        adaptive_cm = build_confusion(
            y_true=
                y_true,

            y_pred=
                adaptive_result[
                    "prediction"
                ],
        )

        full_cm = build_confusion(
            y_true=
                y_true,

            y_pred=
                full_result[
                    "prediction"
                ],
        )

        pooled_adaptive_cm += (
            adaptive_cm
        )

        pooled_full_cm += (
            full_cm
        )

        (
            adaptive_metrics,
            adaptive_per_class,
        ) = classification_metrics_from_confusion(
            cm=
                adaptive_cm,

            class_names=
                class_names,
        )

        (
            full_metrics,
            full_per_class,
        ) = classification_metrics_from_confusion(
            cm=
                full_cm,

            class_names=
                class_names,
        )

        # ----------------------------------------------------
        # PATH AUDIT
        # ----------------------------------------------------

        gate = gate_audit(
            y_true=
                y_true,

            rf_pred=
                rf_pred,

            rf_conf=
                rf_conf,

            adaptive_result=
                adaptive_result,
        )

        gate[
            "client_id"
        ] = client_id

        gate[
            "seen_attack"
        ] = (
            phase14g1.CLIENT_ATTACKS[
                client_id
            ]
        )

        gate_rows.append(
            gate
        )

        # ----------------------------------------------------
        # DISAGREEMENT
        # ----------------------------------------------------

        disagreement = disagreement_audit(
            y_true=
                y_true,

            rf_pred=
                rf_pred,

            fed_pred=
                fed_pred,

            cnn_pred=
                cnn_pred,
        )

        disagreement[
            "client_id"
        ] = client_id

        disagreement[
            "seen_attack"
        ] = (
            phase14g1.CLIENT_ATTACKS[
                client_id
            ]
        )

        disagreement_rows.append(
            disagreement
        )

        # ----------------------------------------------------
        # SEVERITY
        # ----------------------------------------------------

        severity_rows.extend(
            severity_distribution_rows(
                client_id=
                    client_id,

                adaptive_severity=
                    adaptive_result[
                        "severity_id"
                    ],

                full_severity=
                    full_result[
                        "severity_id"
                    ],
            )
        )

        # ----------------------------------------------------
        # PER CLASS
        # ----------------------------------------------------

        for policy, rows in (
            (
                "adaptive_gate",
                adaptive_per_class,
            ),
            (
                "always_run_all",
                full_per_class,
            ),
        ):
            for row in rows:
                per_class_rows.append(
                    {
                        "client_id":
                            client_id,

                        "seen_attack":
                            phase14g1.CLIENT_ATTACKS[
                                client_id
                            ],

                        "policy":
                            policy,

                        **row,
                    }
                )

        # ----------------------------------------------------
        # SAVE CONFUSION MATRICES
        # ----------------------------------------------------

        save_confusion_csv(
            cm=
                adaptive_cm,

            class_names=
                class_names,

            path=
                (
                    RESULT_ROOT
                    /
                    f"phase14h_client_{client_id}_adaptive_confusion.csv"
                ),
        )

        save_confusion_csv(
            cm=
                full_cm,

            class_names=
                class_names,

            path=
                (
                    RESULT_ROOT
                    /
                    f"phase14h_client_{client_id}_always_run_all_confusion.csv"
                ),
        )

        # ----------------------------------------------------
        # SAVE COMPACT FULL PREDICTION EVIDENCE
        # ----------------------------------------------------

        np.savez_compressed(
            (
                RESULT_ROOT
                /
                f"phase14h_client_{client_id}_predictions.npz"
            ),

            y_true=
                y_true,

            rf_pred=
                rf_pred,

            rf_conf=
                rf_conf,

            ae_error=
                ae_error,

            ae_score=
                ae_score,

            ae_anomaly=
                ae_anomaly,

            fedprox70_pred=
                fed_pred,

            fedprox70_conf=
                fed_conf,

            cnn_bilstm_pred=
                cnn_pred,

            cnn_bilstm_conf=
                cnn_conf,

            full_fusion_pred=
                full_result[
                    "prediction"
                ],

            full_fusion_conf=
                full_result[
                    "confidence"
                ],

            full_fusion_risk=
                full_result[
                    "risk"
                ],

            full_fusion_severity_id=
                full_result[
                    "severity_id"
                ],

            adaptive_pred=
                adaptive_result[
                    "prediction"
                ],

            adaptive_conf=
                adaptive_result[
                    "confidence"
                ],

            adaptive_risk=
                adaptive_result[
                    "risk"
                ],

            adaptive_severity_id=
                adaptive_result[
                    "severity_id"
                ],

            fast_path_mask=
                adaptive_result[
                    "fast_mask"
                ],
        )

        # ----------------------------------------------------
        # LATENCY PROFILE
        # ----------------------------------------------------

        if args.latency_samples > 0:
            latency_indices = stratified_latency_indices(
                y=
                    y_true,

                requested=
                    args.latency_samples,

                seed=
                    SEED
                    +
                    client_id
                    *
                    1000,
            )

            print(
                f"Profiling per-event latency on "
                f"{len(latency_indices)} class-stratified samples..."
            )

            client_latency_rows = profile_latency(
                client_id=
                    client_id,

                indices=
                    latency_indices,

                sequences=
                    sequences,

                flows=
                    flows,

                y_true=
                    y_true,

                rf=
                    rf,

                ae=
                    ae,

                fedprox70=
                    fedprox70,

                cnn=
                    cnn,
            )

            latency_rows.extend(
                client_latency_rows
            )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        client_seconds = (
            time.perf_counter()
            -
            client_start
        )

        client_summary_rows.append(
            {
                "client_id":
                    client_id,

                "seen_attack":
                    phase14g1.CLIENT_ATTACKS[
                        client_id
                    ],

                "validation_exposures":
                    len(
                        y_true
                    ),

                "adaptive_accuracy":
                    adaptive_metrics[
                        "accuracy"
                    ],

                "adaptive_balanced_accuracy":
                    adaptive_metrics[
                        "balanced_accuracy"
                    ],

                "adaptive_macro_precision":
                    adaptive_metrics[
                        "macro_precision"
                    ],

                "adaptive_macro_recall":
                    adaptive_metrics[
                        "macro_recall"
                    ],

                "adaptive_macro_f1":
                    adaptive_metrics[
                        "macro_f1"
                    ],

                "adaptive_weighted_f1":
                    adaptive_metrics[
                        "weighted_f1"
                    ],

                "always_accuracy":
                    full_metrics[
                        "accuracy"
                    ],

                "always_balanced_accuracy":
                    full_metrics[
                        "balanced_accuracy"
                    ],

                "always_macro_precision":
                    full_metrics[
                        "macro_precision"
                    ],

                "always_macro_recall":
                    full_metrics[
                        "macro_recall"
                    ],

                "always_macro_f1":
                    full_metrics[
                        "macro_f1"
                    ],

                "always_weighted_f1":
                    full_metrics[
                        "weighted_f1"
                    ],

                "adaptive_minus_always_accuracy":
                    (
                        adaptive_metrics[
                            "accuracy"
                        ]
                        -
                        full_metrics[
                            "accuracy"
                        ]
                    ),

                "adaptive_minus_always_balanced_accuracy":
                    (
                        adaptive_metrics[
                            "balanced_accuracy"
                        ]
                        -
                        full_metrics[
                            "balanced_accuracy"
                        ]
                    ),

                "adaptive_minus_always_macro_f1":
                    (
                        adaptive_metrics[
                            "macro_f1"
                        ]
                        -
                        full_metrics[
                            "macro_f1"
                        ]
                    ),

                "fast_path_rate":
                    gate[
                        "fast_path_rate"
                    ],

                "deep_path_rate":
                    gate[
                        "deep_path_rate"
                    ],

                "rf_false_fast_path_rate":
                    gate[
                        "rf_false_fast_path_rate"
                    ],

                "benign_false_fast_path_rate":
                    gate[
                        "benign_false_fast_path_rate"
                    ],

                "local_batched_inference_seconds":
                    local_batched_seconds,

                "client_total_seconds":
                    client_seconds,
            }
        )

        print()
        print("ADAPTIVE GATE")
        print(
            f"Accuracy / BA / MacroF1      : "
            f"{adaptive_metrics['accuracy']:.6f} / "
            f"{adaptive_metrics['balanced_accuracy']:.6f} / "
            f"{adaptive_metrics['macro_f1']:.6f}"
        )

        print(
            f"Weighted F1                   : "
            f"{adaptive_metrics['weighted_f1']:.6f}"
        )

        print()
        print("ALWAYS-RUN-ALL")
        print(
            f"Accuracy / BA / MacroF1      : "
            f"{full_metrics['accuracy']:.6f} / "
            f"{full_metrics['balanced_accuracy']:.6f} / "
            f"{full_metrics['macro_f1']:.6f}"
        )

        print(
            f"Weighted F1                   : "
            f"{full_metrics['weighted_f1']:.6f}"
        )

        print()
        print("GATE AUDIT")
        print(
            f"Fast path rate                : "
            f"{gate['fast_path_rate']:.4%}"
        )

        print(
            f"Deep path rate                : "
            f"{gate['deep_path_rate']:.4%}"
        )

        print(
            f"Fast correct / wrong          : "
            f"{gate['fast_path_correct']:,} / "
            f"{gate['fast_path_wrong']:,}"
        )

        print(
            f"RF false-fast-path rate       : "
            f"{gate['rf_false_fast_path_rate']:.4%}"
        )

        print(
            f"Benign false-fast-path rate   : "
            f"{gate['benign_false_fast_path_rate']:.4%}"
        )

        print(
            f"Client total time             : "
            f"{client_seconds:.2f}s"
        )

        del rf
        del ae
        del rf_pred
        del rf_conf
        del ae_error
        del ae_score
        del ae_anomaly
        del full_result
        del adaptive_result

        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ========================================================
    # SAVE TABLES
    # ========================================================

    client_summary_df = pd.DataFrame(
        client_summary_rows
    )

    client_summary_df.to_csv(
        CLIENT_SUMMARY_FILE,
        index=False,
    )

    per_class_df = pd.DataFrame(
        per_class_rows
    )

    per_class_df.to_csv(
        PER_CLASS_FILE,
        index=False,
    )

    gate_df = pd.DataFrame(
        gate_rows
    )

    gate_df.to_csv(
        PATH_AUDIT_FILE,
        index=False,
    )

    severity_df = pd.DataFrame(
        severity_rows
    )

    severity_df.to_csv(
        SEVERITY_FILE,
        index=False,
    )

    disagreement_df = pd.DataFrame(
        disagreement_rows
    )

    disagreement_df.to_csv(
        DISAGREEMENT_FILE,
        index=False,
    )

    # ========================================================
    # MEAN CLIENT METRICS
    # ========================================================

    numeric_columns = [
        column
        for column
        in client_summary_df.columns
        if pd.api.types.is_numeric_dtype(
            client_summary_df[
                column
            ]
        )
        and
        column
        not in {
            "client_id",
            "validation_exposures",
        }
    ]

    mean_rows = []

    for column in numeric_columns:
        values = client_summary_df[
            column
        ].to_numpy(
            dtype=np.float64
        )

        mean_rows.append(
            {
                "metric":
                    column,

                "mean":
                    float(
                        np.nanmean(
                            values
                        )
                    ),

                "std_population":
                    float(
                        np.nanstd(
                            values,
                            ddof=0,
                        )
                    ),

                "min":
                    float(
                        np.nanmin(
                            values
                        )
                    ),

                "max":
                    float(
                        np.nanmax(
                            values
                        )
                    ),
            }
        )

    mean_client_df = pd.DataFrame(
        mean_rows
    )

    mean_client_df.to_csv(
        MEAN_CLIENT_METRICS_FILE,
        index=False,
    )

    # ========================================================
    # POOLED CLIENT-EXPOSURE METRICS
    # ========================================================

    (
        pooled_adaptive_metrics,
        pooled_adaptive_per_class,
    ) = classification_metrics_from_confusion(
        pooled_adaptive_cm,
        class_names,
    )

    (
        pooled_full_metrics,
        pooled_full_per_class,
    ) = classification_metrics_from_confusion(
        pooled_full_cm,
        class_names,
    )

    pooled_exposure_count = (
        len(
            y_true
        )
        *
        len(
            clients
        )
    )

    pooled_payload = {
        "important_interpretation":
            (
                "These are pooled CLIENT-EXPOSURE metrics. "
                "The same validation sequences are replayed at "
                "multiple client-local stacks, so exposures are not "
                "independent unique flows."
            ),

        "selected_clients":
            clients,

        "unique_validation_sequences":
            len(
                y_true
            ),

        "pooled_client_exposures":
            pooled_exposure_count,

        "adaptive_gate":
            pooled_adaptive_metrics,

        "always_run_all":
            pooled_full_metrics,

        "adaptive_minus_always":
            {
                "accuracy":
                    (
                        pooled_adaptive_metrics[
                            "accuracy"
                        ]
                        -
                        pooled_full_metrics[
                            "accuracy"
                        ]
                    ),

                "balanced_accuracy":
                    (
                        pooled_adaptive_metrics[
                            "balanced_accuracy"
                        ]
                        -
                        pooled_full_metrics[
                            "balanced_accuracy"
                        ]
                    ),

                "macro_f1":
                    (
                        pooled_adaptive_metrics[
                            "macro_f1"
                        ]
                        -
                        pooled_full_metrics[
                            "macro_f1"
                        ]
                    ),

                "weighted_f1":
                    (
                        pooled_adaptive_metrics[
                            "weighted_f1"
                        ]
                        -
                        pooled_full_metrics[
                            "weighted_f1"
                        ]
                    ),
            },
    }

    save_json(
        POOLED_EXPOSURE_METRICS_FILE,
        pooled_payload,
    )

    save_confusion_csv(
        pooled_adaptive_cm,
        class_names,
        (
            RESULT_ROOT
            /
            "phase14h_pooled_client_exposure_adaptive_confusion.csv"
        ),
    )

    save_confusion_csv(
        pooled_full_cm,
        class_names,
        (
            RESULT_ROOT
            /
            "phase14h_pooled_client_exposure_always_confusion.csv"
        ),
    )

    # ========================================================
    # LATENCY
    # ========================================================

    latency_raw_df = pd.DataFrame(
        latency_rows
    )

    if not latency_raw_df.empty:
        latency_raw_df.to_csv(
            LATENCY_RAW_FILE,
            index=False,
        )

        latency_summary = pd.DataFrame(
            latency_summary_rows(
                latency_raw_df
            )
        )

        latency_summary.to_csv(
            LATENCY_SUMMARY_FILE,
            index=False,
        )

    else:
        latency_summary = pd.DataFrame()

    # ========================================================
    # FINAL CONFIG / SUMMARY
    # ========================================================

    total_seconds = (
        time.perf_counter()
        -
        total_start
    )

    config_payload = {
        "phase":
            "14H",

        "policy":
            POLICY_VERSION,

        "seed":
            SEED,

        "device":
            str(
                device
            ),

        "clients":
            clients,

        "validation_sequence_file":
            VALIDATION_SEQUENCE_FILE,

        "unique_validation_sequences":
            len(
                y_true
            ),

        "full_validation":
            full_validation,

        "max_validation_samples":
            args.max_validation_samples,

        "batch_size":
            args.batch_size,

        "latency_samples_per_client":
            args.latency_samples,

        "feature_count":
            EXPECTED_FEATURES,

        "feature_names":
            active_features,

        "sequence_length":
            SEQUENCE_LENGTH,

        "rf_fast_attack_threshold":
            RF_FAST_ATTACK_THRESHOLD,

        "fast_path_severity":
            "HIGH",

        "critical_requires_full_fusion":
            True,

        "risk_weights":
            RISK_WEIGHTS,

        "autoencoder_score_semantics":
            (
                "threshold-relative engineering fusion score; "
                "not a calibrated probability"
            ),

        "client_replay_design":
            (
                "Same validation set replayed independently at each "
                "selected client-local RF/AE stack. Pooled metrics "
                "are client-exposure metrics."
            ),

        "oversampling":
            False,

        "synthetic_data":
            False,

        "locked_test_used":
            False,

        "scientific_label":
            (
                "SOURCE-ORDERED CNN-BiLSTM sequences; "
                "NOT timestamp-confirmed temporal sequences."
            ),
    }

    save_json(
        CONFIG_FILE,
        config_payload,
    )

    final_payload = {
        "phase":
            "14H",

        "policy":
            POLICY_VERSION,

        "selected_clients":
            clients,

        "unique_validation_sequences":
            len(
                y_true
            ),

        "pooled_client_exposures":
            pooled_exposure_count,

        "full_validation":
            full_validation,

        "pooled_adaptive_gate_metrics":
            pooled_adaptive_metrics,

        "pooled_always_run_all_metrics":
            pooled_full_metrics,

        "pooled_adaptive_minus_always":
            pooled_payload[
                "adaptive_minus_always"
            ],

        "mean_fast_path_rate":
            float(
                np.nanmean(
                    gate_df[
                        "fast_path_rate"
                    ].to_numpy(
                        dtype=np.float64
                    )
                )
            ),

        "mean_deep_path_rate":
            float(
                np.nanmean(
                    gate_df[
                        "deep_path_rate"
                    ].to_numpy(
                        dtype=np.float64
                    )
                )
            ),

        "mean_rf_false_fast_path_rate":
            float(
                np.nanmean(
                    gate_df[
                        "rf_false_fast_path_rate"
                    ].to_numpy(
                        dtype=np.float64
                    )
                )
            ),

        "mean_benign_false_fast_path_rate":
            float(
                np.nanmean(
                    gate_df[
                        "benign_false_fast_path_rate"
                    ].to_numpy(
                        dtype=np.float64
                    )
                )
            ),

        "global_model_batched_inference_seconds":
            global_inference_seconds,

        "phase_total_seconds":
            total_seconds,

        "latency_profiled":
            not latency_raw_df.empty,

        "latency_samples_total":
            len(
                latency_raw_df
            ),

        "locked_test_used":
            False,

        "interpretation_warning":
            (
                "Pooled results are repeated client-exposures, "
                "not independent unique validation flows."
            ),
    }

    if not latency_summary.empty:
        overall_latency = latency_summary[
            latency_summary[
                "path"
            ]
            ==
            "ALL"
        ]

        if not overall_latency.empty:
            final_payload[
                "mean_across_clients_latency"
            ] = {
                "always_mean_ms":
                    float(
                        overall_latency[
                            "always_mean_ms"
                        ].mean()
                    ),

                "adaptive_mean_ms":
                    float(
                        overall_latency[
                            "adaptive_mean_ms"
                        ].mean()
                    ),

                "mean_latency_savings_percent":
                    float(
                        overall_latency[
                            "mean_latency_savings_percent"
                        ].mean()
                    ),
            }

    save_json(
        FINAL_SUMMARY_FILE,
        final_payload,
    )

    # ========================================================
    # FINAL CONSOLE SUMMARY
    # ========================================================

    separator()

    print(
        "PHASE 14H COMPLETE"
    )

    separator()

    print(
        f"Unique validation sequences     : "
        f"{len(y_true):,}"
    )

    print(
        f"Clients evaluated               : "
        f"{len(clients)}"
    )

    print(
        f"Pooled client exposures         : "
        f"{pooled_exposure_count:,}"
    )

    print()

    print("POOLED CLIENT-EXPOSURE METRICS")
    print("-" * 126)

    print(
        "NOTE: these exposures are not independent unique flows."
    )

    print()

    print("Adaptive gate:")
    print(
        f"  Accuracy                      : "
        f"{pooled_adaptive_metrics['accuracy']:.6f}"
    )

    print(
        f"  Balanced Accuracy             : "
        f"{pooled_adaptive_metrics['balanced_accuracy']:.6f}"
    )

    print(
        f"  Macro F1                      : "
        f"{pooled_adaptive_metrics['macro_f1']:.6f}"
    )

    print(
        f"  Weighted F1                   : "
        f"{pooled_adaptive_metrics['weighted_f1']:.6f}"
    )

    print()

    print("Always-run-all:")
    print(
        f"  Accuracy                      : "
        f"{pooled_full_metrics['accuracy']:.6f}"
    )

    print(
        f"  Balanced Accuracy             : "
        f"{pooled_full_metrics['balanced_accuracy']:.6f}"
    )

    print(
        f"  Macro F1                      : "
        f"{pooled_full_metrics['macro_f1']:.6f}"
    )

    print(
        f"  Weighted F1                   : "
        f"{pooled_full_metrics['weighted_f1']:.6f}"
    )

    print()

    print(
        f"Adaptive - Always Macro F1     : "
        f"{pooled_payload['adaptive_minus_always']['macro_f1']:+.6f}"
    )

    print(
        f"Mean fast-path rate            : "
        f"{final_payload['mean_fast_path_rate']:.4%}"
    )

    print(
        f"Mean deep-path rate            : "
        f"{final_payload['mean_deep_path_rate']:.4%}"
    )

    print(
        f"Mean RF false-fast-path rate   : "
        f"{final_payload['mean_rf_false_fast_path_rate']:.4%}"
    )

    if (
        "mean_across_clients_latency"
        in final_payload
    ):
        latency_final = final_payload[
            "mean_across_clients_latency"
        ]

        print()

        print("LATENCY PROFILE")
        print("-" * 126)

        print(
            f"Always-run-all mean             : "
            f"{latency_final['always_mean_ms']:.3f} ms"
        )

        print(
            f"Adaptive mean                   : "
            f"{latency_final['adaptive_mean_ms']:.3f} ms"
        )

        print(
            f"Mean latency saving             : "
            f"{latency_final['mean_latency_savings_percent']:.2f}%"
        )

    print()

    print(
        f"Client summary                 : "
        f"{CLIENT_SUMMARY_FILE}"
    )

    print(
        f"Per-class metrics              : "
        f"{PER_CLASS_FILE}"
    )

    print(
        f"Adaptive gate audit            : "
        f"{PATH_AUDIT_FILE}"
    )

    print(
        f"Severity distribution          : "
        f"{SEVERITY_FILE}"
    )

    print(
        f"Model disagreement             : "
        f"{DISAGREEMENT_FILE}"
    )

    if not latency_raw_df.empty:
        print(
            f"Latency samples                : "
            f"{LATENCY_RAW_FILE}"
        )

        print(
            f"Latency summary                : "
            f"{LATENCY_SUMMARY_FILE}"
        )

    print(
        f"Final summary                  : "
        f"{FINAL_SUMMARY_FILE}"
    )

    print(
        f"Evaluation config              : "
        f"{CONFIG_FILE}"
    )

    print()

    print(
        "LOCKED TEST USED : NO"
    )

    print()

    print(
        "SCIENTIFIC LABEL:"
    )

    print(
        "SOURCE-ORDERED CNN-BiLSTM sequences; "
        "NOT timestamp-confirmed temporal sequences."
    )

    print()

    print(
        "IMPORTANT INTERPRETATION:"
    )

    print(
        "The same validation set was replayed at each client."
    )

    print(
        "Do not report pooled client-exposure count as the number "
        "of independent validation flows."
    )


if __name__ == "__main__":
    main()
