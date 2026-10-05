from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PHASE_NAME = (
    "PHASE 14H1 — ADAPTIVE GATE THRESHOLD SENSITIVITY"
)

SEED = 42

NUM_CLASSES = 6
ANOMALOUS_ID = 6
ANOMALOUS_NAME = "ANOMALOUS_TRAFFIC"

DEFAULT_THRESHOLDS = [
    0.80,
    0.85,
    0.90,
    0.925,
    0.95,
    0.975,
    0.99,
]

BASELINE_THRESHOLD = 0.95

# These are reporting / candidate-screening tolerances only.
# They do not alter any model, threshold, probability, or prediction.
DEFAULT_MAX_MACRO_F1_DROP = 0.0010
DEFAULT_MAX_BALANCED_ACCURACY_DROP = 0.0010
DEFAULT_MAX_FALSE_FAST_RATE = 0.0050

CLASS_NAMES = [
    "BENIGN",
    "DDoS",
    "DoS GoldenEye",
    "DoS Hulk",
    "FTP-Patator",
    "PortScan",
]

CLIENT_ATTACKS = {
    1: "DoS Hulk",
    2: "DDoS",
    3: "PortScan",
    4: "DoS GoldenEye",
    5: "FTP-Patator",
}


# ============================================================
# PHASE14H INPUTS
# ============================================================

PHASE14H_ROOT = (
    PROJECT_ROOT
    / "results"
    / "phase14"
    / "phase14h"
)

CLIENT_PREDICTION_TEMPLATE = (
    PHASE14H_ROOT
    / "phase14h_client_{client_id}_predictions.npz"
)

LATENCY_SAMPLES_FILE = (
    PHASE14H_ROOT
    / "phase14h_latency_samples.csv"
)

PHASE14H_SUMMARY_FILE = (
    PHASE14H_ROOT
    / "phase14h_summary.json"
)


# ============================================================
# OUTPUTS
# ============================================================

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "phase14"
    / "phase14h1"
)

ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "detection"
    / "phase14h1"
)

POOLED_SWEEP_FILE = (
    RESULT_ROOT
    / "phase14h1_pooled_threshold_sensitivity.csv"
)

CLIENT_SWEEP_FILE = (
    RESULT_ROOT
    / "phase14h1_client_threshold_sensitivity.csv"
)

PER_CLASS_FILE = (
    RESULT_ROOT
    / "phase14h1_per_class_recall.csv"
)

LATENCY_FILE = (
    RESULT_ROOT
    / "phase14h1_threshold_latency.csv"
)

PARETO_FILE = (
    RESULT_ROOT
    / "phase14h1_pareto_frontier.csv"
)

CANDIDATE_FILE = (
    RESULT_ROOT
    / "phase14h1_candidate_thresholds.csv"
)

SUMMARY_FILE = (
    RESULT_ROOT
    / "phase14h1_summary.json"
)

CONFIG_FILE = (
    ARTIFACT_ROOT
    / "phase14h1_config.json"
)


# ============================================================
# HELPERS
# ============================================================

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


def save_json(path: Path, payload) -> None:
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


def parse_thresholds(value: str) -> List[float]:
    thresholds = []

    for item in value.split(","):
        threshold = float(
            item.strip()
        )

        if not (
            0.0
            <
            threshold
            <
            1.0
        ):
            raise ValueError(
                "Every threshold must be in (0,1)."
            )

        if threshold not in thresholds:
            thresholds.append(
                threshold
            )

    thresholds = sorted(
        thresholds
    )

    if not thresholds:
        raise ValueError(
            "At least one threshold is required."
        )

    return thresholds


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
# METRICS
# ============================================================

def build_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    return confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(
            NUM_CLASSES + 1
        ),
    )


def metrics_from_confusion(
    cm: np.ndarray,
):
    total = float(
        np.sum(
            cm[
                :NUM_CLASSES,
                :
            ]
        )
    )

    correct = float(
        np.trace(
            cm[
                :NUM_CLASSES,
                :NUM_CLASSES
            ]
        )
    )

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    precisions = []
    recalls = []
    f1s = []
    supports = []

    per_class = []

    for class_id in range(
        NUM_CLASSES
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
                    :NUM_CLASSES,
                    class_id,
                ]
            )
        )

        precision = (
            tp / predicted_count
            if predicted_count > 0
            else 0.0
        )

        recall = (
            tp / support
            if support > 0
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

        per_class.append(
            {
                "class_id":
                    class_id,

                "class_name":
                    CLASS_NAMES[
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

    return {
        "accuracy":
            accuracy,

        "balanced_accuracy":
            float(
                np.mean(
                    recalls
                )
            ),

        "macro_precision":
            float(
                np.mean(
                    precisions
                )
            ),

        "macro_recall":
            float(
                np.mean(
                    recalls
                )
            ),

        "macro_f1":
            float(
                np.mean(
                    f1s
                )
            ),

        "weighted_f1":
            weighted_f1,

        "anomalous_traffic_predictions":
            int(
                np.sum(
                    cm[
                        :NUM_CLASSES,
                        ANOMALOUS_ID,
                    ]
                )
            ),

        "per_class":
            per_class,
    }


# ============================================================
# LOAD PHASE14H FROZEN PREDICTION EVIDENCE
# ============================================================

def load_client_predictions(
    client_id: int,
):
    path = Path(
        str(
            CLIENT_PREDICTION_TEMPLATE
        ).format(
            client_id=
                client_id
        )
    )

    if not path.exists():
        raise FileNotFoundError(
            "\nMissing Phase14H prediction artifact:\n"
            f"{path}\n\n"
            "Run Phase14H first."
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        result = {
            key:
                np.asarray(
                    data[
                        key
                    ]
                )
            for key
            in data.files
        }

    required = {
        "y_true",
        "rf_pred",
        "rf_conf",
        "full_fusion_pred",
        "full_fusion_conf",
        "full_fusion_risk",
        "full_fusion_severity_id",
    }

    missing = (
        required
        -
        set(
            result.keys()
        )
    )

    if missing:
        raise RuntimeError(
            "\nPhase14H prediction NPZ is missing:\n"
            f"{sorted(missing)}"
        )

    return (
        path,
        result,
    )


# ============================================================
# THRESHOLD SWEEP
# ============================================================

def evaluate_threshold(
    *,
    threshold: float,
    y_true: np.ndarray,
    rf_pred: np.ndarray,
    rf_conf: np.ndarray,
    full_pred: np.ndarray,
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
            threshold
        )
    )

    adaptive_pred = np.where(
        fast_mask,
        rf_pred,
        full_pred,
    ).astype(
        np.int64,
        copy=False,
    )

    cm = build_confusion(
        y_true,
        adaptive_pred,
    )

    metrics = metrics_from_confusion(
        cm
    )

    fast_count = int(
        np.sum(
            fast_mask
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

    wrong_family_mask = (
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

    benign_false_fast_mask = (
        fast_mask
        &
        (
            y_true
            ==
            0
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

    wrong_family = int(
        np.sum(
            wrong_family_mask
        )
    )

    benign_false_fast = int(
        np.sum(
            benign_false_fast_mask
        )
    )

    false_fast_rate = (
        fast_wrong
        /
        fast_count
        if fast_count > 0
        else float(
            "nan"
        )
    )

    return {
        "prediction":
            adaptive_pred,

        "confusion":
            cm,

        "metrics":
            metrics,

        "fast_mask":
            fast_mask,

        "fast_path_count":
            fast_count,

        "fast_path_rate":
            fast_count
            /
            len(
                y_true
            ),

        "deep_path_count":
            len(
                y_true
            )
            -
            fast_count,

        "deep_path_rate":
            1.0
            -
            (
                fast_count
                /
                len(
                    y_true
                )
            ),

        "fast_path_correct":
            fast_correct,

        "fast_path_wrong":
            fast_wrong,

        "false_fast_path_rate":
            false_fast_rate,

        "benign_false_fast_path_count":
            benign_false_fast,

        "wrong_attack_family_fast_count":
            wrong_family,
    }


# ============================================================
# LATENCY ESTIMATION FOR THRESHOLD SWEEP
# ============================================================

def load_latency_samples():
    if not LATENCY_SAMPLES_FILE.exists():
        raise FileNotFoundError(
            "\nMissing Phase14H latency samples:\n"
            f"{LATENCY_SAMPLES_FILE}\n\n"
            "Run Phase14H with --latency-samples > 0 first."
        )

    dataframe = pd.read_csv(
        LATENCY_SAMPLES_FILE
    )

    required = {
        "client_id",
        "validation_index",
        "rf_ms",
        "ae_ms",
        "fedprox70_ms",
        "cnn_bilstm_ms",
        "risk_fusion_ms",
        "always_run_all_ms",
    }

    missing = (
        required
        -
        set(
            dataframe.columns
        )
    )

    if missing:
        raise RuntimeError(
            "\nLatency CSV missing columns:\n"
            f"{sorted(missing)}"
        )

    return dataframe


def latency_for_threshold(
    *,
    client_id: int,
    threshold: float,
    latency_df: pd.DataFrame,
    client_prediction_data: Dict[str, np.ndarray],
):
    subset = (
        latency_df[
            latency_df[
                "client_id"
            ]
            ==
            client_id
        ]
        .copy()
    )

    if subset.empty:
        return None

    validation_indices = subset[
        "validation_index"
    ].to_numpy(
        dtype=np.int64
    )

    rf_pred = client_prediction_data[
        "rf_pred"
    ][
        validation_indices
    ]

    rf_conf = client_prediction_data[
        "rf_conf"
    ][
        validation_indices
    ]

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
            threshold
        )
    )

    # Threshold-sweep latency is COMPONENT-BASED and estimated
    # from the SAME Phase14H per-event component timings.
    #
    # FAST:
    #   RF only.
    #
    # DEEP:
    #   RF + AE + FedProx70 + CNN-BiLSTM + risk fusion.
    #
    # This avoids re-running all seven threshold arms with
    # different wall-clock noise and isolates only the gate decision.
    fast_latency = subset[
        "rf_ms"
    ].to_numpy(
        dtype=np.float64
    )

    deep_latency = (
        subset[
            "rf_ms"
        ].to_numpy(
            dtype=np.float64
        )
        +
        subset[
            "ae_ms"
        ].to_numpy(
            dtype=np.float64
        )
        +
        subset[
            "fedprox70_ms"
        ].to_numpy(
            dtype=np.float64
        )
        +
        subset[
            "cnn_bilstm_ms"
        ].to_numpy(
            dtype=np.float64
        )
        +
        subset[
            "risk_fusion_ms"
        ].to_numpy(
            dtype=np.float64
        )
    )

    adaptive_estimated = np.where(
        fast_mask,
        fast_latency,
        deep_latency,
    )

    always_component = deep_latency

    mean_always = float(
        np.mean(
            always_component
        )
    )

    mean_adaptive = float(
        np.mean(
            adaptive_estimated
        )
    )

    aggregate_saving = (
        (
            mean_always
            -
            mean_adaptive
        )
        /
        mean_always
        *
        100.0
        if mean_always > 0
        else float(
            "nan"
        )
    )

    return {
        "client_id":
            client_id,

        "threshold":
            threshold,

        "profile_samples":
            len(
                subset
            ),

        "profile_fast_count":
            int(
                np.sum(
                    fast_mask
                )
            ),

        "profile_fast_rate":
            float(
                np.mean(
                    fast_mask
                )
            ),

        "always_component_mean_ms":
            mean_always,

        "adaptive_estimated_mean_ms":
            mean_adaptive,

        "adaptive_estimated_median_ms":
            float(
                np.median(
                    adaptive_estimated
                )
            ),

        "adaptive_estimated_p95_ms":
            percentile(
                adaptive_estimated,
                95,
            ),

        "adaptive_estimated_p99_ms":
            percentile(
                adaptive_estimated,
                99,
            ),

        "aggregate_mean_latency_saving_percent":
            aggregate_saving,

        "latency_method":
            (
                "component-based estimate from frozen Phase14H "
                "per-event timings; fast=RF only, deep=all components"
            ),
    }


# ============================================================
# PARETO FRONTIER
# ============================================================

def add_pareto_flag(
    dataframe: pd.DataFrame,
):
    if dataframe.empty:
        dataframe[
            "pareto_optimal"
        ] = False
        return dataframe

    pareto = np.ones(
        len(
            dataframe
        ),
        dtype=bool,
    )

    fast = dataframe[
        "fast_path_rate"
    ].to_numpy(
        dtype=np.float64
    )

    macro = dataframe[
        "macro_f1"
    ].to_numpy(
        dtype=np.float64
    )

    false_fast = dataframe[
        "false_fast_path_rate"
    ].to_numpy(
        dtype=np.float64
    )

    latency = dataframe[
        "estimated_mean_latency_ms"
    ].to_numpy(
        dtype=np.float64
    )

    for i in range(
        len(
            dataframe
        )
    ):
        if not pareto[
            i
        ]:
            continue

        for j in range(
            len(
                dataframe
            )
        ):
            if i == j:
                continue

            # j dominates i if it is no worse in all objectives:
            #   higher fast coverage,
            #   higher Macro F1,
            #   lower false-fast rate,
            #   lower estimated latency,
            # and strictly better in at least one.
            no_worse = (
                fast[
                    j
                ]
                >=
                fast[
                    i
                ]
                and
                macro[
                    j
                ]
                >=
                macro[
                    i
                ]
                and
                false_fast[
                    j
                ]
                <=
                false_fast[
                    i
                ]
                and
                latency[
                    j
                ]
                <=
                latency[
                    i
                ]
            )

            strictly_better = (
                fast[
                    j
                ]
                >
                fast[
                    i
                ]
                or
                macro[
                    j
                ]
                >
                macro[
                    i
                ]
                or
                false_fast[
                    j
                ]
                <
                false_fast[
                    i
                ]
                or
                latency[
                    j
                ]
                <
                latency[
                    i
                ]
            )

            if (
                no_worse
                and
                strictly_better
            ):
                pareto[
                    i
                ] = False
                break

    dataframe = dataframe.copy()

    dataframe[
        "pareto_optimal"
    ] = pareto

    return dataframe


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=PHASE_NAME
    )

    parser.add_argument(
        "--thresholds",
        type=str,
        default=
            ",".join(
                str(
                    value
                )
                for value
                in DEFAULT_THRESHOLDS
            ),
        help=(
            "Comma-separated RF fast-path thresholds."
        ),
    )

    parser.add_argument(
        "--max-macro-f1-drop",
        type=float,
        default=
            DEFAULT_MAX_MACRO_F1_DROP,
        help=(
            "Candidate-screening tolerance vs always-run-all."
        ),
    )

    parser.add_argument(
        "--max-balanced-accuracy-drop",
        type=float,
        default=
            DEFAULT_MAX_BALANCED_ACCURACY_DROP,
    )

    parser.add_argument(
        "--max-false-fast-rate",
        type=float,
        default=
            DEFAULT_MAX_FALSE_FAST_RATE,
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    thresholds = parse_thresholds(
        args.thresholds
    )

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARTIFACT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    latency_df = load_latency_samples()

    separator()

    print(
        PHASE_NAME
    )

    separator()

    print(
        f"Seed                          : {SEED}"
    )

    print(
        f"Thresholds                    : {thresholds}"
    )

    print(
        f"Current baseline threshold    : {BASELINE_THRESHOLD}"
    )

    print(
        f"Macro F1 drop tolerance       : "
        f"{args.max_macro_f1_drop:.6f}"
    )

    print(
        f"Balanced Acc drop tolerance   : "
        f"{args.max_balanced_accuracy_drop:.6f}"
    )

    print(
        f"False-fast rate tolerance     : "
        f"{args.max_false_fast_rate:.4%}"
    )

    print(
        "Models / AE thresholds         : FROZEN"
    )

    print(
        "Fusion weights                  : FROZEN"
    )

    print(
        "Locked test                     : NO"
    )

    print()
    print("SCIENTIFIC CONTROL")
    print("-" * 126)

    print(
        "Phase14H1 does NOT run or retrain any model."
    )

    print(
        "It reuses the exact Phase14H predictions and varies ONLY "
        "the RF fast-gate threshold."
    )

    print(
        "Full-fusion predictions remain identical in every arm."
    )

    print(
        "Latency is a component-based estimate from the SAME "
        "Phase14H profiling samples, avoiding cross-arm timing noise."
    )

    # ========================================================
    # LOAD ALL CLIENT ARTIFACTS
    # ========================================================

    client_data = {}

    y_reference = None

    for client_id in range(
        1,
        6,
    ):
        path, data = load_client_predictions(
            client_id
        )

        client_data[
            client_id
        ] = data

        y = data[
            "y_true"
        ].astype(
            np.int64,
            copy=False,
        )

        if y_reference is None:
            y_reference = y

        else:
            if not np.array_equal(
                y_reference,
                y,
            ):
                raise RuntimeError(
                    "\nClient Phase14H prediction files do not "
                    "contain the same validation labels."
                )

        print(
            f"Client {client_id} predictions      : {path}"
        )

    # ========================================================
    # ALWAYS-RUN-ALL BASELINE
    # ========================================================

    pooled_full_cm = np.zeros(
        (
            NUM_CLASSES + 1,
            NUM_CLASSES + 1,
        ),
        dtype=np.int64,
    )

    for client_id in range(
        1,
        6,
    ):
        data = client_data[
            client_id
        ]

        pooled_full_cm += build_confusion(
            data[
                "y_true"
            ],
            data[
                "full_fusion_pred"
            ],
        )

    full_metrics = metrics_from_confusion(
        pooled_full_cm
    )

    print()
    print("ALWAYS-RUN-ALL REFERENCE")
    print("-" * 126)

    print(
        f"Accuracy                      : "
        f"{full_metrics['accuracy']:.6f}"
    )

    print(
        f"Balanced Accuracy             : "
        f"{full_metrics['balanced_accuracy']:.6f}"
    )

    print(
        f"Macro F1                      : "
        f"{full_metrics['macro_f1']:.6f}"
    )

    print(
        f"Weighted F1                   : "
        f"{full_metrics['weighted_f1']:.6f}"
    )

    # ========================================================
    # THRESHOLD SWEEP
    # ========================================================

    pooled_rows = []
    client_rows = []
    per_class_rows = []
    latency_rows = []

    for threshold in thresholds:
        pooled_cm = np.zeros_like(
            pooled_full_cm
        )

        pooled_fast_count = 0
        pooled_fast_wrong = 0
        pooled_fast_correct = 0
        pooled_wrong_family = 0
        pooled_benign_false_fast = 0
        pooled_exposures = 0

        threshold_latency_client_rows = []

        for client_id in range(
            1,
            6,
        ):
            data = client_data[
                client_id
            ]

            result = evaluate_threshold(
                threshold=
                    threshold,

                y_true=
                    data[
                        "y_true"
                    ],

                rf_pred=
                    data[
                        "rf_pred"
                    ],

                rf_conf=
                    data[
                        "rf_conf"
                    ],

                full_pred=
                    data[
                        "full_fusion_pred"
                    ],
            )

            pooled_cm += result[
                "confusion"
            ]

            pooled_fast_count += (
                result[
                    "fast_path_count"
                ]
            )

            pooled_fast_wrong += (
                result[
                    "fast_path_wrong"
                ]
            )

            pooled_fast_correct += (
                result[
                    "fast_path_correct"
                ]
            )

            pooled_wrong_family += (
                result[
                    "wrong_attack_family_fast_count"
                ]
            )

            pooled_benign_false_fast += (
                result[
                    "benign_false_fast_path_count"
                ]
            )

            pooled_exposures += len(
                data[
                    "y_true"
                ]
            )

            metrics = result[
                "metrics"
            ]

            client_rows.append(
                {
                    "threshold":
                        threshold,

                    "client_id":
                        client_id,

                    "seen_attack":
                        CLIENT_ATTACKS[
                            client_id
                        ],

                    "accuracy":
                        metrics[
                            "accuracy"
                        ],

                    "balanced_accuracy":
                        metrics[
                            "balanced_accuracy"
                        ],

                    "macro_f1":
                        metrics[
                            "macro_f1"
                        ],

                    "weighted_f1":
                        metrics[
                            "weighted_f1"
                        ],

                    "fast_path_rate":
                        result[
                            "fast_path_rate"
                        ],

                    "false_fast_path_rate":
                        result[
                            "false_fast_path_rate"
                        ],

                    "fast_path_correct":
                        result[
                            "fast_path_correct"
                        ],

                    "fast_path_wrong":
                        result[
                            "fast_path_wrong"
                        ],

                    "wrong_attack_family_fast_count":
                        result[
                            "wrong_attack_family_fast_count"
                        ],

                    "benign_false_fast_path_count":
                        result[
                            "benign_false_fast_path_count"
                        ],
                }
            )

            for class_row in metrics[
                "per_class"
            ]:
                per_class_rows.append(
                    {
                        "threshold":
                            threshold,

                        "client_id":
                            client_id,

                        "seen_attack":
                            CLIENT_ATTACKS[
                                client_id
                            ],

                        **class_row,
                    }
                )

            latency = latency_for_threshold(
                client_id=
                    client_id,

                threshold=
                    threshold,

                latency_df=
                    latency_df,

                client_prediction_data=
                    data,
            )

            if latency is not None:
                latency_rows.append(
                    latency
                )

                threshold_latency_client_rows.append(
                    latency
                )

        pooled_metrics = metrics_from_confusion(
            pooled_cm
        )

        pooled_false_fast_rate = (
            pooled_fast_wrong
            /
            pooled_fast_count
            if pooled_fast_count > 0
            else float(
                "nan"
            )
        )

        pooled_fast_rate = (
            pooled_fast_count
            /
            pooled_exposures
        )

        if threshold_latency_client_rows:
            estimated_mean_latency_ms = float(
                np.mean(
                    [
                        row[
                            "adaptive_estimated_mean_ms"
                        ]
                        for row
                        in threshold_latency_client_rows
                    ]
                )
            )

            always_mean_latency_ms = float(
                np.mean(
                    [
                        row[
                            "always_component_mean_ms"
                        ]
                        for row
                        in threshold_latency_client_rows
                    ]
                )
            )

            aggregate_saving = (
                (
                    always_mean_latency_ms
                    -
                    estimated_mean_latency_ms
                )
                /
                always_mean_latency_ms
                *
                100.0
                if always_mean_latency_ms > 0
                else float(
                    "nan"
                )
            )
        else:
            estimated_mean_latency_ms = float(
                "nan"
            )

            always_mean_latency_ms = float(
                "nan"
            )

            aggregate_saving = float(
                "nan"
            )

        macro_drop = (
            full_metrics[
                "macro_f1"
            ]
            -
            pooled_metrics[
                "macro_f1"
            ]
        )

        ba_drop = (
            full_metrics[
                "balanced_accuracy"
            ]
            -
            pooled_metrics[
                "balanced_accuracy"
            ]
        )

        eligible = (
            macro_drop
            <=
            args.max_macro_f1_drop
            and
            ba_drop
            <=
            args.max_balanced_accuracy_drop
            and
            pooled_false_fast_rate
            <=
            args.max_false_fast_rate
        )

        pooled_rows.append(
            {
                "threshold":
                    threshold,

                "accuracy":
                    pooled_metrics[
                        "accuracy"
                    ],

                "balanced_accuracy":
                    pooled_metrics[
                        "balanced_accuracy"
                    ],

                "macro_f1":
                    pooled_metrics[
                        "macro_f1"
                    ],

                "weighted_f1":
                    pooled_metrics[
                        "weighted_f1"
                    ],

                "macro_f1_drop_vs_always":
                    macro_drop,

                "balanced_accuracy_drop_vs_always":
                    ba_drop,

                "fast_path_count":
                    pooled_fast_count,

                "fast_path_rate":
                    pooled_fast_rate,

                "deep_path_rate":
                    1.0
                    -
                    pooled_fast_rate,

                "fast_path_correct":
                    pooled_fast_correct,

                "fast_path_wrong":
                    pooled_fast_wrong,

                "false_fast_path_rate":
                    pooled_false_fast_rate,

                "wrong_attack_family_fast_count":
                    pooled_wrong_family,

                "benign_false_fast_path_count":
                    pooled_benign_false_fast,

                "estimated_mean_latency_ms":
                    estimated_mean_latency_ms,

                "always_component_mean_latency_ms":
                    always_mean_latency_ms,

                "estimated_latency_saving_percent":
                    aggregate_saving,

                "candidate_eligible":
                    eligible,
            }
        )

        print()
        print(
            f"THRESHOLD {threshold:.3f}"
        )

        print("-" * 126)

        print(
            f"Fast path rate                : "
            f"{pooled_fast_rate:.4%}"
        )

        print(
            f"False-fast-path rate          : "
            f"{pooled_false_fast_rate:.4%}"
        )

        print(
            f"Wrong attack-family fast      : "
            f"{pooled_wrong_family:,}"
        )

        print(
            f"Balanced Accuracy             : "
            f"{pooled_metrics['balanced_accuracy']:.6f}"
        )

        print(
            f"Macro F1                      : "
            f"{pooled_metrics['macro_f1']:.6f}"
        )

        print(
            f"Macro F1 drop vs always       : "
            f"{macro_drop:+.6f}"
        )

        print(
            f"Estimated mean latency        : "
            f"{estimated_mean_latency_ms:.3f} ms"
        )

        print(
            f"Estimated latency saving      : "
            f"{aggregate_saving:.2f}%"
        )

        print(
            f"Candidate eligible            : "
            f"{eligible}"
        )

    # ========================================================
    # SAVE TABLES
    # ========================================================

    pooled_df = pd.DataFrame(
        pooled_rows
    )

    pooled_df = add_pareto_flag(
        pooled_df
    )

    pooled_df.to_csv(
        POOLED_SWEEP_FILE,
        index=False,
    )

    pd.DataFrame(
        client_rows
    ).to_csv(
        CLIENT_SWEEP_FILE,
        index=False,
    )

    pd.DataFrame(
        per_class_rows
    ).to_csv(
        PER_CLASS_FILE,
        index=False,
    )

    pd.DataFrame(
        latency_rows
    ).to_csv(
        LATENCY_FILE,
        index=False,
    )

    pareto_df = pooled_df[
        pooled_df[
            "pareto_optimal"
        ]
    ].copy()

    pareto_df.to_csv(
        PARETO_FILE,
        index=False,
    )

    candidate_df = pooled_df[
        pooled_df[
            "candidate_eligible"
        ]
    ].copy()

    candidate_df = candidate_df.sort_values(
        by=[
            "estimated_mean_latency_ms",
            "false_fast_path_rate",
            "macro_f1",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )

    candidate_df.to_csv(
        CANDIDATE_FILE,
        index=False,
    )

    # ========================================================
    # RECOMMENDATION LOGIC
    # ========================================================

    recommended_threshold = None
    recommendation_reason = None

    if not candidate_df.empty:
        best = candidate_df.iloc[
            0
        ]

        recommended_threshold = float(
            best[
                "threshold"
            ]
        )

        recommendation_reason = (
            "Lowest component-estimated mean adaptive latency among "
            "thresholds satisfying the predeclared Macro-F1, "
            "Balanced-Accuracy, and false-fast-path tolerances."
        )

    baseline_row = pooled_df[
        np.isclose(
            pooled_df[
                "threshold"
            ],
            BASELINE_THRESHOLD,
        )
    ]

    baseline_payload = None

    if not baseline_row.empty:
        baseline_payload = (
            baseline_row
            .iloc[
                0
            ]
            .to_dict()
        )

    config = {
        "phase":
            "14H1",

        "seed":
            SEED,

        "thresholds":
            thresholds,

        "baseline_threshold":
            BASELINE_THRESHOLD,

        "models_retrained":
            False,

        "models_changed":
            False,

        "ae_thresholds_changed":
            False,

        "fusion_weights_changed":
            False,

        "only_variable":
            "RF adaptive-gate confidence threshold",

        "candidate_tolerances":
            {
                "max_macro_f1_drop":
                    args.max_macro_f1_drop,

                "max_balanced_accuracy_drop":
                    args.max_balanced_accuracy_drop,

                "max_false_fast_path_rate":
                    args.max_false_fast_rate,
            },

        "latency_method":
            (
                "Component-based estimate using the exact same "
                "Phase14H per-event profiling observations. "
                "FAST=RF time only. DEEP=RF+AE+FedProx70+"
                "CNN-BiLSTM+risk-fusion component times."
            ),

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
        config,
    )

    summary = {
        "phase":
            "14H1",

        "always_run_all_reference":
            {
                key:
                    value
                for key, value
                in full_metrics.items()
                if key
                !=
                "per_class"
            },

        "baseline_threshold_0_95":
            baseline_payload,

        "recommended_threshold":
            recommended_threshold,

        "recommendation_reason":
            recommendation_reason,

        "candidate_count":
            len(
                candidate_df
            ),

        "pareto_thresholds":
            pareto_df[
                "threshold"
            ].tolist(),

        "locked_test_used":
            False,

        "important_note":
            (
                "The threshold recommendation is validation-selected "
                "method development, not independent confirmatory "
                "evidence. Final claims require a separate external "
                "or re-held-out confirmatory evaluation."
            ),
    }

    save_json(
        SUMMARY_FILE,
        summary,
    )

    # ========================================================
    # FINAL
    # ========================================================

    separator()

    print(
        "PHASE 14H1 COMPLETE"
    )

    separator()

    display_columns = [
        "threshold",
        "fast_path_rate",
        "false_fast_path_rate",
        "wrong_attack_family_fast_count",
        "balanced_accuracy",
        "macro_f1",
        "macro_f1_drop_vs_always",
        "estimated_mean_latency_ms",
        "estimated_latency_saving_percent",
        "candidate_eligible",
        "pareto_optimal",
    ]

    print(
        pooled_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()

    if recommended_threshold is not None:
        print(
            f"Validation-selected candidate threshold : "
            f"{recommended_threshold:.3f}"
        )

        print(
            "Selection rule                          : "
            "lowest estimated mean latency among eligible arms"
        )
    else:
        print(
            "Validation-selected candidate threshold : NONE"
        )

        print(
            "Reason                                  : "
            "No arm satisfied all predeclared tolerances."
        )

    print()

    print(
        f"Pooled sensitivity table               : "
        f"{POOLED_SWEEP_FILE}"
    )

    print(
        f"Client sensitivity table               : "
        f"{CLIENT_SWEEP_FILE}"
    )

    print(
        f"Per-class recall table                 : "
        f"{PER_CLASS_FILE}"
    )

    print(
        f"Threshold latency table                : "
        f"{LATENCY_FILE}"
    )

    print(
        f"Pareto frontier                        : "
        f"{PARETO_FILE}"
    )

    print(
        f"Eligible candidates                    : "
        f"{CANDIDATE_FILE}"
    )

    print(
        f"Summary                                : "
        f"{SUMMARY_FILE}"
    )

    print(
        f"Config                                 : "
        f"{CONFIG_FILE}"
    )

    print()

    print(
        "LOCKED TEST USED : NO"
    )

    print()

    print(
        "SCIENTIFIC INTERPRETATION:"
    )

    print(
        "Only the RF adaptive-gate confidence threshold changed."
    )

    print(
        "All model outputs, AE thresholds, fusion weights and "
        "full-fusion decisions remained frozen."
    )

    print(
        "Latency values are component-based threshold-sweep estimates "
        "using the same Phase14H timing observations, not fresh "
        "wall-clock remeasurements for each threshold arm."
    )


if __name__ == "__main__":
    main()
