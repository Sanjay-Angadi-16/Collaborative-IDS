from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# ============================================================
# PROJECT MODULES
# ============================================================

try:
    import phase14g1_real_model_wiring as phase14g1
    import phase14h_full_validation_adaptive_gate as phase14h
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "\nPlace this script inside the project's scripts folder.\n\n"
        "Required existing files:\n"
        "  scripts\\phase14g1_real_model_wiring.py\n"
        "  scripts\\phase14h_full_validation_adaptive_gate.py\n"
    ) from exc


# ============================================================
# CONFIG
# ============================================================

PHASE_NAME = "PHASE 17H.4 — FALSE-NEGATIVE ANALYSIS"

SEED = 42
NUM_CLIENTS = 5
NUM_KNOWN_CLASSES = 6
BENIGN_ID = 0
ANOMALOUS_ID = 6

EXPECTED_FEATURES = 70
SEQUENCE_LENGTH = 20
DEFAULT_BATCH_SIZE = 2048

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "dashboard_samples"
    / "ids_5_attacks_1000_sequences.npz"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "dashboard"
    / "phase17h4_false_negative_analysis"
)

ALL_PREDICTIONS_CSV = RESULT_ROOT / "phase17h4_all_predictions.csv"
MISSED_ATTACKS_CSV = RESULT_ROOT / "phase17h4_missed_attacks.csv"
MISSED_BY_CLASS_CSV = RESULT_ROOT / "phase17h4_missed_by_class.csv"
MISSED_BY_CLIENT_CSV = RESULT_ROOT / "phase17h4_missed_by_client.csv"
MODEL_BEHAVIOR_CSV = RESULT_ROOT / "phase17h4_model_behavior_on_misses.csv"
ROUTING_ANALYSIS_CSV = RESULT_ROOT / "phase17h4_routing_analysis.csv"
CONFUSION_CSV = RESULT_ROOT / "phase17h4_confusion_matrix.csv"
SUMMARY_JSON = RESULT_ROOT / "phase17h4_summary.json"


# ============================================================
# HELPERS
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def separator(width: int = 120) -> None:
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


def class_name(
    class_id: int,
    inverse_mapping: dict[int, str],
) -> str:
    class_id = int(class_id)

    if class_id == ANOMALOUS_ID:
        return "ANOMALOUS_TRAFFIC"

    return str(
        inverse_mapping.get(
            class_id,
            f"CLASS_{class_id}",
        )
    )


def severity_name(severity_id: int) -> str:
    names = {
        0: "NORMAL",
        1: "LOW",
        2: "MEDIUM",
        3: "HIGH",
        4: "CRITICAL",
    }

    return names.get(
        int(severity_id),
        f"SEVERITY_{severity_id}",
    )


def load_npz(input_file: Path):
    if not input_file.exists():
        raise FileNotFoundError(
            f"\nInput NPZ not found:\n{input_file}\n\n"
            "Run Phase 17H.2 first."
        )

    with np.load(
        input_file,
        allow_pickle=False,
    ) as data:
        if "X" not in data.files:
            raise KeyError(
                "NPZ must contain array 'X'."
            )

        if "y" not in data.files:
            raise KeyError(
                "NPZ must contain array 'y'."
            )

        X = np.asarray(
            data["X"],
            dtype=np.float32,
        )

        y = np.asarray(
            data["y"],
            dtype=np.int64,
        )

        validation_index = (
            np.asarray(
                data["validation_index"],
                dtype=np.int64,
            )
            if "validation_index" in data.files
            else np.arange(
                len(y),
                dtype=np.int64,
            )
        )

        # Analysis-only metadata.
        # It is NOT used to route predictions.
        specialist_client_id = (
            np.asarray(
                data["specialist_client_id"],
                dtype=np.int64,
            )
            if "specialist_client_id" in data.files
            else None
        )

    if X.ndim != 3:
        raise RuntimeError(
            f"Expected X shape (N,20,70); found {X.shape}."
        )

    if tuple(X.shape[1:]) != (
        SEQUENCE_LENGTH,
        EXPECTED_FEATURES,
    ):
        raise RuntimeError(
            "Expected X shape "
            f"(N,{SEQUENCE_LENGTH},{EXPECTED_FEATURES}); "
            f"found {X.shape}."
        )

    if y.ndim != 1 or len(y) != len(X):
        raise RuntimeError(
            "X/y shape mismatch."
        )

    if np.any(
        y == BENIGN_ID
    ):
        raise RuntimeError(
            "This Phase 17H.4 script expects the attack-only Phase 17H.2 dataset."
        )

    return (
        X,
        y,
        validation_index,
        specialist_client_id,
    )


# ============================================================
# MODEL INFERENCE
# ============================================================

def run_inference(
    X: np.ndarray,
    y_true: np.ndarray,
    validation_index: np.ndarray,
    specialist_client_id: np.ndarray | None,
    label_mapping: dict[str, int],
    inverse_mapping: dict[int, str],
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:

    n = len(
        y_true
    )

    flows = np.asarray(
        X[
            :,
            -1,
            :,
        ],
        dtype=np.float32,
    )

    # Primary routing is label-independent round robin.
    assigned_clients = (
        np.arange(
            n,
            dtype=np.int64,
        )
        %
        NUM_CLIENTS
    ) + 1

    # --------------------------------------------------------
    # Global models
    # --------------------------------------------------------
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

    (
        fed_pred,
        fed_conf,
    ) = phase14h.predict_fedprox70_batch(
        predictor=
            fedprox70,

        flows=
            flows,

        batch_size=
            batch_size,

        device=
            device,
    )

    (
        cnn_pred,
        cnn_conf,
    ) = phase14h.predict_cnn_batch(
        predictor=
            cnn,

        sequences=
            X,

        batch_size=
            batch_size,

        device=
            device,
    )

    # --------------------------------------------------------
    # Local models and final policy
    # --------------------------------------------------------
    final_pred = np.empty(
        n,
        dtype=np.int64,
    )

    final_conf = np.empty(
        n,
        dtype=np.float32,
    )

    fusion_risk = np.empty(
        n,
        dtype=np.float32,
    )

    severity_id = np.empty(
        n,
        dtype=np.int8,
    )

    processing_path = np.empty(
        n,
        dtype=object,
    )

    decision_source = np.empty(
        n,
        dtype=object,
    )

    rf_pred_all = np.empty(
        n,
        dtype=np.int64,
    )

    rf_conf_all = np.empty(
        n,
        dtype=np.float32,
    )

    ae_error_all = np.empty(
        n,
        dtype=np.float32,
    )

    ae_score_all = np.empty(
        n,
        dtype=np.float32,
    )

    ae_anomaly_all = np.empty(
        n,
        dtype=bool,
    )

    for client_id in range(
        1,
        NUM_CLIENTS + 1,
    ):
        positions = np.flatnonzero(
            assigned_clients
            ==
            client_id
        )

        if len(
            positions
        ) == 0:
            continue

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

        (
            rf_pred,
            rf_conf,
        ) = phase14h.predict_rf_batch(
            rf=
                rf,

            flows=
                flows[
                    positions
                ],
        )

        (
            ae_error,
            ae_score,
            ae_anomaly,
        ) = phase14h.predict_ae_batch(
            ae=
                ae,

            flows=
                flows[
                    positions
                ],

            batch_size=
                batch_size,

            device=
                device,
        )

        full_result = phase14h.full_fusion_vectorized(
            rf_pred=
                rf_pred,

            rf_conf=
                rf_conf,

            ae_score=
                ae_score,

            fed_pred=
                fed_pred[
                    positions
                ],

            fed_conf=
                fed_conf[
                    positions
                ],

            cnn_pred=
                cnn_pred[
                    positions
                ],

            cnn_conf=
                cnn_conf[
                    positions
                ],
        )

        adaptive_result = phase14h.adaptive_policy_vectorized(
            rf_pred=
                rf_pred,

            rf_conf=
                rf_conf,

            full_result=
                full_result,
        )

        rf_pred_all[
            positions
        ] = rf_pred

        rf_conf_all[
            positions
        ] = rf_conf

        ae_error_all[
            positions
        ] = ae_error

        ae_score_all[
            positions
        ] = ae_score

        ae_anomaly_all[
            positions
        ] = ae_anomaly

        final_pred[
            positions
        ] = adaptive_result[
            "prediction"
        ]

        final_conf[
            positions
        ] = adaptive_result[
            "confidence"
        ]

        fusion_risk[
            positions
        ] = adaptive_result[
            "risk"
        ]

        severity_id[
            positions
        ] = adaptive_result[
            "severity_id"
        ]

        fast_mask = adaptive_result[
            "fast_mask"
        ]

        processing_path[
            positions
        ] = np.where(
            fast_mask,
            "FAST_ATTACK_PATH",
            "DEEP_ANALYSIS",
        )

        decision_source[
            positions
        ] = np.where(
            fast_mask,
            "Random Forest",
            full_result[
                "decision_source"
            ],
        )

    # --------------------------------------------------------
    # Build dataframe
    # --------------------------------------------------------
    rows = []

    for i in range(
        n
    ):
        risk = fusion_risk[
            i
        ]

        risk_value = (
            None
            if np.isnan(
                risk
            )
            else float(
                risk
            )
        )

        specialist = (
            int(
                specialist_client_id[
                    i
                ]
            )
            if specialist_client_id
            is not None
            else None
        )

        route_match = (
            bool(
                int(
                    assigned_clients[
                        i
                    ]
                )
                ==
                specialist
            )
            if specialist
            is not None
            else None
        )

        true_id = int(
            y_true[
                i
            ]
        )

        final_id = int(
            final_pred[
                i
            ]
        )

        rf_id = int(
            rf_pred_all[
                i
            ]
        )

        fed_id = int(
            fed_pred[
                i
            ]
        )

        cnn_id = int(
            cnn_pred[
                i
            ]
        )

        # Evidence-pattern labels for diagnosis.
        global_benign_votes = int(
            fed_id
            ==
            BENIGN_ID
        ) + int(
            cnn_id
            ==
            BENIGN_ID
        )

        supervised_benign_votes = (
            int(
                rf_id
                ==
                BENIGN_ID
            )
            +
            global_benign_votes
        )

        if supervised_benign_votes == 3:
            evidence_pattern = (
                "RF+FedProx+CNN all BENIGN"
            )
        elif global_benign_votes == 2:
            evidence_pattern = (
                "FedProx+CNN BENIGN"
            )
        elif fed_id == BENIGN_ID:
            evidence_pattern = (
                "FedProx BENIGN only/global disagreement"
            )
        elif cnn_id == BENIGN_ID:
            evidence_pattern = (
                "CNN BENIGN only/global disagreement"
            )
        elif rf_id == BENIGN_ID:
            evidence_pattern = (
                "RF BENIGN only"
            )
        else:
            evidence_pattern = (
                "No supervised BENIGN vote"
            )

        rows.append(
            {
                "sequence_row":
                    int(
                        i
                    ),

                "validation_index":
                    int(
                        validation_index[
                            i
                        ]
                    ),

                "true_class_id":
                    true_id,

                "true_class":
                    class_name(
                        true_id,
                        inverse_mapping,
                    ),

                "assigned_client":
                    int(
                        assigned_clients[
                            i
                        ]
                    ),

                "specialist_client_analysis_only":
                    specialist,

                "routing_matches_specialist_analysis_only":
                    route_match,

                "rf_prediction_id":
                    rf_id,

                "rf_prediction":
                    class_name(
                        rf_id,
                        inverse_mapping,
                    ),

                "rf_confidence":
                    float(
                        rf_conf_all[
                            i
                        ]
                    ),

                "ae_reconstruction_error":
                    float(
                        ae_error_all[
                            i
                        ]
                    ),

                "ae_anomaly_score":
                    float(
                        ae_score_all[
                            i
                        ]
                    ),

                "ae_anomaly":
                    bool(
                        ae_anomaly_all[
                            i
                        ]
                    ),

                "fedprox70_prediction_id":
                    fed_id,

                "fedprox70_prediction":
                    class_name(
                        fed_id,
                        inverse_mapping,
                    ),

                "fedprox70_confidence":
                    float(
                        fed_conf[
                            i
                        ]
                    ),

                "cnn_bilstm_prediction_id":
                    cnn_id,

                "cnn_bilstm_prediction":
                    class_name(
                        cnn_id,
                        inverse_mapping,
                    ),

                "cnn_bilstm_confidence":
                    float(
                        cnn_conf[
                            i
                        ]
                    ),

                "fusion_risk":
                    risk_value,

                "final_prediction_id":
                    final_id,

                "final_prediction":
                    class_name(
                        final_id,
                        inverse_mapping,
                    ),

                "final_confidence":
                    float(
                        final_conf[
                            i
                        ]
                    ),

                "severity":
                    severity_name(
                        int(
                            severity_id[
                                i
                            ]
                        )
                    ),

                "processing_path":
                    str(
                        processing_path[
                            i
                        ]
                    ),

                "decision_source":
                    str(
                        decision_source[
                            i
                        ]
                    ),

                "correct":
                    bool(
                        final_id
                        ==
                        true_id
                    ),

                "missed_attack_to_benign":
                    bool(
                        final_id
                        ==
                        BENIGN_ID
                    ),

                "wrong_known_attack_family":
                    bool(
                        final_id
                        >=
                        1
                        and
                        final_id
                        <
                        NUM_KNOWN_CLASSES
                        and
                        final_id
                        !=
                        true_id
                    ),

                "predicted_anomalous":
                    bool(
                        final_id
                        ==
                        ANOMALOUS_ID
                    ),

                "rf_benign_vote":
                    bool(
                        rf_id
                        ==
                        BENIGN_ID
                    ),

                "fedprox_benign_vote":
                    bool(
                        fed_id
                        ==
                        BENIGN_ID
                    ),

                "cnn_benign_vote":
                    bool(
                        cnn_id
                        ==
                        BENIGN_ID
                    ),

                "global_supervised_benign_consensus":
                    bool(
                        fed_id
                        ==
                        BENIGN_ID
                        and
                        cnn_id
                        ==
                        BENIGN_ID
                    ),

                "all_supervised_benign_consensus":
                    bool(
                        rf_id
                        ==
                        BENIGN_ID
                        and
                        fed_id
                        ==
                        BENIGN_ID
                        and
                        cnn_id
                        ==
                        BENIGN_ID
                    ),

                "evidence_pattern":
                    evidence_pattern,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# SUMMARIES
# ============================================================

def build_summaries(
    df: pd.DataFrame,
    inverse_mapping: dict[int, str],
):
    missed = df[
        df[
            "missed_attack_to_benign"
        ]
    ].copy()

    # --------------------------------------------------------
    # Per-class false-negative summary
    # --------------------------------------------------------
    class_rows = []

    for class_id in sorted(
        df[
            "true_class_id"
        ].unique()
    ):
        class_df = df[
            df[
                "true_class_id"
            ]
            ==
            class_id
        ]

        class_missed = class_df[
            class_df[
                "missed_attack_to_benign"
            ]
        ]

        class_rows.append(
            {
                "true_class_id":
                    int(
                        class_id
                    ),

                "true_class":
                    class_name(
                        int(
                            class_id
                        ),
                        inverse_mapping,
                    ),

                "total_sequences":
                    int(
                        len(
                            class_df
                        )
                    ),

                "correct":
                    int(
                        class_df[
                            "correct"
                        ].sum()
                    ),

                "missed_to_benign":
                    int(
                        len(
                            class_missed
                        )
                    ),

                "miss_rate":
                    float(
                        len(
                            class_missed
                        )
                        /
                        max(
                            len(
                                class_df
                            ),
                            1,
                        )
                    ),

                "wrong_attack_family":
                    int(
                        class_df[
                            "wrong_known_attack_family"
                        ].sum()
                    ),

                "predicted_anomalous":
                    int(
                        class_df[
                            "predicted_anomalous"
                        ].sum()
                    ),

                "rf_benign_on_misses":
                    int(
                        class_missed[
                            "rf_benign_vote"
                        ].sum()
                    ),

                "fedprox_benign_on_misses":
                    int(
                        class_missed[
                            "fedprox_benign_vote"
                        ].sum()
                    ),

                "cnn_benign_on_misses":
                    int(
                        class_missed[
                            "cnn_benign_vote"
                        ].sum()
                    ),

                "global_benign_consensus_on_misses":
                    int(
                        class_missed[
                            "global_supervised_benign_consensus"
                        ].sum()
                    ),

                "ae_anomaly_on_misses":
                    int(
                        class_missed[
                            "ae_anomaly"
                        ].sum()
                    ),

                "mean_ae_score_on_misses":
                    (
                        float(
                            class_missed[
                                "ae_anomaly_score"
                            ].mean()
                        )
                        if len(
                            class_missed
                        )
                        else None
                    ),
            }
        )

    by_class = pd.DataFrame(
        class_rows
    )

    # --------------------------------------------------------
    # Per-client false-negative summary
    # --------------------------------------------------------
    client_rows = []

    for client_id in range(
        1,
        NUM_CLIENTS + 1,
    ):
        client_df = df[
            df[
                "assigned_client"
            ]
            ==
            client_id
        ]

        client_missed = client_df[
            client_df[
                "missed_attack_to_benign"
            ]
        ]

        client_rows.append(
            {
                "assigned_client":
                    client_id,

                "total_sequences":
                    int(
                        len(
                            client_df
                        )
                    ),

                "missed_to_benign":
                    int(
                        len(
                            client_missed
                        )
                    ),

                "miss_rate":
                    float(
                        len(
                            client_missed
                        )
                        /
                        max(
                            len(
                                client_df
                            ),
                            1,
                        )
                    ),

                "correct":
                    int(
                        client_df[
                            "correct"
                        ].sum()
                    ),

                "accuracy":
                    float(
                        client_df[
                            "correct"
                        ].mean()
                    )
                    if len(
                        client_df
                    )
                    else None,
            }
        )

    by_client = pd.DataFrame(
        client_rows
    )

    # --------------------------------------------------------
    # Model behavior specifically on missed attacks
    # --------------------------------------------------------
    missed_count = len(
        missed
    )

    def count_rate(
        column: str,
    ):
        count = int(
            missed[
                column
            ].sum()
        )

        rate = (
            float(
                count
                /
                missed_count
            )
            if missed_count
            else 0.0
        )

        return (
            count,
            rate,
        )

    behavior_rows = []

    behavior_specs = [
        (
            "RF predicted BENIGN",
            "rf_benign_vote",
        ),
        (
            "FedProx70 predicted BENIGN",
            "fedprox_benign_vote",
        ),
        (
            "CNN-BiLSTM predicted BENIGN",
            "cnn_benign_vote",
        ),
        (
            "FedProx70 + CNN both BENIGN",
            "global_supervised_benign_consensus",
        ),
        (
            "RF + FedProx70 + CNN all BENIGN",
            "all_supervised_benign_consensus",
        ),
        (
            "AE flagged anomaly",
            "ae_anomaly",
        ),
    ]

    for label, column in behavior_specs:
        count, rate = count_rate(
            column
        )

        behavior_rows.append(
            {
                "pattern":
                    label,

                "count":
                    count,

                "share_of_221_style_misses":
                    rate,
            }
        )

    model_behavior = pd.DataFrame(
        behavior_rows
    )

    # --------------------------------------------------------
    # Routing diagnostic
    # specialist metadata is used only after inference.
    # --------------------------------------------------------
    routing_rows = []

    if (
        "routing_matches_specialist_analysis_only"
        in df.columns
        and
        df[
            "routing_matches_specialist_analysis_only"
        ].notna().any()
    ):
        for route_match in (
            True,
            False,
        ):
            route_df = df[
                df[
                    "routing_matches_specialist_analysis_only"
                ]
                ==
                route_match
            ]

            route_missed = route_df[
                route_df[
                    "missed_attack_to_benign"
                ]
            ]

            routing_rows.append(
                {
                    "routing_group":
                        (
                            "assigned client matches attack specialist"
                            if route_match
                            else
                            "assigned client does NOT match attack specialist"
                        ),

                    "analysis_only":
                        True,

                    "total_sequences":
                        int(
                            len(
                                route_df
                            )
                        ),

                    "correct":
                        int(
                            route_df[
                                "correct"
                            ].sum()
                        ),

                    "accuracy":
                        float(
                            route_df[
                                "correct"
                            ].mean()
                        )
                        if len(
                            route_df
                        )
                        else None,

                    "missed_to_benign":
                        int(
                            len(
                                route_missed
                            )
                        ),

                    "miss_rate":
                        float(
                            len(
                                route_missed
                            )
                            /
                            max(
                                len(
                                    route_df
                                ),
                                1,
                            )
                        ),
                }
            )

    routing_analysis = pd.DataFrame(
        routing_rows
    )

    return (
        missed,
        by_class,
        by_client,
        model_behavior,
        routing_analysis,
    )


def save_confusion(
    df: pd.DataFrame,
    inverse_mapping: dict[int, str],
):
    labels = list(
        range(
            NUM_KNOWN_CLASSES
        )
    ) + [
        ANOMALOUS_ID
    ]

    cm = confusion_matrix(
        df[
            "true_class_id"
        ].to_numpy(
            dtype=np.int64
        ),

        df[
            "final_prediction_id"
        ].to_numpy(
            dtype=np.int64
        ),

        labels=
            labels,
    )

    names = [
        class_name(
            class_id,
            inverse_mapping,
        )
        for class_id
        in labels
    ]

    cm_df = pd.DataFrame(
        cm,
        index=[
            f"TRUE_{name}"
            for name
            in names
        ],
        columns=[
            f"PRED_{name}"
            for name
            in names
        ],
    )

    cm_df.to_csv(
        CONFUSION_CSV
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=
            PHASE_NAME
    )

    parser.add_argument(
        "--input",
        type=str,
        default=str(
            DEFAULT_INPUT
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=
            DEFAULT_BATCH_SIZE,
    )

    args = parser.parse_args()

    input_file = Path(
        args.input
    )

    if not input_file.is_absolute():
        input_file = (
            PROJECT_ROOT
            /
            input_file
        )

    input_file = input_file.resolve()

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be > 0."
        )

    set_seed(
        SEED
    )

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        label_mapping,
        inverse_mapping,
        active_features,
    ) = phase14g1.load_project_schema()

    if len(
        active_features
    ) != EXPECTED_FEATURES:
        raise RuntimeError(
            f"Expected 70 active features; found {len(active_features)}."
        )

    (
        X,
        y_true,
        validation_index,
        specialist_client_id,
    ) = load_npz(
        input_file
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    separator()

    print(
        PHASE_NAME
    )

    separator()

    print(
        f"Input NPZ                : {input_file}"
    )

    print(
        f"Device                   : {device}"
    )

    print(
        f"Sequences                : {len(y_true):,}"
    )

    print(
        f"Shape                    : {X.shape}"
    )

    print(
        "Primary routing          : deterministic round-robin"
    )

    print(
        "specialist_client_id     : ANALYSIS ONLY — not used for inference"
    )

    print(
        "Locked test used         : NO"
    )

    print(
        "Models retrained         : NO"
    )

    start = time.perf_counter()

    df = run_inference(
        X=
            X,

        y_true=
            y_true,

        validation_index=
            validation_index,

        specialist_client_id=
            specialist_client_id,

        label_mapping=
            label_mapping,

        inverse_mapping=
            inverse_mapping,

        batch_size=
            args.batch_size,

        device=
            device,
    )

    elapsed = (
        time.perf_counter()
        -
        start
    )

    (
        missed,
        by_class,
        by_client,
        model_behavior,
        routing_analysis,
    ) = build_summaries(
        df=
            df,

        inverse_mapping=
            inverse_mapping,
    )

    df.to_csv(
        ALL_PREDICTIONS_CSV,
        index=False,
    )

    missed.to_csv(
        MISSED_ATTACKS_CSV,
        index=False,
    )

    by_class.to_csv(
        MISSED_BY_CLASS_CSV,
        index=False,
    )

    by_client.to_csv(
        MISSED_BY_CLIENT_CSV,
        index=False,
    )

    model_behavior.to_csv(
        MODEL_BEHAVIOR_CSV,
        index=False,
    )

    routing_analysis.to_csv(
        ROUTING_ANALYSIS_CSV,
        index=False,
    )

    save_confusion(
        df=
            df,

        inverse_mapping=
            inverse_mapping,
    )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------
    present_labels = sorted(
        int(
            value
        )
        for value
        in np.unique(
            y_true
        )
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            df[
                "final_prediction_id"
            ].to_numpy(
                dtype=np.int64
            ),
            labels=
                present_labels,
            zero_division=
                0,
        )
    )

    accuracy = float(
        accuracy_score(
            y_true,
            df[
                "final_prediction_id"
            ].to_numpy(
                dtype=np.int64
            ),
        )
    )

    balanced_accuracy = float(
        np.mean(
            recall
        )
    )

    macro_f1 = float(
        np.mean(
            f1
        )
    )

    total_errors = int(
        (
            ~df[
                "correct"
            ]
        ).sum()
    )

    missed_count = int(
        df[
            "missed_attack_to_benign"
        ].sum()
    )

    wrong_family_count = int(
        df[
            "wrong_known_attack_family"
        ].sum()
    )

    anomalous_count = int(
        df[
            "predicted_anomalous"
        ].sum()
    )

    summary = {
        "phase":
            "17H.4",

        "purpose":
            (
                "Analyze attack-to-BENIGN false negatives in the "
                "Phase 17H.2 five-attack sequence-preserving stress test."
            ),

        "input_file":
            str(
                input_file
            ),

        "sequence_count":
            int(
                len(
                    df
                )
            ),

        "sequence_shape":
            list(
                X.shape
            ),

        "accuracy":
            accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "macro_f1":
            macro_f1,

        "total_errors":
            total_errors,

        "missed_attacks_to_benign":
            missed_count,

        "wrong_known_attack_family":
            wrong_family_count,

        "predicted_anomalous":
            anomalous_count,

        "error_decomposition_check":
            {
                "total_errors":
                    total_errors,

                "missed_to_benign_plus_wrong_family_plus_anomalous":
                    (
                        missed_count
                        +
                        wrong_family_count
                        +
                        anomalous_count
                    ),
            },

        "missed_by_class":
            by_class.to_dict(
                orient="records"
            ),

        "model_behavior_on_misses":
            model_behavior.to_dict(
                orient="records"
            ),

        "routing_analysis":
            routing_analysis.to_dict(
                orient="records"
            ),

        "primary_routing_policy":
            (
                "deterministic round-robin; true label not used"
            ),

        "specialist_client_id_policy":
            (
                "analysis only; derived from known attack class; "
                "never used to route primary predictions"
            ),

        "locked_test_used":
            False,

        "models_retrained":
            False,

        "runtime_seconds":
            float(
                elapsed
            ),
    }

    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_safe(
                summary
            ),
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Console result
    # --------------------------------------------------------
    separator()

    print(
        "PHASE 17H.4 COMPLETE"
    )

    separator()

    print(
        f"Accuracy                 : {accuracy:.4%}"
    )

    print(
        f"Balanced Accuracy        : {balanced_accuracy:.4%}"
    )

    print(
        f"Macro F1                 : {macro_f1:.4%}"
    )

    print(
        f"Total errors             : {total_errors:,}"
    )

    print(
        f"Missed attacks -> BENIGN : {missed_count:,}"
    )

    print(
        f"Wrong attack family      : {wrong_family_count:,}"
    )

    print(
        f"Predicted ANOMALOUS      : {anomalous_count:,}"
    )

    print()

    print(
        "FALSE NEGATIVES BY TRUE ATTACK"
    )

    print(
        "-" * 120
    )

    print(
        by_class[
            [
                "true_class",
                "total_sequences",
                "missed_to_benign",
                "miss_rate",
                "wrong_attack_family",
                "predicted_anomalous",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()

    print(
        "MODEL BEHAVIOR ON MISSED ATTACKS"
    )

    print(
        "-" * 120
    )

    print(
        model_behavior.to_string(
            index=False
        )
    )

    if not routing_analysis.empty:
        print()

        print(
            "ROUTING DIAGNOSTIC — ANALYSIS ONLY"
        )

        print(
            "-" * 120
        )

        print(
            routing_analysis.to_string(
                index=False
            )
        )

    print()

    print(
        "OUTPUTS"
    )

    print(
        "-" * 120
    )

    print(
        f"All predictions          : {ALL_PREDICTIONS_CSV}"
    )

    print(
        f"Missed attacks detail    : {MISSED_ATTACKS_CSV}"
    )

    print(
        f"Missed by class          : {MISSED_BY_CLASS_CSV}"
    )

    print(
        f"Missed by client         : {MISSED_BY_CLIENT_CSV}"
    )

    print(
        f"Model behavior           : {MODEL_BEHAVIOR_CSV}"
    )

    print(
        f"Routing analysis         : {ROUTING_ANALYSIS_CSV}"
    )

    print(
        f"Confusion matrix         : {CONFUSION_CSV}"
    )

    print(
        f"Summary JSON             : {SUMMARY_JSON}"
    )

    print()

    print(
        "Locked test used         : NO"
    )

    print(
        "Models retrained         : NO"
    )


if __name__ == "__main__":
    main()
