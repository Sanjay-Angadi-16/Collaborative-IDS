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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


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
        "\nPlace this script inside the project's scripts folder.\n"
        "Required existing files:\n"
        "  scripts\\phase14g1_real_model_wiring.py\n"
        "  scripts\\phase14h_full_validation_adaptive_gate.py\n"
    ) from exc


# ============================================================
# CONFIG
# ============================================================

PHASE_NAME = (
    "PHASE 17H.1 — SAMPLING / CLIENT-ASSIGNMENT COMPARISON"
)

SEED = 42
NUM_CLASSES = 6
NUM_CLIENTS = 5
KNOWN_LABELS = [0, 1, 2, 3, 4, 5]
ANOMALOUS_ID = 6

EXPECTED_FEATURES = 70
SEQUENCE_LENGTH = 20

DEFAULT_SAMPLES = 1000
DEFAULT_BATCH_SIZE = 2048

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "dashboard"
    / "phase17h1_comparison"
)

SUMMARY_CSV = (
    RESULT_ROOT
    / "phase17h1_experiment_summary.csv"
)

CLASS_DISTRIBUTION_CSV = (
    RESULT_ROOT
    / "phase17h1_class_distribution.csv"
)

CLIENT_DETAIL_CSV = (
    RESULT_ROOT
    / "phase17h1_phase14h_style_client_metrics.csv"
)

SUMMARY_JSON = (
    RESULT_ROOT
    / "phase17h1_comparison_summary.json"
)


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


def separator(width: int = 120) -> None:
    print()
    print("=" * width)


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


# ============================================================
# VALIDATION DATA
# ============================================================

def load_full_validation():
    validation_file = (
        phase14h.VALIDATION_SEQUENCE_FILE
    )

    if not validation_file.exists():
        raise FileNotFoundError(
            "\nMissing validation file:\n"
            f"{validation_file}"
        )

    with np.load(
        validation_file,
        allow_pickle=False,
    ) as data:
        X = data["X"].astype(
            np.float32,
            copy=False,
        )

        y = data["y"].astype(
            np.int64,
            copy=False,
        )

    if X.ndim != 3:
        raise RuntimeError(
            f"Expected 3D validation X; found {X.shape}."
        )

    if tuple(X.shape[1:]) != (
        SEQUENCE_LENGTH,
        EXPECTED_FEATURES,
    ):
        raise RuntimeError(
            "Expected validation sequence shape "
            f"(*,{SEQUENCE_LENGTH},{EXPECTED_FEATURES}); "
            f"found {X.shape}."
        )

    if len(X) != len(y):
        raise RuntimeError(
            "Validation X/y count mismatch."
        )

    return X, y, validation_file


def balanced_indices(
    y: np.ndarray,
    requested: int,
) -> np.ndarray:
    """
    Current dashboard sampling:
    approximately equal number from each of the six classes.
    """
    requested = min(
        int(requested),
        len(y),
    )

    rng = np.random.default_rng(
        SEED
    )

    selected: list[int] = []

    per_class = max(
        1,
        requested // NUM_CLASSES,
    )

    for class_id in range(
        NUM_CLASSES
    ):
        class_indices = np.flatnonzero(
            y == class_id
        )

        take = min(
            per_class,
            len(class_indices),
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

    if len(selected) < requested:
        already = set(
            selected
        )

        remaining = np.asarray(
            [
                index
                for index in range(
                    len(y)
                )
                if index not in already
            ],
            dtype=np.int64,
        )

        extra_count = min(
            requested - len(selected),
            len(remaining),
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
        sorted(
            selected[:requested]
        ),
        dtype=np.int64,
    )


def natural_random_indices(
    y: np.ndarray,
    requested: int,
) -> np.ndarray:
    """
    Uniform random sample of validation rows.
    This approximately preserves the natural validation class distribution.
    """
    requested = min(
        int(requested),
        len(y),
    )

    rng = np.random.default_rng(
        SEED
    )

    selected = rng.choice(
        np.arange(
            len(y),
            dtype=np.int64,
        ),
        size=requested,
        replace=False,
    )

    return np.asarray(
        sorted(
            selected.tolist()
        ),
        dtype=np.int64,
    )


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    anomalous_count = int(
        np.sum(
            y_pred == ANOMALOUS_ID
        )
    )

    anomalous_rate = float(
        anomalous_count
        /
        max(
            len(y_pred),
            1,
        )
    )

    macro_precision = float(
        precision_score(
            y_true,
            y_pred,
            labels=KNOWN_LABELS,
            average="macro",
            zero_division=0,
        )
    )

    macro_recall = float(
        recall_score(
            y_true,
            y_pred,
            labels=KNOWN_LABELS,
            average="macro",
            zero_division=0,
        )
    )

    return {
        "samples_or_exposures":
            int(
                len(y_true)
            ),

        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "balanced_accuracy":
            macro_recall,

        "macro_precision":
            macro_precision,

        "macro_recall":
            macro_recall,

        "macro_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=KNOWN_LABELS,
                    average="macro",
                    zero_division=0,
                )
            ),

        "weighted_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=KNOWN_LABELS,
                    average="weighted",
                    zero_division=0,
                )
            ),

        "anomalous_predictions":
            anomalous_count,

        "anomalous_rate":
            anomalous_rate,
    }


def distribution_rows(
    experiment: str,
    y: np.ndarray,
    inverse_mapping: dict[int, str],
) -> list[dict]:
    rows = []

    for class_id in KNOWN_LABELS:
        count = int(
            np.sum(
                y == class_id
            )
        )

        rows.append(
            {
                "experiment":
                    experiment,

                "class_id":
                    class_id,

                "class_name":
                    class_name(
                        class_id,
                        inverse_mapping,
                    ),

                "rows":
                    count,

                "share":
                    float(
                        count
                        /
                        max(
                            len(y),
                            1,
                        )
                    ),
            }
        )

    return rows


# ============================================================
# GLOBAL MODELS
# ============================================================

def run_global_models(
    sequences: np.ndarray,
    flows: np.ndarray,
    inverse_mapping: dict[int, str],
    batch_size: int,
    device: torch.device,
):
    fedprox70 = (
        phase14g1.RealFedProx70Predictor(
            checkpoint_path=
                phase14g1.FEDPROX70_CHECKPOINT,

            inverse_mapping=
                inverse_mapping,

            device=
                device,
        )
    )

    cnn = (
        phase14g1.build_real_cnn_predictor(
            inverse_mapping=
                inverse_mapping,

            device=
                device,
        )
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
            sequences,

        batch_size=
            batch_size,

        device=
            device,
    )

    return (
        fed_pred,
        fed_conf,
        cnn_pred,
        cnn_conf,
    )


# ============================================================
# CURRENT ROUND-ROBIN REPLAY
# ============================================================

def run_round_robin_experiment(
    *,
    experiment_name: str,
    sequences: np.ndarray,
    y_true: np.ndarray,
    label_mapping: dict[str, int],
    inverse_mapping: dict[int, str],
    batch_size: int,
    device: torch.device,
) -> dict:
    flows = np.asarray(
        sequences[
            :,
            -1,
            :,
        ],
        dtype=np.float32,
    )

    (
        fed_pred,
        fed_conf,
        cnn_pred,
        cnn_conf,
    ) = run_global_models(
        sequences=
            sequences,

        flows=
            flows,

        inverse_mapping=
            inverse_mapping,

        batch_size=
            batch_size,

        device=
            device,
    )

    n = len(
        y_true
    )

    # Same assignment used by dashboard replay:
    # true class is deliberately not used.
    assigned_clients = (
        np.arange(
            n
        )
        %
        NUM_CLIENTS
    ) + 1

    adaptive_pred = np.empty(
        n,
        dtype=np.int64,
    )

    processing_fast = np.zeros(
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

        rf = (
            phase14g1.RealRandomForestPredictor(
                client_id=
                    client_id,

                label_mapping=
                    label_mapping,

                inverse_mapping=
                    inverse_mapping,
            )
        )

        ae = (
            phase14g1.RealPersonalizedAutoencoderPredictor(
                client_id=
                    client_id,

                device=
                    device,

                explicit_model_path=
                    None,
            )
        )

        client_flows = flows[
            positions
        ]

        (
            rf_pred,
            rf_conf,
        ) = phase14h.predict_rf_batch(
            rf=
                rf,

            flows=
                client_flows,
        )

        (
            _ae_error,
            ae_score,
            _ae_anomaly,
        ) = phase14h.predict_ae_batch(
            ae=
                ae,

            flows=
                client_flows,

            batch_size=
                batch_size,

            device=
                device,
        )

        full_result = (
            phase14h.full_fusion_vectorized(
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
        )

        adaptive_result = (
            phase14h.adaptive_policy_vectorized(
                rf_pred=
                    rf_pred,

                rf_conf=
                    rf_conf,

                full_result=
                    full_result,
            )
        )

        adaptive_pred[
            positions
        ] = adaptive_result[
            "prediction"
        ]

        processing_fast[
            positions
        ] = adaptive_result[
            "fast_mask"
        ]

    metrics = compute_metrics(
        y_true=
            y_true,

        y_pred=
            adaptive_pred,
    )

    metrics.update(
        {
            "experiment":
                experiment_name,

            "unique_validation_samples":
                int(
                    n
                ),

            "client_exposures":
                int(
                    n
                ),

            "client_assignment":
                (
                    "deterministic round-robin; "
                    "true label not used"
                ),

            "fast_path_count":
                int(
                    np.sum(
                        processing_fast
                    )
                ),

            "fast_path_rate":
                float(
                    np.mean(
                        processing_fast
                    )
                ),

            "deep_path_count":
                int(
                    np.sum(
                        ~processing_fast
                    )
                ),
        }
    )

    return metrics


# ============================================================
# PHASE14H-STYLE ALL-CLIENT REPLAY
# ============================================================

def run_phase14h_style_experiment(
    *,
    experiment_name: str,
    sequences: np.ndarray,
    y_true: np.ndarray,
    label_mapping: dict[str, int],
    inverse_mapping: dict[int, str],
    batch_size: int,
    device: torch.device,
):
    """
    Mirrors the key Phase14H evaluation idea:
    the same validation subset is replayed independently through
    all five client-local RF/AE stacks.

    Therefore 1000 unique validation samples become 5000 client exposures.
    """

    flows = np.asarray(
        sequences[
            :,
            -1,
            :,
        ],
        dtype=np.float32,
    )

    (
        fed_pred,
        fed_conf,
        cnn_pred,
        cnn_conf,
    ) = run_global_models(
        sequences=
            sequences,

        flows=
            flows,

        inverse_mapping=
            inverse_mapping,

        batch_size=
            batch_size,

        device=
            device,
    )

    pooled_y = []
    pooled_pred = []

    client_rows = []

    pooled_fast_count = 0

    for client_id in range(
        1,
        NUM_CLIENTS + 1,
    ):
        rf = (
            phase14g1.RealRandomForestPredictor(
                client_id=
                    client_id,

                label_mapping=
                    label_mapping,

                inverse_mapping=
                    inverse_mapping,
            )
        )

        ae = (
            phase14g1.RealPersonalizedAutoencoderPredictor(
                client_id=
                    client_id,

                device=
                    device,

                explicit_model_path=
                    None,
            )
        )

        (
            rf_pred,
            rf_conf,
        ) = phase14h.predict_rf_batch(
            rf=
                rf,

            flows=
                flows,
        )

        (
            _ae_error,
            ae_score,
            _ae_anomaly,
        ) = phase14h.predict_ae_batch(
            ae=
                ae,

            flows=
                flows,

            batch_size=
                batch_size,

            device=
                device,
        )

        full_result = (
            phase14h.full_fusion_vectorized(
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
        )

        adaptive_result = (
            phase14h.adaptive_policy_vectorized(
                rf_pred=
                    rf_pred,

                rf_conf=
                    rf_conf,

                full_result=
                    full_result,
            )
        )

        pred = adaptive_result[
            "prediction"
        ]

        fast_mask = adaptive_result[
            "fast_mask"
        ]

        client_metrics = compute_metrics(
            y_true=
                y_true,

            y_pred=
                pred,
        )

        client_metrics.update(
            {
                "client_id":
                    client_id,

                "seen_attack":
                    phase14g1.CLIENT_ATTACKS[
                        client_id
                    ],

                "fast_path_count":
                    int(
                        np.sum(
                            fast_mask
                        )
                    ),

                "fast_path_rate":
                    float(
                        np.mean(
                            fast_mask
                        )
                    ),
            }
        )

        client_rows.append(
            client_metrics
        )

        pooled_y.append(
            y_true
        )

        pooled_pred.append(
            pred
        )

        pooled_fast_count += int(
            np.sum(
                fast_mask
            )
        )

    pooled_y = np.concatenate(
        pooled_y
    )

    pooled_pred = np.concatenate(
        pooled_pred
    )

    pooled_metrics = compute_metrics(
        y_true=
            pooled_y,

        y_pred=
            pooled_pred,
    )

    pooled_metrics.update(
        {
            "experiment":
                experiment_name,

            "unique_validation_samples":
                int(
                    len(
                        y_true
                    )
                ),

            "client_exposures":
                int(
                    len(
                        pooled_y
                    )
                ),

            "client_assignment":
                (
                    "Phase14H-style: same validation subset "
                    "replayed independently through all 5 clients"
                ),

            "fast_path_count":
                pooled_fast_count,

            "fast_path_rate":
                float(
                    pooled_fast_count
                    /
                    max(
                        len(
                            pooled_y
                        ),
                        1,
                    )
                ),

            "deep_path_count":
                int(
                    len(
                        pooled_y
                    )
                    -
                    pooled_fast_count
                ),
        }
    )

    return (
        pooled_metrics,
        client_rows,
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
        "--samples",
        type=int,
        default=
            DEFAULT_SAMPLES,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=
            DEFAULT_BATCH_SIZE,
    )

    args = parser.parse_args()

    if args.samples <= 0:
        raise ValueError(
            "--samples must be > 0"
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be > 0"
        )

    set_seed(
        SEED
    )

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

    (
        X_all,
        y_all,
        validation_file,
    ) = load_full_validation()

    requested = min(
        args.samples,
        len(
            y_all
        ),
    )

    balanced_idx = balanced_indices(
        y=
            y_all,

        requested=
            requested,
    )

    natural_idx = natural_random_indices(
        y=
            y_all,

        requested=
            requested,
    )

    X_balanced = X_all[
        balanced_idx
    ]

    y_balanced = y_all[
        balanced_idx
    ]

    X_natural = X_all[
        natural_idx
    ]

    y_natural = y_all[
        natural_idx
    ]

    separator()

    print(
        PHASE_NAME
    )

    separator()

    print(
        f"Device                  : {device}"
    )

    print(
        f"Validation source       : {validation_file}"
    )

    print(
        f"Full validation rows    : {len(y_all):,}"
    )

    print(
        f"Samples per experiment  : {requested:,}"
    )

    print(
        "Locked test used        : NO"
    )

    print()

    print(
        "EXPERIMENT A"
    )

    print(
        "Balanced 6-class sampling + round-robin client assignment"
    )

    start = time.perf_counter()

    exp_a = run_round_robin_experiment(
        experiment_name=
            "A_balanced_round_robin",

        sequences=
            X_balanced,

        y_true=
            y_balanced,

        label_mapping=
            label_mapping,

        inverse_mapping=
            inverse_mapping,

        batch_size=
            args.batch_size,

        device=
            device,
    )

    time_a = (
        time.perf_counter()
        -
        start
    )

    print(
        f"Completed in             : {time_a:.2f}s"
    )

    print()

    print(
        "EXPERIMENT B"
    )

    print(
        "Natural random validation sampling + round-robin client assignment"
    )

    start = time.perf_counter()

    exp_b = run_round_robin_experiment(
        experiment_name=
            "B_natural_round_robin",

        sequences=
            X_natural,

        y_true=
            y_natural,

        label_mapping=
            label_mapping,

        inverse_mapping=
            inverse_mapping,

        batch_size=
            args.batch_size,

        device=
            device,
    )

    time_b = (
        time.perf_counter()
        -
        start
    )

    print(
        f"Completed in             : {time_b:.2f}s"
    )

    print()

    print(
        "EXPERIMENT C"
    )

    print(
        "Natural random validation sampling + Phase14H-style all-client replay"
    )

    start = time.perf_counter()

    (
        exp_c,
        client_rows,
    ) = run_phase14h_style_experiment(
        experiment_name=
            "C_natural_phase14h_all_clients",

        sequences=
            X_natural,

        y_true=
            y_natural,

        label_mapping=
            label_mapping,

        inverse_mapping=
            inverse_mapping,

        batch_size=
            args.batch_size,

        device=
            device,
    )

    time_c = (
        time.perf_counter()
        -
        start
    )

    print(
        f"Completed in             : {time_c:.2f}s"
    )

    # --------------------------------------------------------
    # Save experiment summary
    # --------------------------------------------------------
    summary_rows = [
        exp_a,
        exp_b,
        exp_c,
    ]

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        SUMMARY_CSV,
        index=False,
    )

    client_df = pd.DataFrame(
        client_rows
    )

    client_df.to_csv(
        CLIENT_DETAIL_CSV,
        index=False,
    )

    distribution = []

    distribution.extend(
        distribution_rows(
            experiment=
                "A_balanced_round_robin",

            y=
                y_balanced,

            inverse_mapping=
                inverse_mapping,
        )
    )

    distribution.extend(
        distribution_rows(
            experiment=
                "B_natural_round_robin",

            y=
                y_natural,

            inverse_mapping=
                inverse_mapping,
        )
    )

    distribution.extend(
        distribution_rows(
            experiment=
                "C_natural_phase14h_all_clients_unique_sample_distribution",

            y=
                y_natural,

            inverse_mapping=
                inverse_mapping,
        )
    )

    pd.DataFrame(
        distribution
    ).to_csv(
        CLASS_DISTRIBUTION_CSV,
        index=False,
    )

    payload = {
        "phase":
            "17H.1",

        "purpose":
            (
                "Diagnose whether the dashboard replay accuracy difference "
                "is driven by class sampling, round-robin client assignment, "
                "or the underlying frozen detection stack."
            ),

        "seed":
            SEED,

        "validation_source":
            str(
                validation_file
            ),

        "requested_samples":
            requested,

        "locked_test_used":
            False,

        "experiments":
            summary_rows,

        "phase14h_style_client_metrics":
            client_rows,

        "runtime_seconds":
            {
                "experiment_a":
                    time_a,

                "experiment_b":
                    time_b,

                "experiment_c":
                    time_c,
            },
    }

    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_safe(
                payload
            ),
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Console comparison
    # --------------------------------------------------------
    separator()

    print(
        "COMPARISON COMPLETE"
    )

    separator()

    display_columns = [
        "experiment",
        "unique_validation_samples",
        "client_exposures",
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "anomalous_predictions",
        "anomalous_rate",
        "fast_path_rate",
    ]

    print(
        summary_df[
            display_columns
        ]
        .to_string(
            index=False
        )
    )

    print()

    print(
        "PHASE14H-STYLE PER-CLIENT DETAIL"
    )

    print(
        "-" * 120
    )

    print(
        client_df[
            [
                "client_id",
                "seen_attack",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "anomalous_predictions",
                "anomalous_rate",
                "fast_path_rate",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()

    print(
        f"Summary CSV            : {SUMMARY_CSV}"
    )

    print(
        f"Class distribution CSV : {CLASS_DISTRIBUTION_CSV}"
    )

    print(
        f"Client detail CSV      : {CLIENT_DETAIL_CSV}"
    )

    print(
        f"Summary JSON           : {SUMMARY_JSON}"
    )

    print()

    print(
        "LOCKED TEST USED       : NO"
    )


if __name__ == "__main__":
    main()
