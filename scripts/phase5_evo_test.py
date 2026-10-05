# ============================================================
# file: scripts/phase5_evo_test.py
# ============================================================
#
# PHASE 5 — INDEPENDENT EVO FEATURE-SELECTION TEST
#
# Research flow:
#
#   train.csv
#       ↓
#   remove labels / metadata
#       ↓
#   remove 8 zero-variance features
#       ↓
#   exactly 70 Phase-4-compatible features
#       ↓
#   stratified optimization subset FROM TRAIN ONLY
#       ↓
#   internal optimization train / validation split
#       ↓
#   Energy Valley Optimizer
#       ↓
#   continuous particle
#       ↓
#   EVOFeatureSelector
#       ↓
#   binary feature mask
#       ↓
#   ExtraTrees fitness proxy
#       ↓
#   Macro F1 + Balanced Accuracy + Feature Reduction
#       ↓
#   best feature subset
#       ↓
#   retrain using FULL training.csv
#       ↓
#   evaluate on official validation.csv
#
# IMPORTANT:
#
#   test_LOCKED.csv IS NEVER LOADED.
#
#   Official validation.csv is NOT used during EVO search.
#   It is used only after the EVO subset has been frozen.
#
# Outputs:
#
#   results/phase5_evo/
#       evo_selected_features.csv
#       evo_feature_mask.csv
#       evo_convergence.csv
#       evo_comparison.csv
#       phase5_evo_summary.json
#
# ============================================================


from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from feature_selection.evo_feature_selector import (
    EVOFeatureSelector,
)

from optimization.evo import (
    EnergyValleyOptimizer,
)


# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

RANDOM_STATE = 42

EXPECTED_FEATURE_COUNT = 70


# ============================================================
# DATA PATHS
# ============================================================

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train.csv"
)

VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation.csv"
)

LOCKED_TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_LOCKED.csv"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "phase5_evo"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# TARGET
# ============================================================

TARGET_COLUMN = "Label"


# ============================================================
# NON-FEATURE COLUMNS
# ============================================================

NON_FEATURE_COLUMNS = {

    # Target aliases
    "Label",
    "label",
    "LABEL",

    # Label-derived metadata
    "fine_label",
    "family_label",
    "original_label",

    # Provenance
    "source_file",
    "source_day",
    "flow_id",
    "record_id",
    "stable_flow_id",

    # Experiment metadata
    "split",
    "client",
    "client_id",
    "scenario",
}


# ============================================================
# ZERO-VARIANCE FEATURES
#
# Verified from training.csv.
#
# 78 numerical CICIDS features
# - 8 zero-variance features
# = 70 Phase-4-compatible ML features
# ============================================================

ZERO_VARIANCE_FEATURES = {

    "Bwd PSH Flags",
    "Bwd URG Flags",

    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",

    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
}


# ============================================================
# EVO CONFIGURATION
# ============================================================

EVO_POPULATION_SIZE = 20
EVO_ITERATIONS = 30

# Continuous search domain.
#
# Selector uses:
#
# sigmoid(position) >= 0.5
#
# Therefore negative values naturally represent feature=0,
# while positive values represent feature=1.

EVO_LOWER_BOUND = -6.0
EVO_UPPER_BOUND = 6.0


# ============================================================
# FITNESS WEIGHTS
# ============================================================

MACRO_F1_WEIGHT = 0.50

BALANCED_ACCURACY_WEIGHT = 0.35

FEATURE_REDUCTION_WEIGHT = 0.15


# ============================================================
# SEARCH PROXY CONFIGURATION
#
# DO NOT run hundreds of ExtraTrees fits on 1.75 million
# samples during evolutionary search.
#
# EVO searches using a reproducible stratified subset of the
# TRAIN partition only.
#
# After feature selection is frozen:
#
# FULL TRAIN
#       ↓
# selected features
#       ↓
# FULL OFFICIAL VALIDATION
#
# is used for final Phase-5 verification.
# ============================================================

OPTIMIZATION_SAMPLE_ROWS = 120_000

OPTIMIZATION_VALIDATION_FRACTION = 0.25

SEARCH_N_ESTIMATORS = 40

FINAL_N_ESTIMATORS = 80


# ============================================================
# UTILITY
# ============================================================

def separator():

    print(
        "\n"
        + "=" * 70
    )


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_random_seed(
    seed: int,
):

    np.random.seed(seed)


# ============================================================
# LOCKED TEST PROTECTION
# ============================================================

def assert_locked_test_not_used(
    paths,
):

    locked_path = (
        LOCKED_TEST_FILE
        .resolve()
    )

    for path in paths:

        candidate = (
            Path(path)
            .resolve()
        )

        if candidate == locked_path:

            raise RuntimeError(
                "\n"
                "DATA LEAKAGE DETECTED.\n\n"
                "test_LOCKED.csv cannot be used "
                "during Phase 5 feature optimization."
            )


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(
    path: Path,
):

    if not path.exists():

        raise FileNotFoundError(
            f"\nDataset not found:\n{path}\n"
        )

    print(
        f"Loading: {path}"
    )

    df = pd.read_csv(path)

    print(
        f"Rows     : {len(df):,}"
    )

    print(
        f"Columns  : {len(df.columns)}"
    )

    return df


# ============================================================
# DETECT TARGET
# ============================================================

def detect_target_column(
    train_df,
    val_df,
):

    candidates = [

        TARGET_COLUMN,

        "Label",
        "label",
        "LABEL",

        "target",
        "Target",

        "class",
        "Class",

        "attack",
        "attack_label",

        "label_encoded",
    ]

    for column in candidates:

        if (
            column in train_df.columns
            and
            column in val_df.columns
        ):
            return column

    raise ValueError(
        "\nCould not detect target column.\n"
        f"Available columns:\n"
        f"{list(train_df.columns)}"
    )


# ============================================================
# VERIFY ZERO-VARIANCE FEATURES
# ============================================================

def verify_zero_variance_features(
    train_df,
):

    print(
        "\nZero-variance features excluded:"
    )

    for feature in sorted(
        ZERO_VARIANCE_FEATURES
    ):

        if feature not in train_df.columns:

            raise ValueError(
                f"\nExpected zero-variance feature "
                f"missing from training data:\n"
                f"{feature}"
            )

        unique_count = (
            train_df[feature]
            .nunique(
                dropna=False
            )
        )

        print(
            f"  {feature:<30} "
            f"unique values = {unique_count}"
        )

        if unique_count > 1:

            raise ValueError(
                f"\nFeature '{feature}' is no longer "
                f"zero-variance.\n"
                "The preprocessing schema may have changed.\n"
                "Do not silently exclude it."
            )


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(
    train_df,
    val_df,
):

    target_column = (
        detect_target_column(
            train_df,
            val_df,
        )
    )

    print(
        f"\nTarget column: "
        f"{target_column}"
    )

    # --------------------------------------------------------
    # Verify known Phase-4 zero-variance features
    # --------------------------------------------------------

    verify_zero_variance_features(
        train_df
    )

    # --------------------------------------------------------
    # Construct exclusion set
    # --------------------------------------------------------

    drop_columns = set(
        NON_FEATURE_COLUMNS
    )

    drop_columns.add(
        target_column
    )

    drop_columns.update(
        ZERO_VARIANCE_FEATURES
    )

    # --------------------------------------------------------
    # Candidate feature schema
    # --------------------------------------------------------

    feature_columns = [

        column

        for column
        in train_df.columns

        if column
        not in drop_columns
    ]

    print(
        f"\nCandidate ML features: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Exact Phase-4 compatibility
    # --------------------------------------------------------

    if (
        len(feature_columns)
        !=
        EXPECTED_FEATURE_COUNT
    ):

        print(
            "\nCandidate feature columns:"
        )

        for i, column in enumerate(
            feature_columns,
            start=1,
        ):

            print(
                f"{i:02d}. {column}"
            )

        raise ValueError(
            "\nPhase 5 feature count does not match "
            f"Phase 4.\n"
            f"Expected : {EXPECTED_FEATURE_COUNT}\n"
            f"Found    : {len(feature_columns)}\n"
        )

    print(
        "Phase-4 feature compatibility: PASS"
    )

    # --------------------------------------------------------
    # Validation schema check
    # --------------------------------------------------------

    missing_validation_features = [

        column

        for column
        in feature_columns

        if column
        not in val_df.columns
    ]

    if missing_validation_features:

        raise ValueError(
            "\nValidation data is missing features:\n"
            f"{missing_validation_features}"
        )

    # --------------------------------------------------------
    # Numeric checks
    # --------------------------------------------------------

    train_non_numeric = [

        column

        for column
        in feature_columns

        if not pd.api.types.is_numeric_dtype(
            train_df[column]
        )
    ]

    val_non_numeric = [

        column

        for column
        in feature_columns

        if not pd.api.types.is_numeric_dtype(
            val_df[column]
        )
    ]

    if train_non_numeric:

        raise ValueError(
            "\nTraining features contain "
            "non-numeric columns:\n"
            f"{train_non_numeric}"
        )

    if val_non_numeric:

        raise ValueError(
            "\nValidation features contain "
            "non-numeric columns:\n"
            f"{val_non_numeric}"
        )

    # --------------------------------------------------------
    # Convert to float32
    # --------------------------------------------------------

    X_train = (
        train_df[
            feature_columns
        ]
        .to_numpy(
            dtype=np.float32,
            copy=True,
        )
    )

    X_val = (
        val_df[
            feature_columns
        ]
        .to_numpy(
            dtype=np.float32,
            copy=True,
        )
    )

    y_train = (
        train_df[
            target_column
        ]
        .to_numpy()
    )

    y_val = (
        val_df[
            target_column
        ]
        .to_numpy()
    )

    # --------------------------------------------------------
    # NaN / infinity checks
    # --------------------------------------------------------

    if not np.isfinite(
        X_train
    ).all():

        raise ValueError(
            "\nTraining feature matrix contains "
            "NaN or infinity."
        )

    if not np.isfinite(
        X_val
    ).all():

        raise ValueError(
            "\nValidation feature matrix contains "
            "NaN or infinity."
        )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        feature_columns,
        target_column,
    )


# ============================================================
# CREATE STRATIFIED EVO SEARCH DATA
# ============================================================

def build_evo_search_data(
    X_train,
    y_train,
):

    total_rows = len(
        y_train
    )

    print(
        "\nPreparing EVO optimization subset..."
    )

    # --------------------------------------------------------
    # Sample from TRAIN ONLY
    # --------------------------------------------------------

    if (
        total_rows
        >
        OPTIMIZATION_SAMPLE_ROWS
    ):

        all_indices = np.arange(
            total_rows
        )

        sample_indices, _ = (
            train_test_split(

                all_indices,

                train_size=
                    OPTIMIZATION_SAMPLE_ROWS,

                stratify=
                    y_train,

                random_state=
                    RANDOM_STATE,
            )
        )

        X_sample = (
            X_train[
                sample_indices
            ]
        )

        y_sample = (
            y_train[
                sample_indices
            ]
        )

    else:

        X_sample = X_train
        y_sample = y_train

    # --------------------------------------------------------
    # Internal train / fitness-validation split
    # --------------------------------------------------------

    (
        X_search_train,
        X_search_val,
        y_search_train,
        y_search_val,

    ) = train_test_split(

        X_sample,
        y_sample,

        test_size=
            OPTIMIZATION_VALIDATION_FRACTION,

        stratify=
            y_sample,

        random_state=
            RANDOM_STATE,
    )

    print(
        f"Optimization sample rows : "
        f"{len(X_sample):,}"
    )

    print(
        f"EVO search train rows    : "
        f"{len(X_search_train):,}"
    )

    print(
        f"EVO search val rows      : "
        f"{len(X_search_val):,}"
    )

    print(
        "Official validation used "
        "during EVO search : NO"
    )

    return (
        X_search_train,
        y_search_train,
        X_search_val,
        y_search_val,
    )


# ============================================================
# EVALUATE FEATURE SUBSET
# ============================================================

def evaluate_feature_subset(
    X_train,
    y_train,
    X_val,
    y_val,
    mask=None,
    n_estimators=FINAL_N_ESTIMATORS,
):

    if mask is None:

        X_train_selected = (
            X_train
        )

        X_val_selected = (
            X_val
        )

        selected_count = (
            X_train.shape[1]
        )

    else:

        mask = np.asarray(
            mask,
            dtype=bool,
        )

        X_train_selected = (
            X_train[:, mask]
        )

        X_val_selected = (
            X_val[:, mask]
        )

        selected_count = int(
            np.sum(mask)
        )

    model = (
        ExtraTreesClassifier(

            n_estimators=
                n_estimators,

            class_weight=
                "balanced",

            random_state=
                RANDOM_STATE,

            n_jobs=-1,
        )
    )

    start = (
        time.perf_counter()
    )

    model.fit(
        X_train_selected,
        y_train,
    )

    predictions = (
        model.predict(
            X_val_selected
        )
    )

    runtime = (
        time.perf_counter()
        -
        start
    )

    metrics = {

        "accuracy":
            float(
                accuracy_score(
                    y_val,
                    predictions,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y_val,
                    predictions,
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    y_val,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),

        "weighted_f1":
            float(
                f1_score(
                    y_val,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),

        "selected_count":
            int(
                selected_count
            ),

        "runtime_seconds":
            float(
                runtime
            ),
    }

    return metrics


# ============================================================
# RUN ENERGY VALLEY OPTIMIZER
# ============================================================

def run_evo(
    selector,
    n_features,
):

    print(
        "\nCreating Energy Valley Optimizer..."
    )

    optimizer = (
        EnergyValleyOptimizer(

            population_size=
                EVO_POPULATION_SIZE,

            max_iterations=
                EVO_ITERATIONS,

            random_state=
                RANDOM_STATE,

            verbose=True,
        )
    )

    print(
        "\nStarting Energy Valley Optimization..."
    )

    result = (
        optimizer.optimize(

            objective_function=
                selector.fitness,

            dimension=
                n_features,

            lower_bound=
                EVO_LOWER_BOUND,

            upper_bound=
                EVO_UPPER_BOUND,
        )
    )

    best_position = np.asarray(
        result.best_position,
        dtype=float,
    )

    best_fitness = float(
        result.best_fitness
    )

    convergence_history = list(
        result.convergence_history
    )

    print(
        "\nEVO optimization finished."
    )

    print(
        f"Iterations      : "
        f"{result.iterations}"
    )

    print(
        f"EVO evaluations : "
        f"{result.evaluations}"
    )

    print(
        f"Best fitness    : "
        f"{best_fitness:.6f}"
    )

    return (
        best_position,
        best_fitness,
        convergence_history,
        result,
    )


# ============================================================
# SAVE CONVERGENCE HISTORY
# ============================================================

def save_convergence_history(
    history,
):

    if history is None:
        return

    history = np.asarray(
        history,
        dtype=float,
    ).reshape(-1)

    if len(history) == 0:
        return

    convergence_df = (
        pd.DataFrame({

            "iteration":
                np.arange(
                    1,
                    len(history) + 1,
                ),

            "best_fitness":
                history,
        })
    )

    output_path = (
        OUTPUT_DIR
        / "evo_convergence.csv"
    )

    convergence_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved convergence : "
        f"{output_path}"
    )


# ============================================================
# SAVE SELECTED FEATURES
# ============================================================

def save_selected_features(
    best_result,
):

    selected_df = (
        pd.DataFrame({

            "feature_index":
                list(
                    best_result
                    .selected_indices
                ),

            "feature_name":
                list(
                    best_result
                    .selected_features
                ),
        })
    )

    output_path = (
        OUTPUT_DIR
        / "evo_selected_features.csv"
    )

    selected_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved features    : "
        f"{output_path}"
    )


# ============================================================
# SAVE FEATURE MASK
# ============================================================

def save_feature_mask(
    feature_names,
    mask,
):

    mask_df = (
        pd.DataFrame({

            "feature_index":
                np.arange(
                    len(feature_names)
                ),

            "feature_name":
                feature_names,

            "selected":
                np.asarray(
                    mask,
                    dtype=int,
                ),
        })
    )

    output_path = (
        OUTPUT_DIR
        / "evo_feature_mask.csv"
    )

    mask_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved mask        : "
        f"{output_path}"
    )


# ============================================================
# SAVE COMPARISON CSV
# ============================================================

def save_comparison(
    baseline,
    selected_metrics,
    total_features,
    selected_count,
):

    reduction = (
        1.0
        -
        selected_count
        /
        total_features
    )

    comparison_df = (
        pd.DataFrame([

            {
                "experiment":
                    "All 70 Features",

                "feature_count":
                    total_features,

                "feature_reduction":
                    0.0,

                "accuracy":
                    baseline[
                        "accuracy"
                    ],

                "balanced_accuracy":
                    baseline[
                        "balanced_accuracy"
                    ],

                "macro_f1":
                    baseline[
                        "macro_f1"
                    ],

                "weighted_f1":
                    baseline[
                        "weighted_f1"
                    ],

                "runtime_seconds":
                    baseline[
                        "runtime_seconds"
                    ],
            },

            {
                "experiment":
                    "EVO Selected Features",

                "feature_count":
                    selected_count,

                "feature_reduction":
                    reduction,

                "accuracy":
                    selected_metrics[
                        "accuracy"
                    ],

                "balanced_accuracy":
                    selected_metrics[
                        "balanced_accuracy"
                    ],

                "macro_f1":
                    selected_metrics[
                        "macro_f1"
                    ],

                "weighted_f1":
                    selected_metrics[
                        "weighted_f1"
                    ],

                "runtime_seconds":
                    selected_metrics[
                        "runtime_seconds"
                    ],
            },
        ])
    )

    output_path = (
        OUTPUT_DIR
        / "evo_comparison.csv"
    )

    comparison_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved comparison  : "
        f"{output_path}"
    )


# ============================================================
# SAVE JSON SUMMARY
# ============================================================

def save_summary(
    baseline,
    search_result,
    final_selected_metrics,
    optimization_runtime,
    selector,
    evo_result,
):

    macro_delta = (
        final_selected_metrics[
            "macro_f1"
        ]
        -
        baseline[
            "macro_f1"
        ]
    )

    balanced_delta = (
        final_selected_metrics[
            "balanced_accuracy"
        ]
        -
        baseline[
            "balanced_accuracy"
        ]
    )

    summary = {

        "phase":
            "Phase 5 - Independent EVO Feature Selection",

        "optimizer":
            "Energy Valley Optimizer",

        "random_state":
            RANDOM_STATE,

        "starting_feature_count":
            EXPECTED_FEATURE_COUNT,

        "zero_variance_features_removed":
            sorted(
                ZERO_VARIANCE_FEATURES
            ),

        "search_protocol": {

            "search_source":
                "training.csv only",

            "optimization_sample_rows":
                OPTIMIZATION_SAMPLE_ROWS,

            "optimization_validation_fraction":
                OPTIMIZATION_VALIDATION_FRACTION,

            "official_validation_used_during_search":
                False,

            "locked_test_used":
                False,
        },

        "optimizer_configuration": {

            "population_size":
                EVO_POPULATION_SIZE,

            "max_iterations":
                EVO_ITERATIONS,

            "lower_bound":
                EVO_LOWER_BOUND,

            "upper_bound":
                EVO_UPPER_BOUND,

            "optimizer_iterations":
                int(
                    evo_result.iterations
                ),

            "optimizer_evaluations":
                int(
                    evo_result.evaluations
                ),

            "classifier_evaluations":
                int(
                    selector.evaluation_count
                ),

            "cache_hits":
                int(
                    selector.cache_hits
                ),

            "optimization_runtime_seconds":
                float(
                    optimization_runtime
                ),
        },

        "fitness": {

            "objective":
                "minimize",

            "macro_f1_weight":
                MACRO_F1_WEIGHT,

            "balanced_accuracy_weight":
                BALANCED_ACCURACY_WEIGHT,

            "feature_reduction_weight":
                FEATURE_REDUCTION_WEIGHT,
        },

        "search_best_result": {

            "fitness":
                float(
                    search_result.fitness
                ),

            "utility":
                float(
                    search_result.utility
                ),

            "macro_f1":
                float(
                    search_result.macro_f1
                ),

            "balanced_accuracy":
                float(
                    search_result
                    .balanced_accuracy
                ),

            "feature_reduction":
                float(
                    search_result
                    .feature_reduction
                ),

            "selected_count":
                int(
                    search_result
                    .selected_count
                ),

            "selected_features":
                list(
                    search_result
                    .selected_features
                ),
        },

        "full_validation_baseline": baseline,

        "full_validation_evo_subset":
            final_selected_metrics,

        "full_validation_change": {

            "macro_f1_delta":
                float(
                    macro_delta
                ),

            "balanced_accuracy_delta":
                float(
                    balanced_delta
                ),
        },

        "locked_test_used":
            False,
    }

    output_path = (
        OUTPUT_DIR
        / "phase5_evo_summary.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    print(
        f"Saved summary     : "
        f"{output_path}"
    )


# ============================================================
# PRINT FINAL RESULT
# ============================================================

def print_final_result(
    baseline,
    search_result,
    final_selected_metrics,
    optimization_runtime,
    selector,
    evo_result,
):

    total_features = (
        search_result.total_features
    )

    selected_count = (
        search_result.selected_count
    )

    removed_count = (
        total_features
        -
        selected_count
    )

    feature_reduction = (
        search_result
        .feature_reduction
    )

    macro_delta = (
        final_selected_metrics[
            "macro_f1"
        ]
        -
        baseline[
            "macro_f1"
        ]
    )

    balanced_delta = (
        final_selected_metrics[
            "balanced_accuracy"
        ]
        -
        baseline[
            "balanced_accuracy"
        ]
    )

    separator()

    print(
        "PHASE 5 COMPLETE — "
        "INDEPENDENT EVO FEATURE SELECTION"
    )

    separator()

    print(
        f"\nOriginal features     : "
        f"{total_features}"
    )

    print(
        f"Selected features     : "
        f"{selected_count}"
    )

    print(
        f"Removed features      : "
        f"{removed_count}"
    )

    print(
        f"Feature reduction     : "
        f"{feature_reduction * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Search fitness result
    # --------------------------------------------------------

    print(
        "\n--- EVO SEARCH RESULT ---"
    )

    print(
        f"Search Macro F1       : "
        f"{search_result.macro_f1:.6f}"
    )

    print(
        f"Search Balanced Acc.  : "
        f"{search_result.balanced_accuracy:.6f}"
    )

    print(
        f"Search Utility        : "
        f"{search_result.utility:.6f}"
    )

    print(
        f"Search Fitness        : "
        f"{search_result.fitness:.6f}"
    )

    # --------------------------------------------------------
    # Full validation baseline
    # --------------------------------------------------------

    print(
        "\n--- FULL VALIDATION: 70-FEATURE BASELINE ---"
    )

    print(
        f"Accuracy              : "
        f"{baseline['accuracy']:.6f}"
    )

    print(
        f"Balanced Accuracy     : "
        f"{baseline['balanced_accuracy']:.6f}"
    )

    print(
        f"Macro F1              : "
        f"{baseline['macro_f1']:.6f}"
    )

    print(
        f"Weighted F1           : "
        f"{baseline['weighted_f1']:.6f}"
    )

    print(
        f"Runtime               : "
        f"{baseline['runtime_seconds']:.2f}s"
    )

    # --------------------------------------------------------
    # Full validation EVO subset
    # --------------------------------------------------------

    print(
        "\n--- FULL VALIDATION: EVO SUBSET ---"
    )

    print(
        f"Accuracy              : "
        f"{final_selected_metrics['accuracy']:.6f}"
    )

    print(
        f"Balanced Accuracy     : "
        f"{final_selected_metrics['balanced_accuracy']:.6f}"
    )

    print(
        f"Macro F1              : "
        f"{final_selected_metrics['macro_f1']:.6f}"
    )

    print(
        f"Weighted F1           : "
        f"{final_selected_metrics['weighted_f1']:.6f}"
    )

    print(
        f"Runtime               : "
        f"{final_selected_metrics['runtime_seconds']:.2f}s"
    )

    # --------------------------------------------------------
    # Delta
    # --------------------------------------------------------

    print(
        "\n--- EVO SUBSET VS 70-FEATURE BASELINE ---"
    )

    print(
        f"Macro F1 Delta        : "
        f"{macro_delta:+.6f}"
    )

    print(
        f"Balanced Acc. Delta   : "
        f"{balanced_delta:+.6f}"
    )

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    print(
        "\n--- OPTIMIZATION ---"
    )

    print(
        f"Population             : "
        f"{EVO_POPULATION_SIZE}"
    )

    print(
        f"Configured iterations  : "
        f"{EVO_ITERATIONS}"
    )

    print(
        f"Completed iterations   : "
        f"{evo_result.iterations}"
    )

    print(
        f"EVO evaluations        : "
        f"{evo_result.evaluations}"
    )

    print(
        f"Classifier evaluations : "
        f"{selector.evaluation_count}"
    )

    print(
        f"Cache hits             : "
        f"{selector.cache_hits}"
    )

    print(
        f"Optimization time      : "
        f"{optimization_runtime:.2f}s"
    )

    # --------------------------------------------------------
    # Selected features
    # --------------------------------------------------------

    print(
        "\n--- SELECTED FEATURES ---"
    )

    for number, feature in enumerate(
        search_result.selected_features,
        start=1,
    ):

        print(
            f"{number:02d}. {feature}"
        )

    separator()

    print(
        "\nPHASE-4 STARTING FEATURES : 70"
    )

    print(
        "LOCKED TEST SET USED      : NO"
    )

    print(
        "OFFICIAL VALIDATION IN EVO: NO"
    )

    print(
        "FEDAVG USED               : NO"
    )

    print(
        "EVO VERIFIED ALONE        : YES"
    )

    separator()


# ============================================================
# MAIN
# ============================================================

def main():

    set_random_seed(
        RANDOM_STATE
    )

    separator()

    print(
        "PHASE 5 — EVO FEATURE "
        "SELECTION INDEPENDENT TEST"
    )

    separator()

    print(
        f"\nProject root:\n"
        f"{PROJECT_ROOT}"
    )

    # ========================================================
    # 1. LOCKED TEST PROTECTION
    # ========================================================

    assert_locked_test_not_used([
        TRAIN_FILE,
        VALIDATION_FILE,
    ])

    print(
        "\nLocked test protection: PASS"
    )

    # ========================================================
    # 2. LOAD DATA
    # ========================================================

    train_df = load_dataset(
        TRAIN_FILE
    )

    val_df = load_dataset(
        VALIDATION_FILE
    )

    # ========================================================
    # 3. PREPARE EXACT 70-FEATURE SCHEMA
    # ========================================================

    (
        X_train,
        y_train,
        X_val,
        y_val,
        feature_names,
        target_column,

    ) = prepare_data(

        train_df,
        val_df,
    )

    # DataFrames no longer required.
    del train_df
    del val_df

    gc.collect()

    print(
        f"\nInput features : "
        f"{len(feature_names)}"
    )

    print(
        f"Train rows     : "
        f"{len(X_train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(X_val):,}"
    )

    print(
        f"Classes        : "
        f"{len(np.unique(y_train))}"
    )

    # ========================================================
    # 4. FULL 70-FEATURE BASELINE
    # ========================================================

    separator()

    print(
        "\nEvaluating full "
        "70-feature baseline..."
    )

    baseline = (
        evaluate_feature_subset(

            X_train=
                X_train,

            y_train=
                y_train,

            X_val=
                X_val,

            y_val=
                y_val,

            mask=None,

            n_estimators=
                FINAL_N_ESTIMATORS,
        )
    )

    print(
        f"\nBaseline Accuracy          : "
        f"{baseline['accuracy']:.6f}"
    )

    print(
        f"Baseline Macro F1          : "
        f"{baseline['macro_f1']:.6f}"
    )

    print(
        f"Baseline Balanced Accuracy : "
        f"{baseline['balanced_accuracy']:.6f}"
    )

    # ========================================================
    # 5. BUILD EVO SEARCH SUBSET FROM TRAIN ONLY
    # ========================================================

    (
        X_search_train,
        y_search_train,
        X_search_val,
        y_search_val,

    ) = build_evo_search_data(

        X_train,
        y_train,
    )

    # ========================================================
    # 6. SEARCH CLASSIFIER
    # ========================================================

    search_classifier = (
        ExtraTreesClassifier(

            n_estimators=
                SEARCH_N_ESTIMATORS,

            class_weight=
                "balanced",

            random_state=
                RANDOM_STATE,

            n_jobs=-1,
        )
    )

    # ========================================================
    # 7. CREATE EVO FEATURE SELECTOR
    # ========================================================

    selector = (
        EVOFeatureSelector(

            X_train=
                X_search_train,

            y_train=
                y_search_train,

            X_val=
                X_search_val,

            y_val=
                y_search_val,

            feature_names=
                feature_names,

            transfer_function=
                "sigmoid",

            threshold=
                0.5,

            min_features=
                1,

            macro_f1_weight=
                MACRO_F1_WEIGHT,

            balanced_accuracy_weight=
                BALANCED_ACCURACY_WEIGHT,

            feature_reduction_weight=
                FEATURE_REDUCTION_WEIGHT,

            objective=
                "minimize",

            random_state=
                RANDOM_STATE,

            classifier=
                search_classifier,

            enable_cache=True,
        )
    )

    # ========================================================
    # 8. RUN EVO
    # ========================================================

    separator()

    print(
        "\nStarting EVO optimization..."
    )

    print(
        f"Population       : "
        f"{EVO_POPULATION_SIZE}"
    )

    print(
        f"Iterations       : "
        f"{EVO_ITERATIONS}"
    )

    print(
        f"Dimensions       : "
        f"{len(feature_names)}"
    )

    print(
        f"Search trees     : "
        f"{SEARCH_N_ESTIMATORS}"
    )

    optimization_start = (
        time.perf_counter()
    )

    (
        best_position,
        best_fitness,
        history,
        evo_result,

    ) = run_evo(

        selector=
            selector,

        n_features=
            len(feature_names),
    )

    optimization_runtime = (
        time.perf_counter()
        -
        optimization_start
    )

    print(
        f"\nEVO reported evaluations : "
        f"{evo_result.evaluations}"
    )

    print(
        f"Selector real evaluations : "
        f"{selector.evaluation_count}"
    )

    print(
        f"Selector cache hits       : "
        f"{selector.cache_hits}"
    )

    print(
        f"Optimization runtime      : "
        f"{optimization_runtime:.2f}s"
    )

    # ========================================================
    # 9. FREEZE BEST EVO SUBSET
    # ========================================================

    search_best_result = (
        selector.evaluate(
            best_position
        )
    )

    best_mask = (
        selector.position_to_mask(
            best_position
        )
    )

    # --------------------------------------------------------
    # Fitness consistency
    # --------------------------------------------------------

    if not np.isclose(
        float(best_fitness),
        search_best_result.fitness,
        atol=1e-8,
    ):

        print(
            "\nWARNING:"
        )

        print(
            "Optimizer best fitness differs "
            "from selector re-evaluation."
        )

        print(
            f"EVO fitness      : "
            f"{best_fitness}"
        )

        print(
            f"Selector fitness : "
            f"{search_best_result.fitness}"
        )

    # ========================================================
    # 10. FULL TRAIN → OFFICIAL VALIDATION USING EVO FEATURES
    # ========================================================

    separator()

    print(
        "\nEVO subset frozen."
    )

    print(
        f"Selected features: "
        f"{search_best_result.selected_count}"
    )

    print(
        "\nEvaluating frozen EVO subset "
        "on FULL training + official validation..."
    )

    final_selected_metrics = (
        evaluate_feature_subset(

            X_train=
                X_train,

            y_train=
                y_train,

            X_val=
                X_val,

            y_val=
                y_val,

            mask=
                best_mask,

            n_estimators=
                FINAL_N_ESTIMATORS,
        )
    )

    # ========================================================
    # 11. SAVE RESULTS
    # ========================================================

    separator()

    print(
        "\nSaving Phase 5 artifacts..."
    )

    save_selected_features(
        search_best_result
    )

    save_feature_mask(
        feature_names,
        best_mask,
    )

    save_convergence_history(
        history
    )

    save_comparison(

        baseline=
            baseline,

        selected_metrics=
            final_selected_metrics,

        total_features=
            len(feature_names),

        selected_count=
            search_best_result.selected_count,
    )

    save_summary(

        baseline=
            baseline,

        search_result=
            search_best_result,

        final_selected_metrics=
            final_selected_metrics,

        optimization_runtime=
            optimization_runtime,

        selector=
            selector,

        evo_result=
            evo_result,
    )

    # ========================================================
    # 12. FINAL REPORT
    # ========================================================

    print_final_result(

        baseline=
            baseline,

        search_result=
            search_best_result,

        final_selected_metrics=
            final_selected_metrics,

        optimization_runtime=
            optimization_runtime,

        selector=
            selector,

        evo_result=
            evo_result,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()