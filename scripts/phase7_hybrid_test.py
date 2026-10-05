"""
Phase 7 - Federated-Aware Hybrid EVO-GA Feature Selection
=========================================================

Purpose
-------

Phase 5B:
    Federated-aware EVO
        -> 36 features

Phase 6:
    Federated-aware GA
        -> 34 features

Phase 7:
    Federated-aware Hybrid EVO-GA

Hybrid flow:

    EVO global exploration
            ↓
    strongest EVO candidates
            ↓
    seed GA population
            ↓
    GA refinement
            ↓
    frozen Hybrid feature mask


Scientific control
------------------

The Hybrid experiment uses the SAME:

    - severe non-IID client files
    - 5 clients
    - train-derived proxy sampling
    - train/validation split
    - Phase-4 scaler
    - 70-feature starting space
    - proxy MLP
    - FedAvg
    - proxy rounds
    - local epochs
    - fitness weights
    - zero-attack penalty

as the standalone federated-aware EVO and GA experiments.

Only the optimization strategy changes.


DATA INTEGRITY
--------------

This script DOES NOT use:

    data/processed/validation.csv
    data/processed/test_locked.csv

It uses only:

    data/federated/non_iid/client_1.csv
    ...
    data/federated/non_iid/client_5.csv


Outputs
-------

results/phase7_hybrid/

    hybrid_selected_features.csv
    hybrid_feature_mask.csv

    hybrid_convergence.csv
    hybrid_evo_convergence.csv
    hybrid_ga_convergence.csv

    proxy_client_splits.csv

    hybrid_proxy_method_comparison.csv

    phase7_hybrid_summary.json


Run
---

    python scripts/phase7_hybrid_test.py
"""

from __future__ import annotations

import gc
import json
import logging
import random
import sys
import time

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCRIPT_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SCRIPT_DIR),
    )


# ============================================================
# PROJECT IMPORTS
# ============================================================

import phase4_fedavg as phase4

from optimization.hybrid_evo_ga import (
    HybridEVOGAOptimizer,
)

from feature_selection.hybrid_federated_feature_selector import (
    HybridFederatedFeatureSelector,
)


# ============================================================
# GENERAL CONFIG
# ============================================================

SEED = 42

N_CLIENTS = 5

EXPECTED_FEATURE_COUNT = 70

SCENARIO = "non_iid"


# ============================================================
# SAME PROXY DATA CONFIGURATION AS EVO / GA
# ============================================================

MAX_ROWS_PER_CLIENT = 20_000

CLIENT_VALIDATION_FRACTION = 0.25


# ============================================================
# SAME FEDERATED PROXY CONFIGURATION
# ============================================================

PROXY_FEDERATED_ROUNDS = 2

PROXY_LOCAL_EPOCHS = 1

PROXY_BATCH_SIZE = 1024

PROXY_LEARNING_RATE = 0.001

PROXY_WEIGHT_DECAY = 0.0001

PROXY_GRADIENT_CLIP = 5.0


# ============================================================
# HYBRID SEARCH SPACE
# ============================================================

HYBRID_LOWER_BOUND = -6.0

HYBRID_UPPER_BOUND = 6.0


# ============================================================
# EVO STAGE
# ============================================================

EVO_POPULATION_SIZE = 10

EVO_ITERATIONS = 10


# ============================================================
# GA REFINEMENT STAGE
# ============================================================

GA_POPULATION_SIZE = 10

GA_GENERATIONS = 10

GA_CROSSOVER_RATE = 0.90

GA_MUTATION_RATE = 0.10

GA_MUTATION_SCALE = 0.15

GA_TOURNAMENT_SIZE = 3

GA_ELITISM_COUNT = 1


# ============================================================
# EVO -> GA TRANSFER
# ============================================================

EVO_ELITE_FRACTION = 0.50

BEST_SOLUTION_COPIES = 1

SEEDED_MUTATION_SCALE = 0.05

RANDOM_INJECTION_FRACTION = 0.10


# ============================================================
# FEATURE MASK
# ============================================================

MIN_FEATURES = 5


# ============================================================
# SAME FITNESS AS FEDERATED EVO + GA
# ============================================================

MACRO_F1_WEIGHT = 0.35

BALANCED_ACCURACY_WEIGHT = 0.30

ATTACK_MACRO_RECALL_WEIGHT = 0.20

WORST_ATTACK_RECALL_WEIGHT = 0.10

FEATURE_REDUCTION_WEIGHT = 0.05

ZERO_ATTACK_PENALTY_WEIGHT = 0.15


# ============================================================
# DATA PATHS
# ============================================================

FEDERATED_DIR = (
    PROJECT_ROOT
    / "data"
    / "federated"
    / SCENARIO
)

OFFICIAL_VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation.csv"
)

LOCKED_TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_locked.csv"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "phase7_hybrid"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SELECTED_FEATURES_FILE = (
    OUTPUT_DIR
    / "hybrid_selected_features.csv"
)

FEATURE_MASK_FILE = (
    OUTPUT_DIR
    / "hybrid_feature_mask.csv"
)

COMBINED_CONVERGENCE_FILE = (
    OUTPUT_DIR
    / "hybrid_convergence.csv"
)

EVO_CONVERGENCE_FILE = (
    OUTPUT_DIR
    / "hybrid_evo_convergence.csv"
)

GA_CONVERGENCE_FILE = (
    OUTPUT_DIR
    / "hybrid_ga_convergence.csv"
)

CLIENT_SPLIT_FILE = (
    OUTPUT_DIR
    / "proxy_client_splits.csv"
)

METHOD_COMPARISON_FILE = (
    OUTPUT_DIR
    / "hybrid_proxy_method_comparison.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "phase7_hybrid_summary.json"
)


# ============================================================
# PREVIOUS RESULTS
# ============================================================

EVO_SUMMARY_FILE = (
    PROJECT_ROOT
    / "results"
    / "phase5b_federated_evo"
    / "phase5b_federated_evo_summary.json"
)

GA_SUMMARY_FILE = (
    PROJECT_ROOT
    / "results"
    / "phase6_ga"
    / "phase6_ga_summary.json"
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(
    "phase7_hybrid"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(
    seed: int,
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

    if hasattr(
        torch.backends,
        "cudnn",
    ):

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False


# ============================================================
# DISPLAY
# ============================================================

def separator():

    print()
    print(
        "=" * 90
    )


# ============================================================
# DATA SAFETY
# ============================================================

def assert_search_data_is_safe():

    if not FEDERATED_DIR.exists():

        raise FileNotFoundError(
            "\nFederated scenario directory missing:\n"
            f"{FEDERATED_DIR}"
        )

    print()
    print(
        "Hybrid search data source:"
    )

    print(
        FEDERATED_DIR
    )

    print()

    print(
        "Official validation used: NO"
    )

    print(
        "Locked test used         : NO"
    )


# ============================================================
# STRATIFIED SAMPLE
# ============================================================

def stratified_sample(
    X,
    y,
    max_rows,
    random_state,
):

    if len(
        y
    ) <= max_rows:

        return (
            X,
            y,
        )

    indices = np.arange(
        len(
            y
        )
    )

    try:

        sampled_indices, _ = train_test_split(

            indices,

            train_size=
                max_rows,

            stratify=
                y,

            random_state=
                random_state,
        )

    except ValueError:

        logger.warning(
            "Stratified sampling failed. "
            "Using deterministic random sampling."
        )

        rng = np.random.default_rng(
            random_state
        )

        sampled_indices = rng.choice(

            indices,

            size=
                max_rows,

            replace=False,
        )

    return (
        X[
            sampled_indices
        ],

        y[
            sampled_indices
        ],
    )


# ============================================================
# CLIENT TRAIN/VALIDATION SPLIT
# ============================================================

def split_client_proxy_data(
    X,
    y,
    random_state,
):

    try:

        (
            X_train,
            X_val,
            y_train,
            y_val,

        ) = train_test_split(

            X,
            y,

            test_size=
                CLIENT_VALIDATION_FRACTION,

            stratify=
                y,

            random_state=
                random_state,
        )

    except ValueError:

        logger.warning(
            "Stratified proxy split failed. "
            "Using deterministic non-stratified split."
        )

        (
            X_train,
            X_val,
            y_train,
            y_val,

        ) = train_test_split(

            X,
            y,

            test_size=
                CLIENT_VALIDATION_FRACTION,

            random_state=
                random_state,
        )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
    )


# ============================================================
# LOAD ONE CLIENT
# ============================================================

def load_client_proxy_split(
    client_id,
    client_file,
    scaler,
    feature_columns,
    active_indices,
    label_mapping,
):

    logger.info(
        "Loading client %d: %s",
        client_id,
        client_file,
    )

    df = pd.read_csv(
        client_file
    )

    original_rows = len(
        df
    )


    # --------------------------------------------------------
    # EXACT PHASE-4 PREPROCESSING
    # --------------------------------------------------------

    X = phase4.transform_features(

        df=
            df,

        feature_columns=
            feature_columns,

        scaler=
            scaler,

        active_indices=
            active_indices,
    )

    y = phase4.encode_labels(

        df[
            phase4.LABEL_COL
        ],

        label_mapping,
    )

    del df

    gc.collect()


    # --------------------------------------------------------
    # VERIFY FEATURE SPACE
    # --------------------------------------------------------

    if (
        X.shape[1]
        !=
        EXPECTED_FEATURE_COUNT
    ):

        raise ValueError(
            f"\nClient {client_id} produced "
            f"{X.shape[1]} features.\n"
            f"Expected {EXPECTED_FEATURE_COUNT}."
        )


    # --------------------------------------------------------
    # SAME SAMPLE RULE USED FOR EVO / GA
    # --------------------------------------------------------

    X, y = stratified_sample(

        X=
            X,

        y=
            y,

        max_rows=
            MAX_ROWS_PER_CLIENT,

        random_state=
            SEED
            +
            client_id,
    )

    sampled_rows = len(
        y
    )


    # --------------------------------------------------------
    # SAME INTERNAL TRAIN / VALIDATION SPLIT
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_val,
        y_val,

    ) = split_client_proxy_data(

        X=
            X,

        y=
            y,

        random_state=
            SEED
            +
            client_id
            * 10,
    )

    del X
    del y

    gc.collect()


    # --------------------------------------------------------
    # DISTRIBUTION
    # --------------------------------------------------------

    full_counts = np.bincount(

        np.concatenate(
            [
                y_train,
                y_val,
            ]
        ),

        minlength=
            len(
                label_mapping
            ),
    )


    logger.info(
        "Client %d | original=%d | sampled=%d | "
        "proxy_train=%d | proxy_val=%d",
        client_id,
        original_rows,
        sampled_rows,
        len(
            y_train
        ),
        len(
            y_val
        ),
    )


    return {

        "client_id":
            client_id,

        "X_train":
            X_train.astype(
                np.float32,
                copy=False,
            ),

        "y_train":
            y_train.astype(
                np.int64,
                copy=False,
            ),

        "X_val":
            X_val.astype(
                np.float32,
                copy=False,
            ),

        "y_val":
            y_val.astype(
                np.int64,
                copy=False,
            ),

        "_metadata": {

            "original_rows":
                int(
                    original_rows
                ),

            "sampled_rows":
                int(
                    sampled_rows
                ),

            "train_rows":
                int(
                    len(
                        y_train
                    )
                ),

            "validation_rows":
                int(
                    len(
                        y_val
                    )
                ),

            "class_counts":
                [
                    int(
                        value
                    )

                    for value
                    in full_counts
                ],
        },
    }


# ============================================================
# BUILD ALL CLIENT SPLITS
# ============================================================

def build_client_splits(
    scaler,
    feature_columns,
    active_indices,
    label_mapping,
):

    client_splits = []

    split_rows = []

    inverse_mapping = {

        value:
            label

        for label, value
        in label_mapping.items()
    }


    for client_id in range(
        1,
        N_CLIENTS + 1,
    ):

        client_file = (

            FEDERATED_DIR
            /
            f"client_{client_id}.csv"
        )

        if not client_file.exists():

            raise FileNotFoundError(
                "\nClient file missing:\n"
                f"{client_file}"
            )


        split = load_client_proxy_split(

            client_id=
                client_id,

            client_file=
                client_file,

            scaler=
                scaler,

            feature_columns=
                feature_columns,

            active_indices=
                active_indices,

            label_mapping=
                label_mapping,
        )


        metadata = split.pop(
            "_metadata"
        )

        client_splits.append(
            split
        )


        row = {

            "client":
                f"client_{client_id}",

            "local_seen_attack":
                phase4.CLIENT_ATTACKS.get(
                    client_id,
                    "UNKNOWN",
                ),

            "original_rows":
                metadata[
                    "original_rows"
                ],

            "sampled_rows":
                metadata[
                    "sampled_rows"
                ],

            "proxy_train_rows":
                metadata[
                    "train_rows"
                ],

            "proxy_validation_rows":
                metadata[
                    "validation_rows"
                ],
        }


        for class_id, count in enumerate(
            metadata[
                "class_counts"
            ]
        ):

            class_name = (
                inverse_mapping.get(
                    class_id,
                    str(
                        class_id
                    ),
                )
            )

            row[
                f"count_{class_name}"
            ] = count


        split_rows.append(
            row
        )


    pd.DataFrame(
        split_rows
    ).to_csv(

        CLIENT_SPLIT_FILE,

        index=False,
    )

    return client_splits


# ============================================================
# SAVE SELECTED FEATURES
# ============================================================

def save_selected_features(
    result,
):

    pd.DataFrame({

        "selected_order":
            np.arange(
                1,
                result.selected_count
                + 1,
            ),

        "feature_index":
            list(
                result.selected_indices
            ),

        "feature_name":
            list(
                result.selected_features
            ),

    }).to_csv(

        SELECTED_FEATURES_FILE,

        index=False,
    )


# ============================================================
# SAVE FEATURE MASK
# ============================================================

def save_feature_mask(
    feature_names,
    mask,
):

    mask = np.asarray(
        mask,
        dtype=bool,
    )

    if len(
        mask
    ) != len(
        feature_names
    ):

        raise ValueError(
            "Feature mask length does not match "
            "feature-name count."
        )

    pd.DataFrame({

        "feature_index":
            np.arange(
                len(
                    feature_names
                )
            ),

        "feature_name":
            feature_names,

        "selected":
            mask.astype(
                int
            ),

    }).to_csv(

        FEATURE_MASK_FILE,

        index=False,
    )


# ============================================================
# SAVE CONVERGENCE
# ============================================================

def save_convergence(
    history,
    output_file,
    column_name,
):

    values = np.asarray(
        history,
        dtype=np.float64,
    ).reshape(-1)

    pd.DataFrame({

        "step":
            np.arange(
                1,
                len(
                    values
                )
                + 1,
            ),

        column_name:
            values,

    }).to_csv(

        output_file,

        index=False,
    )


# ============================================================
# RESULT -> DICT
# ============================================================

def result_to_dict(
    result,
    class_names,
):

    data = result.to_dict()

    data[
        "selected_indices"
    ] = list(
        data[
            "selected_indices"
        ]
    )

    data[
        "selected_features"
    ] = list(
        data[
            "selected_features"
        ]
    )

    data[
        "class_recalls"
    ] = list(
        data[
            "class_recalls"
        ]
    )

    data[
        "per_class_recall"
    ] = {

        class_name:
            float(
                recall
            )

        for class_name, recall
        in zip(
            class_names,
            result.class_recalls,
        )
    }

    return data


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(
    title,
    result,
    class_names,
):

    print()
    print(
        "-" * 90
    )

    print(
        title
    )

    print(
        "-" * 90
    )

    print(
        f"Features             : "
        f"{result.selected_count}/"
        f"{result.total_features}"
    )

    print(
        f"Feature reduction    : "
        f"{result.feature_reduction * 100:.2f}%"
    )

    print(
        f"Accuracy             : "
        f"{result.accuracy:.6f}"
    )

    print(
        f"Balanced Accuracy    : "
        f"{result.balanced_accuracy:.6f}"
    )

    print(
        f"Macro Precision      : "
        f"{result.macro_precision:.6f}"
    )

    print(
        f"Macro Recall         : "
        f"{result.macro_recall:.6f}"
    )

    print(
        f"Macro F1             : "
        f"{result.macro_f1:.6f}"
    )

    print(
        f"Attack Macro Recall  : "
        f"{result.attack_macro_recall:.6f}"
    )

    print(
        f"Worst Attack Recall  : "
        f"{result.worst_attack_recall:.6f}"
    )

    print(
        f"Zero-recall attacks  : "
        f"{result.zero_attack_count}"
    )

    print(
        f"Utility              : "
        f"{result.utility:.6f}"
    )

    print(
        f"Fitness              : "
        f"{result.fitness:.6f}"
    )

    print()

    print(
        "Per-class recall:"
    )

    for class_name, recall in zip(
        class_names,
        result.class_recalls,
    ):

        print(
            f"  {class_name:25s} "
            f"{recall:.6f}"
        )


# ============================================================
# LOAD JSON
# ============================================================

def load_json_if_exists(
    path,
):

    if not path.exists():

        return None

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


# ============================================================
# PREVIOUS EVO / GA PROXY RESULTS
# ============================================================

def extract_previous_proxy_results():

    evo_summary = load_json_if_exists(
        EVO_SUMMARY_FILE
    )

    ga_summary = load_json_if_exists(
        GA_SUMMARY_FILE
    )


    evo_result = None

    ga_result = None


    if evo_summary is not None:

        evo_result = (
            evo_summary.get(
                "federated_evo_best"
            )
        )


    if ga_summary is not None:

        ga_result = (
            ga_summary.get(
                "federated_ga_best"
            )
        )


    return (
        evo_result,
        ga_result,
    )


# ============================================================
# SAVE METHOD COMPARISON
# ============================================================

def save_method_comparison(
    baseline_result,
    hybrid_result,
    evo_result,
    ga_result,
):

    rows = []


    # --------------------------------------------------------
    # 70 FEATURES
    # --------------------------------------------------------

    rows.append({

        "method":
            "Federated_Proxy_70",

        "features":
            baseline_result.selected_count,

        "feature_reduction":
            baseline_result.feature_reduction,

        "macro_f1":
            baseline_result.macro_f1,

        "balanced_accuracy":
            baseline_result.balanced_accuracy,

        "attack_macro_recall":
            baseline_result.attack_macro_recall,

        "worst_attack_recall":
            baseline_result.worst_attack_recall,

        "zero_attack_count":
            baseline_result.zero_attack_count,

        "fitness":
            baseline_result.fitness,
    })


    # --------------------------------------------------------
    # EVO
    # --------------------------------------------------------

    if evo_result is not None:

        rows.append({

            "method":
                "Federated_EVO",

            "features":
                evo_result.get(
                    "selected_count"
                ),

            "feature_reduction":
                evo_result.get(
                    "feature_reduction"
                ),

            "macro_f1":
                evo_result.get(
                    "macro_f1"
                ),

            "balanced_accuracy":
                evo_result.get(
                    "balanced_accuracy"
                ),

            "attack_macro_recall":
                evo_result.get(
                    "attack_macro_recall"
                ),

            "worst_attack_recall":
                evo_result.get(
                    "worst_attack_recall"
                ),

            "zero_attack_count":
                evo_result.get(
                    "zero_attack_count"
                ),

            "fitness":
                evo_result.get(
                    "fitness"
                ),
        })


    # --------------------------------------------------------
    # GA
    # --------------------------------------------------------

    if ga_result is not None:

        rows.append({

            "method":
                "Federated_GA",

            "features":
                ga_result.get(
                    "selected_count"
                ),

            "feature_reduction":
                ga_result.get(
                    "feature_reduction"
                ),

            "macro_f1":
                ga_result.get(
                    "macro_f1"
                ),

            "balanced_accuracy":
                ga_result.get(
                    "balanced_accuracy"
                ),

            "attack_macro_recall":
                ga_result.get(
                    "attack_macro_recall"
                ),

            "worst_attack_recall":
                ga_result.get(
                    "worst_attack_recall"
                ),

            "zero_attack_count":
                ga_result.get(
                    "zero_attack_count"
                ),

            "fitness":
                ga_result.get(
                    "fitness"
                ),
        })


    # --------------------------------------------------------
    # HYBRID
    # --------------------------------------------------------

    rows.append({

        "method":
            "Federated_Hybrid_EVO_GA",

        "features":
            hybrid_result.selected_count,

        "feature_reduction":
            hybrid_result.feature_reduction,

        "macro_f1":
            hybrid_result.macro_f1,

        "balanced_accuracy":
            hybrid_result.balanced_accuracy,

        "attack_macro_recall":
            hybrid_result.attack_macro_recall,

        "worst_attack_recall":
            hybrid_result.worst_attack_recall,

        "zero_attack_count":
            hybrid_result.zero_attack_count,

        "fitness":
            hybrid_result.fitness,
    })


    comparison_df = pd.DataFrame(
        rows
    )

    comparison_df.to_csv(

        METHOD_COMPARISON_FILE,

        index=False,
    )

    return comparison_df


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = (
        time.perf_counter()
    )

    set_seed(
        SEED
    )


    # ========================================================
    # HEADER
    # ========================================================

    separator()

    print(
        "PHASE 7 — FEDERATED-AWARE HYBRID EVO-GA"
    )

    separator()

    print(
        f"Scenario                   : "
        f"{SCENARIO}"
    )

    print(
        f"Clients                    : "
        f"{N_CLIENTS}"
    )

    print(
        f"Starting features          : "
        f"{EXPECTED_FEATURE_COUNT}"
    )

    print(
        f"Max rows/client            : "
        f"{MAX_ROWS_PER_CLIENT:,}"
    )

    print(
        f"Proxy FedAvg rounds        : "
        f"{PROXY_FEDERATED_ROUNDS}"
    )

    print(
        f"Proxy local epochs         : "
        f"{PROXY_LOCAL_EPOCHS}"
    )

    print()

    print(
        f"EVO population             : "
        f"{EVO_POPULATION_SIZE}"
    )

    print(
        f"EVO iterations             : "
        f"{EVO_ITERATIONS}"
    )

    print()

    print(
        f"GA population              : "
        f"{GA_POPULATION_SIZE}"
    )

    print(
        f"GA generations             : "
        f"{GA_GENERATIONS}"
    )

    print(
        f"GA crossover               : "
        f"{GA_CROSSOVER_RATE}"
    )

    print(
        f"GA mutation                : "
        f"{GA_MUTATION_RATE}"
    )


    # ========================================================
    # 1. SAFETY
    # ========================================================

    assert_search_data_is_safe()


    # ========================================================
    # 2. PHASE-4 PREPROCESSING
    # ========================================================

    (
        scaler,
        feature_columns,
        label_mapping,
        active_indices,
        active_features,

    ) = phase4.load_preprocessing()


    if (
        len(
            active_features
        )
        !=
        EXPECTED_FEATURE_COUNT
    ):

        raise ValueError(
            "\nExpected Phase-4 feature count "
            f"{EXPECTED_FEATURE_COUNT}, "
            f"found {len(active_features)}."
        )


    # ========================================================
    # 3. BENIGN CLASS
    # ========================================================

    if (
        phase4.BENIGN_LABEL
        not in
        label_mapping
    ):

        raise ValueError(
            "BENIGN label missing from "
            "Phase-4 label mapping."
        )

    benign_class_id = int(

        label_mapping[
            phase4.BENIGN_LABEL
        ]
    )


    # ========================================================
    # 4. CLASS NAMES
    # ========================================================

    inverse_mapping = {

        value:
            label

        for label, value
        in label_mapping.items()
    }

    class_names = [

        inverse_mapping[
            class_id
        ]

        for class_id
        in range(
            len(
                label_mapping
            )
        )
    ]


    # ========================================================
    # 5. PROXY CLIENT SPLITS
    # ========================================================

    separator()

    print(
        "BUILDING TRAIN-DERIVED FEDERATED PROXY SPLITS"
    )

    client_splits = build_client_splits(

        scaler=
            scaler,

        feature_columns=
            feature_columns,

        active_indices=
            active_indices,

        label_mapping=
            label_mapping,
    )


    print()

    for client in client_splits:

        print(
            f"Client {client['client_id']} | "
            f"train={len(client['y_train']):,} | "
            f"val={len(client['y_val']):,}"
        )


    # ========================================================
    # 6. HYBRID FEDERATED SELECTOR
    # ========================================================

    selector = (
        HybridFederatedFeatureSelector(

            client_splits=
                client_splits,

            feature_names=
                active_features,

            num_classes=
                len(
                    label_mapping
                ),

            benign_class_id=
                benign_class_id,

            federated_rounds=
                PROXY_FEDERATED_ROUNDS,

            local_epochs=
                PROXY_LOCAL_EPOCHS,

            batch_size=
                PROXY_BATCH_SIZE,

            learning_rate=
                PROXY_LEARNING_RATE,

            weight_decay=
                PROXY_WEIGHT_DECAY,

            gradient_clip=
                PROXY_GRADIENT_CLIP,

            transfer_function=
                "sigmoid",

            threshold=
                0.5,

            min_features=
                MIN_FEATURES,

            macro_f1_weight=
                MACRO_F1_WEIGHT,

            balanced_accuracy_weight=
                BALANCED_ACCURACY_WEIGHT,

            attack_macro_recall_weight=
                ATTACK_MACRO_RECALL_WEIGHT,

            worst_attack_recall_weight=
                WORST_ATTACK_RECALL_WEIGHT,

            feature_reduction_weight=
                FEATURE_REDUCTION_WEIGHT,

            zero_attack_penalty_weight=
                ZERO_ATTACK_PENALTY_WEIGHT,

            objective=
                "minimize",

            random_state=
                SEED,

            device=None,

            enable_cache=True,

            verbose=True,
        )
    )


    # ========================================================
    # 7. 70-FEATURE PROXY BASELINE
    # ========================================================

    separator()

    print(
        "EVALUATING 70-FEATURE FEDERATED PROXY BASELINE"
    )


    all_feature_position = np.full(

        len(
            active_features
        ),

        6.0,

        dtype=np.float64,
    )


    baseline_start = (
        time.perf_counter()
    )


    baseline_result = (
        selector.evaluate(
            all_feature_position
        )
    )


    baseline_seconds = (

        time.perf_counter()

        -
        baseline_start
    )


    print_result(

        title=
            "70-FEATURE FEDERATED PROXY BASELINE",

        result=
            baseline_result,

        class_names=
            class_names,
    )


    print(
        f"\nBaseline evaluation time: "
        f"{baseline_seconds:.2f}s"
    )


    # ========================================================
    # 8. HYBRID OPTIMIZER
    # ========================================================

    hybrid_optimizer = (
        HybridEVOGAOptimizer(

            # EVO
            evo_population_size=
                EVO_POPULATION_SIZE,

            evo_iterations=
                EVO_ITERATIONS,

            # GA
            ga_population_size=
                GA_POPULATION_SIZE,

            ga_generations=
                GA_GENERATIONS,

            crossover_rate=
                GA_CROSSOVER_RATE,

            mutation_rate=
                GA_MUTATION_RATE,

            mutation_scale=
                GA_MUTATION_SCALE,

            tournament_size=
                GA_TOURNAMENT_SIZE,

            elitism_count=
                GA_ELITISM_COUNT,

            # EVO -> GA
            evo_elite_fraction=
                EVO_ELITE_FRACTION,

            best_solution_copies=
                BEST_SOLUTION_COPIES,

            seeded_mutation_scale=
                SEEDED_MUTATION_SCALE,

            random_injection_fraction=
                RANDOM_INJECTION_FRACTION,

            objective=
                "minimize",

            random_state=
                SEED,

            verbose=True,
        )
    )


    # ========================================================
    # 9. RUN HYBRID SEARCH
    # ========================================================

    separator()

    print(
        "STARTING FEDERATED-AWARE HYBRID EVO-GA SEARCH"
    )

    print(
        f"Dimension : "
        f"{len(active_features)}"
    )

    print(
        f"Bounds    : "
        f"[{HYBRID_LOWER_BOUND}, "
        f"{HYBRID_UPPER_BOUND}]"
    )


    optimization_start = (
        time.perf_counter()
    )


    hybrid_result = (
        hybrid_optimizer.optimize(

            objective_function=
                selector.fitness,

            dimension=
                len(
                    active_features
                ),

            lower_bound=
                HYBRID_LOWER_BOUND,

            upper_bound=
                HYBRID_UPPER_BOUND,
        )
    )


    optimization_seconds = (

        time.perf_counter()

        -
        optimization_start
    )


    # ========================================================
    # 10. FREEZE FINAL HYBRID MASK
    # ========================================================

    best_position = np.asarray(

        hybrid_result.best_position,

        dtype=np.float64,
    )


    best_mask = (
        selector.position_to_mask(
            best_position
        )
    )


    best_result = (
        selector.evaluate(
            best_position
        )
    )


    # ========================================================
    # 11. FITNESS CONSISTENCY
    # ========================================================

    if not np.isclose(

        float(
            hybrid_result.best_fitness
        ),

        float(
            best_result.fitness
        ),

        atol=1e-8,
    ):

        print()

        print(
            "WARNING:"
        )

        print(
            "Hybrid optimizer best fitness "
            "does not exactly match selector."
        )

        print(
            f"Optimizer fitness : "
            f"{hybrid_result.best_fitness:.12f}"
        )

        print(
            f"Selector fitness  : "
            f"{best_result.fitness:.12f}"
        )


    # ========================================================
    # 12. FINAL HYBRID RESULT
    # ========================================================

    print_result(

        title=
            "BEST FEDERATED-AWARE HYBRID EVO-GA SUBSET",

        result=
            best_result,

        class_names=
            class_names,
    )


    # ========================================================
    # 13. PROXY BASELINE VS HYBRID
    # ========================================================

    separator()

    print(
        "FEDERATED PROXY: 70 FEATURES VS HYBRID"
    )

    separator()


    print(
        f"Feature count         : "
        f"{baseline_result.selected_count}"
        f" -> "
        f"{best_result.selected_count}"
    )


    print(
        f"Feature reduction     : "
        f"{best_result.feature_reduction * 100:.2f}%"
    )


    print(
        f"Macro F1 delta        : "
        f"{best_result.macro_f1 - baseline_result.macro_f1:+.6f}"
    )


    print(
        f"Balanced Acc delta    : "
        f"{best_result.balanced_accuracy - baseline_result.balanced_accuracy:+.6f}"
    )


    print(
        f"Attack Recall delta   : "
        f"{best_result.attack_macro_recall - baseline_result.attack_macro_recall:+.6f}"
    )


    print(
        f"Worst Attack delta    : "
        f"{best_result.worst_attack_recall - baseline_result.worst_attack_recall:+.6f}"
    )


    print(
        f"Zero-recall attacks   : "
        f"{baseline_result.zero_attack_count}"
        f" -> "
        f"{best_result.zero_attack_count}"
    )


    # ========================================================
    # 14. PREVIOUS EVO / GA RESULTS
    # ========================================================

    (
        evo_previous,
        ga_previous,

    ) = extract_previous_proxy_results()


    comparison_df = save_method_comparison(

        baseline_result=
            baseline_result,

        hybrid_result=
            best_result,

        evo_result=
            evo_previous,

        ga_result=
            ga_previous,
    )


    # ========================================================
    # 15. DISPLAY EVO VS GA VS HYBRID
    # ========================================================

    separator()

    print(
        "FEDERATED PROXY METHOD COMPARISON"
    )

    separator()


    display_columns = [

        "method",
        "features",
        "feature_reduction",
        "macro_f1",
        "balanced_accuracy",
        "attack_macro_recall",
        "worst_attack_recall",
        "zero_attack_count",
        "fitness",
    ]


    print(

        comparison_df[
            display_columns
        ].to_string(
            index=False
        )
    )


    # ========================================================
    # 16. SAVE FEATURES / MASK
    # ========================================================

    save_selected_features(
        best_result
    )

    save_feature_mask(

        feature_names=
            active_features,

        mask=
            best_mask,
    )


    # ========================================================
    # 17. SAVE CONVERGENCE
    # ========================================================

    save_convergence(

        history=
            hybrid_result.convergence_history,

        output_file=
            COMBINED_CONVERGENCE_FILE,

        column_name=
            "best_fitness",
    )


    save_convergence(

        history=
            hybrid_result.evo_convergence_history,

        output_file=
            EVO_CONVERGENCE_FILE,

        column_name=
            "evo_best_fitness",
    )


    save_convergence(

        history=
            hybrid_result.ga_convergence_history,

        output_file=
            GA_CONVERGENCE_FILE,

        column_name=
            "ga_best_fitness",
    )


    # ========================================================
    # 18. TOTAL RUNTIME
    # ========================================================

    total_seconds = (

        time.perf_counter()

        -
        total_start
    )


    # ========================================================
    # 19. SUMMARY
    # ========================================================

    summary = {

        "phase":
            7,

        "experiment":
            (
                "Federated-Aware "
                "Hybrid EVO-GA Feature Selection"
            ),

        "scenario":
            SCENARIO,

        "seed":
            SEED,


        # ----------------------------------------------------
        # DATA INTEGRITY
        # ----------------------------------------------------

        "data_integrity": {

            "data_source":
                (
                    "Federated client training files only"
                ),

            "official_validation_used":
                False,

            "locked_test_used":
                False,

            "official_validation_path":
                str(
                    OFFICIAL_VALIDATION_FILE
                ),

            "locked_test_path":
                str(
                    LOCKED_TEST_FILE
                ),
        },


        # ----------------------------------------------------
        # FEATURE SPACE
        # ----------------------------------------------------

        "feature_space": {

            "starting_features":
                int(
                    len(
                        active_features
                    )
                ),

            "selected_features":
                int(
                    best_result.selected_count
                ),

            "feature_reduction_ratio":
                float(
                    best_result.feature_reduction
                ),

            "selected_feature_names":
                list(
                    best_result.selected_features
                ),
        },


        # ----------------------------------------------------
        # PROXY
        # ----------------------------------------------------

        "proxy_data": {

            "max_rows_per_client":
                MAX_ROWS_PER_CLIENT,

            "validation_fraction":
                CLIENT_VALIDATION_FRACTION,

            "clients":
                N_CLIENTS,
        },


        "proxy_federated_training": {

            "rounds":
                PROXY_FEDERATED_ROUNDS,

            "local_epochs":
                PROXY_LOCAL_EPOCHS,

            "batch_size":
                PROXY_BATCH_SIZE,

            "learning_rate":
                PROXY_LEARNING_RATE,

            "weight_decay":
                PROXY_WEIGHT_DECAY,

            "gradient_clip":
                PROXY_GRADIENT_CLIP,
        },


        # ----------------------------------------------------
        # FITNESS
        # ----------------------------------------------------

        "fitness_weights": {

            "macro_f1":
                MACRO_F1_WEIGHT,

            "balanced_accuracy":
                BALANCED_ACCURACY_WEIGHT,

            "attack_macro_recall":
                ATTACK_MACRO_RECALL_WEIGHT,

            "worst_attack_recall":
                WORST_ATTACK_RECALL_WEIGHT,

            "feature_reduction":
                FEATURE_REDUCTION_WEIGHT,

            "zero_attack_penalty_weight":
                ZERO_ATTACK_PENALTY_WEIGHT,
        },


        # ----------------------------------------------------
        # HYBRID CONFIG
        # ----------------------------------------------------

        "hybrid": {

            "lower_bound":
                HYBRID_LOWER_BOUND,

            "upper_bound":
                HYBRID_UPPER_BOUND,


            "evo": {

                "population_size":
                    EVO_POPULATION_SIZE,

                "iterations":
                    EVO_ITERATIONS,

                "completed_iterations":
                    int(
                        hybrid_result.evo_iterations
                    ),

                "optimizer_evaluations":
                    int(
                        hybrid_result.evo_evaluations
                    ),

                "best_fitness":
                    float(
                        hybrid_result.evo_best_fitness
                    ),
            },


            "ga_refinement": {

                "population_size":
                    GA_POPULATION_SIZE,

                "generations":
                    GA_GENERATIONS,

                "completed_generations":
                    int(
                        hybrid_result.ga_generations
                    ),

                "optimizer_evaluations":
                    int(
                        hybrid_result.ga_evaluations
                    ),

                "crossover_rate":
                    GA_CROSSOVER_RATE,

                "mutation_rate":
                    GA_MUTATION_RATE,

                "mutation_scale":
                    GA_MUTATION_SCALE,

                "tournament_size":
                    GA_TOURNAMENT_SIZE,

                "elitism_count":
                    GA_ELITISM_COUNT,

                "best_fitness":
                    float(
                        hybrid_result.ga_best_fitness
                    ),
            },


            "evo_to_ga_transfer": {

                "evo_elite_fraction":
                    EVO_ELITE_FRACTION,

                "best_solution_copies":
                    BEST_SOLUTION_COPIES,

                "seeded_mutation_scale":
                    SEEDED_MUTATION_SCALE,

                "random_injection_fraction":
                    RANDOM_INJECTION_FRACTION,
            },


            "final_best_fitness":
                float(
                    hybrid_result.best_fitness
                ),

            "total_optimizer_evaluations":
                int(
                    hybrid_result.total_evaluations
                ),
        },


        # ----------------------------------------------------
        # REAL FL EVALUATIONS
        # ----------------------------------------------------

        "selector_statistics":
            selector.get_search_statistics(),


        # ----------------------------------------------------
        # BASELINE
        # ----------------------------------------------------

        "federated_proxy_baseline":
            result_to_dict(
                baseline_result,
                class_names,
            ),


        # ----------------------------------------------------
        # HYBRID RESULT
        # ----------------------------------------------------

        "federated_hybrid_best":
            result_to_dict(
                best_result,
                class_names,
            ),


        # ----------------------------------------------------
        # DELTA
        # ----------------------------------------------------

        "delta_vs_proxy_baseline": {

            "macro_f1":
                float(
                    best_result.macro_f1
                    -
                    baseline_result.macro_f1
                ),

            "balanced_accuracy":
                float(
                    best_result.balanced_accuracy
                    -
                    baseline_result.balanced_accuracy
                ),

            "attack_macro_recall":
                float(
                    best_result.attack_macro_recall
                    -
                    baseline_result.attack_macro_recall
                ),

            "worst_attack_recall":
                float(
                    best_result.worst_attack_recall
                    -
                    baseline_result.worst_attack_recall
                ),

            "zero_attack_count_change":
                int(
                    best_result.zero_attack_count
                    -
                    baseline_result.zero_attack_count
                ),
        },


        # ----------------------------------------------------
        # PREVIOUS RESULTS
        # ----------------------------------------------------

        "previous_proxy_results": {

            "federated_evo":
                evo_previous,

            "federated_ga":
                ga_previous,
        },


        # ----------------------------------------------------
        # RUNTIME
        # ----------------------------------------------------

        "runtime": {

            "baseline_seconds":
                float(
                    baseline_seconds
                ),

            "optimization_seconds":
                float(
                    optimization_seconds
                ),

            "total_script_seconds":
                float(
                    total_seconds
                ),
        },


        # ----------------------------------------------------
        # OUTPUTS
        # ----------------------------------------------------

        "output_files": {

            "selected_features":
                str(
                    SELECTED_FEATURES_FILE
                ),

            "feature_mask":
                str(
                    FEATURE_MASK_FILE
                ),

            "combined_convergence":
                str(
                    COMBINED_CONVERGENCE_FILE
                ),

            "evo_convergence":
                str(
                    EVO_CONVERGENCE_FILE
                ),

            "ga_convergence":
                str(
                    GA_CONVERGENCE_FILE
                ),

            "client_splits":
                str(
                    CLIENT_SPLIT_FILE
                ),

            "method_comparison":
                str(
                    METHOD_COMPARISON_FILE
                ),
        },
    }


    with open(

        SUMMARY_FILE,

        "w",

        encoding="utf-8",

    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )


    # ========================================================
    # 20. SELECTED FEATURES
    # ========================================================

    separator()

    print(
        "FEDERATED-AWARE HYBRID EVO-GA SELECTED FEATURES"
    )

    separator()


    for number, feature in enumerate(
        best_result.selected_features,
        start=1,
    ):

        print(
            f"{number:02d}. "
            f"{feature}"
        )


    # ========================================================
    # 21. FINAL REPORT
    # ========================================================

    separator()

    print(
        "PHASE 7 COMPLETE — "
        "FEDERATED-AWARE HYBRID EVO-GA"
    )

    separator()


    print(
        f"Scenario                 : "
        f"{SCENARIO}"
    )

    print(
        f"Original features        : "
        f"{best_result.total_features}"
    )

    print(
        f"Selected features        : "
        f"{best_result.selected_count}"
    )

    print(
        f"Feature reduction        : "
        f"{best_result.feature_reduction * 100:.2f}%"
    )


    print()


    print(
        f"Macro F1                 : "
        f"{best_result.macro_f1:.6f}"
    )

    print(
        f"Balanced Accuracy        : "
        f"{best_result.balanced_accuracy:.6f}"
    )

    print(
        f"Attack Macro Recall      : "
        f"{best_result.attack_macro_recall:.6f}"
    )

    print(
        f"Worst Attack Recall      : "
        f"{best_result.worst_attack_recall:.6f}"
    )

    print(
        f"Zero-recall attacks      : "
        f"{best_result.zero_attack_count}"
    )

    print(
        f"Fitness                  : "
        f"{best_result.fitness:.6f}"
    )


    print()


    print(
        f"EVO stage best fitness   : "
        f"{hybrid_result.evo_best_fitness:.6f}"
    )

    print(
        f"GA stage best fitness    : "
        f"{hybrid_result.ga_best_fitness:.6f}"
    )

    print(
        f"Final hybrid fitness     : "
        f"{hybrid_result.best_fitness:.6f}"
    )


    print()


    print(
        f"EVO iterations           : "
        f"{hybrid_result.evo_iterations}"
    )

    print(
        f"EVO evaluations          : "
        f"{hybrid_result.evo_evaluations}"
    )

    print(
        f"GA generations           : "
        f"{hybrid_result.ga_generations}"
    )

    print(
        f"GA evaluations           : "
        f"{hybrid_result.ga_evaluations}"
    )

    print(
        f"Total optimizer evals    : "
        f"{hybrid_result.total_evaluations}"
    )

    print(
        f"Real FL evaluations      : "
        f"{selector.evaluation_count}"
    )

    print(
        f"Cache hits               : "
        f"{selector.cache_hits}"
    )


    print()


    print(
        f"Optimization time        : "
        f"{optimization_seconds:.2f}s"
    )

    print(
        f"Total script time        : "
        f"{total_seconds:.2f}s"
    )


    print()


    print(
        "Official validation used during Hybrid: NO ✅"
    )

    print(
        "Locked test used during Hybrid         : NO ✅"
    )

    print(
        "Same federated proxy as EVO            : YES ✅"
    )

    print(
        "Same federated proxy as GA             : YES ✅"
    )

    print(
        "Same fitness weights                   : YES ✅"
    )

    print(
        "EVO seeds GA population                : YES ✅"
    )

    print(
        "GA refines EVO candidates              : YES ✅"
    )

    print(
        "Sample-weighted FedAvg proxy            : YES ✅"
    )

    print(
        "Attack-collapse penalty                 : YES ✅"
    )


    print()


    print(
        f"Hybrid feature mask:\n"
        f"{FEATURE_MASK_FILE}"
    )

    print()


    print(
        f"Summary:\n"
        f"{SUMMARY_FILE}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()