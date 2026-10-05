"""
Phase 5B - Federated-Aware EVO Feature Selection
=================================================

Purpose:

    Existing Phase 5A:
        Centralized ExtraTrees proxy
            ↓
        EVO selected 10 features
            ↓
        severe non-IID FedAvg collapsed

    Phase 5B:
        Same heterogeneous FL client structure
            ↓
        train-derived proxy splits
            ↓
        federated-aware EVO fitness
            ↓
        feature subset optimized for:
            Macro F1
            Balanced Accuracy
            Attack Macro Recall
            Worst Attack Recall
            Feature Reduction
            Zero-attack penalty

IMPORTANT
---------

This script NEVER reads:

    data/processed/validation.csv
    data/processed/test_locked.csv

It uses only:

    data/federated/<scenario>/client_*.csv

and divides each client internally into:

    proxy train
    proxy validation

The resulting feature mask is saved separately at:

    results/phase5b_federated_evo/
        federated_evo_feature_mask.csv

It does NOT overwrite:

    results/phase5_evo/evo_feature_mask.csv
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

from feature_selection.evo_federated_feature_selector import (
    EVOFederatedFeatureSelector,
)

from optimization.evo import (
    EnergyValleyOptimizer,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

N_CLIENTS = 5

EXPECTED_FEATURE_COUNT = 70

SCENARIO = "non_iid"


# ============================================================
# FEDERATED PROXY DATA CONFIG
#
# We deliberately keep this smaller than the real Phase-4
# client datasets because every EVO fitness calculation itself
# runs federated neural-network training.
# ============================================================

MAX_ROWS_PER_CLIENT = 20_000

CLIENT_VALIDATION_FRACTION = 0.25


# ============================================================
# FEDERATED FITNESS PROXY
# ============================================================

PROXY_FEDERATED_ROUNDS = 2

PROXY_LOCAL_EPOCHS = 1

PROXY_BATCH_SIZE = 1024

PROXY_LEARNING_RATE = 0.001

PROXY_WEIGHT_DECAY = 0.0001


# ============================================================
# EVO SEARCH BUDGET
#
# Start with 10 x 10.
#
# Do NOT immediately use 20 x 30 because every candidate now
# performs real federated neural-network training.
# ============================================================

EVO_POPULATION_SIZE = 10

EVO_ITERATIONS = 10

EVO_LOWER_BOUND = -6.0

EVO_UPPER_BOUND = 6.0


# ============================================================
# FEATURE MASK
# ============================================================

MIN_FEATURES = 5


# ============================================================
# FITNESS WEIGHTS
# ============================================================

MACRO_F1_WEIGHT = 0.35

BALANCED_ACCURACY_WEIGHT = 0.30

ATTACK_MACRO_RECALL_WEIGHT = 0.20

WORST_ATTACK_RECALL_WEIGHT = 0.10

FEATURE_REDUCTION_WEIGHT = 0.05

ZERO_ATTACK_PENALTY_WEIGHT = 0.15


# ============================================================
# PATHS
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
    / "phase5b_federated_evo"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SELECTED_FEATURES_FILE = (
    OUTPUT_DIR
    / "federated_evo_selected_features.csv"
)

FEATURE_MASK_FILE = (
    OUTPUT_DIR
    / "federated_evo_feature_mask.csv"
)

CONVERGENCE_FILE = (
    OUTPUT_DIR
    / "federated_evo_convergence.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "phase5b_federated_evo_summary.json"
)

CLIENT_SPLIT_FILE = (
    OUTPUT_DIR
    / "proxy_client_splits.csv"
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
    "phase5b_federated_evo"
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
# DISPLAY SEPARATOR
# ============================================================

def separator():

    print()
    print(
        "=" * 90
    )


# ============================================================
# DATA-LEAKAGE GUARD
# ============================================================

def assert_search_data_is_safe():

    """
    Phase 5B feature search must NOT consume the official
    validation set or locked test set.

    This function documents and validates the intended design.

    Client source files must come only from:
        data/federated/<scenario>/
    """

    if not FEDERATED_DIR.exists():

        raise FileNotFoundError(
            f"Federated scenario missing:\n"
            f"{FEDERATED_DIR}"
        )

    print(
        "\nSearch data source:"
    )

    print(
        FEDERATED_DIR
    )

    print(
        "\nOfficial validation used: NO"
    )

    print(
        "Locked test used         : NO"
    )


# ============================================================
# SAFE STRATIFIED SAMPLE
# ============================================================

def stratified_sample(
    X,
    y,
    max_rows,
    random_state,
):

    """
    Reduce one client's dataset while maintaining its class
    proportions as much as possible.
    """

    if len(
        y
    ) <= max_rows:

        return (
            X,
            y,
        )

    indices = np.arange(
        len(y)
    )

    # --------------------------------------------------------
    # Normal stratified sampling
    # --------------------------------------------------------

    try:

        sampled_indices, _ = (
            train_test_split(

                indices,

                train_size=
                    max_rows,

                stratify=
                    y,

                random_state=
                    random_state,
            )
        )

    except ValueError:

        # ----------------------------------------------------
        # Fallback for pathological tiny classes
        # ----------------------------------------------------

        logger.warning(
            "Stratified client sampling failed; "
            "falling back to deterministic random sampling."
        )

        rng = np.random.default_rng(
            random_state
        )

        sampled_indices = (
            rng.choice(

                indices,

                size=
                    max_rows,

                replace=False,
            )
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
# SAFE CLIENT TRAIN / VALIDATION SPLIT
# ============================================================

def split_client_proxy_data(
    X,
    y,
    random_state,
):

    """
    Split a client's sampled data into train-derived proxy:

        local training
        local validation
    """

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
            "Stratified client train/validation split failed; "
            "using deterministic non-stratified split."
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
# LOAD ONE FEDERATED CLIENT
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
    # Same preprocessing as Phase 4
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
    # Confirm 70-feature Phase-4 space
    # --------------------------------------------------------

    if (
        X.shape[1]
        !=
        EXPECTED_FEATURE_COUNT
    ):

        raise ValueError(
            f"\nClient {client_id} produced "
            f"{X.shape[1]} features.\n"
            f"Expected Phase-4 feature count: "
            f"{EXPECTED_FEATURE_COUNT}"
        )

    # --------------------------------------------------------
    # Smaller reproducible client subset
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
    # Train-derived client proxy split
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
    # Distribution diagnostics
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
        len(y_train),
        len(y_val),
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
                    len(y_train)
                ),

            "validation_rows":
                int(
                    len(y_val)
                ),

            "class_counts":
                [
                    int(x)
                    for x
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
                f"Client file missing:\n"
                f"{client_file}"
            )

        split = (
            load_client_proxy_split(

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

    split_df = pd.DataFrame(
        split_rows
    )

    split_df.to_csv(
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

    selected_df = pd.DataFrame({

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
    })

    selected_df.to_csv(
        SELECTED_FEATURES_FILE,
        index=False,
    )


# ============================================================
# SAVE FULL 70-FEATURE MASK
# ============================================================

def save_feature_mask(
    feature_names,
    mask,
):

    mask = np.asarray(
        mask,
        dtype=bool,
    )

    mask_df = pd.DataFrame({

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
    })

    mask_df.to_csv(
        FEATURE_MASK_FILE,
        index=False,
    )


# ============================================================
# SAVE CONVERGENCE
# ============================================================

def save_convergence(
    history,
):

    history = np.asarray(
        history,
        dtype=float,
    ).reshape(-1)

    convergence_df = pd.DataFrame({

        "iteration":
            np.arange(
                1,
                len(history)
                + 1,
            ),

        "best_fitness":
            history,
    })

    convergence_df.to_csv(
        CONVERGENCE_FILE,
        index=False,
    )


# ============================================================
# RESULT TO SERIALIZABLE DICT
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
# PRINT ONE RESULT
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

    for name, recall in zip(
        class_names,
        result.class_recalls,
    ):

        print(
            f"  {name:25s} "
            f"{recall:.6f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = time.perf_counter()

    set_seed(
        SEED
    )

    separator()

    print(
        "PHASE 5B — FEDERATED-AWARE EVO FEATURE SELECTION"
    )

    separator()

    print(
        f"Scenario                : "
        f"{SCENARIO}"
    )

    print(
        f"Clients                 : "
        f"{N_CLIENTS}"
    )

    print(
        f"Max rows/client         : "
        f"{MAX_ROWS_PER_CLIENT:,}"
    )

    print(
        f"Proxy FedAvg rounds     : "
        f"{PROXY_FEDERATED_ROUNDS}"
    )

    print(
        f"Proxy local epochs      : "
        f"{PROXY_LOCAL_EPOCHS}"
    )

    print(
        f"EVO population          : "
        f"{EVO_POPULATION_SIZE}"
    )

    print(
        f"EVO iterations          : "
        f"{EVO_ITERATIONS}"
    )

    # ========================================================
    # 1. DATA SAFETY
    # ========================================================

    assert_search_data_is_safe()


    # ========================================================
    # 2. LOAD PHASE-4 PREPROCESSING
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
            "\nExpected Phase-4 active feature count "
            f"{EXPECTED_FEATURE_COUNT}, "
            f"found {len(active_features)}."
        )

    print(
        f"\nPhase-4 active features : "
        f"{len(active_features)}"
    )

    print(
        f"Classes                 : "
        f"{len(label_mapping)}"
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
            f"BENIGN label not found in "
            f"label mapping:\n"
            f"{label_mapping}"
        )

    benign_class_id = int(
        label_mapping[
            phase4.BENIGN_LABEL
        ]
    )


    # ========================================================
    # 4. CLASS NAMES IN ENCODED ORDER
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
    # 5. BUILD CLIENT PROXY SPLITS
    # ========================================================

    separator()

    print(
        "BUILDING TRAIN-DERIVED FEDERATED PROXY SPLITS"
    )

    client_splits = (
        build_client_splits(

            scaler=
                scaler,

            feature_columns=
                feature_columns,

            active_indices=
                active_indices,

            label_mapping=
                label_mapping,
        )
    )

    print()

    for client in client_splits:

        print(
            f"Client {client['client_id']} | "
            f"train={len(client['y_train']):,} | "
            f"val={len(client['y_val']):,}"
        )


    # ========================================================
    # 6. CREATE FEDERATED-AWARE SELECTOR
    # ========================================================

    selector = (
        EVOFederatedFeatureSelector(

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
    # 7. FEDERATED PROXY BASELINE — ALL 70 FEATURES
    #
    # position = positive values
    # sigmoid(position) > 0.5
    # therefore every feature is selected.
    # ========================================================

    separator()

    print(
        "EVALUATING FEDERATED PROXY BASELINE — 70 FEATURES"
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
    # 8. CREATE ENERGY VALLEY OPTIMIZER
    # ========================================================

    optimizer = (
        EnergyValleyOptimizer(

            population_size=
                EVO_POPULATION_SIZE,

            max_iterations=
                EVO_ITERATIONS,

            random_state=
                SEED,

            verbose=True,
        )
    )


    # ========================================================
    # 9. RUN FEDERATED-AWARE EVO
    # ========================================================

    separator()

    print(
        "STARTING FEDERATED-AWARE ENERGY VALLEY OPTIMIZATION"
    )

    print(
        f"Dimension : "
        f"{len(active_features)}"
    )

    print(
        f"Bounds    : "
        f"[{EVO_LOWER_BOUND}, "
        f"{EVO_UPPER_BOUND}]"
    )

    optimization_start = (
        time.perf_counter()
    )

    evo_result = (
        optimizer.optimize(

            objective_function=
                selector.fitness,

            dimension=
                len(
                    active_features
                ),

            lower_bound=
                EVO_LOWER_BOUND,

            upper_bound=
                EVO_UPPER_BOUND,
        )
    )

    optimization_seconds = (

        time.perf_counter()

        -
        optimization_start
    )


    # ========================================================
    # 10. FREEZE BEST CANDIDATE
    # ========================================================

    best_position = np.asarray(

        evo_result.best_position,

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
    # 11. FITNESS CONSISTENCY CHECK
    # ========================================================

    if not np.isclose(

        float(
            evo_result.best_fitness
        ),

        best_result.fitness,

        atol=1e-8,
    ):

        print(
            "\nWARNING:"
        )

        print(
            "EnergyValleyOptimizer best fitness "
            "does not exactly match selector result."
        )

        print(
            f"EVO result       : "
            f"{evo_result.best_fitness}"
        )

        print(
            f"Selector result  : "
            f"{best_result.fitness}"
        )


    # ========================================================
    # 12. FINAL SEARCH RESULT
    # ========================================================

    print_result(

        title=
            "BEST FEDERATED-AWARE EVO SUBSET",

        result=
            best_result,

        class_names=
            class_names,
    )


    # ========================================================
    # 13. CHANGE VS FEDERATED PROXY BASELINE
    # ========================================================

    separator()

    print(
        "FEDERATED PROXY: 70 FEATURES VS EVO SUBSET"
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
    # 14. SAVE FEATURES
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

    save_convergence(
        evo_result.convergence_history
    )


    # ========================================================
    # 15. SAVE SUMMARY
    # ========================================================

    total_seconds = (

        time.perf_counter()

        -
        total_start
    )

    summary = {

        "phase":
            "5B",

        "experiment":
            "Federated-Aware EVO Feature Selection",

        "scenario":
            SCENARIO,

        "seed":
            SEED,

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

        "feature_space": {

            "starting_features":
                len(
                    active_features
                ),

            "selected_features":
                best_result.selected_count,

            "feature_reduction_ratio":
                best_result.feature_reduction,

            "selected_feature_names":
                list(
                    best_result.selected_features
                ),
        },

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
        },

        "evo": {

            "population_size":
                EVO_POPULATION_SIZE,

            "max_iterations":
                EVO_ITERATIONS,

            "lower_bound":
                EVO_LOWER_BOUND,

            "upper_bound":
                EVO_UPPER_BOUND,

            "completed_iterations":
                int(
                    evo_result.iterations
                ),

            "optimizer_evaluations":
                int(
                    evo_result.evaluations
                ),

            "real_selector_evaluations":
                int(
                    selector.evaluation_count
                ),

            "cache_hits":
                int(
                    selector.cache_hits
                ),

            "best_fitness":
                float(
                    evo_result.best_fitness
                ),
        },

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

        "federated_proxy_baseline":
            result_to_dict(
                baseline_result,
                class_names,
            ),

        "federated_evo_best":
            result_to_dict(
                best_result,
                class_names,
            ),

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
        },

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

        "output_files": {

            "selected_features":
                str(
                    SELECTED_FEATURES_FILE
                ),

            "feature_mask":
                str(
                    FEATURE_MASK_FILE
                ),

            "convergence":
                str(
                    CONVERGENCE_FILE
                ),

            "client_splits":
                str(
                    CLIENT_SPLIT_FILE
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
    # 16. PRINT SELECTED FEATURES
    # ========================================================

    separator()

    print(
        "FEDERATED-AWARE EVO SELECTED FEATURES"
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
    # 17. FINAL EXECUTION REPORT
    # ========================================================

    separator()

    print(
        "PHASE 5B COMPLETE — FEDERATED-AWARE EVO"
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
        f"EVO iterations           : "
        f"{evo_result.iterations}"
    )

    print(
        f"EVO evaluations          : "
        f"{evo_result.evaluations}"
    )

    print(
        f"Real FL evaluations      : "
        f"{selector.evaluation_count}"
    )

    print(
        f"Cache hits               : "
        f"{selector.cache_hits}"
    )

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
        "Official validation used during EVO: NO ✅"
    )

    print(
        "Locked test used during EVO         : NO ✅"
    )

    print(
        "Federated client structure used     : YES ✅"
    )

    print(
        "Sample-weighted FedAvg proxy used   : YES ✅"
    )

    print(
        "Attack-collapse penalty used        : YES ✅"
    )

    print()

    print(
        f"Feature mask:\n"
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