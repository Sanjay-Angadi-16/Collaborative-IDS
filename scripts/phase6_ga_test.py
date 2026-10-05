"""
Phase 6 - Federated-Aware GA Feature Selection
==============================================

Purpose:

    Phase 5B:
        Federated-aware EVO
            ↓
        train-derived non-IID client proxy
            ↓
        frozen feature subset

    Phase 6:
        Federated-aware GA
            ↓
        SAME train-derived non-IID client proxy
            ↓
        frozen GA feature subset

This script uses:

    optimization/ga.py

and:

    feature_selection/ga_federated_feature_selector.py

IMPORTANT:
    - Official validation.csv is NOT used.
    - test_locked.csv is NOT used.
    - Only federated client training files are used.
    - Each client is split internally into proxy train/validation.
    - The same proxy design as Phase 5B EVO is preserved.

Output:

    results/phase6_ga/
        ga_selected_features.csv
        ga_feature_mask.csv
        ga_convergence.csv
        proxy_client_splits.csv
        phase6_ga_summary.json
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

from optimization.ga import (
    GeneticAlgorithmOptimizer,
)

from feature_selection.ga_federated_feature_selector import (
    GAFederatedFeatureSelector,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

N_CLIENTS = 5

EXPECTED_FEATURE_COUNT = 70

SCENARIO = "non_iid"


# ============================================================
# SAME PROXY DATA BUDGET AS PHASE 5B EVO
# ============================================================

MAX_ROWS_PER_CLIENT = 20_000

CLIENT_VALIDATION_FRACTION = 0.25


# ============================================================
# SAME FEDERATED PROXY SETTINGS AS PHASE 5B EVO
# ============================================================

PROXY_FEDERATED_ROUNDS = 2

PROXY_LOCAL_EPOCHS = 1

PROXY_BATCH_SIZE = 1024

PROXY_LEARNING_RATE = 0.001

PROXY_WEIGHT_DECAY = 0.0001


# ============================================================
# GA SEARCH BUDGET
#
# Same nominal 10 x 10 search budget as Phase 5B EVO.
# ============================================================

GA_POPULATION_SIZE = 10

GA_GENERATIONS = 10

GA_LOWER_BOUND = -6.0

GA_UPPER_BOUND = 6.0


# ============================================================
# GA PARAMETERS
# ============================================================

GA_CROSSOVER_RATE = 0.90

GA_MUTATION_RATE = 0.10

GA_MUTATION_SCALE = 0.15

GA_TOURNAMENT_SIZE = 3

GA_ELITISM_COUNT = 1


# ============================================================
# FEATURE MASK
# ============================================================

MIN_FEATURES = 5


# ============================================================
# SAME FITNESS AS FEDERATED-AWARE EVO
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
    / "phase6_ga"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SELECTED_FEATURES_FILE = (
    OUTPUT_DIR
    / "ga_selected_features.csv"
)

FEATURE_MASK_FILE = (
    OUTPUT_DIR
    / "ga_feature_mask.csv"
)

CONVERGENCE_FILE = (
    OUTPUT_DIR
    / "ga_convergence.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "phase6_ga_summary.json"
)

CLIENT_SPLIT_FILE = (
    OUTPUT_DIR
    / "proxy_client_splits.csv"
)


# ============================================================
# OPTIONAL EVO COMPARISON
# ============================================================

EVO_SUMMARY_FILE = (
    PROJECT_ROOT
    / "results"
    / "phase5b_federated_evo"
    / "phase5b_federated_evo_summary.json"
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
    "phase6_ga"
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
            f"Federated scenario missing:\n"
            f"{FEDERATED_DIR}"
        )

    print()
    print(
        "Search data source:"
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

    if (
        X.shape[1]
        !=
        EXPECTED_FEATURE_COUNT
    ):

        raise ValueError(
            f"\nClient {client_id} produced "
            f"{X.shape[1]} features.\n"
            f"Expected: "
            f"{EXPECTED_FEATURE_COUNT}"
        )

    # --------------------------------------------------------
    # SAME SAMPLE SIZE AS EVO
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
    # SAME INTERNAL SPLIT RULE AS EVO
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
                f"Client file missing:\n"
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

            class_name = inverse_mapping.get(
                class_id,
                str(
                    class_id
                ),
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
# SAVE MASK
# ============================================================

def save_feature_mask(
    feature_names,
    mask,
):

    mask = np.asarray(
        mask,
        dtype=bool,
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
):

    values = np.asarray(
        history,
        dtype=np.float64,
    ).reshape(-1)

    pd.DataFrame({

        "generation":
            np.arange(
                1,
                len(
                    values
                )
                + 1,
            ),

        "best_fitness":
            values,

    }).to_csv(
        CONVERGENCE_FILE,
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
# LOAD EVO SUMMARY FOR OPTIONAL COMPARISON
# ============================================================

def load_evo_summary():

    if not EVO_SUMMARY_FILE.exists():

        return None

    with open(
        EVO_SUMMARY_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


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

    separator()

    print(
        "PHASE 6 — FEDERATED-AWARE GA FEATURE SELECTION"
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
        f"GA population           : "
        f"{GA_POPULATION_SIZE}"
    )

    print(
        f"GA generations          : "
        f"{GA_GENERATIONS}"
    )

    print(
        f"Crossover rate          : "
        f"{GA_CROSSOVER_RATE}"
    )

    print(
        f"Mutation rate           : "
        f"{GA_MUTATION_RATE}"
    )


    # ========================================================
    # 1. SAFETY
    # ========================================================

    assert_search_data_is_safe()


    # ========================================================
    # 2. LOAD SAME PHASE-4 PREPROCESSING
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


    # ========================================================
    # 3. BENIGN CLASS
    # ========================================================

    if (
        phase4.BENIGN_LABEL
        not in
        label_mapping
    ):

        raise ValueError(
            "BENIGN label missing from label mapping."
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
    # 5. BUILD SAME FEDERATED PROXY SPLITS
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

    for client in (
        client_splits
    ):

        print(
            f"Client {client['client_id']} | "
            f"train={len(client['y_train']):,} | "
            f"val={len(client['y_val']):,}"
        )


    # ========================================================
    # 6. SELECTOR
    # ========================================================

    selector = (
        GAFederatedFeatureSelector(

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
    # 8. GA OPTIMIZER
    # ========================================================

    optimizer = (
        GeneticAlgorithmOptimizer(

            population_size=
                GA_POPULATION_SIZE,

            max_generations=
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

            random_state=
                SEED,

            verbose=True,

            objective=
                "minimize",
        )
    )


    # ========================================================
    # 9. RUN GA
    # ========================================================

    separator()

    print(
        "STARTING FEDERATED-AWARE GENETIC ALGORITHM"
    )

    print(
        f"Dimension : "
        f"{len(active_features)}"
    )

    print(
        f"Bounds    : "
        f"[{GA_LOWER_BOUND}, "
        f"{GA_UPPER_BOUND}]"
    )

    optimization_start = (
        time.perf_counter()
    )

    ga_result = (
        optimizer.optimize(

            objective_function=
                selector.fitness,

            dimension=
                len(
                    active_features
                ),

            lower_bound=
                GA_LOWER_BOUND,

            upper_bound=
                GA_UPPER_BOUND,
        )
    )

    optimization_seconds = (

        time.perf_counter()

        -
        optimization_start
    )


    # ========================================================
    # 10. FREEZE BEST GA MASK
    # ========================================================

    best_position = np.asarray(

        ga_result.best_position,

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
            ga_result.best_fitness
        ),

        best_result.fitness,

        atol=1e-8,
    ):

        print()

        print(
            "WARNING:"
        )

        print(
            "GA best fitness and selector fitness "
            "do not exactly match."
        )

        print(
            f"GA         : "
            f"{ga_result.best_fitness}"
        )

        print(
            f"Selector   : "
            f"{best_result.fitness}"
        )


    # ========================================================
    # 12. BEST GA RESULT
    # ========================================================

    print_result(

        title=
            "BEST FEDERATED-AWARE GA SUBSET",

        result=
            best_result,

        class_names=
            class_names,
    )


    # ========================================================
    # 13. BASELINE VS GA
    # ========================================================

    separator()

    print(
        "FEDERATED PROXY: 70 FEATURES VS GA"
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
    # 14. OPTIONAL EVO VS GA
    # ========================================================

    evo_summary = (
        load_evo_summary()
    )

    evo_vs_ga = None

    if evo_summary is not None:

        evo_best = evo_summary.get(
            "federated_evo_best"
        )

        if evo_best is not None:

            evo_vs_ga = {

                "evo_selected_features":
                    evo_best.get(
                        "selected_count"
                    ),

                "ga_selected_features":
                    best_result.selected_count,

                "evo_macro_f1":
                    evo_best.get(
                        "macro_f1"
                    ),

                "ga_macro_f1":
                    best_result.macro_f1,

                "evo_balanced_accuracy":
                    evo_best.get(
                        "balanced_accuracy"
                    ),

                "ga_balanced_accuracy":
                    best_result.balanced_accuracy,

                "evo_attack_macro_recall":
                    evo_best.get(
                        "attack_macro_recall"
                    ),

                "ga_attack_macro_recall":
                    best_result.attack_macro_recall,

                "evo_worst_attack_recall":
                    evo_best.get(
                        "worst_attack_recall"
                    ),

                "ga_worst_attack_recall":
                    best_result.worst_attack_recall,

                "evo_zero_attack_count":
                    evo_best.get(
                        "zero_attack_count"
                    ),

                "ga_zero_attack_count":
                    best_result.zero_attack_count,

                "evo_fitness":
                    evo_best.get(
                        "fitness"
                    ),

                "ga_fitness":
                    best_result.fitness,
            }

            print()

            print(
                "-" * 90
            )

            print(
                "FEDERATED-AWARE EVO VS GA — PROXY"
            )

            print(
                "-" * 90
            )

            print(
                f"EVO selected features : "
                f"{evo_vs_ga['evo_selected_features']}"
            )

            print(
                f"GA selected features  : "
                f"{evo_vs_ga['ga_selected_features']}"
            )

            print(
                f"EVO Macro F1          : "
                f"{evo_vs_ga['evo_macro_f1']}"
            )

            print(
                f"GA Macro F1           : "
                f"{evo_vs_ga['ga_macro_f1']:.6f}"
            )

            print(
                f"EVO Balanced Accuracy : "
                f"{evo_vs_ga['evo_balanced_accuracy']}"
            )

            print(
                f"GA Balanced Accuracy  : "
                f"{evo_vs_ga['ga_balanced_accuracy']:.6f}"
            )

            print(
                f"EVO Attack Recall     : "
                f"{evo_vs_ga['evo_attack_macro_recall']}"
            )

            print(
                f"GA Attack Recall      : "
                f"{evo_vs_ga['ga_attack_macro_recall']:.6f}"
            )

            print(
                f"EVO zero attacks      : "
                f"{evo_vs_ga['evo_zero_attack_count']}"
            )

            print(
                f"GA zero attacks       : "
                f"{evo_vs_ga['ga_zero_attack_count']}"
            )


    # ========================================================
    # 15. SAVE ARTIFACTS
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
        ga_result.convergence_history
    )


    # ========================================================
    # 16. SUMMARY
    # ========================================================

    total_seconds = (

        time.perf_counter()

        -
        total_start
    )

    summary = {

        "phase":
            6,

        "experiment":
            "Federated-Aware GA Feature Selection",

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

        "ga": {

            "population_size":
                GA_POPULATION_SIZE,

            "generations":
                GA_GENERATIONS,

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

            "lower_bound":
                GA_LOWER_BOUND,

            "upper_bound":
                GA_UPPER_BOUND,

            "completed_generations":
                int(
                    ga_result.generations
                ),

            "optimizer_evaluations":
                int(
                    ga_result.evaluations
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
                    ga_result.best_fitness
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

        "federated_ga_best":
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

            "zero_attack_count_change":
                int(
                    best_result.zero_attack_count
                    -
                    baseline_result.zero_attack_count
                ),
        },

        "evo_vs_ga_proxy":
            evo_vs_ga,

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
    # 17. FEATURES
    # ========================================================

    separator()

    print(
        "FEDERATED-AWARE GA SELECTED FEATURES"
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
    # 18. FINAL
    # ========================================================

    separator()

    print(
        "PHASE 6 COMPLETE — FEDERATED-AWARE GA"
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
        f"GA generations           : "
        f"{ga_result.generations}"
    )

    print(
        f"GA evaluations           : "
        f"{ga_result.evaluations}"
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
        "Official validation used during GA: NO ✅"
    )

    print(
        "Locked test used during GA         : NO ✅"
    )

    print(
        "Same federated proxy as EVO        : YES ✅"
    )

    print(
        "Same fitness weights as EVO        : YES ✅"
    )

    print(
        "Sample-weighted FedAvg proxy used   : YES ✅"
    )

    print(
        "Attack-collapse penalty used        : YES ✅"
    )

    print()

    print(
        f"GA feature mask:\n"
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