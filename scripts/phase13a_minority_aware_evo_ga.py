"""
Phase 13A - Local-Attack-Retention-Aware Hybrid EVO-GA Feature Selection
========================================================================

Research motivation
-------------------
The earlier Phase 13A global proxy showed that severe non-IID aggregation can
erase attack knowledge even when the local client model learns its own attack
very strongly. Therefore Phase 13A must not reject a feature subset merely
because a lightweight aggregated proxy loses that class.

This version separates FEATURE PRESERVATION from AGGREGATION BEHAVIOR.

Primary Phase 13A search flow
-----------------------------

    70-feature local baselines
            ->
    Hybrid EVO-GA candidate mask
            ->
    train each client locally from the same deterministic initialization
            ->
    evaluate that client's locally seen attack recall on validation
            ->
    calculate retention vs that client's 70-feature local baseline
            ->
    require EVERY client/attack retention >= 80% by default
            ->
    among feasible candidates, reward attack preservation + feature reduction
            ->
    freeze exactly ONE feature mask for Phase 13B

All five attack classes are protected symmetrically:

    Client 1 -> DoS Hulk
    Client 2 -> DDoS
    Client 3 -> PortScan
    Client 4 -> DoS GoldenEye
    Client 5 -> FTP-Patator

Search utility
--------------

    0.30 * Mean Local Seen-Attack Recall
    0.25 * Worst Local Seen-Attack Recall
    0.25 * Mean Local Attack Retention (capped at 1 for utility)
    0.10 * Worst Local Attack Retention (capped at 1 for utility)
    0.10 * Feature Reduction

Hard constraint
---------------

For every client k:

    candidate_seen_recall_k
        >=
    retention_ratio * baseline70_seen_recall_k

Default retention_ratio = 0.80.

This is an engineering / preregistered preservation criterion, not a universal
IDS standard.

IMPORTANT DATA POLICY
---------------------
USED:
    data/federated/<scenario>/client_1.csv ... client_5.csv
    data/processed/validation.csv

NOT USED:
    data/processed/test_locked.csv

The locked test is NEVER opened by this script.

The final feature mask is frozen before Phase 13B.

Proxy client construction
-------------------------
Each client proxy keeps approximately:
    80% BENIGN
    20% that client's locally observed attack

No oversampling.
No duplicated attack rows.
No synthetic data.

If a client has fewer than the desired number of attack records, all available
attack records are kept and the remaining capacity is filled with BENIGN.

Local-search training
---------------------
Each 70-feature baseline and every feature-mask candidate:
    - starts from the same deterministic 70-feature reference initialization;
    - slices only the selected first-layer input columns;
    - trains every client independently;
    - uses the existing weighted CrossEntropy + FedProx proximal term;
    - performs NO aggregation for the primary search objective.

The optional local-before-aggregation diagnostic is retained only to inspect
aggregation behavior. It never runs EVO-GA and never uses the locked test.

Recommended run order
---------------------
1. Local 70-feature baseline only:

python scripts\\phase13a_minority_aware_evo_ga.py ^
    --scenario non_iid ^
    --seed 42 ^
    --baseline-only

2. Full local-retention-aware EVO-GA search:

python scripts\\phase13a_minority_aware_evo_ga.py ^
    --scenario non_iid ^
    --seed 42

3. Optional aggregation diagnostic:

python scripts\\phase13a_minority_aware_evo_ga.py ^
    --scenario non_iid ^
    --seed 42 ^
    --proxy-rounds 10 ^
    --diagnose-local-before-aggregation

Outputs
-------
results/phase13/feature_search/

    phase13_candidate_history.csv
    phase13_selected_features.csv
    phase13_feature_mask.csv
    phase13_selected_features.json
    phase13_selected_mask.npy

    phase13_convergence.csv
    phase13_evo_convergence.csv
    phase13_ga_convergence.csv

    phase13_proxy_client_distribution.csv

    phase13_local_70_baseline.csv
    phase13_local_selected_comparison.csv
    phase13_full_validation_local_comparison.csv
    phase13_local_before_aggregation_diagnostic.csv

    phase13a_summary.json
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import logging
import random
import sys
import time

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# ============================================================
# EXISTING PROJECT INFRASTRUCTURE
# ============================================================

try:
    import phase9_fedavg70 as phase4
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "\nCould not import scripts\\phase9_fedavg70.py\n\n"
        "Keep this Phase 13A script inside the scripts folder."
    ) from exc

try:
    from optimization.hybrid_evo_ga import HybridEVOGAOptimizer
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "\nCould not import optimization\\hybrid_evo_ga.py\n\n"
        "Phase 13A reuses the existing HybridEVOGAOptimizer."
    ) from exc


# ============================================================
# GENERAL CONFIGURATION
# ============================================================

DEFAULT_SEED = 42
N_CLIENTS = 5
EXPECTED_FEATURES = 70
DEFAULT_SCENARIO = "non_iid"


# ============================================================
# DATA POLICY
# ============================================================

VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation.csv"
)

# IMPORTANT:
# This path is printed for audit documentation only.
# The file is NEVER opened in Phase 13A.
LOCKED_TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_locked.csv"
)


# ============================================================
# PROXY DATA
# ============================================================

MAX_ROWS_PER_CLIENT = 20_000
ATTACK_FRACTION = 0.20
DEFAULT_VALIDATION_ROWS_PER_CLASS = 10_000


# ============================================================
# LOCAL SEARCH TRAINING
# ============================================================

DEFAULT_LOCAL_SEARCH_EPOCHS = 1
DEFAULT_LOCAL_SEARCH_MU = 0.01

PROXY_BATCH_SIZE = 1024
PROXY_LEARNING_RATE = 0.001
PROXY_WEIGHT_DECAY = 0.0001
PROXY_GRADIENT_CLIP = 5.0


# ============================================================
# OPTIONAL FEDERATED DIAGNOSTIC
# ============================================================

DEFAULT_PROXY_FEDERATED_ROUNDS = 10


# ============================================================
# LOCAL BASELINE VIABILITY
# ============================================================

BASELINE_MIN_ATTACK_RECALL = 1e-6


# ============================================================
# FEATURE SEARCH
# ============================================================

SEARCH_LOWER_BOUND = -6.0
SEARCH_UPPER_BOUND = 6.0
MASK_THRESHOLD = 0.50
MIN_FEATURES = 5


# ============================================================
# HYBRID EVO-GA
# ============================================================

DEFAULT_EVO_POPULATION = 10
DEFAULT_EVO_ITERATIONS = 10

DEFAULT_GA_POPULATION = 10
DEFAULT_GA_GENERATIONS = 10

GA_CROSSOVER_RATE = 0.90
GA_MUTATION_RATE = 0.10
GA_MUTATION_SCALE = 0.15
GA_TOURNAMENT_SIZE = 3
GA_ELITISM_COUNT = 1

EVO_ELITE_FRACTION = 0.50
BEST_SOLUTION_COPIES = 1
SEEDED_MUTATION_SCALE = 0.05
RANDOM_INJECTION_FRACTION = 0.10


# ============================================================
# LOCAL-RETENTION-AWARE FITNESS
# ============================================================

MEAN_LOCAL_ATTACK_RECALL_WEIGHT = 0.30
WORST_LOCAL_ATTACK_RECALL_WEIGHT = 0.25
MEAN_LOCAL_RETENTION_WEIGHT = 0.25
WORST_LOCAL_RETENTION_WEIGHT = 0.10
FEATURE_REDUCTION_WEIGHT = 0.10

FITNESS_WEIGHT_SUM = (
    MEAN_LOCAL_ATTACK_RECALL_WEIGHT
    + WORST_LOCAL_ATTACK_RECALL_WEIGHT
    + MEAN_LOCAL_RETENTION_WEIGHT
    + WORST_LOCAL_RETENTION_WEIGHT
    + FEATURE_REDUCTION_WEIGHT
)

if not np.isclose(FITNESS_WEIGHT_SUM, 1.0):
    raise ValueError(
        "\nPhase 13A fitness weights must sum to 1.0.\n"
        f"Current sum = {FITNESS_WEIGHT_SUM}"
    )


# ============================================================
# RETENTION CONSTRAINT
# ============================================================

DEFAULT_LOCAL_RETENTION = 0.80


# ============================================================
# ATTACK MAP
# ============================================================

# Use the project's existing exact client -> attack mapping.
CLIENT_ATTACKS = {
    int(client_id): str(attack)
    for client_id, attack
    in phase4.CLIENT_ATTACKS.items()
}

if set(CLIENT_ATTACKS.keys()) != set(range(1, N_CLIENTS + 1)):
    raise ValueError(
        "\nPhase 13A expected exactly five client attack mappings "
        f"for clients 1..{N_CLIENTS}.\n"
        f"Found: {CLIENT_ATTACKS}"
    )

ALL_ATTACKS = [
    CLIENT_ATTACKS[client_id]
    for client_id
    in range(1, N_CLIENTS + 1)
]


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "phase13"
    / "feature_search"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

CANDIDATE_HISTORY_FILE = (
    OUTPUT_ROOT
    / "phase13_candidate_history.csv"
)

SELECTED_FEATURES_FILE = (
    OUTPUT_ROOT
    / "phase13_selected_features.csv"
)

FEATURE_MASK_FILE = (
    OUTPUT_ROOT
    / "phase13_feature_mask.csv"
)

SELECTED_FEATURES_JSON = (
    OUTPUT_ROOT
    / "phase13_selected_features.json"
)

SELECTED_MASK_NPY = (
    OUTPUT_ROOT
    / "phase13_selected_mask.npy"
)

CONVERGENCE_FILE = (
    OUTPUT_ROOT
    / "phase13_convergence.csv"
)

EVO_CONVERGENCE_FILE = (
    OUTPUT_ROOT
    / "phase13_evo_convergence.csv"
)

GA_CONVERGENCE_FILE = (
    OUTPUT_ROOT
    / "phase13_ga_convergence.csv"
)

CLIENT_DISTRIBUTION_FILE = (
    OUTPUT_ROOT
    / "phase13_proxy_client_distribution.csv"
)

LOCAL_BASELINE_FILE = (
    OUTPUT_ROOT
    / "phase13_local_70_baseline.csv"
)

LOCAL_SELECTED_COMPARISON_FILE = (
    OUTPUT_ROOT
    / "phase13_local_selected_comparison.csv"
)

FULL_VALIDATION_LOCAL_COMPARISON_FILE = (
    OUTPUT_ROOT
    / "phase13_full_validation_local_comparison.csv"
)

LOCAL_DIAGNOSTIC_FILE = (
    OUTPUT_ROOT
    / "phase13_local_before_aggregation_diagnostic.csv"
)

BASELINE_DIAGNOSTIC_FILE = (
    OUTPUT_ROOT
    / "phase13_local_baseline_diagnostic.json"
)

SUMMARY_FILE = (
    OUTPUT_ROOT
    / "phase13a_summary.json"
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
    "phase13a_local_retention_evo_ga"
)


# ============================================================
# RESULT OBJECT
# ============================================================

@dataclass
class LocalRetentionResult:
    mask_key: str

    selected_count: int
    total_features: int
    feature_reduction: float

    selected_indices: list[int]
    selected_features: list[str]

    local_attack_recalls: dict[str, float]
    local_baseline_recalls: dict[str, float]
    local_thresholds: dict[str, float]
    local_retention_ratios: dict[str, float]

    mean_local_attack_recall: float
    worst_local_attack_recall: float

    mean_local_retention: float
    worst_local_retention: float

    feasible: bool
    total_constraint_violation: float

    utility: float
    fitness: float

    evaluation_seconds: float


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# JSON SAFE
# ============================================================

def json_safe(
    value: Any,
):
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item
            in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item
            in value
        ]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    return value


def save_json(
    path: Path,
    payload: dict,
) -> None:
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


# ============================================================
# DISPLAY
# ============================================================

def separator():
    print()
    print("=" * 100)


# ============================================================
# DATA SAFETY
# ============================================================

def assert_data_integrity(
    federated_dir: Path,
):
    if not federated_dir.exists():
        raise FileNotFoundError(
            "\nFederated training directory missing:\n"
            f"{federated_dir}"
        )

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            "\nValidation file missing:\n"
            f"{VALIDATION_FILE}"
        )

    print()
    print("PHASE 13A DATA POLICY")
    print("-" * 100)
    print(f"Federated TRAIN source : {federated_dir}")
    print(f"Validation source      : {VALIDATION_FILE}")
    print("Locked test used       : NO")
    print(f"Locked test path       : {LOCKED_TEST_FILE}")
    print("Locked-test evaluation : FORBIDDEN IN PHASE 13A")


# ============================================================
# FEATURE SPACE
# ============================================================

def verify_feature_space(
    active_features,
    active_indices,
    feature_columns,
):
    if len(active_features) != EXPECTED_FEATURES:
        raise ValueError(
            "\nPhase 13A requires exactly "
            f"{EXPECTED_FEATURES} active features.\n"
            f"Found {len(active_features)}."
        )

    active_indices = np.asarray(
        active_indices,
        dtype=np.int64,
    )

    reconstructed = [
        feature_columns[int(index)]
        for index
        in active_indices
    ]

    if reconstructed != list(active_features):
        raise ValueError(
            "\nActive feature/index order mismatch."
        )

    return active_indices


# ============================================================
# CLASS NAMES
# ============================================================

def build_class_names(
    label_mapping,
):
    inverse_mapping = {
        int(value): str(label)
        for label, value
        in label_mapping.items()
    }

    return [
        inverse_mapping[class_id]
        for class_id
        in range(len(label_mapping))
    ]


# ============================================================
# FILTER TO SIX PROJECT CLASSES
# ============================================================

def filter_to_project_classes(
    dataframe: pd.DataFrame,
    label_mapping,
):
    if phase4.LABEL_COL not in dataframe.columns:
        raise KeyError(
            "\nMissing label column:\n"
            f"{phase4.LABEL_COL}"
        )

    canonical = {
        str(label).strip().lower():
            str(label).strip()
        for label
        in label_mapping.keys()
    }

    labels = (
        dataframe[phase4.LABEL_COL]
        .astype(str)
        .str.strip()
    )

    keep_mask = (
        labels
        .str.lower()
        .isin(canonical.keys())
    )

    filtered = (
        dataframe
        .loc[keep_mask]
        .copy()
    )

    filtered[phase4.LABEL_COL] = (
        filtered[phase4.LABEL_COL]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(canonical)
    )

    return filtered


# ============================================================
# ATTACK-PRESERVING PROXY SAMPLER
# ============================================================

def build_attack_preserving_proxy(
    X,
    y,
    client_id,
    label_mapping,
    seed,
    max_rows=MAX_ROWS_PER_CLIENT,
    attack_fraction=ATTACK_FRACTION,
):
    """
    Build an 80/20 severe non-IID proxy without oversampling.

    Desired:
        80% BENIGN
        20% client's locally observed attack

    No duplicate attack rows.
    No synthetic records.
    """

    benign_id = int(
        label_mapping[
            phase4.BENIGN_LABEL
        ]
    )

    attack_name = CLIENT_ATTACKS[
        client_id
    ]

    attack_id = int(
        label_mapping[
            attack_name
        ]
    )

    benign_indices = np.flatnonzero(
        y == benign_id
    )

    attack_indices = np.flatnonzero(
        y == attack_id
    )

    if len(attack_indices) == 0:
        raise RuntimeError(
            f"Client {client_id} contains no "
            f"{attack_name} samples."
        )

    rng = np.random.default_rng(
        seed
        +
        client_id
        *
        1000
    )

    desired_attack = int(
        round(
            max_rows
            *
            attack_fraction
        )
    )

    attack_count = min(
        desired_attack,
        len(attack_indices),
    )

    selected_attack = rng.choice(
        attack_indices,
        size=attack_count,
        replace=False,
    )

    remaining_capacity = (
        max_rows
        -
        attack_count
    )

    benign_count = min(
        remaining_capacity,
        len(benign_indices),
    )

    selected_benign = rng.choice(
        benign_indices,
        size=benign_count,
        replace=False,
    )

    selected_indices = np.concatenate(
        [
            selected_benign,
            selected_attack,
        ]
    )

    rng.shuffle(
        selected_indices
    )

    X_proxy = X[
        selected_indices
    ]

    y_proxy = y[
        selected_indices
    ]

    print()
    print(
        f"Client {client_id} attack-preserving proxy"
    )
    print(
        f"  Attack       : {attack_name}"
    )
    print(
        f"  BENIGN       : {benign_count}"
    )
    print(
        f"  Attack       : {attack_count}"
    )
    print(
        f"  Total        : {len(y_proxy)}"
    )
    print(
        f"  Attack ratio : "
        f"{attack_count / len(y_proxy) * 100:.2f}%"
    )

    return (
        X_proxy.astype(
            np.float32,
            copy=False,
        ),
        y_proxy.astype(
            np.int64,
            copy=False,
        ),
    )


# ============================================================
# LOAD CLIENT PROXIES
# ============================================================

def load_proxy_clients(
    federated_dir,
    scaler,
    feature_columns,
    active_indices,
    label_mapping,
    seed,
):
    clients = []
    distribution_rows = []

    inverse_mapping = {
        int(value): label
        for label, value
        in label_mapping.items()
    }

    for client_id in range(
        1,
        N_CLIENTS + 1,
    ):
        client_file = (
            federated_dir
            /
            f"client_{client_id}.csv"
        )

        if not client_file.exists():
            raise FileNotFoundError(
                "\nMissing client file:\n"
                f"{client_file}"
            )

        logger.info(
            "Loading proxy client %d...",
            client_id,
        )

        dataframe = pd.read_csv(
            client_file
        )

        dataframe = filter_to_project_classes(
            dataframe,
            label_mapping,
        )

        original_rows = len(
            dataframe
        )

        X = phase4.transform_features(
            df=dataframe,
            feature_columns=feature_columns,
            scaler=scaler,
            active_indices=active_indices,
        )

        y = phase4.encode_labels(
            dataframe[
                phase4.LABEL_COL
            ],
            label_mapping,
        )

        del dataframe
        gc.collect()

        if X.shape[1] != EXPECTED_FEATURES:
            raise ValueError(
                f"\nClient {client_id} generated "
                f"{X.shape[1]} active features.\n"
                f"Expected {EXPECTED_FEATURES}."
            )

        X, y = build_attack_preserving_proxy(
            X=X,
            y=y,
            client_id=client_id,
            label_mapping=label_mapping,
            seed=seed,
            max_rows=MAX_ROWS_PER_CLIENT,
            attack_fraction=ATTACK_FRACTION,
        )

        class_counts = np.bincount(
            y,
            minlength=len(label_mapping),
        )

        row = {
            "client_id":
                client_id,

            "local_seen_attack":
                CLIENT_ATTACKS[
                    client_id
                ],

            "original_rows":
                original_rows,

            "proxy_rows":
                len(y),
        }

        for class_id, count in enumerate(
            class_counts
        ):
            row[
                f"count_{inverse_mapping[class_id]}"
            ] = int(
                count
            )

        distribution_rows.append(
            row
        )

        clients.append(
            {
                "client_id":
                    client_id,

                "attack_name":
                    CLIENT_ATTACKS[
                        client_id
                    ],

                "X":
                    X.astype(
                        np.float32,
                        copy=False,
                    ),

                "y":
                    y.astype(
                        np.int64,
                        copy=False,
                    ),
            }
        )

        logger.info(
            "Client %d | original=%d | proxy=%d",
            client_id,
            original_rows,
            len(y),
        )

    pd.DataFrame(
        distribution_rows
    ).to_csv(
        CLIENT_DISTRIBUTION_FILE,
        index=False,
    )

    return clients


# ============================================================
# LOAD SEARCH VALIDATION
# ============================================================

def load_validation_data(
    scaler,
    feature_columns,
    active_indices,
    label_mapping,
    rows_per_class,
    seed,
):
    logger.info(
        "Loading official validation data..."
    )

    dataframe = pd.read_csv(
        VALIDATION_FILE
    )

    total_original_rows = len(
        dataframe
    )

    dataframe = filter_to_project_classes(
        dataframe,
        label_mapping,
    )

    retained_six_class_rows = len(
        dataframe
    )

    sampled_frames = []

    for class_name in label_mapping.keys():
        class_frame = dataframe[
            dataframe[
                phase4.LABEL_COL
            ]
            ==
            class_name
        ]

        if class_frame.empty:
            raise RuntimeError(
                "\nValidation data is missing class:\n"
                f"{class_name}"
            )

        if len(class_frame) > rows_per_class:
            class_frame = class_frame.sample(
                n=rows_per_class,
                random_state=(
                    seed
                    +
                    int(
                        label_mapping[
                            class_name
                        ]
                    )
                    *
                    100
                ),
                replace=False,
            )

        sampled_frames.append(
            class_frame
        )

    search_dataframe = (
        pd.concat(
            sampled_frames,
            ignore_index=True,
        )
        .sample(
            frac=1.0,
            random_state=(
                seed
                +
                1300
            ),
        )
        .reset_index(
            drop=True
        )
    )

    logger.info(
        "Validation original rows      : %d",
        total_original_rows,
    )

    logger.info(
        "Validation six-class rows     : %d",
        retained_six_class_rows,
    )

    logger.info(
        "Validation search rows        : %d",
        len(search_dataframe),
    )

    print()
    print(
        "VALIDATION SEARCH CLASS DISTRIBUTION"
    )
    print(
        search_dataframe[
            phase4.LABEL_COL
        ]
        .value_counts()
        .to_string()
    )

    X_validation = phase4.transform_features(
        df=search_dataframe,
        feature_columns=feature_columns,
        scaler=scaler,
        active_indices=active_indices,
    )

    y_validation = phase4.encode_labels(
        search_dataframe[
            phase4.LABEL_COL
        ],
        label_mapping,
    )

    del search_dataframe
    del dataframe
    gc.collect()

    return (
        X_validation.astype(
            np.float32,
            copy=False,
        ),
        y_validation.astype(
            np.int64,
            copy=False,
        ),
        {
            "original_validation_rows":
                int(
                    total_original_rows
                ),

            "six_class_validation_rows":
                int(
                    retained_six_class_rows
                ),

            "search_validation_rows":
                int(
                    len(
                        y_validation
                    )
                ),
        },
    )


# ============================================================
# LOAD FULL SIX-CLASS VALIDATION
# ============================================================

def load_full_validation_data(
    scaler,
    feature_columns,
    active_indices,
    label_mapping,
):
    dataframe = pd.read_csv(
        VALIDATION_FILE
    )

    dataframe = filter_to_project_classes(
        dataframe,
        label_mapping,
    )

    X = phase4.transform_features(
        df=dataframe,
        feature_columns=feature_columns,
        scaler=scaler,
        active_indices=active_indices,
    )

    y = phase4.encode_labels(
        dataframe[
            phase4.LABEL_COL
        ],
        label_mapping,
    )

    del dataframe
    gc.collect()

    return (
        X.astype(
            np.float32,
            copy=False,
        ),
        y.astype(
            np.int64,
            copy=False,
        ),
    )


# ============================================================
# MASK FUNCTIONS
# ============================================================

def sigmoid(
    values,
):
    values = np.clip(
        values,
        -60.0,
        60.0,
    )

    return (
        1.0
        /
        (
            1.0
            +
            np.exp(
                -values
            )
        )
    )


def position_to_mask(
    position,
    min_features,
):
    position = np.asarray(
        position,
        dtype=np.float64,
    )

    probabilities = sigmoid(
        position
    )

    mask = (
        probabilities
        >=
        MASK_THRESHOLD
    )

    selected_count = int(
        mask.sum()
    )

    if selected_count < min_features:
        strongest = np.argsort(
            probabilities
        )[
            -min_features:
        ]

        mask = np.zeros(
            len(position),
            dtype=bool,
        )

        mask[
            strongest
        ] = True

    return mask


def mask_to_key(
    mask,
):
    bits = "".join(
        "1"
        if value
        else "0"
        for value
        in mask
    )

    return hashlib.sha256(
        bits.encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================
# FAIR INITIALIZATION
# ============================================================

def build_initial_global_state(
    selected_positions,
    label_mapping,
    seed,
):
    """
    Build every candidate from the same deterministic 70-feature
    reference model.

    For reduced masks:
        - slice only the input columns of the first Linear layer;
        - copy all remaining layer parameters unchanged.

    This makes feature-mask comparisons much fairer than creating a
    completely independent random network per input dimension.
    """

    set_seed(
        seed
        +
        13_000
    )

    reference_model = phase4.IDSMLP(
        input_dim=
            EXPECTED_FEATURES,

        num_classes=
            len(
                label_mapping
            ),
    )

    selected_model = phase4.IDSMLP(
        input_dim=
            len(
                selected_positions
            ),

        num_classes=
            len(
                label_mapping
            ),
    )

    reference_state = (
        reference_model
        .state_dict()
    )

    selected_state = (
        selected_model
        .state_dict()
    )

    first_layer_key = (
        "network.0.weight"
    )

    if first_layer_key not in reference_state:
        raise KeyError(
            "\nExpected first IDSMLP layer key was not found:\n"
            f"{first_layer_key}\n\n"
            "Check phase9_fedavg70.IDSMLP architecture."
        )

    positions = [
        int(position)
        for position
        in selected_positions
    ]

    for key in selected_state.keys():
        if key == first_layer_key:
            selected_state[
                key
            ] = (
                reference_state[
                    key
                ][
                    :,
                    positions
                ]
                .clone()
            )
        else:
            selected_state[
                key
            ] = (
                reference_state[
                    key
                ]
                .clone()
            )

    del reference_model
    del selected_model

    return {
        key:
            tensor
            .detach()
            .cpu()
            .clone()

        for key, tensor
        in selected_state.items()
    }


# ============================================================
# FEDPROX TERM
# ============================================================

def calculate_proximal_term(
    model,
    global_parameters,
):
    proximal_term = torch.zeros(
        (),
        device=
            next(
                model.parameters()
            ).device,
    )

    for (
        local_parameter,
        global_parameter,
    ) in zip(
        model.parameters(),
        global_parameters,
    ):
        proximal_term = (
            proximal_term
            +
            torch.sum(
                (
                    local_parameter
                    -
                    global_parameter
                )
                ** 2
            )
        )

    return proximal_term


# ============================================================
# TRAIN ONE LOCAL CLIENT
# ============================================================

def train_local_client(
    initial_state,
    X,
    y,
    selected_positions,
    label_mapping,
    class_weights,
    device,
    client_id,
    seed,
    local_epochs,
    local_mu,
    seed_round_number=1,
):
    """
    Train one client independently from the provided initial state.

    There is NO aggregation inside this function.

    The proximal anchor is the initial state supplied to the local client.
    """

    model = (
        phase4.IDSMLP(
            input_dim=
                len(
                    selected_positions
                ),

            num_classes=
                len(
                    label_mapping
                ),
        )
        .to(
            device
        )
    )

    model.load_state_dict(
        initial_state
    )

    global_parameters = [
        parameter
        .detach()
        .clone()
        .to(
            device
        )

        for parameter
        in model.parameters()
    ]

    for parameter in global_parameters:
        parameter.requires_grad_(
            False
        )

    criterion = nn.CrossEntropyLoss(
        weight=
            torch.tensor(
                class_weights,
                dtype=torch.float32,
                device=device,
            )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PROXY_LEARNING_RATE,
        weight_decay=PROXY_WEIGHT_DECAY,
    )

    model.train()

    for local_epoch in range(
        1,
        local_epochs + 1,
    ):
        local_seed = (
            seed
            +
            client_id
            *
            100
            +
            seed_round_number
            *
            1000
            +
            local_epoch
        )

        random.seed(
            local_seed
        )
        np.random.seed(
            local_seed
        )
        torch.manual_seed(
            local_seed
        )

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                local_seed
            )

        rng = np.random.default_rng(
            local_seed
        )

        indices = rng.permutation(
            len(y)
        )

        for start in range(
            0,
            len(indices),
            PROXY_BATCH_SIZE,
        ):
            batch_indices = indices[
                start:
                start
                +
                PROXY_BATCH_SIZE
            ]

            X_batch = torch.from_numpy(
                X[
                    batch_indices
                ][
                    :,
                    selected_positions
                ]
                .astype(
                    np.float32,
                    copy=False,
                )
            ).to(
                device
            )

            y_batch = torch.from_numpy(
                y[
                    batch_indices
                ]
                .astype(
                    np.int64,
                    copy=False,
                )
            ).to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                X_batch
            )

            ce_loss = criterion(
                logits,
                y_batch,
            )

            proximal_term = (
                calculate_proximal_term(
                    model,
                    global_parameters,
                )
            )

            loss = (
                ce_loss
                +
                (
                    local_mu
                    /
                    2.0
                )
                *
                proximal_term
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=
                    PROXY_GRADIENT_CLIP,
            )

            optimizer.step()

    state = {
        key:
            tensor
            .detach()
            .cpu()
            .clone()

        for key, tensor
        in model.state_dict().items()
    }

    del model
    del global_parameters

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return state


# ============================================================
# EVALUATE MODEL / STATE
# ============================================================

@torch.no_grad()
def evaluate_model(
    model,
    X,
    y,
    selected_positions,
    label_mapping,
    device,
):
    model.eval()

    predictions = []

    for start in range(
        0,
        len(y),
        PROXY_BATCH_SIZE,
    ):
        end = min(
            start
            +
            PROXY_BATCH_SIZE,
            len(y),
        )

        X_batch = torch.from_numpy(
            X[
                start:end
            ][
                :,
                selected_positions
            ]
            .astype(
                np.float32,
                copy=False,
            )
        ).to(
            device
        )

        logits = model(
            X_batch
        )

        predictions.append(
            logits
            .argmax(
                dim=1
            )
            .cpu()
            .numpy()
        )

    y_pred = np.concatenate(
        predictions
    )

    (
        overall,
        report,
        confusion_matrix,
        class_names,

    ) = phase4.calculate_metrics(
        y,
        y_pred,
        label_mapping,
    )

    return (
        overall,
        report,
        confusion_matrix,
        class_names,
    )


def evaluate_state_on_validation(
    state,
    X_validation,
    y_validation,
    selected_positions,
    label_mapping,
    device,
):
    model = (
        phase4.IDSMLP(
            input_dim=
                len(
                    selected_positions
                ),

            num_classes=
                len(
                    label_mapping
                ),
        )
        .to(
            device
        )
    )

    model.load_state_dict(
        state
    )

    result = evaluate_model(
        model=
            model,
        X=
            X_validation,
        y=
            y_validation,
        selected_positions=
            selected_positions,
        label_mapping=
            label_mapping,
        device=
            device,
    )

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ============================================================
# LOCAL MASK EVALUATION
# ============================================================

def evaluate_local_mask(
    clients,
    X_validation,
    y_validation,
    selected_positions,
    label_mapping,
    class_weights,
    device,
    seed,
    local_epochs,
    local_mu,
):
    """
    Primary Phase 13A evaluator.

    No aggregation is performed.

    Every client:
        1. starts from the same deterministic candidate initialization,
        2. trains independently,
        3. is evaluated on the common validation data,
        4. contributes only its OWN seen-attack recall to the primary
           feature-preservation objective.
    """

    selected_positions = np.asarray(
        selected_positions,
        dtype=np.int64,
    )

    initial_state = build_initial_global_state(
        selected_positions=
            selected_positions,
        label_mapping=
            label_mapping,
        seed=
            seed,
    )

    per_client = {}

    for client in clients:
        client_id = int(
            client[
                "client_id"
            ]
        )

        attack_name = CLIENT_ATTACKS[
            client_id
        ]

        state = train_local_client(
            initial_state=
                initial_state,
            X=
                client[
                    "X"
                ],
            y=
                client[
                    "y"
                ],
            selected_positions=
                selected_positions,
            label_mapping=
                label_mapping,
            class_weights=
                class_weights,
            device=
                device,
            client_id=
                client_id,
            seed=
                seed,
            local_epochs=
                local_epochs,
            local_mu=
                local_mu,
            seed_round_number=
                1,
        )

        (
            overall,
            report,
            _,
            _,

        ) = evaluate_state_on_validation(
            state=
                state,
            X_validation=
                X_validation,
            y_validation=
                y_validation,
            selected_positions=
                selected_positions,
            label_mapping=
                label_mapping,
            device=
                device,
        )

        seen_recall = float(
            report[
                attack_name
            ][
                "recall"
            ]
        )

        per_client[
            client_id
        ] = {
            "client_id":
                client_id,

            "attack_name":
                attack_name,

            "seen_attack_recall":
                seen_recall,

            "accuracy":
                float(
                    overall[
                        "accuracy"
                    ]
                ),

            "balanced_accuracy":
                float(
                    overall[
                        "balanced_accuracy"
                    ]
                ),

            "macro_f1":
                float(
                    overall[
                        "macro_f1"
                    ]
                ),
        }

        del state

    return per_client


# ============================================================
# OPTIONAL LOCAL-BEFORE-AGGREGATION DIAGNOSTIC
# ============================================================

def run_local_before_aggregation_diagnostic(
    clients,
    X_validation,
    y_validation,
    label_mapping,
    class_weights,
    device,
    seed,
    proxy_rounds,
    local_epochs,
    local_mu,
):
    """
    Diagnostic only.

    For every federated round:
        - train local models from current global state;
        - measure local seen-attack recall BEFORE aggregation;
        - aggregate with the project's normal sample-weighted FedAvg;
        - measure global recall AFTER aggregation.

    This mode never runs EVO-GA and never uses the locked test.
    """

    separator()
    print(
        "PHASE 13A — LOCAL-BEFORE-AGGREGATION DIAGNOSTIC"
    )
    separator()

    print(
        "Purpose: inspect whether attack knowledge is "
        "lost during aggregation."
    )
    print()
    print("EVO-GA executed : NO")
    print("Locked test used: NO")
    print(f"Rounds          : {proxy_rounds}")
    print(f"Local epochs    : {local_epochs}")
    print(f"FedProx mu      : {local_mu}")

    selected_positions = np.arange(
        EXPECTED_FEATURES,
        dtype=np.int64,
    )

    global_state = build_initial_global_state(
        selected_positions=
            selected_positions,
        label_mapping=
            label_mapping,
        seed=
            seed,
    )

    class_names = build_class_names(
        label_mapping
    )

    rows = []

    for round_number in range(
        1,
        proxy_rounds + 1,
    ):
        separator()
        print(
            f"ROUND {round_number}"
        )
        print("-" * 100)

        client_states = []
        client_weights = []

        for client in clients:
            client_id = int(
                client[
                    "client_id"
                ]
            )

            attack_name = CLIENT_ATTACKS[
                client_id
            ]

            state = train_local_client(
                initial_state=
                    global_state,
                X=
                    client[
                        "X"
                    ],
                y=
                    client[
                        "y"
                    ],
                selected_positions=
                    selected_positions,
                label_mapping=
                    label_mapping,
                class_weights=
                    class_weights,
                device=
                    device,
                client_id=
                    client_id,
                seed=
                    seed,
                local_epochs=
                    local_epochs,
                local_mu=
                    local_mu,
                seed_round_number=
                    round_number,
            )

            (
                local_overall,
                local_report,
                _,
                _,

            ) = evaluate_state_on_validation(
                state=
                    state,
                X_validation=
                    X_validation,
                y_validation=
                    y_validation,
                selected_positions=
                    selected_positions,
                label_mapping=
                    label_mapping,
                device=
                    device,
            )

            local_seen_recall = float(
                local_report[
                    attack_name
                ][
                    "recall"
                ]
            )

            print(
                f"Client {client_id} | "
                f"seen attack={attack_name:20s} | "
                f"LOCAL recall before aggregation="
                f"{local_seen_recall:.6f}"
            )

            row = {
                "round":
                    round_number,

                "stage":
                    "local_before_aggregation",

                "client_id":
                    client_id,

                "seen_attack":
                    attack_name,

                "seen_attack_recall":
                    local_seen_recall,

                "accuracy":
                    float(
                        local_overall[
                            "accuracy"
                        ]
                    ),

                "balanced_accuracy":
                    float(
                        local_overall[
                            "balanced_accuracy"
                        ]
                    ),

                "macro_f1":
                    float(
                        local_overall[
                            "macro_f1"
                        ]
                    ),
            }

            for class_name in class_names:
                row[
                    f"recall_{class_name}"
                ] = float(
                    local_report[
                        class_name
                    ][
                        "recall"
                    ]
                )

            rows.append(
                row
            )

            client_states.append(
                state
            )

            client_weights.append(
                len(
                    client[
                        "y"
                    ]
                )
            )

        global_state = phase4.federated_average(
            client_states=
                client_states,
            client_weights=
                client_weights,
        )

        (
            global_overall,
            global_report,
            _,
            _,

        ) = evaluate_state_on_validation(
            state=
                global_state,
            X_validation=
                X_validation,
            y_validation=
                y_validation,
            selected_positions=
                selected_positions,
            label_mapping=
                label_mapping,
            device=
                device,
        )

        print()
        print(
            "GLOBAL MODEL AFTER AGGREGATION"
        )
        print("-" * 100)

        for class_name in class_names:
            print(
                f"{class_name:25s} "
                f"recall="
                f"{global_report[class_name]['recall']:.6f}"
            )

        global_row = {
            "round":
                round_number,

            "stage":
                "global_after_aggregation",

            "client_id":
                0,

            "seen_attack":
                "GLOBAL",

            "seen_attack_recall":
                np.nan,

            "accuracy":
                float(
                    global_overall[
                        "accuracy"
                    ]
                ),

            "balanced_accuracy":
                float(
                    global_overall[
                        "balanced_accuracy"
                    ]
                ),

            "macro_f1":
                float(
                    global_overall[
                        "macro_f1"
                    ]
                ),
        }

        for class_name in class_names:
            global_row[
                f"recall_{class_name}"
            ] = float(
                global_report[
                    class_name
                ][
                    "recall"
                ]
            )

        rows.append(
            global_row
        )

    diagnostic_df = pd.DataFrame(
        rows
    )

    diagnostic_df.to_csv(
        LOCAL_DIAGNOSTIC_FILE,
        index=False,
    )

    separator()
    print(
        "LOCAL-BEFORE-AGGREGATION DIAGNOSTIC COMPLETE"
    )
    separator()

    print()
    print(
        "FINAL-ROUND LOCAL VS GLOBAL SEEN-ATTACK COMPARISON"
    )
    print("-" * 100)

    final_round = proxy_rounds

    final_global_rows = diagnostic_df[
        (
            diagnostic_df[
                "round"
            ]
            ==
            final_round
        )
        &
        (
            diagnostic_df[
                "stage"
            ]
            ==
            "global_after_aggregation"
        )
    ]

    if final_global_rows.empty:
        raise RuntimeError(
            "Final global diagnostic row was not produced."
        )

    final_global_row = final_global_rows.iloc[
        0
    ]

    for client_id in range(
        1,
        N_CLIENTS + 1,
    ):
        attack_name = CLIENT_ATTACKS[
            client_id
        ]

        local_rows = diagnostic_df[
            (
                diagnostic_df[
                    "round"
                ]
                ==
                final_round
            )
            &
            (
                diagnostic_df[
                    "stage"
                ]
                ==
                "local_before_aggregation"
            )
            &
            (
                diagnostic_df[
                    "client_id"
                ]
                ==
                client_id
            )
        ]

        if local_rows.empty:
            continue

        local_recall = float(
            local_rows.iloc[
                0
            ][
                f"recall_{attack_name}"
            ]
        )

        global_recall = float(
            final_global_row[
                f"recall_{attack_name}"
            ]
        )

        aggregation_delta = (
            global_recall
            -
            local_recall
        )

        print(
            f"Client {client_id} | "
            f"{attack_name:20s} | "
            f"local={local_recall:.6f} | "
            f"global={global_recall:.6f} | "
            f"aggregation delta="
            f"{aggregation_delta:+.6f}"
        )

    print()
    print("Diagnostic saved:")
    print(LOCAL_DIAGNOSTIC_FILE)
    print()
    print("EVO-GA executed : NO")
    print("Locked test used: NO")

    return diagnostic_df


# ============================================================
# LOCAL RETENTION SELECTOR
# ============================================================

class LocalAttackRetentionSelector:

    def __init__(
        self,
        clients,
        X_validation,
        y_validation,
        feature_names,
        label_mapping,
        class_weights,
        retention_ratio,
        seed,
        device,
        local_epochs,
        local_mu,
    ):
        self.clients = clients

        self.X_validation = (
            X_validation
        )

        self.y_validation = (
            y_validation
        )

        self.feature_names = list(
            feature_names
        )

        self.total_features = len(
            self.feature_names
        )

        self.label_mapping = (
            label_mapping
        )

        self.class_weights = (
            class_weights
        )

        self.retention_ratio = float(
            retention_ratio
        )

        self.seed = int(
            seed
        )

        self.device = (
            device
        )

        self.local_epochs = int(
            local_epochs
        )

        self.local_mu = float(
            local_mu
        )

        self.cache: dict[
            str,
            LocalRetentionResult
        ] = {}

        self.candidate_history: list[
            dict
        ] = []

        self.unique_evaluations = 0
        self.cache_hits = 0

        self.baseline_result: (
            LocalRetentionResult | None
        ) = None

        self.baseline_recalls: (
            dict[str, float] | None
        ) = None

        self.local_thresholds: (
            dict[str, float] | None
        ) = None


    # ========================================================
    # POSITION -> MASK
    # ========================================================

    def position_to_mask(
        self,
        position,
    ):
        return position_to_mask(
            position,
            min_features=
                MIN_FEATURES,
        )


    # ========================================================
    # RAW LOCAL EVALUATION
    # ========================================================

    def _evaluate_raw(
        self,
        mask,
    ):
        selected_positions = np.flatnonzero(
            mask
        )

        selected_count = int(
            len(
                selected_positions
            )
        )

        feature_reduction = float(
            1.0
            -
            selected_count
            /
            self.total_features
        )

        evaluation_start = (
            time.perf_counter()
        )

        per_client = evaluate_local_mask(
            clients=
                self.clients,
            X_validation=
                self.X_validation,
            y_validation=
                self.y_validation,
            selected_positions=
                selected_positions,
            label_mapping=
                self.label_mapping,
            class_weights=
                self.class_weights,
            device=
                self.device,
            seed=
                self.seed,
            local_epochs=
                self.local_epochs,
            local_mu=
                self.local_mu,
        )

        attack_recalls = {
            CLIENT_ATTACKS[
                client_id
            ]:
                float(
                    per_client[
                        client_id
                    ][
                        "seen_attack_recall"
                    ]
                )

            for client_id
            in range(
                1,
                N_CLIENTS + 1,
            )
        }

        recall_values = np.asarray(
            [
                attack_recalls[
                    CLIENT_ATTACKS[
                        client_id
                    ]
                ]
                for client_id
                in range(
                    1,
                    N_CLIENTS + 1,
                )
            ],
            dtype=np.float64,
        )

        return {
            "selected_positions":
                selected_positions,

            "selected_count":
                selected_count,

            "selected_features":
                [
                    self.feature_names[
                        position
                    ]
                    for position
                    in selected_positions
                ],

            "feature_reduction":
                feature_reduction,

            "per_client":
                per_client,

            "attack_recalls":
                attack_recalls,

            "mean_local_attack_recall":
                float(
                    np.mean(
                        recall_values
                    )
                ),

            "worst_local_attack_recall":
                float(
                    np.min(
                        recall_values
                    )
                ),

            "evaluation_seconds":
                float(
                    time.perf_counter()
                    -
                    evaluation_start
                ),
        }


    # ========================================================
    # BASELINE
    # ========================================================

    def initialize_baseline(
        self,
    ):
        logger.info(
            "Evaluating 70-feature LOCAL attack baselines..."
        )

        mask = np.ones(
            self.total_features,
            dtype=bool,
        )

        raw = self._evaluate_raw(
            mask
        )

        self.baseline_recalls = copy.deepcopy(
            raw[
                "attack_recalls"
            ]
        )

        failed_attacks = {
            attack:
                recall
            for attack, recall
            in self.baseline_recalls.items()
            if recall <= BASELINE_MIN_ATTACK_RECALL
        }

        if failed_attacks:
            print()
            print(
                "70-FEATURE LOCAL BASELINE VIABILITY: FAILED"
            )
            print("-" * 100)

            for client_id in range(
                1,
                N_CLIENTS + 1,
            ):
                attack = CLIENT_ATTACKS[
                    client_id
                ]

                print(
                    f"Client {client_id} | "
                    f"{attack:20s} | "
                    f"local recall="
                    f"{self.baseline_recalls[attack]:.6f}"
                )

            raise RuntimeError(
                "\nPHASE 13A LOCAL BASELINE INVALID\n\n"
                "At least one 70-feature LOCAL baseline has zero "
                "or near-zero seen-attack recall:\n"
                f"{failed_attacks}\n\n"
                "EVO-GA was NOT started.\n"
                "LOCKED TEST WAS NOT USED."
            )

        self.local_thresholds = {
            attack:
                float(
                    self.retention_ratio
                    *
                    baseline_recall
                )

            for attack, baseline_recall
            in self.baseline_recalls.items()
        }

        retention_ratios = {
            attack:
                1.0
            for attack
            in self.baseline_recalls
        }

        mean_retention = 1.0
        worst_retention = 1.0

        utility = self._calculate_utility(
            mean_local_attack_recall=
                raw[
                    "mean_local_attack_recall"
                ],
            worst_local_attack_recall=
                raw[
                    "worst_local_attack_recall"
                ],
            mean_local_retention=
                mean_retention,
            worst_local_retention=
                worst_retention,
            feature_reduction=
                raw[
                    "feature_reduction"
                ],
        )

        result = LocalRetentionResult(
            mask_key=
                mask_to_key(
                    mask
                ),

            selected_count=
                raw[
                    "selected_count"
                ],

            total_features=
                self.total_features,

            feature_reduction=
                raw[
                    "feature_reduction"
                ],

            selected_indices=
                raw[
                    "selected_positions"
                ]
                .tolist(),

            selected_features=
                raw[
                    "selected_features"
                ],

            local_attack_recalls=
                copy.deepcopy(
                    raw[
                        "attack_recalls"
                    ]
                ),

            local_baseline_recalls=
                copy.deepcopy(
                    self.baseline_recalls
                ),

            local_thresholds=
                copy.deepcopy(
                    self.local_thresholds
                ),

            local_retention_ratios=
                retention_ratios,

            mean_local_attack_recall=
                raw[
                    "mean_local_attack_recall"
                ],

            worst_local_attack_recall=
                raw[
                    "worst_local_attack_recall"
                ],

            mean_local_retention=
                mean_retention,

            worst_local_retention=
                worst_retention,

            feasible=
                True,

            total_constraint_violation=
                0.0,

            utility=
                utility,

            fitness=
                float(
                    1.0
                    -
                    utility
                ),

            evaluation_seconds=
                raw[
                    "evaluation_seconds"
                ],
        )

        self.baseline_result = (
            result
        )

        self.cache[
            result.mask_key
        ] = result

        self._append_history(
            result=
                result,
            source=
                "local_70_baseline",
        )

        self._save_local_baseline(
            raw[
                "per_client"
            ]
        )

        print()
        print(
            "70-FEATURE LOCAL BASELINE VIABILITY: PASSED"
        )
        print("-" * 100)

        for client_id in range(
            1,
            N_CLIENTS + 1,
        ):
            attack = CLIENT_ATTACKS[
                client_id
            ]

            print(
                f"Client {client_id} | "
                f"{attack:20s} | "
                f"baseline recall="
                f"{self.baseline_recalls[attack]:.6f} | "
                f"80% minimum="
                f"{self.local_thresholds[attack]:.6f}"
            )

        return result


    # ========================================================
    # SAVE BASELINE
    # ========================================================

    def _save_local_baseline(
        self,
        per_client,
    ):
        rows = []

        for client_id in range(
            1,
            N_CLIENTS + 1,
        ):
            attack = CLIENT_ATTACKS[
                client_id
            ]

            metrics = per_client[
                client_id
            ]

            rows.append(
                {
                    "client_id":
                        client_id,

                    "seen_attack":
                        attack,

                    "baseline_70_seen_attack_recall":
                        metrics[
                            "seen_attack_recall"
                        ],

                    "retention_threshold":
                        self.local_thresholds[
                            attack
                        ],

                    "accuracy_all_6_classes":
                        metrics[
                            "accuracy"
                        ],

                    "balanced_accuracy_all_6_classes":
                        metrics[
                            "balanced_accuracy"
                        ],

                    "macro_f1_all_6_classes":
                        metrics[
                            "macro_f1"
                        ],

                    "locked_test_used":
                        False,
                }
            )

        pd.DataFrame(
            rows
        ).to_csv(
            LOCAL_BASELINE_FILE,
            index=False,
        )


    # ========================================================
    # UTILITY
    # ========================================================

    def _calculate_utility(
        self,
        mean_local_attack_recall,
        worst_local_attack_recall,
        mean_local_retention,
        worst_local_retention,
        feature_reduction,
    ):
        # Retention > 1 means a candidate improved over baseline.
        # It is stored and reported in full, but capped at 1 inside
        # the utility so a large gain in one attack cannot compensate
        # for losing another attack.
        mean_retention_utility = min(
            float(
                mean_local_retention
            ),
            1.0,
        )

        worst_retention_utility = min(
            float(
                worst_local_retention
            ),
            1.0,
        )

        return float(
            MEAN_LOCAL_ATTACK_RECALL_WEIGHT
            *
            mean_local_attack_recall

            +

            WORST_LOCAL_ATTACK_RECALL_WEIGHT
            *
            worst_local_attack_recall

            +

            MEAN_LOCAL_RETENTION_WEIGHT
            *
            mean_retention_utility

            +

            WORST_LOCAL_RETENTION_WEIGHT
            *
            worst_retention_utility

            +

            FEATURE_REDUCTION_WEIGHT
            *
            feature_reduction
        )


    # ========================================================
    # CONSTRAINTS
    # ========================================================

    def _calculate_constraints(
        self,
        raw,
    ):
        if (
            self.baseline_recalls is None
            or
            self.local_thresholds is None
        ):
            raise RuntimeError(
                "\nCall initialize_baseline() "
                "before running EVO-GA."
            )

        retention_ratios = {}

        total_violation = 0.0

        feasible = True

        for client_id in range(
            1,
            N_CLIENTS + 1,
        ):
            attack = CLIENT_ATTACKS[
                client_id
            ]

            baseline = float(
                self.baseline_recalls[
                    attack
                ]
            )

            actual = float(
                raw[
                    "attack_recalls"
                ][
                    attack
                ]
            )

            threshold = float(
                self.local_thresholds[
                    attack
                ]
            )

            if baseline <= BASELINE_MIN_ATTACK_RECALL:
                raise RuntimeError(
                    "\nInvalid local baseline for "
                    f"{attack}: {baseline:.8f}"
                )

            ratio = float(
                actual
                /
                baseline
            )

            retention_ratios[
                attack
            ] = ratio

            shortfall = max(
                0.0,
                threshold
                -
                actual,
            )

            if shortfall > 0.0:
                feasible = False

                normalized_shortfall = (
                    shortfall
                    /
                    max(
                        threshold,
                        1e-12,
                    )
                )

                total_violation += (
                    normalized_shortfall
                )

        retention_values = np.asarray(
            list(
                retention_ratios.values()
            ),
            dtype=np.float64,
        )

        return (
            feasible,
            float(
                total_violation
            ),
            retention_ratios,
            float(
                np.mean(
                    retention_values
                )
            ),
            float(
                np.min(
                    retention_values
                )
            ),
        )


    # ========================================================
    # HISTORY
    # ========================================================

    def _append_history(
        self,
        result,
        source,
    ):
        row = {
            "evaluation":
                len(
                    self.candidate_history
                )
                +
                1,

            "source":
                source,

            "mask_key":
                result.mask_key,

            "selected_count":
                result.selected_count,

            "feature_reduction":
                result.feature_reduction,

            "mean_local_attack_recall":
                result.mean_local_attack_recall,

            "worst_local_attack_recall":
                result.worst_local_attack_recall,

            "mean_local_retention":
                result.mean_local_retention,

            "worst_local_retention":
                result.worst_local_retention,

            "feasible":
                result.feasible,

            "constraint_violation":
                result.total_constraint_violation,

            "utility":
                result.utility,

            "fitness":
                result.fitness,

            "evaluation_seconds":
                result.evaluation_seconds,
        }

        for client_id in range(
            1,
            N_CLIENTS + 1,
        ):
            attack = CLIENT_ATTACKS[
                client_id
            ]

            safe_attack = (
                attack
                .replace(
                    " ",
                    "_"
                )
                .replace(
                    "/",
                    "_"
                )
            )

            row[
                f"client_{client_id}_{safe_attack}_recall"
            ] = float(
                result.local_attack_recalls[
                    attack
                ]
            )

            row[
                f"client_{client_id}_{safe_attack}_baseline"
            ] = float(
                result.local_baseline_recalls[
                    attack
                ]
            )

            row[
                f"client_{client_id}_{safe_attack}_retention"
            ] = float(
                result.local_retention_ratios[
                    attack
                ]
            )

            row[
                f"client_{client_id}_{safe_attack}_minimum"
            ] = float(
                result.local_thresholds[
                    attack
                ]
            )

        self.candidate_history.append(
            row
        )


    # ========================================================
    # CANDIDATE EVALUATION
    # ========================================================

    def evaluate(
        self,
        position,
    ):
        if self.baseline_result is None:
            raise RuntimeError(
                "\n70-feature LOCAL baselines have "
                "not been initialized."
            )

        mask = self.position_to_mask(
            position
        )

        key = mask_to_key(
            mask
        )

        if key in self.cache:
            self.cache_hits += 1
            return self.cache[
                key
            ]

        self.unique_evaluations += 1

        raw = self._evaluate_raw(
            mask
        )

        (
            feasible,
            violation,
            retention_ratios,
            mean_retention,
            worst_retention,

        ) = self._calculate_constraints(
            raw
        )

        utility = self._calculate_utility(
            mean_local_attack_recall=
                raw[
                    "mean_local_attack_recall"
                ],

            worst_local_attack_recall=
                raw[
                    "worst_local_attack_recall"
                ],

            mean_local_retention=
                mean_retention,

            worst_local_retention=
                worst_retention,

            feature_reduction=
                raw[
                    "feature_reduction"
                ],
        )

        if feasible:
            fitness = float(
                1.0
                -
                utility
            )
        else:
            # Always worse than feasible candidates.
            fitness = float(
                1.0
                +
                violation
                +
                (
                    0.01
                    *
                    (
                        1.0
                        -
                        raw[
                            "feature_reduction"
                        ]
                    )
                )
            )

        result = LocalRetentionResult(
            mask_key=
                key,

            selected_count=
                raw[
                    "selected_count"
                ],

            total_features=
                self.total_features,

            feature_reduction=
                raw[
                    "feature_reduction"
                ],

            selected_indices=
                raw[
                    "selected_positions"
                ]
                .tolist(),

            selected_features=
                raw[
                    "selected_features"
                ],

            local_attack_recalls=
                copy.deepcopy(
                    raw[
                        "attack_recalls"
                    ]
                ),

            local_baseline_recalls=
                copy.deepcopy(
                    self.baseline_recalls
                ),

            local_thresholds=
                copy.deepcopy(
                    self.local_thresholds
                ),

            local_retention_ratios=
                retention_ratios,

            mean_local_attack_recall=
                raw[
                    "mean_local_attack_recall"
                ],

            worst_local_attack_recall=
                raw[
                    "worst_local_attack_recall"
                ],

            mean_local_retention=
                mean_retention,

            worst_local_retention=
                worst_retention,

            feasible=
                feasible,

            total_constraint_violation=
                violation,

            utility=
                utility,

            fitness=
                fitness,

            evaluation_seconds=
                raw[
                    "evaluation_seconds"
                ],
        )

        self.cache[
            key
        ] = result

        self._append_history(
            result=
                result,
            source=
                "optimizer",
        )

        logger.info(
            "Candidate %d | "
            "features=%d | "
            "reduction=%.2f%% | "
            "mean_local_recall=%.4f | "
            "worst_local_recall=%.4f | "
            "mean_retention=%.4f | "
            "worst_retention=%.4f | "
            "feasible=%s | "
            "fitness=%.6f",
            self.unique_evaluations,
            result.selected_count,
            result.feature_reduction
            *
            100.0,
            result.mean_local_attack_recall,
            result.worst_local_attack_recall,
            result.mean_local_retention,
            result.worst_local_retention,
            result.feasible,
            result.fitness,
        )

        return result


    # ========================================================
    # OBJECTIVE
    # ========================================================

    def fitness(
        self,
        position,
    ):
        return float(
            self.evaluate(
                position
            ).fitness
        )


    # ========================================================
    # BEST FEASIBLE REDUCED CANDIDATE
    # ========================================================

    def get_best_feasible_candidate(
        self,
    ):
        feasible_results = [
            result

            for result
            in self.cache.values()

            if (
                result.feasible

                and

                result.selected_count
                <
                self.total_features
            )
        ]

        if not feasible_results:
            logger.warning(
                "No reduced candidate preserved all five "
                "local attacks at the required retention. "
                "Returning the 70-feature local baseline."
            )

            return self.baseline_result

        feasible_results.sort(
            key=
                lambda result:
                    (
                        result.fitness,
                        result.selected_count,
                    )
        )

        return feasible_results[
            0
        ]


    # ========================================================
    # STATISTICS
    # ========================================================

    def get_statistics(
        self,
    ):
        return {
            "unique_candidate_evaluations":
                int(
                    self.unique_evaluations
                ),

            "cache_hits":
                int(
                    self.cache_hits
                ),

            "cached_masks":
                int(
                    len(
                        self.cache
                    )
                ),

            "history_rows":
                int(
                    len(
                        self.candidate_history
                    )
                ),
        }


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(
    title,
    result: LocalRetentionResult,
):
    print()
    print("-" * 100)
    print(title)
    print("-" * 100)

    print(
        f"Features                : "
        f"{result.selected_count}/"
        f"{result.total_features}"
    )

    print(
        f"Feature reduction       : "
        f"{result.feature_reduction * 100:.2f}%"
    )

    print(
        f"Mean local attack recall: "
        f"{result.mean_local_attack_recall:.6f}"
    )

    print(
        f"Worst local attack recall: "
        f"{result.worst_local_attack_recall:.6f}"
    )

    print(
        f"Mean local retention    : "
        f"{result.mean_local_retention * 100:.2f}%"
    )

    print(
        f"Worst local retention   : "
        f"{result.worst_local_retention * 100:.2f}%"
    )

    print(
        f"Feasible                : "
        f"{result.feasible}"
    )

    print(
        f"Utility                 : "
        f"{result.utility:.6f}"
    )

    print(
        f"Fitness                 : "
        f"{result.fitness:.6f}"
    )

    print()
    print(
        "Per-client local seen-attack retention:"
    )

    for client_id in range(
        1,
        N_CLIENTS + 1,
    ):
        attack = CLIENT_ATTACKS[
            client_id
        ]

        print(
            f"  Client {client_id} | "
            f"{attack:20s} | "
            f"baseline="
            f"{result.local_baseline_recalls[attack]:.6f} | "
            f"candidate="
            f"{result.local_attack_recalls[attack]:.6f} | "
            f"retention="
            f"{result.local_retention_ratios[attack] * 100:.2f}% | "
            f"minimum="
            f"{result.local_thresholds[attack]:.6f}"
        )


# ============================================================
# SAVE MASK / FEATURES
# ============================================================

def save_mask(
    active_features,
    selected_indices,
):
    selected_set = set(
        selected_indices
    )

    mask = np.asarray(
        [
            index
            in
            selected_set

            for index
            in range(
                len(
                    active_features
                )
            )
        ],
        dtype=bool,
    )

    pd.DataFrame(
        {
            "feature_index":
                np.arange(
                    len(
                        active_features
                    )
                ),

            "feature_name":
                active_features,

            "selected":
                mask.astype(
                    int
                ),
        }
    ).to_csv(
        FEATURE_MASK_FILE,
        index=False,
    )

    np.save(
        SELECTED_MASK_NPY,
        mask.astype(
            np.uint8
        ),
    )

    return mask


def save_selected_features(
    result,
):
    pd.DataFrame(
        {
            "selected_order":
                np.arange(
                    1,
                    result.selected_count
                    +
                    1,
                ),

            "active_feature_index":
                result.selected_indices,

            "feature_name":
                result.selected_features,
        }
    ).to_csv(
        SELECTED_FEATURES_FILE,
        index=False,
    )

    save_json(
        SELECTED_FEATURES_JSON,
        {
            "phase":
                "13A",

            "selection_strategy":
                "local_attack_retention_aware_hybrid_evo_ga",

            "selected_count":
                result.selected_count,

            "total_features":
                result.total_features,

            "feature_reduction":
                result.feature_reduction,

            "selected_indices":
                result.selected_indices,

            "selected_features":
                result.selected_features,

            "local_attack_recalls":
                result.local_attack_recalls,

            "local_baseline_recalls":
                result.local_baseline_recalls,

            "local_retention_ratios":
                result.local_retention_ratios,

            "retention_thresholds":
                result.local_thresholds,

            "fitness":
                result.fitness,

            "utility":
                result.utility,

            "feasible":
                result.feasible,

            "locked_test_used":
                False,
        },
    )


# ============================================================
# SAVE CONVERGENCE
# ============================================================

def save_convergence(
    history,
    path,
    column,
):
    values = np.asarray(
        history,
        dtype=np.float64,
    ).reshape(
        -1
    )

    pd.DataFrame(
        {
            "step":
                np.arange(
                    1,
                    len(values)
                    +
                    1,
                ),

            column:
                values,
        }
    ).to_csv(
        path,
        index=False,
    )


# ============================================================
# SAVE BASELINE-ONLY JSON
# ============================================================

def save_baseline_diagnostic(
    baseline_result,
    args,
):
    save_json(
        BASELINE_DIAGNOSTIC_FILE,
        {
            "phase":
                "13A",

            "mode":
                "local_70_feature_baseline_only",

            "scenario":
                args.scenario,

            "seed":
                args.seed,

            "features":
                EXPECTED_FEATURES,

            "local_search_epochs":
                args.local_search_epochs,

            "local_search_mu":
                args.local_search_mu,

            "retention_requirement":
                args.local_retention,

            "mean_local_attack_recall":
                baseline_result.mean_local_attack_recall,

            "worst_local_attack_recall":
                baseline_result.worst_local_attack_recall,

            "local_attack_recalls":
                baseline_result.local_attack_recalls,

            "local_thresholds":
                baseline_result.local_thresholds,

            "baseline_viability_passed":
                True,

            "locked_test_used":
                False,
        },
    )


# ============================================================
# SAVE SEARCH-SET FINAL LOCAL COMPARISON
# ============================================================

def save_search_local_comparison(
    baseline_result,
    selected_result,
):
    rows = []

    for client_id in range(
        1,
        N_CLIENTS + 1,
    ):
        attack = CLIENT_ATTACKS[
            client_id
        ]

        rows.append(
            {
                "client_id":
                    client_id,

                "seen_attack":
                    attack,

                "baseline_70_recall":
                    baseline_result.local_attack_recalls[
                        attack
                    ],

                "selected_subset_recall":
                    selected_result.local_attack_recalls[
                        attack
                    ],

                "absolute_delta":
                    (
                        selected_result.local_attack_recalls[
                            attack
                        ]
                        -
                        baseline_result.local_attack_recalls[
                            attack
                        ]
                    ),

                "retention_ratio":
                    selected_result.local_retention_ratios[
                        attack
                    ],

                "required_retention_ratio":
                    (
                        selected_result.local_thresholds[
                            attack
                        ]
                        /
                        baseline_result.local_attack_recalls[
                            attack
                        ]
                    ),

                "required_minimum_recall":
                    selected_result.local_thresholds[
                        attack
                    ],

                "constraint_passed":
                    (
                        selected_result.local_attack_recalls[
                            attack
                        ]
                        >=
                        selected_result.local_thresholds[
                            attack
                        ]
                    ),
            }
        )

    dataframe = pd.DataFrame(
        rows
    )

    dataframe.to_csv(
        LOCAL_SELECTED_COMPARISON_FILE,
        index=False,
    )

    return dataframe


# ============================================================
# FULL VALIDATION LOCAL AUDIT
# ============================================================

def run_full_validation_local_audit(
    clients,
    X_full_validation,
    y_full_validation,
    baseline_result,
    selected_result,
    label_mapping,
    class_weights,
    device,
    seed,
    local_epochs,
    local_mu,
):
    """
    Reporting-only audit after the mask is already frozen.

    IMPORTANT:
    These full-validation results MUST NOT be used to modify:
        - fitness weights,
        - retention threshold,
        - optimizer budget,
        - selected mask.

    No locked test is used.
    """

    full70_positions = np.arange(
        EXPECTED_FEATURES,
        dtype=np.int64,
    )

    selected_positions = np.asarray(
        selected_result.selected_indices,
        dtype=np.int64,
    )

    baseline_full = evaluate_local_mask(
        clients=
            clients,
        X_validation=
            X_full_validation,
        y_validation=
            y_full_validation,
        selected_positions=
            full70_positions,
        label_mapping=
            label_mapping,
        class_weights=
            class_weights,
        device=
            device,
        seed=
            seed,
        local_epochs=
            local_epochs,
        local_mu=
            local_mu,
    )

    selected_full = evaluate_local_mask(
        clients=
            clients,
        X_validation=
            X_full_validation,
        y_validation=
            y_full_validation,
        selected_positions=
            selected_positions,
        label_mapping=
            label_mapping,
        class_weights=
            class_weights,
        device=
            device,
        seed=
            seed,
        local_epochs=
            local_epochs,
        local_mu=
            local_mu,
    )

    rows = []

    for client_id in range(
        1,
        N_CLIENTS + 1,
    ):
        attack = CLIENT_ATTACKS[
            client_id
        ]

        baseline_recall = float(
            baseline_full[
                client_id
            ][
                "seen_attack_recall"
            ]
        )

        selected_recall = float(
            selected_full[
                client_id
            ][
                "seen_attack_recall"
            ]
        )

        retention = (
            selected_recall
            /
            baseline_recall

            if
            baseline_recall
            >
            BASELINE_MIN_ATTACK_RECALL

            else
            np.nan
        )

        rows.append(
            {
                "client_id":
                    client_id,

                "seen_attack":
                    attack,

                "baseline_70_full_validation_recall":
                    baseline_recall,

                "selected_full_validation_recall":
                    selected_recall,

                "absolute_delta":
                    selected_recall
                    -
                    baseline_recall,

                "retention_ratio":
                    retention,

                "search_required_retention_ratio":
                    selected_result.local_thresholds[
                        attack
                    ]
                    /
                    selected_result.local_baseline_recalls[
                        attack
                    ],

                "reporting_only":
                    True,

                "used_for_further_tuning":
                    False,

                "locked_test_used":
                    False,
            }
        )

    dataframe = pd.DataFrame(
        rows
    )

    dataframe.to_csv(
        FULL_VALIDATION_LOCAL_COMPARISON_FILE,
        index=False,
    )

    return dataframe


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--scenario",
        choices=[
            "non_iid",
            "controlled_non_iid",
            "iid",
        ],
        default=
            DEFAULT_SCENARIO,
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=
            DEFAULT_SEED,
    )


    parser.add_argument(
        "--local-retention",
        type=float,
        default=
            DEFAULT_LOCAL_RETENTION,
        help=(
            "Minimum candidate local seen-attack recall retention "
            "relative to each client's 70-feature local baseline."
        ),
    )


    # Backward-compatible alias.
    parser.add_argument(
        "--protected-retention",
        dest=
            "local_retention",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )


    parser.add_argument(
        "--validation-rows-per-class",
        type=int,
        default=
            DEFAULT_VALIDATION_ROWS_PER_CLASS,
    )


    parser.add_argument(
        "--local-search-epochs",
        type=int,
        default=
            DEFAULT_LOCAL_SEARCH_EPOCHS,
    )


    parser.add_argument(
        "--local-search-mu",
        type=float,
        default=
            DEFAULT_LOCAL_SEARCH_MU,
    )


    parser.add_argument(
        "--proxy-rounds",
        type=int,
        default=
            DEFAULT_PROXY_FEDERATED_ROUNDS,
        help=(
            "Used only by --diagnose-local-before-aggregation."
        ),
    )


    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help=(
            "Evaluate only the five 70-feature local baselines "
            "and stop before EVO-GA."
        ),
    )


    parser.add_argument(
        "--diagnose-local-before-aggregation",
        action="store_true",
        help=(
            "Run the optional local-vs-global aggregation diagnostic. "
            "EVO-GA and locked test are not used."
        ),
    )


    parser.add_argument(
        "--evo-population",
        type=int,
        default=
            DEFAULT_EVO_POPULATION,
    )


    parser.add_argument(
        "--evo-iterations",
        type=int,
        default=
            DEFAULT_EVO_ITERATIONS,
    )


    parser.add_argument(
        "--ga-population",
        type=int,
        default=
            DEFAULT_GA_POPULATION,
    )


    parser.add_argument(
        "--ga-generations",
        type=int,
        default=
            DEFAULT_GA_GENERATIONS,
    )


    args = parser.parse_args()


    if not (
        0.0
        <
        args.local_retention
        <=
        1.0
    ):
        raise ValueError(
            "\n--local-retention must be in (0, 1]."
        )


    if args.local_search_epochs < 1:
        raise ValueError(
            "--local-search-epochs must be >= 1."
        )


    if args.local_search_mu < 0:
        raise ValueError(
            "--local-search-mu must be >= 0."
        )


    if args.proxy_rounds < 1:
        raise ValueError(
            "--proxy-rounds must be >= 1."
        )


    if args.validation_rows_per_class < 1:
        raise ValueError(
            "--validation-rows-per-class must be >= 1."
        )


    total_start = (
        time.perf_counter()
    )


    set_seed(
        args.seed
    )


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    federated_dir = (
        phase4.FEDERATED_ROOT
        /
        args.scenario
    )


    # ========================================================
    # HEADER
    # ========================================================

    separator()
    print(
        "PHASE 13A — LOCAL-ATTACK-RETENTION-AWARE "
        "HYBRID EVO-GA FEATURE SELECTION"
    )
    separator()

    print(
        f"Scenario                    : "
        f"{args.scenario}"
    )

    print(
        f"Seed                        : "
        f"{args.seed}"
    )

    print(
        f"Device                      : "
        f"{device}"
    )

    print(
        f"Starting features           : "
        f"{EXPECTED_FEATURES}"
    )

    print(
        f"Primary search aggregation  : "
        f"NONE"
    )

    print(
        f"Local search epochs         : "
        f"{args.local_search_epochs}"
    )

    print(
        f"Local FedProx mu            : "
        f"{args.local_search_mu}"
    )

    print(
        f"All protected attacks       : "
        f"{ALL_ATTACKS}"
    )

    print(
        f"Required local retention    : "
        f"{args.local_retention * 100:.1f}%"
    )

    print(
        f"Proxy rows/client           : "
        f"{MAX_ROWS_PER_CLIENT:,}"
    )

    print(
        f"Proxy attack fraction       : "
        f"{ATTACK_FRACTION * 100:.1f}%"
    )

    print(
        f"Validation rows/class       : "
        f"{args.validation_rows_per_class:,}"
    )

    print(
        f"Baseline-only mode          : "
        f"{args.baseline_only}"
    )

    print(
        f"Aggregation diagnostic      : "
        f"{args.diagnose_local_before_aggregation}"
    )

    print()
    print(
        "Fitness:"
    )

    print(
        f"  Mean Local Attack Recall  : "
        f"{MEAN_LOCAL_ATTACK_RECALL_WEIGHT:.2f}"
    )

    print(
        f"  Worst Local Attack Recall : "
        f"{WORST_LOCAL_ATTACK_RECALL_WEIGHT:.2f}"
    )

    print(
        f"  Mean Local Retention      : "
        f"{MEAN_LOCAL_RETENTION_WEIGHT:.2f}"
    )

    print(
        f"  Worst Local Retention     : "
        f"{WORST_LOCAL_RETENTION_WEIGHT:.2f}"
    )

    print(
        f"  Feature Reduction         : "
        f"{FEATURE_REDUCTION_WEIGHT:.2f}"
    )


    # ========================================================
    # 1. DATA SAFETY
    # ========================================================

    assert_data_integrity(
        federated_dir
    )


    # ========================================================
    # 2. PREPROCESSING
    # ========================================================

    (
        scaler,
        feature_columns,
        label_mapping,
        active_indices,
        active_features,

    ) = phase4.load_preprocessing()


    active_indices = verify_feature_space(
        active_features=
            active_features,
        active_indices=
            active_indices,
        feature_columns=
            feature_columns,
    )


    if phase4.BENIGN_LABEL not in label_mapping:
        raise ValueError(
            "\nBENIGN is missing from label mapping."
        )


    for attack in ALL_ATTACKS:
        if attack not in label_mapping:
            raise ValueError(
                "\nAttack is missing from label mapping:\n"
                f"{attack}"
            )


    class_names = build_class_names(
        label_mapping
    )


    print()
    print(
        "Class mapping:"
    )

    for class_name in class_names:
        print(
            f"  {label_mapping[class_name]} "
            f"-> {class_name}"
        )


    # ========================================================
    # 3. CLASS WEIGHTS - GLOBAL TRAIN ONLY
    # ========================================================

    global_counts = phase4.calculate_global_class_counts(
        label_mapping=
            label_mapping,
        chunk_size=
            phase4.DEFAULT_CHUNK_SIZE,
    )

    class_weights = phase4.calculate_global_class_weights(
        global_counts
    )


    print()
    print(
        "Global TRAIN class weights:"
    )

    for class_name in class_names:
        class_id = int(
            label_mapping[
                class_name
            ]
        )

        print(
            f"  {class_name:25s} "
            f"{class_weights[class_id]:.6f}"
        )


    # ========================================================
    # 4. PROXY CLIENTS
    # ========================================================

    separator()
    print(
        "BUILDING ATTACK-PRESERVING LOCAL CLIENT PROXIES"
    )

    clients = load_proxy_clients(
        federated_dir=
            federated_dir,
        scaler=
            scaler,
        feature_columns=
            feature_columns,
        active_indices=
            active_indices,
        label_mapping=
            label_mapping,
        seed=
            args.seed,
    )


    # ========================================================
    # 5. SEARCH VALIDATION
    # ========================================================

    separator()
    print(
        "BUILDING SIX-CLASS VALIDATION SEARCH SET"
    )

    (
        X_validation,
        y_validation,
        validation_metadata,

    ) = load_validation_data(
        scaler=
            scaler,
        feature_columns=
            feature_columns,
        active_indices=
            active_indices,
        label_mapping=
            label_mapping,
        rows_per_class=
            args.validation_rows_per_class,
        seed=
            args.seed,
    )


    # ========================================================
    # OPTIONAL AGGREGATION DIAGNOSTIC
    # ========================================================

    if args.diagnose_local_before_aggregation:
        run_local_before_aggregation_diagnostic(
            clients=
                clients,
            X_validation=
                X_validation,
            y_validation=
                y_validation,
            label_mapping=
                label_mapping,
            class_weights=
                class_weights,
            device=
                device,
            seed=
                args.seed,
            proxy_rounds=
                args.proxy_rounds,
            local_epochs=
                args.local_search_epochs,
            local_mu=
                args.local_search_mu,
        )
        return


    # ========================================================
    # 6. LOCAL RETENTION SELECTOR
    # ========================================================

    selector = LocalAttackRetentionSelector(
        clients=
            clients,
        X_validation=
            X_validation,
        y_validation=
            y_validation,
        feature_names=
            active_features,
        label_mapping=
            label_mapping,
        class_weights=
            class_weights,
        retention_ratio=
            args.local_retention,
        seed=
            args.seed,
        device=
            device,
        local_epochs=
            args.local_search_epochs,
        local_mu=
            args.local_search_mu,
    )


    # ========================================================
    # 7. FIVE 70-FEATURE LOCAL BASELINES
    # ========================================================

    separator()
    print(
        "EVALUATING FIVE 70-FEATURE LOCAL ATTACK BASELINES"
    )

    baseline_result = (
        selector.initialize_baseline()
    )

    print_result(
        title=
            "70-FEATURE LOCAL ATTACK BASELINES",
        result=
            baseline_result,
    )


    # ========================================================
    # BASELINE-ONLY
    # ========================================================

    if args.baseline_only:
        save_baseline_diagnostic(
            baseline_result=
                baseline_result,
            args=
                args,
        )

        separator()
        print(
            "PHASE 13A LOCAL BASELINE-ONLY COMPLETE"
        )
        separator()

        print(
            f"Mean local attack recall : "
            f"{baseline_result.mean_local_attack_recall:.6f}"
        )

        print(
            f"Worst local attack recall: "
            f"{baseline_result.worst_local_attack_recall:.6f}"
        )

        print()
        print("EVO-GA search executed : NO")
        print("Locked test used       : NO")
        print()
        print("Baseline CSV:")
        print(LOCAL_BASELINE_FILE)
        print()
        print("Baseline JSON:")
        print(BASELINE_DIAGNOSTIC_FILE)

        return


    # ========================================================
    # 8. OPTIMIZER
    # ========================================================

    hybrid_optimizer = HybridEVOGAOptimizer(
        evo_population_size=
            args.evo_population,
        evo_iterations=
            args.evo_iterations,

        ga_population_size=
            args.ga_population,
        ga_generations=
            args.ga_generations,

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
            args.seed,

        verbose=True,
    )


    # ========================================================
    # 9. SEARCH
    # ========================================================

    separator()
    print(
        "STARTING LOCAL-RETENTION-AWARE HYBRID EVO-GA SEARCH"
    )

    print(
        f"Dimension : "
        f"{len(active_features)}"
    )

    print(
        f"Bounds    : "
        f"[{SEARCH_LOWER_BOUND}, "
        f"{SEARCH_UPPER_BOUND}]"
    )

    print(
        f"Hard rule : every attack must preserve >= "
        f"{args.local_retention * 100:.1f}% "
        f"of its 70-feature LOCAL baseline"
    )

    optimization_start = (
        time.perf_counter()
    )

    optimizer_result = hybrid_optimizer.optimize(
        objective_function=
            selector.fitness,
        dimension=
            len(
                active_features
            ),
        lower_bound=
            SEARCH_LOWER_BOUND,
        upper_bound=
            SEARCH_UPPER_BOUND,
    )

    optimization_seconds = float(
        time.perf_counter()
        -
        optimization_start
    )


    # ========================================================
    # 10. OPTIMIZER BEST
    # ========================================================

    optimizer_best_result = selector.evaluate(
        np.asarray(
            optimizer_result.best_position,
            dtype=np.float64,
        )
    )

    print_result(
        title=
            "OPTIMIZER BEST CANDIDATE",
        result=
            optimizer_best_result,
    )


    # ========================================================
    # 11. FINAL FREEZE RULE
    # ========================================================

    final_result = selector.get_best_feasible_candidate()

    separator()
    print(
        "FINAL PHASE 13A FROZEN CANDIDATE"
    )

    print_result(
        title=
            "LOCAL-RETENTION-AWARE FROZEN FEATURE SUBSET",
        result=
            final_result,
    )


    # ========================================================
    # 12. SAVE SEARCH ARTIFACTS
    # ========================================================

    pd.DataFrame(
        selector.candidate_history
    ).to_csv(
        CANDIDATE_HISTORY_FILE,
        index=False,
    )

    save_convergence(
        optimizer_result.convergence_history,
        CONVERGENCE_FILE,
        "best_fitness",
    )

    save_convergence(
        optimizer_result.evo_convergence_history,
        EVO_CONVERGENCE_FILE,
        "evo_best_fitness",
    )

    save_convergence(
        optimizer_result.ga_convergence_history,
        GA_CONVERGENCE_FILE,
        "ga_best_fitness",
    )

    save_mask(
        active_features=
            active_features,
        selected_indices=
            final_result.selected_indices,
    )

    save_selected_features(
        final_result
    )

    search_comparison_df = save_search_local_comparison(
        baseline_result=
            baseline_result,
        selected_result=
            final_result,
    )


    # ========================================================
    # 13. FULL VALIDATION LOCAL AUDIT - REPORTING ONLY
    # ========================================================

    separator()
    print(
        "FINAL LOCAL AUDIT ON FULL SIX-CLASS VALIDATION"
    )
    separator()

    print(
        "Feature mask is already frozen."
    )

    print(
        "These numbers are reporting-only and MUST NOT "
        "trigger additional Phase 13A tuning."
    )

    (
        X_full_validation,
        y_full_validation,

    ) = load_full_validation_data(
        scaler=
            scaler,
        feature_columns=
            feature_columns,
        active_indices=
            active_indices,
        label_mapping=
            label_mapping,
    )

    full_validation_df = run_full_validation_local_audit(
        clients=
            clients,
        X_full_validation=
            X_full_validation,
        y_full_validation=
            y_full_validation,
        baseline_result=
            baseline_result,
        selected_result=
            final_result,
        label_mapping=
            label_mapping,
        class_weights=
            class_weights,
        device=
            device,
        seed=
            args.seed,
        local_epochs=
            args.local_search_epochs,
        local_mu=
            args.local_search_mu,
    )

    print()
    print(
        full_validation_df.to_string(
            index=False
        )
    )


    # ========================================================
    # 14. SUMMARY
    # ========================================================

    total_seconds = float(
        time.perf_counter()
        -
        total_start
    )

    summary = {
        "phase":
            "13A",

        "experiment":
            (
                "Local-Attack-Retention-Aware "
                "Hybrid EVO-GA Feature Selection"
            ),

        "scenario":
            args.scenario,

        "seed":
            args.seed,

        "device":
            str(
                device
            ),

        "research_design": {
            "primary_search_unit":
                "local_client_seen_attack_recall",

            "aggregation_used_for_primary_search":
                False,

            "all_five_attacks_protected_symmetrically":
                True,

            "client_attack_mapping":
                CLIENT_ATTACKS,

            "hard_retention_ratio":
                args.local_retention,

            "retention_definition":
                (
                    "candidate local seen-attack recall / "
                    "70-feature local seen-attack recall"
                ),
        },

        "data_integrity": {
            "federated_training_source":
                str(
                    federated_dir
                ),

            "validation_source":
                str(
                    VALIDATION_FILE
                ),

            "locked_test_path_documented_only":
                str(
                    LOCKED_TEST_FILE
                ),

            "locked_test_used":
                False,

            "locked_test_allowed_in_phase13a":
                False,
        },

        "validation_metadata":
            validation_metadata,

        "proxy_data": {
            "max_rows_per_client":
                MAX_ROWS_PER_CLIENT,

            "attack_fraction":
                ATTACK_FRACTION,

            "oversampling":
                False,

            "synthetic_data":
                False,
        },

        "local_training": {
            "local_epochs":
                args.local_search_epochs,

            "fedprox_mu":
                args.local_search_mu,

            "batch_size":
                PROXY_BATCH_SIZE,

            "learning_rate":
                PROXY_LEARNING_RATE,

            "weight_decay":
                PROXY_WEIGHT_DECAY,

            "gradient_clip":
                PROXY_GRADIENT_CLIP,

            "aggregation_during_search":
                False,
        },

        "fitness_weights": {
            "mean_local_attack_recall":
                MEAN_LOCAL_ATTACK_RECALL_WEIGHT,

            "worst_local_attack_recall":
                WORST_LOCAL_ATTACK_RECALL_WEIGHT,

            "mean_local_retention":
                MEAN_LOCAL_RETENTION_WEIGHT,

            "worst_local_retention":
                WORST_LOCAL_RETENTION_WEIGHT,

            "feature_reduction":
                FEATURE_REDUCTION_WEIGHT,
        },

        "feature_space": {
            "starting_features":
                EXPECTED_FEATURES,

            "selected_features":
                final_result.selected_count,

            "feature_reduction_ratio":
                final_result.feature_reduction,

            "selected_feature_names":
                final_result.selected_features,

            "selected_indices":
                final_result.selected_indices,
        },

        "local_70_baselines":
            baseline_result.local_attack_recalls,

        "local_retention_thresholds":
            baseline_result.local_thresholds,

        "frozen_candidate": {
            "local_attack_recalls":
                final_result.local_attack_recalls,

            "local_retention_ratios":
                final_result.local_retention_ratios,

            "mean_local_attack_recall":
                final_result.mean_local_attack_recall,

            "worst_local_attack_recall":
                final_result.worst_local_attack_recall,

            "mean_local_retention":
                final_result.mean_local_retention,

            "worst_local_retention":
                final_result.worst_local_retention,

            "feasible":
                final_result.feasible,

            "utility":
                final_result.utility,

            "fitness":
                final_result.fitness,
        },

        "hybrid_search": {
            "lower_bound":
                SEARCH_LOWER_BOUND,

            "upper_bound":
                SEARCH_UPPER_BOUND,

            "evo_population":
                args.evo_population,

            "evo_iterations":
                args.evo_iterations,

            "ga_population":
                args.ga_population,

            "ga_generations":
                args.ga_generations,

            "optimizer_best_fitness":
                float(
                    optimizer_result.best_fitness
                ),

            "frozen_candidate_fitness":
                final_result.fitness,

            "optimization_seconds":
                optimization_seconds,
        },

        "selector_statistics":
            selector.get_statistics(),

        "search_comparison":
            json_safe(
                search_comparison_df
                .to_dict(
                    orient="records"
                )
            ),

        "full_validation_local_audit": {
            "results":
                json_safe(
                    full_validation_df
                    .to_dict(
                        orient="records"
                    )
                ),

            "used_for_further_tuning":
                False,

            "locked_test_used":
                False,
        },

        "runtime": {
            "optimization_seconds":
                optimization_seconds,

            "total_script_seconds":
                total_seconds,
        },

        "outputs": {
            "candidate_history":
                str(
                    CANDIDATE_HISTORY_FILE
                ),

            "local_70_baseline":
                str(
                    LOCAL_BASELINE_FILE
                ),

            "local_selected_comparison":
                str(
                    LOCAL_SELECTED_COMPARISON_FILE
                ),

            "full_validation_local_comparison":
                str(
                    FULL_VALIDATION_LOCAL_COMPARISON_FILE
                ),

            "selected_features":
                str(
                    SELECTED_FEATURES_FILE
                ),

            "feature_mask":
                str(
                    FEATURE_MASK_FILE
                ),

            "selected_features_json":
                str(
                    SELECTED_FEATURES_JSON
                ),

            "selected_mask_npy":
                str(
                    SELECTED_MASK_NPY
                ),

            "summary":
                str(
                    SUMMARY_FILE
                ),
        },
    }

    save_json(
        SUMMARY_FILE,
        summary,
    )


    # ========================================================
    # 15. FINAL DISPLAY
    # ========================================================

    separator()
    print(
        "PHASE 13A COMPLETE"
    )
    separator()

    print(
        "Selection basis         : "
        "LOCAL attack retention"
    )

    print(
        "Aggregation in search   : NO"
    )

    print(
        f"Starting features       : "
        f"{EXPECTED_FEATURES}"
    )

    print(
        f"Selected features       : "
        f"{final_result.selected_count}"
    )

    print(
        f"Feature reduction       : "
        f"{final_result.feature_reduction * 100:.2f}%"
    )

    print(
        f"Candidate feasible      : "
        f"{final_result.feasible}"
    )

    print(
        f"Required retention      : "
        f"{args.local_retention * 100:.2f}%"
    )

    print(
        f"Mean local recall       : "
        f"{baseline_result.mean_local_attack_recall:.6f}"
        f" -> "
        f"{final_result.mean_local_attack_recall:.6f}"
    )

    print(
        f"Worst local recall      : "
        f"{baseline_result.worst_local_attack_recall:.6f}"
        f" -> "
        f"{final_result.worst_local_attack_recall:.6f}"
    )

    print(
        f"Mean local retention    : "
        f"{final_result.mean_local_retention * 100:.2f}%"
    )

    print(
        f"Worst local retention   : "
        f"{final_result.worst_local_retention * 100:.2f}%"
    )

    print()
    print(
        "Per-client attack preservation:"
    )

    for client_id in range(
        1,
        N_CLIENTS + 1,
    ):
        attack = CLIENT_ATTACKS[
            client_id
        ]

        print(
            f"  Client {client_id} | "
            f"{attack:20s} | "
            f"baseline="
            f"{baseline_result.local_attack_recalls[attack]:.6f} | "
            f"selected="
            f"{final_result.local_attack_recalls[attack]:.6f} | "
            f"retention="
            f"{final_result.local_retention_ratios[attack] * 100:.2f}%"
        )

    print()
    print(
        f"Unique evaluations      : "
        f"{selector.unique_evaluations}"
    )

    print(
        f"Cache hits              : "
        f"{selector.cache_hits}"
    )

    print(
        f"Optimization time       : "
        f"{optimization_seconds:.2f}s"
    )

    print(
        f"Total time              : "
        f"{total_seconds:.2f}s"
    )

    print()
    print("Frozen mask:")
    print(FEATURE_MASK_FILE)

    print()
    print("Selected features:")
    print(SELECTED_FEATURES_FILE)

    print()
    print("Local comparison:")
    print(LOCAL_SELECTED_COMPARISON_FILE)

    print()
    print("Summary:")
    print(SUMMARY_FILE)

    print()
    print("LOCKED TEST USED: NO")

    print()
    print("NEXT:")
    print(
        "Use exactly this ONE frozen mask in Phase 13B "
        "across the repeated FedProx seeds."
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()
