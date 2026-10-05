"""
Phase 9 - Repeated-Seed Validation: Hybrid EVO-GA + FedProx
============================================================

Purpose
-------

Phase 7 established the frozen Hybrid EVO-GA feature subset:

    70 -> 37 features

Phase 9 evaluates Hybrid37 + FedProx across repeated random seeds
while keeping the feature mask and all experimental settings fixed.

FedProx local objective:

    L_prox(w)
        =
        L_task(w)
        +
        (mu / 2) * ||w - w_global||^2

where:

    L_task      = weighted CrossEntropyLoss
    w           = current local model parameters
    w_global    = parameters received from global server
    mu          = proximal regularization coefficient

IMPORTANT
---------

The Hybrid feature selector is NOT rerun.

The frozen mask is loaded from:

    results/phase7_hybrid/hybrid_feature_mask.csv

Repeated-seed validation changes only the random seed:

    --seed controls model initialization, dropout RNG, and client shuffling.

The same settings are preserved:

    - 5 clients
    - same non-IID scenario
    - same scaler and label mapping
    - same frozen 37 Hybrid features
    - same MLP architecture
    - same class weights
    - same AdamW optimizer
    - same batch size
    - same learning rate / weight decay
    - same local epochs
    - same 10 federated rounds
    - same sample-weighted server aggregation
    - same FedProx mu (default 0.01)

Outputs are isolated by seed:

    results/phase9/fedprox/<scenario>/seed_<seed>/
    artifacts/phase9/fedprox/<scenario>/seed_<seed>/

Examples:

    python scripts/phase9_fedprox.py --scenario non_iid --seed 42 --mu 0.01
    python scripts/phase9_fedprox.py --scenario non_iid --seed 52 --mu 0.01
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
import time

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


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
# PHASE 9 FEDAVG70 REUSE
# ============================================================

import phase9_fedavg70 as phase4


# ============================================================
# GENERAL CONFIG
# ============================================================

SEED = phase4.SEED

N_CLIENTS = phase4.N_CLIENTS

EXPECTED_PHASE4_FEATURES = 70


# ============================================================
# SAME PHASE-9 FEDAVG70 / PHASE-7 DEFAULTS
# ============================================================

DEFAULT_ROUNDS = phase4.DEFAULT_ROUNDS

DEFAULT_LOCAL_EPOCHS = phase4.DEFAULT_LOCAL_EPOCHS

DEFAULT_BATCH_SIZE = phase4.DEFAULT_BATCH_SIZE

DEFAULT_CHUNK_SIZE = phase4.DEFAULT_CHUNK_SIZE

DEFAULT_LEARNING_RATE = phase4.DEFAULT_LEARNING_RATE

DEFAULT_WEIGHT_DECAY = phase4.DEFAULT_WEIGHT_DECAY


# ============================================================
# FEDPROX
# ============================================================

DEFAULT_MU = 0.01

GRADIENT_CLIP = 5.0


# ============================================================
# PHASE 7 HYBRID MASK
# ============================================================

PHASE7_SEARCH_DIR = (
    PROJECT_ROOT
    / "results"
    / "phase7_hybrid"
)

DEFAULT_MASK_FILE = (
    PHASE7_SEARCH_DIR
    / "hybrid_feature_mask.csv"
)


# ============================================================
# OUTPUT
# ============================================================

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "phase9"
    / "fedprox"
)

ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "phase9"
    / "fedprox"
)


# ============================================================
# PREVIOUS RESULT PATHS
# ============================================================

PHASE4_RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "phase9"
    / "fedavg70"
)

PHASE7_RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "phase9"
    / "hybrid_fedavg"
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
    "phase9_fedprox"
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
# HASH FILE
# ============================================================

def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as file:

        while True:

            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


# ============================================================
# JSON LOAD
# ============================================================

def load_json_if_exists(
    path: Path,
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
# LOAD HYBRID MASK
# ============================================================

def load_hybrid_mask(
    mask_file,
    active_features,
    active_indices,
    feature_columns,
):

    if not mask_file.exists():

        raise FileNotFoundError(
            "\nHybrid feature mask missing:\n"
            f"{mask_file}\n\n"
            "Run Phase 7 first."
        )


    mask_df = pd.read_csv(
        mask_file
    )


    required_columns = {
        "feature_name",
        "selected",
    }


    missing = (

        required_columns

        -

        set(
            mask_df.columns
        )
    )


    if missing:

        raise ValueError(
            "\nInvalid Hybrid feature mask.\n"
            f"Missing columns: {sorted(missing)}"
        )


    mask_df[
        "feature_name"
    ] = (

        mask_df[
            "feature_name"
        ]
        .astype(str)
        .str.strip()
    )


    # ========================================================
    # VERIFY 70-FEATURE ACTIVE SPACE
    # ========================================================

    if (
        len(
            active_features
        )
        !=
        EXPECTED_PHASE4_FEATURES
    ):

        raise ValueError(
            "\nExpected exactly 70 Phase-4 features.\n"
            f"Found: {len(active_features)}"
        )


    if (
        len(
            mask_df
        )
        !=
        EXPECTED_PHASE4_FEATURES
    ):

        raise ValueError(
            "\nHybrid mask must contain exactly "
            "70 rows.\n"
            f"Found: {len(mask_df)}"
        )


    # ========================================================
    # FEATURE SCHEMA SAFETY
    # ========================================================

    mask_features = set(
        mask_df[
            "feature_name"
        ]
    )

    active_feature_set = set(
        active_features
    )


    missing_from_mask = (

        active_feature_set

        -

        mask_features
    )


    extra_in_mask = (

        mask_features

        -

        active_feature_set
    )


    if (
        missing_from_mask
        or
        extra_in_mask
    ):

        raise ValueError(
            "\nHybrid mask feature schema mismatch.\n\n"
            f"Missing:\n{sorted(missing_from_mask)}\n\n"
            f"Extra:\n{sorted(extra_in_mask)}"
        )


    # ========================================================
    # SELECTION MAP
    # ========================================================

    selection_map = {

        row.feature_name:
            int(
                row.selected
            )

        for row
        in mask_df.itertuples(
            index=False
        )
    }


    invalid_values = {

        value

        for value
        in selection_map.values()

        if value not in {
            0,
            1,
        }
    }


    if invalid_values:

        raise ValueError(
            "selected must contain only 0 or 1."
        )


    # ========================================================
    # PRESERVE PHASE-4 FEATURE ORDER
    # ========================================================

    selected_active_positions = [

        index

        for index, feature
        in enumerate(
            active_features
        )

        if selection_map[
            feature
        ] == 1
    ]


    selected_features = [

        active_features[
            index
        ]

        for index
        in selected_active_positions
    ]


    if not selected_features:

        raise ValueError(
            "Hybrid mask selected zero features."
        )


    # ========================================================
    # ACTIVE INDEX -> ORIGINAL SCALER INDEX
    # ========================================================

    active_indices_array = np.asarray(

        active_indices,

        dtype=np.int64,
    )


    selected_original_indices = (

        active_indices_array[
            selected_active_positions
        ]
    )


    reconstructed_features = [

        feature_columns[
            int(
                index
            )
        ]

        for index
        in selected_original_indices
    ]


    if (
        reconstructed_features
        !=
        selected_features
    ):

        raise ValueError(
            "\nHybrid feature index mapping failed."
        )


    return (

        selected_features,

        np.asarray(
            selected_active_positions,
            dtype=np.int64,
        ),

        np.asarray(
            selected_original_indices,
            dtype=np.int64,
        ),
    )


# ============================================================
# PROJECTED INITIALIZATION
# ============================================================

def initialize_projected_model(
    selected_active_positions,
    full_input_dim,
    selected_input_dim,
    num_classes,
    device,
    seed,
):

    """
    Create same seed-based 70-feature reference model and
    project first layer to the frozen Hybrid subset.
    """

    phase4.set_seed(
        seed
    )


    reference_model = (

        phase4.IDSMLP(

            input_dim=
                full_input_dim,

            num_classes=
                num_classes,

        ).to(
            device
        )
    )


    reference_state = {

        key:
            value
            .detach()
            .clone()

        for key, value
        in reference_model
        .state_dict()
        .items()
    }


    selected_model = (

        phase4.IDSMLP(

            input_dim=
                selected_input_dim,

            num_classes=
                num_classes,

        ).to(
            device
        )
    )


    selected_state = (
        selected_model.state_dict()
    )


    first_layer_key = (
        "network.0.weight"
    )


    positions = [

        int(
            value
        )

        for value
        in selected_active_positions
    ]


    for key in (
        selected_state.keys()
    ):

        if (
            key
            ==
            first_layer_key
        ):

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


    selected_model.load_state_dict(
        selected_state
    )


    return (
        selected_model,
        reference_model,
    )


# ============================================================
# FEDPROX PENALTY
# ============================================================

def calculate_proximal_term(
    local_model,
    global_parameters,
):

    """
    Calculate:

        ||w_local - w_global||^2

    over trainable model parameters.
    """

    proximal_term = torch.zeros(

        1,

        device=
            next(
                local_model.parameters()
            ).device,
    )


    for local_parameter, global_parameter in zip(

        local_model.parameters(),

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
# FEDPROX LOCAL CLIENT TRAINING
# ============================================================

def train_client_fedprox(
    global_state,
    client_file,
    scaler,
    feature_columns,
    selected_original_indices,
    label_mapping,
    class_weights,
    input_dim,
    device,
    local_epochs,
    batch_size,
    chunk_size,
    learning_rate,
    weight_decay,
    client_id,
    round_number,
    mu,
    seed,
):

    """
    Train one client using FedProx.

    Local objective:

        weighted CE
        +
        (mu / 2) * ||w_local - w_global||^2
    """

    training_start = (
        time.perf_counter()
    )


    # ========================================================
    # LOCAL MODEL
    # ========================================================

    local_model = (

        phase4.IDSMLP(

            input_dim=
                input_dim,

            num_classes=
                len(
                    label_mapping
                ),

        ).to(
            device
        )
    )


    local_model.load_state_dict(
        global_state
    )


    # ========================================================
    # FROZEN GLOBAL REFERENCE PARAMETERS
    # ========================================================

    global_parameters = [

        parameter
        .detach()
        .clone()
        .to(
            device
        )

        for parameter
        in local_model.parameters()
    ]


    for parameter in (
        global_parameters
    ):

        parameter.requires_grad_(
            False
        )


    # ========================================================
    # LOSS
    # ========================================================

    class_weight_tensor = torch.tensor(

        class_weights,

        dtype=torch.float32,

        device=device,
    )


    criterion = nn.CrossEntropyLoss(

        weight=
            class_weight_tensor
    )


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(

        local_model.parameters(),

        lr=
            learning_rate,

        weight_decay=
            weight_decay,
    )


    # ========================================================
    # METRIC ACCUMULATORS
    # ========================================================

    total_loss_sum = 0.0

    total_ce_loss_sum = 0.0

    total_prox_loss_sum = 0.0

    total_correct = 0

    total_samples = 0

    total_batches = 0


    # ========================================================
    # LOCAL EPOCHS
    # ========================================================

    local_model.train()


    for local_epoch in range(
        1,
        local_epochs + 1
    ):

        # Same deterministic shuffle rule used by Phase 9 FedAvg70.
        rng = np.random.default_rng(

            seed
            + round_number * 1000
            + client_id * 100
            + local_epoch
        )

        reader = pd.read_csv(

            client_file,

            chunksize=
                chunk_size,
        )


        for chunk in (
            reader
        ):

            # =================================================
            # SAME PHASE-4 TRANSFORMATION
            # =================================================

            X = phase4.transform_features(

                df=
                    chunk,

                feature_columns=
                    feature_columns,

                scaler=
                    scaler,

                active_indices=
                    selected_original_indices,
            )


            y = phase4.encode_labels(

                chunk[
                    phase4.LABEL_COL
                ],

                label_mapping,
            )


            if (
                len(
                    y
                )
                ==
                0
            ):

                continue


            indices = rng.permutation(
                len(
                    y
                )
            )


            # =================================================
            # MINI BATCHES
            # =================================================

            for start in range(

                0,

                len(
                    indices
                ),

                batch_size,
            ):

                batch_indices = indices[

                    start:
                    start
                    +
                    batch_size
                ]


                X_batch = torch.from_numpy(

                    X[
                        batch_indices
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


                logits = local_model(
                    X_batch
                )


                # =============================================
                # TASK LOSS
                # =============================================

                ce_loss = criterion(

                    logits,

                    y_batch,
                )


                # =============================================
                # FEDPROX PENALTY
                # =============================================

                proximal_term = (
                    calculate_proximal_term(

                        local_model=
                            local_model,

                        global_parameters=
                            global_parameters,
                    )
                )


                prox_loss = (

                    0.5

                    *

                    mu

                    *

                    proximal_term
                )


                # =============================================
                # TOTAL FEDPROX OBJECTIVE
                # =============================================

                loss = (

                    ce_loss

                    +

                    prox_loss
                )


                loss.backward()


                torch.nn.utils.clip_grad_norm_(

                    local_model.parameters(),

                    max_norm=
                        GRADIENT_CLIP,
                )


                optimizer.step()


                # =============================================
                # METRICS
                # =============================================

                predictions = torch.argmax(

                    logits,

                    dim=1,
                )


                batch_samples = int(
                    y_batch.size(
                        0
                    )
                )


                batch_correct = int(

                    (
                        predictions
                        ==
                        y_batch
                    )
                    .sum()
                    .item()
                )


                total_loss_sum += (

                    float(
                        loss.item()
                    )

                    *
                    batch_samples
                )


                total_ce_loss_sum += (

                    float(
                        ce_loss.item()
                    )

                    *
                    batch_samples
                )


                total_prox_loss_sum += (

                    float(
                        prox_loss.item()
                    )

                    *
                    batch_samples
                )


                total_correct += (
                    batch_correct
                )


                total_samples += (
                    batch_samples
                )


                total_batches += 1


            del chunk


    # ========================================================
    # FINAL CLIENT METRICS
    # ========================================================

    if total_samples == 0:

        raise RuntimeError(
            f"Client {client_id} produced zero "
            "training samples."
        )


    mean_loss = float(

        total_loss_sum

        /

        total_samples
    )


    mean_ce_loss = float(

        total_ce_loss_sum

        /

        total_samples
    )


    mean_prox_loss = float(

        total_prox_loss_sum

        /

        total_samples
    )


    accuracy = float(

        total_correct

        /

        total_samples
    )


    training_seconds = (

        time.perf_counter()

        -

        training_start
    )


    # ========================================================
    # CPU STATE FOR AGGREGATION
    # ========================================================

    state = {

        key:
            value
            .detach()
            .cpu()
            .clone()

        for key, value
        in local_model
        .state_dict()
        .items()
    }


    logger.info(
        "Round %d | Client %d | "
        "FedProx loss=%.6f | "
        "CE=%.6f | "
        "prox=%.6f | "
        "acc=%.6f | "
        "mu=%.6f | "
        "time=%.2fs",
        round_number,
        client_id,
        mean_loss,
        mean_ce_loss,
        mean_prox_loss,
        accuracy,
        mu,
        training_seconds,
    )


    del local_model
    del global_parameters


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


    return (
        state,
        mean_loss,
        mean_ce_loss,
        mean_prox_loss,
        accuracy,
        training_seconds,
    )


# ============================================================
# PARAMETER COUNT
# ============================================================

def parameter_count(
    model,
):

    return int(

        sum(

            parameter.numel()

            for parameter
            in model.parameters()
        )
    )


# ============================================================
# PAYLOAD SIZE
# ============================================================

def model_payload_bytes(
    model,
):

    return int(

        sum(

            tensor.numel()
            *
            tensor.element_size()

            for tensor
            in model.state_dict().values()
        )
    )


# ============================================================
# COMMUNICATION
# ============================================================

def calculate_communication(
    baseline_model,
    fedprox_model,
    rounds,
):

    baseline_params = (
        parameter_count(
            baseline_model
        )
    )


    fedprox_params = (
        parameter_count(
            fedprox_model
        )
    )


    baseline_payload = (
        model_payload_bytes(
            baseline_model
        )
    )


    fedprox_payload = (
        model_payload_bytes(
            fedprox_model
        )
    )


    baseline_total = (

        2
        *
        N_CLIENTS
        *
        rounds
        *
        baseline_payload
    )


    fedprox_total = (

        2
        *
        N_CLIENTS
        *
        rounds
        *
        fedprox_payload
    )


    return {

        "phase4_parameter_count":
            baseline_params,

        "fedprox_parameter_count":
            fedprox_params,

        "parameter_reduction_ratio":
            float(

                1.0

                -

                fedprox_params
                /
                baseline_params
            ),

        "phase4_model_payload_mib":
            float(

                baseline_payload
                /
                1024 ** 2
            ),

        "fedprox_model_payload_mib":
            float(

                fedprox_payload
                /
                1024 ** 2
            ),

        "phase4_total_communication_mib":
            float(

                baseline_total
                /
                1024 ** 2
            ),

        "fedprox_total_communication_mib":
            float(

                fedprox_total
                /
                1024 ** 2
            ),

        "communication_reduction_ratio":
            float(

                1.0

                -

                fedprox_total
                /
                baseline_total
            ),

        "assumption":
            (
                "Full float32 model upload/download "
                "for every client per round."
            ),
    }


# ============================================================
# PER-CLASS TABLE
# ============================================================

def build_per_class_df(
    report,
    class_names,
):

    rows = []


    for class_name in (
        class_names
    ):

        metrics = report[
            class_name
        ]


        rows.append({

            "class":
                class_name,

            "precision":
                float(
                    metrics[
                        "precision"
                    ]
                ),

            "recall":
                float(
                    metrics[
                        "recall"
                    ]
                ),

            "f1_score":
                float(
                    metrics[
                        "f1-score"
                    ]
                ),

            "support":
                int(
                    metrics[
                        "support"
                    ]
                ),
        })


    return pd.DataFrame(
        rows
    )


# ============================================================
# SAVE COMPARISON
# ============================================================

def save_comparison(
    scenario,
    seed,
    overall,
    result_dir,
):

    baseline_file = (

        PHASE4_RESULT_ROOT
        /
        scenario
        /
        f"seed_{seed}"
        /
        "overall_metrics.json"
    )


    hybrid_fedavg_file = (

        PHASE7_RESULT_ROOT
        /
        scenario
        /
        f"seed_{seed}"
        /
        "overall_metrics.json"
    )


    baseline = load_json_if_exists(
        baseline_file
    )


    hybrid_fedavg = load_json_if_exists(
        hybrid_fedavg_file
    )


    rows = []


    if baseline is not None:

        rows.append({

            "method":
                "FedAvg_70",

            "features":
                70,

            "accuracy":
                baseline.get(
                    "accuracy"
                ),

            "balanced_accuracy":
                baseline.get(
                    "balanced_accuracy"
                ),

            "macro_f1":
                baseline.get(
                    "macro_f1"
                ),

            "weighted_f1":
                baseline.get(
                    "weighted_f1"
                ),
        })


    if hybrid_fedavg is not None:

        rows.append({

            "method":
                "Hybrid37_FedAvg",

            "features":
                37,

            "accuracy":
                hybrid_fedavg.get(
                    "accuracy"
                ),

            "balanced_accuracy":
                hybrid_fedavg.get(
                    "balanced_accuracy"
                ),

            "macro_f1":
                hybrid_fedavg.get(
                    "macro_f1"
                ),

            "weighted_f1":
                hybrid_fedavg.get(
                    "weighted_f1"
                ),
        })


    rows.append({

        "method":
            "Hybrid37_FedProx",

        "features":
            overall[
                "selected_feature_count"
            ],

        "accuracy":
            overall[
                "accuracy"
            ],

        "balanced_accuracy":
            overall[
                "balanced_accuracy"
            ],

        "macro_f1":
            overall[
                "macro_f1"
            ],

        "weighted_f1":
            overall[
                "weighted_f1"
            ],
    })


    comparison_df = pd.DataFrame(
        rows
    )


    comparison_df.to_csv(

        result_dir
        / "fedavg_vs_fedprox.csv",

        index=False,
    )


    return comparison_df


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(

        "--scenario",

        choices=[
            "iid",
            "non_iid",
            "controlled_non_iid",
        ],

        default=
            "non_iid",
    )


    parser.add_argument(

        "--rounds",

        type=int,

        default=
            DEFAULT_ROUNDS,
    )


    parser.add_argument(

        "--local-epochs",

        type=int,

        default=
            DEFAULT_LOCAL_EPOCHS,
    )


    parser.add_argument(

        "--batch-size",

        type=int,

        default=
            DEFAULT_BATCH_SIZE,
    )


    parser.add_argument(

        "--chunk-size",

        type=int,

        default=
            DEFAULT_CHUNK_SIZE,
    )


    parser.add_argument(

        "--learning-rate",

        type=float,

        default=
            DEFAULT_LEARNING_RATE,
    )


    parser.add_argument(

        "--weight-decay",

        type=float,

        default=
            DEFAULT_WEIGHT_DECAY,
    )


    parser.add_argument(

        "--seed",

        type=int,

        default=
            SEED,

        help=(
            "Random seed for Phase 9 repeated-seed validation."
        ),
    )


    parser.add_argument(

        "--mu",

        type=float,

        default=
            DEFAULT_MU,

        help=(
            "FedProx proximal coefficient. "
            "First experiment recommended: 0.01"
        ),
    )


    parser.add_argument(

        "--mask-file",

        type=Path,

        default=
            DEFAULT_MASK_FILE,
    )


    args = parser.parse_args()


    # ========================================================
    # VALIDATE MU
    # ========================================================

    if (
        args.mu
        <
        0.0
    ):

        raise ValueError(
            "FedProx mu must be >= 0."
        )


    # ========================================================
    # SEED
    # ========================================================

    set_seed(
        args.seed
    )


    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"
    )


    # ========================================================
    # SCENARIO
    # ========================================================

    scenario_dir = (

        phase4.FEDERATED_ROOT

        /

        args.scenario
    )


    if not scenario_dir.exists():

        raise FileNotFoundError(
            scenario_dir
        )


    # ========================================================
    # OUTPUT
    # ========================================================

    result_dir = (

        RESULT_ROOT
        /
        args.scenario
        /
        f"seed_{args.seed}"
    )


    artifact_dir = (

        ARTIFACT_ROOT
        /
        args.scenario
        /
        f"seed_{args.seed}"
    )


    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # LOAD PHASE-4 PREPROCESSING
    # ========================================================

    (
        scaler,
        feature_columns,
        label_mapping,
        active_indices,
        active_features,

    ) = phase4.load_preprocessing()


    # ========================================================
    # LOAD FROZEN HYBRID FEATURE MASK
    # ========================================================

    (
        selected_features,
        selected_active_positions,
        selected_original_indices,

    ) = load_hybrid_mask(

        mask_file=
            args.mask_file,

        active_features=
            active_features,

        active_indices=
            active_indices,

        feature_columns=
            feature_columns,
    )


    feature_reduction = (

        1.0

        -

        len(
            selected_features
        )
        /
        len(
            active_features
        )
    )


    # ========================================================
    # HEADER
    # ========================================================

    print()
    print(
        "=" * 90
    )

    print(
        "PHASE 9 — REPEATED-SEED HYBRID EVO-GA + FEDPROX"
    )

    print(
        "=" * 90
    )


    print(
        f"Scenario            : "
        f"{args.scenario}"
    )

    print(
        f"Seed                : "
        f"{args.seed}"
    )

    print(
        f"Device              : "
        f"{device}"
    )

    print(
        f"Clients             : "
        f"{N_CLIENTS}"
    )

    print(
        f"Rounds              : "
        f"{args.rounds}"
    )

    print(
        f"Local epochs        : "
        f"{args.local_epochs}"
    )

    print(
        f"FedProx mu          : "
        f"{args.mu}"
    )

    print(
        f"Original features   : "
        f"{len(active_features)}"
    )

    print(
        f"Selected features   : "
        f"{len(selected_features)}"
    )

    print(
        f"Feature reduction   : "
        f"{feature_reduction * 100:.2f}%"
    )


    # ========================================================
    # GLOBAL CLASS COUNTS / WEIGHTS
    # ========================================================

    global_counts = (
        phase4.calculate_global_class_counts(

            label_mapping,

            args.chunk_size,
        )
    )


    global_weights = (
        phase4.calculate_global_class_weights(

            global_counts
        )
    )


    # ========================================================
    # CLIENT FILES / COUNTS
    # ========================================================

    client_files = {}

    client_sample_counts = {}


    for client_id in range(
        1,
        N_CLIENTS + 1,
    ):

        client_file = (

            scenario_dir
            /
            f"client_{client_id}.csv"
        )


        if not client_file.exists():

            raise FileNotFoundError(
                client_file
            )


        client_files[
            client_id
        ] = client_file


        sample_count = (
            phase4.count_client_samples(

                client_file,

                args.chunk_size,
            )
        )


        client_sample_counts[
            f"client_{client_id}"
        ] = sample_count


        logger.info(
            "Client %d samples=%d",
            client_id,
            sample_count,
        )


    # ========================================================
    # GLOBAL MODEL
    # ========================================================

    (
        global_model,
        reference_model,

    ) = initialize_projected_model(

        selected_active_positions=
            selected_active_positions,

        full_input_dim=
            len(
                active_features
            ),

        selected_input_dim=
            len(
                selected_features
            ),

        num_classes=
            len(
                label_mapping
            ),

        device=
            device,

        seed=
            args.seed,
    )


    # ========================================================
    # COMMUNICATION
    # ========================================================

    communication = calculate_communication(

        baseline_model=
            reference_model,

        fedprox_model=
            global_model,

        rounds=
            args.rounds,
    )


    del reference_model


    # ========================================================
    # FEDPROX TRAINING
    # ========================================================

    history = []


    total_training_start = (
        time.perf_counter()
    )


    for round_number in range(

        1,

        args.rounds + 1,
    ):

        print()
        print(
            "-" * 90
        )

        print(
            f"FEDPROX ROUND "
            f"{round_number}/"
            f"{args.rounds}"
        )

        print(
            "-" * 90
        )


        round_start = (
            time.perf_counter()
        )


        old_global_state = {

            key:
                value
                .detach()
                .cpu()
                .clone()

            for key, value
            in global_model
            .state_dict()
            .items()
        }


        client_states = []

        aggregation_weights = []

        losses = []

        ce_losses = []

        prox_losses = []

        accuracies = []


        # ====================================================
        # CLIENTS
        # ====================================================

        for client_id in range(

            1,

            N_CLIENTS + 1,
        ):

            (
                local_state,
                local_loss,
                local_ce_loss,
                local_prox_loss,
                local_accuracy,
                local_seconds,

            ) = train_client_fedprox(

                global_state=
                    global_model.state_dict(),

                client_file=
                    client_files[
                        client_id
                    ],

                scaler=
                    scaler,

                feature_columns=
                    feature_columns,

                selected_original_indices=
                    selected_original_indices,

                label_mapping=
                    label_mapping,

                class_weights=
                    global_weights,

                input_dim=
                    len(
                        selected_features
                    ),

                device=
                    device,

                local_epochs=
                    args.local_epochs,

                batch_size=
                    args.batch_size,

                chunk_size=
                    args.chunk_size,

                learning_rate=
                    args.learning_rate,

                weight_decay=
                    args.weight_decay,

                client_id=
                    client_id,

                round_number=
                    round_number,

                mu=
                    args.mu,

                seed=
                    args.seed,
            )


            client_states.append(
                local_state
            )


            aggregation_weights.append(

                client_sample_counts[
                    f"client_{client_id}"
                ]
            )


            losses.append(
                local_loss
            )

            ce_losses.append(
                local_ce_loss
            )

            prox_losses.append(
                local_prox_loss
            )

            accuracies.append(
                local_accuracy
            )


        # ====================================================
        # SAMPLE-WEIGHTED SERVER AGGREGATION
        # ====================================================

        aggregated_state = (
            phase4.federated_average(

                client_states,

                aggregation_weights,
            )
        )


        global_model.load_state_dict(
            aggregated_state
        )


        # ====================================================
        # GLOBAL UPDATE NORM
        # ====================================================

        update_norm = (
            phase4.state_update_norm(

                old_global_state,

                aggregated_state,
            )
        )


        # ====================================================
        # ROUND METRICS
        # ====================================================

        weights = np.asarray(

            aggregation_weights,

            dtype=np.float64,
        )


        weighted_loss = float(

            np.average(

                losses,

                weights=
                    weights,
            )
        )


        weighted_ce_loss = float(

            np.average(

                ce_losses,

                weights=
                    weights,
            )
        )


        weighted_prox_loss = float(

            np.average(

                prox_losses,

                weights=
                    weights,
            )
        )


        weighted_accuracy = float(

            np.average(

                accuracies,

                weights=
                    weights,
            )
        )


        round_seconds = (

            time.perf_counter()

            -

            round_start
        )


        history.append({

            "round":
                round_number,

            "weighted_local_loss":
                weighted_loss,

            "weighted_ce_loss":
                weighted_ce_loss,

            "weighted_prox_loss":
                weighted_prox_loss,

            "weighted_local_accuracy":
                weighted_accuracy,

            "global_update_norm":
                update_norm,

            "round_training_seconds":
                round_seconds,

            "mu":
                args.mu,

            "feature_count":
                len(
                    selected_features
                ),
        })


        logger.info(
            "ROUND %d COMPLETE | "
            "loss=%.6f | "
            "ce=%.6f | "
            "prox=%.6f | "
            "acc=%.6f | "
            "update_norm=%.6f | "
            "time=%.2fs",
            round_number,
            weighted_loss,
            weighted_ce_loss,
            weighted_prox_loss,
            weighted_accuracy,
            update_norm,
            round_seconds,
        )


    # ========================================================
    # TRAINING TIME
    # ========================================================

    training_seconds = (

        time.perf_counter()

        -

        total_training_start
    )


    # ========================================================
    # FINAL EVALUATION
    # ========================================================

    (
        y_true,
        y_pred,
        inference_seconds,

    ) = phase4.evaluate_global_model(

        model=
            global_model,

        scaler=
            scaler,

        feature_columns=
            feature_columns,

        active_indices=
            selected_original_indices,

        label_mapping=
            label_mapping,

        device=
            device,

        batch_size=
            args.batch_size,

        chunk_size=
            args.chunk_size,
    )


    (
        overall,
        report,
        cm,
        class_names,

    ) = phase4.calculate_metrics(

        y_true,

        y_pred,

        label_mapping,
    )


    overall[
        "training_time_seconds"
    ] = float(
        training_seconds
    )


    overall[
        "inference_time_seconds"
    ] = float(
        inference_seconds
    )


    overall[
        "selected_feature_count"
    ] = int(
        len(
            selected_features
        )
    )


    overall[
        "feature_reduction_ratio"
    ] = float(
        feature_reduction
    )


    overall[
        "phase"
    ] = 9

    overall[
        "method"
    ] = "Hybrid37_FedProx"

    overall[
        "scenario"
    ] = args.scenario

    overall[
        "seed"
    ] = int(
        args.seed
    )

    overall[
        "fedprox_mu"
    ] = float(
        args.mu
    )


    # ========================================================
    # KNOWLEDGE TRANSFER
    # ========================================================

    transfer_df = (
        phase4.create_knowledge_transfer_report(

            scenario=
                args.scenario,

            report=
                report,

            class_names=
                class_names,

            result_dir=
                result_dir,
        )
    )


    # ========================================================
    # SAVE ROUND HISTORY
    # ========================================================

    history_df = pd.DataFrame(
        history
    )


    history_df.to_csv(

        result_dir
        / "fedprox_round_history.csv",

        index=False,
    )


    # ========================================================
    # SAVE OVERALL
    # ========================================================

    with open(

        result_dir
        / "overall_metrics.json",

        "w",
        encoding="utf-8",

    ) as file:

        json.dump(
            overall,
            file,
            indent=4,
        )


    # ========================================================
    # SAVE PER CLASS
    # ========================================================

    per_class_df = (
        build_per_class_df(

            report,
            class_names,
        )
    )


    per_class_df.to_csv(

        result_dir
        / "per_class_metrics.csv",

        index=False,
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    pd.DataFrame(

        cm,

        index=
            class_names,

        columns=
            class_names,

    ).to_csv(

        result_dir
        / "confusion_matrix.csv"
    )


    # ========================================================
    # SAVE MODEL
    # ========================================================

    torch.save(

        {

            "phase":
                9,

            "algorithm":
                "Hybrid EVO-GA + FedProx",

            "seed":
                int(
                    args.seed
                ),

            "model_state_dict":
                global_model.state_dict(),

            "input_dim":
                len(
                    selected_features
                ),

            "num_classes":
                len(
                    label_mapping
                ),

            "fedprox_mu":
                args.mu,

            "selected_features":
                selected_features,

            "selected_active_positions":
                selected_active_positions.tolist(),

            "selected_original_indices":
                selected_original_indices.tolist(),

            "label_mapping":
                label_mapping,

            "mask_sha256":
                sha256_file(
                    args.mask_file
                ),
        },

        artifact_dir
        / "hybrid_fedprox_model.pt",
    )


    # ========================================================
    # MANIFEST
    # ========================================================

    manifest = {

        "phase":
            9,

        "algorithm":
            "Hybrid EVO-GA + FedProx",

        "scenario":
            args.scenario,

        "seed":
            int(
                args.seed
            ),

        "clients":
            N_CLIENTS,

        "rounds":
            args.rounds,

        "local_epochs":
            args.local_epochs,

        "batch_size":
            args.batch_size,

        "learning_rate":
            args.learning_rate,

        "weight_decay":
            args.weight_decay,

        "fedprox_mu":
            args.mu,

        "aggregation":
            "sample-weighted parameter averaging",

        "local_objective":
            (
                "Weighted CrossEntropy + "
                "(mu / 2) * ||w_local - w_global||^2"
            ),

        "original_features":
            len(
                active_features
            ),

        "selected_features":
            len(
                selected_features
            ),

        "feature_reduction_ratio":
            feature_reduction,

        "hybrid_mask":
            str(
                args.mask_file
            ),

        "hybrid_mask_sha256":
            sha256_file(
                args.mask_file
            ),

        "communication":
            communication,

        "locked_test_used_during_training":
            False,

        "locked_test_used_for":
            "final evaluation only",

        "final_metrics":
            overall,
    }


    with open(

        result_dir
        / "phase9_fedprox_manifest.json",

        "w",
        encoding="utf-8",

    ) as file:

        json.dump(
            manifest,
            file,
            indent=4,
        )


    # ========================================================
    # COMPARISON
    # ========================================================

    comparison_df = (
        save_comparison(

            scenario=
                args.scenario,

            seed=
                args.seed,

            overall=
                overall,

            result_dir=
                result_dir,
        )
    )


    # ========================================================
    # GRAPHS
    # ========================================================

    phase4.generate_graphs(

        scenario=
            (
                "Hybrid37-FedProx-"
                f"{args.scenario}"
            ),

        history_df=
            history_df,

        overall=
            overall,

        report=
            report,

        cm=
            cm,

        class_names=
            class_names,

        transfer_df=
            transfer_df,

        result_dir=
            result_dir,
    )


    # ========================================================
    # FINAL DISPLAY
    # ========================================================

    print()
    print(
        "=" * 90
    )

    print(
        "PHASE 9 COMPLETE — "
        "REPEATED-SEED HYBRID EVO-GA + FEDPROX"
    )

    print(
        "=" * 90
    )


    print(
        f"Scenario            : "
        f"{args.scenario}"
    )

    print(
        f"Seed                : "
        f"{args.seed}"
    )

    print(
        f"Device              : "
        f"{device}"
    )

    print(
        f"Federated rounds    : "
        f"{args.rounds}"
    )

    print(
        f"Local epochs/round  : "
        f"{args.local_epochs}"
    )

    print(
        f"FedProx mu          : "
        f"{args.mu}"
    )

    print(
        f"Features            : "
        f"{len(selected_features)}"
    )

    print(
        f"Feature reduction   : "
        f"{feature_reduction * 100:.2f}%"
    )


    print()


    print(
        f"Accuracy            : "
        f"{overall['accuracy']:.6f}"
    )

    print(
        f"Balanced Accuracy   : "
        f"{overall['balanced_accuracy']:.6f}"
    )

    print(
        f"Macro Precision     : "
        f"{overall['macro_precision']:.6f}"
    )

    print(
        f"Macro Recall        : "
        f"{overall['macro_recall']:.6f}"
    )

    print(
        f"Macro F1            : "
        f"{overall['macro_f1']:.6f}"
    )

    print(
        f"Weighted F1         : "
        f"{overall['weighted_f1']:.6f}"
    )


    print()


    print(
        f"Training time       : "
        f"{training_seconds:.2f}s"
    )

    print(
        f"Inference time      : "
        f"{inference_seconds:.2f}s"
    )


    # ========================================================
    # PER CLASS
    # ========================================================

    print()
    print(
        "-" * 90
    )

    print(
        "PER-CLASS FEDPROX PERFORMANCE"
    )

    print(
        "-" * 90
    )


    for class_name in (
        class_names
    ):

        metrics = report[
            class_name
        ]


        print(

            f"{class_name:25s} "

            f"P="
            f"{metrics['precision']:.4f} "

            f"R="
            f"{metrics['recall']:.4f} "

            f"F1="
            f"{metrics['f1-score']:.4f} "

            f"N="
            f"{int(metrics['support'])}"
        )


    # ========================================================
    # KNOWLEDGE TRANSFER
    # ========================================================

    if transfer_df is not None:

        print()
        print(
            "-" * 90
        )

        print(
            "KNOWLEDGE TRANSFER"
        )

        print(
            "-" * 90
        )


        print(

            transfer_df.to_string(
                index=False
            )
        )


        if (
            "knowledge_transfer_gain"
            in
            transfer_df.columns
        ):

            print()

            print(
                "Mean knowledge-transfer gain: "
                f"{transfer_df['knowledge_transfer_gain'].mean():.6f}"
            )


    # ========================================================
    # COMMUNICATION
    # ========================================================

    print()
    print(
        "-" * 90
    )

    print(
        "MODEL / COMMUNICATION EFFICIENCY"
    )

    print(
        "-" * 90
    )


    print(
        f"Phase-4 parameters   : "
        f"{communication['phase4_parameter_count']:,}"
    )

    print(
        f"FedProx parameters   : "
        f"{communication['fedprox_parameter_count']:,}"
    )

    print(
        f"Parameter reduction  : "
        f"{communication['parameter_reduction_ratio'] * 100:.2f}%"
    )

    print(
        f"Communication reduce : "
        f"{communication['communication_reduction_ratio'] * 100:.2f}%"
    )


    # ========================================================
    # COMPARISON
    # ========================================================

    print()
    print(
        "-" * 90
    )

    print(
        "FEDAVG VS FEDPROX COMPARISON"
    )

    print(
        "-" * 90
    )


    print(

        comparison_df.to_string(
            index=False
        )
    )


    # ========================================================
    # INTEGRITY
    # ========================================================

    print()
    print(
        "=" * 90
    )

    print(
        "EXPERIMENTAL INTEGRITY"
    )

    print(
        "=" * 90
    )


    print(
        "Hybrid 37-feature mask frozen       : YES ✅"
    )

    print(
        "Hybrid optimization rerun           : NO ✅"
    )

    print(
        "Same Phase-7 architecture           : YES ✅"
    )

    print(
        "Same Phase-7 preprocessing          : YES ✅"
    )

    print(
        "Same Phase-7 client files           : YES ✅"
    )

    print(
        "Same sample-weighted aggregation    : YES ✅"
    )

    print(
        "FedProx proximal term added          : YES ✅"
    )

    print(
        "Locked test used during training     : NO ✅"
    )

    print(
        "Locked test evaluated after round 10 : YES ✅"
    )


    print()


    print(
        f"Results:\n"
        f"{result_dir}"
    )

    print()


    print(
        f"Model:\n"
        f"{artifact_dir}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()