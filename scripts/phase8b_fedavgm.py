"""
Phase 8B - Hybrid EVO-GA + FedAvgM
==================================

Purpose
-------

Evaluate server-side momentum under the same severe non-IID
configuration used by:

    Phase 7:
        Hybrid EVO-GA + FedAvg

    Phase 8:
        Hybrid EVO-GA + FedProx

Frozen feature subset:

    70 -> 37 Hybrid EVO-GA features

FedAvgM
-------

Standard FedAvg:

    delta_t = FedAvg(client_models) - global_model

    w_(t+1) = w_t + delta_t


FedAvgM:

    delta_t = FedAvg(client_models) - global_model

    v_(t+1) = beta * v_t + delta_t

    w_(t+1) = w_t + v_(t+1)


where:

    beta = server momentum coefficient

Default:

    beta = 0.9


Scientific control
------------------

Keep identical:

    - Hybrid 37-feature mask
    - 5 clients
    - non-IID scenario
    - 10 rounds
    - 1 local epoch
    - scaler
    - MLP
    - class weights
    - AdamW local optimizer
    - learning rate
    - weight decay
    - batch size
    - initialization
    - client files

Only change:

    Server aggregation:
        FedAvg -> FedAvgM


Run
---

    python scripts/phase8b_fedavgm.py --scenario non_iid --server-momentum 0.9


Outputs
-------

    results/fedavgm/non_iid/

    artifacts/fedavgm/non_iid/
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
# REUSE PHASE 4
# ============================================================

import phase4_fedavg as phase4


# ============================================================
# CONFIGURATION
# ============================================================

SEED = phase4.SEED

N_CLIENTS = phase4.N_CLIENTS

EXPECTED_PHASE4_FEATURES = 70


# ============================================================
# SAME TRAINING DEFAULTS
# ============================================================

DEFAULT_ROUNDS = (
    phase4.DEFAULT_ROUNDS
)

DEFAULT_LOCAL_EPOCHS = (
    phase4.DEFAULT_LOCAL_EPOCHS
)

DEFAULT_BATCH_SIZE = (
    phase4.DEFAULT_BATCH_SIZE
)

DEFAULT_CHUNK_SIZE = (
    phase4.DEFAULT_CHUNK_SIZE
)

DEFAULT_LEARNING_RATE = (
    phase4.DEFAULT_LEARNING_RATE
)

DEFAULT_WEIGHT_DECAY = (
    phase4.DEFAULT_WEIGHT_DECAY
)


# ============================================================
# FEDAVGM
# ============================================================

DEFAULT_SERVER_MOMENTUM = 0.90


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
    / "fedavgm"
)

ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "fedavgm"
)


# ============================================================
# PREVIOUS RESULTS
# ============================================================

PHASE4_RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "fedavg"
)

PHASE7_RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "hybrid_fedavg"
)

PHASE8_FEDPROX_ROOT = (
    PROJECT_ROOT
    / "results"
    / "fedprox"
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
    "phase8b_fedavgm"
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
# SHA256
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


    missing_columns = (

        required_columns

        -

        set(
            mask_df.columns
        )
    )


    if missing_columns:

        raise ValueError(
            "\nInvalid Hybrid feature mask.\n"
            f"Missing columns: "
            f"{sorted(missing_columns)}"
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
    # VERIFY 70-FEATURE SPACE
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
            "\nHybrid feature mask must contain "
            "exactly 70 rows.\n"
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
            "\nHybrid feature schema mismatch.\n\n"
            f"Missing:\n"
            f"{sorted(missing_from_mask)}\n\n"
            f"Extra:\n"
            f"{sorted(extra_in_mask)}"
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

        if value
        not in {
            0,
            1,
        }
    }


    if invalid_values:

        raise ValueError(
            "Hybrid mask 'selected' must "
            "contain only 0 or 1."
        )


    # ========================================================
    # PRESERVE PHASE-4 ORDER
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
    # ACTIVE -> ORIGINAL SCALER INDEX
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


    reconstructed = [

        feature_columns[
            int(
                index
            )
        ]

        for index
        in selected_original_indices
    ]


    if (
        reconstructed
        !=
        selected_features
    ):

        raise ValueError(
            "\nHybrid feature-index mapping failed."
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
):

    """
    Use the same Phase-4 seed-42 reference initialization,
    then project its first-layer columns to the 37-feature
    Hybrid subset.
    """

    phase4.set_seed(
        SEED
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
# INITIALIZE SERVER MOMENTUM
# ============================================================

def initialize_server_velocity(
    global_state,
):

    """
    Create zero momentum tensors matching the floating-point
    model state.

    Non-floating tensors, if any, are excluded from momentum.
    """

    velocity = {}


    for key, tensor in (
        global_state.items()
    ):

        if torch.is_floating_point(
            tensor
        ):

            velocity[
                key
            ] = torch.zeros_like(
                tensor
            )


    return velocity


# ============================================================
# FEDAVGM SERVER UPDATE
# ============================================================

def apply_fedavgm(
    old_global_state,
    averaged_state,
    velocity,
    beta,
):

    """
    Apply server momentum.

    delta_t =
        averaged_client_state - old_global_state

    velocity_t =
        beta * velocity_(t-1) + delta_t

    new_global =
        old_global_state + velocity_t
    """

    new_state = {}


    delta_norm_squared = 0.0

    velocity_norm_squared = 0.0


    for key in (
        old_global_state.keys()
    ):

        old_tensor = (
            old_global_state[
                key
            ]
        )


        averaged_tensor = (
            averaged_state[
                key
            ]
        )


        # ====================================================
        # FLOATING PARAMETERS
        # ====================================================

        if torch.is_floating_point(
            old_tensor
        ):

            delta = (

                averaged_tensor

                -

                old_tensor
            )


            velocity[
                key
            ] = (

                beta
                *
                velocity[
                    key
                ]

                +

                delta
            )


            new_state[
                key
            ] = (

                old_tensor

                +

                velocity[
                    key
                ]
            )


            delta_norm_squared += float(

                torch.sum(
                    delta.double()
                    ** 2
                ).item()
            )


            velocity_norm_squared += float(

                torch.sum(

                    velocity[
                        key
                    ]
                    .double()
                    ** 2

                ).item()
            )


        # ====================================================
        # NON-FLOATING STATE
        # ====================================================

        else:

            new_state[
                key
            ] = (
                averaged_tensor.clone()
            )


    delta_norm = float(
        np.sqrt(
            delta_norm_squared
        )
    )


    velocity_norm = float(
        np.sqrt(
            velocity_norm_squared
        )
    )


    return (
        new_state,
        velocity,
        delta_norm,
        velocity_norm,
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
# MODEL PAYLOAD
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
# COMMUNICATION ANALYSIS
# ============================================================

def calculate_communication(
    baseline_model,
    fedavgm_model,
    rounds,
):

    baseline_params = (
        parameter_count(
            baseline_model
        )
    )


    fedavgm_params = (
        parameter_count(
            fedavgm_model
        )
    )


    baseline_payload = (
        model_payload_bytes(
            baseline_model
        )
    )


    fedavgm_payload = (
        model_payload_bytes(
            fedavgm_model
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


    fedavgm_total = (

        2
        *
        N_CLIENTS
        *
        rounds
        *
        fedavgm_payload
    )


    return {

        "phase4_parameter_count":
            baseline_params,

        "fedavgm_parameter_count":
            fedavgm_params,

        "parameter_reduction_ratio":
            float(

                1.0

                -

                fedavgm_params
                /
                baseline_params
            ),

        "phase4_payload_mib":
            float(

                baseline_payload

                /

                1024 ** 2
            ),

        "fedavgm_payload_mib":
            float(

                fedavgm_payload

                /

                1024 ** 2
            ),

        "phase4_total_communication_mib":
            float(

                baseline_total

                /

                1024 ** 2
            ),

        "fedavgm_total_communication_mib":
            float(

                fedavgm_total

                /

                1024 ** 2
            ),

        "communication_reduction_ratio":
            float(

                1.0

                -

                fedavgm_total
                /
                baseline_total
            ),

        "server_momentum_extra_client_communication":
            0,

        "note":
            (
                "Server momentum is maintained centrally, "
                "therefore FedAvgM adds no extra client-server "
                "payload under this implementation."
            ),
    }


# ============================================================
# PER-CLASS DATAFRAME
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
# LOAD PREVIOUS METHOD
# ============================================================

def read_method_metrics(
    root,
    scenario,
):

    path = (

        root
        /
        scenario
        /
        "overall_metrics.json"
    )


    return load_json_if_exists(
        path
    )


# ============================================================
# METHOD COMPARISON
# ============================================================

def build_method_comparison(
    scenario,
    fedavgm_overall,
    result_dir,
):

    phase4_metrics = read_method_metrics(

        PHASE4_RESULT_ROOT,

        scenario,
    )


    hybrid_fedavg_metrics = read_method_metrics(

        PHASE7_RESULT_ROOT,

        scenario,
    )


    fedprox_metrics = read_method_metrics(

        PHASE8_FEDPROX_ROOT,

        scenario,
    )


    rows = []


    # ========================================================
    # PHASE-4 70-FEATURE FEDAVG
    # ========================================================

    if phase4_metrics is not None:

        rows.append({

            "method":
                "FedAvg_70",

            "features":
                70,

            "accuracy":
                phase4_metrics.get(
                    "accuracy"
                ),

            "balanced_accuracy":
                phase4_metrics.get(
                    "balanced_accuracy"
                ),

            "macro_f1":
                phase4_metrics.get(
                    "macro_f1"
                ),

            "weighted_f1":
                phase4_metrics.get(
                    "weighted_f1"
                ),
        })


    # ========================================================
    # HYBRID + FEDAVG
    # ========================================================

    if hybrid_fedavg_metrics is not None:

        rows.append({

            "method":
                "Hybrid37_FedAvg",

            "features":
                37,

            "accuracy":
                hybrid_fedavg_metrics.get(
                    "accuracy"
                ),

            "balanced_accuracy":
                hybrid_fedavg_metrics.get(
                    "balanced_accuracy"
                ),

            "macro_f1":
                hybrid_fedavg_metrics.get(
                    "macro_f1"
                ),

            "weighted_f1":
                hybrid_fedavg_metrics.get(
                    "weighted_f1"
                ),
        })


    # ========================================================
    # HYBRID + FEDPROX
    # ========================================================

    if fedprox_metrics is not None:

        rows.append({

            "method":
                "Hybrid37_FedProx",

            "features":
                37,

            "accuracy":
                fedprox_metrics.get(
                    "accuracy"
                ),

            "balanced_accuracy":
                fedprox_metrics.get(
                    "balanced_accuracy"
                ),

            "macro_f1":
                fedprox_metrics.get(
                    "macro_f1"
                ),

            "weighted_f1":
                fedprox_metrics.get(
                    "weighted_f1"
                ),
        })


    # ========================================================
    # HYBRID + FEDAVGM
    # ========================================================

    rows.append({

        "method":
            "Hybrid37_FedAvgM",

        "features":
            fedavgm_overall[
                "selected_feature_count"
            ],

        "accuracy":
            fedavgm_overall[
                "accuracy"
            ],

        "balanced_accuracy":
            fedavgm_overall[
                "balanced_accuracy"
            ],

        "macro_f1":
            fedavgm_overall[
                "macro_f1"
            ],

        "weighted_f1":
            fedavgm_overall[
                "weighted_f1"
            ],
    })


    comparison_df = pd.DataFrame(
        rows
    )


    comparison_df.to_csv(

        result_dir
        / "fl_method_comparison.csv",

        index=False,
    )


    return comparison_df


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    # ========================================================
    # ARGUMENTS
    # ========================================================

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

        "--server-momentum",

        type=float,

        default=
            DEFAULT_SERVER_MOMENTUM,

        help=(
            "FedAvgM server momentum beta. "
            "Default = 0.9"
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
    # VALIDATE MOMENTUM
    # ========================================================

    if not (
        0.0
        <=
        args.server_momentum
        <
        1.0
    ):

        raise ValueError(
            "server-momentum must satisfy "
            "0 <= beta < 1."
        )


    # ========================================================
    # REPRODUCIBILITY
    # ========================================================

    set_seed(
        SEED
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
            f"\nScenario missing:\n"
            f"{scenario_dir}"
        )


    if not phase4.LOCKED_TEST_FILE.exists():

        raise FileNotFoundError(
            f"\nLocked test missing:\n"
            f"{phase4.LOCKED_TEST_FILE}"
        )


    # ========================================================
    # OUTPUT
    # ========================================================

    result_dir = (

        RESULT_ROOT

        /

        args.scenario
    )


    artifact_dir = (

        ARTIFACT_ROOT

        /

        args.scenario
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
    # LOAD PREPROCESSING
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
        EXPECTED_PHASE4_FEATURES
    ):

        raise ValueError(
            "\nExpected 70 Phase-4 active features."
        )


    # ========================================================
    # LOAD FROZEN 37-FEATURE HYBRID MASK
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
        "PHASE 8B — HYBRID EVO-GA + FEDAVGM"
    )

    print(
        "=" * 90
    )


    print(
        f"Scenario            : "
        f"{args.scenario}"
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
        f"Server momentum β   : "
        f"{args.server_momentum}"
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
    # CLIENT FILES
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
    # INITIAL MODEL
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
    )


    # ========================================================
    # COMMUNICATION
    # ========================================================

    communication = (
        calculate_communication(

            baseline_model=
                reference_model,

            fedavgm_model=
                global_model,

            rounds=
                args.rounds,
        )
    )


    del reference_model


    # ========================================================
    # SERVER MOMENTUM INITIALIZATION
    # ========================================================

    initial_state = {

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


    velocity = (
        initialize_server_velocity(
            initial_state
        )
    )


    # ========================================================
    # TRAINING
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
            f"FEDAVGM ROUND "
            f"{round_number}/"
            f"{args.rounds}"
        )

        print(
            "-" * 90
        )


        round_start = (
            time.perf_counter()
        )


        # ====================================================
        # OLD GLOBAL STATE
        # ====================================================

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

        client_losses = []

        client_accuracies = []

        client_times = []


        # ====================================================
        # CLIENT TRAINING
        # ====================================================

        for client_id in range(

            1,

            N_CLIENTS + 1,
        ):

            logger.info(
                "Round %d | Client %d | "
                "FedAvgM local training",
                round_number,
                client_id,
            )


            (
                local_state,
                local_loss,
                local_accuracy,
                local_seconds,

            ) = phase4.train_client(

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

                active_indices=
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
            )


            client_states.append(
                local_state
            )


            samples = (

                client_sample_counts[
                    f"client_{client_id}"
                ]
            )


            aggregation_weights.append(
                samples
            )


            client_losses.append(
                local_loss
            )


            client_accuracies.append(
                local_accuracy
            )


            client_times.append(
                local_seconds
            )


        # ====================================================
        # STEP 1: NORMAL SAMPLE-WEIGHTED FEDAVG
        # ====================================================

        averaged_state = (
            phase4.federated_average(

                client_states,

                aggregation_weights,
            )
        )


        # ====================================================
        # STEP 2: SERVER MOMENTUM
        # ====================================================

        (
            momentum_state,
            velocity,
            raw_fedavg_delta_norm,
            server_velocity_norm,

        ) = apply_fedavgm(

            old_global_state=
                old_global_state,

            averaged_state=
                averaged_state,

            velocity=
                velocity,

            beta=
                args.server_momentum,
        )


        # ====================================================
        # LOAD FEDAVGM GLOBAL MODEL
        # ====================================================

        global_model.load_state_dict(
            momentum_state
        )


        # ====================================================
        # ACTUAL GLOBAL UPDATE NORM
        # ====================================================

        actual_update_norm = (
            phase4.state_update_norm(

                old_global_state,

                momentum_state,
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

                np.asarray(
                    client_losses,
                    dtype=np.float64,
                ),

                weights=
                    weights,
            )
        )


        weighted_accuracy = float(

            np.average(

                np.asarray(
                    client_accuracies,
                    dtype=np.float64,
                ),

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

            "weighted_local_accuracy":
                weighted_accuracy,

            "raw_fedavg_delta_norm":
                raw_fedavg_delta_norm,

            "server_velocity_norm":
                server_velocity_norm,

            "global_update_norm":
                actual_update_norm,

            "server_momentum":
                args.server_momentum,

            "round_training_seconds":
                round_seconds,

            "client_1_loss":
                client_losses[0],

            "client_2_loss":
                client_losses[1],

            "client_3_loss":
                client_losses[2],

            "client_4_loss":
                client_losses[3],

            "client_5_loss":
                client_losses[4],

            "feature_count":
                len(
                    selected_features
                ),
        })


        logger.info(
            "ROUND %d COMPLETE | "
            "loss=%.6f | "
            "acc=%.6f | "
            "raw_delta=%.6f | "
            "velocity=%.6f | "
            "actual_update=%.6f | "
            "time=%.2fs",
            round_number,
            weighted_loss,
            weighted_accuracy,
            raw_fedavg_delta_norm,
            server_velocity_norm,
            actual_update_norm,
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
    # FINAL LOCKED TEST
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


    # ========================================================
    # FINAL METRICS
    # ========================================================

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
        "server_momentum"
    ] = float(
        args.server_momentum
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
    # SAVE HISTORY
    # ========================================================

    history_df = pd.DataFrame(
        history
    )


    history_df.to_csv(

        result_dir
        / "fedavgm_round_history.csv",

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
    # SAVE PER-CLASS
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
    # SAVE CONFUSION MATRIX
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
    # SAVE COMMUNICATION
    # ========================================================

    with open(

        result_dir
        / "communication_metrics.json",

        "w",
        encoding="utf-8",

    ) as file:

        json.dump(
            communication,
            file,
            indent=4,
        )


    # ========================================================
    # SAVE MODEL
    # ========================================================

    torch.save(

        {

            "phase":
                "8B",

            "algorithm":
                "Hybrid EVO-GA + FedAvgM",

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

            "server_momentum":
                args.server_momentum,

            "selected_features":
                selected_features,

            "selected_active_positions":
                selected_active_positions.tolist(),

            "selected_original_indices":
                selected_original_indices.tolist(),

            "label_mapping":
                label_mapping,

            "mask_file":
                str(
                    args.mask_file
                ),

            "mask_sha256":
                sha256_file(
                    args.mask_file
                ),
        },

        artifact_dir
        / "hybrid_fedavgm_model.pt",
    )


    # ========================================================
    # MANIFEST
    # ========================================================

    manifest = {

        "phase":
            "8B",

        "algorithm":
            "Hybrid EVO-GA + FedAvgM",

        "scenario":
            args.scenario,

        "seed":
            SEED,

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

        "server_momentum":
            args.server_momentum,

        "local_training":
            "Same local AdamW training as Phase 7 FedAvg",

        "aggregation":
            (
                "Sample-weighted FedAvg followed by "
                "server momentum"
            ),

        "fedavgm_equation":
            (
                "v_t = beta * v_(t-1) + "
                "(FedAvg(client_states) - w_t); "
                "w_(t+1) = w_t + v_t"
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
        / "phase8b_fedavgm_manifest.json",

        "w",
        encoding="utf-8",

    ) as file:

        json.dump(
            manifest,
            file,
            indent=4,
        )


    # ========================================================
    # METHOD COMPARISON
    # ========================================================

    comparison_df = (
        build_method_comparison(

            scenario=
                args.scenario,

            fedavgm_overall=
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
                "Hybrid37-FedAvgM-"
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
        "PHASE 8B COMPLETE — "
        "HYBRID EVO-GA + FEDAVGM"
    )

    print(
        "=" * 90
    )


    print(
        f"Scenario            : "
        f"{args.scenario}"
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
        f"Server momentum β   : "
        f"{args.server_momentum}"
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
        "PER-CLASS FEDAVGM PERFORMANCE"
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

            mean_transfer = float(

                transfer_df[
                    "knowledge_transfer_gain"
                ]
                .mean()
            )


            print()

            print(
                "Mean knowledge-transfer gain: "
                f"{mean_transfer:.6f}"
            )


    # ========================================================
    # MOMENTUM DIAGNOSTICS
    # ========================================================

    print()
    print(
        "-" * 90
    )

    print(
        "FEDAVGM MOMENTUM DIAGNOSTICS"
    )

    print(
        "-" * 90
    )


    print(
        f"Final raw FedAvg delta norm : "
        f"{history_df['raw_fedavg_delta_norm'].iloc[-1]:.6f}"
    )

    print(
        f"Final server velocity norm  : "
        f"{history_df['server_velocity_norm'].iloc[-1]:.6f}"
    )

    print(
        f"Final global update norm    : "
        f"{history_df['global_update_norm'].iloc[-1]:.6f}"
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
        f"FedAvgM parameters   : "
        f"{communication['fedavgm_parameter_count']:,}"
    )

    print(
        f"Parameter reduction  : "
        f"{communication['parameter_reduction_ratio'] * 100:.2f}%"
    )

    print(
        f"Communication reduce : "
        f"{communication['communication_reduction_ratio'] * 100:.2f}%"
    )

    print(
        "Extra momentum communication: 0 bytes"
    )


    # ========================================================
    # METHOD COMPARISON
    # ========================================================

    print()
    print(
        "-" * 90
    )

    print(
        "CURRENT FEDERATED METHOD COMPARISON"
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
    # EXPERIMENTAL INTEGRITY
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
        "Same Phase-7 local model            : YES ✅"
    )

    print(
        "Same local optimizer                : YES ✅"
    )

    print(
        "Same class weights                  : YES ✅"
    )

    print(
        "Same client files                   : YES ✅"
    )

    print(
        "Same local epochs                   : YES ✅"
    )

    print(
        "Same federated rounds               : YES ✅"
    )

    print(
        "Sample-weighted client averaging    : YES ✅"
    )

    print(
        "Server momentum added               : YES ✅"
    )

    print(
        "Extra client communication          : NO ✅"
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