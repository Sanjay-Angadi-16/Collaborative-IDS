"""
Phase 8E - Hybrid EVO-GA + Coordinate-wise Median
=================================================

Purpose
-------

Evaluate a robust aggregation rule under the same severe non-IID setup used by:

    Hybrid37 + FedAvg
    Hybrid37 + FedProx
    Hybrid37 + FedAvgM
    Hybrid37 + FedNova
    Hybrid37 + SCAFFOLD

Frozen feature subset:

    70 -> 37 Hybrid EVO-GA features


Coordinate-wise Median
----------------------

For every model parameter coordinate j:

    w_global[j] = median(
        w_client_1[j],
        w_client_2[j],
        ...,
        w_client_N[j]
    )

Unlike FedAvg, client sample counts are NOT used in the aggregation itself.

This makes coordinate-wise Median more robust to extreme/outlier
client updates, but it may sacrifice performance when all clients
are honest and highly heterogeneous.

IMPORTANT
---------

Keep exactly the same:

    - 37 Hybrid features
    - 5 clients
    - severe non-IID scenario
    - 10 rounds
    - 1 local epoch
    - same MLP
    - same scaler
    - same class weights
    - same AdamW
    - same learning rate
    - same weight decay
    - same client datasets
    - same initialization

Only change:

    Server aggregation:
        sample-weighted FedAvg
            ->
        coordinate-wise Median


Run
---

    python scripts/phase8e_median.py --scenario non_iid


Outputs
-------

    results/median/non_iid/

    artifacts/median/non_iid/
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
# CONFIG
# ============================================================

SEED = phase4.SEED

N_CLIENTS = phase4.N_CLIENTS

EXPECTED_PHASE4_FEATURES = 70


DEFAULT_ROUNDS = phase4.DEFAULT_ROUNDS
DEFAULT_LOCAL_EPOCHS = phase4.DEFAULT_LOCAL_EPOCHS
DEFAULT_BATCH_SIZE = phase4.DEFAULT_BATCH_SIZE
DEFAULT_CHUNK_SIZE = phase4.DEFAULT_CHUNK_SIZE
DEFAULT_LEARNING_RATE = phase4.DEFAULT_LEARNING_RATE
DEFAULT_WEIGHT_DECAY = phase4.DEFAULT_WEIGHT_DECAY


# ============================================================
# HYBRID MASK
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
    / "median"
)

ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "median"
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

PHASE8B_FEDAVGM_ROOT = (
    PROJECT_ROOT
    / "results"
    / "fedavgm"
)

PHASE8C_FEDNOVA_ROOT = (
    PROJECT_ROOT
    / "results"
    / "fednova"
)

PHASE8D_SCAFFOLD_ROOT = (
    PROJECT_ROOT
    / "results"
    / "scaffold"
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
    "phase8e_median"
)


# ============================================================
# SEED
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
# FILE HASH
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
# LOAD JSON
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
            "\nHybrid mask not found:\n"
            f"{mask_file}"
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
            "\nInvalid Hybrid mask.\n"
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


    if len(active_features) != EXPECTED_PHASE4_FEATURES:

        raise ValueError(
            "\nExpected exactly 70 Phase-4 features."
        )


    if len(mask_df) != EXPECTED_PHASE4_FEATURES:

        raise ValueError(
            "\nHybrid mask must contain exactly 70 rows."
        )


    mask_features = set(
        mask_df[
            "feature_name"
        ]
    )

    active_feature_set = set(
        active_features
    )


    if mask_features != active_feature_set:

        raise ValueError(
            "\nHybrid feature schema does not match Phase-4."
        )


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
            "selected column must contain only 0 or 1."
        )


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


    if reconstructed != selected_features:

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

    phase4.set_seed(
        SEED
    )


    reference_model = phase4.IDSMLP(
        input_dim=
            full_input_dim,
        num_classes=
            num_classes,
    ).to(
        device
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


    selected_model = phase4.IDSMLP(
        input_dim=
            selected_input_dim,
        num_classes=
            num_classes,
    ).to(
        device
    )


    selected_state = selected_model.state_dict()


    first_layer_key = "network.0.weight"


    positions = [

        int(
            value
        )

        for value
        in selected_active_positions
    ]


    for key in selected_state.keys():

        if key == first_layer_key:

            projected = (
                reference_state[
                    key
                ][
                    :,
                    positions
                ]
                .clone()
            )

            if projected.shape != selected_state[key].shape:

                raise ValueError(
                    "\nProjected layer shape mismatch."
                )

            selected_state[
                key
            ] = projected

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
# COORDINATE-WISE MEDIAN
# ============================================================

def coordinate_wise_median(
    client_states,
):

    """
    Robust aggregation:

        median(
            client_1_parameter,
            ...
            client_N_parameter
        )

    independently for every parameter coordinate.
    """

    if not client_states:

        raise ValueError(
            "No client states supplied."
        )


    keys = client_states[0].keys()

    aggregated_state = {}


    for key in keys:

        reference_tensor = client_states[0][
            key
        ]


        # ====================================================
        # FLOAT PARAMETERS
        # ====================================================

        if torch.is_floating_point(
            reference_tensor
        ):

            stacked = torch.stack(

                [

                    client_state[
                        key
                    ]
                    .float()

                    for client_state
                    in client_states
                ],

                dim=0,
            )


            median_tensor = torch.median(

                stacked,

                dim=0,
            ).values


            aggregated_state[
                key
            ] = median_tensor.to(
                dtype=
                    reference_tensor.dtype
            )


        # ====================================================
        # NON-FLOATING STATE
        # ====================================================

        else:

            aggregated_state[
                key
            ] = (
                reference_tensor.clone()
            )


    return aggregated_state


# ============================================================
# CLIENT UPDATE NORMS
# ============================================================

def calculate_client_update_norms(
    old_global_state,
    client_states,
):

    norms = []


    for client_state in (
        client_states
    ):

        norm_squared = 0.0


        for key in (
            old_global_state.keys()
        ):

            old_tensor = (
                old_global_state[
                    key
                ]
            )


            client_tensor = (
                client_state[
                    key
                ]
            )


            if not torch.is_floating_point(
                old_tensor
            ):

                continue


            delta = (

                client_tensor

                -

                old_tensor
            )


            norm_squared += float(

                torch.sum(

                    delta.double()
                    ** 2

                ).item()
            )


        norms.append(
            float(
                np.sqrt(
                    norm_squared
                )
            )
        )


    return norms


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
# COMMUNICATION
# ============================================================

def calculate_communication(
    baseline_model,
    median_model,
    rounds,
):

    baseline_params = (
        parameter_count(
            baseline_model
        )
    )


    median_params = (
        parameter_count(
            median_model
        )
    )


    baseline_payload = (
        model_payload_bytes(
            baseline_model
        )
    )


    median_payload = (
        model_payload_bytes(
            median_model
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


    median_total = (
        2
        *
        N_CLIENTS
        *
        rounds
        *
        median_payload
    )


    return {

        "phase4_parameter_count":
            baseline_params,

        "median_parameter_count":
            median_params,

        "parameter_reduction_ratio":
            float(
                1.0
                -
                median_params
                /
                baseline_params
            ),

        "phase4_model_payload_mib":
            float(
                baseline_payload
                /
                1024 ** 2
            ),

        "median_model_payload_mib":
            float(
                median_payload
                /
                1024 ** 2
            ),

        "phase4_total_communication_mib":
            float(
                baseline_total
                /
                1024 ** 2
            ),

        "median_total_communication_mib":
            float(
                median_total
                /
                1024 ** 2
            ),

        "communication_reduction_ratio":
            float(
                1.0
                -
                median_total
                /
                baseline_total
            ),

        "extra_client_communication":
            0,

        "note":
            (
                "Coordinate-wise Median uses the same model "
                "payload as FedAvg; only server aggregation "
                "changes."
            ),
    }


# ============================================================
# PER CLASS
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
# READ PREVIOUS METHOD
# ============================================================

def read_method_metrics(
    root,
    scenario,
):

    return load_json_if_exists(

        root
        /
        scenario
        /
        "overall_metrics.json"
    )


# ============================================================
# COMPARISON TABLE
# ============================================================

def build_method_comparison(
    scenario,
    median_overall,
    result_dir,
):

    previous_methods = [

        (
            "FedAvg_70",
            70,
            PHASE4_RESULT_ROOT,
        ),

        (
            "Hybrid37_FedAvg",
            37,
            PHASE7_RESULT_ROOT,
        ),

        (
            "Hybrid37_FedProx",
            37,
            PHASE8_FEDPROX_ROOT,
        ),

        (
            "Hybrid37_FedAvgM",
            37,
            PHASE8B_FEDAVGM_ROOT,
        ),

        (
            "Hybrid37_FedNova",
            37,
            PHASE8C_FEDNOVA_ROOT,
        ),

        (
            "Hybrid37_SCAFFOLD",
            37,
            PHASE8D_SCAFFOLD_ROOT,
        ),
    ]


    rows = []


    for (
        method_name,
        feature_count,
        result_root,

    ) in previous_methods:

        metrics = read_method_metrics(
            result_root,
            scenario,
        )


        if metrics is None:
            continue


        rows.append({

            "method":
                method_name,

            "features":
                feature_count,

            "accuracy":
                metrics.get(
                    "accuracy"
                ),

            "balanced_accuracy":
                metrics.get(
                    "balanced_accuracy"
                ),

            "macro_f1":
                metrics.get(
                    "macro_f1"
                ),

            "weighted_f1":
                metrics.get(
                    "weighted_f1"
                ),
        })


    rows.append({

        "method":
            "Hybrid37_Median",

        "features":
            median_overall[
                "selected_feature_count"
            ],

        "accuracy":
            median_overall[
                "accuracy"
            ],

        "balanced_accuracy":
            median_overall[
                "balanced_accuracy"
            ],

        "macro_f1":
            median_overall[
                "macro_f1"
            ],

        "weighted_f1":
            median_overall[
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

        "--mask-file",

        type=Path,

        default=
            DEFAULT_MASK_FILE,
    )


    args = parser.parse_args()


    # ========================================================
    # SEED
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
            "\nLocked test missing."
        )


    # ========================================================
    # OUTPUT DIRS
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


    # ========================================================
    # LOAD HYBRID MASK
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
        "PHASE 8E — HYBRID EVO-GA + COORDINATE-WISE MEDIAN"
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

    print(
        "Aggregation         : coordinate-wise median"
    )


    # ========================================================
    # CLASS WEIGHTS
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
    # INITIAL GLOBAL MODEL
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

            median_model=
                global_model,

            rounds=
                args.rounds,
        )
    )


    del reference_model


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
            f"MEDIAN ROUND "
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

        client_losses = []

        client_accuracies = []

        client_times = []


        # ====================================================
        # LOCAL TRAINING
        # ====================================================

        for client_id in range(
            1,
            N_CLIENTS + 1,
        ):

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
        # CLIENT UPDATE DIAGNOSTICS
        # ====================================================

        update_norms = (
            calculate_client_update_norms(

                old_global_state=
                    old_global_state,

                client_states=
                    client_states,
            )
        )


        # ====================================================
        # COORDINATE-WISE MEDIAN
        # ====================================================

        aggregated_state = (
            coordinate_wise_median(

                client_states=
                    client_states
            )
        )


        global_model.load_state_dict(
            aggregated_state
        )


        # ====================================================
        # GLOBAL UPDATE NORM
        # ====================================================

        global_update_norm = (
            phase4.state_update_norm(

                old_global_state,

                aggregated_state,
            )
        )


        # ====================================================
        # ROUND METRICS
        #
        # These are reporting diagnostics only.
        # Since Median does not sample-weight aggregation,
        # we report both equal-client and sample-weighted stats.
        # ====================================================

        sample_weights = np.asarray(

            [

                client_sample_counts[
                    f"client_{client_id}"
                ]

                for client_id
                in range(
                    1,
                    N_CLIENTS + 1,
                )
            ],

            dtype=np.float64,
        )


        sample_weighted_loss = float(

            np.average(

                np.asarray(
                    client_losses,
                    dtype=np.float64,
                ),

                weights=
                    sample_weights,
            )
        )


        sample_weighted_accuracy = float(

            np.average(

                np.asarray(
                    client_accuracies,
                    dtype=np.float64,
                ),

                weights=
                    sample_weights,
            )
        )


        equal_client_loss = float(
            np.mean(
                client_losses
            )
        )


        equal_client_accuracy = float(
            np.mean(
                client_accuracies
            )
        )


        round_seconds = (
            time.perf_counter()
            -
            round_start
        )


        round_record = {

            "round":
                round_number,

            "sample_weighted_local_loss":
                sample_weighted_loss,

            "sample_weighted_local_accuracy":
                sample_weighted_accuracy,

            "equal_client_local_loss":
                equal_client_loss,

            "equal_client_local_accuracy":
                equal_client_accuracy,

            "global_update_norm":
                global_update_norm,

            "mean_client_update_norm":
                float(
                    np.mean(
                        update_norms
                    )
                ),

            "median_client_update_norm":
                float(
                    np.median(
                        update_norms
                    )
                ),

            "max_client_update_norm":
                float(
                    np.max(
                        update_norms
                    )
                ),

            "min_client_update_norm":
                float(
                    np.min(
                        update_norms
                    )
                ),

            "round_training_seconds":
                round_seconds,

            "feature_count":
                len(
                    selected_features
                ),
        }


        for index, norm in enumerate(
            update_norms,
            start=1,
        ):

            round_record[
                f"client_{index}_update_norm"
            ] = float(
                norm
            )


        history.append(
            round_record
        )


        logger.info(
            "ROUND %d COMPLETE | "
            "sample_loss=%.6f | "
            "sample_acc=%.6f | "
            "median_update=%.6f | "
            "max_update=%.6f | "
            "global_update=%.6f | "
            "time=%.2fs",
            round_number,
            sample_weighted_loss,
            sample_weighted_accuracy,
            float(
                np.median(
                    update_norms
                )
            ),
            float(
                np.max(
                    update_norms
                )
            ),
            global_update_norm,
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
    # METRICS
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
        "aggregation"
    ] = "coordinate-wise median"


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
        / "median_round_history.csv",

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
    # PER CLASS
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
    # COMMUNICATION
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
                "8E",

            "algorithm":
                "Hybrid EVO-GA + Coordinate-wise Median",

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
        / "hybrid_median_model.pt",
    )


    # ========================================================
    # MANIFEST
    # ========================================================

    manifest = {

        "phase":
            "8E",

        "algorithm":
            "Hybrid EVO-GA + Coordinate-wise Median",

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

        "local_optimizer":
            "AdamW",

        "aggregation":
            "coordinate-wise median",

        "aggregation_weighting":
            (
                "No sample-count weighting in median aggregation"
            ),

        "scientific_note":
            (
                "Coordinate-wise Median is primarily a robust "
                "aggregation baseline. Performance under clean "
                "non-IID data should be interpreted separately "
                "from later poisoned-client robustness tests."
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

        "client_sample_counts":
            client_sample_counts,

        "communication":
            communication,

        "hybrid_mask":
            str(
                args.mask_file
            ),

        "hybrid_mask_sha256":
            sha256_file(
                args.mask_file
            ),

        "locked_test_used_during_training":
            False,

        "locked_test_used_for":
            "final evaluation only",

        "final_metrics":
            overall,
    }


    with open(

        result_dir
        / "phase8e_median_manifest.json",

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
        build_method_comparison(

            scenario=
                args.scenario,

            median_overall=
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
                "Hybrid37-Median-"
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
    # FINAL REPORT
    # ========================================================

    print()
    print(
        "=" * 90
    )

    print(
        "PHASE 8E COMPLETE — "
        "HYBRID EVO-GA + COORDINATE-WISE MEDIAN"
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
        f"Features            : "
        f"{len(selected_features)}"
    )

    print(
        f"Feature reduction   : "
        f"{feature_reduction * 100:.2f}%"
    )

    print(
        "Aggregation         : coordinate-wise median"
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
        "PER-CLASS MEDIAN PERFORMANCE"
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
    # MEDIAN DIAGNOSTICS
    # ========================================================

    print()
    print(
        "-" * 90
    )

    print(
        "MEDIAN AGGREGATION DIAGNOSTICS"
    )

    print(
        "-" * 90
    )


    print(
        f"Final mean client update norm   : "
        f"{history_df['mean_client_update_norm'].iloc[-1]:.6f}"
    )

    print(
        f"Final median client update norm : "
        f"{history_df['median_client_update_norm'].iloc[-1]:.6f}"
    )

    print(
        f"Final max client update norm    : "
        f"{history_df['max_client_update_norm'].iloc[-1]:.6f}"
    )

    print(
        f"Final min client update norm    : "
        f"{history_df['min_client_update_norm'].iloc[-1]:.6f}"
    )

    print(
        f"Final global update norm        : "
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
        f"Median parameters    : "
        f"{communication['median_parameter_count']:,}"
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
        "Extra client communication: 0 bytes"
    )


    # ========================================================
    # CURRENT COMPARISON
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
        "Hybrid optimizer rerun              : NO ✅"
    )

    print(
        "Same MLP architecture               : YES ✅"
    )

    print(
        "Same AdamW local optimizer          : YES ✅"
    )

    print(
        "Same preprocessing                  : YES ✅"
    )

    print(
        "Same class weights                  : YES ✅"
    )

    print(
        "Same client files                   : YES ✅"
    )

    print(
        "Same federated rounds               : YES ✅"
    )

    print(
        "Same local epochs                   : YES ✅"
    )

    print(
        "Coordinate-wise median used         : YES ✅"
    )

    print(
        "Sample weighting in aggregation     : NO ✅"
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
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()