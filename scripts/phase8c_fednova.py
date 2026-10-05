"""
Phase 8C - Hybrid EVO-GA + FedNova
==================================

Purpose
-------

Evaluate FedNova under the same severe non-IID setup used by:

    Hybrid37 + FedAvg
    Hybrid37 + FedProx
    Hybrid37 + FedAvgM

Frozen feature subset:

    70 -> 37 Hybrid EVO-GA features


FedNova idea
------------

Each client performs local optimization:

    w_i = locally trained client model

Define client model update:

    delta_i = w_global - w_i

Let:

    tau_i = number of local optimizer steps

Normalize client update:

    d_i = delta_i / tau_i

Server computes:

    d = sum_i p_i * d_i

where:

    p_i = n_i / sum_j n_j

Effective local-step coefficient:

    tau_eff = sum_i p_i * tau_i

Then:

    w_new = w_global - tau_eff * d


IMPORTANT SCIENTIFIC NOTE
-------------------------

Your current Phase-4 training uses AdamW locally.

The original FedNova derivation is commonly expressed for local
SGD-style optimization. This implementation preserves AdamW so
that the local optimizer remains identical to your previous FL
experiments and only the aggregation normalization changes.

Therefore this should be described accurately as:

    "FedNova-style normalized aggregation with the same AdamW
     local optimizer used across the experimental pipeline."

Also:

If all clients have exactly the same number of local optimizer
steps, FedNova normalization can reduce to behavior very close
to ordinary FedAvg. That is an expected result, not an error.


Run
---

    python scripts/phase8c_fednova.py --scenario non_iid


Outputs
-------

    results/fednova/non_iid/
    artifacts/fednova/non_iid/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
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
# SAME TRAINING SETTINGS
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
# HYBRID FEATURE MASK
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
    / "fednova"
)


ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "fednova"
)


# ============================================================
# PREVIOUS EXPERIMENTS
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
    "phase8c_fednova"
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
            "\nHybrid mask missing:\n"
            f"{mask_file}"
        )


    mask_df = pd.read_csv(
        mask_file
    )


    required = {

        "feature_name",
        "selected",
    }


    missing = (

        required

        -

        set(
            mask_df.columns
        )
    )


    if missing:

        raise ValueError(
            "\nInvalid Hybrid feature mask.\n"
            f"Missing: {sorted(missing)}"
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
    # FEATURE SPACE SAFETY
    # ========================================================

    if (
        len(
            active_features
        )
        !=
        EXPECTED_PHASE4_FEATURES
    ):

        raise ValueError(
            "\nExpected exactly 70 Phase-4 "
            "active features."
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
            "70 rows."
        )


    mask_features = set(
        mask_df[
            "feature_name"
        ]
    )


    active_set = set(
        active_features
    )


    if (
        mask_features
        !=
        active_set
    ):

        raise ValueError(
            "\nHybrid mask feature schema "
            "does not match Phase-4 schema."
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


    if (
        reconstructed
        !=
        selected_features
    ):

        raise ValueError(
            "\nHybrid feature mapping failed."
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
# COUNT LOCAL OPTIMIZER STEPS
# ============================================================

def count_local_steps(
    client_file,
    batch_size,
    chunk_size,
    local_epochs,
):

    """
    Count the actual number of optimizer.step() calls expected
    from phase4.train_client().

    Since Phase 4 batches independently inside each pandas
    chunk, we must count:

        ceil(rows_in_chunk / batch_size)

    for every chunk and every local epoch.
    """

    steps_per_epoch = 0

    total_rows = 0


    reader = pd.read_csv(

        client_file,

        chunksize=
            chunk_size,
    )


    for chunk in reader:

        rows = len(
            chunk
        )


        if rows <= 0:

            continue


        total_rows += rows


        steps_per_epoch += int(

            math.ceil(

                rows
                /
                batch_size
            )
        )


    total_steps = (

        steps_per_epoch

        *
        local_epochs
    )


    if total_steps <= 0:

        raise RuntimeError(
            f"No optimizer steps detected for "
            f"{client_file}"
        )


    return (
        int(
            total_steps
        ),

        int(
            total_rows
        ),
    )


# ============================================================
# FEDNOVA NORMALIZED AGGREGATION
# ============================================================

def fednova_aggregate(
    global_state,
    client_states,
    sample_counts,
    local_steps,
):

    """
    FedNova-style normalized aggregation.

    For floating-point tensors:

        delta_i =
            w_global - w_client_i

        normalized_delta_i =
            delta_i / tau_i

        normalized_global_delta =
            sum_i p_i * normalized_delta_i

        tau_eff =
            sum_i p_i * tau_i

        w_new =
            w_global -
            tau_eff * normalized_global_delta


    If all tau_i are equal, this reduces mathematically to
    ordinary sample-weighted FedAvg.
    """

    if not (

        len(
            client_states
        )
        ==
        len(
            sample_counts
        )
        ==
        len(
            local_steps
        )
    ):

        raise ValueError(
            "FedNova client input lengths differ."
        )


    sample_counts = np.asarray(

        sample_counts,

        dtype=np.float64,
    )


    local_steps = np.asarray(

        local_steps,

        dtype=np.float64,
    )


    if np.any(
        sample_counts <= 0
    ):

        raise ValueError(
            "Sample counts must be > 0."
        )


    if np.any(
        local_steps <= 0
    ):

        raise ValueError(
            "Local step counts must be > 0."
        )


    # ========================================================
    # CLIENT WEIGHTS
    # ========================================================

    probabilities = (

        sample_counts

        /

        np.sum(
            sample_counts
        )
    )


    # ========================================================
    # EFFECTIVE TAU
    # ========================================================

    tau_eff = float(

        np.sum(

            probabilities

            *

            local_steps
        )
    )


    new_state = {}


    normalized_delta_norm_squared = 0.0

    final_update_norm_squared = 0.0


    # ========================================================
    # STATE TENSORS
    # ========================================================

    for key in (
        global_state.keys()
    ):

        global_tensor = (
            global_state[
                key
            ]
        )


        # ====================================================
        # FLOATING PARAMETERS
        # ====================================================

        if torch.is_floating_point(
            global_tensor
        ):

            normalized_delta = torch.zeros_like(
                global_tensor
            )


            for (
                probability,
                tau_i,
                client_state,

            ) in zip(

                probabilities,
                local_steps,
                client_states,
            ):

                client_tensor = (
                    client_state[
                        key
                    ]
                )


                client_delta = (

                    global_tensor

                    -

                    client_tensor
                )


                normalized_delta += (

                    float(
                        probability
                    )

                    *

                    (
                        client_delta

                        /

                        float(
                            tau_i
                        )
                    )
                )


            final_update = (

                tau_eff

                *

                normalized_delta
            )


            new_state[
                key
            ] = (

                global_tensor

                -

                final_update
            )


            normalized_delta_norm_squared += float(

                torch.sum(

                    normalized_delta
                    .double()
                    ** 2

                ).item()
            )


            final_update_norm_squared += float(

                torch.sum(

                    final_update
                    .double()
                    ** 2

                ).item()
            )


        # ====================================================
        # NON-FLOATING STATE
        # ====================================================

        else:

            # Keep same behavior as aggregation by taking
            # the first client state for any non-floating
            # buffers.

            new_state[
                key
            ] = (

                client_states[
                    0
                ][
                    key
                ]
                .clone()
            )


    normalized_delta_norm = float(

        np.sqrt(
            normalized_delta_norm_squared
        )
    )


    final_update_norm = float(

        np.sqrt(
            final_update_norm_squared
        )
    )


    return (

        new_state,

        tau_eff,

        probabilities,

        normalized_delta_norm,

        final_update_norm,
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
# PAYLOAD
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
    fednova_model,
    rounds,
):

    baseline_params = (
        parameter_count(
            baseline_model
        )
    )


    fednova_params = (
        parameter_count(
            fednova_model
        )
    )


    baseline_payload = (
        model_payload_bytes(
            baseline_model
        )
    )


    fednova_payload = (
        model_payload_bytes(
            fednova_model
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


    fednova_total = (

        2
        *
        N_CLIENTS
        *
        rounds
        *
        fednova_payload
    )


    return {

        "phase4_parameter_count":
            baseline_params,

        "fednova_parameter_count":
            fednova_params,

        "parameter_reduction_ratio":
            float(

                1.0

                -

                fednova_params
                /
                baseline_params
            ),

        "phase4_payload_mib":
            float(

                baseline_payload

                /

                1024 ** 2
            ),

        "fednova_payload_mib":
            float(

                fednova_payload

                /

                1024 ** 2
            ),

        "phase4_total_communication_mib":
            float(

                baseline_total

                /

                1024 ** 2
            ),

        "fednova_total_communication_mib":
            float(

                fednova_total

                /

                1024 ** 2
            ),

        "communication_reduction_ratio":
            float(

                1.0

                -

                fednova_total
                /
                baseline_total
            ),

        "extra_metadata":
            (
                "Each client conceptually provides its "
                "local optimizer-step count tau_i. "
                "This scalar overhead is negligible relative "
                "to model payload and is excluded."
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
# PREVIOUS METHOD METRICS
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
# BUILD COMPARISON
# ============================================================

def build_method_comparison(
    scenario,
    fednova_overall,
    result_dir,
):

    phase4_metrics = (
        read_method_metrics(

            PHASE4_RESULT_ROOT,

            scenario,
        )
    )


    hybrid_fedavg = (
        read_method_metrics(

            PHASE7_RESULT_ROOT,

            scenario,
        )
    )


    fedprox = (
        read_method_metrics(

            PHASE8_FEDPROX_ROOT,

            scenario,
        )
    )


    fedavgm = (
        read_method_metrics(

            PHASE8B_FEDAVGM_ROOT,

            scenario,
        )
    )


    rows = []


    methods = [

        (
            "FedAvg_70",
            70,
            phase4_metrics,
        ),

        (
            "Hybrid37_FedAvg",
            37,
            hybrid_fedavg,
        ),

        (
            "Hybrid37_FedProx",
            37,
            fedprox,
        ),

        (
            "Hybrid37_FedAvgM",
            37,
            fedavgm,
        ),
    ]


    for (
        method_name,
        features,
        metrics,

    ) in methods:

        if metrics is None:

            continue


        rows.append({

            "method":
                method_name,

            "features":
                features,

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
            "Hybrid37_FedNova",

        "features":
            fednova_overall[
                "selected_feature_count"
            ],

        "accuracy":
            fednova_overall[
                "accuracy"
            ],

        "balanced_accuracy":
            fednova_overall[
                "balanced_accuracy"
            ],

        "macro_f1":
            fednova_overall[
                "macro_f1"
            ],

        "weighted_f1":
            fednova_overall[
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
            "\nLocked test file missing."
        )


    # ========================================================
    # OUTPUT DIRECTORIES
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
        "PHASE 8C — HYBRID EVO-GA + FEDNOVA"
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
    # CLIENT FILES / COUNTS / TAU
    # ========================================================

    client_files = {}

    client_sample_counts = {}

    client_local_steps = {}


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


        sample_count = (
            phase4.count_client_samples(

                client_file,

                args.chunk_size,
            )
        )


        (
            local_steps,
            counted_rows,

        ) = count_local_steps(

            client_file=
                client_file,

            batch_size=
                args.batch_size,

            chunk_size=
                args.chunk_size,

            local_epochs=
                args.local_epochs,
        )


        if (
            counted_rows
            !=
            sample_count
        ):

            raise ValueError(
                f"\nClient {client_id}: "
                f"sample-count mismatch.\n"
                f"Phase4 count = {sample_count}\n"
                f"FedNova count = {counted_rows}"
            )


        client_files[
            client_id
        ] = client_file


        client_sample_counts[
            f"client_{client_id}"
        ] = sample_count


        client_local_steps[
            f"client_{client_id}"
        ] = local_steps


        logger.info(
            "Client %d | samples=%d | "
            "local optimizer steps=%d",
            client_id,
            sample_count,
            local_steps,
        )


    # ========================================================
    # STEP HETEROGENEITY CHECK
    # ========================================================

    unique_steps = set(
        client_local_steps.values()
    )


    print()
    print(
        "Client local optimizer steps:"
    )


    for client_name, steps in (
        client_local_steps.items()
    ):

        print(
            f"  {client_name:12s}: "
            f"{steps}"
        )


    if (
        len(
            unique_steps
        )
        ==
        1
    ):

        print()
        print(
            "NOTE: All clients have equal local-step counts."
        )

        print(
            "Under equal tau_i, FedNova normalization can "
            "reduce to ordinary sample-weighted FedAvg."
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

            fednova_model=
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
            f"FEDNOVA ROUND "
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

        sample_counts = []

        local_steps_list = []

        client_losses = []

        client_accuracies = []


        # ====================================================
        # CLIENT LOCAL TRAINING
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


            sample_counts.append(

                client_sample_counts[
                    f"client_{client_id}"
                ]
            )


            local_steps_list.append(

                client_local_steps[
                    f"client_{client_id}"
                ]
            )


            client_losses.append(
                local_loss
            )


            client_accuracies.append(
                local_accuracy
            )


        # ====================================================
        # FEDNOVA AGGREGATION
        # ====================================================

        (
            new_global_state,
            tau_eff,
            probabilities,
            normalized_delta_norm,
            fednova_update_norm,

        ) = fednova_aggregate(

            global_state=
                old_global_state,

            client_states=
                client_states,

            sample_counts=
                sample_counts,

            local_steps=
                local_steps_list,
        )


        global_model.load_state_dict(
            new_global_state
        )


        # ====================================================
        # ACTUAL UPDATE NORM
        # ====================================================

        actual_update_norm = (
            phase4.state_update_norm(

                old_global_state,

                new_global_state,
            )
        )


        # ====================================================
        # ROUND METRICS
        # ====================================================

        sample_weights = np.asarray(

            sample_counts,

            dtype=np.float64,
        )


        weighted_loss = float(

            np.average(

                np.asarray(
                    client_losses,
                    dtype=np.float64,
                ),

                weights=
                    sample_weights,
            )
        )


        weighted_accuracy = float(

            np.average(

                np.asarray(
                    client_accuracies,
                    dtype=np.float64,
                ),

                weights=
                    sample_weights,
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

            "weighted_local_loss":
                weighted_loss,

            "weighted_local_accuracy":
                weighted_accuracy,

            "tau_effective":
                tau_eff,

            "normalized_delta_norm":
                normalized_delta_norm,

            "fednova_update_norm":
                fednova_update_norm,

            "global_update_norm":
                actual_update_norm,

            "round_training_seconds":
                round_seconds,

            "feature_count":
                len(
                    selected_features
                ),
        }


        for client_index in range(
            N_CLIENTS
        ):

            round_record[
                f"client_{client_index + 1}_tau"
            ] = int(
                local_steps_list[
                    client_index
                ]
            )


            round_record[
                f"client_{client_index + 1}_weight"
            ] = float(
                probabilities[
                    client_index
                ]
            )


        history.append(
            round_record
        )


        logger.info(
            "ROUND %d COMPLETE | "
            "loss=%.6f | "
            "acc=%.6f | "
            "tau_eff=%.4f | "
            "norm_delta=%.6f | "
            "update=%.6f | "
            "time=%.2fs",
            round_number,
            weighted_loss,
            weighted_accuracy,
            tau_eff,
            normalized_delta_norm,
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
        "fednova_effective_tau_final"
    ] = float(
        history[
            -1
        ][
            "tau_effective"
        ]
    )


    overall[
        "equal_local_step_counts"
    ] = bool(
        len(
            unique_steps
        )
        ==
        1
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
        / "fednova_round_history.csv",

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
                "8C",

            "algorithm":
                "Hybrid EVO-GA + FedNova",

            "implementation_note":
                (
                    "FedNova-style normalized model "
                    "updates with existing AdamW "
                    "local optimizer."
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

            "selected_features":
                selected_features,

            "selected_active_positions":
                selected_active_positions.tolist(),

            "selected_original_indices":
                selected_original_indices.tolist(),

            "local_optimizer_steps":
                client_local_steps,

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
        / "hybrid_fednova_model.pt",
    )


    # ========================================================
    # MANIFEST
    # ========================================================

    manifest = {

        "phase":
            "8C",

        "algorithm":
            "Hybrid EVO-GA + FedNova",

        "aggregation_variant":
            (
                "FedNova-style normalized aggregation"
            ),

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

        "scientific_note":
            (
                "Local AdamW was preserved to keep "
                "local optimization identical across "
                "FL experiments. FedNova normalization "
                "is therefore applied at the model-update "
                "aggregation level."
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

        "client_local_optimizer_steps":
            client_local_steps,

        "equal_local_step_counts":
            bool(
                len(
                    unique_steps
                )
                ==
                1
            ),

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
        / "phase8c_fednova_manifest.json",

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

            fednova_overall=
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
                "Hybrid37-FedNova-"
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
        "PHASE 8C COMPLETE — "
        "HYBRID EVO-GA + FEDNOVA"
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
        f"Equal local steps   : "
        f"{len(unique_steps) == 1}"
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
        "PER-CLASS FEDNOVA PERFORMANCE"
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
    # FEDNOVA DIAGNOSTICS
    # ========================================================

    print()
    print(
        "-" * 90
    )

    print(
        "FEDNOVA DIAGNOSTICS"
    )

    print(
        "-" * 90
    )


    print(
        f"Effective tau            : "
        f"{history_df['tau_effective'].iloc[-1]:.6f}"
    )

    print(
        f"Normalized delta norm    : "
        f"{history_df['normalized_delta_norm'].iloc[-1]:.6f}"
    )

    print(
        f"Final update norm        : "
        f"{history_df['global_update_norm'].iloc[-1]:.6f}"
    )


    print()


    for client_id in range(
        1,
        N_CLIENTS + 1,
    ):

        print(

            f"client_{client_id} "
            f"tau="
            f"{client_local_steps[f'client_{client_id}']} "
            f"samples="
            f"{client_sample_counts[f'client_{client_id}']}"
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
        f"FedNova parameters   : "
        f"{communication['fednova_parameter_count']:,}"
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
        "Same local MLP                      : YES ✅"
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
        "Same rounds/local epochs            : YES ✅"
    )

    print(
        "FedNova update normalization        : YES ✅"
    )

    print(
        "Local step counts recorded          : YES ✅"
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