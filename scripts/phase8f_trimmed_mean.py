"""
Phase 8F - Hybrid EVO-GA + Coordinate-wise Trimmed Mean
=======================================================

Purpose
-------

Evaluate Trimmed Mean robust aggregation under the same severe
non-IID configuration used by:

    Hybrid37 + FedAvg
    Hybrid37 + FedProx
    Hybrid37 + FedAvgM
    Hybrid37 + FedNova
    Hybrid37 + SCAFFOLD
    Hybrid37 + Coordinate-wise Median

Frozen feature subset:

    70 -> 37 Hybrid EVO-GA features


Trimmed Mean
------------

For each parameter coordinate:

    1. Collect values from all N clients
    2. Sort values
    3. Remove the lowest trim_count values
    4. Remove the highest trim_count values
    5. Average the remaining values

For 5 clients and trim_count = 1:

    [x1, x2, x3, x4, x5]
             sort
              ↓
    [min, v2, v3, v4, max]
              ↓
        remove min/max
              ↓
         mean(v2,v3,v4)

This robust aggregation rule reduces the influence of extreme
client updates.

IMPORTANT
---------

With N_CLIENTS = 5:

    trim_count = 1

leaves exactly:

    5 - 2 = 3

client values per parameter coordinate.

Do NOT use trim_count >= 2 because that would leave at most one
client value and would not provide a meaningful trimmed mean.


Scientific control
------------------

Keep exactly the same:

    - Hybrid 37-feature subset
    - same 5 clients
    - same non-IID client files
    - same 10 rounds
    - same 1 local epoch
    - same MLP
    - same projected initialization
    - same scaler
    - same class weighting
    - same AdamW local optimizer
    - same learning rate
    - same weight decay
    - same batch size

Only change:

    Server aggregation:
        sample-weighted FedAvg
            ->
        coordinate-wise Trimmed Mean


Run
---

    python scripts/phase8f_trimmed_mean.py --scenario non_iid --trim-count 1


Outputs
-------

    results/trimmed_mean/non_iid/

    artifacts/trimmed_mean/non_iid/
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
# GENERAL CONFIG
# ============================================================

SEED = phase4.SEED

N_CLIENTS = phase4.N_CLIENTS

EXPECTED_PHASE4_FEATURES = 70


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


DEFAULT_TRIM_COUNT = 1


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
    / "trimmed_mean"
)


ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "trimmed_mean"
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


PHASE8E_MEDIAN_ROOT = (
    PROJECT_ROOT
    / "results"
    / "median"
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
    "phase8f_trimmed_mean"
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
# HASH
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
            "\nHybrid feature mask missing:\n"
            f"{mask_file}"
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
    # VERIFY FEATURE SPACE
    # ========================================================

    if (
        len(
            active_features
        )
        !=
        EXPECTED_PHASE4_FEATURES
    ):

        raise ValueError(
            "\nExpected exactly 70 Phase-4 active features."
        )


    if (
        len(
            mask_df
        )
        !=
        EXPECTED_PHASE4_FEATURES
    ):

        raise ValueError(
            "\nHybrid mask must contain exactly 70 rows."
        )


    mask_feature_set = set(
        mask_df[
            "feature_name"
        ]
    )


    active_feature_set = set(
        active_features
    )


    if (
        mask_feature_set
        !=
        active_feature_set
    ):

        raise ValueError(
            "\nHybrid mask feature schema mismatch."
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
            "Hybrid mask selected column must contain 0/1."
        )


    # ========================================================
    # PRESERVE ACTIVE FEATURE ORDER
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
    # MAP ACTIVE -> ORIGINAL SCALER INDEX
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
            "\nSelected feature index mapping failed."
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
    Initialize the full 70-feature reference model using
    the Phase-4 seed and project the first layer onto the
    frozen 37-feature subset.
    """

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


    first_layer_key = (
        "network.0.weight"
    )


    positions = [

        int(
            position
        )

        for position
        in selected_active_positions
    ]


    for key in (
        selected_state.keys()
    ):

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


            if (
                projected.shape
                !=
                selected_state[
                    key
                ].shape
            ):

                raise ValueError(
                    "\nProjected first-layer shape mismatch."
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
# VALIDATE TRIM COUNT
# ============================================================

def validate_trim_count(
    trim_count,
    n_clients,
):

    if trim_count < 0:

        raise ValueError(
            "trim-count must be >= 0."
        )


    remaining = (
        n_clients
        -
        2 * trim_count
    )


    if remaining <= 0:

        raise ValueError(
            "\nInvalid trim-count.\n"
            f"Clients={n_clients}\n"
            f"trim_count={trim_count}\n"
            f"Remaining values={remaining}"
        )


    if (
        n_clients == 5
        and
        trim_count > 1
    ):

        raise ValueError(
            "\nFor this 5-client experiment use "
            "--trim-count 1.\n"
            "A larger trim count is not suitable "
            "for the current comparison."
        )


    return remaining


# ============================================================
# COORDINATE-WISE TRIMMED MEAN
# ============================================================

def coordinate_wise_trimmed_mean(
    client_states,
    trim_count,
):

    """
    For each floating-point parameter coordinate:

        values =
            stack(client_parameter_values)

        sorted_values =
            sort(values over client dimension)

        trimmed =
            sorted_values[
                trim_count :
                N - trim_count
            ]

        aggregate =
            mean(trimmed)

    Example for 5 clients and trim_count=1:

        [1, 5, 2, 100, 4]
             ↓ sort
        [1, 2, 4, 5, 100]
             ↓ trim
        [2, 4, 5]
             ↓ mean
        3.6667
    """

    if not client_states:

        raise ValueError(
            "No client states supplied."
        )


    n_clients = len(
        client_states
    )


    remaining = validate_trim_count(

        trim_count=
            trim_count,

        n_clients=
            n_clients,
    )


    aggregated_state = {}


    for key in (
        client_states[0].keys()
    ):

        reference_tensor = (
            client_states[
                0
            ][
                key
            ]
        )


        # ====================================================
        # FLOATING MODEL STATE
        # ====================================================

        if torch.is_floating_point(
            reference_tensor
        ):

            stacked = torch.stack(

                [

                    client_state[
                        key
                    ].float()

                    for client_state
                    in client_states
                ],

                dim=0,
            )


            sorted_values = torch.sort(

                stacked,

                dim=0,
            ).values


            if trim_count == 0:

                trimmed = (
                    sorted_values
                )

            else:

                trimmed = sorted_values[
                    trim_count:
                    n_clients
                    -
                    trim_count
                ]


            if (
                trimmed.shape[
                    0
                ]
                !=
                remaining
            ):

                raise RuntimeError(
                    "\nUnexpected number of retained "
                    "client values after trimming."
                )


            trimmed_mean = torch.mean(

                trimmed,

                dim=0,
            )


            aggregated_state[
                key
            ] = trimmed_mean.to(

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


            if not torch.is_floating_point(
                old_tensor
            ):

                continue


            client_tensor = (
                client_state[
                    key
                ]
            )


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
    trimmed_model,
    rounds,
):

    baseline_params = (
        parameter_count(
            baseline_model
        )
    )


    trimmed_params = (
        parameter_count(
            trimmed_model
        )
    )


    baseline_payload = (
        model_payload_bytes(
            baseline_model
        )
    )


    trimmed_payload = (
        model_payload_bytes(
            trimmed_model
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


    trimmed_total = (
        2
        *
        N_CLIENTS
        *
        rounds
        *
        trimmed_payload
    )


    return {

        "phase4_parameter_count":
            baseline_params,

        "trimmed_mean_parameter_count":
            trimmed_params,

        "parameter_reduction_ratio":
            float(
                1.0
                -
                trimmed_params
                /
                baseline_params
            ),

        "phase4_model_payload_mib":
            float(
                baseline_payload
                /
                1024 ** 2
            ),

        "trimmed_mean_model_payload_mib":
            float(
                trimmed_payload
                /
                1024 ** 2
            ),

        "phase4_total_communication_mib":
            float(
                baseline_total
                /
                1024 ** 2
            ),

        "trimmed_mean_total_communication_mib":
            float(
                trimmed_total
                /
                1024 ** 2
            ),

        "communication_reduction_ratio":
            float(
                1.0
                -
                trimmed_total
                /
                baseline_total
            ),

        "extra_client_communication":
            0,

        "note":
            (
                "Trimmed Mean changes only server aggregation. "
                "Model payload is identical to other 37-feature "
                "methods."
            ),
    }


# ============================================================
# BUILD PER-CLASS DATAFRAME
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
# READ PREVIOUS METRICS
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
# METHOD COMPARISON
# ============================================================

def build_method_comparison(
    scenario,
    trimmed_overall,
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

        (
            "Hybrid37_Median",
            37,
            PHASE8E_MEDIAN_ROOT,
        ),
    ]


    rows = []


    for (
        method_name,
        feature_count,
        result_root,

    ) in previous_methods:

        metrics = (
            read_method_metrics(

                root=
                    result_root,

                scenario=
                    scenario,
            )
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
            "Hybrid37_TrimmedMean",

        "features":
            trimmed_overall[
                "selected_feature_count"
            ],

        "accuracy":
            trimmed_overall[
                "accuracy"
            ],

        "balanced_accuracy":
            trimmed_overall[
                "balanced_accuracy"
            ],

        "macro_f1":
            trimmed_overall[
                "macro_f1"
            ],

        "weighted_f1":
            trimmed_overall[
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

        "--trim-count",

        type=int,

        default=
            DEFAULT_TRIM_COUNT,
    )


    parser.add_argument(

        "--mask-file",

        type=Path,

        default=
            DEFAULT_MASK_FILE,
    )


    args = parser.parse_args()


    # ========================================================
    # VALIDATE TRIM
    # ========================================================

    retained_clients = (
        validate_trim_count(

            trim_count=
                args.trim_count,

            n_clients=
                N_CLIENTS,
        )
    )


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
            "\nLocked test file missing."
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
    # PREPROCESSING
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
        "PHASE 8F — HYBRID EVO-GA + TRIMMED MEAN"
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
        f"Trim count          : "
        f"{args.trim_count}"
    )

    print(
        f"Values retained     : "
        f"{retained_clients}/"
        f"{N_CLIENTS} per coordinate"
    )

    print(
        "Aggregation         : coordinate-wise trimmed mean"
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

            trimmed_model=
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
            f"TRIMMED MEAN ROUND "
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
        # CLIENT TRAINING
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
        # UPDATE NORM DIAGNOSTICS
        # ====================================================

        client_update_norms = (
            calculate_client_update_norms(

                old_global_state=
                    old_global_state,

                client_states=
                    client_states,
            )
        )


        # ====================================================
        # TRIMMED MEAN AGGREGATION
        # ====================================================

        aggregated_state = (
            coordinate_wise_trimmed_mean(

                client_states=
                    client_states,

                trim_count=
                    args.trim_count,
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
        # REPORTING METRICS
        #
        # Retain Phase-4-compatible column names so that
        # phase4.generate_graphs() works correctly.
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


        weighted_local_loss = float(

            np.average(

                np.asarray(
                    client_losses,
                    dtype=np.float64,
                ),

                weights=
                    sample_weights,
            )
        )


        weighted_local_accuracy = float(

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


        history_row = {

            "round":
                round_number,

            # =================================================
            # REQUIRED BY PHASE-4 GRAPH CODE
            # =================================================

            "weighted_local_loss":
                weighted_local_loss,

            "weighted_local_accuracy":
                weighted_local_accuracy,

            # =================================================
            # TRIMMED-MEAN-SPECIFIC DIAGNOSTICS
            # =================================================

            "sample_weighted_local_loss":
                weighted_local_loss,

            "sample_weighted_local_accuracy":
                weighted_local_accuracy,

            "equal_client_local_loss":
                equal_client_loss,

            "equal_client_local_accuracy":
                equal_client_accuracy,

            "global_update_norm":
                global_update_norm,

            "mean_client_update_norm":
                float(
                    np.mean(
                        client_update_norms
                    )
                ),

            "median_client_update_norm":
                float(
                    np.median(
                        client_update_norms
                    )
                ),

            "max_client_update_norm":
                float(
                    np.max(
                        client_update_norms
                    )
                ),

            "min_client_update_norm":
                float(
                    np.min(
                        client_update_norms
                    )
                ),

            "trim_count":
                int(
                    args.trim_count
                ),

            "retained_client_values":
                int(
                    retained_clients
                ),

            "round_training_seconds":
                float(
                    round_seconds
                ),

            "feature_count":
                int(
                    len(
                        selected_features
                    )
                ),
        }


        for index, update_norm in enumerate(
            client_update_norms,
            start=1,
        ):

            history_row[
                f"client_{index}_update_norm"
            ] = float(
                update_norm
            )


        history.append(
            history_row
        )


        logger.info(
            "ROUND %d COMPLETE | "
            "loss=%.6f | "
            "acc=%.6f | "
            "median_client_update=%.6f | "
            "max_client_update=%.6f | "
            "global_update=%.6f | "
            "time=%.2fs",
            round_number,
            weighted_local_loss,
            weighted_local_accuracy,
            float(
                np.median(
                    client_update_norms
                )
            ),
            float(
                np.max(
                    client_update_norms
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
        "aggregation"
    ] = (
        "coordinate-wise trimmed mean"
    )


    overall[
        "trim_count"
    ] = int(
        args.trim_count
    )


    overall[
        "retained_client_values_per_coordinate"
    ] = int(
        retained_clients
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
        / "trimmed_mean_round_history.csv",

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
                "8F",

            "algorithm":
                "Hybrid EVO-GA + Trimmed Mean",

            "aggregation":
                "coordinate-wise trimmed mean",

            "trim_count":
                int(
                    args.trim_count
                ),

            "retained_client_values":
                int(
                    retained_clients
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
        / "hybrid_trimmed_mean_model.pt",
    )


    # ========================================================
    # MANIFEST
    # ========================================================

    manifest = {

        "phase":
            "8F",

        "algorithm":
            "Hybrid EVO-GA + Trimmed Mean",

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
            "coordinate-wise trimmed mean",

        "trim_count":
            args.trim_count,

        "retained_client_values_per_coordinate":
            retained_clients,

        "sample_weighted_aggregation":
            False,

        "scientific_note":
            (
                "For each model coordinate, the smallest "
                "and largest client values are discarded "
                "when trim_count=1, and the remaining "
                "three client values are averaged."
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
        / "phase8f_trimmed_mean_manifest.json",

        "w",
        encoding="utf-8",

    ) as file:

        json.dump(
            manifest,
            file,
            indent=4,
        )


    # ========================================================
    # COMPARISON TABLE
    # ========================================================

    comparison_df = (
        build_method_comparison(

            scenario=
                args.scenario,

            trimmed_overall=
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
                "Hybrid37-TrimmedMean-"
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
        "PHASE 8F COMPLETE — "
        "HYBRID EVO-GA + TRIMMED MEAN"
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
        f"Trim count          : "
        f"{args.trim_count}"
    )

    print(
        f"Values retained     : "
        f"{retained_clients}/"
        f"{N_CLIENTS}"
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
        "PER-CLASS TRIMMED MEAN PERFORMANCE"
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
    # ROBUST AGGREGATION DIAGNOSTICS
    # ========================================================

    print()
    print(
        "-" * 90
    )

    print(
        "TRIMMED MEAN AGGREGATION DIAGNOSTICS"
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
        f"Phase-4 parameters      : "
        f"{communication['phase4_parameter_count']:,}"
    )

    print(
        f"TrimmedMean parameters  : "
        f"{communication['trimmed_mean_parameter_count']:,}"
    )

    print(
        f"Parameter reduction     : "
        f"{communication['parameter_reduction_ratio'] * 100:.2f}%"
    )

    print(
        f"Communication reduction : "
        f"{communication['communication_reduction_ratio'] * 100:.2f}%"
    )

    print(
        "Extra client communication: 0 bytes"
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
        "Coordinate-wise Trimmed Mean        : YES ✅"
    )

    print(
        f"Trim count                          : "
        f"{args.trim_count} ✅"
    )

    print(
        f"Values retained / coordinate        : "
        f"{retained_clients} ✅"
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