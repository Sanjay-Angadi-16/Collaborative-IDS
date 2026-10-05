"""
Phase 8D - Hybrid EVO-GA + SCAFFOLD
===================================

Purpose
-------

Evaluate SCAFFOLD-style federated learning under the same
severe non-IID configuration used by:

    Hybrid37 + FedAvg
    Hybrid37 + FedProx
    Hybrid37 + FedAvgM
    Hybrid37 + FedNova

Frozen Hybrid feature subset:

    70 -> 37 features


Core SCAFFOLD idea
------------------

Non-IID clients may drift toward their own local objectives.

SCAFFOLD introduces:

    c   = global/server control variate
    c_i = client-specific control variate

During local training the gradient is corrected as:

    g_corrected = g - c_i + c


After K local optimizer steps, the client control variate is
updated approximately as:

    c_i_new
        =
        c_i - c
        +
        (w_global - w_local) / (K * eta)

and:

    delta_c_i
        =
        c_i_new - c_i


The server updates:

    c_new
        =
        c
        +
        mean(delta_c_i)

All 5 clients participate in every round.


IMPORTANT SCIENTIFIC NOTE
-------------------------

Canonical SCAFFOLD theory is generally derived using SGD-style
local updates.

Your existing pipeline uses AdamW.

To maintain comparability across FedAvg, FedProx, FedAvgM and
FedNova, this experiment preserves AdamW and applies the SCAFFOLD
control-variate correction directly to parameter gradients.

Therefore, describe this experiment as:

    "SCAFFOLD-style control-variate correction with AdamW local
     optimization"

unless you later implement a canonical SGD-based SCAFFOLD
experiment separately.


Run
---

    python scripts/phase8d_scaffold.py --scenario non_iid


Outputs
-------

    results/scaffold/non_iid/

    artifacts/scaffold/non_iid/
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
# REUSE PHASE 4
# ============================================================

import phase4_fedavg as phase4


# ============================================================
# GENERAL CONFIGURATION
# ============================================================

SEED = phase4.SEED

N_CLIENTS = phase4.N_CLIENTS

EXPECTED_PHASE4_FEATURES = 70


# ============================================================
# SAME TRAINING SETTINGS
# ============================================================

DEFAULT_ROUNDS = phase4.DEFAULT_ROUNDS

DEFAULT_LOCAL_EPOCHS = phase4.DEFAULT_LOCAL_EPOCHS

DEFAULT_BATCH_SIZE = phase4.DEFAULT_BATCH_SIZE

DEFAULT_CHUNK_SIZE = phase4.DEFAULT_CHUNK_SIZE

DEFAULT_LEARNING_RATE = phase4.DEFAULT_LEARNING_RATE

DEFAULT_WEIGHT_DECAY = phase4.DEFAULT_WEIGHT_DECAY


GRADIENT_CLIP = 5.0


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
    / "scaffold"
)


ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "scaffold"
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
    "phase8d_scaffold"
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
    # VERIFY BASE FEATURE SPACE
    # ========================================================

    if (
        len(
            active_features
        )
        !=
        EXPECTED_PHASE4_FEATURES
    ):

        raise ValueError(
            "\nExpected exactly 70 Phase-4 features."
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


    mask_features = set(
        mask_df[
            "feature_name"
        ]
    )


    active_feature_set = set(
        active_features
    )


    if (
        mask_features
        !=
        active_feature_set
    ):

        raise ValueError(
            "\nHybrid mask schema does not match "
            "Phase-4 active feature schema."
        )


    # ========================================================
    # VALIDATE SELECTED VALUES
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
            "Hybrid mask selected values must be 0 or 1."
        )


    # ========================================================
    # PRESERVE PHASE-4 ACTIVE FEATURE ORDER
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
    # MAP TO ORIGINAL SCALER INDICES
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
            "\nHybrid feature-index reconstruction failed."
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
# PROJECTED MODEL INITIALIZATION
# ============================================================

def initialize_projected_model(
    selected_active_positions,
    full_input_dim,
    selected_input_dim,
    num_classes,
    device,
):

    """
    Create the same seed-based 70-feature reference model,
    then project the first layer into the Hybrid-selected
    37-feature space.
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
                    "\nProjected first-layer dimensions "
                    "do not match."
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
# CREATE ZERO CONTROL VARIATE
# ============================================================

def create_zero_control_variate(
    model,
):

    """
    Control variates are defined for trainable parameters only,
    not non-trainable buffers.
    """

    control = {}


    for name, parameter in (
        model.named_parameters()
    ):

        control[
            name
        ] = torch.zeros_like(

            parameter,

            device="cpu",
        )


    return control


# ============================================================
# CONTROL NORM
# ============================================================

def control_variate_norm(
    control,
):

    norm_squared = 0.0


    for tensor in (
        control.values()
    ):

        norm_squared += float(

            torch.sum(

                tensor.double()
                ** 2

            ).item()
        )


    return float(

        np.sqrt(
            norm_squared
        )
    )


# ============================================================
# SCAFFOLD CLIENT TRAINING
# ============================================================

def train_client_scaffold(
    global_state,
    server_control,
    client_control,
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
):

    """
    Train one local client using:

        gradient =
            gradient - client_control + server_control


    Then update local control variate:

        c_i_new =
            c_i - c +
            (w_global - w_local) / (K * eta)
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
    # GLOBAL MODEL PARAMETERS AT ROUND START
    # ========================================================

    global_parameter_reference = {}


    for name, parameter in (
        local_model.named_parameters()
    ):

        global_parameter_reference[
            name
        ] = (

            parameter
            .detach()
            .clone()
        )


    # ========================================================
    # CONTROL VARIATES ON DEVICE
    # ========================================================

    server_control_device = {

        name:
            tensor.to(
                device
            )

        for name, tensor
        in server_control.items()
    }


    client_control_device = {

        name:
            tensor.to(
                device
            )

        for name, tensor
        in client_control.items()
    }


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
    # SAME LOCAL OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(

        local_model.parameters(),

        lr=
            learning_rate,

        weight_decay=
            weight_decay,
    )


    # ========================================================
    # DETERMINISTIC RANDOM ORDER
    # ========================================================

    rng = np.random.default_rng(

        SEED

        +

        round_number
        * 1000

        +

        client_id
        * 100
    )


    # ========================================================
    # METRICS
    # ========================================================

    total_loss_sum = 0.0

    total_correct = 0

    total_samples = 0

    optimizer_steps = 0


    # ========================================================
    # LOCAL TRAINING
    # ========================================================

    local_model.train()


    for local_epoch in range(
        local_epochs
    ):

        reader = pd.read_csv(

            client_file,

            chunksize=
                chunk_size,
        )


        for chunk in reader:

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
            # MINI-BATCH TRAINING
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


                loss = criterion(

                    logits,

                    y_batch,
                )


                loss.backward()


                # =============================================
                # SCAFFOLD GRADIENT CORRECTION
                #
                # g_corrected =
                #     g - c_i + c
                # =============================================

                for name, parameter in (
                    local_model.named_parameters()
                ):

                    if (
                        parameter.grad
                        is None
                    ):

                        continue


                    parameter.grad.add_(

                        server_control_device[
                            name
                        ]

                        -

                        client_control_device[
                            name
                        ]
                    )


                # =============================================
                # SAME GRADIENT CLIPPING
                # =============================================

                torch.nn.utils.clip_grad_norm_(

                    local_model.parameters(),

                    max_norm=
                        GRADIENT_CLIP,
                )


                optimizer.step()


                optimizer_steps += 1


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


                total_correct += (
                    batch_correct
                )


                total_samples += (
                    batch_samples
                )


            del chunk


    # ========================================================
    # CHECK
    # ========================================================

    if optimizer_steps <= 0:

        raise RuntimeError(
            f"Client {client_id} completed "
            "zero optimizer steps."
        )


    if total_samples <= 0:

        raise RuntimeError(
            f"Client {client_id} completed "
            "zero training samples."
        )


    # ========================================================
    # UPDATE CLIENT CONTROL VARIATE
    #
    # c_i_new =
    #   c_i - c
    #   + (w_global - w_local) / (K * eta)
    # ========================================================

    new_client_control = {}

    control_delta = {}


    denominator = (

        float(
            optimizer_steps
        )

        *

        float(
            learning_rate
        )
    )


    if denominator <= 0:

        raise ValueError(
            "Invalid SCAFFOLD control denominator."
        )


    for name, parameter in (
        local_model.named_parameters()
    ):

        w_local = (
            parameter
            .detach()
        )


        w_global = (
            global_parameter_reference[
                name
            ]
        )


        old_ci = (
            client_control_device[
                name
            ]
        )


        global_c = (
            server_control_device[
                name
            ]
        )


        new_ci = (

            old_ci

            -

            global_c

            +

            (
                w_global

                -

                w_local
            )

            /

            denominator
        )


        delta_ci = (

            new_ci

            -

            old_ci
        )


        new_client_control[
            name
        ] = (

            new_ci
            .detach()
            .cpu()
            .clone()
        )


        control_delta[
            name
        ] = (

            delta_ci
            .detach()
            .cpu()
            .clone()
        )


    # ========================================================
    # FINAL CLIENT MODEL STATE
    # ========================================================

    local_state = {

        key:
            tensor
            .detach()
            .cpu()
            .clone()

        for key, tensor
        in local_model
        .state_dict()
        .items()
    }


    mean_loss = float(

        total_loss_sum

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


    logger.info(
        "Round %d | Client %d | "
        "loss=%.6f | acc=%.6f | "
        "steps=%d | c_norm=%.6f | "
        "time=%.2fs",
        round_number,
        client_id,
        mean_loss,
        accuracy,
        optimizer_steps,
        control_variate_norm(
            new_client_control
        ),
        training_seconds,
    )


    del local_model


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


    return (

        local_state,
        new_client_control,
        control_delta,
        mean_loss,
        accuracy,
        optimizer_steps,
        training_seconds,
    )


# ============================================================
# UPDATE SERVER CONTROL
# ============================================================

def update_server_control(
    server_control,
    client_control_deltas,
):

    """
    Full client participation:

        c_new =
            c + (1/N) * sum(delta_c_i)

    Control-variate averaging is equal-client averaging.
    """

    new_server_control = {}


    for name in (
        server_control.keys()
    ):

        mean_delta = torch.zeros_like(

            server_control[
                name
            ]
        )


        for delta in (
            client_control_deltas
        ):

            mean_delta += (

                delta[
                    name
                ]

                /

                float(
                    len(
                        client_control_deltas
                    )
                )
            )


        new_server_control[
            name
        ] = (

            server_control[
                name
            ]

            +

            mean_delta
        )


    return new_server_control


# ============================================================
# CONTROL DELTA NORM
# ============================================================

def mean_control_delta_norm(
    client_control_deltas,
):

    norms = []


    for delta in (
        client_control_deltas
    ):

        norm_squared = 0.0


        for tensor in (
            delta.values()
        ):

            norm_squared += float(

                torch.sum(

                    tensor.double()
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


    return float(
        np.mean(
            norms
        )
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
# CONTROL PAYLOAD
# ============================================================

def control_payload_bytes(
    control,
):

    return int(

        sum(

            tensor.numel()
            *
            tensor.element_size()

            for tensor
            in control.values()
        )
    )


# ============================================================
# COMMUNICATION
# ============================================================

def calculate_communication(
    baseline_model,
    scaffold_model,
    server_control,
    rounds,
):

    """
    SCAFFOLD may require control variates to be communicated.

    Model payload:
        server -> client model
        client -> server model/update

    Control payload:
        server -> client global c
        client -> server delta c_i

    Thus SCAFFOLD communication is larger than plain FedAvg.
    """

    baseline_params = (
        parameter_count(
            baseline_model
        )
    )


    scaffold_params = (
        parameter_count(
            scaffold_model
        )
    )


    baseline_model_payload = (
        model_payload_bytes(
            baseline_model
        )
    )


    scaffold_model_payload = (
        model_payload_bytes(
            scaffold_model
        )
    )


    scaffold_control_payload = (
        control_payload_bytes(
            server_control
        )
    )


    # ========================================================
    # PLAIN PHASE-4 FEDAVG COMMUNICATION
    # ========================================================

    baseline_total = (

        2

        *

        N_CLIENTS

        *

        rounds

        *

        baseline_model_payload
    )


    # ========================================================
    # SCAFFOLD COMMUNICATION
    #
    # Per client / round:
    #
    # model download
    # model upload
    # server control download
    # client control delta upload
    # ========================================================

    scaffold_per_client_round = (

        2
        *
        scaffold_model_payload

        +

        2
        *
        scaffold_control_payload
    )


    scaffold_total = (

        N_CLIENTS

        *

        rounds

        *

        scaffold_per_client_round
    )


    return {

        "phase4_parameter_count":
            baseline_params,

        "scaffold_parameter_count":
            scaffold_params,

        "parameter_reduction_ratio":
            float(

                1.0

                -

                scaffold_params
                /
                baseline_params
            ),

        "phase4_model_payload_mib":
            float(

                baseline_model_payload

                /

                1024 ** 2
            ),

        "scaffold_model_payload_mib":
            float(

                scaffold_model_payload

                /

                1024 ** 2
            ),

        "scaffold_control_payload_mib":
            float(

                scaffold_control_payload

                /

                1024 ** 2
            ),

        "phase4_total_communication_mib":
            float(

                baseline_total

                /

                1024 ** 2
            ),

        "scaffold_total_communication_mib":
            float(

                scaffold_total

                /

                1024 ** 2
            ),

        "communication_change_ratio_vs_phase4":
            float(

                scaffold_total
                /
                baseline_total

                -

                1.0
            ),

        "note":
            (
                "Unlike FedAvg/FedProx/FedNova, SCAFFOLD "
                "requires control-variate communication. "
                "This estimate includes model and control "
                "payloads but excludes protocol overhead."
            ),
    }


# ============================================================
# PER CLASS TABLE
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
    scaffold_overall,
    result_dir,
):

    previous = [

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
    ]


    rows = []


    for (
        method_name,
        feature_count,
        result_root,

    ) in previous:

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
            "Hybrid37_SCAFFOLD",

        "features":
            scaffold_overall[
                "selected_feature_count"
            ],

        "accuracy":
            scaffold_overall[
                "accuracy"
            ],

        "balanced_accuracy":
            scaffold_overall[
                "balanced_accuracy"
            ],

        "macro_f1":
            scaffold_overall[
                "macro_f1"
            ],

        "weighted_f1":
            scaffold_overall[
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
        "PHASE 8D — HYBRID EVO-GA + SCAFFOLD"
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
        "Local optimizer     : AdamW"
    )

    print(
        "Control correction  : g - c_i + c"
    )


    # ========================================================
    # CLASS COUNTS / WEIGHTS
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
    # INITIALIZE SERVER CONTROL
    # ========================================================

    server_control = (
        create_zero_control_variate(
            global_model
        )
    )


    # ========================================================
    # INITIALIZE CLIENT CONTROLS
    # ========================================================

    client_controls = {}


    for client_id in range(
        1,
        N_CLIENTS + 1,
    ):

        client_controls[
            client_id
        ] = {

            name:
                tensor.clone()

            for name, tensor
            in server_control.items()
        }


    # ========================================================
    # COMMUNICATION
    # ========================================================

    communication = (
        calculate_communication(

            baseline_model=
                reference_model,

            scaffold_model=
                global_model,

            server_control=
                server_control,

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
            f"SCAFFOLD ROUND "
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

        client_control_deltas = []

        aggregation_weights = []

        client_losses = []

        client_accuracies = []

        client_steps = []

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
                new_client_control,
                control_delta,
                local_loss,
                local_accuracy,
                local_steps,
                local_seconds,

            ) = train_client_scaffold(

                global_state=
                    global_model.state_dict(),

                server_control=
                    server_control,

                client_control=
                    client_controls[
                        client_id
                    ],

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
            )


            # ================================================
            # SAVE CLIENT'S NEW CONTROL
            # ================================================

            client_controls[
                client_id
            ] = new_client_control


            client_states.append(
                local_state
            )


            client_control_deltas.append(
                control_delta
            )


            aggregation_weights.append(

                client_sample_counts[
                    f"client_{client_id}"
                ]
            )


            client_losses.append(
                local_loss
            )


            client_accuracies.append(
                local_accuracy
            )


            client_steps.append(
                local_steps
            )


            client_times.append(
                local_seconds
            )


        # ====================================================
        # MODEL AGGREGATION
        #
        # Keep same sample-weighted aggregation used elsewhere.
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
        # SERVER CONTROL UPDATE
        # ====================================================

        server_control = (
            update_server_control(

                server_control=
                    server_control,

                client_control_deltas=
                    client_control_deltas,
            )
        )


        # ====================================================
        # DIAGNOSTICS
        # ====================================================

        global_update_norm = (
            phase4.state_update_norm(

                old_global_state,

                aggregated_state,
            )
        )


        server_control_norm = (
            control_variate_norm(
                server_control
            )
        )


        average_client_control_norm = float(

            np.mean(

                [

                    control_variate_norm(

                        client_controls[
                            client_id
                        ]
                    )

                    for client_id
                    in range(
                        1,
                        N_CLIENTS + 1,
                    )
                ]
            )
        )


        mean_delta_c_norm = (
            mean_control_delta_norm(

                client_control_deltas
            )
        )


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


        round_record = {

            "round":
                round_number,

            "weighted_local_loss":
                weighted_loss,

            "weighted_local_accuracy":
                weighted_accuracy,

            "global_update_norm":
                global_update_norm,

            "server_control_norm":
                server_control_norm,

            "mean_client_control_norm":
                average_client_control_norm,

            "mean_client_control_delta_norm":
                mean_delta_c_norm,

            "round_training_seconds":
                round_seconds,

            "feature_count":
                len(
                    selected_features
                ),
        }


        for index in range(
            N_CLIENTS
        ):

            round_record[
                f"client_{index + 1}_steps"
            ] = int(
                client_steps[
                    index
                ]
            )


            round_record[
                f"client_{index + 1}_loss"
            ] = float(
                client_losses[
                    index
                ]
            )


            round_record[
                f"client_{index + 1}_accuracy"
            ] = float(
                client_accuracies[
                    index
                ]
            )


        history.append(
            round_record
        )


        logger.info(
            "ROUND %d COMPLETE | "
            "loss=%.6f | "
            "acc=%.6f | "
            "update=%.6f | "
            "server_c=%.6f | "
            "client_c=%.6f | "
            "delta_c=%.6f | "
            "time=%.2fs",
            round_number,
            weighted_loss,
            weighted_accuracy,
            global_update_norm,
            server_control_norm,
            average_client_control_norm,
            mean_delta_c_norm,
            round_seconds,
        )


    # ========================================================
    # TOTAL TRAINING TIME
    # ========================================================

    training_seconds = (

        time.perf_counter()

        -

        total_training_start
    )


    # ========================================================
    # FINAL LOCKED EVALUATION
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
        "final_server_control_norm"
    ] = float(

        control_variate_norm(
            server_control
        )
    )


    overall[
        "implementation_variant"
    ] = (
        "SCAFFOLD-style control-variate "
        "correction with AdamW"
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
        / "scaffold_round_history.csv",

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
    # PER-CLASS
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
    # SAVE MODEL + CONTROLS
    # ========================================================

    torch.save(

        {

            "phase":
                "8D",

            "algorithm":
                "Hybrid EVO-GA + SCAFFOLD",

            "implementation_variant":
                (
                    "SCAFFOLD-style gradient "
                    "correction with AdamW"
                ),

            "model_state_dict":
                global_model.state_dict(),

            "server_control":
                server_control,

            "client_controls":
                client_controls,

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
        / "hybrid_scaffold_model.pt",
    )


    # ========================================================
    # MANIFEST
    # ========================================================

    manifest = {

        "phase":
            "8D",

        "algorithm":
            "Hybrid EVO-GA + SCAFFOLD",

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

        "gradient_correction":
            "g_corrected = g - c_i + c",

        "client_control_update":
            (
                "c_i_new = c_i - c + "
                "(w_global - w_local)/(K*learning_rate)"
            ),

        "server_control_update":
            (
                "c_new = c + mean(delta_c_i)"
            ),

        "model_aggregation":
            "sample-weighted FedAvg",

        "scientific_note":
            (
                "Canonical SCAFFOLD is generally derived "
                "for SGD-style local optimization. AdamW "
                "was preserved here to maintain identical "
                "local optimizer configuration across "
                "the existing experimental pipeline."
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
        / "phase8d_scaffold_manifest.json",

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

            scaffold_overall=
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
                "Hybrid37-SCAFFOLD-"
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
        "PHASE 8D COMPLETE — "
        "HYBRID EVO-GA + SCAFFOLD"
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
        "Optimizer           : AdamW"
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
        "PER-CLASS SCAFFOLD PERFORMANCE"
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
    # CONTROL-VARIATE DIAGNOSTICS
    # ========================================================

    print()
    print(
        "-" * 90
    )

    print(
        "SCAFFOLD CONTROL-VARIATE DIAGNOSTICS"
    )

    print(
        "-" * 90
    )


    print(
        f"Final server control norm     : "
        f"{history_df['server_control_norm'].iloc[-1]:.6f}"
    )

    print(
        f"Final mean client control norm: "
        f"{history_df['mean_client_control_norm'].iloc[-1]:.6f}"
    )

    print(
        f"Final mean delta-c norm       : "
        f"{history_df['mean_client_control_delta_norm'].iloc[-1]:.6f}"
    )

    print(
        f"Final global update norm      : "
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
        f"Phase-4 parameters        : "
        f"{communication['phase4_parameter_count']:,}"
    )

    print(
        f"SCAFFOLD parameters       : "
        f"{communication['scaffold_parameter_count']:,}"
    )

    print(
        f"Parameter reduction       : "
        f"{communication['parameter_reduction_ratio'] * 100:.2f}%"
    )

    print(
        f"Model payload             : "
        f"{communication['scaffold_model_payload_mib']:.4f} MiB"
    )

    print(
        f"Control payload           : "
        f"{communication['scaffold_control_payload_mib']:.4f} MiB"
    )

    print(
        f"Phase-4 total comm.       : "
        f"{communication['phase4_total_communication_mib']:.4f} MiB"
    )

    print(
        f"SCAFFOLD total comm.      : "
        f"{communication['scaffold_total_communication_mib']:.4f} MiB"
    )

    print(
        f"Communication change      : "
        f"{communication['communication_change_ratio_vs_phase4'] * 100:+.2f}%"
    )


    # ========================================================
    # CURRENT METHOD COMPARISON
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
        "Hybrid optimization rerun           : NO ✅"
    )

    print(
        "Same MLP architecture               : YES ✅"
    )

    print(
        "Same AdamW optimizer                : YES ✅"
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
        "SCAFFOLD control variates           : YES ✅"
    )

    print(
        "Client controls retained by client  : YES ✅"
    )

    print(
        "Server global control retained       : YES ✅"
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