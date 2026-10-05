"""
Phase 5 - EVO Feature Selection + Federated Learning with FedAvg
================================================================

Research objective:

    Phase 4:
        FedAvg + 70 active features

                    VS

    Phase 5:
        Frozen EVO feature subset + same FedAvg

The EVO optimizer has already been executed independently by:

    scripts/phase5_evo_test.py

using:

    optimization/evo.py
    feature_selection/evo_feature_selector.py

This script DOES NOT rerun EVO.

Instead it loads the frozen EVO mask from:

    results/phase5_evo/evo_feature_mask.csv

and applies exactly those selected features to the existing
Phase-4 federated learning pipeline.

IMPORTANT
---------

1. Client construction is unchanged.
2. MLP architecture is unchanged except input dimension.
3. FedAvg is unchanged.
4. Class weighting is unchanged.
5. Training hyperparameters are unchanged.
6. Locked test is NEVER used during federated training.
7. Locked test is evaluated only after the final round.
8. The existing Phase-4 preprocessing scaler is reused.
9. The EVO mask must map exactly to the 70 active Phase-4 features.

Recommended research run:

    python scripts/phase5_evo_fedavg.py --scenario non_iid

Other supported runs:

    python scripts/phase5_evo_fedavg.py --scenario controlled_non_iid

    python scripts/phase5_evo_fedavg.py --scenario iid
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


# ============================================================
# PROJECT ROOT / SCRIPT ROOT
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
# REUSE PHASE 4 IMPLEMENTATION
#
# We intentionally reuse Phase 4 instead of rewriting:
#
#   IDSMLP
#   preprocessing
#   label encoding
#   class weighting
#   local training
#   FedAvg
#   metrics
#   locked-test evaluation
#
# This keeps the ablation controlled.
# ============================================================

import phase4_fedavg as phase4


# ============================================================
# CONFIGURATION
# ============================================================

SEED = phase4.SEED

N_CLIENTS = phase4.N_CLIENTS

EXPECTED_PHASE4_FEATURES = 70

BENIGN_LABEL = phase4.BENIGN_LABEL


# ============================================================
# DEFAULT TRAINING PARAMETERS
# ============================================================

DEFAULT_ROUNDS = phase4.DEFAULT_ROUNDS

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
# EVO ARTIFACTS
# ============================================================

EVO_RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "phase5_evo"
)

DEFAULT_EVO_MASK_FILE = (
    EVO_RESULT_DIR
    / "evo_feature_mask.csv"
)

EVO_SUMMARY_FILE = (
    EVO_RESULT_DIR
    / "phase5_evo_summary.json"
)


# ============================================================
# PHASE 5 OUTPUTS
# ============================================================

ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "evo_fedavg"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "evo_fedavg"
)


# ============================================================
# PHASE 4 BASELINE RESULTS
# ============================================================

PHASE4_RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "fedavg"
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
    "phase5_evo_fedavg"
)


# ============================================================
# FILE HASH
# ============================================================

def sha256_file(
    path: Path,
):

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
# LOAD FROZEN EVO FEATURE SUBSET
# ============================================================

def load_evo_feature_selection(
    mask_file: Path,
    active_features,
    active_indices,
    feature_columns,
):

    """
    Load the feature mask created by phase5_evo_test.py.

    The mask operates over the 70 Phase-4 active features.

    We convert:

        EVO active-feature positions
                    ↓
        original scaler feature indices

    so that Phase-4 transform_features() can be reused
    without modification.
    """

    if not mask_file.exists():

        raise FileNotFoundError(
            "\nFrozen EVO feature mask not found:\n"
            f"{mask_file}\n\n"
            "Run first:\n"
            "python scripts/phase5_evo_test.py"
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
            "\nInvalid EVO mask file.\n"
            f"Missing columns: "
            f"{missing_columns}"
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

    if (
        mask_df[
            "feature_name"
        ]
        .duplicated()
        .any()
    ):

        duplicates = (
            mask_df[
                mask_df[
                    "feature_name"
                ].duplicated(
                    keep=False
                )
            ][
                "feature_name"
            ]
            .tolist()
        )

        raise ValueError(
            "\nDuplicate features found "
            "inside EVO mask:\n"
            f"{duplicates}"
        )

    # --------------------------------------------------------
    # Phase 4 should contain exactly 70 active features
    # --------------------------------------------------------

    if (
        len(active_features)
        !=
        EXPECTED_PHASE4_FEATURES
    ):

        raise ValueError(
            "\nPhase-4 feature schema changed.\n"
            f"Expected: "
            f"{EXPECTED_PHASE4_FEATURES}\n"
            f"Found   : "
            f"{len(active_features)}"
        )

    # --------------------------------------------------------
    # Mask should also describe exactly 70 features
    # --------------------------------------------------------

    if (
        len(mask_df)
        !=
        len(active_features)
    ):

        raise ValueError(
            "\nEVO feature mask size does not "
            "match Phase-4 feature space.\n"
            f"Mask rows           : "
            f"{len(mask_df)}\n"
            f"Phase-4 active feats: "
            f"{len(active_features)}"
        )

    mask_feature_set = set(
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
        mask_feature_set
    )

    extra_in_mask = (
        mask_feature_set
        -
        active_feature_set
    )

    if (
        missing_from_mask
        or
        extra_in_mask
    ):

        raise ValueError(
            "\nEVO mask feature schema differs "
            "from Phase 4.\n\n"
            f"Missing from EVO mask:\n"
            f"{sorted(missing_from_mask)}\n\n"
            f"Extra in EVO mask:\n"
            f"{sorted(extra_in_mask)}"
        )

    # --------------------------------------------------------
    # Map selected flag by feature name
    # --------------------------------------------------------

    selection_map = {

        row.feature_name:
            int(row.selected)

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
            "EVO selected column "
            "must contain only 0 or 1."
        )

    # --------------------------------------------------------
    # Preserve Phase-4 active feature order
    # --------------------------------------------------------

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

    if len(
        selected_features
    ) == 0:

        raise ValueError(
            "EVO mask selected zero features."
        )

    # --------------------------------------------------------
    # Convert selected active positions back to the
    # ORIGINAL scaler feature positions.
    #
    # Example:
    #
    # scaler input = 78 original features
    #
    # active_indices =
    #     70 non-zero-variance positions
    #
    # EVO mask =
    #     positions among those 70
    #
    # selected_original_indices =
    #     final indices used after scaler.transform()
    # --------------------------------------------------------

    active_indices_array = np.asarray(
        active_indices,
        dtype=np.int64,
    )

    selected_original_indices = (
        active_indices_array[
            selected_active_positions
        ]
    )

    # --------------------------------------------------------
    # Final mapping integrity check
    # --------------------------------------------------------

    reconstructed_features = [

        feature_columns[
            int(index)
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
            "\nEVO feature mapping failed.\n\n"
            f"Expected:\n"
            f"{selected_features}\n\n"
            f"Reconstructed:\n"
            f"{reconstructed_features}"
        )

    # --------------------------------------------------------
    # Optional independent-EVO summary verification
    # --------------------------------------------------------

    evo_summary = None

    if EVO_SUMMARY_FILE.exists():

        with open(
            EVO_SUMMARY_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            evo_summary = json.load(
                file
            )

        expected_selected_count = None

        # New Phase-5 summary structure
        if (
            "search_best_result"
            in evo_summary
        ):

            expected_selected_count = (
                evo_summary[
                    "search_best_result"
                ]
                .get(
                    "selected_count"
                )
            )

        # Older structure compatibility
        elif "evo" in evo_summary:

            expected_selected_count = (
                evo_summary[
                    "evo"
                ]
                .get(
                    "selected_count"
                )
            )

        if (
            expected_selected_count
            is not None
            and
            int(
                expected_selected_count
            )
            !=
            len(selected_features)
        ):

            raise ValueError(
                "\nFrozen EVO mask conflicts "
                "with Phase-5 EVO summary.\n"
                f"Summary count: "
                f"{expected_selected_count}\n"
                f"Mask count   : "
                f"{len(selected_features)}"
            )

    logger.info(
        "EVO mask verified."
    )

    logger.info(
        "Phase-4 active features : %d",
        len(active_features),
    )

    logger.info(
        "EVO selected features   : %d",
        len(selected_features),
    )

    logger.info(
        "Feature reduction       : %.2f%%",
        (
            1
            -
            len(selected_features)
            /
            len(active_features)
        )
        * 100,
    )

    return (
        selected_features,
        np.asarray(
            selected_active_positions,
            dtype=np.int64,
        ),
        selected_original_indices,
        evo_summary,
    )


# ============================================================
# INITIALIZE EVO MODEL FROM PHASE-4 INITIALIZATION
# ============================================================

def initialize_projected_global_model(
    selected_active_positions,
    full_input_dim,
    selected_input_dim,
    num_classes,
    device,
):

    """
    Stronger ablation control.

    Phase 4 initializes:

        IDSMLP(70, classes)
        with SEED=42

    Here we recreate that exact 70-feature initial model.

    We then create the 10-feature model and:

        - take only selected columns from first Linear layer
        - copy first-layer bias
        - copy all later layers exactly

    Therefore common parameters start from the same Phase-4
    initialization instead of independently sampling a new
    network due to the smaller input layer.
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

        ).to(device)
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

    # Creation RNG no longer matters because every parameter
    # is overwritten immediately below.

    selected_model = (
        phase4.IDSMLP(

            input_dim=
                selected_input_dim,

            num_classes=
                num_classes,

        ).to(device)
    )

    selected_state = (
        selected_model
        .state_dict()
    )

    first_weight_key = (
        "network.0.weight"
    )

    positions = [
        int(x)
        for x
        in selected_active_positions
    ]

    for key in (
        selected_state.keys()
    ):

        if key == first_weight_key:

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
                    "\nProjected first-layer "
                    "weight has incorrect shape.\n"
                    f"Projected: "
                    f"{projected.shape}\n"
                    f"Expected : "
                    f"{selected_state[key].shape}"
                )

            selected_state[
                key
            ] = projected

        else:

            if (
                reference_state[
                    key
                ].shape
                !=
                selected_state[
                    key
                ].shape
            ):

                raise ValueError(
                    "\nUnexpected architecture "
                    "difference for parameter:\n"
                    f"{key}\n"
                    f"Phase4 shape: "
                    f"{reference_state[key].shape}\n"
                    f"Phase5 shape: "
                    f"{selected_state[key].shape}"
                )

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
# MODEL SIZE / COMMUNICATION COST
# ============================================================

def model_parameter_count(
    model,
):

    return int(
        sum(
            parameter.numel()

            for parameter
            in model.parameters()
        )
    )


def model_payload_bytes(
    model,
):

    """
    Theoretical serialized parameter payload.

    Protocol headers, compression and transport overhead
    are intentionally excluded.
    """

    return int(
        sum(

            tensor.numel()
            *
            tensor.element_size()

            for tensor
            in model.state_dict().values()
        )
    )


def calculate_communication_metrics(
    baseline_model,
    evo_model,
    rounds,
):

    baseline_parameters = (
        model_parameter_count(
            baseline_model
        )
    )

    evo_parameters = (
        model_parameter_count(
            evo_model
        )
    )

    baseline_bytes = (
        model_payload_bytes(
            baseline_model
        )
    )

    evo_bytes = (
        model_payload_bytes(
            evo_model
        )
    )

    # One complete FL round:
    #
    # Server -> N clients : model download
    # N clients -> Server : model upload
    #
    # Total:
    # 2 * N_CLIENTS * model_size

    baseline_round_bytes = (
        2
        *
        N_CLIENTS
        *
        baseline_bytes
    )

    evo_round_bytes = (
        2
        *
        N_CLIENTS
        *
        evo_bytes
    )

    baseline_total_bytes = (
        baseline_round_bytes
        *
        rounds
    )

    evo_total_bytes = (
        evo_round_bytes
        *
        rounds
    )

    communication_reduction = (

        1.0

        -
        (
            evo_total_bytes
            /
            baseline_total_bytes
        )
    )

    parameter_reduction = (

        1.0

        -
        (
            evo_parameters
            /
            baseline_parameters
        )
    )

    return {

        "assumption":
            (
                "Full float32 model downloaded and "
                "uploaded by every client every round; "
                "protocol/compression overhead excluded."
            ),

        "clients":
            N_CLIENTS,

        "rounds":
            int(rounds),

        "phase4_parameter_count":
            baseline_parameters,

        "evo_parameter_count":
            evo_parameters,

        "parameter_reduction_ratio":
            float(
                parameter_reduction
            ),

        "phase4_model_payload_bytes":
            baseline_bytes,

        "evo_model_payload_bytes":
            evo_bytes,

        "phase4_model_payload_mib":
            float(
                baseline_bytes
                /
                (
                    1024 ** 2
                )
            ),

        "evo_model_payload_mib":
            float(
                evo_bytes
                /
                (
                    1024 ** 2
                )
            ),

        "phase4_communication_per_round_bytes":
            baseline_round_bytes,

        "evo_communication_per_round_bytes":
            evo_round_bytes,

        "phase4_total_communication_bytes":
            baseline_total_bytes,

        "evo_total_communication_bytes":
            evo_total_bytes,

        "phase4_total_communication_mib":
            float(
                baseline_total_bytes
                /
                (
                    1024 ** 2
                )
            ),

        "evo_total_communication_mib":
            float(
                evo_total_bytes
                /
                (
                    1024 ** 2
                )
            ),

        "communication_reduction_ratio":
            float(
                communication_reduction
            ),
    }


# ============================================================
# CREATE PER-CLASS DATAFRAME
# ============================================================

def build_per_class_dataframe(
    report,
    class_names,
):

    rows = []

    for name in class_names:

        metrics = report[
            name
        ]

        rows.append({

            "class":
                name,

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
# SAVE PHASE 4 VS PHASE 5 COMPARISONS
# ============================================================

def save_phase4_comparisons(
    scenario,
    overall,
    per_class_df,
    transfer_df,
    selected_feature_count,
    communication,
    result_dir,
):

    baseline_dir = (
        PHASE4_RESULT_ROOT
        / scenario
    )

    baseline_metrics_file = (
        baseline_dir
        / "overall_metrics.json"
    )

    if not baseline_metrics_file.exists():

        logger.warning(
            "Phase-4 baseline results not found: %s",
            baseline_metrics_file,
        )

        return None

    with open(
        baseline_metrics_file,
        "r",
        encoding="utf-8",
    ) as file:

        baseline = json.load(
            file
        )

    # --------------------------------------------------------
    # Overall comparison
    # --------------------------------------------------------

    comparison_rows = [

        {
            "method":
                "Phase4_FedAvg",

            "features":
                EXPECTED_PHASE4_FEATURES,

            "accuracy":
                baseline.get(
                    "accuracy"
                ),

            "balanced_accuracy":
                baseline.get(
                    "balanced_accuracy"
                ),

            "macro_precision":
                baseline.get(
                    "macro_precision"
                ),

            "macro_recall":
                baseline.get(
                    "macro_recall"
                ),

            "macro_f1":
                baseline.get(
                    "macro_f1"
                ),

            "weighted_f1":
                baseline.get(
                    "weighted_f1"
                ),

            "training_time_seconds":
                baseline.get(
                    "training_time_seconds"
                ),

            "inference_time_seconds":
                baseline.get(
                    "inference_time_seconds"
                ),

            "parameter_count":
                communication[
                    "phase4_parameter_count"
                ],

            "model_payload_bytes":
                communication[
                    "phase4_model_payload_bytes"
                ],

            "total_communication_bytes":
                communication[
                    "phase4_total_communication_bytes"
                ],
        },

        {
            "method":
                "Phase5_EVO_FedAvg",

            "features":
                selected_feature_count,

            "accuracy":
                overall.get(
                    "accuracy"
                ),

            "balanced_accuracy":
                overall.get(
                    "balanced_accuracy"
                ),

            "macro_precision":
                overall.get(
                    "macro_precision"
                ),

            "macro_recall":
                overall.get(
                    "macro_recall"
                ),

            "macro_f1":
                overall.get(
                    "macro_f1"
                ),

            "weighted_f1":
                overall.get(
                    "weighted_f1"
                ),

            "training_time_seconds":
                overall.get(
                    "training_time_seconds"
                ),

            "inference_time_seconds":
                overall.get(
                    "inference_time_seconds"
                ),

            "parameter_count":
                communication[
                    "evo_parameter_count"
                ],

            "model_payload_bytes":
                communication[
                    "evo_model_payload_bytes"
                ],

            "total_communication_bytes":
                communication[
                    "evo_total_communication_bytes"
                ],
        },
    ]

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_df.to_csv(

        result_dir
        / "phase4_vs_evo_fedavg_overall.csv",

        index=False,
    )

    # --------------------------------------------------------
    # Delta JSON
    # --------------------------------------------------------

    def metric_delta(
        metric,
    ):

        baseline_value = baseline.get(
            metric
        )

        evo_value = overall.get(
            metric
        )

        if (
            baseline_value is None
            or
            evo_value is None
        ):

            return None

        return float(
            evo_value
            -
            baseline_value
        )

    delta = {

        "accuracy_delta":
            metric_delta(
                "accuracy"
            ),

        "balanced_accuracy_delta":
            metric_delta(
                "balanced_accuracy"
            ),

        "macro_precision_delta":
            metric_delta(
                "macro_precision"
            ),

        "macro_recall_delta":
            metric_delta(
                "macro_recall"
            ),

        "macro_f1_delta":
            metric_delta(
                "macro_f1"
            ),

        "weighted_f1_delta":
            metric_delta(
                "weighted_f1"
            ),

        "feature_reduction_ratio":
            float(
                1
                -
                selected_feature_count
                /
                EXPECTED_PHASE4_FEATURES
            ),

        "parameter_reduction_ratio":
            communication[
                "parameter_reduction_ratio"
            ],

        "communication_reduction_ratio":
            communication[
                "communication_reduction_ratio"
            ],
    }

    if (
        baseline.get(
            "training_time_seconds"
        )
        is not None
        and
        overall.get(
            "training_time_seconds"
        )
        is not None
    ):

        delta[
            "training_time_delta_seconds"
        ] = float(

            overall[
                "training_time_seconds"
            ]

            -
            baseline[
                "training_time_seconds"
            ]
        )

    if (
        baseline.get(
            "inference_time_seconds"
        )
        is not None
        and
        overall.get(
            "inference_time_seconds"
        )
        is not None
    ):

        delta[
            "inference_time_delta_seconds"
        ] = float(

            overall[
                "inference_time_seconds"
            ]

            -
            baseline[
                "inference_time_seconds"
            ]
        )

    with open(

        result_dir
        / "phase4_vs_evo_fedavg_delta.json",

        "w",
        encoding="utf-8",

    ) as file:

        json.dump(
            delta,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # Per-class comparison
    # --------------------------------------------------------

    baseline_per_class_file = (
        baseline_dir
        / "per_class_metrics.csv"
    )

    if baseline_per_class_file.exists():

        baseline_per_class = (
            pd.read_csv(
                baseline_per_class_file
            )
        )

        merged = (
            baseline_per_class
            .merge(

                per_class_df,

                on="class",

                how="outer",

                suffixes=(
                    "_phase4",
                    "_evo_fedavg",
                ),
            )
        )

        for metric in [
            "precision",
            "recall",
            "f1_score",
        ]:

            left = (
                f"{metric}_phase4"
            )

            right = (
                f"{metric}_evo_fedavg"
            )

            if (
                left in merged.columns
                and
                right in merged.columns
            ):

                merged[
                    f"{metric}_delta"
                ] = (

                    merged[
                        right
                    ]

                    -
                    merged[
                        left
                    ]
                )

        merged.to_csv(

            result_dir
            / "phase4_vs_evo_fedavg_per_class.csv",

            index=False,
        )

    # --------------------------------------------------------
    # Knowledge-transfer comparison
    # --------------------------------------------------------

    baseline_transfer_file = (
        baseline_dir
        / "knowledge_transfer.csv"
    )

    if (
        transfer_df is not None
        and
        baseline_transfer_file.exists()
    ):

        baseline_transfer = (
            pd.read_csv(
                baseline_transfer_file
            )
        )

        transfer_comparison = (
            baseline_transfer
            .merge(

                transfer_df,

                on=[
                    "client",
                    "local_seen_attack",
                ],

                how="outer",

                suffixes=(
                    "_phase4",
                    "_evo_fedavg",
                ),
            )
        )

        phase4_col = (
            "knowledge_transfer_gain_phase4"
        )

        evo_col = (
            "knowledge_transfer_gain_evo_fedavg"
        )

        if (
            phase4_col
            in transfer_comparison.columns
            and
            evo_col
            in transfer_comparison.columns
        ):

            transfer_comparison[
                "knowledge_transfer_gain_delta"
            ] = (

                transfer_comparison[
                    evo_col
                ]

                -
                transfer_comparison[
                    phase4_col
                ]
            )

        transfer_comparison.to_csv(

            result_dir
            / "phase4_vs_evo_fedavg_knowledge_transfer.csv",

            index=False,
        )

    return delta


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    scenario,
    global_model,
    history,
    overall,
    report,
    cm,
    class_names,
    active_features,
    selected_features,
    selected_active_positions,
    selected_original_indices,
    label_mapping,
    args,
    client_sample_counts,
    total_training_seconds,
    inference_seconds,
    communication,
    evo_summary,
    mask_file,
    result_dir,
    artifact_dir,
):

    # --------------------------------------------------------
    # Add runtime and research metadata
    # --------------------------------------------------------

    overall[
        "training_time_seconds"
    ] = float(
        total_training_seconds
    )

    overall[
        "inference_time_seconds"
    ] = float(
        inference_seconds
    )

    overall[
        "original_feature_count"
    ] = int(
        len(active_features)
    )

    overall[
        "selected_feature_count"
    ] = int(
        len(selected_features)
    )

    overall[
        "feature_reduction_ratio"
    ] = float(

        1.0

        -
        len(selected_features)
        /
        len(active_features)
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    torch.save(

        {

            "model_state_dict":
                global_model.state_dict(),

            "phase":
                5,

            "algorithm":
                "EVO + FedAvg",

            "input_dim":
                len(
                    selected_features
                ),

            "num_classes":
                len(
                    label_mapping
                ),

            "phase4_active_features":
                active_features,

            "selected_features":
                selected_features,

            "selected_active_positions":
                [
                    int(x)
                    for x
                    in selected_active_positions
                ],

            "selected_original_indices":
                [
                    int(x)
                    for x
                    in selected_original_indices
                ],

            "label_mapping":
                label_mapping,

            "aggregation":
                "sample-weighted FedAvg",

            "feature_selection":
                "Frozen Energy Valley Optimizer mask",

            "evo_mask_file":
                str(
                    mask_file
                ),

            "evo_mask_sha256":
                sha256_file(
                    mask_file
                ),
        },

        artifact_dir
        / "evo_fedavg_global_model.pt",
    )

    # --------------------------------------------------------
    # Round history
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(

        result_dir
        / "fedavg_round_history.csv",

        index=False,
    )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Per-class metrics
    # --------------------------------------------------------

    per_class_df = (
        build_per_class_dataframe(
            report,
            class_names,
        )
    )

    per_class_df.to_csv(

        result_dir
        / "per_class_metrics.csv",

        index=False,
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Frozen feature subset
    # --------------------------------------------------------

    selected_df = pd.DataFrame({

        "selected_order":
            np.arange(
                1,
                len(
                    selected_features
                )
                + 1,
            ),

        "active_feature_position":
            selected_active_positions,

        "original_scaler_position":
            selected_original_indices,

        "feature_name":
            selected_features,
    })

    selected_df.to_csv(

        result_dir
        / "selected_evo_features.csv",

        index=False,
    )

    # --------------------------------------------------------
    # Complete 70-feature mask actually applied
    # --------------------------------------------------------

    selected_set = set(
        selected_features
    )

    applied_mask = pd.DataFrame({

        "active_feature_index":
            np.arange(
                len(
                    active_features
                )
            ),

        "feature_name":
            active_features,

        "selected":
            [
                int(
                    feature
                    in
                    selected_set
                )

                for feature
                in active_features
            ],
    })

    applied_mask.to_csv(

        result_dir
        / "applied_evo_feature_mask.csv",

        index=False,
    )

    # --------------------------------------------------------
    # Communication metrics
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = {

        "phase":
            5,

        "algorithm":
            "EVO + FedAvg",

        "scenario":
            scenario,

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

        "chunk_size":
            args.chunk_size,

        "learning_rate":
            args.learning_rate,

        "weight_decay":
            args.weight_decay,

        "aggregation":
            "sample-weighted FedAvg",

        "feature_selection":
            {

                "optimizer":
                    "Energy Valley Optimizer",

                "mode":
                    "frozen feature mask",

                "mask_file":
                    str(
                        mask_file
                    ),

                "mask_sha256":
                    sha256_file(
                        mask_file
                    ),

                "starting_features":
                    len(
                        active_features
                    ),

                "selected_features":
                    len(
                        selected_features
                    ),

                "feature_reduction_ratio":
                    float(

                        1.0

                        -
                        len(
                            selected_features
                        )
                        /
                        len(
                            active_features
                        )
                    ),

                "feature_names":
                    selected_features,

                "evo_reexecuted_during_fedavg":
                    False,
            },

        "architecture":
            [

                len(
                    selected_features
                ),

                256,
                128,
                64,

                len(
                    label_mapping
                ),
            ],

        "initialization":
            (
                "Projected from Phase-4 70-feature "
                "seed-42 initialization; selected "
                "columns retained in first layer "
                "and all shared parameters copied."
            ),

        "client_sample_counts":
            client_sample_counts,

        "total_federated_training_seconds":
            float(
                total_training_seconds
            ),

        "final_inference_seconds":
            float(
                inference_seconds
            ),

        "communication":
            communication,

        "locked_test_used_during_training":
            False,

        "locked_test_used_for":
            "final evaluation only",

        "final_metrics":
            overall,

        "independent_evo_summary":
            evo_summary,
    }

    with open(

        result_dir
        / "phase5_evo_fedavg_manifest.json",

        "w",
        encoding="utf-8",

    ) as file:

        json.dump(
            manifest,
            file,
            indent=4,
        )

    return (
        history_df,
        per_class_df,
    )


# ============================================================
# PHASE 4 VS EVO-FEDAVG GRAPH
# ============================================================

def generate_phase4_comparison_graph(
    scenario,
    result_dir,
):

    comparison_file = (

        result_dir
        / "phase4_vs_evo_fedavg_overall.csv"
    )

    if not comparison_file.exists():

        return

    comparison = pd.read_csv(
        comparison_file
    )

    if len(
        comparison
    ) != 2:

        return

    metric_columns = [

        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
    ]

    labels = [

        "Accuracy",
        "Balanced Accuracy",
        "Macro F1",
        "Weighted F1",
    ]

    baseline_values = [

        float(
            comparison.iloc[0][
                metric
            ]
        )
        * 100

        for metric
        in metric_columns
    ]

    evo_values = [

        float(
            comparison.iloc[1][
                metric
            ]
        )
        * 100

        for metric
        in metric_columns
    ]

    x = np.arange(
        len(labels)
    )

    width = 0.35

    plt.figure(
        figsize=(
            11,
            6,
        )
    )

    plt.bar(

        x
        -
        width
        /
        2,

        baseline_values,

        width,

        label=
            "FedAvg - 70 features",
    )

    plt.bar(

        x
        +
        width
        /
        2,

        evo_values,

        width,

        label=
            "EVO + FedAvg",
    )

    plt.xticks(
        x,
        labels,
    )

    plt.ylim(
        0,
        105,
    )

    plt.ylabel(
        "Score (%)"
    )

    plt.xlabel(
        "Metric"
    )

    plt.title(
        "Phase 4 FedAvg vs "
        f"EVO + FedAvg - {scenario}"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    graph_dir = (
        result_dir
        / "graphs"
    )

    graph_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(

        graph_dir
        / "07_phase4_vs_evo_fedavg.png",

        dpi=300,
    )

    plt.close()


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

        # Severe non-IID is the main Phase-5 research case.
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
            DEFAULT_EVO_MASK_FILE,
    )

    args = parser.parse_args()


    # ========================================================
    # REPRODUCIBILITY
    # ========================================================

    phase4.set_seed(
        SEED
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
            f"Missing federated scenario: "
            f"{scenario_dir}"
        )

    if not phase4.LOCKED_TEST_FILE.exists():

        raise FileNotFoundError(
            f"Locked test missing: "
            f"{phase4.LOCKED_TEST_FILE}"
        )


    # ========================================================
    # OUTPUT DIRECTORIES
    # ========================================================

    artifact_dir = (

        ARTIFACT_ROOT
        /
        args.scenario
    )

    result_dir = (

        RESULT_ROOT
        /
        args.scenario
    )

    artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"
    )

    logger.info(
        "Device: %s",
        device,
    )


    # ========================================================
    # LOAD SAME PHASE-4 PREPROCESSING
    # ========================================================

    (
        scaler,
        feature_columns,
        label_mapping,
        active_indices,
        active_features,

    ) = phase4.load_preprocessing()


    # ========================================================
    # VERIFY 70-FEATURE PHASE-4 BASE SPACE
    # ========================================================

    if (
        len(active_features)
        !=
        EXPECTED_PHASE4_FEATURES
    ):

        raise ValueError(
            "\nExpected exactly 70 Phase-4 "
            "active features.\n"
            f"Found: {len(active_features)}"
        )


    # ========================================================
    # LOAD FROZEN EVO MASK
    # ========================================================

    (
        selected_features,
        selected_active_positions,
        selected_original_indices,
        evo_summary,

    ) = load_evo_feature_selection(

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
    # DISPLAY EVO SUBSET
    # ========================================================

    print()
    print(
        "=" * 90
    )

    print(
        "PHASE 5 - FROZEN EVO FEATURE SET"
    )

    print(
        "=" * 90
    )

    print(
        f"Phase-4 features : "
        f"{len(active_features)}"
    )

    print(
        f"EVO features     : "
        f"{len(selected_features)}"
    )

    print(
        f"Reduction        : "
        f"{feature_reduction * 100:.2f}%"
    )

    print()

    for number, feature in enumerate(
        selected_features,
        start=1,
    ):

        print(
            f"{number:02d}. "
            f"{feature}"
        )

    print()


    # ========================================================
    # GLOBAL CLASS COUNTS
    # SAME AS PHASE 4
    # ========================================================

    global_counts = (
        phase4
        .calculate_global_class_counts(

            label_mapping,

            args.chunk_size,
        )
    )


    # ========================================================
    # GLOBAL CLASS WEIGHTS
    # SAME AS PHASE 4
    # ========================================================

    global_weights = (
        phase4
        .calculate_global_class_weights(
            global_counts
        )
    )

    logger.info(
        "Global class weights:"
    )

    for label, index in (
        label_mapping.items()
    ):

        logger.info(

            "%-20s count=%d weight=%.4f",

            label,

            global_counts[
                index
            ],

            global_weights[
                index
            ],
        )


    # ========================================================
    # CLIENT FILES + SAMPLE COUNTS
    # SAME EXACT CLIENT DATA AS PHASE 4
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

        samples = (
            phase4
            .count_client_samples(

                client_file,

                args.chunk_size,
            )
        )

        client_sample_counts[
            f"client_{client_id}"
        ] = samples

        logger.info(

            "Client %d samples = %d",

            client_id,
            samples,
        )


    # ========================================================
    # INITIAL GLOBAL MODEL
    #
    # Controlled initialization:
    #
    # Reproduce the original Phase-4 70-input network and
    # project the first layer onto the selected EVO columns.
    # ========================================================

    (
        global_model,
        phase4_reference_model,

    ) = initialize_projected_global_model(

        selected_active_positions=
            selected_active_positions,

        full_input_dim=
            len(active_features),

        selected_input_dim=
            len(selected_features),

        num_classes=
            len(label_mapping),

        device=
            device,
    )


    # ========================================================
    # PARAMETER / COMMUNICATION ANALYSIS
    # ========================================================

    communication = (
        calculate_communication_metrics(

            baseline_model=
                phase4_reference_model,

            evo_model=
                global_model,

            rounds=
                args.rounds,
        )
    )

    logger.info(
        "Phase-4 model parameters : %d",
        communication[
            "phase4_parameter_count"
        ],
    )

    logger.info(
        "EVO model parameters     : %d",
        communication[
            "evo_parameter_count"
        ],
    )

    logger.info(
        "Parameter reduction      : %.2f%%",
        communication[
            "parameter_reduction_ratio"
        ]
        * 100,
    )

    logger.info(
        "Communication reduction  : %.2f%%",
        communication[
            "communication_reduction_ratio"
        ]
        * 100,
    )

    # Reference model no longer needed.
    del phase4_reference_model


    # ========================================================
    # FEDERATED TRAINING
    # ========================================================

    history = []

    total_training_start = (
        time.perf_counter()
    )

    print()
    print(
        "=" * 90
    )

    print(
        "PHASE 5 - EVO + FEDAVG"
    )

    print(
        "=" * 90
    )

    print(
        f"Scenario      : "
        f"{args.scenario}"
    )

    print(
        f"Clients       : "
        f"{N_CLIENTS}"
    )

    print(
        f"Rounds        : "
        f"{args.rounds}"
    )

    print(
        f"Local epochs  : "
        f"{args.local_epochs}"
    )

    print(
        f"Original feats: "
        f"{len(active_features)}"
    )

    print(
        f"EVO features  : "
        f"{len(selected_features)}"
    )

    print(
        f"Classes       : "
        f"{len(label_mapping)}"
    )

    print()


    # ========================================================
    # FEDERATED ROUNDS
    # ========================================================

    for round_number in range(
        1,
        args.rounds + 1,
    ):

        print()
        print(
            "-" * 90
        )

        print(
            f"EVO + FEDAVG ROUND "
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

        client_losses = []

        client_accuracies = []

        client_times = []


        # ====================================================
        # LOCAL CLIENT TRAINING
        #
        # Reuses Phase-4 train_client().
        #
        # The only feature-related change is:
        #
        # active_indices
        #       ↓
        # selected_original_indices
        #
        # Therefore transform_features() scales exactly as
        # Phase 4 and returns only the frozen EVO features.
        # ====================================================

        for client_id in range(
            1,
            N_CLIENTS + 1,
        ):

            logger.info(

                "Round %d | "
                "training Client %d "
                "with %d EVO features...",

                round_number,

                client_id,

                len(
                    selected_features
                ),
            )

            (
                local_state,
                local_loss,
                local_accuracy,
                training_seconds,

            ) = phase4.train_client(

                global_state=
                    global_model
                    .state_dict(),

                client_file=
                    client_files[
                        client_id
                    ],

                scaler=
                    scaler,

                feature_columns=
                    feature_columns,

                # CRITICAL:
                # use EVO-selected original scaler indices.
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

            client_samples = (
                client_sample_counts[
                    f"client_{client_id}"
                ]
            )

            aggregation_weights.append(
                client_samples
            )

            client_losses.append(
                local_loss
            )

            client_accuracies.append(
                local_accuracy
            )

            client_times.append(
                training_seconds
            )

            logger.info(

                "Round %d | Client %d | "
                "loss=%.6f | "
                "accuracy=%.6f | "
                "time=%.2fs",

                round_number,
                client_id,
                local_loss,
                local_accuracy,
                training_seconds,
            )


        # ====================================================
        # SAMPLE-WEIGHTED FEDAVG
        # IDENTICAL TO PHASE 4
        # ====================================================

        aggregated_state = (
            phase4
            .federated_average(

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
            phase4
            .state_update_norm(

                old_global_state,

                aggregated_state,
            )
        )


        # ====================================================
        # WEIGHTED TRAINING STATISTICS
        # ====================================================

        weight_array = np.asarray(

            aggregation_weights,

            dtype=np.float64,
        )

        loss_array = np.asarray(

            client_losses,

            dtype=np.float64,
        )

        accuracy_array = np.asarray(

            client_accuracies,

            dtype=np.float64,
        )

        weighted_loss = float(

            np.average(

                loss_array,

                weights=
                    weight_array,
            )
        )

        weighted_accuracy = float(

            np.average(

                accuracy_array,

                weights=
                    weight_array,
            )
        )

        round_seconds = (

            time.perf_counter()

            -
            round_start
        )


        # ====================================================
        # ROUND HISTORY
        # ====================================================

        history.append({

            "round":
                round_number,

            "weighted_local_loss":
                weighted_loss,

            "weighted_local_accuracy":
                weighted_accuracy,

            "global_update_norm":
                update_norm,

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

            "client_1_training_seconds":
                client_times[0],

            "client_2_training_seconds":
                client_times[1],

            "client_3_training_seconds":
                client_times[2],

            "client_4_training_seconds":
                client_times[3],

            "client_5_training_seconds":
                client_times[4],

            "feature_count":
                len(
                    selected_features
                ),
        })

        logger.info(

            "ROUND %d COMPLETE | "
            "weighted_loss=%.6f | "
            "weighted_accuracy=%.6f | "
            "update_norm=%.6f | "
            "time=%.2fs",

            round_number,
            weighted_loss,
            weighted_accuracy,
            update_norm,
            round_seconds,
        )


    # ========================================================
    # TOTAL FEDERATED TRAINING TIME
    # ========================================================

    total_training_seconds = (

        time.perf_counter()

        -
        total_training_start
    )


    # ========================================================
    # FINAL LOCKED TEST
    #
    # This is the FIRST point at which the locked test is read.
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

        # Frozen EVO subset
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


    # Add timing before comparisons
    overall[
        "training_time_seconds"
    ] = float(
        total_training_seconds
    )

    overall[
        "inference_time_seconds"
    ] = float(
        inference_seconds
    )


    # ========================================================
    # KNOWLEDGE TRANSFER
    # ========================================================

    transfer_df = (
        phase4
        .create_knowledge_transfer_report(

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
    # SAVE PHASE 5 RESULTS
    # ========================================================

    (
        history_df,
        per_class_df,

    ) = save_results(

        scenario=
            args.scenario,

        global_model=
            global_model,

        history=
            history,

        overall=
            overall,

        report=
            report,

        cm=
            cm,

        class_names=
            class_names,

        active_features=
            active_features,

        selected_features=
            selected_features,

        selected_active_positions=
            selected_active_positions,

        selected_original_indices=
            selected_original_indices,

        label_mapping=
            label_mapping,

        args=
            args,

        client_sample_counts=
            client_sample_counts,

        total_training_seconds=
            total_training_seconds,

        inference_seconds=
            inference_seconds,

        communication=
            communication,

        evo_summary=
            evo_summary,

        mask_file=
            args.mask_file,

        result_dir=
            result_dir,

        artifact_dir=
            artifact_dir,
    )


    # ========================================================
    # PHASE 4 VS PHASE 5 COMPARISONS
    # ========================================================

    delta = save_phase4_comparisons(

        scenario=
            args.scenario,

        overall=
            overall,

        per_class_df=
            per_class_df,

        transfer_df=
            transfer_df,

        selected_feature_count=
            len(
                selected_features
            ),

        communication=
            communication,

        result_dir=
            result_dir,
    )


    # ========================================================
    # STANDARD GRAPHS
    #
    # Reuse Phase-4 graph logic.
    # ========================================================

    phase4.generate_graphs(

        scenario=
            f"EVO+FedAvg-{args.scenario}",

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
    # DIRECT PHASE 4 VS EVO-FEDAVG GRAPH
    # ========================================================

    generate_phase4_comparison_graph(

        scenario=
            args.scenario,

        result_dir=
            result_dir,
    )


    # ========================================================
    # DISPLAY FINAL RESULT
    # ========================================================

    print()
    print(
        "=" * 90
    )

    print(
        "PHASE 5 COMPLETE — EVO + FEDAVG"
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
        f"Phase-4 features    : "
        f"{len(active_features)}"
    )

    print(
        f"EVO features        : "
        f"{len(selected_features)}"
    )

    print(
        f"Feature reduction   : "
        f"{feature_reduction * 100:.2f}%"
    )

    print(
        f"Classes             : "
        f"{len(label_mapping)}"
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
        f"{total_training_seconds:.2f}s"
    )

    print(
        f"Inference time      : "
        f"{inference_seconds:.2f}s"
    )


    # ========================================================
    # MODEL / COMMUNICATION EFFICIENCY
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
        f"Phase-4 parameters  : "
        f"{communication['phase4_parameter_count']:,}"
    )

    print(
        f"EVO parameters      : "
        f"{communication['evo_parameter_count']:,}"
    )

    print(
        f"Parameter reduction : "
        f"{communication['parameter_reduction_ratio'] * 100:.2f}%"
    )

    print(
        f"Phase-4 payload     : "
        f"{communication['phase4_model_payload_mib']:.4f} MiB"
    )

    print(
        f"EVO payload         : "
        f"{communication['evo_model_payload_mib']:.4f} MiB"
    )

    print(
        f"Comm. reduction     : "
        f"{communication['communication_reduction_ratio'] * 100:.2f}%"
    )

    print(
        f"Phase-4 total comm. : "
        f"{communication['phase4_total_communication_mib']:.4f} MiB"
    )

    print(
        f"EVO total comm.     : "
        f"{communication['evo_total_communication_mib']:.4f} MiB"
    )


    # ========================================================
    # PER CLASS
    # ========================================================

    print()
    print(
        "-" * 90
    )

    print(
        "PER-CLASS EVO + FEDAVG PERFORMANCE"
    )

    print(
        "-" * 90
    )

    for class_name in class_names:

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
            "KNOWLEDGE TRANSFER — EVO + FEDAVG"
        )

        print(
            "-" * 90
        )

        print(

            transfer_df.to_string(
                index=False
            )
        )

        print()

        print(
            "Mean knowledge-transfer gain: "
            f"{transfer_df['knowledge_transfer_gain'].mean():.6f}"
        )


    # ========================================================
    # PHASE 4 DELTA
    # ========================================================

    if delta is not None:

        print()

        print(
            "-" * 90
        )

        print(
            "PHASE 4 FEDAVG VS PHASE 5 EVO + FEDAVG"
        )

        print(
            "-" * 90
        )

        for key in [

            "accuracy_delta",
            "balanced_accuracy_delta",
            "macro_precision_delta",
            "macro_recall_delta",
            "macro_f1_delta",
            "weighted_f1_delta",

        ]:

            value = delta.get(
                key
            )

            if value is not None:

                print(
                    f"{key:30s}: "
                    f"{value:+.6f}"
                )

        print()

        print(
            "Feature reduction             : "
            f"{delta['feature_reduction_ratio'] * 100:.2f}%"
        )

        print(
            "Parameter reduction           : "
            f"{delta['parameter_reduction_ratio'] * 100:.2f}%"
        )

        print(
            "Communication reduction       : "
            f"{delta['communication_reduction_ratio'] * 100:.2f}%"
        )


    # ========================================================
    # SELECTED FEATURES
    # ========================================================

    print()

    print(
        "-" * 90
    )

    print(
        "FROZEN EVO FEATURES"
    )

    print(
        "-" * 90
    )

    for number, feature in enumerate(
        selected_features,
        start=1,
    ):

        print(
            f"{number:02d}. "
            f"{feature}"
        )


    # ========================================================
    # FINAL INTEGRITY REPORT
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
        "Existing optimization/evo.py used "
        "to create frozen feature set: YES ✅"
    )

    print(
        "EVO rerun during FedAvg training: NO ✅"
    )

    print(
        "Frozen EVO mask applied: YES ✅"
    )

    print(
        "Same Phase-4 scaler used: YES ✅"
    )

    print(
        "Same Phase-4 MLP used: YES ✅"
    )

    print(
        "Same Phase-4 class weights used: YES ✅"
    )

    print(
        "Same Phase-4 FedAvg used: YES ✅"
    )

    print(
        "Same client files used: YES ✅"
    )

    print(
        "Raw client data shared between clients: NO ✅"
    )

    print(
        "Locked test used during FL training: NO ✅"
    )

    print(
        "Locked test used for final evaluation only: YES ✅"
    )

    print()

    print(
        f"Results:\n{result_dir}"
    )

    print()

    print(
        f"Model artifact:\n{artifact_dir}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()