"""
Phase 9 - Repeated-Seed Validation: Hybrid EVO-GA + FedAvg
===============================================================

Purpose
-------

Phase 4:
    FedAvg
    70 features

Phase 5B:
    Federated-aware EVO
    36 features
    + full FedAvg

Phase 6:
    Federated-aware GA
    34 features
    + full FedAvg

Phase 7 source:
    Hybrid EVO-GA
    37 selected features
        ↓
    frozen Hybrid feature mask
        ↓

Phase 9 validation:
    reuse the SAME frozen 37-feature mask
        ↓
    rerun full 10-round FedAvg with a supplied seed


IMPORTANT
---------

This script DOES NOT rerun EVO or GA.

It loads:

    results/phase7_hybrid/
        hybrid_feature_mask.csv

and runs the SAME Phase-4 FL pipeline:

    - same 5 clients
    - same non-IID scenario
    - same scaler
    - same label mapping
    - same MLP architecture
    - same global class weights
    - same AdamW optimizer
    - same batch size
    - same learning rate
    - same weight decay
    - same sample-weighted FedAvg
    - same 10 rounds
    - same 1 local epoch


Outputs
-------

    results/phase9/hybrid_fedavg/<scenario>/seed_<seed>/

    artifacts/phase9/hybrid_fedavg/<scenario>/seed_<seed>/


Run
---

    python scripts/phase9_hybrid_fedavg.py --scenario non_iid --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
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
# REUSE PHASE 9 FEDAVG70 BASE
# ============================================================
#
# IMPORTANT:
# Phase 9 repeated-seed validation needs the seed-aware
# train_client(..., seed=...) implementation.  That implementation
# lives in phase9_fedavg70.py.  We reuse it as the common FedAvg
# base while keeping the frozen 37-feature Hybrid EVO-GA mask.
#
# Alias it as ``phase4`` so the original Hybrid script can keep
# using the same preprocessing/model/FedAvg helper calls without
# changing the scientific pipeline.
# ============================================================

import phase9_fedavg70 as phase4


# ============================================================
# CONFIGURATION
# ============================================================

SEED = phase4.SEED

N_CLIENTS = phase4.N_CLIENTS

EXPECTED_PHASE4_FEATURES = 70


# ============================================================
# SAME PHASE-4 TRAINING DEFAULTS
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

PHASE7_SEARCH_SUMMARY = (
    PHASE7_SEARCH_DIR
    / "phase7_hybrid_summary.json"
)


# ============================================================
# OUTPUT
# ============================================================

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "phase9"
    / "hybrid_fedavg"
)

ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "phase9"
    / "hybrid_fedavg"
)


# ============================================================
# PREVIOUS EXPERIMENT RESULTS
# ============================================================

PHASE4_RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "fedavg"
)

PHASE5B_RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "federated_evo_fedavg"
)

PHASE6_RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ga_fedavg"
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
    "phase9_hybrid_fedavg"
)


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
            "\nHybrid EVO-GA mask not found:\n"
            f"{mask_file}\n\n"
            "Run first:\n"
            "python scripts\\phase7_hybrid_test.py"
        )


    mask_df = pd.read_csv(
        mask_file
    )


    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

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


    # ========================================================
    # CLEAN NAMES
    # ========================================================

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
    # VERIFY 70-FEATURE BASE SPACE
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
            "active features.\n"
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
    # SCHEMA VALIDATION
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
            "\nHybrid feature mask schema differs "
            "from Phase 4.\n\n"
            f"Missing:\n"
            f"{sorted(missing_from_mask)}\n\n"
            f"Extra:\n"
            f"{sorted(extra_in_mask)}"
        )


    # ========================================================
    # SELECTION VALUES
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
            "\nHybrid mask 'selected' column "
            "must contain only 0 or 1."
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
            "Hybrid EVO-GA selected zero features."
        )


    # ========================================================
    # MAP ACTIVE POSITIONS -> ORIGINAL SCALER POSITIONS
    #
    # Phase 4 preprocessing:
    #
    # full original feature set
    #       ↓ scaler.transform()
    # active_indices
    #       ↓
    # 70 Phase-4 features
    #
    # Hybrid mask operates on those 70.
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


    # ========================================================
    # RECONSTRUCT NAMES FOR SAFETY
    # ========================================================

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
            "\nHybrid feature-index mapping failed.\n\n"
            f"Expected:\n"
            f"{selected_features}\n\n"
            f"Reconstructed:\n"
            f"{reconstructed_features}"
        )


    logger.info(
        "Hybrid EVO-GA feature mask verified."
    )

    logger.info(
        "Phase-4 features : %d",
        len(
            active_features
        ),
    )

    logger.info(
        "Hybrid features  : %d",
        len(
            selected_features
        ),
    )

    logger.info(
        "Feature reduction: %.2f%%",
        (
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
        *
        100.0,
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
# PROJECT PHASE-4 INITIALIZATION
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
    Initialize a full 70-feature Phase-4 reference model using
    the same seed, then project the first-layer columns to the
    Hybrid-selected features.

    This reduces initialization confounding between:

        Phase 4
        EVO + FedAvg
        GA + FedAvg
        Hybrid + FedAvg
    """

    phase4.set_seed(
        seed
    )


    # ========================================================
    # FULL 70-FEATURE REFERENCE MODEL
    # ========================================================

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


    # ========================================================
    # HYBRID MODEL
    # ========================================================

    hybrid_model = (
        phase4.IDSMLP(

            input_dim=
                selected_input_dim,

            num_classes=
                num_classes,

        ).to(
            device
        )
    )


    hybrid_state = (
        hybrid_model.state_dict()
    )


    first_layer_key = (
        "network.0.weight"
    )


    positions = [

        int(
            index
        )

        for index
        in selected_active_positions
    ]


    # ========================================================
    # PROJECT INITIALIZATION
    # ========================================================

    for key in (
        hybrid_state.keys()
    ):

        if (
            key
            ==
            first_layer_key
        ):

            projected_weights = (

                reference_state[
                    key
                ][
                    :,
                    positions
                ]
                .clone()
            )


            if (
                projected_weights.shape
                !=
                hybrid_state[
                    key
                ].shape
            ):

                raise ValueError(
                    "\nProjected first-layer shape "
                    "does not match Hybrid model.\n"
                    f"Projected: "
                    f"{projected_weights.shape}\n"
                    f"Expected: "
                    f"{hybrid_state[key].shape}"
                )


            hybrid_state[
                key
            ] = projected_weights


        else:

            if (
                reference_state[
                    key
                ].shape
                !=
                hybrid_state[
                    key
                ].shape
            ):

                raise ValueError(
                    "\nUnexpected shared parameter "
                    f"shape mismatch: {key}"
                )


            hybrid_state[
                key
            ] = (

                reference_state[
                    key
                ]
                .clone()
            )


    hybrid_model.load_state_dict(
        hybrid_state
    )


    return (
        hybrid_model,
        reference_model,
    )


# ============================================================
# PARAMETER COUNT
# ============================================================

def parameter_count(
    model,
) -> int:

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
) -> int:

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
    hybrid_model,
    rounds,
):

    phase4_parameters = (
        parameter_count(
            baseline_model
        )
    )

    hybrid_parameters = (
        parameter_count(
            hybrid_model
        )
    )


    phase4_payload = (
        model_payload_bytes(
            baseline_model
        )
    )

    hybrid_payload = (
        model_payload_bytes(
            hybrid_model
        )
    )


    # ========================================================
    # ASSUMPTION
    #
    # Per client per round:
    #
    # server -> client download
    # client -> server upload
    #
    # Therefore:
    # 2 * clients * model-size
    # ========================================================

    phase4_per_round = (

        2

        *

        N_CLIENTS

        *

        phase4_payload
    )


    hybrid_per_round = (

        2

        *

        N_CLIENTS

        *

        hybrid_payload
    )


    phase4_total = (

        phase4_per_round

        *

        rounds
    )


    hybrid_total = (

        hybrid_per_round

        *

        rounds
    )


    return {

        "phase4_parameter_count":
            phase4_parameters,

        "phase7_parameter_count":
            hybrid_parameters,

        "parameter_reduction_ratio":
            float(

                1.0

                -

                hybrid_parameters
                /
                phase4_parameters
            ),


        "phase4_model_payload_bytes":
            phase4_payload,

        "phase7_model_payload_bytes":
            hybrid_payload,


        "phase4_model_payload_mib":
            float(

                phase4_payload

                /

                1024 ** 2
            ),

        "phase7_model_payload_mib":
            float(

                hybrid_payload

                /

                1024 ** 2
            ),


        "phase4_total_communication_bytes":
            phase4_total,

        "phase7_total_communication_bytes":
            hybrid_total,


        "phase4_total_communication_mib":
            float(

                phase4_total

                /

                1024 ** 2
            ),

        "phase7_total_communication_mib":
            float(

                hybrid_total

                /

                1024 ** 2
            ),


        "communication_reduction_ratio":
            float(

                1.0

                -

                hybrid_total
                /
                phase4_total
            ),


        "assumption":
            (
                "Full float32 model upload and download "
                "for every client in every federated round. "
                "Protocol, compression and transport "
                "overhead are excluded."
            ),
    }


# ============================================================
# PER CLASS DATAFRAME
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
# SAVE FINAL METHOD COMPARISON
# ============================================================

def save_method_comparison(
    scenario,
    overall,
    selected_feature_count,
    communication,
    result_dir,
):

    phase4_file = (

        PHASE4_RESULT_ROOT
        /
        scenario
        /
        "overall_metrics.json"
    )


    evo_file = (

        PHASE5B_RESULT_ROOT
        /
        scenario
        /
        "overall_metrics.json"
    )


    ga_file = (

        PHASE6_RESULT_ROOT
        /
        scenario
        /
        "overall_metrics.json"
    )


    phase4_metrics = (
        load_json_if_exists(
            phase4_file
        )
    )

    evo_metrics = (
        load_json_if_exists(
            evo_file
        )
    )

    ga_metrics = (
        load_json_if_exists(
            ga_file
        )
    )


    metrics = [

        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    ]


    rows = []


    # ========================================================
    # PHASE 4
    # ========================================================

    if phase4_metrics is not None:

        rows.append({

            "method":
                "FedAvg",

            "features":
                70,

            "feature_reduction":
                0.0,

            **{

                metric:
                    phase4_metrics.get(
                        metric
                    )

                for metric
                in metrics
            },

            "training_time_seconds":
                phase4_metrics.get(
                    "training_time_seconds"
                ),

            "inference_time_seconds":
                phase4_metrics.get(
                    "inference_time_seconds"
                ),

            "parameter_reduction":
                0.0,

            "communication_reduction":
                0.0,
        })


    # ========================================================
    # EVO
    # ========================================================

    if evo_metrics is not None:

        rows.append({

            "method":
                "Federated_EVO_FedAvg",

            "features":
                evo_metrics.get(
                    "selected_feature_count",
                    36,
                ),

            "feature_reduction":
                evo_metrics.get(
                    "feature_reduction_ratio",
                    1.0 - 36.0 / 70.0,
                ),

            **{

                metric:
                    evo_metrics.get(
                        metric
                    )

                for metric
                in metrics
            },

            "training_time_seconds":
                evo_metrics.get(
                    "training_time_seconds"
                ),

            "inference_time_seconds":
                evo_metrics.get(
                    "inference_time_seconds"
                ),

            "parameter_reduction":
                0.1436,

            "communication_reduction":
                0.1436,
        })


    # ========================================================
    # GA
    # ========================================================

    if ga_metrics is not None:

        rows.append({

            "method":
                "Federated_GA_FedAvg",

            "features":
                ga_metrics.get(
                    "selected_feature_count",
                    34,
                ),

            "feature_reduction":
                ga_metrics.get(
                    "feature_reduction_ratio",
                    1.0 - 34.0 / 70.0,
                ),

            **{

                metric:
                    ga_metrics.get(
                        metric
                    )

                for metric
                in metrics
            },

            "training_time_seconds":
                ga_metrics.get(
                    "training_time_seconds"
                ),

            "inference_time_seconds":
                ga_metrics.get(
                    "inference_time_seconds"
                ),

            "parameter_reduction":
                0.1520,

            "communication_reduction":
                0.1520,
        })


    # ========================================================
    # HYBRID
    # ========================================================

    rows.append({

        "method":
            "Hybrid_EVO_GA_FedAvg",

        "features":
            selected_feature_count,

        "feature_reduction":
            float(

                1.0

                -

                selected_feature_count
                /
                EXPECTED_PHASE4_FEATURES
            ),

        **{

            metric:
                overall.get(
                    metric
                )

            for metric
            in metrics
        },

        "training_time_seconds":
            overall.get(
                "training_time_seconds"
            ),

        "inference_time_seconds":
            overall.get(
                "inference_time_seconds"
            ),

        "parameter_reduction":
            communication[
                "parameter_reduction_ratio"
            ],

        "communication_reduction":
            communication[
                "communication_reduction_ratio"
            ],
    })


    comparison_df = pd.DataFrame(
        rows
    )


    comparison_df.to_csv(

        result_dir
        / "final_method_comparison.csv",

        index=False,
    )


    # ========================================================
    # HYBRID DELTA VS PHASE 4
    # ========================================================

    delta_vs_phase4 = None


    if phase4_metrics is not None:

        delta_vs_phase4 = {}


        for metric in (
            metrics
        ):

            if (
                phase4_metrics.get(
                    metric
                )
                is not None

                and

                overall.get(
                    metric
                )
                is not None
            ):

                delta_vs_phase4[
                    f"{metric}_delta"
                ] = float(

                    overall[
                        metric
                    ]

                    -

                    phase4_metrics[
                        metric
                    ]
                )


        delta_vs_phase4[
            "feature_reduction_ratio"
        ] = float(

            1.0

            -

            selected_feature_count
            /
            EXPECTED_PHASE4_FEATURES
        )


        delta_vs_phase4[
            "parameter_reduction_ratio"
        ] = communication[
            "parameter_reduction_ratio"
        ]


        delta_vs_phase4[
            "communication_reduction_ratio"
        ] = communication[
            "communication_reduction_ratio"
        ]


        with open(

            result_dir
            / "phase4_vs_phase7_delta.json",

            "w",
            encoding="utf-8",

        ) as file:

            json.dump(
                delta_vs_phase4,
                file,
                indent=4,
            )


    # ========================================================
    # HYBRID DELTA VS EVO
    # ========================================================

    delta_vs_evo = None


    if evo_metrics is not None:

        delta_vs_evo = {}


        for metric in (
            metrics
        ):

            if (
                evo_metrics.get(
                    metric
                )
                is not None

                and

                overall.get(
                    metric
                )
                is not None
            ):

                delta_vs_evo[
                    f"{metric}_delta"
                ] = float(

                    overall[
                        metric
                    ]

                    -

                    evo_metrics[
                        metric
                    ]
                )


        with open(

            result_dir
            / "phase5b_evo_vs_phase7_hybrid_delta.json",

            "w",
            encoding="utf-8",

        ) as file:

            json.dump(
                delta_vs_evo,
                file,
                indent=4,
            )


    # ========================================================
    # HYBRID DELTA VS GA
    # ========================================================

    delta_vs_ga = None


    if ga_metrics is not None:

        delta_vs_ga = {}


        for metric in (
            metrics
        ):

            if (
                ga_metrics.get(
                    metric
                )
                is not None

                and

                overall.get(
                    metric
                )
                is not None
            ):

                delta_vs_ga[
                    f"{metric}_delta"
                ] = float(

                    overall[
                        metric
                    ]

                    -

                    ga_metrics[
                        metric
                    ]
                )


        with open(

            result_dir
            / "phase6_ga_vs_phase7_hybrid_delta.json",

            "w",
            encoding="utf-8",

        ) as file:

            json.dump(
                delta_vs_ga,
                file,
                indent=4,
            )


    return (

        comparison_df,

        delta_vs_phase4,

        delta_vs_evo,

        delta_vs_ga,
    )


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
    mask_file,
    result_dir,
    artifact_dir,
):

    # ========================================================
    # OVERALL METADATA
    # ========================================================

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
        len(
            active_features
        )
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
    # PHASE 9 RUN METADATA
    # ========================================================

    overall[
        "phase"
    ] = 9

    overall[
        "method"
    ] = "hybrid_fedavg"

    overall[
        "seed"
    ] = int(
        args.seed
    )

    overall[
        "scenario"
    ] = scenario


    # ========================================================
    # SAVE MODEL
    # ========================================================

    torch.save(

        {

            "phase":
                9,

            "algorithm":
                "Phase 9 Repeated-Seed Hybrid EVO-GA + FedAvg",

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

            "selected_features":
                selected_features,

            "selected_active_positions":
                [

                    int(
                        value
                    )

                    for value
                    in selected_active_positions
                ],

            "selected_original_indices":
                [

                    int(
                        value
                    )

                    for value
                    in selected_original_indices
                ],

            "label_mapping":
                label_mapping,

            "mask_file":
                str(
                    mask_file
                ),

            "mask_sha256":
                sha256_file(
                    mask_file
                ),
        },

        artifact_dir
        / "hybrid_evo_ga_fedavg_model.pt",
    )


    # ========================================================
    # ROUND HISTORY
    # ========================================================

    history_df = pd.DataFrame(
        history
    )


    history_df.to_csv(

        result_dir
        / "fedavg_round_history.csv",

        index=False,
    )


    # ========================================================
    # OVERALL METRICS
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
    # PER-CLASS METRICS
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
    # SELECTED FEATURES
    # ========================================================

    pd.DataFrame({

        "selected_order":
            np.arange(

                1,

                len(
                    selected_features
                )
                +
                1,
            ),

        "active_feature_index":
            selected_active_positions,

        "original_scaler_index":
            selected_original_indices,

        "feature_name":
            selected_features,

    }).to_csv(

        result_dir
        / "selected_hybrid_features.csv",

        index=False,
    )


    # ========================================================
    # COMMUNICATION METRICS
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
    # MANIFEST
    # ========================================================

    manifest = {

        "phase":
            9,

        "algorithm":
            (
                "Phase 9 Repeated-Seed "
                "Hybrid EVO-GA + FedAvg"
            ),

        "scenario":
            scenario,

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

        "chunk_size":
            args.chunk_size,

        "learning_rate":
            args.learning_rate,

        "weight_decay":
            args.weight_decay,

        "aggregation":
            "sample-weighted FedAvg",

        "original_features":
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

        "selected_feature_names":
            selected_features,

        "feature_mask_file":
            str(
                mask_file
            ),

        "feature_mask_sha256":
            sha256_file(
                mask_file
            ),

        "hybrid_optimizer_rerun_during_fedavg":
            False,

        "client_sample_counts":
            client_sample_counts,

        "training_time_seconds":
            float(
                total_training_seconds
            ),

        "inference_time_seconds":
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
    }


    with open(

        result_dir
        / "phase9_manifest.json",

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

        "--mask-file",

        type=Path,

        default=
            DEFAULT_MASK_FILE,
    )


    parser.add_argument(

        "--seed",

        type=int,

        default=
            SEED,

        help=(
            "Random seed for Phase 9 repeated-seed "
            "statistical validation."
        ),
    )


    args = parser.parse_args()


    # ========================================================
    # REPRODUCIBILITY
    # ========================================================

    phase4.set_seed(
        args.seed
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
            "\nFederated scenario missing:\n"
            f"{scenario_dir}"
        )


    if not phase4.LOCKED_TEST_FILE.exists():

        raise FileNotFoundError(
            "\nLocked test missing:\n"
            f"{phase4.LOCKED_TEST_FILE}"
        )


    # ========================================================
    # OUTPUT DIRECTORIES
    # ========================================================

    seed_folder = (
        f"seed_{args.seed}"
    )


    result_dir = (

        RESULT_ROOT

        /

        args.scenario

        /

        seed_folder
    )


    artifact_dir = (

        ARTIFACT_ROOT

        /

        args.scenario

        /

        seed_folder
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


    # ========================================================
    # LOAD FROZEN HYBRID MASK
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
    # PRINT SELECTED FEATURE SET
    # ========================================================

    print()
    print(
        "=" * 90
    )

    print(
        "FROZEN PHASE 7 HYBRID EVO-GA FEATURE SET"
    )

    print(
        "=" * 90
    )

    print(
        f"Original features : "
        f"{len(active_features)}"
    )

    print(
        f"Selected features : "
        f"{len(selected_features)}"
    )

    print(
        f"Reduction         : "
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


    # ========================================================
    # GLOBAL CLASS COUNTS
    # ========================================================

    global_counts = (
        phase4
        .calculate_global_class_counts(

            label_mapping,

            args.chunk_size,
        )
    )


    # ========================================================
    # SAME PHASE-4 CLASS WEIGHTS
    # ========================================================

    global_weights = (
        phase4
        .calculate_global_class_weights(

            global_counts
        )
    )


    # ========================================================
    # CLIENT FILES / SAMPLE COUNTS
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
                "\nClient file missing:\n"
                f"{client_file}"
            )


        client_files[
            client_id
        ] = client_file


        sample_count = (
            phase4
            .count_client_samples(

                client_file,

                args.chunk_size,
            )
        )


        client_sample_counts[
            f"client_{client_id}"
        ] = sample_count


        logger.info(
            "Client %d samples = %d",
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

        seed=
            args.seed,
    )


    # ========================================================
    # MODEL / COMMUNICATION EFFICIENCY
    # ========================================================

    communication = (
        calculate_communication(

            baseline_model=
                reference_model,

            hybrid_model=
                global_model,

            rounds=
                args.rounds,
        )
    )


    del reference_model


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
        "PHASE 9 - REPEATED-SEED "
        "HYBRID EVO-GA + FEDAVG"
    )

    print(
        "=" * 90
    )


    print(
        f"Scenario      : "
        f"{args.scenario}"
    )

    print(
        f"Seed          : "
        f"{args.seed}"
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
        f"Features      : "
        f"{len(selected_features)}"
    )

    print(
        f"Classes       : "
        f"{len(label_mapping)}"
    )


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
            f"FEDERATED ROUND "
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

        client_training_times = []


        # ====================================================
        # LOCAL CLIENT TRAINING
        # ====================================================

        for client_id in range(
            1,
            N_CLIENTS + 1,
        ):

            logger.info(
                "Round %d | Client %d | "
                "Hybrid features=%d",
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
                local_training_seconds,

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

                # ============================================
                # IMPORTANT:
                # HYBRID-SELECTED ORIGINAL SCALER INDICES
                # ============================================

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

                seed=
                    args.seed,
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


            client_training_times.append(
                local_training_seconds
            )


        # ====================================================
        # SAMPLE-WEIGHTED FEDAVG
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
        # ROUND WEIGHTED METRICS
        # ====================================================

        weight_array = np.asarray(

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
                    weight_array,
            )
        )


        weighted_accuracy = float(

            np.average(

                np.asarray(
                    client_accuracies,
                    dtype=np.float64,
                ),

                weights=
                    weight_array,
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
                client_training_times[0],

            "client_2_training_seconds":
                client_training_times[1],

            "client_3_training_seconds":
                client_training_times[2],

            "client_4_training_seconds":
                client_training_times[3],

            "client_5_training_seconds":
                client_training_times[4],

            "feature_count":
                len(
                    selected_features
                ),
        })


        logger.info(
            "ROUND %d COMPLETE | "
            "loss=%.6f | "
            "accuracy=%.6f | "
            "update_norm=%.6f | "
            "time=%.2fs",
            round_number,
            weighted_loss,
            weighted_accuracy,
            update_norm,
            round_seconds,
        )


    # ========================================================
    # TOTAL TRAINING TIME
    # ========================================================

    total_training_seconds = (

        time.perf_counter()

        -

        total_training_start
    )


    # ========================================================
    # FINAL LOCKED TEST EVALUATION
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
    # SAVE RESULTS
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

        mask_file=
            args.mask_file,

        result_dir=
            result_dir,

        artifact_dir=
            artifact_dir,
    )


    # ========================================================
    # PHASE 9 CROSS-METHOD COMPARISON
    # ========================================================
    #
    # IMPORTANT:
    # Do NOT compare this seed against the old Phase 4/5/6
    # single-run result files here.  Phase 9 statistics will
    # compare matching seeds across all five methods after all
    # repeated-seed runs are complete.

    comparison_df = pd.DataFrame()
    delta_vs_phase4 = None
    delta_vs_evo = None
    delta_vs_ga = None


    # ========================================================
    # GENERATE PHASE-4 STYLE GRAPHS
    # ========================================================

    phase4.generate_graphs(

        scenario=
            (
                "HybridEVO-GA+FedAvg-"
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
        "PHASE 9 RUN COMPLETE — "
        "HYBRID EVO-GA + FEDAVG"
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
        f"Phase-4 features    : "
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
        f"Phase-4 parameters  : "
        f"{communication['phase4_parameter_count']:,}"
    )

    print(
        f"Phase-7 parameters  : "
        f"{communication['phase7_parameter_count']:,}"
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
        f"Phase-7 payload     : "
        f"{communication['phase7_model_payload_mib']:.4f} MiB"
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
        f"Hybrid total comm.  : "
        f"{communication['phase7_total_communication_mib']:.4f} MiB"
    )


    # ========================================================
    # PER-CLASS
    # ========================================================

    print()

    print(
        "-" * 90
    )

    print(
        "PER-CLASS HYBRID EVO-GA + FEDAVG"
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


        print()


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


            print(
                "Mean knowledge-transfer gain: "
                f"{mean_transfer:.6f}"
            )


    # ========================================================
    # PHASE 4 VS HYBRID
    # ========================================================

    if delta_vs_phase4 is not None:

        print()

        print(
            "-" * 90
        )

        print(
            "PHASE 4 VS PHASE 7 HYBRID"
        )

        print(
            "-" * 90
        )


        for key, value in (
            delta_vs_phase4.items()
        ):

            if (
                key.endswith(
                    "_delta"
                )
                and
                value is not None
            ):

                print(
                    f"{key:32s}: "
                    f"{value:+.6f}"
                )


        print()


        print(
            f"Feature reduction       : "
            f"{delta_vs_phase4['feature_reduction_ratio'] * 100:.2f}%"
        )

        print(
            f"Parameter reduction     : "
            f"{delta_vs_phase4['parameter_reduction_ratio'] * 100:.2f}%"
        )

        print(
            f"Communication reduction : "
            f"{delta_vs_phase4['communication_reduction_ratio'] * 100:.2f}%"
        )


    # ========================================================
    # EVO VS HYBRID
    # ========================================================

    if delta_vs_evo is not None:

        print()

        print(
            "-" * 90
        )

        print(
            "PHASE 5B EVO VS PHASE 7 HYBRID"
        )

        print(
            "-" * 90
        )


        for key, value in (
            delta_vs_evo.items()
        ):

            print(
                f"{key:32s}: "
                f"{value:+.6f}"
            )


    # ========================================================
    # GA VS HYBRID
    # ========================================================

    if delta_vs_ga is not None:

        print()

        print(
            "-" * 90
        )

        print(
            "PHASE 6 GA VS PHASE 7 HYBRID"
        )

        print(
            "-" * 90
        )


        for key, value in (
            delta_vs_ga.items()
        ):

            print(
                f"{key:32s}: "
                f"{value:+.6f}"
            )


    # ========================================================
    # PHASE 9 COMPARISON NOTE
    # ========================================================

    print()

    print(
        "-" * 90
    )

    print(
        "PHASE 9 CROSS-METHOD COMPARISON"
    )

    print(
        "-" * 90
    )

    print(
        "Deferred to phase9_statistics.py after all "
        "five methods and all seeds are complete."
    )


    # ========================================================
    # SELECTED FEATURES
    # ========================================================

    print()

    print(
        "-" * 90
    )

    print(
        "HYBRID EVO-GA SELECTED FEATURES"
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
        "Hybrid EVO-GA mask loaded: YES ✅"
    )

    print(
        "EVO rerun during FedAvg: NO ✅"
    )

    print(
        "GA rerun during FedAvg: NO ✅"
    )

    print(
        "Hybrid search rerun during FedAvg: NO ✅"
    )

    print(
        "Same Phase-4 scaler: YES ✅"
    )

    print(
        "Same Phase-4 MLP: YES ✅"
    )

    print(
        "Same Phase-4 optimizer: YES ✅"
    )

    print(
        "Same Phase-4 global class weights: YES ✅"
    )

    print(
        "Same sample-weighted FedAvg: YES ✅"
    )

    print(
        "Same non-IID client files: YES ✅"
    )

    print(
        "Same federated rounds: YES ✅"
    )

    print(
        "Same local epochs: YES ✅"
    )

    print(
        "Locked test used during training: NO ✅"
    )

    print(
        "Evaluation performed only after final round: YES ✅"
    )


    print()


    print(
        f"Hybrid mask:\n"
        f"{args.mask_file}"
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