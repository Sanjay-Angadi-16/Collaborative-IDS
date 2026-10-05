"""
Phase 6 - Federated-Aware GA + FedAvg
=====================================

Purpose:

    Phase 4 baseline:
        FedAvg + 70 active features

    Phase 6:
        Federated-aware GA
            ↓
        frozen GA feature mask
            ↓
        FedAvg with selected features

IMPORTANT:
    This script DOES NOT rerun GA.

It loads:

    results/phase6_ga/ga_feature_mask.csv

Then runs the same Phase-4 federated pipeline:

    - Same 5 clients
    - Same severe non-IID scenario
    - Same scaler
    - Same label mapping
    - Same MLP
    - Same class weighting
    - Same optimizer
    - Same batch size
    - Same learning rate
    - Same weight decay
    - Same sample-weighted FedAvg
    - Same 10 rounds
    - Same 1 local epoch
    - Locked test only after final round

Outputs are isolated under:

    results/ga_fedavg/<scenario>/
    artifacts/ga_fedavg/<scenario>/
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
    sys.path.insert(0, str(PROJECT_ROOT))

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


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
# TRAINING DEFAULTS
# ============================================================

DEFAULT_ROUNDS = phase4.DEFAULT_ROUNDS
DEFAULT_LOCAL_EPOCHS = phase4.DEFAULT_LOCAL_EPOCHS
DEFAULT_BATCH_SIZE = phase4.DEFAULT_BATCH_SIZE
DEFAULT_CHUNK_SIZE = phase4.DEFAULT_CHUNK_SIZE
DEFAULT_LEARNING_RATE = phase4.DEFAULT_LEARNING_RATE
DEFAULT_WEIGHT_DECAY = phase4.DEFAULT_WEIGHT_DECAY


# ============================================================
# PHASE 6 GA MASK
# ============================================================

PHASE6_DIR = (
    PROJECT_ROOT
    / "results"
    / "phase6_ga"
)

DEFAULT_MASK_FILE = (
    PHASE6_DIR
    / "ga_feature_mask.csv"
)

PHASE6_SUMMARY_FILE = (
    PHASE6_DIR
    / "phase6_ga_summary.json"
)


# ============================================================
# OUTPUT
# ============================================================

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ga_fedavg"
)

ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "ga_fedavg"
)


# ============================================================
# PHASE 4 BASELINE
# ============================================================

PHASE4_RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "fedavg"
)


# ============================================================
# OPTIONAL EVO RESULT
# ============================================================

PHASE5B_RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "federated_evo_fedavg"
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(
    "phase6_ga_fedavg"
)


# ============================================================
# HASH FILE
# ============================================================

def sha256_file(path: Path) -> str:

    digest = hashlib.sha256()

    with open(path, "rb") as file:

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
# LOAD GA MASK
# ============================================================

def load_ga_mask(
    mask_file,
    active_features,
    active_indices,
    feature_columns,
):

    if not mask_file.exists():

        raise FileNotFoundError(
            "\nGA mask not found:\n"
            f"{mask_file}\n\n"
            "Run first:\n"
            "python scripts\\phase6_ga_test.py"
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
        set(mask_df.columns)
    )

    if missing:

        raise ValueError(
            "\nInvalid GA feature mask.\n"
            f"Missing columns: {missing}"
        )

    mask_df["feature_name"] = (
        mask_df["feature_name"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Verify Phase-4 active space
    # --------------------------------------------------------

    if len(active_features) != EXPECTED_PHASE4_FEATURES:

        raise ValueError(
            "\nExpected 70 Phase-4 active features.\n"
            f"Found: {len(active_features)}"
        )

    if len(mask_df) != EXPECTED_PHASE4_FEATURES:

        raise ValueError(
            "\nGA mask must contain exactly 70 rows.\n"
            f"Found: {len(mask_df)}"
        )

    # --------------------------------------------------------
    # Schema validation
    # --------------------------------------------------------

    mask_features = set(
        mask_df["feature_name"]
    )

    active_set = set(
        active_features
    )

    missing_from_mask = (
        active_set
        -
        mask_features
    )

    extra_in_mask = (
        mask_features
        -
        active_set
    )

    if missing_from_mask or extra_in_mask:

        raise ValueError(
            "\nGA mask feature schema differs from Phase 4.\n\n"
            f"Missing:\n{sorted(missing_from_mask)}\n\n"
            f"Extra:\n{sorted(extra_in_mask)}"
        )

    # --------------------------------------------------------
    # Selection map
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

        if value not in {0, 1}
    }

    if invalid_values:

        raise ValueError(
            "selected column must contain only 0 and 1."
        )

    # --------------------------------------------------------
    # Preserve Phase-4 order
    # --------------------------------------------------------

    selected_active_positions = [

        index

        for index, feature
        in enumerate(active_features)

        if selection_map[feature] == 1
    ]

    selected_features = [

        active_features[index]

        for index
        in selected_active_positions
    ]

    if not selected_features:

        raise ValueError(
            "GA selected zero features."
        )

    # --------------------------------------------------------
    # Active position -> original scaler index
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

    reconstructed = [

        feature_columns[
            int(index)
        ]

        for index
        in selected_original_indices
    ]

    if reconstructed != selected_features:

        raise ValueError(
            "\nGA feature-index mapping failed.\n\n"
            f"Expected:\n{selected_features}\n\n"
            f"Reconstructed:\n{reconstructed}"
        )

    logger.info(
        "GA feature mask verified."
    )

    logger.info(
        "Original features : %d",
        len(active_features),
    )

    logger.info(
        "Selected features : %d",
        len(selected_features),
    )

    logger.info(
        "Feature reduction : %.2f%%",
        (
            1.0
            -
            len(selected_features)
            /
            len(active_features)
        )
        * 100.0,
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
):

    """
    Recreate the same seed-42 full Phase-4 model, then project
    its first-layer columns to the GA-selected features.

    This keeps initialization comparable across experiments.
    """

    phase4.set_seed(
        SEED
    )

    reference_model = phase4.IDSMLP(

        input_dim=
            full_input_dim,

        num_classes=
            num_classes,

    ).to(device)

    reference_state = {

        key:
            value
            .detach()
            .clone()

        for key, value
        in reference_model.state_dict().items()
    }

    selected_model = phase4.IDSMLP(

        input_dim=
            selected_input_dim,

        num_classes=
            num_classes,

    ).to(device)

    selected_state = (
        selected_model.state_dict()
    )

    first_layer_key = (
        "network.0.weight"
    )

    positions = [

        int(index)

        for index
        in selected_active_positions
    ]

    for key in selected_state.keys():

        if key == first_layer_key:

            selected_state[key] = (

                reference_state[key][
                    :,
                    positions
                ]
                .clone()
            )

        else:

            selected_state[key] = (
                reference_state[key]
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
# PARAMETER COUNT
# ============================================================

def parameter_count(model):

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

def model_payload_bytes(model):

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
    ga_model,
    rounds,
):

    baseline_params = (
        parameter_count(
            baseline_model
        )
    )

    ga_params = (
        parameter_count(
            ga_model
        )
    )

    baseline_payload = (
        model_payload_bytes(
            baseline_model
        )
    )

    ga_payload = (
        model_payload_bytes(
            ga_model
        )
    )

    baseline_per_round = (

        2
        *
        N_CLIENTS
        *
        baseline_payload
    )

    ga_per_round = (

        2
        *
        N_CLIENTS
        *
        ga_payload
    )

    baseline_total = (

        baseline_per_round
        *
        rounds
    )

    ga_total = (

        ga_per_round
        *
        rounds
    )

    return {

        "phase4_parameter_count":
            baseline_params,

        "phase6_parameter_count":
            ga_params,

        "parameter_reduction_ratio":
            float(
                1.0
                -
                ga_params
                /
                baseline_params
            ),

        "phase4_model_payload_bytes":
            baseline_payload,

        "phase6_model_payload_bytes":
            ga_payload,

        "phase4_model_payload_mib":
            float(
                baseline_payload
                /
                1024 ** 2
            ),

        "phase6_model_payload_mib":
            float(
                ga_payload
                /
                1024 ** 2
            ),

        "phase4_total_communication_bytes":
            baseline_total,

        "phase6_total_communication_bytes":
            ga_total,

        "phase4_total_communication_mib":
            float(
                baseline_total
                /
                1024 ** 2
            ),

        "phase6_total_communication_mib":
            float(
                ga_total
                /
                1024 ** 2
            ),

        "communication_reduction_ratio":
            float(
                1.0
                -
                ga_total
                /
                baseline_total
            ),

        "assumption":
            (
                "Full float32 model upload and download "
                "for every client in every round; "
                "protocol/compression overhead excluded."
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

    for class_name in class_names:

        metrics = report[
            class_name
        ]

        rows.append({

            "class":
                class_name,

            "precision":
                float(
                    metrics["precision"]
                ),

            "recall":
                float(
                    metrics["recall"]
                ),

            "f1_score":
                float(
                    metrics["f1-score"]
                ),

            "support":
                int(
                    metrics["support"]
                ),
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# LOAD JSON SAFELY
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
# SAVE BASELINE / EVO / GA COMPARISON
# ============================================================

def save_comparison(
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

    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    ]

    rows = []

    if phase4_metrics is not None:

        rows.append({

            "method":
                "Phase4_FedAvg_70",

            "features":
                EXPECTED_PHASE4_FEATURES,

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
        })

    if evo_metrics is not None:

        evo_feature_count = (
            evo_metrics.get(
                "selected_feature_count",
                36,
            )
        )

        rows.append({

            "method":
                "Phase5B_FederatedEVO_FedAvg",

            "features":
                evo_feature_count,

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
        })

    rows.append({

        "method":
            "Phase6_FederatedGA_FedAvg",

        "features":
            selected_feature_count,

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
    })

    comparison_df = pd.DataFrame(
        rows
    )

    comparison_df.to_csv(

        result_dir
        / "fedavg_method_comparison.csv",

        index=False,
    )

    delta_vs_phase4 = None

    if phase4_metrics is not None:

        delta_vs_phase4 = {}

        for metric in metrics:

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
            / "phase4_vs_phase6_delta.json",

            "w",
            encoding="utf-8",

        ) as file:

            json.dump(
                delta_vs_phase4,
                file,
                indent=4,
            )

    delta_vs_evo = None

    if evo_metrics is not None:

        delta_vs_evo = {}

        for metric in metrics:

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
            / "phase5b_evo_vs_phase6_ga_delta.json",

            "w",
            encoding="utf-8",

        ) as file:

            json.dump(
                delta_vs_evo,
                file,
                indent=4,
            )

    return (
        comparison_df,
        delta_vs_phase4,
        delta_vs_evo,
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
    # MODEL
    # --------------------------------------------------------

    torch.save(

        {

            "phase":
                6,

            "algorithm":
                "Federated-aware GA + FedAvg",

            "model_state_dict":
                global_model.state_dict(),

            "input_dim":
                len(selected_features),

            "num_classes":
                len(label_mapping),

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

            "mask_file":
                str(mask_file),

            "mask_sha256":
                sha256_file(
                    mask_file
                ),
        },

        artifact_dir
        / "ga_fedavg_model.pt",
    )

    # --------------------------------------------------------
    # ROUND HISTORY
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
    # OVERALL
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
    # PER CLASS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # CONFUSION MATRIX
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
    # SELECTED FEATURES
    # --------------------------------------------------------

    pd.DataFrame({

        "selected_order":
            np.arange(
                1,
                len(selected_features) + 1,
            ),

        "active_feature_index":
            selected_active_positions,

        "original_scaler_index":
            selected_original_indices,

        "feature_name":
            selected_features,

    }).to_csv(

        result_dir
        / "selected_ga_features.csv",

        index=False,
    )

    # --------------------------------------------------------
    # COMMUNICATION
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
    # MANIFEST
    # --------------------------------------------------------

    manifest = {

        "phase":
            6,

        "algorithm":
            "Federated-aware GA + FedAvg",

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

        "learning_rate":
            args.learning_rate,

        "weight_decay":
            args.weight_decay,

        "aggregation":
            "sample-weighted FedAvg",

        "original_features":
            len(active_features),

        "selected_features":
            len(selected_features),

        "feature_reduction_ratio":
            float(

                1.0

                -
                len(selected_features)
                /
                len(active_features)
            ),

        "selected_feature_names":
            selected_features,

        "feature_mask_file":
            str(mask_file),

        "feature_mask_sha256":
            sha256_file(
                mask_file
            ),

        "ga_rerun_during_fedavg":
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
        / "phase6_manifest.json",

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

    parser.add_argument(
        "--scenario",
        choices=[
            "iid",
            "non_iid",
            "controlled_non_iid",
        ],
        default="non_iid",
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_ROUNDS,
    )

    parser.add_argument(
        "--local-epochs",
        type=int,
        default=DEFAULT_LOCAL_EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )

    parser.add_argument(
        "--mask-file",
        type=Path,
        default=DEFAULT_MASK_FILE,
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
            f"Missing scenario:\n"
            f"{scenario_dir}"
        )

    if not phase4.LOCKED_TEST_FILE.exists():

        raise FileNotFoundError(
            f"Locked test missing:\n"
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
    # SAME PHASE-4 PREPROCESSING
    # ========================================================

    (
        scaler,
        feature_columns,
        label_mapping,
        active_indices,
        active_features,

    ) = phase4.load_preprocessing()

    if len(active_features) != EXPECTED_PHASE4_FEATURES:

        raise ValueError(
            "\nExpected exactly 70 Phase-4 active features."
        )


    # ========================================================
    # LOAD GA MASK
    # ========================================================

    (
        selected_features,
        selected_active_positions,
        selected_original_indices,

    ) = load_ga_mask(

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
        len(selected_features)
        /
        len(active_features)
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    print()
    print("=" * 90)

    print(
        "PHASE 6 - FEDERATED-AWARE GA FEATURE SET"
    )

    print("=" * 90)

    print(
        f"Original features : {len(active_features)}"
    )

    print(
        f"Selected features : {len(selected_features)}"
    )

    print(
        f"Reduction         : {feature_reduction * 100:.2f}%"
    )

    print()

    for number, feature in enumerate(
        selected_features,
        start=1,
    ):

        print(
            f"{number:02d}. {feature}"
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
            len(active_features),

        selected_input_dim=
            len(selected_features),

        num_classes=
            len(label_mapping),

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

            ga_model=
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
    print("=" * 90)

    print(
        "PHASE 6 - FEDERATED-AWARE GA + FEDAVG"
    )

    print("=" * 90)

    print(
        f"Scenario      : {args.scenario}"
    )

    print(
        f"Clients       : {N_CLIENTS}"
    )

    print(
        f"Rounds        : {args.rounds}"
    )

    print(
        f"Local epochs  : {args.local_epochs}"
    )

    print(
        f"Features      : {len(selected_features)}"
    )

    print(
        f"Classes       : {len(label_mapping)}"
    )


    # ========================================================
    # FEDERATED ROUNDS
    # ========================================================

    for round_number in range(
        1,
        args.rounds + 1,
    ):

        print()
        print("-" * 90)

        print(
            f"FEDERATED ROUND "
            f"{round_number}/"
            f"{args.rounds}"
        )

        print("-" * 90)

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
            in global_model.state_dict().items()
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
                "training with %d GA-selected features",
                round_number,
                client_id,
                len(selected_features),
            )

            (
                local_state,
                local_loss,
                local_accuracy,
                training_seconds,

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

                # GA-selected original scaler indices
                active_indices=
                    selected_original_indices,

                label_mapping=
                    label_mapping,

                class_weights=
                    global_weights,

                input_dim=
                    len(selected_features),

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
                training_seconds
            )


        # ====================================================
        # FEDAVG
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
        # UPDATE NORM
        # ====================================================

        update_norm = (
            phase4.state_update_norm(

                old_global_state,

                aggregated_state,
            )
        )


        # ====================================================
        # WEIGHTED LOCAL METRICS
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

            "feature_count":
                len(selected_features),
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
    # TRAINING TIME
    # ========================================================

    total_training_seconds = (

        time.perf_counter()
        -
        total_training_start
    )


    # ========================================================
    # FINAL LOCKED-TEST EVALUATION
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
    # COMPARISON
    # ========================================================

    (
        comparison_df,
        delta_vs_phase4,
        delta_vs_evo,

    ) = save_comparison(

        scenario=
            args.scenario,

        overall=
            overall,

        selected_feature_count=
            len(selected_features),

        communication=
            communication,

        result_dir=
            result_dir,
    )


    # ========================================================
    # GRAPHS
    # ========================================================

    phase4.generate_graphs(

        scenario=
            (
                "FederatedGA+FedAvg-"
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
    print("=" * 90)

    print(
        "PHASE 6 COMPLETE — "
        "FEDERATED-AWARE GA + FEDAVG"
    )

    print("=" * 90)

    print(
        f"Scenario            : {args.scenario}"
    )

    print(
        f"Device              : {device}"
    )

    print(
        f"Federated rounds    : {args.rounds}"
    )

    print(
        f"Local epochs/round  : {args.local_epochs}"
    )

    print(
        f"Phase-4 features    : {len(active_features)}"
    )

    print(
        f"Selected features   : {len(selected_features)}"
    )

    print(
        f"Feature reduction   : {feature_reduction * 100:.2f}%"
    )

    print(
        f"Classes             : {len(label_mapping)}"
    )

    print()

    print(
        f"Accuracy            : {overall['accuracy']:.6f}"
    )

    print(
        f"Balanced Accuracy   : {overall['balanced_accuracy']:.6f}"
    )

    print(
        f"Macro Precision     : {overall['macro_precision']:.6f}"
    )

    print(
        f"Macro Recall        : {overall['macro_recall']:.6f}"
    )

    print(
        f"Macro F1            : {overall['macro_f1']:.6f}"
    )

    print(
        f"Weighted F1         : {overall['weighted_f1']:.6f}"
    )

    print()

    print(
        f"Training time       : {total_training_seconds:.2f}s"
    )

    print(
        f"Inference time      : {inference_seconds:.2f}s"
    )


    # ========================================================
    # COMMUNICATION
    # ========================================================

    print()
    print("-" * 90)

    print(
        "MODEL / COMMUNICATION EFFICIENCY"
    )

    print("-" * 90)

    print(
        f"Phase-4 parameters  : "
        f"{communication['phase4_parameter_count']:,}"
    )

    print(
        f"Phase-6 parameters  : "
        f"{communication['phase6_parameter_count']:,}"
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
        f"Phase-6 payload     : "
        f"{communication['phase6_model_payload_mib']:.4f} MiB"
    )

    print(
        f"Comm. reduction     : "
        f"{communication['communication_reduction_ratio'] * 100:.2f}%"
    )


    # ========================================================
    # PER CLASS
    # ========================================================

    print()
    print("-" * 90)

    print(
        "PER-CLASS FEDERATED-AWARE GA + FEDAVG"
    )

    print("-" * 90)

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
        print("-" * 90)

        print(
            "KNOWLEDGE TRANSFER"
        )

        print("-" * 90)

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
    # DELTA VS PHASE 4
    # ========================================================

    if delta_vs_phase4 is not None:

        print()
        print("-" * 90)

        print(
            "PHASE 4 VS PHASE 6"
        )

        print("-" * 90)

        for key, value in delta_vs_phase4.items():

            if (
                key.endswith("_delta")
                and
                value is not None
            ):

                print(
                    f"{key:32s}: {value:+.6f}"
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
    # DELTA VS EVO
    # ========================================================

    if delta_vs_evo is not None:

        print()
        print("-" * 90)

        print(
            "PHASE 5B EVO VS PHASE 6 GA"
        )

        print("-" * 90)

        for key, value in delta_vs_evo.items():

            print(
                f"{key:32s}: {value:+.6f}"
            )


    # ========================================================
    # FEATURES
    # ========================================================

    print()
    print("-" * 90)

    print(
        "FEDERATED-AWARE GA FEATURES"
    )

    print("-" * 90)

    for number, feature in enumerate(
        selected_features,
        start=1,
    ):

        print(
            f"{number:02d}. {feature}"
        )


    # ========================================================
    # INTEGRITY
    # ========================================================

    print()
    print("=" * 90)

    print(
        "EXPERIMENTAL INTEGRITY"
    )

    print("=" * 90)

    print(
        "Federated-aware GA mask loaded: YES ✅"
    )

    print(
        "GA rerun during FedAvg: NO ✅"
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
        "Same Phase-4 class weights: YES ✅"
    )

    print(
        "Same sample-weighted FedAvg: YES ✅"
    )

    print(
        "Same client files: YES ✅"
    )

    print(
        "Locked test used during training: NO ✅"
    )

    print(
        "Locked test used only after final round: YES ✅"
    )

    print()

    print(
        f"Results:\n{result_dir}"
    )

    print()

    print(
        f"Model:\n{artifact_dir}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()