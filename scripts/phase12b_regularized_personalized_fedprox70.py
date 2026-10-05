r"""
Phase 12B - Knowledge-Preserving Personalized FedProx70
========================================================

Background
----------
Phase 11:
    FedProx70 transfers unseen attack knowledge,
    but reduces local specialization.

Phase 12A:
    Naive local fine-tuning restores local specialization,
    but catastrophically forgets global knowledge.

Phase 12B:
    Personalize FedProx70 while anchoring the personalized
    model to the original global FedProx70 parameters.

Objective
---------
For client k:

    Loss =
        CrossEntropy(local_data)
        +
        lambda / 2 * || w_personalized - w_global ||^2

Where:
    w_personalized = client-personalized model parameters
    w_global       = frozen Phase-10 FedProx70 parameters


Validation search
-----------------
Lambdas:
    0.001
    0.01
    0.1
    1.0

Epoch budgets:
    1
    2
    3
    5

The locked test is NOT used for model selection.

Candidate constraint:
    Validation knowledge preservation >= 75%

AND:
    Personalized seen recall >= Global seen recall

Knowledge preservation:
    Personalized unseen recall
    --------------------------
       Global unseen recall


Validation selection score
--------------------------
    0.50 * Harmonic(Seen Recall, Unseen Recall)
    +
    0.25 * Macro F1
    +
    0.25 * Balanced Accuracy


Final test metrics
------------------
Personalization Gain =
    Personalized Seen Recall
    -
    Global Seen Recall

Global Knowledge Loss =
    Global Unseen Recall
    -
    Personalized Unseen Recall

Knowledge Preservation Ratio =
    Personalized Unseen Recall
    /
    Global Unseen Recall

Local Recovery Ratio =
    Personalized Seen Recall
    /
    LocalOnly Seen Recall


Required prior results
----------------------
Phase 10:
artifacts/phase10/fedprox70/<scenario>/seed_<seed>/fedprox70_model.pt

Phase 11:
results/phase11/knowledge_retention/<scenario>/seed_<seed>/
    local_only_per_class_metrics.csv


Outputs
-------
results/phase12b/regularized_personalized_fedprox70/<scenario>/seed_<seed>/

    validation_sweep.csv
    regularized_training_history.csv
    selected_training_history.csv
    client_training_distribution.csv

    final_regularized_personalization_results.csv
    final_regularized_personalized_per_class_metrics.csv

    phase12b_summary.json


Artifacts
---------
artifacts/phase12b/regularized_personalized_fedprox70/<scenario>/seed_<seed>/

    client_1/
        regularized_personalized_fedprox70_model.pt

    ...

Aggregate
---------
results/phase12b/regularized_personalized_fedprox70/<scenario>/aggregate/

    final_regularized_personalization_all_seeds.csv
    final_regularized_per_class_all_seeds.csv
    validation_sweep_all_seeds.csv

    selected_hyperparameter_frequency.csv
    per_client_summary_across_seeds.csv
    seed_level_summary.csv
    seed_level_paired_tests.csv

    phase12b_final_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn


# ============================================================
# OPTIONAL SCIPY
# ============================================================

try:
    from scipy.stats import t as student_t
    from scipy.stats import wilcoxon

    SCIPY_AVAILABLE = True

except ImportError:
    student_t = None
    wilcoxon = None

    SCIPY_AVAILABLE = False


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

SCRIPT_DIR = Path(
    __file__
).resolve().parent


for path in (
    PROJECT_ROOT,
    SCRIPT_DIR,
):
    if str(path) not in sys.path:
        sys.path.insert(
            0,
            str(path),
        )


# ============================================================
# REUSE PHASE 9 INFRASTRUCTURE
# ============================================================

try:
    import phase9_fedavg70 as phase4

except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "\nMissing:\n"
        "scripts\\phase9_fedavg70.py\n\n"
        "Keep this script inside the scripts folder."
    ) from exc


# ============================================================
# CONFIG
# ============================================================

DEFAULT_SEED = phase4.SEED

N_CLIENTS = phase4.N_CLIENTS

EXPECTED_FEATURES = 70

DEFAULT_BATCH_SIZE = phase4.DEFAULT_BATCH_SIZE

DEFAULT_CHUNK_SIZE = phase4.DEFAULT_CHUNK_SIZE

DEFAULT_WEIGHT_DECAY = phase4.DEFAULT_WEIGHT_DECAY

DEFAULT_PERSONALIZATION_LR = 0.0001


DEFAULT_LAMBDAS = [
    0.001,
    0.01,
    0.1,
    1.0,
]


DEFAULT_EPOCH_BUDGETS = [
    1,
    2,
    3,
    5,
]


DEFAULT_MIN_KNOWLEDGE_PRESERVATION = 0.75


GRADIENT_CLIP = 5.0


# ============================================================
# PATHS
# ============================================================

PHASE10_MODEL_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "phase10"
    / "fedprox70"
)


PHASE11_RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "phase11"
    / "knowledge_retention"
)


RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "phase12b"
    / "regularized_personalized_fedprox70"
)


ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "phase12b"
    / "regularized_personalized_fedprox70"
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
    "phase12b_regularized_personalized_fedprox70"
)


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed: int) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(
        torch.backends,
        "cudnn",
    ):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# JSON HELPERS
# ============================================================

def json_safe(value: Any):

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(
        value,
        np.integer,
    ):
        return int(value)

    if isinstance(
        value,
        np.floating,
    ):
        if np.isnan(value):
            return None

        return float(value)

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(value)

    if (
        isinstance(value, float)
        and
        math.isnan(value)
    ):
        return None

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
# TORCH LOAD
# ============================================================

def load_torch(
    path: Path,
    device,
):

    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )

    except TypeError:
        return torch.load(
            path,
            map_location=device,
        )


# ============================================================
# SAFE RATIOS
# ============================================================

def safe_ratio(
    numerator: float,
    denominator: float,
):

    if (
        not np.isfinite(denominator)
        or
        np.isclose(
            denominator,
            0.0,
        )
    ):
        return np.nan

    return float(
        numerator
        /
        denominator
    )


def knowledge_preservation_ratio(
    personalized_unseen: float,
    global_unseen: float,
):

    # If the global model itself has zero unseen recall,
    # personalization cannot destroy that knowledge.

    if np.isclose(
        global_unseen,
        0.0,
    ):

        if personalized_unseen >= global_unseen:
            return 1.0

        return 0.0

    return float(
        personalized_unseen
        /
        global_unseen
    )


# ============================================================
# HARMONIC MEAN
# ============================================================

def harmonic_mean(
    value_a: float,
    value_b: float,
) -> float:

    denominator = (
        value_a
        +
        value_b
    )

    if np.isclose(
        denominator,
        0.0,
    ):
        return 0.0

    return float(
        2.0
        *
        value_a
        *
        value_b
        /
        denominator
    )


# ============================================================
# VALIDATION SELECTION SCORE
# ============================================================

def calculate_selection_score(
    seen_recall: float,
    unseen_recall: float,
    macro_f1: float,
    balanced_accuracy: float,
) -> float:

    balance = harmonic_mean(
        seen_recall,
        unseen_recall,
    )

    return float(
        0.50
        *
        balance
        +
        0.25
        *
        macro_f1
        +
        0.25
        *
        balanced_accuracy
    )


# ============================================================
# RESOLVE VALIDATION FILE
# ============================================================

def resolve_validation_file(
    user_path: Path | None,
) -> Path:

    if user_path is not None:

        path = user_path.expanduser()

        if not path.is_absolute():
            path = (
                PROJECT_ROOT
                /
                path
            )

        path = path.resolve()

        if not path.exists():
            raise FileNotFoundError(
                "\nValidation file not found:\n"
                f"{path}"
            )

        if (
            "test_locked"
            in
            path.name.lower()
        ):
            raise ValueError(
                "\nLocked test cannot be used "
                "for personalization selection."
            )

        return path


    candidates = [
        PROJECT_ROOT
        / "data"
        / "processed"
        / "validation.csv",

        PROJECT_ROOT
        / "data"
        / "validation.csv",

        PROJECT_ROOT
        / "validation.csv",
    ]


    candidates = [
        path.resolve()
        for path in candidates
        if path.exists()
    ]


    if len(candidates) == 1:
        return candidates[0]


    if not candidates:
        raise FileNotFoundError(
            "\nvalidation.csv was not found.\n\n"
            "Use:\n"
            "--validation-file data\\processed\\validation.csv"
        )


    raise RuntimeError(
        "\nMultiple validation files found.\n"
        "Pass the correct file explicitly."
    )


# ============================================================
# VERIFY 70 FEATURES
# ============================================================

def verify_70_features(
    active_features,
    active_indices,
    feature_columns,
):

    if len(active_features) != EXPECTED_FEATURES:
        raise ValueError(
            "\nExpected 70 active features.\n"
            f"Found: {len(active_features)}"
        )


    active_indices_array = np.asarray(
        active_indices,
        dtype=np.int64,
    )


    if len(active_indices_array) != EXPECTED_FEATURES:
        raise ValueError(
            "\nExpected 70 active indices."
        )


    reconstructed = [
        feature_columns[int(index)]
        for index in active_indices_array
    ]


    if list(reconstructed) != list(active_features):
        raise ValueError(
            "\nFeature/index mismatch."
        )


    return (
        list(active_features),
        active_indices_array,
    )


# ============================================================
# DETECT BENIGN LABEL
# ============================================================

def detect_benign_label(
    label_mapping,
):

    labels = list(
        label_mapping.keys()
    )


    for preferred in (
        "BENIGN",
        "Benign",
        "benign",
        "NORMAL",
        "Normal",
        "normal",
    ):

        if preferred in labels:
            return preferred


    for label in labels:

        normalized = (
            str(label)
            .strip()
            .lower()
        )

        if (
            "benign" in normalized
            or
            "normal" in normalized
        ):
            return label


    raise ValueError(
        "\nCould not identify benign class."
    )


# ============================================================
# CLIENT CLASS COUNTS
# ============================================================

def read_client_label_counts(
    client_file: Path,
    chunk_size: int,
    label_mapping,
):

    canonical = {
        str(label)
        .strip()
        .lower():
            str(label).strip()

        for label
        in label_mapping.keys()
    }


    counts = {
        str(label).strip(): 0
        for label
        in label_mapping.keys()
    }


    reader = pd.read_csv(
        client_file,
        usecols=[
            phase4.LABEL_COL
        ],
        chunksize=chunk_size,
    )


    for chunk in reader:

        values = (
            chunk[
                phase4.LABEL_COL
            ]
            .astype(str)
            .str.strip()
            .value_counts()
        )


        for raw_label, count in values.items():

            key = (
                str(raw_label)
                .strip()
                .lower()
            )


            if key not in canonical:
                raise ValueError(
                    "\nUnknown client label:\n"
                    f"{raw_label}"
                )


            counts[
                canonical[key]
            ] += int(count)


    return counts


# ============================================================
# LOAD PHASE 10 GLOBAL CHECKPOINT
# ============================================================

def load_global_checkpoint(
    scenario: str,
    seed: int,
    device,
    active_features,
):

    model_file = (
        PHASE10_MODEL_ROOT
        / scenario
        / f"seed_{seed}"
        / "fedprox70_model.pt"
    )


    if not model_file.exists():
        raise FileNotFoundError(
            "\nPhase 10 FedProx70 model missing:\n"
            f"{model_file}"
        )


    checkpoint = load_torch(
        model_file,
        device,
    )


    checkpoint_seed = checkpoint.get(
        "seed"
    )


    if (
        checkpoint_seed is not None
        and
        int(checkpoint_seed) != int(seed)
    ):
        raise ValueError(
            "Phase 10 checkpoint seed mismatch."
        )


    checkpoint_features = checkpoint.get(
        "features"
    )


    if (
        checkpoint_features is not None
        and
        list(checkpoint_features)
        !=
        list(active_features)
    ):
        raise ValueError(
            "Phase 10 feature schema mismatch."
        )


    if (
        "model_state_dict"
        not in checkpoint
    ):
        raise KeyError(
            "Checkpoint missing model_state_dict."
        )


    return checkpoint, model_file


# ============================================================
# LOAD PHASE 11 LOCAL REFERENCE
# ============================================================

def load_phase11_local_metrics(
    scenario: str,
    seed: int,
):

    path = (
        PHASE11_RESULT_ROOT
        / scenario
        / f"seed_{seed}"
        / "local_only_per_class_metrics.csv"
    )


    if not path.exists():
        raise FileNotFoundError(
            "\nPhase 11 metrics missing:\n"
            f"{path}"
        )


    dataframe = pd.read_csv(
        path
    )


    if (
        "f1" in dataframe.columns
        and
        "f1_score" not in dataframe.columns
    ):
        dataframe = dataframe.rename(
            columns={
                "f1": "f1_score"
            }
        )


    dataframe[
        "client_id"
    ] = dataframe[
        "client_id"
    ].astype(int)


    dataframe[
        "class"
    ] = (
        dataframe[
            "class"
        ]
        .astype(str)
        .str.strip()
    )


    return dataframe


# ============================================================
# BUILD MLP
# ============================================================

def build_model(
    state_dict,
    input_dim,
    num_classes,
    device,
):

    model = phase4.IDSMLP(
        input_dim=input_dim,
        num_classes=num_classes,
    ).to(device)


    model.load_state_dict(
        state_dict
    )


    return model


# ============================================================
# FILTERED VALIDATION EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_model_on_csv(
    model,
    csv_file: Path,
    scaler,
    feature_columns,
    active_indices,
    label_mapping,
    device,
    batch_size,
    chunk_size,
):

    model.eval()


    canonical = {
        str(label)
        .strip()
        .lower():
            str(label).strip()

        for label
        in label_mapping.keys()
    }


    y_true_parts = []
    y_pred_parts = []

    total_read = 0
    total_used = 0
    total_filtered = 0

    excluded_counts = {}


    started = time.perf_counter()


    reader = pd.read_csv(
        csv_file,
        chunksize=chunk_size,
    )


    for chunk in reader:

        total_read += len(chunk)


        if phase4.LABEL_COL not in chunk.columns:
            raise KeyError(
                f"Missing label column: "
                f"{phase4.LABEL_COL}"
            )


        labels = (
            chunk[
                phase4.LABEL_COL
            ]
            .astype(str)
            .str.strip()
        )


        lower_labels = (
            labels
            .str.lower()
        )


        keep_mask = lower_labels.isin(
            canonical.keys()
        )


        excluded = (
            labels[
                ~keep_mask
            ]
            .value_counts()
        )


        for label, count in excluded.items():

            excluded_counts[
                str(label)
            ] = (
                excluded_counts.get(
                    str(label),
                    0,
                )
                +
                int(count)
            )


        total_filtered += int(
            (~keep_mask).sum()
        )


        chunk = (
            chunk.loc[
                keep_mask
            ]
            .copy()
        )


        if chunk.empty:
            continue


        chunk[
            phase4.LABEL_COL
        ] = (
            chunk[
                phase4.LABEL_COL
            ]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(canonical)
        )


        total_used += len(chunk)


        X = phase4.transform_features(
            df=chunk,
            feature_columns=feature_columns,
            scaler=scaler,
            active_indices=active_indices,
        )


        y = phase4.encode_labels(
            chunk[
                phase4.LABEL_COL
            ],
            label_mapping,
        )


        if len(y) == 0:
            continue


        y_true_parts.append(
            y.astype(
                np.int64,
                copy=False,
            )
        )


        chunk_predictions = []


        for start in range(
            0,
            len(y),
            batch_size,
        ):

            X_batch = torch.from_numpy(
                X[
                    start:
                    start + batch_size
                ].astype(
                    np.float32,
                    copy=False,
                )
            ).to(device)


            logits = model(
                X_batch
            )


            predictions = (
                torch.argmax(
                    logits,
                    dim=1,
                )
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.int64,
                    copy=False,
                )
            )


            chunk_predictions.append(
                predictions
            )


        y_pred_parts.append(
            np.concatenate(
                chunk_predictions
            )
        )


    if not y_true_parts:
        raise RuntimeError(
            "Validation produced zero "
            "six-class samples."
        )


    logger.info(
        "Validation rows read     : %d",
        total_read,
    )

    logger.info(
        "Validation rows retained : %d",
        total_used,
    )

    logger.info(
        "Validation rows excluded : %d",
        total_filtered,
    )


    if excluded_counts:

        logger.info(
            "Excluded classes      : %s",
            excluded_counts,
        )


    y_true = np.concatenate(
        y_true_parts
    )

    y_pred = np.concatenate(
        y_pred_parts
    )


    (
        overall,
        report,
        confusion_matrix,
        class_names,
    ) = phase4.calculate_metrics(
        y_true,
        y_pred,
        label_mapping,
    )


    return (
        float(
            time.perf_counter()
            -
            started
        ),
        overall,
        report,
        confusion_matrix,
        class_names,
    )


# ============================================================
# LOCKED TEST EVALUATION
# ============================================================

def evaluate_locked_test(
    model,
    scaler,
    feature_columns,
    active_indices,
    label_mapping,
    device,
    batch_size,
    chunk_size,
):

    (
        y_true,
        y_pred,
        inference_seconds,
    ) = phase4.evaluate_global_model(
        model=model,
        scaler=scaler,
        feature_columns=feature_columns,
        active_indices=active_indices,
        label_mapping=label_mapping,
        device=device,
        batch_size=batch_size,
        chunk_size=chunk_size,
    )


    (
        overall,
        report,
        confusion_matrix,
        class_names,
    ) = phase4.calculate_metrics(
        y_true,
        y_pred,
        label_mapping,
    )


    return (
        inference_seconds,
        overall,
        report,
        confusion_matrix,
        class_names,
    )


# ============================================================
# MEAN CLASS METRIC
# ============================================================

def mean_attack_metric(
    report,
    attacks,
    metric="recall",
):

    if not attacks:
        return np.nan


    key = (
        "f1-score"
        if metric == "f1"
        else metric
    )


    return float(
        np.mean(
            [
                float(
                    report[
                        attack
                    ][key]
                )
                for attack in attacks
            ]
        )
    )


# ============================================================
# CREATE GLOBAL ANCHOR
# ============================================================

def create_global_anchor(
    model,
):

    return {
        name:
            parameter
            .detach()
            .clone()

        for name, parameter
        in model.named_parameters()
    }


# ============================================================
# PROXIMAL / GLOBAL ANCHOR PENALTY
# ============================================================

def calculate_anchor_penalty(
    model,
    global_anchor,
):

    penalty = torch.zeros(
        (),
        device=next(
            model.parameters()
        ).device,
    )


    for name, parameter in model.named_parameters():

        anchor_parameter = global_anchor[
            name
        ]


        penalty = (
            penalty
            +
            torch.sum(
                (
                    parameter
                    -
                    anchor_parameter
                )
                ** 2
            )
        )


    return penalty


# ============================================================
# TRAIN ONE REGULARIZED PERSONALIZATION EPOCH
# ============================================================

def train_regularized_epoch(
    model,
    optimizer,
    client_file,
    scaler,
    feature_columns,
    active_indices,
    label_mapping,
    class_weights,
    global_anchor,
    lambda_value,
    device,
    batch_size,
    chunk_size,
    seed,
    client_id,
    epoch_number,
):

    model.train()


    weight_tensor = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )


    criterion = nn.CrossEntropyLoss(
        weight=weight_tensor
    )


    rng = np.random.default_rng(
        120000
        +
        seed
        +
        client_id * 1000
        +
        epoch_number * 10
    )


    total_combined_loss = 0.0

    total_ce_loss = 0.0

    total_anchor_loss = 0.0

    total_correct = 0

    total_samples = 0


    started = time.perf_counter()


    reader = pd.read_csv(
        client_file,
        chunksize=chunk_size,
    )


    for chunk in reader:

        X = phase4.transform_features(
            df=chunk,
            feature_columns=feature_columns,
            scaler=scaler,
            active_indices=active_indices,
        )


        y = phase4.encode_labels(
            chunk[
                phase4.LABEL_COL
            ],
            label_mapping,
        )


        if len(y) == 0:
            continue


        indices = rng.permutation(
            len(y)
        )


        for start in range(
            0,
            len(indices),
            batch_size,
        ):

            batch_indices = indices[
                start:
                start + batch_size
            ]


            X_batch = torch.from_numpy(
                X[
                    batch_indices
                ].astype(
                    np.float32,
                    copy=False,
                )
            ).to(device)


            y_batch = torch.from_numpy(
                y[
                    batch_indices
                ].astype(
                    np.int64,
                    copy=False,
                )
            ).to(device)


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


            anchor_penalty = (
                calculate_anchor_penalty(
                    model,
                    global_anchor,
                )
            )


            regularization_loss = (
                0.5
                *
                lambda_value
                *
                anchor_penalty
            )


            loss = (
                ce_loss
                +
                regularization_loss
            )


            loss.backward()


            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRADIENT_CLIP,
            )


            optimizer.step()


            predictions = torch.argmax(
                logits,
                dim=1,
            )


            batch_samples = int(
                y_batch.size(0)
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


            total_combined_loss += (
                float(
                    loss.item()
                )
                *
                batch_samples
            )


            total_ce_loss += (
                float(
                    ce_loss.item()
                )
                *
                batch_samples
            )


            total_anchor_loss += (
                float(
                    regularization_loss.item()
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


    if total_samples == 0:

        raise RuntimeError(
            f"Client {client_id} "
            "produced zero training samples."
        )


    return {
        "epoch":
            epoch_number,

        "lambda":
            lambda_value,

        "combined_loss":
            total_combined_loss
            /
            total_samples,

        "ce_loss":
            total_ce_loss
            /
            total_samples,

        "anchor_loss":
            total_anchor_loss
            /
            total_samples,

        "training_accuracy":
            total_correct
            /
            total_samples,

        "samples":
            total_samples,

        "training_seconds":
            time.perf_counter()
            -
            started,
    }


# ============================================================
# RUN ONE LAMBDA SWEEP
# ============================================================

def run_lambda_sweep(
    global_state,
    lambda_value,
    epoch_budgets,
    client_id,
    client_file,
    validation_file,
    seen_attacks,
    unseen_attacks,
    scaler,
    feature_columns,
    active_indices,
    label_mapping,
    class_weights,
    input_dim,
    device,
    batch_size,
    chunk_size,
    personalization_lr,
    weight_decay,
    seed,
    global_validation_seen,
    global_validation_unseen,
    min_knowledge_preservation,
):

    model = build_model(
        global_state,
        input_dim=input_dim,
        num_classes=len(
            label_mapping
        ),
        device=device,
    )


    global_anchor = create_global_anchor(
        model
    )


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=personalization_lr,
        weight_decay=weight_decay,
    )


    sweep_rows = []

    history_rows = []


    maximum_epochs = max(
        epoch_budgets
    )


    for epoch in range(
        1,
        maximum_epochs + 1,
    ):

        train_result = train_regularized_epoch(
            model=model,
            optimizer=optimizer,
            client_file=client_file,
            scaler=scaler,
            feature_columns=feature_columns,
            active_indices=active_indices,
            label_mapping=label_mapping,
            class_weights=class_weights,
            global_anchor=global_anchor,
            lambda_value=lambda_value,
            device=device,
            batch_size=batch_size,
            chunk_size=chunk_size,
            seed=seed,
            client_id=client_id,
            epoch_number=epoch,
        )


        train_result.update(
            {
                "seed":
                    seed,

                "client_id":
                    client_id,
            }
        )


        history_rows.append(
            train_result
        )


        if epoch not in epoch_budgets:
            continue


        (
            validation_seconds,
            overall,
            report,
            _,
            _,
        ) = evaluate_model_on_csv(
            model=model,
            csv_file=validation_file,
            scaler=scaler,
            feature_columns=feature_columns,
            active_indices=active_indices,
            label_mapping=label_mapping,
            device=device,
            batch_size=batch_size,
            chunk_size=chunk_size,
        )


        seen_recall = mean_attack_metric(
            report,
            seen_attacks,
            "recall",
        )


        unseen_recall = mean_attack_metric(
            report,
            unseen_attacks,
            "recall",
        )


        seen_f1 = mean_attack_metric(
            report,
            seen_attacks,
            "f1",
        )


        unseen_f1 = mean_attack_metric(
            report,
            unseen_attacks,
            "f1",
        )


        preservation_ratio = (
            knowledge_preservation_ratio(
                unseen_recall,
                global_validation_unseen,
            )
        )


        seen_gain = (
            seen_recall
            -
            global_validation_seen
        )


        balance = harmonic_mean(
            seen_recall,
            unseen_recall,
        )


        selection_score = (
            calculate_selection_score(
                seen_recall,
                unseen_recall,
                float(
                    overall[
                        "macro_f1"
                    ]
                ),
                float(
                    overall[
                        "balanced_accuracy"
                    ]
                ),
            )
        )


        passes_preservation = bool(
            preservation_ratio
            >=
            min_knowledge_preservation
        )


        passes_seen_gain = bool(
            seen_gain
            >=
            0.0
        )


        feasible = bool(
            passes_preservation
            and
            passes_seen_gain
        )


        sweep_rows.append(
            {
                "seed":
                    seed,

                "client_id":
                    client_id,

                "lambda":
                    lambda_value,

                "personalization_epochs":
                    epoch,

                "validation_seen_recall":
                    seen_recall,

                "validation_unseen_recall":
                    unseen_recall,

                "global_validation_seen_recall":
                    global_validation_seen,

                "global_validation_unseen_recall":
                    global_validation_unseen,

                "validation_personalization_gain":
                    seen_gain,

                "validation_knowledge_preservation_ratio":
                    preservation_ratio,

                "validation_seen_f1":
                    seen_f1,

                "validation_unseen_f1":
                    unseen_f1,

                "validation_harmonic_balance":
                    balance,

                "validation_macro_f1":
                    float(
                        overall[
                            "macro_f1"
                        ]
                    ),

                "validation_balanced_accuracy":
                    float(
                        overall[
                            "balanced_accuracy"
                        ]
                    ),

                "selection_score":
                    selection_score,

                "passes_knowledge_constraint":
                    passes_preservation,

                "passes_seen_gain_constraint":
                    passes_seen_gain,

                "feasible":
                    feasible,

                "validation_inference_seconds":
                    validation_seconds,
            }
        )


        logger.info(
            "Validation | "
            "seed=%d | "
            "client=%d | "
            "lambda=%.4g | "
            "epoch=%d | "
            "seen=%.6f | "
            "unseen=%.6f | "
            "preserve=%.2f%% | "
            "gain=%+.6f | "
            "score=%.6f | "
            "feasible=%s",
            seed,
            client_id,
            lambda_value,
            epoch,
            seen_recall,
            unseen_recall,
            preservation_ratio
            *
            100.0,
            seen_gain,
            selection_score,
            feasible,
        )


    del model


    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    return (
        pd.DataFrame(
            sweep_rows
        ),
        pd.DataFrame(
            history_rows
        ),
    )


# ============================================================
# RUN COMPLETE VALIDATION GRID
# ============================================================

def run_validation_grid(
    global_state,
    lambdas,
    epoch_budgets,
    client_id,
    client_file,
    validation_file,
    seen_attacks,
    unseen_attacks,
    scaler,
    feature_columns,
    active_indices,
    label_mapping,
    class_weights,
    input_dim,
    device,
    batch_size,
    chunk_size,
    personalization_lr,
    weight_decay,
    seed,
    global_validation_seen,
    global_validation_unseen,
    min_knowledge_preservation,
):

    sweep_frames = []

    history_frames = []


    for lambda_value in lambdas:

        (
            sweep_df,
            history_df,
        ) = run_lambda_sweep(
            global_state=global_state,
            lambda_value=lambda_value,
            epoch_budgets=epoch_budgets,
            client_id=client_id,
            client_file=client_file,
            validation_file=validation_file,
            seen_attacks=seen_attacks,
            unseen_attacks=unseen_attacks,
            scaler=scaler,
            feature_columns=feature_columns,
            active_indices=active_indices,
            label_mapping=label_mapping,
            class_weights=class_weights,
            input_dim=input_dim,
            device=device,
            batch_size=batch_size,
            chunk_size=chunk_size,
            personalization_lr=personalization_lr,
            weight_decay=weight_decay,
            seed=seed,
            global_validation_seen=global_validation_seen,
            global_validation_unseen=global_validation_unseen,
            min_knowledge_preservation=(
                min_knowledge_preservation
            ),
        )


        sweep_frames.append(
            sweep_df
        )


        history_frames.append(
            history_df
        )


    return (
        pd.concat(
            sweep_frames,
            ignore_index=True,
        ),
        pd.concat(
            history_frames,
            ignore_index=True,
        ),
    )


# ============================================================
# SELECT BEST CANDIDATE
# ============================================================

def choose_best_candidate(
    sweep_df: pd.DataFrame,
    fail_if_no_feasible: bool,
):

    feasible = sweep_df[
        sweep_df[
            "feasible"
        ]
        ==
        True
    ].copy()


    if not feasible.empty:

        ranked = feasible.sort_values(
            by=[
                "selection_score",
                "validation_harmonic_balance",
                "validation_personalization_gain",
                "validation_macro_f1",
                "personalization_epochs",
                "lambda",
            ],
            ascending=[
                False,
                False,
                False,
                False,
                True,
                False,
            ],
        )


        best = ranked.iloc[
            0
        ].to_dict()


        best[
            "selection_status"
        ] = "feasible_candidate"


        return best


    if fail_if_no_feasible:

        raise RuntimeError(
            "\nNo personalization candidate "
            "satisfied the knowledge-preservation "
            "and seen-gain constraints."
        )


    logger.warning(
        "No feasible candidate found. "
        "Using best fallback candidate."
    )


    # Fallback prioritizes knowledge preservation first.

    ranked = sweep_df.sort_values(
        by=[
            "validation_knowledge_preservation_ratio",
            "validation_harmonic_balance",
            "selection_score",
            "validation_personalization_gain",
            "personalization_epochs",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            True,
        ],
    )


    best = ranked.iloc[
        0
    ].to_dict()


    best[
        "selection_status"
    ] = "fallback_no_feasible_candidate"


    return best


# ============================================================
# TRAIN FINAL SELECTED MODEL
# ============================================================

def train_selected_model(
    global_state,
    selected_lambda,
    selected_epochs,
    client_id,
    client_file,
    scaler,
    feature_columns,
    active_indices,
    label_mapping,
    class_weights,
    input_dim,
    device,
    batch_size,
    chunk_size,
    personalization_lr,
    weight_decay,
    seed,
):

    # Start again from ORIGINAL FedProx70

    model = build_model(
        global_state,
        input_dim=input_dim,
        num_classes=len(
            label_mapping
        ),
        device=device,
    )


    global_anchor = create_global_anchor(
        model
    )


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=personalization_lr,
        weight_decay=weight_decay,
    )


    history_rows = []


    started = time.perf_counter()


    for epoch in range(
        1,
        selected_epochs + 1,
    ):

        result = train_regularized_epoch(
            model=model,
            optimizer=optimizer,
            client_file=client_file,
            scaler=scaler,
            feature_columns=feature_columns,
            active_indices=active_indices,
            label_mapping=label_mapping,
            class_weights=class_weights,
            global_anchor=global_anchor,
            lambda_value=selected_lambda,
            device=device,
            batch_size=batch_size,
            chunk_size=chunk_size,
            seed=seed,
            client_id=client_id,
            epoch_number=epoch,
        )


        result.update(
            {
                "seed":
                    seed,

                "client_id":
                    client_id,

                "selected_lambda":
                    selected_lambda,

                "selected_epochs":
                    selected_epochs,
            }
        )


        history_rows.append(
            result
        )


    return (
        model,
        pd.DataFrame(
            history_rows
        ),
        float(
            time.perf_counter()
            -
            started
        ),
    )


# ============================================================
# PER CLASS DATAFRAME
# ============================================================

def report_to_dataframe(
    report,
    class_names,
    seed,
    client_id,
    selected_lambda,
    selected_epochs,
):

    rows = []


    for class_name in class_names:

        metrics = report[
            class_name
        ]


        rows.append(
            {
                "seed":
                    seed,

                "client_id":
                    client_id,

                "selected_lambda":
                    selected_lambda,

                "selected_epochs":
                    selected_epochs,

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
            }
        )


    return pd.DataFrame(
        rows
    )


# ============================================================
# CI
# ============================================================

def mean_std_ci95(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )


    values = values[
        np.isfinite(
            values
        )
    ]


    if len(values) == 0:
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )


    mean_value = float(
        values.mean()
    )


    if len(values) == 1:
        return (
            mean_value,
            0.0,
            mean_value,
            mean_value,
        )


    std_value = float(
        values.std(
            ddof=1
        )
    )


    if SCIPY_AVAILABLE:

        critical = float(
            student_t.ppf(
                0.975,
                df=len(values)-1,
            )
        )

    else:

        critical = 1.96


    margin = (
        critical
        *
        std_value
        /
        math.sqrt(
            len(values)
        )
    )


    return (
        mean_value,
        std_value,
        mean_value-margin,
        mean_value+margin,
    )


# ============================================================
# SAFE WILCOXON
# ============================================================

def safe_wilcoxon(
    left,
    right,
):

    if not SCIPY_AVAILABLE:
        return (
            np.nan,
            np.nan,
        )


    left = np.asarray(
        left,
        dtype=float,
    )


    right = np.asarray(
        right,
        dtype=float,
    )


    valid = (
        np.isfinite(left)
        &
        np.isfinite(right)
    )


    left = left[
        valid
    ]

    right = right[
        valid
    ]


    if len(left) == 0:
        return (
            np.nan,
            np.nan,
        )


    differences = (
        left
        -
        right
    )


    if np.allclose(
        differences,
        0.0,
    ):
        return (
            0.0,
            1.0,
        )


    result = wilcoxon(
        left,
        right,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto",
    )


    return (
        float(
            result.statistic
        ),
        float(
            result.pvalue
        ),
    )


# ============================================================
# AGGREGATE COMPLETED SEEDS
# ============================================================

def aggregate_completed_seeds(
    scenario,
):

    scenario_root = (
        RESULT_ROOT
        /
        scenario
    )


    aggregate_dir = (
        scenario_root
        /
        "aggregate"
    )


    aggregate_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    final_frames = []

    class_frames = []

    sweep_frames = []

    completed_seeds = []


    for seed_dir in scenario_root.glob(
        "seed_*"
    ):

        if not seed_dir.is_dir():
            continue


        final_file = (
            seed_dir
            /
            "final_regularized_personalization_results.csv"
        )


        class_file = (
            seed_dir
            /
            "final_regularized_personalized_per_class_metrics.csv"
        )


        sweep_file = (
            seed_dir
            /
            "validation_sweep.csv"
        )


        if (
            not final_file.exists()
            or
            not class_file.exists()
            or
            not sweep_file.exists()
        ):
            continue


        seed = int(
            seed_dir.name.split(
                "_"
            )[-1]
        )


        completed_seeds.append(
            seed
        )


        final_frames.append(
            pd.read_csv(
                final_file
            )
        )


        class_frames.append(
            pd.read_csv(
                class_file
            )
        )


        sweep_frames.append(
            pd.read_csv(
                sweep_file
            )
        )


    if not completed_seeds:
        return None


    completed_seeds = sorted(
        completed_seeds
    )


    final_all = pd.concat(
        final_frames,
        ignore_index=True,
    )


    class_all = pd.concat(
        class_frames,
        ignore_index=True,
    )


    sweep_all = pd.concat(
        sweep_frames,
        ignore_index=True,
    )


    final_all.to_csv(
        aggregate_dir
        /
        "final_regularized_personalization_all_seeds.csv",
        index=False,
    )


    class_all.to_csv(
        aggregate_dir
        /
        "final_regularized_per_class_all_seeds.csv",
        index=False,
    )


    sweep_all.to_csv(
        aggregate_dir
        /
        "validation_sweep_all_seeds.csv",
        index=False,
    )


    # ========================================================
    # SELECTED HYPERPARAMETER FREQUENCY
    # ========================================================

    (
        final_all
        .groupby(
            [
                "client_id",
                "selected_lambda",
                "selected_epochs",
                "selection_status",
            ]
        )
        .size()
        .reset_index(
            name="count"
        )
        .sort_values(
            [
                "client_id",
                "count",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .to_csv(
            aggregate_dir
            /
            "selected_hyperparameter_frequency.csv",
            index=False,
        )
    )


    # ========================================================
    # PER CLIENT SUMMARY
    # ========================================================

    metric_columns = [
        "local_only_seen_recall",
        "global_seen_recall",
        "personalized_seen_recall",
        "personalization_gain",

        "global_unseen_recall",
        "personalized_unseen_recall",
        "global_knowledge_loss",

        "global_knowledge_preservation_ratio",
        "local_recovery_ratio",

        "global_utility_score",
        "personalized_utility_score",

        "personalized_macro_f1",
        "personalized_balanced_accuracy",
    ]


    summary_rows = []


    for client_id, group in final_all.groupby(
        "client_id"
    ):

        for metric in metric_columns:

            (
                mean_value,
                std_value,
                ci_lower,
                ci_upper,
            ) = mean_std_ci95(
                group[
                    metric
                ].to_numpy(
                    dtype=float
                )
            )


            summary_rows.append(
                {
                    "client_id":
                        int(
                            client_id
                        ),

                    "metric":
                        metric,

                    "n":
                        int(
                            group[
                                metric
                            ]
                            .notna()
                            .sum()
                        ),

                    "mean":
                        mean_value,

                    "std":
                        std_value,

                    "ci95_lower":
                        ci_lower,

                    "ci95_upper":
                        ci_upper,
                }
            )


    pd.DataFrame(
        summary_rows
    ).to_csv(
        aggregate_dir
        /
        "per_client_summary_across_seeds.csv",
        index=False,
    )


    # ========================================================
    # STATISTICAL UNIT = SEED
    # ========================================================

    seed_level = (
        final_all
        .groupby(
            "seed"
        )[
            [
                "local_only_seen_recall",
                "global_seen_recall",
                "personalized_seen_recall",

                "global_unseen_recall",
                "personalized_unseen_recall",

                "global_knowledge_preservation_ratio",
                "local_recovery_ratio",

                "global_utility_score",
                "personalized_utility_score",

                "personalization_gain",
                "global_knowledge_loss",
            ]
        ]
        .mean()
        .reset_index()
        .sort_values(
            "seed"
        )
    )


    seed_level.to_csv(
        aggregate_dir
        /
        "seed_level_summary.csv",
        index=False,
    )


    comparisons = [
        (
            "personalized_seen_vs_global_seen",
            "personalized_seen_recall",
            "global_seen_recall",
        ),

        (
            "personalized_unseen_vs_global_unseen",
            "personalized_unseen_recall",
            "global_unseen_recall",
        ),

        (
            "personalized_seen_vs_local_seen",
            "personalized_seen_recall",
            "local_only_seen_recall",
        ),

        (
            "personalized_utility_vs_global_utility",
            "personalized_utility_score",
            "global_utility_score",
        ),
    ]


    test_rows = []


    for (
        comparison,
        left_column,
        right_column,
    ) in comparisons:

        left = seed_level[
            left_column
        ].to_numpy(
            dtype=float
        )


        right = seed_level[
            right_column
        ].to_numpy(
            dtype=float
        )


        difference = (
            left
            -
            right
        )


        statistic, p_value = safe_wilcoxon(
            left,
            right,
        )


        test_rows.append(
            {
                "comparison":
                    comparison,

                "n_pairs":
                    len(
                        seed_level
                    ),

                "left_mean":
                    float(
                        np.nanmean(
                            left
                        )
                    ),

                "right_mean":
                    float(
                        np.nanmean(
                            right
                        )
                    ),

                "mean_delta":
                    float(
                        np.nanmean(
                            difference
                        )
                    ),

                "wins_left":
                    int(
                        np.sum(
                            difference
                            >
                            0
                        )
                    ),

                "losses_left":
                    int(
                        np.sum(
                            difference
                            <
                            0
                        )
                    ),

                "ties":
                    int(
                        np.sum(
                            np.isclose(
                                difference,
                                0.0,
                            )
                        )
                    ),

                "wilcoxon_statistic":
                    statistic,

                "p_value":
                    p_value,
            }
        )


    pd.DataFrame(
        test_rows
    ).to_csv(
        aggregate_dir
        /
        "seed_level_paired_tests.csv",
        index=False,
    )


    preservation = (
        final_all[
            "global_knowledge_preservation_ratio"
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )


    recovery = (
        final_all[
            "local_recovery_ratio"
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )


    summary = {
        "phase":
            "12B",

        "experiment":
            "Regularized Personalized FedProx70",

        "scenario":
            scenario,

        "completed_seeds":
            completed_seeds,

        "number_of_completed_seeds":
            len(
                completed_seeds
            ),

        "mean_personalization_gain":
            float(
                final_all[
                    "personalization_gain"
                ].mean()
            ),

        "mean_global_knowledge_loss":
            float(
                final_all[
                    "global_knowledge_loss"
                ].mean()
            ),

        "mean_global_knowledge_preservation_ratio":
            (
                float(
                    preservation.mean()
                )
                if len(
                    preservation
                ) > 0
                else None
            ),

        "mean_local_recovery_ratio":
            (
                float(
                    recovery.mean()
                )
                if len(
                    recovery
                ) > 0
                else None
            ),

        "mean_global_utility":
            float(
                final_all[
                    "global_utility_score"
                ].mean()
            ),

        "mean_personalized_utility":
            float(
                final_all[
                    "personalized_utility_score"
                ].mean()
            ),

        "feasible_selection_rate":
            float(
                (
                    final_all[
                        "selection_status"
                    ]
                    ==
                    "feasible_candidate"
                )
                .mean()
            ),

        "locked_test_used_for_selection":
            False,

        "selection_basis":
            "validation only",
    }


    save_json(
        aggregate_dir
        /
        "phase12b_final_summary.json",
        summary,
    )


    return summary


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
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )


    parser.add_argument(
        "--validation-file",
        type=Path,
        default=None,
    )


    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=DEFAULT_LAMBDAS,
    )


    parser.add_argument(
        "--epoch-budgets",
        nargs="+",
        type=int,
        default=DEFAULT_EPOCH_BUDGETS,
    )


    parser.add_argument(
        "--personalization-lr",
        type=float,
        default=DEFAULT_PERSONALIZATION_LR,
    )


    parser.add_argument(
        "--min-knowledge-preservation",
        type=float,
        default=DEFAULT_MIN_KNOWLEDGE_PRESERVATION,
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
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )


    parser.add_argument(
        "--fail-if-no-feasible",
        action="store_true",
    )


    args = parser.parse_args()


    lambdas = sorted(
        set(
            args.lambdas
        )
    )


    epoch_budgets = sorted(
        set(
            args.epoch_budgets
        )
    )


    if any(
        value <= 0
        for value in lambdas
    ):
        raise ValueError(
            "All lambda values must be > 0."
        )


    if any(
        epoch < 1
        for epoch in epoch_budgets
    ):
        raise ValueError(
            "Epoch budgets must be >= 1."
        )


    if not (
        0.0
        <
        args.min_knowledge_preservation
        <=
        1.0
    ):
        raise ValueError(
            "--min-knowledge-preservation "
            "must be in (0, 1]."
        )


    set_seed(
        args.seed
    )


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    validation_file = resolve_validation_file(
        args.validation_file
    )


    scenario_dir = (
        phase4.FEDERATED_ROOT
        /
        args.scenario
    )


    if not scenario_dir.exists():
        raise FileNotFoundError(
            scenario_dir
        )


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
    # PREPROCESSING
    # ========================================================

    (
        scaler,
        feature_columns,
        label_mapping,
        active_indices,
        active_features,
    ) = phase4.load_preprocessing()


    (
        all_features,
        all_active_indices,
    ) = verify_70_features(
        active_features,
        active_indices,
        feature_columns,
    )


    benign_label = detect_benign_label(
        label_mapping
    )


    attack_classes = [
        label
        for label in label_mapping.keys()
        if label != benign_label
    ]


    # ========================================================
    # GLOBAL CLASS WEIGHTS
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
    # LOAD ORIGINAL PHASE 10 GLOBAL MODEL
    # ========================================================

    (
        checkpoint,
        global_model_file,
    ) = load_global_checkpoint(
        scenario=args.scenario,
        seed=args.seed,
        device=device,
        active_features=all_features,
    )


    global_state = {
        name:
            tensor
            .detach()
            .cpu()
            .clone()

        for name, tensor
        in checkpoint[
            "model_state_dict"
        ].items()
    }


    # ========================================================
    # PHASE 11 LOCAL-ONLY REFERENCE
    # ========================================================

    local_only_df = (
        load_phase11_local_metrics(
            args.scenario,
            args.seed,
        )
    )


    # ========================================================
    # GLOBAL MODEL
    # ========================================================

    global_model = build_model(
        global_state,
        input_dim=len(
            all_features
        ),
        num_classes=len(
            label_mapping
        ),
        device=device,
    )


    # ========================================================
    # GLOBAL VALIDATION REPORT
    # ========================================================

    (
        _,
        global_validation_overall,
        global_validation_report,
        _,
        _,
    ) = evaluate_model_on_csv(
        model=global_model,
        csv_file=validation_file,
        scaler=scaler,
        feature_columns=feature_columns,
        active_indices=all_active_indices,
        label_mapping=label_mapping,
        device=device,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )


    # ========================================================
    # GLOBAL LOCKED-TEST REPORT
    # ========================================================

    (
        global_test_seconds,
        global_test_overall,
        global_test_report,
        _,
        global_class_names,
    ) = evaluate_locked_test(
        model=global_model,
        scaler=scaler,
        feature_columns=feature_columns,
        active_indices=all_active_indices,
        label_mapping=label_mapping,
        device=device,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )


    del global_model


    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    # ========================================================
    # HEADER
    # ========================================================

    print()

    print("=" * 100)

    print(
        "PHASE 12B — "
        "KNOWLEDGE-PRESERVING "
        "PERSONALIZED FEDPROX70"
    )

    print("=" * 100)

    print(
        f"Scenario                     : "
        f"{args.scenario}"
    )

    print(
        f"Seed                         : "
        f"{args.seed}"
    )

    print(
        f"Device                       : "
        f"{device}"
    )

    print(
        f"Features                     : "
        f"{len(all_features)}"
    )

    print(
        f"Validation                   : "
        f"{validation_file}"
    )

    print(
        f"Lambdas                      : "
        f"{lambdas}"
    )

    print(
        f"Epoch budgets                : "
        f"{epoch_budgets}"
    )

    print(
        f"Personalization LR           : "
        f"{args.personalization_lr}"
    )

    print(
        f"Minimum knowledge preserved  : "
        f"{args.min_knowledge_preservation * 100:.1f}%"
    )

    print(
        "Locked test used for select : NO"
    )


    # ========================================================
    # COLLECTIONS
    # ========================================================

    sweep_frames = []

    history_frames = []

    selected_history_frames = []

    final_rows = []

    per_class_frames = []

    distribution_rows = []


    # ========================================================
    # CLIENT LOOP
    # ========================================================

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


        counts = read_client_label_counts(
            client_file,
            args.chunk_size,
            label_mapping,
        )


        seen_attacks = [
            attack
            for attack in attack_classes
            if counts.get(
                attack,
                0,
            ) > 0
        ]


        unseen_attacks = [
            attack
            for attack in attack_classes
            if counts.get(
                attack,
                0,
            ) == 0
        ]


        if (
            not seen_attacks
            or
            not unseen_attacks
        ):
            raise RuntimeError(
                f"Invalid seen/unseen split "
                f"for client {client_id}."
            )


        print()

        print("-" * 100)

        print(
            f"CLIENT {client_id}"
        )

        print("-" * 100)

        print(
            f"Seen attacks   : "
            f"{seen_attacks}"
        )

        print(
            f"Unseen attacks : "
            f"{unseen_attacks}"
        )


        # ====================================================
        # CLIENT DISTRIBUTION
        # ====================================================

        for class_name in label_mapping.keys():

            distribution_rows.append(
                {
                    "seed":
                        args.seed,

                    "client_id":
                        client_id,

                    "class":
                        class_name,

                    "count":
                        int(
                            counts.get(
                                class_name,
                                0,
                            )
                        ),

                    "is_seen_attack":
                        class_name in seen_attacks,

                    "is_unseen_attack":
                        class_name in unseen_attacks,
                }
            )


        # ====================================================
        # PHASE 11 LOCAL BASELINE
        # ====================================================

        local_client = (
            local_only_df[
                local_only_df[
                    "client_id"
                ]
                ==
                client_id
            ]
            .set_index(
                "class"
            )
        )


        local_seen_recall = float(
            np.mean(
                [
                    float(
                        local_client.loc[
                            attack,
                            "recall",
                        ]
                    )
                    for attack in seen_attacks
                ]
            )
        )


        local_unseen_recall = float(
            np.mean(
                [
                    float(
                        local_client.loc[
                            attack,
                            "recall",
                        ]
                    )
                    for attack in unseen_attacks
                ]
            )
        )


        # ====================================================
        # GLOBAL VALIDATION SEEN/UNSEEN
        # ====================================================

        global_validation_seen = (
            mean_attack_metric(
                global_validation_report,
                seen_attacks,
                "recall",
            )
        )


        global_validation_unseen = (
            mean_attack_metric(
                global_validation_report,
                unseen_attacks,
                "recall",
            )
        )


        # ====================================================
        # GLOBAL LOCKED TEST SEEN/UNSEEN
        # ====================================================

        global_seen_recall = (
            mean_attack_metric(
                global_test_report,
                seen_attacks,
                "recall",
            )
        )


        global_unseen_recall = (
            mean_attack_metric(
                global_test_report,
                unseen_attacks,
                "recall",
            )
        )


        # ====================================================
        # VALIDATION GRID
        # ====================================================

        (
            sweep_df,
            history_df,
        ) = run_validation_grid(
            global_state=global_state,
            lambdas=lambdas,
            epoch_budgets=epoch_budgets,
            client_id=client_id,
            client_file=client_file,
            validation_file=validation_file,
            seen_attacks=seen_attacks,
            unseen_attacks=unseen_attacks,
            scaler=scaler,
            feature_columns=feature_columns,
            active_indices=all_active_indices,
            label_mapping=label_mapping,
            class_weights=global_weights,
            input_dim=len(
                all_features
            ),
            device=device,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            personalization_lr=(
                args.personalization_lr
            ),
            weight_decay=args.weight_decay,
            seed=args.seed,
            global_validation_seen=(
                global_validation_seen
            ),
            global_validation_unseen=(
                global_validation_unseen
            ),
            min_knowledge_preservation=(
                args.min_knowledge_preservation
            ),
        )


        sweep_df[
            "seen_attacks"
        ] = " | ".join(
            seen_attacks
        )


        sweep_df[
            "unseen_attacks"
        ] = " | ".join(
            unseen_attacks
        )


        sweep_frames.append(
            sweep_df
        )


        history_frames.append(
            history_df
        )


        # ====================================================
        # SELECT BEST VALIDATION CANDIDATE
        # ====================================================

        best = choose_best_candidate(
            sweep_df,
            fail_if_no_feasible=(
                args.fail_if_no_feasible
            ),
        )


        selected_lambda = float(
            best[
                "lambda"
            ]
        )


        selected_epochs = int(
            best[
                "personalization_epochs"
            ]
        )


        selection_status = str(
            best[
                "selection_status"
            ]
        )


        print()

        print(
            "VALIDATION SELECTION"
        )

        print(
            f"Selected lambda     : "
            f"{selected_lambda}"
        )

        print(
            f"Selected epochs     : "
            f"{selected_epochs}"
        )

        print(
            f"Selection status    : "
            f"{selection_status}"
        )

        print(
            f"Validation seen     : "
            f"{best['validation_seen_recall']:.6f}"
        )

        print(
            f"Validation unseen   : "
            f"{best['validation_unseen_recall']:.6f}"
        )

        print(
            f"Knowledge preserved : "
            f"{best['validation_knowledge_preservation_ratio'] * 100:.2f}%"
        )

        print(
            f"Selection score     : "
            f"{best['selection_score']:.6f}"
        )


        # ====================================================
        # RETRAIN SELECTED CONFIG FROM ORIGINAL FEDPROX70
        # ====================================================

        (
            personalized_model,
            selected_history,
            personalization_training_seconds,
        ) = train_selected_model(
            global_state=global_state,
            selected_lambda=selected_lambda,
            selected_epochs=selected_epochs,
            client_id=client_id,
            client_file=client_file,
            scaler=scaler,
            feature_columns=feature_columns,
            active_indices=all_active_indices,
            label_mapping=label_mapping,
            class_weights=global_weights,
            input_dim=len(
                all_features
            ),
            device=device,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            personalization_lr=(
                args.personalization_lr
            ),
            weight_decay=args.weight_decay,
            seed=args.seed,
        )


        selected_history_frames.append(
            selected_history
        )


        # ====================================================
        # LOCKED TEST ONCE
        # ====================================================

        (
            personalized_test_seconds,
            personalized_overall,
            personalized_report,
            _,
            personalized_class_names,
        ) = evaluate_locked_test(
            model=personalized_model,
            scaler=scaler,
            feature_columns=feature_columns,
            active_indices=all_active_indices,
            label_mapping=label_mapping,
            device=device,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
        )


        personalized_seen = (
            mean_attack_metric(
                personalized_report,
                seen_attacks,
                "recall",
            )
        )


        personalized_unseen = (
            mean_attack_metric(
                personalized_report,
                unseen_attacks,
                "recall",
            )
        )


        personalization_gain = (
            personalized_seen
            -
            global_seen_recall
        )


        global_knowledge_loss = (
            global_unseen_recall
            -
            personalized_unseen
        )


        preservation_ratio = (
            knowledge_preservation_ratio(
                personalized_unseen,
                global_unseen_recall,
            )
        )


        local_recovery_ratio = (
            safe_ratio(
                personalized_seen,
                local_seen_recall,
            )
        )


        global_utility = (
            0.5
            *
            global_seen_recall
            +
            0.5
            *
            global_unseen_recall
        )


        personalized_utility = (
            0.5
            *
            personalized_seen
            +
            0.5
            *
            personalized_unseen
        )


        # ====================================================
        # FINAL ROW
        # ====================================================

        final_rows.append(
            {
                "seed":
                    args.seed,

                "client_id":
                    client_id,

                "seen_attacks":
                    " | ".join(
                        seen_attacks
                    ),

                "unseen_attacks":
                    " | ".join(
                        unseen_attacks
                    ),

                "selected_lambda":
                    selected_lambda,

                "selected_epochs":
                    selected_epochs,

                "selection_status":
                    selection_status,

                "validation_selection_score":
                    float(
                        best[
                            "selection_score"
                        ]
                    ),

                "validation_knowledge_preservation_ratio":
                    float(
                        best[
                            "validation_knowledge_preservation_ratio"
                        ]
                    ),

                "local_only_seen_recall":
                    local_seen_recall,

                "local_only_unseen_recall":
                    local_unseen_recall,

                "global_seen_recall":
                    global_seen_recall,

                "global_unseen_recall":
                    global_unseen_recall,

                "personalized_seen_recall":
                    personalized_seen,

                "personalized_unseen_recall":
                    personalized_unseen,

                "personalization_gain":
                    personalization_gain,

                "global_knowledge_loss":
                    global_knowledge_loss,

                "global_knowledge_preservation_ratio":
                    preservation_ratio,

                "local_recovery_ratio":
                    local_recovery_ratio,

                "global_utility_score":
                    global_utility,

                "personalized_utility_score":
                    personalized_utility,

                "personalized_harmonic_balance":
                    harmonic_mean(
                        personalized_seen,
                        personalized_unseen,
                    ),

                "global_macro_f1":
                    float(
                        global_test_overall[
                            "macro_f1"
                        ]
                    ),

                "personalized_macro_f1":
                    float(
                        personalized_overall[
                            "macro_f1"
                        ]
                    ),

                "personalized_balanced_accuracy":
                    float(
                        personalized_overall[
                            "balanced_accuracy"
                        ]
                    ),

                "personalized_accuracy":
                    float(
                        personalized_overall[
                            "accuracy"
                        ]
                    ),

                "personalized_weighted_f1":
                    float(
                        personalized_overall[
                            "weighted_f1"
                        ]
                    ),

                "personalization_training_seconds":
                    personalization_training_seconds,

                "personalized_test_inference_seconds":
                    personalized_test_seconds,
            }
        )


        # ====================================================
        # PER CLASS
        # ====================================================

        per_class_frames.append(
            report_to_dataframe(
                personalized_report,
                personalized_class_names,
                seed=args.seed,
                client_id=client_id,
                selected_lambda=(
                    selected_lambda
                ),
                selected_epochs=(
                    selected_epochs
                ),
            )
        )


        # ====================================================
        # SAVE MODEL
        # ====================================================

        client_artifact_dir = (
            artifact_dir
            /
            f"client_{client_id}"
        )


        client_artifact_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        torch.save(
            {
                "phase":
                    "12B",

                "algorithm":
                    (
                        "Regularized "
                        "Personalized FedProx70"
                    ),

                "scenario":
                    args.scenario,

                "seed":
                    args.seed,

                "client_id":
                    client_id,

                "selected_lambda":
                    selected_lambda,

                "selected_epochs":
                    selected_epochs,

                "selection_status":
                    selection_status,

                "min_knowledge_preservation":
                    args.min_knowledge_preservation,

                "personalization_learning_rate":
                    args.personalization_lr,

                "seen_attacks":
                    seen_attacks,

                "unseen_attacks":
                    unseen_attacks,

                "features":
                    all_features,

                "active_indices":
                    all_active_indices.tolist(),

                "label_mapping":
                    label_mapping,

                "model_state_dict":
                    personalized_model.state_dict(),

                "test_metrics":
                    personalized_overall,
            },

            client_artifact_dir
            /
            "regularized_personalized_fedprox70_model.pt",
        )


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        print()

        print(
            "LOCKED TEST RESULT"
        )

        print(
            f"Local-only seen       : "
            f"{local_seen_recall:.6f}"
        )

        print(
            f"Global seen           : "
            f"{global_seen_recall:.6f}"
        )

        print(
            f"Personalized seen     : "
            f"{personalized_seen:.6f}"
        )

        print(
            f"Personalization gain  : "
            f"{personalization_gain:+.6f}"
        )

        print(
            f"Global unseen         : "
            f"{global_unseen_recall:.6f}"
        )

        print(
            f"Personalized unseen   : "
            f"{personalized_unseen:.6f}"
        )

        print(
            f"Knowledge loss        : "
            f"{global_knowledge_loss:+.6f}"
        )

        print(
            f"Knowledge preserved   : "
            f"{preservation_ratio * 100:.2f}%"
        )

        if np.isfinite(
            local_recovery_ratio
        ):
            print(
                f"Local recovery        : "
                f"{local_recovery_ratio * 100:.2f}%"
            )

        print(
            f"Global utility        : "
            f"{global_utility:.6f}"
        )

        print(
            f"Personalized utility  : "
            f"{personalized_utility:.6f}"
        )


        del personalized_model


        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    # ========================================================
    # COMBINE OUTPUTS
    # ========================================================

    validation_sweep_df = pd.concat(
        sweep_frames,
        ignore_index=True,
    )


    training_history_df = pd.concat(
        history_frames,
        ignore_index=True,
    )


    selected_history_df = pd.concat(
        selected_history_frames,
        ignore_index=True,
    )


    final_df = pd.DataFrame(
        final_rows
    )


    per_class_df = pd.concat(
        per_class_frames,
        ignore_index=True,
    )


    distribution_df = pd.DataFrame(
        distribution_rows
    )


    # ========================================================
    # SAVE SEED OUTPUTS
    # ========================================================

    validation_sweep_df.to_csv(
        result_dir
        /
        "validation_sweep.csv",
        index=False,
    )


    training_history_df.to_csv(
        result_dir
        /
        "regularized_training_history.csv",
        index=False,
    )


    selected_history_df.to_csv(
        result_dir
        /
        "selected_training_history.csv",
        index=False,
    )


    distribution_df.to_csv(
        result_dir
        /
        "client_training_distribution.csv",
        index=False,
    )


    final_df.to_csv(
        result_dir
        /
        "final_regularized_personalization_results.csv",
        index=False,
    )


    per_class_df.to_csv(
        result_dir
        /
        "final_regularized_personalized_per_class_metrics.csv",
        index=False,
    )


    # ========================================================
    # SEED SUMMARY
    # ========================================================

    preservation_values = (
        final_df[
            "global_knowledge_preservation_ratio"
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )


    recovery_values = (
        final_df[
            "local_recovery_ratio"
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )


    phase12b_summary = {
        "phase":
            "12B",

        "experiment":
            (
                "Knowledge-Preserving "
                "Regularized Personalized FedProx70"
            ),

        "scenario":
            args.scenario,

        "seed":
            args.seed,

        "global_model":
            str(
                global_model_file
            ),

        "validation_file":
            str(
                validation_file
            ),

        "features":
            len(
                all_features
            ),

        "candidate_lambdas":
            lambdas,

        "candidate_epochs":
            epoch_budgets,

        "personalization_learning_rate":
            args.personalization_lr,

        "minimum_validation_knowledge_preservation":
            args.min_knowledge_preservation,

        "selection_score":
            (
                "0.50 * harmonic(seen, unseen) + "
                "0.25 * macro_f1 + "
                "0.25 * balanced_accuracy"
            ),

        "mean_personalization_gain":
            float(
                final_df[
                    "personalization_gain"
                ].mean()
            ),

        "mean_global_knowledge_loss":
            float(
                final_df[
                    "global_knowledge_loss"
                ].mean()
            ),

        "mean_global_knowledge_preservation_ratio":
            (
                float(
                    preservation_values.mean()
                )
                if len(
                    preservation_values
                ) > 0
                else None
            ),

        "mean_local_recovery_ratio":
            (
                float(
                    recovery_values.mean()
                )
                if len(
                    recovery_values
                ) > 0
                else None
            ),

        "mean_global_utility":
            float(
                final_df[
                    "global_utility_score"
                ].mean()
            ),

        "mean_personalized_utility":
            float(
                final_df[
                    "personalized_utility_score"
                ].mean()
            ),

        "feasible_clients":
            int(
                (
                    final_df[
                        "selection_status"
                    ]
                    ==
                    "feasible_candidate"
                )
                .sum()
            ),

        "locked_test_used_for_selection":
            False,

        "locked_test_usage":
            "final evaluation only",
    }


    save_json(
        result_dir
        /
        "phase12b_summary.json",
        phase12b_summary,
    )


    # ========================================================
    # AGGREGATE COMPLETED SEEDS
    # ========================================================

    aggregate_summary = (
        aggregate_completed_seeds(
            args.scenario
        )
    )


    # ========================================================
    # FINAL DISPLAY
    # ========================================================

    print()

    print("=" * 100)

    print(
        "PHASE 12B SEED COMPLETE"
    )

    print("=" * 100)


    display_columns = [
        "client_id",
        "seen_attacks",

        "selected_lambda",
        "selected_epochs",
        "selection_status",

        "local_only_seen_recall",
        "global_seen_recall",
        "personalized_seen_recall",

        "personalization_gain",

        "global_unseen_recall",
        "personalized_unseen_recall",

        "global_knowledge_loss",
        "global_knowledge_preservation_ratio",

        "local_recovery_ratio",

        "global_utility_score",
        "personalized_utility_score",
    ]


    print()

    print(
        final_df[
            display_columns
        ].to_string(
            index=False
        )
    )


    print()

    print(
        f"Results:\n"
        f"{result_dir}"
    )


    print()

    print(
        f"Models:\n"
        f"{artifact_dir}"
    )


    if aggregate_summary is not None:

        print()

        print("=" * 100)

        print(
            "CURRENT MULTI-SEED AGGREGATE"
        )

        print("=" * 100)


        print(
            "Completed seeds            : "
            f"{aggregate_summary['completed_seeds']}"
        )


        print(
            "Mean personalization gain  : "
            f"{aggregate_summary['mean_personalization_gain']:.6f}"
        )


        print(
            "Mean global knowledge loss : "
            f"{aggregate_summary['mean_global_knowledge_loss']:.6f}"
        )


        preservation = (
            aggregate_summary[
                "mean_global_knowledge_preservation_ratio"
            ]
        )


        if preservation is not None:

            print(
                "Mean knowledge preserved   : "
                f"{preservation * 100:.2f}%"
            )


        recovery = (
            aggregate_summary[
                "mean_local_recovery_ratio"
            ]
        )


        if recovery is not None:

            print(
                "Mean local recovery         : "
                f"{recovery * 100:.2f}%"
            )


        print(
            "Global utility              : "
            f"{aggregate_summary['mean_global_utility']:.6f}"
        )


        print(
            "Personalized utility        : "
            f"{aggregate_summary['mean_personalized_utility']:.6f}"
        )


        print(
            "Feasible selection rate     : "
            f"{aggregate_summary['feasible_selection_rate'] * 100:.2f}%"
        )


    if not SCIPY_AVAILABLE:

        print()

        print(
            "Install SciPy for "
            "multi-seed Wilcoxon tests:"
        )

        print(
            "pip install scipy"
        )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()