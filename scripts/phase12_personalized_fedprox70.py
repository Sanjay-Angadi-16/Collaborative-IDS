r"""
Phase 12 - Personalized FedProx70
=================================

Research question
-----------------
Phase 11 showed:

Local-only:
    Seen attack recall   -> very high
    Unseen attack recall -> approximately zero

FedProx70:
    Seen attack recall   -> lower than local-only
    Unseen attack recall -> substantially improved

Phase 12 asks:

    Can local personalization recover client-specific attack
    specialization while preserving the global knowledge
    transferred by FedProx70?

Protocol
--------
For each seed and each client:

    FedProx70 global model
            |
            +--> fine-tune 1 epoch
            +--> fine-tune 2 epochs
            +--> fine-tune 3 epochs
            +--> fine-tune 5 epochs

The best epoch budget is selected ONLY using validation.csv.

Then:

    Reload original FedProx70
            |
            +--> retrain selected epoch budget
            |
            +--> evaluate ONCE on locked test

This prevents locked-test hyperparameter tuning.

Metrics
-------
Personalization Gain:

    Personalized Seen Recall
    -
    Global FedProx70 Seen Recall

Global Knowledge Loss:

    Global FedProx70 Unseen Recall
    -
    Personalized Unseen Recall

Knowledge Preservation Ratio:

    Personalized Unseen Recall
    /
    Global FedProx70 Unseen Recall

Local Recovery Ratio:

    Personalized Seen Recall
    /
    LocalOnly Seen Recall

Utility:

    0.5 * Seen Recall
    +
    0.5 * Unseen Recall


Required previous phases
------------------------
Phase 10:
    artifacts/phase10/fedprox70/<scenario>/seed_<seed>/
        fedprox70_model.pt

Phase 11:
    results/phase11/knowledge_retention/<scenario>/seed_<seed>/
        local_only_per_class_metrics.csv


Outputs
-------
results/phase12/personalized_fedprox70/<scenario>/seed_<seed>/

    validation_sweep.csv
    personalization_training_history.csv
    selected_personalization_training_history.csv

    final_personalization_results.csv
    final_personalized_per_class_metrics.csv

    client_training_distribution.csv

    phase12_summary.json


Models
------
artifacts/phase12/personalized_fedprox70/<scenario>/seed_<seed>/

    client_1/
        personalized_fedprox70_model.pt

    client_2/
        personalized_fedprox70_model.pt

    ...

Aggregate results
-----------------
results/phase12/personalized_fedprox70/<scenario>/aggregate/

    final_personalization_all_seeds.csv
    final_personalized_per_class_all_seeds.csv
    validation_sweep_all_seeds.csv

    selected_epoch_frequency.csv
    per_client_summary_across_seeds.csv
    seed_level_summary.csv
    seed_level_paired_tests.csv

    phase12_final_summary.json
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

    if str(
        path
    ) not in sys.path:

        sys.path.insert(

            0,

            str(
                path
            ),

        )


# ============================================================
# REUSE PHASE-9 INFRASTRUCTURE
# ============================================================

try:

    import phase9_fedavg70 as phase4

except ModuleNotFoundError as exc:

    raise ModuleNotFoundError(

        "\nMissing:\n"

        "scripts\\phase9_fedavg70.py\n\n"

        "Keep phase12_personalized_fedprox70.py "

        "inside the same scripts folder."

    ) from exc


# ============================================================
# CONFIG
# ============================================================

DEFAULT_SEED = (
    phase4.SEED
)


N_CLIENTS = (
    phase4.N_CLIENTS
)


EXPECTED_FEATURES = 70


DEFAULT_BATCH_SIZE = (
    phase4.DEFAULT_BATCH_SIZE
)


DEFAULT_CHUNK_SIZE = (
    phase4.DEFAULT_CHUNK_SIZE
)


DEFAULT_WEIGHT_DECAY = (
    phase4.DEFAULT_WEIGHT_DECAY
)


DEFAULT_PERSONALIZATION_LR = (
    0.0001
)


DEFAULT_EPOCH_BUDGETS = [

    1,

    2,

    3,

    5,

]


DEFAULT_SEEN_WEIGHT = (
    0.50
)


GRADIENT_CLIP = (
    5.0
)


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

    / "phase12"

    / "personalized_fedprox70"

)


ARTIFACT_ROOT = (

    PROJECT_ROOT

    / "artifacts"

    / "phase12"

    / "personalized_fedprox70"

)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(

    level=
        logging.INFO,

    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),

)


logger = logging.getLogger(

    "phase12_personalized_fedprox70"

)


# ============================================================
# SEED
# ============================================================

def set_seed(

    seed: int,

) -> None:


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

        torch.backends.cudnn.deterministic = (
            True
        )

        torch.backends.cudnn.benchmark = (
            False
        )


# ============================================================
# JSON HELPERS
# ============================================================

def json_safe(

    value: Any,

):


    if isinstance(

        value,

        dict,

    ):

        return {

            str(
                key
            ):

                json_safe(
                    item
                )

            for key, item
            in value.items()

        }


    if isinstance(

        value,

        (list, tuple),

    ):

        return [

            json_safe(
                item
            )

            for item
            in value

        ]


    if isinstance(

        value,

        np.integer,

    ):

        return int(
            value
        )


    if isinstance(

        value,

        np.floating,

    ):

        if np.isnan(
            value
        ):

            return None


        return float(
            value
        )


    if isinstance(

        value,

        np.bool_,

    ):

        return bool(
            value
        )


    if (

        isinstance(
            value,
            float,
        )

        and

        math.isnan(
            value
        )

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

        encoding=
            "utf-8",

    ) as file:


        json.dump(

            json_safe(
                payload
            ),

            file,

            indent=
                4,

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

            map_location=
                device,

            weights_only=
                False,

        )


    except TypeError:

        return torch.load(

            path,

            map_location=
                device,

        )


# ============================================================
# SAFE RATIO
# ============================================================

def safe_ratio(

    numerator: float,

    denominator: float,

):


    if (

        not np.isfinite(
            denominator
        )

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


# ============================================================
# HARMONIC BALANCE
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
# SELECTION SCORE
# ============================================================

def weighted_seen_unseen_score(

    seen_recall: float,

    unseen_recall: float,

    seen_weight: float,

) -> float:


    return float(

        seen_weight

        *

        seen_recall

        +

        (

            1.0

            -

            seen_weight

        )

        *

        unseen_recall

    )


# ============================================================
# VALIDATION FILE
# ============================================================

def resolve_validation_file(

    user_path: Path | None,

) -> Path:


    if user_path is not None:


        validation_file = (

            user_path

            .expanduser()

        )


        if not validation_file.is_absolute():

            validation_file = (

                PROJECT_ROOT

                /

                validation_file

            )


        validation_file = (

            validation_file

            .resolve()

        )


        if not validation_file.exists():

            raise FileNotFoundError(

                "\nValidation file not found:\n"

                f"{validation_file}"

            )


        if (

            "test_locked"

            in

            validation_file.name.lower()

        ):

            raise ValueError(

                "\nRefusing to use the locked-test file "

                "for personalization selection."

            )


        return validation_file


    candidates = []


    # ========================================================
    # TRY PHASE-4 CONSTANTS
    # ========================================================

    for attribute in (

        "VALIDATION_FILE",

        "VALIDATION_CSV",

        "VAL_FILE",

        "VAL_CSV",

    ):


        value = getattr(

            phase4,

            attribute,

            None,

        )


        if value:


            path = Path(
                value
            )


            if not path.is_absolute():

                path = (

                    PROJECT_ROOT

                    /

                    path

                )


            if path.exists():

                candidates.append(

                    path.resolve()

                )


    # ========================================================
    # COMMON LOCATIONS
    # ========================================================

    common_paths = [

        PROJECT_ROOT
        / "data"
        / "validation.csv",

        PROJECT_ROOT
        / "data"
        / "processed"
        / "validation.csv",

        PROJECT_ROOT
        / "data"
        / "splits"
        / "validation.csv",

        PROJECT_ROOT
        / "validation.csv",

    ]


    for path in common_paths:

        if path.exists():

            candidates.append(

                path.resolve()

            )


    # Remove duplicates

    candidates = list(

        dict.fromkeys(

            candidates

        )

    )


    if len(
        candidates
    ) == 1:

        return candidates[
            0
        ]


    if len(
        candidates
    ) == 0:

        raise FileNotFoundError(

            "\nCould not find validation.csv automatically.\n\n"

            "Pass it explicitly:\n\n"

            "python scripts\\phase12_personalized_fedprox70.py "

            "--scenario non_iid "

            "--seed 42 "

            "--validation-file <path-to-validation.csv>\n"

        )


    raise RuntimeError(

        "\nMultiple validation files found:\n\n"

        +

        "\n".join(

            f"  {path}"

            for path
            in candidates

        )

        +

        "\n\nPass the correct one using --validation-file."

    )


# ============================================================
# VERIFY ALL 70 FEATURES
# ============================================================

def verify_70_features(

    active_features,

    active_indices,

    feature_columns,

):


    if (

        len(
            active_features
        )

        !=

        EXPECTED_FEATURES

    ):

        raise ValueError(

            "\nExpected exactly 70 active features.\n"

            f"Found: {len(active_features)}"

        )


    active_indices_array = np.asarray(

        active_indices,

        dtype=
            np.int64,

    )


    if (

        len(
            active_indices_array
        )

        !=

        EXPECTED_FEATURES

    ):

        raise ValueError(

            "\nExpected exactly 70 active indices.\n"

            f"Found: {len(active_indices_array)}"

        )


    reconstructed_features = [

        feature_columns[

            int(
                index
            )

        ]

        for index
        in active_indices_array

    ]


    if (

        list(
            reconstructed_features
        )

        !=

        list(
            active_features
        )

    ):

        raise ValueError(

            "\nFeature/index mapping mismatch."

        )


    return (

        list(
            active_features
        ),

        active_indices_array,

    )


# ============================================================
# DETECT BENIGN CLASS
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

            str(
                label
            )

            .strip()

            .lower()

        )


        if (

            "benign"

            in

            normalized

            or

            "normal"

            in

            normalized

        ):

            return label


    raise ValueError(

        "\nCould not detect benign label.\n"

        f"Labels: {labels}"

    )


# ============================================================
# CLIENT LABEL DISTRIBUTION
# ============================================================

def read_client_label_counts(

    client_file: Path,

    chunk_size: int,

    label_mapping,

):


    canonical = {

        str(
            label
        )
        .strip()
        .lower():

            str(
                label
            ).strip()

        for label
        in label_mapping.keys()

    }


    counts = {

        str(
            label
        ).strip():

            0

        for label
        in label_mapping.keys()

    }


    reader = pd.read_csv(

        client_file,

        usecols=[

            phase4.LABEL_COL

        ],

        chunksize=
            chunk_size,

    )


    for chunk in reader:


        value_counts = (

            chunk[

                phase4.LABEL_COL

            ]

            .astype(
                str
            )

            .str.strip()

            .value_counts()

        )


        for (

            raw_label,

            count,

        ) in value_counts.items():


            key = (

                str(
                    raw_label
                )

                .strip()

                .lower()

            )


            if key not in canonical:

                raise ValueError(

                    "\nUnknown label in client file:\n"

                    f"{raw_label}"

                )


            counts[

                canonical[
                    key
                ]

            ] += int(

                count

            )


    return counts


# ============================================================
# LOAD PHASE-10 FEDPROX70
# ============================================================

def load_fedprox70_checkpoint(

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

            "\nPhase-10 FedProx70 model missing:\n"

            f"{model_file}"

        )


    checkpoint = (

        load_torch(

            model_file,

            device,

        )

    )


    checkpoint_seed = (

        checkpoint.get(

            "seed"

        )

    )


    if (

        checkpoint_seed

        is not None

        and

        int(
            checkpoint_seed
        )

        !=

        int(
            seed
        )

    ):

        raise ValueError(

            "\nPhase-10 checkpoint seed mismatch."

        )


    checkpoint_features = (

        checkpoint.get(

            "features"

        )

    )


    if (

        checkpoint_features

        is not None

        and

        list(
            checkpoint_features
        )

        !=

        list(
            active_features
        )

    ):

        raise ValueError(

            "\nPhase-10 feature schema mismatch."

        )


    if (

        "model_state_dict"

        not in

        checkpoint

    ):

        raise KeyError(

            "\nCheckpoint missing model_state_dict."

        )


    return (

        checkpoint,

        model_file,

    )


# ============================================================
# PHASE-11 LOCAL BASELINE
# ============================================================

def load_phase11_local_metrics(

    scenario: str,

    seed: int,

) -> pd.DataFrame:


    metrics_file = (

        PHASE11_RESULT_ROOT

        / scenario

        / f"seed_{seed}"

        / "local_only_per_class_metrics.csv"

    )


    if not metrics_file.exists():

        raise FileNotFoundError(

            "\nPhase-11 local-only metrics missing:\n"

            f"{metrics_file}\n\n"

            "Run Phase 11 first."

        )


    dataframe = pd.read_csv(

        metrics_file

    )


    if (

        "f1"

        in

        dataframe.columns

        and

        "f1_score"

        not in

        dataframe.columns

    ):

        dataframe = (

            dataframe.rename(

                columns={

                    "f1":
                        "f1_score"

                }

            )

        )


    required = {

        "client_id",

        "class",

        "precision",

        "recall",

        "f1_score",

        "support",

    }


    missing = (

        required

        -

        set(
            dataframe.columns
        )

    )


    if missing:

        raise ValueError(

            "\nInvalid Phase-11 local metrics.\n"

            f"Missing: {sorted(missing)}"

        )


    dataframe[

        "client_id"

    ] = (

        dataframe[

            "client_id"

        ]

        .astype(
            int
        )

    )


    dataframe[

        "class"

    ] = (

        dataframe[

            "class"

        ]

        .astype(
            str
        )

        .str.strip()

    )


    return dataframe


# ============================================================
# BUILD MODEL
# ============================================================

def build_model_from_state(

    state_dict,

    input_dim: int,

    num_classes: int,

    device,

):


    model = (

        phase4.IDSMLP(

            input_dim=
                input_dim,

            num_classes=
                num_classes,

        )

        .to(
            device
        )

    )


    model.load_state_dict(

        state_dict

    )


    return model


# ============================================================
# EVALUATE ANY CSV
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
    """
    Evaluate on a CSV using ONLY the six classes present
    in the frozen Phase-9/10 label mapping.

    This is required because the original validation.csv
    may contain additional CICIDS2017 attack classes that
    are outside the locked six-class experiment.
    """

    model.eval()

    y_true_parts = []
    y_pred_parts = []

    started = time.perf_counter()

    # ========================================================
    # ALLOWED SIX CLASSES
    # ========================================================

    allowed_labels = {
        str(label).strip()
        for label in label_mapping.keys()
    }

    allowed_labels_lower = {
        str(label).strip().lower():
            str(label).strip()
        for label in label_mapping.keys()
    }

    total_rows_read = 0
    total_rows_used = 0
    total_rows_filtered = 0

    filtered_label_counts = {}

    reader = pd.read_csv(
        csv_file,
        chunksize=chunk_size,
    )

    for chunk in reader:

        total_rows_read += len(chunk)

        # ====================================================
        # VALIDATE LABEL COLUMN
        # ====================================================

        if phase4.LABEL_COL not in chunk.columns:

            raise KeyError(
                f"\nValidation file is missing label column:\n"
                f"{phase4.LABEL_COL}\n"
                f"File: {csv_file}"
            )

        # Normalize labels for safe comparison.
        normalized_labels = (
            chunk[phase4.LABEL_COL]
            .astype(str)
            .str.strip()
        )

        normalized_lower = (
            normalized_labels
            .str.lower()
        )

        # ====================================================
        # KEEP ONLY THE FROZEN SIX CLASSES
        # ====================================================

        keep_mask = normalized_lower.isin(
            allowed_labels_lower.keys()
        )

        # Track excluded classes for auditability.
        excluded = (
            normalized_labels[
                ~keep_mask
            ]
            .value_counts()
        )

        for label, count in excluded.items():

            filtered_label_counts[
                str(label)
            ] = (
                filtered_label_counts.get(
                    str(label),
                    0,
                )
                +
                int(count)
            )

        chunk = (
            chunk.loc[
                keep_mask
            ]
            .copy()
        )

        total_rows_filtered += int(
            (~keep_mask).sum()
        )

        if chunk.empty:

            continue

        # ====================================================
        # CANONICALIZE LABEL SPELLING
        # ====================================================

        chunk[
            phase4.LABEL_COL
        ] = (
            chunk[
                phase4.LABEL_COL
            ]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(
                allowed_labels_lower
            )
        )

        # Safety check.
        if (
            chunk[
                phase4.LABEL_COL
            ]
            .isna()
            .any()
        ):

            raise ValueError(
                "\nLabel canonicalization failed "
                "inside validation evaluation."
            )

        total_rows_used += len(chunk)

        # ====================================================
        # SAME PHASE-9/10 FEATURE TRANSFORMATION
        # ====================================================

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

        prediction_parts = []

        # ====================================================
        # BATCH INFERENCE
        # ====================================================

        for start in range(
            0,
            len(y),
            batch_size,
        ):

            X_batch = torch.from_numpy(
                X[
                    start:
                    start + batch_size
                ]
                .astype(
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

            prediction_parts.append(
                predictions
            )

        y_pred_parts.append(
            np.concatenate(
                prediction_parts
            )
        )

    # ========================================================
    # VALIDATION SAFETY
    # ========================================================

    if not y_true_parts:

        raise RuntimeError(
            "\nValidation evaluation produced zero "
            "samples from the selected six classes.\n"
            f"File: {csv_file}"
        )

    y_true = np.concatenate(
        y_true_parts
    )

    y_pred = np.concatenate(
        y_pred_parts
    )

    # ========================================================
    # AUDIT LOG
    # ========================================================

    logger.info(
        "Validation rows read     : %d",
        total_rows_read,
    )

    logger.info(
        "Validation rows retained : %d",
        total_rows_used,
    )

    logger.info(
        "Validation rows excluded : %d",
        total_rows_filtered,
    )

    if filtered_label_counts:

        logger.info(
            "Excluded validation classes: %s",
            filtered_label_counts,
        )

    # ========================================================
    # METRICS
    # ========================================================

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
# LOCKED TEST
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

        model=
            model,

        scaler=
            scaler,

        feature_columns=
            feature_columns,

        active_indices=
            active_indices,

        label_mapping=
            label_mapping,

        device=
            device,

        batch_size=
            batch_size,

        chunk_size=
            chunk_size,

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
# MEAN ATTACK METRIC
# ============================================================

def mean_attack_metric(

    report,

    attacks,

    metric=
        "recall",

):


    if not attacks:

        return np.nan


    if metric == "f1":

        metric_key = (
            "f1-score"
        )

    else:

        metric_key = (
            metric
        )


    values = [

        float(

            report[

                attack

            ][

                metric_key

            ]

        )

        for attack
        in attacks

    ]


    return float(

        np.mean(

            values

        )

    )


# ============================================================
# PERSONALIZATION — ONE EPOCH
# ============================================================

def train_one_epoch(

    model,

    optimizer,

    client_file,

    scaler,

    feature_columns,

    active_indices,

    label_mapping,

    class_weights,

    device,

    batch_size,

    chunk_size,

    seed,

    client_id,

    epoch_number,

):


    model.train()


    class_weight_tensor = (

        torch.tensor(

            class_weights,

            dtype=
                torch.float32,

            device=
                device,

        )

    )


    criterion = (

        nn.CrossEntropyLoss(

            weight=
                class_weight_tensor

        )

    )


    # ========================================================
    # DETERMINISTIC PERSONALIZATION SHUFFLE
    # ========================================================

    rng = (

        np.random.default_rng(

            120000

            +

            seed

            +

            client_id
            *
            1000

            +

            epoch_number
            *
            10

        )

    )


    total_loss = 0.0

    total_correct = 0

    total_samples = 0


    started = (

        time.perf_counter()

    )


    reader = pd.read_csv(

        client_file,

        chunksize=
            chunk_size,

    )


    for chunk in reader:


        X = (

            phase4.transform_features(

                df=
                    chunk,

                feature_columns=
                    feature_columns,

                scaler=
                    scaler,

                active_indices=
                    active_indices,

            )

        )


        y = (

            phase4.encode_labels(

                chunk[

                    phase4.LABEL_COL

                ],

                label_mapping,

            )

        )


        if len(
            y
        ) == 0:

            continue


        indices = (

            rng.permutation(

                len(
                    y
                )

            )

        )


        for start in range(

            0,

            len(
                indices
            ),

            batch_size,

        ):


            batch_indices = (

                indices[

                    start:

                    start
                    +
                    batch_size

                ]

            )


            X_batch = (

                torch.from_numpy(

                    X[

                        batch_indices

                    ]

                    .astype(

                        np.float32,

                        copy=
                            False,

                    )

                )

                .to(
                    device
                )

            )


            y_batch = (

                torch.from_numpy(

                    y[

                        batch_indices

                    ]

                    .astype(

                        np.int64,

                        copy=
                            False,

                    )

                )

                .to(
                    device
                )

            )


            optimizer.zero_grad(

                set_to_none=
                    True

            )


            logits = model(

                X_batch

            )


            loss = criterion(

                logits,

                y_batch,

            )


            loss.backward()


            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                max_norm=
                    GRADIENT_CLIP,

            )


            optimizer.step()


            predictions = (

                torch.argmax(

                    logits,

                    dim=
                        1,

                )

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


            total_loss += (

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


    if total_samples == 0:

        raise RuntimeError(

            f"\nClient {client_id} produced zero samples."

        )


    return {

        "epoch":

            epoch_number,

        "loss":

            float(

                total_loss

                /

                total_samples

            ),

        "accuracy":

            float(

                total_correct

                /

                total_samples

            ),

        "samples":

            total_samples,

        "training_seconds":

            float(

                time.perf_counter()

                -

                started

            ),

    }


# ============================================================
# VALIDATION SWEEP
# ============================================================

def run_validation_sweep(

    global_state,

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

    seen_weight,

):


    # Always start from original FedProx70

    model = (

        build_model_from_state(

            global_state,

            input_dim=
                input_dim,

            num_classes=
                len(
                    label_mapping
                ),

            device=
                device,

        )

    )


    optimizer = (

        torch.optim.AdamW(

            model.parameters(),

            lr=
                personalization_lr,

            weight_decay=
                weight_decay,

        )

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


        training_result = (

            train_one_epoch(

                model=
                    model,

                optimizer=
                    optimizer,

                client_file=
                    client_file,

                scaler=
                    scaler,

                feature_columns=
                    feature_columns,

                active_indices=
                    active_indices,

                label_mapping=
                    label_mapping,

                class_weights=
                    class_weights,

                device=
                    device,

                batch_size=
                    batch_size,

                chunk_size=
                    chunk_size,

                seed=
                    seed,

                client_id=
                    client_id,

                epoch_number=
                    epoch,

            )

        )


        training_result.update({

            "seed":

                seed,

            "client_id":

                client_id,

        })


        history_rows.append(

            training_result

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

            model=
                model,

            csv_file=
                validation_file,

            scaler=
                scaler,

            feature_columns=
                feature_columns,

            active_indices=
                active_indices,

            label_mapping=
                label_mapping,

            device=
                device,

            batch_size=
                batch_size,

            chunk_size=
                chunk_size,

        )


        seen_recall = (

            mean_attack_metric(

                report,

                seen_attacks,

                "recall",

            )

        )


        unseen_recall = (

            mean_attack_metric(

                report,

                unseen_attacks,

                "recall",

            )

        )


        seen_f1 = (

            mean_attack_metric(

                report,

                seen_attacks,

                "f1",

            )

        )


        unseen_f1 = (

            mean_attack_metric(

                report,

                unseen_attacks,

                "f1",

            )

        )


        score = (

            weighted_seen_unseen_score(

                seen_recall,

                unseen_recall,

                seen_weight,

            )

        )


        sweep_rows.append({

            "seed":

                seed,

            "client_id":

                client_id,

            "personalization_epochs":

                epoch,

            "validation_seen_recall":

                seen_recall,

            "validation_unseen_recall":

                unseen_recall,

            "validation_seen_f1":

                seen_f1,

            "validation_unseen_f1":

                unseen_f1,

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

                score,

            "harmonic_balance":

                harmonic_mean(

                    seen_recall,

                    unseen_recall,

                ),

            "validation_inference_seconds":

                validation_seconds,

        })


        logger.info(

            "Validation | "

            "seed=%d | "

            "client=%d | "

            "epochs=%d | "

            "seen=%.6f | "

            "unseen=%.6f | "

            "score=%.6f | "

            "macro_f1=%.6f",

            seed,

            client_id,

            epoch,

            seen_recall,

            unseen_recall,

            score,

            overall[

                "macro_f1"

            ],

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
# SELECT BEST VALIDATION EPOCH
# ============================================================

def choose_best_epoch(

    sweep_df: pd.DataFrame,

) -> dict:


    if sweep_df.empty:

        raise RuntimeError(

            "\nValidation sweep is empty."

        )


    ranked = (

        sweep_df

        .sort_values(

            by=[

                "selection_score",

                "validation_macro_f1",

                "harmonic_balance",

                "personalization_epochs",

            ],

            ascending=[

                False,

                False,

                False,

                True,

            ],

        )

    )


    return ranked.iloc[

        0

    ].to_dict()


# ============================================================
# RETRAIN SELECTED PERSONALIZED MODEL
# ============================================================

def train_selected_personalized_model(

    global_state,

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


    # ========================================================
    # IMPORTANT:
    # Start again from original global model
    # ========================================================

    model = (

        build_model_from_state(

            global_state,

            input_dim=
                input_dim,

            num_classes=
                len(
                    label_mapping
                ),

            device=
                device,

        )

    )


    optimizer = (

        torch.optim.AdamW(

            model.parameters(),

            lr=
                personalization_lr,

            weight_decay=
                weight_decay,

        )

    )


    history_rows = []


    started = (

        time.perf_counter()

    )


    for epoch in range(

        1,

        selected_epochs + 1,

    ):


        row = (

            train_one_epoch(

                model=
                    model,

                optimizer=
                    optimizer,

                client_file=
                    client_file,

                scaler=
                    scaler,

                feature_columns=
                    feature_columns,

                active_indices=
                    active_indices,

                label_mapping=
                    label_mapping,

                class_weights=
                    class_weights,

                device=
                    device,

                batch_size=
                    batch_size,

                chunk_size=
                    chunk_size,

                seed=
                    seed,

                client_id=
                    client_id,

                epoch_number=
                    epoch,

            )

        )


        row.update({

            "seed":

                seed,

            "client_id":

                client_id,

            "selected_epochs":

                selected_epochs,

        })


        history_rows.append(

            row

        )


    training_seconds = float(

        time.perf_counter()

        -

        started

    )


    return (

        model,

        pd.DataFrame(

            history_rows

        ),

        training_seconds,

    )


# ============================================================
# PER-CLASS OUTPUT
# ============================================================

def report_to_df(

    report,

    class_names,

    seed,

    client_id,

    selected_epochs,

):


    rows = []


    for class_name in class_names:


        metrics = report[

            class_name

        ]


        rows.append({

            "seed":

                seed,

            "client_id":

                client_id,

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

        })


    return pd.DataFrame(

        rows

    )


# ============================================================
# STATISTICS
# ============================================================

def mean_std_ci95(

    values,

):


    values = np.asarray(

        values,

        dtype=
            float,

    )


    values = values[

        np.isfinite(

            values

        )

    ]


    if len(
        values
    ) == 0:

        return (

            np.nan,

            np.nan,

            np.nan,

            np.nan,

        )


    mean_value = float(

        values.mean()

    )


    if len(
        values
    ) == 1:

        return (

            mean_value,

            0.0,

            mean_value,

            mean_value,

        )


    std_value = float(

        values.std(

            ddof=
                1

        )

    )


    if SCIPY_AVAILABLE:

        critical = float(

            student_t.ppf(

                0.975,

                df=

                    len(
                        values
                    )

                    -

                    1,

            )

        )


    else:

        critical = (
            1.96
        )


    margin = (

        critical

        *

        std_value

        /

        math.sqrt(

            len(
                values
            )

        )

    )


    return (

        mean_value,

        std_value,

        mean_value
        -
        margin,

        mean_value
        +
        margin,

    )


# ============================================================
# WILCOXON
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

        dtype=
            float,

    )


    right = np.asarray(

        right,

        dtype=
            float,

    )


    valid = (

        np.isfinite(
            left
        )

        &

        np.isfinite(
            right
        )

    )


    left = left[

        valid

    ]


    right = right[

        valid

    ]


    if len(
        left
    ) == 0:

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


    result = (

        wilcoxon(

            left,

            right,

            alternative=
                "two-sided",

            zero_method=
                "wilcox",

            method=
                "auto",

        )

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

    scenario: str,

):


    scenario_root = (

        RESULT_ROOT

        / scenario

    )


    aggregate_dir = (

        scenario_root

        / "aggregate"

    )


    aggregate_dir.mkdir(

        parents=
            True,

        exist_ok=
            True,

    )


    final_frames = []

    per_class_frames = []

    sweep_frames = []

    completed_seeds = []


    for seed_dir in (

        scenario_root.glob(

            "seed_*"

        )

    ):


        if not seed_dir.is_dir():

            continue


        final_file = (

            seed_dir

            / "final_personalization_results.csv"

        )


        per_class_file = (

            seed_dir

            / "final_personalized_per_class_metrics.csv"

        )


        sweep_file = (

            seed_dir

            / "validation_sweep.csv"

        )


        if (

            not final_file.exists()

            or

            not per_class_file.exists()

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


        per_class_frames.append(

            pd.read_csv(

                per_class_file

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


    final_all = (

        pd.concat(

            final_frames,

            ignore_index=
                True,

        )

    )


    per_class_all = (

        pd.concat(

            per_class_frames,

            ignore_index=
                True,

        )

    )


    sweep_all = (

        pd.concat(

            sweep_frames,

            ignore_index=
                True,

        )

    )


    # ========================================================
    # SAVE COMBINED FILES
    # ========================================================

    final_all.to_csv(

        aggregate_dir

        / "final_personalization_all_seeds.csv",

        index=
            False,

    )


    per_class_all.to_csv(

        aggregate_dir

        / "final_personalized_per_class_all_seeds.csv",

        index=
            False,

    )


    sweep_all.to_csv(

        aggregate_dir

        / "validation_sweep_all_seeds.csv",

        index=
            False,

    )


    # ========================================================
    # SELECTED EPOCH FREQUENCY
    # ========================================================

    (

        final_all

        .groupby(

            [

                "client_id",

                "selected_epochs",

            ]

        )

        .size()

        .reset_index(

            name=
                "count"

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

            / "selected_epoch_frequency.csv",

            index=
                False,

        )

    )


    # ========================================================
    # PER-CLIENT SUMMARY
    # ========================================================

    metrics = [

        "local_only_seen_recall",

        "global_seen_recall",

        "personalized_seen_recall",

        "personalization_gain",

        "local_recovery_ratio",

        "local_only_unseen_recall",

        "global_unseen_recall",

        "personalized_unseen_recall",

        "global_knowledge_loss",

        "global_knowledge_preservation_ratio",

        "global_utility_score",

        "personalized_utility_score",

        "personalized_macro_f1",

        "personalized_balanced_accuracy",

    ]


    summary_rows = []


    for (

        client_id,

        group,

    ) in final_all.groupby(

        "client_id"

    ):


        for metric in metrics:


            (

                mean_value,

                std_value,

                ci_lower,

                ci_upper,

            ) = mean_std_ci95(

                group[

                    metric

                ]

                .to_numpy(

                    dtype=
                        float

                )

            )


            summary_rows.append({

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

            })


    pd.DataFrame(

        summary_rows

    ).to_csv(

        aggregate_dir

        / "per_client_summary_across_seeds.csv",

        index=
            False,

    )


    # ========================================================
    # SEED-LEVEL SUMMARY
    #
    # Average clients within each seed before significance
    # testing so the statistical unit remains the seed.
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

                "local_only_unseen_recall",

                "global_unseen_recall",

                "personalized_unseen_recall",

                "global_utility_score",

                "personalized_utility_score",

                "personalization_gain",

                "global_knowledge_loss",

                "global_knowledge_preservation_ratio",

                "local_recovery_ratio",

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

        / "seed_level_summary.csv",

        index=
            False,

    )


    # ========================================================
    # PAIRED TESTS
    # ========================================================

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

        comparison_name,

        left_column,

        right_column,

    ) in comparisons:


        left = (

            seed_level[

                left_column

            ]

            .to_numpy(

                dtype=
                    float

            )

        )


        right = (

            seed_level[

                right_column

            ]

            .to_numpy(

                dtype=
                    float

            )

        )


        differences = (

            left

            -

            right

        )


        (

            statistic,

            p_value,

        ) = safe_wilcoxon(

            left,

            right,

        )


        test_rows.append({

            "comparison":

                comparison_name,

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

                        differences

                    )

                ),

            "wins_left":

                int(

                    np.sum(

                        differences

                        >

                        0

                    )

                ),

            "losses_left":

                int(

                    np.sum(

                        differences

                        <

                        0

                    )

                ),

            "ties":

                int(

                    np.sum(

                        np.isclose(

                            differences,

                            0.0,

                        )

                    )

                ),

            "wilcoxon_statistic":

                statistic,

            "p_value":

                p_value,

        })


    pd.DataFrame(

        test_rows

    ).to_csv(

        aggregate_dir

        / "seed_level_paired_tests.csv",

        index=
            False,

    )


    # ========================================================
    # FINAL AGGREGATE SUMMARY
    # ========================================================

    preservation_values = (

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


    local_recovery_values = (

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

            12,

        "experiment":

            "Personalized FedProx70",

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

                ]

                .mean()

            ),

        "mean_global_knowledge_loss":

            float(

                final_all[

                    "global_knowledge_loss"

                ]

                .mean()

            ),

        "mean_global_knowledge_preservation_ratio":

            float(

                preservation_values.mean()

            )

            if

            len(
                preservation_values
            )

            >

            0

            else

            None,

        "mean_local_recovery_ratio":

            float(

                local_recovery_values.mean()

            )

            if

            len(
                local_recovery_values
            )

            >

            0

            else

            None,

        "mean_global_utility":

            float(

                final_all[

                    "global_utility_score"

                ]

                .mean()

            ),

        "mean_personalized_utility":

            float(

                final_all[

                    "personalized_utility_score"

                ]

                .mean()

            ),

        "locked_test_used_for_selection":

            False,

        "selection_basis":

            "validation set only",

        "scipy_available":

            SCIPY_AVAILABLE,

    }


    save_json(

        aggregate_dir

        / "phase12_final_summary.json",

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

        default=
            "non_iid",

    )


    parser.add_argument(

        "--seed",

        type=
            int,

        default=
            DEFAULT_SEED,

    )


    parser.add_argument(

        "--validation-file",

        type=
            Path,

        default=
            None,

        help=(
            "Held-out validation.csv. "
            "Used only for choosing personalization epochs."
        ),

    )


    parser.add_argument(

        "--epoch-budgets",

        nargs=
            "+",

        type=
            int,

        default=
            DEFAULT_EPOCH_BUDGETS,

    )


    parser.add_argument(

        "--personalization-lr",

        type=
            float,

        default=
            DEFAULT_PERSONALIZATION_LR,

    )


    parser.add_argument(

        "--seen-weight",

        type=
            float,

        default=
            DEFAULT_SEEN_WEIGHT,

        help=(
            "Validation weighting. "
            "0.5 gives equal importance to "
            "seen and unseen recall."
        ),

    )


    parser.add_argument(

        "--batch-size",

        type=
            int,

        default=
            DEFAULT_BATCH_SIZE,

    )


    parser.add_argument(

        "--chunk-size",

        type=
            int,

        default=
            DEFAULT_CHUNK_SIZE,

    )


    parser.add_argument(

        "--weight-decay",

        type=
            float,

        default=
            DEFAULT_WEIGHT_DECAY,

    )


    args = parser.parse_args()


    epoch_budgets = sorted(

        set(

            args.epoch_budgets

        )

    )


    if (

        not epoch_budgets

        or

        min(
            epoch_budgets
        )

        <

        1

    ):

        raise ValueError(

            "--epoch-budgets must contain positive integers."

        )


    if not (

        0.0

        <=

        args.seen_weight

        <=

        1.0

    ):

        raise ValueError(

            "--seen-weight must be between 0 and 1."

        )


    if (

        args.personalization_lr

        <=

        0.0

    ):

        raise ValueError(

            "--personalization-lr must be > 0."

        )


    set_seed(

        args.seed

    )


    device = torch.device(

        "cuda"

        if

        torch.cuda.is_available()

        else

        "cpu"

    )


    validation_file = (

        resolve_validation_file(

            args.validation_file

        )

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

        parents=
            True,

        exist_ok=
            True,

    )


    artifact_dir.mkdir(

        parents=
            True,

        exist_ok=
            True,

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


    benign_label = (

        detect_benign_label(

            label_mapping

        )

    )


    attack_classes = [

        label

        for label
        in label_mapping.keys()

        if

        label

        !=

        benign_label

    ]


    # ========================================================
    # SAME CLASS WEIGHTS
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
    # LOAD PHASE-10 GLOBAL MODEL
    # ========================================================

    (

        checkpoint,

        global_model_file,

    ) = load_fedprox70_checkpoint(

        scenario=
            args.scenario,

        seed=
            args.seed,

        device=
            device,

        active_features=
            all_features,

    )


    global_state = {

        key:

            value

            .detach()

            .cpu()

            .clone()

        for key, value
        in checkpoint[

            "model_state_dict"

        ].items()

    }


    # ========================================================
    # PHASE-11 LOCAL-ONLY REFERENCE
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

    global_model = (

        build_model_from_state(

            global_state,

            input_dim=
                len(
                    all_features
                ),

            num_classes=
                len(
                    label_mapping
                ),

            device=
                device,

        )

    )


    # ========================================================
    # GLOBAL VALIDATION BASELINE
    # ========================================================

    (

        _,

        global_validation_overall,

        global_validation_report,

        _,

        _,

    ) = evaluate_model_on_csv(

        model=
            global_model,

        csv_file=
            validation_file,

        scaler=
            scaler,

        feature_columns=
            feature_columns,

        active_indices=
            all_active_indices,

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
    # GLOBAL LOCKED-TEST BASELINE
    # ========================================================

    (

        global_test_seconds,

        global_test_overall,

        global_test_report,

        _,

        global_class_names,

    ) = evaluate_locked_test(

        model=
            global_model,

        scaler=
            scaler,

        feature_columns=
            feature_columns,

        active_indices=
            all_active_indices,

        label_mapping=
            label_mapping,

        device=
            device,

        batch_size=
            args.batch_size,

        chunk_size=
            args.chunk_size,

    )


    del global_model


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


    # ========================================================
    # HEADER
    # ========================================================

    print()

    print(
        "=" * 100
    )


    print(
        "PHASE 12 — PERSONALIZED FEDPROX70"
    )


    print(
        "=" * 100
    )


    print(
        f"Scenario              : {args.scenario}"
    )


    print(
        f"Seed                  : {args.seed}"
    )


    print(
        f"Device                : {device}"
    )


    print(
        f"Features              : {len(all_features)}"
    )


    print(
        f"Validation file       : {validation_file}"
    )


    print(
        f"Candidate epochs      : {epoch_budgets}"
    )


    print(
        f"Personalization LR    : {args.personalization_lr}"
    )


    print(
        f"Seen weight           : {args.seen_weight}"
    )


    print(
        "Locked test selection : NO ✅"
    )


    # ========================================================
    # OUTPUT COLLECTIONS
    # ========================================================

    validation_sweep_frames = []

    training_history_frames = []

    final_history_frames = []

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


        counts = (

            read_client_label_counts(

                client_file,

                args.chunk_size,

                label_mapping,

            )

        )


        seen_attacks = [

            attack

            for attack
            in attack_classes

            if

            counts.get(

                attack,

                0,

            )

            >

            0

        ]


        unseen_attacks = [

            attack

            for attack
            in attack_classes

            if

            counts.get(

                attack,

                0,

            )

            ==

            0

        ]


        if (

            not seen_attacks

            or

            not unseen_attacks

        ):

            raise RuntimeError(

                f"\nClient {client_id}: "

                "invalid seen/unseen attack split."

            )


        print()

        print(
            "-" * 100
        )


        print(
            f"CLIENT {client_id}"
        )


        print(
            "-" * 100
        )


        print(
            f"Seen attacks   : {seen_attacks}"
        )


        print(
            f"Unseen attacks : {unseen_attacks}"
        )


        # ====================================================
        # DISTRIBUTION
        # ====================================================

        for class_name in (

            label_mapping.keys()

        ):


            distribution_rows.append({

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

                    bool(

                        class_name

                        in

                        seen_attacks

                    ),

                "is_unseen_attack":

                    bool(

                        class_name

                        in

                        unseen_attacks

                    ),

            })


        # ====================================================
        # LOCAL-ONLY REFERENCE
        # ====================================================

        local_client_df = (

            local_only_df[

                local_only_df[

                    "client_id"

                ]

                ==

                client_id

            ]

        )


        local_map = (

            local_client_df

            .set_index(

                "class"

            )

        )


        local_seen_recall = float(

            np.mean(

                [

                    float(

                        local_map.loc[

                            attack,

                            "recall",

                        ]

                    )

                    for attack
                    in seen_attacks

                ]

            )

        )


        local_unseen_recall = float(

            np.mean(

                [

                    float(

                        local_map.loc[

                            attack,

                            "recall",

                        ]

                    )

                    for attack
                    in unseen_attacks

                ]

            )

        )


        # ====================================================
        # GLOBAL LOCKED-TEST SEEN / UNSEEN
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
        # VALIDATION SWEEP
        # ====================================================

        (

            validation_sweep_df,

            training_history_df,

        ) = run_validation_sweep(

            global_state=
                global_state,

            epoch_budgets=
                epoch_budgets,

            client_id=
                client_id,

            client_file=
                client_file,

            validation_file=
                validation_file,

            seen_attacks=
                seen_attacks,

            unseen_attacks=
                unseen_attacks,

            scaler=
                scaler,

            feature_columns=
                feature_columns,

            active_indices=
                all_active_indices,

            label_mapping=
                label_mapping,

            class_weights=
                global_weights,

            input_dim=
                len(
                    all_features
                ),

            device=
                device,

            batch_size=
                args.batch_size,

            chunk_size=
                args.chunk_size,

            personalization_lr=
                args.personalization_lr,

            weight_decay=
                args.weight_decay,

            seed=
                args.seed,

            seen_weight=
                args.seen_weight,

        )


        validation_sweep_df[

            "seen_attacks"

        ] = (

            " | ".join(

                seen_attacks

            )

        )


        validation_sweep_df[

            "unseen_attacks"

        ] = (

            " | ".join(

                unseen_attacks

            )

        )


        validation_sweep_frames.append(

            validation_sweep_df

        )


        training_history_frames.append(

            training_history_df

        )


        best = (

            choose_best_epoch(

                validation_sweep_df

            )

        )


        selected_epochs = int(

            best[

                "personalization_epochs"

            ]

        )


        print(

            "Selected using VALIDATION only: "

            f"{selected_epochs} epoch(s) | "

            f"score={float(best['selection_score']):.6f}"

        )


        # ====================================================
        # RETRAIN FROM ORIGINAL GLOBAL MODEL
        # ====================================================

        (

            personalized_model,

            final_training_history,

            personalization_training_seconds,

        ) = train_selected_personalized_model(

            global_state=
                global_state,

            selected_epochs=
                selected_epochs,

            client_id=
                client_id,

            client_file=
                client_file,

            scaler=
                scaler,

            feature_columns=
                feature_columns,

            active_indices=
                all_active_indices,

            label_mapping=
                label_mapping,

            class_weights=
                global_weights,

            input_dim=
                len(
                    all_features
                ),

            device=
                device,

            batch_size=
                args.batch_size,

            chunk_size=
                args.chunk_size,

            personalization_lr=
                args.personalization_lr,

            weight_decay=
                args.weight_decay,

            seed=
                args.seed,

        )


        final_history_frames.append(

            final_training_history

        )


        # ====================================================
        # LOCKED TEST — SELECTED MODEL ONLY
        # ====================================================

        (

            personalized_test_seconds,

            personalized_overall,

            personalized_report,

            _,

            personalized_class_names,

        ) = evaluate_locked_test(

            model=
                personalized_model,

            scaler=
                scaler,

            feature_columns=
                feature_columns,

            active_indices=
                all_active_indices,

            label_mapping=
                label_mapping,

            device=
                device,

            batch_size=
                args.batch_size,

            chunk_size=
                args.chunk_size,

        )


        personalized_seen_recall = (

            mean_attack_metric(

                personalized_report,

                seen_attacks,

                "recall",

            )

        )


        personalized_unseen_recall = (

            mean_attack_metric(

                personalized_report,

                unseen_attacks,

                "recall",

            )

        )


        personalization_gain = (

            personalized_seen_recall

            -

            global_seen_recall

        )


        global_knowledge_loss = (

            global_unseen_recall

            -

            personalized_unseen_recall

        )


        knowledge_preservation_ratio = (

            safe_ratio(

                personalized_unseen_recall,

                global_unseen_recall,

            )

        )


        local_recovery_ratio = (

            safe_ratio(

                personalized_seen_recall,

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

            personalized_seen_recall

            +

            0.5

            *

            personalized_unseen_recall

        )


        # ====================================================
        # FINAL RESULT ROW
        # ====================================================

        final_rows.append({

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

            "selected_epochs":

                selected_epochs,

            "validation_selection_score":

                float(

                    best[

                        "selection_score"

                    ]

                ),

            "validation_seen_recall":

                float(

                    best[

                        "validation_seen_recall"

                    ]

                ),

            "validation_unseen_recall":

                float(

                    best[

                        "validation_unseen_recall"

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

                personalized_seen_recall,

            "personalized_unseen_recall":

                personalized_unseen_recall,

            "personalization_gain":

                personalization_gain,

            "global_knowledge_loss":

                global_knowledge_loss,

            "global_knowledge_preservation_ratio":

                knowledge_preservation_ratio,

            "local_recovery_ratio":

                local_recovery_ratio,

            "global_utility_score":

                global_utility,

            "personalized_utility_score":

                personalized_utility,

            "personalized_harmonic_balance":

                harmonic_mean(

                    personalized_seen_recall,

                    personalized_unseen_recall,

                ),

            "personalized_accuracy":

                float(

                    personalized_overall[

                        "accuracy"

                    ]

                ),

            "personalized_balanced_accuracy":

                float(

                    personalized_overall[

                        "balanced_accuracy"

                    ]

                ),

            "personalized_macro_f1":

                float(

                    personalized_overall[

                        "macro_f1"

                    ]

                ),

            "personalized_weighted_f1":

                float(

                    personalized_overall[

                        "weighted_f1"

                    ]

                ),

            "global_accuracy":

                float(

                    global_test_overall[

                        "accuracy"

                    ]

                ),

            "global_balanced_accuracy":

                float(

                    global_test_overall[

                        "balanced_accuracy"

                    ]

                ),

            "global_macro_f1":

                float(

                    global_test_overall[

                        "macro_f1"

                    ]

                ),

            "personalization_training_seconds":

                personalization_training_seconds,

            "personalized_test_inference_seconds":

                personalized_test_seconds,

        })


        # ====================================================
        # PER-CLASS
        # ====================================================

        per_class_frames.append(

            report_to_df(

                personalized_report,

                personalized_class_names,

                seed=
                    args.seed,

                client_id=
                    client_id,

                selected_epochs=
                    selected_epochs,

            )

        )


        # ====================================================
        # SAVE PERSONALIZED MODEL
        # ====================================================

        client_artifact_dir = (

            artifact_dir

            / f"client_{client_id}"

        )


        client_artifact_dir.mkdir(

            parents=
                True,

            exist_ok=
                True,

        )


        torch.save(

            {

                "phase":

                    12,

                "algorithm":

                    "Personalized FedProx70",

                "scenario":

                    args.scenario,

                "seed":

                    args.seed,

                "client_id":

                    client_id,

                "selected_epochs":

                    selected_epochs,

                "personalization_learning_rate":

                    args.personalization_lr,

                "seen_weight":

                    args.seen_weight,

                "validation_file":

                    str(

                        validation_file

                    ),

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

            / "personalized_fedprox70_model.pt",

        )


        # ====================================================
        # DISPLAY CLIENT RESULT
        # ====================================================

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

            f"{personalized_seen_recall:.6f}"

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

            f"{personalized_unseen_recall:.6f}"

        )


        print(

            f"Knowledge loss        : "

            f"{global_knowledge_loss:+.6f}"

        )


        if np.isfinite(

            knowledge_preservation_ratio

        ):

            print(

                "Knowledge preserved   : "

                f"{knowledge_preservation_ratio * 100:.2f}%"

            )


        if np.isfinite(

            local_recovery_ratio

        ):

            print(

                "Local recovery        : "

                f"{local_recovery_ratio * 100:.2f}%"

            )


        del personalized_model


        if torch.cuda.is_available():

            torch.cuda.empty_cache()


    # ========================================================
    # BUILD OUTPUT DATAFRAMES
    # ========================================================

    validation_sweep_df = (

        pd.concat(

            validation_sweep_frames,

            ignore_index=
                True,

        )

    )


    training_history_df = (

        pd.concat(

            training_history_frames,

            ignore_index=
                True,

        )

    )


    final_training_history_df = (

        pd.concat(

            final_history_frames,

            ignore_index=
                True,

        )

    )


    final_df = (

        pd.DataFrame(

            final_rows

        )

    )


    per_class_df = (

        pd.concat(

            per_class_frames,

            ignore_index=
                True,

        )

    )


    distribution_df = (

        pd.DataFrame(

            distribution_rows

        )

    )


    # ========================================================
    # SAVE FILES
    # ========================================================

    validation_sweep_df.to_csv(

        result_dir

        / "validation_sweep.csv",

        index=
            False,

    )


    training_history_df.to_csv(

        result_dir

        / "personalization_training_history.csv",

        index=
            False,

    )


    final_training_history_df.to_csv(

        result_dir

        / "selected_personalization_training_history.csv",

        index=
            False,

    )


    final_df.to_csv(

        result_dir

        / "final_personalization_results.csv",

        index=
            False,

    )


    per_class_df.to_csv(

        result_dir

        / "final_personalized_per_class_metrics.csv",

        index=
            False,

    )


    distribution_df.to_csv(

        result_dir

        / "client_training_distribution.csv",

        index=
            False,

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


    seed_summary = {

        "phase":

            12,

        "experiment":

            "Validation-selected Personalized FedProx70",

        "scenario":

            args.scenario,

        "seed":

            args.seed,

        "features":

            70,

        "global_model":

            str(

                global_model_file

            ),

        "validation_file":

            str(

                validation_file

            ),

        "candidate_epochs":

            epoch_budgets,

        "personalization_learning_rate":

            args.personalization_lr,

        "seen_weight":

            args.seen_weight,

        "unseen_weight":

            1.0

            -

            args.seen_weight,

        "selection_formula":

            (

                "seen_weight * validation_seen_recall + "

                "(1-seen_weight) * validation_unseen_recall"

            ),

        "global_validation_metrics":

            global_validation_overall,

        "global_locked_test_metrics":

            global_test_overall,

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

            float(

                preservation_values.mean()

            )

            if

            len(
                preservation_values
            )

            >

            0

            else

            None,

        "mean_local_recovery_ratio":

            float(

                recovery_values.mean()

            )

            if

            len(
                recovery_values
            )

            >

            0

            else

            None,

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

        "locked_test_used_for_selection":

            False,

        "locked_test_usage":

            (

                "Final evaluation only after "

                "validation selected personalization epochs."

            ),

    }


    save_json(

        result_dir

        / "phase12_summary.json",

        seed_summary,

    )


    # ========================================================
    # AGGREGATE ALL COMPLETED SEEDS
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

    print(
        "=" * 100
    )


    print(
        "PHASE 12 SEED COMPLETE"
    )


    print(
        "=" * 100
    )


    display_columns = [

        "client_id",

        "seen_attacks",

        "selected_epochs",

        "local_only_seen_recall",

        "global_seen_recall",

        "personalized_seen_recall",

        "personalization_gain",

        "global_unseen_recall",

        "personalized_unseen_recall",

        "global_knowledge_loss",

        "global_knowledge_preservation_ratio",

        "local_recovery_ratio",

        "personalized_macro_f1",

    ]


    print()

    print(

        final_df[

            display_columns

        ]

        .to_string(

            index=
                False

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

        print(
            "=" * 100
        )


        print(
            "CURRENT MULTI-SEED AGGREGATE"
        )


        print(
            "=" * 100
        )


        print(

            "Completed seeds           : "

            f"{aggregate_summary['completed_seeds']}"

        )


        print(

            "Mean personalization gain : "

            f"{aggregate_summary['mean_personalization_gain']:.6f}"

        )


        print(

            "Mean global knowledge loss: "

            f"{aggregate_summary['mean_global_knowledge_loss']:.6f}"

        )


        preservation = (

            aggregate_summary[

                "mean_global_knowledge_preservation_ratio"

            ]

        )


        if preservation is not None:

            print(

                "Mean knowledge preserved  : "

                f"{preservation * 100:.2f}%"

            )


        recovery = (

            aggregate_summary[

                "mean_local_recovery_ratio"

            ]

        )


        if recovery is not None:

            print(

                "Mean local recovery        : "

                f"{recovery * 100:.2f}%"

            )


        print(

            "Global utility             : "

            f"{aggregate_summary['mean_global_utility']:.6f}"

        )


        print(

            "Personalized utility       : "

            f"{aggregate_summary['mean_personalized_utility']:.6f}"

        )


    if not SCIPY_AVAILABLE:


        print()

        print(

            "SciPy is not installed."

        )


        print(

            "Install it for aggregate Wilcoxon tests:"

        )


        print(

            "pip install scipy"

        )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()