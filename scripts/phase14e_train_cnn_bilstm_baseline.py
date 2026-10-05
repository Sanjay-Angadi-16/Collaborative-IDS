from __future__ import annotations

import gc
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

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# ============================================================
# EXISTING PROJECT INFRASTRUCTURE
# ============================================================

try:
    import phase9_fedavg70 as phase4
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "\nCould not import scripts\\phase9_fedavg70.py\n"
        "Keep this script inside the scripts folder."
    ) from exc


# ============================================================
# PHASE
# ============================================================

PHASE_NAME = (
    "PHASE 14E — CENTRALIZED CNN-BILSTM BASELINE TRAINING"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42


# ============================================================
# DATA / MODEL CONFIGURATION
# ============================================================

EXPECTED_SEQUENCE_LENGTH = 20
EXPECTED_FEATURES = 70

CONV1_CHANNELS = 128
CONV2_CHANNELS = 64

LSTM_HIDDEN_SIZE = 64
LSTM_LAYERS = 1
BIDIRECTIONAL = True

FC_HIDDEN_SIZE = 64

DROPOUT = 0.25

NUM_CLASSES = 6


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 256

MAX_EPOCHS = 15

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 4

NUM_WORKERS = 0

GRADIENT_CLIP_NORM = 5.0


# ============================================================
# CLASS WEIGHT CONFIGURATION
# ============================================================

CLASS_WEIGHT_MODE = "sqrt_max_over_count"

CLASS_WEIGHT_CAP = 20.0


# ============================================================
# PATHS
# ============================================================

SEQUENCE_ROOT = (
    PROJECT_ROOT
    /
    "data"
    /
    "sequences"
)


GLOBAL_TRAIN_FILE = (
    SEQUENCE_ROOT
    /
    "global_train_sequences.npz"
)


VALIDATION_FILE = (
    SEQUENCE_ROOT
    /
    "validation_sequences.npz"
)


RESULT_ROOT = (
    PROJECT_ROOT
    /
    "results"
    /
    "phase14"
)


ARTIFACT_ROOT = (
    PROJECT_ROOT
    /
    "artifacts"
    /
    "detection"
)


MODEL_FILE = (
    ARTIFACT_ROOT
    /
    "cnn_bilstm_centralized.pt"
)


METADATA_FILE = (
    ARTIFACT_ROOT
    /
    "cnn_bilstm_centralized_metadata.json"
)


TRAINING_HISTORY_FILE = (
    RESULT_ROOT
    /
    "phase14e_training_history.csv"
)


VALIDATION_METRICS_FILE = (
    RESULT_ROOT
    /
    "phase14e_validation_metrics.json"
)


PER_CLASS_METRICS_FILE = (
    RESULT_ROOT
    /
    "phase14e_per_class_metrics.csv"
)


CONFUSION_MATRIX_FILE = (
    RESULT_ROOT
    /
    "phase14e_confusion_matrix.csv"
)


CLASS_WEIGHTS_FILE = (
    RESULT_ROOT
    /
    "phase14e_class_weights.csv"
)


LOCKED_TEST_FILE = (
    PROJECT_ROOT
    /
    "data"
    /
    "processed"
    /
    "test_locked.csv"
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
    "phase14e_cnn_bilstm"
)


# ============================================================
# HELPERS
# ============================================================

def set_seed(seed: int) -> None:

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


def separator() -> None:

    print()

    print(
        "=" * 112
    )


def json_safe(value):

    if isinstance(
        value,
        dict,
    ):

        return {

            str(key):
                json_safe(item)

            for key, item
            in value.items()
        }


    if isinstance(
        value,
        (list, tuple),
    ):

        return [

            json_safe(item)

            for item
            in value
        ]


    if isinstance(
        value,
        np.ndarray,
    ):

        return value.tolist()


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


    if isinstance(
        value,
        Path,
    ):

        return str(
            value
        )


    return value


def save_json(
    path: Path,
    payload,
) -> None:

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            json_safe(
                payload
            ),
            file,
            indent=4,
        )


def resolve_device() -> torch.device:

    if torch.cuda.is_available():

        return torch.device(
            "cuda"
        )


    return torch.device(
        "cpu"
    )


def get_inverse_mapping(
    label_mapping,
):

    return {

        int(class_id):
            str(label)

        for label, class_id
        in label_mapping.items()
    }


# ============================================================
# DATA LOADING
# ============================================================

def load_sequence_npz(
    path: Path,
    dataset_name: str,
):

    if not path.exists():

        raise FileNotFoundError(
            f"\nMissing {dataset_name} NPZ:\n"
            f"{path}"
        )


    logger.info(
        "Loading %s...",
        dataset_name,
    )


    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        if (
            "X" not in data

            or

            "y" not in data
        ):

            raise KeyError(
                f"\n{dataset_name} NPZ must contain X and y."
            )


        X = data[
            "X"
        ].astype(
            np.float32,
            copy=False,
        )


        y = data[
            "y"
        ].astype(
            np.int64,
            copy=False,
        )


    if X.ndim != 3:

        raise ValueError(
            f"\n{dataset_name}: expected X to be 3D, "
            f"found shape {X.shape}."
        )


    if tuple(
        X.shape[
            1:
        ]
    ) != (
        EXPECTED_SEQUENCE_LENGTH,
        EXPECTED_FEATURES,
    ):

        raise ValueError(
            f"\n{dataset_name}: expected shape "
            f"(*, {EXPECTED_SEQUENCE_LENGTH}, "
            f"{EXPECTED_FEATURES}), "
            f"found {X.shape}."
        )


    if y.ndim != 1:

        raise ValueError(
            f"\n{dataset_name}: expected y to be 1D, "
            f"found shape {y.shape}."
        )


    if len(
        X
    ) != len(
        y
    ):

        raise ValueError(
            f"\n{dataset_name}: X/y count mismatch."
        )


    unique_classes = np.unique(
        y
    )


    if np.any(
        unique_classes
        <
        0
    ):

        raise ValueError(
            f"\n{dataset_name}: negative class id found."
        )


    if np.any(
        unique_classes
        >=
        NUM_CLASSES
    ):

        raise ValueError(
            f"\n{dataset_name}: class id outside "
            f"0..{NUM_CLASSES - 1}."
        )


    return (
        X,
        y,
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

def compute_class_weights(
    y_train: np.ndarray,
):

    counts = np.bincount(
        y_train,
        minlength=NUM_CLASSES,
    ).astype(
        np.float64
    )


    if np.any(
        counts
        <=
        0
    ):

        missing = np.flatnonzero(
            counts
            <=
            0
        )

        raise RuntimeError(
            "\nGlobal Phase14E training sequences are "
            "missing class ids:\n"
            f"{missing.tolist()}"
        )


    max_count = float(
        np.max(
            counts
        )
    )


    weights = np.sqrt(
        max_count
        /
        counts
    )


    weights = np.minimum(
        weights,
        CLASS_WEIGHT_CAP,
    )


    weights = (
        weights
        /
        np.mean(
            weights
        )
    )


    return (
        counts,
        weights.astype(
            np.float32
        ),
    )


# ============================================================
# MODEL
# ============================================================

class CNNBiLSTM(
    nn.Module,
):

    def __init__(
        self,
        feature_count: int,
        num_classes: int,
    ):

        super().__init__()


        self.feature_count = feature_count

        self.num_classes = num_classes


        # ----------------------------------------------------
        # CNN extracts local patterns across neighbouring flows.
        #
        # Input to Conv1d:
        # (batch, features, sequence_length)
        # ----------------------------------------------------

        self.conv1 = nn.Conv1d(

            in_channels=
                feature_count,

            out_channels=
                CONV1_CHANNELS,

            kernel_size=
                3,

            padding=
                1,
        )


        self.conv2 = nn.Conv1d(

            in_channels=
                CONV1_CHANNELS,

            out_channels=
                CONV2_CHANNELS,

            kernel_size=
                3,

            padding=
                1,
        )


        self.relu = nn.ReLU()


        # ----------------------------------------------------
        # LayerNorm is FL-friendly compared with BatchNorm
        # because it has no running global batch statistics.
        #
        # Applied after converting back to:
        # (batch, sequence_length, channels)
        # ----------------------------------------------------

        self.layer_norm = nn.LayerNorm(
            CONV2_CHANNELS
        )


        # ----------------------------------------------------
        # BiLSTM models ordered dependencies inside the
        # 20-flow source-ordered window.
        # ----------------------------------------------------

        self.bilstm = nn.LSTM(

            input_size=
                CONV2_CHANNELS,

            hidden_size=
                LSTM_HIDDEN_SIZE,

            num_layers=
                LSTM_LAYERS,

            batch_first=
                True,

            bidirectional=
                BIDIRECTIONAL,
        )


        representation_size = (

            LSTM_HIDDEN_SIZE
            *
            (
                2
                if BIDIRECTIONAL
                else 1
            )
        )


        self.dropout = nn.Dropout(
            DROPOUT
        )


        self.fc1 = nn.Linear(

            representation_size,

            FC_HIDDEN_SIZE,
        )


        self.fc2 = nn.Linear(

            FC_HIDDEN_SIZE,

            num_classes,
        )


    def forward(
        self,
        x,
    ):

        # ----------------------------------------------------
        # x:
        # (B, 20, 70)
        # ----------------------------------------------------

        x = x.transpose(
            1,
            2,
        )

        # ----------------------------------------------------
        # (B, 70, 20)
        # ->
        # (B, 128, 20)
        # ----------------------------------------------------

        x = self.relu(
            self.conv1(
                x
            )
        )


        # ----------------------------------------------------
        # (B, 128, 20)
        # ->
        # (B, 64, 20)
        # ----------------------------------------------------

        x = self.relu(
            self.conv2(
                x
            )
        )


        # ----------------------------------------------------
        # Back to sequence-major feature representation:
        # (B, 20, 64)
        # ----------------------------------------------------

        x = x.transpose(
            1,
            2,
        )


        x = self.layer_norm(
            x
        )


        # ----------------------------------------------------
        # h_n shape:
        # (num_layers * num_directions, B, hidden_size)
        #
        # For 1-layer BiLSTM:
        # h_n[-2] = final forward state
        # h_n[-1] = final backward state
        #
        # Concatenating them gives a complete bidirectional
        # representation of the full 20-flow sequence.
        # ----------------------------------------------------

        lstm_output, (
            h_n,
            c_n,
        ) = self.bilstm(
            x
        )


        if BIDIRECTIONAL:

            representation = torch.cat(
                (
                    h_n[
                        -2
                    ],
                    h_n[
                        -1
                    ],
                ),
                dim=1,
            )

        else:

            representation = h_n[
                -1
            ]


        representation = self.dropout(
            representation
        )


        representation = self.relu(
            self.fc1(
                representation
            )
        )


        representation = self.dropout(
            representation
        )


        logits = self.fc2(
            representation
        )


        return logits


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )


    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred,
    )


    (
        precision,
        recall,
        f1,
        support,

    ) = precision_recall_fscore_support(

        y_true,
        y_pred,

        labels=
            np.arange(
                NUM_CLASSES
            ),

        zero_division=
            0,
    )


    macro_precision = float(
        np.mean(
            precision
        )
    )


    macro_recall = float(
        np.mean(
            recall
        )
    )


    macro_f1 = float(
        np.mean(
            f1
        )
    )


    total_support = float(
        np.sum(
            support
        )
    )


    weighted_f1 = float(

        np.sum(
            f1
            *
            support
        )
        /
        total_support
    )


    return {

        "accuracy":
            float(
                accuracy
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy
            ),

        "macro_precision":
            macro_precision,

        "macro_recall":
            macro_recall,

        "macro_f1":
            macro_f1,

        "weighted_f1":
            weighted_f1,

        "per_class_precision":
            precision,

        "per_class_recall":
            recall,

        "per_class_f1":
            f1,

        "per_class_support":
            support,
    }


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
):

    model.train()


    total_loss = 0.0

    total_samples = 0

    all_true = []

    all_pred = []


    for X_batch, y_batch in dataloader:

        X_batch = X_batch.to(
            device,
            non_blocking=True,
        )


        y_batch = y_batch.to(
            device,
            non_blocking=True,
        )


        optimizer.zero_grad(
            set_to_none=True
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
                GRADIENT_CLIP_NORM,
        )


        optimizer.step()


        batch_size = int(
            y_batch.shape[
                0
            ]
        )


        total_loss += (

            float(
                loss.item()
            )
            *
            batch_size
        )


        total_samples += batch_size


        predictions = torch.argmax(
            logits,
            dim=1,
        )


        all_true.append(

            y_batch
            .detach()
            .cpu()
            .numpy()
        )


        all_pred.append(

            predictions
            .detach()
            .cpu()
            .numpy()
        )


    y_true = np.concatenate(
        all_true
    )


    y_pred = np.concatenate(
        all_pred
    )


    metrics = calculate_metrics(
        y_true,
        y_pred,
    )


    average_loss = (

        total_loss
        /
        total_samples
    )


    return (
        float(
            average_loss
        ),
        metrics,
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    dataloader,
    criterion,
    device,
):

    model.eval()


    total_loss = 0.0

    total_samples = 0

    all_true = []

    all_pred = []


    for X_batch, y_batch in dataloader:

        X_batch = X_batch.to(
            device,
            non_blocking=True,
        )


        y_batch = y_batch.to(
            device,
            non_blocking=True,
        )


        logits = model(
            X_batch
        )


        loss = criterion(
            logits,
            y_batch,
        )


        batch_size = int(
            y_batch.shape[
                0
            ]
        )


        total_loss += (

            float(
                loss.item()
            )
            *
            batch_size
        )


        total_samples += batch_size


        predictions = torch.argmax(
            logits,
            dim=1,
        )


        all_true.append(

            y_batch
            .cpu()
            .numpy()
        )


        all_pred.append(

            predictions
            .cpu()
            .numpy()
        )


    y_true = np.concatenate(
        all_true
    )


    y_pred = np.concatenate(
        all_pred
    )


    metrics = calculate_metrics(
        y_true,
        y_pred,
    )


    average_loss = (

        total_loss
        /
        total_samples
    )


    return (
        float(
            average_loss
        ),
        metrics,
        y_true,
        y_pred,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(
        SEED
    )


    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    ARTIFACT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    device = resolve_device()


    separator()

    print(
        PHASE_NAME
    )

    separator()


    print(
        f"Device                    : "
        f"{device}"
    )


    print(
        f"Seed                      : "
        f"{SEED}"
    )


    print(
        f"Training NPZ              : "
        f"{GLOBAL_TRAIN_FILE}"
    )


    print(
        f"Validation NPZ            : "
        f"{VALIDATION_FILE}"
    )


    print(
        f"Sequence shape            : "
        f"({EXPECTED_SEQUENCE_LENGTH}, "
        f"{EXPECTED_FEATURES})"
    )


    print(
        f"Classes                   : "
        f"{NUM_CLASSES}"
    )


    print(
        f"Batch size                : "
        f"{BATCH_SIZE}"
    )


    print(
        f"Max epochs                : "
        f"{MAX_EPOCHS}"
    )


    print(
        f"Learning rate             : "
        f"{LEARNING_RATE}"
    )


    print(
        f"Weight decay              : "
        f"{WEIGHT_DECAY}"
    )


    print(
        f"Early stopping patience   : "
        f"{EARLY_STOPPING_PATIENCE}"
    )


    print(
        f"Primary selection metric  : "
        f"validation Macro F1"
    )


    print()

    print(
        "DATA / RESEARCH POLICY"
    )

    print(
        "-" * 112
    )


    print(
        "1. Uses Phase14E0 UNIQUE global training sequences."
    )


    print(
        "2. Uses Phase14D validation_sequences.npz."
    )


    print(
        "3. Class weights are calculated from TRAIN only."
    )


    print(
        "4. No oversampling."
    )


    print(
        "5. No synthetic data."
    )


    print(
        "6. No locked-test evaluation or model selection."
    )


    print(
        "7. Validation Macro F1 controls early stopping."
    )


    print(
        "8. Architecture uses LayerNorm, not BatchNorm, "
        "to remain suitable for later federated training."
    )


    print(
        "9. Scientific label: SOURCE-ORDERED sequences; "
        "not timestamp-confirmed temporal sequences."
    )


    # ========================================================
    # LOAD LABEL MAPPING
    # ========================================================

    (
        _,
        _,
        label_mapping,
        _,
        active_features,

    ) = phase4.load_preprocessing()


    if len(
        active_features
    ) != EXPECTED_FEATURES:

        raise ValueError(
            "\nExpected 70 active features."
        )


    inverse_mapping = get_inverse_mapping(
        label_mapping
    )


    if len(
        inverse_mapping
    ) != NUM_CLASSES:

        raise ValueError(
            f"\nExpected {NUM_CLASSES} classes, "
            f"found {len(inverse_mapping)}."
        )


    class_names = [

        inverse_mapping[
            class_id
        ]

        for class_id
        in range(
            NUM_CLASSES
        )
    ]


    # ========================================================
    # LOAD DATA
    # ========================================================

    data_start = time.perf_counter()


    (
        X_train,
        y_train,

    ) = load_sequence_npz(

        GLOBAL_TRAIN_FILE,

        "global training sequences",
    )


    (
        X_validation,
        y_validation,

    ) = load_sequence_npz(

        VALIDATION_FILE,

        "validation sequences",
    )


    data_elapsed = (

        time.perf_counter()
        -
        data_start
    )


    print()

    print(
        f"Training X shape          : "
        f"{X_train.shape}"
    )


    print(
        f"Training y shape          : "
        f"{y_train.shape}"
    )


    print(
        f"Validation X shape        : "
        f"{X_validation.shape}"
    )


    print(
        f"Validation y shape        : "
        f"{y_validation.shape}"
    )


    print(
        f"Data load time            : "
        f"{data_elapsed:.2f}s"
    )


    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    (
        class_counts,
        class_weights,

    ) = compute_class_weights(
        y_train
    )


    class_weight_rows = []


    print()

    print(
        "TRAIN CLASS WEIGHTS"
    )

    print(
        "-" * 112
    )


    for class_id in range(
        NUM_CLASSES
    ):

        row = {

            "class_id":
                class_id,

            "class_name":
                class_names[
                    class_id
                ],

            "train_sequences":
                int(
                    class_counts[
                        class_id
                    ]
                ),

            "weight":
                float(
                    class_weights[
                        class_id
                    ]
                ),
        }


        class_weight_rows.append(
            row
        )


        print(
            f"{class_id} | "
            f"{class_names[class_id]:18s} | "
            f"count={int(class_counts[class_id]):7d} | "
            f"weight={class_weights[class_id]:.6f}"
        )


    pd.DataFrame(
        class_weight_rows
    ).to_csv(
        CLASS_WEIGHTS_FILE,
        index=False,
    )


    # ========================================================
    # TORCH DATASETS
    # ========================================================

    train_dataset = TensorDataset(

        torch.from_numpy(
            X_train
        ),

        torch.from_numpy(
            y_train
        ),
    )


    validation_dataset = TensorDataset(

        torch.from_numpy(
            X_validation
        ),

        torch.from_numpy(
            y_validation
        ),
    )


    data_generator = torch.Generator()

    data_generator.manual_seed(
        SEED
    )


    train_loader = DataLoader(

        train_dataset,

        batch_size=
            BATCH_SIZE,

        shuffle=
            True,

        num_workers=
            NUM_WORKERS,

        pin_memory=
            (
                device.type
                ==
                "cuda"
            ),

        generator=
            data_generator,

        drop_last=
            False,
    )


    validation_loader = DataLoader(

        validation_dataset,

        batch_size=
            BATCH_SIZE,

        shuffle=
            False,

        num_workers=
            NUM_WORKERS,

        pin_memory=
            (
                device.type
                ==
                "cuda"
            ),

        drop_last=
            False,
    )


    # ========================================================
    # MODEL
    # ========================================================

    model = CNNBiLSTM(

        feature_count=
            EXPECTED_FEATURES,

        num_classes=
            NUM_CLASSES,
    ).to(
        device
    )


    parameter_count = sum(

        parameter.numel()

        for parameter
        in model.parameters()
    )


    trainable_parameter_count = sum(

        parameter.numel()

        for parameter
        in model.parameters()

        if parameter.requires_grad
    )


    print()

    print(
        "MODEL"
    )

    print(
        "-" * 112
    )


    print(
        model
    )


    print()

    print(
        f"Total parameters          : "
        f"{parameter_count:,}"
    )


    print(
        f"Trainable parameters      : "
        f"{trainable_parameter_count:,}"
    )


    # ========================================================
    # LOSS / OPTIMIZER
    # ========================================================

    criterion = nn.CrossEntropyLoss(

        weight=
            torch.tensor(
                class_weights,
                dtype=torch.float32,
                device=device,
            )
    )


    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=
            LEARNING_RATE,

        weight_decay=
            WEIGHT_DECAY,
    )


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    history_rows = []


    best_macro_f1 = -np.inf

    best_epoch = 0

    patience_counter = 0


    training_start = time.perf_counter()


    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        epoch_start = time.perf_counter()


        (
            train_loss,
            train_metrics,

        ) = train_one_epoch(

            model=
                model,

            dataloader=
                train_loader,

            criterion=
                criterion,

            optimizer=
                optimizer,

            device=
                device,
        )


        (
            validation_loss,
            validation_metrics,
            _,
            _,

        ) = evaluate(

            model=
                model,

            dataloader=
                validation_loader,

            criterion=
                criterion,

            device=
                device,
        )


        epoch_seconds = (

            time.perf_counter()
            -
            epoch_start
        )


        validation_macro_f1 = float(

            validation_metrics[
                "macro_f1"
            ]
        )


        improved = (

            validation_macro_f1

            >

            best_macro_f1
            +
            1e-8
        )


        if improved:

            best_macro_f1 = (
                validation_macro_f1
            )


            best_epoch = epoch


            patience_counter = 0


            checkpoint = {

                "phase":
                    "14E",

                "seed":
                    SEED,

                "epoch":
                    epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "validation_macro_f1":
                    best_macro_f1,

                "feature_count":
                    EXPECTED_FEATURES,

                "sequence_length":
                    EXPECTED_SEQUENCE_LENGTH,

                "num_classes":
                    NUM_CLASSES,

                "class_names":
                    class_names,

                "class_weights":
                    class_weights.tolist(),

                "architecture":
                    {

                        "conv1_channels":
                            CONV1_CHANNELS,

                        "conv2_channels":
                            CONV2_CHANNELS,

                        "lstm_hidden_size":
                            LSTM_HIDDEN_SIZE,

                        "lstm_layers":
                            LSTM_LAYERS,

                        "bidirectional":
                            BIDIRECTIONAL,

                        "fc_hidden_size":
                            FC_HIDDEN_SIZE,

                        "dropout":
                            DROPOUT,

                        "normalization":
                            "LayerNorm",
                    },
            }


            torch.save(

                checkpoint,

                MODEL_FILE,
            )


        else:

            patience_counter += 1


        history_rows.append(
            {

                "epoch":
                    epoch,

                "train_loss":
                    train_loss,

                "train_accuracy":
                    train_metrics[
                        "accuracy"
                    ],

                "train_balanced_accuracy":
                    train_metrics[
                        "balanced_accuracy"
                    ],

                "train_macro_f1":
                    train_metrics[
                        "macro_f1"
                    ],

                "validation_loss":
                    validation_loss,

                "validation_accuracy":
                    validation_metrics[
                        "accuracy"
                    ],

                "validation_balanced_accuracy":
                    validation_metrics[
                        "balanced_accuracy"
                    ],

                "validation_macro_precision":
                    validation_metrics[
                        "macro_precision"
                    ],

                "validation_macro_recall":
                    validation_metrics[
                        "macro_recall"
                    ],

                "validation_macro_f1":
                    validation_metrics[
                        "macro_f1"
                    ],

                "validation_weighted_f1":
                    validation_metrics[
                        "weighted_f1"
                    ],

                "is_best":
                    bool(
                        improved
                    ),

                "patience_counter":
                    patience_counter,

                "epoch_seconds":
                    epoch_seconds,
            }
        )


        print()

        print(
            f"EPOCH {epoch:02d}/{MAX_EPOCHS}"
        )


        print(
            "-" * 112
        )


        print(
            f"Train loss               : "
            f"{train_loss:.6f}"
        )


        print(
            f"Train Acc / BA / MacroF1 : "
            f"{train_metrics['accuracy']:.6f} / "
            f"{train_metrics['balanced_accuracy']:.6f} / "
            f"{train_metrics['macro_f1']:.6f}"
        )


        print(
            f"Val loss                 : "
            f"{validation_loss:.6f}"
        )


        print(
            f"Val Acc / BA / MacroF1   : "
            f"{validation_metrics['accuracy']:.6f} / "
            f"{validation_metrics['balanced_accuracy']:.6f} / "
            f"{validation_metrics['macro_f1']:.6f}"
        )


        print(
            f"Val Weighted F1          : "
            f"{validation_metrics['weighted_f1']:.6f}"
        )


        print(
            f"Best epoch / MacroF1     : "
            f"{best_epoch} / "
            f"{best_macro_f1:.6f}"
        )


        print(
            f"Patience                 : "
            f"{patience_counter}/"
            f"{EARLY_STOPPING_PATIENCE}"
        )


        print(
            f"Epoch time               : "
            f"{epoch_seconds:.2f}s"
        )


        pd.DataFrame(
            history_rows
        ).to_csv(
            TRAINING_HISTORY_FILE,
            index=False,
        )


        if (
            patience_counter

            >=

            EARLY_STOPPING_PATIENCE
        ):

            print()

            print(
                "EARLY STOPPING TRIGGERED"
            )


            break


    training_seconds = (

        time.perf_counter()
        -
        training_start
    )


    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    if not MODEL_FILE.exists():

        raise RuntimeError(
            "\nNo Phase14E checkpoint was saved."
        )


    checkpoint = torch.load(

        MODEL_FILE,

        map_location=
            device,
    )


    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]
    )


    # ========================================================
    # FINAL BEST-MODEL VALIDATION
    # ========================================================

    inference_start = time.perf_counter()


    (
        best_validation_loss,
        best_metrics,
        y_true,
        y_pred,

    ) = evaluate(

        model=
            model,

        dataloader=
            validation_loader,

        criterion=
            criterion,

        device=
            device,
    )


    inference_seconds = (

        time.perf_counter()
        -
        inference_start
    )


    # ========================================================
    # SAVE PER-CLASS METRICS
    # ========================================================

    per_class_rows = []


    for class_id in range(
        NUM_CLASSES
    ):

        per_class_rows.append(
            {

                "class_id":
                    class_id,

                "class_name":
                    class_names[
                        class_id
                    ],

                "precision":
                    float(
                        best_metrics[
                            "per_class_precision"
                        ][
                            class_id
                        ]
                    ),

                "recall":
                    float(
                        best_metrics[
                            "per_class_recall"
                        ][
                            class_id
                        ]
                    ),

                "f1":
                    float(
                        best_metrics[
                            "per_class_f1"
                        ][
                            class_id
                        ]
                    ),

                "support":
                    int(
                        best_metrics[
                            "per_class_support"
                        ][
                            class_id
                        ]
                    ),
            }
        )


    per_class_dataframe = pd.DataFrame(
        per_class_rows
    )


    per_class_dataframe.to_csv(

        PER_CLASS_METRICS_FILE,

        index=False,
    )


    # ========================================================
    # SAVE CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(

        y_true,
        y_pred,

        labels=
            np.arange(
                NUM_CLASSES
            ),
    )


    confusion_dataframe = pd.DataFrame(

        cm,

        index=[
            f"true_{name}"
            for name
            in class_names
        ],

        columns=[
            f"pred_{name}"
            for name
            in class_names
        ],
    )


    confusion_dataframe.to_csv(

        CONFUSION_MATRIX_FILE
    )


    # ========================================================
    # SAVE VALIDATION METRICS
    # ========================================================

    validation_payload = {

        "phase":
            "14E",

        "seed":
            SEED,

        "device":
            str(
                device
            ),

        "best_epoch":
            int(
                best_epoch
            ),

        "best_validation_loss":
            float(
                best_validation_loss
            ),

        "accuracy":
            best_metrics[
                "accuracy"
            ],

        "balanced_accuracy":
            best_metrics[
                "balanced_accuracy"
            ],

        "macro_precision":
            best_metrics[
                "macro_precision"
            ],

        "macro_recall":
            best_metrics[
                "macro_recall"
            ],

        "macro_f1":
            best_metrics[
                "macro_f1"
            ],

        "weighted_f1":
            best_metrics[
                "weighted_f1"
            ],

        "training_seconds":
            float(
                training_seconds
            ),

        "final_validation_inference_seconds":
            float(
                inference_seconds
            ),

        "locked_test_used":
            False,

        "scientific_label":
            (
                "source_ordered_sequences_"
                "not_timestamp_confirmed"
            ),
    }


    save_json(

        VALIDATION_METRICS_FILE,

        validation_payload,
    )


    # ========================================================
    # SAVE MODEL METADATA
    # ========================================================

    metadata = {

        "phase":
            "14E",

        "component":
            "centralized_cnn_bilstm_baseline",

        "model_file":
            str(
                MODEL_FILE
            ),

        "training_sequence_file":
            str(
                GLOBAL_TRAIN_FILE
            ),

        "validation_sequence_file":
            str(
                VALIDATION_FILE
            ),

        "seed":
            SEED,

        "device":
            str(
                device
            ),

        "sequence_length":
            EXPECTED_SEQUENCE_LENGTH,

        "feature_count":
            EXPECTED_FEATURES,

        "feature_names":
            list(
                active_features
            ),

        "num_classes":
            NUM_CLASSES,

        "class_names":
            class_names,

        "parameter_count":
            int(
                parameter_count
            ),

        "trainable_parameter_count":
            int(
                trainable_parameter_count
            ),

        "architecture":
            {

                "conv1_channels":
                    CONV1_CHANNELS,

                "conv2_channels":
                    CONV2_CHANNELS,

                "convolution_kernel":
                    3,

                "normalization":
                    "LayerNorm",

                "lstm_hidden_size":
                    LSTM_HIDDEN_SIZE,

                "lstm_layers":
                    LSTM_LAYERS,

                "bidirectional":
                    BIDIRECTIONAL,

                "fc_hidden_size":
                    FC_HIDDEN_SIZE,

                "dropout":
                    DROPOUT,
            },

        "training":
            {

                "batch_size":
                    BATCH_SIZE,

                "max_epochs":
                    MAX_EPOCHS,

                "learning_rate":
                    LEARNING_RATE,

                "weight_decay":
                    WEIGHT_DECAY,

                "gradient_clip_norm":
                    GRADIENT_CLIP_NORM,

                "early_stopping_patience":
                    EARLY_STOPPING_PATIENCE,

                "selection_metric":
                    "validation_macro_f1",

                "best_epoch":
                    int(
                        best_epoch
                    ),

                "class_weight_mode":
                    CLASS_WEIGHT_MODE,

                "class_weight_cap":
                    CLASS_WEIGHT_CAP,

                "class_weights":
                    class_weights.tolist(),

                "oversampling":
                    False,

                "synthetic_data":
                    False,
            },

        "validation_metrics":
            validation_payload,

        "locked_test_path":
            str(
                LOCKED_TEST_FILE
            ),

        "locked_test_used":
            False,

        "scientific_label":
            (
                "SOURCE-ORDERED flow sequences; "
                "NOT timestamp-confirmed temporal sequences."
            ),
    }


    save_json(

        METADATA_FILE,

        metadata,
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    separator()

    print(
        "PHASE 14E COMPLETE"
    )

    separator()


    print(
        f"Best epoch                : "
        f"{best_epoch}"
    )


    print(
        f"Validation loss           : "
        f"{best_validation_loss:.6f}"
    )


    print(
        f"Accuracy                  : "
        f"{best_metrics['accuracy']:.6f}"
    )


    print(
        f"Balanced Accuracy         : "
        f"{best_metrics['balanced_accuracy']:.6f}"
    )


    print(
        f"Macro Precision           : "
        f"{best_metrics['macro_precision']:.6f}"
    )


    print(
        f"Macro Recall              : "
        f"{best_metrics['macro_recall']:.6f}"
    )


    print(
        f"Macro F1                  : "
        f"{best_metrics['macro_f1']:.6f}"
    )


    print(
        f"Weighted F1               : "
        f"{best_metrics['weighted_f1']:.6f}"
    )


    print()

    print(
        "PER-CLASS VALIDATION METRICS"
    )

    print(
        "-" * 112
    )


    print(

        per_class_dataframe[
            [
                "class_name",
                "precision",
                "recall",
                "f1",
                "support",
            ]
        ]

        .to_string(
            index=False
        )
    )


    print()

    print(
        f"Total training time       : "
        f"{training_seconds:.2f}s"
    )


    print(
        f"Final inference time      : "
        f"{inference_seconds:.2f}s"
    )


    print(
        f"Model                     : "
        f"{MODEL_FILE}"
    )


    print(
        f"Training history          : "
        f"{TRAINING_HISTORY_FILE}"
    )


    print(
        f"Validation metrics        : "
        f"{VALIDATION_METRICS_FILE}"
    )


    print(
        f"Per-class metrics         : "
        f"{PER_CLASS_METRICS_FILE}"
    )


    print(
        f"Confusion matrix          : "
        f"{CONFUSION_MATRIX_FILE}"
    )


    print(
        f"Metadata                  : "
        f"{METADATA_FILE}"
    )


    print()

    print(
        "LOCKED TEST USED : NO"
    )


    print()

    print(
        "SCIENTIFIC LABEL:"
    )


    print(
        "SOURCE-ORDERED flow sequences; "
        "NOT timestamp-confirmed temporal sequences."
    )


    # ========================================================
    # CLEANUP
    # ========================================================

    del X_train

    del y_train

    del X_validation

    del y_validation

    del train_dataset

    del validation_dataset

    del train_loader

    del validation_loader

    gc.collect()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
