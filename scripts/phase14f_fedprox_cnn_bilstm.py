from __future__ import annotations

import gc
import json
import logging
import random
import sys
import time
from collections import OrderedDict
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
# PROJECT ROOT / EXISTING INFRASTRUCTURE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

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
    "PHASE 14F — FEDPROX CNN-BILSTM FEDERATED TRAINING"
)


# ============================================================
# EXACT USER-SELECTED FEDERATED CONFIGURATION
# ============================================================

SEED = 42

NUM_CLIENTS = 5
FEDERATED_ROUNDS = 10
LOCAL_EPOCHS = 1

BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

FEDPROX_MU = 0.01
GRADIENT_CLIP_NORM = 5.0

NUM_WORKERS = 0

AGGREGATION_METHOD = "FedProx"
AGGREGATION_WEIGHTING = "client_sequence_count"

# No early stop: run all requested FL rounds.
MODEL_SELECTION_METRIC = "validation_macro_f1"


# ============================================================
# CNN-BILSTM ARCHITECTURE
# Same Phase14E architecture = 126,982 parameters
# ============================================================

EXPECTED_SEQUENCE_LENGTH = 20
EXPECTED_FEATURES = 70
NUM_CLASSES = 6

CONV1_CHANNELS = 128
CONV2_CHANNELS = 64

LSTM_HIDDEN_SIZE = 64
LSTM_LAYERS = 1
BIDIRECTIONAL = True

FC_HIDDEN_SIZE = 64
DROPOUT = 0.25


# ============================================================
# CLASS-WEIGHT POLICY
#
# We use ONE fixed global class-weight vector derived only
# from the unique Phase14E0 TRAIN sequence labels.
# This avoids unstable client-specific class weights under
# severe Non-IID clients that contain only BENIGN + one attack.
# ============================================================

CLASS_WEIGHT_MODE = "sqrt_max_over_count"
CLASS_WEIGHT_CAP = 20.0


# ============================================================
# PATHS
# ============================================================

SEQUENCE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "sequences"
)

CLIENT_SEQUENCE_ROOT = (
    SEQUENCE_ROOT
    / "non_iid"
)

GLOBAL_TRAIN_SEQUENCE_FILE = (
    SEQUENCE_ROOT
    / "global_train_sequences.npz"
)

VALIDATION_FILE = (
    SEQUENCE_ROOT
    / "validation_sequences.npz"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "phase14"
)

ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "detection"
)

BEST_MODEL_FILE = (
    ARTIFACT_ROOT
    / "cnn_bilstm_fedprox.pt"
)

FINAL_ROUND_MODEL_FILE = (
    ARTIFACT_ROOT
    / "cnn_bilstm_fedprox_final_round.pt"
)

METADATA_FILE = (
    ARTIFACT_ROOT
    / "cnn_bilstm_fedprox_metadata.json"
)

ROUND_HISTORY_FILE = (
    RESULT_ROOT
    / "phase14f_round_history.csv"
)

CLIENT_ROUND_METRICS_FILE = (
    RESULT_ROOT
    / "phase14f_client_round_metrics.csv"
)

BEST_METRICS_FILE = (
    RESULT_ROOT
    / "phase14f_best_validation_metrics.json"
)

FINAL_ROUND_METRICS_FILE = (
    RESULT_ROOT
    / "phase14f_final_round_validation_metrics.json"
)

BEST_PER_CLASS_FILE = (
    RESULT_ROOT
    / "phase14f_best_per_class_metrics.csv"
)

FINAL_PER_CLASS_FILE = (
    RESULT_ROOT
    / "phase14f_final_round_per_class_metrics.csv"
)

BEST_CONFUSION_MATRIX_FILE = (
    RESULT_ROOT
    / "phase14f_best_confusion_matrix.csv"
)

FINAL_CONFUSION_MATRIX_FILE = (
    RESULT_ROOT
    / "phase14f_final_round_confusion_matrix.csv"
)

CLASS_WEIGHTS_FILE = (
    RESULT_ROOT
    / "phase14f_class_weights.csv"
)

LOCKED_TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_locked.csv"
)


# ============================================================
# CLIENT ATTACK MAP
# ============================================================

CLIENT_ATTACKS = {
    1: "DoS Hulk",
    2: "DDoS",
    3: "PortScan",
    4: "DoS GoldenEye",
    5: "FTP-Patator",
}


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(
    "phase14f_fedprox_cnn_bilstm"
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def separator() -> None:
    print()
    print("=" * 118)


def json_safe(value):
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, Path):
        return str(value)

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
            json_safe(payload),
            file,
            indent=4,
        )


def resolve_device() -> torch.device:
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def get_inverse_mapping(label_mapping):
    return {
        int(class_id): str(label)
        for label, class_id
        in label_mapping.items()
    }


def clone_state_dict_to_cpu(model: nn.Module):
    return OrderedDict(
        (
            key,
            value.detach().cpu().clone(),
        )
        for key, value
        in model.state_dict().items()
    )


# ============================================================
# LOAD SEQUENCE NPZ
# ============================================================

def load_sequence_npz(
    path: Path,
    dataset_name: str,
):
    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing {dataset_name}:\n{path}"
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
                f"\n{dataset_name} must contain X and y."
            )

        X = data["X"].astype(
            np.float32,
            copy=False,
        )

        y = data["y"].astype(
            np.int64,
            copy=False,
        )

    if X.ndim != 3:
        raise ValueError(
            f"\n{dataset_name}: expected 3D X, "
            f"found {X.shape}."
        )

    if tuple(X.shape[1:]) != (
        EXPECTED_SEQUENCE_LENGTH,
        EXPECTED_FEATURES,
    ):
        raise ValueError(
            f"\n{dataset_name}: expected shape "
            f"(*, {EXPECTED_SEQUENCE_LENGTH}, "
            f"{EXPECTED_FEATURES}), found {X.shape}."
        )

    if y.ndim != 1:
        raise ValueError(
            f"\n{dataset_name}: expected 1D y, "
            f"found {y.shape}."
        )

    if len(X) != len(y):
        raise ValueError(
            f"\n{dataset_name}: X/y count mismatch."
        )

    unique_classes = np.unique(y)

    if np.any(unique_classes < 0):
        raise ValueError(
            f"\n{dataset_name}: negative class id found."
        )

    if np.any(unique_classes >= NUM_CLASSES):
        raise ValueError(
            f"\n{dataset_name}: class id outside "
            f"0..{NUM_CLASSES - 1}."
        )

    return X, y


def load_y_only(
    path: Path,
    dataset_name: str,
):
    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing {dataset_name}:\n{path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        if "y" not in data:
            raise KeyError(
                f"\n{dataset_name} must contain y."
            )

        y = data["y"].astype(
            np.int64,
            copy=False,
        )

    return y


# ============================================================
# FIXED GLOBAL CLASS WEIGHTS FROM UNIQUE TRAIN SEQUENCES
# ============================================================

def compute_global_class_weights():
    y_global = load_y_only(
        GLOBAL_TRAIN_SEQUENCE_FILE,
        "Phase14E0 unique global training sequences",
    )

    counts = np.bincount(
        y_global,
        minlength=NUM_CLASSES,
    ).astype(np.float64)

    del y_global
    gc.collect()

    if np.any(counts <= 0):
        missing = np.flatnonzero(
            counts <= 0
        )

        raise RuntimeError(
            "\nGlobal Phase14E0 sequence training data "
            "is missing class ids:\n"
            f"{missing.tolist()}"
        )

    max_count = float(
        np.max(counts)
    )

    weights = np.sqrt(
        max_count / counts
    )

    weights = np.minimum(
        weights,
        CLASS_WEIGHT_CAP,
    )

    weights = (
        weights
        /
        np.mean(weights)
    )

    return (
        counts,
        weights.astype(np.float32),
    )


# ============================================================
# MODEL
# ============================================================

class CNNBiLSTM(nn.Module):

    def __init__(
        self,
        feature_count: int,
        num_classes: int,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=feature_count,
            out_channels=CONV1_CHANNELS,
            kernel_size=3,
            padding=1,
        )

        self.conv2 = nn.Conv1d(
            in_channels=CONV1_CHANNELS,
            out_channels=CONV2_CHANNELS,
            kernel_size=3,
            padding=1,
        )

        self.relu = nn.ReLU()

        # Same FL-friendly normalization used in Phase14E.
        self.layer_norm = nn.LayerNorm(
            CONV2_CHANNELS
        )

        self.bilstm = nn.LSTM(
            input_size=CONV2_CHANNELS,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            bidirectional=BIDIRECTIONAL,
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

    def forward(self, x):
        # (B, 20, 70) -> (B, 70, 20)
        x = x.transpose(1, 2)

        # (B, 70, 20) -> (B, 128, 20)
        x = self.relu(
            self.conv1(x)
        )

        # (B, 128, 20) -> (B, 64, 20)
        x = self.relu(
            self.conv2(x)
        )

        # (B, 64, 20) -> (B, 20, 64)
        x = x.transpose(1, 2)

        x = self.layer_norm(x)

        lstm_output, (
            h_n,
            c_n,
        ) = self.bilstm(x)

        if BIDIRECTIONAL:
            representation = torch.cat(
                (
                    h_n[-2],
                    h_n[-1],
                ),
                dim=1,
            )
        else:
            representation = h_n[-1]

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

        return self.fc2(
            representation
        )


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
        labels=np.arange(NUM_CLASSES),
        zero_division=0,
    )

    macro_precision = float(
        np.mean(precision)
    )

    macro_recall = float(
        np.mean(recall)
    )

    macro_f1 = float(
        np.mean(f1)
    )

    total_support = float(
        np.sum(support)
    )

    weighted_f1 = float(
        np.sum(
            f1 * support
        )
        /
        total_support
    )

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy":
            float(balanced_accuracy),
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
# FEDPROX LOCAL TRAINING
# ============================================================

def train_local_fedprox(
    global_model: nn.Module,
    X_client: np.ndarray,
    y_client: np.ndarray,
    class_weights: np.ndarray,
    device: torch.device,
    round_number: int,
    client_id: int,
):
    # Every client begins this round from the exact same
    # current global parameters.
    local_model = CNNBiLSTM(
        feature_count=EXPECTED_FEATURES,
        num_classes=NUM_CLASSES,
    ).to(device)

    local_model.load_state_dict(
        global_model.state_dict()
    )

    # Frozen reference point for the FedProx proximal term.
    global_reference_parameters = [
        parameter.detach().clone()
        for parameter
        in local_model.parameters()
    ]

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            class_weights,
            dtype=torch.float32,
            device=device,
        )
    )

    optimizer = torch.optim.AdamW(
        local_model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    dataset = TensorDataset(
        torch.from_numpy(X_client),
        torch.from_numpy(y_client),
    )

    generator = torch.Generator()
    generator.manual_seed(
        SEED
        +
        round_number * 10_000
        +
        client_id * 100
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        generator=generator,
        drop_last=False,
    )

    local_model.train()

    total_ce_loss = 0.0
    total_prox_loss = 0.0
    total_objective_loss = 0.0
    total_samples = 0

    all_true = []
    all_pred = []

    local_start = time.perf_counter()

    for local_epoch in range(
        1,
        LOCAL_EPOCHS + 1,
    ):
        for X_batch, y_batch in loader:
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

            logits = local_model(
                X_batch
            )

            ce_loss = criterion(
                logits,
                y_batch,
            )

            proximal_sum = torch.zeros(
                (),
                dtype=ce_loss.dtype,
                device=device,
            )

            for local_parameter, global_parameter in zip(
                local_model.parameters(),
                global_reference_parameters,
            ):
                proximal_sum = (
                    proximal_sum
                    +
                    torch.sum(
                        (
                            local_parameter
                            -
                            global_parameter
                        )
                        ** 2
                    )
                )

            prox_loss = (
                0.5
                *
                FEDPROX_MU
                *
                proximal_sum
            )

            total_loss = (
                ce_loss
                +
                prox_loss
            )

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                local_model.parameters(),
                max_norm=GRADIENT_CLIP_NORM,
            )

            optimizer.step()

            batch_size = int(
                y_batch.shape[0]
            )

            total_ce_loss += (
                float(
                    ce_loss.item()
                )
                *
                batch_size
            )

            total_prox_loss += (
                float(
                    prox_loss.item()
                )
                *
                batch_size
            )

            total_objective_loss += (
                float(
                    total_loss.item()
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

    local_seconds = (
        time.perf_counter()
        -
        local_start
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

    local_state = clone_state_dict_to_cpu(
        local_model
    )

    result = {
        "ce_loss":
            float(
                total_ce_loss
                /
                total_samples
            ),
        "prox_loss":
            float(
                total_prox_loss
                /
                total_samples
            ),
        "objective_loss":
            float(
                total_objective_loss
                /
                total_samples
            ),
        "accuracy":
            metrics[
                "accuracy"
            ],
        "balanced_accuracy":
            metrics[
                "balanced_accuracy"
            ],
        "macro_f1":
            metrics[
                "macro_f1"
            ],
        "local_seconds":
            float(
                local_seconds
            ),
    }

    del loader
    del dataset
    del local_model
    del global_reference_parameters
    del all_true
    del all_pred
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return (
        local_state,
        result,
    )


# ============================================================
# SAMPLE-WEIGHTED FEDERATED AVERAGING
# ============================================================

def weighted_federated_average(
    client_states,
    client_sequence_counts,
):
    if len(client_states) == 0:
        raise ValueError(
            "No client states supplied."
        )

    if len(client_states) != len(
        client_sequence_counts
    ):
        raise ValueError(
            "Client states / weights length mismatch."
        )

    weights = np.asarray(
        client_sequence_counts,
        dtype=np.float64,
    )

    if np.any(weights <= 0):
        raise ValueError(
            "All aggregation weights must be > 0."
        )

    normalized_weights = (
        weights
        /
        np.sum(weights)
    )

    averaged_state = OrderedDict()

    first_state = client_states[0]

    for key in first_state.keys():
        reference_tensor = first_state[key]

        if not torch.is_floating_point(
            reference_tensor
        ):
            averaged_state[key] = (
                reference_tensor.clone()
            )
            continue

        averaged_tensor = torch.zeros_like(
            reference_tensor,
            dtype=reference_tensor.dtype,
        )

        for client_state, client_weight in zip(
            client_states,
            normalized_weights,
        ):
            averaged_tensor.add_(
                client_state[key],
                alpha=float(
                    client_weight
                ),
            )

        averaged_state[key] = averaged_tensor

    return averaged_state


# ============================================================
# GLOBAL VALIDATION
# ============================================================

@torch.no_grad()
def evaluate_global(
    model,
    validation_loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_true = []
    all_pred = []

    for X_batch, y_batch in validation_loader:
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
            y_batch.shape[0]
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
# RESULT HELPERS
# ============================================================

def per_class_dataframe(
    metrics,
    class_names,
):
    rows = []

    for class_id in range(
        NUM_CLASSES
    ):
        rows.append(
            {
                "class_id":
                    class_id,
                "class_name":
                    class_names[
                        class_id
                    ],
                "precision":
                    float(
                        metrics[
                            "per_class_precision"
                        ][
                            class_id
                        ]
                    ),
                "recall":
                    float(
                        metrics[
                            "per_class_recall"
                        ][
                            class_id
                        ]
                    ),
                "f1":
                    float(
                        metrics[
                            "per_class_f1"
                        ][
                            class_id
                        ]
                    ),
                "support":
                    int(
                        metrics[
                            "per_class_support"
                        ][
                            class_id
                        ]
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def save_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    path,
):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(
            NUM_CLASSES
        ),
    )

    dataframe = pd.DataFrame(
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

    dataframe.to_csv(
        path
    )


def metrics_payload(
    round_number,
    loss,
    metrics,
    training_seconds,
    inference_seconds,
):
    return {
        "phase":
            "14F",
        "aggregation":
            AGGREGATION_METHOD,
        "round":
            int(
                round_number
            ),
        "validation_loss":
            float(
                loss
            ),
        "accuracy":
            metrics[
                "accuracy"
            ],
        "balanced_accuracy":
            metrics[
                "balanced_accuracy"
            ],
        "macro_precision":
            metrics[
                "macro_precision"
            ],
        "macro_recall":
            metrics[
                "macro_recall"
            ],
        "macro_f1":
            metrics[
                "macro_f1"
            ],
        "weighted_f1":
            metrics[
                "weighted_f1"
            ],
        "training_seconds":
            float(
                training_seconds
            ),
        "validation_inference_seconds":
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
    print(PHASE_NAME)
    separator()

    print(
        f"Scenario                  : non_iid"
    )

    print(
        f"Device                    : {device}"
    )

    print(
        f"Seed                      : {SEED}"
    )

    print(
        f"Aggregation               : {AGGREGATION_METHOD}"
    )

    print(
        f"Clients                   : {NUM_CLIENTS}"
    )

    print(
        f"Federated rounds          : {FEDERATED_ROUNDS}"
    )

    print(
        f"Local epochs / round      : {LOCAL_EPOCHS}"
    )

    print(
        f"Batch size                : {BATCH_SIZE}"
    )

    print(
        f"Learning rate             : {LEARNING_RATE}"
    )

    print(
        f"Weight decay              : {WEIGHT_DECAY}"
    )

    print(
        f"FedProx mu                : {FEDPROX_MU}"
    )

    print(
        f"Gradient clipping         : {GRADIENT_CLIP_NORM}"
    )

    print(
        f"Aggregation weight        : "
        f"{AGGREGATION_WEIGHTING}"
    )

    print(
        f"Initialization            : "
        f"fresh common random initialization"
    )

    print(
        f"Sequence shape            : "
        f"({EXPECTED_SEQUENCE_LENGTH}, "
        f"{EXPECTED_FEATURES})"
    )

    print(
        f"Validation                : {VALIDATION_FILE}"
    )

    print(
        "Locked test               : NO"
    )

    print()
    print("RESEARCH POLICY")
    print("-" * 118)

    print(
        "1. Round 1 starts from a fresh common random global model."
    )

    print(
        "2. The centralized Phase14E checkpoint is NOT loaded."
    )

    print(
        "3. Every client starts each round from the same current global model."
    )

    print(
        "4. Local objective = weighted CE + FedProx proximal penalty."
    )

    print(
        "5. One fixed global TRAIN-only class-weight vector is used on all clients."
    )

    print(
        "6. Aggregation is weighted by actual client sequence count."
    )

    print(
        "7. All 10 requested rounds run; no round-level early stopping."
    )

    print(
        "8. Validation is used for monitoring and best-round checkpoint selection."
    )

    print(
        "9. No oversampling and no synthetic data."
    )

    print(
        "10. Locked test is never read."
    )

    print(
        "11. SOURCE-ORDERED sequences; not timestamp-confirmed temporal sequences."
    )

    # ========================================================
    # LABEL MAPPING / FEATURE CHECK
    # ========================================================

    (
        _scaler,
        _feature_columns,
        label_mapping,
        _active_indices,
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

    label_to_id = {
        str(label): int(
            class_id
        )
        for label, class_id
        in label_mapping.items()
    }

    # ========================================================
    # FIXED GLOBAL CLASS WEIGHTS
    # ========================================================

    (
        global_class_counts,
        class_weights,
    ) = compute_global_class_weights()

    class_weight_rows = []

    print()
    print("GLOBAL TRAIN-ONLY CLASS WEIGHTS")
    print("-" * 118)

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
            "unique_global_train_sequences":
                int(
                    global_class_counts[
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
            f"count={int(global_class_counts[class_id]):7d} | "
            f"weight={class_weights[class_id]:.6f}"
        )

    pd.DataFrame(
        class_weight_rows
    ).to_csv(
        CLASS_WEIGHTS_FILE,
        index=False,
    )

    # ========================================================
    # CLIENT FILE VALIDATION + COUNTS
    # ========================================================

    client_files = {
        client_id:
            (
                CLIENT_SEQUENCE_ROOT
                /
                f"client_{client_id}_sequences.npz"
            )
        for client_id
        in range(
            1,
            NUM_CLIENTS + 1,
        )
    }

    client_sequence_counts = {}

    print()
    print("CLIENT SEQUENCE CHECK")
    print("-" * 118)

    for client_id, path in client_files.items():
        y_client = load_y_only(
            path,
            f"client_{client_id}",
        )

        count = int(
            len(
                y_client
            )
        )

        client_sequence_counts[
            client_id
        ] = count

        unique, counts = np.unique(
            y_client,
            return_counts=True,
        )

        distribution = {
            class_names[
                int(class_id)
            ]:
                int(class_count)
            for class_id, class_count
            in zip(
                unique,
                counts,
            )
        }

        expected_attack = CLIENT_ATTACKS[
            client_id
        ]

        expected_attack_id = label_to_id[
            expected_attack
        ]

        allowed_ids = {
            label_to_id[
                "BENIGN"
            ],
            expected_attack_id,
        }

        if not set(
            int(item)
            for item
            in unique
        ).issubset(
            allowed_ids
        ):
            raise RuntimeError(
                f"\nclient_{client_id} contains unexpected classes."
            )

        if expected_attack_id not in set(
            int(item)
            for item
            in unique
        ):
            raise RuntimeError(
                f"\nclient_{client_id} is missing "
                f"{expected_attack}."
            )

        print(
            f"client_{client_id} | "
            f"sequences={count:,} | "
            f"attack={expected_attack:17s} | "
            f"distribution={distribution}"
        )

        del y_client
        gc.collect()

    # ========================================================
    # VALIDATION DATA
    # ========================================================

    (
        X_validation,
        y_validation,
    ) = load_sequence_npz(
        VALIDATION_FILE,
        "validation sequences",
    )

    validation_classes = set(
        int(item)
        for item
        in np.unique(
            y_validation
        )
    )

    if validation_classes != set(
        range(
            NUM_CLASSES
        )
    ):
        raise RuntimeError(
            "\nValidation does not contain all 6 classes."
        )

    validation_dataset = TensorDataset(
        torch.from_numpy(
            X_validation
        ),
        torch.from_numpy(
            y_validation
        ),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type
            ==
            "cuda"
        ),
        drop_last=False,
    )

    validation_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            class_weights,
            dtype=torch.float32,
            device=device,
        )
    )

    # ========================================================
    # FRESH COMMON GLOBAL INITIALIZATION
    # ========================================================

    global_model = CNNBiLSTM(
        feature_count=EXPECTED_FEATURES,
        num_classes=NUM_CLASSES,
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter
        in global_model.parameters()
    )

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter
        in global_model.parameters()
        if parameter.requires_grad
    )

    if parameter_count != 126_982:
        raise RuntimeError(
            "\nArchitecture drift detected.\n"
            f"Expected 126,982 parameters; "
            f"found {parameter_count:,}."
        )

    print()
    print("GLOBAL CNN-BILSTM")
    print("-" * 118)

    print(global_model)

    print()
    print(
        f"Total parameters          : "
        f"{parameter_count:,}"
    )

    print(
        f"Trainable parameters      : "
        f"{trainable_parameter_count:,}"
    )

    print(
        "Centralized checkpoint loaded : NO"
    )

    # ========================================================
    # OPTIONAL ROUND-0 BASELINE
    # ========================================================

    (
        round0_loss,
        round0_metrics,
        _round0_true,
        _round0_pred,
    ) = evaluate_global(
        model=global_model,
        validation_loader=validation_loader,
        criterion=validation_criterion,
        device=device,
    )

    print()
    print(
        "ROUND 0 — FRESH INITIALIZATION VALIDATION"
    )
    print("-" * 118)

    print(
        f"Acc / BA / MacroF1       : "
        f"{round0_metrics['accuracy']:.6f} / "
        f"{round0_metrics['balanced_accuracy']:.6f} / "
        f"{round0_metrics['macro_f1']:.6f}"
    )

    # ========================================================
    # FEDERATED TRAINING
    # ========================================================

    round_rows = []

    client_round_rows = []

    best_round = 0
    best_macro_f1 = -np.inf
    best_state = None
    best_validation_loss = None
    best_metrics = None
    best_y_true = None
    best_y_pred = None

    total_training_start = time.perf_counter()

    last_round_loss = None
    last_round_metrics = None
    last_round_y_true = None
    last_round_y_pred = None
    last_round_inference_seconds = None

    for round_number in range(
        1,
        FEDERATED_ROUNDS + 1,
    ):
        separator()

        print(
            f"FEDERATED ROUND "
            f"{round_number:02d}/"
            f"{FEDERATED_ROUNDS}"
        )

        print(
            "-" * 118
        )

        round_start = time.perf_counter()

        client_states = []
        aggregation_counts = []

        # ----------------------------------------------------
        # CLIENT LOCAL TRAINING
        # ----------------------------------------------------

        for client_id in range(
            1,
            NUM_CLIENTS + 1,
        ):
            client_path = client_files[
                client_id
            ]

            (
                X_client,
                y_client,
            ) = load_sequence_npz(
                client_path,
                f"client_{client_id} sequences",
            )

            actual_count = int(
                len(
                    y_client
                )
            )

            if actual_count != client_sequence_counts[
                client_id
            ]:
                raise RuntimeError(
                    f"\nclient_{client_id} count changed "
                    "between audit and training."
                )

            (
                local_state,
                local_result,
            ) = train_local_fedprox(
                global_model=global_model,
                X_client=X_client,
                y_client=y_client,
                class_weights=class_weights,
                device=device,
                round_number=round_number,
                client_id=client_id,
            )

            client_states.append(
                local_state
            )

            aggregation_counts.append(
                actual_count
            )

            client_round_rows.append(
                {
                    "round":
                        round_number,
                    "client_id":
                        client_id,
                    "attack":
                        CLIENT_ATTACKS[
                            client_id
                        ],
                    "sequence_count":
                        actual_count,
                    "aggregation_weight":
                        float(
                            actual_count
                            /
                            sum(
                                client_sequence_counts.values()
                            )
                        ),
                    "ce_loss":
                        local_result[
                            "ce_loss"
                        ],
                    "prox_loss":
                        local_result[
                            "prox_loss"
                        ],
                    "objective_loss":
                        local_result[
                            "objective_loss"
                        ],
                    "local_accuracy":
                        local_result[
                            "accuracy"
                        ],
                    "local_balanced_accuracy":
                        local_result[
                            "balanced_accuracy"
                        ],
                    "local_macro_f1":
                        local_result[
                            "macro_f1"
                        ],
                    "local_seconds":
                        local_result[
                            "local_seconds"
                        ],
                }
            )

            print(
                f"client_{client_id} | "
                f"{CLIENT_ATTACKS[client_id]:17s} | "
                f"n={actual_count:6d} | "
                f"CE={local_result['ce_loss']:.6f} | "
                f"prox={local_result['prox_loss']:.6f} | "
                f"obj={local_result['objective_loss']:.6f} | "
                f"BA={local_result['balanced_accuracy']:.6f} | "
                f"F1={local_result['macro_f1']:.6f} | "
                f"{local_result['local_seconds']:.2f}s"
            )

            del X_client
            del y_client
            del local_state
            gc.collect()

            if device.type == "cuda":
                torch.cuda.empty_cache()

        # ----------------------------------------------------
        # SAMPLE-WEIGHTED AGGREGATION
        # ----------------------------------------------------

        aggregated_state = weighted_federated_average(
            client_states=client_states,
            client_sequence_counts=
                aggregation_counts,
        )

        global_model.load_state_dict(
            aggregated_state
        )

        del client_states
        del aggregated_state
        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

        # ----------------------------------------------------
        # GLOBAL VALIDATION
        # ----------------------------------------------------

        inference_start = time.perf_counter()

        (
            validation_loss,
            validation_metrics,
            y_true,
            y_pred,
        ) = evaluate_global(
            model=global_model,
            validation_loader=validation_loader,
            criterion=validation_criterion,
            device=device,
        )

        inference_seconds = (
            time.perf_counter()
            -
            inference_start
        )

        round_seconds = (
            time.perf_counter()
            -
            round_start
        )

        recalls = validation_metrics[
            "per_class_recall"
        ]

        row = {
            "round":
                round_number,
            "validation_loss":
                validation_loss,
            "accuracy":
                validation_metrics[
                    "accuracy"
                ],
            "balanced_accuracy":
                validation_metrics[
                    "balanced_accuracy"
                ],
            "macro_precision":
                validation_metrics[
                    "macro_precision"
                ],
            "macro_recall":
                validation_metrics[
                    "macro_recall"
                ],
            "macro_f1":
                validation_metrics[
                    "macro_f1"
                ],
            "weighted_f1":
                validation_metrics[
                    "weighted_f1"
                ],
            "recall_BENIGN":
                float(
                    recalls[0]
                ),
            "recall_DDoS":
                float(
                    recalls[1]
                ),
            "recall_DoS_GoldenEye":
                float(
                    recalls[2]
                ),
            "recall_DoS_Hulk":
                float(
                    recalls[3]
                ),
            "recall_FTP_Patator":
                float(
                    recalls[4]
                ),
            "recall_PortScan":
                float(
                    recalls[5]
                ),
            "validation_inference_seconds":
                inference_seconds,
            "round_seconds":
                round_seconds,
        }

        round_rows.append(
            row
        )

        current_macro_f1 = float(
            validation_metrics[
                "macro_f1"
            ]
        )

        improved = (
            current_macro_f1
            >
            best_macro_f1
            +
            1e-8
        )

        if improved:
            best_round = round_number
            best_macro_f1 = (
                current_macro_f1
            )

            best_validation_loss = (
                validation_loss
            )

            best_metrics = {
                key:
                    (
                        value.copy()
                        if isinstance(
                            value,
                            np.ndarray,
                        )
                        else value
                    )
                for key, value
                in validation_metrics.items()
            }

            best_y_true = (
                y_true.copy()
            )

            best_y_pred = (
                y_pred.copy()
            )

            best_state = clone_state_dict_to_cpu(
                global_model
            )

            torch.save(
                {
                    "phase":
                        "14F",
                    "aggregation":
                        AGGREGATION_METHOD,
                    "round":
                        round_number,
                    "seed":
                        SEED,
                    "model_state_dict":
                        best_state,
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
                    "fedprox_mu":
                        FEDPROX_MU,
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
                },
                BEST_MODEL_FILE,
            )

        last_round_loss = validation_loss

        last_round_metrics = {
            key:
                (
                    value.copy()
                    if isinstance(
                        value,
                        np.ndarray,
                    )
                    else value
                )
            for key, value
            in validation_metrics.items()
        }

        last_round_y_true = y_true.copy()
        last_round_y_pred = y_pred.copy()

        last_round_inference_seconds = (
            inference_seconds
        )

        print()
        print(
            f"Round {round_number:02d} global validation"
        )

        print(
            f"Loss                     : "
            f"{validation_loss:.6f}"
        )

        print(
            f"Acc / BA / MacroF1       : "
            f"{validation_metrics['accuracy']:.6f} / "
            f"{validation_metrics['balanced_accuracy']:.6f} / "
            f"{validation_metrics['macro_f1']:.6f}"
        )

        print(
            f"Weighted F1              : "
            f"{validation_metrics['weighted_f1']:.6f}"
        )

        print(
            "Per-class recall         : "
            f"BENIGN={recalls[0]:.4f}, "
            f"DDoS={recalls[1]:.4f}, "
            f"GoldenEye={recalls[2]:.4f}, "
            f"Hulk={recalls[3]:.4f}, "
            f"FTP={recalls[4]:.4f}, "
            f"PortScan={recalls[5]:.4f}"
        )

        print(
            f"Best round / MacroF1     : "
            f"{best_round} / "
            f"{best_macro_f1:.6f}"
        )

        print(
            f"Round time               : "
            f"{round_seconds:.2f}s"
        )

        pd.DataFrame(
            round_rows
        ).to_csv(
            ROUND_HISTORY_FILE,
            index=False,
        )

        pd.DataFrame(
            client_round_rows
        ).to_csv(
            CLIENT_ROUND_METRICS_FILE,
            index=False,
        )

    total_training_seconds = (
        time.perf_counter()
        -
        total_training_start
    )

    # ========================================================
    # SAVE FINAL ROUND CHECKPOINT
    # ========================================================

    final_state = clone_state_dict_to_cpu(
        global_model
    )

    torch.save(
        {
            "phase":
                "14F",
            "aggregation":
                AGGREGATION_METHOD,
            "round":
                FEDERATED_ROUNDS,
            "seed":
                SEED,
            "model_state_dict":
                final_state,
            "validation_macro_f1":
                last_round_metrics[
                    "macro_f1"
                ],
            "feature_count":
                EXPECTED_FEATURES,
            "sequence_length":
                EXPECTED_SEQUENCE_LENGTH,
            "num_classes":
                NUM_CLASSES,
            "class_names":
                class_names,
            "fedprox_mu":
                FEDPROX_MU,
            "class_weights":
                class_weights.tolist(),
        },
        FINAL_ROUND_MODEL_FILE,
    )

    # ========================================================
    # SAVE BEST / FINAL METRICS
    # ========================================================

    if best_state is None:
        raise RuntimeError(
            "\nNo best federated checkpoint was produced."
        )

    best_payload = metrics_payload(
        round_number=best_round,
        loss=best_validation_loss,
        metrics=best_metrics,
        training_seconds=
            total_training_seconds,
        inference_seconds=0.0,
    )

    final_payload = metrics_payload(
        round_number=
            FEDERATED_ROUNDS,
        loss=
            last_round_loss,
        metrics=
            last_round_metrics,
        training_seconds=
            total_training_seconds,
        inference_seconds=
            last_round_inference_seconds,
    )

    save_json(
        BEST_METRICS_FILE,
        best_payload,
    )

    save_json(
        FINAL_ROUND_METRICS_FILE,
        final_payload,
    )

    best_per_class = per_class_dataframe(
        best_metrics,
        class_names,
    )

    final_per_class = per_class_dataframe(
        last_round_metrics,
        class_names,
    )

    best_per_class.to_csv(
        BEST_PER_CLASS_FILE,
        index=False,
    )

    final_per_class.to_csv(
        FINAL_PER_CLASS_FILE,
        index=False,
    )

    save_confusion_matrix(
        best_y_true,
        best_y_pred,
        class_names,
        BEST_CONFUSION_MATRIX_FILE,
    )

    save_confusion_matrix(
        last_round_y_true,
        last_round_y_pred,
        class_names,
        FINAL_CONFUSION_MATRIX_FILE,
    )

    # ========================================================
    # COMMUNICATION ACCOUNTING
    # Full model downlink + uplink for every client/round.
    # ========================================================

    model_bytes = (
        parameter_count
        *
        4
    )

    model_size_mib = (
        model_bytes
        /
        1024
        /
        1024
    )

    communication_bytes = (
        model_bytes
        *
        NUM_CLIENTS
        *
        FEDERATED_ROUNDS
        *
        2
    )

    communication_mib = (
        communication_bytes
        /
        1024
        /
        1024
    )

    # ========================================================
    # OPTIONAL CENTRALIZED REFERENCE GAP
    # ========================================================

    centralized_reference_file = (
        RESULT_ROOT
        /
        "phase14e_validation_metrics.json"
    )

    centralized_reference = None

    if centralized_reference_file.exists():
        with open(
            centralized_reference_file,
            "r",
            encoding="utf-8",
        ) as file:
            centralized_reference = json.load(
                file
            )

    centralized_gap = None

    if centralized_reference is not None:
        centralized_gap = {
            "balanced_accuracy_gap_best_federated_minus_centralized":
                float(
                    best_metrics[
                        "balanced_accuracy"
                    ]
                    -
                    centralized_reference[
                        "balanced_accuracy"
                    ]
                ),
            "macro_f1_gap_best_federated_minus_centralized":
                float(
                    best_metrics[
                        "macro_f1"
                    ]
                    -
                    centralized_reference[
                        "macro_f1"
                    ]
                ),
        }

    # ========================================================
    # METADATA
    # ========================================================

    metadata = {
        "phase":
            "14F",

        "component":
            "federated_fedprox_cnn_bilstm",

        "scenario":
            "non_iid",

        "seed":
            SEED,

        "device":
            str(
                device
            ),

        "aggregation":
            AGGREGATION_METHOD,

        "aggregation_weighting":
            AGGREGATION_WEIGHTING,

        "clients":
            NUM_CLIENTS,

        "client_sequence_counts":
            client_sequence_counts,

        "rounds":
            FEDERATED_ROUNDS,

        "local_epochs_per_round":
            LOCAL_EPOCHS,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "fedprox_mu":
            FEDPROX_MU,

        "gradient_clip_norm":
            GRADIENT_CLIP_NORM,

        "initialization":
            "fresh_common_random_initialization",

        "centralized_checkpoint_loaded":
            False,

        "validation_file":
            str(
                VALIDATION_FILE
            ),

        "global_class_weight_source":
            str(
                GLOBAL_TRAIN_SEQUENCE_FILE
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

        "locked_test_path":
            str(
                LOCKED_TEST_FILE
            ),

        "locked_test_used":
            False,

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

        "best_round":
            int(
                best_round
            ),

        "best_validation_metrics":
            best_payload,

        "final_round_validation_metrics":
            final_payload,

        "model_size_mib_float32":
            float(
                model_size_mib
            ),

        "communication_assumption":
            (
                "full float32 model downlink and uplink "
                "for all 5 clients each of 10 rounds"
            ),

        "estimated_total_communication_mib":
            float(
                communication_mib
            ),

        "centralized_reference":
            centralized_reference,

        "centralized_gap":
            centralized_gap,

        "total_training_seconds":
            float(
                total_training_seconds
            ),

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
    print("PHASE 14F COMPLETE")
    separator()

    print(
        f"Aggregation               : "
        f"{AGGREGATION_METHOD}"
    )

    print(
        f"Rounds                    : "
        f"{FEDERATED_ROUNDS}"
    )

    print(
        f"Local epochs / round      : "
        f"{LOCAL_EPOCHS}"
    )

    print(
        f"FedProx mu                : "
        f"{FEDPROX_MU}"
    )

    print(
        f"Parameters                : "
        f"{parameter_count:,}"
    )

    print(
        f"Model size (float32)      : "
        f"{model_size_mib:.4f} MiB"
    )

    print(
        f"Estimated communication   : "
        f"{communication_mib:.4f} MiB"
    )

    print()
    print("BEST VALIDATION ROUND")
    print("-" * 118)

    print(
        f"Best round                : "
        f"{best_round}"
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
    print("BEST-ROUND PER-CLASS VALIDATION METRICS")
    print("-" * 118)

    print(
        best_per_class[
            [
                "class_name",
                "precision",
                "recall",
                "f1",
                "support",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("FINAL ROUND")
    print("-" * 118)

    print(
        f"Round                     : "
        f"{FEDERATED_ROUNDS}"
    )

    print(
        f"Accuracy                  : "
        f"{last_round_metrics['accuracy']:.6f}"
    )

    print(
        f"Balanced Accuracy         : "
        f"{last_round_metrics['balanced_accuracy']:.6f}"
    )

    print(
        f"Macro F1                  : "
        f"{last_round_metrics['macro_f1']:.6f}"
    )

    print(
        f"Weighted F1               : "
        f"{last_round_metrics['weighted_f1']:.6f}"
    )

    if centralized_gap is not None:
        print()
        print("CENTRALIZED PHASE14E REFERENCE GAP")
        print("-" * 118)

        print(
            f"Best FedProx BA - centralized BA : "
            f"{centralized_gap['balanced_accuracy_gap_best_federated_minus_centralized']:+.6f}"
        )

        print(
            f"Best FedProx F1 - centralized F1 : "
            f"{centralized_gap['macro_f1_gap_best_federated_minus_centralized']:+.6f}"
        )

    print()
    print(
        f"Total federated time      : "
        f"{total_training_seconds:.2f}s"
    )

    print(
        f"Best model                : "
        f"{BEST_MODEL_FILE}"
    )

    print(
        f"Final-round model         : "
        f"{FINAL_ROUND_MODEL_FILE}"
    )

    print(
        f"Round history             : "
        f"{ROUND_HISTORY_FILE}"
    )

    print(
        f"Client-round metrics      : "
        f"{CLIENT_ROUND_METRICS_FILE}"
    )

    print(
        f"Best per-class metrics    : "
        f"{BEST_PER_CLASS_FILE}"
    )

    print(
        f"Final per-class metrics   : "
        f"{FINAL_PER_CLASS_FILE}"
    )

    print(
        f"Metadata                  : "
        f"{METADATA_FILE}"
    )

    print()
    print("LOCKED TEST USED : NO")

    print()
    print("SCIENTIFIC LABEL:")
    print(
        "SOURCE-ORDERED flow sequences; "
        "NOT timestamp-confirmed temporal sequences."
    )

    # ========================================================
    # CLEANUP
    # ========================================================

    del X_validation
    del y_validation
    del validation_dataset
    del validation_loader
    del global_model
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
