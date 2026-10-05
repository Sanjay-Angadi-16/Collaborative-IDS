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
    "PHASE 14F2 — FEDPROX CNN-BILSTM CONVERGENCE EXTENSION"
)


# ============================================================
# FROZEN CONFIGURATION
# ============================================================

SEED = 42

NUM_CLIENTS = 5

# Phase14F1 already completed rounds 1..20.
START_ROUND = 21
FINAL_ROUND = 30

LOCAL_EPOCHS = 1

BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

FEDPROX_MU = 0.01
GRADIENT_CLIP_NORM = 5.0

NUM_WORKERS = 0

AGGREGATION_METHOD = "FedProx"
AGGREGATION_WEIGHTING = "client_sequence_count"

MODEL_SELECTION_METRIC = "validation_macro_f1"


# ============================================================
# CNN-BILSTM ARCHITECTURE
# EXACTLY SAME AS PHASE14E / 14F / 14F1
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

EXPECTED_PARAMETER_COUNT = 126_982


# ============================================================
# CLASS WEIGHT POLICY
# Same fixed TRAIN-only weights used by Phase14E / 14F / 14F1.
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

PHASE14F1_RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "phase14"
    / "phase14f1"
)

PHASE14F1_ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "detection"
    / "phase14f1"
)

# We explicitly resume from the FINAL ROUND-20 model,
# not merely from a "best" model.
SOURCE_CHECKPOINT = (
    PHASE14F1_ARTIFACT_ROOT
    / "fedprox_mu001"
    / "final_round_model.pt"
)

SOURCE_ROUND_HISTORY_FILE = (
    PHASE14F1_RESULT_ROOT
    / "fedprox_mu001"
    / "round_history.csv"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "phase14"
    / "phase14f2"
)

ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "detection"
    / "phase14f2"
)

BEST_MODEL_FILE = (
    ARTIFACT_ROOT
    / "cnn_bilstm_fedprox_mu001_best_round_1_to_30.pt"
)

FINAL_MODEL_FILE = (
    ARTIFACT_ROOT
    / "cnn_bilstm_fedprox_mu001_round30.pt"
)

EXTENSION_HISTORY_FILE = (
    RESULT_ROOT
    / "phase14f2_round21_to_30_history.csv"
)

COMBINED_HISTORY_FILE = (
    RESULT_ROOT
    / "phase14f2_combined_round1_to_30_history.csv"
)

CLIENT_ROUND_METRICS_FILE = (
    RESULT_ROOT
    / "phase14f2_client_round21_to_30_metrics.csv"
)

BEST_METRICS_FILE = (
    RESULT_ROOT
    / "phase14f2_best_validation_metrics.json"
)

FINAL_METRICS_FILE = (
    RESULT_ROOT
    / "phase14f2_round30_validation_metrics.json"
)

BEST_PER_CLASS_FILE = (
    RESULT_ROOT
    / "phase14f2_best_per_class_metrics.csv"
)

FINAL_PER_CLASS_FILE = (
    RESULT_ROOT
    / "phase14f2_round30_per_class_metrics.csv"
)

BEST_CONFUSION_MATRIX_FILE = (
    RESULT_ROOT
    / "phase14f2_best_confusion_matrix.csv"
)

FINAL_CONFUSION_MATRIX_FILE = (
    RESULT_ROOT
    / "phase14f2_round30_confusion_matrix.csv"
)

CLASS_WEIGHTS_FILE = (
    RESULT_ROOT
    / "phase14f2_class_weights.csv"
)

METADATA_FILE = (
    ARTIFACT_ROOT
    / "phase14f2_metadata.json"
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
    "phase14f2_cnn_bilstm"
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


def separator(width: int = 122) -> None:
    print()
    print("=" * width)


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


def clone_state_dict_to_cpu(
    model: nn.Module,
):
    return OrderedDict(
        (
            key,
            value.detach().cpu().clone(),
        )
        for key, value
        in model.state_dict().items()
    )


def copy_metric_dict(metrics):
    return {
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
        in metrics.items()
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
            f"\n{dataset_name}: expected "
            f"(*,{EXPECTED_SEQUENCE_LENGTH},"
            f"{EXPECTED_FEATURES}); "
            f"found {X.shape}."
        )

    if y.ndim != 1:
        raise ValueError(
            f"\n{dataset_name}: expected 1D y."
        )

    if len(X) != len(y):
        raise ValueError(
            f"\n{dataset_name}: X/y count mismatch."
        )

    unique_classes = np.unique(y)

    if np.any(unique_classes < 0):
        raise ValueError(
            f"\n{dataset_name}: negative class id."
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
# GLOBAL TRAIN-ONLY CLASS WEIGHTS
# ============================================================

def compute_global_class_weights():
    y_global = load_y_only(
        GLOBAL_TRAIN_SEQUENCE_FILE,
        "Phase14E0 global training sequences",
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
            "\nGlobal training sequences are missing "
            f"class ids: {missing.tolist()}"
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
        x = x.transpose(
            1,
            2,
        )

        x = self.relu(
            self.conv1(x)
        )

        x = self.relu(
            self.conv2(x)
        )

        x = x.transpose(
            1,
            2,
        )

        x = self.layer_norm(
            x
        )

        lstm_output, (
            h_n,
            c_n,
        ) = self.bilstm(
            x
        )

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

def calculate_global_metrics(
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

    weighted_f1 = float(
        np.sum(
            f1
            *
            support
        )
        /
        np.sum(support)
    )

    return {
        "accuracy":
            float(accuracy),

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


def calculate_present_class_metrics(
    y_true,
    y_pred,
):
    present_labels = np.unique(
        y_true
    )

    (
        precision,
        recall,
        f1,
        support,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=present_labels,
        zero_division=0,
    )

    return {
        "present_class_balanced_accuracy":
            float(
                np.mean(recall)
            ),

        "present_class_macro_precision":
            float(
                np.mean(precision)
            ),

        "present_class_macro_f1":
            float(
                np.mean(f1)
            ),

        "present_labels":
            present_labels.tolist(),
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
    local_model = CNNBiLSTM(
        feature_count=EXPECTED_FEATURES,
        num_classes=NUM_CLASSES,
    ).to(device)

    local_model.load_state_dict(
        global_model.state_dict()
    )

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
        torch.from_numpy(
            X_client
        ),
        torch.from_numpy(
            y_client
        ),
    )

    generator = torch.Generator()

    # EXACT continuation of the Phase14F1 shuffle schedule:
    # round 21 uses the same formula that a single 30-round
    # run would have used from the beginning.
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
        pin_memory=(
            device.type
            ==
            "cuda"
        ),
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

    local_start = (
        time.perf_counter()
    )

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

            for (
                local_parameter,
                global_parameter,
            ) in zip(
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

            objective_loss = (
                ce_loss
                +
                prox_loss
            )

            objective_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                local_model.parameters(),
                max_norm=
                    GRADIENT_CLIP_NORM,
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
                    objective_loss.item()
                )
                *
                batch_size
            )

            total_samples += (
                batch_size
            )

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

    present_metrics = (
        calculate_present_class_metrics(
            y_true,
            y_pred,
        )
    )

    local_state = (
        clone_state_dict_to_cpu(
            local_model
        )
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

        "present_class_balanced_accuracy":
            present_metrics[
                "present_class_balanced_accuracy"
            ],

        "present_class_macro_precision":
            present_metrics[
                "present_class_macro_precision"
            ],

        "present_class_macro_f1":
            present_metrics[
                "present_class_macro_f1"
            ],

        "present_labels":
            json.dumps(
                present_metrics[
                    "present_labels"
                ]
            ),

        "local_seconds":
            float(
                local_seconds
            ),
    }

    del local_model
    del global_reference_parameters
    del loader
    del dataset
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
# SAMPLE-WEIGHTED AGGREGATION
# ============================================================

def weighted_federated_average(
    client_states,
    client_sequence_counts,
):
    if not client_states:
        raise ValueError(
            "No client states supplied."
        )

    weights = np.asarray(
        client_sequence_counts,
        dtype=np.float64,
    )

    if len(weights) != len(
        client_states
    ):
        raise ValueError(
            "State/count length mismatch."
        )

    if np.any(weights <= 0):
        raise ValueError(
            "Aggregation weights must be positive."
        )

    normalized_weights = (
        weights
        /
        np.sum(weights)
    )

    first_state = client_states[0]

    averaged_state = OrderedDict()

    for key in first_state.keys():
        reference_tensor = (
            first_state[key]
        )

        if not torch.is_floating_point(
            reference_tensor
        ):
            averaged_state[key] = (
                reference_tensor.clone()
            )

            continue

        averaged_tensor = (
            torch.zeros_like(
                reference_tensor
            )
        )

        for (
            client_state,
            normalized_weight,
        ) in zip(
            client_states,
            normalized_weights,
        ):
            averaged_tensor.add_(
                client_state[key],
                alpha=float(
                    normalized_weight
                ),
            )

        averaged_state[key] = (
            averaged_tensor
        )

    return averaged_state


# ============================================================
# GLOBAL VALIDATION
# ============================================================

@torch.no_grad()
def evaluate_global(
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
            y_batch.shape[0]
        )

        total_loss += (
            float(
                loss.item()
            )
            *
            batch_size
        )

        total_samples += (
            batch_size
        )

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

    metrics = calculate_global_metrics(
        y_true,
        y_pred,
    )

    return (
        float(
            total_loss
            /
            total_samples
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
        f"Device                    : {device}"
    )

    print(
        f"Seed                      : {SEED}"
    )

    print(
        f"Aggregation               : {AGGREGATION_METHOD}"
    )

    print(
        f"FedProx mu                : {FEDPROX_MU}"
    )

    print(
        f"Clients                   : {NUM_CLIENTS}"
    )

    print(
        f"Resume from round         : {START_ROUND - 1}"
    )

    print(
        f"Extension rounds          : "
        f"{START_ROUND}..{FINAL_ROUND}"
    )

    print(
        f"Local epochs              : {LOCAL_EPOCHS}"
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
        f"Gradient clipping         : "
        f"{GRADIENT_CLIP_NORM}"
    )

    print(
        f"Architecture parameters   : "
        f"{EXPECTED_PARAMETER_COUNT:,}"
    )

    print(
        f"Source checkpoint         : "
        f"{SOURCE_CHECKPOINT}"
    )

    print(
        f"Validation                : "
        f"{VALIDATION_FILE}"
    )

    print(
        "Locked test               : NO"
    )

    print()
    print("CONVERGENCE-EXTENSION POLICY")
    print("-" * 122)

    print(
        "1. Continues the Phase14F1 winner: FedProx mu=0.01."
    )

    print(
        "2. Loads the FINAL round-20 checkpoint, not a centralized model."
    )

    print(
        "3. Continues rounds 21..30 only; does not retrain rounds 1..20."
    )

    print(
        "4. Same frozen client sequences and validation sequences."
    )

    print(
        "5. Same architecture, seed, LR, weight decay, local epochs and batch size."
    )

    print(
        "6. Same per-round/client shuffle-seed formula as Phase14F1."
    )

    print(
        "7. Same fixed TRAIN-only global class weights."
    )

    print(
        "8. No early stopping; round 30 is always reached."
    )

    print(
        "9. Best round across 1..30 is selected by validation Macro F1."
    )

    print(
        "10. Locked test is never read."
    )

    print(
        "11. SOURCE-ORDERED; not timestamp-confirmed temporal sequences."
    )

    # ========================================================
    # PREPROCESSING / LABELS
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

    inverse_mapping = (
        get_inverse_mapping(
            label_mapping
        )
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
        str(label):
            int(class_id)
        for label, class_id
        in label_mapping.items()
    }

    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    (
        global_class_counts,
        class_weights,
    ) = compute_global_class_weights()

    class_weight_rows = []

    print()
    print("GLOBAL TRAIN-ONLY CLASS WEIGHTS")
    print("-" * 122)

    for class_id in range(
        NUM_CLASSES
    ):
        class_weight_rows.append(
            {
                "class_id":
                    class_id,

                "class_name":
                    class_names[
                        class_id
                    ],

                "train_sequences":
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
    # CLIENT DATA AUDIT
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
    print("-" * 122)

    for client_id, path in client_files.items():
        y_client = load_y_only(
            path,
            f"client_{client_id}",
        )

        unique, counts = np.unique(
            y_client,
            return_counts=True,
        )

        client_sequence_counts[
            client_id
        ] = int(
            len(
                y_client
            )
        )

        expected_ids = {
            label_to_id[
                "BENIGN"
            ],
            label_to_id[
                CLIENT_ATTACKS[
                    client_id
                ]
            ],
        }

        actual_ids = {
            int(item)
            for item
            in unique
        }

        if not actual_ids.issubset(
            expected_ids
        ):
            raise RuntimeError(
                f"\nclient_{client_id} contains "
                "unexpected classes."
            )

        distribution = {
            class_names[
                int(class_id)
            ]:
                int(count)
            for class_id, count
            in zip(
                unique,
                counts,
            )
        }

        print(
            f"client_{client_id} | "
            f"n={len(y_client):,} | "
            f"{distribution}"
        )

        del y_client

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

    if set(
        int(item)
        for item
        in np.unique(
            y_validation
        )
    ) != set(
        range(
            NUM_CLASSES
        )
    ):
        raise RuntimeError(
            "\nValidation must contain all 6 classes."
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
    # LOAD ROUND-20 WINNER CHECKPOINT
    # ========================================================

    if not SOURCE_CHECKPOINT.exists():
        raise FileNotFoundError(
            "\nPhase14F1 round-20 checkpoint missing:\n"
            f"{SOURCE_CHECKPOINT}"
        )

    checkpoint = torch.load(
        SOURCE_CHECKPOINT,
        map_location=
            device,
    )

    checkpoint_round = int(
        checkpoint.get(
            "round",
            -1,
        )
    )

    checkpoint_mu = float(
        checkpoint.get(
            "mu",
            FEDPROX_MU,
        )
    )

    checkpoint_aggregation = str(
        checkpoint.get(
            "aggregation",
            AGGREGATION_METHOD,
        )
    )

    if checkpoint_round != (
        START_ROUND - 1
    ):
        raise RuntimeError(
            "\nExpected source checkpoint at round "
            f"{START_ROUND - 1}, "
            f"found round {checkpoint_round}."
        )

    if abs(
        checkpoint_mu
        -
        FEDPROX_MU
    ) > 1e-12:
        raise RuntimeError(
            "\nSource checkpoint mu mismatch.\n"
            f"Expected {FEDPROX_MU}, "
            f"found {checkpoint_mu}."
        )

    if checkpoint_aggregation.lower() != (
        AGGREGATION_METHOD.lower()
    ):
        raise RuntimeError(
            "\nSource checkpoint aggregation mismatch."
        )

    global_model = CNNBiLSTM(
        feature_count=EXPECTED_FEATURES,
        num_classes=NUM_CLASSES,
    ).to(device)

    global_model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    parameter_count = sum(
        parameter.numel()
        for parameter
        in global_model.parameters()
    )

    if parameter_count != (
        EXPECTED_PARAMETER_COUNT
    ):
        raise RuntimeError(
            "\nArchitecture drift detected.\n"
            f"Expected {EXPECTED_PARAMETER_COUNT:,}; "
            f"found {parameter_count:,}."
        )

    print()
    print("ROUND-20 CHECKPOINT VERIFIED")
    print("-" * 122)

    print(
        f"Checkpoint round          : "
        f"{checkpoint_round}"
    )

    print(
        f"Checkpoint aggregation    : "
        f"{checkpoint_aggregation}"
    )

    print(
        f"Checkpoint mu             : "
        f"{checkpoint_mu}"
    )

    print(
        f"Parameters                : "
        f"{parameter_count:,}"
    )

    # ========================================================
    # LOAD EXISTING ROUND1..20 HISTORY
    # ========================================================

    if not SOURCE_ROUND_HISTORY_FILE.exists():
        raise FileNotFoundError(
            "\nPhase14F1 round history missing:\n"
            f"{SOURCE_ROUND_HISTORY_FILE}"
        )

    prior_history = pd.read_csv(
        SOURCE_ROUND_HISTORY_FILE
    )

    if len(
        prior_history
    ) != 20:
        raise RuntimeError(
            "\nExpected exactly 20 prior rounds in "
            "Phase14F1 history."
        )

    if int(
        prior_history[
            "round"
        ].min()
    ) != 1 or int(
        prior_history[
            "round"
        ].max()
    ) != 20:
        raise RuntimeError(
            "\nPhase14F1 history must cover rounds 1..20."
        )

    # Recover the best prior round from the actual saved history.
    prior_best_index = (
        prior_history[
            "macro_f1"
        ]
        .astype(float)
        .idxmax()
    )

    prior_best_row = (
        prior_history
        .loc[
            prior_best_index
        ]
    )

    best_round = int(
        prior_best_row[
            "round"
        ]
    )

    best_macro_f1 = float(
        prior_best_row[
            "macro_f1"
        ]
    )

    # We will obtain full best metrics from either:
    # - current round-20 model if prior best is 20, or
    # - the Phase14F1 best checkpoint if ever needed.
    if best_round != 20:
        raise RuntimeError(
            "\nPhase14F2 expects Phase14F1 winner's best "
            "round to be 20 based on the confirmed result.\n"
            f"History reports best round {best_round}."
        )

    (
        round20_validation_loss,
        round20_metrics,
        round20_y_true,
        round20_y_pred,
    ) = evaluate_global(
        model=global_model,
        dataloader=
            validation_loader,
        criterion=
            validation_criterion,
        device=
            device,
    )

    best_validation_loss = (
        round20_validation_loss
    )

    best_metrics = (
        copy_metric_dict(
            round20_metrics
        )
    )

    best_y_true = (
        round20_y_true.copy()
    )

    best_y_pred = (
        round20_y_pred.copy()
    )

    best_state = (
        clone_state_dict_to_cpu(
            global_model
        )
    )

    # Sanity-check history vs re-evaluated checkpoint.
    if abs(
        best_metrics[
            "macro_f1"
        ]
        -
        best_macro_f1
    ) > 1e-5:
        raise RuntimeError(
            "\nRound-20 checkpoint Macro F1 does not "
            "match Phase14F1 history."
        )

    # ========================================================
    # CONTINUE ROUNDS 21..30
    # ========================================================

    extension_rows = []

    client_rows = []

    total_extension_start = (
        time.perf_counter()
    )

    last_validation_loss = None
    last_metrics = None
    last_y_true = None
    last_y_pred = None

    for round_number in range(
        START_ROUND,
        FINAL_ROUND + 1,
    ):
        separator()

        print(
            f"FEDPROX CONVERGENCE ROUND "
            f"{round_number:02d}/"
            f"{FINAL_ROUND}"
        )

        print(
            "-" * 122
        )

        round_start = (
            time.perf_counter()
        )

        client_states = []
        aggregation_counts = []

        for client_id in range(
            1,
            NUM_CLIENTS + 1,
        ):
            (
                X_client,
                y_client,
            ) = load_sequence_npz(
                client_files[
                    client_id
                ],
                f"round{round_number} client_{client_id}",
            )

            actual_count = int(
                len(
                    y_client
                )
            )

            if actual_count != (
                client_sequence_counts[
                    client_id
                ]
            ):
                raise RuntimeError(
                    f"\nclient_{client_id} "
                    "sequence count changed."
                )

            (
                local_state,
                local_result,
            ) = train_local_fedprox(
                global_model=
                    global_model,

                X_client=
                    X_client,

                y_client=
                    y_client,

                class_weights=
                    class_weights,

                device=
                    device,

                round_number=
                    round_number,

                client_id=
                    client_id,
            )

            client_states.append(
                local_state
            )

            aggregation_counts.append(
                actual_count
            )

            client_rows.append(
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

                    "present_class_balanced_accuracy":
                        local_result[
                            "present_class_balanced_accuracy"
                        ],

                    "present_class_macro_precision":
                        local_result[
                            "present_class_macro_precision"
                        ],

                    "present_class_macro_f1":
                        local_result[
                            "present_class_macro_f1"
                        ],

                    "present_labels":
                        local_result[
                            "present_labels"
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
                f"CE={local_result['ce_loss']:.6f} | "
                f"prox={local_result['prox_loss']:.6f} | "
                f"present-BA="
                f"{local_result['present_class_balanced_accuracy']:.6f} | "
                f"present-F1="
                f"{local_result['present_class_macro_f1']:.6f} | "
                f"{local_result['local_seconds']:.2f}s"
            )

            del X_client
            del y_client
            del local_state

            gc.collect()

            if device.type == "cuda":
                torch.cuda.empty_cache()

        aggregated_state = (
            weighted_federated_average(
                client_states=
                    client_states,

                client_sequence_counts=
                    aggregation_counts,
            )
        )

        global_model.load_state_dict(
            aggregated_state
        )

        del client_states
        del aggregated_state

        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

        validation_start = (
            time.perf_counter()
        )

        (
            validation_loss,
            validation_metrics,
            y_true,
            y_pred,
        ) = evaluate_global(
            model=
                global_model,

            dataloader=
                validation_loader,

            criterion=
                validation_criterion,

            device=
                device,
        )

        validation_seconds = (
            time.perf_counter()
            -
            validation_start
        )

        round_seconds = (
            time.perf_counter()
            -
            round_start
        )

        recalls = (
            validation_metrics[
                "per_class_recall"
            ]
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
            best_round = (
                round_number
            )

            best_macro_f1 = (
                current_macro_f1
            )

            best_validation_loss = (
                validation_loss
            )

            best_metrics = (
                copy_metric_dict(
                    validation_metrics
                )
            )

            best_y_true = (
                y_true.copy()
            )

            best_y_pred = (
                y_pred.copy()
            )

            best_state = (
                clone_state_dict_to_cpu(
                    global_model
                )
            )

        last_validation_loss = (
            validation_loss
        )

        last_metrics = (
            copy_metric_dict(
                validation_metrics
            )
        )

        last_y_true = (
            y_true.copy()
        )

        last_y_pred = (
            y_pred.copy()
        )

        extension_rows.append(
            {
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

                "is_best_across_1_to_30":
                    bool(
                        improved
                    ),

                "validation_seconds":
                    validation_seconds,

                "round_seconds":
                    round_seconds,
            }
        )

        print()
        print(
            f"GLOBAL VALIDATION | "
            f"Acc={validation_metrics['accuracy']:.6f} | "
            f"BA={validation_metrics['balanced_accuracy']:.6f} | "
            f"MacroF1={validation_metrics['macro_f1']:.6f} | "
            f"WeightedF1={validation_metrics['weighted_f1']:.6f}"
        )

        print(
            "Recall | "
            f"BENIGN={recalls[0]:.4f} | "
            f"DDoS={recalls[1]:.4f} | "
            f"GoldenEye={recalls[2]:.4f} | "
            f"Hulk={recalls[3]:.4f} | "
            f"FTP={recalls[4]:.4f} | "
            f"PortScan={recalls[5]:.4f}"
        )

        print(
            f"Best round across 1..30  : "
            f"{best_round} | "
            f"MacroF1={best_macro_f1:.6f}"
        )

        pd.DataFrame(
            extension_rows
        ).to_csv(
            EXTENSION_HISTORY_FILE,
            index=False,
        )

        pd.DataFrame(
            client_rows
        ).to_csv(
            CLIENT_ROUND_METRICS_FILE,
            index=False,
        )

    extension_seconds = (
        time.perf_counter()
        -
        total_extension_start
    )

    # ========================================================
    # COMBINE ROUND 1..20 + 21..30 HISTORY
    # ========================================================

    extension_history = pd.DataFrame(
        extension_rows
    )

    prior_history_for_merge = (
        prior_history.copy()
    )

    # Keep only columns shared with the extension table,
    # then concatenate to form a clean round1..30 trajectory.
    shared_columns = [
        column
        for column
        in extension_history.columns
        if column
        in prior_history_for_merge.columns
    ]

    if "round" not in shared_columns:
        raise RuntimeError(
            "\nCould not merge Phase14F1 and Phase14F2 histories."
        )

    combined_history = pd.concat(
        [
            prior_history_for_merge[
                shared_columns
            ],
            extension_history[
                shared_columns
            ],
        ],
        ignore_index=True,
    )

    combined_history = (
        combined_history
        .sort_values(
            "round"
        )
        .reset_index(
            drop=True
        )
    )

    combined_history.to_csv(
        COMBINED_HISTORY_FILE,
        index=False,
    )

    # ========================================================
    # SAVE BEST MODEL ACROSS ROUNDS 1..30
    # ========================================================

    torch.save(
        {
            "phase":
                "14F2",

            "aggregation":
                AGGREGATION_METHOD,

            "mu":
                FEDPROX_MU,

            "round":
                best_round,

            "seed":
                SEED,

            "model_state_dict":
                best_state,

            "validation_macro_f1":
                best_metrics[
                    "macro_f1"
                ],

            "validation_balanced_accuracy":
                best_metrics[
                    "balanced_accuracy"
                ],

            "feature_count":
                EXPECTED_FEATURES,

            "sequence_length":
                EXPECTED_SEQUENCE_LENGTH,

            "num_classes":
                NUM_CLASSES,

            "class_names":
                class_names,
        },
        BEST_MODEL_FILE,
    )

    final_state = (
        clone_state_dict_to_cpu(
            global_model
        )
    )

    torch.save(
        {
            "phase":
                "14F2",

            "aggregation":
                AGGREGATION_METHOD,

            "mu":
                FEDPROX_MU,

            "round":
                FINAL_ROUND,

            "seed":
                SEED,

            "model_state_dict":
                final_state,

            "validation_macro_f1":
                last_metrics[
                    "macro_f1"
                ],

            "validation_balanced_accuracy":
                last_metrics[
                    "balanced_accuracy"
                ],
        },
        FINAL_MODEL_FILE,
    )

    # ========================================================
    # SAVE BEST / FINAL METRICS
    # ========================================================

    best_payload = {
        "phase":
            "14F2",

        "aggregation":
            AGGREGATION_METHOD,

        "mu":
            FEDPROX_MU,

        "best_round":
            int(
                best_round
            ),

        "validation_loss":
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

        "extension_seconds":
            float(
                extension_seconds
            ),

        "locked_test_used":
            False,
    }

    final_payload = {
        "phase":
            "14F2",

        "aggregation":
            AGGREGATION_METHOD,

        "mu":
            FEDPROX_MU,

        "round":
            FINAL_ROUND,

        "validation_loss":
            float(
                last_validation_loss
            ),

        "accuracy":
            last_metrics[
                "accuracy"
            ],

        "balanced_accuracy":
            last_metrics[
                "balanced_accuracy"
            ],

        "macro_precision":
            last_metrics[
                "macro_precision"
            ],

        "macro_recall":
            last_metrics[
                "macro_recall"
            ],

        "macro_f1":
            last_metrics[
                "macro_f1"
            ],

        "weighted_f1":
            last_metrics[
                "weighted_f1"
            ],

        "extension_seconds":
            float(
                extension_seconds
            ),

        "locked_test_used":
            False,
    }

    save_json(
        BEST_METRICS_FILE,
        best_payload,
    )

    save_json(
        FINAL_METRICS_FILE,
        final_payload,
    )

    best_per_class = (
        per_class_dataframe(
            best_metrics,
            class_names,
        )
    )

    final_per_class = (
        per_class_dataframe(
            last_metrics,
            class_names,
        )
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
        last_y_true,
        last_y_pred,
        class_names,
        FINAL_CONFUSION_MATRIX_FILE,
    )

    # ========================================================
    # SIMPLE CONVERGENCE DIAGNOSTIC
    # ========================================================

    tail = (
        combined_history
        .tail(5)
        .copy()
    )

    if len(tail) == 5:
        tail_macro_f1_change = float(
            tail[
                "macro_f1"
            ]
            .iloc[-1]
            -
            tail[
                "macro_f1"
            ]
            .iloc[0]
        )

        tail_ba_change = float(
            tail[
                "balanced_accuracy"
            ]
            .iloc[-1]
            -
            tail[
                "balanced_accuracy"
            ]
            .iloc[0]
        )

    else:
        tail_macro_f1_change = None
        tail_ba_change = None

    # ========================================================
    # METADATA
    # ========================================================

    model_bytes = (
        parameter_count
        *
        4
    )

    extension_communication_bytes = (
        model_bytes
        *
        NUM_CLIENTS
        *
        (
            FINAL_ROUND
            -
            START_ROUND
            +
            1
        )
        *
        2
    )

    total_round1_to_30_communication_bytes = (
        model_bytes
        *
        NUM_CLIENTS
        *
        FINAL_ROUND
        *
        2
    )

    metadata = {
        "phase":
            "14F2",

        "component":
            "fedprox_cnn_bilstm_convergence_extension",

        "seed":
            SEED,

        "device":
            str(
                device
            ),

        "aggregation":
            AGGREGATION_METHOD,

        "fedprox_mu":
            FEDPROX_MU,

        "clients":
            NUM_CLIENTS,

        "resume_from_round":
            START_ROUND - 1,

        "extension_start_round":
            START_ROUND,

        "final_round":
            FINAL_ROUND,

        "source_checkpoint":
            str(
                SOURCE_CHECKPOINT
            ),

        "source_round_history":
            str(
                SOURCE_ROUND_HISTORY_FILE
            ),

        "local_epochs":
            LOCAL_EPOCHS,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "gradient_clip_norm":
            GRADIENT_CLIP_NORM,

        "aggregation_weighting":
            AGGREGATION_WEIGHTING,

        "class_weight_mode":
            CLASS_WEIGHT_MODE,

        "class_weight_cap":
            CLASS_WEIGHT_CAP,

        "class_weights":
            class_weights.tolist(),

        "client_sequence_counts":
            client_sequence_counts,

        "validation_file":
            str(
                VALIDATION_FILE
            ),

        "parameter_count":
            parameter_count,

        "architecture":
            {
                "conv1_channels":
                    CONV1_CHANNELS,

                "conv2_channels":
                    CONV2_CHANNELS,

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

        "best_round_1_to_30":
            best_round,

        "best_validation_metrics":
            best_payload,

        "round30_validation_metrics":
            final_payload,

        "last5_round_macro_f1_change":
            tail_macro_f1_change,

        "last5_round_balanced_accuracy_change":
            tail_ba_change,

        "extension_training_seconds":
            extension_seconds,

        "extension_communication_mib":
            float(
                extension_communication_bytes
                /
                1024
                /
                1024
            ),

        "estimated_round1_to_30_communication_mib":
            float(
                total_round1_to_30_communication_bytes
                /
                1024
                /
                1024
            ),

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
        "PHASE 14F2 COMPLETE"
    )

    separator()

    print(
        f"FedProx mu                : "
        f"{FEDPROX_MU}"
    )

    print(
        f"Continuation              : "
        f"round {START_ROUND - 1} -> "
        f"round {FINAL_ROUND}"
    )

    print(
        f"Extension time            : "
        f"{extension_seconds:.2f}s"
    )

    print()
    print("BEST ROUND ACROSS 1..30")
    print("-" * 122)

    print(
        f"Best round                : "
        f"{best_round}"
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
    print("BEST-ROUND PER-CLASS METRICS")
    print("-" * 122)

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
    print("ROUND 30")
    print("-" * 122)

    print(
        f"Accuracy                  : "
        f"{last_metrics['accuracy']:.6f}"
    )

    print(
        f"Balanced Accuracy         : "
        f"{last_metrics['balanced_accuracy']:.6f}"
    )

    print(
        f"Macro F1                  : "
        f"{last_metrics['macro_f1']:.6f}"
    )

    print(
        f"Weighted F1               : "
        f"{last_metrics['weighted_f1']:.6f}"
    )

    print()

    print(
        f"Last-5-round MacroF1 Δ    : "
        f"{tail_macro_f1_change:+.6f}"
    )

    print(
        f"Last-5-round BA Δ         : "
        f"{tail_ba_change:+.6f}"
    )

    print()

    if best_round == FINAL_ROUND:
        print(
            "CONVERGENCE STATUS:"
        )

        print(
            "Best validation Macro F1 is still at the "
            "final available round (30)."
        )

        print(
            "The trajectory may still be improving; "
            "consider a controlled extension to round 40."
        )

    else:
        print(
            "CONVERGENCE STATUS:"
        )

        print(
            f"Best validation Macro F1 occurred at "
            f"round {best_round}, before round 30."
        )

        print(
            "Do not extend automatically; inspect rounds "
            "around the best point and freeze the best checkpoint "
            "if the trajectory has plateaued or degraded."
        )

    print()

    print(
        f"Best model                : "
        f"{BEST_MODEL_FILE}"
    )

    print(
        f"Round-30 model            : "
        f"{FINAL_MODEL_FILE}"
    )

    print(
        f"Combined round history    : "
        f"{COMBINED_HISTORY_FILE}"
    )

    print(
        f"Extension round history   : "
        f"{EXTENSION_HISTORY_FILE}"
    )

    print(
        f"Client metrics            : "
        f"{CLIENT_ROUND_METRICS_FILE}"
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

    del X_validation
    del y_validation
    del validation_dataset
    del validation_loader
    del global_model
    del best_state
    del final_state

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
