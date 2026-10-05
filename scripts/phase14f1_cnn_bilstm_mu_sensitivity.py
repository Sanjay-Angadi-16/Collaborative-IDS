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
        "Keep this file inside the scripts folder."
    ) from exc


# ============================================================
# PHASE
# ============================================================

PHASE_NAME = (
    "PHASE 14F1 — FEDERATED CNN-BILSTM "
    "FEDAVG / FEDPROX-MU SENSITIVITY STUDY"
)


# ============================================================
# FAIR-COMPARISON CONFIGURATION
# ============================================================

SEED = 42

NUM_CLIENTS = 5
FEDERATED_ROUNDS = 20
LOCAL_EPOCHS = 1

BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 5.0

NUM_WORKERS = 0

AGGREGATION_WEIGHTING = "client_sequence_count"

MODEL_SELECTION_METRIC = "validation_macro_f1"

# ------------------------------------------------------------
# Controlled study:
# Only the proximal coefficient changes.
#
# mu = 0.0   => FedAvg-equivalent local objective
# mu = 0.001 => weak FedProx
# mu = 0.01  => current Phase14F FedProx setting
# ------------------------------------------------------------

EXPERIMENTS = [
    {
        "experiment_id": "fedavg_mu0",
        "aggregation": "FedAvg",
        "mu": 0.0,
    },
    {
        "experiment_id": "fedprox_mu0001",
        "aggregation": "FedProx",
        "mu": 0.001,
    },
    {
        "experiment_id": "fedprox_mu001",
        "aggregation": "FedProx",
        "mu": 0.01,
    },
]


# ============================================================
# CNN-BILSTM ARCHITECTURE
# EXACTLY SAME AS PHASE 14E / 14F
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
# Fixed global TRAIN-only weights from unique Phase14E0 data.
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
    / "phase14f1"
)

ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "detection"
    / "phase14f1"
)

COMPARISON_SUMMARY_FILE = (
    RESULT_ROOT
    / "phase14f1_comparison_summary.csv"
)

ALL_ROUND_HISTORY_FILE = (
    RESULT_ROOT
    / "phase14f1_all_round_history.csv"
)

ALL_CLIENT_ROUND_FILE = (
    RESULT_ROOT
    / "phase14f1_all_client_round_metrics.csv"
)

CLASS_WEIGHTS_FILE = (
    RESULT_ROOT
    / "phase14f1_class_weights.csv"
)

CONFIG_FILE = (
    ARTIFACT_ROOT
    / "phase14f1_study_config.json"
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
    "phase14f1_cnn_bilstm"
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


def clone_state_dict(
    state_dict,
):
    return OrderedDict(
        (
            key,
            value.clone(),
        )
        for key, value
        in state_dict.items()
    )


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
            f"\n{dataset_name}: expected shape "
            f"(*, {EXPECTED_SEQUENCE_LENGTH}, "
            f"{EXPECTED_FEATURES}), found {X.shape}."
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
        # (B, 20, 70) -> (B, 70, 20)
        x = x.transpose(1, 2)

        x = self.relu(
            self.conv1(x)
        )

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
            f1 * support
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
    # Client-local diagnostic only.
    #
    # We evaluate only classes that are genuinely present in
    # y_true so a 2-class Non-IID client is not artificially
    # penalized by four absent classes.
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

    balanced_accuracy = float(
        np.mean(recall)
    )

    macro_f1 = float(
        np.mean(f1)
    )

    macro_precision = float(
        np.mean(precision)
    )

    return {
        "present_labels":
            present_labels.tolist(),

        "present_class_balanced_accuracy":
            balanced_accuracy,

        "present_class_macro_precision":
            macro_precision,

        "present_class_macro_f1":
            macro_f1,
    }


# ============================================================
# LOCAL TRAINING
# ============================================================

def train_local(
    global_model: nn.Module,
    X_client: np.ndarray,
    y_client: np.ndarray,
    class_weights: np.ndarray,
    device: torch.device,
    round_number: int,
    client_id: int,
    mu: float,
):
    local_model = CNNBiLSTM(
        feature_count=EXPECTED_FEATURES,
        num_classes=NUM_CLASSES,
    ).to(device)

    local_model.load_state_dict(
        global_model.state_dict()
    )

    # Frozen round-start global reference for FedProx.
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

    # IMPORTANT:
    # Same round/client shuffle seed for every mu experiment.
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

            if mu > 0.0:
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
                    mu
                    *
                    proximal_sum
                )

            else:
                prox_loss = torch.zeros(
                    (),
                    dtype=ce_loss.dtype,
                    device=device,
                )

            objective_loss = (
                ce_loss
                +
                prox_loss
            )

            objective_loss.backward()

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
                    objective_loss.item()
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

    present_metrics = (
        calculate_present_class_metrics(
            y_true,
            y_pred,
        )
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
# AGGREGATION
# ============================================================

def weighted_federated_average(
    client_states,
    client_sequence_counts,
):
    if not client_states:
        raise ValueError(
            "No client states supplied."
        )

    if len(client_states) != len(
        client_sequence_counts
    ):
        raise ValueError(
            "Client state/count length mismatch."
        )

    weights = np.asarray(
        client_sequence_counts,
        dtype=np.float64,
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

        averaged_tensor = torch.zeros_like(
            reference_tensor
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
            y_batch.cpu().numpy()
        )

        all_pred.append(
            predictions.cpu().numpy()
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
# ONE COMPLETE FEDERATED EXPERIMENT
# ============================================================

def run_experiment(
    experiment,
    common_initial_state,
    client_files,
    client_sequence_counts,
    class_weights,
    class_names,
    validation_loader,
    validation_criterion,
    device,
):
    experiment_id = experiment[
        "experiment_id"
    ]

    aggregation = experiment[
        "aggregation"
    ]

    mu = float(
        experiment[
            "mu"
        ]
    )

    experiment_result_root = (
        RESULT_ROOT
        /
        experiment_id
    )

    experiment_artifact_root = (
        ARTIFACT_ROOT
        /
        experiment_id
    )

    experiment_result_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    experiment_artifact_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_model_file = (
        experiment_artifact_root
        /
        "best_model.pt"
    )

    final_model_file = (
        experiment_artifact_root
        /
        "final_round_model.pt"
    )

    round_history_file = (
        experiment_result_root
        /
        "round_history.csv"
    )

    client_round_file = (
        experiment_result_root
        /
        "client_round_metrics.csv"
    )

    best_per_class_file = (
        experiment_result_root
        /
        "best_per_class_metrics.csv"
    )

    final_per_class_file = (
        experiment_result_root
        /
        "final_per_class_metrics.csv"
    )

    best_cm_file = (
        experiment_result_root
        /
        "best_confusion_matrix.csv"
    )

    final_cm_file = (
        experiment_result_root
        /
        "final_confusion_matrix.csv"
    )

    metrics_file = (
        experiment_result_root
        /
        "experiment_summary.json"
    )

    # --------------------------------------------------------
    # FAIRNESS:
    # Every experiment gets exactly the same initial parameters.
    # --------------------------------------------------------

    global_model = CNNBiLSTM(
        feature_count=EXPECTED_FEATURES,
        num_classes=NUM_CLASSES,
    ).to(device)

    global_model.load_state_dict(
        common_initial_state
    )

    separator()

    print(
        f"EXPERIMENT: {experiment_id}"
    )

    print(
        "-" * 122
    )

    print(
        f"Aggregation               : {aggregation}"
    )

    print(
        f"FedProx mu                : {mu}"
    )

    print(
        f"Rounds                    : {FEDERATED_ROUNDS}"
    )

    print(
        "Initialization            : exact same common random state"
    )

    round_rows = []
    client_rows = []

    best_round = 0
    best_macro_f1 = -np.inf
    best_validation_loss = None
    best_metrics = None
    best_y_true = None
    best_y_pred = None
    best_state = None

    last_validation_loss = None
    last_metrics = None
    last_y_true = None
    last_y_pred = None

    experiment_start = (
        time.perf_counter()
    )

    for round_number in range(
        1,
        FEDERATED_ROUNDS + 1,
    ):
        round_start = (
            time.perf_counter()
        )

        client_states = []
        aggregation_counts = []

        print()
        print(
            f"{experiment_id} | "
            f"ROUND {round_number:02d}/"
            f"{FEDERATED_ROUNDS}"
        )

        print(
            "-" * 122
        )

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
                f"{experiment_id} client_{client_id}",
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
                    f"\nclient_{client_id} sequence count changed."
                )

            (
                local_state,
                local_result,
            ) = train_local(
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

                mu=
                    mu,
            )

            client_states.append(
                local_state
            )

            aggregation_counts.append(
                actual_count
            )

            aggregation_weight = (
                actual_count
                /
                sum(
                    client_sequence_counts.values()
                )
            )

            client_rows.append(
                {
                    "experiment_id":
                        experiment_id,

                    "aggregation":
                        aggregation,

                    "mu":
                        mu,

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
                        aggregation_weight,

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

            torch.save(
                {
                    "phase":
                        "14F1",

                    "experiment_id":
                        experiment_id,

                    "aggregation":
                        aggregation,

                    "mu":
                        mu,

                    "round":
                        round_number,

                    "seed":
                        SEED,

                    "model_state_dict":
                        best_state,

                    "validation_macro_f1":
                        best_macro_f1,

                    "validation_balanced_accuracy":
                        validation_metrics[
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
                best_model_file,
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

        round_rows.append(
            {
                "experiment_id":
                    experiment_id,

                "aggregation":
                    aggregation,

                "mu":
                    mu,

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

                "is_best_so_far":
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
            f"Best round / MacroF1     : "
            f"{best_round} / "
            f"{best_macro_f1:.6f}"
        )

        pd.DataFrame(
            round_rows
        ).to_csv(
            round_history_file,
            index=False,
        )

        pd.DataFrame(
            client_rows
        ).to_csv(
            client_round_file,
            index=False,
        )

    experiment_seconds = (
        time.perf_counter()
        -
        experiment_start
    )

    if best_state is None:
        raise RuntimeError(
            f"\n{experiment_id}: no best checkpoint."
        )

    final_state = (
        clone_state_dict_to_cpu(
            global_model
        )
    )

    torch.save(
        {
            "phase":
                "14F1",

            "experiment_id":
                experiment_id,

            "aggregation":
                aggregation,

            "mu":
                mu,

            "round":
                FEDERATED_ROUNDS,

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
        final_model_file,
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
        best_per_class_file,
        index=False,
    )

    final_per_class.to_csv(
        final_per_class_file,
        index=False,
    )

    save_confusion_matrix(
        best_y_true,
        best_y_pred,
        class_names,
        best_cm_file,
    )

    save_confusion_matrix(
        last_y_true,
        last_y_pred,
        class_names,
        final_cm_file,
    )

    best_recalls = (
        best_metrics[
            "per_class_recall"
        ]
    )

    final_recalls = (
        last_metrics[
            "per_class_recall"
        ]
    )

    summary = {
        "experiment_id":
            experiment_id,

        "aggregation":
            aggregation,

        "mu":
            mu,

        "rounds":
            FEDERATED_ROUNDS,

        "local_epochs":
            LOCAL_EPOCHS,

        "best_round":
            best_round,

        "best_accuracy":
            best_metrics[
                "accuracy"
            ],

        "best_balanced_accuracy":
            best_metrics[
                "balanced_accuracy"
            ],

        "best_macro_precision":
            best_metrics[
                "macro_precision"
            ],

        "best_macro_recall":
            best_metrics[
                "macro_recall"
            ],

        "best_macro_f1":
            best_metrics[
                "macro_f1"
            ],

        "best_weighted_f1":
            best_metrics[
                "weighted_f1"
            ],

        "best_recall_BENIGN":
            float(
                best_recalls[0]
            ),

        "best_recall_DDoS":
            float(
                best_recalls[1]
            ),

        "best_recall_DoS_GoldenEye":
            float(
                best_recalls[2]
            ),

        "best_recall_DoS_Hulk":
            float(
                best_recalls[3]
            ),

        "best_recall_FTP_Patator":
            float(
                best_recalls[4]
            ),

        "best_recall_PortScan":
            float(
                best_recalls[5]
            ),

        "final_accuracy":
            last_metrics[
                "accuracy"
            ],

        "final_balanced_accuracy":
            last_metrics[
                "balanced_accuracy"
            ],

        "final_macro_f1":
            last_metrics[
                "macro_f1"
            ],

        "final_weighted_f1":
            last_metrics[
                "weighted_f1"
            ],

        "final_recall_BENIGN":
            float(
                final_recalls[0]
            ),

        "final_recall_DDoS":
            float(
                final_recalls[1]
            ),

        "final_recall_DoS_GoldenEye":
            float(
                final_recalls[2]
            ),

        "final_recall_DoS_Hulk":
            float(
                final_recalls[3]
            ),

        "final_recall_FTP_Patator":
            float(
                final_recalls[4]
            ),

        "final_recall_PortScan":
            float(
                final_recalls[5]
            ),

        "experiment_seconds":
            float(
                experiment_seconds
            ),

        "best_model_file":
            str(
                best_model_file
            ),

        "final_model_file":
            str(
                final_model_file
            ),

        "locked_test_used":
            False,
    }

    save_json(
        metrics_file,
        summary,
    )

    separator()

    print(
        f"{experiment_id} COMPLETE"
    )

    print(
        "-" * 122
    )

    print(
        f"Best round                : "
        f"{best_round}"
    )

    print(
        f"Best BA                   : "
        f"{summary['best_balanced_accuracy']:.6f}"
    )

    print(
        f"Best Macro F1             : "
        f"{summary['best_macro_f1']:.6f}"
    )

    print(
        "Best recalls              : "
        f"BENIGN={summary['best_recall_BENIGN']:.4f}, "
        f"DDoS={summary['best_recall_DDoS']:.4f}, "
        f"GoldenEye={summary['best_recall_DoS_GoldenEye']:.4f}, "
        f"Hulk={summary['best_recall_DoS_Hulk']:.4f}, "
        f"FTP={summary['best_recall_FTP_Patator']:.4f}, "
        f"PortScan={summary['best_recall_PortScan']:.4f}"
    )

    print(
        f"Experiment time           : "
        f"{experiment_seconds:.2f}s"
    )

    del global_model
    del best_state
    del final_state

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return (
        summary,
        round_rows,
        client_rows,
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
        f"Clients                   : {NUM_CLIENTS}"
    )

    print(
        f"Rounds / experiment       : {FEDERATED_ROUNDS}"
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
        f"Gradient clip             : {GRADIENT_CLIP_NORM}"
    )

    print(
        f"Aggregation weighting     : "
        f"{AGGREGATION_WEIGHTING}"
    )

    print(
        f"Validation                : {VALIDATION_FILE}"
    )

    print(
        "Locked test               : NO"
    )

    print()
    print("CONTROLLED EXPERIMENTS")
    print("-" * 122)

    for experiment in EXPERIMENTS:
        print(
            f"{experiment['experiment_id']:18s} | "
            f"{experiment['aggregation']:7s} | "
            f"mu={experiment['mu']}"
        )

    print()
    print("SCIENTIFIC CONTROLS")
    print("-" * 122)

    print(
        "1. Same frozen Phase14D client sequence data for all experiments."
    )

    print(
        "2. Same 126,982-parameter CNN-BiLSTM architecture."
    )

    print(
        "3. Same seed, batch size, LR, weight decay and local epochs."
    )

    print(
        "4. Same exact common random initial model parameters."
    )

    print(
        "5. Same client shuffle seed in corresponding round/client."
    )

    print(
        "6. Same fixed TRAIN-only global class weights."
    )

    print(
        "7. Only proximal coefficient changes: 0, 0.001, 0.01."
    )

    print(
        "8. All 20 rounds run; no early stopping."
    )

    print(
        "9. Best round selected by validation Macro F1."
    )

    print(
        "10. Locked test is not read."
    )

    print(
        "11. Local diagnostics use PRESENT-CLASS metrics "
        "to avoid misleading six-class F1 on two-class clients."
    )

    print(
        "12. SOURCE-ORDERED; not timestamp-confirmed temporal sequences."
    )

    # ========================================================
    # PREPROCESSING / LABEL MAPPING
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
            f"\nExpected 6 classes, "
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
        str(label):
            int(class_id)
        for label, class_id
        in label_mapping.items()
    }

    # ========================================================
    # GLOBAL TRAIN-ONLY CLASS WEIGHTS
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
    # CLIENT DATA CHECK
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

        actual_count = int(
            len(
                y_client
            )
        )

        client_sequence_counts[
            client_id
        ] = actual_count

        expected_attack = (
            CLIENT_ATTACKS[
                client_id
            ]
        )

        expected_ids = {
            label_to_id[
                "BENIGN"
            ],
            label_to_id[
                expected_attack
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
                f"\nclient_{client_id} contains unexpected classes."
            )

        if label_to_id[
            expected_attack
        ] not in actual_ids:
            raise RuntimeError(
                f"\nclient_{client_id} missing {expected_attack}."
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
            f"n={actual_count:,} | "
            f"{distribution}"
        )

        del y_client

    # ========================================================
    # VALIDATION
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
    # ONE COMMON RANDOM INITIALIZATION FOR ALL 3 EXPERIMENTS
    # ========================================================

    set_seed(
        SEED
    )

    initial_model = CNNBiLSTM(
        feature_count=EXPECTED_FEATURES,
        num_classes=NUM_CLASSES,
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter
        in initial_model.parameters()
    )

    if parameter_count != (
        EXPECTED_PARAMETER_COUNT
    ):
        raise RuntimeError(
            "\nArchitecture drift detected.\n"
            f"Expected {EXPECTED_PARAMETER_COUNT:,}; "
            f"found {parameter_count:,}."
        )

    common_initial_state = (
        clone_state_dict_to_cpu(
            initial_model
        )
    )

    del initial_model

    gc.collect()

    print()
    print(
        f"Common initial parameters : "
        f"{parameter_count:,}"
    )

    print(
        "Common initialization     : frozen once and reused for all experiments"
    )

    # ========================================================
    # SAVE STUDY CONFIG
    # ========================================================

    study_config = {
        "phase":
            "14F1",

        "study":
            "cnn_bilstm_fedavg_fedprox_mu_sensitivity",

        "seed":
            SEED,

        "device":
            str(
                device
            ),

        "experiments":
            EXPERIMENTS,

        "clients":
            NUM_CLIENTS,

        "client_sequence_counts":
            client_sequence_counts,

        "rounds_per_experiment":
            FEDERATED_ROUNDS,

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

        "same_initial_state":
            True,

        "same_client_shuffle_schedule":
            True,

        "class_weight_mode":
            CLASS_WEIGHT_MODE,

        "class_weight_cap":
            CLASS_WEIGHT_CAP,

        "class_weights":
            class_weights.tolist(),

        "validation_file":
            str(
                VALIDATION_FILE
            ),

        "locked_test_path":
            str(
                LOCKED_TEST_FILE
            ),

        "locked_test_used":
            False,

        "parameter_count":
            parameter_count,

        "sequence_length":
            EXPECTED_SEQUENCE_LENGTH,

        "feature_count":
            EXPECTED_FEATURES,

        "scientific_label":
            (
                "SOURCE-ORDERED flow sequences; "
                "NOT timestamp-confirmed temporal sequences."
            ),
    }

    save_json(
        CONFIG_FILE,
        study_config,
    )

    # ========================================================
    # RUN ALL CONTROLLED EXPERIMENTS
    # ========================================================

    all_summaries = []
    all_round_rows = []
    all_client_rows = []

    study_start = (
        time.perf_counter()
    )

    for experiment in EXPERIMENTS:
        # Reset all RNGs to the same study seed before each
        # experiment. Client shuffle seeds are also explicitly
        # fixed per corresponding round/client.
        set_seed(
            SEED
        )

        (
            summary,
            round_rows,
            client_rows,
        ) = run_experiment(
            experiment=
                experiment,

            common_initial_state=
                clone_state_dict(
                    common_initial_state
                ),

            client_files=
                client_files,

            client_sequence_counts=
                client_sequence_counts,

            class_weights=
                class_weights,

            class_names=
                class_names,

            validation_loader=
                validation_loader,

            validation_criterion=
                validation_criterion,

            device=
                device,
        )

        all_summaries.append(
            summary
        )

        all_round_rows.extend(
            round_rows
        )

        all_client_rows.extend(
            client_rows
        )

        pd.DataFrame(
            all_summaries
        ).to_csv(
            COMPARISON_SUMMARY_FILE,
            index=False,
        )

        pd.DataFrame(
            all_round_rows
        ).to_csv(
            ALL_ROUND_HISTORY_FILE,
            index=False,
        )

        pd.DataFrame(
            all_client_rows
        ).to_csv(
            ALL_CLIENT_ROUND_FILE,
            index=False,
        )

    study_seconds = (
        time.perf_counter()
        -
        study_start
    )

    # ========================================================
    # RANK BY MACRO F1, THEN BALANCED ACCURACY
    # ========================================================

    comparison = pd.DataFrame(
        all_summaries
    )

    comparison = (
        comparison
        .sort_values(
            by=[
                "best_macro_f1",
                "best_balanced_accuracy",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    comparison.insert(
        0,
        "rank",
        np.arange(
            1,
            len(comparison) + 1,
        ),
    )

    comparison.to_csv(
        COMPARISON_SUMMARY_FILE,
        index=False,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    separator()

    print(
        "PHASE 14F1 COMPLETE"
    )

    separator()

    display_columns = [
        "rank",
        "experiment_id",
        "aggregation",
        "mu",
        "best_round",
        "best_balanced_accuracy",
        "best_macro_f1",
        "best_recall_DoS_GoldenEye",
        "best_recall_FTP_Patator",
        "best_recall_PortScan",
        "best_recall_DoS_Hulk",
        "best_recall_DDoS",
        "best_recall_BENIGN",
    ]

    print(
        comparison[
            display_columns
        ].to_string(
            index=False
        )
    )

    winner = comparison.iloc[
        0
    ]

    print()
    print("VALIDATION WINNER")
    print("-" * 122)

    print(
        f"Experiment                : "
        f"{winner['experiment_id']}"
    )

    print(
        f"Aggregation               : "
        f"{winner['aggregation']}"
    )

    print(
        f"mu                        : "
        f"{winner['mu']}"
    )

    print(
        f"Best round                : "
        f"{int(winner['best_round'])}"
    )

    print(
        f"Balanced Accuracy         : "
        f"{winner['best_balanced_accuracy']:.6f}"
    )

    print(
        f"Macro F1                  : "
        f"{winner['best_macro_f1']:.6f}"
    )

    print(
        "Critical recalls          : "
        f"GoldenEye="
        f"{winner['best_recall_DoS_GoldenEye']:.4f}, "
        f"FTP="
        f"{winner['best_recall_FTP_Patator']:.4f}, "
        f"PortScan="
        f"{winner['best_recall_PortScan']:.4f}"
    )

    print()
    print(
        f"Total Phase14F1 time      : "
        f"{study_seconds:.2f}s"
    )

    print(
        f"Comparison summary        : "
        f"{COMPARISON_SUMMARY_FILE}"
    )

    print(
        f"All round history         : "
        f"{ALL_ROUND_HISTORY_FILE}"
    )

    print(
        f"All client metrics        : "
        f"{ALL_CLIENT_ROUND_FILE}"
    )

    print(
        f"Study config              : "
        f"{CONFIG_FILE}"
    )

    print()
    print(
        "IMPORTANT: This is validation-only method development."
    )

    print(
        "LOCKED TEST USED : NO"
    )

    print()
    print("SCIENTIFIC LABEL:")

    print(
        "SOURCE-ORDERED flow sequences; "
        "NOT timestamp-confirmed temporal sequences."
    )

    del X_validation
    del y_validation
    del validation_dataset
    del validation_loader
    del common_initial_state

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
