"""
Phase 4 - Federated Learning with FedAvg
========================================

Supports:
    1. IID
    2. Natural Non-IID
    3. Controlled Non-IID

Research configuration:
    Clients      = 5
    Classes      = 6
    Architecture = Same MLP as Phase 3B / Phase 3C
    Aggregation  = Sample-weighted FedAvg

IMPORTANT:
    Locked test set is NEVER used during federated training.
    It is evaluated ONCE after the final round.

Examples:

    python scripts/phase4_fedavg.py --scenario controlled_non_iid

    python scripts/phase4_fedavg.py --scenario non_iid

    python scripts/phase4_fedavg.py --scenario iid
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

N_CLIENTS = 5

LABEL_COL = "Label"

BENIGN_LABEL = "BENIGN"

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# DATA PATHS
# ============================================================

FEDERATED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "federated"
)

GLOBAL_TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train.csv"
)

LOCKED_TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_locked.csv"
)


# ============================================================
# PREPROCESSING
# ============================================================

PREPROCESSING_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "preprocessing"
)


# ============================================================
# OUTPUT
# ============================================================

ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "fedavg"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "fedavg"
)


# ============================================================
# DEFAULT TRAINING PARAMETERS
# ============================================================

DEFAULT_ROUNDS = 10

DEFAULT_LOCAL_EPOCHS = 1

DEFAULT_BATCH_SIZE = 2048

DEFAULT_CHUNK_SIZE = 100_000

DEFAULT_LEARNING_RATE = 0.001

DEFAULT_WEIGHT_DECAY = 0.0001


# ============================================================
# NON-IID RESEARCH DESIGN
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
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("phase4_fedavg")


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    if hasattr(
        torch.backends,
        "cudnn"
    ):

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False


# ============================================================
# LOAD PREPROCESSING
# ============================================================

def load_preprocessing():

    scaler_file = (
        PREPROCESSING_DIR
        / "scaler.joblib"
    )

    features_file = (
        PREPROCESSING_DIR
        / "feature_columns.json"
    )

    labels_file = (
        PREPROCESSING_DIR
        / "label_mapping.json"
    )

    for file in [
        scaler_file,
        features_file,
        labels_file
    ]:

        if not file.exists():

            raise FileNotFoundError(
                f"Missing preprocessing artifact: {file}"
            )

    scaler = joblib.load(
        scaler_file
    )

    with open(
        features_file,
        "r",
        encoding="utf-8"
    ) as f:

        feature_columns = json.load(f)

    with open(
        labels_file,
        "r",
        encoding="utf-8"
    ) as f:

        label_mapping = json.load(f)

    variances = np.asarray(
        scaler.var_
    )

    active_indices = np.where(
        variances > 1e-12
    )[0]

    active_features = [

        feature_columns[i]

        for i in active_indices
    ]

    logger.info(
        "Original features : %d",
        len(feature_columns)
    )

    logger.info(
        "Active features   : %d",
        len(active_features)
    )

    logger.info(
        "Classes           : %d",
        len(label_mapping)
    )

    return (
        scaler,
        feature_columns,
        label_mapping,
        active_indices,
        active_features
    )


# ============================================================
# PREPROCESS FEATURES
# ============================================================

def transform_features(
    df,
    feature_columns,
    scaler,
    active_indices
):

    df.columns = [

        str(c).strip()

        for c in df.columns
    ]

    X = df[
        feature_columns
    ].copy()

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(
        0.0
    )

    X = scaler.transform(
        X
    )

    X = X[
        :,
        active_indices
    ]

    return X.astype(
        np.float32,
        copy=False
    )


# ============================================================
# LABEL ENCODING
# ============================================================

def encode_labels(
    labels,
    label_mapping
):

    labels = (
        labels
        .astype(str)
        .str.strip()
    )

    unknown = (

        set(
            labels.unique()
        )

        - set(
            label_mapping.keys()
        )
    )

    if unknown:

        raise ValueError(
            f"Unknown labels detected: {unknown}"
        )

    return (
        labels
        .map(label_mapping)
        .to_numpy(
            dtype=np.int64
        )
    )


# ============================================================
# MODEL
# ============================================================

class IDSMLP(nn.Module):

    """
    Same architecture as Phase 3B and Phase 3C.
    """

    def __init__(
        self,
        input_dim,
        num_classes
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                256
            ),

            nn.LayerNorm(
                256
            ),

            nn.ReLU(),

            nn.Dropout(
                0.20
            ),

            nn.Linear(
                256,
                128
            ),

            nn.LayerNorm(
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                0.20
            ),

            nn.Linear(
                128,
                64
            ),

            nn.LayerNorm(
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.10
            ),

            nn.Linear(
                64,
                num_classes
            )
        )

    def forward(
        self,
        x
    ):

        return self.network(
            x
        )


# ============================================================
# GLOBAL CLASS COUNTS
# ============================================================

def calculate_global_class_counts(
    label_mapping,
    chunk_size
):

    logger.info(
        "Calculating global class counts..."
    )

    counts = np.zeros(
        len(label_mapping),
        dtype=np.int64
    )

    for chunk in pd.read_csv(
        GLOBAL_TRAIN_FILE,
        usecols=[LABEL_COL],
        chunksize=chunk_size
    ):

        y = encode_labels(
            chunk[LABEL_COL],
            label_mapping
        )

        counts += np.bincount(
            y,
            minlength=len(label_mapping)
        )

    return counts


# ============================================================
# GLOBAL CLASS WEIGHTS
# ============================================================

def calculate_global_class_weights(
    counts
):

    """
    Same moderated inverse-frequency strategy as Phase 3B.

    weight =
        sqrt(max_count / class_count)

    capped at 20
    then normalized to mean 1.
    """

    safe_counts = np.maximum(
        counts.astype(
            np.float64
        ),
        1.0
    )

    maximum = (
        safe_counts.max()
    )

    weights = np.sqrt(
        maximum
        / safe_counts
    )

    weights = np.minimum(
        weights,
        20.0
    )

    weights = (
        weights
        / weights.mean()
    )

    return weights.astype(
        np.float32
    )


# ============================================================
# COUNT CLIENT DATA
# ============================================================

def count_client_samples(
    client_file,
    chunk_size
):

    count = 0

    for chunk in pd.read_csv(
        client_file,
        usecols=[LABEL_COL],
        chunksize=chunk_size
    ):

        count += len(
            chunk
        )

    return count


# ============================================================
# LOCAL TRAINING
# ============================================================

def train_client(
    global_state,
    client_file,
    scaler,
    feature_columns,
    active_indices,
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
    round_number
):

    """
    Simulates one FL client.

    Client receives global parameters,
    trains only on local data,
    returns updated parameters.
    """

    # --------------------------------------------------------
    # Create local model
    # --------------------------------------------------------

    local_model = IDSMLP(

        input_dim=
            input_dim,

        num_classes=
            len(label_mapping)

    ).to(device)

    local_model.load_state_dict(
        global_state
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    weight_tensor = torch.tensor(

        class_weights,

        dtype=torch.float32,

        device=device
    )

    criterion = nn.CrossEntropyLoss(
        weight=
            weight_tensor
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(

        local_model.parameters(),

        lr=
            learning_rate,

        weight_decay=
            weight_decay
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    local_model.train()

    total_loss = 0.0

    total_samples = 0

    total_correct = 0

    training_start = (
        time.perf_counter()
    )

    for local_epoch in range(
        1,
        local_epochs + 1
    ):

        rng = np.random.default_rng(

            SEED
            + round_number * 1000
            + client_id * 100
            + local_epoch
        )

        for chunk in pd.read_csv(
            client_file,
            chunksize=chunk_size
        ):

            X = transform_features(

                chunk,
                feature_columns,
                scaler,
                active_indices
            )

            y = encode_labels(
                chunk[LABEL_COL],
                label_mapping
            )

            indices = rng.permutation(
                len(X)
            )

            X = X[
                indices
            ]

            y = y[
                indices
            ]

            for start in range(
                0,
                len(X),
                batch_size
            ):

                end = min(
                    start + batch_size,
                    len(X)
                )

                xb = torch.from_numpy(
                    X[start:end]
                ).to(device)

                yb = torch.from_numpy(
                    y[start:end]
                ).to(device)

                optimizer.zero_grad(
                    set_to_none=True
                )

                logits = local_model(
                    xb
                )

                loss = criterion(
                    logits,
                    yb
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(

                    local_model.parameters(),

                    max_norm=5.0
                )

                optimizer.step()

                n = len(
                    yb
                )

                total_loss += (
                    loss.item()
                    * n
                )

                predictions = (
                    logits.argmax(
                        dim=1
                    )
                )

                total_correct += (

                    predictions
                    .eq(yb)
                    .sum()
                    .item()
                )

                total_samples += n

    training_seconds = (

        time.perf_counter()

        - training_start
    )

    mean_loss = (
        total_loss
        / max(
            total_samples,
            1
        )
    )

    accuracy = (
        total_correct
        / max(
            total_samples,
            1
        )
    )

    # --------------------------------------------------------
    # Send only model parameters back to server
    # --------------------------------------------------------

    state = {

        key:
            value
            .detach()
            .cpu()
            .clone()

        for key, value
        in local_model.state_dict().items()
    }

    return (
        state,
        mean_loss,
        accuracy,
        training_seconds
    )


# ============================================================
# FEDAVG
# ============================================================

def federated_average(
    client_states,
    client_weights
):

    """
    Standard sample-weighted FedAvg.

        w_global =
            sum(n_k * w_k)
            ----------------
            sum(n_k)
    """

    total_samples = float(
        sum(
            client_weights
        )
    )

    global_state = {}

    keys = (
        client_states[0]
        .keys()
    )

    for key in keys:

        aggregated = torch.zeros_like(

            client_states[0][key],

            dtype=torch.float32
        )

        for state, samples in zip(
            client_states,
            client_weights
        ):

            aggregated += (

                state[key]
                .float()

                * (
                    samples
                    / total_samples
                )
            )

        global_state[
            key
        ] = aggregated

    return global_state


# ============================================================
# MODEL UPDATE MAGNITUDE
# ============================================================

def state_update_norm(
    old_state,
    new_state
):

    total = 0.0

    for key in old_state.keys():

        old = (
            old_state[key]
            .float()
            .cpu()
        )

        new = (
            new_state[key]
            .float()
            .cpu()
        )

        difference = (
            new
            - old
        )

        total += float(
            torch.sum(
                difference ** 2
            ).item()
        )

    return float(
        np.sqrt(
            total
        )
    )


# ============================================================
# FINAL LOCKED TEST EVALUATION
# ============================================================

def evaluate_global_model(
    model,
    scaler,
    feature_columns,
    active_indices,
    label_mapping,
    device,
    batch_size,
    chunk_size
):

    logger.info(
        "Evaluating final global model on LOCKED TEST..."
    )

    model.eval()

    true_values = []

    predictions = []

    start_time = (
        time.perf_counter()
    )

    with torch.no_grad():

        for chunk in pd.read_csv(
            LOCKED_TEST_FILE,
            chunksize=chunk_size
        ):

            X = transform_features(

                chunk,
                feature_columns,
                scaler,
                active_indices
            )

            y = encode_labels(
                chunk[LABEL_COL],
                label_mapping
            )

            chunk_predictions = []

            for start in range(
                0,
                len(X),
                batch_size
            ):

                end = min(
                    start + batch_size,
                    len(X)
                )

                xb = torch.from_numpy(
                    X[start:end]
                ).to(device)

                logits = model(
                    xb
                )

                pred = (
                    logits
                    .argmax(
                        dim=1
                    )
                    .cpu()
                    .numpy()
                )

                chunk_predictions.append(
                    pred
                )

            true_values.append(
                y
            )

            predictions.append(

                np.concatenate(
                    chunk_predictions
                )
            )

    inference_seconds = (

        time.perf_counter()

        - start_time
    )

    y_true = np.concatenate(
        true_values
    )

    y_pred = np.concatenate(
        predictions
    )

    return (
        y_true,
        y_pred,
        inference_seconds
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    label_mapping
):

    inverse_mapping = {

        value: key

        for key, value
        in label_mapping.items()
    }

    class_ids = list(
        range(
            len(label_mapping)
        )
    )

    class_names = [

        inverse_mapping[i]

        for i in class_ids
    ]

    overall = {

        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y_true,
                    y_pred
                )
            ),

        "macro_precision":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0
                )
            ),

        "macro_recall":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0
                )
            ),

        "weighted_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0
                )
            ),

        "test_samples":
            int(
                len(
                    y_true
                )
            )
    }

    report = classification_report(

        y_true,
        y_pred,

        labels=
            class_ids,

        target_names=
            class_names,

        output_dict=True,

        zero_division=0
    )

    cm = confusion_matrix(

        y_true,
        y_pred,

        labels=
            class_ids
    )

    return (
        overall,
        report,
        cm,
        class_names
    )


# ============================================================
# KNOWLEDGE TRANSFER ANALYSIS
# ============================================================

def create_knowledge_transfer_report(
    scenario,
    report,
    class_names,
    result_dir
):

    if scenario == "iid":

        return None

    local_summary_file = (

        PROJECT_ROOT
        / "results"
        / "local_baselines"
        / scenario
        / "local_client_summary.csv"
    )

    if not local_summary_file.exists():

        logger.warning(
            "Local baseline summary missing. "
            "Knowledge-transfer table skipped."
        )

        return None

    local_summary = pd.read_csv(
        local_summary_file
    )

    rows = []

    attack_classes = [

        name

        for name in class_names

        if name != BENIGN_LABEL
    ]

    for client_id in range(
        1,
        N_CLIENTS + 1
    ):

        local_attack = (
            CLIENT_ATTACKS[
                client_id
            ]
        )

        unseen_attacks = [

            attack

            for attack
            in attack_classes

            if attack != local_attack
        ]

        federated_unseen_recall = float(

            np.mean(
                [
                    report[
                        attack
                    ]["recall"]

                    for attack
                    in unseen_attacks
                ]
            )
        )

        local_row = local_summary[

            local_summary[
                "client"
            ]
            == f"client_{client_id}"

        ]

        if len(
            local_row
        ) == 0:

            local_unseen_recall = 0.0

        else:

            local_unseen_recall = float(

                local_row.iloc[0][
                    "mean_unseen_attack_recall"
                ]
            )

        gain = (

            federated_unseen_recall

            - local_unseen_recall
        )

        rows.append({

            "client":
                f"client_{client_id}",

            "local_seen_attack":
                local_attack,

            "local_unseen_attack_recall":
                local_unseen_recall,

            "federated_unseen_attack_recall":
                federated_unseen_recall,

            "knowledge_transfer_gain":
                gain
        })

    transfer_df = pd.DataFrame(
        rows
    )

    transfer_file = (

        result_dir
        / "knowledge_transfer.csv"
    )

    transfer_df.to_csv(
        transfer_file,
        index=False
    )

    return transfer_df


# ============================================================
# GRAPHS
# ============================================================

def generate_graphs(
    scenario,
    history_df,
    overall,
    report,
    cm,
    class_names,
    transfer_df,
    result_dir
):

    graph_dir = (
        result_dir
        / "graphs"
    )

    graph_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # GRAPH 1 - FEDAVG TRAINING LOSS
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(

        history_df[
            "round"
        ],

        history_df[
            "weighted_local_loss"
        ],

        marker="o"
    )

    plt.title(
        f"FedAvg Training Convergence - {scenario}"
    )

    plt.xlabel(
        "Federated Round"
    )

    plt.ylabel(
        "Weighted Local Training Loss"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(

        graph_dir
        / "01_fedavg_training_loss.png",

        dpi=300
    )

    plt.close()


    # --------------------------------------------------------
    # GRAPH 2 - GLOBAL UPDATE NORM
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(

        history_df[
            "round"
        ],

        history_df[
            "global_update_norm"
        ],

        marker="o"
    )

    plt.title(
        f"FedAvg Global Model Update Magnitude - {scenario}"
    )

    plt.xlabel(
        "Federated Round"
    )

    plt.ylabel(
        "Parameter Update Norm"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(

        graph_dir
        / "02_global_update_norm.png",

        dpi=300
    )

    plt.close()


    # --------------------------------------------------------
    # GRAPH 3 - PER CLASS PERFORMANCE
    # --------------------------------------------------------

    precision = [

        report[
            name
        ]["precision"]
        * 100

        for name in class_names
    ]

    recall = [

        report[
            name
        ]["recall"]
        * 100

        for name in class_names
    ]

    f1 = [

        report[
            name
        ]["f1-score"]
        * 100

        for name in class_names
    ]

    x = np.arange(
        len(
            class_names
        )
    )

    width = 0.25

    plt.figure(
        figsize=(14, 7)
    )

    plt.bar(
        x - width,
        precision,
        width,
        label="Precision"
    )

    plt.bar(
        x,
        recall,
        width,
        label="Recall"
    )

    plt.bar(
        x + width,
        f1,
        width,
        label="F1"
    )

    plt.xticks(
        x,
        class_names,
        rotation=35,
        ha="right"
    )

    plt.ylim(
        0,
        105
    )

    plt.ylabel(
        "Score (%)"
    )

    plt.xlabel(
        "Class"
    )

    plt.title(
        f"FedAvg Per-Class Performance - {scenario}"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(

        graph_dir
        / "03_per_class_metrics.png",

        dpi=300
    )

    plt.close()


    # --------------------------------------------------------
    # GRAPH 4 - CONFUSION MATRIX
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 9)
    )

    image = plt.imshow(
        cm,
        aspect="auto"
    )

    plt.colorbar(
        image
    )

    plt.xticks(
        range(
            len(
                class_names
            )
        ),
        class_names,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(
            len(
                class_names
            )
        ),
        class_names
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "True Class"
    )

    plt.title(
        f"FedAvg Confusion Matrix - {scenario}"
    )

    max_value = (
        cm.max()
        if cm.size
        else 1
    )

    threshold = (
        max_value
        * 0.50
    )

    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            value = cm[
                i,
                j
            ]

            if value > 0:

                plt.text(

                    j,
                    i,

                    f"{int(value):,}",

                    ha="center",
                    va="center",

                    fontsize=8,

                    color=(
                        "white"
                        if value > threshold
                        else "black"
                    )
                )

    plt.tight_layout()

    plt.savefig(

        graph_dir
        / "04_confusion_matrix.png",

        dpi=300
    )

    plt.close()


    # --------------------------------------------------------
    # GRAPH 5 - KNOWLEDGE TRANSFER
    # --------------------------------------------------------

    if transfer_df is not None:

        labels = (
            transfer_df[
                "client"
            ]
            .tolist()
        )

        local_values = (

            transfer_df[
                "local_unseen_attack_recall"
            ]

            * 100
        )

        fed_values = (

            transfer_df[
                "federated_unseen_attack_recall"
            ]

            * 100
        )

        x = np.arange(
            len(
                labels
            )
        )

        width = 0.35

        plt.figure(
            figsize=(11, 6)
        )

        plt.bar(

            x - width / 2,

            local_values,

            width,

            label=
                "Local-only unseen recall"
        )

        plt.bar(

            x + width / 2,

            fed_values,

            width,

            label=
                "FedAvg unseen recall"
        )

        plt.xticks(
            x,
            labels
        )

        plt.ylim(
            0,
            105
        )

        plt.xlabel(
            "Client"
        )

        plt.ylabel(
            "Mean Unseen-Attack Recall (%)"
        )

        plt.title(
            f"Knowledge Transfer Through FedAvg - {scenario}"
        )

        plt.legend()

        plt.grid(
            axis="y",
            alpha=0.25
        )

        plt.tight_layout()

        plt.savefig(

            graph_dir
            / "05_knowledge_transfer.png",

            dpi=300
        )

        plt.close()


    # --------------------------------------------------------
    # GRAPH 6 - CENTRALIZED VS FEDAVG
    # --------------------------------------------------------

    centralized_file = (

        PROJECT_ROOT
        / "results"
        / "centralized_baseline"
        / "overall_metrics.json"
    )

    if centralized_file.exists():

        with open(
            centralized_file,
            "r",
            encoding="utf-8"
        ) as f:

            centralized = json.load(
                f
            )

        metrics = [

            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1"
        ]

        labels = [

            "Accuracy",
            "Balanced Accuracy",
            "Macro F1",
            "Weighted F1"
        ]

        centralized_values = [

            centralized[
                metric
            ] * 100

            for metric in metrics
        ]

        fedavg_values = [

            overall[
                metric
            ] * 100

            for metric in metrics
        ]

        x = np.arange(
            len(
                labels
            )
        )

        width = 0.35

        plt.figure(
            figsize=(11, 6)
        )

        plt.bar(

            x - width / 2,

            centralized_values,

            width,

            label=
                "Centralized MLP"
        )

        plt.bar(

            x + width / 2,

            fedavg_values,

            width,

            label=
                f"FedAvg-{scenario}"
        )

        plt.xticks(
            x,
            labels
        )

        plt.ylim(
            0,
            105
        )

        plt.ylabel(
            "Score (%)"
        )

        plt.title(
            "Centralized vs Federated IDS Performance"
        )

        plt.legend()

        plt.grid(
            axis="y",
            alpha=0.25
        )

        plt.tight_layout()

        plt.savefig(

            graph_dir
            / "06_centralized_vs_fedavg.png",

            dpi=300
        )

        plt.close()


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
    label_mapping,
    args,
    client_sample_counts,
    total_training_seconds,
    inference_seconds
):

    artifact_dir = (
        ARTIFACT_ROOT
        / scenario
    )

    result_dir = (
        RESULT_ROOT
        / scenario
    )

    artifact_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # GLOBAL MODEL
    # --------------------------------------------------------

    torch.save(

        {
            "model_state_dict":
                global_model.state_dict(),

            "input_dim":
                len(active_features),

            "num_classes":
                len(label_mapping),

            "active_features":
                active_features,

            "label_mapping":
                label_mapping,

            "aggregation":
                "FedAvg"
        },

        artifact_dir
        / "fedavg_global_model.pt"
    )

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(

        result_dir
        / "fedavg_round_history.csv",

        index=False
    )

    # --------------------------------------------------------
    # OVERALL METRICS
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

    with open(

        result_dir
        / "overall_metrics.json",

        "w",
        encoding="utf-8"

    ) as f:

        json.dump(
            overall,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # PER CLASS
    # --------------------------------------------------------

    rows = []

    for name in class_names:

        metrics = report[
            name
        ]

        rows.append({

            "class":
                name,

            "precision":
                metrics[
                    "precision"
                ],

            "recall":
                metrics[
                    "recall"
                ],

            "f1_score":
                metrics[
                    "f1-score"
                ],

            "support":
                metrics[
                    "support"
                ]
        })

    pd.DataFrame(
        rows
    ).to_csv(

        result_dir
        / "per_class_metrics.csv",

        index=False
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    pd.DataFrame(

        cm,

        index=
            class_names,

        columns=
            class_names

    ).to_csv(

        result_dir
        / "confusion_matrix.csv"
    )

    # --------------------------------------------------------
    # MANIFEST
    # --------------------------------------------------------

    manifest = {

        "phase":
            4,

        "algorithm":
            "FedAvg",

        "scenario":
            scenario,

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

        "architecture":
            [
                len(active_features),
                256,
                128,
                64,
                len(label_mapping)
            ],

        "client_sample_counts":
            client_sample_counts,

        "total_federated_training_seconds":
            total_training_seconds,

        "final_inference_seconds":
            inference_seconds,

        "locked_test_used_during_training":
            False,

        "locked_test_used_for":
            "final evaluation only",

        "final_metrics":
            overall
    }

    with open(

        result_dir
        / "phase4_manifest.json",

        "w",
        encoding="utf-8"

    ) as f:

        json.dump(
            manifest,
            f,
            indent=4
        )

    return (
        history_df,
        result_dir
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
            "controlled_non_iid"
        ],

        default=
            "controlled_non_iid"
    )

    parser.add_argument(

        "--rounds",

        type=int,

        default=
            DEFAULT_ROUNDS
    )

    parser.add_argument(

        "--local-epochs",

        type=int,

        default=
            DEFAULT_LOCAL_EPOCHS
    )

    parser.add_argument(

        "--batch-size",

        type=int,

        default=
            DEFAULT_BATCH_SIZE
    )

    parser.add_argument(

        "--chunk-size",

        type=int,

        default=
            DEFAULT_CHUNK_SIZE
    )

    parser.add_argument(

        "--learning-rate",

        type=float,

        default=
            DEFAULT_LEARNING_RATE
    )

    parser.add_argument(

        "--weight-decay",

        type=float,

        default=
            DEFAULT_WEIGHT_DECAY
    )

    args = parser.parse_args()


    # ========================================================
    # INITIALIZATION
    # ========================================================

    set_seed(
        SEED
    )

    scenario_dir = (

        FEDERATED_ROOT
        / args.scenario
    )

    if not scenario_dir.exists():

        raise FileNotFoundError(
            f"Missing scenario: {scenario_dir}"
        )

    if not LOCKED_TEST_FILE.exists():

        raise FileNotFoundError(
            f"Locked test missing: {LOCKED_TEST_FILE}"
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
        device
    )


    # ========================================================
    # PREPROCESSING
    # ========================================================

    (
        scaler,
        feature_columns,
        label_mapping,
        active_indices,
        active_features

    ) = load_preprocessing()


    # ========================================================
    # GLOBAL CLASS WEIGHTS
    # ========================================================

    global_counts = (
        calculate_global_class_counts(
            label_mapping,
            args.chunk_size
        )
    )

    global_weights = (
        calculate_global_class_weights(
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
            ]
        )


    # ========================================================
    # CLIENT FILES + SAMPLE COUNTS
    # ========================================================

    client_files = {}

    client_sample_counts = {}

    for client_id in range(
        1,
        N_CLIENTS + 1
    ):

        client_file = (

            scenario_dir
            / f"client_{client_id}.csv"
        )

        if not client_file.exists():

            raise FileNotFoundError(
                client_file
            )

        client_files[
            client_id
        ] = client_file

        samples = count_client_samples(

            client_file,

            args.chunk_size
        )

        client_sample_counts[
            f"client_{client_id}"
        ] = samples

        logger.info(

            "Client %d samples = %d",

            client_id,
            samples
        )


    # ========================================================
    # INITIAL GLOBAL MODEL
    # ========================================================

    set_seed(
        SEED
    )

    global_model = IDSMLP(

        input_dim=
            len(active_features),

        num_classes=
            len(label_mapping)

    ).to(device)

    parameter_count = sum(

        p.numel()

        for p
        in global_model.parameters()
    )

    logger.info(
        "Global model parameters: %d",
        parameter_count
    )


    # ========================================================
    # FEDERATED TRAINING
    # ========================================================

    history = []

    total_training_start = (
        time.perf_counter()
    )

    print()
    print("=" * 90)
    print("PHASE 4 - FEDAVG")
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
        f"Features      : {len(active_features)}"
    )

    print(
        f"Classes       : {len(label_mapping)}"
    )

    print()


    # ========================================================
    # FEDAVG ROUNDS
    # ========================================================

    for round_number in range(
        1,
        args.rounds + 1
    ):

        print()
        print("-" * 90)

        print(
            f"FEDERATED ROUND "
            f"{round_number}/{args.rounds}"
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
        # LOCAL CLIENT TRAINING
        # ====================================================

        for client_id in range(
            1,
            N_CLIENTS + 1
        ):

            logger.info(

                "Round %d | "
                "training Client %d...",

                round_number,
                client_id
            )

            (
                local_state,
                local_loss,
                local_accuracy,
                training_seconds

            ) = train_client(

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

                active_indices=
                    active_indices,

                label_mapping=
                    label_mapping,

                class_weights=
                    global_weights,

                input_dim=
                    len(active_features),

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
                    round_number
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
                training_seconds
            )


        # ====================================================
        # SERVER FEDAVG
        # ====================================================

        aggregated_state = (
            federated_average(

                client_states,
                aggregation_weights
            )
        )

        global_model.load_state_dict(
            aggregated_state
        )

        update_norm = (
            state_update_norm(

                old_global_state,
                aggregated_state
            )
        )


        # ====================================================
        # WEIGHTED TRAINING STATISTICS
        # ====================================================

        weight_array = np.asarray(

            aggregation_weights,

            dtype=np.float64
        )

        loss_array = np.asarray(

            client_losses,

            dtype=np.float64
        )

        accuracy_array = np.asarray(

            client_accuracies,

            dtype=np.float64
        )

        weighted_loss = float(

            np.average(

                loss_array,

                weights=
                    weight_array
            )
        )

        weighted_accuracy = float(

            np.average(

                accuracy_array,

                weights=
                    weight_array
            )
        )

        round_seconds = (

            time.perf_counter()

            - round_start
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
                client_losses[4]
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
            round_seconds
        )


    # ========================================================
    # TOTAL TRAINING TIME
    # ========================================================

    total_training_seconds = (

        time.perf_counter()

        - total_training_start
    )


    # ========================================================
    # FINAL LOCKED TEST ONLY
    # ========================================================

    (
        y_true,
        y_pred,
        inference_seconds

    ) = evaluate_global_model(

        model=
            global_model,

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
            args.batch_size,

        chunk_size=
            args.chunk_size
    )


    # ========================================================
    # FINAL METRICS
    # ========================================================

    (
        overall,
        report,
        cm,
        class_names

    ) = calculate_metrics(

        y_true,
        y_pred,
        label_mapping
    )


    # ========================================================
    # SAVE
    # ========================================================

    (
        history_df,
        result_dir

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

        label_mapping=
            label_mapping,

        args=
            args,

        client_sample_counts=
            client_sample_counts,

        total_training_seconds=
            total_training_seconds,

        inference_seconds=
            inference_seconds
    )


    # ========================================================
    # KNOWLEDGE TRANSFER
    # ========================================================

    transfer_df = (
        create_knowledge_transfer_report(

            scenario=
                args.scenario,

            report=
                report,

            class_names=
                class_names,

            result_dir=
                result_dir
        )
    )


    # ========================================================
    # GRAPHS
    # ========================================================

    generate_graphs(

        scenario=
            args.scenario,

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
            result_dir
    )


    # ========================================================
    # DISPLAY FINAL RESULTS
    # ========================================================

    print()
    print("=" * 90)
    print("PHASE 4 COMPLETE — FEDAVG")
    print("=" * 90)

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
        f"Input features      : "
        f"{len(active_features)}"
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

    print()

    print("-" * 90)
    print("PER-CLASS FEDAVG PERFORMANCE")
    print("-" * 90)

    for class_name in class_names:

        metrics = report[
            class_name
        ]

        print(

            f"{class_name:25s} "

            f"P={metrics['precision']:.4f} "

            f"R={metrics['recall']:.4f} "

            f"F1={metrics['f1-score']:.4f} "

            f"N={int(metrics['support'])}"
        )

    if transfer_df is not None:

        print()

        print("-" * 90)
        print("KNOWLEDGE TRANSFER")
        print("-" * 90)

        print(
            transfer_df.to_string(
                index=False
            )
        )

    print()

    print(
        f"Results:\n{result_dir}"
    )

    print()

    print(
        "Raw client data shared between clients: NO ✅"
    )

    print(
        "FedAvg parameter aggregation performed: YES ✅"
    )

    print(
        "Locked test used during FL training: NO ✅"
    )

    print(
        "Locked test used for final evaluation only: YES ✅"
    )


if __name__ == "__main__":

    main()