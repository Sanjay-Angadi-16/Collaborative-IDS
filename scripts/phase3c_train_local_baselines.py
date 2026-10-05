"""
Phase 3C - Local-Only Federated Client Baselines
=================================================

Each federated client trains independently.

NO:
    - FedAvg
    - FedProx
    - parameter sharing
    - model aggregation
    - communication between clients

Every local model:
    1. Uses the same Phase-3A scaler
    2. Uses the same 70 active features
    3. Uses the same MLP architecture as Phase 3B
    4. Has 6 output neurons
    5. Is evaluated on the SAME locked global test set

Supported scenarios:

    non_iid
    controlled_non_iid

Examples:

    python scripts/phase3c_train_local_baselines.py --scenario controlled_non_iid

    python scripts/phase3c_train_local_baselines.py --scenario non_iid
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
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


# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

FEDERATED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "federated"
)

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_locked.csv"
)


# ------------------------------------------------------------
# PREPROCESSING
# ------------------------------------------------------------

PREPROCESSING_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "preprocessing"
)


# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "local_baselines"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "local_baselines"
)


# ------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------

DEFAULT_EPOCHS = 5

DEFAULT_BATCH_SIZE = 2048

DEFAULT_CHUNK_SIZE = 100_000

DEFAULT_LEARNING_RATE = 0.001

DEFAULT_WEIGHT_DECAY = 0.0001


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("phase3c")


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# LOAD PHASE 3A PREPROCESSING
# ============================================================

def load_preprocessing():

    scaler_file = (
        PREPROCESSING_DIR
        / "scaler.joblib"
    )

    feature_file = (
        PREPROCESSING_DIR
        / "feature_columns.json"
    )

    label_file = (
        PREPROCESSING_DIR
        / "label_mapping.json"
    )

    for file in [
        scaler_file,
        feature_file,
        label_file
    ]:

        if not file.exists():

            raise FileNotFoundError(
                f"Missing Phase-3A artifact: {file}"
            )

    scaler = joblib.load(
        scaler_file
    )

    with open(
        feature_file,
        "r",
        encoding="utf-8"
    ) as f:

        feature_columns = json.load(f)

    with open(
        label_file,
        "r",
        encoding="utf-8"
    ) as f:

        label_mapping = json.load(f)

    return (
        scaler,
        feature_columns,
        label_mapping
    )


# ============================================================
# ACTIVE FEATURES
# ============================================================

def get_active_features(
    scaler,
    feature_columns
):

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
        "Original features = %d",
        len(feature_columns)
    )

    logger.info(
        "Active features   = %d",
        len(active_features)
    )

    return (
        active_indices,
        active_features
    )


# ============================================================
# CLEAN + SCALE
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
# LABELS
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
            f"Unknown labels: {unknown}"
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
    Same architecture as centralized Phase 3B.
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
# DETECT CLIENT ATTACK
# ============================================================

def detect_client_attack(
    client_file
):

    labels = set()

    for chunk in pd.read_csv(
        client_file,
        usecols=[LABEL_COL],
        chunksize=100_000
    ):

        labels.update(

            chunk[
                LABEL_COL
            ]
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

    attacks = [

        label

        for label in labels

        if label != BENIGN_LABEL
    ]

    if len(attacks) != 1:

        raise RuntimeError(
            f"{client_file} should contain exactly "
            f"one attack class. Found: {attacks}"
        )

    return attacks[0]


# ============================================================
# LOCAL CLASS COUNTS
# ============================================================

def get_local_class_counts(
    client_file,
    label_mapping,
    chunk_size
):

    counts = np.zeros(
        len(label_mapping),
        dtype=np.int64
    )

    for chunk in pd.read_csv(
        client_file,
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
# LOCAL CLASS WEIGHTS
# ============================================================

def calculate_local_weights(
    counts
):

    """
    We calculate weights only for classes present locally.

    Classes never observed by the client receive weight 0,
    because they never appear as training targets.
    """

    weights = np.zeros(
        len(counts),
        dtype=np.float32
    )

    present = (
        counts > 0
    )

    present_counts = (
        counts[present]
        .astype(np.float64)
    )

    maximum = (
        present_counts.max()
    )

    local_weights = np.sqrt(
        maximum
        / present_counts
    )

    local_weights = np.minimum(
        local_weights,
        20.0
    )

    local_weights = (
        local_weights
        / local_weights.mean()
    )

    weights[
        present
    ] = local_weights

    return weights


# ============================================================
# TRAIN LOCAL MODEL
# ============================================================

def train_local_model(
    model,
    client_file,
    scaler,
    feature_columns,
    active_indices,
    label_mapping,
    criterion,
    optimizer,
    device,
    epochs,
    batch_size,
    chunk_size
):

    history = []

    for epoch in range(
        1,
        epochs + 1
    ):

        model.train()

        epoch_loss = 0.0

        epoch_correct = 0

        epoch_samples = 0

        rng = np.random.default_rng(
            SEED + epoch
        )

        for chunk_number, chunk in enumerate(

            pd.read_csv(
                client_file,
                chunksize=chunk_size
            ),

            start=1
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

                logits = model(
                    xb
                )

                loss = criterion(
                    logits,
                    yb
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0
                )

                optimizer.step()

                n = len(
                    yb
                )

                epoch_loss += (
                    loss.item()
                    * n
                )

                predictions = (
                    logits.argmax(
                        dim=1
                    )
                )

                epoch_correct += (

                    predictions
                    .eq(yb)
                    .sum()
                    .item()
                )

                epoch_samples += n

        loss_value = (
            epoch_loss
            / epoch_samples
        )

        accuracy_value = (
            epoch_correct
            / epoch_samples
        )

        history.append({

            "epoch":
                epoch,

            "train_loss":
                loss_value,

            "train_accuracy":
                accuracy_value
        })

        logger.info(

            "Epoch %d/%d | "
            "loss=%.6f | "
            "local_accuracy=%.6f",

            epoch,
            epochs,
            loss_value,
            accuracy_value
        )

    return history


# ============================================================
# TEST LOCAL MODEL
# ============================================================

def evaluate_model(
    model,
    test_file,
    scaler,
    feature_columns,
    active_indices,
    label_mapping,
    device,
    batch_size,
    chunk_size
):

    model.eval()

    true_list = []

    prediction_list = []

    start_time = time.perf_counter()

    with torch.no_grad():

        for chunk in pd.read_csv(
            test_file,
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

                output = model(
                    xb
                )

                predictions = (

                    output
                    .argmax(
                        dim=1
                    )
                    .cpu()
                    .numpy()
                )

                chunk_predictions.append(
                    predictions
                )

            true_list.append(
                y
            )

            prediction_list.append(

                np.concatenate(
                    chunk_predictions
                )
            )

    inference_seconds = (

        time.perf_counter()
        - start_time
    )

    y_true = np.concatenate(
        true_list
    )

    y_pred = np.concatenate(
        prediction_list
    )

    return (
        y_true,
        y_pred,
        inference_seconds
    )


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    label_mapping,
    local_attack
):

    class_ids = list(
        range(
            len(label_mapping)
        )
    )

    inverse_mapping = {

        index: label

        for label, index
        in label_mapping.items()
    }

    class_names = [

        inverse_mapping[i]

        for i in class_ids
    ]

    report = classification_report(
        y_true,
        y_pred,
        labels=class_ids,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

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
            )
    }


    # ========================================================
    # SEEN ATTACK PERFORMANCE
    # ========================================================

    seen_recall = float(
        report[
            local_attack
        ]["recall"]
    )

    seen_f1 = float(
        report[
            local_attack
        ]["f1-score"]
    )


    # ========================================================
    # UNSEEN ATTACK PERFORMANCE
    # ========================================================

    unseen_attacks = [

        name

        for name in class_names

        if (
            name != BENIGN_LABEL
            and
            name != local_attack
        )
    ]

    unseen_recalls = [

        report[
            attack
        ]["recall"]

        for attack
        in unseen_attacks
    ]

    unseen_f1s = [

        report[
            attack
        ]["f1-score"]

        for attack
        in unseen_attacks
    ]

    overall[
        "local_seen_attack"
    ] = local_attack

    overall[
        "seen_attack_recall"
    ] = seen_recall

    overall[
        "seen_attack_f1"
    ] = seen_f1

    overall[
        "mean_unseen_attack_recall"
    ] = float(
        np.mean(
            unseen_recalls
        )
    )

    overall[
        "mean_unseen_attack_f1"
    ] = float(
        np.mean(
            unseen_f1s
        )
    )

    return (
        overall,
        report,
        class_names
    )


# ============================================================
# SAVE CLIENT RESULTS
# ============================================================

def save_client_results(
    client_id,
    scenario,
    model,
    history,
    overall,
    report,
    class_names,
    active_features,
    label_mapping,
    train_seconds,
    inference_seconds
):

    artifact_dir = (

        ARTIFACT_ROOT
        / scenario
        / f"client_{client_id}"
    )

    result_dir = (

        RESULT_ROOT
        / scenario
        / f"client_{client_id}"
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
    # MODEL
    # --------------------------------------------------------

    torch.save(

        {
            "model_state_dict":
                model.state_dict(),

            "active_features":
                active_features,

            "label_mapping":
                label_mapping
        },

        artifact_dir
        / "local_model.pt"
    )

    # --------------------------------------------------------
    # TRAINING HISTORY
    # --------------------------------------------------------

    pd.DataFrame(
        history
    ).to_csv(

        result_dir
        / "training_history.csv",

        index=False
    )

    # --------------------------------------------------------
    # PER-CLASS RESULTS
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
    # OVERALL
    # --------------------------------------------------------

    overall[
        "training_time_seconds"
    ] = float(
        train_seconds
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


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--scenario",

        choices=[
            "non_iid",
            "controlled_non_iid"
        ],

        default=
            "controlled_non_iid"
    )

    parser.add_argument(

        "--epochs",

        type=int,

        default=
            DEFAULT_EPOCHS
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

    args = parser.parse_args()


    # ========================================================
    # FILES
    # ========================================================

    scenario_dir = (

        FEDERATED_ROOT
        / args.scenario
    )

    if not scenario_dir.exists():

        raise FileNotFoundError(
            f"Scenario directory missing: "
            f"{scenario_dir}"
        )

    if not TEST_FILE.exists():

        raise FileNotFoundError(
            f"Locked test missing: {TEST_FILE}"
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
        label_mapping

    ) = load_preprocessing()

    (
        active_indices,
        active_features

    ) = get_active_features(
        scaler,
        feature_columns
    )


    # ========================================================
    # RESULTS SUMMARY
    # ========================================================

    summary_rows = []


    print()
    print("=" * 90)
    print(
        f"PHASE 3C - LOCAL BASELINES "
        f"[{args.scenario.upper()}]"
    )
    print("=" * 90)


    # ========================================================
    # TRAIN EACH CLIENT
    # ========================================================

    for client_id in range(
        1,
        N_CLIENTS + 1
    ):

        print()
        print("=" * 90)
        print(
            f"CLIENT {client_id}"
        )
        print("=" * 90)

        client_file = (

            scenario_dir
            / f"client_{client_id}.csv"
        )

        if not client_file.exists():

            raise FileNotFoundError(
                client_file
            )


        # ----------------------------------------------------
        # ATTACK
        # ----------------------------------------------------

        local_attack = (
            detect_client_attack(
                client_file
            )
        )

        logger.info(
            "Client %d local attack: %s",
            client_id,
            local_attack
        )


        # ----------------------------------------------------
        # CLASS COUNTS
        # ----------------------------------------------------

        counts = (
            get_local_class_counts(
                client_file,
                label_mapping,
                args.chunk_size
            )
        )

        weights = (
            calculate_local_weights(
                counts
            )
        )


        # ----------------------------------------------------
        # SAME INITIALIZATION FOR EVERY CLIENT
        # ----------------------------------------------------

        set_seed(
            SEED
        )

        model = IDSMLP(

            input_dim=
                len(active_features),

            num_classes=
                len(label_mapping)

        ).to(device)


        # ----------------------------------------------------
        # LOSS
        # ----------------------------------------------------

        weight_tensor = torch.tensor(

            weights,

            dtype=torch.float32,

            device=device
        )

        criterion = nn.CrossEntropyLoss(
            weight=
                weight_tensor
        )


        # ----------------------------------------------------
        # OPTIMIZER
        # ----------------------------------------------------

        optimizer = torch.optim.AdamW(

            model.parameters(),

            lr=
                DEFAULT_LEARNING_RATE,

            weight_decay=
                DEFAULT_WEIGHT_DECAY
        )


        # ----------------------------------------------------
        # TRAINING TIMER
        # ----------------------------------------------------

        training_start = (
            time.perf_counter()
        )

        history = train_local_model(

            model=model,

            client_file=client_file,

            scaler=scaler,

            feature_columns=
                feature_columns,

            active_indices=
                active_indices,

            label_mapping=
                label_mapping,

            criterion=
                criterion,

            optimizer=
                optimizer,

            device=
                device,

            epochs=
                args.epochs,

            batch_size=
                args.batch_size,

            chunk_size=
                args.chunk_size
        )

        training_seconds = (

            time.perf_counter()

            - training_start
        )


        # ----------------------------------------------------
        # GLOBAL LOCKED TEST
        # ----------------------------------------------------

        (
            y_true,
            y_pred,
            inference_seconds

        ) = evaluate_model(

            model=model,

            test_file=
                TEST_FILE,

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


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        (
            overall,
            report,
            class_names

        ) = calculate_metrics(

            y_true,
            y_pred,
            label_mapping,
            local_attack
        )


        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        save_client_results(

            client_id=
                client_id,

            scenario=
                args.scenario,

            model=
                model,

            history=
                history,

            overall=
                overall,

            report=
                report,

            class_names=
                class_names,

            active_features=
                active_features,

            label_mapping=
                label_mapping,

            train_seconds=
                training_seconds,

            inference_seconds=
                inference_seconds
        )


        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        summary_rows.append({

            "client":
                f"client_{client_id}",

            "local_attack":
                local_attack,

            "accuracy":
                overall[
                    "accuracy"
                ],

            "balanced_accuracy":
                overall[
                    "balanced_accuracy"
                ],

            "macro_f1":
                overall[
                    "macro_f1"
                ],

            "weighted_f1":
                overall[
                    "weighted_f1"
                ],

            "seen_attack_recall":
                overall[
                    "seen_attack_recall"
                ],

            "seen_attack_f1":
                overall[
                    "seen_attack_f1"
                ],

            "mean_unseen_attack_recall":
                overall[
                    "mean_unseen_attack_recall"
                ],

            "mean_unseen_attack_f1":
                overall[
                    "mean_unseen_attack_f1"
                ],

            "training_seconds":
                training_seconds,

            "inference_seconds":
                inference_seconds
        })


        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        print()

        print(
            f"Local attack          : "
            f"{local_attack}"
        )

        print(
            f"Accuracy              : "
            f"{overall['accuracy']:.6f}"
        )

        print(
            f"Balanced Accuracy     : "
            f"{overall['balanced_accuracy']:.6f}"
        )

        print(
            f"Macro F1              : "
            f"{overall['macro_f1']:.6f}"
        )

        print(
            f"Weighted F1           : "
            f"{overall['weighted_f1']:.6f}"
        )

        print(
            f"Seen attack recall    : "
            f"{overall['seen_attack_recall']:.6f}"
        )

        print(
            f"Unseen mean recall    : "
            f"{overall['mean_unseen_attack_recall']:.6f}"
        )

        print()

        print(
            "PER-CLASS RESULTS"
        )

        for class_name in class_names:

            metrics = report[
                class_name
            ]

            print(

                f"{class_name:25s} "

                f"P={metrics['precision']:.4f} "

                f"R={metrics['recall']:.4f} "

                f"F1={metrics['f1-score']:.4f}"
            )


    # ========================================================
    # SAVE COMBINED SUMMARY
    # ========================================================

    scenario_result_dir = (

        RESULT_ROOT
        / args.scenario
    )

    scenario_result_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_file = (

        scenario_result_dir
        / "local_client_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )


    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 90)
    print("PHASE 3C COMPLETE")
    print("=" * 90)

    print()

    print(
        f"Scenario : {args.scenario}"
    )

    print(
        f"Clients  : {N_CLIENTS}"
    )

    print(
        f"Classes  : {len(label_mapping)}"
    )

    print(
        f"Features : {len(active_features)}"
    )

    print()

    print(
        f"Summary:\n{summary_file}"
    )

    print()

    print(
        "Federated aggregation performed: NO ✅"
    )

    print(
        "Model sharing performed: NO ✅"
    )

    print(
        "Locked test used for training: NO ✅"
    )

    print(
        "Locked test used for evaluation: YES ✅"
    )


if __name__ == "__main__":

    main()