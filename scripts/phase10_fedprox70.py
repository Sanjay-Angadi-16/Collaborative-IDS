"""
Phase 10 - FedProx70 Feature-Preservation Ablation
===================================================

Purpose
-------
Phase 9 selected FedProx as the strongest overall federated method
using the frozen Hybrid 37-feature subset.

This Phase 10 experiment answers one focused question:

    Is the near-zero PortScan recall caused or amplified by reducing
    the model input from the original 70 Phase-4 features to the
    frozen Hybrid 37-feature subset?

Controlled comparison
---------------------
Existing:
    Phase 9 FedProx37

New:
    Phase 10 FedProx70

Only the feature set changes:

    FedProx37 -> 37 frozen Hybrid features
    FedProx70 -> all 70 active Phase-4 features

The following remain fixed:
    - same client files / non-IID partition
    - same scaler and label mapping
    - same 5 clients
    - same MLP family
    - same global class weights
    - same AdamW optimizer
    - same learning rate / weight decay
    - same batch size / chunk size
    - same local epochs
    - same number of federated rounds
    - same sample-weighted server aggregation
    - same FedProx proximal coefficient (default mu=0.01)
    - same locked test set
    - same seed-aware initialization/shuffling protocol

FedProx objective
-----------------
    L = WeightedCrossEntropy
        + (mu / 2) * ||w_local - w_global||^2

Recommended seeds
-----------------
    42 52 62 72 82 92 102 112 122 132

Output
------
    results/phase10/fedprox70/<scenario>/seed_<seed>/
    artifacts/phase10/fedprox70/<scenario>/seed_<seed>/

Examples
--------
    python scripts/phase10_fedprox70.py --scenario non_iid --seed 42 --mu 0.01
    python scripts/phase10_fedprox70.py --scenario non_iid --seed 52 --mu 0.01
"""

from __future__ import annotations

import argparse
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
# REUSE SEED-AWARE PHASE 9 FEDAVG70 INFRASTRUCTURE
# ============================================================

try:
    import phase9_fedavg70 as phase4
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "\nphase10_fedprox70.py requires:\n"
        "    scripts\\phase9_fedavg70.py\n\n"
        "Place this file inside the same scripts folder as "
        "phase9_fedavg70.py."
    ) from exc


# ============================================================
# CONFIG
# ============================================================

DEFAULT_SEED = phase4.SEED
N_CLIENTS = phase4.N_CLIENTS
EXPECTED_FEATURES = 70

DEFAULT_ROUNDS = phase4.DEFAULT_ROUNDS
DEFAULT_LOCAL_EPOCHS = phase4.DEFAULT_LOCAL_EPOCHS
DEFAULT_BATCH_SIZE = phase4.DEFAULT_BATCH_SIZE
DEFAULT_CHUNK_SIZE = phase4.DEFAULT_CHUNK_SIZE
DEFAULT_LEARNING_RATE = phase4.DEFAULT_LEARNING_RATE
DEFAULT_WEIGHT_DECAY = phase4.DEFAULT_WEIGHT_DECAY

DEFAULT_MU = 0.01
GRADIENT_CLIP = 5.0


# ============================================================
# OUTPUT PATHS
# ============================================================

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "phase10"
    / "fedprox70"
)

ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "phase10"
    / "fedprox70"
)

PHASE9_FEDAVG70_ROOT = (
    PROJECT_ROOT
    / "results"
    / "phase9"
    / "fedavg70"
)

PHASE9_FEDPROX37_ROOT = (
    PROJECT_ROOT
    / "results"
    / "phase9"
    / "fedprox"
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("phase10_fedprox70")


# ============================================================
# REPRODUCIBILITY
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


# ============================================================
# JSON HELPERS
# ============================================================

def load_json_if_exists(path: Path):
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# VERIFY ALL 70 FEATURES
# ============================================================

def prepare_all_70_features(
    active_features,
    active_indices,
    feature_columns,
):
    """
    Use every active Phase-4 feature.

    This deliberately does NOT load the Hybrid feature mask.
    """

    if len(active_features) != EXPECTED_FEATURES:
        raise ValueError(
            "\nFedProx70 requires exactly 70 active Phase-4 features.\n"
            f"Found: {len(active_features)}"
        )

    active_indices_array = np.asarray(
        active_indices,
        dtype=np.int64,
    )

    if len(active_indices_array) != EXPECTED_FEATURES:
        raise ValueError(
            "\nFedProx70 requires exactly 70 active feature indices.\n"
            f"Found: {len(active_indices_array)}"
        )

    reconstructed_features = [
        feature_columns[int(index)]
        for index in active_indices_array
    ]

    if list(reconstructed_features) != list(active_features):
        raise ValueError(
            "\nPhase-4 active feature/index mapping mismatch.\n"
            "Refusing to run the ablation with an inconsistent schema."
        )

    return (
        list(active_features),
        active_indices_array,
    )


# ============================================================
# INITIALIZE THE FULL 70-FEATURE MODEL
# ============================================================

def initialize_full_model(
    input_dim: int,
    num_classes: int,
    device,
    seed: int,
):
    """
    Match the seed-based initialization used by Phase 9 FedAvg70.

    No projection is performed because all 70 features are retained.
    """

    phase4.set_seed(seed)

    model = phase4.IDSMLP(
        input_dim=input_dim,
        num_classes=num_classes,
    ).to(device)

    return model


# ============================================================
# FEDPROX PENALTY
# ============================================================

def calculate_proximal_term(
    local_model,
    global_parameters,
):
    proximal_term = torch.zeros(
        1,
        device=next(local_model.parameters()).device,
    )

    for local_parameter, global_parameter in zip(
        local_model.parameters(),
        global_parameters,
    ):
        proximal_term = (
            proximal_term
            + torch.sum(
                (local_parameter - global_parameter) ** 2
            )
        )

    return proximal_term


# ============================================================
# LOCAL FEDPROX TRAINING
# ============================================================

def train_client_fedprox(
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
    round_number,
    mu,
    seed,
):
    """
    Train one client using all 70 features and the FedProx objective.
    """

    training_start = time.perf_counter()

    local_model = phase4.IDSMLP(
        input_dim=input_dim,
        num_classes=len(label_mapping),
    ).to(device)

    local_model.load_state_dict(global_state)

    # Frozen reference to the received global model.
    global_parameters = [
        parameter.detach().clone().to(device)
        for parameter in local_model.parameters()
    ]

    for parameter in global_parameters:
        parameter.requires_grad_(False)

    class_weight_tensor = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weight_tensor
    )

    optimizer = torch.optim.AdamW(
        local_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    total_loss_sum = 0.0
    total_ce_loss_sum = 0.0
    total_prox_loss_sum = 0.0
    total_correct = 0
    total_samples = 0
    total_batches = 0

    local_model.train()

    for local_epoch in range(1, local_epochs + 1):

        # Exactly the repeated-seed shuffle form used in Phase 9.
        rng = np.random.default_rng(
            seed
            + round_number * 1000
            + client_id * 100
            + local_epoch
        )

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
                chunk[phase4.LABEL_COL],
                label_mapping,
            )

            if len(y) == 0:
                continue

            indices = rng.permutation(len(y))

            for start in range(
                0,
                len(indices),
                batch_size,
            ):
                batch_indices = indices[
                    start:start + batch_size
                ]

                X_batch = torch.from_numpy(
                    X[batch_indices].astype(
                        np.float32,
                        copy=False,
                    )
                ).to(device)

                y_batch = torch.from_numpy(
                    y[batch_indices].astype(
                        np.int64,
                        copy=False,
                    )
                ).to(device)

                optimizer.zero_grad(
                    set_to_none=True
                )

                logits = local_model(X_batch)

                ce_loss = criterion(
                    logits,
                    y_batch,
                )

                proximal_term = calculate_proximal_term(
                    local_model=local_model,
                    global_parameters=global_parameters,
                )

                prox_loss = (
                    0.5
                    * mu
                    * proximal_term
                )

                loss = (
                    ce_loss
                    + prox_loss
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    local_model.parameters(),
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
                    (predictions == y_batch)
                    .sum()
                    .item()
                )

                total_loss_sum += (
                    float(loss.item())
                    * batch_samples
                )

                total_ce_loss_sum += (
                    float(ce_loss.item())
                    * batch_samples
                )

                total_prox_loss_sum += (
                    float(prox_loss.item())
                    * batch_samples
                )

                total_correct += batch_correct
                total_samples += batch_samples
                total_batches += 1

            del chunk

    if total_samples == 0:
        raise RuntimeError(
            f"Client {client_id} produced zero training samples."
        )

    mean_loss = float(
        total_loss_sum / total_samples
    )

    mean_ce_loss = float(
        total_ce_loss_sum / total_samples
    )

    mean_prox_loss = float(
        total_prox_loss_sum / total_samples
    )

    accuracy = float(
        total_correct / total_samples
    )

    training_seconds = (
        time.perf_counter()
        - training_start
    )

    state = {
        key: value.detach().cpu().clone()
        for key, value
        in local_model.state_dict().items()
    }

    logger.info(
        "Round %d | Client %d | "
        "FedProx70 loss=%.6f | "
        "CE=%.6f | prox=%.6f | "
        "acc=%.6f | mu=%.6f | time=%.2fs",
        round_number,
        client_id,
        mean_loss,
        mean_ce_loss,
        mean_prox_loss,
        accuracy,
        mu,
        training_seconds,
    )

    del local_model
    del global_parameters

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        state,
        mean_loss,
        mean_ce_loss,
        mean_prox_loss,
        accuracy,
        training_seconds,
    )


# ============================================================
# MODEL / COMMUNICATION SIZE
# ============================================================

def parameter_count(model) -> int:
    return int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
        )
    )


def model_payload_bytes(model) -> int:
    return int(
        sum(
            tensor.numel()
            * tensor.element_size()
            for tensor
            in model.state_dict().values()
        )
    )


def calculate_communication(
    model,
    rounds,
):
    """
    FedProx70 uses the same 70-feature model size as FedAvg70.
    """

    params = parameter_count(model)
    payload = model_payload_bytes(model)

    total = (
        2
        * N_CLIENTS
        * rounds
        * payload
    )

    return {
        "parameter_count": params,
        "model_payload_mib": float(
            payload / 1024 ** 2
        ),
        "total_communication_mib": float(
            total / 1024 ** 2
        ),
        "feature_count": EXPECTED_FEATURES,
        "feature_reduction_ratio": 0.0,
        "parameter_reduction_vs_70_ratio": 0.0,
        "communication_reduction_vs_70_ratio": 0.0,
        "assumption": (
            "Full float32 model upload/download "
            "for every client per round."
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
        metrics = report[class_name]

        rows.append({
            "class": class_name,
            "precision": float(
                metrics["precision"]
            ),
            "recall": float(
                metrics["recall"]
            ),
            "f1_score": float(
                metrics["f1-score"]
            ),
            "support": int(
                metrics["support"]
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# PREVIOUS SAME-SEED PER-CLASS METRICS
# ============================================================

def load_per_class_if_exists(
    root,
    scenario,
    seed,
):
    path = (
        root
        / scenario
        / f"seed_{seed}"
        / "per_class_metrics.csv"
    )

    if not path.exists():
        return None

    df = pd.read_csv(path)

    # Normalize f1 column across result versions.
    if (
        "f1_score" not in df.columns
        and "f1" in df.columns
    ):
        df = df.rename(
            columns={"f1": "f1_score"}
        )

    return df


# ============================================================
# SAME-SEED FEATURE ABLATION COMPARISON
# ============================================================

def save_feature_ablation_comparison(
    scenario,
    seed,
    overall,
    per_class_df,
    result_dir,
):
    """
    Compare:
        FedAvg70  - Phase 9 baseline
        FedProx37 - Phase 9 selected method
        FedProx70 - current Phase 10 ablation
    """

    rows = []

    previous_methods = [
        (
            "FedAvg70",
            70,
            PHASE9_FEDAVG70_ROOT,
        ),
        (
            "FedProx37",
            37,
            PHASE9_FEDPROX37_ROOT,
        ),
    ]

    for method_name, feature_count, root in previous_methods:
        metrics = load_json_if_exists(
            root
            / scenario
            / f"seed_{seed}"
            / "overall_metrics.json"
        )

        if metrics is None:
            continue

        rows.append({
            "method": method_name,
            "seed": int(seed),
            "features": feature_count,
            "accuracy": metrics.get("accuracy"),
            "balanced_accuracy": metrics.get(
                "balanced_accuracy"
            ),
            "macro_precision": metrics.get(
                "macro_precision"
            ),
            "macro_recall": metrics.get(
                "macro_recall"
            ),
            "macro_f1": metrics.get("macro_f1"),
            "weighted_f1": metrics.get(
                "weighted_f1"
            ),
            "training_time_seconds": metrics.get(
                "training_time_seconds"
            ),
            "inference_time_seconds": metrics.get(
                "inference_time_seconds"
            ),
        })

    rows.append({
        "method": "FedProx70",
        "seed": int(seed),
        "features": 70,
        "accuracy": overall["accuracy"],
        "balanced_accuracy": overall[
            "balanced_accuracy"
        ],
        "macro_precision": overall[
            "macro_precision"
        ],
        "macro_recall": overall[
            "macro_recall"
        ],
        "macro_f1": overall["macro_f1"],
        "weighted_f1": overall["weighted_f1"],
        "training_time_seconds": overall[
            "training_time_seconds"
        ],
        "inference_time_seconds": overall[
            "inference_time_seconds"
        ],
    })

    overall_comparison = pd.DataFrame(rows)

    overall_comparison.to_csv(
        result_dir
        / "same_seed_feature_ablation_overall.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Per-class comparison
    # --------------------------------------------------------

    per_class_rows = []

    for method_name, feature_count, root in previous_methods:
        previous_df = load_per_class_if_exists(
            root=root,
            scenario=scenario,
            seed=seed,
        )

        if previous_df is None:
            continue

        for _, row in previous_df.iterrows():
            per_class_rows.append({
                "method": method_name,
                "seed": int(seed),
                "features": feature_count,
                "class": row["class"],
                "precision": row.get(
                    "precision",
                    np.nan,
                ),
                "recall": row.get(
                    "recall",
                    np.nan,
                ),
                "f1_score": row.get(
                    "f1_score",
                    np.nan,
                ),
                "support": row.get(
                    "support",
                    np.nan,
                ),
            })

    for _, row in per_class_df.iterrows():
        per_class_rows.append({
            "method": "FedProx70",
            "seed": int(seed),
            "features": 70,
            "class": row["class"],
            "precision": row["precision"],
            "recall": row["recall"],
            "f1_score": row["f1_score"],
            "support": row["support"],
        })

    per_class_comparison = pd.DataFrame(
        per_class_rows
    )

    if not per_class_comparison.empty:
        per_class_comparison.to_csv(
            result_dir
            / "same_seed_feature_ablation_per_class.csv",
            index=False,
        )

        focus_classes = [
            "DDoS",
            "DoS GoldenEye",
            "FTP-Patator",
            "PortScan",
        ]

        focus_df = per_class_comparison[
            per_class_comparison["class"].isin(
                focus_classes
            )
        ].copy()

        focus_df.to_csv(
            result_dir
            / "same_seed_focus_attack_comparison.csv",
            index=False,
        )

    return (
        overall_comparison,
        per_class_comparison,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Phase 10 FedProx70 feature-preservation ablation"
        )
    )

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
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "Random seed. Use the same 10 seeds as Phase 9."
        ),
    )

    parser.add_argument(
        "--mu",
        type=float,
        default=DEFAULT_MU,
        help=(
            "FedProx proximal coefficient. "
            "Keep fixed at 0.01 for this ablation."
        ),
    )

    args = parser.parse_args()

    if args.mu < 0.0:
        raise ValueError(
            "FedProx mu must be >= 0."
        )

    # This experiment should preserve the selected Phase 9 mu.
    if not np.isclose(
        args.mu,
        DEFAULT_MU,
    ):
        logger.warning(
            "Ablation fairness warning: mu=%.6f differs "
            "from the Phase 9 value %.6f.",
            args.mu,
            DEFAULT_MU,
        )

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    scenario_dir = (
        phase4.FEDERATED_ROOT
        / args.scenario
    )

    if not scenario_dir.exists():
        raise FileNotFoundError(
            scenario_dir
        )

    result_dir = (
        RESULT_ROOT
        / args.scenario
        / f"seed_{args.seed}"
    )

    artifact_dir = (
        ARTIFACT_ROOT
        / args.scenario
        / f"seed_{args.seed}"
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
    # LOAD PHASE-4 PREPROCESSING
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
    ) = prepare_all_70_features(
        active_features=active_features,
        active_indices=active_indices,
        feature_columns=feature_columns,
    )

    feature_reduction = 0.0

    # ========================================================
    # HEADER
    # ========================================================

    print()
    print("=" * 96)
    print(
        "PHASE 10 — FEDPROX70 FEATURE-PRESERVATION ABLATION"
    )
    print("=" * 96)

    print(f"Scenario            : {args.scenario}")
    print(f"Seed                : {args.seed}")
    print(f"Device              : {device}")
    print(f"Clients             : {N_CLIENTS}")
    print(f"Rounds              : {args.rounds}")
    print(f"Local epochs        : {args.local_epochs}")
    print(f"FedProx mu          : {args.mu}")
    print(f"Input features      : {len(all_features)}")
    print(f"Feature reduction   : {feature_reduction * 100:.2f}%")
    print("Hybrid mask used    : NO")
    print("Ablation target     : FedProx37 vs FedProx70")

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
    # CLIENT FILES
    # ========================================================

    client_files = {}
    client_sample_counts = {}

    for client_id in range(
        1,
        N_CLIENTS + 1,
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
            "Client %d samples=%d",
            client_id,
            sample_count,
        )

    # ========================================================
    # GLOBAL MODEL
    # ========================================================

    global_model = initialize_full_model(
        input_dim=len(all_features),
        num_classes=len(label_mapping),
        device=device,
        seed=args.seed,
    )

    communication = calculate_communication(
        model=global_model,
        rounds=args.rounds,
    )

    # ========================================================
    # FEDPROX TRAINING
    # ========================================================

    history = []

    total_training_start = (
        time.perf_counter()
    )

    for round_number in range(
        1,
        args.rounds + 1,
    ):

        print()
        print("-" * 96)
        print(
            f"FEDPROX70 ROUND "
            f"{round_number}/{args.rounds}"
        )
        print("-" * 96)

        round_start = (
            time.perf_counter()
        )

        old_global_state = {
            key: value.detach().cpu().clone()
            for key, value
            in global_model.state_dict().items()
        }

        client_states = []
        aggregation_weights = []

        losses = []
        ce_losses = []
        prox_losses = []
        accuracies = []
        client_training_times = []

        for client_id in range(
            1,
            N_CLIENTS + 1,
        ):
            (
                local_state,
                local_loss,
                local_ce_loss,
                local_prox_loss,
                local_accuracy,
                local_seconds,
            ) = train_client_fedprox(
                global_state=global_model.state_dict(),
                client_file=client_files[
                    client_id
                ],
                scaler=scaler,
                feature_columns=feature_columns,
                active_indices=all_active_indices,
                label_mapping=label_mapping,
                class_weights=global_weights,
                input_dim=len(all_features),
                device=device,
                local_epochs=args.local_epochs,
                batch_size=args.batch_size,
                chunk_size=args.chunk_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                client_id=client_id,
                round_number=round_number,
                mu=args.mu,
                seed=args.seed,
            )

            client_states.append(
                local_state
            )

            aggregation_weights.append(
                client_sample_counts[
                    f"client_{client_id}"
                ]
            )

            losses.append(
                local_loss
            )

            ce_losses.append(
                local_ce_loss
            )

            prox_losses.append(
                local_prox_loss
            )

            accuracies.append(
                local_accuracy
            )

            client_training_times.append(
                local_seconds
            )

        # ----------------------------------------------------
        # SAME SAMPLE-WEIGHTED SERVER AGGREGATION
        # ----------------------------------------------------

        aggregated_state = (
            phase4.federated_average(
                client_states,
                aggregation_weights,
            )
        )

        global_model.load_state_dict(
            aggregated_state
        )

        update_norm = (
            phase4.state_update_norm(
                old_global_state,
                aggregated_state,
            )
        )

        weights = np.asarray(
            aggregation_weights,
            dtype=np.float64,
        )

        weighted_loss = float(
            np.average(
                losses,
                weights=weights,
            )
        )

        weighted_ce_loss = float(
            np.average(
                ce_losses,
                weights=weights,
            )
        )

        weighted_prox_loss = float(
            np.average(
                prox_losses,
                weights=weights,
            )
        )

        weighted_accuracy = float(
            np.average(
                accuracies,
                weights=weights,
            )
        )

        round_seconds = (
            time.perf_counter()
            - round_start
        )

        history.append({
            "round": round_number,
            "seed": int(args.seed),
            "feature_count": 70,
            "weighted_local_loss": weighted_loss,
            "weighted_ce_loss": weighted_ce_loss,
            "weighted_prox_loss": weighted_prox_loss,
            "weighted_local_accuracy": weighted_accuracy,
            "global_update_norm": update_norm,
            "round_training_seconds": round_seconds,
            "mean_client_training_seconds": float(
                np.mean(
                    client_training_times
                )
            ),
            "mu": float(args.mu),
        })

        logger.info(
            "ROUND %d COMPLETE | "
            "loss=%.6f | CE=%.6f | "
            "prox=%.6f | acc=%.6f | "
            "update_norm=%.6f | time=%.2fs",
            round_number,
            weighted_loss,
            weighted_ce_loss,
            weighted_prox_loss,
            weighted_accuracy,
            update_norm,
            round_seconds,
        )

    training_seconds = (
        time.perf_counter()
        - total_training_start
    )

    # ========================================================
    # LOCKED TEST EVALUATION
    # ========================================================

    (
        y_true,
        y_pred,
        inference_seconds,
    ) = phase4.evaluate_global_model(
        model=global_model,
        scaler=scaler,
        feature_columns=feature_columns,
        active_indices=all_active_indices,
        label_mapping=label_mapping,
        device=device,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )

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
        training_seconds
    )

    overall[
        "inference_time_seconds"
    ] = float(
        inference_seconds
    )

    overall[
        "selected_feature_count"
    ] = 70

    overall[
        "feature_reduction_ratio"
    ] = 0.0

    overall[
        "phase"
    ] = 10

    overall[
        "method"
    ] = "FedProx70"

    overall[
        "scenario"
    ] = args.scenario

    overall[
        "seed"
    ] = int(
        args.seed
    )

    overall[
        "fedprox_mu"
    ] = float(
        args.mu
    )

    overall[
        "feature_mode"
    ] = "all_70_phase4_features"

    overall[
        "ablation"
    ] = "FedProx37_vs_FedProx70"

    # ========================================================
    # KNOWLEDGE TRANSFER
    # ========================================================

    transfer_df = (
        phase4.create_knowledge_transfer_report(
            scenario=args.scenario,
            report=report,
            class_names=class_names,
            result_dir=result_dir,
        )
    )

    # ========================================================
    # SAVE HISTORY
    # ========================================================

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        result_dir
        / "fedprox70_round_history.csv",
        index=False,
    )

    # ========================================================
    # SAVE OVERALL
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
    # SAVE PER CLASS
    # ========================================================

    per_class_df = build_per_class_df(
        report,
        class_names,
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
        index=class_names,
        columns=class_names,
    ).to_csv(
        result_dir
        / "confusion_matrix.csv"
    )

    # ========================================================
    # SAME-SEED FEATURE ABLATION FILES
    # ========================================================

    (
        comparison_df,
        per_class_comparison_df,
    ) = save_feature_ablation_comparison(
        scenario=args.scenario,
        seed=args.seed,
        overall=overall,
        per_class_df=per_class_df,
        result_dir=result_dir,
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    torch.save(
        {
            "phase": 10,
            "algorithm": "FedProx70",
            "experiment": (
                "FedProx feature-preservation ablation"
            ),
            "seed": int(args.seed),
            "model_state_dict":
                global_model.state_dict(),
            "input_dim": 70,
            "num_classes": len(
                label_mapping
            ),
            "fedprox_mu": float(
                args.mu
            ),
            "feature_mode":
                "all_70_phase4_features",
            "features": all_features,
            "active_indices":
                all_active_indices.tolist(),
            "label_mapping":
                label_mapping,
        },
        artifact_dir
        / "fedprox70_model.pt",
    )

    # ========================================================
    # MANIFEST
    # ========================================================

    manifest = {
        "phase": 10,
        "algorithm": "FedProx70",
        "experiment":
            "FedProx37_vs_FedProx70_feature_ablation",
        "scenario": args.scenario,
        "seed": int(args.seed),
        "clients": N_CLIENTS,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "fedprox_mu": float(args.mu),
        "aggregation":
            "sample-weighted parameter averaging",
        "local_objective": (
            "Weighted CrossEntropy + "
            "(mu / 2) * ||w_local - w_global||^2"
        ),
        "original_features": 70,
        "selected_features": 70,
        "feature_reduction_ratio": 0.0,
        "hybrid_feature_mask_used": False,
        "feature_schema":
            "all active Phase-4 features",
        "features": all_features,
        "active_indices":
            all_active_indices.tolist(),
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
        / "phase10_fedprox70_manifest.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=4,
        )

    # ========================================================
    # GRAPHS
    # ========================================================

    phase4.generate_graphs(
        scenario=(
            "FedProx70-"
            f"{args.scenario}"
            f"-seed{args.seed}"
        ),
        history_df=history_df,
        overall=overall,
        report=report,
        cm=cm,
        class_names=class_names,
        transfer_df=transfer_df,
        result_dir=result_dir,
    )

    # ========================================================
    # FINAL DISPLAY
    # ========================================================

    print()
    print("=" * 96)
    print(
        "PHASE 10 COMPLETE — FEDPROX70 FEATURE-PRESERVATION ABLATION"
    )
    print("=" * 96)

    print(f"Scenario            : {args.scenario}")
    print(f"Seed                : {args.seed}")
    print(f"Device              : {device}")
    print(f"Federated rounds    : {args.rounds}")
    print(f"Local epochs/round  : {args.local_epochs}")
    print(f"FedProx mu          : {args.mu}")
    print("Features            : 70")
    print("Feature reduction   : 0.00%")
    print("Hybrid mask         : NOT USED")

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
        f"{training_seconds:.2f}s"
    )

    print(
        f"Inference time      : "
        f"{inference_seconds:.2f}s"
    )

    # ========================================================
    # PER CLASS
    # ========================================================

    print()
    print("-" * 96)
    print(
        "PER-CLASS FEDPROX70 PERFORMANCE"
    )
    print("-" * 96)

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

    # ========================================================
    # FOCUS CLASSES
    # ========================================================

    print()
    print("-" * 96)
    print(
        "FEATURE-ABLATION FOCUS CLASSES"
    )
    print("-" * 96)

    for focus_class in [
        "DDoS",
        "DoS GoldenEye",
        "FTP-Patator",
        "PortScan",
    ]:
        if focus_class in report:
            metrics = report[
                focus_class
            ]

            print(
                f"{focus_class:25s} "
                f"Recall={metrics['recall']:.6f} "
                f"F1={metrics['f1-score']:.6f}"
            )

    # ========================================================
    # KNOWLEDGE TRANSFER
    # ========================================================

    if transfer_df is not None:

        print()
        print("-" * 96)
        print(
            "KNOWLEDGE TRANSFER"
        )
        print("-" * 96)

        print(
            transfer_df.to_string(
                index=False
            )
        )

        if (
            "knowledge_transfer_gain"
            in transfer_df.columns
        ):
            print()
            print(
                "Mean knowledge-transfer gain: "
                f"{transfer_df['knowledge_transfer_gain'].mean():.6f}"
            )

    # ========================================================
    # COMMUNICATION
    # ========================================================

    print()
    print("-" * 96)
    print(
        "MODEL / COMMUNICATION EFFICIENCY"
    )
    print("-" * 96)

    print(
        f"FedProx70 parameters : "
        f"{communication['parameter_count']:,}"
    )

    print(
        f"Model payload        : "
        f"{communication['model_payload_mib']:.4f} MiB"
    )

    print(
        f"Total communication  : "
        f"{communication['total_communication_mib']:.4f} MiB"
    )

    print(
        "Reduction vs 70      : 0.00%"
    )

    # ========================================================
    # SAME-SEED COMPARISON
    # ========================================================

    print()
    print("-" * 96)
    print(
        "SAME-SEED FEATURE ABLATION"
    )
    print("-" * 96)

    print(
        comparison_df.to_string(
            index=False
        )
    )

    # ========================================================
    # INTEGRITY
    # ========================================================

    print()
    print("=" * 96)
    print(
        "EXPERIMENTAL INTEGRITY"
    )
    print("=" * 96)

    print(
        "All 70 active Phase-4 features used  : YES ✅"
    )
    print(
        "Hybrid 37-feature mask used          : NO ✅"
    )
    print(
        "Hybrid optimizer rerun               : NO ✅"
    )
    print(
        "Same scaler / preprocessing          : YES ✅"
    )
    print(
        "Same client files                    : YES ✅"
    )
    print(
        "Same MLP family                      : YES ✅"
    )
    print(
        "Same class weights                   : YES ✅"
    )
    print(
        "Same AdamW optimizer                 : YES ✅"
    )
    print(
        "Same sample-weighted aggregation     : YES ✅"
    )
    print(
        "FedProx mu fixed                     : "
        f"{args.mu} ✅"
    )
    print(
        "Locked test used during training     : NO ✅"
    )
    print(
        "Locked test used for final eval only : YES ✅"
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
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()
