from __future__ import annotations

import gc
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
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
        "\nCould not import:\n"
        "scripts\\phase9_fedavg70.py\n\n"
        "Keep this script inside the scripts folder."
    ) from exc


# ============================================================
# PHASE CONFIGURATION
# ============================================================

PHASE_NAME = "PHASE 14B.1 — AUTOENCODER ERROR DISTRIBUTION & THRESHOLD STUDY"

SCENARIO = "non_iid"
SEED = 42

N_CLIENTS = 5
EXPECTED_FEATURES = 70

BENIGN_VALIDATION_RATIO = 0.20

INFERENCE_BATCH_SIZE = 4096

# Thresholds to compare.
PERCENTILE_THRESHOLDS = (95.0, 97.0, 98.0, 99.0)

# Robust threshold:
# threshold = median + MAD_K * 1.4826 * MAD
MAD_K = 3.5
MAD_NORMALIZATION = 1.4826

# IMPORTANT:
# ORACLE thresholds use attack labels and therefore are DIAGNOSTIC ONLY.
# They must NOT be used as the final zero-day/open-set threshold.
INCLUDE_ORACLE_THRESHOLDS = True


# ============================================================
# CLIENT ATTACK MAPPING
# ============================================================

CLIENT_ATTACKS = {
    1: "DoS Hulk",
    2: "DDoS",
    3: "PortScan",
    4: "DoS GoldenEye",
    5: "FTP-Patator",
}


# ============================================================
# AUTOENCODER ARCHITECTURE
# Must match Phase 14B.
# ============================================================

HIDDEN_DIM_1 = 48
HIDDEN_DIM_2 = 24
LATENT_DIM = 12


class PersonalizedAutoencoder(nn.Module):
    def __init__(self, input_dim=70):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM_1),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM_2, LATENT_DIM),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM_2, HIDDEN_DIM_1),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM_1, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ============================================================
# OUTPUT PATHS
# ============================================================

ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "detection"

RESULT_ROOT = PROJECT_ROOT / "results" / "phase14" / "phase14b1"
GRAPH_ROOT = RESULT_ROOT / "graphs"
ERROR_ROOT = RESULT_ROOT / "errors"

THRESHOLD_COMPARISON_FILE = RESULT_ROOT / "phase14b1_threshold_comparison.csv"
CLIENT_SUMMARY_FILE = RESULT_ROOT / "phase14b1_client_summary.csv"
ERROR_SUMMARY_FILE = RESULT_ROOT / "phase14b1_error_distribution_summary.csv"
FINAL_REPORT_FILE = RESULT_ROOT / "phase14b1_final_report.txt"
SUMMARY_JSON_FILE = RESULT_ROOT / "phase14b1_summary.json"


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("phase14b1_ae_threshold_study")


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DISPLAY
# ============================================================

def separator():
    print()
    print("=" * 110)


# ============================================================
# PROJECT CLASS FILTER
# ============================================================

def filter_to_project_classes(dataframe, label_mapping):
    canonical = {
        str(label).strip().lower(): str(label).strip()
        for label in label_mapping.keys()
    }

    labels = (
        dataframe[phase4.LABEL_COL]
        .astype(str)
        .str.strip()
    )

    keep_mask = labels.str.lower().isin(canonical.keys())

    dataframe = dataframe.loc[keep_mask].copy()

    dataframe[phase4.LABEL_COL] = (
        dataframe[phase4.LABEL_COL]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(canonical)
    )

    return dataframe


# ============================================================
# PREPROCESSING VERIFICATION
# ============================================================

def verify_preprocessing(active_features, active_indices, feature_columns):
    if len(active_features) != EXPECTED_FEATURES:
        raise ValueError(
            "\nExpected exactly "
            f"{EXPECTED_FEATURES} active features.\n"
            f"Found: {len(active_features)}"
        )

    active_indices = np.asarray(active_indices, dtype=np.int64)

    reconstructed = [
        feature_columns[int(index)]
        for index in active_indices
    ]

    if reconstructed != list(active_features):
        raise ValueError("\nFeature ordering mismatch.")

    return active_indices


# ============================================================
# DATA LOADER
# ============================================================

def create_loader(X, batch_size, seed):
    tensor = torch.from_numpy(
        X.astype(np.float32, copy=False)
    )

    dataset = TensorDataset(tensor)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=None,
    )


# ============================================================
# RECONSTRUCTION ERRORS
# ============================================================

@torch.no_grad()
def reconstruction_errors(model, X, device):
    loader = create_loader(
        X=X,
        batch_size=INFERENCE_BATCH_SIZE,
        seed=SEED,
    )

    model.eval()

    all_errors = []

    for (batch,) in loader:
        batch = batch.to(device)

        reconstructed = model(batch)

        row_errors = (
            (reconstructed - batch) ** 2
        ).mean(dim=1)

        all_errors.append(
            row_errors.detach().cpu().numpy()
        )

    return np.concatenate(all_errors)


# ============================================================
# LOAD SAVED PHASE 14B MODEL
# ============================================================

def load_client_autoencoder(client_id, device):
    model_file = ARTIFACT_ROOT / f"autoencoder_client_{client_id}.pt"

    if not model_file.exists():
        raise FileNotFoundError(
            "\nMissing Phase 14B Autoencoder model:\n"
            f"{model_file}\n\n"
            "Run Phase 14B first."
        )

    checkpoint = torch.load(
        model_file,
        map_location=device,
    )

    input_dim = int(
        checkpoint.get(
            "input_dim",
            EXPECTED_FEATURES,
        )
    )

    if input_dim != EXPECTED_FEATURES:
        raise ValueError(
            f"\nClient {client_id}: checkpoint input_dim={input_dim}; "
            f"expected {EXPECTED_FEATURES}."
        )

    model = PersonalizedAutoencoder(
        input_dim=input_dim
    ).to(device)

    state_dict = checkpoint.get("state_dict")

    if state_dict is None:
        raise KeyError(
            f"\nClient {client_id}: checkpoint has no state_dict."
        )

    model.load_state_dict(state_dict)
    model.eval()

    return model, checkpoint, model_file


# ============================================================
# ERROR DISTRIBUTION SUMMARY
# ============================================================

def distribution_summary(
    client_id,
    attack_name,
    group_name,
    errors,
):
    percentiles = [
        0,
        1,
        5,
        25,
        50,
        75,
        90,
        95,
        97,
        98,
        99,
        99.5,
        99.9,
        100,
    ]

    row = {
        "client_id": client_id,
        "seen_attack": attack_name,
        "group": group_name,
        "rows": int(len(errors)),
        "mean": float(np.mean(errors)),
        "std": float(np.std(errors)),
        "median": float(np.median(errors)),
        "min": float(np.min(errors)),
        "max": float(np.max(errors)),
    }

    for p in percentiles:
        safe_name = str(p).replace(".", "_")
        row[f"p{safe_name}"] = float(
            np.percentile(errors, p)
        )

    return row


# ============================================================
# THRESHOLD BUILDERS
# ============================================================

def benign_percentile_thresholds(benign_errors):
    thresholds = {}

    for percentile in PERCENTILE_THRESHOLDS:
        thresholds[f"P{int(percentile)}"] = {
            "threshold": float(
                np.percentile(
                    benign_errors,
                    percentile,
                )
            ),
            "uses_attack_labels": False,
            "threshold_source": (
                f"benign_validation_percentile_{percentile:g}"
            ),
        }

    return thresholds


def mad_threshold(benign_errors):
    median = float(
        np.median(benign_errors)
    )

    mad = float(
        np.median(
            np.abs(
                benign_errors - median
            )
        )
    )

    robust_sigma = (
        MAD_NORMALIZATION * mad
    )

    threshold = (
        median
        +
        MAD_K * robust_sigma
    )

    # Safe fallback in the rare case MAD is zero.
    if threshold <= median:
        threshold = float(
            np.percentile(
                benign_errors,
                99.0,
            )
        )

    return {
        "MAD": {
            "threshold": float(threshold),
            "uses_attack_labels": False,
            "threshold_source": (
                f"benign_median_plus_{MAD_K:g}_robust_sigma"
            ),
            "benign_median": median,
            "benign_mad": mad,
            "robust_sigma": robust_sigma,
        }
    }


def oracle_thresholds(
    benign_errors,
    attack_errors,
):
    """
    DIAGNOSTIC ONLY.
    Uses attack labels and therefore MUST NOT be used as the final
    zero-day/open-set threshold.
    """

    y_true = np.concatenate(
        [
            np.zeros(
                len(benign_errors),
                dtype=np.int64,
            ),
            np.ones(
                len(attack_errors),
                dtype=np.int64,
            ),
        ]
    )

    scores = np.concatenate(
        [
            benign_errors,
            attack_errors,
        ]
    )

    output = {}

    # --------------------------------------------------------
    # ORACLE 1: Youden J from ROC
    # --------------------------------------------------------
    fpr, tpr, roc_thresholds = roc_curve(
        y_true,
        scores,
    )

    finite_mask = np.isfinite(
        roc_thresholds
    )

    fpr_f = fpr[finite_mask]
    tpr_f = tpr[finite_mask]
    th_f = roc_thresholds[finite_mask]

    if len(th_f) > 0:
        j = tpr_f - fpr_f
        best_index = int(
            np.argmax(j)
        )

        output["ORACLE_YOUDEN"] = {
            "threshold": float(
                th_f[best_index]
            ),
            "uses_attack_labels": True,
            "threshold_source": (
                "oracle_roc_youden_j_diagnostic_only"
            ),
        }

    # --------------------------------------------------------
    # ORACLE 2: Best F1 from PR curve
    # --------------------------------------------------------
    precision, recall, pr_thresholds = (
        precision_recall_curve(
            y_true,
            scores,
        )
    )

    if len(pr_thresholds) > 0:
        # precision/recall arrays have one extra element.
        p = precision[:-1]
        r = recall[:-1]

        f1_values = (
            2.0 * p * r
            /
            np.maximum(
                p + r,
                1e-12,
            )
        )

        best_index = int(
            np.argmax(f1_values)
        )

        output["ORACLE_BEST_F1"] = {
            "threshold": float(
                pr_thresholds[best_index]
            ),
            "uses_attack_labels": True,
            "threshold_source": (
                "oracle_pr_best_f1_diagnostic_only"
            ),
        }

    return output


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

def evaluate_threshold(
    client_id,
    attack_name,
    threshold_name,
    threshold_info,
    benign_errors,
    attack_errors,
):
    threshold = float(
        threshold_info["threshold"]
    )

    y_true = np.concatenate(
        [
            np.zeros(
                len(benign_errors),
                dtype=np.int64,
            ),
            np.ones(
                len(attack_errors),
                dtype=np.int64,
            ),
        ]
    )

    scores = np.concatenate(
        [
            benign_errors,
            attack_errors,
        ]
    )

    y_pred = (
        scores > threshold
    ).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    benign_fpr = (
        fp / max(tn + fp, 1)
    )

    attack_recall = (
        tp / max(tp + fn, 1)
    )

    specificity = (
        tn / max(tn + fp, 1)
    )

    return {
        "client_id": client_id,
        "seen_attack": attack_name,
        "threshold_method": threshold_name,
        "threshold": threshold,
        "uses_attack_labels": bool(
            threshold_info.get(
                "uses_attack_labels",
                False,
            )
        ),
        "threshold_source": threshold_info.get(
            "threshold_source",
            "",
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "benign_fpr": float(
            benign_fpr
        ),
        "specificity": float(
            specificity
        ),
        "attack_detection_rate": float(
            attack_recall
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                scores,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                scores,
            )
        ),
        "mean_benign_error": float(
            np.mean(benign_errors)
        ),
        "mean_attack_error": float(
            np.mean(attack_errors)
        ),
        "median_benign_error": float(
            np.median(benign_errors)
        ),
        "median_attack_error": float(
            np.median(attack_errors)
        ),
    }


# ============================================================
# GRAPHING
# ============================================================

def save_histogram(
    client_id,
    attack_name,
    benign_errors,
    attack_errors,
    threshold_dictionary,
):
    """
    Distribution figure for visual diagnosis.
    Uses no locked-test data.
    """

    # Avoid one extreme value making the whole x-axis unreadable.
    combined = np.concatenate(
        [
            benign_errors,
            attack_errors,
        ]
    )

    x_max = float(
        np.percentile(
            combined,
            99.9,
        )
    )

    if not np.isfinite(x_max) or x_max <= 0:
        x_max = float(
            np.max(combined)
        )

    benign_plot = benign_errors[
        benign_errors <= x_max
    ]

    attack_plot = attack_errors[
        attack_errors <= x_max
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.hist(
        benign_plot,
        bins=100,
        alpha=0.55,
        density=True,
        label="BENIGN validation",
    )

    ax.hist(
        attack_plot,
        bins=100,
        alpha=0.55,
        density=True,
        label=attack_name,
    )

    for name in (
        "P95",
        "P97",
        "P98",
        "P99",
        "MAD",
    ):
        info = threshold_dictionary.get(
            name
        )

        if info is None:
            continue

        threshold = float(
            info["threshold"]
        )

        if threshold <= x_max:
            ax.axvline(
                threshold,
                linestyle="--",
                linewidth=1.2,
                label=f"{name}={threshold:.4f}",
            )

    ax.set_title(
        f"Client {client_id} — Reconstruction Error Distribution\n"
        f"Attack: {attack_name}"
    )

    ax.set_xlabel(
        "Reconstruction MSE"
    )

    ax.set_ylabel(
        "Density"
    )

    ax.legend(
        fontsize=8
    )

    fig.tight_layout()

    graph_file = (
        GRAPH_ROOT
        /
        f"client_{client_id}_error_distribution.png"
    )

    fig.savefig(
        graph_file,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return graph_file


def save_ecdf(
    client_id,
    attack_name,
    benign_errors,
    attack_errors,
):
    """
    ECDF is useful when error distributions have long tails.
    """

    def ecdf(values):
        x = np.sort(values)
        y = (
            np.arange(
                1,
                len(x) + 1,
            )
            /
            len(x)
        )
        return x, y

    benign_x, benign_y = ecdf(
        benign_errors
    )

    attack_x, attack_y = ecdf(
        attack_errors
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        benign_x,
        benign_y,
        label="BENIGN validation",
    )

    ax.plot(
        attack_x,
        attack_y,
        label=attack_name,
    )

    ax.set_title(
        f"Client {client_id} — Reconstruction Error ECDF\n"
        f"Attack: {attack_name}"
    )

    ax.set_xlabel(
        "Reconstruction MSE"
    )

    ax.set_ylabel(
        "Empirical CDF"
    )

    ax.legend()

    fig.tight_layout()

    graph_file = (
        GRAPH_ROOT
        /
        f"client_{client_id}_error_ecdf.png"
    )

    fig.savefig(
        graph_file,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return graph_file


# ============================================================
# SAVE RAW ERRORS
# ============================================================

def save_raw_errors(
    client_id,
    attack_name,
    benign_errors,
    attack_errors,
):
    benign_df = pd.DataFrame(
        {
            "client_id": client_id,
            "seen_attack": attack_name,
            "group": "BENIGN_VALIDATION",
            "reconstruction_error": benign_errors,
        }
    )

    attack_df = pd.DataFrame(
        {
            "client_id": client_id,
            "seen_attack": attack_name,
            "group": "ATTACK",
            "reconstruction_error": attack_errors,
        }
    )

    output = pd.concat(
        [
            benign_df,
            attack_df,
        ],
        ignore_index=True,
    )

    output_file = (
        ERROR_ROOT
        /
        f"client_{client_id}_reconstruction_errors.csv"
    )

    output.to_csv(
        output_file,
        index=False,
    )

    return output_file


# ============================================================
# PROCESS ONE CLIENT
# ============================================================

def process_client(
    client_id,
    client_file,
    scaler,
    feature_columns,
    active_indices,
    active_features,
    label_mapping,
    device,
):
    separator()

    attack_name = CLIENT_ATTACKS[
        client_id
    ]

    print(
        f"CLIENT {client_id} — {attack_name}"
    )

    print("-" * 110)

    # --------------------------------------------------------
    # Load and filter data
    # --------------------------------------------------------
    dataframe = pd.read_csv(
        client_file
    )

    dataframe = filter_to_project_classes(
        dataframe,
        label_mapping,
    )

    benign_dataframe = (
        dataframe[
            dataframe[
                phase4.LABEL_COL
            ]
            == "BENIGN"
        ]
        .copy()
    )

    attack_dataframe = (
        dataframe[
            dataframe[
                phase4.LABEL_COL
            ]
            == attack_name
        ]
        .copy()
    )

    if len(benign_dataframe) == 0:
        raise ValueError(
            f"Client {client_id} has no BENIGN rows."
        )

    if len(attack_dataframe) == 0:
        raise ValueError(
            f"Client {client_id} has no {attack_name} rows."
        )

    # --------------------------------------------------------
    # Use EXACTLY the same frozen preprocessing as Phase 14B.
    # --------------------------------------------------------
    X_benign = phase4.transform_features(
        df=benign_dataframe,
        feature_columns=feature_columns,
        scaler=scaler,
        active_indices=active_indices,
    ).astype(
        np.float32,
        copy=False,
    )

    X_attack = phase4.transform_features(
        df=attack_dataframe,
        feature_columns=feature_columns,
        scaler=scaler,
        active_indices=active_indices,
    ).astype(
        np.float32,
        copy=False,
    )

    if X_benign.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            "Unexpected benign feature count."
        )

    if X_attack.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            "Unexpected attack feature count."
        )

    # --------------------------------------------------------
    # Reproduce the exact Phase 14B BENIGN split.
    #
    # Only the held-out BENIGN validation subset is used for
    # threshold calibration.
    # --------------------------------------------------------
    (
        _X_benign_train,
        X_benign_validation,
    ) = train_test_split(
        X_benign,
        test_size=BENIGN_VALIDATION_RATIO,
        random_state=SEED + client_id,
        shuffle=True,
    )

    # --------------------------------------------------------
    # Load existing Phase 14B model.
    # NO RETRAINING.
    # --------------------------------------------------------
    (
        model,
        checkpoint,
        model_file,
    ) = load_client_autoencoder(
        client_id=client_id,
        device=device,
    )

    print(
        f"Model          : {model_file}"
    )

    print(
        f"BENIGN val rows: {len(X_benign_validation):,}"
    )

    print(
        f"Attack rows    : {len(X_attack):,}"
    )

    # --------------------------------------------------------
    # Reconstruction errors
    # --------------------------------------------------------
    benign_errors = reconstruction_errors(
        model=model,
        X=X_benign_validation,
        device=device,
    )

    attack_errors = reconstruction_errors(
        model=model,
        X=X_attack,
        device=device,
    )

    # --------------------------------------------------------
    # Global ranking metrics.
    # These do not depend on one threshold.
    # --------------------------------------------------------
    y_true = np.concatenate(
        [
            np.zeros(
                len(benign_errors),
                dtype=np.int64,
            ),
            np.ones(
                len(attack_errors),
                dtype=np.int64,
            ),
        ]
    )

    scores = np.concatenate(
        [
            benign_errors,
            attack_errors,
        ]
    )

    roc_auc = float(
        roc_auc_score(
            y_true,
            scores,
        )
    )

    pr_auc = float(
        average_precision_score(
            y_true,
            scores,
        )
    )

    # --------------------------------------------------------
    # Thresholds
    # --------------------------------------------------------
    thresholds = {}

    thresholds.update(
        benign_percentile_thresholds(
            benign_errors
        )
    )

    thresholds.update(
        mad_threshold(
            benign_errors
        )
    )

    if INCLUDE_ORACLE_THRESHOLDS:
        thresholds.update(
            oracle_thresholds(
                benign_errors,
                attack_errors,
            )
        )

    comparison_rows = []

    for (
        threshold_name,
        threshold_info,
    ) in thresholds.items():

        result = evaluate_threshold(
            client_id=client_id,
            attack_name=attack_name,
            threshold_name=threshold_name,
            threshold_info=threshold_info,
            benign_errors=benign_errors,
            attack_errors=attack_errors,
        )

        comparison_rows.append(
            result
        )

    # --------------------------------------------------------
    # Distribution summaries
    # --------------------------------------------------------
    distribution_rows = [
        distribution_summary(
            client_id,
            attack_name,
            "BENIGN_VALIDATION",
            benign_errors,
        ),
        distribution_summary(
            client_id,
            attack_name,
            "ATTACK",
            attack_errors,
        ),
    ]

    # --------------------------------------------------------
    # Save raw errors and figures
    # --------------------------------------------------------
    raw_error_file = save_raw_errors(
        client_id,
        attack_name,
        benign_errors,
        attack_errors,
    )

    histogram_file = save_histogram(
        client_id,
        attack_name,
        benign_errors,
        attack_errors,
        thresholds,
    )

    ecdf_file = save_ecdf(
        client_id,
        attack_name,
        benign_errors,
        attack_errors,
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------
    print()
    print(
        f"Mean BENIGN error : {np.mean(benign_errors):.8f}"
    )

    print(
        f"Mean ATTACK error : {np.mean(attack_errors):.8f}"
    )

    print(
        f"Median BENIGN     : {np.median(benign_errors):.8f}"
    )

    print(
        f"Median ATTACK     : {np.median(attack_errors):.8f}"
    )

    print(
        f"ROC-AUC           : {roc_auc:.6f}"
    )

    print(
        f"PR-AUC            : {pr_auc:.6f}"
    )

    client_comparison = pd.DataFrame(
        comparison_rows
    )

    print()
    print(
        client_comparison[
            [
                "threshold_method",
                "threshold",
                "benign_fpr",
                "recall",
                "precision",
                "f1",
                "balanced_accuracy",
                "uses_attack_labels",
            ]
        ]
        .to_string(
            index=False
        )
    )

    client_summary = {
        "client_id": client_id,
        "seen_attack": attack_name,
        "benign_validation_rows": int(
            len(benign_errors)
        ),
        "attack_rows": int(
            len(attack_errors)
        ),
        "mean_benign_error": float(
            np.mean(benign_errors)
        ),
        "mean_attack_error": float(
            np.mean(attack_errors)
        ),
        "median_benign_error": float(
            np.median(benign_errors)
        ),
        "median_attack_error": float(
            np.median(attack_errors)
        ),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "phase14b_checkpoint_threshold": float(
            checkpoint.get(
                "reconstruction_threshold",
                np.nan,
            )
        ),
        "raw_error_file": str(
            raw_error_file
        ),
        "histogram_file": str(
            histogram_file
        ),
        "ecdf_file": str(
            ecdf_file
        ),
        "locked_test_used": False,
    }

    del model
    del dataframe
    del benign_dataframe
    del attack_dataframe
    del X_benign
    del X_attack
    del _X_benign_train
    del X_benign_validation
    del benign_errors
    del attack_errors

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        comparison_rows,
        distribution_rows,
        client_summary,
    )


# ============================================================
# FINAL REPORT
# ============================================================

def build_text_report(
    comparison_df,
    client_summary_df,
):
    lines = []

    lines.append("=" * 110)
    lines.append(PHASE_NAME)
    lines.append("=" * 110)
    lines.append("")
    lines.append("PURPOSE")
    lines.append("-" * 110)
    lines.append(
        "Diagnose whether poor Autoencoder attack recall is mainly caused by the fixed P99 threshold "
        "or by weak reconstruction-error separation."
    )
    lines.append("")
    lines.append("IMPORTANT SCIENTIFIC CONTROL")
    lines.append("-" * 110)
    lines.append("1. Existing Phase 14B models are reused; there is NO retraining.")
    lines.append("2. Existing frozen 70-feature preprocessing is reused.")
    lines.append("3. Only held-out BENIGN validation errors are used for deployable threshold candidates.")
    lines.append("4. ORACLE_YOUDEN and ORACLE_BEST_F1 use attack labels and are DIAGNOSTIC ONLY.")
    lines.append("5. Locked test is NOT used.")
    lines.append("")
    lines.append("CLIENT-LEVEL RANKING QUALITY")
    lines.append("-" * 110)

    for _, row in client_summary_df.iterrows():
        lines.append(
            f"Client {int(row['client_id'])} | "
            f"{row['seen_attack']:<16} | "
            f"ROC-AUC={row['roc_auc']:.6f} | "
            f"PR-AUC={row['pr_auc']:.6f} | "
            f"Mean benign={row['mean_benign_error']:.6f} | "
            f"Mean attack={row['mean_attack_error']:.6f}"
        )

    lines.append("")
    lines.append("DEPLOYABLE BENIGN-ONLY THRESHOLDS")
    lines.append("-" * 110)

    deployable = comparison_df[
        comparison_df["uses_attack_labels"] == False
    ].copy()

    for client_id in sorted(
        deployable["client_id"].unique()
    ):
        subset = deployable[
            deployable["client_id"]
            == client_id
        ].sort_values(
            "threshold"
        )

        attack = subset.iloc[0][
            "seen_attack"
        ]

        lines.append("")
        lines.append(
            f"Client {client_id} — {attack}"
        )

        for _, row in subset.iterrows():
            lines.append(
                f"  {row['threshold_method']:<8} "
                f"threshold={row['threshold']:.6f} | "
                f"FPR={row['benign_fpr']:.4f} | "
                f"Recall={row['recall']:.4f} | "
                f"Precision={row['precision']:.4f} | "
                f"F1={row['f1']:.4f} | "
                f"BalAcc={row['balanced_accuracy']:.4f}"
            )

    if INCLUDE_ORACLE_THRESHOLDS:
        lines.append("")
        lines.append("DIAGNOSTIC ORACLE THRESHOLDS — NOT FOR DEPLOYMENT")
        lines.append("-" * 110)

        oracle = comparison_df[
            comparison_df["uses_attack_labels"] == True
        ].copy()

        for client_id in sorted(
            oracle["client_id"].unique()
        ):
            subset = oracle[
                oracle["client_id"]
                == client_id
            ]

            attack = subset.iloc[0][
                "seen_attack"
            ]

            lines.append("")
            lines.append(
                f"Client {client_id} — {attack}"
            )

            for _, row in subset.iterrows():
                lines.append(
                    f"  {row['threshold_method']:<18} "
                    f"threshold={row['threshold']:.6f} | "
                    f"FPR={row['benign_fpr']:.4f} | "
                    f"Recall={row['recall']:.4f} | "
                    f"F1={row['f1']:.4f}"
                )

    lines.append("")
    lines.append("INTERPRETATION GUIDE")
    lines.append("-" * 110)
    lines.append(
        "A. If P95/P97/P98 greatly improve recall while ROC-AUC is already high, "
        "the major issue is threshold strictness / calibration."
    )
    lines.append(
        "B. If even oracle thresholds cannot achieve useful recall and precision, "
        "the reconstruction representation itself is weak for that attack family."
    )
    lines.append(
        "C. If attack reconstruction errors are systematically LOWER than benign errors, "
        "one-sided 'high reconstruction error = attack' is fundamentally unsuitable for that attack."
    )
    lines.append(
        "D. In Phase 18, the Autoencoder should remain one uncertainty signal rather than the sole final classifier."
    )
    lines.append("")
    lines.append("LOCKED TEST USED : NO")
    lines.append("=" * 110)

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(SEED)

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    GRAPH_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    ERROR_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    separator()
    print(PHASE_NAME)
    separator()

    print(
        f"Scenario                 : {SCENARIO}"
    )

    print(
        f"Clients                  : {N_CLIENTS}"
    )

    print(
        f"Device                   : {device}"
    )

    print(
        f"Input features           : {EXPECTED_FEATURES}"
    )

    print(
        f"Percentile thresholds    : {PERCENTILE_THRESHOLDS}"
    )

    print(
        f"MAD threshold            : median + {MAD_K} * 1.4826 * MAD"
    )

    print(
        f"Oracle diagnostics       : {'YES' if INCLUDE_ORACLE_THRESHOLDS else 'NO'}"
    )

    print()
    print(
        "NO MODEL RETRAINING."
    )

    print(
        "LOCKED TEST IS NOT USED."
    )

    print(
        "ORACLE thresholds use attack labels and are diagnostic only."
    )

    # --------------------------------------------------------
    # Load frozen preprocessing
    # --------------------------------------------------------
    (
        scaler,
        feature_columns,
        label_mapping,
        active_indices,
        active_features,
    ) = phase4.load_preprocessing()

    active_indices = verify_preprocessing(
        active_features=active_features,
        active_indices=active_indices,
        feature_columns=feature_columns,
    )

    # --------------------------------------------------------
    # Client directory
    # --------------------------------------------------------
    federated_dir = (
        phase4.FEDERATED_ROOT
        /
        SCENARIO
    )

    if not federated_dir.exists():
        raise FileNotFoundError(
            "\nFederated directory missing:\n"
            f"{federated_dir}"
        )

    all_comparison_rows = []
    all_distribution_rows = []
    all_client_summaries = []

    for client_id in range(
        1,
        N_CLIENTS + 1,
    ):
        client_file = (
            federated_dir
            /
            f"client_{client_id}.csv"
        )

        if not client_file.exists():
            raise FileNotFoundError(
                "\nMissing client file:\n"
                f"{client_file}"
            )

        (
            comparison_rows,
            distribution_rows,
            client_summary,
        ) = process_client(
            client_id=client_id,
            client_file=client_file,
            scaler=scaler,
            feature_columns=feature_columns,
            active_indices=active_indices,
            active_features=active_features,
            label_mapping=label_mapping,
            device=device,
        )

        all_comparison_rows.extend(
            comparison_rows
        )

        all_distribution_rows.extend(
            distribution_rows
        )

        all_client_summaries.append(
            client_summary
        )

    # --------------------------------------------------------
    # Save CSV outputs
    # --------------------------------------------------------
    comparison_df = pd.DataFrame(
        all_comparison_rows
    )

    distribution_df = pd.DataFrame(
        all_distribution_rows
    )

    client_summary_df = pd.DataFrame(
        all_client_summaries
    )

    comparison_df.to_csv(
        THRESHOLD_COMPARISON_FILE,
        index=False,
    )

    distribution_df.to_csv(
        ERROR_SUMMARY_FILE,
        index=False,
    )

    client_summary_df.to_csv(
        CLIENT_SUMMARY_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Save JSON summary
    # --------------------------------------------------------
    summary_payload = {
        "phase": "14B.1",
        "phase_name": PHASE_NAME,
        "scenario": SCENARIO,
        "seed": SEED,
        "clients": N_CLIENTS,
        "features": EXPECTED_FEATURES,
        "percentile_thresholds": list(
            PERCENTILE_THRESHOLDS
        ),
        "mad_k": MAD_K,
        "oracle_thresholds_included": (
            INCLUDE_ORACLE_THRESHOLDS
        ),
        "oracle_thresholds_are_diagnostic_only": True,
        "locked_test_used": False,
        "model_retraining": False,
        "threshold_comparison_file": str(
            THRESHOLD_COMPARISON_FILE
        ),
        "client_summary_file": str(
            CLIENT_SUMMARY_FILE
        ),
        "error_summary_file": str(
            ERROR_SUMMARY_FILE
        ),
        "graph_directory": str(
            GRAPH_ROOT
        ),
        "raw_error_directory": str(
            ERROR_ROOT
        ),
    }

    with open(
        SUMMARY_JSON_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary_payload,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # Final text report
    # --------------------------------------------------------
    report_text = build_text_report(
        comparison_df,
        client_summary_df,
    )

    with open(
        FINAL_REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            report_text
        )

    # --------------------------------------------------------
    # Final console summary
    # --------------------------------------------------------
    separator()
    print("PHASE 14B.1 COMPLETE")
    separator()

    print()
    print(
        "CLIENT RANKING QUALITY"
    )

    print("-" * 110)

    print(
        client_summary_df[
            [
                "client_id",
                "seen_attack",
                "mean_benign_error",
                "mean_attack_error",
                "roc_auc",
                "pr_auc",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        "P99 BASELINE"
    )

    print("-" * 110)

    p99 = comparison_df[
        comparison_df[
            "threshold_method"
        ]
        == "P99"
    ]

    print(
        p99[
            [
                "client_id",
                "seen_attack",
                "threshold",
                "benign_fpr",
                "recall",
                "precision",
                "f1",
                "balanced_accuracy",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "P95/P97/P98/P99 and MAD are benign-only candidate thresholds."
    )

    print(
        "ORACLE_YOUDEN and ORACLE_BEST_F1 use attack labels and are diagnostic only."
    )

    print()
    print(
        f"Threshold comparison : {THRESHOLD_COMPARISON_FILE}"
    )

    print(
        f"Client summary       : {CLIENT_SUMMARY_FILE}"
    )

    print(
        f"Error summary        : {ERROR_SUMMARY_FILE}"
    )

    print(
        f"Graphs               : {GRAPH_ROOT}"
    )

    print(
        f"Raw errors           : {ERROR_ROOT}"
    )

    print(
        f"Text report          : {FINAL_REPORT_FILE}"
    )

    print(
        f"JSON summary         : {SUMMARY_JSON_FILE}"
    )

    print()
    print(
        "LOCKED TEST USED : NO"
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()
