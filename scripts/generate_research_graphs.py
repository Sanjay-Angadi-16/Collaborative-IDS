"""
Research Graph Generator
========================

Generates publication-ready IDS experiment graphs.

Current support:
    Phase 3B - Centralized MLP

Later:
    Local Baseline
    FedAvg
    FedProx
    EVO
    GA
    EVO-GA

Run:
    python scripts/generate_research_graphs.py
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_ROOT = PROJECT_ROOT / "results"

CENTRALIZED_DIR = (
    RESULTS_ROOT
    / "centralized_baseline"
)

GRAPH_DIR = (
    RESULTS_ROOT
    / "graphs"
)

GRAPH_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# GRAPH SETTINGS
# ============================================================

DPI = 300


def save_graph(filename):

    output = GRAPH_DIR / filename

    plt.tight_layout()

    plt.savefig(
        output,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output}")


# ============================================================
# LOAD CENTRALIZED RESULTS
# ============================================================

def load_centralized():

    metrics_file = (
        CENTRALIZED_DIR
        / "overall_metrics.json"
    )

    class_file = (
        CENTRALIZED_DIR
        / "per_class_metrics.csv"
    )

    confusion_file = (
        CENTRALIZED_DIR
        / "confusion_matrix.csv"
    )

    history_file = (
        CENTRALIZED_DIR
        / "training_history.csv"
    )

    if not metrics_file.exists():

        raise FileNotFoundError(
            "Run Phase 3B first. Missing:\n"
            f"{metrics_file}"
        )

    with open(
        metrics_file,
        "r",
        encoding="utf-8"
    ) as f:

        metrics = json.load(f)

    per_class = pd.read_csv(
        class_file
    )

    confusion = pd.read_csv(
        confusion_file,
        index_col=0
    )

    history = pd.read_csv(
        history_file
    )

    return (
        metrics,
        per_class,
        confusion,
        history
    )


# ============================================================
# 1. OVERALL MODEL PERFORMANCE
# ============================================================

def graph_overall_performance(metrics):

    labels = [
        "Accuracy",
        "Balanced\nAccuracy",
        "Weighted F1",
        "Macro F1"
    ]

    values = [

        metrics["accuracy"] * 100,

        metrics[
            "balanced_accuracy"
        ] * 100,

        metrics[
            "weighted_f1"
        ] * 100,

        metrics[
            "macro_f1"
        ] * 100
    ]

    plt.figure(
        figsize=(10, 6)
    )

    bars = plt.bar(
        labels,
        values
    )

    plt.title(
        "Centralized MLP IDS Performance"
    )

    plt.ylabel(
        "Score (%)"
    )

    plt.xlabel(
        "Metric"
    )

    plt.ylim(
        0,
        100
    )

    plt.grid(
        axis="y",
        alpha=0.25
    )

    for bar, value in zip(
        bars,
        values
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,

            value + 1,

            f"{value:.2f}%",

            ha="center"
        )

    save_graph(
        "01_centralized_overall_performance.png"
    )


# ============================================================
# 2. PER CLASS F1
# ============================================================

def graph_per_class_f1(per_class):

    names = (
        per_class["class"]
        .tolist()
    )

    values = (
        per_class["f1_score"]
        .to_numpy()
        * 100
    )

    plt.figure(
        figsize=(18, 8)
    )

    bars = plt.bar(
        names,
        values
    )

    plt.title(
        "Centralized MLP Per-Class F1 Score"
    )

    plt.ylabel(
        "F1 Score (%)"
    )

    plt.xlabel(
        "Traffic / Attack Class"
    )

    plt.ylim(
        0,
        100
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.grid(
        axis="y",
        alpha=0.25
    )

    for bar, value in zip(
        bars,
        values
    ):

        plt.text(
            bar.get_x()
            + bar.get_width()/2,

            value + 1,

            f"{value:.1f}",

            ha="center",
            fontsize=8
        )

    save_graph(
        "02_per_class_f1.png"
    )


# ============================================================
# 3. PRECISION RECALL F1 BY CLASS
# ============================================================

def graph_per_class_metrics(per_class):

    classes = (
        per_class["class"]
        .tolist()
    )

    precision = (
        per_class["precision"]
        .to_numpy()
        * 100
    )

    recall = (
        per_class["recall"]
        .to_numpy()
        * 100
    )

    f1 = (
        per_class["f1_score"]
        .to_numpy()
        * 100
    )

    x = np.arange(
        len(classes)
    )

    width = 0.25

    plt.figure(
        figsize=(20, 9)
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
        label="F1 Score"
    )

    plt.title(
        "Centralized MLP Per-Class Performance"
    )

    plt.xlabel(
        "Traffic / Attack Class"
    )

    plt.ylabel(
        "Score (%)"
    )

    plt.xticks(
        x,
        classes,
        rotation=45,
        ha="right"
    )

    plt.ylim(
        0,
        100
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.25
    )

    save_graph(
        "03_per_class_precision_recall_f1.png"
    )


# ============================================================
# 4. CONFUSION MATRIX
# ============================================================

def graph_confusion_matrix(
    confusion
):

    # Clean names produced by Phase 3B
    row_names = [
        str(name).replace(
            "TRUE::",
            ""
        )
        for name
        in confusion.index
    ]

    column_names = [
        str(name).replace(
            "PRED::",
            ""
        )
        for name
        in confusion.columns
    ]

    cm = confusion.to_numpy()

    plt.figure(
        figsize=(15, 13)
    )

    image = plt.imshow(
        cm,
        interpolation="nearest",
        aspect="auto"
    )

    plt.colorbar(
        image
    )

    plt.title(
        "Centralized MLP Locked-Test Confusion Matrix"
    )

    plt.ylabel(
        "True Class"
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.xticks(
        range(len(column_names)),
        column_names,
        rotation=55,
        ha="right",
        fontsize=8
    )

    plt.yticks(
        range(len(row_names)),
        row_names,
        fontsize=8
    )

    # Only print cell value when non-zero.
    # This prevents the 15 x 15 matrix becoming unreadable.

    max_value = (
        cm.max()
        if cm.size
        else 1
    )

    threshold = (
        max_value * 0.5
    )

    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            value = cm[i, j]

            if value > 0:

                plt.text(
                    j,
                    i,
                    f"{int(value):,}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=(
                        "white"
                        if value > threshold
                        else "black"
                    )
                )

    save_graph(
        "04_confusion_matrix.png"
    )


# ============================================================
# 5. PREDICTION DISTRIBUTION
# ============================================================

def graph_prediction_distribution(
    confusion
):

    # Column sum = number of samples predicted
    # as each particular class.

    predicted_counts = (
        confusion.sum(
            axis=0
        )
    )

    names = [

        str(name).replace(
            "PRED::",
            ""
        )

        for name
        in predicted_counts.index
    ]

    values = (
        predicted_counts
        .to_numpy()
    )

    plt.figure(
        figsize=(18, 8)
    )

    bars = plt.bar(
        names,
        values
    )

    plt.title(
        "Centralized MLP Prediction Distribution"
    )

    plt.ylabel(
        "Predicted Samples"
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.grid(
        axis="y",
        alpha=0.25
    )

    for bar, value in zip(
        bars,
        values
    ):

        if value > 0:

            plt.text(
                bar.get_x()
                + bar.get_width()/2,

                value,

                f"{int(value):,}",

                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90
            )

    save_graph(
        "05_prediction_distribution.png"
    )


# ============================================================
# 6. TRAINING LOSS
# ============================================================

def graph_training_loss(
    history
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        history["epoch"],
        history["train_loss"],
        marker="o"
    )

    plt.title(
        "Centralized MLP Training Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Training Loss"
    )

    plt.xticks(
        history["epoch"]
    )

    plt.grid(
        alpha=0.25
    )

    save_graph(
        "06_training_loss.png"
    )


# ============================================================
# 7. TRAINING ACCURACY
# ============================================================

def graph_training_accuracy(
    history
):

    values = (
        history[
            "train_accuracy"
        ]
        * 100
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        history["epoch"],
        values,
        marker="o"
    )

    plt.title(
        "Centralized MLP Training Accuracy"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Training Accuracy (%)"
    )

    plt.ylim(
        0,
        100
    )

    plt.xticks(
        history["epoch"]
    )

    plt.grid(
        alpha=0.25
    )

    save_graph(
        "07_training_accuracy.png"
    )


# ============================================================
# 8. CLASS SUPPORT / IMBALANCE
# ============================================================

def graph_class_support(
    per_class
):

    classes = (
        per_class["class"]
        .tolist()
    )

    support = (
        per_class["support"]
        .to_numpy()
    )

    plt.figure(
        figsize=(18, 8)
    )

    plt.bar(
        classes,
        support
    )

    plt.title(
        "Locked Test Set Class Distribution"
    )

    plt.xlabel(
        "Traffic / Attack Class"
    )

    plt.ylabel(
        "Number of Samples"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.grid(
        axis="y",
        alpha=0.25
    )

    save_graph(
        "08_test_class_distribution.png"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("GENERATING IDS RESEARCH GRAPHS")
    print("=" * 80)

    (
        metrics,
        per_class,
        confusion,
        history
    ) = load_centralized()

    graph_overall_performance(
        metrics
    )

    graph_per_class_f1(
        per_class
    )

    graph_per_class_metrics(
        per_class
    )

    graph_confusion_matrix(
        confusion
    )

    graph_prediction_distribution(
        confusion
    )

    graph_training_loss(
        history
    )

    graph_training_accuracy(
        history
    )

    graph_class_support(
        per_class
    )

    print()
    print("=" * 80)
    print("GRAPH GENERATION COMPLETE")
    print("=" * 80)

    print(
        f"Graphs saved at:\n{GRAPH_DIR}"
    )


if __name__ == "__main__":
    main()