from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .phase16d_gnn import (
    AlertCorrelationGCN,
    CorrelationGraphDataset,
    collate_graphs,
    rule_baseline_prediction,
    set_all_seeds,
)
from .phase16f_rule_gap import (
    generate_rule_gap_benchmark,
)
from .rule_temporal_correlator import CorrelationConfig


PHASE16G_VERSION = "phase16g_v1"


@dataclass(frozen=True)
class SeedConfig:
    seed: int
    train_graphs: int
    validation_graphs: int
    test_graphs: int
    hidden_dim: int
    dropout: float
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    patience: int
    rule_positive_fraction_of_positives: float
    hybrid_threshold_candidates: Tuple[float, ...]
    hybrid_min_precision: float


def _confusion_counts(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> Tuple[int, int, int, int]:
    tp = fp = tn = fn = 0

    for truth, pred in zip(
        y_true,
        y_pred,
    ):
        if truth == 1 and pred == 1:
            tp += 1
        elif truth == 0 and pred == 1:
            fp += 1
        elif truth == 0 and pred == 0:
            tn += 1
        elif truth == 1 and pred == 0:
            fn += 1

    return tp, fp, tn, fn


def binary_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    scores: Sequence[float] | None = None,
) -> Dict[str, Any]:
    tp, fp, tn, fn = _confusion_counts(
        y_true,
        y_pred,
    )

    total = max(
        tp + fp + tn + fn,
        1,
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp)
        else 0.0
    )

    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    negative_precision = (
        tn / (tn + fn)
        if (tn + fn)
        else 0.0
    )

    negative_recall = specificity

    negative_f1 = (
        2.0
        * negative_precision
        * negative_recall
        /
        (
            negative_precision
            +
            negative_recall
        )
        if (
            negative_precision
            +
            negative_recall
        )
        else 0.0
    )

    macro_f1 = (
        f1
        +
        negative_f1
    ) / 2.0

    balanced_accuracy = (
        recall
        +
        specificity
    ) / 2.0

    result: Dict[str, Any] = {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": (
            tp + tn
        ) / total,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "macro_f1": macro_f1,
        "false_positive_rate": (
            fp / (fp + tn)
            if (fp + tn)
            else 0.0
        ),
        "false_negative_rate": (
            fn / (fn + tp)
            if (fn + tp)
            else 0.0
        ),
        "confusion_matrix": [
            [tn, fp],
            [fn, tp],
        ],
    }

    if (
        scores is not None
        and
        len(
            set(
                y_true
            )
        )
        ==
        2
    ):
        # Dependency-light AUROC implementation using rank statistics.
        positives = [
            float(
                score
            )
            for truth, score
            in zip(
                y_true,
                scores,
            )
            if truth == 1
        ]

        negatives = [
            float(
                score
            )
            for truth, score
            in zip(
                y_true,
                scores,
            )
            if truth == 0
        ]

        wins = 0.0

        for p in positives:
            for n in negatives:
                if p > n:
                    wins += 1.0
                elif p == n:
                    wins += 0.5

        denominator = (
            len(
                positives
            )
            *
            len(
                negatives
            )
        )

        result[
            "roc_auc"
        ] = (
            wins
            /
            denominator
            if denominator
            else None
        )

    else:
        result[
            "roc_auc"
        ] = None

    return result


def rule_gap_recall(
    *,
    subtypes: Sequence[str],
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> float:
    indices = [
        i
        for i, (
            subtype,
            truth,
        )
        in enumerate(
            zip(
                subtypes,
                y_true,
            )
        )
        if (
            truth == 1
            and
            subtype.startswith(
                "RULE_GAP_"
            )
        )
    ]

    if not indices:
        return 0.0

    hits = sum(
        1
        for i
        in indices
        if y_pred[
            i
        ]
        ==
        1
    )

    return float(
        hits
        /
        len(
            indices
        )
    )


@torch.no_grad()
def predict_gnn(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()

    y_true: List[int] = []
    y_pred: List[int] = []
    probabilities: List[float] = []
    subtypes: List[str] = []
    alert_batches: List[Any] = []

    for batch in loader:
        x = batch[
            "x"
        ].to(
            device
        )

        adjacency = batch[
            "adjacency"
        ].to(
            device
        )

        mask = batch[
            "mask"
        ].to(
            device
        )

        logits = model(
            x,
            adjacency,
            mask,
        )

        probs = torch.softmax(
            logits,
            dim=1,
        )[
            :,
            1
        ]

        pred = torch.argmax(
            logits,
            dim=1,
        )

        y_true.extend(
            batch[
                "labels"
            ].tolist()
        )

        y_pred.extend(
            pred.cpu().tolist()
        )

        probabilities.extend(
            probs.cpu().tolist()
        )

        subtypes.extend(
            batch[
                "subtypes"
            ]
        )

        alert_batches.extend(
            batch[
                "alerts"
            ]
        )

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "probabilities": probabilities,
        "subtypes": subtypes,
        "alerts": alert_batches,
    }


def choose_hybrid_threshold_validation_only(
    *,
    y_true: Sequence[int],
    rule_pred: Sequence[int],
    gnn_probabilities: Sequence[float],
    candidates: Sequence[float],
    min_precision: float,
) -> Dict[str, Any]:
    """
    Validation-only threshold selection.

    Selection order:
      1. precision >= min_precision
      2. highest Macro F1
      3. highest precision
      4. highest recall
      5. highest threshold (conservative tie-break)

    Test labels/probabilities are never used here.
    """
    rows = []

    for threshold in candidates:
        hybrid = [
            1
            if (
                rule_value == 1
                or
                probability
                >=
                threshold
            )
            else
            0
            for rule_value, probability
            in zip(
                rule_pred,
                gnn_probabilities,
            )
        ]

        metrics = binary_metrics(
            y_true,
            hybrid,
            [
                max(
                    float(
                        rule_value
                    ),
                    float(
                        probability
                    ),
                )
                for rule_value, probability
                in zip(
                    rule_pred,
                    gnn_probabilities,
                )
            ],
        )

        rows.append(
            {
                "threshold": float(
                    threshold
                ),
                **metrics,
            }
        )

    eligible = [
        row
        for row
        in rows
        if row[
            "precision"
        ]
        >=
        min_precision
    ]

    if not eligible:
        # Safe fallback: choose highest threshold among candidates.
        selected = max(
            rows,
            key=lambda row: row[
                "threshold"
            ],
        )

        reason = (
            "No validation threshold met the minimum precision constraint; "
            "selected the most conservative highest threshold."
        )

    else:
        selected = sorted(
            eligible,
            key=lambda row: (
                -row[
                    "macro_f1"
                ],
                -row[
                    "precision"
                ],
                -row[
                    "recall"
                ],
                -row[
                    "threshold"
                ],
            ),
        )[
            0
        ]

        reason = (
            "Selected using validation data only: precision constraint first, "
            "then Macro F1, precision, recall, and conservative threshold tie-break."
        )

    return {
        "selected_threshold": selected[
            "threshold"
        ],
        "selected_validation_metrics": selected,
        "all_candidates": rows,
        "selection_reason": reason,
        "test_data_used_for_selection": False,
    }


def train_one_seed(
    *,
    config: SeedConfig,
    rule_config: CorrelationConfig,
    device: torch.device,
) -> Dict[str, Any]:
    set_all_seeds(
        config.seed
    )

    train_seed = (
        161000
        +
        config.seed
        *
        10
        +
        1
    )

    validation_seed = (
        161000
        +
        config.seed
        *
        10
        +
        2
    )

    test_seed = (
        161000
        +
        config.seed
        *
        10
        +
        3
    )

    train_scenarios = generate_rule_gap_benchmark(
        total_graphs=config.train_graphs,
        seed=train_seed,
        window_seconds=rule_config.window_seconds,
        rule_positive_fraction_of_positives=
            config.rule_positive_fraction_of_positives,
    )

    validation_scenarios = generate_rule_gap_benchmark(
        total_graphs=config.validation_graphs,
        seed=validation_seed,
        window_seconds=rule_config.window_seconds,
        rule_positive_fraction_of_positives=
            config.rule_positive_fraction_of_positives,
    )

    test_scenarios = generate_rule_gap_benchmark(
        total_graphs=config.test_graphs,
        seed=test_seed,
        window_seconds=rule_config.window_seconds,
        rule_positive_fraction_of_positives=
            config.rule_positive_fraction_of_positives,
    )

    train_ds = CorrelationGraphDataset(
        train_scenarios
    )

    validation_ds = CorrelationGraphDataset(
        validation_scenarios
    )

    test_ds = CorrelationGraphDataset(
        test_scenarios
    )

    generator = torch.Generator()
    generator.manual_seed(
        config.seed
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_graphs,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
    )

    feature_dim = int(
        train_ds[
            0
        ].x.shape[
            1
        ]
    )

    model = AlertCorrelationGCN(
        input_dim=feature_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_epoch = 0
    best_validation_macro_f1 = -1.0
    patience_counter = 0
    history = []

    for epoch in range(
        1,
        config.epochs
        +
        1,
    ):
        model.train()

        running_loss = 0.0
        sample_count = 0

        for batch in train_loader:
            optimizer.zero_grad(
                set_to_none=True
            )

            labels = batch[
                "labels"
            ].to(
                device
            )

            logits = model(
                batch[
                    "x"
                ].to(
                    device
                ),
                batch[
                    "adjacency"
                ].to(
                    device
                ),
                batch[
                    "mask"
                ].to(
                    device
                ),
            )

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()
            optimizer.step()

            n = int(
                labels.shape[
                    0
                ]
            )

            running_loss += float(
                loss.item()
            ) * n

            sample_count += n

        validation_pred = predict_gnn(
            model,
            validation_loader,
            device,
        )

        validation_metrics = binary_metrics(
            validation_pred[
                "y_true"
            ],
            validation_pred[
                "y_pred"
            ],
            validation_pred[
                "probabilities"
            ],
        )

        train_loss = (
            running_loss
            /
            max(
                sample_count,
                1,
            )
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_macro_f1":
                    validation_metrics[
                        "macro_f1"
                    ],
            }
        )

        if (
            validation_metrics[
                "macro_f1"
            ]
            >
            best_validation_macro_f1
            +
            1e-12
        ):
            best_validation_macro_f1 = float(
                validation_metrics[
                    "macro_f1"
                ]
            )

            best_epoch = epoch

            best_state = {
                key:
                    value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            patience_counter = 0

        else:
            patience_counter += 1

        if (
            patience_counter
            >=
            config.patience
        ):
            break

    if best_state is None:
        raise RuntimeError(
            "No valid GNN state selected."
        )

    model.load_state_dict(
        best_state
    )

    # ------------------------------------------------------------
    # Validation-only hybrid threshold selection
    # ------------------------------------------------------------
    validation_pred = predict_gnn(
        model,
        validation_loader,
        device,
    )

    validation_rule = [
        rule_baseline_prediction(
            alerts,
            rule_config,
        )
        for alerts
        in validation_pred[
            "alerts"
        ]
    ]

    threshold_selection = choose_hybrid_threshold_validation_only(
        y_true=validation_pred[
            "y_true"
        ],
        rule_pred=validation_rule,
        gnn_probabilities=validation_pred[
            "probabilities"
        ],
        candidates=config.hybrid_threshold_candidates,
        min_precision=config.hybrid_min_precision,
    )

    selected_threshold = float(
        threshold_selection[
            "selected_threshold"
        ]
    )

    # ------------------------------------------------------------
    # Test exactly once after threshold freeze
    # ------------------------------------------------------------
    test_pred = predict_gnn(
        model,
        test_loader,
        device,
    )

    y_true = test_pred[
        "y_true"
    ]

    gnn_y = test_pred[
        "y_pred"
    ]

    probabilities = test_pred[
        "probabilities"
    ]

    subtypes = test_pred[
        "subtypes"
    ]

    rule_y = [
        rule_baseline_prediction(
            alerts,
            rule_config,
        )
        for alerts
        in test_pred[
            "alerts"
        ]
    ]

    hybrid_y = [
        1
        if (
            rule_value == 1
            or
            probability
            >=
            selected_threshold
        )
        else
        0
        for rule_value, probability
        in zip(
            rule_y,
            probabilities,
        )
    ]

    rule_metrics = binary_metrics(
        y_true,
        rule_y,
        [
            float(
                value
            )
            for value
            in rule_y
        ],
    )

    gnn_metrics = binary_metrics(
        y_true,
        gnn_y,
        probabilities,
    )

    hybrid_metrics = binary_metrics(
        y_true,
        hybrid_y,
        [
            max(
                float(
                    rule_value
                ),
                float(
                    probability
                ),
            )
            for rule_value, probability
            in zip(
                rule_y,
                probabilities,
            )
        ],
    )

    for name, predictions, metrics in (
        (
            "rule",
            rule_y,
            rule_metrics,
        ),
        (
            "gnn",
            gnn_y,
            gnn_metrics,
        ),
        (
            "hybrid",
            hybrid_y,
            hybrid_metrics,
        ),
    ):
        metrics[
            "rule_gap_recall"
        ] = rule_gap_recall(
            subtypes=subtypes,
            y_true=y_true,
            y_pred=predictions,
        )

    return {
        "seed": config.seed,
        "dataset_seeds": {
            "train": train_seed,
            "validation": validation_seed,
            "test": test_seed,
        },
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_validation_macro_f1,
        "hybrid_threshold_selection": threshold_selection,
        "rule": rule_metrics,
        "gnn": gnn_metrics,
        "hybrid": hybrid_metrics,
        "history": history,
    }


def mean_std(
    values: Sequence[float],
) -> Dict[str, float]:
    array = np.asarray(
        values,
        dtype=float,
    )

    return {
        "mean": float(
            array.mean()
        ),
        "std": float(
            array.std(
                ddof=1
            )
            if len(
                array
            )
            >
            1
            else
            0.0
        ),
        "min": float(
            array.min()
        ),
        "max": float(
            array.max()
        ),
    }


def exact_sign_test(
    a: Sequence[float],
    b: Sequence[float],
    *,
    tolerance: float = 1e-12,
) -> Dict[str, Any]:
    wins = 0
    losses = 0
    ties = 0

    for left, right in zip(
        a,
        b,
    ):
        diff = float(
            left
            -
            right
        )

        if diff > tolerance:
            wins += 1
        elif diff < -tolerance:
            losses += 1
        else:
            ties += 1

    n = wins + losses

    if n == 0:
        p_value = 1.0

    else:
        k = min(
            wins,
            losses,
        )

        tail = sum(
            comb(
                n,
                i,
            )
            for i
            in range(
                0,
                k
                +
                1,
            )
        ) / (
            2
            **
            n
        )

        p_value = min(
            1.0,
            2.0
            *
            tail,
        )

    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "non_tied_pairs": n,
        "two_sided_exact_p": float(
            p_value
        ),
    }


def summarize_runs(
    runs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    methods = (
        "rule",
        "gnn",
        "hybrid",
    )

    metrics = (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "macro_f1",
        "false_positive_rate",
        "false_negative_rate",
        "rule_gap_recall",
    )

    summary: Dict[
        str,
        Any,
    ] = {}

    for method in methods:
        summary[
            method
        ] = {}

        for metric in metrics:
            values = [
                float(
                    run[
                        method
                    ][
                        metric
                    ]
                )
                for run
                in runs
            ]

            summary[
                method
            ][
                metric
            ] = mean_std(
                values
            )

    threshold_values = [
        float(
            run[
                "hybrid_threshold_selection"
            ][
                "selected_threshold"
            ]
        )
        for run
        in runs
    ]

    summary[
        "hybrid_thresholds"
    ] = {
        "values": threshold_values,
        "mean_std": mean_std(
            threshold_values
        ),
    }

    def vector(
        method: str,
        metric: str,
    ) -> List[float]:
        return [
            float(
                run[
                    method
                ][
                    metric
                ]
            )
            for run
            in runs
        ]

    summary[
        "paired_tests"
    ] = {
        "gnn_vs_rule_macro_f1":
            exact_sign_test(
                vector(
                    "gnn",
                    "macro_f1",
                ),
                vector(
                    "rule",
                    "macro_f1",
                ),
            ),

        "hybrid_vs_rule_macro_f1":
            exact_sign_test(
                vector(
                    "hybrid",
                    "macro_f1",
                ),
                vector(
                    "rule",
                    "macro_f1",
                ),
            ),

        "gnn_vs_rule_rule_gap_recall":
            exact_sign_test(
                vector(
                    "gnn",
                    "rule_gap_recall",
                ),
                vector(
                    "rule",
                    "rule_gap_recall",
                ),
            ),

        "hybrid_vs_rule_rule_gap_recall":
            exact_sign_test(
                vector(
                    "hybrid",
                    "rule_gap_recall",
                ),
                vector(
                    "rule",
                    "rule_gap_recall",
                ),
            ),

        "hybrid_vs_gnn_macro_f1":
            exact_sign_test(
                vector(
                    "hybrid",
                    "macro_f1",
                ),
                vector(
                    "gnn",
                    "macro_f1",
                ),
            ),
    }

    return summary


def decide_dual_correlation_evidence(
    *,
    summary: Mapping[str, Any],
    min_macro_f1_gain: float,
    min_rule_gap_recall_gain: float,
    min_precision: float,
    alpha: float,
) -> Dict[str, Any]:
    """
    Predeclared research conclusion for this synthetic benchmark.

    This does NOT change the frozen deterministic rule contract.
    It decides only whether repeated-seed evidence supports keeping a GNN
    correlation branch for rule-gap campaigns.
    """
    rule_macro = float(
        summary[
            "rule"
        ][
            "macro_f1"
        ][
            "mean"
        ]
    )

    gnn_macro = float(
        summary[
            "gnn"
        ][
            "macro_f1"
        ][
            "mean"
        ]
    )

    rule_gap = float(
        summary[
            "rule"
        ][
            "rule_gap_recall"
        ][
            "mean"
        ]
    )

    gnn_gap = float(
        summary[
            "gnn"
        ][
            "rule_gap_recall"
        ][
            "mean"
        ]
    )

    gnn_precision = float(
        summary[
            "gnn"
        ][
            "precision"
        ][
            "mean"
        ]
    )

    macro_test = summary[
        "paired_tests"
    ][
        "gnn_vs_rule_macro_f1"
    ]

    gap_test = summary[
        "paired_tests"
    ][
        "gnn_vs_rule_rule_gap_recall"
    ]

    qualifies = (
        (
            gnn_macro
            -
            rule_macro
        )
        >=
        min_macro_f1_gain
        and
        (
            gnn_gap
            -
            rule_gap
        )
        >=
        min_rule_gap_recall_gain
        and
        gnn_precision
        >=
        min_precision
        and
        float(
            macro_test[
                "two_sided_exact_p"
            ]
        )
        <
        alpha
        and
        float(
            gap_test[
                "two_sided_exact_p"
            ]
        )
        <
        alpha
    )

    return {
        "dual_correlation_evidence_supported":
            bool(
                qualifies
            ),

        "recommended_phase16_research_architecture":
            (
                "FROZEN_RULES_PLUS_GNN_RULE_GAP_BRANCH"
                if qualifies
                else
                "FROZEN_RULES_ONLY_PENDING_MORE_EVIDENCE"
            ),

        "rule_macro_f1_mean":
            rule_macro,

        "gnn_macro_f1_mean":
            gnn_macro,

        "macro_f1_gain":
            gnn_macro
            -
            rule_macro,

        "rule_gap_recall_mean_rule":
            rule_gap,

        "rule_gap_recall_mean_gnn":
            gnn_gap,

        "rule_gap_recall_gain":
            gnn_gap
            -
            rule_gap,

        "gnn_precision_mean":
            gnn_precision,

        "macro_f1_p_value":
            float(
                macro_test[
                    "two_sided_exact_p"
                ]
            ),

        "rule_gap_recall_p_value":
            float(
                gap_test[
                    "two_sided_exact_p"
                ]
            ),

        "criteria": {
            "minimum_macro_f1_gain":
                min_macro_f1_gain,

            "minimum_rule_gap_recall_gain":
                min_rule_gap_recall_gain,

            "minimum_gnn_precision":
                min_precision,

            "significance_alpha":
                alpha,
        },

        "scope":
            (
                "Controlled synthetic privacy-safe rule-gap benchmark only. "
                "No production or real-world superiority claim."
            ),
    }
