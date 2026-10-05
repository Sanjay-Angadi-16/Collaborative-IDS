from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

from .phase16d_gnn import (
    AlertCorrelationGCN,
    CorrelationGraphDataset,
    collate_graphs,
    generate_graph_scenarios,
    rule_baseline_prediction,
    set_all_seeds,
)
from .rule_temporal_correlator import CorrelationConfig


PHASE16E_VERSION = "phase16e_v1"


@dataclass(frozen=True)
class SeedRunConfig:
    seed: int
    train_graphs: int
    validation_graphs: int
    test_graphs: int
    epochs: int
    batch_size: int
    hidden_dim: int
    dropout: float
    learning_rate: float
    weight_decay: float
    patience: int
    hybrid_gnn_threshold: float


def binary_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    scores: Sequence[float] | None = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
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
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).tolist(),
    }

    tn, fp, fn, tp = np.asarray(
        result["confusion_matrix"],
        dtype=int,
    ).ravel()

    result["false_positive_rate"] = float(
        fp / (fp + tn)
        if (fp + tn)
        else 0.0
    )

    result["false_negative_rate"] = float(
        fn / (fn + tp)
        if (fn + tp)
        else 0.0
    )

    if (
        scores is not None
        and len(set(y_true)) == 2
    ):
        result["roc_auc"] = float(
            roc_auc_score(
                y_true,
                scores,
            )
        )
    else:
        result["roc_auc"] = None

    return result


@torch.no_grad()
def predict_gnn(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[int], List[float]]:
    model.eval()

    y_true: List[int] = []
    y_pred: List[int] = []
    probabilities: List[float] = []

    for batch in loader:
        x = batch["x"].to(
            device
        )

        adjacency = batch["adjacency"].to(
            device
        )

        mask = batch["mask"].to(
            device
        )

        labels = batch["labels"].to(
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
        )[:, 1]

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        y_true.extend(
            labels.cpu().tolist()
        )

        y_pred.extend(
            predictions.cpu().tolist()
        )

        probabilities.extend(
            probs.cpu().tolist()
        )

    return (
        y_true,
        y_pred,
        probabilities,
    )


def train_single_seed(
    *,
    config: SeedRunConfig,
    rule_config: CorrelationConfig,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Train one GNN seed and evaluate:
      1) frozen Phase16C rules
      2) GNN
      3) rule-first hybrid

    Hybrid policy is predeclared:
      - if frozen rules correlate -> positive
      - otherwise GNN may rescue only when P(correlated) >= threshold

    This preserves deterministic rule positives and allows the GNN only to add
    high-confidence positives where the frozen rule engine is silent.
    """
    set_all_seeds(
        config.seed
    )

    train_seed = (
        160000
        +
        config.seed
        *
        10
        +
        1
    )

    validation_seed = (
        160000
        +
        config.seed
        *
        10
        +
        2
    )

    test_seed = (
        160000
        +
        config.seed
        *
        10
        +
        3
    )

    train_scenarios = generate_graph_scenarios(
        count=config.train_graphs,
        seed=train_seed,
        window_seconds=rule_config.window_seconds,
    )

    validation_scenarios = generate_graph_scenarios(
        count=config.validation_graphs,
        seed=validation_seed,
        window_seconds=rule_config.window_seconds,
    )

    test_scenarios = generate_graph_scenarios(
        count=config.test_graphs,
        seed=test_seed,
        window_seconds=rule_config.window_seconds,
    )

    train_dataset = CorrelationGraphDataset(
        train_scenarios
    )

    validation_dataset = CorrelationGraphDataset(
        validation_scenarios
    )

    test_dataset = CorrelationGraphDataset(
        test_scenarios
    )

    generator = torch.Generator()
    generator.manual_seed(
        config.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_graphs,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
    )

    feature_dim = int(
        train_dataset[0].x.shape[1]
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
    history: List[Dict[str, Any]] = []

    for epoch in range(
        1,
        config.epochs + 1,
    ):
        model.train()

        running_loss = 0.0
        sample_count = 0

        for batch in train_loader:
            optimizer.zero_grad(
                set_to_none=True
            )

            x = batch["x"].to(
                device
            )

            adjacency = batch["adjacency"].to(
                device
            )

            mask = batch["mask"].to(
                device
            )

            labels = batch["labels"].to(
                device
            )

            logits = model(
                x,
                adjacency,
                mask,
            )

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()
            optimizer.step()

            batch_count = int(
                labels.shape[0]
            )

            running_loss += float(
                loss.item()
            ) * batch_count

            sample_count += batch_count

        train_loss = (
            running_loss
            /
            max(
                sample_count,
                1,
            )
        )

        val_true, val_pred, val_probs = predict_gnn(
            model,
            validation_loader,
            device,
        )

        val_metrics = binary_metrics(
            val_true,
            val_pred,
            val_probs,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_macro_f1": val_metrics[
                    "macro_f1"
                ],
            }
        )

        if (
            val_metrics["macro_f1"]
            >
            best_validation_macro_f1
            +
            1e-12
        ):
            best_validation_macro_f1 = float(
                val_metrics[
                    "macro_f1"
                ]
            )

            best_epoch = epoch

            best_state = {
                key: value.detach().cpu().clone()
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
            "No GNN checkpoint was selected."
        )

    model.load_state_dict(
        best_state
    )

    test_true, gnn_pred, gnn_probs = predict_gnn(
        model,
        test_loader,
        device,
    )

    gnn_metrics = binary_metrics(
        test_true,
        gnn_pred,
        gnn_probs,
    )

    rule_pred = [
        rule_baseline_prediction(
            scenario.alerts,
            rule_config,
        )
        for scenario
        in test_scenarios
    ]

    rule_metrics = binary_metrics(
        test_true,
        rule_pred,
        [
            float(
                value
            )
            for value
            in rule_pred
        ],
    )

    hybrid_pred = []

    for rule_value, gnn_probability in zip(
        rule_pred,
        gnn_probs,
    ):
        if rule_value == 1:
            hybrid_pred.append(
                1
            )
        else:
            hybrid_pred.append(
                1
                if gnn_probability
                >=
                config.hybrid_gnn_threshold
                else
                0
            )

    hybrid_metrics = binary_metrics(
        test_true,
        hybrid_pred,
        [
            max(
                float(rule_value),
                float(gnn_probability),
            )
            for rule_value, gnn_probability
            in zip(
                rule_pred,
                gnn_probs,
            )
        ],
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
            if len(array)
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


def paired_sign_test(
    a: Sequence[float],
    b: Sequence[float],
    *,
    tolerance: float = 1e-12,
) -> Dict[str, Any]:
    """
    Exact two-sided sign test for paired differences a - b.
    Ties are removed.
    """
    if len(a) != len(b):
        raise ValueError(
            "Paired arrays must have equal length."
        )

    positive = 0
    negative = 0
    ties = 0

    for left, right in zip(
        a,
        b,
    ):
        difference = float(
            left
            -
            right
        )

        if difference > tolerance:
            positive += 1

        elif difference < -tolerance:
            negative += 1

        else:
            ties += 1

    n = (
        positive
        +
        negative
    )

    if n == 0:
        p_value = 1.0

    else:
        k = min(
            positive,
            negative,
        )

        tail = sum(
            comb(
                n,
                i,
            )
            for i
            in range(
                0,
                k + 1,
            )
        ) / (
            2 ** n
        )

        p_value = min(
            1.0,
            2.0
            *
            tail,
        )

    return {
        "wins": positive,
        "losses": negative,
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

    rule_macro = [
        float(
            run["rule"]["macro_f1"]
        )
        for run
        in runs
    ]

    gnn_macro = [
        float(
            run["gnn"]["macro_f1"]
        )
        for run
        in runs
    ]

    hybrid_macro = [
        float(
            run["hybrid"]["macro_f1"]
        )
        for run
        in runs
    ]

    summary["paired_tests"] = {
        "gnn_vs_rule_macro_f1": paired_sign_test(
            gnn_macro,
            rule_macro,
        ),
        "hybrid_vs_rule_macro_f1": paired_sign_test(
            hybrid_macro,
            rule_macro,
        ),
        "hybrid_vs_gnn_macro_f1": paired_sign_test(
            hybrid_macro,
            gnn_macro,
        ),
    }

    return summary


def select_operational_policy(
    *,
    summary: Mapping[str, Any],
    minimum_material_gain: float,
    significance_alpha: float,
    max_precision_drop: float,
) -> Dict[str, Any]:
    """
    Predeclared conservative policy selection.

    Rule is incumbent because it was frozen in Phase16C.

    An alternative replaces it only if:
      - mean Macro F1 improves by at least minimum_material_gain
      - exact paired sign-test p < alpha
      - mean precision does not fall by more than max_precision_drop

    If neither alternative qualifies, keep RULE_ONLY.

    This selection is valid only for the synthetic Phase16 benchmark and must
    not be advertised as production superiority.
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

    rule_precision = float(
        summary[
            "rule"
        ][
            "precision"
        ][
            "mean"
        ]
    )

    candidates = []

    for method, test_key in (
        (
            "gnn",
            "gnn_vs_rule_macro_f1",
        ),
        (
            "hybrid",
            "hybrid_vs_rule_macro_f1",
        ),
    ):
        macro = float(
            summary[
                method
            ][
                "macro_f1"
            ][
                "mean"
            ]
        )

        precision = float(
            summary[
                method
            ][
                "precision"
            ][
                "mean"
            ]
        )

        gain = (
            macro
            -
            rule_macro
        )

        precision_drop = (
            rule_precision
            -
            precision
        )

        paired = summary[
            "paired_tests"
        ][
            test_key
        ]

        qualifies = (
            gain
            >=
            minimum_material_gain
            and
            float(
                paired[
                    "two_sided_exact_p"
                ]
            )
            <
            significance_alpha
            and
            precision_drop
            <=
            max_precision_drop
        )

        candidates.append(
            {
                "method": method,
                "mean_macro_f1": macro,
                "mean_precision": precision,
                "macro_f1_gain_vs_rule": gain,
                "precision_drop_vs_rule": precision_drop,
                "paired_sign_test": paired,
                "qualifies": qualifies,
            }
        )

    qualifying = [
        candidate
        for candidate
        in candidates
        if candidate[
            "qualifies"
        ]
    ]

    if qualifying:
        qualifying.sort(
            key=lambda item: (
                -item[
                    "mean_macro_f1"
                ],
                -item[
                    "mean_precision"
                ],
            )
        )

        selected = qualifying[
            0
        ][
            "method"
        ].upper()

        reason = (
            "Alternative met the predeclared material-gain, paired-significance, "
            "and precision-preservation criteria."
        )

    else:
        selected = "RULE_ONLY"

        reason = (
            "No alternative met all predeclared replacement criteria; the "
            "frozen Phase16C rule correlator remains the operational policy."
        )

    return {
        "selected_policy": selected,
        "reason": reason,
        "candidates": candidates,
        "selection_criteria": {
            "minimum_material_macro_f1_gain": minimum_material_gain,
            "significance_alpha": significance_alpha,
            "max_precision_drop": max_precision_drop,
        },
        "scope": (
            "Synthetic Phase16 correlation benchmark only; not a production "
            "or real-world superiority claim."
        ),
    }
