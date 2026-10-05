from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def class_names_from_mapping(
    label_mapping: Mapping[str, int],
) -> list[str]:
    inverse = {
        int(index): str(label)
        for label, index
        in label_mapping.items()
    }

    expected = list(
        range(
            len(
                inverse
            )
        )
    )

    if sorted(
        inverse.keys()
    ) != expected:
        raise ValueError(
            "Label mapping ids must be contiguous from 0."
        )

    return [
        inverse[
            index
        ]
        for index
        in expected
    ]


def calculate_robustness_metrics(
    *,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_mapping: Mapping[str, int],
    benign_label: str = "BENIGN",
) -> Dict[str, Any]:
    class_names = class_names_from_mapping(
        label_mapping
    )

    labels = list(
        range(
            len(
                class_names
            )
        )
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    overall = {
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
        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
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
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "validation_samples": int(
            len(
                y_true
            )
        ),
    }

    per_class = {}

    for class_name in class_names:
        entry = report[
            class_name
        ]

        per_class[
            class_name
        ] = {
            "precision": float(
                entry[
                    "precision"
                ]
            ),
            "recall": float(
                entry[
                    "recall"
                ]
            ),
            "f1": float(
                entry[
                    "f1-score"
                ]
            ),
            "support": int(
                entry[
                    "support"
                ]
            ),
        }

    recalls = [
        per_class[
            class_name
        ][
            "recall"
        ]
        for class_name
        in class_names
    ]

    attack_recalls = [
        per_class[
            class_name
        ][
            "recall"
        ]
        for class_name
        in class_names
        if class_name
        !=
        benign_label
    ]

    overall[
        "worst_class_recall"
    ] = float(
        min(
            recalls
        )
        if recalls
        else 0.0
    )

    overall[
        "worst_attack_recall"
    ] = float(
        min(
            attack_recalls
        )
        if attack_recalls
        else 0.0
    )

    if benign_label not in label_mapping:
        raise KeyError(
            f"Benign label {benign_label!r} not found."
        )

    benign_id = int(
        label_mapping[
            benign_label
        ]
    )

    benign_support = int(
        cm[
            benign_id,
            :
        ].sum()
    )

    benign_correct = int(
        cm[
            benign_id,
            benign_id
        ]
    )

    # IDS false positive / false alarm rate:
    # benign traffic incorrectly classified as any attack.
    benign_false_alarms = (
        benign_support
        -
        benign_correct
    )

    overall[
        "benign_false_positive_rate"
    ] = float(
        benign_false_alarms
        /
        benign_support
        if benign_support
        else 0.0
    )

    return {
        "overall": overall,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }


def compare_to_clean(
    *,
    clean: Mapping[str, float],
    candidate: Mapping[str, float],
) -> Dict[str, float]:
    def drop(
        key: str,
    ) -> float:
        return float(
            clean[
                key
            ]
            -
            candidate[
                key
            ]
        )

    macro_clean = float(
        clean[
            "macro_f1"
        ]
    )

    macro_drop = drop(
        "macro_f1"
    )

    return {
        "accuracy_drop": drop(
            "accuracy"
        ),
        "balanced_accuracy_drop": drop(
            "balanced_accuracy"
        ),
        "macro_f1_drop": macro_drop,
        "macro_f1_relative_drop_pct": float(
            (
                macro_drop
                /
                macro_clean
                *
                100.0
            )
            if macro_clean
            else 0.0
        ),
        "weighted_f1_drop": drop(
            "weighted_f1"
        ),
        "worst_class_recall_drop": drop(
            "worst_class_recall"
        ),
        "worst_attack_recall_drop": drop(
            "worst_attack_recall"
        ),
        "benign_fpr_increase": float(
            candidate[
                "benign_false_positive_rate"
            ]
            -
            clean[
                "benign_false_positive_rate"
            ]
        ),
    }
