from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np


PHASE17E_ANALYSIS_VERSION = "phase17e_analysis_v1"


def mean_or_none(
    values: Sequence[float],
) -> float | None:
    if not values:
        return None

    return float(
        np.mean(
            np.asarray(
                values,
                dtype=np.float64,
            )
        )
    )


def target_class_summary(
    *,
    validation: Mapping[str, Any],
    malicious_clients: Sequence[int],
    client_attacks: Mapping[int, str],
) -> Dict[str, Any]:
    per_class = validation[
        "per_class"
    ]

    target_recalls = {}

    for client_id in malicious_clients:
        attack_name = str(
            client_attacks[
                int(
                    client_id
                )
            ]
        )

        if attack_name not in per_class:
            raise KeyError(
                f"Target class {attack_name!r} missing from validation metrics."
            )

        target_recalls[
            attack_name
        ] = float(
            per_class[
                attack_name
            ][
                "recall"
            ]
        )

    values = list(
        target_recalls.values()
    )

    return {
        "target_attack_recalls":
            target_recalls,
        "target_attack_mean_recall":
            mean_or_none(
                values
            ),
        "target_attack_worst_recall":
            (
                float(
                    min(
                        values
                    )
                )
                if values
                else None
            ),
    }


def build_identity_table(
    *,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []

    for row in rows:
        if (
            row[
                "malicious_ratio"
            ]
            !=
            0.20
        ):
            continue

        if len(
            row[
                "malicious_clients"
            ]
        ) != 1:
            continue

        output.append(
            {
                "attack_type":
                    row[
                        "attack_type"
                    ],
                "magnitude":
                    row[
                        "magnitude"
                    ],
                "malicious_client":
                    row[
                        "malicious_clients"
                    ][
                        0
                    ],
                "specialized_attack":
                    row[
                        "specialized_attacks"
                    ][
                        0
                    ],
                "macro_f1":
                    row[
                        "macro_f1"
                    ],
                "balanced_accuracy":
                    row[
                        "balanced_accuracy"
                    ],
                "weighted_f1":
                    row[
                        "weighted_f1"
                    ],
                "worst_attack_recall":
                    row[
                        "worst_attack_recall"
                    ],
                "target_attack_mean_recall":
                    row[
                        "target_attack_mean_recall"
                    ],
                "target_attack_worst_recall":
                    row[
                        "target_attack_worst_recall"
                    ],
                "macro_f1_drop_vs_clean":
                    row[
                        "macro_f1_drop_vs_clean"
                    ],
                "balanced_accuracy_drop_vs_clean":
                    row[
                        "balanced_accuracy_drop_vs_clean"
                    ],
                "attack_audit_pass":
                    row[
                        "attack_audit_pass"
                    ],
            }
        )

    output.sort(
        key=lambda item: (
            item[
                "attack_type"
            ],
            item[
                "malicious_client"
            ],
        )
    )

    return output


def build_ratio_table(
    *,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []

    clean_rows = [
        row
        for row
        in rows
        if row[
            "attack_type"
        ]
        ==
        "NONE"
    ]

    if clean_rows:
        clean = clean_rows[
            0
        ]

        output.append(
            {
                "attack_type":
                    "CLEAN",
                "magnitude":
                    1.0,
                "malicious_clients":
                    [],
                "malicious_ratio":
                    0.0,
                "macro_f1":
                    clean[
                        "macro_f1"
                    ],
                "balanced_accuracy":
                    clean[
                        "balanced_accuracy"
                    ],
                "weighted_f1":
                    clean[
                        "weighted_f1"
                    ],
                "worst_attack_recall":
                    clean[
                        "worst_attack_recall"
                    ],
                "target_attack_mean_recall":
                    None,
                "target_attack_worst_recall":
                    None,
                "macro_f1_drop_vs_clean":
                    0.0,
                "balanced_accuracy_drop_vs_clean":
                    0.0,
                "attack_audit_pass":
                    clean[
                        "attack_audit_pass"
                    ],
            }
        )

    for attack_type in (
        "SCALE",
        "RANDOM",
    ):
        selected = [
            row
            for row
            in rows
            if row[
                "attack_type"
            ]
            ==
            attack_type
            and row[
                "malicious_clients"
            ]
            in (
                [
                    1
                ],
                [
                    1,
                    2,
                ],
            )
        ]

        for row in selected:
            output.append(
                {
                    "attack_type":
                        attack_type,
                    "magnitude":
                        row[
                            "magnitude"
                        ],
                    "malicious_clients":
                        row[
                            "malicious_clients"
                        ],
                    "malicious_ratio":
                        row[
                            "malicious_ratio"
                        ],
                    "macro_f1":
                        row[
                            "macro_f1"
                        ],
                    "balanced_accuracy":
                        row[
                            "balanced_accuracy"
                        ],
                    "weighted_f1":
                        row[
                            "weighted_f1"
                        ],
                    "worst_attack_recall":
                        row[
                            "worst_attack_recall"
                        ],
                    "target_attack_mean_recall":
                        row[
                            "target_attack_mean_recall"
                        ],
                    "target_attack_worst_recall":
                        row[
                            "target_attack_worst_recall"
                        ],
                    "macro_f1_drop_vs_clean":
                        row[
                            "macro_f1_drop_vs_clean"
                        ],
                    "balanced_accuracy_drop_vs_clean":
                        row[
                            "balanced_accuracy_drop_vs_clean"
                        ],
                    "attack_audit_pass":
                        row[
                            "attack_audit_pass"
                        ],
                }
            )

    attack_rank = {
        "CLEAN": 0,
        "SCALE": 1,
        "RANDOM": 2,
    }

    output.sort(
        key=lambda item: (
            attack_rank[
                item[
                    "attack_type"
                ]
            ],
            item[
                "malicious_ratio"
            ],
        )
    )

    return output


def build_summary(
    *,
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    identity = build_identity_table(
        rows=rows
    )

    ratio = build_ratio_table(
        rows=rows
    )

    worst_identity_by_attack = {}

    for attack_type in (
        "SCALE",
        "RANDOM",
    ):
        subset = [
            row
            for row
            in identity
            if row[
                "attack_type"
            ]
            ==
            attack_type
        ]

        if not subset:
            continue

        worst_macro = min(
            subset,
            key=lambda row:
                float(
                    row[
                        "macro_f1"
                    ]
                ),
        )

        worst_target = min(
            subset,
            key=lambda row:
                float(
                    row[
                        "target_attack_worst_recall"
                    ]
                ),
        )

        worst_identity_by_attack[
            attack_type
        ] = {
            "lowest_macro_f1_client":
                int(
                    worst_macro[
                        "malicious_client"
                    ]
                ),
            "lowest_macro_f1":
                float(
                    worst_macro[
                        "macro_f1"
                    ]
                ),
            "lowest_target_recall_client":
                int(
                    worst_target[
                        "malicious_client"
                    ]
                ),
            "lowest_target_recall_attack":
                str(
                    worst_target[
                        "specialized_attack"
                    ]
                ),
            "lowest_target_recall":
                float(
                    worst_target[
                        "target_attack_worst_recall"
                    ]
                ),
        }

    return {
        "identity_rows":
            identity,
        "ratio_rows":
            ratio,
        "worst_identity_by_attack":
            worst_identity_by_attack,
    }
