from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.correlation.phase16d_gnn import (  # noqa: E402
    AlertCorrelationGCN,
    CorrelationGraphDataset,
    collate_graphs,
    rule_baseline_prediction,
    set_all_seeds,
)
from src.correlation.phase16f_rule_gap import (  # noqa: E402
    PHASE16F_VERSION,
    audit_rule_gap,
    generate_rule_gap_benchmark,
    subtype_recall,
)
from src.correlation.rule_temporal_correlator import CorrelationConfig  # noqa: E402


PHASE_NAME = (
    "PHASE 16F — INDEPENDENT RULE-GAP CAMPAIGN BENCHMARK + GNN VALUE TEST"
)

RESULT_ROOT = PROJECT_ROOT / "results" / "phase16"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "correlation"

POLICY_FILE = PROJECT_ROOT / "configs" / "phase16f_rule_gap_policy.json"
PHASE16C_FREEZE_FILE = (
    ARTIFACT_ROOT
    / "phase16c_rule_correlator_freeze_manifest.json"
)

DATASET_FILE = RESULT_ROOT / "phase16f_rule_gap_dataset_manifest.json"
GNN_FILE = RESULT_ROOT / "phase16f_gnn_results.json"
RULE_FILE = RESULT_ROOT / "phase16f_rule_results.json"
HYBRID_FILE = RESULT_ROOT / "phase16f_hybrid_results.json"
AUDIT_FILE = RESULT_ROOT / "phase16f_audit.json"
CHECKPOINT_FILE = ARTIFACT_ROOT / "phase16f_rule_gap_gnn_baseline.pt"
MANIFEST_FILE = ARTIFACT_ROOT / "phase16f_rule_gap_manifest.json"


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=4,
        )


def metrics(
    y_true: List[int],
    y_pred: List[int],
    scores: List[float] | None,
) -> Dict[str, Any]:
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1,
        ],
    )

    tn, fp, fn, tp = cm.ravel()

    result = {
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
        "false_positive_rate": float(
            fp
            /
            max(
                fp + tn,
                1,
            )
        ),
        "false_negative_rate": float(
            fn
            /
            max(
                fn + tp,
                1,
            )
        ),
        "confusion_matrix": cm.tolist(),
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
        result[
            "roc_auc"
        ] = float(
            roc_auc_score(
                y_true,
                scores,
            )
        )
    else:
        result[
            "roc_auc"
        ] = None

    return result


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

    for batch in loader:
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

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "probabilities": probabilities,
        "subtypes": subtypes,
    }


def gap_positive_recall(
    subtypes: List[str],
    y_true: List[int],
    y_pred: List[int],
) -> float:
    indices = [
        i
        for i, subtype
        in enumerate(
            subtypes
        )
        if (
            y_true[
                i
            ]
            ==
            1
            and
            subtype.startswith(
                "RULE_GAP_"
            )
        )
    ]

    if not indices:
        return 0.0

    return float(
        sum(
            1
            for i
            in indices
            if y_pred[
                i
            ]
            ==
            1
        )
        /
        len(
            indices
        )
    )


def main():
    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARTIFACT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    policy = load_json(
        POLICY_FILE
    )

    freeze = load_json(
        PHASE16C_FREEZE_FILE
    )

    if freeze.get(
        "status"
    ) != "FROZEN":
        raise RuntimeError(
            "Phase16C rule correlator is not frozen."
        )

    if freeze.get(
        "locked_test_used",
        True,
    ):
        raise RuntimeError(
            "Locked-test use detected. Phase16F is prohibited."
        )

    frozen = freeze[
        "frozen_rule_correlator_contract"
    ]

    rule_config = CorrelationConfig(
        window_seconds=int(
            frozen[
                "window_seconds"
            ]
        ),
        min_cross_client_clients=int(
            frozen[
                "r1_min_clients"
            ]
        ),
        min_multi_class_clients=int(
            frozen[
                "r2_min_clients"
            ]
        ),
        min_multi_class_classes=int(
            frozen[
                "r2_min_classes"
            ]
        ),
        min_destination_clients=int(
            frozen[
                "r3_min_clients"
            ]
        ),
        min_destination_sources=int(
            frozen[
                "r3_min_sources"
            ]
        ),
    ).validate()

    dataset_cfg = policy[
        "dataset"
    ]

    gnn_cfg = policy[
        "gnn"
    ]

    hybrid_cfg = policy[
        "hybrid"
    ]

    set_all_seeds(
        int(
            gnn_cfg[
                "training_seed"
            ]
        )
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    train = generate_rule_gap_benchmark(
        total_graphs=int(
            dataset_cfg[
                "train_graphs"
            ]
        ),
        seed=int(
            dataset_cfg[
                "train_seed"
            ]
        ),
        window_seconds=rule_config.window_seconds,
        rule_positive_fraction_of_positives=float(
            dataset_cfg[
                "rule_positive_fraction_of_positives"
            ]
        ),
    )

    validation = generate_rule_gap_benchmark(
        total_graphs=int(
            dataset_cfg[
                "validation_graphs"
            ]
        ),
        seed=int(
            dataset_cfg[
                "validation_seed"
            ]
        ),
        window_seconds=rule_config.window_seconds,
        rule_positive_fraction_of_positives=float(
            dataset_cfg[
                "rule_positive_fraction_of_positives"
            ]
        ),
    )

    test = generate_rule_gap_benchmark(
        total_graphs=int(
            dataset_cfg[
                "test_graphs"
            ]
        ),
        seed=int(
            dataset_cfg[
                "test_seed"
            ]
        ),
        window_seconds=rule_config.window_seconds,
        rule_positive_fraction_of_positives=float(
            dataset_cfg[
                "rule_positive_fraction_of_positives"
            ]
        ),
    )

    train_audit = audit_rule_gap(
        train,
        rule_config,
    )

    validation_audit = audit_rule_gap(
        validation,
        rule_config,
    )

    test_audit = audit_rule_gap(
        test,
        rule_config,
    )

    if not all(
        [
            train_audit[
                "passed"
            ],
            validation_audit[
                "passed"
            ],
            test_audit[
                "passed"
            ],
        ]
    ):
        raise RuntimeError(
            "Rule-gap construction audit failed: at least one intended "
            "rule-gap positive or hard negative unexpectedly triggered "
            "the frozen Phase16C rules."
        )

    print()
    print("=" * 132)
    print(PHASE_NAME)
    print("=" * 132)

    print(
        f"Device                         : {device}"
    )
    print(
        f"Frozen Phase16C window         : {rule_config.window_seconds} seconds"
    )
    print(
        f"Train / validation / test      : "
        f"{len(train)} / {len(validation)} / {len(test)} graphs"
    )
    print(
        f"Rule-gap positive audit        : PASS"
    )
    print(
        f"Hard-negative rule-silence     : PASS"
    )
    print(
        f"Locked IDS test                : NO"
    )

    print()
    print("SCIENTIFIC NOTE:")
    print(
        "Unlike Phase16D/E, positive rule-gap labels in this benchmark are defined "
        "by predeclared multi-stage campaign grammars rather than by R1/R2/R3."
    )
    print(
        "The benchmark is still synthetic. A GNN advantage here is evidence of "
        "additional modeled correlation capacity, not proof of real-world superiority."
    )

    dataset_manifest = {
        "phase": "16F",
        "version": PHASE16F_VERSION,
        "frozen_rule_contract": frozen,
        "train_graphs": len(
            train
        ),
        "validation_graphs": len(
            validation
        ),
        "test_graphs": len(
            test
        ),
        "train_subtypes": dict(
            Counter(
                scenario.subtype
                for scenario
                in train
            )
        ),
        "validation_subtypes": dict(
            Counter(
                scenario.subtype
                for scenario
                in validation
            )
        ),
        "test_subtypes": dict(
            Counter(
                scenario.subtype
                for scenario
                in test
            )
        ),
        "train_rule_gap_audit": train_audit,
        "validation_rule_gap_audit": validation_audit,
        "test_rule_gap_audit": test_audit,
        "campaign_grammars": [
            "CHAIN_A: PortScan -> FTP-Patator -> DoS GoldenEye -> DDoS",
            "CHAIN_B: PortScan -> DoS GoldenEye -> DoS Hulk",
            "CHAIN_C: FTP-Patator -> PortScan -> DDoS",
        ],
        "hard_negative_controls": [
            "same timing/client pattern but shuffled attack stages",
            "same attack stages but wrong client progression",
            "same stage order but spread over a long time",
            "random attack burst with similar size/timing",
        ],
        "raw_identifiers_used": False,
        "locked_test_used": False,
    }

    save_json(
        DATASET_FILE,
        dataset_manifest,
    )

    train_ds = CorrelationGraphDataset(
        train
    )

    validation_ds = CorrelationGraphDataset(
        validation
    )

    test_ds = CorrelationGraphDataset(
        test
    )

    generator = torch.Generator()
    generator.manual_seed(
        int(
            gnn_cfg[
                "training_seed"
            ]
        )
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(
            gnn_cfg[
                "batch_size"
            ]
        ),
        shuffle=True,
        collate_fn=collate_graphs,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_ds,
        batch_size=int(
            gnn_cfg[
                "batch_size"
            ]
        ),
        shuffle=False,
        collate_fn=collate_graphs,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=int(
            gnn_cfg[
                "batch_size"
            ]
        ),
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
        hidden_dim=int(
            gnn_cfg[
                "hidden_dim"
            ]
        ),
        dropout=float(
            gnn_cfg[
                "dropout"
            ]
        ),
    ).to(
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(
            gnn_cfg[
                "learning_rate"
            ]
        ),
        weight_decay=float(
            gnn_cfg[
                "weight_decay"
            ]
        ),
    )

    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_epoch = 0
    best_val_macro_f1 = -1.0
    patience_counter = 0
    history = []

    print()
    print("GNN TRAINING")
    print("-" * 132)

    for epoch in range(
        1,
        int(
            gnn_cfg[
                "epochs"
            ]
        )
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

        val_metrics = metrics(
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
                "validation_macro_f1": val_metrics[
                    "macro_f1"
                ],
            }
        )

        print(
            f"Epoch {epoch:02d} | "
            f"loss={train_loss:.6f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.6f}"
        )

        if (
            val_metrics[
                "macro_f1"
            ]
            >
            best_val_macro_f1
            +
            1e-12
        ):
            best_val_macro_f1 = float(
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
            int(
                gnn_cfg[
                    "early_stopping_patience"
                ]
            )
        ):
            print(
                f"Early stopping at epoch {epoch}; best epoch={best_epoch}."
            )
            break

    if best_state is None:
        raise RuntimeError(
            "No GNN checkpoint selected."
        )

    model.load_state_dict(
        best_state
    )

    torch.save(
        {
            "phase": "16F",
            "version": PHASE16F_VERSION,
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_val_macro_f1,
            "input_dim": feature_dim,
            "hidden_dim": int(
                gnn_cfg[
                    "hidden_dim"
                ]
            ),
            "dropout": float(
                gnn_cfg[
                    "dropout"
                ]
            ),
            "state_dict": best_state,
            "frozen_rule_contract": frozen,
        },
        CHECKPOINT_FILE,
    )

    gnn_pred = predict_gnn(
        model,
        test_loader,
        device,
    )

    y_true = gnn_pred[
        "y_true"
    ]

    gnn_y = gnn_pred[
        "y_pred"
    ]

    probabilities = gnn_pred[
        "probabilities"
    ]

    subtypes = gnn_pred[
        "subtypes"
    ]

    rule_y = [
        rule_baseline_prediction(
            scenario.alerts,
            rule_config,
        )
        for scenario
        in test
    ]

    threshold = float(
        hybrid_cfg[
            "gnn_rescue_threshold"
        ]
    )

    hybrid_y = [
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
            rule_y,
            probabilities,
        )
    ]

    gnn_metrics = metrics(
        y_true,
        gnn_y,
        probabilities,
    )

    rule_metrics = metrics(
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

    hybrid_metrics = metrics(
        y_true,
        hybrid_y,
        [
            max(
                float(
                    rule_value
                ),
                probability,
            )
            for rule_value, probability
            in zip(
                rule_y,
                probabilities,
            )
        ],
    )

    gnn_metrics[
        "rule_gap_positive_recall"
    ] = gap_positive_recall(
        subtypes,
        y_true,
        gnn_y,
    )

    rule_metrics[
        "rule_gap_positive_recall"
    ] = gap_positive_recall(
        subtypes,
        y_true,
        rule_y,
    )

    hybrid_metrics[
        "rule_gap_positive_recall"
    ] = gap_positive_recall(
        subtypes,
        y_true,
        hybrid_y,
    )

    gnn_metrics[
        "per_subtype"
    ] = subtype_recall(
        subtypes=subtypes,
        y_true=y_true,
        y_pred=gnn_y,
    )

    rule_metrics[
        "per_subtype"
    ] = subtype_recall(
        subtypes=subtypes,
        y_true=y_true,
        y_pred=rule_y,
    )

    hybrid_metrics[
        "per_subtype"
    ] = subtype_recall(
        subtypes=subtypes,
        y_true=y_true,
        y_pred=hybrid_y,
    )

    save_json(
        GNN_FILE,
        gnn_metrics,
    )

    save_json(
        RULE_FILE,
        rule_metrics,
    )

    save_json(
        HYBRID_FILE,
        hybrid_metrics,
    )

    print()
    print("HELD-OUT RULE-GAP BENCHMARK")
    print("-" * 132)

    print(
        f"Frozen Rule Macro F1           : {rule_metrics['macro_f1']:.6f}"
    )
    print(
        f"GNN Macro F1                   : {gnn_metrics['macro_f1']:.6f}"
    )
    print(
        f"Rule+GNN Hybrid Macro F1       : {hybrid_metrics['macro_f1']:.6f}"
    )

    print()
    print(
        f"Frozen Rule Precision / Recall : "
        f"{rule_metrics['precision']:.6f} / {rule_metrics['recall']:.6f}"
    )
    print(
        f"GNN Precision / Recall         : "
        f"{gnn_metrics['precision']:.6f} / {gnn_metrics['recall']:.6f}"
    )
    print(
        f"Hybrid Precision / Recall      : "
        f"{hybrid_metrics['precision']:.6f} / {hybrid_metrics['recall']:.6f}"
    )

    print()
    print(
        f"Rule-gap positive recall — Rule   : "
        f"{rule_metrics['rule_gap_positive_recall']:.6f}"
    )
    print(
        f"Rule-gap positive recall — GNN    : "
        f"{gnn_metrics['rule_gap_positive_recall']:.6f}"
    )
    print(
        f"Rule-gap positive recall — Hybrid : "
        f"{hybrid_metrics['rule_gap_positive_recall']:.6f}"
    )

    audit = {
        "phase": "16F",
        "version": PHASE16F_VERSION,
        "phase_name": PHASE_NAME,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_val_macro_f1,
        "frozen_rule": rule_metrics,
        "gnn": gnn_metrics,
        "hybrid": hybrid_metrics,
        "rule_gap_dataset_audit": {
            "train": train_audit,
            "validation": validation_audit,
            "test": test_audit,
        },
        "frozen_rules_modified": False,
        "benchmark_is_synthetic": True,
        "real_world_superiority_claimed": False,
        "locked_test_used": False,
        "all_tests_passed": True,
    }

    save_json(
        AUDIT_FILE,
        audit,
    )

    manifest = {
        "phase": "16F",
        "version": PHASE16F_VERSION,
        "status": "PASS",
        "purpose": (
            "Test whether the GNN adds correlation capacity on privacy-safe "
            "multi-stage campaigns deliberately outside frozen R1/R2/R3."
        ),
        "frozen_rule_contract": frozen,
        "gnn_architecture": (
            "Phase16D 2-layer dense GCN with mean/max pooling"
        ),
        "hybrid_policy": hybrid_cfg,
        "important_limitations": [
            "The benchmark is synthetic.",
            "The multi-stage campaign grammars are predeclared engineered patterns.",
            "A positive GNN result demonstrates capacity on this benchmark, not production superiority.",
            "No locked IDS test data are used.",
            "Fresh-alert_id semantic replay remains outside this phase.",
        ],
        "recommended_next_step": (
            "If GNN or hybrid materially improves rule-gap recall without excessive false positives, "
            "run repeated-seed Phase16G on this independent benchmark before any final GNN claim."
        ),
        "locked_test_used": False,
    }

    save_json(
        MANIFEST_FILE,
        manifest,
    )

    print()
    print("=" * 132)
    print("PHASE 16F COMPLETE")
    print("=" * 132)

    print(
        f"Best GNN epoch                 : {best_epoch}"
    )
    print(
        f"Best validation Macro F1       : {best_val_macro_f1:.6f}"
    )
    print(
        f"Frozen Rule Macro F1           : {rule_metrics['macro_f1']:.6f}"
    )
    print(
        f"GNN Macro F1                   : {gnn_metrics['macro_f1']:.6f}"
    )
    print(
        f"Hybrid Macro F1                : {hybrid_metrics['macro_f1']:.6f}"
    )
    print(
        f"Rule-gap recall Rule/GNN/Hybrid: "
        f"{rule_metrics['rule_gap_positive_recall']:.6f} / "
        f"{gnn_metrics['rule_gap_positive_recall']:.6f} / "
        f"{hybrid_metrics['rule_gap_positive_recall']:.6f}"
    )
    print(
        f"Real-world superiority claim   : NO"
    )
    print(
        f"Locked test used               : NO"
    )
    print(
        f"Dataset manifest               : {DATASET_FILE}"
    )
    print(
        f"Audit                          : {AUDIT_FILE}"
    )
    print(
        f"Checkpoint                     : {CHECKPOINT_FILE}"
    )
    print(
        f"Manifest                       : {MANIFEST_FILE}"
    )

    print()
    print("STATUS                         : PASS")

    print()
    print("NEXT:")
    print(
        "If rule-gap recall improves materially, Phase 16G should repeat this independent "
        "benchmark across multiple seeds and statistically compare Rule vs GNN vs Hybrid."
    )


if __name__ == "__main__":
    main()
