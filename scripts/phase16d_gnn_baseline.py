from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
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
    PHASE16D_VERSION,
    AlertCorrelationGCN,
    CorrelationGraphDataset,
    collate_graphs,
    generate_graph_scenarios,
    privacy_audit_scenarios,
    rule_baseline_prediction,
    set_all_seeds,
)
from src.correlation.rule_temporal_correlator import CorrelationConfig  # noqa: E402


PHASE_NAME = (
    "PHASE 16D — PRIVACY-SAFE GNN ALERT-CORRELATION GRAPH DATASET + GCN BASELINE"
)

RESULT_ROOT = PROJECT_ROOT / "results" / "phase16"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "correlation"

POLICY_FILE = PROJECT_ROOT / "configs" / "phase16d_gnn_policy.json"
PHASE16C_FREEZE_FILE = (
    ARTIFACT_ROOT
    / "phase16c_rule_correlator_freeze_manifest.json"
)

TRAINING_HISTORY_FILE = RESULT_ROOT / "phase16d_gnn_training_history.json"
TEST_RESULTS_FILE = RESULT_ROOT / "phase16d_gnn_test_results.json"
RULE_RESULTS_FILE = RESULT_ROOT / "phase16d_rule_baseline_on_gnn_test.json"
DATASET_MANIFEST_FILE = RESULT_ROOT / "phase16d_graph_dataset_manifest.json"
AUDIT_FILE = RESULT_ROOT / "phase16d_audit.json"

CHECKPOINT_FILE = (
    ARTIFACT_ROOT
    / "phase16d_gnn_baseline.pt"
)

MANIFEST_FILE = (
    ARTIFACT_ROOT
    / "phase16d_gnn_manifest.json"
)


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


def metrics_from_predictions(
    y_true: List[int],
    y_pred: List[int],
    probabilities: List[float],
) -> Dict[str, Any]:
    result = {
        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "precision":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "recall":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),

        "confusion_matrix":
            confusion_matrix(
                y_true,
                y_pred,
                labels=[
                    0,
                    1,
                ],
            ).tolist(),
    }

    if len(
        set(
            y_true
        )
    ) == 2:
        result[
            "roc_auc"
        ] = float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        )

    else:
        result[
            "roc_auc"
        ] = None

    return result


def per_subtype_metrics(
    subtypes: List[str],
    y_true: List[int],
    y_pred: List[int],
) -> Dict[str, Any]:
    grouped = defaultdict(
        lambda: {
            "true": [],
            "pred": [],
        }
    )

    for subtype, truth, prediction in zip(
        subtypes,
        y_true,
        y_pred,
    ):
        grouped[
            subtype
        ][
            "true"
        ].append(
            truth
        )

        grouped[
            subtype
        ][
            "pred"
        ].append(
            prediction
        )

    output = {}

    for subtype, values in sorted(
        grouped.items()
    ):
        truth = values[
            "true"
        ]

        prediction = values[
            "pred"
        ]

        output[
            subtype
        ] = {
            "count":
                len(
                    truth
                ),

            "accuracy":
                float(
                    accuracy_score(
                        truth,
                        prediction,
                    )
                ),
        }

    return output


@torch.no_grad()
def evaluate_model(
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

        labels = batch[
            "labels"
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

        subtypes.extend(
            batch[
                "subtypes"
            ]
        )

    metrics = metrics_from_predictions(
        y_true,
        y_pred,
        probabilities,
    )

    metrics[
        "per_subtype"
    ] = per_subtype_metrics(
        subtypes,
        y_true,
        y_pred,
    )

    metrics[
        "count"
    ] = len(
        y_true
    )

    return metrics


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
            "Phase16C rule correlator is not frozen. "
            "Phase16D must not proceed."
        )

    if freeze.get(
        "locked_test_used",
        True,
    ):
        raise RuntimeError(
            "Phase16C indicates locked-test usage. "
            "Phase16D must not proceed."
        )

    frozen_contract = freeze[
        "frozen_rule_correlator_contract"
    ]

    rule_config = CorrelationConfig(
        window_seconds=int(
            frozen_contract[
                "window_seconds"
            ]
        ),
        min_cross_client_clients=int(
            frozen_contract[
                "r1_min_clients"
            ]
        ),
        min_multi_class_clients=int(
            frozen_contract[
                "r2_min_clients"
            ]
        ),
        min_multi_class_classes=int(
            frozen_contract[
                "r2_min_classes"
            ]
        ),
        min_destination_clients=int(
            frozen_contract[
                "r3_min_clients"
            ]
        ),
        min_destination_sources=int(
            frozen_contract[
                "r3_min_sources"
            ]
        ),
    ).validate()

    dataset_policy = policy[
        "dataset"
    ]

    gnn_policy = policy[
        "gnn"
    ]

    set_all_seeds(
        int(
            gnn_policy[
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

    train_scenarios = generate_graph_scenarios(
        count=int(
            dataset_policy[
                "train_graphs"
            ]
        ),
        seed=int(
            dataset_policy[
                "train_seed"
            ]
        ),
        window_seconds=
            rule_config.window_seconds,
    )

    validation_scenarios = generate_graph_scenarios(
        count=int(
            dataset_policy[
                "validation_graphs"
            ]
        ),
        seed=int(
            dataset_policy[
                "validation_seed"
            ]
        ),
        window_seconds=
            rule_config.window_seconds,
    )

    test_scenarios = generate_graph_scenarios(
        count=int(
            dataset_policy[
                "test_graphs"
            ]
        ),
        seed=int(
            dataset_policy[
                "test_seed"
            ]
        ),
        window_seconds=
            rule_config.window_seconds,
    )

    privacy_train = privacy_audit_scenarios(
        train_scenarios
    )

    privacy_validation = privacy_audit_scenarios(
        validation_scenarios
    )

    privacy_test = privacy_audit_scenarios(
        test_scenarios
    )

    privacy_pass = all(
        [
            privacy_train[
                "passed"
            ],
            privacy_validation[
                "passed"
            ],
            privacy_test[
                "passed"
            ],
        ]
    )

    if not privacy_pass:
        raise RuntimeError(
            "Phase16D synthetic graph privacy audit failed."
        )

    print()
    print("=" * 132)
    print(PHASE_NAME)
    print("=" * 132)

    print(
        f"Device                        : {device}"
    )
    print(
        f"Frozen rule window            : {rule_config.window_seconds} seconds"
    )
    print(
        f"Train / validation / test     : "
        f"{len(train_scenarios)} / "
        f"{len(validation_scenarios)} / "
        f"{len(test_scenarios)} graphs"
    )
    print(
        f"Graph privacy audit           : PASS"
    )
    print(
        f"Raw identifiers               : NO"
    )
    print(
        f"Locked IDS test               : NO"
    )

    print()
    print("SCIENTIFIC NOTE:")
    print(
        "Phase16D trains a GNN on separately generated synthetic privacy-safe "
        "correlation graphs."
    )
    print(
        "This is a GNN correlation baseline. It is NOT IDS attack-classification "
        "evidence and does not replace the frozen Phase16C rule baseline."
    )

    dataset_manifest = {
        "phase":
            "16D",

        "version":
            PHASE16D_VERSION,

        "frozen_rule_contract":
            frozen_contract,

        "train_graphs":
            len(
                train_scenarios
            ),

        "validation_graphs":
            len(
                validation_scenarios
            ),

        "test_graphs":
            len(
                test_scenarios
            ),

        "train_label_counts":
            dict(
                Counter(
                    scenario.label
                    for scenario
                    in train_scenarios
                )
            ),

        "validation_label_counts":
            dict(
                Counter(
                    scenario.label
                    for scenario
                    in validation_scenarios
                )
            ),

        "test_label_counts":
            dict(
                Counter(
                    scenario.label
                    for scenario
                    in test_scenarios
                )
            ),

        "train_subtypes":
            dict(
                Counter(
                    scenario.subtype
                    for scenario
                    in train_scenarios
                )
            ),

        "validation_subtypes":
            dict(
                Counter(
                    scenario.subtype
                    for scenario
                    in validation_scenarios
                )
            ),

        "test_subtypes":
            dict(
                Counter(
                    scenario.subtype
                    for scenario
                    in test_scenarios
                )
            ),

        "privacy_audit":
            {
                "train":
                    privacy_train,

                "validation":
                    privacy_validation,

                "test":
                    privacy_test,
            },

        "raw_identifiers_used":
            False,

        "locked_test_used":
            False,
    }

    save_json(
        DATASET_MANIFEST_FILE,
        dataset_manifest,
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

    batch_size = int(
        gnn_policy[
            "batch_size"
        ]
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(
        int(
            gnn_policy[
                "training_seed"
            ]
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_graphs,
        generator=train_generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
    )

    feature_dim = int(
        train_dataset[
            0
        ].x.shape[
            1
        ]
    )

    model = AlertCorrelationGCN(
        input_dim=
            feature_dim,

        hidden_dim=
            int(
                gnn_policy[
                    "hidden_dim"
                ]
            ),

        dropout=
            float(
                gnn_policy[
                    "dropout"
                ]
            ),
    ).to(
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(
            gnn_policy[
                "learning_rate"
            ]
        ),
        weight_decay=float(
            gnn_policy[
                "weight_decay"
            ]
        ),
    )

    criterion = nn.CrossEntropyLoss()

    history = []
    best_val_macro_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    epochs = int(
        gnn_policy[
            "epochs"
        ]
    )

    patience = int(
        gnn_policy[
            "early_stopping_patience"
        ]
    )

    print()
    print("GNN TRAINING")
    print("-" * 132)

    training_start = time.perf_counter()

    for epoch in range(
        1,
        epochs
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

            labels = batch[
                "labels"
            ].to(
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
                labels.shape[
                    0
                ]
            )

            running_loss += (
                float(
                    loss.item()
                )
                *
                batch_count
            )

            sample_count += batch_count

        train_loss = (
            running_loss
            /
            max(
                sample_count,
                1,
            )
        )

        validation_metrics = evaluate_model(
            model,
            validation_loader,
            device,
        )

        val_macro_f1 = float(
            validation_metrics[
                "macro_f1"
            ]
        )

        history.append(
            {
                "epoch":
                    epoch,

                "train_loss":
                    train_loss,

                "validation_macro_f1":
                    val_macro_f1,

                "validation_accuracy":
                    validation_metrics[
                        "accuracy"
                    ],

                "validation_balanced_accuracy":
                    validation_metrics[
                        "balanced_accuracy"
                    ],
            }
        )

        improved = (
            val_macro_f1
            >
            best_val_macro_f1
            +
            1e-12
        )

        if improved:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "phase":
                        "16D",

                    "version":
                        PHASE16D_VERSION,

                    "epoch":
                        epoch,

                    "input_dim":
                        feature_dim,

                    "hidden_dim":
                        int(
                            gnn_policy[
                                "hidden_dim"
                            ]
                        ),

                    "dropout":
                        float(
                            gnn_policy[
                                "dropout"
                            ]
                        ),

                    "state_dict":
                        model.state_dict(),

                    "frozen_rule_contract":
                        frozen_contract,

                    "training_seed":
                        int(
                            gnn_policy[
                                "training_seed"
                            ]
                        ),
                },
                CHECKPOINT_FILE,
            )

        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:02d} | "
            f"loss={train_loss:.6f} | "
            f"val_macro_f1={val_macro_f1:.6f}"
        )

        if patience_counter >= patience:
            print(
                f"Early stopping at epoch {epoch}; best epoch={best_epoch}."
            )
            break

    training_seconds = (
        time.perf_counter()
        -
        training_start
    )

    save_json(
        TRAINING_HISTORY_FILE,
        {
            "best_epoch":
                best_epoch,

            "best_validation_macro_f1":
                best_val_macro_f1,

            "training_seconds":
                training_seconds,

            "history":
                history,
        },
    )

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "state_dict"
        ]
    )

    test_start = time.perf_counter()

    gnn_test = evaluate_model(
        model,
        test_loader,
        device,
    )

    inference_seconds = (
        time.perf_counter()
        -
        test_start
    )

    gnn_test[
        "best_epoch"
    ] = best_epoch

    gnn_test[
        "best_validation_macro_f1"
    ] = best_val_macro_f1

    gnn_test[
        "training_seconds"
    ] = training_seconds

    gnn_test[
        "test_inference_seconds"
    ] = inference_seconds

    gnn_test[
        "mean_graph_inference_ms"
    ] = (
        inference_seconds
        /
        max(
            len(
                test_scenarios
            ),
            1,
        )
        *
        1000.0
    )

    save_json(
        TEST_RESULTS_FILE,
        gnn_test,
    )

    # ============================================================
    # Frozen Phase16C deterministic baseline on exact same test set
    # ============================================================
    rule_y_true = []
    rule_y_pred = []
    rule_subtypes = []

    rule_start = time.perf_counter()

    for scenario in test_scenarios:
        truth = int(
            scenario.label
        )

        prediction = rule_baseline_prediction(
            scenario.alerts,
            rule_config,
        )

        rule_y_true.append(
            truth
        )

        rule_y_pred.append(
            prediction
        )

        rule_subtypes.append(
            scenario.subtype
        )

    rule_seconds = (
        time.perf_counter()
        -
        rule_start
    )

    # Rule outputs are hard labels, so use them as 0/1 "scores" only for
    # a descriptive AUROC calculation. Do not treat them as probabilities.
    rule_metrics = metrics_from_predictions(
        rule_y_true,
        rule_y_pred,
        [
            float(
                value
            )
            for value
            in rule_y_pred
        ],
    )

    rule_metrics[
        "per_subtype"
    ] = per_subtype_metrics(
        rule_subtypes,
        rule_y_true,
        rule_y_pred,
    )

    rule_metrics[
        "count"
    ] = len(
        rule_y_true
    )

    rule_metrics[
        "test_inference_seconds"
    ] = rule_seconds

    rule_metrics[
        "mean_graph_inference_ms"
    ] = (
        rule_seconds
        /
        max(
            len(
                test_scenarios
            ),
            1,
        )
        *
        1000.0
    )

    rule_metrics[
        "scores_are_probabilities"
    ] = False

    save_json(
        RULE_RESULTS_FILE,
        rule_metrics,
    )

    print()
    print("HELD-OUT SYNTHETIC GRAPH TEST")
    print("-" * 132)

    print(
        f"GNN Accuracy                   : {gnn_test['accuracy']:.6f}"
    )
    print(
        f"GNN Balanced Accuracy          : {gnn_test['balanced_accuracy']:.6f}"
    )
    print(
        f"GNN Precision                  : {gnn_test['precision']:.6f}"
    )
    print(
        f"GNN Recall                     : {gnn_test['recall']:.6f}"
    )
    print(
        f"GNN F1                         : {gnn_test['f1']:.6f}"
    )
    print(
        f"GNN Macro F1                   : {gnn_test['macro_f1']:.6f}"
    )
    print(
        f"GNN ROC AUC                    : {gnn_test['roc_auc']:.6f}"
    )

    print()
    print(
        f"Frozen Rule Accuracy           : {rule_metrics['accuracy']:.6f}"
    )
    print(
        f"Frozen Rule Balanced Accuracy  : {rule_metrics['balanced_accuracy']:.6f}"
    )
    print(
        f"Frozen Rule Precision          : {rule_metrics['precision']:.6f}"
    )
    print(
        f"Frozen Rule Recall             : {rule_metrics['recall']:.6f}"
    )
    print(
        f"Frozen Rule F1                 : {rule_metrics['f1']:.6f}"
    )
    print(
        f"Frozen Rule Macro F1           : {rule_metrics['macro_f1']:.6f}"
    )

    comparison = {
        "gnn_macro_f1":
            gnn_test[
                "macro_f1"
            ],

        "rule_macro_f1":
            rule_metrics[
                "macro_f1"
            ],

        "gnn_minus_rule_macro_f1":
            gnn_test[
                "macro_f1"
            ]
            -
            rule_metrics[
                "macro_f1"
            ],

        "gnn_mean_graph_inference_ms":
            gnn_test[
                "mean_graph_inference_ms"
            ],

        "rule_mean_graph_inference_ms":
            rule_metrics[
                "mean_graph_inference_ms"
            ],

        "superiority_claimed":
            False,
    }

    all_passed = all(
        [
            privacy_pass,
            CHECKPOINT_FILE.exists(),
            len(
                test_scenarios
            )
            ==
            int(
                dataset_policy[
                    "test_graphs"
                ]
            ),
            math.isfinite(
                float(
                    gnn_test[
                        "macro_f1"
                    ]
                )
            ),
            math.isfinite(
                float(
                    rule_metrics[
                        "macro_f1"
                    ]
                )
            ),
        ]
    )

    audit = {
        "phase":
            "16D",

        "phase_version":
            PHASE16D_VERSION,

        "phase_name":
            PHASE_NAME,

        "frozen_phase16c_contract":
            frozen_contract,

        "graph_privacy_audit_passed":
            privacy_pass,

        "best_epoch":
            best_epoch,

        "best_validation_macro_f1":
            best_val_macro_f1,

        "gnn_test":
            gnn_test,

        "frozen_rule_test":
            rule_metrics,

        "comparison":
            comparison,

        "gnn_checkpoint_is_final_model":
            False,

        "phase16c_rule_baseline_modified":
            False,

        "phase14_risk_semantics_modified":
            False,

        "raw_identifiers_used":
            False,

        "locked_test_used":
            False,

        "all_tests_passed":
            all_passed,
    }

    save_json(
        AUDIT_FILE,
        audit,
    )

    manifest = {
        "phase":
            "16D",

        "version":
            PHASE16D_VERSION,

        "status":
            (
                "PASS"
                if all_passed
                else
                "FAIL"
            ),

        "model_type":
            "2-layer dense GCN + mean/max graph pooling",

        "task":
            "binary graph-level correlated-campaign classification",

        "node_features":
            [
                "attack-class one-hot",
                "client one-hot",
                "severity one-hot",
                "protocol one-hot",
                "processing-path one-hot",
                "risk-mode one-hot",
                "normalized destination port",
                "confidence",
                "evidence score",
                "risk score with zero for null FAST path",
                "relative event-time position",
            ],

        "edge_relations":
            [
                "same source pseudonym",
                "same destination pseudonym + port",
                "same client weak relation",
                "continuous temporal proximity",
            ],

        "important_privacy_note":
            (
                "HMAC tokens are used only for equality relations. "
                "Opaque token characters are not converted into semantic numeric features."
            ),

        "scientific_limitations":
            [
                "Dataset is synthetic and privacy-safe.",
                "This does not measure IDS attack-classification accuracy.",
                "One training seed is a baseline only.",
                "The GNN checkpoint is not frozen as the final correlator.",
                "Phase16C deterministic rules remain frozen and unchanged.",
                "Cross-role source-to-destination entity linking is unavailable because Phase15 HMAC uses domain-separated source/destination pseudonyms.",
                "No superiority claim over rules is allowed from Phase16D alone.",
            ],

        "locked_test_used":
            False,

        "ready_for_phase16e":
            all_passed,
    }

    save_json(
        MANIFEST_FILE,
        manifest,
    )

    print()
    print("=" * 132)
    print("PHASE 16D COMPLETE")
    print("=" * 132)

    print(
        f"Graph privacy audit            : {'PASS' if privacy_pass else 'FAIL'}"
    )
    print(
        f"Best GNN epoch                 : {best_epoch}"
    )
    print(
        f"Best validation Macro F1       : {best_val_macro_f1:.6f}"
    )
    print(
        f"Held-out GNN Macro F1          : {gnn_test['macro_f1']:.6f}"
    )
    print(
        f"Held-out frozen-rule Macro F1  : {rule_metrics['macro_f1']:.6f}"
    )
    print(
        f"GNN - Rule Macro F1            : {comparison['gnn_minus_rule_macro_f1']:+.6f}"
    )
    print(
        f"GNN mean graph inference       : {gnn_test['mean_graph_inference_ms']:.4f} ms"
    )
    print(
        f"Rule mean graph inference      : {rule_metrics['mean_graph_inference_ms']:.4f} ms"
    )
    print(
        f"Locked test used               : NO"
    )
    print(
        f"GNN checkpoint                 : {CHECKPOINT_FILE}"
    )
    print(
        f"Audit                          : {AUDIT_FILE}"
    )
    print(
        f"Dataset manifest               : {DATASET_MANIFEST_FILE}"
    )
    print(
        f"GNN test results               : {TEST_RESULTS_FILE}"
    )
    print(
        f"Rule baseline results          : {RULE_RESULTS_FILE}"
    )
    print(
        f"Manifest                       : {MANIFEST_FILE}"
    )

    print()
    print(
        f"STATUS                         : {'PASS' if all_passed else 'FAIL'}"
    )

    print()
    print("IMPORTANT:")
    print(
        "Phase16D is a single-seed GNN baseline. Do not claim that GNN is superior "
        "to the frozen rule correlator from this phase alone."
    )

    print()
    print("NEXT:")
    print(
        "Phase 16E — repeated-seed Rule vs GNN vs Rule+GNN hybrid comparison, "
        "followed by a predeclared final correlation-policy decision."
    )

    if not all_passed:
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()
