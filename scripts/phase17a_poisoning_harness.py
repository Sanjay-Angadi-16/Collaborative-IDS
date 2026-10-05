from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SCRIPT_DIR),
    )

try:
    import phase10_fedprox70 as phase10
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "\nPhase17A requires the existing script:\n"
        "    scripts\\phase10_fedprox70.py\n"
    ) from exc

phase4 = phase10.phase4

from src.federated.poisoning.attacks import (  # noqa: E402
    AttackSpec,
    apply_client_attack,
    state_max_abs_diff,
)
from src.federated.poisoning.metrics import (  # noqa: E402
    calculate_robustness_metrics,
    compare_to_clean,
)


PHASE_NAME = (
    "PHASE 17A — POISONING THREAT-MODEL FREEZE + CLEAN-vs-MALICIOUS HARNESS VALIDATION"
)

CONFIG_FILE = (
    PROJECT_ROOT
    / "configs"
    / "phase17a_threat_model.json"
)

VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation.csv"
)

LOCKED_TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_locked.csv"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "phase17"
    / "phase17a"
)

ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "poisoning"
    / "phase17a"
)

logger = logging.getLogger(
    "phase17a_poisoning_harness"
)


def load_json(
    path: Path,
) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            path
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )


def save_json(
    path: Path,
    payload: Any,
) -> None:
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


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as file:
        for block in iter(
            lambda: file.read(
                1024
                *
                1024
            ),
            b"",
        ):
            digest.update(
                block
            )

    return digest.hexdigest()


def sha256_state(
    state: Mapping[str, torch.Tensor],
) -> str:
    digest = hashlib.sha256()

    for key in sorted(
        state.keys()
    ):
        digest.update(
            key.encode(
                "utf-8"
            )
        )

        tensor = (
            state[
                key
            ]
            .detach()
            .cpu()
            .contiguous()
        )

        digest.update(
            str(
                tensor.dtype
            ).encode(
                "utf-8"
            )
        )

        digest.update(
            str(
                tuple(
                    tensor.shape
                )
            ).encode(
                "utf-8"
            )
        )

        digest.update(
            tensor.numpy().tobytes()
        )

    return digest.hexdigest()


def verify_frozen_sources(
    config: Mapping[str, Any],
    *,
    allow_mismatch: bool,
) -> Dict[str, Any]:
    expected = config[
        "frozen_source_hashes"
    ]

    rows = {}

    for relative_path, expected_hash in expected.items():
        path = (
            PROJECT_ROOT
            /
            relative_path
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Frozen source missing: {path}"
            )

        actual_hash = sha256_file(
            path
        )

        match = (
            actual_hash
            ==
            expected_hash
        )

        rows[
            relative_path
        ] = {
            "expected_sha256":
                expected_hash,
            "actual_sha256":
                actual_hash,
            "match":
                match,
        }

        if (
            not match
            and
            not allow_mismatch
        ):
            raise RuntimeError(
                "\nFrozen source hash mismatch.\n\n"
                f"File: {path}\n"
                f"Expected: {expected_hash}\n"
                f"Actual  : {actual_hash}\n\n"
                "Phase17A is designed around the exact current FedProx70 "
                "pipeline. If you intentionally changed the baseline, create "
                "a new Phase17A version instead of silently proceeding.\n"
                "For debugging only, --allow-source-hash-mismatch can bypass "
                "this gate."
            )

    return rows


def ensure_data_policy() -> None:
    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            f"Validation file missing: {VALIDATION_FILE}"
        )

    if (
        VALIDATION_FILE.resolve()
        ==
        LOCKED_TEST_FILE.resolve()
    ):
        raise RuntimeError(
            "Validation path resolves to locked test. Refusing to run."
        )


@torch.no_grad()
def evaluate_on_validation(
    *,
    model: torch.nn.Module,
    scaler,
    feature_columns,
    active_indices,
    label_mapping,
    device: torch.device,
    batch_size: int,
    chunk_size: int,
) -> Dict[str, Any]:
    """
    Stream the official development validation.csv.

    IMPORTANT:
        test_locked.csv is never opened here.
    """
    model.eval()

    all_true: List[
        np.ndarray
    ] = []

    all_pred: List[
        np.ndarray
    ] = []

    inference_start = (
        time.perf_counter()
    )

    reader = pd.read_csv(
        VALIDATION_FILE,
        chunksize=chunk_size,
    )

    # The official validation.csv contains more CICIDS2017 classes than
    # the frozen six-class project model. Phase13A already established
    # the correct development policy: filter validation rows to the
    # project's six-class label_mapping before encoding/evaluation.
    #
    # IMPORTANT:
    # We do NOT remap unsupported attacks to BENIGN or to another class.
    # They are excluded from this six-class FedProx70 evaluation and
    # explicitly counted for the audit.
    canonical = {
        str(label).strip().lower():
            str(label).strip()
        for label
        in label_mapping.keys()
    }

    total_validation_rows = 0
    evaluated_validation_rows = 0
    ignored_validation_rows = 0
    ignored_label_counts: Dict[str, int] = {}

    for chunk in reader:
        total_validation_rows += int(
            len(
                chunk
            )
        )

        if phase4.LABEL_COL not in chunk.columns:
            raise KeyError(
                f"Validation file is missing label column: {phase4.LABEL_COL}"
            )

        raw_labels = (
            chunk[
                phase4.LABEL_COL
            ]
            .astype(str)
            .str.strip()
        )

        keep_mask = (
            raw_labels
            .str.lower()
            .isin(
                canonical.keys()
            )
        )

        ignored = (
            raw_labels
            .loc[
                ~keep_mask
            ]
            .value_counts()
        )

        for label, count in ignored.items():
            ignored_label_counts[
                str(
                    label
                )
            ] = (
                ignored_label_counts.get(
                    str(
                        label
                    ),
                    0,
                )
                +
                int(
                    count
                )
            )

        ignored_validation_rows += int(
            (
                ~keep_mask
            ).sum()
        )

        chunk = (
            chunk
            .loc[
                keep_mask
            ]
            .copy()
        )

        if chunk.empty:
            continue

        chunk[
            phase4.LABEL_COL
        ] = (
            chunk[
                phase4.LABEL_COL
            ]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(
                canonical
            )
        )

        evaluated_validation_rows += int(
            len(
                chunk
            )
        )

        X = phase4.transform_features(
            df=chunk,
            feature_columns=feature_columns,
            scaler=scaler,
            active_indices=active_indices,
        )

        y = phase4.encode_labels(
            chunk[
                phase4.LABEL_COL
            ],
            label_mapping,
        )

        if len(
            y
        ) == 0:
            continue

        predictions = []

        for start in range(
            0,
            len(
                y
            ),
            batch_size,
        ):
            end = min(
                start
                +
                batch_size,
                len(
                    y
                ),
            )

            X_batch = torch.from_numpy(
                X[
                    start:end
                ].astype(
                    np.float32,
                    copy=False,
                )
            ).to(
                device
            )

            logits = model(
                X_batch
            )

            pred = torch.argmax(
                logits,
                dim=1,
            ).cpu().numpy()

            predictions.append(
                pred
            )

        all_true.append(
            y.astype(
                np.int64,
                copy=False,
            )
        )

        all_pred.append(
            np.concatenate(
                predictions
            ).astype(
                np.int64,
                copy=False,
            )
        )

    if not all_true:
        raise RuntimeError(
            "Validation evaluation produced zero samples."
        )

    y_true = np.concatenate(
        all_true
    )

    y_pred = np.concatenate(
        all_pred
    )

    metrics = calculate_robustness_metrics(
        y_true=y_true,
        y_pred=y_pred,
        label_mapping=label_mapping,
        benign_label=phase4.BENIGN_LABEL,
    )

    metrics[
        "overall"
    ][
        "inference_time_seconds"
    ] = float(
        time.perf_counter()
        -
        inference_start
    )

    metrics[
        "evaluation_source"
    ] = str(
        VALIDATION_FILE
    )

    metrics[
        "validation_filter_policy"
    ] = (
        "Only labels present in the frozen six-class label_mapping are "
        "evaluated. Unsupported CICIDS2017 validation labels are excluded, "
        "never relabeled."
    )

    metrics[
        "total_validation_rows"
    ] = int(
        total_validation_rows
    )

    metrics[
        "evaluated_project_class_rows"
    ] = int(
        evaluated_validation_rows
    )

    metrics[
        "ignored_non_project_rows"
    ] = int(
        ignored_validation_rows
    )

    metrics[
        "ignored_non_project_labels"
    ] = dict(
        sorted(
            ignored_label_counts.items()
        )
    )

    if (
        evaluated_validation_rows
        <=
        0
    ):
        raise RuntimeError(
            "No six-class project rows remained after validation filtering."
        )

    metrics[
        "locked_test_used"
    ] = False

    return metrics


def run_experiment(
    *,
    experiment: Mapping[str, Any],
    args,
    scaler,
    feature_columns,
    label_mapping,
    all_active_indices,
    all_features,
    global_weights,
    client_files,
    client_sample_counts,
    device: torch.device,
) -> Dict[str, Any]:
    experiment_id = str(
        experiment[
            "id"
        ]
    )

    spec = AttackSpec(
        attack_type=str(
            experiment[
                "attack_type"
            ]
        ),
        malicious_clients=tuple(
            int(
                client_id
            )
            for client_id
            in experiment[
                "malicious_clients"
            ]
        ),
        update_scale=float(
            experiment.get(
                "update_scale",
                1.0,
            )
        ),
    ).validate()

    print()
    print("=" * 120)
    print(
        f"{experiment_id}"
    )
    print("=" * 120)
    print(
        f"Attack type          : {spec.attack_type}"
    )
    print(
        f"Malicious clients    : {list(spec.malicious_clients)}"
    )
    print(
        f"Update scale         : {spec.update_scale}"
    )
    print(
        f"Locked test used     : NO"
    )

    global_model = (
        phase10.initialize_full_model(
            input_dim=len(
                all_features
            ),
            num_classes=len(
                label_mapping
            ),
            device=device,
            seed=args.seed,
        )
    )

    history = []
    attack_events = []

    training_start = (
        time.perf_counter()
    )

    for round_number in range(
        1,
        args.rounds
        +
        1,
    ):
        old_global_state = {
            key:
                value.detach().cpu().clone()
            for key, value
            in global_model.state_dict().items()
        }

        client_states = []
        aggregation_weights = []
        local_losses = []
        local_ce_losses = []
        local_prox_losses = []
        local_accuracies = []
        local_times = []

        for client_id in range(
            1,
            phase10.N_CLIENTS
            +
            1,
        ):
            (
                local_state,
                local_loss,
                local_ce_loss,
                local_prox_loss,
                local_accuracy,
                local_seconds,
            ) = phase10.train_client_fedprox(
                global_state=
                    global_model.state_dict(),
                client_file=
                    client_files[
                        client_id
                    ],
                scaler=
                    scaler,
                feature_columns=
                    feature_columns,
                active_indices=
                    all_active_indices,
                label_mapping=
                    label_mapping,
                class_weights=
                    global_weights,
                input_dim=
                    len(
                        all_features
                    ),
                device=
                    device,
                local_epochs=
                    args.local_epochs,
                batch_size=
                    args.batch_size,
                chunk_size=
                    args.chunk_size,
                learning_rate=
                    args.learning_rate,
                weight_decay=
                    args.weight_decay,
                client_id=
                    client_id,
                round_number=
                    round_number,
                mu=
                    args.mu,
                seed=
                    args.seed,
            )

            routed_state, attack_event = (
                apply_client_attack(
                    global_state=
                        global_model.state_dict(),
                    local_state=
                        local_state,
                    client_id=
                        client_id,
                    spec=
                        spec,
                    round_number=
                        round_number,
                )
            )

            attack_events.append(
                attack_event
            )

            client_states.append(
                routed_state
            )

            aggregation_weights.append(
                client_sample_counts[
                    f"client_{client_id}"
                ]
            )

            local_losses.append(
                local_loss
            )

            local_ce_losses.append(
                local_ce_loss
            )

            local_prox_losses.append(
                local_prox_loss
            )

            local_accuracies.append(
                local_accuracy
            )

            local_times.append(
                local_seconds
            )

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

        history.append(
            {
                "round":
                    int(
                        round_number
                    ),
                "weighted_local_loss":
                    float(
                        np.average(
                            local_losses,
                            weights=
                                weights,
                        )
                    ),
                "weighted_ce_loss":
                    float(
                        np.average(
                            local_ce_losses,
                            weights=
                                weights,
                        )
                    ),
                "weighted_prox_loss":
                    float(
                        np.average(
                            local_prox_losses,
                            weights=
                                weights,
                        )
                    ),
                "weighted_local_accuracy":
                    float(
                        np.average(
                            local_accuracies,
                            weights=
                                weights,
                        )
                    ),
                "global_update_norm":
                    float(
                        update_norm
                    ),
                "mean_client_training_seconds":
                    float(
                        np.mean(
                            local_times
                        )
                    ),
            }
        )

        print(
            f"Round {round_number:02d}/{args.rounds} | "
            f"loss={history[-1]['weighted_local_loss']:.6f} | "
            f"acc={history[-1]['weighted_local_accuracy']:.6f} | "
            f"update_norm={history[-1]['global_update_norm']:.6f}"
        )

    training_seconds = float(
        time.perf_counter()
        -
        training_start
    )

    validation_metrics = evaluate_on_validation(
        model=global_model,
        scaler=scaler,
        feature_columns=feature_columns,
        active_indices=all_active_indices,
        label_mapping=label_mapping,
        device=device,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )

    validation_metrics[
        "overall"
    ][
        "training_time_seconds"
    ] = training_seconds

    final_state = {
        key:
            value.detach().cpu().clone()
        for key, value
        in global_model.state_dict().items()
    }

    model_hash = sha256_state(
        final_state
    )

    return {
        "experiment_id":
            experiment_id,
        "attack_spec":
            {
                "attack_type":
                    spec.attack_type,
                "malicious_clients":
                    list(
                        spec.malicious_clients
                    ),
                "update_scale":
                    float(
                        spec.update_scale
                    ),
            },
        "history":
            history,
        "attack_events":
            attack_events,
        "validation":
            validation_metrics,
        "final_state":
            final_state,
        "final_state_sha256":
            model_hash,
    }


def save_experiment(
    *,
    run: Mapping[str, Any],
    seed: int,
) -> None:
    experiment_id = run[
        "experiment_id"
    ]

    result_dir = (
        RESULT_ROOT
        /
        f"seed_{seed}"
        /
        experiment_id
    )

    artifact_dir = (
        ARTIFACT_ROOT
        /
        f"seed_{seed}"
        /
        experiment_id
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        run[
            "history"
        ]
    ).to_csv(
        result_dir
        /
        "round_history.csv",
        index=False,
    )

    save_json(
        result_dir
        /
        "attack_events.json",
        run[
            "attack_events"
        ],
    )

    save_json(
        result_dir
        /
        "validation_metrics.json",
        run[
            "validation"
        ],
    )

    per_class = run[
        "validation"
    ][
        "per_class"
    ]

    pd.DataFrame(
        [
            {
                "class":
                    class_name,
                **values,
            }
            for class_name, values
            in per_class.items()
        ]
    ).to_csv(
        result_dir
        /
        "per_class_validation.csv",
        index=False,
    )

    pd.DataFrame(
        run[
            "validation"
        ][
            "confusion_matrix"
        ],
        index=
            run[
                "validation"
            ][
                "class_names"
            ],
        columns=
            run[
                "validation"
            ][
                "class_names"
            ],
    ).to_csv(
        result_dir
        /
        "validation_confusion_matrix.csv",
    )

    torch.save(
        {
            "phase":
                "17A",
            "experiment_id":
                experiment_id,
            "seed":
                int(
                    seed
                ),
            "attack_spec":
                run[
                    "attack_spec"
                ],
            "model_state_dict":
                run[
                    "final_state"
                ],
            "state_sha256":
                run[
                    "final_state_sha256"
                ],
        },
        artifact_dir
        /
        "fedprox70_phase17a_model.pt",
    )


def metric_max_abs_diff(
    clean: Mapping[str, float],
    candidate: Mapping[str, float],
    keys: Sequence[str],
) -> float:
    return max(
        abs(
            float(
                clean[
                    key
                ]
            )
            -
            float(
                candidate[
                    key
                ]
            )
        )
        for key
        in keys
    )


def per_class_recall_max_diff(
    clean: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> float:
    classes = sorted(
        clean.keys()
    )

    if classes != sorted(
        candidate.keys()
    ):
        return float(
            "inf"
        )

    return max(
        abs(
            float(
                clean[
                    class_name
                ][
                    "recall"
                ]
            )
            -
            float(
                candidate[
                    class_name
                ][
                    "recall"
                ]
            )
        )
        for class_name
        in classes
    )


def main():
    parser = argparse.ArgumentParser(
        description=PHASE_NAME
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
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--local-epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001,
    )

    parser.add_argument(
        "--mu",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--matrix",
        choices=[
            "full",
            "smoke",
        ],
        default="full",
        help=(
            "full = A1-A5; smoke = A1 clean + A2 one-client NOOP only"
        ),
    )

    parser.add_argument(
        "--allow-source-hash-mismatch",
        action="store_true",
        help=(
            "Debug-only override. Do not use for the final Phase17A evidence."
        ),
    )

    args = parser.parse_args()

    config = load_json(
        CONFIG_FILE
    )

    source_audit = verify_frozen_sources(
        config,
        allow_mismatch=
            args.allow_source_hash_mismatch,
    )

    ensure_data_policy()

    if args.mu != 0.01:
        raise ValueError(
            "Phase17A freezes FedProx mu=0.01."
        )

    phase10.set_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    scenario_dir = (
        phase4.FEDERATED_ROOT
        /
        args.scenario
    )

    if not scenario_dir.exists():
        raise FileNotFoundError(
            scenario_dir
        )

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
    ) = phase10.prepare_all_70_features(
        active_features=
            active_features,
        active_indices=
            active_indices,
        feature_columns=
            feature_columns,
    )

    if len(
        all_features
    ) != 70:
        raise RuntimeError(
            f"Expected 70 active features, got {len(all_features)}."
        )

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

    client_files = {}
    client_sample_counts = {}

    for client_id in range(
        1,
        phase10.N_CLIENTS
        +
        1,
    ):
        path = (
            scenario_dir
            /
            f"client_{client_id}.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        client_files[
            client_id
        ] = path

        client_sample_counts[
            f"client_{client_id}"
        ] = (
            phase4.count_client_samples(
                path,
                args.chunk_size,
            )
        )

    experiments = list(
        config[
            "phase17a_experiments"
        ]
    )

    if args.matrix == "smoke":
        experiments = [
            experiment
            for experiment
            in experiments
            if experiment[
                "id"
            ]
            in {
                "A1_CLEAN",
                "A2_ONE_MALICIOUS_NOOP",
                "A4_ONE_MALICIOUS_SCALE1",
            }
        ]

    print()
    print("=" * 120)
    print(PHASE_NAME)
    print("=" * 120)
    print(
        f"Scenario              : {args.scenario}"
    )
    print(
        f"Seed                  : {args.seed}"
    )
    print(
        f"Device                : {device}"
    )
    print(
        f"Features              : 70"
    )
    print(
        f"Clients               : {phase10.N_CLIENTS}"
    )
    print(
        f"Rounds                : {args.rounds}"
    )
    print(
        f"Local epochs          : {args.local_epochs}"
    )
    print(
        f"FedProx mu            : {args.mu}"
    )
    print(
        f"Evaluation            : {VALIDATION_FILE}"
    )
    print(
        f"Locked test used      : NO"
    )
    print(
        f"Matrix                : {args.matrix}"
    )
    print(
        f"Frozen source hashes  : "
        f"{'PASS' if all(row['match'] for row in source_audit.values()) else 'OVERRIDDEN'}"
    )

    print()
    print("SCIENTIFIC NOTE:")
    print(
        "Phase17A DOES NOT inject damaging poisoning. "
        "It validates the attack-routing boundary using NOOP and scale=1.0 identity transforms."
    )
    print(
        "Any later degradation in Phase17B+ should therefore be attributable to the attack, "
        "not to an accidental change in the FedProx70 harness."
    )

    runs = {}

    for experiment in experiments:
        run = run_experiment(
            experiment=
                experiment,
            args=
                args,
            scaler=
                scaler,
            feature_columns=
                feature_columns,
            label_mapping=
                label_mapping,
            all_active_indices=
                all_active_indices,
            all_features=
                all_features,
            global_weights=
                global_weights,
            client_files=
                client_files,
            client_sample_counts=
                client_sample_counts,
            device=
                device,
        )

        runs[
            run[
                "experiment_id"
            ]
        ] = run

        save_experiment(
            run=run,
            seed=args.seed,
        )

        o = run[
            "validation"
        ][
            "overall"
        ]

        print()
        print(
            f"{run['experiment_id']} VALIDATION"
        )
        print(
            f"Accuracy              : {o['accuracy']:.6f}"
        )
        print(
            f"Balanced Accuracy     : {o['balanced_accuracy']:.6f}"
        )
        print(
            f"Macro F1              : {o['macro_f1']:.6f}"
        )
        print(
            f"Weighted F1           : {o['weighted_f1']:.6f}"
        )
        print(
            f"Worst-class Recall    : {o['worst_class_recall']:.6f}"
        )
        print(
            f"Worst-attack Recall   : {o['worst_attack_recall']:.6f}"
        )
        print(
            f"Benign FPR            : {o['benign_false_positive_rate']:.6f}"
        )
        print(
            f"Validation rows used  : "
            f"{run['validation']['evaluated_project_class_rows']} / "
            f"{run['validation']['total_validation_rows']}"
        )
        print(
            f"Non-project rows skip : "
            f"{run['validation']['ignored_non_project_rows']}"
        )

    clean = runs[
        "A1_CLEAN"
    ]

    clean_overall = (
        clean[
            "validation"
        ][
            "overall"
        ]
    )

    clean_per_class = (
        clean[
            "validation"
        ][
            "per_class"
        ]
    )

    acceptance = config[
        "identity_acceptance"
    ]

    metric_keys = [
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "worst_class_recall",
        "worst_attack_recall",
        "benign_false_positive_rate",
    ]

    comparison_rows = []
    audit_rows = []

    for experiment_id, run in runs.items():
        candidate_overall = (
            run[
                "validation"
            ][
                "overall"
            ]
        )

        candidate_per_class = (
            run[
                "validation"
            ][
                "per_class"
            ]
        )

        state_diff = state_max_abs_diff(
            clean[
                "final_state"
            ],
            run[
                "final_state"
            ],
        )

        metric_diff = metric_max_abs_diff(
            clean_overall,
            candidate_overall,
            metric_keys,
        )

        class_recall_diff = (
            per_class_recall_max_diff(
                clean_per_class,
                candidate_per_class,
            )
        )

        max_event_identity_error = max(
            [
                float(
                    event.get(
                        "identity_error",
                        0.0,
                    )
                )
                for event
                in run[
                    "attack_events"
                ]
            ]
            or
            [
                0.0
            ]
        )

        is_clean = (
            experiment_id
            ==
            "A1_CLEAN"
        )

        is_noop = (
            run[
                "attack_spec"
            ][
                "attack_type"
            ]
            ==
            "NOOP"
        )

        is_scale1 = (
            run[
                "attack_spec"
            ][
                "attack_type"
            ]
            ==
            "UPDATE_SCALE"
        )

        if is_clean:
            passed = True
            reason = "clean reference"

        elif is_noop:
            passed = (
                run[
                    "final_state_sha256"
                ]
                ==
                clean[
                    "final_state_sha256"
                ]
                and
                metric_diff
                <=
                float(
                    acceptance[
                        "overall_metric_abs_tolerance"
                    ]
                )
                and
                class_recall_diff
                <=
                float(
                    acceptance[
                        "per_class_recall_abs_tolerance"
                    ]
                )
            )

            reason = (
                "NOOP must reproduce the clean model exactly and preserve metrics."
            )

        elif is_scale1:
            scale1_hash_match = (
                run[
                    "final_state_sha256"
                ]
                ==
                clean[
                    "final_state_sha256"
                ]
            )

            require_scale1_hash_match = bool(
                acceptance.get(
                    "scale1_requires_exact_state_hash_match",
                    True,
                )
            )

            passed = (
                (
                    scale1_hash_match
                    or
                    not require_scale1_hash_match
                )
                and
                state_diff
                <=
                float(
                    acceptance[
                        "scale1_max_state_abs_diff"
                    ]
                )
                and
                metric_diff
                <=
                float(
                    acceptance[
                        "overall_metric_abs_tolerance"
                    ]
                )
                and
                class_recall_diff
                <=
                float(
                    acceptance[
                        "per_class_recall_abs_tolerance"
                    ]
                )
                and
                max_event_identity_error
                <=
                float(
                    acceptance[
                        "attack_event_identity_error_tolerance"
                    ]
                )
            )

            reason = (
                "UPDATE_SCALE=1.0 must be an exact identity control: "
                "same final state hash, zero state drift, and preserved metrics."
            )

        else:
            passed = False
            reason = "unexpected experiment type"

        degradation = compare_to_clean(
            clean=clean_overall,
            candidate=candidate_overall,
        )

        comparison_rows.append(
            {
                "experiment_id":
                    experiment_id,
                "attack_type":
                    run[
                        "attack_spec"
                    ][
                        "attack_type"
                    ],
                "malicious_clients":
                    ",".join(
                        str(
                            value
                        )
                        for value
                        in run[
                            "attack_spec"
                        ][
                            "malicious_clients"
                        ]
                    ),
                "state_sha256":
                    run[
                        "final_state_sha256"
                    ],
                "exact_state_hash_match_vs_clean":
                    bool(
                        run[
                            "final_state_sha256"
                        ]
                        ==
                        clean[
                            "final_state_sha256"
                        ]
                    ),
                "max_state_abs_diff_vs_clean":
                    float(
                        state_diff
                    ),
                "max_overall_metric_abs_diff_vs_clean":
                    float(
                        metric_diff
                    ),
                "max_per_class_recall_abs_diff_vs_clean":
                    float(
                        class_recall_diff
                    ),
                "max_attack_event_identity_error":
                    float(
                        max_event_identity_error
                    ),
                **{
                    key:
                        float(
                            value
                        )
                    for key, value
                    in degradation.items()
                },
                "identity_gate_pass":
                    bool(
                        passed
                    ),
            }
        )

        audit_rows.append(
            {
                "experiment_id":
                    experiment_id,
                "passed":
                    bool(
                        passed
                    ),
                "reason":
                    reason,
            }
        )

    result_seed_dir = (
        RESULT_ROOT
        /
        f"seed_{args.seed}"
    )

    result_seed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        comparison_rows
    ).to_csv(
        result_seed_dir
        /
        "phase17a_clean_vs_harness_comparison.csv",
        index=False,
    )

    all_passed = all(
        row[
            "passed"
        ]
        for row
        in audit_rows
    )

    audit = {
        "phase":
            "17A",
        "version":
            config[
                "version"
            ],
        "scenario":
            args.scenario,
        "seed":
            int(
                args.seed
            ),
        "matrix":
            args.matrix,
        "source_hash_audit":
            source_audit,
        "validation_file":
            str(
                VALIDATION_FILE
            ),
        "locked_test_file":
            str(
                LOCKED_TEST_FILE
            ),
        "locked_test_used":
            False,
        "experiments":
            audit_rows,
        "identity_acceptance":
            acceptance,
        "all_tests_passed":
            bool(
                all_passed
            ),
        "damaging_poisoning_injected":
            False,
    }

    save_json(
        result_seed_dir
        /
        "phase17a_audit.json",
        audit,
    )

    manifest = {
        "phase":
            "17A",
        "version":
            config[
                "version"
            ],
        "status":
            (
                "FROZEN"
                if all_passed
                else
                "NOT_FROZEN"
            ),
        "threat_model":
            config[
                "threat_model"
            ],
        "baseline":
            {
                **config[
                    "baseline"
                ],
                "scenario":
                    args.scenario,
                "seed":
                    int(
                        args.seed
                    ),
                "rounds":
                    int(
                        args.rounds
                    ),
                "local_epochs":
                    int(
                        args.local_epochs
                    ),
            },
        "data_policy":
            config[
                "data_policy"
            ],
        "experiment_matrix":
            [
                {
                    "id":
                        run[
                            "experiment_id"
                        ],
                    "attack_spec":
                        run[
                            "attack_spec"
                        ],
                    "final_state_sha256":
                        run[
                            "final_state_sha256"
                        ],
                }
                for run
                in runs.values()
            ],
        "scientific_interpretation":
            (
                "Phase17A validates the malicious-client routing/evaluation harness only. "
                "It does not measure poisoning vulnerability because NOOP and scale=1.0 "
                "are intentionally non-damaging identity controls."
            ),
        "locked_test_used":
            False,
        "ready_for_phase17b":
            bool(
                all_passed
            ),
    }

    ARTIFACT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        ARTIFACT_ROOT
        /
        "phase17a_threat_model_manifest.json",
        manifest,
    )

    print()
    print("=" * 120)
    print("PHASE 17A COMPLETE")
    print("=" * 120)

    for row in comparison_rows:
        print(
            f"[{'PASS' if row['identity_gate_pass'] else 'FAIL'}] "
            f"{row['experiment_id']:<30} | "
            f"metric_diff={row['max_overall_metric_abs_diff_vs_clean']:.8f} | "
            f"state_diff={row['max_state_abs_diff_vs_clean']:.8f}"
        )

    print()
    print(
        f"Threat model frozen      : {'YES' if all_passed else 'NO'}"
    )
    print(
        f"Damaging poisoning       : NO"
    )
    print(
        f"Locked test used         : NO"
    )
    print(
        f"Audit                    : "
        f"{result_seed_dir / 'phase17a_audit.json'}"
    )
    print(
        f"Comparison               : "
        f"{result_seed_dir / 'phase17a_clean_vs_harness_comparison.csv'}"
    )
    print(
        f"Manifest                 : "
        f"{ARTIFACT_ROOT / 'phase17a_threat_model_manifest.json'}"
    )
    print()
    print(
        f"STATUS                    : {'PASS' if all_passed else 'FAIL'}"
    )

    print()
    print("NEXT:")
    print(
        "Phase 17B — controlled label-flipping poisoning using this frozen harness. "
        "Use validation data for development; keep test_locked.csv out of attack tuning."
    )

    if not all_passed:
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()
