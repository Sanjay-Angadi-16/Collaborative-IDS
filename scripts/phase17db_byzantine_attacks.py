from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

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
        "\nPhase17D-B requires:\n"
        "    scripts\\phase10_fedprox70.py\n"
    ) from exc

phase4 = phase10.phase4

from src.federated.poisoning.byzantine_attacks import (  # noqa: E402
    ByzantineAttackSpec,
    apply_byzantine_attack,
)
from src.federated.poisoning.metrics import (  # noqa: E402
    calculate_robustness_metrics,
    compare_to_clean,
)


PHASE_NAME = (
    "PHASE 17D-B — SCALING + RANDOM BYZANTINE UPDATE CHARACTERIZATION"
)

CONFIG_FILE = (
    PROJECT_ROOT
    / "configs"
    / "phase17db_byzantine_attack_policy.json"
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

PHASE17A_MANIFEST = (
    PROJECT_ROOT
    / "artifacts"
    / "poisoning"
    / "phase17a"
    / "phase17a_threat_model_manifest.json"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "phase17"
    / "phase17db"
)

ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "poisoning"
    / "phase17db"
)

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(
    "phase17db"
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
        tensor = (
            state[
                key
            ]
            .detach()
            .cpu()
            .contiguous()
        )

        digest.update(
            key.encode(
                "utf-8"
            )
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


def verify_prerequisites(
    config: Mapping[str, Any],
    *,
    allow_source_hash_mismatch: bool,
) -> Dict[str, Any]:
    if not PHASE17A_MANIFEST.exists():
        raise FileNotFoundError(
            "\nPhase17A frozen manifest is missing:\n"
            f"{PHASE17A_MANIFEST}"
        )

    phase17a = load_json(
        PHASE17A_MANIFEST
    )

    if phase17a.get(
        "status"
    ) != "FROZEN":
        raise RuntimeError(
            "Phase17A threat model is not frozen."
        )

    if phase17a.get(
        "locked_test_used",
        True,
    ):
        raise RuntimeError(
            "Phase17A manifest reports locked-test use."
        )

    source_audit = {}

    for relative, expected in config[
        "frozen_source_hashes"
    ].items():
        path = (
            PROJECT_ROOT
            /
            relative
        )

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        actual = sha256_file(
            path
        )

        match = (
            actual
            ==
            expected
        )

        source_audit[
            relative
        ] = {
            "expected_sha256":
                expected,
            "actual_sha256":
                actual,
            "match":
                match,
        }

        if (
            not match
            and
            not allow_source_hash_mismatch
        ):
            raise RuntimeError(
                "\nFrozen baseline source hash mismatch.\n"
                f"File     : {path}\n"
                f"Expected : {expected}\n"
                f"Actual   : {actual}\n\n"
                "Do not silently continue with a changed baseline. "
                "If the change was intentional, version the experiment. "
                "Use --allow-source-hash-mismatch only for debugging."
            )

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            VALIDATION_FILE
        )

    return {
        "phase17a_status":
            phase17a.get(
                "status"
            ),
        "phase17a_locked_test_used":
            phase17a.get(
                "locked_test_used"
            ),
        "source_hashes":
            source_audit,
    }


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
    model.eval()

    canonical = {
        str(
            label
        )
        .strip()
        .lower():
            str(
                label
            ).strip()
        for label
        in label_mapping.keys()
    }

    total_rows = 0
    evaluated_rows = 0
    ignored_rows = 0
    ignored_labels: Dict[
        str,
        int,
    ] = {}

    all_true: List[
        np.ndarray
    ] = []

    all_pred: List[
        np.ndarray
    ] = []

    inference_start = (
        time.perf_counter()
    )

    for chunk in pd.read_csv(
        VALIDATION_FILE,
        chunksize=chunk_size,
    ):
        total_rows += int(
            len(
                chunk
            )
        )

        raw_labels = (
            chunk[
                phase4.LABEL_COL
            ]
            .astype(
                str
            )
            .str
            .strip()
        )

        keep_mask = (
            raw_labels
            .str
            .lower()
            .isin(
                canonical.keys()
            )
        )

        ignored = (
            raw_labels[
                ~keep_mask
            ]
            .value_counts()
        )

        for label, count in ignored.items():
            ignored_labels[
                str(
                    label
                )
            ] = (
                ignored_labels.get(
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

        ignored_rows += int(
            (
                ~keep_mask
            ).sum()
        )

        chunk = (
            chunk[
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
            .astype(
                str
            )
            .str
            .strip()
            .str
            .lower()
            .map(
                canonical
            )
        )

        evaluated_rows += int(
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

            predictions.append(
                torch.argmax(
                    logits,
                    dim=1,
                )
                .cpu()
                .numpy()
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
            "Validation produced zero six-class samples."
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

    metrics.update(
        {
            "evaluation_source":
                str(
                    VALIDATION_FILE
                ),
            "total_validation_rows":
                int(
                    total_rows
                ),
            "evaluated_project_class_rows":
                int(
                    evaluated_rows
                ),
            "ignored_non_project_rows":
                int(
                    ignored_rows
                ),
            "ignored_non_project_labels":
                dict(
                    sorted(
                        ignored_labels.items()
                    )
                ),
            "inference_time_seconds":
                float(
                    time.perf_counter()
                    -
                    inference_start
                ),
            "locked_test_used":
                False,
        }
    )

    return metrics


def run_experiment(
    *,
    experiment: Mapping[str, Any],
    args,
    scaler,
    feature_columns,
    label_mapping,
    active_indices,
    active_features,
    global_weights,
    client_files,
    client_sample_counts,
    device: torch.device,
) -> Dict[str, Any]:
    spec = ByzantineAttackSpec(
        attack_type=str(
            experiment[
                "attack_type"
            ]
        ),
        malicious_clients=tuple(
            int(
                value
            )
            for value
            in experiment[
                "malicious_clients"
            ]
        ),
        magnitude=float(
            experiment[
                "magnitude"
            ]
        ),
        attack_seed=int(
            args.attack_seed
        ),
    ).validate()

    experiment_id = str(
        experiment[
            "id"
        ]
    )

    print()
    print("=" * 120)
    print(
        experiment_id
    )
    print("=" * 120)
    print(
        f"Attack type          : {spec.attack_type}"
    )
    print(
        f"Malicious clients    : {list(spec.malicious_clients)}"
    )
    print(
        f"Magnitude            : {spec.magnitude}"
    )
    print(
        f"Aggregation          : sample-weighted mean"
    )
    print(
        f"Robust defense       : NONE"
    )
    print(
        f"Locked test used     : NO"
    )

    global_model = (
        phase10.initialize_full_model(
            input_dim=len(
                active_features
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
        local_accuracies = []

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
                    active_indices,
                label_mapping=
                    label_mapping,
                class_weights=
                    global_weights,
                input_dim=
                    len(
                        active_features
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

            attacked_state, event = (
                apply_byzantine_attack(
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
                    experiment_seed=
                        args.seed,
                )
            )

            attack_events.append(
                event
            )

            client_states.append(
                attacked_state
            )

            aggregation_weights.append(
                client_sample_counts[
                    f"client_{client_id}"
                ]
            )

            local_losses.append(
                float(
                    local_loss
                )
            )

            local_accuracies.append(
                float(
                    local_accuracy
                )
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

        global_update_norm = (
            phase4.state_update_norm(
                old_global_state,
                aggregated_state,
            )
        )

        weight_array = np.asarray(
            aggregation_weights,
            dtype=np.float64,
        )

        malicious_events = [
            event
            for event
            in attack_events
            if (
                event[
                    "round"
                ]
                ==
                round_number
                and
                event[
                    "declared_malicious"
                ]
            )
        ]

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
                                weight_array,
                        )
                    ),
                "weighted_local_accuracy":
                    float(
                        np.average(
                            local_accuracies,
                            weights=
                                weight_array,
                        )
                    ),
                "global_update_norm":
                    float(
                        global_update_norm
                    ),
                "malicious_update_mean_norm_ratio":
                    (
                        float(
                            np.mean(
                                [
                                    event[
                                        "observed_norm_ratio"
                                    ]
                                    for event
                                    in malicious_events
                                ]
                            )
                        )
                        if malicious_events
                        else
                        1.0
                    ),
                "malicious_update_mean_cosine":
                    (
                        float(
                            np.mean(
                                [
                                    event[
                                        "update_cosine_similarity"
                                    ]
                                    for event
                                    in malicious_events
                                ]
                            )
                        )
                        if malicious_events
                        else
                        1.0
                    ),
            }
        )

        print(
            f"Round {round_number:02d}/{args.rounds} | "
            f"loss={history[-1]['weighted_local_loss']:.6f} | "
            f"acc={history[-1]['weighted_local_accuracy']:.6f} | "
            f"global_norm={history[-1]['global_update_norm']:.6f} | "
            f"mal_ratio={history[-1]['malicious_update_mean_norm_ratio']:.4f}"
        )

    validation = evaluate_on_validation(
        model=global_model,
        scaler=scaler,
        feature_columns=feature_columns,
        active_indices=active_indices,
        label_mapping=label_mapping,
        device=device,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )

    validation[
        "training_time_seconds"
    ] = float(
        time.perf_counter()
        -
        training_start
    )

    final_state = {
        key:
            value.detach().cpu().clone()
        for key, value
        in global_model.state_dict().items()
    }

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
                "magnitude":
                    float(
                        spec.magnitude
                    ),
                "attack_seed":
                    int(
                        spec.attack_seed
                    ),
            },
        "history":
            history,
        "attack_events":
            attack_events,
        "validation":
            validation,
        "final_state":
            final_state,
        "final_state_sha256":
            sha256_state(
                final_state
            ),
    }


def audit_attack_events(
    *,
    run: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    spec = run[
        "attack_spec"
    ]

    attack_type = spec[
        "attack_type"
    ]

    magnitude = float(
        spec[
            "magnitude"
        ]
    )

    malicious = [
        event
        for event
        in run[
            "attack_events"
        ]
        if event[
            "declared_malicious"
        ]
    ]

    if attack_type == "NONE":
        return {
            "attack_type":
                "NONE",
            "malicious_events":
                0,
            "passed":
                (
                    len(
                        malicious
                    )
                    ==
                    0
                ),
            "reason":
                "Clean control contains no malicious transformation.",
        }

    expected_count = (
        len(
            spec[
                "malicious_clients"
            ]
        )
        *
        len(
            run[
                "history"
            ]
        )
    )

    if len(
        malicious
    ) != expected_count:
        return {
            "attack_type":
                attack_type,
            "malicious_events":
                len(
                    malicious
                ),
            "expected_malicious_events":
                expected_count,
            "passed":
                False,
            "reason":
                "Unexpected malicious-event count.",
        }

    if attack_type == "SCALE":
        ratio_tolerance = float(
            config[
                "attack_audit"
            ][
                "scale_norm_ratio_abs_tolerance"
            ]
        )

        cosine_minimum = float(
            config[
                "attack_audit"
            ][
                "scale_cosine_minimum"
            ]
        )

        ratio_error = max(
            float(
                event[
                    "norm_ratio_abs_error"
                ]
            )
            for event
            in malicious
        )

        min_cosine = min(
            float(
                event[
                    "update_cosine_similarity"
                ]
            )
            for event
            in malicious
        )

        passed = (
            ratio_error
            <=
            ratio_tolerance
            and
            min_cosine
            >=
            cosine_minimum
        )

        return {
            "attack_type":
                attack_type,
            "malicious_events":
                len(
                    malicious
                ),
            "requested_magnitude":
                magnitude,
            "max_norm_ratio_abs_error":
                ratio_error,
            "minimum_update_cosine":
                min_cosine,
            "passed":
                bool(
                    passed
                ),
            "reason":
                (
                    "Positive scaling must preserve update direction "
                    "and multiply the full-model update norm by the requested factor."
                ),
        }

    if attack_type == "RANDOM":
        ratio_tolerance = float(
            config[
                "attack_audit"
            ][
                "random_norm_ratio_abs_tolerance"
            ]
        )

        ratio_error = max(
            float(
                event[
                    "norm_ratio_abs_error"
                ]
            )
            for event
            in malicious
        )

        all_seeded = all(
            event.get(
                "random_seed_used"
            )
            is not None
            for event
            in malicious
        )

        passed = (
            ratio_error
            <=
            ratio_tolerance
            and
            all_seeded
        )

        return {
            "attack_type":
                attack_type,
            "malicious_events":
                len(
                    malicious
                ),
            "requested_magnitude":
                magnitude,
            "max_norm_ratio_abs_error":
                ratio_error,
            "all_random_events_seeded":
                bool(
                    all_seeded
                ),
            "mean_abs_cosine_with_legitimate_update":
                float(
                    np.mean(
                        [
                            abs(
                                float(
                                    event[
                                        "update_cosine_similarity"
                                    ]
                                )
                            )
                            for event
                            in malicious
                        ]
                    )
                ),
            "passed":
                bool(
                    passed
                ),
            "reason":
                (
                    "Random Byzantine direction must be deterministic and "
                    "have the requested norm relative to the legitimate update."
                ),
        }

    return {
        "passed":
            False,
        "reason":
            "Unknown attack type.",
    }


def save_run(
    *,
    run: Mapping[str, Any],
    seed: int,
) -> None:
    result_dir = (
        RESULT_ROOT
        /
        f"seed_{seed}"
        /
        run[
            "experiment_id"
        ]
    )

    artifact_dir = (
        ARTIFACT_ROOT
        /
        f"seed_{seed}"
        /
        run[
            "experiment_id"
        ]
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        result_dir
        /
        "attack_spec.json",
        run[
            "attack_spec"
        ],
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

    pd.DataFrame(
        [
            {
                "class":
                    name,
                **metrics,
            }
            for name, metrics
            in run[
                "validation"
            ][
                "per_class"
            ].items()
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
                "17D-B",
            "experiment_id":
                run[
                    "experiment_id"
                ],
            "seed":
                int(
                    seed
                ),
            "attack_spec":
                run[
                    "attack_spec"
                ],
            "state_sha256":
                run[
                    "final_state_sha256"
                ],
            "model_state_dict":
                run[
                    "final_state"
                ],
        },
        artifact_dir
        /
        "fedprox70_phase17db.pt",
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
        "--attack-seed",
        type=int,
        default=1704,
    )

    parser.add_argument(
        "--matrix",
        choices=[
            "full",
            "smoke",
            "scale",
            "random",
        ],
        default="full",
    )

    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help=(
            "Run exactly one experiment id, e.g. DB_SCALE_X5."
        ),
    )

    parser.add_argument(
        "--allow-source-hash-mismatch",
        action="store_true",
    )

    args = parser.parse_args()

    if args.mu != 0.01:
        raise ValueError(
            "Phase17D-B freezes FedProx mu=0.01."
        )

    config = load_json(
        CONFIG_FILE
    )

    prerequisite_audit = (
        verify_prerequisites(
            config,
            allow_source_hash_mismatch=
                args.allow_source_hash_mismatch,
        )
    )

    experiments = list(
        config[
            "experiments"
        ]
    )

    if args.only is not None:
        experiments = [
            experiment
            for experiment
            in experiments
            if experiment[
                "id"
            ]
            ==
            args.only
        ]

        if not experiments:
            raise ValueError(
                f"Unknown --only experiment id: {args.only}"
            )

    elif args.matrix == "smoke":
        allowed = {
            "DB0_CLEAN",
            "DB_SCALE_X2",
            "DB_RANDOM_X1",
        }

        experiments = [
            experiment
            for experiment
            in experiments
            if experiment[
                "id"
            ]
            in allowed
        ]

    elif args.matrix == "scale":
        experiments = [
            experiment
            for experiment
            in experiments
            if (
                experiment[
                    "attack_type"
                ]
                in {
                    "NONE",
                    "SCALE",
                }
            )
        ]

    elif args.matrix == "random":
        experiments = [
            experiment
            for experiment
            in experiments
            if (
                experiment[
                    "attack_type"
                ]
                in {
                    "NONE",
                    "RANDOM",
                }
            )
        ]

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
            f"Expected 70 active features; found {len(all_features)}."
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
        f"Malicious client      : client_1 (20%)"
    )
    print(
        f"Client_1 attack class : {phase4.CLIENT_ATTACKS[1]}"
    )
    print(
        f"Rounds                : {args.rounds}"
    )
    print(
        f"FedProx mu            : {args.mu}"
    )
    print(
        f"Aggregation           : sample-weighted mean"
    )
    print(
        f"Robust defense        : NONE"
    )
    print(
        f"Locked test used      : NO"
    )
    print(
        f"Matrix                : {args.matrix}"
    )

    print()
    print("SCIENTIFIC NOTE:")
    print(
        "This phase characterizes attacks only. Median/Trimmed Mean or other "
        "robust defenses are intentionally NOT enabled."
    )
    print(
        "SCALE changes update magnitude while preserving direction. RANDOM "
        "replaces the direction with deterministic seeded noise at a controlled norm."
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
            active_indices=
                all_active_indices,
            active_features=
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

        run[
            "attack_audit"
        ] = audit_attack_events(
            run=run,
            config=config,
        )

        runs[
            run[
                "experiment_id"
            ]
        ] = run

        save_run(
            run=run,
            seed=args.seed,
        )

        overall = run[
            "validation"
        ][
            "overall"
        ]

        hulk_recall = (
            run[
                "validation"
            ][
                "per_class"
            ][
                phase4.CLIENT_ATTACKS[
                    1
                ]
            ][
                "recall"
            ]
        )

        print()
        print(
            f"{run['experiment_id']} VALIDATION"
        )
        print(
            f"Accuracy             : {overall['accuracy']:.6f}"
        )
        print(
            f"Balanced Accuracy    : {overall['balanced_accuracy']:.6f}"
        )
        print(
            f"Macro F1             : {overall['macro_f1']:.6f}"
        )
        print(
            f"Weighted F1          : {overall['weighted_f1']:.6f}"
        )
        print(
            f"Worst attack Recall  : {overall['worst_attack_recall']:.6f}"
        )
        print(
            f"DoS Hulk Recall      : {hulk_recall:.6f}"
        )
        print(
            f"Attack audit         : "
            f"{'PASS' if run['attack_audit']['passed'] else 'FAIL'}"
        )

    clean = runs.get(
        "DB0_CLEAN"
    )

    comparison_rows = []

    if clean is not None:
        clean_overall = (
            clean[
                "validation"
            ][
                "overall"
            ]
        )

        clean_hulk = float(
            clean[
                "validation"
            ][
                "per_class"
            ][
                phase4.CLIENT_ATTACKS[
                    1
                ]
            ][
                "recall"
            ]
        )

        for experiment_id, run in runs.items():
            overall = (
                run[
                    "validation"
                ][
                    "overall"
                ]
            )

            hulk = float(
                run[
                    "validation"
                ][
                    "per_class"
                ][
                    phase4.CLIENT_ATTACKS[
                        1
                    ]
                ][
                    "recall"
                ]
            )

            degradation = compare_to_clean(
                clean=
                    clean_overall,
                candidate=
                    overall,
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
                    "magnitude":
                        run[
                            "attack_spec"
                        ][
                            "magnitude"
                        ],
                    "accuracy":
                        overall[
                            "accuracy"
                        ],
                    "balanced_accuracy":
                        overall[
                            "balanced_accuracy"
                        ],
                    "macro_f1":
                        overall[
                            "macro_f1"
                        ],
                    "weighted_f1":
                        overall[
                            "weighted_f1"
                        ],
                    "worst_attack_recall":
                        overall[
                            "worst_attack_recall"
                        ],
                    "dos_hulk_recall":
                        hulk,
                    "dos_hulk_recall_drop":
                        float(
                            clean_hulk
                            -
                            hulk
                        ),
                    **degradation,
                    "attack_audit_pass":
                        bool(
                            run[
                                "attack_audit"
                            ][
                                "passed"
                            ]
                        ),
                }
            )

        seed_result_dir = (
            RESULT_ROOT
            /
            f"seed_{args.seed}"
        )

        seed_result_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        pd.DataFrame(
            comparison_rows
        ).to_csv(
            seed_result_dir
            /
            "phase17db_clean_vs_byzantine_comparison.csv",
            index=False,
        )

    all_audits_passed = all(
        bool(
            run[
                "attack_audit"
            ][
                "passed"
            ]
        )
        for run
        in runs.values()
    )

    seed_result_dir = (
        RESULT_ROOT
        /
        f"seed_{args.seed}"
    )

    seed_result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit = {
        "phase":
            "17D-B",
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
        "rounds":
            int(
                args.rounds
            ),
        "matrix":
            args.matrix,
        "experiments_completed":
            list(
                runs.keys()
            ),
        "attack_audits":
            {
                experiment_id:
                    run[
                        "attack_audit"
                    ]
                for experiment_id, run
                in runs.items()
            },
        "prerequisite_audit":
            prerequisite_audit,
        "robust_defense_used":
            False,
        "locked_test_used":
            False,
        "all_tests_passed":
            bool(
                all_audits_passed
            ),
        "status_meaning":
            (
                "PASS means attack implementation/audit integrity passed. "
                "It does not imply attack effectiveness or model robustness."
            ),
    }

    save_json(
        seed_result_dir
        /
        "phase17db_audit.json",
        audit,
    )

    ARTIFACT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        "phase":
            "17D-B",
        "version":
            config[
                "version"
            ],
        "status":
            (
                "PASS"
                if all_audits_passed
                else
                "FAIL"
            ),
        "baseline":
            config[
                "baseline"
            ],
        "threat_model":
            config[
                "threat_model"
            ],
        "experiments":
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
                    "attack_audit":
                        run[
                            "attack_audit"
                        ],
                    "final_state_sha256":
                        run[
                            "final_state_sha256"
                        ],
                }
                for run
                in runs.values()
            ],
        "scientific_rules":
            config[
                "scientific_rules"
            ],
        "locked_test_used":
            False,
        "recommended_next_step":
            (
                "After reviewing attack effectiveness, proceed to Phase17E "
                "malicious-client identity/ratio analysis. Do not tune a defense "
                "from these test-free development runs unless the defense phase is explicitly versioned."
            ),
    }

    save_json(
        ARTIFACT_ROOT
        /
        "phase17db_manifest.json",
        manifest,
    )

    print()
    print("=" * 120)
    print("PHASE 17D-B COMPLETE")
    print("=" * 120)

    for run in runs.values():
        overall = run[
            "validation"
        ][
            "overall"
        ]

        print(
            f"{run['experiment_id']:<20} | "
            f"MacroF1={overall['macro_f1']:.6f} | "
            f"BA={overall['balanced_accuracy']:.6f} | "
            f"Audit={'PASS' if run['attack_audit']['passed'] else 'FAIL'}"
        )

    print()
    print(
        f"Attack implementation audit : "
        f"{'PASS' if all_audits_passed else 'FAIL'}"
    )
    print(
        f"Robust defense used         : NO"
    )
    print(
        f"Locked test used            : NO"
    )
    print(
        f"Results                     : {seed_result_dir}"
    )
    print(
        f"Manifest                    : "
        f"{ARTIFACT_ROOT / 'phase17db_manifest.json'}"
    )
    print()
    print(
        f"STATUS                      : "
        f"{'PASS' if all_audits_passed else 'FAIL'}"
    )

    if not all_audits_passed:
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()
