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
        "\nPhase17E requires:\n"
        "    scripts\\phase10_fedprox70.py\n"
    ) from exc

phase4 = phase10.phase4

try:
    from src.federated.poisoning.byzantine_attacks import (
        ByzantineAttackSpec,
        apply_byzantine_attack,
    )
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "\nPhase17E requires the Phase17D-B helper:\n"
        "    src\\federated\\poisoning\\byzantine_attacks.py\n\n"
        "Extract the Phase17D-B ZIP into the project root first."
    ) from exc

try:
    from src.federated.poisoning.metrics import (
        calculate_robustness_metrics,
        compare_to_clean,
    )
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "\nPhase17E requires the Phase17A metrics helper:\n"
        "    src\\federated\\poisoning\\metrics.py\n"
    ) from exc

from src.federated.poisoning.phase17e_analysis import (
    build_summary,
    target_class_summary,
)


PHASE_NAME = (
    "PHASE 17E — MALICIOUS-CLIENT IDENTITY + 0%/20%/40% RATIO ANALYSIS"
)

CONFIG_FILE = (
    PROJECT_ROOT
    / "configs"
    / "phase17e_identity_ratio_policy.json"
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

PHASE17DB_MANIFEST = (
    PROJECT_ROOT
    / "artifacts"
    / "poisoning"
    / "phase17db"
    / "phase17db_manifest.json"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "phase17"
    / "phase17e"
)

ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "poisoning"
    / "phase17e"
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
    "phase17e"
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
            "Phase17A reports locked-test use."
        )

    phase17db = load_json(
        PHASE17DB_MANIFEST
    )

    if phase17db.get(
        "status"
    ) != "PASS":
        raise RuntimeError(
            "Phase17D-B attack characterization did not PASS."
        )

    if phase17db.get(
        "locked_test_used",
        True,
    ):
        raise RuntimeError(
            "Phase17D-B reports locked-test use."
        )

    source_audit = {}

    for relative_path, expected_hash in config[
        "frozen_source_hashes"
    ].items():
        path = (
            PROJECT_ROOT
            /
            relative_path
        )

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        actual_hash = sha256_file(
            path
        )

        match = (
            actual_hash
            ==
            expected_hash
        )

        source_audit[
            relative_path
        ] = {
            "expected_sha256":
                expected_hash,
            "actual_sha256":
                actual_hash,
            "match":
                bool(
                    match
                ),
        }

        if (
            not match
            and
            not allow_source_hash_mismatch
        ):
            raise RuntimeError(
                "\nFrozen source hash mismatch.\n"
                f"File     : {path}\n"
                f"Expected : {expected_hash}\n"
                f"Actual   : {actual_hash}\n\n"
                "Version the experiment if the baseline changed. "
                "Use --allow-source-hash-mismatch only for debugging."
            )

    return {
        "phase17a_status":
            phase17a.get(
                "status"
            ),
        "phase17db_status":
            phase17db.get(
                "status"
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

    start_time = time.perf_counter()

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
            "No six-class validation rows were evaluated."
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
                    start_time
                ),
            "locked_test_used":
                False,
        }
    )

    return metrics


def audit_attack_events(
    *,
    attack_type: str,
    magnitude: float,
    malicious_clients: Sequence[int],
    attack_events: Sequence[Mapping[str, Any]],
    rounds: int,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    malicious_events = [
        event
        for event
        in attack_events
        if event[
            "declared_malicious"
        ]
    ]

    if attack_type == "NONE":
        return {
            "attack_type":
                "NONE",
            "malicious_events":
                len(
                    malicious_events
                ),
            "passed":
                (
                    len(
                        malicious_events
                    )
                    ==
                    0
                ),
        }

    expected_events = (
        len(
            malicious_clients
        )
        *
        int(
            rounds
        )
    )

    if len(
        malicious_events
    ) != expected_events:
        return {
            "attack_type":
                attack_type,
            "expected_malicious_events":
                expected_events,
            "observed_malicious_events":
                len(
                    malicious_events
                ),
            "passed":
                False,
            "reason":
                "Unexpected malicious-event count.",
        }

    ratio_errors = [
        float(
            event[
                "norm_ratio_abs_error"
            ]
        )
        for event
        in malicious_events
    ]

    if attack_type == "SCALE":
        max_ratio_error = max(
            ratio_errors
        )

        min_cosine = min(
            float(
                event[
                    "update_cosine_similarity"
                ]
            )
            for event
            in malicious_events
        )

        passed = (
            max_ratio_error
            <=
            float(
                config[
                    "attack_audit"
                ][
                    "scale_norm_ratio_abs_tolerance"
                ]
            )
            and
            min_cosine
            >=
            float(
                config[
                    "attack_audit"
                ][
                    "scale_cosine_minimum"
                ]
            )
        )

        return {
            "attack_type":
                attack_type,
            "magnitude":
                float(
                    magnitude
                ),
            "malicious_events":
                len(
                    malicious_events
                ),
            "max_norm_ratio_abs_error":
                float(
                    max_ratio_error
                ),
            "minimum_update_cosine":
                float(
                    min_cosine
                ),
            "passed":
                bool(
                    passed
                ),
        }

    if attack_type == "RANDOM":
        max_ratio_error = max(
            ratio_errors
        )

        all_seeded = all(
            event.get(
                "random_seed_used"
            )
            is not None
            for event
            in malicious_events
        )

        passed = (
            max_ratio_error
            <=
            float(
                config[
                    "attack_audit"
                ][
                    "random_norm_ratio_abs_tolerance"
                ]
            )
            and
            all_seeded
        )

        return {
            "attack_type":
                attack_type,
            "magnitude":
                float(
                    magnitude
                ),
            "malicious_events":
                len(
                    malicious_events
                ),
            "max_norm_ratio_abs_error":
                float(
                    max_ratio_error
                ),
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
                            in malicious_events
                        ]
                    )
                ),
            "passed":
                bool(
                    passed
                ),
        }

    return {
        "passed":
            False,
        "reason":
            "Unknown attack type.",
    }


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
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    experiment_id = str(
        experiment[
            "id"
        ]
    )

    malicious_clients = tuple(
        int(
            value
        )
        for value
        in experiment[
            "malicious_clients"
        ]
    )

    attack_type = str(
        experiment[
            "attack_type"
        ]
    )

    magnitude = float(
        experiment[
            "magnitude"
        ]
    )

    spec = ByzantineAttackSpec(
        attack_type=
            attack_type,
        malicious_clients=
            malicious_clients,
        magnitude=
            magnitude,
        attack_seed=
            int(
                args.attack_seed
            ),
    ).validate()

    malicious_ratio = float(
        len(
            malicious_clients
        )
        /
        phase10.N_CLIENTS
    )

    specialized_attacks = [
        phase4.CLIENT_ATTACKS[
            int(
                client_id
            )
        ]
        for client_id
        in malicious_clients
    ]

    print()
    print("=" * 124)
    print(
        experiment_id
    )
    print("=" * 124)
    print(
        f"Attack type          : {attack_type}"
    )
    print(
        f"Magnitude            : {magnitude}"
    )
    print(
        f"Malicious clients    : {list(malicious_clients)}"
    )
    print(
        f"Malicious ratio      : {malicious_ratio * 100:.0f}%"
    )
    print(
        f"Specialized attacks  : {specialized_attacks}"
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

            routed_state, event = (
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
                routed_state
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

        malicious_round_events = [
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

        weight_array = np.asarray(
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
                                    in malicious_round_events
                                ]
                            )
                        )
                        if malicious_round_events
                        else
                        1.0
                    ),
                "malicious_update_mean_abs_cosine":
                    (
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
                                    in malicious_round_events
                                ]
                            )
                        )
                        if malicious_round_events
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

    target_summary = target_class_summary(
        validation=validation,
        malicious_clients=malicious_clients,
        client_attacks=phase4.CLIENT_ATTACKS,
    )

    attack_audit = audit_attack_events(
        attack_type=attack_type,
        magnitude=magnitude,
        malicious_clients=malicious_clients,
        attack_events=attack_events,
        rounds=args.rounds,
        config=config,
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
        "attack_type":
            attack_type,
        "magnitude":
            magnitude,
        "malicious_clients":
            list(
                malicious_clients
            ),
        "malicious_ratio":
            malicious_ratio,
        "specialized_attacks":
            specialized_attacks,
        "history":
            history,
        "attack_events":
            attack_events,
        "attack_audit":
            attack_audit,
        "validation":
            validation,
        "target_summary":
            target_summary,
        "final_state":
            final_state,
        "final_state_sha256":
            sha256_state(
                final_state
            ),
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
        "experiment.json",
        {
            "experiment_id":
                run[
                    "experiment_id"
                ],
            "attack_type":
                run[
                    "attack_type"
                ],
            "magnitude":
                run[
                    "magnitude"
                ],
            "malicious_clients":
                run[
                    "malicious_clients"
                ],
            "malicious_ratio":
                run[
                    "malicious_ratio"
                ],
            "specialized_attacks":
                run[
                    "specialized_attacks"
                ],
        },
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
        "attack_audit.json",
        run[
            "attack_audit"
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

    save_json(
        result_dir
        /
        "target_attack_summary.json",
        run[
            "target_summary"
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
                    class_name,
                **values,
            }
            for class_name, values
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

    torch.save(
        {
            "phase":
                "17E",
            "experiment_id":
                run[
                    "experiment_id"
                ],
            "seed":
                int(
                    seed
                ),
            "attack_type":
                run[
                    "attack_type"
                ],
            "magnitude":
                run[
                    "magnitude"
                ],
            "malicious_clients":
                run[
                    "malicious_clients"
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
        "fedprox70_phase17e.pt",
    )


def make_experiment_set(
    *,
    config: Mapping[str, Any],
    matrix: str,
    only: str | None,
) -> list[dict[str, Any]]:
    clean = {
        "id": "E0_CLEAN",
        "attack_type": "NONE",
        "magnitude": 1.0,
        "malicious_clients": [],
    }

    identity = list(
        config[
            "identity_experiments"
        ]
    )

    ratio = list(
        config[
            "ratio_experiments"
        ]
    )

    if matrix == "identity":
        experiments = [
            clean,
            *identity,
        ]

    elif matrix == "ratio":
        experiments = ratio

    elif matrix == "smoke":
        experiments = [
            clean,
            next(
                item
                for item
                in identity
                if item[
                    "id"
                ]
                ==
                "E_SCALE_C1"
            ),
            next(
                item
                for item
                in identity
                if item[
                    "id"
                ]
                ==
                "E_RANDOM_C1"
            ),
            next(
                item
                for item
                in ratio
                if item[
                    "id"
                ]
                ==
                "E_SCALE_C1_C2"
            ),
        ]

    elif matrix == "full":
        by_id = {
            clean[
                "id"
            ]:
                clean,
        }

        for item in [
            *identity,
            *ratio,
        ]:
            by_id[
                item[
                    "id"
                ]
            ] = item

        experiments = list(
            by_id.values()
        )

    else:
        raise ValueError(
            matrix
        )

    if only is not None:
        selected = [
            item
            for item
            in experiments
            if item[
                "id"
            ]
            ==
            only
        ]

        if not selected:
            # allow --only against the complete policy, even if matrix is different
            all_items = {
                clean[
                    "id"
                ]:
                    clean,
            }

            for item in [
                *identity,
                *ratio,
            ]:
                all_items[
                    item[
                        "id"
                    ]
                ] = item

            if only not in all_items:
                raise ValueError(
                    f"Unknown experiment id: {only}"
                )

            selected = [
                all_items[
                    only
                ]
            ]

        experiments = selected

    return experiments


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
            "smoke",
            "identity",
            "ratio",
            "full",
        ],
        default="full",
    )

    parser.add_argument(
        "--only",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--allow-source-hash-mismatch",
        action="store_true",
    )

    args = parser.parse_args()

    if args.mu != 0.01:
        raise ValueError(
            "Phase17E freezes FedProx mu=0.01."
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

    experiments = make_experiment_set(
        config=config,
        matrix=args.matrix,
        only=args.only,
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

    print()
    print("=" * 124)
    print(PHASE_NAME)
    print("=" * 124)
    print(
        f"Scenario             : {args.scenario}"
    )
    print(
        f"Seed                 : {args.seed}"
    )
    print(
        f"Device               : {device}"
    )
    print(
        f"Features             : 70"
    )
    print(
        f"Clients              : {phase10.N_CLIENTS}"
    )
    print(
        f"Rounds               : {args.rounds}"
    )
    print(
        f"Local epochs         : {args.local_epochs}"
    )
    print(
        f"FedProx mu           : {args.mu}"
    )
    print(
        f"Selected attacks     : SCALE x5 and RANDOM x5"
    )
    print(
        f"Identity rotation    : client_1 ... client_5"
    )
    print(
        f"Ratio policy         : 0% / 20% / 40%"
    )
    print(
        f"40% malicious pair   : client_1 + client_2"
    )
    print(
        f"Robust defense       : NONE"
    )
    print(
        f"Locked test used     : NO"
    )
    print(
        f"Matrix               : {args.matrix}"
    )
    print(
        f"Experiments          : {[item['id'] for item in experiments]}"
    )

    print()
    print("SCIENTIFIC NOTE:")
    print(
        "Identity analysis rotates exactly one malicious client while keeping "
        "attack type and x5 magnitude fixed."
    )
    print(
        "Ratio analysis keeps malicious identities predeclared: 20%=client_1, "
        "40%=client_1+client_2. The same x5 strength is used at both ratios."
    )

    runs: Dict[
        str,
        Dict[
            str,
            Any,
        ]
    ] = {}

    for experiment in experiments:
        run = run_experiment(
            experiment=experiment,
            args=args,
            scaler=scaler,
            feature_columns=feature_columns,
            label_mapping=label_mapping,
            active_indices=all_active_indices,
            active_features=all_features,
            global_weights=global_weights,
            client_files=client_files,
            client_sample_counts=client_sample_counts,
            device=device,
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

        target = run[
            "target_summary"
        ]

        print()
        print(
            f"{run['experiment_id']} VALIDATION"
        )
        print(
            f"Accuracy                  : {overall['accuracy']:.6f}"
        )
        print(
            f"Balanced Accuracy         : {overall['balanced_accuracy']:.6f}"
        )
        print(
            f"Macro F1                  : {overall['macro_f1']:.6f}"
        )
        print(
            f"Weighted F1               : {overall['weighted_f1']:.6f}"
        )
        print(
            f"Worst attack Recall       : {overall['worst_attack_recall']:.6f}"
        )
        print(
            f"Target attack mean Recall : "
            f"{target['target_attack_mean_recall'] if target['target_attack_mean_recall'] is not None else 'N/A'}"
        )
        print(
            f"Target attack recalls     : {target['target_attack_recalls']}"
        )
        print(
            f"Attack audit              : "
            f"{'PASS' if run['attack_audit']['passed'] else 'FAIL'}"
        )

    # Build the clean-relative table only when a clean run is part of this invocation.
    clean = runs.get(
        "E0_CLEAN"
    )

    summary_rows = []

    if clean is not None:
        clean_overall = (
            clean[
                "validation"
            ][
                "overall"
            ]
        )

        for run in runs.values():
            overall = (
                run[
                    "validation"
                ][
                    "overall"
                ]
            )

            degradation = compare_to_clean(
                clean=clean_overall,
                candidate=overall,
            )

            target = run[
                "target_summary"
            ]

            summary_rows.append(
                {
                    "experiment_id":
                        run[
                            "experiment_id"
                        ],
                    "attack_type":
                        run[
                            "attack_type"
                        ],
                    "magnitude":
                        float(
                            run[
                                "magnitude"
                            ]
                        ),
                    "malicious_clients":
                        run[
                            "malicious_clients"
                        ],
                    "malicious_ratio":
                        float(
                            run[
                                "malicious_ratio"
                            ]
                        ),
                    "specialized_attacks":
                        run[
                            "specialized_attacks"
                        ],
                    "accuracy":
                        float(
                            overall[
                                "accuracy"
                            ]
                        ),
                    "balanced_accuracy":
                        float(
                            overall[
                                "balanced_accuracy"
                            ]
                        ),
                    "macro_f1":
                        float(
                            overall[
                                "macro_f1"
                            ]
                        ),
                    "weighted_f1":
                        float(
                            overall[
                                "weighted_f1"
                            ]
                        ),
                    "worst_attack_recall":
                        float(
                            overall[
                                "worst_attack_recall"
                            ]
                        ),
                    "target_attack_mean_recall":
                        target[
                            "target_attack_mean_recall"
                        ],
                    "target_attack_worst_recall":
                        target[
                            "target_attack_worst_recall"
                        ],
                    "target_attack_recalls":
                        target[
                            "target_attack_recalls"
                        ],
                    "macro_f1_drop_vs_clean":
                        float(
                            degradation[
                                "macro_f1_drop"
                            ]
                        ),
                    "macro_f1_relative_drop_pct":
                        float(
                            degradation[
                                "macro_f1_relative_drop_pct"
                            ]
                        ),
                    "balanced_accuracy_drop_vs_clean":
                        float(
                            degradation[
                                "balanced_accuracy_drop"
                            ]
                        ),
                    "worst_attack_recall_drop_vs_clean":
                        float(
                            degradation[
                                "worst_attack_recall_drop"
                            ]
                        ),
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

        analysis = build_summary(
            rows=summary_rows
        )

    else:
        analysis = {
            "identity_rows": [],
            "ratio_rows": [],
            "worst_identity_by_attack": {},
            "note": (
                "Clean-relative analysis not generated because E0_CLEAN was not part "
                "of this --only invocation."
            ),
        }

    seed_result_dir = (
        RESULT_ROOT
        /
        f"seed_{args.seed}"
    )

    seed_result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        seed_result_dir
        /
        "phase17e_analysis.json",
        analysis,
    )

    if summary_rows:
        dataframe_rows = []

        for row in summary_rows:
            flat = dict(
                row
            )

            flat[
                "malicious_clients"
            ] = ",".join(
                str(
                    value
                )
                for value
                in row[
                    "malicious_clients"
                ]
            )

            flat[
                "specialized_attacks"
            ] = " | ".join(
                row[
                    "specialized_attacks"
                ]
            )

            flat[
                "target_attack_recalls"
            ] = json.dumps(
                row[
                    "target_attack_recalls"
                ],
                sort_keys=True,
            )

            dataframe_rows.append(
                flat
            )

        pd.DataFrame(
            dataframe_rows
        ).to_csv(
            seed_result_dir
            /
            "phase17e_all_experiments.csv",
            index=False,
        )

        pd.DataFrame(
            analysis[
                "identity_rows"
            ]
        ).to_csv(
            seed_result_dir
            /
            "phase17e_identity_rotation.csv",
            index=False,
        )

        ratio_rows = []

        for row in analysis[
            "ratio_rows"
        ]:
            flat = dict(
                row
            )

            flat[
                "malicious_clients"
            ] = ",".join(
                str(
                    value
                )
                for value
                in row[
                    "malicious_clients"
                ]
            )

            ratio_rows.append(
                flat
            )

        pd.DataFrame(
            ratio_rows
        ).to_csv(
            seed_result_dir
            /
            "phase17e_ratio_curve.csv",
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

    audit = {
        "phase":
            "17E",
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
        "prerequisite_audit":
            prerequisite_audit,
        "attack_audits":
            {
                experiment_id:
                    run[
                        "attack_audit"
                    ]
                for experiment_id, run
                in runs.items()
            },
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
                "PASS means requested experiments completed and attack transformations "
                "passed integrity audits. It does not mean the system is robust."
            ),
    }

    save_json(
        seed_result_dir
        /
        "phase17e_audit.json",
        audit,
    )

    ARTIFACT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        "phase":
            "17E",
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
        "client_specialization":
            config[
                "client_specialization"
            ],
        "selected_attacks":
            config[
                "selected_attacks"
            ],
        "scientific_rules":
            config[
                "scientific_rules"
            ],
        "experiments_completed":
            [
                {
                    "id":
                        run[
                            "experiment_id"
                        ],
                    "attack_type":
                        run[
                            "attack_type"
                        ],
                    "magnitude":
                        run[
                            "magnitude"
                        ],
                    "malicious_clients":
                        run[
                            "malicious_clients"
                        ],
                    "malicious_ratio":
                        run[
                            "malicious_ratio"
                        ],
                    "attack_audit_pass":
                        run[
                            "attack_audit"
                        ][
                            "passed"
                        ],
                    "final_state_sha256":
                        run[
                            "final_state_sha256"
                        ],
                }
                for run
                in runs.values()
            ],
        "locked_test_used":
            False,
        "recommended_next_step":
            (
                "Use Phase17E evidence to freeze representative malicious-client identity/ratio "
                "scenarios, then run Phase17F defense comparison with identical attack definitions."
            ),
    }

    save_json(
        ARTIFACT_ROOT
        /
        "phase17e_manifest.json",
        manifest,
    )

    print()
    print("=" * 124)
    print("PHASE 17E COMPLETE")
    print("=" * 124)

    if summary_rows:
        for row in summary_rows:
            print(
                f"{row['experiment_id']:<18} | "
                f"mal={row['malicious_ratio']*100:>4.0f}% | "
                f"MF1={row['macro_f1']:.6f} | "
                f"BA={row['balanced_accuracy']:.6f} | "
                f"worst_attack={row['worst_attack_recall']:.6f} | "
                f"audit={'PASS' if row['attack_audit_pass'] else 'FAIL'}"
            )

    print()
    print(
        f"Attack audits passed       : "
        f"{'YES' if all_audits_passed else 'NO'}"
    )
    print(
        f"Robust defense used        : NO"
    )
    print(
        f"Locked test used           : NO"
    )
    print(
        f"Results                    : {seed_result_dir}"
    )
    print(
        f"Manifest                   : "
        f"{ARTIFACT_ROOT / 'phase17e_manifest.json'}"
    )
    print()
    print(
        f"STATUS                     : "
        f"{'PASS' if all_audits_passed else 'FAIL'}"
    )

    print()
    print("NEXT:")
    print(
        "Review identity-specific target-class failures and the 0%/20%/40% degradation curves. "
        "Then freeze Phase17F matched defense scenarios: baseline mean vs coordinate Median vs Trimmed Mean."
    )

    if not all_audits_passed:
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()
