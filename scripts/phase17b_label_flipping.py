from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import phase17a_poisoning_harness as phase17a
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "\nPhase17B requires the validated Phase17A script:\n"
        "    scripts\\phase17a_poisoning_harness.py\n"
    ) from exc

phase10 = phase17a.phase10
phase4 = phase17a.phase4

from src.federated.poisoning.label_flip import (  # noqa: E402
    LabelFlipAudit,
    create_label_flipped_csv,
)
from src.federated.poisoning.metrics import compare_to_clean  # noqa: E402


PHASE_NAME = "PHASE 17B — CONTROLLED LABEL-FLIPPING POISONING"

RESULT_ROOT = PROJECT_ROOT / "results" / "phase17" / "phase17b"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "poisoning" / "phase17b"

logger = logging.getLogger("phase17b_label_flipping")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def normalize_label(value: Any) -> str:
    return str(value).strip()


def resolve_benign_label(label_mapping: Mapping[str, int]) -> str:
    preferred = normalize_label(getattr(phase4, "BENIGN_LABEL", "BENIGN"))

    for label in label_mapping:
        if normalize_label(label).lower() == preferred.lower():
            return str(label)

    for label in label_mapping:
        if normalize_label(label).lower() == "benign":
            return str(label)

    raise KeyError(
        f"Could not resolve BENIGN label from mapping={dict(label_mapping)}"
    )


def make_experiments(rates: list[float], malicious_client: int) -> list[dict[str, Any]]:
    experiments = [
        {
            "id": "B0_CLEAN",
            "poison_rate": 0.0,
            "malicious_client": None,
        }
    ]

    for rate in rates:
        pct = int(round(rate * 100))
        experiments.append(
            {
                "id": f"B_LABEL_FLIP_{pct:02d}",
                "poison_rate": float(rate),
                "malicious_client": int(malicious_client),
            }
        )

    return experiments


def save_phase17b_experiment(
    *,
    run: Mapping[str, Any],
    seed: int,
    poisoning_audit: Mapping[str, Any] | None,
    data_poisoning_spec: Mapping[str, Any],
) -> None:
    experiment_id = str(run["experiment_id"])

    result_dir = RESULT_ROOT / f"seed_{seed}" / experiment_id
    artifact_dir = ARTIFACT_ROOT / f"seed_{seed}" / experiment_id

    result_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(run["history"]).to_csv(
        result_dir / "round_history.csv",
        index=False,
    )

    save_json(
        result_dir / "validation_metrics.json",
        run["validation"],
    )

    save_json(
        result_dir / "data_poisoning_spec.json",
        dict(data_poisoning_spec),
    )

    if poisoning_audit is not None:
        save_json(
            result_dir / "label_flip_audit.json",
            dict(poisoning_audit),
        )

    per_class = run["validation"]["per_class"]
    pd.DataFrame(
        [
            {
                "class": class_name,
                **values,
            }
            for class_name, values in per_class.items()
        ]
    ).to_csv(
        result_dir / "per_class_validation.csv",
        index=False,
    )

    pd.DataFrame(
        run["validation"]["confusion_matrix"],
        index=run["validation"]["class_names"],
        columns=run["validation"]["class_names"],
    ).to_csv(
        result_dir / "validation_confusion_matrix.csv"
    )

    torch.save(
        {
            "phase": "17B",
            "experiment_id": experiment_id,
            "seed": int(seed),
            "data_poisoning_spec": dict(data_poisoning_spec),
            "model_state_dict": run["final_state"],
            "state_sha256": run["final_state_sha256"],
        },
        artifact_dir / "fedprox70_phase17b_model.pt",
    )


def print_poison_validation(audit: LabelFlipAudit, experiment_id: str) -> None:
    print()
    print("=" * 70)
    print("PHASE 17B POISON VALIDATION")
    print("=" * 70)
    print(f"Experiment              : {experiment_id}")
    print(f"Seed                    : {audit.seed}")
    print(f"Eligible attack rows    : {audit.eligible_attack_rows}")
    print(f"Requested poison rate   : {audit.poison_rate_requested * 100:.2f}%")
    print(f"Rows poisoned           : {audit.poisoned_rows}")
    print(f"Actual poison rate      : {audit.poison_rate_actual * 100:.2f}%")
    print(f"Source attack labels    : {audit.source_attack_labels}")
    print(f"Target label            : {audit.benign_label}")
    print(f"Feature modification    : {'YES' if audit.features_modified else 'NO'}")
    print(f"Benign-label changes    : {audit.benign_rows_modified}")
    print(f"Unexpected label changes: {audit.unexpected_label_changes}")
    print("Validation poisoned     : NO")
    print("Locked test used        : NO")
    print("STATUS                  : PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=PHASE_NAME)

    parser.add_argument(
        "--scenario",
        choices=["iid", "non_iid", "controlled_non_iid"],
        default="non_iid",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=100000)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--mu", type=float, default=0.01)
    parser.add_argument(
        "--malicious-client",
        type=int,
        default=1,
        help="Phase17B attack-strength study uses one fixed malicious client. Ratio study is Phase17E.",
    )
    parser.add_argument(
        "--rates",
        type=float,
        nargs="+",
        default=[0.10, 0.25, 0.50],
        help="Poison fractions over eligible attack rows. Default: 0.10 0.25 0.50",
    )
    parser.add_argument(
        "--source-attack-labels",
        nargs="*",
        default=None,
        help=(
            "Optional exact attack labels to flip. If omitted, every non-BENIGN label "
            "present in the malicious client's local training CSV is eligible."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run B0 clean + first poison rate for 1 round unless --rounds is explicitly changed.",
    )
    parser.add_argument(
        "--allow-source-hash-mismatch",
        action="store_true",
        help="Debug-only. Final evidence should use the frozen Phase17A source hashes.",
    )

    args = parser.parse_args()

    if args.malicious_client < 1 or args.malicious_client > phase10.N_CLIENTS:
        raise ValueError(
            f"malicious-client must be 1..{phase10.N_CLIENTS}, got {args.malicious_client}"
        )

    for rate in args.rates:
        if not (0.0 < float(rate) <= 1.0):
            raise ValueError(f"Every poisoning rate must be in (0, 1], got {rate}")

    if args.mu != 0.01:
        raise ValueError("Phase17B keeps the frozen FedProx mu=0.01 baseline.")

    # Verify Phase17A frozen baseline/harness files before damaging poisoning.
    phase17a_config = phase17a.load_json(phase17a.CONFIG_FILE)
    source_audit = phase17a.verify_frozen_sources(
        phase17a_config,
        allow_mismatch=args.allow_source_hash_mismatch,
    )
    phase17a.ensure_data_policy()

    phase10.set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    scenario_dir = phase4.FEDERATED_ROOT / args.scenario
    if not scenario_dir.exists():
        raise FileNotFoundError(scenario_dir)

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
        active_features=active_features,
        active_indices=active_indices,
        feature_columns=feature_columns,
    )

    if len(all_features) != 70:
        raise RuntimeError(f"Expected 70 active features, got {len(all_features)}")

    global_counts = phase4.calculate_global_class_counts(
        label_mapping,
        args.chunk_size,
    )
    global_weights = phase4.calculate_global_class_weights(global_counts)

    benign_label = resolve_benign_label(label_mapping)
    label_column = phase4.LABEL_COL

    original_client_files: Dict[int, Path] = {}
    client_sample_counts: Dict[str, int] = {}

    for client_id in range(1, phase10.N_CLIENTS + 1):
        path = scenario_dir / f"client_{client_id}.csv"
        if not path.exists():
            raise FileNotFoundError(path)

        original_client_files[client_id] = path
        client_sample_counts[f"client_{client_id}"] = phase4.count_client_samples(
            path,
            args.chunk_size,
        )

    experiments = make_experiments(
        rates=[float(rate) for rate in args.rates],
        malicious_client=args.malicious_client,
    )

    if args.smoke:
        experiments = experiments[:2]
        if args.rounds == 10:
            args.rounds = 1

    print()
    print("=" * 100)
    print(PHASE_NAME)
    print("=" * 100)
    print(f"Scenario              : {args.scenario}")
    print(f"Seed                  : {args.seed}")
    print(f"Device                : {device}")
    print(f"Rounds                : {args.rounds}")
    print(f"Local epochs          : {args.local_epochs}")
    print(f"Malicious client      : client_{args.malicious_client}")
    print(f"Poison rates          : {args.rates}")
    print(f"Target label          : {benign_label}")
    print("Aggregation           : Frozen Phase17A FedProx/FedAvg path")
    print("Validation data       : CLEAN")
    print("Locked test used      : NO")

    runs: list[dict[str, Any]] = []

    for experiment in experiments:
        experiment_id = str(experiment["id"])
        poison_rate = float(experiment["poison_rate"])
        malicious_client = experiment["malicious_client"]

        working_client_files = dict(original_client_files)
        poisoning_audit: LabelFlipAudit | None = None

        if poison_rate > 0.0:
            poisoned_file = (
                ARTIFACT_ROOT
                / f"seed_{args.seed}"
                / experiment_id
                / "poisoned_data"
                / f"client_{malicious_client}.csv"
            )

            poisoning_audit = create_label_flipped_csv(
                source_csv=original_client_files[int(malicious_client)],
                destination_csv=poisoned_file,
                label_column=label_column,
                benign_label=benign_label,
                poison_rate=poison_rate,
                seed=args.seed,
                source_attack_labels=args.source_attack_labels,
            )

            print_poison_validation(poisoning_audit, experiment_id)
            working_client_files[int(malicious_client)] = poisoned_file

        # Reuse the validated Phase17A FL runner unchanged.
        # UPDATE poisoning is NONE here because Phase17B is DATA poisoning only.
        phase17a_experiment = {
            "id": experiment_id,
            "attack_type": "NONE",
            "malicious_clients": [],
            "update_scale": 1.0,
        }

        run = phase17a.run_experiment(
            experiment=phase17a_experiment,
            args=args,
            scaler=scaler,
            feature_columns=feature_columns,
            label_mapping=label_mapping,
            all_active_indices=all_active_indices,
            all_features=all_features,
            global_weights=global_weights,
            client_files=working_client_files,
            client_sample_counts=client_sample_counts,
            device=device,
        )

        data_poisoning_spec = {
            "phase": "17B",
            "attack_type": "LABEL_FLIP",
            "malicious_client": (
                int(malicious_client)
                if malicious_client is not None
                else None
            ),
            "poison_rate": float(poison_rate),
            "rate_denominator": "eligible_attack_rows",
            "target_label": benign_label,
            "source_attack_labels": (
                poisoning_audit.source_attack_labels
                if poisoning_audit is not None
                else []
            ),
            "feature_modification": False,
            "validation_poisoned": False,
            "locked_test_used": False,
            "seed": int(args.seed),
        }

        run["data_poisoning_spec"] = data_poisoning_spec
        run["label_flip_audit"] = (
            poisoning_audit.to_dict()
            if poisoning_audit is not None
            else None
        )

        save_phase17b_experiment(
            run=run,
            seed=args.seed,
            poisoning_audit=run["label_flip_audit"],
            data_poisoning_spec=data_poisoning_spec,
        )

        runs.append(run)

    clean_run = next(run for run in runs if run["experiment_id"] == "B0_CLEAN")
    clean_overall = clean_run["validation"]["overall"]

    comparison_rows = []

    print()
    print("=" * 100)
    print("PHASE 17B RESULTS")
    print("=" * 100)

    for run in runs:
        overall = run["validation"]["overall"]
        spec = run["data_poisoning_spec"]

        if run["experiment_id"] == "B0_CLEAN":
            drops = {
                "accuracy_drop": 0.0,
                "balanced_accuracy_drop": 0.0,
                "macro_f1_drop": 0.0,
                "macro_f1_relative_drop_pct": 0.0,
                "weighted_f1_drop": 0.0,
                "worst_class_recall_drop": 0.0,
                "worst_attack_recall_drop": 0.0,
                "benign_fpr_increase": 0.0,
            }
        else:
            drops = compare_to_clean(
                clean=clean_overall,
                candidate=overall,
            )

        target_attack_recalls: Dict[str, float] = {}
        for attack_label in spec["source_attack_labels"]:
            if attack_label in run["validation"]["per_class"]:
                target_attack_recalls[attack_label] = float(
                    run["validation"]["per_class"][attack_label]["recall"]
                )

        print()
        print(f"{run['experiment_id']}")
        print(f"  Poison rate          : {spec['poison_rate'] * 100:.2f}%")
        print(f"  Accuracy             : {overall['accuracy']:.6f}")
        print(f"  Balanced Accuracy    : {overall['balanced_accuracy']:.6f}")
        print(f"  Macro F1             : {overall['macro_f1']:.6f}")
        print(f"  Worst attack Recall  : {overall['worst_attack_recall']:.6f}")
        print(f"  Macro F1 drop        : {drops['macro_f1_drop']:.6f}")
        if target_attack_recalls:
            print(f"  Target attack Recall : {target_attack_recalls}")

        audit = run.get("label_flip_audit") or {}

        row = {
            "experiment_id": run["experiment_id"],
            "seed": int(args.seed),
            "malicious_client": spec["malicious_client"],
            "poison_rate": float(spec["poison_rate"]),
            "eligible_attack_rows": int(audit.get("eligible_attack_rows", 0)),
            "poisoned_rows": int(audit.get("poisoned_rows", 0)),
            "accuracy": float(overall["accuracy"]),
            "balanced_accuracy": float(overall["balanced_accuracy"]),
            "macro_precision": float(overall["macro_precision"]),
            "macro_recall": float(overall["macro_recall"]),
            "macro_f1": float(overall["macro_f1"]),
            "weighted_f1": float(overall["weighted_f1"]),
            "worst_class_recall": float(overall["worst_class_recall"]),
            "worst_attack_recall": float(overall["worst_attack_recall"]),
            **{key: float(value) for key, value in drops.items()},
            "target_attack_recalls_json": json.dumps(target_attack_recalls, sort_keys=True),
            "locked_test_used": False,
        }
        comparison_rows.append(row)

    summary_dir = RESULT_ROOT / f"seed_{args.seed}"
    summary_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(comparison_rows).to_csv(
        summary_dir / "phase17b_clean_vs_label_flip_comparison.csv",
        index=False,
    )

    manifest = {
        "phase": "17B",
        "status": "PASS",
        "scenario": args.scenario,
        "seed": int(args.seed),
        "rounds": int(args.rounds),
        "local_epochs": int(args.local_epochs),
        "malicious_client": int(args.malicious_client),
        "poison_rates": [float(x) for x in args.rates],
        "attack": "controlled label flipping: eligible attack label -> BENIGN",
        "poison_rate_denominator": "eligible attack rows",
        "feature_set_frozen": True,
        "client_partitions_frozen": True,
        "model_architecture_frozen": True,
        "aggregation_path_frozen_from_phase17a": True,
        "validation_poisoned": False,
        "locked_test_used": False,
        "source_hash_audit": source_audit,
        "experiments": [
            {
                "experiment_id": run["experiment_id"],
                "data_poisoning_spec": run["data_poisoning_spec"],
                "final_state_sha256": run["final_state_sha256"],
            }
            for run in runs
        ],
        "next": "Phase 17C — controlled sign-flip/model-update poisoning.",
    }

    manifest_path = ARTIFACT_ROOT / "phase17b_manifest.json"
    save_json(manifest_path, manifest)

    print()
    print("=" * 70)
    print("PHASE 17B COMPLETE")
    print("=" * 70)
    print(f"Experiments completed   : {len(runs)}")
    print(f"Original data modified  : NO")
    print(f"Validation poisoned      : NO")
    print(f"Locked test used         : NO")
    print(f"Comparison               : {summary_dir / 'phase17b_clean_vs_label_flip_comparison.csv'}")
    print(f"Manifest                 : {manifest_path}")
    print("STATUS                   : PASS")
    print()
    print("NEXT:")
    print("Phase 17C — controlled sign-flip/model-update poisoning.")


if __name__ == "__main__":
    main()
