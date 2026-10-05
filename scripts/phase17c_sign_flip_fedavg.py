from __future__ import annotations

r"""
Phase 17C — Controlled Sign-Flip / Model-Update Poisoning
=========================================================

This runner reuses the validated Phase 17A / Phase 10 federated-training
pipeline, but applies a real damaging sign-flip attack AFTER local training
and BEFORE FedAvg aggregation.

Attack for malicious client i:

    delta_i = W_i(local) - W_global
    delta_i_poisoned = -scale * delta_i
    W_i(poisoned) = W_global + delta_i_poisoned
                  = W_global - scale * (W_i(local) - W_global)

Important project policy:
- Original client CSV files are NOT modified.
- Validation data remains CLEAN.
- Locked test data is NOT used.
- Client partitions, 70-feature set, model architecture and FedProx training
  path remain frozen from the validated Phase 17A baseline.

Run from project root, for example:

    python scripts\phase17c_sign_flip_fedavg.py --scenario non_iid --seed 42 \
        --malicious-client 1 --scale 1.0

The malicious-client argument is 1-based to match the existing project:
client_1 => --malicious-client 1
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

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
        "\nPhase17C requires the validated Phase17A script:\n"
        "    scripts\\phase17a_poisoning_harness.py\n"
        "Place this file inside the project's scripts folder and run it "
        "from the project root."
    ) from exc

phase10 = phase17a.phase10
phase4 = phase17a.phase4

from src.federated.poisoning.metrics import compare_to_clean  # noqa: E402


# ---------------------------------------------------------------------------
# Phase constants
# ---------------------------------------------------------------------------

PHASE_NAME = "PHASE 17C — CONTROLLED SIGN-FLIP MODEL-UPDATE POISONING"

RESULT_ROOT = PROJECT_ROOT / "results" / "phase17" / "phase17c"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "poisoning" / "phase17c"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def save_json(path: Path, payload: Any) -> None:
    """Save a JSON file, creating parent directories when required."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def clone_state(
    state: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Clone a model state to CPU without sharing tensor storage."""
    return {
        key: value.detach().cpu().clone()
        for key, value in state.items()
    }


def state_max_abs_diff(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> float:
    """Maximum absolute difference between two compatible model states."""
    if set(left.keys()) != set(right.keys()):
        raise ValueError("State dictionaries have different keys.")

    maximum = 0.0

    for key in left:
        a = left[key].detach().cpu()
        b = right[key].detach().cpu()

        if a.shape != b.shape:
            raise ValueError(
                f"Shape mismatch for {key}: {tuple(a.shape)} vs {tuple(b.shape)}"
            )

        if not torch.is_floating_point(a):
            if not torch.equal(a, b):
                return float("inf")
            continue

        diff = torch.max(
            torch.abs(a.to(torch.float64) - b.to(torch.float64))
        ).item()
        maximum = max(maximum, float(diff))

    return maximum


def model_update_l2_norm(
    *,
    global_state: Mapping[str, torch.Tensor],
    local_state: Mapping[str, torch.Tensor],
) -> float:
    """Compute ||W_local - W_global||_2 over floating-point tensors."""
    squared_sum = 0.0

    for key in global_state:
        g = global_state[key].detach().cpu()
        l = local_state[key].detach().cpu()

        if g.shape != l.shape:
            raise ValueError(f"State shape mismatch for key={key}")

        if not torch.is_floating_point(g):
            continue

        delta = l.to(torch.float64) - g.to(torch.float64)
        squared_sum += float(torch.sum(delta * delta).item())

    return float(np.sqrt(squared_sum))


# ---------------------------------------------------------------------------
# Sign-flip attack
# ---------------------------------------------------------------------------


def sign_flip_model_update(
    *,
    global_state: Mapping[str, torch.Tensor],
    local_state: Mapping[str, torch.Tensor],
    scale: float,
) -> Dict[str, torch.Tensor]:
    """
    Apply controlled sign-flip poisoning in UPDATE SPACE.

        honest_delta   = local - global
        poisoned_delta = -scale * honest_delta
        poisoned_state = global + poisoned_delta

    scale=1.0 reverses the update with equal magnitude.
    scale>1.0 reverses and amplifies the update.

    Non-floating buffers are copied from the local state unchanged.
    """
    if scale <= 0.0:
        raise ValueError(f"Sign-flip scale must be > 0, got {scale}")

    attacked: Dict[str, torch.Tensor] = {}

    for key in local_state:
        local_tensor = local_state[key].detach().cpu()
        global_tensor = global_state[key].detach().cpu()

        if local_tensor.shape != global_tensor.shape:
            raise ValueError(f"State shape mismatch for key={key}")

        if torch.is_floating_point(local_tensor):
            attacked[key] = (
                global_tensor
                - float(scale) * (local_tensor - global_tensor)
            ).clone()
        else:
            # Preserve integer/non-floating buffers.
            attacked[key] = local_tensor.clone()

    return attacked


def route_client_state(
    *,
    global_state: Mapping[str, torch.Tensor],
    local_state: Mapping[str, torch.Tensor],
    client_id: int,
    malicious_client: int | None,
    scale: float,
    round_number: int,
) -> tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Route an honest or malicious client through the Phase 17C boundary."""
    honest_norm = model_update_l2_norm(
        global_state=global_state,
        local_state=local_state,
    )

    is_malicious = (
        malicious_client is not None
        and int(client_id) == int(malicious_client)
    )

    if not is_malicious:
        routed = clone_state(local_state)
        event = {
            "round": int(round_number),
            "client_id": int(client_id),
            "declared_malicious": False,
            "attack_type": "HONEST",
            "sign_flip_scale": None,
            "honest_update_l2_norm": float(honest_norm),
            "routed_update_l2_norm": float(honest_norm),
            "expected_routed_update_l2_norm": float(honest_norm),
            "norm_error": 0.0,
            "state_modified": False,
        }
        return routed, event

    routed = sign_flip_model_update(
        global_state=global_state,
        local_state=local_state,
        scale=scale,
    )

    routed_norm = model_update_l2_norm(
        global_state=global_state,
        local_state=routed,
    )

    expected_norm = float(scale) * float(honest_norm)
    norm_error = abs(float(routed_norm) - expected_norm)
    modification = state_max_abs_diff(local_state, routed)

    event = {
        "round": int(round_number),
        "client_id": int(client_id),
        "declared_malicious": True,
        "attack_type": "SIGN_FLIP",
        "sign_flip_scale": float(scale),
        "honest_update_l2_norm": float(honest_norm),
        "routed_update_l2_norm": float(routed_norm),
        "expected_routed_update_l2_norm": float(expected_norm),
        "norm_error": float(norm_error),
        "local_vs_poisoned_state_max_abs_diff": float(modification),
        "state_modified": bool(modification > 0.0),
    }

    return routed, event


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------


def scale_tag(scale: float) -> str:
    """Convert a scale into a filesystem/experiment-friendly tag."""
    text = f"{float(scale):g}".replace("-", "M").replace(".", "P")
    return text


def make_experiments(
    *,
    scales: list[float],
    malicious_client: int,
) -> list[dict[str, Any]]:
    """Create one clean baseline plus one sign-flip experiment per scale."""
    experiments: list[dict[str, Any]] = [
        {
            "id": "C0_CLEAN",
            "attack_type": "NONE",
            "malicious_client": None,
            "scale": 1.0,
        }
    ]

    for scale in scales:
        experiments.append(
            {
                "id": f"C_SIGN_FLIP_X{scale_tag(scale)}",
                "attack_type": "SIGN_FLIP",
                "malicious_client": int(malicious_client),
                "scale": float(scale),
            }
        )

    return experiments


# ---------------------------------------------------------------------------
# Federated training runner
# ---------------------------------------------------------------------------


def run_sign_flip_experiment(
    *,
    experiment: Mapping[str, Any],
    args: argparse.Namespace,
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
    """Run one clean or sign-flip FL experiment using the frozen pipeline."""
    experiment_id = str(experiment["id"])
    attack_type = str(experiment["attack_type"])
    malicious_client = experiment["malicious_client"]
    scale = float(experiment["scale"])

    if attack_type not in {"NONE", "SIGN_FLIP"}:
        raise ValueError(f"Unsupported Phase17C attack_type={attack_type!r}")

    # Reset reproducibility state so every experiment starts from the same
    # deterministic baseline for this seed.
    phase10.set_seed(args.seed)

    print()
    print("=" * 120)
    print(experiment_id)
    print("=" * 120)
    print(f"Attack type          : {attack_type}")
    print(
        "Malicious client     : "
        + (
            f"client_{malicious_client}"
            if malicious_client is not None
            else "NONE"
        )
    )
    print(f"Sign-flip scale      : {scale if attack_type == 'SIGN_FLIP' else 'N/A'}")
    print("Validation data      : CLEAN")
    print("Locked test used     : NO")

    global_model = phase10.initialize_full_model(
        input_dim=len(all_features),
        num_classes=len(label_mapping),
        device=device,
        seed=args.seed,
    )

    history: list[dict[str, Any]] = []
    attack_events: list[dict[str, Any]] = []

    training_start = time.perf_counter()

    for round_number in range(1, args.rounds + 1):
        old_global_state = clone_state(global_model.state_dict())

        client_states: list[Dict[str, torch.Tensor]] = []
        aggregation_weights: list[int] = []
        local_losses: list[float] = []
        local_ce_losses: list[float] = []
        local_prox_losses: list[float] = []
        local_accuracies: list[float] = []
        local_times: list[float] = []

        for client_id in range(1, phase10.N_CLIENTS + 1):
            (
                local_state,
                local_loss,
                local_ce_loss,
                local_prox_loss,
                local_accuracy,
                local_seconds,
            ) = phase10.train_client_fedprox(
                global_state=global_model.state_dict(),
                client_file=client_files[client_id],
                scaler=scaler,
                feature_columns=feature_columns,
                active_indices=all_active_indices,
                label_mapping=label_mapping,
                class_weights=global_weights,
                input_dim=len(all_features),
                device=device,
                local_epochs=args.local_epochs,
                batch_size=args.batch_size,
                chunk_size=args.chunk_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                client_id=client_id,
                round_number=round_number,
                mu=args.mu,
                seed=args.seed,
            )

            # ---------------------------------------------------------------
            # PHASE 17C ATTACK BOUNDARY
            # Local training has finished. Poison the selected client's update
            # BEFORE server-side FedAvg aggregation.
            # ---------------------------------------------------------------
            if attack_type == "SIGN_FLIP":
                routed_state, attack_event = route_client_state(
                    global_state=global_model.state_dict(),
                    local_state=local_state,
                    client_id=client_id,
                    malicious_client=int(malicious_client),
                    scale=scale,
                    round_number=round_number,
                )
            else:
                routed_state, attack_event = route_client_state(
                    global_state=global_model.state_dict(),
                    local_state=local_state,
                    client_id=client_id,
                    malicious_client=None,
                    scale=1.0,
                    round_number=round_number,
                )

            attack_events.append(attack_event)
            client_states.append(routed_state)

            aggregation_weights.append(
                int(client_sample_counts[f"client_{client_id}"])
            )

            local_losses.append(float(local_loss))
            local_ce_losses.append(float(local_ce_loss))
            local_prox_losses.append(float(local_prox_loss))
            local_accuracies.append(float(local_accuracy))
            local_times.append(float(local_seconds))

        # Server aggregation remains the existing validated FedAvg function.
        aggregated_state = phase4.federated_average(
            client_states,
            aggregation_weights,
        )

        global_model.load_state_dict(aggregated_state)

        update_norm = phase4.state_update_norm(
            old_global_state,
            aggregated_state,
        )

        weights = np.asarray(aggregation_weights, dtype=np.float64)

        round_malicious_events = [
            event
            for event in attack_events
            if event["round"] == round_number
            and event["declared_malicious"]
        ]

        malicious_honest_norm = None
        malicious_routed_norm = None
        if round_malicious_events:
            malicious_honest_norm = float(
                round_malicious_events[0]["honest_update_l2_norm"]
            )
            malicious_routed_norm = float(
                round_malicious_events[0]["routed_update_l2_norm"]
            )

        history.append(
            {
                "round": int(round_number),
                "weighted_local_loss": float(
                    np.average(local_losses, weights=weights)
                ),
                "weighted_ce_loss": float(
                    np.average(local_ce_losses, weights=weights)
                ),
                "weighted_prox_loss": float(
                    np.average(local_prox_losses, weights=weights)
                ),
                "weighted_local_accuracy": float(
                    np.average(local_accuracies, weights=weights)
                ),
                "global_update_norm": float(update_norm),
                "mean_client_training_seconds": float(np.mean(local_times)),
                "malicious_honest_update_norm": malicious_honest_norm,
                "malicious_routed_update_norm": malicious_routed_norm,
            }
        )

        if round_malicious_events:
            attack_text = (
                f" | malicious_norm={malicious_honest_norm:.6f}"
                f"->{malicious_routed_norm:.6f}"
            )
        else:
            attack_text = ""

        print(
            f"Round {round_number:02d}/{args.rounds} | "
            f"loss={history[-1]['weighted_local_loss']:.6f} | "
            f"acc={history[-1]['weighted_local_accuracy']:.6f} | "
            f"global_update_norm={history[-1]['global_update_norm']:.6f}"
            f"{attack_text}"
        )

    training_seconds = float(time.perf_counter() - training_start)

    # Same clean validation path used by Phase 17A.
    validation_metrics = phase17a.evaluate_on_validation(
        model=global_model,
        scaler=scaler,
        feature_columns=feature_columns,
        active_indices=all_active_indices,
        label_mapping=label_mapping,
        device=device,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )

    validation_metrics["overall"]["training_time_seconds"] = training_seconds

    final_state = clone_state(global_model.state_dict())
    final_state_sha256 = phase17a.sha256_state(final_state)

    return {
        "experiment_id": experiment_id,
        "attack_spec": {
            "attack_type": attack_type,
            "malicious_client": (
                int(malicious_client)
                if malicious_client is not None
                else None
            ),
            "sign_flip_scale": (
                float(scale)
                if attack_type == "SIGN_FLIP"
                else None
            ),
            "formula": (
                "poisoned_state = global_state - scale * "
                "(local_state - global_state)"
                if attack_type == "SIGN_FLIP"
                else "identity"
            ),
        },
        "history": history,
        "attack_events": attack_events,
        "validation": validation_metrics,
        "final_state": final_state,
        "final_state_sha256": final_state_sha256,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_phase17c_experiment(
    *,
    run: Mapping[str, Any],
    seed: int,
) -> None:
    """Save all Phase 17C evidence without overwriting other seeds."""
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
        result_dir / "attack_events.json",
        run["attack_events"],
    )

    save_json(
        result_dir / "attack_spec.json",
        run["attack_spec"],
    )

    save_json(
        result_dir / "validation_metrics.json",
        run["validation"],
    )

    per_class = run["validation"]["per_class"]
    pd.DataFrame(
        [
            {"class": class_name, **values}
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
            "phase": "17C",
            "experiment_id": experiment_id,
            "seed": int(seed),
            "attack_spec": run["attack_spec"],
            "model_state_dict": run["final_state"],
            "state_sha256": run["final_state_sha256"],
        },
        artifact_dir / "fedprox70_phase17c_model.pt",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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
        help=(
            "1-based malicious client id. Example: 1 means client_1. "
            "Keep one fixed malicious client for the Phase17C attack-strength study."
        ),
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help=(
            "Single sign-flip magnitude. Example: --scale 1.0. "
            "Use --scales for an attack-strength sweep."
        ),
    )

    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional sign-flip magnitude sweep. "
            "Example: --scales 1.0 2.0 5.0"
        ),
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run clean + first sign-flip experiment for 1 round unless "
            "--rounds was explicitly changed from its default."
        ),
    )

    parser.add_argument(
        "--allow-source-hash-mismatch",
        action="store_true",
        help=(
            "Debug-only. Final evidence should use the frozen Phase17A source hashes."
        ),
    )

    args = parser.parse_args()

    if args.malicious_client < 1 or args.malicious_client > phase10.N_CLIENTS:
        raise ValueError(
            f"malicious-client must be 1..{phase10.N_CLIENTS}, "
            f"got {args.malicious_client}"
        )

    if args.scale is not None and args.scales is not None:
        raise ValueError("Use either --scale or --scales, not both.")

    if args.scales is not None:
        resolved_scales = [float(x) for x in args.scales]
    elif args.scale is not None:
        resolved_scales = [float(args.scale)]
    else:
        resolved_scales = [1.0]

    for scale in resolved_scales:
        if not np.isfinite(float(scale)) or float(scale) <= 0.0:
            raise ValueError(
                f"Every sign-flip scale must be finite and > 0, got {scale}"
            )

    # Normalize to the list form used internally by the experiment loop.
    args.scales = resolved_scales

    if args.mu != 0.01:
        raise ValueError("Phase17C keeps the frozen FedProx mu=0.01 baseline.")

    # -----------------------------------------------------------------------
    # Verify Phase17A frozen baseline and data policy before damaging attack.
    # -----------------------------------------------------------------------
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
        raise RuntimeError(
            f"Expected frozen 70 active features, got {len(all_features)}"
        )

    global_counts = phase4.calculate_global_class_counts(
        label_mapping,
        args.chunk_size,
    )
    global_weights = phase4.calculate_global_class_weights(global_counts)

    client_files: Dict[int, Path] = {}
    client_sample_counts: Dict[str, int] = {}

    for client_id in range(1, phase10.N_CLIENTS + 1):
        client_file = scenario_dir / f"client_{client_id}.csv"
        if not client_file.exists():
            raise FileNotFoundError(client_file)

        client_files[client_id] = client_file
        client_sample_counts[f"client_{client_id}"] = phase4.count_client_samples(
            client_file,
            args.chunk_size,
        )

    experiments = make_experiments(
        scales=[float(scale) for scale in args.scales],
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
    print(f"Clients               : {phase10.N_CLIENTS}")
    print(f"Malicious client      : client_{args.malicious_client}")
    print(f"Sign-flip scales      : {[float(x) for x in args.scales]}")
    print("Local optimizer       : Frozen Phase17A/Phase10 FedProx path")
    print("Server aggregation    : Existing FedAvg")
    print("Feature set           : Frozen 70 features")
    print("Client data modified  : NO")
    print("Validation data       : CLEAN")
    print("Locked test used      : NO")

    runs: list[dict[str, Any]] = []

    for experiment in experiments:
        run = run_sign_flip_experiment(
            experiment=experiment,
            args=args,
            scaler=scaler,
            feature_columns=feature_columns,
            label_mapping=label_mapping,
            all_active_indices=all_active_indices,
            all_features=all_features,
            global_weights=global_weights,
            client_files=client_files,
            client_sample_counts=client_sample_counts,
            device=device,
        )

        save_phase17c_experiment(
            run=run,
            seed=args.seed,
        )

        runs.append(run)

    clean_run = next(
        run for run in runs
        if run["experiment_id"] == "C0_CLEAN"
    )
    clean_overall = clean_run["validation"]["overall"]

    comparison_rows: list[dict[str, Any]] = []

    print()
    print("=" * 100)
    print("PHASE 17C RESULTS")
    print("=" * 100)

    for run in runs:
        overall = run["validation"]["overall"]
        spec = run["attack_spec"]

        if run["experiment_id"] == "C0_CLEAN":
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

        malicious_events = [
            event
            for event in run["attack_events"]
            if event["declared_malicious"]
        ]

        mean_honest_norm = (
            float(np.mean([
                event["honest_update_l2_norm"]
                for event in malicious_events
            ]))
            if malicious_events
            else 0.0
        )

        mean_routed_norm = (
            float(np.mean([
                event["routed_update_l2_norm"]
                for event in malicious_events
            ]))
            if malicious_events
            else 0.0
        )

        max_norm_error = (
            float(max([
                event["norm_error"]
                for event in malicious_events
            ]))
            if malicious_events
            else 0.0
        )

        print()
        print(run["experiment_id"])
        print(f"  Attack type          : {spec['attack_type']}")
        print(f"  Malicious client     : {spec['malicious_client']}")
        print(f"  Sign-flip scale      : {spec['sign_flip_scale']}")
        print(f"  Accuracy             : {overall['accuracy']:.6f}")
        print(f"  Balanced Accuracy    : {overall['balanced_accuracy']:.6f}")
        print(f"  Macro Precision      : {overall['macro_precision']:.6f}")
        print(f"  Macro Recall         : {overall['macro_recall']:.6f}")
        print(f"  Macro F1             : {overall['macro_f1']:.6f}")
        print(f"  Weighted F1          : {overall['weighted_f1']:.6f}")
        print(f"  Worst attack Recall  : {overall['worst_attack_recall']:.6f}")
        print(f"  Macro F1 drop        : {drops['macro_f1_drop']:.6f}")
        if malicious_events:
            print(f"  Mean honest upd norm : {mean_honest_norm:.6f}")
            print(f"  Mean poison upd norm : {mean_routed_norm:.6f}")
            print(f"  Max norm check error : {max_norm_error:.12f}")

        comparison_rows.append(
            {
                "experiment_id": run["experiment_id"],
                "seed": int(args.seed),
                "scenario": args.scenario,
                "attack_type": spec["attack_type"],
                "malicious_client": spec["malicious_client"],
                "sign_flip_scale": spec["sign_flip_scale"],
                "accuracy": float(overall["accuracy"]),
                "balanced_accuracy": float(overall["balanced_accuracy"]),
                "macro_precision": float(overall["macro_precision"]),
                "macro_recall": float(overall["macro_recall"]),
                "macro_f1": float(overall["macro_f1"]),
                "weighted_f1": float(overall["weighted_f1"]),
                "worst_class_recall": float(overall["worst_class_recall"]),
                "worst_attack_recall": float(overall["worst_attack_recall"]),
                "benign_false_positive_rate": float(
                    overall["benign_false_positive_rate"]
                ),
                "training_time_seconds": float(
                    overall.get("training_time_seconds", 0.0)
                ),
                "mean_malicious_honest_update_norm": mean_honest_norm,
                "mean_malicious_routed_update_norm": mean_routed_norm,
                "max_sign_flip_norm_error": max_norm_error,
                **{key: float(value) for key, value in drops.items()},
                "validation_poisoned": False,
                "locked_test_used": False,
            }
        )

    summary_dir = RESULT_ROOT / f"seed_{args.seed}"
    summary_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = (
        summary_dir / "phase17c_clean_vs_sign_flip_comparison.csv"
    )
    pd.DataFrame(comparison_rows).to_csv(
        comparison_path,
        index=False,
    )

    manifest = {
        "phase": "17C",
        "status": "PASS",
        "scenario": args.scenario,
        "seed": int(args.seed),
        "rounds": int(args.rounds),
        "local_epochs": int(args.local_epochs),
        "malicious_client": int(args.malicious_client),
        "sign_flip_scales": [float(x) for x in args.scales],
        "attack": (
            "controlled sign flip in client model-update space: "
            "delta_poisoned = -scale * delta_honest"
        ),
        "attack_location": "after local training, before FedAvg aggregation",
        "original_client_data_modified": False,
        "feature_set_frozen": True,
        "client_partitions_frozen": True,
        "model_architecture_frozen": True,
        "local_training_path_frozen_from_phase17a": True,
        "server_aggregation": "existing phase4.federated_average",
        "validation_poisoned": False,
        "locked_test_used": False,
        "source_hash_audit": source_audit,
        "experiments": [
            {
                "experiment_id": run["experiment_id"],
                "attack_spec": run["attack_spec"],
                "final_state_sha256": run["final_state_sha256"],
            }
            for run in runs
        ],
        "next": (
            "After seed 42 is validated, repeat Phase17C with seeds "
            "52, 62, 72 and 82 using identical configuration."
        ),
    }

    manifest_path = ARTIFACT_ROOT / f"seed_{args.seed}" / "phase17c_manifest.json"
    save_json(manifest_path, manifest)

    print()
    print("=" * 70)
    print("PHASE 17C COMPLETE")
    print("=" * 70)
    print(f"Experiments completed   : {len(runs)}")
    print("Original data modified  : NO")
    print("Validation poisoned      : NO")
    print("Locked test used         : NO")
    print(f"Comparison               : {comparison_path}")
    print(f"Manifest                 : {manifest_path}")
    print("STATUS                   : PASS")
    print()
    print("NEXT:")
    print("1. Inspect seed 42 results and attack_events.json.")
    print("2. If correct, repeat seeds 52, 62, 72 and 82.")


if __name__ == "__main__":
    main()
