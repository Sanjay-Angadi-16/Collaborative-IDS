from __future__ import annotations

r"""
Phase 17F — Matched Defense Comparison
======================================

Compare three server aggregation rules under the SAME Phase 17E Byzantine
attack conditions:

    1. FEDAVG            : sample-weighted parameter averaging
    2. COORDINATE_MEDIAN : coordinate-wise median
    3. TRIMMED_MEAN      : coordinate-wise trimmed mean

Client-side training is unchanged:
    - FedProx local training
    - mu = 0.01
    - frozen 70-feature IDS
    - same non-IID client partitions
    - same attack helper used by Phase 17E

Default matched experiment matrix (from Phase 17E ratio policy):
    E0_CLEAN              : 0% malicious
    E_SCALE_C1            : SCALE x5, 20% malicious
    E_SCALE_C1_C2         : SCALE x5, 40% malicious
    E_RANDOM_C1           : RANDOM x5, 20% malicious
    E_RANDOM_C1_C2        : RANDOM x5, 40% malicious

Important scientific control:
    Every (experiment, aggregator) run resets the SAME seed before model
    initialization and local training. Therefore the intended changed factor
    is the server aggregation rule.

Locked test data is NOT used.

Recommended seed-42 run:

    python scripts\phase17f_fedavg_vs_median_vs_trimmed_mean.py \
        --scenario non_iid \
        --seed 42 \
        --matrix matched \
        --trim-count 1

Smoke test:

    python scripts\phase17f_fedavg_vs_median_vs_trimmed_mean.py \
        --scenario non_iid \
        --seed 42 \
        --matrix smoke \
        --rounds 1
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


# =============================================================================
# PROJECT IMPORTS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import phase17e_identity_ratio as phase17e
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "\nPhase 17F requires your completed Phase 17E runner:\n"
        "    scripts\\phase17e_identity_ratio.py\n"
        "and the Phase 17E helper/config files.\n"
    ) from exc

phase10 = phase17e.phase10
phase4 = phase17e.phase4

ByzantineAttackSpec = phase17e.ByzantineAttackSpec
apply_byzantine_attack = phase17e.apply_byzantine_attack


# =============================================================================
# CONSTANTS
# =============================================================================

PHASE_NAME = "PHASE 17F — FEDAVG vs COORDINATE MEDIAN vs TRIMMED MEAN"

RESULT_ROOT = PROJECT_ROOT / "results" / "phase17" / "phase17f"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "poisoning" / "phase17f"

AGGREGATORS = (
    "fedavg",
    "coordinate_median",
    "trimmed_mean",
)

DISPLAY_NAMES = {
    "fedavg": "FedAvg",
    "coordinate_median": "Coordinate Median",
    "trimmed_mean": "Trimmed Mean",
}


# =============================================================================
# HELPERS
# =============================================================================

def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def clone_state(
    state: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in state.items()
    }


def max_state_abs_diff(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> float:
    if set(left.keys()) != set(right.keys()):
        raise ValueError("State dictionaries have different keys.")

    maximum = 0.0

    for key in left:
        a = left[key].detach().cpu()
        b = right[key].detach().cpu()

        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch for {key}")

        if torch.is_floating_point(a):
            error = torch.max(
                torch.abs(
                    a.to(torch.float64)
                    - b.to(torch.float64)
                )
            ).item()
            maximum = max(maximum, float(error))
        elif not torch.equal(a, b):
            return float("inf")

    return float(maximum)


def scale_tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "M").replace(".", "P")


def event_float(
    event: Mapping[str, Any],
    *keys: str,
) -> float:
    """
    Read a numeric value from an attack-event dictionary using the first
    available key.

    Phase 17E Byzantine SCALE/RANDOM events use:
        - legitimate_update_l2
        - attacked_update_l2

    Earlier Phase 17C/17D sign-flip helpers used names such as:
        - honest_update_l2_norm
        - routed_update_l2_norm

    Keeping both schemas here makes Phase 17F robust without changing the
    attack implementation itself.
    """
    for key in keys:
        if key in event and event[key] is not None:
            return float(event[key])

    raise KeyError(
        "Attack event is missing all expected keys: "
        + ", ".join(keys)
        + f". Available keys: {sorted(event.keys())}"
    )


# =============================================================================
# AGGREGATION RULES
# =============================================================================

def validate_trim_count(num_clients: int, trim_count: int) -> None:
    if trim_count < 0:
        raise ValueError("trim_count must be >= 0")

    if num_clients - 2 * trim_count <= 0:
        raise ValueError(
            "Invalid Trimmed Mean configuration: "
            f"num_clients={num_clients}, trim_count={trim_count}"
        )


def coordinate_median(
    *,
    client_states: Sequence[Mapping[str, torch.Tensor]],
    global_state: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Coordinate-wise median across client model states.

    With 5 clients, every floating-point model coordinate is replaced by the
    third value after sorting the five submitted values.

    Non-floating buffers are preserved from the current global model.
    """
    if not client_states:
        raise ValueError("client_states cannot be empty")

    output: Dict[str, torch.Tensor] = {}
    expected_keys = set(global_state.keys())

    for index, state in enumerate(client_states, start=1):
        if set(state.keys()) != expected_keys:
            raise ValueError(
                f"Client state {index} has different state_dict keys."
            )

    for key, global_tensor_original in global_state.items():
        global_tensor = global_tensor_original.detach().cpu()

        tensors = [
            state[key].detach().cpu()
            for state in client_states
        ]

        if any(tensor.shape != global_tensor.shape for tensor in tensors):
            raise ValueError(f"Shape mismatch for state key {key!r}")

        if not torch.is_floating_point(global_tensor):
            output[key] = global_tensor.clone()
            continue

        stacked = torch.stack(
            [tensor.to(torch.float64) for tensor in tensors],
            dim=0,
        )

        median = torch.median(stacked, dim=0).values
        output[key] = median.to(dtype=global_tensor.dtype)

    return output


def coordinate_trimmed_mean(
    *,
    client_states: Sequence[Mapping[str, torch.Tensor]],
    global_state: Mapping[str, torch.Tensor],
    trim_count: int,
) -> Dict[str, torch.Tensor]:
    """
    Coordinate-wise Trimmed Mean.

    For n=5, trim_count=1:
        sort 5 values -> remove smallest 1 and largest 1 -> mean remaining 3.

    This robust defense is intentionally unweighted. Sample-weighted averaging
    is used only by the FedAvg baseline.
    """
    if not client_states:
        raise ValueError("client_states cannot be empty")

    num_clients = len(client_states)
    validate_trim_count(num_clients, trim_count)

    output: Dict[str, torch.Tensor] = {}
    expected_keys = set(global_state.keys())

    for index, state in enumerate(client_states, start=1):
        if set(state.keys()) != expected_keys:
            raise ValueError(
                f"Client state {index} has different state_dict keys."
            )

    for key, global_tensor_original in global_state.items():
        global_tensor = global_tensor_original.detach().cpu()

        tensors = [
            state[key].detach().cpu()
            for state in client_states
        ]

        if any(tensor.shape != global_tensor.shape for tensor in tensors):
            raise ValueError(f"Shape mismatch for state key {key!r}")

        if not torch.is_floating_point(global_tensor):
            output[key] = global_tensor.clone()
            continue

        stacked = torch.stack(
            [tensor.to(torch.float64) for tensor in tensors],
            dim=0,
        )

        sorted_values, _ = torch.sort(stacked, dim=0)

        if trim_count == 0:
            retained = sorted_values
        else:
            retained = sorted_values[
                trim_count : num_clients - trim_count
            ]

        mean_value = retained.mean(dim=0)
        output[key] = mean_value.to(dtype=global_tensor.dtype)

    return output


def aggregate_client_states(
    *,
    aggregator: str,
    client_states: Sequence[Mapping[str, torch.Tensor]],
    sample_weights: Sequence[int],
    global_state: Mapping[str, torch.Tensor],
    trim_count: int,
) -> Dict[str, torch.Tensor]:
    if aggregator == "fedavg":
        # Exact same sample-weighted server path used by Phase 17E.
        return phase4.federated_average(
            client_states,
            sample_weights,
        )

    if aggregator == "coordinate_median":
        return coordinate_median(
            client_states=client_states,
            global_state=global_state,
        )

    if aggregator == "trimmed_mean":
        return coordinate_trimmed_mean(
            client_states=client_states,
            global_state=global_state,
            trim_count=trim_count,
        )

    raise ValueError(f"Unknown aggregator: {aggregator}")


# =============================================================================
# AGGREGATION AUDIT
# =============================================================================

def audit_aggregation(
    *,
    aggregator: str,
    client_states: Sequence[Mapping[str, torch.Tensor]],
    sample_weights: Sequence[int],
    global_state: Mapping[str, torch.Tensor],
    aggregated_state: Mapping[str, torch.Tensor],
    trim_count: int,
) -> Dict[str, Any]:
    """
    Recompute the expected server aggregate and compare state dictionaries.

    This is intentionally redundant: its purpose is evidence/audit, not speed.
    """
    expected = aggregate_client_states(
        aggregator=aggregator,
        client_states=client_states,
        sample_weights=sample_weights,
        global_state=global_state,
        trim_count=trim_count,
    )

    max_error = max_state_abs_diff(expected, aggregated_state)

    return {
        "aggregator": aggregator,
        "max_recomputation_abs_error": float(max_error),
        "passed": bool(max_error <= 1e-7),
    }


# =============================================================================
# EXPERIMENT MATRIX
# =============================================================================

def matched_experiments(
    *,
    config: Mapping[str, Any],
    matrix: str,
) -> list[dict[str, Any]]:
    clean = {
        "id": "E0_CLEAN",
        "attack_type": "NONE",
        "magnitude": 1.0,
        "malicious_clients": [],
    }

    identity = list(config["identity_experiments"])
    ratio = list(config["ratio_experiments"])

    if matrix == "matched":
        # Phase 17F core comparison: 0%, 20%, 40% under SCALE/RANDOM x5.
        wanted = {
            "E0_CLEAN",
            "E_SCALE_C1",
            "E_SCALE_C1_C2",
            "E_RANDOM_C1",
            "E_RANDOM_C1_C2",
        }

        by_id = {clean["id"]: clean}
        for item in ratio:
            by_id[str(item["id"])] = dict(item)

        return [
            by_id[key]
            for key in (
                "E0_CLEAN",
                "E_SCALE_C1",
                "E_SCALE_C1_C2",
                "E_RANDOM_C1",
                "E_RANDOM_C1_C2",
            )
            if key in wanted
        ]

    if matrix == "identity":
        return [clean, *identity]

    if matrix == "full":
        by_id = {clean["id"]: clean}
        for item in [*identity, *ratio]:
            by_id[str(item["id"])] = dict(item)
        return list(by_id.values())

    if matrix == "smoke":
        by_id = {str(item["id"]): dict(item) for item in ratio}
        return [
            clean,
            by_id["E_SCALE_C1"],
            by_id["E_RANDOM_C1"],
        ]

    raise ValueError(matrix)


# =============================================================================
# ONE MATCHED RUN
# =============================================================================

def run_one(
    *,
    experiment: Mapping[str, Any],
    aggregator: str,
    args: argparse.Namespace,
    scaler,
    feature_columns,
    label_mapping,
    active_indices,
    active_features,
    global_weights,
    client_files: Mapping[int, Path],
    client_sample_counts: Mapping[str, int],
    device: torch.device,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    experiment_id = str(experiment["id"])
    attack_type = str(experiment["attack_type"])
    magnitude = float(experiment["magnitude"])
    malicious_clients = tuple(
        int(value)
        for value in experiment["malicious_clients"]
    )

    # Critical matched-comparison control: reset seed for EVERY aggregator run.
    phase10.set_seed(args.seed)

    spec = ByzantineAttackSpec(
        attack_type=attack_type,
        malicious_clients=malicious_clients,
        magnitude=magnitude,
        attack_seed=int(args.attack_seed),
    ).validate()

    malicious_ratio = float(
        len(malicious_clients) / phase10.N_CLIENTS
    )

    specialized_attacks = [
        phase4.CLIENT_ATTACKS[int(client_id)]
        for client_id in malicious_clients
    ]

    print()
    print("=" * 118)
    print(f"{experiment_id} | {DISPLAY_NAMES[aggregator]}")
    print("=" * 118)
    print(f"Attack type          : {attack_type}")
    print(f"Magnitude            : {magnitude:g}")
    print(f"Malicious clients    : {list(malicious_clients)}")
    print(f"Malicious ratio      : {malicious_ratio * 100:.0f}%")
    print(f"Specialized attacks  : {specialized_attacks}")
    print(f"Local training       : FedProx (mu={args.mu})")
    print(f"Server aggregation   : {DISPLAY_NAMES[aggregator]}")
    if aggregator == "trimmed_mean":
        print(f"Trim count           : {args.trim_count}")
        print(
            f"Values retained      : "
            f"{phase10.N_CLIENTS - 2 * args.trim_count}/{phase10.N_CLIENTS}"
        )
    print("Validation data      : CLEAN")
    print("Locked test used     : NO")

    global_model = phase10.initialize_full_model(
        input_dim=len(active_features),
        num_classes=len(label_mapping),
        device=device,
        seed=args.seed,
    )

    history: list[dict[str, Any]] = []
    attack_events: list[dict[str, Any]] = []
    aggregation_audits: list[dict[str, Any]] = []

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
                active_indices=active_indices,
                label_mapping=label_mapping,
                class_weights=global_weights,
                input_dim=len(active_features),
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

            routed_state, event = apply_byzantine_attack(
                global_state=global_model.state_dict(),
                local_state=local_state,
                client_id=client_id,
                spec=spec,
                round_number=round_number,
                experiment_seed=args.seed,
            )

            client_states.append(routed_state)
            attack_events.append(event)

            aggregation_weights.append(
                int(client_sample_counts[f"client_{client_id}"])
            )

            local_losses.append(float(local_loss))
            local_ce_losses.append(float(local_ce_loss))
            local_prox_losses.append(float(local_prox_loss))
            local_accuracies.append(float(local_accuracy))
            local_times.append(float(local_seconds))

        pre_aggregation_global = clone_state(global_model.state_dict())

        aggregated_state = aggregate_client_states(
            aggregator=aggregator,
            client_states=client_states,
            sample_weights=aggregation_weights,
            global_state=pre_aggregation_global,
            trim_count=args.trim_count,
        )

        aggregation_audit = audit_aggregation(
            aggregator=aggregator,
            client_states=client_states,
            sample_weights=aggregation_weights,
            global_state=pre_aggregation_global,
            aggregated_state=aggregated_state,
            trim_count=args.trim_count,
        )
        aggregation_audit["round"] = int(round_number)
        aggregation_audits.append(aggregation_audit)

        if not aggregation_audit["passed"]:
            raise RuntimeError(
                f"Aggregation audit failed in round {round_number}: "
                f"{aggregation_audit}"
            )

        global_model.load_state_dict(aggregated_state)

        global_update_norm = phase4.state_update_norm(
            old_global_state,
            aggregated_state,
        )

        weights_array = np.asarray(
            aggregation_weights,
            dtype=np.float64,
        )

        malicious_round_events = [
            event
            for event in attack_events
            if event["round"] == round_number
            and event["declared_malicious"]
        ]

        mean_malicious_honest_norm = None
        mean_malicious_routed_norm = None

        if malicious_round_events:
            mean_malicious_honest_norm = float(
                np.mean(
                    [
                        event_float(
                            event,
                            "legitimate_update_l2",
                            "honest_update_l2_norm",
                            "honest_update_l2",
                        )
                        for event in malicious_round_events
                    ]
                )
            )
            mean_malicious_routed_norm = float(
                np.mean(
                    [
                        event_float(
                            event,
                            "attacked_update_l2",
                            "routed_update_l2_norm",
                            "poisoned_update_l2_norm",
                            "submitted_update_l2",
                        )
                        for event in malicious_round_events
                    ]
                )
            )

        history.append(
            {
                "round": int(round_number),
                "aggregator": aggregator,
                "weighted_local_loss": float(
                    np.average(local_losses, weights=weights_array)
                ),
                "weighted_ce_loss": float(
                    np.average(local_ce_losses, weights=weights_array)
                ),
                "weighted_prox_loss": float(
                    np.average(local_prox_losses, weights=weights_array)
                ),
                "weighted_local_accuracy": float(
                    np.average(local_accuracies, weights=weights_array)
                ),
                "global_update_norm": float(global_update_norm),
                "mean_client_training_seconds": float(np.mean(local_times)),
                "mean_malicious_honest_update_norm": mean_malicious_honest_norm,
                "mean_malicious_routed_update_norm": mean_malicious_routed_norm,
                "aggregation_audit_max_error": float(
                    aggregation_audit["max_recomputation_abs_error"]
                ),
            }
        )

        attack_note = ""
        if malicious_round_events:
            attack_note = (
                f" | malicious_norm="
                f"{mean_malicious_honest_norm:.5f}"
                f"->{mean_malicious_routed_norm:.5f}"
            )

        print(
            f"Round {round_number:02d}/{args.rounds} | "
            f"loss={history[-1]['weighted_local_loss']:.6f} | "
            f"local_acc={history[-1]['weighted_local_accuracy']:.6f} | "
            f"global_norm={global_update_norm:.6f} | "
            f"agg_audit={aggregation_audit['max_recomputation_abs_error']:.2e}"
            f"{attack_note}"
        )

    training_seconds = float(time.perf_counter() - training_start)

    validation = phase17e.evaluate_on_validation(
        model=global_model,
        scaler=scaler,
        feature_columns=feature_columns,
        active_indices=active_indices,
        label_mapping=label_mapping,
        device=device,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )

    validation["overall"]["training_time_seconds"] = training_seconds

    attack_audit = phase17e.audit_attack_events(
        attack_type=attack_type,
        magnitude=magnitude,
        malicious_clients=malicious_clients,
        attack_events=attack_events,
        rounds=args.rounds,
        config=config,
    )

    if not attack_audit.get("passed", False):
        raise RuntimeError(
            f"Attack audit failed for {experiment_id}/{aggregator}: "
            f"{attack_audit}"
        )

    target_summary = phase17e.target_class_summary(
        validation=validation,
        malicious_clients=malicious_clients,
        client_attacks=phase4.CLIENT_ATTACKS,
    )

    final_state = clone_state(global_model.state_dict())
    state_sha256 = phase17e.sha256_state(final_state)

    max_aggregation_audit_error = max(
        float(item["max_recomputation_abs_error"])
        for item in aggregation_audits
    )

    return {
        "experiment_id": experiment_id,
        "aggregator": aggregator,
        "aggregator_display": DISPLAY_NAMES[aggregator],
        "seed": int(args.seed),
        "scenario": args.scenario,
        "attack_type": attack_type,
        "magnitude": magnitude,
        "malicious_clients": list(malicious_clients),
        "malicious_ratio": malicious_ratio,
        "specialized_attacks": specialized_attacks,
        "trim_count": (
            int(args.trim_count)
            if aggregator == "trimmed_mean"
            else None
        ),
        "history": history,
        "attack_events": attack_events,
        "attack_audit": attack_audit,
        "aggregation_audit": aggregation_audits,
        "max_aggregation_audit_error": max_aggregation_audit_error,
        "validation": validation,
        "target_summary": target_summary,
        "final_state": final_state,
        "final_state_sha256": state_sha256,
    }


# =============================================================================
# SAVE RUN
# =============================================================================

def save_run(run: Mapping[str, Any]) -> None:
    seed = int(run["seed"])
    aggregator = str(run["aggregator"])
    experiment_id = str(run["experiment_id"])

    result_dir = (
        RESULT_ROOT
        / f"seed_{seed}"
        / aggregator
        / experiment_id
    )

    artifact_dir = (
        ARTIFACT_ROOT
        / f"seed_{seed}"
        / aggregator
        / experiment_id
    )

    result_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(run["history"]).to_csv(
        result_dir / "round_history.csv",
        index=False,
    )

    pd.DataFrame(run["aggregation_audit"]).to_csv(
        result_dir / "aggregation_audit.csv",
        index=False,
    )

    save_json(
        result_dir / "attack_events.json",
        run["attack_events"],
    )

    save_json(
        result_dir / "attack_audit.json",
        run["attack_audit"],
    )

    save_json(
        result_dir / "target_attack_summary.json",
        run["target_summary"],
    )

    save_json(
        result_dir / "validation_metrics.json",
        run["validation"],
    )

    metadata = {
        key: value
        for key, value in run.items()
        if key not in {
            "history",
            "attack_events",
            "attack_audit",
            "aggregation_audit",
            "validation",
            "target_summary",
            "final_state",
        }
    }

    save_json(
        result_dir / "experiment.json",
        metadata,
    )

    per_class = run["validation"]["per_class"]
    pd.DataFrame(
        [
            {"class": class_name, **metrics}
            for class_name, metrics in per_class.items()
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
            "phase": "17F",
            "seed": seed,
            "scenario": run["scenario"],
            "experiment_id": experiment_id,
            "aggregator": aggregator,
            "attack_type": run["attack_type"],
            "magnitude": run["magnitude"],
            "malicious_clients": run["malicious_clients"],
            "model_state_dict": run["final_state"],
            "state_sha256": run["final_state_sha256"],
        },
        artifact_dir / "fedprox70_phase17f_model.pt",
    )


# =============================================================================
# COMPARISON TABLE
# =============================================================================

def safe_damage_recovery(
    *,
    fedavg_clean: float,
    fedavg_attacked: float,
    defended_attacked: float,
) -> float | None:
    denominator = float(fedavg_clean) - float(fedavg_attacked)

    if abs(denominator) < 1e-12:
        return None

    return float(
        (float(defended_attacked) - float(fedavg_attacked))
        / denominator
    )


def build_comparison_rows(
    runs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {
        (str(run["aggregator"]), str(run["experiment_id"])): run
        for run in runs
    }

    fedavg_clean_run = by_key.get(("fedavg", "E0_CLEAN"))
    if fedavg_clean_run is None:
        raise RuntimeError("FedAvg clean run is required for comparison.")

    fedavg_clean_overall = fedavg_clean_run["validation"]["overall"]

    rows: list[dict[str, Any]] = []

    for run in runs:
        overall = run["validation"]["overall"]
        aggregator = str(run["aggregator"])
        experiment_id = str(run["experiment_id"])

        own_clean_run = by_key.get((aggregator, "E0_CLEAN"))
        if own_clean_run is None:
            raise RuntimeError(
                f"Missing clean baseline for aggregator {aggregator}"
            )

        own_clean = own_clean_run["validation"]["overall"]

        fedavg_matched_run = by_key.get(("fedavg", experiment_id))
        fedavg_matched = (
            fedavg_matched_run["validation"]["overall"]
            if fedavg_matched_run is not None
            else None
        )

        is_clean = run["attack_type"] == "NONE"

        macro_f1_drop_vs_own_clean = float(
            own_clean["macro_f1"] - overall["macro_f1"]
        )

        balanced_drop_vs_own_clean = float(
            own_clean["balanced_accuracy"]
            - overall["balanced_accuracy"]
        )

        weighted_f1_drop_vs_own_clean = float(
            own_clean["weighted_f1"] - overall["weighted_f1"]
        )

        macro_f1_recovery_vs_fedavg = None
        balanced_recovery_vs_fedavg = None
        weighted_f1_recovery_vs_fedavg = None
        worst_attack_recall_recovery_vs_fedavg = None
        damage_recovery_fraction = None
        damage_recovery_pct = None

        if fedavg_matched is not None:
            macro_f1_recovery_vs_fedavg = float(
                overall["macro_f1"] - fedavg_matched["macro_f1"]
            )
            balanced_recovery_vs_fedavg = float(
                overall["balanced_accuracy"]
                - fedavg_matched["balanced_accuracy"]
            )
            weighted_f1_recovery_vs_fedavg = float(
                overall["weighted_f1"] - fedavg_matched["weighted_f1"]
            )
            worst_attack_recall_recovery_vs_fedavg = float(
                overall["worst_attack_recall"]
                - fedavg_matched["worst_attack_recall"]
            )

            if not is_clean and aggregator != "fedavg":
                damage_recovery_fraction = safe_damage_recovery(
                    fedavg_clean=float(fedavg_clean_overall["macro_f1"]),
                    fedavg_attacked=float(fedavg_matched["macro_f1"]),
                    defended_attacked=float(overall["macro_f1"]),
                )

                if damage_recovery_fraction is not None:
                    damage_recovery_pct = float(
                        100.0 * damage_recovery_fraction
                    )

        rows.append(
            {
                "seed": int(run["seed"]),
                "scenario": run["scenario"],
                "experiment_id": experiment_id,
                "attack_type": run["attack_type"],
                "magnitude": float(run["magnitude"]),
                "malicious_clients": "+".join(
                    str(value) for value in run["malicious_clients"]
                ),
                "malicious_ratio": float(run["malicious_ratio"]),
                "aggregator": aggregator,
                "aggregator_display": run["aggregator_display"],
                "trim_count": run["trim_count"],
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
                "target_attack_mean_recall": (
                    run["target_summary"]["target_attack_mean_recall"]
                ),
                "target_attack_worst_recall": (
                    run["target_summary"]["target_attack_worst_recall"]
                ),
                "macro_f1_drop_vs_own_clean": macro_f1_drop_vs_own_clean,
                "balanced_accuracy_drop_vs_own_clean": balanced_drop_vs_own_clean,
                "weighted_f1_drop_vs_own_clean": weighted_f1_drop_vs_own_clean,
                "clean_macro_f1_cost_vs_fedavg": (
                    float(
                        overall["macro_f1"]
                        - fedavg_clean_overall["macro_f1"]
                    )
                    if is_clean
                    else None
                ),
                "macro_f1_recovery_vs_fedavg": macro_f1_recovery_vs_fedavg,
                "balanced_accuracy_recovery_vs_fedavg": (
                    balanced_recovery_vs_fedavg
                ),
                "weighted_f1_recovery_vs_fedavg": (
                    weighted_f1_recovery_vs_fedavg
                ),
                "worst_attack_recall_recovery_vs_fedavg": (
                    worst_attack_recall_recovery_vs_fedavg
                ),
                "macro_f1_damage_recovery_fraction": (
                    damage_recovery_fraction
                ),
                "macro_f1_damage_recovery_pct": damage_recovery_pct,
                "attack_audit_pass": bool(run["attack_audit"]["passed"]),
                "max_aggregation_audit_error": float(
                    run["max_aggregation_audit_error"]
                ),
                "validation_poisoned": False,
                "locked_test_used": False,
            }
        )

    return rows


# =============================================================================
# MAIN
# =============================================================================

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
    parser.add_argument("--attack-seed", type=int, default=1704)

    parser.add_argument(
        "--matrix",
        choices=["smoke", "matched", "identity", "full"],
        default="matched",
        help=(
            "matched = clean + SCALE/RANDOM x5 at 20% and 40%; "
            "identity = client identity rotation; full = union."
        ),
    )

    parser.add_argument(
        "--aggregators",
        nargs="+",
        choices=list(AGGREGATORS),
        default=list(AGGREGATORS),
        help="Server aggregators to compare.",
    )

    parser.add_argument(
        "--trim-count",
        type=int,
        default=1,
        help=(
            "Number removed from each tail for Trimmed Mean. "
            "For 5 clients / up to 1 Byzantine client, use 1."
        ),
    )

    parser.add_argument(
        "--allow-source-hash-mismatch",
        action="store_true",
    )

    args = parser.parse_args()

    if not math.isclose(float(args.mu), 0.01, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Phase17F freezes FedProx mu=0.01.")

    validate_trim_count(
        phase10.N_CLIENTS,
        args.trim_count,
    )

    config = phase17e.load_json(
        phase17e.CONFIG_FILE
    )

    prerequisite_audit = phase17e.verify_prerequisites(
        config,
        allow_source_hash_mismatch=args.allow_source_hash_mismatch,
    )

    experiments = matched_experiments(
        config=config,
        matrix=args.matrix,
    )

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

    global_weights = phase4.calculate_global_class_weights(
        global_counts
    )

    client_files: Dict[int, Path] = {}
    client_sample_counts: Dict[str, int] = {}

    for client_id in range(1, phase10.N_CLIENTS + 1):
        path = scenario_dir / f"client_{client_id}.csv"
        if not path.exists():
            raise FileNotFoundError(path)

        client_files[client_id] = path
        client_sample_counts[f"client_{client_id}"] = (
            phase4.count_client_samples(
                path,
                args.chunk_size,
            )
        )

    print()
    print("=" * 118)
    print(PHASE_NAME)
    print("=" * 118)
    print(f"Scenario             : {args.scenario}")
    print(f"Seed                 : {args.seed}")
    print(f"Device               : {device}")
    print("Features             : 70")
    print(f"Clients              : {phase10.N_CLIENTS}")
    print(f"Rounds               : {args.rounds}")
    print(f"Local epochs         : {args.local_epochs}")
    print(f"FedProx mu           : {args.mu}")
    print(f"Matrix               : {args.matrix}")
    print(f"Aggregators          : {[DISPLAY_NAMES[x] for x in args.aggregators]}")
    print(f"Trim count           : {args.trim_count}")
    print("Attacks              : Phase17E SCALE x5 / RANDOM x5")
    print("Locked test used     : NO")
    print(f"Experiments          : {[item['id'] for item in experiments]}")

    print()
    print("MATCHED-COMPARISON RULE:")
    print(
        "For every experiment, the same seed and same client attack are rerun "
        "for each server aggregator; only aggregation changes."
    )

    runs: list[dict[str, Any]] = []

    for experiment in experiments:
        for aggregator in args.aggregators:
            run = run_one(
                experiment=experiment,
                aggregator=aggregator,
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

            save_run(run)
            runs.append(run)

    comparison_rows = build_comparison_rows(runs)
    comparison_frame = pd.DataFrame(comparison_rows)

    summary_dir = RESULT_ROOT / f"seed_{args.seed}"
    summary_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = summary_dir / "phase17f_matched_defense_comparison.csv"
    comparison_frame.to_csv(comparison_path, index=False)

    # Easy-to-read pivot for Macro F1.
    macro_f1_pivot = comparison_frame.pivot_table(
        index=[
            "experiment_id",
            "attack_type",
            "malicious_ratio",
        ],
        columns="aggregator_display",
        values="macro_f1",
        aggfunc="first",
    ).reset_index()

    macro_f1_path = summary_dir / "phase17f_macro_f1_pivot.csv"
    macro_f1_pivot.to_csv(macro_f1_path, index=False)

    # Per-class comparison across all runs.
    per_class_rows: list[dict[str, Any]] = []

    for run in runs:
        for class_name, metrics in run["validation"]["per_class"].items():
            per_class_rows.append(
                {
                    "seed": int(run["seed"]),
                    "experiment_id": run["experiment_id"],
                    "attack_type": run["attack_type"],
                    "malicious_ratio": run["malicious_ratio"],
                    "aggregator": run["aggregator"],
                    "class": class_name,
                    **metrics,
                }
            )

    per_class_path = summary_dir / "phase17f_per_class_comparison.csv"
    pd.DataFrame(per_class_rows).to_csv(per_class_path, index=False)

    max_aggregation_error = max(
        float(run["max_aggregation_audit_error"])
        for run in runs
    )

    attack_audits_pass = all(
        bool(run["attack_audit"].get("passed", False))
        for run in runs
    )

    manifest = {
        "phase": "17F",
        "status": (
            "PASS"
            if attack_audits_pass and max_aggregation_error <= 1e-7
            else "FAIL"
        ),
        "scenario": args.scenario,
        "seed": int(args.seed),
        "rounds": int(args.rounds),
        "local_epochs": int(args.local_epochs),
        "num_clients": int(phase10.N_CLIENTS),
        "features": 70,
        "local_training": {
            "algorithm": "FedProx",
            "mu": float(args.mu),
        },
        "aggregators": list(args.aggregators),
        "trim_count": int(args.trim_count),
        "matrix": args.matrix,
        "experiments": [dict(item) for item in experiments],
        "attacks": "Phase17E SCALE x5 / RANDOM x5",
        "matched_seed_reset_per_run": True,
        "validation_poisoned": False,
        "locked_test_used": False,
        "prerequisite_audit": prerequisite_audit,
        "attack_audits_pass": bool(attack_audits_pass),
        "max_aggregation_audit_error": float(max_aggregation_error),
        "comparison_csv": str(comparison_path),
        "macro_f1_pivot_csv": str(macro_f1_path),
        "per_class_csv": str(per_class_path),
    }

    manifest_path = ARTIFACT_ROOT / f"seed_{args.seed}" / "phase17f_manifest.json"
    save_json(manifest_path, manifest)

    print()
    print("=" * 118)
    print("PHASE 17F RESULTS — MACRO F1")
    print("=" * 118)

    for _, row in macro_f1_pivot.iterrows():
        print()
        print(
            f"{row['experiment_id']} | "
            f"attack={row['attack_type']} | "
            f"malicious={100.0 * float(row['malicious_ratio']):.0f}%"
        )

        for display_name in (
            "FedAvg",
            "Coordinate Median",
            "Trimmed Mean",
        ):
            if display_name in row.index and pd.notna(row[display_name]):
                print(
                    f"  {display_name:<20}: "
                    f"{float(row[display_name]):.6f}"
                )

    print()
    print("=" * 78)
    print("PHASE 17F COMPLETE")
    print("=" * 78)
    print(f"Runs completed           : {len(runs)}")
    print(f"Attack audits PASS       : {attack_audits_pass}")
    print(f"Max aggregation error    : {max_aggregation_error:.12g}")
    print("Validation poisoned      : NO")
    print("Locked test used         : NO")
    print(f"Comparison               : {comparison_path}")
    print(f"Macro-F1 pivot           : {macro_f1_path}")
    print(f"Per-class comparison     : {per_class_path}")
    print(f"Manifest                 : {manifest_path}")
    print(f"STATUS                   : {manifest['status']}")


if __name__ == "__main__":
    main()
