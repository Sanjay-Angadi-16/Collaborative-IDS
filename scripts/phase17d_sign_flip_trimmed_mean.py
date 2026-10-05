from __future__ import annotations

r"""
Phase 17D — Sign-Flip Attack + Trimmed-Mean Defense
====================================================

Purpose
-------
Evaluate whether coordinate-wise Trimmed Mean can defend the existing
Federated IDS against the Phase 17C Byzantine sign-flip attack.

The experiment intentionally keeps the Phase 17C client-side path frozen:

    1. Global model is sent to all clients.
    2. Every client performs the same FedProx local training (mu=0.01).
    3. The selected malicious client applies the same Phase 17C sign flip:

           delta_i          = W_i(local) - W_global
           delta_i_poisoned = -scale * delta_i
           W_i(poisoned)    = W_global + delta_i_poisoned

    4. ONLY the server aggregation changes:

           Phase 17C: sample-weighted FedAvg
           Phase 17D (trim_count=1): coordinate-wise Trimmed Mean

Important control mode:
    - trim_count=0 is NOT treated as a defense.
    - trim_count=0 deliberately calls the exact same sample-weighted FedAvg
      aggregation function used by Phase 17C. This is a reproducibility control.

With 5 clients and trim_count=1, each floating-point model coordinate:
    - sorts the 5 submitted client values,
    - removes the smallest 1,
    - removes the largest 1,
    - averages the remaining 3 values.

This is equivalent to applying coordinate-wise Trimmed Mean in update space,
because every client update is measured relative to the same global model.

Research controls
-----------------
- Original client CSV files are NOT modified.
- Validation data remains CLEAN.
- Locked test data is NOT used.
- Feature set remains the frozen 70-feature set.
- Model architecture remains unchanged.
- Local training remains FedProx with mu=0.01.
- Same malicious client / attack scales as Phase 17C can be reused.
- For trim_count=1, only the server aggregation rule is changed from FedAvg
  to unweighted coordinate-wise Trimmed Mean.
- For trim_count=0, the exact Phase 17C sample-weighted FedAvg path is used
  to verify reproducibility before interpreting the defense.

Recommended first run
---------------------
    python scripts\phase17d_sign_flip_trimmed_mean.py \
        --scenario non_iid \
        --seed 42 \
        --malicious-client 1 \
        --scales 1.0 2.0 5.0 \
        --trim-count 1

After seed 42 passes, repeat seeds 52, 62, 72 and 82.
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
    import phase17c_sign_flip_fedavg as phase17c
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "\nPhase 17D requires the validated Phase 17C runner:\n"
        "    scripts\\phase17c_sign_flip_fedavg.py\n"
        "Place this file inside the project's scripts folder."
    ) from exc

phase17a = phase17c.phase17a
phase10 = phase17c.phase10
phase4 = phase17c.phase4

from src.federated.poisoning.metrics import compare_to_clean  # noqa: E402


# =============================================================================
# CONSTANTS
# =============================================================================

PHASE_NAME = "PHASE 17D — SIGN-FLIP + TRIMMED-MEAN DEFENSE"

RESULT_ROOT = PROJECT_ROOT / "results" / "phase17" / "phase17d"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "poisoning" / "phase17d"

# Existing Phase 17C FedAvg evidence, used only for comparison if available.
PHASE17C_RESULT_ROOT = PROJECT_ROOT / "results" / "phase17" / "phase17c"


# =============================================================================
# BASIC HELPERS
# =============================================================================


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def clone_state(
    state: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return phase17c.clone_state(state)


def state_max_abs_diff(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> float:
    return phase17c.state_max_abs_diff(left, right)


def aggregation_mode_name(trim_count: int) -> str:
    """Human-readable server aggregation mode."""
    if int(trim_count) == 0:
        return "phase17c_sample_weighted_fedavg_control"
    return "coordinate_wise_trimmed_mean"


# =============================================================================
# TRIMMED-MEAN AGGREGATION
# =============================================================================


def validate_trim_configuration(
    *,
    num_clients: int,
    trim_count: int,
) -> None:
    """Validate coordinate-wise Trimmed Mean configuration."""
    if num_clients < 1:
        raise ValueError("num_clients must be >= 1")

    if trim_count < 0:
        raise ValueError("trim_count must be >= 0")

    retained = num_clients - 2 * trim_count

    if retained <= 0:
        raise ValueError(
            "Invalid Trimmed Mean configuration: "
            f"num_clients={num_clients}, trim_count={trim_count}. "
            "Need num_clients - 2*trim_count >= 1."
        )


def coordinate_wise_trimmed_mean(
    *,
    client_states: Sequence[Mapping[str, torch.Tensor]],
    global_state: Mapping[str, torch.Tensor],
    trim_count: int,
) -> Dict[str, torch.Tensor]:
    """
    Coordinate-wise Trimmed Mean over submitted client model states.

    For each floating-point coordinate:

        values = [v_1, ..., v_n]
        values = sort(values)
        retained = values[trim_count : n - trim_count]
        aggregate = mean(retained)

    With 5 clients and trim_count=1, 3/5 values are retained.

    Non-floating state entries are copied from the current global model.
    They are not meaningful targets for coordinate-wise robust averaging.

    Notes
    -----
    Because all client updates share the same current global model G,

        trimmed_mean(W_i) - G == trimmed_mean(W_i - G)

    for floating-point coordinates. Therefore model-state-space aggregation
    here is equivalent to update-space Trimmed Mean.
    """
    if not client_states:
        raise ValueError("client_states cannot be empty")

    num_clients = len(client_states)
    validate_trim_configuration(
        num_clients=num_clients,
        trim_count=trim_count,
    )

    reference_keys = set(global_state.keys())

    for idx, state in enumerate(client_states, start=1):
        if set(state.keys()) != reference_keys:
            raise ValueError(
                f"Client state {idx} has different state_dict keys."
            )

    aggregated: Dict[str, torch.Tensor] = {}

    for key in global_state:
        global_tensor = global_state[key].detach().cpu()

        tensors = [
            state[key].detach().cpu()
            for state in client_states
        ]

        for tensor in tensors:
            if tensor.shape != global_tensor.shape:
                raise ValueError(
                    f"Shape mismatch for parameter {key!r}: "
                    f"expected {tuple(global_tensor.shape)}, "
                    f"got {tuple(tensor.shape)}"
                )

        if not torch.is_floating_point(global_tensor):
            # Preserve non-floating buffers exactly from the current global
            # model. The IDS MLP is expected to be floating-point parameters,
            # but this keeps the aggregator safe for state_dict buffers.
            aggregated[key] = global_tensor.clone()
            continue

        # Use float64 internally for numerically stable sorting/averaging.
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

        aggregated[key] = mean_value.to(
            dtype=global_tensor.dtype
        )

    return aggregated


def aggregate_server_state(
    *,
    client_states: Sequence[Mapping[str, torch.Tensor]],
    global_state: Mapping[str, torch.Tensor],
    sample_weights: Sequence[int],
    trim_count: int,
) -> Dict[str, torch.Tensor]:
    """
    Server aggregation used by Phase 17D.

    trim_count == 0
        Exact Phase 17C reproduction control. Calls the SAME sample-weighted
        FedAvg function as Phase 17C:

            phase4.federated_average(client_states, sample_weights)

        This is intentionally different from an unweighted arithmetic mean.

    trim_count > 0
        Actual Phase 17D defense: unweighted coordinate-wise Trimmed Mean.
    """
    if int(trim_count) == 0:
        return phase4.federated_average(
            list(client_states),
            list(sample_weights),
        )

    return coordinate_wise_trimmed_mean(
        client_states=client_states,
        global_state=global_state,
        trim_count=int(trim_count),
    )


# =============================================================================
# AGGREGATION AUDIT
# =============================================================================


def calculate_coordinate_envelope_violations(
    *,
    client_states: Sequence[Mapping[str, torch.Tensor]],
    aggregated_state: Mapping[str, torch.Tensor],
    trim_count: int,
) -> Dict[str, float]:
    """
    Audit that each floating-point aggregate lies inside the retained
    coordinate interval after trimming.

    For a valid coordinate-wise Trimmed Mean:

        retained_min <= aggregate <= retained_max

    up to floating-point tolerance.
    """
    num_clients = len(client_states)
    validate_trim_configuration(
        num_clients=num_clients,
        trim_count=trim_count,
    )

    max_lower_violation = 0.0
    max_upper_violation = 0.0
    checked_coordinates = 0

    for key, aggregate_tensor in aggregated_state.items():
        if not torch.is_floating_point(aggregate_tensor):
            continue

        stacked = torch.stack(
            [
                state[key].detach().cpu().to(torch.float64)
                for state in client_states
            ],
            dim=0,
        )

        sorted_values, _ = torch.sort(stacked, dim=0)

        if trim_count == 0:
            retained = sorted_values
        else:
            retained = sorted_values[
                trim_count : num_clients - trim_count
            ]

        lower = retained[0]
        upper = retained[-1]
        aggregate = aggregate_tensor.detach().cpu().to(torch.float64)

        lower_violation = torch.clamp(lower - aggregate, min=0.0)
        upper_violation = torch.clamp(aggregate - upper, min=0.0)

        if lower_violation.numel() > 0:
            max_lower_violation = max(
                max_lower_violation,
                float(lower_violation.max().item()),
            )
            max_upper_violation = max(
                max_upper_violation,
                float(upper_violation.max().item()),
            )
            checked_coordinates += int(lower_violation.numel())

    return {
        "checked_coordinates": float(checked_coordinates),
        "max_lower_envelope_violation": float(max_lower_violation),
        "max_upper_envelope_violation": float(max_upper_violation),
        "max_envelope_violation": float(
            max(max_lower_violation, max_upper_violation)
        ),
    }


# =============================================================================
# EXPERIMENT DEFINITIONS
# =============================================================================


def scale_tag(scale: float) -> str:
    return phase17c.scale_tag(scale)


def make_experiments(
    *,
    scales: Sequence[float],
    malicious_client: int,
) -> list[dict[str, Any]]:
    """Clean Trimmed Mean baseline + attacked Trimmed Mean experiments."""
    experiments: list[dict[str, Any]] = [
        {
            "id": "D0_TRIMMED_MEAN_CLEAN",
            "attack_type": "NONE",
            "malicious_client": None,
            "scale": 1.0,
        }
    ]

    for scale in scales:
        experiments.append(
            {
                "id": f"D_SIGN_FLIP_X{scale_tag(float(scale))}_TRIMMED_MEAN",
                "attack_type": "SIGN_FLIP",
                "malicious_client": int(malicious_client),
                "scale": float(scale),
            }
        )

    return experiments


# =============================================================================
# FEDERATED TRAINING
# =============================================================================


def run_trimmed_mean_experiment(
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
    """
    Run one Phase 17D experiment.

    Client side:
        Frozen FedProx training + optional Phase17C sign flip.

    Server side:
        Coordinate-wise Trimmed Mean.
    """
    experiment_id = str(experiment["id"])
    attack_type = str(experiment["attack_type"])
    malicious_client = experiment["malicious_client"]
    scale = float(experiment["scale"])

    if attack_type not in {"NONE", "SIGN_FLIP"}:
        raise ValueError(f"Unsupported Phase17D attack_type={attack_type!r}")

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
    print(
        "Sign-flip scale      : "
        + (
            f"{scale:g}"
            if attack_type == "SIGN_FLIP"
            else "N/A"
        )
    )
    print("Local training       : FedProx (frozen)")
    if args.trim_count == 0:
        print("Server aggregation   : Phase17C sample-weighted FedAvg CONTROL")
        print("Defense active       : NO")
    else:
        print("Server aggregation   : Coordinate-wise Trimmed Mean")
        print("Defense active       : YES")
    print(f"Trim count           : {args.trim_count}")
    print(
        f"Values retained      : "
        f"{phase10.N_CLIENTS - 2 * args.trim_count}/{phase10.N_CLIENTS}"
    )
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
    aggregation_audit: list[dict[str, Any]] = []

    training_start = time.perf_counter()

    for round_number in range(1, args.rounds + 1):
        old_global_state = clone_state(global_model.state_dict())

        client_states: list[Dict[str, torch.Tensor]] = []
        sample_weights: list[int] = []
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

            # -------------------------------------------------------------
            # SAME PHASE 17C ATTACK BOUNDARY
            # -------------------------------------------------------------
            # Attack is applied AFTER FedProx local training and BEFORE
            # server aggregation. Only the selected client is modified.
            if attack_type == "SIGN_FLIP":
                routed_state, attack_event = phase17c.route_client_state(
                    global_state=global_model.state_dict(),
                    local_state=local_state,
                    client_id=client_id,
                    malicious_client=int(malicious_client),
                    scale=scale,
                    round_number=round_number,
                )
            else:
                routed_state, attack_event = phase17c.route_client_state(
                    global_state=global_model.state_dict(),
                    local_state=local_state,
                    client_id=client_id,
                    malicious_client=None,
                    scale=1.0,
                    round_number=round_number,
                )

            attack_events.append(attack_event)
            client_states.append(routed_state)

            # Keep sample counts for weighted local-training summaries.
            # They are ALSO used by trim_count=0 so that the reproduction
            # control calls the exact sample-weighted Phase17C FedAvg path.
            # For trim_count>0, robust Trimmed Mean remains unweighted.
            sample_weights.append(
                int(client_sample_counts[f"client_{client_id}"])
            )

            local_losses.append(float(local_loss))
            local_ce_losses.append(float(local_ce_loss))
            local_prox_losses.append(float(local_prox_loss))
            local_accuracies.append(float(local_accuracy))
            local_times.append(float(local_seconds))

        # -------------------------------------------------------------
        # SERVER AGGREGATION BOUNDARY
        # -------------------------------------------------------------
        # trim_count == 0:
        #     exact Phase17C sample-weighted FedAvg reproduction control.
        # trim_count > 0:
        #     actual coordinate-wise Trimmed Mean defense.
        aggregated_state = aggregate_server_state(
            client_states=client_states,
            global_state=global_model.state_dict(),
            sample_weights=sample_weights,
            trim_count=args.trim_count,
        )

        audit = calculate_coordinate_envelope_violations(
            client_states=client_states,
            aggregated_state=aggregated_state,
            trim_count=args.trim_count,
        )
        audit["round"] = int(round_number)
        aggregation_audit.append(audit)

        global_model.load_state_dict(aggregated_state)

        update_norm = phase4.state_update_norm(
            old_global_state,
            aggregated_state,
        )

        weights = np.asarray(sample_weights, dtype=np.float64)

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
                "server_aggregation_mode": aggregation_mode_name(args.trim_count),
                "trim_count": int(args.trim_count),
                "values_retained": int(
                    len(client_states) - 2 * args.trim_count
                ),
                "max_trimmed_mean_envelope_violation": float(
                    audit["max_envelope_violation"]
                ),
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
            f"global_update_norm={history[-1]['global_update_norm']:.6f} | "
            f"trim_audit={audit['max_envelope_violation']:.3e}"
            f"{attack_text}"
        )

    training_seconds = float(time.perf_counter() - training_start)

    # Same clean validation path as Phase 17A / Phase 17C.
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
        "defense_spec": {
            "aggregation": aggregation_mode_name(args.trim_count),
            "defense_active": bool(args.trim_count > 0),
            "num_clients": int(phase10.N_CLIENTS),
            "trim_count": int(args.trim_count),
            "values_retained": int(
                phase10.N_CLIENTS - 2 * args.trim_count
            ),
            "weighted": bool(args.trim_count == 0),
            "weighting_note": (
                "Phase17C sample-count weights"
                if args.trim_count == 0
                else "unweighted robust coordinate aggregation"
            ),
            "non_float_policy": "preserve_current_global_state",
        },
        "history": history,
        "attack_events": attack_events,
        "aggregation_audit": aggregation_audit,
        "validation": validation_metrics,
        "final_state": final_state,
        "final_state_sha256": final_state_sha256,
    }


# =============================================================================
# PERSISTENCE
# =============================================================================


def save_phase17d_experiment(
    *,
    run: Mapping[str, Any],
    seed: int,
) -> None:
    experiment_id = str(run["experiment_id"])

    trim_count = int(run["defense_spec"]["trim_count"])
    trim_dir = f"trim_{trim_count}"

    result_dir = RESULT_ROOT / f"seed_{seed}" / trim_dir / experiment_id
    artifact_dir = ARTIFACT_ROOT / f"seed_{seed}" / trim_dir / experiment_id

    result_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(run["history"]).to_csv(
        result_dir / "round_history.csv",
        index=False,
    )

    pd.DataFrame(run["aggregation_audit"]).to_csv(
        result_dir / "trimmed_mean_audit.csv",
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
        result_dir / "defense_spec.json",
        run["defense_spec"],
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
            "phase": "17D",
            "experiment_id": experiment_id,
            "seed": int(seed),
            "attack_spec": run["attack_spec"],
            "defense_spec": run["defense_spec"],
            "model_state_dict": run["final_state"],
            "state_sha256": run["final_state_sha256"],
        },
        artifact_dir / "fedprox70_phase17d_trimmed_mean_model.pt",
    )


# =============================================================================
# PHASE 17C FEDAVG COMPARISON
# =============================================================================


def load_phase17c_fedavg_results(
    *,
    seed: int,
) -> pd.DataFrame | None:
    """
    Load same-seed Phase 17C FedAvg results if they exist.

    Phase 17D does NOT depend on these files for training. They are used only
    to quantify how much performance the robust aggregator recovers relative
    to the already-completed Phase 17C attack baseline.
    """
    path = (
        PHASE17C_RESULT_ROOT
        / f"seed_{seed}"
        / "phase17c_clean_vs_sign_flip_comparison.csv"
    )

    if not path.exists():
        return None

    frame = pd.read_csv(path)

    required = {
        "attack_type",
        "sign_flip_scale",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "worst_attack_recall",
    }

    missing = required - set(frame.columns)
    if missing:
        print(
            "WARNING: Phase17C comparison exists but is missing columns: "
            + ", ".join(sorted(missing))
        )
        return None

    return frame


def find_phase17c_row(
    *,
    phase17c_frame: pd.DataFrame,
    attack_type: str,
    scale: float | None,
) -> pd.Series | None:
    if attack_type == "NONE":
        rows = phase17c_frame[
            phase17c_frame["attack_type"].astype(str).eq("NONE")
        ]
    else:
        rows = phase17c_frame[
            phase17c_frame["attack_type"].astype(str).eq("SIGN_FLIP")
        ].copy()

        if rows.empty:
            return None

        numeric_scale = pd.to_numeric(
            rows["sign_flip_scale"],
            errors="coerce",
        )

        rows = rows[
            np.isclose(
                numeric_scale.to_numpy(dtype=float),
                float(scale),
                rtol=0.0,
                atol=1e-12,
            )
        ]

    if rows.empty:
        return None

    return rows.iloc[0]


def safe_recovery_fraction(
    *,
    clean_value: float,
    attacked_value: float,
    defended_value: float,
) -> float | None:
    """
    Fraction of attack-induced damage recovered by the defense.

        recovery_fraction =
            (defended - attacked) / (clean - attacked)

    1.0  -> full recovery to the clean FedAvg level
    0.0  -> no recovery
    >1.0 -> defense exceeds clean FedAvg performance
    <0.0 -> defense is worse than attacked FedAvg
    """
    denominator = float(clean_value) - float(attacked_value)

    if abs(denominator) < 1e-12:
        return None

    return float(
        (float(defended_value) - float(attacked_value))
        / denominator
    )


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

    parser.add_argument(
        "--malicious-client",
        type=int,
        default=1,
        help="1-based malicious client id. Default: client_1.",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Run one sign-flip magnitude, e.g. --scale 1.0.",
    )

    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=None,
        help="Attack-strength sweep, e.g. --scales 1.0 2.0 5.0.",
    )

    parser.add_argument(
        "--trim-count",
        type=int,
        default=1,
        help=(
            "Number of lowest and highest client values removed per "
            "coordinate. With 5 clients and 1 Byzantine client use 1."
        ),
    )

    parser.add_argument(
        "--control-tolerance",
        type=float,
        default=1e-6,
        help=(
            "Maximum allowed absolute metric difference between trim_count=0 "
            "and same-seed Phase17C FedAvg before the reproduction control is "
            "marked FAIL. Default: 1e-6."
        ),
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run clean + first attacked experiment for one round unless "
            "--rounds was explicitly changed."
        ),
    )

    parser.add_argument(
        "--allow-source-hash-mismatch",
        action="store_true",
        help=(
            "Debug-only. Final evidence should use the frozen Phase17A "
            "source hashes."
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
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"Every sign-flip scale must be finite and > 0, got {scale}"
            )

    args.scales = resolved_scales

    if not np.isfinite(args.control_tolerance) or args.control_tolerance < 0.0:
        raise ValueError("--control-tolerance must be finite and >= 0.")

    if not math.isclose(float(args.mu), 0.01, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "Phase17D keeps the frozen FedProx mu=0.01 local-training path."
        )

    validate_trim_configuration(
        num_clients=phase10.N_CLIENTS,
        trim_count=args.trim_count,
    )

    if args.smoke and args.rounds == 10:
        args.rounds = 1

    # ---------------------------------------------------------------------
    # Verify frozen Phase 17A baseline and data policy.
    # ---------------------------------------------------------------------
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

    global_weights = phase4.calculate_global_class_weights(
        global_counts
    )

    client_files: Dict[int, Path] = {}
    client_sample_counts: Dict[str, int] = {}

    for client_id in range(1, phase10.N_CLIENTS + 1):
        client_file = scenario_dir / f"client_{client_id}.csv"

        if not client_file.exists():
            raise FileNotFoundError(client_file)

        client_files[client_id] = client_file

        client_sample_counts[f"client_{client_id}"] = (
            phase4.count_client_samples(
                client_file,
                args.chunk_size,
            )
        )

    experiments = make_experiments(
        scales=args.scales,
        malicious_client=args.malicious_client,
    )

    if args.smoke:
        experiments = experiments[:2]

    retained_count = phase10.N_CLIENTS - 2 * args.trim_count

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
    print(f"Sign-flip scales      : {args.scales}")
    print("Local optimizer       : Frozen FedProx path (mu=0.01)")
    if args.trim_count == 0:
        print("Server aggregation    : Phase17C sample-weighted FedAvg CONTROL")
        print("Defense active        : NO")
        print("Aggregation weighting : Phase17C sample-count weights")
    else:
        print("Server aggregation    : Coordinate-wise Trimmed Mean")
        print("Defense active        : YES")
        print("Aggregation weighting : Unweighted robust coordinate mean")
    print(f"Trim count            : {args.trim_count}")
    print(f"Values retained       : {retained_count}/{phase10.N_CLIENTS}")
    print("Feature set           : Frozen 70 features")
    print("Client data modified  : NO")
    print("Validation data       : CLEAN")
    print("Locked test used      : NO")

    runs: list[dict[str, Any]] = []

    for experiment in experiments:
        run = run_trimmed_mean_experiment(
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

        save_phase17d_experiment(
            run=run,
            seed=args.seed,
        )

        runs.append(run)

    clean_run = next(
        run
        for run in runs
        if run["experiment_id"] == "D0_TRIMMED_MEAN_CLEAN"
    )

    clean_overall = clean_run["validation"]["overall"]

    phase17c_frame = load_phase17c_fedavg_results(seed=args.seed)

    phase17c_clean_row = None
    if phase17c_frame is not None:
        phase17c_clean_row = find_phase17c_row(
            phase17c_frame=phase17c_frame,
            attack_type="NONE",
            scale=None,
        )

    comparison_rows: list[dict[str, Any]] = []
    control_differences: list[float] = []
    control_pass = True

    print()
    print("=" * 100)
    print("PHASE 17D RESULTS")
    print("=" * 100)

    for run in runs:
        overall = run["validation"]["overall"]
        attack_spec = run["attack_spec"]
        defense_spec = run["defense_spec"]

        if attack_spec["attack_type"] == "NONE":
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
            float(
                np.mean(
                    [
                        event["honest_update_l2_norm"]
                        for event in malicious_events
                    ]
                )
            )
            if malicious_events
            else 0.0
        )

        mean_routed_norm = (
            float(
                np.mean(
                    [
                        event["routed_update_l2_norm"]
                        for event in malicious_events
                    ]
                )
            )
            if malicious_events
            else 0.0
        )

        max_sign_flip_norm_error = (
            float(
                max(
                    event["norm_error"]
                    for event in malicious_events
                )
            )
            if malicious_events
            else 0.0
        )

        max_trim_audit_error = float(
            max(
                audit["max_envelope_violation"]
                for audit in run["aggregation_audit"]
            )
        )

        # -------------------------------------------------------------
        # Optional direct comparison to Phase 17C FedAvg.
        # -------------------------------------------------------------
        fedavg_row = None

        if phase17c_frame is not None:
            fedavg_row = find_phase17c_row(
                phase17c_frame=phase17c_frame,
                attack_type=attack_spec["attack_type"],
                scale=attack_spec["sign_flip_scale"],
            )

        fedavg_accuracy = None
        fedavg_balanced_accuracy = None
        fedavg_macro_f1 = None
        fedavg_weighted_f1 = None
        fedavg_worst_attack_recall = None

        accuracy_recovery = None
        balanced_accuracy_recovery = None
        macro_f1_recovery = None
        weighted_f1_recovery = None
        worst_attack_recall_recovery = None
        macro_f1_recovery_fraction = None
        macro_f1_recovery_pct = None

        if fedavg_row is not None:
            fedavg_accuracy = float(fedavg_row["accuracy"])
            fedavg_balanced_accuracy = float(
                fedavg_row["balanced_accuracy"]
            )
            fedavg_macro_f1 = float(fedavg_row["macro_f1"])
            fedavg_weighted_f1 = float(fedavg_row["weighted_f1"])
            fedavg_worst_attack_recall = float(
                fedavg_row["worst_attack_recall"]
            )

            accuracy_recovery = float(
                overall["accuracy"] - fedavg_accuracy
            )
            balanced_accuracy_recovery = float(
                overall["balanced_accuracy"]
                - fedavg_balanced_accuracy
            )
            macro_f1_recovery = float(
                overall["macro_f1"] - fedavg_macro_f1
            )
            weighted_f1_recovery = float(
                overall["weighted_f1"] - fedavg_weighted_f1
            )
            worst_attack_recall_recovery = float(
                overall["worst_attack_recall"]
                - fedavg_worst_attack_recall
            )

            if args.trim_count == 0:
                # This is a reproduction control, not a defense. Track the
                # absolute differences from the Phase17C FedAvg evidence.
                current_diffs = [
                    abs(accuracy_recovery),
                    abs(balanced_accuracy_recovery),
                    abs(macro_f1_recovery),
                    abs(weighted_f1_recovery),
                    abs(worst_attack_recall_recovery),
                ]
                control_differences.extend(current_diffs)
                if max(current_diffs) > args.control_tolerance:
                    control_pass = False
            elif (
                attack_spec["attack_type"] == "SIGN_FLIP"
                and phase17c_clean_row is not None
            ):
                macro_f1_recovery_fraction = safe_recovery_fraction(
                    clean_value=float(phase17c_clean_row["macro_f1"]),
                    attacked_value=fedavg_macro_f1,
                    defended_value=float(overall["macro_f1"]),
                )

                if macro_f1_recovery_fraction is not None:
                    macro_f1_recovery_pct = float(
                        100.0 * macro_f1_recovery_fraction
                    )

        print()
        print(run["experiment_id"])
        print(f"  Attack type          : {attack_spec['attack_type']}")
        print(f"  Malicious client     : {attack_spec['malicious_client']}")
        print(f"  Sign-flip scale      : {attack_spec['sign_flip_scale']}")
        print(f"  Trim count           : {defense_spec['trim_count']}")
        print(f"  Accuracy             : {overall['accuracy']:.6f}")
        print(f"  Balanced Accuracy    : {overall['balanced_accuracy']:.6f}")
        print(f"  Macro Precision      : {overall['macro_precision']:.6f}")
        print(f"  Macro Recall         : {overall['macro_recall']:.6f}")
        print(f"  Macro F1             : {overall['macro_f1']:.6f}")
        print(f"  Weighted F1          : {overall['weighted_f1']:.6f}")
        print(f"  Worst attack Recall  : {overall['worst_attack_recall']:.6f}")
        print(f"  Macro F1 drop        : {drops['macro_f1_drop']:.6f}")
        print(f"  Trim audit max error : {max_trim_audit_error:.12f}")

        if malicious_events:
            print(f"  Mean honest upd norm : {mean_honest_norm:.6f}")
            print(f"  Mean poison upd norm : {mean_routed_norm:.6f}")
            print(f"  Sign-flip norm error : {max_sign_flip_norm_error:.12f}")

        if fedavg_row is not None:
            print(f"  Phase17C FedAvg F1   : {fedavg_macro_f1:.6f}")
            if args.trim_count == 0:
                print(f"  Macro F1 control diff: {macro_f1_recovery:+.9f}")
                print(
                    "  Control status       : "
                    + (
                        "PASS"
                        if abs(macro_f1_recovery) <= args.control_tolerance
                        else "FAIL"
                    )
                )
            else:
                print(f"  Macro F1 recovery    : {macro_f1_recovery:+.6f}")
                if macro_f1_recovery_pct is not None:
                    print(
                        f"  Damage recovered     : "
                        f"{macro_f1_recovery_pct:.2f}%"
                    )

        comparison_rows.append(
            {
                "experiment_id": run["experiment_id"],
                "seed": int(args.seed),
                "scenario": args.scenario,
                "attack_type": attack_spec["attack_type"],
                "malicious_client": attack_spec["malicious_client"],
                "sign_flip_scale": attack_spec["sign_flip_scale"],
                "server_aggregation": defense_spec["aggregation"],
                "defense_active": bool(defense_spec["defense_active"]),
                "trim_count": int(defense_spec["trim_count"]),
                "values_retained": int(defense_spec["values_retained"]),
                "accuracy": float(overall["accuracy"]),
                "balanced_accuracy": float(overall["balanced_accuracy"]),
                "macro_precision": float(overall["macro_precision"]),
                "macro_recall": float(overall["macro_recall"]),
                "macro_f1": float(overall["macro_f1"]),
                "weighted_f1": float(overall["weighted_f1"]),
                "worst_class_recall": float(overall["worst_class_recall"]),
                "worst_attack_recall": float(
                    overall["worst_attack_recall"]
                ),
                "benign_false_positive_rate": float(
                    overall["benign_false_positive_rate"]
                ),
                "training_time_seconds": float(
                    overall.get("training_time_seconds", 0.0)
                ),
                "mean_malicious_honest_update_norm": mean_honest_norm,
                "mean_malicious_routed_update_norm": mean_routed_norm,
                "max_sign_flip_norm_error": max_sign_flip_norm_error,
                "max_trimmed_mean_envelope_violation": max_trim_audit_error,
                **{
                    key: float(value)
                    for key, value in drops.items()
                },
                "phase17c_fedavg_accuracy": fedavg_accuracy,
                "phase17c_fedavg_balanced_accuracy": (
                    fedavg_balanced_accuracy
                ),
                "phase17c_fedavg_macro_f1": fedavg_macro_f1,
                "phase17c_fedavg_weighted_f1": fedavg_weighted_f1,
                "phase17c_fedavg_worst_attack_recall": (
                    fedavg_worst_attack_recall
                ),
                "accuracy_recovery_vs_fedavg": accuracy_recovery,
                "balanced_accuracy_recovery_vs_fedavg": (
                    balanced_accuracy_recovery
                ),
                "macro_f1_recovery_vs_fedavg": macro_f1_recovery,
                "weighted_f1_recovery_vs_fedavg": weighted_f1_recovery,
                "worst_attack_recall_recovery_vs_fedavg": (
                    worst_attack_recall_recovery
                ),
                "macro_f1_damage_recovery_fraction": (
                    macro_f1_recovery_fraction
                ),
                "macro_f1_damage_recovery_pct": (
                    macro_f1_recovery_pct
                ),
                "is_fedavg_reproduction_control": bool(args.trim_count == 0),
                "control_tolerance": float(args.control_tolerance),
                "validation_poisoned": False,
                "locked_test_used": False,
            }
        )

    summary_dir = RESULT_ROOT / f"seed_{args.seed}" / f"trim_{args.trim_count}"
    summary_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = (
        summary_dir
        / "phase17d_trimmed_mean_defense_comparison.csv"
    )

    pd.DataFrame(comparison_rows).to_csv(
        comparison_path,
        index=False,
    )

    max_attack_norm_error = max(
        (
            float(row["max_sign_flip_norm_error"])
            for row in comparison_rows
        ),
        default=0.0,
    )

    max_trim_error = max(
        (
            float(row["max_trimmed_mean_envelope_violation"])
            for row in comparison_rows
        ),
        default=0.0,
    )

    manifest = {
        "phase": "17D",
        "status": (
            "FAIL"
            if args.trim_count == 0
            and phase17c_frame is not None
            and not control_pass
            else "PASS"
        ),
        "scenario": args.scenario,
        "seed": int(args.seed),
        "rounds": int(args.rounds),
        "local_epochs": int(args.local_epochs),
        "num_clients": int(phase10.N_CLIENTS),
        "malicious_client": int(args.malicious_client),
        "sign_flip_scales": [float(x) for x in args.scales],
        "local_training": {
            "algorithm": "FedProx",
            "mu": float(args.mu),
            "frozen_from_phase17c": True,
        },
        "attack": {
            "type": "Byzantine model-update sign flip",
            "formula": (
                "delta_poisoned = -scale * "
                "(local_state - global_state)"
            ),
            "location": (
                "after local FedProx training, before server aggregation"
            ),
        },
        "defense": {
            "aggregation": aggregation_mode_name(args.trim_count),
            "defense_active": bool(args.trim_count > 0),
            "trim_count": int(args.trim_count),
            "values_retained": int(retained_count),
            "total_client_values": int(phase10.N_CLIENTS),
            "weighted": bool(args.trim_count == 0),
            "control_mode_note": (
                "trim_count=0 reproduces Phase17C sample-weighted FedAvg"
                if args.trim_count == 0
                else None
            ),
        },
        "fedavg_reproduction_control": {
            "active": bool(args.trim_count == 0),
            "tolerance": float(args.control_tolerance),
            "max_metric_abs_difference": (
                float(max(control_differences))
                if control_differences
                else None
            ),
            "passed": (
                bool(control_pass)
                if args.trim_count == 0 and phase17c_frame is not None
                else None
            ),
        },
        "original_client_data_modified": False,
        "validation_poisoned": False,
        "locked_test_used": False,
        "feature_set_frozen": True,
        "client_partitions_frozen": True,
        "model_architecture_frozen": True,
        "source_hash_audit": source_audit,
        "phase17c_fedavg_comparison_available": bool(
            phase17c_frame is not None
        ),
        "max_sign_flip_norm_error": float(max_attack_norm_error),
        "max_trimmed_mean_envelope_violation": float(max_trim_error),
        "experiments": [
            {
                "experiment_id": run["experiment_id"],
                "attack_spec": run["attack_spec"],
                "defense_spec": run["defense_spec"],
                "final_state_sha256": run["final_state_sha256"],
            }
            for run in runs
        ],
        "next": (
            "Run/verify trim_count=0 reproduction control, then run "
            "trim_count=1 defense for seed 42. Only after seed 42 is "
            "validated should seeds 52, 62, 72 and 82 be repeated."
        ),
    }

    manifest_path = (
        ARTIFACT_ROOT
        / f"seed_{args.seed}"
        / f"trim_{args.trim_count}"
        / "phase17d_manifest.json"
    )

    save_json(manifest_path, manifest)

    print()
    print("=" * 78)
    print("PHASE 17D COMPLETE")
    print("=" * 78)
    print(f"Experiments completed   : {len(runs)}")
    print("Local training           : FedProx (unchanged)")
    if args.trim_count == 0:
        print("Server mode              : Phase17C sample-weighted FedAvg CONTROL")
        print("Server defense           : NONE")
    else:
        print("Server mode              : Coordinate-wise Trimmed Mean")
        print("Server defense           : Trimmed Mean")
    print(f"Trim count               : {args.trim_count}")
    print(f"Values retained          : {retained_count}/{phase10.N_CLIENTS}")
    print("Original data modified  : NO")
    print("Validation poisoned      : NO")
    print("Locked test used         : NO")
    print(f"Max attack audit error   : {max_attack_norm_error:.12g}")
    print(f"Max trim audit error     : {max_trim_error:.12g}")
    print(f"Phase17C comparison      : {'YES' if phase17c_frame is not None else 'NO'}")
    print(f"Comparison               : {comparison_path}")
    print(f"Manifest                 : {manifest_path}")

    if args.trim_count == 0:
        if phase17c_frame is None:
            print("CONTROL STATUS            : NOT CHECKED (Phase17C comparison missing)")
            print("STATUS                    : PASS WITH WARNING")
        else:
            max_control_diff = (
                max(control_differences)
                if control_differences
                else float("inf")
            )
            print(f"Control tolerance         : {args.control_tolerance:.3e}")
            print(f"Max control metric diff   : {max_control_diff:.12g}")
            print(f"CONTROL STATUS            : {'PASS' if control_pass else 'FAIL'}")
            print(f"STATUS                    : {'PASS' if control_pass else 'FAIL'}")
    else:
        print("STATUS                    : PASS")

    print()
    print("NEXT:")
    if args.trim_count == 0:
        print("1. The trim=0 control must PASS before interpreting Trimmed Mean.")
        print("2. Then rerun seed 42 with --trim-count 1.")
    else:
        print("1. Inspect seed 42 defense recovery against Phase17C FedAvg.")
        print("2. If scientifically valid, repeat seeds 52, 62, 72 and 82.")

    if args.trim_count == 0 and phase17c_frame is not None and not control_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
