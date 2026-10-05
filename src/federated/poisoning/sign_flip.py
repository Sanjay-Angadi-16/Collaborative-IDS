from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Tuple

import torch


@dataclass
class SignFlipAudit:
    strength: float
    floating_tensors_modified: int
    non_floating_tensors_preserved: int
    clean_update_l2: float
    poisoned_update_l2: float
    expected_l2_ratio: float
    actual_l2_ratio: float
    max_verification_error: float
    verification_passed: bool


def _clone_state(
    state: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in state.items()
    }


def _update_l2_norm(
    *,
    global_state: Mapping[str, torch.Tensor],
    local_state: Mapping[str, torch.Tensor],
) -> float:
    total_sq = 0.0

    for key in global_state:
        global_tensor = global_state[key].detach().cpu()
        local_tensor = local_state[key].detach().cpu()

        if not torch.is_floating_point(local_tensor):
            continue

        delta = (
            local_tensor.to(torch.float64)
            - global_tensor.to(torch.float64)
        )

        total_sq += float(
            torch.sum(delta * delta).item()
        )

    return total_sq ** 0.5


def sign_flip_model_update(
    *,
    global_state: Mapping[str, torch.Tensor],
    local_state: Mapping[str, torch.Tensor],
    strength: float = 1.0,
    verification_tolerance: float = 1e-6,
) -> Tuple[
    Dict[str, torch.Tensor],
    SignFlipAudit,
]:
    """
    Controlled Phase 17C sign-flip attack.

    Normal client update:

        delta = local_state - global_state

    Poisoned update:

        poisoned_delta = -strength * delta

    Reconstructed poisoned client state:

        poisoned_state =
            global_state + poisoned_delta

    Equivalent:

        poisoned_state =
            global_state
            - strength * (local_state - global_state)

    Example:
        strength = 1.0

        clean delta:
            +0.20

        malicious delta:
            -0.20
    """

    strength = float(strength)

    if strength <= 0.0:
        raise ValueError(
            "Sign-flip strength must be > 0."
        )

    global_keys = set(global_state.keys())
    local_keys = set(local_state.keys())

    if global_keys != local_keys:
        missing_global = sorted(
            local_keys - global_keys
        )
        missing_local = sorted(
            global_keys - local_keys
        )

        raise ValueError(
            "State-dict keys do not match. "
            f"Missing global={missing_global}, "
            f"missing local={missing_local}"
        )

    attacked: Dict[str, torch.Tensor] = {}

    floating_modified = 0
    non_floating_preserved = 0
    max_verification_error = 0.0

    for key in global_state:
        global_tensor = (
            global_state[key]
            .detach()
            .cpu()
        )

        local_tensor = (
            local_state[key]
            .detach()
            .cpu()
        )

        if global_tensor.shape != local_tensor.shape:
            raise ValueError(
                f"Shape mismatch for {key}: "
                f"global={tuple(global_tensor.shape)}, "
                f"local={tuple(local_tensor.shape)}"
            )

        # --------------------------------------------------
        # Only floating-point model parameters/buffers are
        # sign-flipped.
        #
        # Integer buffers such as BatchNorm
        # num_batches_tracked must remain valid.
        # --------------------------------------------------
        if not torch.is_floating_point(local_tensor):
            attacked[key] = local_tensor.clone()
            non_floating_preserved += 1
            continue

        # Use float calculation in original dtype so that
        # aggregation receives compatible tensors.
        clean_delta = (
            local_tensor - global_tensor
        )

        poisoned_delta = (
            -strength * clean_delta
        )

        poisoned_tensor = (
            global_tensor + poisoned_delta
        )

        attacked[key] = poisoned_tensor.clone()

        floating_modified += 1

        # --------------------------------------------------
        # Verify:
        #
        # poisoned_delta ≈ -strength * clean_delta
        # --------------------------------------------------
        actual_delta = (
            attacked[key] - global_tensor
        )

        expected_delta = (
            -strength * clean_delta
        )

        error = torch.max(
            torch.abs(
                actual_delta.to(torch.float64)
                - expected_delta.to(torch.float64)
            )
        ).item()

        max_verification_error = max(
            max_verification_error,
            float(error),
        )

    clean_l2 = _update_l2_norm(
        global_state=global_state,
        local_state=local_state,
    )

    poisoned_l2 = _update_l2_norm(
        global_state=global_state,
        local_state=attacked,
    )

    if clean_l2 > 0.0:
        actual_ratio = (
            poisoned_l2 / clean_l2
        )
    else:
        actual_ratio = 0.0

    audit = SignFlipAudit(
        strength=strength,
        floating_tensors_modified=(
            floating_modified
        ),
        non_floating_tensors_preserved=(
            non_floating_preserved
        ),
        clean_update_l2=clean_l2,
        poisoned_update_l2=poisoned_l2,
        expected_l2_ratio=strength,
        actual_l2_ratio=actual_ratio,
        max_verification_error=(
            max_verification_error
        ),
        verification_passed=(
            max_verification_error
            <= verification_tolerance
        ),
    )

    if not audit.verification_passed:
        raise RuntimeError(
            "Sign-flip verification failed: "
            f"max_error="
            f"{audit.max_verification_error:.12f}"
        )

    return attacked, audit


def audit_to_dict(
    audit: SignFlipAudit,
) -> dict:
    return asdict(audit)