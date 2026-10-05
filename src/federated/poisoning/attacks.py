from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import torch

from src.federated.poisoning.sign_flip import (
    sign_flip_model_update,
)


# Keep this unchanged if Phase17A harness checks this value.
PHASE17A_ATTACK_API_VERSION = "phase17a_attack_api_v1"


@dataclass(frozen=True)
class AttackSpec:
    """
    Attack specification used by the Phase 17 poisoning framework.

    Supported attacks:

    Phase 17A
    ----------
    NONE
        No attack.

    NOOP
        Client is declared malicious, but its model state is not changed.

    UPDATE_SCALE
        Phase17A identity control only.

        update_scale must remain exactly 1.0.

        Actual damaging model scaling is reserved for Phase17D.

    Phase 17C
    ----------
    SIGN_FLIP
        Model-update sign-flipping attack.

        clean update:
            delta = local_state - global_state

        poisoned update:
            poisoned_delta = -attack_strength * delta

        poisoned state:
            attacked_state =
                global_state + poisoned_delta
    """

    attack_type: str

    malicious_clients: Tuple[int, ...]

    # ---------------------------------------------------------
    # Phase 17A / future Phase 17D
    # ---------------------------------------------------------
    update_scale: float = 1.0

    # ---------------------------------------------------------
    # Phase 17C
    # ---------------------------------------------------------
    attack_strength: float = 1.0

    def validate(self) -> "AttackSpec":
        """
        Validate attack configuration before it is applied.
        """

        allowed = {
            "NONE",
            "NOOP",
            "UPDATE_SCALE",
            "SIGN_FLIP",
        }

        if self.attack_type not in allowed:
            raise ValueError(
                f"Unsupported attack_type="
                f"{self.attack_type!r}. "
                f"Allowed={sorted(allowed)}"
            )

        # -----------------------------------------------------
        # Client ids are expected to be:
        #
        # client_1 -> 1
        # client_2 -> 2
        # ...
        # -----------------------------------------------------
        if any(
            client_id < 1
            for client_id
            in self.malicious_clients
        ):
            raise ValueError(
                "Client ids must be >= 1."
            )

        # -----------------------------------------------------
        # Phase 17A identity control
        # -----------------------------------------------------
        if self.attack_type == "UPDATE_SCALE":

            if float(self.update_scale) != 1.0:
                raise ValueError(
                    "Phase17A allows UPDATE_SCALE only "
                    "with scale=1.0. "
                    "Damaging model scaling is "
                    "reserved for Phase17D."
                )

        # -----------------------------------------------------
        # Phase 17C sign-flip attack
        # -----------------------------------------------------
        if self.attack_type == "SIGN_FLIP":

            if float(
                self.attack_strength
            ) <= 0.0:
                raise ValueError(
                    "SIGN_FLIP attack_strength "
                    "must be > 0."
                )

        return self


# =============================================================
# STATE UTILITIES
# =============================================================


def clone_state(
    state: Mapping[
        str,
        torch.Tensor,
    ],
) -> Dict[
    str,
    torch.Tensor,
]:
    """
    Create an independent CPU clone of a model state dictionary.

    This prevents an attack implementation from accidentally
    modifying the original local/global state in-place.
    """

    return {
        key: (
            value
            .detach()
            .cpu()
            .clone()
        )
        for key, value
        in state.items()
    }


def state_max_abs_diff(
    left: Mapping[
        str,
        torch.Tensor,
    ],
    right: Mapping[
        str,
        torch.Tensor,
    ],
) -> float:
    """
    Return the maximum absolute tensor difference between
    two state dictionaries.

    For non-floating tensors, exact equality is required.
    """

    maximum = 0.0

    if set(
        left.keys()
    ) != set(
        right.keys()
    ):
        raise ValueError(
            "State dictionaries have different keys."
        )

    for key in left:

        a = (
            left[key]
            .detach()
            .cpu()
        )

        b = (
            right[key]
            .detach()
            .cpu()
        )

        if a.shape != b.shape:
            raise ValueError(
                f"Shape mismatch for {key}: "
                f"{a.shape} vs {b.shape}"
            )

        # -----------------------------------------------------
        # Integer / boolean buffers require exact equality.
        # -----------------------------------------------------
        if not torch.is_floating_point(a):

            if not torch.equal(
                a,
                b,
            ):
                return float(
                    "inf"
                )

            continue

        diff = torch.max(
            torch.abs(
                a.to(
                    torch.float64
                )
                -
                b.to(
                    torch.float64
                )
            )
        ).item()

        maximum = max(
            maximum,
            float(diff),
        )

    return maximum


# =============================================================
# PHASE 17A
# UPDATE SCALE IDENTITY CONTROL
# =============================================================


def scale_model_update(
    *,
    global_state: Mapping[
        str,
        torch.Tensor,
    ],
    local_state: Mapping[
        str,
        torch.Tensor,
    ],
    scale: float,
) -> Tuple[
    Dict[
        str,
        torch.Tensor,
    ],
    float,
]:
    """
    Update-space scaling.

    Normal update:

        delta =
            local_state - global_state

    Scaled update:

        scaled_delta =
            scale * delta

    Reconstructed state:

        attacked_state =
            global_state + scaled_delta


    Phase17A uses scale=1.0 only as an identity control.

    Instead of calculating:

        global + 1.0 * (local - global)

    we return an exact clone of local_state.

    This avoids floating-point round-off that could otherwise
    accumulate across federated rounds.
    """

    scale = float(
        scale
    )

    # ---------------------------------------------------------
    # IMPORTANT:
    # Exact Phase17A identity path.
    # ---------------------------------------------------------
    if scale == 1.0:

        return (
            clone_state(
                local_state
            ),
            0.0,
        )

    attacked: Dict[
        str,
        torch.Tensor,
    ] = {}

    for key in local_state:

        local_tensor = (
            local_state[
                key
            ]
            .detach()
            .cpu()
        )

        global_tensor = (
            global_state[
                key
            ]
            .detach()
            .cpu()
        )

        if (
            local_tensor.shape
            !=
            global_tensor.shape
        ):
            raise ValueError(
                f"State shape mismatch "
                f"for key={key}"
            )

        # -----------------------------------------------------
        # Only floating-point tensors are mathematically
        # transformed.
        #
        # Integer buffers are preserved.
        # -----------------------------------------------------
        if torch.is_floating_point(
            local_tensor
        ):

            clean_delta = (
                local_tensor
                -
                global_tensor
            )

            attacked[
                key
            ] = (
                global_tensor
                +
                scale
                *
                clean_delta
            ).clone()

        else:

            attacked[
                key
            ] = (
                local_tensor
                .clone()
            )

    identity_error = (
        state_max_abs_diff(
            attacked,
            local_state,
        )
    )

    return (
        attacked,
        identity_error,
    )


# =============================================================
# MAIN ATTACK ROUTER
# =============================================================


def apply_client_attack(
    *,
    global_state: Mapping[
        str,
        torch.Tensor,
    ],
    local_state: Mapping[
        str,
        torch.Tensor,
    ],
    client_id: int,
    spec: AttackSpec,
    round_number: int,
) -> Tuple[
    Dict[
        str,
        torch.Tensor,
    ],
    Dict[
        str,
        Any,
    ],
]:
    """
    Route every local client state through one explicit
    attack boundary.

    ---------------------------------------------------------
    HONEST CLIENT
    ---------------------------------------------------------

        local_state
            |
            v
        unchanged clone


    ---------------------------------------------------------
    PHASE 17A NOOP
    ---------------------------------------------------------

        malicious client
            |
            v
        unchanged clone


    ---------------------------------------------------------
    PHASE 17A UPDATE_SCALE(1.0)
    ---------------------------------------------------------

        malicious client
            |
            v
        exact identity clone


    ---------------------------------------------------------
    PHASE 17C SIGN_FLIP
    ---------------------------------------------------------

        clean delta =
            local_state - global_state

        poisoned delta =
            -attack_strength * clean_delta

        attacked state =
            global_state + poisoned_delta
    """

    spec.validate()

    client_id = int(
        client_id
    )

    round_number = int(
        round_number
    )

    malicious_client_set = set(
        int(client)
        for client
        in spec.malicious_clients
    )

    is_malicious = (
        client_id
        in
        malicious_client_set
    )

    # =========================================================
    # COMMON ATTACK AUDIT EVENT
    # =========================================================

    event: Dict[
        str,
        Any,
    ] = {

        "round": (
            round_number
        ),

        "client_id": (
            client_id
        ),

        "declared_malicious": bool(
            is_malicious
        ),

        "attack_type": (
            spec.attack_type
            if is_malicious
            else "HONEST"
        ),

        # -----------------------------------------------------
        # Phase17A / Phase17D field
        # -----------------------------------------------------
        "update_scale": (
            float(
                spec.update_scale
            )
            if (
                is_malicious
                and
                spec.attack_type
                ==
                "UPDATE_SCALE"
            )
            else None
        ),

        # -----------------------------------------------------
        # Phase17C field
        # -----------------------------------------------------
        "attack_strength": (
            float(
                spec.attack_strength
            )
            if (
                is_malicious
                and
                spec.attack_type
                ==
                "SIGN_FLIP"
            )
            else None
        ),

        # -----------------------------------------------------
        # Phase17A identity audit
        # -----------------------------------------------------
        "identity_error": 0.0,

        # -----------------------------------------------------
        # General state modification indicator
        # -----------------------------------------------------
        "state_modified": False,

        # -----------------------------------------------------
        # Phase17C sign-flip audit
        # -----------------------------------------------------
        "verification_passed": None,

        "verification_error": None,

        "clean_update_l2": None,

        "poisoned_update_l2": None,

        "update_l2_ratio": None,
    }

    # =========================================================
    # HONEST CLIENT
    # =========================================================

    if not is_malicious:

        return (
            clone_state(
                local_state
            ),
            event,
        )

    # =========================================================
    # PHASE 17A
    # NONE / NOOP
    # =========================================================

    if spec.attack_type in {
        "NONE",
        "NOOP",
    }:

        return (
            clone_state(
                local_state
            ),
            event,
        )

    # =========================================================
    # PHASE 17A
    # UPDATE SCALE IDENTITY CONTROL
    # =========================================================

    if (
        spec.attack_type
        ==
        "UPDATE_SCALE"
    ):

        (
            attacked_state,
            identity_error,
        ) = scale_model_update(
            global_state=(
                global_state
            ),
            local_state=(
                local_state
            ),
            scale=(
                spec.update_scale
            ),
        )

        event[
            "identity_error"
        ] = float(
            identity_error
        )

        event[
            "state_modified"
        ] = bool(
            identity_error
            >
            0.0
        )

        return (
            attacked_state,
            event,
        )

    # =========================================================
    # PHASE 17C
    # SIGN-FLIP / MODEL-UPDATE POISONING
    # =========================================================

    if (
        spec.attack_type
        ==
        "SIGN_FLIP"
    ):

        (
            attacked_state,
            sign_flip_audit,
        ) = sign_flip_model_update(
            global_state=(
                global_state
            ),
            local_state=(
                local_state
            ),
            strength=(
                spec.attack_strength
            ),
        )

        # -----------------------------------------------------
        # Compare attacked client state with original
        # local client state.
        # -----------------------------------------------------
        attacked_vs_local_diff = (
            state_max_abs_diff(
                attacked_state,
                local_state,
            )
        )

        event[
            "attack_strength"
        ] = float(
            spec.attack_strength
        )

        event[
            "state_modified"
        ] = bool(
            attacked_vs_local_diff
            >
            0.0
        )

        event[
            "verification_passed"
        ] = bool(
            sign_flip_audit
            .verification_passed
        )

        event[
            "verification_error"
        ] = float(
            sign_flip_audit
            .max_verification_error
        )

        event[
            "clean_update_l2"
        ] = float(
            sign_flip_audit
            .clean_update_l2
        )

        event[
            "poisoned_update_l2"
        ] = float(
            sign_flip_audit
            .poisoned_update_l2
        )

        event[
            "update_l2_ratio"
        ] = float(
            sign_flip_audit
            .actual_l2_ratio
        )

        return (
            attacked_state,
            event,
        )

    # =========================================================
    # SAFETY CHECK
    # =========================================================

    raise RuntimeError(
        "Unreachable attack branch."
    )