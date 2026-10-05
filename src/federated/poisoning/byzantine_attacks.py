from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Mapping, Tuple

import torch


PHASE17DB_ATTACK_API_VERSION = "phase17db_attack_api_v1"


@dataclass(frozen=True)
class ByzantineAttackSpec:
    """
    Controlled update-level attacks for Phase17D-B.

    NONE:
        Honest control.

    SCALE:
        Delta_attack = magnitude * Delta_local

    RANDOM:
        Replace the legitimate update direction with a deterministic
        Gaussian random direction whose total L2 norm is:

            ||Delta_attack||_2
                =
            magnitude * ||Delta_local||_2

    All transformations operate only on the designated malicious client's
    model update. Honest clients are unchanged.
    """

    attack_type: str
    malicious_clients: Tuple[int, ...]
    magnitude: float = 1.0
    attack_seed: int = 1704

    def validate(self) -> "ByzantineAttackSpec":
        allowed = {
            "NONE",
            "SCALE",
            "RANDOM",
        }

        if self.attack_type not in allowed:
            raise ValueError(
                f"Unsupported attack_type={self.attack_type!r}. "
                f"Allowed={sorted(allowed)}"
            )

        if any(
            int(client_id) < 1
            for client_id
            in self.malicious_clients
        ):
            raise ValueError(
                "Client IDs must be >= 1."
            )

        if self.attack_type == "SCALE":
            if float(self.magnitude) <= 1.0:
                raise ValueError(
                    "Damaging SCALE attack requires magnitude > 1. "
                    "Phase17A already preserved scale=1 as the identity control."
                )

        if self.attack_type == "RANDOM":
            if float(self.magnitude) <= 0.0:
                raise ValueError(
                    "RANDOM magnitude must be > 0."
                )

        if self.attack_type == "NONE":
            if self.malicious_clients:
                raise ValueError(
                    "NONE control must have no malicious clients."
                )

        return self


def clone_state(
    state: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value
        in state.items()
    }


def _floating_update_tensors(
    *,
    global_state: Mapping[str, torch.Tensor],
    local_state: Mapping[str, torch.Tensor],
):
    if set(global_state.keys()) != set(local_state.keys()):
        raise ValueError(
            "Global/local state dictionaries have different keys."
        )

    for key in local_state:
        local = local_state[key].detach().cpu()
        global_tensor = global_state[key].detach().cpu()

        if local.shape != global_tensor.shape:
            raise ValueError(
                f"State shape mismatch for {key}: "
                f"{local.shape} vs {global_tensor.shape}"
            )

        if torch.is_floating_point(local):
            yield (
                key,
                global_tensor,
                local,
                local.to(torch.float64)
                -
                global_tensor.to(torch.float64),
            )


def update_statistics(
    *,
    global_state: Mapping[str, torch.Tensor],
    local_state: Mapping[str, torch.Tensor],
) -> Dict[str, float]:
    sum_sq = 0.0
    max_abs = 0.0
    parameter_count = 0

    for _, _, _, delta in _floating_update_tensors(
        global_state=global_state,
        local_state=local_state,
    ):
        sum_sq += float(
            torch.sum(
                delta * delta
            ).item()
        )

        if delta.numel():
            max_abs = max(
                max_abs,
                float(
                    torch.max(
                        torch.abs(
                            delta
                        )
                    ).item()
                ),
            )

        parameter_count += int(
            delta.numel()
        )

    return {
        "l2_norm": float(
            math.sqrt(
                max(
                    sum_sq,
                    0.0,
                )
            )
        ),
        "max_abs": float(
            max_abs
        ),
        "floating_parameter_count": int(
            parameter_count
        ),
    }


def update_cosine_similarity(
    *,
    global_state: Mapping[str, torch.Tensor],
    left_state: Mapping[str, torch.Tensor],
    right_state: Mapping[str, torch.Tensor],
) -> float:
    dot = 0.0
    left_sq = 0.0
    right_sq = 0.0

    for key, global_tensor, left_tensor, left_delta in (
        _floating_update_tensors(
            global_state=global_state,
            local_state=left_state,
        )
    ):
        right_tensor = (
            right_state[
                key
            ]
            .detach()
            .cpu()
            .to(
                torch.float64
            )
        )

        right_delta = (
            right_tensor
            -
            global_tensor.to(
                torch.float64
            )
        )

        dot += float(
            torch.sum(
                left_delta
                *
                right_delta
            ).item()
        )

        left_sq += float(
            torch.sum(
                left_delta
                *
                left_delta
            ).item()
        )

        right_sq += float(
            torch.sum(
                right_delta
                *
                right_delta
            ).item()
        )

    denominator = math.sqrt(
        max(
            left_sq,
            0.0,
        )
        *
        max(
            right_sq,
            0.0,
        )
    )

    if denominator <= 0.0:
        return 0.0

    return float(
        dot
        /
        denominator
    )


def scale_update(
    *,
    global_state: Mapping[str, torch.Tensor],
    local_state: Mapping[str, torch.Tensor],
    magnitude: float,
) -> Dict[str, torch.Tensor]:
    """
    Positive update scaling:

        Delta = W_local - W_global

        W_attack
            =
        W_global + magnitude * Delta

    The direction is preserved; only the magnitude increases.
    """
    attacked: Dict[
        str,
        torch.Tensor,
    ] = {}

    for key in local_state:
        local = (
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

        if torch.is_floating_point(
            local
        ):
            attacked[
                key
            ] = (
                global_tensor
                +
                float(
                    magnitude
                )
                *
                (
                    local
                    -
                    global_tensor
                )
            ).clone()

        else:
            attacked[
                key
            ] = local.clone()

    return attacked


def random_byzantine_update(
    *,
    global_state: Mapping[str, torch.Tensor],
    local_state: Mapping[str, torch.Tensor],
    magnitude: float,
    deterministic_seed: int,
) -> Dict[str, torch.Tensor]:
    """
    Deterministic random Byzantine update.

    1. Measure the legitimate full-model update norm.
    2. Generate a seeded Gaussian direction over all floating tensors.
    3. Normalize that random direction globally.
    4. Scale it to magnitude * legitimate_update_norm.
    5. Add it to the current global model.

    This prevents arbitrary random noise scale from being confounded with
    attack direction: the requested magnitude has a precise meaning.
    """
    legitimate = update_statistics(
        global_state=global_state,
        local_state=local_state,
    )

    legitimate_norm = float(
        legitimate[
            "l2_norm"
        ]
    )

    if legitimate_norm <= 0.0:
        raise RuntimeError(
            "Legitimate update norm is zero; RANDOM attack cannot be normalized."
        )

    generator = torch.Generator(
        device="cpu"
    )
    generator.manual_seed(
        int(
            deterministic_seed
        )
    )

    noise: Dict[
        str,
        torch.Tensor,
    ] = {}

    noise_sum_sq = 0.0

    for key in local_state:
        local = (
            local_state[
                key
            ]
            .detach()
            .cpu()
        )

        if not torch.is_floating_point(
            local
        ):
            continue

        sample = torch.randn(
            local.shape,
            generator=generator,
            dtype=torch.float64,
            device="cpu",
        )

        noise[
            key
        ] = sample

        noise_sum_sq += float(
            torch.sum(
                sample
                *
                sample
            ).item()
        )

    noise_norm = math.sqrt(
        max(
            noise_sum_sq,
            0.0,
        )
    )

    if noise_norm <= 0.0:
        raise RuntimeError(
            "Generated random direction has zero norm."
        )

    target_norm = (
        float(
            magnitude
        )
        *
        legitimate_norm
    )

    normalization = (
        target_norm
        /
        noise_norm
    )

    attacked: Dict[
        str,
        torch.Tensor,
    ] = {}

    for key in local_state:
        local = (
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

        if torch.is_floating_point(
            local
        ):
            delta = (
                noise[
                    key
                ]
                *
                normalization
            )

            attacked[
                key
            ] = (
                global_tensor.to(
                    torch.float64
                )
                +
                delta
            ).to(
                dtype=local.dtype
            ).clone()

        else:
            attacked[
                key
            ] = local.clone()

    return attacked


def apply_byzantine_attack(
    *,
    global_state: Mapping[str, torch.Tensor],
    local_state: Mapping[str, torch.Tensor],
    client_id: int,
    spec: ByzantineAttackSpec,
    round_number: int,
    experiment_seed: int,
) -> Tuple[
    Dict[str, torch.Tensor],
    Dict[str, Any],
]:
    spec.validate()

    is_malicious = (
        int(
            client_id
        )
        in
        set(
            int(
                value
            )
            for value
            in spec.malicious_clients
        )
    )

    legitimate_stats = update_statistics(
        global_state=global_state,
        local_state=local_state,
    )

    event: Dict[
        str,
        Any,
    ] = {
        "round": int(
            round_number
        ),
        "client_id": int(
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
        "requested_magnitude": (
            float(
                spec.magnitude
            )
            if is_malicious
            else None
        ),
        "legitimate_update_l2":
            float(
                legitimate_stats[
                    "l2_norm"
                ]
            ),
        "legitimate_update_max_abs":
            float(
                legitimate_stats[
                    "max_abs"
                ]
            ),
        "attacked_update_l2":
            float(
                legitimate_stats[
                    "l2_norm"
                ]
            ),
        "attacked_update_max_abs":
            float(
                legitimate_stats[
                    "max_abs"
                ]
            ),
        "observed_norm_ratio":
            1.0,
        "norm_ratio_abs_error":
            0.0,
        "update_cosine_similarity":
            1.0,
        "random_seed_used":
            None,
        "state_modified":
            False,
    }

    if not is_malicious:
        return (
            clone_state(
                local_state
            ),
            event,
        )

    if spec.attack_type == "SCALE":
        attacked = scale_update(
            global_state=
                global_state,
            local_state=
                local_state,
            magnitude=
                spec.magnitude,
        )

    elif spec.attack_type == "RANDOM":
        random_seed = (
            int(
                spec.attack_seed
            )
            +
            int(
                experiment_seed
            )
            *
            1_000_003
            +
            int(
                round_number
            )
            *
            10_007
            +
            int(
                client_id
            )
            *
            101
        )

        attacked = random_byzantine_update(
            global_state=
                global_state,
            local_state=
                local_state,
            magnitude=
                spec.magnitude,
            deterministic_seed=
                random_seed,
        )

        event[
            "random_seed_used"
        ] = int(
            random_seed
        )

    else:
        raise ValueError(
            "A malicious client requires SCALE or RANDOM in Phase17D-B."
        )

    attacked_stats = update_statistics(
        global_state=global_state,
        local_state=attacked,
    )

    legitimate_norm = float(
        legitimate_stats[
            "l2_norm"
        ]
    )

    attacked_norm = float(
        attacked_stats[
            "l2_norm"
        ]
    )

    ratio = (
        attacked_norm
        /
        legitimate_norm
        if legitimate_norm
        >
        0.0
        else
        float(
            "inf"
        )
    )

    cosine = update_cosine_similarity(
        global_state=
            global_state,
        left_state=
            local_state,
        right_state=
            attacked,
    )

    event.update(
        {
            "attacked_update_l2":
                attacked_norm,
            "attacked_update_max_abs":
                float(
                    attacked_stats[
                        "max_abs"
                    ]
                ),
            "observed_norm_ratio":
                float(
                    ratio
                ),
            "norm_ratio_abs_error":
                float(
                    abs(
                        ratio
                        -
                        float(
                            spec.magnitude
                        )
                    )
                ),
            "update_cosine_similarity":
                float(
                    cosine
                ),
            "state_modified":
                True,
        }
    )

    return (
        attacked,
        event,
    )
