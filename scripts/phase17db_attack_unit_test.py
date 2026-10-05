from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.federated.poisoning.byzantine_attacks import (
    ByzantineAttackSpec,
    apply_byzantine_attack,
)


def build_states():
    global_state = {
        "a": torch.tensor(
            [
                1.0,
                2.0,
                3.0,
            ],
            dtype=torch.float32,
        ),
        "b": torch.tensor(
            [
                [0.5, -0.5],
                [1.5, -1.5],
            ],
            dtype=torch.float32,
        ),
    }

    local_state = {
        "a": torch.tensor(
            [
                2.0,
                1.0,
                5.0,
            ],
            dtype=torch.float32,
        ),
        "b": torch.tensor(
            [
                [1.5, 0.5],
                [1.0, -2.5],
            ],
            dtype=torch.float32,
        ),
    }

    return (
        global_state,
        local_state,
    )


def main():
    global_state, local_state = build_states()

    scale_spec = ByzantineAttackSpec(
        attack_type="SCALE",
        malicious_clients=(1,),
        magnitude=5.0,
    )

    _, scale_event = apply_byzantine_attack(
        global_state=global_state,
        local_state=local_state,
        client_id=1,
        spec=scale_spec,
        round_number=1,
        experiment_seed=42,
    )

    assert (
        scale_event[
            "norm_ratio_abs_error"
        ]
        <
        1e-6
    )

    assert (
        scale_event[
            "update_cosine_similarity"
        ]
        >
        0.999999
    )

    random_spec = ByzantineAttackSpec(
        attack_type="RANDOM",
        malicious_clients=(1,),
        magnitude=2.0,
        attack_seed=1704,
    )

    state_1, event_1 = apply_byzantine_attack(
        global_state=global_state,
        local_state=local_state,
        client_id=1,
        spec=random_spec,
        round_number=3,
        experiment_seed=42,
    )

    state_2, event_2 = apply_byzantine_attack(
        global_state=global_state,
        local_state=local_state,
        client_id=1,
        spec=random_spec,
        round_number=3,
        experiment_seed=42,
    )

    assert (
        event_1[
            "norm_ratio_abs_error"
        ]
        <
        1e-5
    )

    assert (
        event_1[
            "random_seed_used"
        ]
        ==
        event_2[
            "random_seed_used"
        ]
    )

    for key in state_1:
        assert torch.equal(
            state_1[
                key
            ],
            state_2[
                key
            ],
        )

    print("=" * 88)
    print("PHASE 17D-B ATTACK UNIT TEST")
    print("=" * 88)
    print(
        f"SCALE X5 norm ratio      : "
        f"{scale_event['observed_norm_ratio']:.8f}"
    )
    print(
        f"SCALE X5 cosine          : "
        f"{scale_event['update_cosine_similarity']:.8f}"
    )
    print(
        f"RANDOM X2 norm ratio     : "
        f"{event_1['observed_norm_ratio']:.8f}"
    )
    print(
        f"RANDOM deterministic seed: "
        f"{event_1['random_seed_used']}"
    )
    print("Deterministic replay     : PASS")
    print("STATUS                   : PASS")


if __name__ == "__main__":
    main()
