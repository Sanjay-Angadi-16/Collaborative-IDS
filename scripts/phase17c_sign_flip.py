"""
phase17c_sign_flip.py
---------------------

Phase 17C - Sign-Flip Poisoning Attack

Purpose:
    Simulate malicious federated-learning clients that reverse the
    direction of their local model updates before sending them to
    the aggregation server.

Normal client:
    update = local_model - global_model

Malicious client:
    malicious_update = -scale * update

Then:
    poisoned_model = global_model + malicious_update

Example:
    Honest update:
        +0.20

    Sign-flipped update:
        -0.20

    If scale = 5:
        -1.00

This file does NOT modify:
    - Training dataset
    - Locked test dataset
    - Labels
    - Global model directly
    - Honest client updates
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import numpy as np
import torch


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_SEED = 42

DEFAULT_NUM_CLIENTS = 5

DEFAULT_NUM_MALICIOUS_CLIENTS = 1

DEFAULT_SIGN_FLIP_SCALE = 1.0

DEFAULT_OUTPUT_DIR = Path(
    "results/phase17c_sign_flip"
)


# ============================================================
# LOGGING
# ============================================================

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("phase17c_sign_flip")

    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )

        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger


logger = setup_logger()


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int) -> None:
    """
    Set all common random seeds.

    This makes malicious-client selection reproducible.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info("Random seed set to %d", seed)


# ============================================================
# MALICIOUS CLIENT SELECTION
# ============================================================

def select_malicious_clients(
    num_clients: int,
    num_malicious_clients: int,
    seed: int = DEFAULT_SEED,
    explicit_client_ids: Optional[Iterable[int]] = None,
) -> Set[int]:
    """
    Select malicious clients.

    Client IDs are zero based internally.

    Example:
        num_clients = 5
        malicious_clients = 1

        Possible result:
            {0}

    Parameters
    ----------
    num_clients:
        Total number of federated clients.

    num_malicious_clients:
        Number of clients participating in the attack.

    seed:
        Seed controlling random client selection.

    explicit_client_ids:
        Optional fixed malicious client IDs.

        Example:
            [0, 2]

        means:
            client_1 and client_3
    """

    if num_clients <= 0:
        raise ValueError("num_clients must be greater than 0.")

    if explicit_client_ids is not None:
        malicious = set(explicit_client_ids)

        for client_id in malicious:
            if client_id < 0 or client_id >= num_clients:
                raise ValueError(
                    f"Invalid malicious client ID: {client_id}. "
                    f"Valid range = 0 to {num_clients - 1}"
                )

        return malicious

    if num_malicious_clients < 0:
        raise ValueError(
            "num_malicious_clients cannot be negative."
        )

    if num_malicious_clients > num_clients:
        raise ValueError(
            "num_malicious_clients cannot exceed num_clients."
        )

    rng = random.Random(seed)

    selected = rng.sample(
        range(num_clients),
        num_malicious_clients,
    )

    return set(selected)


# ============================================================
# MODEL UPDATE CALCULATION
# ============================================================

def calculate_model_update(
    global_state: Dict[str, torch.Tensor],
    local_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Calculate:

        delta = local_model - global_model

    Only floating-point tensors are treated as trainable updates.

    Integer buffers such as BatchNorm counters are preserved.
    """

    update: Dict[str, torch.Tensor] = {}

    for key in global_state:

        global_tensor = global_state[key]
        local_tensor = local_state[key]

        if torch.is_floating_point(global_tensor):

            update[key] = (
                local_tensor.detach().clone()
                - global_tensor.detach().clone()
            )

        else:
            # Integer/non-floating tensors should not be sign flipped.
            update[key] = local_tensor.detach().clone()

    return update


# ============================================================
# SIGN-FLIP ATTACK
# ============================================================

def sign_flip_update(
    update: Dict[str, torch.Tensor],
    scale: float = DEFAULT_SIGN_FLIP_SCALE,
) -> Dict[str, torch.Tensor]:
    """
    Reverse the direction of a model update.

    Honest update:

        delta

    Malicious update:

        -scale * delta

    Example:

        delta = +0.25
        scale = 1

        poisoned = -0.25

    Stronger attack:

        delta = +0.25
        scale = 5

        poisoned = -1.25
    """

    if scale < 0:
        raise ValueError(
            "scale must be >= 0. "
            "The function already applies the negative sign."
        )

    poisoned_update: Dict[str, torch.Tensor] = {}

    for key, tensor in update.items():

        if torch.is_floating_point(tensor):

            poisoned_update[key] = (
                -scale * tensor.detach().clone()
            )

        else:
            poisoned_update[key] = tensor.detach().clone()

    return poisoned_update


# ============================================================
# CONVERT UPDATE BACK TO MODEL
# ============================================================

def apply_update_to_global_model(
    global_state: Dict[str, torch.Tensor],
    update: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Reconstruct client model from:

        client_model = global_model + update
    """

    poisoned_state: Dict[str, torch.Tensor] = {}

    for key in global_state:

        global_tensor = global_state[key]

        if torch.is_floating_point(global_tensor):

            poisoned_state[key] = (
                global_tensor.detach().clone()
                + update[key].detach().clone()
            )

        else:
            poisoned_state[key] = update[key].detach().clone()

    return poisoned_state


# ============================================================
# COMPLETE CLIENT ATTACK
# ============================================================

def poison_client_model_sign_flip(
    global_state: Dict[str, torch.Tensor],
    local_state: Dict[str, torch.Tensor],
    scale: float = DEFAULT_SIGN_FLIP_SCALE,
) -> Dict[str, torch.Tensor]:
    """
    Complete sign-flip pipeline.

    Step 1:
        Calculate honest local update.

            delta = local - global

    Step 2:
        Reverse the update.

            poisoned_delta = -scale * delta

    Step 3:
        Construct malicious client model.

            poisoned_local =
                global + poisoned_delta

    Returns
    -------
    Poisoned client state_dict.
    """

    update = calculate_model_update(
        global_state=global_state,
        local_state=local_state,
    )

    poisoned_update = sign_flip_update(
        update=update,
        scale=scale,
    )

    poisoned_state = apply_update_to_global_model(
        global_state=global_state,
        update=poisoned_update,
    )

    return poisoned_state


# ============================================================
# UPDATE-NORM CALCULATION
# ============================================================

def calculate_update_l2_norm(
    global_state: Dict[str, torch.Tensor],
    local_state: Dict[str, torch.Tensor],
) -> float:
    """
    Calculate the L2 norm of a client's model update.

    Useful for analysing whether malicious updates have
    unusually large magnitudes.
    """

    squared_sum = 0.0

    for key in global_state:

        if not torch.is_floating_point(global_state[key]):
            continue

        delta = (
            local_state[key].detach().float()
            - global_state[key].detach().float()
        )

        squared_sum += torch.sum(
            delta * delta
        ).item()

    return float(np.sqrt(squared_sum))


# ============================================================
# APPLY ATTACK TO MULTIPLE CLIENTS
# ============================================================

def apply_sign_flip_attack(
    global_state: Dict[str, torch.Tensor],
    client_states: List[Dict[str, torch.Tensor]],
    malicious_clients: Set[int],
    scale: float = DEFAULT_SIGN_FLIP_SCALE,
) -> List[Dict[str, torch.Tensor]]:
    """
    Apply sign-flip poisoning only to malicious clients.

    Honest clients remain completely unchanged.
    """

    attacked_states = []

    logger.info("=" * 70)
    logger.info("PHASE 17C - SIGN-FLIP ATTACK")
    logger.info("=" * 70)

    logger.info(
        "Total clients      : %d",
        len(client_states),
    )

    logger.info(
        "Malicious clients  : %s",
        [
            f"client_{client_id + 1}"
            for client_id in sorted(malicious_clients)
        ],
    )

    logger.info(
        "Sign-flip scale    : %.4f",
        scale,
    )

    for client_id, local_state in enumerate(client_states):

        before_norm = calculate_update_l2_norm(
            global_state,
            local_state,
        )

        if client_id in malicious_clients:

            logger.info(
                "client_%d -> MALICIOUS",
                client_id + 1,
            )

            poisoned_state = poison_client_model_sign_flip(
                global_state=global_state,
                local_state=local_state,
                scale=scale,
            )

            after_norm = calculate_update_l2_norm(
                global_state,
                poisoned_state,
            )

            logger.info(
                "client_%d | update norm before = %.6f | "
                "after = %.6f",
                client_id + 1,
                before_norm,
                after_norm,
            )

            attacked_states.append(
                poisoned_state
            )

        else:

            logger.info(
                "client_%d -> HONEST | update norm = %.6f",
                client_id + 1,
                before_norm,
            )

            attacked_states.append(
                copy.deepcopy(local_state)
            )

    return attacked_states


# ============================================================
# ATTACK MANIFEST
# ============================================================

def save_attack_manifest(
    output_dir: Path,
    seed: int,
    num_clients: int,
    malicious_clients: Set[int],
    scale: float,
) -> None:
    """
    Save attack configuration for research reproducibility.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        "phase": "17C",
        "attack": "sign_flip",
        "seed": seed,
        "num_clients": num_clients,
        "num_malicious_clients": len(malicious_clients),

        "malicious_client_ids_zero_based":
            sorted(malicious_clients),

        "malicious_client_names": [
            f"client_{client_id + 1}"
            for client_id in sorted(malicious_clients)
        ],

        "sign_flip_scale": scale,

        "formula": (
            "poisoned_update = "
            "-scale * (local_model - global_model)"
        ),

        "data_modified": False,
        "labels_modified": False,
        "locked_test_modified": False,
    }

    path = (
        output_dir
        / "sign_flip_attack_manifest.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=4,
        )

    logger.info(
        "Attack manifest saved: %s",
        path,
    )


# ============================================================
# SIMPLE DEMONSTRATION
# ============================================================

def demo_sign_flip(
    scale: float = 1.0,
) -> None:
    """
    Small demonstration showing exactly what the attack does.
    """

    global_state = {
        "weight": torch.tensor(
            [1.0, 2.0, 3.0]
        )
    }

    local_state = {
        "weight": torch.tensor(
            [1.2, 1.8, 3.5]
        )
    }

    poisoned_state = poison_client_model_sign_flip(
        global_state=global_state,
        local_state=local_state,
        scale=scale,
    )

    honest_update = (
        local_state["weight"]
        - global_state["weight"]
    )

    malicious_update = (
        poisoned_state["weight"]
        - global_state["weight"]
    )

    print("\nSIGN-FLIP EXAMPLE")
    print("=" * 60)

    print(
        "Global model      :",
        global_state["weight"].tolist(),
    )

    print(
        "Local model       :",
        local_state["weight"].tolist(),
    )

    print(
        "Honest update     :",
        honest_update.tolist(),
    )

    print(
        "Malicious update  :",
        malicious_update.tolist(),
    )

    print(
        "Poisoned model    :",
        poisoned_state["weight"].tolist(),
    )

    print("=" * 60)


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Phase 17C - Federated Learning "
            "Sign-Flip Poisoning Attack"
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed",
    )

    parser.add_argument(
        "--num-clients",
        type=int,
        default=DEFAULT_NUM_CLIENTS,
        help="Total federated clients",
    )

    parser.add_argument(
        "--num-malicious",
        type=int,
        default=DEFAULT_NUM_MALICIOUS_CLIENTS,
        help="Number of malicious clients",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_SIGN_FLIP_SCALE,
        help=(
            "Sign-flip magnitude. "
            "1.0 reverses the update with equal magnitude."
        ),
    )

    parser.add_argument(
        "--malicious-client",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional fixed zero-based malicious client IDs. "
            "Example: --malicious-client 0 2"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory",
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run simple sign-flip demonstration",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    args = parse_args()

    set_seed(args.seed)

    malicious_clients = select_malicious_clients(
        num_clients=args.num_clients,
        num_malicious_clients=args.num_malicious,
        seed=args.seed,
        explicit_client_ids=args.malicious_client,
    )

    logger.info("=" * 70)
    logger.info("PHASE 17C CONFIGURATION")
    logger.info("=" * 70)

    logger.info(
        "Seed              : %d",
        args.seed,
    )

    logger.info(
        "Total clients     : %d",
        args.num_clients,
    )

    logger.info(
        "Malicious clients : %d",
        len(malicious_clients),
    )

    logger.info(
        "Selected clients  : %s",
        [
            f"client_{client_id + 1}"
            for client_id in sorted(malicious_clients)
        ],
    )

    logger.info(
        "Sign-flip scale   : %.4f",
        args.scale,
    )

    output_dir = Path(args.output_dir)

    save_attack_manifest(
        output_dir=output_dir,
        seed=args.seed,
        num_clients=args.num_clients,
        malicious_clients=malicious_clients,
        scale=args.scale,
    )

    if args.demo:
        demo_sign_flip(
            scale=args.scale
        )

    logger.info("=" * 70)
    logger.info(
        "Phase 17C configuration completed."
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()