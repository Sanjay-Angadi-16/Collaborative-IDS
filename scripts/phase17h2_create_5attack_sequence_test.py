from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# ============================================================
# PROJECT MODULES
# ============================================================

try:
    import phase14g1_real_model_wiring as phase14g1
    import phase14h_full_validation_adaptive_gate as phase14h
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "\nPlace this script inside the project's scripts folder.\n\n"
        "Required existing files:\n"
        "  scripts\\phase14g1_real_model_wiring.py\n"
        "  scripts\\phase14h_full_validation_adaptive_gate.py\n"
    ) from exc


# ============================================================
# CONFIG
# ============================================================

PHASE_NAME = (
    "PHASE 17H.2 — CREATE 5-ATTACK SEQUENCE TEST"
)

SEED = 42

NUM_CLASSES = 6
EXPECTED_FEATURES = 70
SEQUENCE_LENGTH = 20

ROWS_PER_ATTACK = 200

TARGET_ATTACKS = [
    "DoS Hulk",
    "DDoS",
    "PortScan",
    "DoS GoldenEye",
    "FTP-Patator",
]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "dashboard_samples"
)

OUTPUT_NPZ = (
    OUTPUT_DIR
    / "ids_5_attacks_1000_sequences.npz"
)

OUTPUT_MANIFEST = (
    OUTPUT_DIR
    / "ids_5_attacks_1000_sequences_manifest.json"
)


# ============================================================
# HELPERS
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def normalize_name(value: str) -> str:
    return (
        str(
            value
        )
        .strip()
        .lower()
        .replace(
            "_",
            " ",
        )
        .replace(
            "-",
            " ",
        )
    )


def json_safe(value):
    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        np.integer,
    ):
        return int(
            value
        )

    if isinstance(
        value,
        np.floating,
    ):
        return float(
            value
        )

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(
            value
        )

    if isinstance(
        value,
        dict,
    ):
        return {
            str(
                key
            ):
            json_safe(
                item
            )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            json_safe(
                item
            )
            for item
            in value
        ]

    return value


def separator(
    width: int = 120,
) -> None:
    print()
    print(
        "="
        *
        width
    )


def load_validation_sequences():
    validation_file = (
        phase14h.VALIDATION_SEQUENCE_FILE
    )

    if not validation_file.exists():
        raise FileNotFoundError(
            "\nMissing Phase-14 validation sequence file:\n"
            f"{validation_file}"
        )

    with np.load(
        validation_file,
        allow_pickle=False,
    ) as data:

        if "X" not in data.files:
            raise KeyError(
                "Validation NPZ does not contain array 'X'."
            )

        if "y" not in data.files:
            raise KeyError(
                "Validation NPZ does not contain array 'y'."
            )

        X = data[
            "X"
        ].astype(
            np.float32,
            copy=False,
        )

        y = data[
            "y"
        ].astype(
            np.int64,
            copy=False,
        )

    if X.ndim != 3:
        raise RuntimeError(
            f"Expected X shape (N,20,70); found {X.shape}."
        )

    if tuple(
        X.shape[
            1:
        ]
    ) != (
        SEQUENCE_LENGTH,
        EXPECTED_FEATURES,
    ):
        raise RuntimeError(
            "Expected validation sequence shape "
            f"(N,{SEQUENCE_LENGTH},{EXPECTED_FEATURES}); "
            f"found {X.shape}."
        )

    if len(
        X
    ) != len(
        y
    ):
        raise RuntimeError(
            "Validation X/y count mismatch."
        )

    return (
        X,
        y,
        validation_file,
    )


def resolve_target_class_ids(
    inverse_mapping: dict[int, str],
) -> dict[str, int]:

    normalized_to_id = {
        normalize_name(
            class_name
        ):
        int(
            class_id
        )
        for class_id, class_name
        in inverse_mapping.items()
    }

    resolved = {}

    for attack_name in TARGET_ATTACKS:
        normalized = normalize_name(
            attack_name
        )

        if normalized not in normalized_to_id:
            raise RuntimeError(
                "\nCould not map required attack class:\n"
                f"  {attack_name}\n\n"
                "Available classes:\n"
                +
                "\n".join(
                    f"  {class_id}: {class_name}"
                    for class_id, class_name
                    in sorted(
                        inverse_mapping.items()
                    )
                )
            )

        class_id = normalized_to_id[
            normalized
        ]

        if class_id == 0:
            raise RuntimeError(
                f"{attack_name} resolved to BENIGN class ID 0."
            )

        resolved[
            attack_name
        ] = class_id

    return resolved


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a validation-only, sequence-preserving, "
            "5-attack balanced test set with exactly 1000 sequences."
        )
    )

    parser.add_argument(
        "--rows-per-attack",
        type=int,
        default=
            ROWS_PER_ATTACK,
        help=(
            "Sequences per attack class. "
            "Default 200 => total 1000 sequences."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=
            SEED,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=
            str(
                OUTPUT_NPZ
            ),
    )

    args = parser.parse_args()

    if args.rows_per_attack <= 0:
        raise ValueError(
            "--rows-per-attack must be > 0."
        )

    set_seed(
        args.seed
    )

    output_file = Path(
        args.output
    )

    if not output_file.is_absolute():
        output_file = (
            PROJECT_ROOT
            /
            output_file
        )

    output_file = output_file.resolve()

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        label_mapping,
        inverse_mapping,
        active_features,
    ) = phase14g1.load_project_schema()

    if len(
        active_features
    ) != EXPECTED_FEATURES:
        raise RuntimeError(
            f"Expected 70 active features; found {len(active_features)}."
        )

    (
        X_all,
        y_all,
        validation_file,
    ) = load_validation_sequences()

    target_ids = resolve_target_class_ids(
        inverse_mapping
    )

    separator()

    print(
        PHASE_NAME
    )

    separator()

    print(
        f"Validation source       : {validation_file}"
    )

    print(
        f"Available sequences     : {len(y_all):,}"
    )

    print(
        f"Sequence shape          : {X_all.shape}"
    )

    print(
        f"Rows per attack         : {args.rows_per_attack:,}"
    )

    print(
        f"Target total            : "
        f"{args.rows_per_attack * len(TARGET_ATTACKS):,}"
    )

    print(
        "BENIGN included         : NO"
    )

    print(
        "Locked test used        : NO"
    )

    print(
        "Training rows used      : NO"
    )

    print(
        "Sequence order modified : NO"
    )

    print()

    print(
        "AVAILABLE VALIDATION SEQUENCES"
    )

    print(
        "-" * 120
    )

    for class_id in range(
        NUM_CLASSES
    ):
        count = int(
            np.sum(
                y_all
                ==
                class_id
            )
        )

        print(
            f"{class_id} | "
            f"{inverse_mapping[class_id]:20s} | "
            f"{count:,}"
        )

    rng = np.random.default_rng(
        args.seed
    )

    selected_indices = []
    selected_class_ids = []
    selected_class_names = []
    specialist_client_ids = []

    # Each attack's specialist client in the frozen non-IID design.
    specialist_lookup = {
        "DoS Hulk":
            1,
        "DDoS":
            2,
        "PortScan":
            3,
        "DoS GoldenEye":
            4,
        "FTP-Patator":
            5,
    }

    print()

    print(
        "SAMPLING"
    )

    print(
        "-" * 120
    )

    for attack_name in TARGET_ATTACKS:
        class_id = target_ids[
            attack_name
        ]

        candidates = np.flatnonzero(
            y_all
            ==
            class_id
        )

        if len(
            candidates
        ) < args.rows_per_attack:
            raise RuntimeError(
                f"\nNot enough validation sequences for {attack_name}.\n"
                f"Required : {args.rows_per_attack}\n"
                f"Available: {len(candidates)}"
            )

        sampled = rng.choice(
            candidates,
            size=
                args.rows_per_attack,
            replace=False,
        )

        # Sort the selected source indices for reproducible source-order
        # preservation within each class block.
        sampled = np.asarray(
            sorted(
                sampled.tolist()
            ),
            dtype=np.int64,
        )

        selected_indices.extend(
            sampled.tolist()
        )

        selected_class_ids.extend(
            [
                class_id
            ]
            *
            len(
                sampled
            )
        )

        selected_class_names.extend(
            [
                attack_name
            ]
            *
            len(
                sampled
            )
        )

        specialist_client_ids.extend(
            [
                specialist_lookup[
                    attack_name
                ]
            ]
            *
            len(
                sampled
            )
        )

        print(
            f"{attack_name:20s} | "
            f"class_id={class_id} | "
            f"selected={len(sampled):,}"
        )

    selected_indices = np.asarray(
        selected_indices,
        dtype=np.int64,
    )

    selected_class_ids = np.asarray(
        selected_class_ids,
        dtype=np.int64,
    )

    specialist_client_ids = np.asarray(
        specialist_client_ids,
        dtype=np.int64,
    )

    # Extract full 20x70 sequences exactly as they exist in validation.
    X_selected = X_all[
        selected_indices
    ]

    y_selected = y_all[
        selected_indices
    ]

    # Scientific consistency checks.
    expected_total = (
        args.rows_per_attack
        *
        len(
            TARGET_ATTACKS
        )
    )

    if len(
        X_selected
    ) != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} sequences; found {len(X_selected)}."
        )

    if np.any(
        y_selected
        ==
        0
    ):
        raise RuntimeError(
            "BENIGN unexpectedly appeared in the 5-attack dataset."
        )

    if not np.array_equal(
        y_selected,
        selected_class_ids,
    ):
        raise RuntimeError(
            "Internal label alignment check failed."
        )

    if tuple(
        X_selected.shape[
            1:
        ]
    ) != (
        SEQUENCE_LENGTH,
        EXPECTED_FEATURES,
    ):
        raise RuntimeError(
            "Selected sequence shape changed unexpectedly."
        )

    unique_counts = {
        attack_name:
            int(
                np.sum(
                    y_selected
                    ==
                    target_ids[
                        attack_name
                    ]
                )
            )
        for attack_name in TARGET_ATTACKS
    }

    for attack_name, count in unique_counts.items():
        if count != args.rows_per_attack:
            raise RuntimeError(
                f"Unexpected count for {attack_name}: {count}"
            )

    # Save without shuffling sequence internals.
    #
    # X                -> shape (1000,20,70)
    # y                -> true known-class labels
    # validation_index -> original row in validation_sequences.npz
    # specialist_client_id -> client that owns that attack family
    #
    # This lets the next dashboard test either:
    #   A) round-robin client exposure, or
    #   B) correct specialist-client routing.
    np.savez_compressed(
        output_file,
        X=
            X_selected.astype(
                np.float32,
                copy=False,
            ),

        y=
            y_selected.astype(
                np.int64,
                copy=False,
            ),

        validation_index=
            selected_indices.astype(
                np.int64,
                copy=False,
            ),

        specialist_client_id=
            specialist_client_ids.astype(
                np.int64,
                copy=False,
            ),
    )

    manifest_file = (
        output_file.parent
        /
        (
            output_file.stem
            +
            "_manifest.json"
        )
    )

    manifest = {
        "phase":
            "17H.2",

        "purpose":
            (
                "Balanced five-attack validation-only "
                "sequence-preserving dashboard stress test."
            ),

        "source_validation_file":
            str(
                validation_file
            ),

        "output_file":
            str(
                output_file
            ),

        "seed":
            int(
                args.seed
            ),

        "sequence_count":
            int(
                len(
                    X_selected
                )
            ),

        "sequence_shape":
            list(
                X_selected.shape
            ),

        "sequence_length":
            SEQUENCE_LENGTH,

        "feature_count":
            EXPECTED_FEATURES,

        "rows_per_attack":
            int(
                args.rows_per_attack
            ),

        "contains_benign":
            False,

        "locked_test_used":
            False,

        "training_rows_used":
            False,

        "sequence_internals_shuffled":
            False,

        "classes":
            {
                attack_name:
                    {
                        "class_id":
                            int(
                                target_ids[
                                    attack_name
                                ]
                            ),

                        "count":
                            int(
                                unique_counts[
                                    attack_name
                                ]
                            ),

                        "specialist_client_id":
                            int(
                                specialist_lookup[
                                    attack_name
                                ]
                            ),
                    }
                for attack_name
                in TARGET_ATTACKS
            },

        "arrays":
            {
                "X":
                    "Full source-ordered 20x70 validation sequences.",

                "y":
                    "Known-class ground-truth IDs.",

                "validation_index":
                    (
                        "Original sequence index in "
                        "validation_sequences.npz."
                    ),

                "specialist_client_id":
                    (
                        "Client associated with that attack family "
                        "under the frozen severe non-IID design."
                    ),
            },
    }

    with open(
        manifest_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_safe(
                manifest
            ),
            file,
            indent=2,
        )

    separator()

    print(
        "PHASE 17H.2 DATASET CREATED"
    )

    separator()

    print(
        f"Output NPZ             : {output_file}"
    )

    print(
        f"Manifest               : {manifest_file}"
    )

    print(
        f"Final shape            : {X_selected.shape}"
    )

    print(
        f"Total sequences        : {len(X_selected):,}"
    )

    print(
        f"Features/flow          : {X_selected.shape[2]}"
    )

    print(
        f"Flows/sequence         : {X_selected.shape[1]}"
    )

    print()

    print(
        "FINAL CLASS DISTRIBUTION"
    )

    print(
        "-" * 120
    )

    for attack_name in TARGET_ATTACKS:
        print(
            f"{attack_name:20s} | "
            f"{unique_counts[attack_name]:,} | "
            f"specialist client "
            f"{specialist_lookup[attack_name]}"
        )

    print()

    print(
        "BENIGN rows/sequences : 0"
    )

    print(
        "Locked test used      : NO"
    )

    print(
        "Training rows used    : NO"
    )

    print(
        "Sequence internals    : PRESERVED"
    )

    print()

    print(
        "NEXT STEP"
    )

    print(
        "-" * 120
    )

    print(
        "Update the dashboard to accept this NPZ directly."
    )

    print(
        "RF / AE / FedProx70 should use X[:, -1, :]."
    )

    print(
        "CNN-BiLSTM should use the full X with shape (N,20,70)."
    )

    print(
        "Use y only for evaluation metrics, never for prediction."
    )


if __name__ == "__main__":
    main()
