from __future__ import annotations

import argparse
import importlib
import random
import sys
from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# ============================================================
# CONFIG
# ============================================================

SEED = 42

TARGET_CLASSES = [
    "DoS Hulk",
    "DDoS",
    "PortScan",
    "DoS GoldenEye",
    "FTP-Patator",
]

ROWS_PER_CLASS = 200

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "dashboard_samples"
    / "ids_5_attacks_1000.csv"
)

CHUNK_SIZE = 100_000


# ============================================================
# HELPERS
# ============================================================

def normalize_text(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


TARGET_LOOKUP = {
    normalize_text(name): name
    for name in TARGET_CLASSES
}


def find_label_column(columns) -> str:
    lookup = {
        str(column).strip().lower(): column
        for column in columns
    }

    candidates = [
        "label",
        "class",
        "attack",
        "attack_type",
        "attack type",
    ]

    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]

    raise ValueError(
        "\nCould not find a label column.\n"
        "Expected one of: Label, class, attack, attack_type."
    )


def try_phase14_validation_file() -> Path | None:
    """
    First try to reuse the exact Phase-14 validation source file
    if the existing project module exposes its path.
    """
    module_names = [
        "phase14d0_build_ordered_sequence_dataset_v2",
        "phase14d_build_ordered_sequence_dataset",
        "phase14h_full_validation_adaptive_gate",
    ]

    candidate_attributes = [
        "VALIDATION_FILE",
        "VALIDATION_CSV",
        "VALIDATION_SOURCE_FILE",
        "SOURCE_VALIDATION_FILE",
        "VALIDATION_INPUT_FILE",
    ]

    for module_name in module_names:
        try:
            module = importlib.import_module(
                module_name
            )
        except Exception:
            continue

        for attribute in candidate_attributes:
            value = getattr(
                module,
                attribute,
                None,
            )

            if value is None:
                continue

            path = Path(
                value
            )

            if (
                path.exists()
                and
                path.is_file()
                and
                path.suffix.lower()
                == ".csv"
            ):
                return path

    return None


def discover_validation_csv() -> Path:
    """
    Conservative fallback discovery.

    We only accept CSV files that look like validation files.
    We reject obvious train/test/client files so this generator
    cannot silently sample from the wrong split.
    """
    data_root = (
        PROJECT_ROOT
        / "data"
    )

    if not data_root.exists():
        raise FileNotFoundError(
            f"\nData directory not found:\n{data_root}"
        )

    candidates = []

    for path in data_root.rglob(
        "*.csv"
    ):
        lowered = str(
            path
        ).lower()

        name = path.name.lower()

        if "validation" not in lowered and "val" not in name:
            continue

        if any(
            token in lowered
            for token in [
                "locked_test",
                "locked-test",
                "\\test\\",
                "/test/",
                "train",
                "client_1",
                "client_2",
                "client_3",
                "client_4",
                "client_5",
                "dashboard_samples",
            ]
        ):
            continue

        candidates.append(
            path
        )

    if not candidates:
        raise FileNotFoundError(
            "\nCould not automatically locate a validation CSV.\n\n"
            "Run again with the exact validation CSV path:\n"
            "python scripts\\create_ids_5_attacks_1000.py "
            "--input \"data\\...\\your_validation.csv\""
        )

    # Prefer source-ordered / Phase14-looking files.
    candidates.sort(
        key=lambda p: (
            0
            if (
                "source" in str(p).lower()
                and
                "order" in str(p).lower()
            )
            else 1,
            len(str(p)),
        )
    )

    if len(candidates) > 1:
        print(
            "\nValidation CSV candidates found:"
        )

        for index, path in enumerate(
            candidates,
            start=1,
        ):
            print(
                f"  {index}. {path}"
            )

        print(
            "\nUsing the highest-priority candidate:"
        )

    return candidates[0]


def resolve_input_file(
    explicit_input: str | None,
) -> Path:
    if explicit_input:
        path = Path(
            explicit_input
        )

        if not path.is_absolute():
            path = (
                PROJECT_ROOT
                /
                path
            )

        path = path.resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"\nInput file not found:\n{path}"
            )

        if path.suffix.lower() != ".csv":
            raise ValueError(
                "\nInput must be a CSV file."
            )

        return path

    phase14_path = (
        try_phase14_validation_file()
    )

    if phase14_path is not None:
        return phase14_path

    return discover_validation_csv()


# ============================================================
# RESERVOIR SAMPLING
# ============================================================

def sample_five_attacks(
    input_file: Path,
    output_file: Path,
):
    rng = random.Random(
        SEED
    )

    reservoirs: dict[str, list[dict]] = {
        class_name: []
        for class_name in TARGET_CLASSES
    }

    seen_counts: dict[str, int] = {
        class_name: 0
        for class_name in TARGET_CLASSES
    }

    label_column = None
    output_columns = None

    print(
        f"\nReading validation source:\n{input_file}"
    )

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            input_file,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ),
        start=1,
    ):
        # Remove accidental whitespace around column names.
        chunk.columns = [
            str(column).strip()
            for column in chunk.columns
        ]

        if label_column is None:
            label_column = find_label_column(
                chunk.columns
            )

            output_columns = list(
                chunk.columns
            )

            print(
                f"Label column       : {label_column}"
            )

        normalized_labels = (
            chunk[
                label_column
            ]
            .astype(str)
            .map(
                normalize_text
            )
        )

        target_mask = (
            normalized_labels
            .isin(
                TARGET_LOOKUP.keys()
            )
        )

        filtered = chunk.loc[
            target_mask
        ].copy()

        if filtered.empty:
            continue

        filtered[
            "__canonical_label__"
        ] = (
            normalized_labels.loc[
                filtered.index
            ]
            .map(
                TARGET_LOOKUP
            )
        )

        for _, row in filtered.iterrows():
            class_name = row[
                "__canonical_label__"
            ]

            seen_counts[
                class_name
            ] += 1

            # Preserve original columns exactly, but normalize
            # the Label value to the project's canonical class name.
            record = {
                column: row[
                    column
                ]
                for column in output_columns
            }

            record[
                label_column
            ] = class_name

            reservoir = reservoirs[
                class_name
            ]

            if len(
                reservoir
            ) < ROWS_PER_CLASS:
                reservoir.append(
                    record
                )
            else:
                # Standard reservoir sampling:
                # every matching validation row has equal probability.
                replacement_index = rng.randrange(
                    seen_counts[
                        class_name
                    ]
                )

                if replacement_index < ROWS_PER_CLASS:
                    reservoir[
                        replacement_index
                    ] = record

        if chunk_number % 5 == 0:
            print(
                f"Processed chunks   : {chunk_number}"
            )

    print()
    print(
        "AVAILABLE VALIDATION ROWS"
    )
    print(
        "-" * 55
    )

    for class_name in TARGET_CLASSES:
        print(
            f"{class_name:20s}: "
            f"{seen_counts[class_name]:,}"
        )

    insufficient = [
        class_name
        for class_name in TARGET_CLASSES
        if len(
            reservoirs[
                class_name
            ]
        ) < ROWS_PER_CLASS
    ]

    if insufficient:
        details = "\n".join(
            f"  {name}: "
            f"{len(reservoirs[name])}/"
            f"{ROWS_PER_CLASS}"
            for name in insufficient
        )

        raise RuntimeError(
            "\nThe validation source does not contain enough rows "
            "for one or more target classes:\n"
            f"{details}"
        )

    # Combine exactly 200 from each attack family.
    rows = []

    for class_name in TARGET_CLASSES:
        rows.extend(
            reservoirs[
                class_name
            ]
        )

    result = pd.DataFrame(
        rows,
        columns=output_columns,
    )

    # Shuffle final order deterministically so the dashboard does not
    # receive 200 contiguous rows from each attack class.
    result = (
        result.sample(
            frac=1.0,
            random_state=SEED,
        )
        .reset_index(
            drop=True
        )
    )

    # Scientific assertions.
    if len(
        result
    ) != 1000:
        raise RuntimeError(
            f"Expected 1000 output rows, found {len(result)}."
        )

    final_counts = (
        result[
            label_column
        ]
        .value_counts()
        .to_dict()
    )

    for class_name in TARGET_CLASSES:
        if int(
            final_counts.get(
                class_name,
                0,
            )
        ) != ROWS_PER_CLASS:
            raise RuntimeError(
                f"Unexpected count for {class_name}: "
                f"{final_counts.get(class_name, 0)}"
            )

    if (
        result[
            label_column
        ]
        .astype(str)
        .str.strip()
        .str.upper()
        .eq(
            "BENIGN"
        )
        .any()
    ):
        raise RuntimeError(
            "BENIGN unexpectedly appeared in attack-only dataset."
        )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output_file,
        index=False,
    )

    print()
    print(
        "=" * 70
    )

    print(
        "5-ATTACK DASHBOARD SAMPLE CREATED"
    )

    print(
        "=" * 70
    )

    print(
        f"Output             : {output_file}"
    )

    print(
        f"Rows               : {len(result):,}"
    )

    print(
        f"Columns            : {len(result.columns)}"
    )

    print(
        f"BENIGN rows        : 0"
    )

    print()
    print(
        "FINAL CLASS DISTRIBUTION"
    )
    print(
        "-" * 55
    )

    for class_name in TARGET_CLASSES:
        print(
            f"{class_name:20s}: "
            f"{final_counts[class_name]:,}"
        )

    print()
    print(
        "Source split       : VALIDATION"
    )

    print(
        "Locked test used   : NO"
    )

    print(
        "Training rows used : NO"
    )

    print(
        "Random seed        : 42"
    )


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a 1000-row validation-only dashboard CSV "
            "containing 200 rows from each of the five attack classes."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "Optional exact validation CSV path. "
            "If omitted, the script tries the Phase-14 validation "
            "source path and then conservative auto-discovery."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(
            OUTPUT_FILE
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    input_file = resolve_input_file(
        args.input
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

    sample_five_attacks(
        input_file=input_file,
        output_file=output_file,
    )
