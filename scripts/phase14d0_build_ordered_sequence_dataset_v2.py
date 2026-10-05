from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# ============================================================
# EXISTING PROJECT CODE
# ============================================================

try:
    import phase9_fedavg70 as phase4
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "\nCould not import scripts\\phase9_fedavg70.py"
    ) from exc


# ============================================================
# PHASE
# ============================================================

PHASE_NAME = (
    "PHASE 14D0 — SOURCE-ORDERED CNN-BiLSTM DATASET V2"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

N_CLIENTS = 5

SEQUENCE_LENGTH = 20

# Raw source blocks are kept intact inside TRAIN or VALIDATION.
# Phase14D will later create 20-flow windows inside these blocks.
BLOCK_SIZE = 2_000

# Approximate validation fraction at the block level.
VALIDATION_BLOCK_FRACTION = 0.20

# At least this many rows must exist in a block to be useful.
MIN_BLOCK_ROWS = SEQUENCE_LENGTH

# We trim this many rows from the beginning of every block after
# the first block of a source segment. This creates a safety gap
# between adjacent source-order blocks.
PURGE_GAP = SEQUENCE_LENGTH - 1


# ============================================================
# SIX PROJECT CLASSES
# ============================================================

PROJECT_CLASSES = [
    "BENIGN",
    "DDoS",
    "DoS GoldenEye",
    "DoS Hulk",
    "FTP-Patator",
    "PortScan",
]


# ============================================================
# NON-IID CLIENT ATTACK MAPPING
# ============================================================

CLIENT_ATTACKS = {
    1: "DoS Hulk",
    2: "DDoS",
    3: "PortScan",
    4: "DoS GoldenEye",
    5: "FTP-Patator",
}

ATTACK_TO_CLIENT = {
    attack: client_id
    for client_id, attack
    in CLIENT_ATTACKS.items()
}


# ============================================================
# PATHS
# ============================================================

RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

TEMPORAL_ROOT = (
    PROJECT_ROOT
    / "data"
    / "temporal_phase14"
)

NON_IID_ROOT = (
    TEMPORAL_ROOT
    / "non_iid"
)

TRAIN_FILE = (
    TEMPORAL_ROOT
    / "train_source_ordered.csv"
)

VALIDATION_FILE = (
    TEMPORAL_ROOT
    / "validation.csv"
)

METADATA_ROOT = (
    TEMPORAL_ROOT
    / "metadata"
)

MANIFEST_FILE = (
    METADATA_ROOT
    / "temporal_dataset_manifest.csv"
)

BLOCK_MANIFEST_FILE = (
    METADATA_ROOT
    / "source_ordered_block_manifest.csv"
)

CLIENT_DISTRIBUTION_FILE = (
    METADATA_ROOT
    / "client_class_distribution.csv"
)

CLASS_DISTRIBUTION_FILE = (
    METADATA_ROOT
    / "train_validation_class_distribution.csv"
)

CONFIG_FILE = (
    METADATA_ROOT
    / "temporal_dataset_config.json"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# DISPLAY
# ============================================================

def separator():
    print()
    print("=" * 112)


# ============================================================
# JSON SAFE
# ============================================================

def json_safe(value):
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item
            in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, Path):
        return str(value)

    return value


# ============================================================
# SAVE JSON
# ============================================================

def save_json(path: Path, payload):
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_safe(payload),
            file,
            indent=4,
        )


# ============================================================
# APPEND CSV
# ============================================================

def append_csv(
    dataframe: pd.DataFrame,
    path: Path,
):
    if dataframe is None or len(dataframe) == 0:
        return

    dataframe.to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_columns(
    dataframe: pd.DataFrame,
):
    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    return dataframe


def normalize_label(value):
    return (
        str(value)
        .strip()
        .replace("\ufeff", "")
    )


# ============================================================
# VERIFY REAL ACTIVE FEATURES
# ============================================================

def verify_and_prepare_feature_columns(
    dataframe: pd.DataFrame,
    active_features,
    feature_columns,
):
    """
    fine_label and family_label are not original CICIDS2017
    ML features. They were introduced in the earlier project
    preprocessing branch.

    Only the final 70 active ML features are mandatory in raw
    CICIDS files.

    Missing inactive compatibility columns are added as 0.0 so
    phase4.transform_features() remains structurally compatible.
    """

    metadata_columns = {
        "fine_label",
        "family_label",
    }

    metadata_in_active = [
        column
        for column in active_features
        if column in metadata_columns
    ]

    if metadata_in_active:
        raise RuntimeError(
            "\nSafety failure: metadata columns are active "
            "model inputs:\n"
            + "\n".join(metadata_in_active)
        )

    missing_active = [
        feature
        for feature in active_features
        if feature not in dataframe.columns
    ]

    if missing_active:
        raise ValueError(
            "\nRaw source is missing ACTIVE ML features:\n"
            + "\n".join(missing_active[:30])
        )

    compatibility_added = []

    for column in (
        "fine_label",
        "family_label",
    ):
        if (
            column in feature_columns
            and column not in dataframe.columns
        ):
            dataframe[column] = 0.0
            compatibility_added.append(column)

    print(
        f"Active ML features found  : "
        f"{len(active_features)}/{len(active_features)}"
    )

    if compatibility_added:
        print(
            "Compatibility columns     : "
            + ", ".join(compatibility_added)
            + " -> added as 0.0 (inactive)"
        )
    else:
        print(
            "Compatibility columns     : "
            "already present / not required"
        )

    return dataframe


# ============================================================
# CREATE SOURCE CONTIGUOUS SEGMENTS
# ============================================================

def add_source_segments(
    dataframe: pd.DataFrame,
):
    """
    IMPORTANT CHANGE FROM V1:

    We DO NOT break when Label changes.

    A segment breaks only when rows are no longer consecutive
    in the original raw source after six-class filtering.

    Valid:
        BENIGN -> BENIGN -> FTP -> FTP -> BENIGN

    Invalid:
        raw index 100 -> 101 -> 500
    """

    dataframe = (
        dataframe
        .sort_values(
            "raw_row_index",
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    row_gap = (
        dataframe["raw_row_index"]
        .diff()
        .fillna(1)
        != 1
    )

    dataframe["_source_segment_id"] = (
        row_gap
        .cumsum()
        .astype(np.int64)
    )

    return dataframe


# ============================================================
# SPLIT SOURCE SEGMENTS INTO FIXED SOURCE-ORDERED BLOCKS
# ============================================================

def build_blocks(
    dataframe: pd.DataFrame,
    raw_file_name: str,
):
    """
    Returns a list of independent source-ordered blocks.

    Blocks never cross:
    - raw source files,
    - a gap caused by filtering unsupported classes,
    - fixed BLOCK_SIZE boundaries.

    A 19-row purge is removed at boundaries between consecutive
    blocks inside the same contiguous source segment.
    """

    blocks = []

    for segment_id, segment in dataframe.groupby(
        "_source_segment_id",
        sort=False,
    ):
        segment = (
            segment
            .sort_values(
                "raw_row_index",
                kind="mergesort",
            )
            .reset_index(drop=True)
        )

        n = len(segment)

        block_number = 0

        for start in range(
            0,
            n,
            BLOCK_SIZE,
        ):
            end = min(
                start + BLOCK_SIZE,
                n,
            )

            block = (
                segment
                .iloc[start:end]
                .copy()
            )

            # Purge a short boundary at the beginning of every
            # block except the first block in a source segment.
            if block_number > 0:
                if len(block) <= PURGE_GAP:
                    block_number += 1
                    continue

                block = (
                    block
                    .iloc[PURGE_GAP:]
                    .copy()
                )

            if len(block) < MIN_BLOCK_ROWS:
                block_number += 1
                continue

            block_id = (
                f"{raw_file_name}"
                f"::segment_{int(segment_id):06d}"
                f"::block_{block_number:06d}"
            )

            block["source_file"] = block_id
            block["original_source_file"] = (
                raw_file_name
            )
            block["record_id"] = (
                block["raw_row_index"]
                .astype(np.int64)
            )
            block["block_id"] = block_id

            blocks.append(block)

            block_number += 1

    return blocks


# ============================================================
# CLASS-AWARE BLOCK SPLIT
# ============================================================

def choose_validation_blocks(
    blocks,
    seed: int,
):
    """
    Select approximately 20% of blocks for validation.

    Scientific goals:
    1. No block exists in both train and validation.
    2. Ensure each attack class present in this raw source gets
       validation coverage whenever at least one block contains it.
    3. Keep the overall split near 80/20.

    Labels are used only to stratify the TRAIN/VALIDATION split.
    Locked test is never accessed.
    """

    if not blocks:
        return set()

    rng = random.Random(seed)

    block_indices = list(
        range(len(blocks))
    )

    target_validation_blocks = max(
        1,
        int(
            round(
                len(blocks)
                * VALIDATION_BLOCK_FRACTION
            )
        ),
    )

    target_validation_blocks = min(
        target_validation_blocks,
        max(
            1,
            len(blocks) - 1,
        ),
    )

    validation_indices = set()

    # --------------------------------------------------------
    # Guarantee validation coverage for attack classes present
    # in this raw source.
    # --------------------------------------------------------

    for attack_class in PROJECT_CLASSES:
        if attack_class == "BENIGN":
            continue

        candidates = [
            index
            for index, block in enumerate(blocks)
            if (
                block[phase4.LABEL_COL]
                == attack_class
            ).any()
        ]

        if not candidates:
            continue

        # Prefer a later attack-containing block to create a
        # source-order holdout flavour.
        chosen = candidates[-1]

        validation_indices.add(
            chosen
        )

    # --------------------------------------------------------
    # Ensure BENIGN is represented in validation too.
    # --------------------------------------------------------

    benign_candidates = [
        index
        for index, block in enumerate(blocks)
        if (
            block[phase4.LABEL_COL]
            == "BENIGN"
        ).any()
    ]

    if benign_candidates:
        validation_indices.add(
            benign_candidates[-1]
        )

    # --------------------------------------------------------
    # Fill remaining validation quota deterministically.
    # --------------------------------------------------------

    remaining = [
        index
        for index in block_indices
        if index not in validation_indices
    ]

    rng.shuffle(
        remaining
    )

    desired_total = max(
        target_validation_blocks,
        len(validation_indices),
    )

    for index in remaining:
        if len(validation_indices) >= desired_total:
            break

        validation_indices.add(
            index
        )

    # --------------------------------------------------------
    # Never allow every block into validation.
    # --------------------------------------------------------

    if (
        len(validation_indices)
        >= len(blocks)
        and len(blocks) > 1
    ):
        # Prefer moving the earliest benign-only validation block
        # back to train.
        removable = None

        for index in sorted(validation_indices):
            labels = set(
                blocks[index][
                    phase4.LABEL_COL
                ].unique()
            )

            if labels == {"BENIGN"}:
                removable = index
                break

        if removable is None:
            removable = min(
                validation_indices
            )

        validation_indices.remove(
            removable
        )

    return validation_indices


# ============================================================
# CLIENT-SPECIFIC ORDERED STREAMS
# ============================================================

def write_client_streams_from_train_block(
    train_block: pd.DataFrame,
    client_distribution,
):
    """
    Each client receives:
        BENIGN + its own attack.

    BENIGN context is intentionally available to every simulated
    client in this sequence-learning branch.

    This is explicitly recorded as shared benign reference traffic.
    Attack flows remain attack-family-specific.

    After filtering out foreign attacks, any raw-row gap creates a
    new client source group so Phase14D cannot create a false window
    across removed rows.
    """

    for client_id in range(
        1,
        N_CLIENTS + 1,
    ):
        attack_name = CLIENT_ATTACKS[
            client_id
        ]

        allowed_labels = {
            "BENIGN",
            attack_name,
        }

        client_block = (
            train_block[
                train_block[
                    phase4.LABEL_COL
                ].isin(
                    allowed_labels
                )
            ]
            .copy()
        )

        if len(client_block) < SEQUENCE_LENGTH:
            continue

        client_block = (
            client_block
            .sort_values(
                "raw_row_index",
                kind="mergesort",
            )
            .reset_index(drop=True)
        )

        gap = (
            client_block[
                "raw_row_index"
            ]
            .diff()
            .fillna(1)
            != 1
        )

        client_block[
            "_client_segment_id"
        ] = (
            gap
            .cumsum()
            .astype(np.int64)
        )

        # Write each contiguous client segment independently.
        for (
            client_segment_id,
            client_segment,
        ) in client_block.groupby(
            "_client_segment_id",
            sort=False,
        ):
            if (
                len(client_segment)
                < SEQUENCE_LENGTH
            ):
                continue

            original_block_id = str(
                client_segment[
                    "block_id"
                ].iloc[0]
            )

            client_group_id = (
                f"client_{client_id}"
                f"::{original_block_id}"
                f"::sub_{int(client_segment_id):06d}"
            )

            client_segment = (
                client_segment
                .copy()
            )

            client_segment[
                "source_file"
            ] = client_group_id

            client_segment[
                "client_id"
            ] = client_id

            client_file = (
                NON_IID_ROOT
                / f"client_{client_id}.csv"
            )

            append_csv(
                client_segment,
                client_file,
            )

            counts = (
                client_segment[
                    phase4.LABEL_COL
                ]
                .value_counts()
                .to_dict()
            )

            for label, count in counts.items():
                if (
                    label
                    in client_distribution[
                        client_id
                    ]
                ):
                    client_distribution[
                        client_id
                    ][
                        label
                    ] += int(count)


# ============================================================
# VALIDATE FINAL CLASS COVERAGE
# ============================================================

def verify_class_coverage(
    train_counts,
    validation_counts,
):
    missing_train = [
        label
        for label in PROJECT_CLASSES
        if train_counts.get(label, 0) <= 0
    ]

    missing_validation = [
        label
        for label in PROJECT_CLASSES
        if validation_counts.get(
            label,
            0,
        ) <= 0
    ]

    if missing_train:
        raise RuntimeError(
            "\nPhase14D0 produced no TRAIN rows for:\n"
            + "\n".join(
                missing_train
            )
        )

    if missing_validation:
        raise RuntimeError(
            "\nPhase14D0 produced no VALIDATION rows for:\n"
            + "\n".join(
                missing_validation
            )
            + "\n\n"
            "Do not continue to CNN-BiLSTM until all six "
            "validation classes are present."
        )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete and rebuild only "
            "data\\temporal_phase14."
        ),
    )

    args = parser.parse_args()

    set_seed(
        SEED
    )

    # ========================================================
    # VERIFY RAW INPUT
    # ========================================================

    if not RAW_ROOT.exists():
        raise FileNotFoundError(
            f"\nRaw data directory not found:\n"
            f"{RAW_ROOT}"
        )

    raw_files = sorted(
        RAW_ROOT.glob(
            "*.csv"
        )
    )

    if not raw_files:
        raise FileNotFoundError(
            f"\nNo raw CSV files found under:\n"
            f"{RAW_ROOT}"
        )

    # ========================================================
    # OUTPUT POLICY
    # ========================================================

    if TEMPORAL_ROOT.exists():
        if not args.overwrite:
            raise RuntimeError(
                "\nTemporal Phase14 directory already exists:\n"
                f"{TEMPORAL_ROOT}\n\n"
                "Rerun intentionally with:\n\n"
                "    --overwrite\n"
            )

        shutil.rmtree(
            TEMPORAL_ROOT
        )

    NON_IID_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    METADATA_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # LOAD EXISTING PREPROCESSING METADATA
    # ========================================================

    (
        scaler,
        feature_columns,
        label_mapping,
        active_indices,
        active_features,
    ) = phase4.load_preprocessing()

    if len(active_features) != 70:
        raise ValueError(
            "\nExpected exactly 70 active features."
        )

    # ========================================================
    # HEADER
    # ========================================================

    separator()
    print(PHASE_NAME)
    separator()

    print(
        f"Raw directory             : "
        f"{RAW_ROOT}"
    )

    print(
        f"Raw CSV files             : "
        f"{len(raw_files)}"
    )

    print(
        f"Active features           : "
        f"{len(active_features)}"
    )

    print(
        f"Sequence length           : "
        f"{SEQUENCE_LENGTH}"
    )

    print(
        f"Source block size         : "
        f"{BLOCK_SIZE:,}"
    )

    print(
        f"Validation block fraction : "
        f"{VALIDATION_BLOCK_FRACTION:.2f}"
    )

    print(
        f"Boundary purge            : "
        f"{PURGE_GAP}"
    )

    print()

    print("CRITICAL METHOD CHANGES")
    print("-" * 112)

    print(
        "1. Label changes DO NOT break source-ordered segments."
    )

    print(
        "2. Only raw-row discontinuities break a source segment."
    )

    print(
        "3. Source segments are divided into independent 2,000-row blocks."
    )

    print(
        "4. Train/validation are separated at the block level."
    )

    print(
        "5. Validation block selection preserves minority attack coverage."
    )

    print(
        "6. Client streams preserve BENIGN -> attack -> BENIGN transitions."
    )

    print(
        "7. Each client receives only BENIGN + its designated attack."
    )

    print(
        "8. BENIGN reference traffic is shared across simulated clients "
        "for this sequence-learning branch."
    )

    print(
        "9. Locked test is NOT used."
    )

    print(
        "10. Sequences are source-ordered, NOT timestamp-confirmed."
    )

    # ========================================================
    # STATE
    # ========================================================

    overall_train_counts = {
        label: 0
        for label in PROJECT_CLASSES
    }

    overall_validation_counts = {
        label: 0
        for label in PROJECT_CLASSES
    }

    client_distribution = {
        client_id: {
            label: 0
            for label in PROJECT_CLASSES
        }
        for client_id
        in range(
            1,
            N_CLIENTS + 1,
        )
    }

    manifest_rows = []

    block_manifest_rows = []

    total_start = time.perf_counter()

    # ========================================================
    # PROCESS RAW FILES
    # ========================================================

    for (
        source_index,
        raw_file,
    ) in enumerate(
        raw_files,
        start=1,
    ):
        separator()

        print(
            f"RAW FILE {source_index}/"
            f"{len(raw_files)}"
        )

        print("-" * 112)
        print(raw_file.name)

        source_start = time.perf_counter()

        dataframe = pd.read_csv(
            raw_file,
            low_memory=False,
        )

        dataframe = normalize_columns(
            dataframe
        )

        if phase4.LABEL_COL not in dataframe.columns:
            raise KeyError(
                f"\nMissing Label in:\n{raw_file}"
            )

        original_rows = len(
            dataframe
        )

        dataframe[
            "raw_row_index"
        ] = np.arange(
            original_rows,
            dtype=np.int64,
        )

        dataframe[
            phase4.LABEL_COL
        ] = (
            dataframe[
                phase4.LABEL_COL
            ]
            .map(
                normalize_label
            )
        )

        dataframe = (
            dataframe[
                dataframe[
                    phase4.LABEL_COL
                ].isin(
                    PROJECT_CLASSES
                )
            ]
            .copy()
        )

        selected_rows = len(
            dataframe
        )

        print(
            f"Original rows             : "
            f"{original_rows:,}"
        )

        print(
            f"Selected six-class rows   : "
            f"{selected_rows:,}"
        )

        if selected_rows == 0:
            print(
                "No selected rows; skipped."
            )
            continue

        dataframe = (
            verify_and_prepare_feature_columns(
                dataframe=dataframe,
                active_features=active_features,
                feature_columns=feature_columns,
            )
        )

        dataframe = add_source_segments(
            dataframe
        )

        blocks = build_blocks(
            dataframe=dataframe,
            raw_file_name=raw_file.name,
        )

        print(
            f"Source-order blocks       : "
            f"{len(blocks):,}"
        )

        if not blocks:
            print(
                "No usable blocks; skipped."
            )
            continue

        validation_indices = (
            choose_validation_blocks(
                blocks=blocks,
                seed=(
                    SEED
                    + source_index
                    * 1000
                ),
            )
        )

        source_train_rows = 0
        source_validation_rows = 0

        for block_index, block in enumerate(
            blocks
        ):
            split = (
                "validation"
                if block_index
                in validation_indices
                else
                "train"
            )

            block_counts = (
                block[
                    phase4.LABEL_COL
                ]
                .value_counts()
                .to_dict()
            )

            block_manifest_rows.append(
                {
                    "source_file":
                        raw_file.name,

                    "block_id":
                        str(
                            block[
                                "block_id"
                            ].iloc[0]
                        ),

                    "split":
                        split,

                    "rows":
                        int(
                            len(block)
                        ),

                    "raw_index_min":
                        int(
                            block[
                                "raw_row_index"
                            ].min()
                        ),

                    "raw_index_max":
                        int(
                            block[
                                "raw_row_index"
                            ].max()
                        ),

                    **{
                        f"class_{label}":
                            int(
                                block_counts
                                .get(
                                    label,
                                    0,
                                )
                            )
                        for label
                        in PROJECT_CLASSES
                    },
                }
            )

            if split == "validation":
                validation_block = (
                    block.copy()
                )

                validation_block[
                    "client_id"
                ] = -1

                append_csv(
                    validation_block,
                    VALIDATION_FILE,
                )

                source_validation_rows += (
                    len(validation_block)
                )

                for label, count in block_counts.items():
                    if label in overall_validation_counts:
                        overall_validation_counts[
                            label
                        ] += int(count)

            else:
                train_block = block.copy()

                train_block[
                    "client_id"
                ] = 0

                append_csv(
                    train_block,
                    TRAIN_FILE,
                )

                source_train_rows += len(
                    train_block
                )

                for label, count in block_counts.items():
                    if label in overall_train_counts:
                        overall_train_counts[
                            label
                        ] += int(count)

                # Build severe Non-IID client streams while
                # preserving benign/attack context.
                write_client_streams_from_train_block(
                    train_block=train_block,
                    client_distribution=client_distribution,
                )

        source_elapsed = (
            time.perf_counter()
            - source_start
        )

        source_counts = (
            dataframe[
                phase4.LABEL_COL
            ]
            .value_counts()
            .to_dict()
        )

        manifest_rows.append(
            {
                "source_file":
                    raw_file.name,

                "original_rows":
                    int(
                        original_rows
                    ),

                "selected_rows":
                    int(
                        selected_rows
                    ),

                "source_segments":
                    int(
                        dataframe[
                            "_source_segment_id"
                        ].nunique()
                    ),

                "blocks":
                    int(
                        len(blocks)
                    ),

                "validation_blocks":
                    int(
                        len(
                            validation_indices
                        )
                    ),

                "train_rows":
                    int(
                        source_train_rows
                    ),

                "validation_rows":
                    int(
                        source_validation_rows
                    ),

                "processing_seconds":
                    float(
                        source_elapsed
                    ),

                **{
                    f"class_{label}":
                        int(
                            source_counts
                            .get(
                                label,
                                0,
                            )
                        )

                    for label
                    in PROJECT_CLASSES
                },
            }
        )

        print(
            f"Train rows                : "
            f"{source_train_rows:,}"
        )

        print(
            f"Validation rows           : "
            f"{source_validation_rows:,}"
        )

        print(
            f"Validation blocks         : "
            f"{len(validation_indices):,}"
        )

        print(
            f"Processing time           : "
            f"{source_elapsed:.2f}s"
        )

        del dataframe
        del blocks

    # ========================================================
    # VERIFY OUTPUTS
    # ========================================================

    if not TRAIN_FILE.exists():
        raise RuntimeError(
            "\nNo source-ordered training file generated."
        )

    if not VALIDATION_FILE.exists():
        raise RuntimeError(
            "\nNo source-ordered validation file generated."
        )

    for client_id in range(
        1,
        N_CLIENTS + 1,
    ):
        client_file = (
            NON_IID_ROOT
            / f"client_{client_id}.csv"
        )

        if not client_file.exists():
            raise RuntimeError(
                f"\nClient {client_id} file was not generated."
            )

    verify_class_coverage(
        train_counts=overall_train_counts,
        validation_counts=overall_validation_counts,
    )

    # ========================================================
    # VERIFY EACH CLIENT HAS BENIGN + OWN ATTACK
    # ========================================================

    for client_id in range(
        1,
        N_CLIENTS + 1,
    ):
        attack_name = CLIENT_ATTACKS[
            client_id
        ]

        benign_count = client_distribution[
            client_id
        ]["BENIGN"]

        attack_count = client_distribution[
            client_id
        ][
            attack_name
        ]

        if benign_count < SEQUENCE_LENGTH:
            raise RuntimeError(
                f"\nClient {client_id} has insufficient "
                "BENIGN source-ordered rows."
            )

        if attack_count < SEQUENCE_LENGTH:
            raise RuntimeError(
                f"\nClient {client_id} has insufficient "
                f"{attack_name} rows: {attack_count}"
            )

    # ========================================================
    # SAVE METADATA
    # ========================================================

    manifest_dataframe = pd.DataFrame(
        manifest_rows
    )

    manifest_dataframe.to_csv(
        MANIFEST_FILE,
        index=False,
    )

    block_manifest_dataframe = pd.DataFrame(
        block_manifest_rows
    )

    block_manifest_dataframe.to_csv(
        BLOCK_MANIFEST_FILE,
        index=False,
    )

    client_distribution_rows = []

    for client_id in range(
        1,
        N_CLIENTS + 1,
    ):
        for label in PROJECT_CLASSES:
            client_distribution_rows.append(
                {
                    "client_id":
                        client_id,

                    "assigned_attack":
                        CLIENT_ATTACKS[
                            client_id
                        ],

                    "class_name":
                        label,

                    "rows":
                        int(
                            client_distribution[
                                client_id
                            ][
                                label
                            ]
                        ),
                }
            )

    client_distribution_dataframe = (
        pd.DataFrame(
            client_distribution_rows
        )
    )

    client_distribution_dataframe.to_csv(
        CLIENT_DISTRIBUTION_FILE,
        index=False,
    )

    distribution_rows = []

    for label in PROJECT_CLASSES:
        distribution_rows.append(
            {
                "split":
                    "train",

                "class_name":
                    label,

                "rows":
                    int(
                        overall_train_counts[
                            label
                        ]
                    ),
            }
        )

        distribution_rows.append(
            {
                "split":
                    "validation",

                "class_name":
                    label,

                "rows":
                    int(
                        overall_validation_counts[
                            label
                        ]
                    ),
            }
        )

    distribution_dataframe = pd.DataFrame(
        distribution_rows
    )

    distribution_dataframe.to_csv(
        CLASS_DISTRIBUTION_FILE,
        index=False,
    )

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    config = {
        "phase":
            "14D0-v2",

        "component":
            "source-ordered CNN-BiLSTM dataset construction",

        "timestamp_available":
            False,

        "pcap_available":
            False,

        "scientific_description":
            (
                "Source-ordered flow blocks reconstructed from "
                "the original CICIDS2017 machine-learning CSV "
                "row order. Label transitions are preserved. "
                "The sequences are not timestamp-confirmed."
            ),

        "seed":
            SEED,

        "active_feature_count":
            len(
                active_features
            ),

        "sequence_length":
            SEQUENCE_LENGTH,

        "block_size":
            BLOCK_SIZE,

        "validation_block_fraction":
            VALIDATION_BLOCK_FRACTION,

        "boundary_purge":
            PURGE_GAP,

        "segment_break_policy":
            "raw_row_gap_only",

        "label_change_breaks_segment":
            False,

        "validation_policy":
            (
                "block-level split with attack-class "
                "coverage guarantee"
            ),

        "client_policy":
            {
                "type":
                    "severe_non_iid",

                "attack_mapping":
                    CLIENT_ATTACKS,

                "allowed_classes":
                    "BENIGN + own attack",

                "shared_benign_reference":
                    True,

                "attack_rows_shared_between_clients":
                    False,
            },

        "classes":
            PROJECT_CLASSES,

        "raw_directory":
            str(
                RAW_ROOT
            ),

        "train_file":
            str(
                TRAIN_FILE
            ),

        "validation_file":
            str(
                VALIDATION_FILE
            ),

        "client_directory":
            str(
                NON_IID_ROOT
            ),

        "manifest_file":
            str(
                MANIFEST_FILE
            ),

        "block_manifest_file":
            str(
                BLOCK_MANIFEST_FILE
            ),

        "locked_test_used":
            False,

        "existing_processed_data_modified":
            False,

        "existing_federated_data_modified":
            False,

        "total_seconds":
            float(
                total_seconds
            ),
    }

    save_json(
        CONFIG_FILE,
        config,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    separator()
    print("PHASE 14D0 V2 COMPLETE")
    separator()

    print(
        "TRAIN / VALIDATION DISTRIBUTION"
    )

    print("-" * 112)

    split_summary = (
        distribution_dataframe
        .pivot_table(
            index="class_name",
            columns="split",
            values="rows",
            fill_value=0,
        )
    )

    print(
        split_summary.to_string()
    )

    print()

    print(
        "CLIENT DISTRIBUTIONS"
    )

    print("-" * 112)

    nonzero = (
        client_distribution_dataframe[
            client_distribution_dataframe[
                "rows"
            ] > 0
        ]
    )

    client_summary = (
        nonzero
        .pivot_table(
            index=[
                "client_id",
                "assigned_attack",
            ],
            columns="class_name",
            values="rows",
            aggfunc="sum",
            fill_value=0,
        )
    )

    print(
        client_summary.to_string()
    )

    print()

    print("OUTPUTS")
    print("-" * 112)

    print(
        f"Train          : {TRAIN_FILE}"
    )

    print(
        f"Validation     : {VALIDATION_FILE}"
    )

    print(
        f"Clients        : {NON_IID_ROOT}"
    )

    print(
        f"Manifest       : {MANIFEST_FILE}"
    )

    print(
        f"Block manifest : {BLOCK_MANIFEST_FILE}"
    )

    print(
        f"Config         : {CONFIG_FILE}"
    )

    print()

    print(
        f"Total time     : {total_seconds:.2f}s"
    )

    print()

    print(
        "LOCKED TEST USED : NO"
    )

    print()

    print(
        "SCIENTIFIC LABEL:"
    )

    print(
        "SOURCE-ORDERED flow sequences; "
        "NOT timestamp-confirmed temporal sequences."
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()
