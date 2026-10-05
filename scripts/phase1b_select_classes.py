"""
Phase 1B - Reduce CICIDS Dataset to 5 Attack Classes + BENIGN
=============================================================

FINAL CLASSES:
    BENIGN
    DoS Hulk
    DDoS
    PortScan
    DoS GoldenEye
    FTP-Patator

IMPORTANT:
    This script OVERWRITES:
        data/processed/train.csv
        data/processed/locked_test.csv

    It also removes old:
        data/federated/
        artifacts/preprocessing/
        artifacts/centralized_baseline/
        results/centralized_baseline/
        results/graphs/

Run:
    python scripts/phase1b_select_classes.py
"""

from pathlib import Path
import json
import logging
import shutil

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

TRAIN_FILE = (
    PROCESSED_DIR
    / "train.csv"
)


# ============================================================
# FINAL RESEARCH CLASSES
# ============================================================

SELECTED_CLASSES = [

    "BENIGN",

    "DoS Hulk",

    "DDoS",

    "PortScan",

    "DoS GoldenEye",

    "FTP-Patator",
]


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("phase1b")


# ============================================================
# FIND LOCKED TEST SET
# ============================================================

def find_locked_test_file():

    candidates = [

        PROCESSED_DIR
        / "locked_test.csv",

        PROCESSED_DIR
        / "test_locked.csv",

        PROCESSED_DIR
        / "global_test.csv",

        PROCESSED_DIR
        / "test.csv",
    ]

    for file in candidates:

        if file.exists():

            return file

    raise FileNotFoundError(
        "\nLocked test set was not found.\n"
        "Expected one of:\n"
        "  data/processed/locked_test.csv\n"
        "  data/processed/test_locked.csv\n"
        "  data/processed/global_test.csv\n"
        "  data/processed/test.csv\n"
    )


# ============================================================
# VALIDATE DATASET
# ============================================================

def validate_dataset(
    df,
    name
):

    if "Label" not in df.columns:

        raise ValueError(
            f"'Label' column missing from {name}"
        )

    df["Label"] = (
        df["Label"]
        .astype(str)
        .str.strip()
    )

    available = set(
        df["Label"].unique()
    )

    missing = [

        label
        for label
        in SELECTED_CLASSES

        if label
        not in available
    ]

    if missing:

        raise ValueError(
            f"{name} is missing required classes: "
            f"{missing}"
        )

    return df


# ============================================================
# FILTER DATASET
# ============================================================

def filter_dataset(
    df,
    name
):

    before = len(df)

    df = df[
        df["Label"].isin(
            SELECTED_CLASSES
        )
    ].copy()

    df.reset_index(
        drop=True,
        inplace=True
    )

    after = len(df)

    removed = (
        before
        - after
    )

    logger.info(
        "%s | before=%d | after=%d | removed=%d",
        name,
        before,
        after,
        removed
    )

    return df


# ============================================================
# DISPLAY CLASS DISTRIBUTION
# ============================================================

def print_distribution(
    df,
    title
):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    counts = (
        df["Label"]
        .value_counts()
    )

    total = len(df)

    for label in SELECTED_CLASSES:

        count = int(
            counts.get(
                label,
                0
            )
        )

        percentage = (
            count
            / total
            * 100
            if total
            else 0
        )

        print(
            f"{label:25s}"
            f"{count:12,d}"
            f"  {percentage:8.4f}%"
        )

    print("-" * 80)

    print(
        f"{'TOTAL':25s}"
        f"{total:12,d}"
    )


# ============================================================
# SAFE OVERWRITE
# ============================================================

def safe_overwrite_csv(
    df,
    target
):

    """
    Write to temporary file first.

    Only replace original file after successful write.
    """

    temp_file = (
        target.parent
        / f"{target.stem}_phase1b_tmp.csv"
    )

    logger.info(
        "Writing temporary file: %s",
        temp_file
    )

    df.to_csv(
        temp_file,
        index=False
    )

    # Quick validation
    check = pd.read_csv(
        temp_file,
        usecols=["Label"]
    )

    unexpected = set(
        check["Label"].unique()
    ) - set(
        SELECTED_CLASSES
    )

    if unexpected:

        temp_file.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            "Unexpected classes detected after filtering: "
            f"{unexpected}"
        )

    # Replace original
    temp_file.replace(
        target
    )

    logger.info(
        "OVERWRITTEN: %s",
        target
    )


# ============================================================
# REMOVE OLD EXPERIMENT OUTPUTS
# ============================================================

def remove_directory(
    path
):

    if path.exists():

        logger.info(
            "Removing old output: %s",
            path
        )

        shutil.rmtree(
            path
        )


def clean_old_outputs():

    """
    Delete all outputs generated from the previous
    15-class experiment.
    """

    directories = [

        # Old federated clients
        PROJECT_ROOT
        / "data"
        / "federated",

        # Old Phase 3A preprocessing
        PROJECT_ROOT
        / "artifacts"
        / "preprocessing",

        # Old centralized model
        PROJECT_ROOT
        / "artifacts"
        / "centralized_baseline",

        # Old results
        PROJECT_ROOT
        / "results"
        / "centralized_baseline",

        # Old graphs
        PROJECT_ROOT
        / "results"
        / "graphs",
    ]

    for directory in directories:

        remove_directory(
            directory
        )


# ============================================================
# CREATE MANIFEST
# ============================================================

def create_manifest(
    train_df,
    test_df,
    test_file
):

    manifest = {

        "phase":
            "1B",

        "description":
            "Reduced CICIDS multiclass dataset",

        "selection_policy":
            (
                "Five attack classes selected based on "
                "training-set sample support, plus BENIGN"
            ),

        "number_of_output_classes":
            len(SELECTED_CLASSES),

        "number_of_attack_classes":
            5,

        "includes_benign":
            True,

        "selected_classes":
            SELECTED_CLASSES,

        "train_rows":
            len(train_df),

        "locked_test_rows":
            len(test_df),

        "train_file":
            str(TRAIN_FILE),

        "locked_test_file":
            str(test_file),

        "original_files_overwritten":
            True,

        "old_experiment_outputs_removed":
            True,
    }

    manifest_file = (
        PROCESSED_DIR
        / "phase1b_manifest.json"
    )

    with open(
        manifest_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            manifest,
            f,
            indent=4
        )

    return manifest_file


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("PHASE 1B - FINAL CLASS SELECTION")
    print("=" * 80)

    print()
    print(
        "WARNING: Existing 15-class processed files "
        "will be overwritten."
    )

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    if not TRAIN_FILE.exists():

        raise FileNotFoundError(
            f"Training file not found: {TRAIN_FILE}"
        )

    test_file = (
        find_locked_test_file()
    )

    logger.info(
        "Training file: %s",
        TRAIN_FILE
    )

    logger.info(
        "Locked test file: %s",
        test_file
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    logger.info(
        "Loading training dataset..."
    )

    train_df = pd.read_csv(
        TRAIN_FILE
    )

    logger.info(
        "Loading locked test dataset..."
    )

    test_df = pd.read_csv(
        test_file
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    train_df = validate_dataset(
        train_df,
        "TRAIN"
    )

    test_df = validate_dataset(
        test_df,
        "LOCKED TEST"
    )

    # --------------------------------------------------------
    # Filter
    # --------------------------------------------------------

    train_filtered = filter_dataset(
        train_df,
        "TRAIN"
    )

    test_filtered = filter_dataset(
        test_df,
        "LOCKED TEST"
    )

    # Free original large frames
    del train_df
    del test_df

    # --------------------------------------------------------
    # Verify class availability
    # --------------------------------------------------------

    train_classes = set(
        train_filtered[
            "Label"
        ].unique()
    )

    test_classes = set(
        test_filtered[
            "Label"
        ].unique()
    )

    expected = set(
        SELECTED_CLASSES
    )

    if train_classes != expected:

        raise RuntimeError(
            "Training dataset class validation failed.\n"
            f"Expected: {sorted(expected)}\n"
            f"Found: {sorted(train_classes)}"
        )

    if test_classes != expected:

        logger.warning(
            "Not every selected class is represented "
            "in locked test data."
        )

        logger.warning(
            "Locked-test classes: %s",
            sorted(test_classes)
        )

    # --------------------------------------------------------
    # Print BEFORE overwrite
    # --------------------------------------------------------

    print_distribution(
        train_filtered,
        "FINAL TRAINING DISTRIBUTION"
    )

    print_distribution(
        test_filtered,
        "FINAL LOCKED TEST DISTRIBUTION"
    )

    # --------------------------------------------------------
    # Overwrite processed data
    # --------------------------------------------------------

    logger.info(
        "Overwriting Phase-1 training dataset..."
    )

    safe_overwrite_csv(
        train_filtered,
        TRAIN_FILE
    )

    logger.info(
        "Overwriting locked test dataset..."
    )

    safe_overwrite_csv(
        test_filtered,
        test_file
    )

    # --------------------------------------------------------
    # Delete old experiment outputs
    # --------------------------------------------------------

    clean_old_outputs()

    # --------------------------------------------------------
    # Save manifest
    # --------------------------------------------------------

    manifest_file = create_manifest(
        train_filtered,
        test_filtered,
        test_file
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("PHASE 1B COMPLETE")
    print("=" * 80)

    print()
    print("FINAL DATASET:")

    for index, label in enumerate(
        SELECTED_CLASSES
    ):

        print(
            f"  {index} -> {label}"
        )

    print()

    print(
        f"Training rows    : "
        f"{len(train_filtered):,}"
    )

    print(
        f"Locked test rows : "
        f"{len(test_filtered):,}"
    )

    print()

    print(
        f"Training file OVERWRITTEN:\n"
        f"{TRAIN_FILE}"
    )

    print()

    print(
        f"Locked test OVERWRITTEN:\n"
        f"{test_file}"
    )

    print()

    print(
        "Old federated/preprocessing/model/results "
        "outputs: REMOVED ✅"
    )

    print(
        "Original 15-class derived experiment: REMOVED ✅"
    )

    print()

    print(
        f"Manifest:\n{manifest_file}"
    )

    print()

    print("=" * 80)
    print("NEXT STEP: PHASE 2")
    print("=" * 80)


if __name__ == "__main__":
    main()