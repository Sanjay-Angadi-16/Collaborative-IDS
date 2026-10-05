"""
Phase 3A - Common IDS Preprocessing

Fits preprocessing ONLY on Phase-1 training data.

Creates:
    artifacts/preprocessing/scaler.joblib
    artifacts/preprocessing/feature_columns.json
    artifacts/preprocessing/label_mapping.json
    artifacts/preprocessing/phase3_preprocessing_manifest.json

The locked test set is NEVER used for fitting.

Run:
    python scripts/phase3_prepare_preprocessing.py
"""

from pathlib import Path
import json
import logging

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train.csv"
)

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "preprocessing"
)

LABEL_COL = "Label"

CHUNK_SIZE = 100_000


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("phase3")


# ============================================================
# LABEL NORMALIZATION
# ============================================================

def normalize_label(label):

    label = str(label).strip()

    replacements = {
        "Web Attack � Brute Force":
            "Web Attack - Brute Force",

        "Web Attack � Sql Injection":
            "Web Attack - SQL Injection",

        "Web Attack � XSS":
            "Web Attack - XSS",

        "Web Attack – Brute Force":
            "Web Attack - Brute Force",

        "Web Attack – Sql Injection":
            "Web Attack - SQL Injection",

        "Web Attack – XSS":
            "Web Attack - XSS",
    }

    return replacements.get(label, label)


# ============================================================
# CLEAN FEATURES
# ============================================================

def clean_features(df):

    # Convert everything to numeric.
    df = df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Infinity -> NaN
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Missing/invalid -> 0
    #
    # We use deterministic replacement so exactly the same
    # operation can later be applied to every federated client
    # and the locked test set.
    df = df.fillna(0.0)

    return df


# ============================================================
# DETECT FEATURES + LABELS
# ============================================================

def inspect_training_data():

    logger.info(
        "Inspecting training file: %s",
        TRAIN_FILE
    )

    sample = pd.read_csv(
        TRAIN_FILE,
        nrows=5000
    )

    # Clean column names
    sample.columns = [
        str(col).strip()
        for col in sample.columns
    ]

    if LABEL_COL not in sample.columns:

        raise ValueError(
            f"Label column '{LABEL_COL}' not found.\n"
            f"Columns: {list(sample.columns)}"
        )

    feature_columns = [
        col
        for col in sample.columns
        if col != LABEL_COL
    ]

    logger.info(
        "Feature columns detected: %d",
        len(feature_columns)
    )

    return feature_columns


# ============================================================
# BUILD LABEL MAPPING
# ============================================================

def build_label_mapping():

    logger.info(
        "Collecting training labels..."
    )

    labels = set()

    for chunk in pd.read_csv(
        TRAIN_FILE,
        usecols=[LABEL_COL],
        chunksize=CHUNK_SIZE
    ):

        normalized = (
            chunk[LABEL_COL]
            .astype(str)
            .map(normalize_label)
        )

        labels.update(
            normalized.unique().tolist()
        )

    labels = sorted(labels)

    label_mapping = {
        label: index
        for index, label in enumerate(labels)
    }

    logger.info(
        "Detected %d classes.",
        len(label_mapping)
    )

    for label, index in label_mapping.items():

        logger.info(
            "Class %02d -> %s",
            index,
            label
        )

    return label_mapping


# ============================================================
# FIT SCALER
# ============================================================

def fit_scaler(feature_columns):

    logger.info(
        "Fitting StandardScaler using training data ONLY..."
    )

    scaler = StandardScaler()

    processed_rows = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            TRAIN_FILE,
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):

        chunk.columns = [
            str(col).strip()
            for col in chunk.columns
        ]

        X = chunk[
            feature_columns
        ].copy()

        X = clean_features(X)

        scaler.partial_fit(X)

        processed_rows += len(chunk)

        logger.info(
            "Scaler chunk %d | rows=%d | total=%d",
            chunk_number,
            len(chunk),
            processed_rows
        )

    logger.info(
        "Scaler fitted on %d training rows.",
        processed_rows
    )

    return scaler, processed_rows


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info(
        "Starting Phase 3A preprocessing."
    )

    if not TRAIN_FILE.exists():

        raise FileNotFoundError(
            f"Training file not found: {TRAIN_FILE}"
        )

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Feature list
    # --------------------------------------------------------

    feature_columns = inspect_training_data()

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    label_mapping = build_label_mapping()

    # --------------------------------------------------------
    # Scaler
    # --------------------------------------------------------

    scaler, rows_used = fit_scaler(
        feature_columns
    )

    # --------------------------------------------------------
    # Save scaler
    # --------------------------------------------------------

    scaler_file = (
        ARTIFACT_DIR
        / "scaler.joblib"
    )

    joblib.dump(
        scaler,
        scaler_file
    )

    # --------------------------------------------------------
    # Save feature columns
    # --------------------------------------------------------

    feature_file = (
        ARTIFACT_DIR
        / "feature_columns.json"
    )

    with open(
        feature_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            feature_columns,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Save label mapping
    # --------------------------------------------------------

    label_file = (
        ARTIFACT_DIR
        / "label_mapping.json"
    )

    with open(
        label_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            label_mapping,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = {

        "phase": "3A",

        "description":
            "Common IDS preprocessing fitted on "
            "training data only",

        "training_file":
            str(TRAIN_FILE),

        "training_rows_used":
            rows_used,

        "number_of_features":
            len(feature_columns),

        "number_of_classes":
            len(label_mapping),

        "scaling":
            "StandardScaler",

        "scaler_method":
            "chunked partial_fit",

        "chunk_size":
            CHUNK_SIZE,

        "invalid_values":
            "Infinity -> NaN -> 0",

        "label_normalization":
            True,

        "locked_test_used_for_fitting":
            False
    }

    manifest_file = (
        ARTIFACT_DIR
        / "phase3_preprocessing_manifest.json"
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

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("PHASE 3A COMPLETE")
    print("=" * 80)

    print(
        f"Training rows : {rows_used:,}"
    )

    print(
        f"Features      : {len(feature_columns)}"
    )

    print(
        f"Classes       : {len(label_mapping)}"
    )

    print(
        f"Scaler        : {scaler_file}"
    )

    print(
        f"Features file : {feature_file}"
    )

    print(
        f"Labels file   : {label_file}"
    )

    print(
        f"Manifest      : {manifest_file}"
    )

    print(
        "\nLocked test used for scaler fitting: NO ✅"
    )

    print("\nLabel mapping:")

    for label, index in label_mapping.items():

        print(
            f"  {index:2d} -> {label}"
        )

    logger.info(
        "Phase 3A complete."
    )


if __name__ == "__main__":
    main()