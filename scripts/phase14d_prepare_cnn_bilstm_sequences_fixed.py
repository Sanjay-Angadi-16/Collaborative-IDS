from __future__ import annotations

import argparse
import gc
import json
import logging
import random
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

for path in (
    PROJECT_ROOT,
    SCRIPT_DIR,
):
    if str(path) not in sys.path:
        sys.path.insert(
            0,
            str(path),
        )


# ============================================================
# EXISTING PROJECT INFRASTRUCTURE
# ============================================================

try:
    import phase9_fedavg70 as phase4

except ModuleNotFoundError as exc:

    raise ModuleNotFoundError(
        "\nCould not import:\n"
        "scripts\\phase9_fedavg70.py\n\n"
        "Keep this Phase 14D script inside the scripts folder."
    ) from exc


# ============================================================
# PHASE CONFIGURATION
# ============================================================

PHASE_NAME = (
    "PHASE 14D — CNN-BILSTM SOURCE-ORDERED SEQUENCE PREPARATION"
)

SCENARIO = "non_iid"

SEED = 42

N_CLIENTS = 5

EXPECTED_FEATURES = 70


# ============================================================
# SEQUENCE CONFIGURATION
# ============================================================

# Each CNN-BiLSTM sample:
#
# 20 flows
# ×
# 70 features
#
# shape:
# (20, 70)
#
SEQUENCE_LENGTH = 20


# ------------------------------------------------------------
# STRIDE
#
# STRIDE = 1
# gives heavily overlapping windows and very large files.
#
# STRIDE = 10
# still preserves temporal order while substantially reducing
# storage/training cost.
# ------------------------------------------------------------

SEQUENCE_STRIDE = 10


# ------------------------------------------------------------
# LABEL STRATEGY
#
# The target is the class of the LAST flow in the sequence.
#
# Example:
#
# Flow 1
# Flow 2
# ...
# Flow 20 -> TARGET LABEL
# ------------------------------------------------------------

LABEL_STRATEGY = "last_flow"


# ============================================================
# SAFETY LIMITS
# ============================================================

MAX_SEQUENCES_PER_CLIENT = 50_000

MAX_VALIDATION_SEQUENCES = 50_000


# ============================================================
# TEMPORAL / ORDER COLUMN CANDIDATES
# ============================================================

TIMESTAMP_COLUMN_CANDIDATES = [

    "Timestamp",
    "timestamp",
    "TimeStamp",
    "time_stamp",
    "flow_timestamp",
    "Flow Timestamp",
    "Flow Start",
    "FlowStart",
    "start_time",
    "Start Time",
]


# ============================================================
# ORDER FALLBACK CANDIDATES
#
# These are accepted only when a true timestamp is unavailable.
# ============================================================

ORDER_COLUMN_CANDIDATES = [

    "record_id",
    "stable_flow_id",
    "flow_index",
    "row_id",
    "sequence_id",
]


# ============================================================
# GROUP BOUNDARY CANDIDATES
#
# A sequence MUST NOT join traffic from unrelated source files
# or days if provenance is available.
# ============================================================

GROUP_COLUMN_CANDIDATES = [

    "source_file",
    "source_day",
    "day",
    "SourceFile",
    "Source Day",
]


# ============================================================
# OUTPUT PATHS
# ============================================================

SEQUENCE_ROOT = (
    PROJECT_ROOT
    /
    "data"
    /
    "sequences"
)

CLIENT_SEQUENCE_ROOT = (
    SEQUENCE_ROOT
    /
    SCENARIO
)

VALIDATION_SEQUENCE_FILE = (
    SEQUENCE_ROOT
    /
    "validation_sequences.npz"
)


RESULT_ROOT = (
    PROJECT_ROOT
    /
    "results"
    /
    "phase14"
)


ARTIFACT_ROOT = (
    PROJECT_ROOT
    /
    "artifacts"
    /
    "detection"
)


SEQUENCE_MANIFEST_FILE = (
    RESULT_ROOT
    /
    "cnn_bilstm_sequence_manifest.csv"
)


SEQUENCE_CLASS_DISTRIBUTION_FILE = (
    RESULT_ROOT
    /
    "cnn_bilstm_sequence_class_distribution.csv"
)


SEQUENCE_CONFIG_FILE = (
    ARTIFACT_ROOT
    /
    "cnn_bilstm_sequence_config.json"
)


# ============================================================
# PHASE 14D0 V2 SOURCE-ORDERED INPUTS
# ============================================================

TEMPORAL_ROOT = (
    PROJECT_ROOT
    /
    "data"
    /
    "temporal_phase14"
)


FEDERATED_SEQUENCE_SOURCE = (
    TEMPORAL_ROOT
    /
    "non_iid"
)


VALIDATION_FILE = (
    TEMPORAL_ROOT
    /
    "validation.csv"
)


# The existing locked test path is recorded only for audit
# metadata. Phase 14D DOES NOT read or transform this file.
LOCKED_TEST_FILE = (
    PROJECT_ROOT
    /
    "data"
    /
    "processed"
    /
    "test_locked.csv"
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(
    "phase14d_prepare_sequences"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(
    seed: int,
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )


# ============================================================
# DISPLAY
# ============================================================

def separator():

    print()

    print(
        "=" * 110
    )


# ============================================================
# JSON SAFE
# ============================================================

def json_safe(
    value,
):

    if isinstance(
        value,
        dict,
    ):

        return {
            str(key):
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
        Path,
    ):

        return str(
            value
        )


    return value


# ============================================================
# SAVE JSON
# ============================================================

def save_json(
    path,
    payload,
):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            json_safe(
                payload
            ),
            file,
            indent=4,
        )


# ============================================================
# FILTER TO SIX PROJECT CLASSES
# ============================================================

def filter_to_project_classes(
    dataframe,
    label_mapping,
):

    if phase4.LABEL_COL not in dataframe.columns:

        raise KeyError(
            "\nMissing label column:\n"
            f"{phase4.LABEL_COL}"
        )


    canonical = {

        str(label)
        .strip()
        .lower():

            str(label)
            .strip()

        for label
        in label_mapping.keys()
    }


    labels = (
        dataframe[
            phase4.LABEL_COL
        ]
        .astype(
            str
        )
        .str.strip()
    )


    keep_mask = (
        labels
        .str.lower()
        .isin(
            canonical.keys()
        )
    )


    dataframe = (
        dataframe
        .loc[
            keep_mask
        ]
        .copy()
    )


    dataframe[
        phase4.LABEL_COL
    ] = (
        dataframe[
            phase4.LABEL_COL
        ]
        .astype(
            str
        )
        .str.strip()
        .str.lower()
        .map(
            canonical
        )
    )


    return dataframe


# ============================================================
# VERIFY PREPROCESSING
# ============================================================

def verify_preprocessing(
    active_features,
    active_indices,
    feature_columns,
):

    if len(
        active_features
    ) != EXPECTED_FEATURES:

        raise ValueError(
            "\nExpected exactly "
            f"{EXPECTED_FEATURES} active features.\n"
            f"Found: {len(active_features)}"
        )


    active_indices = np.asarray(
        active_indices,
        dtype=np.int64,
    )


    reconstructed = [

        feature_columns[
            int(index)
        ]

        for index
        in active_indices
    ]


    if reconstructed != list(
        active_features
    ):

        raise ValueError(
            "\nFeature order mismatch between "
            "active_features and active_indices."
        )


    return active_indices


# ============================================================
# FIND FIRST EXISTING COLUMN
# ============================================================

def first_existing_column(
    dataframe,
    candidates,
):

    columns = {
        str(column).strip().lower():
            column

        for column
        in dataframe.columns
    }


    for candidate in candidates:

        key = (
            str(candidate)
            .strip()
            .lower()
        )

        if key in columns:

            return columns[
                key
            ]


    return None


# ============================================================
# RESOLVE GROUP COLUMN
# ============================================================

def resolve_group_column(
    dataframe,
):

    group_column = first_existing_column(
        dataframe,
        GROUP_COLUMN_CANDIDATES,
    )


    return group_column


# ============================================================
# PREPARE TEMPORAL ORDER
# ============================================================

def prepare_temporal_order(
    dataframe,
    allow_row_order=False,
):

    # --------------------------------------------------------
    # 1. TRY REAL TIMESTAMP
    # --------------------------------------------------------

    timestamp_column = first_existing_column(
        dataframe,
        TIMESTAMP_COLUMN_CANDIDATES,
    )


    if timestamp_column is not None:

        logger.info(
            "Temporal column detected: %s",
            timestamp_column,
        )


        original_values = dataframe[
            timestamp_column
        ]


        # ----------------------------------------------------
        # First parsing attempt
        # ----------------------------------------------------

        parsed_time = pd.to_datetime(
            original_values,
            errors="coerce",
        )


        parse_ratio = float(
            parsed_time
            .notna()
            .mean()
        )


        # ----------------------------------------------------
        # Try day-first parsing if normal parse was weak
        # ----------------------------------------------------

        if parse_ratio < 0.90:

            parsed_dayfirst = pd.to_datetime(
                original_values,
                errors="coerce",
                dayfirst=True,
            )


            dayfirst_ratio = float(
                parsed_dayfirst
                .notna()
                .mean()
            )


            if dayfirst_ratio > parse_ratio:

                parsed_time = parsed_dayfirst

                parse_ratio = dayfirst_ratio


        print(
            f"Timestamp parse success : "
            f"{parse_ratio * 100:.2f}%"
        )


        if parse_ratio < 0.90:

            raise ValueError(
                "\nTimestamp column was found but fewer than "
                "90% of values could be parsed.\n\n"
                f"Column: {timestamp_column}\n"
                f"Success ratio: {parse_ratio:.4f}\n\n"
                "Do not silently build temporal CNN-BiLSTM "
                "windows from unreliable timestamps."
            )


        dataframe = dataframe.copy()


        dataframe[
            "__temporal_order"
        ] = parsed_time


        dataframe = dataframe[
            dataframe[
                "__temporal_order"
            ]
            .notna()
        ].copy()


        return (
            dataframe,
            "__temporal_order",
            "timestamp",
            str(
                timestamp_column
            ),
        )


    # --------------------------------------------------------
    # 2. TRY STABLE ORDER COLUMN
    # --------------------------------------------------------

    order_column = first_existing_column(
        dataframe,
        ORDER_COLUMN_CANDIDATES,
    )


    if order_column is not None:

        logger.warning(
            "No timestamp found. "
            "Using stable order column: %s",
            order_column,
        )


        values = pd.to_numeric(
            dataframe[
                order_column
            ],
            errors="coerce",
        )


        valid_ratio = float(
            values
            .notna()
            .mean()
        )


        if valid_ratio < 0.95:

            raise ValueError(
                "\nOrder column exists but is not sufficiently numeric:\n"
                f"{order_column}\n"
                f"Valid ratio: {valid_ratio:.4f}"
            )


        dataframe = dataframe.copy()


        dataframe[
            "__temporal_order"
        ] = values


        dataframe = dataframe[
            dataframe[
                "__temporal_order"
            ]
            .notna()
        ].copy()


        return (
            dataframe,
            "__temporal_order",
            "stable_order",
            str(
                order_column
            ),
        )


    # --------------------------------------------------------
    # 3. OPTIONAL ROW-ORDER FALLBACK
    #
    # This should ONLY be used if you have independently
    # verified that CSV row order preserves chronology.
    # --------------------------------------------------------

    if allow_row_order:

        logger.warning(
            "No timestamp/order column found. "
            "Using existing CSV row order because "
            "--allow-row-order was explicitly provided."
        )


        dataframe = dataframe.copy()


        dataframe[
            "__temporal_order"
        ] = np.arange(
            len(
                dataframe
            ),
            dtype=np.int64,
        )


        return (
            dataframe,
            "__temporal_order",
            "csv_row_order",
            "generated_row_index",
        )


    # --------------------------------------------------------
    # HARD STOP
    # --------------------------------------------------------

    raise RuntimeError(
        "\nNo trustworthy temporal/order column was found.\n\n"
        "CNN-BiLSTM must not be described as temporal if the "
        "client files were previously shuffled.\n\n"
        "Inspect your CSV columns first.\n\n"
        "If you KNOW the current CSV row order is chronological, "
        "rerun with:\n\n"
        "    --allow-row-order\n"
    )


# ============================================================
# SORT DATAFRAME
# ============================================================

def sort_dataframe_temporally(
    dataframe,
    order_column,
    group_column,
):

    if group_column is not None:

        dataframe = (
            dataframe
            .sort_values(
                by=[
                    group_column,
                    order_column,
                ],
                kind="mergesort",
            )
            .reset_index(
                drop=True
            )
        )

    else:

        dataframe = (
            dataframe
            .sort_values(
                by=[
                    order_column,
                ],
                kind="mergesort",
            )
            .reset_index(
                drop=True
            )
        )


    return dataframe


# ============================================================
# BUILD WINDOWS FOR ONE GROUP
# ============================================================

def create_windows_for_group(
    X,
    y,
    original_row_indices,
):

    number_of_rows = len(
        y
    )


    if number_of_rows < SEQUENCE_LENGTH:

        return (
            None,
            None,
            None,
        )


    starts = np.arange(
        0,
        number_of_rows
        -
        SEQUENCE_LENGTH
        +
        1,
        SEQUENCE_STRIDE,
        dtype=np.int64,
    )


    number_of_sequences = len(
        starts
    )


    X_sequences = np.empty(
        (
            number_of_sequences,
            SEQUENCE_LENGTH,
            EXPECTED_FEATURES,
        ),
        dtype=np.float32,
    )


    y_sequences = np.empty(
        number_of_sequences,
        dtype=np.int64,
    )


    end_row_indices = np.empty(
        number_of_sequences,
        dtype=np.int64,
    )


    for sequence_index, start in enumerate(
        starts
    ):

        end = (
            start
            +
            SEQUENCE_LENGTH
        )


        X_sequences[
            sequence_index
        ] = X[
            start:end
        ]


        # ----------------------------------------------------
        # TARGET = LABEL OF FINAL FLOW
        # ----------------------------------------------------

        y_sequences[
            sequence_index
        ] = y[
            end
            -
            1
        ]


        end_row_indices[
            sequence_index
        ] = original_row_indices[
            end
            -
            1
        ]


    return (
        X_sequences,
        y_sequences,
        end_row_indices,
    )


# ============================================================
# CREATE SEQUENCES WITHOUT CROSSING GROUP BOUNDARIES
# ============================================================

def create_sequences(
    X,
    y,
    dataframe,
    group_column,
    max_sequences,
):

    all_X = []

    all_y = []

    all_end_indices = []


    # --------------------------------------------------------
    # SAVE ORIGINAL SORTED ROW POSITION
    # --------------------------------------------------------

    row_positions = np.arange(
        len(
            dataframe
        ),
        dtype=np.int64,
    )


    # --------------------------------------------------------
    # CASE 1:
    # GROUP COLUMN AVAILABLE
    # --------------------------------------------------------

    if group_column is not None:

        grouped_indices = (
            dataframe
            .groupby(
                group_column,
                sort=False,
                dropna=False,
            )
            .indices
        )


        print(
            f"Sequence groups         : "
            f"{len(grouped_indices):,}"
        )


        skipped_groups = 0


        for group_name, positions in grouped_indices.items():

            positions = np.asarray(
                positions,
                dtype=np.int64,
            )


            if len(
                positions
            ) < SEQUENCE_LENGTH:

                skipped_groups += 1

                continue


            X_group = X[
                positions
            ]


            y_group = y[
                positions
            ]


            row_group = row_positions[
                positions
            ]


            (
                X_seq,
                y_seq,
                end_seq,

            ) = create_windows_for_group(

                X=
                    X_group,

                y=
                    y_group,

                original_row_indices=
                    row_group,
            )


            if X_seq is None:

                continue


            all_X.append(
                X_seq
            )


            all_y.append(
                y_seq
            )


            all_end_indices.append(
                end_seq
            )


        print(
            f"Groups too small        : "
            f"{skipped_groups:,}"
        )


    # --------------------------------------------------------
    # CASE 2:
    # NO GROUP COLUMN
    #
    # Whole file is one temporal stream.
    # --------------------------------------------------------

    else:

        (
            X_seq,
            y_seq,
            end_seq,

        ) = create_windows_for_group(

            X=
                X,

            y=
                y,

            original_row_indices=
                row_positions,
        )


        if X_seq is not None:

            all_X.append(
                X_seq
            )


            all_y.append(
                y_seq
            )


            all_end_indices.append(
                end_seq
            )


    # --------------------------------------------------------
    # CHECK
    # --------------------------------------------------------

    if not all_X:

        raise RuntimeError(
            "\nNo valid CNN-BiLSTM sequences were generated."
        )


    # --------------------------------------------------------
    # CONCATENATE
    # --------------------------------------------------------

    X_sequences = np.concatenate(
        all_X,
        axis=0,
    )


    y_sequences = np.concatenate(
        all_y,
        axis=0,
    )


    end_indices = np.concatenate(
        all_end_indices,
        axis=0,
    )


    del all_X

    del all_y

    del all_end_indices

    gc.collect()


    # --------------------------------------------------------
    # OPTIONAL DETERMINISTIC LIMIT
    #
    # We preserve the natural label distribution using a
    # deterministic random subset when the cap is exceeded.
    #
    # This is NOT oversampling.
    # --------------------------------------------------------

    if (
        max_sequences is not None

        and

        len(
            y_sequences
        )
        >
        max_sequences
    ):

        rng = np.random.default_rng(
            SEED
        )


        selected = rng.choice(
            len(
                y_sequences
            ),
            size=
                max_sequences,
            replace=
                False,
        )


        selected = np.sort(
            selected
        )


        X_sequences = X_sequences[
            selected
        ]


        y_sequences = y_sequences[
            selected
        ]


        end_indices = end_indices[
            selected
        ]


    return (
        X_sequences,
        y_sequences,
        end_indices,
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def sequence_class_distribution(
    y_sequences,
    label_mapping,
    dataset_name,
):

    inverse_mapping = {

        int(value):
            str(label)

        for label, value
        in label_mapping.items()
    }


    unique, counts = np.unique(
        y_sequences,
        return_counts=True,
    )


    rows = []


    for class_id, count in zip(
        unique,
        counts,
    ):

        rows.append(
            {
                "dataset":
                    dataset_name,

                "class_id":
                    int(
                        class_id
                    ),

                "class_name":
                    inverse_mapping[
                        int(
                            class_id
                        )
                    ],

                "sequences":
                    int(
                        count
                    ),

                "percentage":
                    float(
                        count
                        /
                        len(
                            y_sequences
                        )
                        *
                        100.0
                    ),
            }
        )


    return rows


# ============================================================
# PREPARE ONE DATASET
# ============================================================

def prepare_dataset(
    dataset_name,
    source_file,
    output_file,
    scaler,
    feature_columns,
    active_indices,
    active_features,
    label_mapping,
    max_sequences,
    allow_row_order,
):

    separator()

    print(
        f"PREPARING: {dataset_name}"
    )

    print(
        "-" * 110
    )


    print(
        f"Source file             : "
        f"{source_file}"
    )


    print(
        f"Output file             : "
        f"{output_file}"
    )


    if not source_file.exists():

        raise FileNotFoundError(
            f"\nSource file not found:\n"
            f"{source_file}"
        )


    start_time = time.perf_counter()


    # ========================================================
    # LOAD
    # ========================================================

    dataframe = pd.read_csv(
        source_file
    )


    original_rows = len(
        dataframe
    )


    dataframe = filter_to_project_classes(
        dataframe,
        label_mapping,
    )


    filtered_rows = len(
        dataframe
    )


    print(
        f"Original rows           : "
        f"{original_rows:,}"
    )


    print(
        f"Six-class rows          : "
        f"{filtered_rows:,}"
    )


    # ========================================================
    # DETECT GROUPING / TEMPORAL ORDER
    # ========================================================

    group_column = resolve_group_column(
        dataframe
    )


    if group_column is None:

        print(
            "Group boundary column   : NONE"
        )

    else:

        print(
            f"Group boundary column   : "
            f"{group_column}"
        )


    (
        dataframe,
        temporal_order_column,
        order_type,
        source_order_column,

    ) = prepare_temporal_order(

        dataframe=
            dataframe,

        allow_row_order=
            allow_row_order,
    )


    print(
        f"Temporal order type     : "
        f"{order_type}"
    )


    print(
        f"Temporal source column  : "
        f"{source_order_column}"
    )


    # ========================================================
    # SORT
    # ========================================================

    dataframe = sort_dataframe_temporally(

        dataframe=
            dataframe,

        order_column=
            temporal_order_column,

        group_column=
            group_column,
    )


    # ========================================================
    # APPLY EXISTING 70-FEATURE PREPROCESSING
    # ========================================================

    logger.info(
        "%s | transforming 70 active features...",
        dataset_name,
    )


    X = phase4.transform_features(

        df=
            dataframe,

        feature_columns=
            feature_columns,

        scaler=
            scaler,

        active_indices=
            active_indices,
    )


    y = phase4.encode_labels(

        dataframe[
            phase4.LABEL_COL
        ],

        label_mapping,
    )


    X = X.astype(
        np.float32,
        copy=False,
    )


    y = y.astype(
        np.int64,
        copy=False,
    )


    if X.shape[
        1
    ] != EXPECTED_FEATURES:

        raise ValueError(
            f"\n{dataset_name}: expected "
            f"{EXPECTED_FEATURES} features, "
            f"found {X.shape[1]}."
        )


    # ========================================================
    # CREATE TEMPORAL WINDOWS
    # ========================================================

    logger.info(
        "%s | creating sequence windows...",
        dataset_name,
    )


    (
        X_sequences,
        y_sequences,
        end_indices,

    ) = create_sequences(

        X=
            X,

        y=
            y,

        dataframe=
            dataframe,

        group_column=
            group_column,

        max_sequences=
            max_sequences,
    )


    # ========================================================
    # VERIFY SHAPE
    # ========================================================

    expected_shape = (
        SEQUENCE_LENGTH,
        EXPECTED_FEATURES,
    )


    if tuple(
        X_sequences.shape[
            1:
        ]
    ) != expected_shape:

        raise RuntimeError(
            "\nUnexpected sequence shape:\n"
            f"{X_sequences.shape}\n\n"
            f"Expected (*, {SEQUENCE_LENGTH}, "
            f"{EXPECTED_FEATURES})"
        )


    if len(
        X_sequences
    ) != len(
        y_sequences
    ):

        raise RuntimeError(
            "\nSequence X/y count mismatch."
        )


    # ========================================================
    # SAVE NPZ
    # ========================================================

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    logger.info(
        "%s | saving compressed NPZ...",
        dataset_name,
    )


    np.savez_compressed(

        output_file,

        X=
            X_sequences,

        y=
            y_sequences,

        end_row_index=
            end_indices,

        sequence_length=
            np.asarray(
                [
                    SEQUENCE_LENGTH
                ],
                dtype=np.int64,
            ),

        feature_count=
            np.asarray(
                [
                    EXPECTED_FEATURES
                ],
                dtype=np.int64,
            ),

        stride=
            np.asarray(
                [
                    SEQUENCE_STRIDE
                ],
                dtype=np.int64,
            ),
    )


    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    distribution_rows = sequence_class_distribution(

        y_sequences=
            y_sequences,

        label_mapping=
            label_mapping,

        dataset_name=
            dataset_name,
    )


    print()

    print(
        "SEQUENCE LABEL DISTRIBUTION"
    )

    print(
        "-" * 110
    )


    distribution_dataframe = pd.DataFrame(
        distribution_rows
    )


    print(

        distribution_dataframe[
            [
                "class_name",
                "sequences",
                "percentage",
            ]
        ]

        .to_string(
            index=False
        )
    )


    elapsed = (
        time.perf_counter()
        -
        start_time
    )


    file_size_mb = (
        output_file
        .stat()
        .st_size
        /
        1024
        /
        1024
    )


    print()

    print(
        f"Final sequence shape    : "
        f"{X_sequences.shape}"
    )


    print(
        f"Final label shape       : "
        f"{y_sequences.shape}"
    )


    print(
        f"Sequence count          : "
        f"{len(y_sequences):,}"
    )


    print(
        f"Saved size              : "
        f"{file_size_mb:.2f} MB"
    )


    print(
        f"Preparation time        : "
        f"{elapsed:.2f}s"
    )


    manifest_row = {

        "dataset":
            dataset_name,

        "source_file":
            str(
                source_file
            ),

        "output_file":
            str(
                output_file
            ),

        "original_rows":
            int(
                original_rows
            ),

        "filtered_rows":
            int(
                filtered_rows
            ),

        "sequence_count":
            int(
                len(
                    y_sequences
                )
            ),

        "sequence_length":
            SEQUENCE_LENGTH,

        "stride":
            SEQUENCE_STRIDE,

        "feature_count":
            EXPECTED_FEATURES,

        "label_strategy":
            LABEL_STRATEGY,

        "group_column":
            str(
                group_column
            )
            if group_column is not None
            else None,

        "temporal_order_type":
            order_type,

        "temporal_source_column":
            source_order_column,

        "max_sequence_limit":
            max_sequences,

        "file_size_mb":
            float(
                file_size_mb
            ),

        "preparation_seconds":
            float(
                elapsed
            ),

        "locked_test_used":
            False,
    }


    # ========================================================
    # CLEANUP
    # ========================================================

    del dataframe

    del X

    del y

    del X_sequences

    del y_sequences

    del end_indices

    gc.collect()


    return (
        manifest_row,
        distribution_rows,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(

        "--allow-row-order",

        action="store_true",

        help=(
            "Use current CSV row order only when no timestamp "
            "or stable order column exists. Use this only if "
            "you have verified that the CSV was not shuffled."
        ),
    )


    args = parser.parse_args()


    set_seed(
        SEED
    )


    CLIENT_SEQUENCE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    ARTIFACT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    separator()

    print(
        PHASE_NAME
    )

    separator()


    print(
        f"Scenario                 : "
        f"{SCENARIO}"
    )


    print(
        f"Phase14D0 source         : "
        f"{TEMPORAL_ROOT}"
    )


    print(
        f"Clients                  : "
        f"{N_CLIENTS}"
    )


    print(
        f"Input features           : "
        f"{EXPECTED_FEATURES}"
    )


    print(
        f"Sequence length          : "
        f"{SEQUENCE_LENGTH}"
    )


    print(
        f"Sequence stride          : "
        f"{SEQUENCE_STRIDE}"
    )


    print(
        f"Label strategy           : "
        f"{LABEL_STRATEGY}"
    )


    print(
        f"Client sequence cap      : "
        f"{MAX_SEQUENCES_PER_CLIENT:,}"
    )


    print(
        f"Validation sequence cap  : "
        f"{MAX_VALIDATION_SEQUENCES:,}"
    )


    print(
        f"Allow raw row order      : "
        f"{args.allow_row_order}"
    )


    print()

    print(
        "CRITICAL DATA POLICY"
    )

    print(
        "-" * 110
    )


    print(
        "TRAIN client sequences are generated only from "
        "Phase 14D0 V2 source-ordered client files."
    )


    print(
        "VALIDATION sequences are generated independently "
        "from Phase 14D0 V2 validation.csv."
    )


    print(
        "LOCKED TEST sequences are NOT generated in Phase 14D."
    )


    print(
        "No source-ordered sequence is allowed to cross "
        "Phase 14D0 V2 source/block boundaries."
    )


    print(
        "Target label = label of the final flow in each "
        "sequence."
    )


    # ========================================================
    # LOAD PREPROCESSING
    # ========================================================

    (
        scaler,
        feature_columns,
        label_mapping,
        active_indices,
        active_features,

    ) = phase4.load_preprocessing()


    active_indices = verify_preprocessing(

        active_features=
            active_features,

        active_indices=
            active_indices,

        feature_columns=
            feature_columns,
    )


    # ========================================================
    # FEDERATED DIRECTORY
    # ========================================================

    federated_dir = (
        FEDERATED_SEQUENCE_SOURCE
    )


    if not federated_dir.exists():

        raise FileNotFoundError(
            "\nFederated client directory missing:\n"
            f"{federated_dir}"
        )


    if not VALIDATION_FILE.exists():

        raise FileNotFoundError(
            "\nValidation file missing:\n"
            f"{VALIDATION_FILE}"
        )


    # ========================================================
    # PREPARE CLIENT SEQUENCES
    # ========================================================

    all_manifest_rows = []

    all_distribution_rows = []


    total_start = time.perf_counter()


    for client_id in range(
        1,
        N_CLIENTS + 1,
    ):

        client_file = (

            federated_dir
            /
            f"client_{client_id}.csv"
        )


        output_file = (

            CLIENT_SEQUENCE_ROOT
            /
            f"client_{client_id}_sequences.npz"
        )


        (
            manifest_row,
            distribution_rows,

        ) = prepare_dataset(

            dataset_name=
                f"client_{client_id}",

            source_file=
                client_file,

            output_file=
                output_file,

            scaler=
                scaler,

            feature_columns=
                feature_columns,

            active_indices=
                active_indices,

            active_features=
                active_features,

            label_mapping=
                label_mapping,

            max_sequences=
                MAX_SEQUENCES_PER_CLIENT,

            allow_row_order=
                args.allow_row_order,
        )


        all_manifest_rows.append(
            manifest_row
        )


        all_distribution_rows.extend(
            distribution_rows
        )


    # ========================================================
    # PREPARE VALIDATION SEQUENCES
    # ========================================================

    (
        validation_manifest,
        validation_distribution,

    ) = prepare_dataset(

        dataset_name=
            "validation",

        source_file=
            VALIDATION_FILE,

        output_file=
            VALIDATION_SEQUENCE_FILE,

        scaler=
            scaler,

        feature_columns=
            feature_columns,

        active_indices=
            active_indices,

        active_features=
            active_features,

        label_mapping=
            label_mapping,

        max_sequences=
            MAX_VALIDATION_SEQUENCES,

        allow_row_order=
            args.allow_row_order,
    )


    all_manifest_rows.append(
        validation_manifest
    )


    all_distribution_rows.extend(
        validation_distribution
    )


    # ========================================================
    # SAVE MANIFEST
    # ========================================================

    manifest_dataframe = pd.DataFrame(
        all_manifest_rows
    )


    manifest_dataframe.to_csv(

        SEQUENCE_MANIFEST_FILE,

        index=False,
    )


    # ========================================================
    # SAVE CLASS DISTRIBUTIONS
    # ========================================================

    distribution_dataframe = pd.DataFrame(
        all_distribution_rows
    )


    distribution_dataframe.to_csv(

        SEQUENCE_CLASS_DISTRIBUTION_FILE,

        index=False,
    )


    # ========================================================
    # CONFIG JSON
    # ========================================================

    total_seconds = (
        time.perf_counter()
        -
        total_start
    )


    config = {

        "phase":
            "14D",

        "component":
            "CNN-BiLSTM source-ordered sequence preparation",

        "scenario":
            SCENARIO,

        "seed":
            SEED,

        "feature_count":
            EXPECTED_FEATURES,

        "feature_names":
            list(
                active_features
            ),

        "sequence_length":
            SEQUENCE_LENGTH,

        "sequence_stride":
            SEQUENCE_STRIDE,

        "label_strategy":
            LABEL_STRATEGY,

        "target_definition":
            "class_of_last_flow_in_window",

        "sequence_semantics":
            "source_ordered_not_timestamp_confirmed",

        "phase14d0_source":
            str(
                TEMPORAL_ROOT
            ),

        "client_sequence_limit":
            MAX_SEQUENCES_PER_CLIENT,

        "validation_sequence_limit":
            MAX_VALIDATION_SEQUENCES,

        "allow_row_order":
            bool(
                args.allow_row_order
            ),

        "client_sequence_directory":
            str(
                CLIENT_SEQUENCE_ROOT
            ),

        "validation_sequence_file":
            str(
                VALIDATION_SEQUENCE_FILE
            ),

        "manifest_file":
            str(
                SEQUENCE_MANIFEST_FILE
            ),

        "class_distribution_file":
            str(
                SEQUENCE_CLASS_DISTRIBUTION_FILE
            ),

        "total_preparation_seconds":
            float(
                total_seconds
            ),

        "locked_test_path":
            str(
                LOCKED_TEST_FILE
            ),

        "locked_test_used":
            False,
    }


    save_json(

        SEQUENCE_CONFIG_FILE,

        config,
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    separator()

    print(
        "PHASE 14D COMPLETE"
    )

    separator()


    print(

        manifest_dataframe[
            [
                "dataset",
                "filtered_rows",
                "sequence_count",
                "sequence_length",
                "stride",
                "feature_count",
                "temporal_order_type",
                "temporal_source_column",
                "group_column",
                "file_size_mb",
            ]
        ]

        .to_string(
            index=False
        )
    )


    print()

    print(
        "CNN-BILSTM INPUT SHAPE"
    )

    print(
        "-" * 110
    )


    print(
        f"One sequence : "
        f"({SEQUENCE_LENGTH}, "
        f"{EXPECTED_FEATURES})"
    )


    print(
        "Training batch shape will be:"
    )


    print(
        "(batch_size, "
        f"{SEQUENCE_LENGTH}, "
        f"{EXPECTED_FEATURES})"
    )


    print()

    print(
        "CLIENT SEQUENCES:"
    )

    print(
        CLIENT_SEQUENCE_ROOT
    )


    print()

    print(
        "VALIDATION SEQUENCES:"
    )

    print(
        VALIDATION_SEQUENCE_FILE
    )


    print()

    print(
        "MANIFEST:"
    )

    print(
        SEQUENCE_MANIFEST_FILE
    )


    print()

    print(
        "CLASS DISTRIBUTION:"
    )

    print(
        SEQUENCE_CLASS_DISTRIBUTION_FILE
    )


    print()

    print(
        "CONFIG:"
    )

    print(
        SEQUENCE_CONFIG_FILE
    )


    print()

    print(
        f"Total preparation time : "
        f"{total_seconds:.2f}s"
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
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()