from __future__ import annotations

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
# PROJECT / IMPORTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

for p in (PROJECT_ROOT, SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    import phase9_fedavg70 as phase4
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "\nCould not import scripts\\phase9_fedavg70.py"
    ) from exc


# ============================================================
# PHASE 14E0 CONFIG
# ============================================================

PHASE_NAME = (
    "PHASE 14E0 — GLOBAL CNN-BILSTM SOURCE-ORDERED "
    "SEQUENCE PREPARATION"
)

SEED = 42
EXPECTED_FEATURES = 70
SEQUENCE_LENGTH = 20
SEQUENCE_STRIDE = 10
MAX_GLOBAL_TRAIN_SEQUENCES = 100_000

TEMPORAL_ROOT = PROJECT_ROOT / "data" / "temporal_phase14"
GLOBAL_TRAIN_SOURCE = TEMPORAL_ROOT / "train_source_ordered.csv"

SEQUENCE_ROOT = PROJECT_ROOT / "data" / "sequences"
GLOBAL_SEQUENCE_FILE = SEQUENCE_ROOT / "global_train_sequences.npz"

RESULT_ROOT = PROJECT_ROOT / "results" / "phase14"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "detection"

DISTRIBUTION_FILE = (
    RESULT_ROOT
    / "phase14e0_global_sequence_class_distribution.csv"
)

CAP_AUDIT_FILE = (
    RESULT_ROOT
    / "phase14e0_global_sequence_cap_audit.csv"
)

CONFIG_FILE = (
    ARTIFACT_ROOT
    / "phase14e0_global_sequence_config.json"
)

LOCKED_TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_locked.csv"
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("phase14e0")


# ============================================================
# HELPERS
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def separator() -> None:
    print()
    print("=" * 110)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
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


def save_json(path: Path, payload) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=4)


def resolve_benign_id(label_mapping) -> int:
    for label, class_id in label_mapping.items():
        if str(label).strip().upper() == "BENIGN":
            return int(class_id)
    raise KeyError("BENIGN not found in label_mapping.")


def verify_preprocessing(
    feature_columns,
    active_indices,
    active_features,
):
    if len(active_features) != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} active features, "
            f"found {len(active_features)}."
        )

    active_indices = np.asarray(active_indices, dtype=np.int64)

    reconstructed = [
        feature_columns[int(i)]
        for i in active_indices
    ]

    if reconstructed != list(active_features):
        raise ValueError(
            "active_features / active_indices order mismatch."
        )

    return active_indices


def filter_to_project_classes(
    df: pd.DataFrame,
    label_mapping,
) -> pd.DataFrame:
    if phase4.LABEL_COL not in df.columns:
        raise KeyError(
            f"Missing label column: {phase4.LABEL_COL}"
        )

    canonical = {
        str(label).strip().lower(): str(label).strip()
        for label in label_mapping.keys()
    }

    labels = (
        df[phase4.LABEL_COL]
        .astype(str)
        .str.strip()
    )

    keep = labels.str.lower().isin(canonical.keys())

    df = df.loc[keep].copy()

    df[phase4.LABEL_COL] = (
        df[phase4.LABEL_COL]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(canonical)
    )

    return df


# ============================================================
# WINDOW CREATION
# ============================================================

def windows_for_group(
    X_group: np.ndarray,
    y_group: np.ndarray,
    row_positions: np.ndarray,
):
    n = len(y_group)

    if n < SEQUENCE_LENGTH:
        return None, None, None

    starts = np.arange(
        0,
        n - SEQUENCE_LENGTH + 1,
        SEQUENCE_STRIDE,
        dtype=np.int64,
    )

    count = len(starts)

    X_seq = np.empty(
        (
            count,
            SEQUENCE_LENGTH,
            EXPECTED_FEATURES,
        ),
        dtype=np.float32,
    )

    y_seq = np.empty(
        count,
        dtype=np.int64,
    )

    end_rows = np.empty(
        count,
        dtype=np.int64,
    )

    for i, start in enumerate(starts):
        end = start + SEQUENCE_LENGTH

        X_seq[i] = X_group[start:end]
        y_seq[i] = y_group[end - 1]
        end_rows[i] = row_positions[end - 1]

    return X_seq, y_seq, end_rows


def create_global_sequences(
    X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
):
    grouped = (
        df.groupby(
            "source_file",
            sort=False,
            dropna=False,
        ).indices
    )

    print(f"Sequence groups           : {len(grouped):,}")

    all_X = []
    all_y = []
    all_end = []
    skipped = 0

    sorted_positions = np.arange(
        len(df),
        dtype=np.int64,
    )

    for positions in grouped.values():
        positions = np.asarray(
            positions,
            dtype=np.int64,
        )

        if len(positions) < SEQUENCE_LENGTH:
            skipped += 1
            continue

        X_seq, y_seq, end_seq = windows_for_group(
            X[positions],
            y[positions],
            sorted_positions[positions],
        )

        if X_seq is None:
            continue

        all_X.append(X_seq)
        all_y.append(y_seq)
        all_end.append(end_seq)

    print(f"Groups too small          : {skipped:,}")

    if not all_X:
        raise RuntimeError(
            "No valid global sequences were generated."
        )

    X_seq = np.concatenate(all_X, axis=0)
    y_seq = np.concatenate(all_y, axis=0)
    end_seq = np.concatenate(all_end, axis=0)

    del all_X, all_y, all_end
    gc.collect()

    return X_seq, y_seq, end_seq


# ============================================================
# MINORITY-PRESERVING CAP
# ============================================================

def class_counts(y: np.ndarray) -> dict[int, int]:
    unique, counts = np.unique(
        y,
        return_counts=True,
    )
    return {
        int(k): int(v)
        for k, v in zip(unique, counts)
    }


def apply_class_aware_cap(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    end_seq: np.ndarray,
    benign_id: int,
):
    benign_idx = np.flatnonzero(y_seq == benign_id)
    attack_idx = np.flatnonzero(y_seq != benign_id)

    audit = {
        "requested_cap": MAX_GLOBAL_TRAIN_SEQUENCES,
        "pre_cap_sequence_count": int(len(y_seq)),
        "pre_cap_benign_count": int(len(benign_idx)),
        "pre_cap_non_benign_count": int(len(attack_idx)),
        "pre_cap_class_counts_json": json.dumps(
            class_counts(y_seq),
            sort_keys=True,
        ),
        "cap_policy":
            "keep_all_non_benign_sample_benign_without_replacement",
        "oversampling": False,
        "synthetic_data": False,
    }

    if len(y_seq) <= MAX_GLOBAL_TRAIN_SEQUENCES:
        selected = np.arange(len(y_seq), dtype=np.int64)

    elif len(attack_idx) >= MAX_GLOBAL_TRAIN_SEQUENCES:
        logger.warning(
            "Attack sequences (%s) exceed/equal cap (%s). "
            "Keeping ALL attacks; final size may exceed cap.",
            f"{len(attack_idx):,}",
            f"{MAX_GLOBAL_TRAIN_SEQUENCES:,}",
        )
        selected = np.sort(attack_idx)

    else:
        benign_capacity = (
            MAX_GLOBAL_TRAIN_SEQUENCES
            -
            len(attack_idx)
        )

        rng = np.random.default_rng(SEED + 140500)

        benign_keep = min(
            len(benign_idx),
            benign_capacity,
        )

        selected_benign = rng.choice(
            benign_idx,
            size=benign_keep,
            replace=False,
        )

        selected = np.sort(
            np.concatenate(
                [
                    attack_idx,
                    selected_benign,
                ]
            )
        )

    X_seq = X_seq[selected]
    y_seq = y_seq[selected]
    end_seq = end_seq[selected]

    post_benign = int(np.sum(y_seq == benign_id))
    post_attack = int(np.sum(y_seq != benign_id))

    audit.update(
        {
            "post_cap_sequence_count": int(len(y_seq)),
            "post_cap_benign_count": post_benign,
            "post_cap_non_benign_count": post_attack,
            "benign_removed":
                int(len(benign_idx) - post_benign),
            "non_benign_removed":
                int(len(attack_idx) - post_attack),
            "post_cap_class_counts_json": json.dumps(
                class_counts(y_seq),
                sort_keys=True,
            ),
        }
    )

    if audit["non_benign_removed"] != 0:
        raise RuntimeError(
            "Class-aware cap removed attack sequences."
        )

    return X_seq, y_seq, end_seq, audit


# ============================================================
# DISTRIBUTION
# ============================================================

def make_distribution(
    y_seq: np.ndarray,
    label_mapping,
) -> pd.DataFrame:
    inverse_mapping = {
        int(v): str(k)
        for k, v in label_mapping.items()
    }

    rows = []

    unique, counts = np.unique(
        y_seq,
        return_counts=True,
    )

    for class_id, count in zip(unique, counts):
        rows.append(
            {
                "dataset": "global_train",
                "class_id": int(class_id),
                "class_name": inverse_mapping[int(class_id)],
                "sequences": int(count),
                "percentage":
                    float(
                        count
                        /
                        len(y_seq)
                        *
                        100.0
                    ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(SEED)

    SEQUENCE_ROOT.mkdir(
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
    print(PHASE_NAME)
    separator()

    print(f"Source file               : {GLOBAL_TRAIN_SOURCE}")
    print(f"Output file               : {GLOBAL_SEQUENCE_FILE}")
    print(f"Sequence length           : {SEQUENCE_LENGTH}")
    print(f"Sequence stride           : {SEQUENCE_STRIDE}")
    print(f"Feature count             : {EXPECTED_FEATURES}")
    print(
        f"Global sequence cap       : "
        f"{MAX_GLOBAL_TRAIN_SEQUENCES:,}"
    )

    print()
    print("DATA POLICY")
    print("-" * 110)
    print("1. Uses UNIQUE global Phase14D0 V2 TRAIN rows.")
    print("2. Does NOT pool the five client NPZ files.")
    print("3. Shared BENIGN client references are therefore not duplicated.")
    print("4. source_file is a hard sequence boundary.")
    print("5. record_id supplies stable source order.")
    print("6. ALL non-BENIGN sequences are retained.")
    print("7. BENIGN is downsampled only if needed.")
    print("8. Oversampling: NO.")
    print("9. Synthetic data: NO.")
    print("10. Locked test: NOT USED.")
    print(
        "11. SOURCE-ORDERED; not timestamp-confirmed temporal data."
    )

    if not GLOBAL_TRAIN_SOURCE.exists():
        raise FileNotFoundError(
            f"\nMissing:\n{GLOBAL_TRAIN_SOURCE}"
        )

    (
        scaler,
        feature_columns,
        label_mapping,
        active_indices,
        active_features,
    ) = phase4.load_preprocessing()

    active_indices = verify_preprocessing(
        feature_columns,
        active_indices,
        active_features,
    )

    benign_id = resolve_benign_id(
        label_mapping
    )

    start_time = time.perf_counter()

    logger.info(
        "Loading unique global source-ordered TRAIN CSV..."
    )

    df = pd.read_csv(
        GLOBAL_TRAIN_SOURCE,
        low_memory=False,
    )

    original_rows = len(df)

    df = filter_to_project_classes(
        df,
        label_mapping,
    )

    filtered_rows = len(df)

    required_order_columns = {
        "source_file",
        "record_id",
    }

    missing = required_order_columns.difference(
        df.columns
    )

    if missing:
        raise RuntimeError(
            "\nMissing Phase14D0 V2 ordering columns:\n"
            + "\n".join(sorted(missing))
        )

    df["__source_order"] = pd.to_numeric(
        df["record_id"],
        errors="coerce",
    )

    valid_ratio = float(
        df["__source_order"]
        .notna()
        .mean()
    )

    if valid_ratio < 0.99:
        raise ValueError(
            f"record_id numeric success too low: "
            f"{valid_ratio:.6f}"
        )

    df = (
        df[
            df["__source_order"].notna()
        ]
        .sort_values(
            [
                "source_file",
                "__source_order",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    print()
    print(f"Original rows             : {original_rows:,}")
    print(f"Six-class rows            : {filtered_rows:,}")
    print("Group boundary column     : source_file")
    print("Stable order column       : record_id")

    logger.info(
        "Transforming 70 active features..."
    )

    X = phase4.transform_features(
        df=df,
        feature_columns=feature_columns,
        scaler=scaler,
        active_indices=active_indices,
    ).astype(
        np.float32,
        copy=False,
    )

    y = phase4.encode_labels(
        df[phase4.LABEL_COL],
        label_mapping,
    ).astype(
        np.int64,
        copy=False,
    )

    if X.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected 70 features; found {X.shape[1]}."
        )

    logger.info(
        "Creating global source-ordered windows..."
    )

    X_seq, y_seq, end_seq = create_global_sequences(
        X,
        y,
        df,
    )

    print()
    print(
        f"Pre-cap sequence count    : "
        f"{len(y_seq):,}"
    )

    (
        X_seq,
        y_seq,
        end_seq,
        audit,
    ) = apply_class_aware_cap(
        X_seq,
        y_seq,
        end_seq,
        benign_id,
    )

    print()
    print("MINORITY-PRESERVING CAP AUDIT")
    print("-" * 110)
    print(
        f"Pre-cap sequences         : "
        f"{audit['pre_cap_sequence_count']:,}"
    )
    print(
        f"Post-cap sequences        : "
        f"{audit['post_cap_sequence_count']:,}"
    )
    print(
        f"Pre-cap BENIGN            : "
        f"{audit['pre_cap_benign_count']:,}"
    )
    print(
        f"Post-cap BENIGN           : "
        f"{audit['post_cap_benign_count']:,}"
    )
    print(
        f"Pre-cap non-BENIGN        : "
        f"{audit['pre_cap_non_benign_count']:,}"
    )
    print(
        f"Post-cap non-BENIGN       : "
        f"{audit['post_cap_non_benign_count']:,}"
    )
    print(
        f"BENIGN removed            : "
        f"{audit['benign_removed']:,}"
    )
    print(
        f"Non-BENIGN removed        : "
        f"{audit['non_benign_removed']:,}"
    )

    if tuple(X_seq.shape[1:]) != (
        SEQUENCE_LENGTH,
        EXPECTED_FEATURES,
    ):
        raise RuntimeError(
            f"Unexpected sequence shape: {X_seq.shape}"
        )

    logger.info(
        "Saving global training NPZ..."
    )

    np.savez_compressed(
        GLOBAL_SEQUENCE_FILE,
        X=X_seq,
        y=y_seq,
        end_row_index=end_seq,
        sequence_length=np.asarray(
            [SEQUENCE_LENGTH],
            dtype=np.int64,
        ),
        feature_count=np.asarray(
            [EXPECTED_FEATURES],
            dtype=np.int64,
        ),
        stride=np.asarray(
            [SEQUENCE_STRIDE],
            dtype=np.int64,
        ),
    )

    distribution = make_distribution(
        y_seq,
        label_mapping,
    )

    distribution.to_csv(
        DISTRIBUTION_FILE,
        index=False,
    )

    pd.DataFrame(
        [audit]
    ).to_csv(
        CAP_AUDIT_FILE,
        index=False,
    )

    print()
    print("GLOBAL TRAIN SEQUENCE DISTRIBUTION")
    print("-" * 110)
    print(
        distribution[
            [
                "class_name",
                "sequences",
                "percentage",
            ]
        ].to_string(index=False)
    )

    elapsed = (
        time.perf_counter()
        -
        start_time
    )

    file_size_mb = (
        GLOBAL_SEQUENCE_FILE.stat().st_size
        /
        1024
        /
        1024
    )

    config = {
        "phase": "14E0",
        "component":
            "unique_global_cnn_bilstm_source_ordered_sequence_preparation",
        "seed": SEED,
        "source_file": str(GLOBAL_TRAIN_SOURCE),
        "output_file": str(GLOBAL_SEQUENCE_FILE),
        "original_rows": original_rows,
        "filtered_rows": filtered_rows,
        "feature_count": EXPECTED_FEATURES,
        "feature_names": list(active_features),
        "sequence_length": SEQUENCE_LENGTH,
        "sequence_stride": SEQUENCE_STRIDE,
        "label_strategy": "last_flow",
        "target_definition": "class_of_last_flow_in_window",
        "group_column": "source_file",
        "order_column": "record_id",
        "max_global_train_sequences":
            MAX_GLOBAL_TRAIN_SEQUENCES,
        "cap_policy": audit["cap_policy"],
        "oversampling": False,
        "synthetic_data": False,
        "client_npz_files_pooled": False,
        "shared_benign_client_duplication": False,
        "locked_test_path": str(LOCKED_TEST_FILE),
        "locked_test_used": False,
        "scientific_label":
            "source_ordered_not_timestamp_confirmed",
        "distribution_file": str(DISTRIBUTION_FILE),
        "cap_audit_file": str(CAP_AUDIT_FILE),
        "elapsed_seconds": elapsed,
    }

    save_json(
        CONFIG_FILE,
        config,
    )

    separator()
    print("PHASE 14E0 COMPLETE")
    separator()

    print(f"Global X shape            : {X_seq.shape}")
    print(f"Global y shape            : {y_seq.shape}")
    print(
        f"Final sequence count      : "
        f"{len(y_seq):,}"
    )
    print(
        f"Sequence dimensions       : "
        f"({SEQUENCE_LENGTH}, {EXPECTED_FEATURES})"
    )
    print(f"NPZ size                  : {file_size_mb:.2f} MB")
    print(f"Preparation time          : {elapsed:.2f}s")

    print()
    print(f"Global training NPZ       : {GLOBAL_SEQUENCE_FILE}")
    print(f"Distribution              : {DISTRIBUTION_FILE}")
    print(f"Cap audit                 : {CAP_AUDIT_FILE}")
    print(f"Config                    : {CONFIG_FILE}")

    print()
    print("LOCKED TEST USED : NO")

    print()
    print("READY FOR PHASE 14E:")
    print("Centralized CNN-BiLSTM baseline training.")

    print()
    print("SCIENTIFIC LABEL:")
    print(
        "SOURCE-ORDERED flow sequences; "
        "NOT timestamp-confirmed temporal sequences."
    )

    del df, X, y, X_seq, y_seq, end_seq
    gc.collect()


if __name__ == "__main__":
    main()
