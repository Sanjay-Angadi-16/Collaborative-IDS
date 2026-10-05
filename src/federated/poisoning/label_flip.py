from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd


PHASE17B_LABEL_FLIP_API_VERSION = "phase17b_label_flip_api_v1"


def _normalize_label(value: Any) -> str:
    return str(value).strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_indices(indices: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(indices, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _distribution(series: pd.Series) -> Dict[str, int]:
    counts = series.astype(str).str.strip().value_counts(dropna=False)
    return {
        str(label): int(count)
        for label, count in counts.items()
    }


@dataclass(frozen=True)
class LabelFlipAudit:
    api_version: str
    source_file: str
    poisoned_file: str
    source_sha256: str
    poisoned_sha256: str
    label_column: str
    benign_label: str
    source_attack_labels: list[str]
    seed: int
    poison_rate_requested: float
    poison_rate_actual: float
    total_rows: int
    eligible_attack_rows: int
    poisoned_rows: int
    selected_row_indices_sha256: str
    selected_row_indices_preview: list[int]
    class_distribution_before: Dict[str, int]
    class_distribution_after: Dict[str, int]
    features_modified: bool
    benign_rows_modified: int
    unexpected_label_changes: int
    original_file_modified: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def create_label_flipped_csv(
    *,
    source_csv: Path | str,
    destination_csv: Path | str,
    label_column: str,
    benign_label: str,
    poison_rate: float,
    seed: int,
    source_attack_labels: Optional[Sequence[str]] = None,
) -> LabelFlipAudit:
    """
    Create a deterministic label-flipped COPY of one client CSV.

    Policy
    ------
    - Original source CSV is never modified.
    - Feature columns are never modified.
    - Only eligible attack labels are changed.
    - Target label is BENIGN.
    - poison_rate is measured over eligible attack rows, not total rows.
    - For the same seed, rates 10% / 25% / 50% are nested because one
      deterministic permutation is generated and each experiment uses a prefix.
    """
    source_csv = Path(source_csv).resolve()
    destination_csv = Path(destination_csv).resolve()

    if not source_csv.exists():
        raise FileNotFoundError(source_csv)

    if source_csv == destination_csv:
        raise ValueError(
            "Refusing to poison in place. source_csv and destination_csv must differ."
        )

    if not (0.0 <= float(poison_rate) <= 1.0):
        raise ValueError("poison_rate must be between 0.0 and 1.0 inclusive.")

    original_sha256_before = _sha256_file(source_csv)

    frame = pd.read_csv(source_csv)
    if label_column not in frame.columns:
        raise KeyError(
            f"Label column {label_column!r} not found in {source_csv}. "
            f"Available columns={list(frame.columns)}"
        )

    original = frame.copy(deep=True)
    poisoned = frame.copy(deep=True)

    original_labels = original[label_column].map(_normalize_label)
    benign_norm = _normalize_label(benign_label)

    if source_attack_labels:
        allowed_sources = {
            _normalize_label(label)
            for label in source_attack_labels
        }
        if benign_norm in allowed_sources:
            raise ValueError("source_attack_labels must not include the benign label.")
        eligible_mask = original_labels.isin(allowed_sources)
    else:
        allowed_sources = set(
            original_labels.loc[original_labels != benign_norm].unique().tolist()
        )
        eligible_mask = original_labels != benign_norm

    eligible_indices = np.flatnonzero(eligible_mask.to_numpy())

    if len(eligible_indices) == 0 and poison_rate > 0.0:
        raise RuntimeError(
            "No eligible attack rows were found in the selected malicious client."
        )

    poison_count = int(len(eligible_indices) * float(poison_rate))

    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(eligible_indices)
    selected_indices = np.sort(permutation[:poison_count])

    if poison_count > 0:
        poisoned.loc[selected_indices, label_column] = benign_label

    # -----------------------------
    # Strict safety/audit assertions
    # -----------------------------
    feature_columns = [
        column
        for column in original.columns
        if column != label_column
    ]

    features_modified = not original[feature_columns].equals(
        poisoned[feature_columns]
    )

    benign_original_mask = original_labels == benign_norm
    poisoned_labels = poisoned[label_column].map(_normalize_label)

    benign_rows_modified = int(
        (
            original.loc[benign_original_mask, label_column].map(_normalize_label)
            != poisoned.loc[benign_original_mask, label_column].map(_normalize_label)
        ).sum()
    )

    changed_mask = original_labels != poisoned_labels
    changed_indices = np.flatnonzero(changed_mask.to_numpy())

    selected_set = set(int(index) for index in selected_indices.tolist())
    changed_set = set(int(index) for index in changed_indices.tolist())

    unexpected_indices = changed_set.symmetric_difference(selected_set)
    unexpected_label_changes = len(unexpected_indices)

    if features_modified:
        raise RuntimeError("Safety violation: feature columns changed during label flipping.")

    if benign_rows_modified != 0:
        raise RuntimeError(
            f"Safety violation: {benign_rows_modified} original BENIGN rows were modified."
        )

    if unexpected_label_changes != 0:
        raise RuntimeError(
            "Safety violation: changed label rows do not exactly match selected poison rows. "
            f"unexpected_count={unexpected_label_changes}"
        )

    if len(changed_indices) != poison_count:
        raise RuntimeError(
            f"Expected {poison_count} changed labels, found {len(changed_indices)}."
        )

    if poison_count > 0:
        if not np.all(
            poisoned_labels.iloc[selected_indices].to_numpy() == benign_norm
        ):
            raise RuntimeError("Not all selected attack rows were flipped to BENIGN.")

    destination_csv.parent.mkdir(parents=True, exist_ok=True)
    poisoned.to_csv(destination_csv, index=False)

    original_sha256_after = _sha256_file(source_csv)
    if original_sha256_after != original_sha256_before:
        raise RuntimeError("Original client CSV changed during poisoning. Refusing to continue.")

    actual_rate = (
        float(poison_count / len(eligible_indices))
        if len(eligible_indices)
        else 0.0
    )

    return LabelFlipAudit(
        api_version=PHASE17B_LABEL_FLIP_API_VERSION,
        source_file=str(source_csv),
        poisoned_file=str(destination_csv),
        source_sha256=original_sha256_before,
        poisoned_sha256=_sha256_file(destination_csv),
        label_column=str(label_column),
        benign_label=str(benign_label),
        source_attack_labels=sorted(str(x) for x in allowed_sources),
        seed=int(seed),
        poison_rate_requested=float(poison_rate),
        poison_rate_actual=float(actual_rate),
        total_rows=int(len(original)),
        eligible_attack_rows=int(len(eligible_indices)),
        poisoned_rows=int(poison_count),
        selected_row_indices_sha256=_sha256_indices(selected_indices),
        selected_row_indices_preview=[
            int(index)
            for index in selected_indices[:20].tolist()
        ],
        class_distribution_before=_distribution(original[label_column]),
        class_distribution_after=_distribution(poisoned[label_column]),
        features_modified=False,
        benign_rows_modified=0,
        unexpected_label_changes=0,
        original_file_modified=False,
    )
