from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

history_file = (
    PROJECT_ROOT
    / "results"
    / "median"
    / "non_iid"
    / "median_round_history.csv"
)


if not history_file.exists():
    raise FileNotFoundError(
        f"History file not found:\n{history_file}"
    )


df = pd.read_csv(history_file)


# ============================================================
# ADD PHASE-4 COMPATIBILITY COLUMNS
# ============================================================

if "weighted_local_loss" not in df.columns:

    if "sample_weighted_local_loss" not in df.columns:
        raise ValueError(
            "sample_weighted_local_loss column is missing."
        )

    df["weighted_local_loss"] = (
        df["sample_weighted_local_loss"]
    )


if "weighted_local_accuracy" not in df.columns:

    if "sample_weighted_local_accuracy" not in df.columns:
        raise ValueError(
            "sample_weighted_local_accuracy column is missing."
        )

    df["weighted_local_accuracy"] = (
        df["sample_weighted_local_accuracy"]
    )


df.to_csv(
    history_file,
    index=False,
)


print("=" * 80)
print("PHASE 8E MEDIAN HISTORY FIXED")
print("=" * 80)

print()
print(f"File:")
print(history_file)

print()
print("Added/verified:")
print("  weighted_local_loss")
print("  weighted_local_accuracy")

print()
print("Rows:", len(df))
print()
print("Columns:")
for column in df.columns:
    print(" ", column)