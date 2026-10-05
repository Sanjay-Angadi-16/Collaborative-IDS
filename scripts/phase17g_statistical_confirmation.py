"""
PHASE 17G
Final 10-Seed Statistical Confirmation + Presentation Graphs

Compares:
    1. FedAvg
    2. Coordinate Median
    3. Trimmed Mean

Conditions:
    E0_CLEAN
    E_RANDOM_C1
    E_RANDOM_C1_C2
    E_SCALE_C1
    E_SCALE_C1_C2

Expected seeds:
    42, 52, 62, 72, 82, 92, 102, 112, 122, 132

Outputs:
    results/phase17/phase17g/
        phase17g_all_seed_results.csv
        phase17g_statistical_summary.csv
        phase17g_recovery_seedwise.csv
        phase17g_recovery_summary.csv
        phase17g_per_class_summary.csv
        phase17g_seedwise_highest_macro_f1.csv
        phase17g_highest_macro_f1_counts.csv
        phase17g_statistical_tests.csv
        phase17g_summary.json
        phase17g_final_report.txt

        graphs/
            01_macro_f1_comparison.png
            02_boxplot_e0_clean.png
            02_boxplot_e_random_c1.png
            02_boxplot_e_random_c1_c2.png
            02_boxplot_e_scale_c1.png
            02_boxplot_e_scale_c1_c2.png
            03_defense_improvement_vs_fedavg.png
            04_highest_macro_f1_count.png
            05_worst_attack_recall.png

This script DOES NOT train any model.
It only reads completed Phase 17F results.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_SEEDS = [
    42,
    52,
    62,
    72,
    82,
    92,
    102,
    112,
    122,
    132,
]

EXPECTED_EXPERIMENTS = [
    "E0_CLEAN",
    "E_RANDOM_C1",
    "E_RANDOM_C1_C2",
    "E_SCALE_C1",
    "E_SCALE_C1_C2",
]

EXPECTED_AGGREGATORS = [
    "fedavg",
    "coordinate_median",
    "trimmed_mean",
]

AGGREGATOR_NAMES = {
    "fedavg": "FedAvg",
    "coordinate_median": "Coordinate Median",
    "trimmed_mean": "Trimmed Mean",
}

EXPERIMENT_LABELS = {
    "E0_CLEAN": "Clean",
    "E_RANDOM_C1": "Random ×5\n20%",
    "E_RANDOM_C1_C2": "Random ×5\n40%",
    "E_SCALE_C1": "Scale ×5\n20%",
    "E_SCALE_C1_C2": "Scale ×5\n40%",
}

METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "worst_class_recall",
    "worst_attack_recall",
    "benign_false_positive_rate",
    "target_attack_mean_recall",
    "target_attack_worst_recall",
]


# ============================================================
# HELPERS
# ============================================================

def sample_sd(values):
    values = pd.Series(values).dropna().astype(float)
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1))


def confidence_interval_95(values):
    """
    95% confidence interval for the mean.

    Uses Student's t distribution if scipy is available.
    Falls back to common t critical values otherwise.
    """
    values = pd.Series(values).dropna().astype(float)
    n = len(values)

    if n == 0:
        return np.nan, np.nan

    mean = float(values.mean())

    if n == 1:
        return mean, mean

    sd = float(values.std(ddof=1))
    se = sd / math.sqrt(n)

    try:
        from scipy.stats import t
        t_critical = float(t.ppf(0.975, df=n - 1))
    except Exception:
        t_table = {
            1: 12.706,
            2: 4.303,
            3: 3.182,
            4: 2.776,
            5: 2.571,
            6: 2.447,
            7: 2.365,
            8: 2.306,
            9: 2.262,
            10: 2.228,
            11: 2.201,
            12: 2.179,
            13: 2.160,
            14: 2.145,
            15: 2.131,
            16: 2.120,
            17: 2.110,
            18: 2.101,
            19: 2.093,
            20: 2.086,
        }
        t_critical = t_table.get(n - 1, 1.96)

    margin = t_critical * se
    return mean - margin, mean + margin


def safe_bool_series(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps Phase 17G compatible with small naming differences
    between Phase 17F result-file revisions.
    """
    df = df.copy()

    aliases = {
        "condition": "experiment_id",
        "experiment": "experiment_id",
        "aggregation": "aggregator",
        "aggregation_method": "aggregator",
        "macro_f1_score": "macro_f1",
        "balanced_acc": "balanced_accuracy",
        "weighted_f1_score": "weighted_f1",
    }

    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    if "aggregator_display" not in df.columns and "aggregator" in df.columns:
        df["aggregator_display"] = (
            df["aggregator"]
            .map(AGGREGATOR_NAMES)
            .fillna(df["aggregator"].astype(str))
        )

    return df


# ============================================================
# LOAD PHASE 17F RESULTS
# ============================================================

def load_phase17f_results(phase17f_root: Path, seeds: list[int]):
    all_results = []
    per_class_results = []
    missing_files = []

    print()
    print("=" * 72)
    print("PHASE 17G — LOADING PHASE 17F RESULTS")
    print("=" * 72)

    for seed in seeds:
        seed_dir = phase17f_root / f"seed_{seed}"

        summary_file = (
            seed_dir / "phase17f_matched_defense_comparison.csv"
        )

        per_class_file = (
            seed_dir / "phase17f_per_class_comparison.csv"
        )

        print()
        print(f"Seed {seed}")

        if not summary_file.exists():
            print(f"  [MISSING] {summary_file}")
            missing_files.append(str(summary_file))
            continue

        summary_df = normalize_columns(pd.read_csv(summary_file))

        if "seed" not in summary_df.columns:
            summary_df["seed"] = seed

        summary_df["source_seed"] = seed
        all_results.append(summary_df)

        print(f"  Summary rows : {len(summary_df)}")

        if per_class_file.exists():
            class_df = normalize_columns(pd.read_csv(per_class_file))

            if "seed" not in class_df.columns:
                class_df["seed"] = seed

            class_df["source_seed"] = seed
            per_class_results.append(class_df)

            print(f"  Per-class rows: {len(class_df)}")
        else:
            print("  [WARNING] Per-class file missing")

    if missing_files:
        print()
        print("Missing Phase 17F files:")
        for file in missing_files:
            print("  ", file)

        raise FileNotFoundError(
            "One or more expected Phase 17F seed results are missing."
        )

    if not all_results:
        raise RuntimeError("No Phase 17F result files were found.")

    all_df = pd.concat(all_results, ignore_index=True)

    if per_class_results:
        class_df = pd.concat(per_class_results, ignore_index=True)
    else:
        class_df = pd.DataFrame()

    return all_df, class_df


# ============================================================
# VALIDATE EXPERIMENT MATRIX
# ============================================================

def validate_matrix(df: pd.DataFrame, seeds: list[int]):
    print()
    print("=" * 72)
    print("VALIDATING PHASE 17F MATRIX")
    print("=" * 72)

    required_columns = [
        "seed",
        "experiment_id",
        "aggregator",
        "macro_f1",
    ]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "Required Phase 17F columns missing: "
            + ", ".join(missing_columns)
        )

    errors = []

    expected_rows_per_seed = (
        len(EXPECTED_EXPERIMENTS) * len(EXPECTED_AGGREGATORS)
    )

    for seed in seeds:
        seed_df = df[df["seed"] == seed]

        print()
        print(f"Seed {seed}: {len(seed_df)} rows")

        if len(seed_df) != expected_rows_per_seed:
            errors.append(
                f"Seed {seed}: expected {expected_rows_per_seed} rows, "
                f"found {len(seed_df)}"
            )

        for experiment in EXPECTED_EXPERIMENTS:
            for aggregator in EXPECTED_AGGREGATORS:
                count = len(
                    seed_df[
                        (seed_df["experiment_id"] == experiment)
                        & (seed_df["aggregator"] == aggregator)
                    ]
                )

                if count != 1:
                    errors.append(
                        f"Seed={seed}, experiment={experiment}, "
                        f"aggregator={aggregator}, count={count}"
                    )

    if "attack_audit_pass" in df.columns:
        audit = safe_bool_series(df["attack_audit_pass"])
        if not audit.fillna(False).all():
            errors.append("At least one attack audit failed.")

    if "validation_poisoned" in df.columns:
        poisoned = safe_bool_series(df["validation_poisoned"])
        if poisoned.fillna(False).any():
            errors.append("Validation poisoning detected.")

    if "locked_test_used" in df.columns:
        locked = safe_bool_series(df["locked_test_used"])
        if locked.fillna(False).any():
            errors.append("Locked test set was used.")

    if "max_aggregation_audit_error" in df.columns:
        max_error = float(
            pd.to_numeric(
                df["max_aggregation_audit_error"],
                errors="coerce",
            )
            .fillna(0)
            .max()
        )

        print()
        print(
            "Maximum aggregation audit error:",
            f"{max_error:.12e}",
        )

    if errors:
        print()
        print("VALIDATION FAILED")

        for error in errors:
            print("  -", error)

        raise RuntimeError("Phase 17F matrix validation failed.")

    print()
    print("Matrix validation: PASS")
    print("Seeds             :", len(seeds))
    print("Experiments/seed  :", len(EXPECTED_EXPERIMENTS))
    print("Aggregators       :", len(EXPECTED_AGGREGATORS))
    print("Runs/seed         :", expected_rows_per_seed)
    print("Total runs        :", len(df))


# ============================================================
# STATISTICAL SUMMARY
# ============================================================

def build_statistical_summary(df: pd.DataFrame):
    rows = []

    group_columns = [
        "experiment_id",
        "aggregator",
        "aggregator_display",
    ]

    optional_group_columns = [
        "attack_type",
        "malicious_ratio",
    ]

    for col in optional_group_columns:
        if col in df.columns:
            group_columns.insert(-2, col)

    grouping = df.groupby(group_columns, dropna=False)

    for group_key, group in grouping:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        row = dict(zip(group_columns, group_key))
        row["n_seeds"] = int(group["seed"].nunique())

        for metric in METRICS:
            if metric not in group.columns:
                continue

            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            ).dropna()

            if len(values) == 0:
                continue

            ci_low, ci_high = confidence_interval_95(values)

            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = sample_sd(values)
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
            row[f"{metric}_ci95_low"] = ci_low
            row[f"{metric}_ci95_high"] = ci_high

        rows.append(row)

    result = pd.DataFrame(rows)

    experiment_order = {
        experiment: index
        for index, experiment in enumerate(EXPECTED_EXPERIMENTS)
    }

    aggregator_order = {
        aggregator: index
        for index, aggregator in enumerate(EXPECTED_AGGREGATORS)
    }

    result["_experiment_order"] = (
        result["experiment_id"].map(experiment_order)
    )
    result["_aggregator_order"] = (
        result["aggregator"].map(aggregator_order)
    )

    result = (
        result.sort_values(
            ["_experiment_order", "_aggregator_order"]
        )
        .drop(
            columns=[
                "_experiment_order",
                "_aggregator_order",
            ]
        )
        .reset_index(drop=True)
    )

    return result


# ============================================================
# RECOVERY / DEFENSE SUMMARY
# ============================================================

def build_recovery_summary(df: pd.DataFrame):
    rows = []

    for seed in sorted(df["seed"].unique()):
        seed_df = df[df["seed"] == seed]

        clean_fedavg = seed_df[
            (seed_df["experiment_id"] == "E0_CLEAN")
            & (seed_df["aggregator"] == "fedavg")
        ]

        if clean_fedavg.empty:
            continue

        clean_fedavg_f1 = float(
            clean_fedavg.iloc[0]["macro_f1"]
        )

        for experiment in EXPECTED_EXPERIMENTS:
            experiment_df = seed_df[
                seed_df["experiment_id"] == experiment
            ]

            fedavg_row = experiment_df[
                experiment_df["aggregator"] == "fedavg"
            ]

            if fedavg_row.empty:
                continue

            fedavg_f1 = float(
                fedavg_row.iloc[0]["macro_f1"]
            )

            attack_damage = clean_fedavg_f1 - fedavg_f1

            for _, row in experiment_df.iterrows():
                aggregator = row["aggregator"]
                macro_f1 = float(row["macro_f1"])
                improvement = macro_f1 - fedavg_f1

                clean_own = seed_df[
                    (seed_df["experiment_id"] == "E0_CLEAN")
                    & (seed_df["aggregator"] == aggregator)
                ]

                if not clean_own.empty:
                    own_clean_f1 = float(
                        clean_own.iloc[0]["macro_f1"]
                    )
                    own_clean_cost = (
                        own_clean_f1 - clean_fedavg_f1
                    )
                    attack_drop_from_own_clean = (
                        own_clean_f1 - macro_f1
                    )
                else:
                    own_clean_f1 = np.nan
                    own_clean_cost = np.nan
                    attack_drop_from_own_clean = np.nan

                if (
                    experiment != "E0_CLEAN"
                    and attack_damage > 0
                ):
                    damage_recovery_fraction = (
                        improvement / attack_damage
                    )
                    damage_recovery_pct = (
                        damage_recovery_fraction * 100
                    )
                else:
                    damage_recovery_fraction = np.nan
                    damage_recovery_pct = np.nan

                rows.append(
                    {
                        "seed": seed,
                        "experiment_id": experiment,
                        "attack_type": row.get(
                            "attack_type", np.nan
                        ),
                        "malicious_ratio": row.get(
                            "malicious_ratio", np.nan
                        ),
                        "aggregator": aggregator,
                        "aggregator_display": row.get(
                            "aggregator_display",
                            AGGREGATOR_NAMES.get(
                                aggregator, aggregator
                            ),
                        ),
                        "macro_f1": macro_f1,
                        "fedavg_macro_f1_same_condition": fedavg_f1,
                        "fedavg_clean_macro_f1": clean_fedavg_f1,
                        "own_clean_macro_f1": own_clean_f1,
                        "macro_f1_improvement_vs_fedavg": improvement,
                        "clean_macro_f1_cost_vs_fedavg": own_clean_cost,
                        "macro_f1_drop_from_own_clean": (
                            attack_drop_from_own_clean
                        ),
                        "fedavg_attack_damage": attack_damage,
                        "damage_recovery_fraction": (
                            damage_recovery_fraction
                        ),
                        "damage_recovery_pct": damage_recovery_pct,
                    }
                )

    raw = pd.DataFrame(rows)
    summary_rows = []

    grouping = raw.groupby(
        [
            "experiment_id",
            "attack_type",
            "malicious_ratio",
            "aggregator",
            "aggregator_display",
        ],
        dropna=False,
    )

    recovery_metrics = [
        "macro_f1_improvement_vs_fedavg",
        "clean_macro_f1_cost_vs_fedavg",
        "macro_f1_drop_from_own_clean",
        "damage_recovery_fraction",
        "damage_recovery_pct",
    ]

    for keys, group in grouping:
        (
            experiment,
            attack,
            ratio,
            aggregator,
            display,
        ) = keys

        row = {
            "experiment_id": experiment,
            "attack_type": attack,
            "malicious_ratio": ratio,
            "aggregator": aggregator,
            "aggregator_display": display,
            "n_seeds": int(group["seed"].nunique()),
        }

        for metric in recovery_metrics:
            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            ).dropna()

            if values.empty:
                continue

            low, high = confidence_interval_95(values)

            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = sample_sd(values)
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    return raw, summary


# ============================================================
# PER-CLASS SUMMARY
# ============================================================

def build_per_class_summary(class_df: pd.DataFrame):
    if class_df.empty:
        return pd.DataFrame()

    rows = []

    class_column = None

    for candidate in ["class", "class_name", "label_name", "label"]:
        if candidate in class_df.columns:
            class_column = candidate
            break

    if class_column is None:
        print(
            "[WARNING] Per-class CSV found but no class column detected. "
            "Per-class summary skipped."
        )
        return pd.DataFrame()

    group_columns = [
        "experiment_id",
        "aggregator",
        class_column,
    ]

    if "attack_type" in class_df.columns:
        group_columns.insert(1, "attack_type")

    if "malicious_ratio" in class_df.columns:
        group_columns.insert(2, "malicious_ratio")

    grouping = class_df.groupby(
        group_columns,
        dropna=False,
    )

    for keys, group in grouping:
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_columns, keys))
        row["class"] = row.pop(class_column)
        row["aggregator_display"] = AGGREGATOR_NAMES.get(
            row["aggregator"],
            row["aggregator"],
        )
        row["n_seeds"] = int(group["seed"].nunique())

        for metric in ["precision", "recall", "f1"]:
            if metric not in group.columns:
                continue

            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            ).dropna()

            if values.empty:
                continue

            low, high = confidence_interval_95(values)

            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = sample_sd(values)
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high

        if "support" in group.columns:
            row["support_mean"] = float(
                pd.to_numeric(
                    group["support"],
                    errors="coerce",
                ).mean()
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# SEED-WISE HIGHEST MACRO-F1
# ============================================================

def build_seedwise_highest_macro_f1(df: pd.DataFrame):
    rows = []

    for (seed, experiment), group in df.groupby(
        ["seed", "experiment_id"]
    ):
        max_f1 = float(group["macro_f1"].max())

        highest_rows = group[
            np.isclose(
                group["macro_f1"],
                max_f1,
                atol=1e-12,
            )
        ]

        for _, row in highest_rows.iterrows():
            rows.append(
                {
                    "seed": seed,
                    "experiment_id": experiment,
                    "attack_type": row.get(
                        "attack_type", np.nan
                    ),
                    "malicious_ratio": row.get(
                        "malicious_ratio", np.nan
                    ),
                    "aggregator": row["aggregator"],
                    "aggregator_display": row.get(
                        "aggregator_display",
                        AGGREGATOR_NAMES.get(
                            row["aggregator"],
                            row["aggregator"],
                        ),
                    ),
                    "macro_f1": row["macro_f1"],
                    "tie": len(highest_rows) > 1,
                }
            )

    return pd.DataFrame(rows)


def build_highest_count_summary(
    highest_df: pd.DataFrame,
):
    if highest_df.empty:
        return pd.DataFrame()

    return (
        highest_df
        .groupby(
            [
                "experiment_id",
                "aggregator",
                "aggregator_display",
            ]
        )
        .size()
        .reset_index(
            name="highest_macro_f1_count"
        )
    )


# ============================================================
# STATISTICAL SIGNIFICANCE TESTS
# ============================================================

def run_statistical_tests(df: pd.DataFrame):
    rows = []

    pairs = [
        ("fedavg", "coordinate_median"),
        ("fedavg", "trimmed_mean"),
        ("coordinate_median", "trimmed_mean"),
    ]

    try:
        from scipy.stats import (
            friedmanchisquare,
            wilcoxon,
        )
        scipy_available = True
    except Exception:
        scipy_available = False

    for experiment in EXPECTED_EXPERIMENTS:
        condition = df[
            df["experiment_id"] == experiment
        ]

        pivot = condition.pivot(
            index="seed",
            columns="aggregator",
            values="macro_f1",
        )

        if (
            scipy_available
            and all(
                agg in pivot.columns
                for agg in EXPECTED_AGGREGATORS
            )
        ):
            complete = pivot[
                EXPECTED_AGGREGATORS
            ].dropna()

            if len(complete) >= 3:
                statistic, p_value = friedmanchisquare(
                    complete["fedavg"],
                    complete["coordinate_median"],
                    complete["trimmed_mean"],
                )

                rows.append(
                    {
                        "experiment_id": experiment,
                        "test": "Friedman",
                        "comparison": (
                            "FedAvg vs Coordinate Median "
                            "vs Trimmed Mean"
                        ),
                        "n": len(complete),
                        "statistic": statistic,
                        "p_value": p_value,
                        "significant_0_05": p_value < 0.05,
                    }
                )

        for first, second in pairs:
            if (
                first not in pivot.columns
                or second not in pivot.columns
            ):
                continue

            paired = pivot[[first, second]].dropna()

            if len(paired) == 0:
                continue

            first_values = paired[first]
            second_values = paired[second]
            difference = second_values - first_values

            result_row = {
                "experiment_id": experiment,
                "test": "Wilcoxon signed-rank",
                "comparison": (
                    f"{AGGREGATOR_NAMES[first]} "
                    f"vs {AGGREGATOR_NAMES[second]}"
                ),
                "n": len(paired),
                "mean_difference_second_minus_first": float(
                    difference.mean()
                ),
                "median_difference_second_minus_first": float(
                    difference.median()
                ),
            }

            if scipy_available:
                try:
                    statistic, p_value = wilcoxon(
                        first_values,
                        second_values,
                        alternative="two-sided",
                        zero_method="wilcox",
                    )

                    result_row["statistic"] = statistic
                    result_row["p_value"] = p_value
                    result_row[
                        "significant_0_05"
                    ] = p_value < 0.05

                except ValueError:
                    result_row["statistic"] = np.nan
                    result_row["p_value"] = np.nan
                    result_row["significant_0_05"] = False
            else:
                result_row["statistic"] = np.nan
                result_row["p_value"] = np.nan
                result_row["significant_0_05"] = np.nan

            rows.append(result_row)

    return pd.DataFrame(rows), scipy_available


# ============================================================
# GRAPH HELPERS
# ============================================================

def save_figure(fig, path: Path):
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def graph_macro_f1_comparison(
    summary: pd.DataFrame,
    output_dir: Path,
):
    graph_dir = output_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    pivot = summary.pivot(
        index="experiment_id",
        columns="aggregator",
        values="macro_f1_mean",
    ).reindex(EXPECTED_EXPERIMENTS)

    errors = summary.pivot(
        index="experiment_id",
        columns="aggregator",
        values="macro_f1_sd",
    ).reindex(EXPECTED_EXPERIMENTS)

    x = np.arange(len(EXPECTED_EXPERIMENTS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, aggregator in enumerate(EXPECTED_AGGREGATORS):
        values = pivot[aggregator].values
        error_values = errors[aggregator].values
        positions = x + (i - 1) * width

        bars = ax.bar(
            positions,
            values,
            width,
            yerr=error_values,
            capsize=4,
            label=AGGREGATOR_NAMES[aggregator],
        )

        for bar, value in zip(bars, values):
            if not np.isnan(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(bar.get_height() + 0.018, 0.98),
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_ylabel("Macro F1")
    ax.set_xlabel("Experimental condition")
    ax.set_title(
        "Phase 17G — Byzantine Defense Comparison\n"
        "Mean Macro-F1 Across 10 Seeds"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            EXPERIMENT_LABELS[e]
            for e in EXPECTED_EXPERIMENTS
        ]
    )

    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    save_figure(
        fig,
        graph_dir / "01_macro_f1_comparison.png",
    )


def graph_macro_f1_boxplots(
    all_df: pd.DataFrame,
    output_dir: Path,
):
    graph_dir = output_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    for experiment in EXPECTED_EXPERIMENTS:
        subset = all_df[
            all_df["experiment_id"] == experiment
        ]

        values = []
        labels = []

        for aggregator in EXPECTED_AGGREGATORS:
            data = pd.to_numeric(
                subset[
                    subset["aggregator"] == aggregator
                ]["macro_f1"],
                errors="coerce",
            ).dropna()

            values.append(data.values)
            labels.append(
                AGGREGATOR_NAMES[aggregator]
            )

        fig, ax = plt.subplots(figsize=(8, 6))

        # matplotlib compatibility: try tick_labels first,
        # fall back to labels for older versions.
        try:
            ax.boxplot(
                values,
                tick_labels=labels,
                showmeans=True,
            )
        except TypeError:
            ax.boxplot(
                values,
                labels=labels,
                showmeans=True,
            )

        ax.set_ylabel("Macro F1")
        ax.set_ylim(0, 1.0)

        ax.set_title(
            "Macro-F1 Distribution Across 10 Seeds\n"
            + EXPERIMENT_LABELS[experiment].replace("\n", " ")
        )

        ax.grid(axis="y", alpha=0.25)

        filename = (
            "02_boxplot_"
            + experiment.lower()
            + ".png"
        )

        save_figure(
            fig,
            graph_dir / filename,
        )


def graph_defense_improvement(
    recovery_summary: pd.DataFrame,
    output_dir: Path,
):
    graph_dir = output_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    attack_experiments = [
        "E_RANDOM_C1",
        "E_RANDOM_C1_C2",
        "E_SCALE_C1",
        "E_SCALE_C1_C2",
    ]

    defenses = [
        "coordinate_median",
        "trimmed_mean",
    ]

    x = np.arange(len(attack_experiments))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))

    for index, defense in enumerate(defenses):
        values = []

        for experiment in attack_experiments:
            row = recovery_summary[
                (
                    recovery_summary["experiment_id"]
                    == experiment
                )
                & (
                    recovery_summary["aggregator"]
                    == defense
                )
            ]

            if row.empty:
                values.append(np.nan)
            else:
                values.append(
                    float(
                        row.iloc[0][
                            "macro_f1_improvement_vs_fedavg_mean"
                        ]
                    )
                )

        positions = x + (index - 0.5) * width

        bars = ax.bar(
            positions,
            values,
            width,
            label=AGGREGATOR_NAMES[defense],
        )

        for bar, value in zip(bars, values):
            if not np.isnan(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{value:+.3f}",
                    ha="center",
                    fontsize=9,
                )

    ax.axhline(0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            EXPERIMENT_LABELS[e]
            for e in attack_experiments
        ]
    )

    ax.set_ylabel(
        "Macro-F1 improvement over FedAvg"
    )
    ax.set_xlabel(
        "Byzantine attack condition"
    )
    ax.set_title(
        "Robust Aggregation Recovery Over FedAvg"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    save_figure(
        fig,
        graph_dir
        / "03_defense_improvement_vs_fedavg.png",
    )


def graph_highest_counts(
    highest_counts: pd.DataFrame,
    output_dir: Path,
):
    graph_dir = output_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    attack_experiments = [
        "E_RANDOM_C1",
        "E_RANDOM_C1_C2",
        "E_SCALE_C1",
        "E_SCALE_C1_C2",
    ]

    pivot = (
        highest_counts
        .pivot(
            index="experiment_id",
            columns="aggregator",
            values="highest_macro_f1_count",
        )
        .fillna(0)
        .reindex(attack_experiments)
        .fillna(0)
    )

    x = np.arange(len(attack_experiments))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))

    for index, aggregator in enumerate(
        EXPECTED_AGGREGATORS
    ):
        if aggregator in pivot.columns:
            values = pivot[aggregator].values
        else:
            values = np.zeros(
                len(attack_experiments)
            )

        positions = x + (index - 1) * width

        bars = ax.bar(
            positions,
            values,
            width,
            label=AGGREGATOR_NAMES[aggregator],
        )

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(int(value)),
                ha="center",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            EXPERIMENT_LABELS[e].replace("×5\n", "")
            for e in attack_experiments
        ]
    )

    ax.set_ylim(0, 11)
    ax.set_ylabel(
        "Number of seeds with highest Macro F1"
    )
    ax.set_xlabel(
        "Byzantine attack condition"
    )
    ax.set_title(
        "Defense Consistency Across 10 Seeds"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    save_figure(
        fig,
        graph_dir
        / "04_highest_macro_f1_count.png",
    )


def graph_worst_attack_recall(
    summary: pd.DataFrame,
    output_dir: Path,
):
    if (
        "worst_attack_recall_mean"
        not in summary.columns
    ):
        print(
            "[WARNING] worst_attack_recall_mean "
            "not available. Graph 5 skipped."
        )
        return

    graph_dir = output_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    pivot = summary.pivot(
        index="experiment_id",
        columns="aggregator",
        values="worst_attack_recall_mean",
    ).reindex(EXPECTED_EXPERIMENTS)

    x = np.arange(len(EXPECTED_EXPERIMENTS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))

    for index, aggregator in enumerate(
        EXPECTED_AGGREGATORS
    ):
        if aggregator not in pivot.columns:
            continue

        values = pivot[aggregator].values
        positions = x + (index - 1) * width

        bars = ax.bar(
            positions,
            values,
            width,
            label=AGGREGATOR_NAMES[aggregator],
        )

        for bar, value in zip(bars, values):
            if not np.isnan(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(bar.get_height() + 0.015, 0.98),
                    f"{value:.2f}",
                    ha="center",
                    fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            EXPERIMENT_LABELS[e]
            for e in EXPECTED_EXPERIMENTS
        ]
    )

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Worst attack-class recall")
    ax.set_xlabel("Experimental condition")
    ax.set_title(
        "Worst Attack-Class Recall Across Byzantine Conditions"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    save_figure(
        fig,
        graph_dir / "05_worst_attack_recall.png",
    )


def generate_phase17g_graphs(
    all_df,
    statistical_summary,
    recovery_summary,
    highest_counts,
    output_dir,
):
    print()
    print("=" * 72)
    print("GENERATING PHASE 17G PRESENTATION GRAPHS")
    print("=" * 72)

    graph_macro_f1_comparison(
        statistical_summary,
        output_dir,
    )

    graph_macro_f1_boxplots(
        all_df,
        output_dir,
    )

    graph_defense_improvement(
        recovery_summary,
        output_dir,
    )

    graph_highest_counts(
        highest_counts,
        output_dir,
    )

    graph_worst_attack_recall(
        statistical_summary,
        output_dir,
    )

    print()
    print("Graphs saved to:")
    print(output_dir / "graphs")


# ============================================================
# CONSOLE SUMMARY
# ============================================================

def print_macro_f1_summary(
    summary: pd.DataFrame,
):
    print()
    print("=" * 72)
    print("MACRO-F1 — 10-SEED SUMMARY")
    print("=" * 72)

    for experiment in EXPECTED_EXPERIMENTS:
        subset = summary[
            summary["experiment_id"] == experiment
        ]

        print()
        print(experiment)

        for _, row in subset.iterrows():
            mean = row.get(
                "macro_f1_mean",
                np.nan,
            )
            sd = row.get(
                "macro_f1_sd",
                np.nan,
            )
            low = row.get(
                "macro_f1_ci95_low",
                np.nan,
            )
            high = row.get(
                "macro_f1_ci95_high",
                np.nan,
            )

            print(
                f"  {row['aggregator_display']:<20} "
                f"{mean:.6f} ± {sd:.6f} "
                f"95% CI [{low:.6f}, {high:.6f}]"
            )


# ============================================================
# TEXT REPORT
# ============================================================

def build_final_report(
    df,
    statistical_summary,
    recovery_summary,
    highest_counts,
    tests,
    scipy_available,
):
    lines = []

    lines.append(
        "PHASE 17G COMPLETE — FINAL STATISTICAL CONFIRMATION"
    )
    lines.append("=" * 72)
    lines.append("")

    lines.append(
        f"Seeds evaluated      : {df['seed'].nunique()}"
    )
    lines.append(
        f"Total matched runs   : {len(df)}"
    )
    lines.append(
        "Aggregators          : "
        "FedAvg, Coordinate Median, Trimmed Mean"
    )
    lines.append(
        "Evaluation data      : Clean validation only"
    )
    lines.append(
        "Locked test used     : NO"
    )
    lines.append(
        "Validation poisoned  : NO"
    )

    lines.append("")
    lines.append("MACRO-F1 STATISTICAL SUMMARY")
    lines.append("-" * 72)

    for experiment in EXPECTED_EXPERIMENTS:
        lines.append("")
        lines.append(experiment)

        subset = statistical_summary[
            statistical_summary["experiment_id"]
            == experiment
        ]

        for _, row in subset.iterrows():
            lines.append(
                "  "
                f"{row['aggregator_display']:<20} "
                f"{row['macro_f1_mean']:.6f} "
                f"± {row['macro_f1_sd']:.6f} "
                f"95% CI "
                f"[{row['macro_f1_ci95_low']:.6f}, "
                f"{row['macro_f1_ci95_high']:.6f}]"
            )

    lines.append("")
    lines.append("HIGHEST MACRO-F1 COUNTS")
    lines.append("-" * 72)

    for experiment in EXPECTED_EXPERIMENTS:
        lines.append("")
        lines.append(experiment)

        subset = highest_counts[
            highest_counts["experiment_id"]
            == experiment
        ]

        for _, row in subset.iterrows():
            lines.append(
                f"  {row['aggregator_display']:<20} "
                f"{int(row['highest_macro_f1_count'])}"
            )

    lines.append("")
    lines.append("DEFENSE RECOVERY SUMMARY")
    lines.append("-" * 72)

    for experiment in EXPECTED_EXPERIMENTS:
        if experiment == "E0_CLEAN":
            continue

        lines.append("")
        lines.append(experiment)

        subset = recovery_summary[
            recovery_summary["experiment_id"]
            == experiment
        ]

        for _, row in subset.iterrows():
            improvement = row.get(
                "macro_f1_improvement_vs_fedavg_mean",
                np.nan,
            )
            recovery_pct = row.get(
                "damage_recovery_pct_mean",
                np.nan,
            )

            lines.append(
                f"  {row['aggregator_display']:<20} "
                f"ΔF1 vs FedAvg={improvement:+.6f}"
                f" | Damage recovery={recovery_pct:.2f}%"
            )

    lines.append("")
    lines.append("STATISTICAL TESTS")
    lines.append("-" * 72)

    if not scipy_available:
        lines.append(
            "SciPy not available. "
            "Statistical significance tests were skipped."
        )
    elif tests.empty:
        lines.append(
            "No valid statistical tests generated."
        )
    else:
        for _, row in tests.iterrows():
            p_value = row.get("p_value", np.nan)

            if pd.isna(p_value):
                p_text = "NA"
            else:
                p_text = f"{p_value:.6g}"

            lines.append(
                f"{row['experiment_id']} | "
                f"{row['test']} | "
                f"{row['comparison']} | "
                f"p={p_text}"
            )

    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 72)

    lines.append(
        "Phase 17G evaluates whether the Phase 17F defense "
        "comparison is stable across repeated random seeds."
    )
    lines.append(
        "Higher Macro-F1 under attack indicates better preservation "
        "of global IDS classification performance."
    )
    lines.append(
        "The results should be interpreted together with clean "
        "performance, worst attack recall, per-class recall, "
        "confidence intervals, and paired statistical tests."
    )
    lines.append(
        "A robust aggregation method mitigates Byzantine poisoning "
        "when it consistently preserves performance better than "
        "FedAvg across repeated seeds."
    )
    lines.append(
        "This constitutes experimental mitigation under the frozen "
        "Phase 17 threat model; it is not a claim of universal "
        "Byzantine immunity."
    )

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Phase 17G — Final statistical "
            "confirmation of Phase 17F"
        )
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(
            "results/phase17/phase17f"
        ),
        help="Phase 17F results root directory",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/phase17/phase17g"
        ),
        help="Phase 17G output directory",
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Seeds to analyse",
    )

    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # LOAD
    # ========================================================

    all_df, class_df = load_phase17f_results(
        phase17f_root=args.input_root,
        seeds=args.seeds,
    )

    # ========================================================
    # VALIDATE
    # ========================================================

    validate_matrix(
        all_df,
        args.seeds,
    )

    # ========================================================
    # STATISTICAL SUMMARY
    # ========================================================

    statistical_summary = build_statistical_summary(
        all_df
    )

    # ========================================================
    # RECOVERY
    # ========================================================

    recovery_raw, recovery_summary = (
        build_recovery_summary(all_df)
    )

    # ========================================================
    # PER CLASS
    # ========================================================

    per_class_summary = build_per_class_summary(
        class_df
    )

    # ========================================================
    # HIGHEST MACRO-F1 COUNTS
    # ========================================================

    highest_df = build_seedwise_highest_macro_f1(
        all_df
    )

    highest_counts = build_highest_count_summary(
        highest_df
    )

    # ========================================================
    # STATISTICAL TESTS
    # ========================================================

    tests, scipy_available = run_statistical_tests(
        all_df
    )

    # ========================================================
    # SAVE CSV FILES
    # ========================================================

    all_df.to_csv(
        output_dir / "phase17g_all_seed_results.csv",
        index=False,
    )

    statistical_summary.to_csv(
        output_dir / "phase17g_statistical_summary.csv",
        index=False,
    )

    recovery_raw.to_csv(
        output_dir / "phase17g_recovery_seedwise.csv",
        index=False,
    )

    recovery_summary.to_csv(
        output_dir / "phase17g_recovery_summary.csv",
        index=False,
    )

    if not per_class_summary.empty:
        per_class_summary.to_csv(
            output_dir / "phase17g_per_class_summary.csv",
            index=False,
        )

    highest_df.to_csv(
        output_dir
        / "phase17g_seedwise_highest_macro_f1.csv",
        index=False,
    )

    highest_counts.to_csv(
        output_dir
        / "phase17g_highest_macro_f1_counts.csv",
        index=False,
    )

    tests.to_csv(
        output_dir / "phase17g_statistical_tests.csv",
        index=False,
    )

    # ========================================================
    # JSON SUMMARY
    # ========================================================

    if "attack_audit_pass" in all_df.columns:
        attack_audits_pass = bool(
            safe_bool_series(
                all_df["attack_audit_pass"]
            )
            .fillna(False)
            .all()
        )
    else:
        attack_audits_pass = None

    summary_json = {
        "phase": "17G",
        "title": (
            "Final 10-Seed Statistical Confirmation"
        ),
        "seeds": [
            int(seed)
            for seed in args.seeds
        ],
        "number_of_seeds": len(args.seeds),
        "total_runs": int(len(all_df)),
        "experiments": EXPECTED_EXPERIMENTS,
        "aggregators": EXPECTED_AGGREGATORS,
        "validation_poisoned": False,
        "locked_test_used": False,
        "attack_audits_pass": attack_audits_pass,
        "scipy_available": scipy_available,
        "graphs_generated": True,
        "status": "PASS",
    }

    with open(
        output_dir / "phase17g_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary_json,
            file,
            indent=4,
        )

    # ========================================================
    # REPORT
    # ========================================================

    report = build_final_report(
        df=all_df,
        statistical_summary=statistical_summary,
        recovery_summary=recovery_summary,
        highest_counts=highest_counts,
        tests=tests,
        scipy_available=scipy_available,
    )

    report_file = (
        output_dir
        / "phase17g_final_report.txt"
    )

    report_file.write_text(
        report,
        encoding="utf-8",
    )

    # ========================================================
    # GRAPHS
    # ========================================================

    generate_phase17g_graphs(
        all_df=all_df,
        statistical_summary=statistical_summary,
        recovery_summary=recovery_summary,
        highest_counts=highest_counts,
        output_dir=output_dir,
    )

    # ========================================================
    # CONSOLE
    # ========================================================

    print_macro_f1_summary(
        statistical_summary
    )

    print()
    print("=" * 72)
    print("HIGHEST MACRO-F1 COUNT")
    print("=" * 72)

    print(
        highest_counts.to_string(
            index=False
        )
    )

    print()
    print("=" * 72)
    print("PHASE 17G COMPLETE")
    print("=" * 72)

    print("Input root :", args.input_root)
    print("Output dir :", output_dir)
    print("Seeds      :", args.seeds)
    print("Total runs :", len(all_df))
    print("Status     : PASS")

    print()
    print("Generated files:")

    for file in sorted(
        output_dir.iterdir()
    ):
        print("  ", file)

    graph_dir = output_dir / "graphs"

    if graph_dir.exists():
        print()
        print("Generated graphs:")
        for file in sorted(
            graph_dir.iterdir()
        ):
            print("  ", file)


if __name__ == "__main__":
    main()
