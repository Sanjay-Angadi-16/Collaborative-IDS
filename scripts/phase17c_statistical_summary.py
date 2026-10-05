from __future__ import annotations

"""
Phase 17C — Repeated-Seed Statistical Summary
=============================================

Aggregates the completed Phase 17C Sign-Flip Byzantine experiments across
multiple random seeds without modifying any training artifacts.

Expected Phase 17C layout
-------------------------
results/phase17/phase17c/
    seed_42/
        phase17c_clean_vs_sign_flip_comparison.csv
        C0_CLEAN/per_class_validation.csv
        C_SIGN_FLIP_X1/per_class_validation.csv
        C_SIGN_FLIP_X2/per_class_validation.csv
        C_SIGN_FLIP_X5/per_class_validation.csv
    seed_52/
    seed_62/
    seed_72/
    seed_82/

Outputs
-------
results/phase17/phase17c/statistical_summary/
    phase17c_all_seed_results.csv
    phase17c_statistical_summary.csv
    phase17c_per_class_all_seeds.csv
    phase17c_per_class_statistical_summary.csv
    phase17c_summary.json
    phase17c_final_report.txt

Statistics
----------
For each experiment (Clean, Sign-Flip x1, x2, x5), the script computes:
    - N
    - Mean
    - Sample standard deviation (ddof=1)
    - Min / Max
    - 95% confidence interval for the mean

For the default five seeds, the CI uses Student's t critical value for df=4.
No SciPy dependency is required.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# DEFAULTS
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "phase17" / "phase17c"
DEFAULT_SEEDS = [42, 52, 62, 72, 82]
DEFAULT_EXPERIMENTS = [
    "C0_CLEAN",
    "C_SIGN_FLIP_X1",
    "C_SIGN_FLIP_X2",
    "C_SIGN_FLIP_X5",
]

# Metrics that should be summarized across seeds.
CORE_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "worst_class_recall",
    "worst_attack_recall",
    "benign_false_positive_rate",
    "training_time_seconds",
]

DROP_METRICS = [
    "accuracy_drop",
    "balanced_accuracy_drop",
    "macro_f1_drop",
    "macro_f1_relative_drop_pct",
    "weighted_f1_drop",
    "worst_class_recall_drop",
    "worst_attack_recall_drop",
    "benign_fpr_increase",
]

ATTACK_AUDIT_METRICS = [
    "mean_malicious_honest_update_norm",
    "mean_malicious_routed_update_norm",
    "max_sign_flip_norm_error",
]

PER_CLASS_METRICS = ["precision", "recall", "f1"]


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def t_critical_95(df: int) -> float:
    """
    Two-sided Student-t critical value for a 95% CI.

    Covers the small sample sizes normally used in repeated-seed ML studies.
    Falls back to the normal 1.96 approximation for larger df.
    """
    table = {
        1: 12.706205,
        2: 4.302653,
        3: 3.182446,
        4: 2.776445,
        5: 2.570582,
        6: 2.446912,
        7: 2.364624,
        8: 2.306004,
        9: 2.262157,
        10: 2.228139,
        11: 2.200985,
        12: 2.178813,
        13: 2.160369,
        14: 2.144787,
        15: 2.131450,
        16: 2.119905,
        17: 2.109816,
        18: 2.100922,
        19: 2.093024,
        20: 2.085963,
        21: 2.079614,
        22: 2.073873,
        23: 2.068658,
        24: 2.063899,
        25: 2.059539,
        26: 2.055529,
        27: 2.051831,
        28: 2.048407,
        29: 2.045230,
        30: 2.042272,
    }
    if df <= 0:
        return float("nan")
    return table.get(df, 1.959964)


def summarize_series(values: pd.Series) -> Dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    n = int(clean.size)

    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "ci95_half_width": float("nan"),
        }

    mean = float(clean.mean())
    minimum = float(clean.min())
    maximum = float(clean.max())

    if n == 1:
        std = 0.0
        half = 0.0
    else:
        std = float(clean.std(ddof=1))
        critical = t_critical_95(n - 1)
        half = float(critical * std / math.sqrt(n))

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "min": minimum,
        "max": maximum,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "ci95_half_width": half,
    }


def pct(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{100.0 * value:.2f}%"


def metric_label(metric: str) -> str:
    labels = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced Accuracy",
        "macro_precision": "Macro Precision",
        "macro_recall": "Macro Recall",
        "macro_f1": "Macro F1",
        "weighted_f1": "Weighted F1",
        "worst_class_recall": "Worst Class Recall",
        "worst_attack_recall": "Worst Attack Recall",
        "benign_false_positive_rate": "Benign FPR",
        "training_time_seconds": "Training Time (s)",
        "accuracy_drop": "Accuracy Drop",
        "balanced_accuracy_drop": "Balanced Accuracy Drop",
        "macro_f1_drop": "Macro F1 Drop",
        "macro_f1_relative_drop_pct": "Macro F1 Relative Drop (%)",
        "weighted_f1_drop": "Weighted F1 Drop",
        "worst_class_recall_drop": "Worst Class Recall Drop",
        "worst_attack_recall_drop": "Worst Attack Recall Drop",
        "benign_fpr_increase": "Benign FPR Increase",
        "mean_malicious_honest_update_norm": "Mean Honest Update Norm",
        "mean_malicious_routed_update_norm": "Mean Routed Update Norm",
        "max_sign_flip_norm_error": "Max Sign-Flip Norm Error",
    }
    return labels.get(metric, metric)


def experiment_label(experiment_id: str) -> str:
    labels = {
        "C0_CLEAN": "Clean",
        "C_SIGN_FLIP_X1": "Sign-Flip x1",
        "C_SIGN_FLIP_X2": "Sign-Flip x2",
        "C_SIGN_FLIP_X5": "Sign-Flip x5",
    }
    return labels.get(experiment_id, experiment_id)


# ---------------------------------------------------------------------------
# LOAD + VALIDATE
# ---------------------------------------------------------------------------

def load_seed_comparison(results_root: Path, seed: int) -> pd.DataFrame:
    path = (
        results_root
        / f"seed_{seed}"
        / "phase17c_clean_vs_sign_flip_comparison.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing Phase 17C comparison file for seed {seed}:\n{path}"
        )

    df = pd.read_csv(path)
    df["seed"] = int(seed)
    df["source_file"] = str(path)
    return df


def validate_seed_frame(
    df: pd.DataFrame,
    *,
    seed: int,
    expected_experiments: Iterable[str],
    norm_error_tolerance: float,
) -> None:
    required_columns = {
        "experiment_id",
        "seed",
        "scenario",
        "attack_type",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "worst_attack_recall",
        "validation_poisoned",
        "locked_test_used",
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Seed {seed} comparison CSV is missing columns: "
            f"{sorted(missing_columns)}"
        )

    observed = set(df["experiment_id"].astype(str))
    expected = set(expected_experiments)
    missing_experiments = expected - observed

    if missing_experiments:
        raise ValueError(
            f"Seed {seed} is missing experiments: {sorted(missing_experiments)}"
        )

    duplicates = df["experiment_id"].astype(str).duplicated()
    if duplicates.any():
        duplicate_ids = sorted(
            df.loc[duplicates, "experiment_id"].astype(str).unique()
        )
        raise ValueError(
            f"Seed {seed} contains duplicate experiment rows: {duplicate_ids}"
        )

    # Audit requirement: validation must remain clean.
    poisoned = df["validation_poisoned"].fillna(False).astype(bool)
    if poisoned.any():
        bad = df.loc[poisoned, "experiment_id"].astype(str).tolist()
        raise RuntimeError(
            f"Seed {seed}: validation_poisoned=True for {bad}."
        )

    # Audit requirement: the locked test set must not be used in Phase 17C.
    locked = df["locked_test_used"].fillna(False).astype(bool)
    if locked.any():
        bad = df.loc[locked, "experiment_id"].astype(str).tolist()
        raise RuntimeError(
            f"Seed {seed}: locked_test_used=True for {bad}."
        )

    # Sign-flip norm consistency check.
    if "max_sign_flip_norm_error" in df.columns:
        attacked = df[df["attack_type"].astype(str) == "SIGN_FLIP"]
        max_error = pd.to_numeric(
            attacked["max_sign_flip_norm_error"], errors="coerce"
        ).max()

        if pd.notna(max_error) and float(max_error) > norm_error_tolerance:
            raise RuntimeError(
                f"Seed {seed}: max sign-flip norm error {max_error:.12g} "
                f"exceeds tolerance {norm_error_tolerance:.12g}."
            )


def load_per_class(
    results_root: Path,
    seeds: Iterable[int],
    experiments: Iterable[str],
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for seed in seeds:
        for experiment_id in experiments:
            path = (
                results_root
                / f"seed_{seed}"
                / experiment_id
                / "per_class_validation.csv"
            )

            if not path.exists():
                raise FileNotFoundError(
                    f"Missing per-class file:\n{path}"
                )

            df = pd.read_csv(path)

            if "class" not in df.columns:
                raise ValueError(
                    f"Missing 'class' column in {path}"
                )

            df.insert(0, "seed", int(seed))
            df.insert(1, "experiment_id", experiment_id)
            df.insert(2, "experiment", experiment_label(experiment_id))
            frames.append(df)

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# AGGREGATION
# ---------------------------------------------------------------------------

def build_statistical_summary(
    all_results: pd.DataFrame,
    experiments: Iterable[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    metrics = CORE_METRICS + DROP_METRICS + ATTACK_AUDIT_METRICS

    for experiment_id in experiments:
        subset = all_results[
            all_results["experiment_id"].astype(str) == experiment_id
        ]

        for metric in metrics:
            if metric not in subset.columns:
                continue

            stats = summarize_series(subset[metric])

            # Skip a metric if every value is absent for this experiment.
            if stats["n"] == 0:
                continue

            row: Dict[str, Any] = {
                "experiment_id": experiment_id,
                "experiment": experiment_label(experiment_id),
                "metric": metric,
                "metric_label": metric_label(metric),
                **stats,
            }
            rows.append(row)

    return pd.DataFrame(rows)


def build_per_class_summary(per_class: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for (experiment_id, class_name), group in per_class.groupby(
        ["experiment_id", "class"], sort=False
    ):
        for metric in PER_CLASS_METRICS:
            if metric not in group.columns:
                continue

            stats = summarize_series(group[metric])
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "experiment": experiment_label(experiment_id),
                    "class": class_name,
                    "metric": metric,
                    **stats,
                }
            )

    return pd.DataFrame(rows)


def mean_for(
    summary: pd.DataFrame,
    experiment_id: str,
    metric: str,
) -> float:
    match = summary[
        (summary["experiment_id"] == experiment_id)
        & (summary["metric"] == metric)
    ]
    if match.empty:
        return float("nan")
    return float(match.iloc[0]["mean"])


def std_for(
    summary: pd.DataFrame,
    experiment_id: str,
    metric: str,
) -> float:
    match = summary[
        (summary["experiment_id"] == experiment_id)
        & (summary["metric"] == metric)
    ]
    if match.empty:
        return float("nan")
    return float(match.iloc[0]["std"])


# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------

def build_text_report(
    *,
    all_results: pd.DataFrame,
    summary: pd.DataFrame,
    per_class_summary: pd.DataFrame,
    seeds: List[int],
    scenario: str,
    norm_error_tolerance: float,
) -> str:
    lines: List[str] = []

    lines.append("=" * 92)
    lines.append("PHASE 17C — REPEATED-SEED STATISTICAL SUMMARY")
    lines.append("=" * 92)
    lines.append("")
    lines.append(f"Scenario             : {scenario}")
    lines.append(f"Seeds                : {', '.join(map(str, seeds))}")
    lines.append(f"Number of seeds      : {len(seeds)}")
    lines.append("Attack               : Byzantine model-update sign flip")
    lines.append("Formula              : delta_attack = -scale * delta")
    lines.append("Validation poisoned  : NO")
    lines.append("Locked test used     : NO")
    lines.append("")

    lines.append("OVERALL PERFORMANCE (mean ± sample SD)")
    lines.append("-" * 92)
    lines.append(
        f"{'Experiment':<18} {'Accuracy':>17} {'Balanced Acc':>17} "
        f"{'Macro F1':>17} {'Weighted F1':>17}"
    )

    for experiment_id in DEFAULT_EXPERIMENTS:
        if experiment_id not in set(all_results["experiment_id"].astype(str)):
            continue

        acc_m = mean_for(summary, experiment_id, "accuracy")
        acc_s = std_for(summary, experiment_id, "accuracy")
        bal_m = mean_for(summary, experiment_id, "balanced_accuracy")
        bal_s = std_for(summary, experiment_id, "balanced_accuracy")
        f1_m = mean_for(summary, experiment_id, "macro_f1")
        f1_s = std_for(summary, experiment_id, "macro_f1")
        wf1_m = mean_for(summary, experiment_id, "weighted_f1")
        wf1_s = std_for(summary, experiment_id, "weighted_f1")

        lines.append(
            f"{experiment_label(experiment_id):<18} "
            f"{acc_m:.6f} ± {acc_s:.6f}  "
            f"{bal_m:.6f} ± {bal_s:.6f}  "
            f"{f1_m:.6f} ± {f1_s:.6f}  "
            f"{wf1_m:.6f} ± {wf1_s:.6f}"
        )

    lines.append("")
    lines.append("ATTACK DEGRADATION VS SAME-SEED CLEAN BASELINE")
    lines.append("-" * 92)

    for experiment_id in [
        "C_SIGN_FLIP_X1",
        "C_SIGN_FLIP_X2",
        "C_SIGN_FLIP_X5",
    ]:
        if experiment_id not in set(all_results["experiment_id"].astype(str)):
            continue

        acc_drop = mean_for(summary, experiment_id, "accuracy_drop")
        bal_drop = mean_for(summary, experiment_id, "balanced_accuracy_drop")
        f1_drop = mean_for(summary, experiment_id, "macro_f1_drop")
        rel_drop = mean_for(
            summary,
            experiment_id,
            "macro_f1_relative_drop_pct",
        )
        worst_drop = mean_for(
            summary,
            experiment_id,
            "worst_attack_recall_drop",
        )

        lines.append(
            f"{experiment_label(experiment_id):<18} | "
            f"Accuracy drop={acc_drop:.6f} | "
            f"Balanced-Acc drop={bal_drop:.6f} | "
            f"Macro-F1 drop={f1_drop:.6f} | "
            f"Relative Macro-F1 drop={rel_drop:.2f}% | "
            f"Worst-attack-recall drop={worst_drop:.6f}"
        )

    lines.append("")
    lines.append("BYZANTINE UPDATE AUDIT")
    lines.append("-" * 92)

    attacked = all_results[
        all_results["attack_type"].astype(str) == "SIGN_FLIP"
    ].copy()

    max_norm_error = pd.to_numeric(
        attacked.get("max_sign_flip_norm_error", pd.Series(dtype=float)),
        errors="coerce",
    ).max()

    if pd.notna(max_norm_error):
        lines.append(f"Maximum norm-check error : {float(max_norm_error):.12g}")
        lines.append(f"Allowed tolerance        : {norm_error_tolerance:.12g}")
        lines.append(
            "Norm validation          : "
            + (
                "PASS"
                if float(max_norm_error) <= norm_error_tolerance
                else "FAIL"
            )
        )

    lines.append("")
    lines.append("WORST ATTACK-CLASS RECALL")
    lines.append("-" * 92)

    for experiment_id in DEFAULT_EXPERIMENTS:
        value = mean_for(summary, experiment_id, "worst_attack_recall")
        if not pd.isna(value):
            lines.append(
                f"{experiment_label(experiment_id):<18}: {value:.6f}"
            )

    # Identify classes with mean recall == 0 under each attack.
    lines.append("")
    lines.append("CLASSES WITH ZERO MEAN RECALL")
    lines.append("-" * 92)

    recall_rows = per_class_summary[
        per_class_summary["metric"] == "recall"
    ]

    for experiment_id in [
        "C_SIGN_FLIP_X1",
        "C_SIGN_FLIP_X2",
        "C_SIGN_FLIP_X5",
    ]:
        subset = recall_rows[
            recall_rows["experiment_id"] == experiment_id
        ]
        zero_classes = subset.loc[
            np.isclose(subset["mean"].astype(float), 0.0, atol=1e-12),
            "class",
        ].astype(str).tolist()

        lines.append(
            f"{experiment_label(experiment_id):<18}: "
            + (", ".join(zero_classes) if zero_classes else "None")
        )

    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 92)
    lines.append(
        "Increasing sign-flip magnitude causes progressively stronger degradation of the "
        "FedAvg global model. The x1 condition already causes a substantial reduction in "
        "balanced accuracy and Macro F1, while x2 produces severe class-wise degradation. "
        "The x5 condition represents an extreme stress-test regime and can drive the model "
        "toward near-collapse."
    )
    lines.append("")
    lines.append(
        "The repeated-seed analysis should be used for the final Phase 17C conclusion rather "
        "than relying on a single seed."
    )
    lines.append("")
    lines.append("STATUS: PASS")
    lines.append("=" * 92)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 17C repeated-seed statistical summary"
    )

    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=(
            "Phase 17C results root containing seed_42, seed_52, ... "
            f"Default: {DEFAULT_RESULTS_ROOT}"
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Seeds to aggregate. Default: 42 52 62 72 82",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Summary output directory. Default: "
            "<results-root>/statistical_summary"
        ),
    )
    parser.add_argument(
        "--norm-error-tolerance",
        type=float,
        default=1e-5,
        help="Maximum allowed sign-flip update-norm verification error.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results_root = args.results_root.resolve()
    seeds = [int(seed) for seed in args.seeds]
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else results_root / "statistical_summary"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(seeds) < 2:
        raise ValueError(
            "Repeated-seed statistical analysis requires at least two seeds."
        )

    print("=" * 92)
    print("PHASE 17C — REPEATED-SEED STATISTICAL SUMMARY")
    print("=" * 92)
    print(f"Results root : {results_root}")
    print(f"Seeds        : {seeds}")
    print(f"Output dir   : {output_dir}")
    print()

    seed_frames: List[pd.DataFrame] = []

    for seed in seeds:
        df = load_seed_comparison(results_root, seed)
        validate_seed_frame(
            df,
            seed=seed,
            expected_experiments=DEFAULT_EXPERIMENTS,
            norm_error_tolerance=float(args.norm_error_tolerance),
        )
        seed_frames.append(df)
        print(f"seed_{seed}: PASS")

    all_results = pd.concat(seed_frames, ignore_index=True)

    # Keep rows in a reproducible experiment/seed order.
    experiment_order = {
        experiment_id: index
        for index, experiment_id in enumerate(DEFAULT_EXPERIMENTS)
    }
    all_results["_experiment_order"] = all_results["experiment_id"].map(
        experiment_order
    )
    all_results = all_results.sort_values(
        ["_experiment_order", "seed"]
    ).drop(columns=["_experiment_order"])

    observed_scenarios = sorted(
        all_results["scenario"].dropna().astype(str).unique().tolist()
    )

    if len(observed_scenarios) != 1:
        raise RuntimeError(
            "Expected one consistent scenario across all seeds; found "
            f"{observed_scenarios}"
        )

    scenario = observed_scenarios[0]

    # Main repeated-seed statistics.
    summary = build_statistical_summary(
        all_results,
        DEFAULT_EXPERIMENTS,
    )

    # Per-class repeated-seed statistics.
    per_class_all = load_per_class(
        results_root,
        seeds,
        DEFAULT_EXPERIMENTS,
    )
    per_class_summary = build_per_class_summary(per_class_all)

    # Write machine-readable tables.
    all_results_path = output_dir / "phase17c_all_seed_results.csv"
    summary_path = output_dir / "phase17c_statistical_summary.csv"
    per_class_all_path = output_dir / "phase17c_per_class_all_seeds.csv"
    per_class_summary_path = (
        output_dir / "phase17c_per_class_statistical_summary.csv"
    )

    all_results.to_csv(all_results_path, index=False)
    summary.to_csv(summary_path, index=False)
    per_class_all.to_csv(per_class_all_path, index=False)
    per_class_summary.to_csv(per_class_summary_path, index=False)

    # Compact JSON summary for research/audit use.
    json_payload: Dict[str, Any] = {
        "phase": "17C",
        "scenario": scenario,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "attack": "Byzantine model-update sign flip",
        "formula": "delta_attack = -scale * delta",
        "validation_poisoned": False,
        "locked_test_used": False,
        "experiments": {},
    }

    for experiment_id in DEFAULT_EXPERIMENTS:
        json_payload["experiments"][experiment_id] = {}
        for metric in CORE_METRICS + DROP_METRICS + ATTACK_AUDIT_METRICS:
            match = summary[
                (summary["experiment_id"] == experiment_id)
                & (summary["metric"] == metric)
            ]
            if match.empty:
                continue

            row = match.iloc[0]
            json_payload["experiments"][experiment_id][metric] = {
                "n": int(row["n"]),
                "mean": float(row["mean"]),
                "std": float(row["std"]),
                "min": float(row["min"]),
                "max": float(row["max"]),
                "ci95_low": float(row["ci95_low"]),
                "ci95_high": float(row["ci95_high"]),
            }

    json_path = output_dir / "phase17c_summary.json"
    save_json(json_path, json_payload)

    report = build_text_report(
        all_results=all_results,
        summary=summary,
        per_class_summary=per_class_summary,
        seeds=seeds,
        scenario=scenario,
        norm_error_tolerance=float(args.norm_error_tolerance),
    )

    report_path = output_dir / "phase17c_final_report.txt"
    report_path.write_text(report, encoding="utf-8")

    print()
    print(report)
    print()
    print("Generated files:")
    print(f"  {all_results_path}")
    print(f"  {summary_path}")
    print(f"  {per_class_all_path}")
    print(f"  {per_class_summary_path}")
    print(f"  {json_path}")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()
