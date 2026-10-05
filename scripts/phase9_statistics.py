"""
PHASE 9 - REPEATED-SEED STATISTICAL VALIDATION
===============================================

Methods
-------
1. FedAvg70
2. Hybrid FedAvg
3. FedProx
4. Median
5. Trimmed Mean

Seeds
-----
42, 52, 62, 72, 82

Main objectives
---------------
1. Verify all 25 runs exist.
2. Collect overall metrics.
3. Collect class-wise Precision / Recall / F1 / Support.
4. Calculate:
      Mean
      Standard deviation
      Median
      Minimum
      Maximum
      95% confidence interval
      Coefficient of variation
5. Rank algorithms.
6. Perform paired Wilcoxon tests vs FedAvg70.
7. Perform Friedman tests across all methods.
8. Perform class-wise statistical analysis.
9. Detect classes that are repeatedly missed.
10. Save all outputs as CSV/JSON.

Run
---

    python scripts\\phase9_statistics.py --scenario non_iid

Optional:

    python scripts\\phase9_statistics.py \
        --scenario non_iid \
        --seeds 42 52 62 72 82


Expected structure
------------------

results/
└── phase9/
    ├── fedavg70/
    │   └── non_iid/
    │       ├── seed_42/
    │       ├── seed_52/
    │       ├── seed_62/
    │       ├── seed_72/
    │       └── seed_82/
    │
    ├── hybrid_fedavg/
    ├── fedprox/
    ├── median/
    └── trimmed_mean/


Statistics output
-----------------

results/
└── phase9/
    └── statistics/
        └── non_iid/
            ├── all_seed_results.csv
            ├── overall_summary_statistics.csv
            ├── overall_rankings.csv
            ├── paired_deltas_vs_fedavg70.csv
            ├── wilcoxon_vs_fedavg70.csv
            ├── friedman_overall.csv
            ├── per_class_all_runs.csv
            ├── per_class_summary.csv
            ├── per_class_recall_ranking.csv
            ├── per_class_f1_ranking.csv
            ├── per_class_wilcoxon_vs_fedavg70.csv
            ├── per_class_friedman.csv
            ├── class_detection_stability.csv
            ├── best_methods.json
            └── phase9_statistics_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# SCIPY
# ============================================================

try:
    from scipy.stats import (
        friedmanchisquare,
        wilcoxon,
    )

    from scipy.stats import t as student_t

except ImportError as exc:

    raise ImportError(
        "\nSciPy is required.\n\n"
        "Install using:\n\n"
        "    pip install scipy\n"
    ) from exc


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


PHASE9_ROOT = (
    PROJECT_ROOT
    / "results"
    / "phase9"
)


# ============================================================
# METHODS
# ============================================================

METHODS = {

    "fedavg70": {

        "display_name":
            "FedAvg70",

        "features":
            70,
    },


    "hybrid_fedavg": {

        "display_name":
            "Hybrid FedAvg",

        "features":
            37,
    },


    "fedprox": {

        "display_name":
            "FedProx",

        "features":
            37,
    },


    "median": {

        "display_name":
            "Median",

        "features":
            37,
    },


    "trimmed_mean": {

        "display_name":
            "Trimmed Mean",

        "features":
            37,
    },
}


# ============================================================
# SEEDS
# ============================================================

DEFAULT_SEEDS = [

    42,
    52,
    62,
    72,
    82,
]


# ============================================================
# CLASSES
# ============================================================

CLASS_NAMES = [

    "BENIGN",

    "DDoS",

    "DoS GoldenEye",

    "DoS Hulk",

    "FTP-Patator",

    "PortScan",
]


EXPECTED_CLASS_SUPPORT = {

    "BENIGN":
        314259,

    "DDoS":
        19088,

    "DoS GoldenEye":
        1540,

    "DoS Hulk":
        26057,

    "FTP-Patator":
        882,

    "PortScan":
        13607,
}


# ============================================================
# OVERALL METRICS
# ============================================================

PERFORMANCE_METRICS = [

    "accuracy",

    "balanced_accuracy",

    "macro_precision",

    "macro_recall",

    "macro_f1",

    "weighted_f1",
]


EFFICIENCY_METRICS = [

    "training_time_seconds",

    "inference_time_seconds",
]


ALL_METRICS = (

    PERFORMANCE_METRICS

    +

    EFFICIENCY_METRICS
)


# ============================================================
# JSON
# ============================================================

def load_json(
    path: Path,
) -> dict:

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


# ============================================================
# SAFE NUMBER
# ============================================================

def safe_float(
    value: Any,
):

    if value is None:

        return np.nan


    try:

        return float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return np.nan


# ============================================================
# RUN DIRECTORY
# ============================================================

def get_run_directory(
    method: str,
    scenario: str,
    seed: int,
) -> Path:

    return (

        PHASE9_ROOT

        / method

        / scenario

        / f"seed_{seed}"
    )


# ============================================================
# OVERALL RESULT FILE
# ============================================================

def get_overall_file(
    method: str,
    scenario: str,
    seed: int,
) -> Path:

    return (

        get_run_directory(
            method,
            scenario,
            seed,
        )

        / "overall_metrics.json"
    )


# ============================================================
# VERIFY 25 RUNS
# ============================================================

def verify_all_runs(
    scenario: str,
    seeds: list[int],
):

    missing = []


    print()

    print(
        "=" * 100
    )

    print(
        "VERIFYING PHASE 9 RUNS"
    )

    print(
        "=" * 100
    )


    for method, config in METHODS.items():

        display_name = (
            config[
                "display_name"
            ]
        )


        for seed in seeds:

            file_path = get_overall_file(

                method=
                    method,

                scenario=
                    scenario,

                seed=
                    seed,
            )


            if file_path.exists():

                print(

                    f"✅ "
                    f"{display_name:18s} "
                    f"seed={seed}"
                )

            else:

                print(

                    f"❌ "
                    f"{display_name:18s} "
                    f"seed={seed} "
                    f"MISSING"
                )


                missing.append(
                    file_path
                )


    if missing:

        print()

        print(
            "Missing files:"
        )


        for file_path in missing:

            print(
                file_path
            )


        raise FileNotFoundError(

            "\nPhase 9 is incomplete.\n"
            f"Missing {len(missing)} runs.\n"
        )


# ============================================================
# FIND METRIC
# ============================================================

def get_metric_value(
    metrics: dict,
    metric: str,
):

    # Direct key
    if metric in metrics:

        return safe_float(
            metrics[
                metric
            ]
        )


    # Some scripts may use alternative names

    aliases = {

        "training_time_seconds": [

            "training_time",

            "train_time_seconds",

            "training_seconds",
        ],

        "inference_time_seconds": [

            "inference_time",

            "test_time_seconds",

            "inference_seconds",
        ],
    }


    for alias in aliases.get(
        metric,
        [],
    ):

        if alias in metrics:

            return safe_float(
                metrics[
                    alias
                ]
            )


    return np.nan


# ============================================================
# COLLECT OVERALL RESULTS
# ============================================================

def collect_overall_results(
    scenario: str,
    seeds: list[int],
) -> pd.DataFrame:

    rows = []


    for method, config in METHODS.items():

        for seed in seeds:

            file_path = get_overall_file(

                method,
                scenario,
                seed,
            )


            metrics = load_json(
                file_path
            )


            row = {

                "method_key":
                    method,

                "method":
                    config[
                        "display_name"
                    ],

                "seed":
                    seed,

                "features":
                    config[
                        "features"
                    ],
            }


            for metric in ALL_METRICS:

                row[
                    metric
                ] = get_metric_value(

                    metrics,
                    metric,
                )


            row[
                "selected_feature_count"
            ] = metrics.get(

                "selected_feature_count",

                config[
                    "features"
                ],
            )


            row[
                "feature_reduction_percent"
            ] = (

                100.0
                *
                (
                    1.0
                    -
                    config[
                        "features"
                    ]
                    /
                    70.0
                )
            )


            rows.append(
                row
            )


    return pd.DataFrame(
        rows
    )


# ============================================================
# 95% CONFIDENCE INTERVAL
# ============================================================

def confidence_interval_95(
    values,
):

    values = np.asarray(

        values,

        dtype=float,
    )


    values = values[

        np.isfinite(
            values
        )
    ]


    n = len(
        values
    )


    if n == 0:

        return (
            np.nan,
            np.nan,
            np.nan,
        )


    mean_value = float(

        np.mean(
            values
        )
    )


    if n == 1:

        return (

            mean_value,

            mean_value,

            mean_value,
        )


    std_value = float(

        np.std(
            values,
            ddof=1,
        )
    )


    standard_error = (

        std_value

        /

        math.sqrt(
            n
        )
    )


    critical_value = float(

        student_t.ppf(

            0.975,

            df=
                n - 1,
        )
    )


    margin = (

        critical_value

        *

        standard_error
    )


    return (

        mean_value,

        mean_value
        -
        margin,

        mean_value
        +
        margin,
    )


# ============================================================
# GENERIC SUMMARY
# ============================================================

def summarize_values(
    values,
):

    values = np.asarray(

        values,

        dtype=float,
    )


    values = values[

        np.isfinite(
            values
        )
    ]


    n = len(
        values
    )


    if n == 0:

        return None


    mean_value = float(

        np.mean(
            values
        )
    )


    std_value = float(

        np.std(
            values,
            ddof=1,
        )

        if n > 1

        else

        0.0
    )


    median_value = float(

        np.median(
            values
        )
    )


    minimum = float(

        np.min(
            values
        )
    )


    maximum = float(

        np.max(
            values
        )
    )


    _, ci_lower, ci_upper = (

        confidence_interval_95(
            values
        )
    )


    coefficient_of_variation = (

        std_value

        /

        abs(
            mean_value
        )

        if mean_value != 0

        else np.nan
    )


    return {

        "n":
            n,

        "mean":
            mean_value,

        "std":
            std_value,

        "median":
            median_value,

        "min":
            minimum,

        "max":
            maximum,

        "ci95_lower":
            ci_lower,

        "ci95_upper":
            ci_upper,

        "coefficient_of_variation":
            coefficient_of_variation,
    }


# ============================================================
# OVERALL SUMMARY
# ============================================================

def build_overall_summary(
    results_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []


    for method, config in METHODS.items():

        subset = results_df[

            results_df[
                "method_key"
            ]
            ==
            method
        ]


        for metric in ALL_METRICS:

            summary = summarize_values(

                subset[
                    metric
                ]
                .to_numpy()
            )


            if summary is None:

                continue


            rows.append({

                "method_key":
                    method,

                "method":
                    config[
                        "display_name"
                    ],

                "features":
                    config[
                        "features"
                    ],

                "metric":
                    metric,

                **summary,
            })


    return pd.DataFrame(
        rows
    )


# ============================================================
# RANK OVERALL METHODS
# ============================================================

def build_overall_rankings(
    summary_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []


    for metric in PERFORMANCE_METRICS:

        subset = summary_df[

            summary_df[
                "metric"
            ]
            ==
            metric

        ].copy()


        subset[
            "rank"
        ] = (

            subset[
                "mean"
            ]

            .rank(

                ascending=False,

                method="min",
            )
        )


        subset = subset.sort_values(

            [
                "rank",
                "mean",
            ],

            ascending=[
                True,
                False,
            ],
        )


        for _, row in subset.iterrows():

            rows.append({

                "metric":
                    metric,

                "rank":
                    int(
                        row[
                            "rank"
                        ]
                    ),

                "method":
                    row[
                        "method"
                    ],

                "features":
                    int(
                        row[
                            "features"
                        ]
                    ),

                "mean":
                    row[
                        "mean"
                    ],

                "std":
                    row[
                        "std"
                    ],

                "ci95_lower":
                    row[
                        "ci95_lower"
                    ],

                "ci95_upper":
                    row[
                        "ci95_upper"
                    ],
            })


    return pd.DataFrame(
        rows
    )


# ============================================================
# WILCOXON
# ============================================================

def safe_wilcoxon(
    method_values,
    baseline_values,
):

    method_values = np.asarray(

        method_values,

        dtype=float,
    )


    baseline_values = np.asarray(

        baseline_values,

        dtype=float,
    )


    mask = (

        np.isfinite(
            method_values
        )

        &

        np.isfinite(
            baseline_values
        )
    )


    method_values = (
        method_values[
            mask
        ]
    )


    baseline_values = (
        baseline_values[
            mask
        ]
    )


    if len(
        method_values
    ) == 0:

        return (
            np.nan,
            np.nan,
        )


    differences = (

        method_values

        -

        baseline_values
    )


    if np.allclose(

        differences,

        0.0,
    ):

        return (
            0.0,
            1.0,
        )


    try:

        result = wilcoxon(

            method_values,

            baseline_values,

            alternative="two-sided",
        )


        return (

            float(
                result.statistic
            ),

            float(
                result.pvalue
            ),
        )


    except ValueError:

        return (
            np.nan,
            np.nan,
        )


# ============================================================
# PAIRED DELTAS VS FEDAVG70
# ============================================================

def build_paired_deltas(
    results_df: pd.DataFrame,
    seeds: list[int],
) -> pd.DataFrame:

    rows = []


    baseline = (

        results_df[

            results_df[
                "method_key"
            ]
            ==
            "fedavg70"

        ]

        .set_index(
            "seed"
        )
    )


    for method, config in METHODS.items():

        if method == "fedavg70":

            continue


        method_df = (

            results_df[

                results_df[
                    "method_key"
                ]
                ==
                method
            ]

            .set_index(
                "seed"
            )
        )


        for seed in seeds:

            for metric in PERFORMANCE_METRICS:

                baseline_value = (

                    baseline.loc[
                        seed,
                        metric
                    ]
                )


                method_value = (

                    method_df.loc[
                        seed,
                        metric
                    ]
                )


                delta = (

                    method_value

                    -

                    baseline_value
                )


                rows.append({

                    "method":
                        config[
                            "display_name"
                        ],

                    "baseline":
                        "FedAvg70",

                    "seed":
                        seed,

                    "metric":
                        metric,

                    "baseline_value":
                        baseline_value,

                    "method_value":
                        method_value,

                    "delta":
                        delta,

                    "improved":
                        bool(
                            delta > 0
                        ),
                })


    return pd.DataFrame(
        rows
    )


# ============================================================
# OVERALL WILCOXON
# ============================================================

def build_overall_wilcoxon(
    results_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []


    baseline = (

        results_df[

            results_df[
                "method_key"
            ]
            ==
            "fedavg70"

        ]

        .sort_values(
            "seed"
        )
    )


    for method, config in METHODS.items():

        if method == "fedavg70":

            continue


        method_df = (

            results_df[

                results_df[
                    "method_key"
                ]
                ==
                method

            ]

            .sort_values(
                "seed"
            )
        )


        for metric in PERFORMANCE_METRICS:

            base_values = (

                baseline[
                    metric
                ]
                .to_numpy()
            )


            method_values = (

                method_df[
                    metric
                ]
                .to_numpy()
            )


            statistic, p_value = (

                safe_wilcoxon(

                    method_values,
                    base_values,
                )
            )


            differences = (

                method_values

                -

                base_values
            )


            rows.append({

                "baseline":
                    "FedAvg70",

                "method":
                    config[
                        "display_name"
                    ],

                "metric":
                    metric,

                "wilcoxon_statistic":
                    statistic,

                "p_value":
                    p_value,

                "significant_at_0_05":
                    bool(
                        p_value < 0.05
                    )
                    if np.isfinite(
                        p_value
                    )
                    else False,

                "mean_delta":
                    float(
                        np.mean(
                            differences
                        )
                    ),

                "median_delta":
                    float(
                        np.median(
                            differences
                        )
                    ),

                "wins":
                    int(
                        np.sum(
                            differences > 0
                        )
                    ),

                "losses":
                    int(
                        np.sum(
                            differences < 0
                        )
                    ),

                "ties":
                    int(
                        np.sum(
                            np.isclose(
                                differences,
                                0.0,
                            )
                        )
                    ),
            })


    return pd.DataFrame(
        rows
    )


# ============================================================
# FRIEDMAN OVERALL
# ============================================================

def build_overall_friedman(
    results_df: pd.DataFrame,
    seeds: list[int],
) -> pd.DataFrame:

    rows = []


    for metric in PERFORMANCE_METRICS:

        samples = []


        for method in METHODS:

            subset = (

                results_df[

                    results_df[
                        "method_key"
                    ]
                    ==
                    method

                ]

                .set_index(
                    "seed"
                )

                .loc[
                    seeds
                ]
            )


            samples.append(

                subset[
                    metric
                ]
                .to_numpy()
            )


        try:

            result = friedmanchisquare(
                *samples
            )


            statistic = float(
                result.statistic
            )

            p_value = float(
                result.pvalue
            )


        except ValueError:

            statistic = np.nan

            p_value = np.nan


        rows.append({

            "metric":
                metric,

            "methods":
                len(
                    METHODS
                ),

            "seeds":
                len(
                    seeds
                ),

            "friedman_statistic":
                statistic,

            "p_value":
                p_value,

            "significant_at_0_05":
                bool(
                    p_value < 0.05
                )
                if np.isfinite(
                    p_value
                )
                else False,
        })


    return pd.DataFrame(
        rows
    )


# ============================================================
# PER-CLASS FILE CANDIDATES
# ============================================================

PER_CLASS_FILE_CANDIDATES = [

    "per_class_metrics.csv",

    "per_class_results.csv",

    "classification_report.csv",

    "per_class_report.csv",

    "class_metrics.csv",
]


# ============================================================
# NORMALIZE CLASS NAME
# ============================================================

def normalize_class_name(
    value: Any,
) -> str:

    text = str(
        value
    ).strip()


    replacements = {

        "benign":
            "BENIGN",

        "ddos":
            "DDoS",

        "dos goldeneye":
            "DoS GoldenEye",

        "dos hulk":
            "DoS Hulk",

        "ftp-patator":
            "FTP-Patator",

        "portscan":
            "PortScan",
    }


    return replacements.get(

        text.lower(),

        text,
    )


# ============================================================
# EXTRACT CLASS METRICS FROM JSON
# ============================================================

def extract_per_class_from_json(
    metrics: dict,
):

    candidate_keys = [

        "per_class",

        "per_class_metrics",

        "classification_report",

        "class_metrics",

        "per_class_performance",
    ]


    for key in candidate_keys:

        data = metrics.get(
            key
        )


        if not isinstance(
            data,
            dict,
        ):

            continue


        rows = []


        for class_name, values in data.items():

            normalized = normalize_class_name(
                class_name
            )


            if normalized not in CLASS_NAMES:

                continue


            if not isinstance(
                values,
                dict,
            ):

                continue


            precision = safe_float(

                values.get(

                    "precision",

                    values.get(
                        "P"
                    ),
                )
            )


            recall = safe_float(

                values.get(

                    "recall",

                    values.get(
                        "R"
                    ),
                )
            )


            f1 = safe_float(

                values.get(

                    "f1",

                    values.get(

                        "f1_score",

                        values.get(
                            "F1"
                        ),
                    ),
                )
            )


            support = safe_float(

                values.get(

                    "support",

                    values.get(
                        "N"
                    ),
                )
            )


            rows.append({

                "class":
                    normalized,

                "precision":
                    precision,

                "recall":
                    recall,

                "f1":
                    f1,

                "support":
                    support,
            })


        if rows:

            return pd.DataFrame(
                rows
            )


    return None


# ============================================================
# EXTRACT CLASS METRICS FROM CSV
# ============================================================

def extract_per_class_from_csv(
    run_directory: Path,
):

    for filename in PER_CLASS_FILE_CANDIDATES:

        path = (
            run_directory
            /
            filename
        )


        if not path.exists():

            continue


        try:

            df = pd.read_csv(
                path
            )

        except Exception:

            continue


        columns_lower = {

            str(column).lower():
                column

            for column
            in df.columns
        }


        class_column = None


        for candidate in [

            "class",

            "class_name",

            "label",

            "attack",

            "attack_name",
        ]:

            if candidate in columns_lower:

                class_column = (
                    columns_lower[
                        candidate
                    ]
                )

                break


        if class_column is None:

            continue


        precision_column = None
        recall_column = None
        f1_column = None
        support_column = None


        for candidate in [

            "precision",

            "p",
        ]:

            if candidate in columns_lower:

                precision_column = (
                    columns_lower[
                        candidate
                    ]
                )

                break


        for candidate in [

            "recall",

            "r",
        ]:

            if candidate in columns_lower:

                recall_column = (
                    columns_lower[
                        candidate
                    ]
                )

                break


        for candidate in [

            "f1",

            "f1_score",

            "f1-score",
        ]:

            if candidate in columns_lower:

                f1_column = (
                    columns_lower[
                        candidate
                    ]
                )

                break


        for candidate in [

            "support",

            "n",
        ]:

            if candidate in columns_lower:

                support_column = (
                    columns_lower[
                        candidate
                    ]
                )

                break


        if (

            precision_column is None

            or recall_column is None

            or f1_column is None

        ):

            continue


        rows = []


        for _, row in df.iterrows():

            class_name = normalize_class_name(

                row[
                    class_column
                ]
            )


            if class_name not in CLASS_NAMES:

                continue


            rows.append({

                "class":
                    class_name,

                "precision":
                    safe_float(
                        row[
                            precision_column
                        ]
                    ),

                "recall":
                    safe_float(
                        row[
                            recall_column
                        ]
                    ),

                "f1":
                    safe_float(
                        row[
                            f1_column
                        ]
                    ),

                "support":
                    safe_float(
                        row[
                            support_column
                        ]
                    )
                    if support_column
                    is not None
                    else np.nan,
            })


        if rows:

            return pd.DataFrame(
                rows
            )


    return None


# ============================================================
# FIND CLASS METRICS
# ============================================================

def load_per_class_metrics(
    method: str,
    scenario: str,
    seed: int,
):

    run_directory = get_run_directory(

        method,
        scenario,
        seed,
    )


    overall_file = (

        run_directory

        / "overall_metrics.json"
    )


    metrics = load_json(
        overall_file
    )


    # ------------------------------------------
    # Try JSON first
    # ------------------------------------------

    class_df = extract_per_class_from_json(
        metrics
    )


    if class_df is not None:

        return class_df


    # ------------------------------------------
    # Try CSV files
    # ------------------------------------------

    class_df = extract_per_class_from_csv(
        run_directory
    )


    if class_df is not None:

        return class_df


    return None


# ============================================================
# COLLECT PER-CLASS RESULTS
# ============================================================

def collect_per_class_results(
    scenario: str,
    seeds: list[int],
) -> pd.DataFrame:

    rows = []

    missing = []


    for method, config in METHODS.items():

        for seed in seeds:

            class_df = load_per_class_metrics(

                method,
                scenario,
                seed,
            )


            if class_df is None:

                missing.append(

                    (
                        method,
                        seed,
                    )
                )

                continue


            for _, row in class_df.iterrows():

                class_name = normalize_class_name(

                    row[
                        "class"
                    ]
                )


                rows.append({

                    "method_key":
                        method,

                    "method":
                        config[
                            "display_name"
                        ],

                    "seed":
                        seed,

                    "features":
                        config[
                            "features"
                        ],

                    "class":
                        class_name,

                    "precision":
                        safe_float(
                            row[
                                "precision"
                            ]
                        ),

                    "recall":
                        safe_float(
                            row[
                                "recall"
                            ]
                        ),

                    "f1":
                        safe_float(
                            row[
                                "f1"
                            ]
                        ),

                    "support":
                        safe_float(
                            row[
                                "support"
                            ]
                        ),
                })


    if missing:

        print()

        print(
            "WARNING:"
        )

        print(
            "Per-class files were not found "
            "for these runs:"
        )


        for method, seed in missing:

            print(

                f"  "
                f"{method} "
                f"seed={seed}"
            )


        print()

        print(
            "Overall statistics will still run."
        )

        print(
            "Per-class statistics require either:"
        )

        print(
            "  per_class_metrics.csv"
        )

        print(
            "or class metrics inside overall_metrics.json."
        )


    return pd.DataFrame(
        rows
    )


# ============================================================
# VERIFY CLASS SUPPORT
# ============================================================

def verify_class_supports(
    class_df: pd.DataFrame,
):

    if class_df.empty:

        return


    print()

    print(
        "=" * 100
    )

    print(
        "VERIFYING LOCKED TEST CLASS SUPPORT"
    )

    print(
        "=" * 100
    )


    for class_name in CLASS_NAMES:

        expected = (

            EXPECTED_CLASS_SUPPORT[
                class_name
            ]
        )


        values = (

            class_df[

                class_df[
                    "class"
                ]
                ==
                class_name

            ][
                "support"
            ]

            .dropna()

            .unique()
        )


        if len(
            values
        ) == 0:

            print(

                f"⚠️ "
                f"{class_name:18s} "
                f"support unavailable"
            )

            continue


        valid = all(

            int(
                value
            )
            ==
            expected

            for value
            in values
        )


        if valid:

            print(

                f"✅ "
                f"{class_name:18s} "
                f"N={expected}"
            )

        else:

            print(

                f"❌ "
                f"{class_name:18s} "
                f"Expected={expected}, "
                f"Found={values}"
            )


# ============================================================
# PER-CLASS SUMMARY
# ============================================================

def build_per_class_summary(
    class_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []


    if class_df.empty:

        return pd.DataFrame()


    for method, config in METHODS.items():

        for class_name in CLASS_NAMES:

            subset = class_df[

                (
                    class_df[
                        "method_key"
                    ]
                    ==
                    method
                )

                &

                (
                    class_df[
                        "class"
                    ]
                    ==
                    class_name
                )
            ]


            if subset.empty:

                continue


            for metric in [

                "precision",

                "recall",

                "f1",
            ]:

                summary = summarize_values(

                    subset[
                        metric
                    ]
                    .to_numpy()
                )


                if summary is None:

                    continue


                rows.append({

                    "method_key":
                        method,

                    "method":
                        config[
                            "display_name"
                        ],

                    "class":
                        class_name,

                    "metric":
                        metric,

                    **summary,
                })


    return pd.DataFrame(
        rows
    )


# ============================================================
# PER-CLASS RANKINGS
# ============================================================

def build_per_class_ranking(
    summary_df: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:

    rows = []


    if summary_df.empty:

        return pd.DataFrame()


    for class_name in CLASS_NAMES:

        subset = summary_df[

            (
                summary_df[
                    "class"
                ]
                ==
                class_name
            )

            &

            (
                summary_df[
                    "metric"
                ]
                ==
                metric
            )

        ].copy()


        if subset.empty:

            continue


        subset[
            "rank"
        ] = (

            subset[
                "mean"
            ]

            .rank(

                ascending=False,

                method="min",
            )
        )


        subset = subset.sort_values(

            [
                "rank",
                "mean",
            ]
        )


        for _, row in subset.iterrows():

            rows.append({

                "class":
                    class_name,

                "metric":
                    metric,

                "rank":
                    int(
                        row[
                            "rank"
                        ]
                    ),

                "method":
                    row[
                        "method"
                    ],

                "mean":
                    row[
                        "mean"
                    ],

                "std":
                    row[
                        "std"
                    ],

                "ci95_lower":
                    row[
                        "ci95_lower"
                    ],

                "ci95_upper":
                    row[
                        "ci95_upper"
                    ],
            })


    return pd.DataFrame(
        rows
    )


# ============================================================
# CLASS DETECTION STABILITY
# ============================================================

def build_detection_stability(
    class_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []


    if class_df.empty:

        return pd.DataFrame()


    for method, config in METHODS.items():

        for class_name in CLASS_NAMES:

            subset = class_df[

                (
                    class_df[
                        "method_key"
                    ]
                    ==
                    method
                )

                &

                (
                    class_df[
                        "class"
                    ]
                    ==
                    class_name
                )
            ]


            if subset.empty:

                continue


            recalls = (

                subset[
                    "recall"
                ]
                .to_numpy(
                    dtype=float
                )
            )


            rows.append({

                "method":
                    config[
                        "display_name"
                    ],

                "class":
                    class_name,

                "mean_recall":
                    float(
                        np.mean(
                            recalls
                        )
                    ),

                "std_recall":
                    float(
                        np.std(
                            recalls,
                            ddof=1,
                        )
                    ),

                "min_recall":
                    float(
                        np.min(
                            recalls
                        )
                    ),

                "max_recall":
                    float(
                        np.max(
                            recalls
                        )
                    ),

                "zero_recall_runs":
                    int(

                        np.sum(

                            np.isclose(
                                recalls,
                                0.0,
                            )
                        )
                    ),

                "detected_runs":
                    int(

                        np.sum(
                            recalls > 0
                        )
                    ),

                "recall_ge_0_10_runs":
                    int(

                        np.sum(
                            recalls >= 0.10
                        )
                    ),

                "recall_ge_0_50_runs":
                    int(

                        np.sum(
                            recalls >= 0.50
                        )
                    ),
            })


    return pd.DataFrame(
        rows
    )


# ============================================================
# PER-CLASS WILCOXON
# ============================================================

def build_per_class_wilcoxon(
    class_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []


    if class_df.empty:

        return pd.DataFrame()


    for class_name in CLASS_NAMES:

        baseline = (

            class_df[

                (
                    class_df[
                        "method_key"
                    ]
                    ==
                    "fedavg70"
                )

                &

                (
                    class_df[
                        "class"
                    ]
                    ==
                    class_name
                )
            ]

            .sort_values(
                "seed"
            )
        )


        for method, config in METHODS.items():

            if method == "fedavg70":

                continue


            method_df = (

                class_df[

                    (
                        class_df[
                            "method_key"
                        ]
                        ==
                        method
                    )

                    &

                    (
                        class_df[
                            "class"
                        ]
                        ==
                        class_name
                    )
                ]

                .sort_values(
                    "seed"
                )
            )


            for metric in [

                "recall",

                "f1",
            ]:

                base_values = (

                    baseline[
                        metric
                    ]
                    .to_numpy()
                )


                method_values = (

                    method_df[
                        metric
                    ]
                    .to_numpy()
                )


                if (

                    len(
                        base_values
                    )
                    !=
                    len(
                        method_values
                    )

                    or

                    len(
                        base_values
                    )
                    == 0
                ):

                    continue


                statistic, p_value = (

                    safe_wilcoxon(

                        method_values,

                        base_values,
                    )
                )


                differences = (

                    method_values

                    -

                    base_values
                )


                rows.append({

                    "class":
                        class_name,

                    "metric":
                        metric,

                    "baseline":
                        "FedAvg70",

                    "method":
                        config[
                            "display_name"
                        ],

                    "mean_delta":
                        float(
                            np.mean(
                                differences
                            )
                        ),

                    "wilcoxon_statistic":
                        statistic,

                    "p_value":
                        p_value,

                    "significant_at_0_05":
                        bool(
                            p_value < 0.05
                        )
                        if np.isfinite(
                            p_value
                        )
                        else False,

                    "wins":
                        int(
                            np.sum(
                                differences > 0
                            )
                        ),

                    "losses":
                        int(
                            np.sum(
                                differences < 0
                            )
                        ),

                    "ties":
                        int(
                            np.sum(
                                np.isclose(
                                    differences,
                                    0.0,
                                )
                            )
                        ),
                })


    return pd.DataFrame(
        rows
    )


# ============================================================
# PER-CLASS FRIEDMAN
# ============================================================

def build_per_class_friedman(
    class_df: pd.DataFrame,
    seeds: list[int],
) -> pd.DataFrame:

    rows = []


    if class_df.empty:

        return pd.DataFrame()


    for class_name in CLASS_NAMES:

        for metric in [

            "recall",

            "f1",
        ]:

            samples = []


            valid = True


            for method in METHODS:

                subset = (

                    class_df[

                        (
                            class_df[
                                "method_key"
                            ]
                            ==
                            method
                        )

                        &

                        (
                            class_df[
                                "class"
                            ]
                            ==
                            class_name
                        )
                    ]

                    .set_index(
                        "seed"
                    )
                )


                try:

                    values = (

                        subset

                        .loc[
                            seeds,
                            metric
                        ]

                        .to_numpy(
                            dtype=float
                        )
                    )

                except KeyError:

                    valid = False

                    break


                samples.append(
                    values
                )


            if not valid:

                continue


            try:

                result = friedmanchisquare(
                    *samples
                )


                statistic = float(
                    result.statistic
                )

                p_value = float(
                    result.pvalue
                )


            except ValueError:

                statistic = np.nan

                p_value = np.nan


            rows.append({

                "class":
                    class_name,

                "metric":
                    metric,

                "friedman_statistic":
                    statistic,

                "p_value":
                    p_value,

                "significant_at_0_05":
                    bool(
                        p_value < 0.05
                    )
                    if np.isfinite(
                        p_value
                    )
                    else False,
            })


    return pd.DataFrame(
        rows
    )


# ============================================================
# BEST METHODS
# ============================================================

def determine_best_methods(
    overall_summary: pd.DataFrame,
    class_summary: pd.DataFrame,
):

    output = {

        "overall":
            {},

        "per_class":
            {},
    }


    for metric in PERFORMANCE_METRICS:

        subset = overall_summary[

            overall_summary[
                "metric"
            ]
            ==
            metric
        ]


        if subset.empty:

            continue


        best = subset.loc[

            subset[
                "mean"
            ]
            .idxmax()
        ]


        output[
            "overall"
        ][
            metric
        ] = {

            "method":
                best[
                    "method"
                ],

            "mean":
                float(
                    best[
                        "mean"
                    ]
                ),

            "std":
                float(
                    best[
                        "std"
                    ]
                ),
        }


    if not class_summary.empty:

        for class_name in CLASS_NAMES:

            output[
                "per_class"
            ][
                class_name
            ] = {}


            for metric in [

                "recall",

                "f1",
            ]:

                subset = class_summary[

                    (
                        class_summary[
                            "class"
                        ]
                        ==
                        class_name
                    )

                    &

                    (
                        class_summary[
                            "metric"
                        ]
                        ==
                        metric
                    )
                ]


                if subset.empty:

                    continue


                best = subset.loc[

                    subset[
                        "mean"
                    ]
                    .idxmax()
                ]


                output[
                    "per_class"
                ][
                    class_name
                ][
                    metric
                ] = {

                    "method":
                        best[
                            "method"
                        ],

                    "mean":
                        float(
                            best[
                                "mean"
                            ]
                        ),

                    "std":
                        float(
                            best[
                                "std"
                            ]
                        ),
                }


    return output


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

            key:
                json_safe(
                    item
                )

            for key, item
            in value.items()
        }


    if isinstance(
        value,
        list,
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
        np.integer,
    ):

        return int(
            value
        )


    if isinstance(
        value,
        np.floating,
    ):

        if np.isnan(
            value
        ):

            return None


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


    return value


# ============================================================
# PRINT OVERALL SUMMARY
# ============================================================

def print_overall_summary(
    summary_df: pd.DataFrame,
):

    important_metrics = [

        "accuracy",

        "balanced_accuracy",

        "macro_f1",

        "weighted_f1",
    ]


    print()

    print(
        "=" * 100
    )

    print(
        "OVERALL PHASE 9 SUMMARY"
    )

    print(
        "=" * 100
    )


    for metric in important_metrics:

        subset = (

            summary_df[

                summary_df[
                    "metric"
                ]
                ==
                metric
            ]

            .sort_values(

                "mean",

                ascending=False,
            )
        )


        print()

        print(
            metric.upper()
        )

        print(
            "-" * 100
        )


        print(

            subset[
                [
                    "method",
                    "mean",
                    "std",
                    "ci95_lower",
                    "ci95_upper",
                ]
            ]

            .to_string(

                index=False,

                float_format=
                    lambda x:
                        f"{x:.6f}",
            )
        )


# ============================================================
# PRINT CLASS RECALL SUMMARY
# ============================================================

def print_class_summary(
    class_summary: pd.DataFrame,
):

    if class_summary.empty:

        return


    print()

    print(
        "=" * 100
    )

    print(
        "PER-CLASS MEAN RECALL"
    )

    print(
        "=" * 100
    )


    recall_df = class_summary[

        class_summary[
            "metric"
        ]
        ==
        "recall"
    ]


    pivot = recall_df.pivot(

        index=
            "method",

        columns=
            "class",

        values=
            "mean",
    )


    existing_classes = [

        class_name

        for class_name
        in CLASS_NAMES

        if class_name
        in pivot.columns
    ]


    pivot = pivot[
        existing_classes
    ]


    print(

        pivot.to_string(

            float_format=
                lambda x:
                    f"{x:.4f}",
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Phase 9 repeated-seed statistical validation"
        )
    )


    parser.add_argument(

        "--scenario",

        choices=[

            "iid",

            "non_iid",

            "controlled_non_iid",
        ],

        default=
            "non_iid",
    )


    parser.add_argument(

        "--seeds",

        nargs="+",

        type=int,

        default=
            DEFAULT_SEEDS,
    )


    args = parser.parse_args()


    seeds = [

        int(
            seed
        )

        for seed
        in args.seeds
    ]


    if len(
        seeds
    ) < 2:

        raise ValueError(
            "At least two seeds are required."
        )


    if len(
        set(
            seeds
        )
    ) != len(
        seeds
    ):

        raise ValueError(
            "Duplicate seed values are not allowed."
        )


    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    output_dir = (

        PHASE9_ROOT

        / "statistics"

        / args.scenario
    )


    output_dir.mkdir(

        parents=True,

        exist_ok=True,
    )


    # ========================================================
    # HEADER
    # ========================================================

    print()

    print(
        "=" * 100
    )

    print(
        "PHASE 9 — REPEATED-SEED STATISTICAL VALIDATION"
    )

    print(
        "=" * 100
    )


    print(
        f"Scenario       : "
        f"{args.scenario}"
    )


    print(
        f"Seeds          : "
        f"{seeds}"
    )


    print(
        f"Methods        : "
        f"{len(METHODS)}"
    )


    print(
        f"Expected runs  : "
        f"{len(METHODS) * len(seeds)}"
    )


    # ========================================================
    # VERIFY
    # ========================================================

    verify_all_runs(

        scenario=
            args.scenario,

        seeds=
            seeds,
    )


    # ========================================================
    # OVERALL RESULTS
    # ========================================================

    overall_df = collect_overall_results(

        scenario=
            args.scenario,

        seeds=
            seeds,
    )


    overall_df.to_csv(

        output_dir
        / "all_seed_results.csv",

        index=False,
    )


    # ========================================================
    # OVERALL SUMMARY
    # ========================================================

    overall_summary = build_overall_summary(
        overall_df
    )


    overall_summary.to_csv(

        output_dir
        / "overall_summary_statistics.csv",

        index=False,
    )


    # ========================================================
    # OVERALL RANKINGS
    # ========================================================

    overall_rankings = build_overall_rankings(
        overall_summary
    )


    overall_rankings.to_csv(

        output_dir
        / "overall_rankings.csv",

        index=False,
    )


    # ========================================================
    # DELTAS
    # ========================================================

    paired_deltas = build_paired_deltas(

        overall_df,

        seeds,
    )


    paired_deltas.to_csv(

        output_dir
        / "paired_deltas_vs_fedavg70.csv",

        index=False,
    )


    # ========================================================
    # WILCOXON
    # ========================================================

    wilcoxon_df = build_overall_wilcoxon(
        overall_df
    )


    wilcoxon_df.to_csv(

        output_dir
        / "wilcoxon_vs_fedavg70.csv",

        index=False,
    )


    # ========================================================
    # FRIEDMAN
    # ========================================================

    friedman_df = build_overall_friedman(

        overall_df,

        seeds,
    )


    friedman_df.to_csv(

        output_dir
        / "friedman_overall.csv",

        index=False,
    )


    # ========================================================
    # PER CLASS
    # ========================================================

    class_df = collect_per_class_results(

        scenario=
            args.scenario,

        seeds=
            seeds,
    )


    if not class_df.empty:

        class_df.to_csv(

            output_dir
            / "per_class_all_runs.csv",

            index=False,
        )


        # ----------------------------------------------
        # Verify same test support
        # ----------------------------------------------

        verify_class_supports(
            class_df
        )


        # ----------------------------------------------
        # Summary
        # ----------------------------------------------

        class_summary = build_per_class_summary(
            class_df
        )


        class_summary.to_csv(

            output_dir
            / "per_class_summary.csv",

            index=False,
        )


        # ----------------------------------------------
        # Recall ranking
        # ----------------------------------------------

        recall_ranking = build_per_class_ranking(

            class_summary,

            "recall",
        )


        recall_ranking.to_csv(

            output_dir
            / "per_class_recall_ranking.csv",

            index=False,
        )


        # ----------------------------------------------
        # F1 ranking
        # ----------------------------------------------

        f1_ranking = build_per_class_ranking(

            class_summary,

            "f1",
        )


        f1_ranking.to_csv(

            output_dir
            / "per_class_f1_ranking.csv",

            index=False,
        )


        # ----------------------------------------------
        # Detection stability
        # ----------------------------------------------

        detection_stability = (

            build_detection_stability(
                class_df
            )
        )


        detection_stability.to_csv(

            output_dir
            / "class_detection_stability.csv",

            index=False,
        )


        # ----------------------------------------------
        # Per-class Wilcoxon
        # ----------------------------------------------

        class_wilcoxon = (

            build_per_class_wilcoxon(
                class_df
            )
        )


        class_wilcoxon.to_csv(

            output_dir
            / "per_class_wilcoxon_vs_fedavg70.csv",

            index=False,
        )


        # ----------------------------------------------
        # Per-class Friedman
        # ----------------------------------------------

        class_friedman = (

            build_per_class_friedman(

                class_df,

                seeds,
            )
        )


        class_friedman.to_csv(

            output_dir
            / "per_class_friedman.csv",

            index=False,
        )


    else:

        class_summary = (
            pd.DataFrame()
        )

        recall_ranking = (
            pd.DataFrame()
        )

        f1_ranking = (
            pd.DataFrame()
        )

        detection_stability = (
            pd.DataFrame()
        )

        class_wilcoxon = (
            pd.DataFrame()
        )

        class_friedman = (
            pd.DataFrame()
        )


    # ========================================================
    # BEST METHODS
    # ========================================================

    best_methods = determine_best_methods(

        overall_summary,

        class_summary,
    )


    with open(

        output_dir
        / "best_methods.json",

        "w",

        encoding="utf-8",

    ) as file:

        json.dump(

            json_safe(
                best_methods
            ),

            file,

            indent=4,
        )


    # ========================================================
    # MASTER SUMMARY
    # ========================================================

    summary_json = {

        "phase":
            9,

        "scenario":
            args.scenario,

        "seeds":
            seeds,

        "number_of_seeds":
            len(
                seeds
            ),

        "number_of_methods":
            len(
                METHODS
            ),

        "total_runs":
            len(
                METHODS
            )
            *
            len(
                seeds
            ),

        "methods": {

            key:
                value[
                    "display_name"
                ]

            for key, value
            in METHODS.items()
        },

        "classes":
            CLASS_NAMES,

        "class_support":
            EXPECTED_CLASS_SUPPORT,

        "statistical_tests": {

            "confidence_interval":
                "95% Student-t confidence interval",

            "paired_test":
                "Wilcoxon signed-rank",

            "global_test":
                "Friedman repeated-measures",

            "significance_level":
                0.05,
        },

        "best_methods":
            best_methods,
    }


    with open(

        output_dir
        / "phase9_statistics_summary.json",

        "w",

        encoding="utf-8",

    ) as file:

        json.dump(

            json_safe(
                summary_json
            ),

            file,

            indent=4,
        )


    # ========================================================
    # DISPLAY
    # ========================================================

    print_overall_summary(
        overall_summary
    )


    print_class_summary(
        class_summary
    )


    # ========================================================
    # COMPLETE
    # ========================================================

    print()

    print(
        "=" * 100
    )

    print(
        "PHASE 9 STATISTICAL VALIDATION COMPLETE"
    )

    print(
        "=" * 100
    )


    print()

    print(
        "Results saved to:"
    )

    print(
        output_dir
    )


    print()

    print(
        "Main files:"
    )


    print(
        "  overall_summary_statistics.csv"
    )

    print(
        "  overall_rankings.csv"
    )

    print(
        "  wilcoxon_vs_fedavg70.csv"
    )

    print(
        "  friedman_overall.csv"
    )


    if not class_df.empty:

        print(
            "  per_class_summary.csv"
        )

        print(
            "  per_class_recall_ranking.csv"
        )

        print(
            "  per_class_f1_ranking.csv"
        )

        print(
            "  class_detection_stability.csv"
        )

        print(
            "  per_class_wilcoxon_vs_fedavg70.csv"
        )

        print(
            "  per_class_friedman.csv"
        )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()