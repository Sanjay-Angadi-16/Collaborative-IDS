"""
Phase 10 - FedProx Feature-Preservation Statistical Analysis
============================================================

Compares the same repeated seeds for:

    1. Phase 9  FedAvg70
    2. Phase 9  FedProx37
    3. Phase 10 FedProx70

Primary research question
-------------------------
Does restoring all 70 Phase-4 features improve detection relative to
FedProx37, especially PortScan, while preserving the benefits of FedProx?

Default seeds
-------------
42, 52, 62, 72, 82, 92, 102, 112, 122, 132

Expected project structure
--------------------------
results/
├── phase9/
│   ├── fedavg70/non_iid/seed_<seed>/
│   └── fedprox/non_iid/seed_<seed>/
└── phase10/
    └── fedprox70/non_iid/seed_<seed>/

Each run should contain:
    overall_metrics.json
    per_class_metrics.csv
    confusion_matrix.csv
    knowledge_transfer.csv

Outputs
-------
results/phase10/statistics/<scenario>/

    all_overall_runs.csv
    overall_summary.csv
    overall_rankings.csv
    overall_pairwise_tests.csv
    overall_friedman.csv

    per_class_all_runs.csv
    per_class_summary.csv
    per_class_rankings.csv
    per_class_pairwise_tests.csv
    per_class_friedman.csv

    portscan_seed_comparison.csv
    portscan_confusion_summary.csv
    ftp_seed_comparison.csv

    knowledge_transfer_all_runs.csv
    knowledge_transfer_run_means.csv
    knowledge_transfer_summary.csv
    knowledge_transfer_pairwise_tests.csv

    final_winner_summary.json
    phase10_statistics_summary.json

Run
---
python scripts/phase10_statistics.py --scenario non_iid
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.stats import (
        friedmanchisquare,
        rankdata,
        wilcoxon,
    )

    from scipy.stats import t as student_t

except ImportError as exc:

    raise ImportError(
        "\nSciPy is required.\n\n"
        "Install using:\n\n"
        "    pip install scipy pandas numpy\n"
    ) from exc


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


DEFAULT_PHASE9_ROOT = (
    PROJECT_ROOT
    / "results"
    / "phase9"
)


DEFAULT_PHASE10_ROOT = (
    PROJECT_ROOT
    / "results"
    / "phase10"
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

        "phase":
            9,

        "relative_root":
            Path("fedavg70"),
    },


    "fedprox37": {

        "display_name":
            "FedProx37",

        "features":
            37,

        "phase":
            9,

        "relative_root":
            Path("fedprox"),
    },


    "fedprox70": {

        "display_name":
            "FedProx70",

        "features":
            70,

        "phase":
            10,

        "relative_root":
            Path("fedprox70"),
    },
}


# ============================================================
# PAIRED COMPARISONS
# ============================================================

PAIRWISE_COMPARISONS = [

    (
        "fedprox70",
        "fedprox37",
    ),

    (
        "fedprox70",
        "fedavg70",
    ),

    (
        "fedprox37",
        "fedavg70",
    ),
]


# ============================================================
# SEEDS
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


EXPECTED_SUPPORT = {

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
# METRICS
# ============================================================

OVERALL_METRICS = [

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


ALL_OVERALL_METRICS = (

    OVERALL_METRICS

    +

    EFFICIENCY_METRICS
)


PER_CLASS_METRICS = [

    "precision",

    "recall",

    "f1_score",
]


PRIMARY_SELECTION_METRICS = [

    "balanced_accuracy",

    "macro_f1",
]


# ============================================================
# SAFE FLOAT
# ============================================================

def safe_float(
    value: Any,
) -> float:

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
# LOAD JSON
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
# JSON SAFE
# ============================================================

def json_safe(
    value: Any,
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
        tuple,
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


    if (
        isinstance(
            value,
            float,
        )
        and
        math.isnan(
            value
        )
    ):

        return None


    return value


# ============================================================
# METHOD ROOT
# ============================================================

def method_base_root(
    method_key: str,
    phase9_root: Path,
    phase10_root: Path,
) -> Path:

    config = METHODS[
        method_key
    ]


    if (
        config[
            "phase"
        ]
        ==
        9
    ):

        phase_root = (
            phase9_root
        )

    else:

        phase_root = (
            phase10_root
        )


    return (

        phase_root

        /

        config[
            "relative_root"
        ]
    )


# ============================================================
# RUN DIRECTORY
# ============================================================

def run_dir(
    method_key: str,
    scenario: str,
    seed: int,
    phase9_root: Path,
    phase10_root: Path,
) -> Path:

    return (

        method_base_root(

            method_key,

            phase9_root,

            phase10_root,
        )

        / scenario

        / f"seed_{seed}"
    )


# ============================================================
# VERIFY ALL RUNS
# ============================================================

def verify_runs(
    scenario: str,
    seeds: list[int],
    phase9_root: Path,
    phase10_root: Path,
) -> None:

    required_files = [

        "overall_metrics.json",

        "per_class_metrics.csv",

        "confusion_matrix.csv",

        "knowledge_transfer.csv",
    ]


    missing = []


    print()

    print(
        "=" * 105
    )

    print(
        "VERIFYING PHASE 9 + PHASE 10 PAIRED RUNS"
    )

    print(
        "=" * 105
    )


    for (
        method_key,
        config,

    ) in METHODS.items():

        for seed in seeds:

            directory = run_dir(

                method_key,

                scenario,

                seed,

                phase9_root,

                phase10_root,
            )


            missing_in_run = [

                directory
                /
                filename

                for filename
                in required_files

                if not (
                    directory
                    /
                    filename
                ).exists()
            ]


            if missing_in_run:

                print(

                    f"❌ "
                    f"{config['display_name']:12s} "
                    f"seed={seed:<3d} "
                    f"missing "
                    f"{len(missing_in_run)} "
                    f"file(s)"
                )


                missing.extend(
                    missing_in_run
                )


            else:

                print(

                    f"✅ "
                    f"{config['display_name']:12s} "
                    f"seed={seed:<3d} "
                    f"COMPLETE"
                )


    if missing:

        print()

        print(
            "Missing files:"
        )


        for path in missing:

            print(
                f"  {path}"
            )


        raise FileNotFoundError(

            "\nStatistical analysis stopped. "
            f"Missing {len(missing)} required files."
        )


# ============================================================
# 95% CONFIDENCE INTERVAL
# ============================================================

def confidence_interval_95(
    values,
):

    values = np.asarray(

        values,

        dtype=np.float64,
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
# SUMMARY
# ============================================================

def summarize(
    values,
):

    values = np.asarray(

        values,

        dtype=np.float64,
    )


    values = values[

        np.isfinite(
            values
        )
    ]


    if len(
        values
    ) == 0:

        return None


    n = len(
        values
    )


    (
        mean_value,
        ci_lower,
        ci_upper,

    ) = confidence_interval_95(
        values
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


    return {

        "n":
            int(
                n
            ),

        "mean":
            float(
                mean_value
            ),

        "std":
            std_value,

        "median":
            float(
                np.median(
                    values
                )
            ),

        "min":
            float(
                np.min(
                    values
                )
            ),

        "max":
            float(
                np.max(
                    values
                )
            ),

        "ci95_lower":
            float(
                ci_lower
            ),

        "ci95_upper":
            float(
                ci_upper
            ),

        "coefficient_of_variation":
            (

                float(

                    std_value

                    /

                    abs(
                        mean_value
                    )
                )

                if mean_value != 0

                else np.nan
            ),
    }


# ============================================================
# PAIRED COHEN DZ
# ============================================================

def paired_cohen_dz(
    differences,
) -> float:

    differences = np.asarray(

        differences,

        dtype=np.float64,
    )


    differences = differences[

        np.isfinite(
            differences
        )
    ]


    if len(
        differences
    ) < 2:

        return np.nan


    sd = float(

        np.std(
            differences,
            ddof=1,
        )
    )


    mean_difference = float(

        np.mean(
            differences
        )
    )


    if np.isclose(
        sd,
        0.0,
    ):

        if np.isclose(
            mean_difference,
            0.0,
        ):

            return 0.0


        return float(

            np.inf

            if mean_difference > 0

            else -np.inf
        )


    return (

        mean_difference

        /

        sd
    )


# ============================================================
# RANK BISERIAL
# ============================================================

def matched_rank_biserial(
    differences,
) -> float:

    differences = np.asarray(

        differences,

        dtype=np.float64,
    )


    differences = differences[

        np.isfinite(
            differences
        )
    ]


    differences = differences[

        ~np.isclose(
            differences,
            0.0,
        )
    ]


    if len(
        differences
    ) == 0:

        return 0.0


    ranks = rankdata(

        np.abs(
            differences
        )
    )


    positive_rank = float(

        np.sum(

            ranks[
                differences > 0
            ]
        )
    )


    negative_rank = float(

        np.sum(

            ranks[
                differences < 0
            ]
        )
    )


    denominator = (

        positive_rank

        +

        negative_rank
    )


    if denominator == 0:

        return 0.0


    return (

        positive_rank

        -

        negative_rank

    ) / denominator


# ============================================================
# WILCOXON
# ============================================================

def safe_wilcoxon(
    a,
    b,
):

    a = np.asarray(

        a,

        dtype=np.float64,
    )


    b = np.asarray(

        b,

        dtype=np.float64,
    )


    valid = (

        np.isfinite(
            a
        )

        &

        np.isfinite(
            b
        )
    )


    a = a[
        valid
    ]

    b = b[
        valid
    ]


    if len(
        a
    ) == 0:

        return (
            np.nan,
            np.nan,
        )


    differences = (

        a

        -

        b
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

            a,

            b,

            alternative=
                "two-sided",

            zero_method=
                "wilcox",

            method=
                "auto",
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
# HOLM CORRECTION
# ============================================================

def holm_adjust(
    p_values,
):

    p_values = np.asarray(

        p_values,

        dtype=np.float64,
    )


    adjusted = np.full(

        len(
            p_values
        ),

        np.nan,

        dtype=np.float64,
    )


    finite_indices = np.where(

        np.isfinite(
            p_values
        )
    )[0]


    if len(
        finite_indices
    ) == 0:

        return adjusted.tolist()


    finite_p = (

        p_values[
            finite_indices
        ]
    )


    order_local = np.argsort(
        finite_p
    )


    ordered_indices = (

        finite_indices[
            order_local
        ]
    )


    m = len(
        ordered_indices
    )


    running_max = 0.0


    for (
        position,
        original_index,

    ) in enumerate(
        ordered_indices
    ):

        multiplier = (

            m

            -

            position
        )


        candidate = min(

            1.0,

            multiplier

            *

            p_values[
                original_index
            ],
        )


        running_max = max(

            running_max,

            candidate,
        )


        adjusted[
            original_index
        ] = running_max


    return adjusted.tolist()


# ============================================================
# COLLECT OVERALL
# ============================================================

def collect_overall(
    scenario,
    seeds,
    phase9_root,
    phase10_root,
):

    rows = []


    for (
        method_key,
        config,

    ) in METHODS.items():

        for seed in seeds:

            directory = run_dir(

                method_key,

                scenario,

                seed,

                phase9_root,

                phase10_root,
            )


            metrics = load_json(

                directory
                /
                "overall_metrics.json"
            )


            row = {

                "method_key":
                    method_key,

                "method":
                    config[
                        "display_name"
                    ],

                "seed":
                    int(
                        seed
                    ),

                "features":
                    int(
                        config[
                            "features"
                        ]
                    ),

                "phase":
                    int(
                        config[
                            "phase"
                        ]
                    ),
            }


            for metric in (
                ALL_OVERALL_METRICS
            ):

                row[
                    metric
                ] = safe_float(

                    metrics.get(
                        metric
                    )
                )


            row[
                "test_samples"
            ] = safe_float(

                metrics.get(
                    "test_samples"
                )
            )


            row[
                "fedprox_mu"
            ] = safe_float(

                metrics.get(
                    "fedprox_mu"
                )
            )


            row[
                "feature_reduction_ratio"
            ] = safe_float(

                metrics.get(
                    "feature_reduction_ratio"
                )
            )


            rows.append(
                row
            )


    return (

        pd.DataFrame(
            rows
        )

        .sort_values(
            [
                "method",
                "seed",
            ]
        )

        .reset_index(
            drop=True
        )
    )


# ============================================================
# OVERALL SUMMARY
# ============================================================

def build_overall_summary(
    overall_df,
):

    rows = []


    for (
        method_key,
        config,

    ) in METHODS.items():

        subset = overall_df[

            overall_df[
                "method_key"
            ]
            ==
            method_key
        ]


        for metric in (
            ALL_OVERALL_METRICS
        ):

            statistics = summarize(

                subset[
                    metric
                ]
                .to_numpy()
            )


            if statistics is None:

                continue


            rows.append({

                "method_key":
                    method_key,

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

                **statistics,
            })


    return pd.DataFrame(
        rows
    )


# ============================================================
# OVERALL RANKINGS
# ============================================================

def build_overall_rankings(
    summary_df,
):

    rows = []


    for metric in (
        OVERALL_METRICS
    ):

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
                    float(
                        row[
                            "mean"
                        ]
                    ),

                "std":
                    float(
                        row[
                            "std"
                        ]
                    ),

                "ci95_lower":
                    float(
                        row[
                            "ci95_lower"
                        ]
                    ),

                "ci95_upper":
                    float(
                        row[
                            "ci95_upper"
                        ]
                    ),
            })


    return pd.DataFrame(
        rows
    )


# ============================================================
# GENERIC PAIRED TEST ROW
# ============================================================

def paired_test_row(
    left_key,
    right_key,
    metric,
    left_values,
    right_values,
):

    left_values = np.asarray(

        left_values,

        dtype=np.float64,
    )


    right_values = np.asarray(

        right_values,

        dtype=np.float64,
    )


    differences = (

        left_values

        -

        right_values
    )


    (
        statistic,
        p_value,

    ) = safe_wilcoxon(

        left_values,

        right_values,
    )


    return {

        "left_method":
            METHODS[
                left_key
            ][
                "display_name"
            ],

        "right_method":
            METHODS[
                right_key
            ][
                "display_name"
            ],

        "metric":
            metric,

        "n_pairs":
            int(
                len(
                    differences
                )
            ),

        "left_mean":
            float(
                np.mean(
                    left_values
                )
            ),

        "right_mean":
            float(
                np.mean(
                    right_values
                )
            ),

        "mean_delta_left_minus_right":
            float(
                np.mean(
                    differences
                )
            ),

        "median_delta_left_minus_right":
            float(
                np.median(
                    differences
                )
            ),

        "wins_left":
            int(
                np.sum(
                    differences > 0
                )
            ),

        "losses_left":
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

        "wilcoxon_statistic":
            statistic,

        "p_value":
            p_value,

        "cohen_dz":
            paired_cohen_dz(
                differences
            ),

        "rank_biserial":
            matched_rank_biserial(
                differences
            ),
    }


# ============================================================
# OVERALL PAIRED TESTS
# ============================================================

def build_overall_pairwise_tests(
    overall_df,
    seeds,
):

    rows = []


    indexed = {

        method_key:

            (
                overall_df[

                    overall_df[
                        "method_key"
                    ]
                    ==
                    method_key
                ]

                .set_index(
                    "seed"
                )

                .loc[
                    seeds
                ]
            )

        for method_key
        in METHODS
    }


    for metric in (
        OVERALL_METRICS
    ):

        metric_rows = []


        for (
            left_key,
            right_key,

        ) in PAIRWISE_COMPARISONS:

            metric_rows.append(

                paired_test_row(

                    left_key,

                    right_key,

                    metric,

                    indexed[
                        left_key
                    ][
                        metric
                    ].to_numpy(
                        dtype=float
                    ),

                    indexed[
                        right_key
                    ][
                        metric
                    ].to_numpy(
                        dtype=float
                    ),
                )
            )


        adjusted_values = holm_adjust(

            [

                row[
                    "p_value"
                ]

                for row
                in metric_rows
            ]
        )


        for (
            row,
            adjusted_p,

        ) in zip(

            metric_rows,

            adjusted_values,
        ):

            row[
                "holm_adjusted_p"
            ] = adjusted_p


            row[
                "significant_raw_0_05"
            ] = bool(

                np.isfinite(
                    row[
                        "p_value"
                    ]
                )

                and

                row[
                    "p_value"
                ]
                <
                0.05
            )


            row[
                "significant_holm_0_05"
            ] = bool(

                np.isfinite(
                    adjusted_p
                )

                and

                adjusted_p
                <
                0.05
            )


            rows.append(
                row
            )


    return pd.DataFrame(
        rows
    )


# ============================================================
# FRIEDMAN OVERALL
# ============================================================

def build_overall_friedman(
    overall_df,
    seeds,
):

    rows = []

    k = len(
        METHODS
    )

    n = len(
        seeds
    )


    for metric in (
        OVERALL_METRICS
    ):

        samples = []


        for method_key in METHODS:

            values = (

                overall_df[

                    overall_df[
                        "method_key"
                    ]
                    ==
                    method_key
                ]

                .set_index(
                    "seed"
                )

                .loc[
                    seeds,
                    metric
                ]

                .to_numpy(
                    dtype=float
                )
            )


            samples.append(
                values
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


            kendalls_w = (

                statistic

                /

                (
                    n
                    *
                    (
                        k - 1
                    )
                )
            )


        except ValueError:

            statistic = np.nan

            p_value = np.nan

            kendalls_w = np.nan


        rows.append({

            "metric":
                metric,

            "methods":
                k,

            "seeds":
                n,

            "friedman_statistic":
                statistic,

            "p_value":
                p_value,

            "kendalls_w":
                kendalls_w,

            "significant_at_0_05":
                bool(

                    np.isfinite(
                        p_value
                    )

                    and

                    p_value
                    <
                    0.05
                ),
        })


    return pd.DataFrame(
        rows
    )


# ============================================================
# NORMALIZE PER CLASS
# ============================================================

def normalize_per_class(
    df,
):

    df = df.copy()


    rename_map = {}


    if (
        "f1"
        in df.columns
        and
        "f1_score"
        not in df.columns
    ):

        rename_map[
            "f1"
        ] = "f1_score"


    if (
        "f1-score"
        in df.columns
        and
        "f1_score"
        not in df.columns
    ):

        rename_map[
            "f1-score"
        ] = "f1_score"


    if rename_map:

        df = df.rename(
            columns=rename_map
        )


    required = {

        "class",

        "precision",

        "recall",

        "f1_score",

        "support",
    }


    missing = (

        required

        -

        set(
            df.columns
        )
    )


    if missing:

        raise ValueError(

            "Invalid per_class_metrics.csv. "
            f"Missing columns: {sorted(missing)}"
        )


    df[
        "class"
    ] = (

        df[
            "class"
        ]

        .astype(
            str
        )

        .str.strip()
    )


    return df


# ============================================================
# COLLECT PER CLASS
# ============================================================

def collect_per_class(
    scenario,
    seeds,
    phase9_root,
    phase10_root,
):

    rows = []


    for (
        method_key,
        config,

    ) in METHODS.items():

        for seed in seeds:

            directory = run_dir(

                method_key,

                scenario,

                seed,

                phase9_root,

                phase10_root,
            )


            df = normalize_per_class(

                pd.read_csv(

                    directory
                    /
                    "per_class_metrics.csv"
                )
            )


            for _, row in df.iterrows():

                rows.append({

                    "method_key":
                        method_key,

                    "method":
                        config[
                            "display_name"
                        ],

                    "features":
                        config[
                            "features"
                        ],

                    "seed":
                        int(
                            seed
                        ),

                    "class":
                        row[
                            "class"
                        ],

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

                    "f1_score":
                        safe_float(
                            row[
                                "f1_score"
                            ]
                        ),

                    "support":
                        int(
                            row[
                                "support"
                            ]
                        ),
                })


    return (

        pd.DataFrame(
            rows
        )

        .sort_values(
            [
                "class",
                "method",
                "seed",
            ]
        )

        .reset_index(
            drop=True
        )
    )


# ============================================================
# VERIFY CLASS SUPPORT
# ============================================================

def verify_class_support(
    per_class_df,
):

    print()

    print(
        "=" * 105
    )

    print(
        "VERIFYING LOCKED TEST CLASS SUPPORT"
    )

    print(
        "=" * 105
    )


    problems = []


    for (
        class_name,
        expected,

    ) in EXPECTED_SUPPORT.items():

        values = sorted(

            per_class_df.loc[

                per_class_df[
                    "class"
                ]
                ==
                class_name,

                "support",

            ]

            .unique()

            .tolist()
        )


        if values == [
            expected
        ]:

            print(

                f"✅ "
                f"{class_name:18s} "
                f"N={expected}"
            )


        else:

            print(

                f"❌ "
                f"{class_name:18s} "
                f"expected={expected}, "
                f"found={values}"
            )


            problems.append(
                class_name
            )


    if problems:

        raise ValueError(

            "Locked-test support is not identical "
            "across compared runs."
        )


# ============================================================
# PER CLASS SUMMARY
# ============================================================

def build_per_class_summary(
    per_class_df,
):

    rows = []


    for class_name in (
        CLASS_NAMES
    ):

        for (
            method_key,
            config,

        ) in METHODS.items():

            subset = per_class_df[

                (
                    per_class_df[
                        "class"
                    ]
                    ==
                    class_name
                )

                &

                (
                    per_class_df[
                        "method_key"
                    ]
                    ==
                    method_key
                )
            ]


            for metric in (
                PER_CLASS_METRICS
            ):

                statistics = summarize(

                    subset[
                        metric
                    ]
                    .to_numpy()
                )


                if statistics is None:

                    continue


                rows.append({

                    "class":
                        class_name,

                    "method_key":
                        method_key,

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

                    **statistics,
                })


    return pd.DataFrame(
        rows
    )


# ============================================================
# PER CLASS RANKING
# ============================================================

def build_per_class_rankings(
    summary_df,
):

    rows = []


    for class_name in (
        CLASS_NAMES
    ):

        for metric in [

            "recall",

            "f1_score",
        ]:

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

                    "features":
                        int(
                            row[
                                "features"
                            ]
                        ),

                    "mean":
                        float(
                            row[
                                "mean"
                            ]
                        ),

                    "std":
                        float(
                            row[
                                "std"
                            ]
                        ),

                    "ci95_lower":
                        float(
                            row[
                                "ci95_lower"
                            ]
                        ),

                    "ci95_upper":
                        float(
                            row[
                                "ci95_upper"
                            ]
                        ),
                })


    return pd.DataFrame(
        rows
    )


# ============================================================
# PER CLASS PAIRED TESTS
# ============================================================

def build_per_class_pairwise_tests(
    per_class_df,
    seeds,
):

    rows = []


    for class_name in (
        CLASS_NAMES
    ):

        class_df = per_class_df[

            per_class_df[
                "class"
            ]
            ==
            class_name
        ]


        indexed = {

            method_key:

                (
                    class_df[

                        class_df[
                            "method_key"
                        ]
                        ==
                        method_key
                    ]

                    .set_index(
                        "seed"
                    )

                    .loc[
                        seeds
                    ]
                )

            for method_key
            in METHODS
        }


        for metric in [

            "recall",

            "f1_score",
        ]:

            metric_rows = []


            for (
                left_key,
                right_key,

            ) in PAIRWISE_COMPARISONS:

                row = paired_test_row(

                    left_key,

                    right_key,

                    metric,

                    indexed[
                        left_key
                    ][
                        metric
                    ].to_numpy(
                        dtype=float
                    ),

                    indexed[
                        right_key
                    ][
                        metric
                    ].to_numpy(
                        dtype=float
                    ),
                )


                row[
                    "class"
                ] = class_name


                metric_rows.append(
                    row
                )


            adjusted_values = holm_adjust(

                [

                    row[
                        "p_value"
                    ]

                    for row
                    in metric_rows
                ]
            )


            for (
                row,
                adjusted_p,

            ) in zip(

                metric_rows,

                adjusted_values,
            ):

                row[
                    "holm_adjusted_p"
                ] = adjusted_p


                row[
                    "significant_raw_0_05"
                ] = bool(

                    np.isfinite(
                        row[
                            "p_value"
                        ]
                    )

                    and

                    row[
                        "p_value"
                    ]
                    <
                    0.05
                )


                row[
                    "significant_holm_0_05"
                ] = bool(

                    np.isfinite(
                        adjusted_p
                    )

                    and

                    adjusted_p
                    <
                    0.05
                )


                rows.append(
                    row
                )


    return pd.DataFrame(
        rows
    )


# ============================================================
# PER CLASS FRIEDMAN
# ============================================================

def build_per_class_friedman(
    per_class_df,
    seeds,
):

    rows = []


    k = len(
        METHODS
    )


    n = len(
        seeds
    )


    for class_name in (
        CLASS_NAMES
    ):

        class_df = per_class_df[

            per_class_df[
                "class"
            ]
            ==
            class_name
        ]


        for metric in [

            "recall",

            "f1_score",
        ]:

            samples = []


            for method_key in METHODS:

                values = (

                    class_df[

                        class_df[
                            "method_key"
                        ]
                        ==
                        method_key
                    ]

                    .set_index(
                        "seed"
                    )

                    .loc[
                        seeds,
                        metric
                    ]

                    .to_numpy(
                        dtype=float
                    )
                )


                samples.append(
                    values
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


                kendalls_w = (

                    statistic

                    /

                    (
                        n
                        *
                        (
                            k - 1
                        )
                    )
                )


            except ValueError:

                statistic = np.nan

                p_value = np.nan

                kendalls_w = np.nan


            rows.append({

                "class":
                    class_name,

                "metric":
                    metric,

                "friedman_statistic":
                    statistic,

                "p_value":
                    p_value,

                "kendalls_w":
                    kendalls_w,

                "significant_at_0_05":
                    bool(

                        np.isfinite(
                            p_value
                        )

                        and

                        p_value
                        <
                        0.05
                    ),
            })


    return pd.DataFrame(
        rows
    )


# ============================================================
# PORTSCAN / FTP SEED TABLE
# ============================================================

def build_focus_seed_table(
    per_class_df,
    class_name,
    seeds,
):

    subset = per_class_df[

        per_class_df[
            "class"
        ]
        ==
        class_name

    ].copy()


    output_rows = []


    for seed in seeds:

        seed_df = (

            subset[

                subset[
                    "seed"
                ]
                ==
                seed
            ]

            .set_index(
                "method_key"
            )
        )


        row = {

            "seed":
                int(
                    seed
                ),

            "class":
                class_name,
        }


        for method_key in METHODS:

            row[
                f"{method_key}_recall"
            ] = float(

                seed_df.loc[
                    method_key,
                    "recall"
                ]
            )


            row[
                f"{method_key}_f1"
            ] = float(

                seed_df.loc[
                    method_key,
                    "f1_score"
                ]
            )


        row[
            "fedprox70_minus_fedprox37_recall"
        ] = (

            row[
                "fedprox70_recall"
            ]

            -

            row[
                "fedprox37_recall"
            ]
        )


        row[
            "fedprox70_minus_fedprox37_f1"
        ] = (

            row[
                "fedprox70_f1"
            ]

            -

            row[
                "fedprox37_f1"
            ]
        )


        row[
            "fedprox70_minus_fedavg70_recall"
        ] = (

            row[
                "fedprox70_recall"
            ]

            -

            row[
                "fedavg70_recall"
            ]
        )


        output_rows.append(
            row
        )


    return pd.DataFrame(
        output_rows
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def read_confusion_matrix(
    path,
):

    df = pd.read_csv(

        path,

        index_col=0,
    )


    df.index = (

        df.index
        .astype(str)
        .str.strip()
    )


    df.columns = (

        df.columns
        .astype(str)
        .str.strip()
    )


    missing_rows = (

        set(
            CLASS_NAMES
        )

        -

        set(
            df.index
        )
    )


    missing_columns = (

        set(
            CLASS_NAMES
        )

        -

        set(
            df.columns
        )
    )


    if (
        missing_rows
        or
        missing_columns
    ):

        raise ValueError(

            f"Confusion matrix mismatch: {path}"
        )


    return (

        df.loc[

            CLASS_NAMES,

            CLASS_NAMES,
        ]

        .astype(
            np.int64
        )
    )


# ============================================================
# PORTSCAN CONFUSION SUMMARY
# ============================================================

def build_portscan_confusion_summary(
    scenario,
    seeds,
    phase9_root,
    phase10_root,
):

    rows = []


    for (
        method_key,
        config,

    ) in METHODS.items():

        aggregate = np.zeros(

            len(
                CLASS_NAMES
            ),

            dtype=np.int64,
        )


        for seed in seeds:

            directory = run_dir(

                method_key,

                scenario,

                seed,

                phase9_root,

                phase10_root,
            )


            cm = read_confusion_matrix(

                directory
                /
                "confusion_matrix.csv"
            )


            aggregate += (

                cm.loc[

                    "PortScan",

                    CLASS_NAMES,
                ]

                .to_numpy(
                    dtype=np.int64
                )
            )


        total = int(

            np.sum(
                aggregate
            )
        )


        for (
            predicted_class,
            count,

        ) in zip(

            CLASS_NAMES,

            aggregate,
        ):

            rows.append({

                "method":
                    config[
                        "display_name"
                    ],

                "features":
                    config[
                        "features"
                    ],

                "true_class":
                    "PortScan",

                "predicted_class":
                    predicted_class,

                "count_across_seeds":
                    int(
                        count
                    ),

                "share_of_portscan":
                    float(

                        count

                        /

                        total
                    )

                    if total

                    else np.nan,

                "total_portscan_across_seeds":
                    total,
            })


    return pd.DataFrame(
        rows
    )


# ============================================================
# KNOWLEDGE TRANSFER
# ============================================================

def collect_knowledge_transfer(
    scenario,
    seeds,
    phase9_root,
    phase10_root,
):

    detail_rows = []

    run_rows = []


    for (
        method_key,
        config,

    ) in METHODS.items():

        for seed in seeds:

            directory = run_dir(

                method_key,

                scenario,

                seed,

                phase9_root,

                phase10_root,
            )


            df = pd.read_csv(

                directory
                /
                "knowledge_transfer.csv"
            )


            required = {

                "client",

                "local_seen_attack",

                "local_unseen_attack_recall",

                "federated_unseen_attack_recall",

                "knowledge_transfer_gain",
            }


            missing = (

                required

                -

                set(
                    df.columns
                )
            )


            if missing:

                raise ValueError(

                    "Invalid knowledge_transfer.csv "
                    f"in {directory}. "
                    f"Missing {sorted(missing)}"
                )


            for _, row in df.iterrows():

                detail_rows.append({

                    "method_key":
                        method_key,

                    "method":
                        config[
                            "display_name"
                        ],

                    "features":
                        config[
                            "features"
                        ],

                    "seed":
                        int(
                            seed
                        ),

                    "client":
                        row[
                            "client"
                        ],

                    "local_seen_attack":
                        row[
                            "local_seen_attack"
                        ],

                    "local_unseen_attack_recall":
                        safe_float(

                            row[
                                "local_unseen_attack_recall"
                            ]
                        ),

                    "federated_unseen_attack_recall":
                        safe_float(

                            row[
                                "federated_unseen_attack_recall"
                            ]
                        ),

                    "knowledge_transfer_gain":
                        safe_float(

                            row[
                                "knowledge_transfer_gain"
                            ]
                        ),
                })


            run_rows.append({

                "method_key":
                    method_key,

                "method":
                    config[
                        "display_name"
                    ],

                "features":
                    config[
                        "features"
                    ],

                "seed":
                    int(
                        seed
                    ),

                "mean_knowledge_transfer_gain":
                    float(

                        df[
                            "knowledge_transfer_gain"
                        ]
                        .mean()
                    ),

                "mean_federated_unseen_attack_recall":
                    float(

                        df[
                            "federated_unseen_attack_recall"
                        ]
                        .mean()
                    ),
            })


    return (

        pd.DataFrame(
            detail_rows
        ),

        pd.DataFrame(
            run_rows
        ),
    )


# ============================================================
# KNOWLEDGE TRANSFER SUMMARY
# ============================================================

def build_knowledge_transfer_summary(
    run_df,
):

    rows = []


    for (
        method_key,
        config,

    ) in METHODS.items():

        subset = run_df[

            run_df[
                "method_key"
            ]
            ==
            method_key
        ]


        for metric in [

            "mean_knowledge_transfer_gain",

            "mean_federated_unseen_attack_recall",
        ]:

            statistics = summarize(

                subset[
                    metric
                ]
                .to_numpy()
            )


            rows.append({

                "method_key":
                    method_key,

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

                **statistics,
            })


    return pd.DataFrame(
        rows
    )


# ============================================================
# KNOWLEDGE TRANSFER TESTS
# ============================================================

def build_knowledge_transfer_pairwise_tests(
    run_df,
    seeds,
):

    rows = []


    indexed = {

        method_key:

            (
                run_df[

                    run_df[
                        "method_key"
                    ]
                    ==
                    method_key
                ]

                .set_index(
                    "seed"
                )

                .loc[
                    seeds
                ]
            )

        for method_key
        in METHODS
    }


    for metric in [

        "mean_knowledge_transfer_gain",

        "mean_federated_unseen_attack_recall",
    ]:

        metric_rows = []


        for (
            left_key,
            right_key,

        ) in PAIRWISE_COMPARISONS:

            metric_rows.append(

                paired_test_row(

                    left_key,

                    right_key,

                    metric,

                    indexed[
                        left_key
                    ][
                        metric
                    ]
                    .to_numpy(
                        dtype=float
                    ),

                    indexed[
                        right_key
                    ][
                        metric
                    ]
                    .to_numpy(
                        dtype=float
                    ),
                )
            )


        adjusted_values = holm_adjust(

            [

                row[
                    "p_value"
                ]

                for row
                in metric_rows
            ]
        )


        for (
            row,
            adjusted_p,

        ) in zip(

            metric_rows,

            adjusted_values,
        ):

            row[
                "holm_adjusted_p"
            ] = adjusted_p


            row[
                "significant_raw_0_05"
            ] = bool(

                np.isfinite(
                    row[
                        "p_value"
                    ]
                )

                and

                row[
                    "p_value"
                ]
                <
                0.05
            )


            row[
                "significant_holm_0_05"
            ] = bool(

                np.isfinite(
                    adjusted_p
                )

                and

                adjusted_p
                <
                0.05
            )


            rows.append(
                row
            )


    return pd.DataFrame(
        rows
    )


# ============================================================
# GET SUMMARY VALUE
# ============================================================

def get_summary_value(
    summary_df,
    method_key,
    metric,
    field="mean",
):

    row = summary_df[

        (
            summary_df[
                "method_key"
            ]
            ==
            method_key
        )

        &

        (
            summary_df[
                "metric"
            ]
            ==
            metric
        )
    ]


    if row.empty:

        return np.nan


    return float(

        row.iloc[
            0
        ][
            field
        ]
    )


# ============================================================
# GET CLASS VALUE
# ============================================================

def get_class_summary_value(
    summary_df,
    method_key,
    class_name,
    metric,
    field="mean",
):

    row = summary_df[

        (
            summary_df[
                "method_key"
            ]
            ==
            method_key
        )

        &

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
    ]


    if row.empty:

        return np.nan


    return float(

        row.iloc[
            0
        ][
            field
        ]
    )


# ============================================================
# WINNER SUMMARY
# ============================================================

def build_winner_summary(
    overall_summary,
    overall_tests,
    per_class_summary,
    knowledge_transfer_summary,
):

    balanced_means = {

        method_key:

            get_summary_value(

                overall_summary,

                method_key,

                "balanced_accuracy",
            )

        for method_key
        in METHODS
    }


    recommended_key = max(

        balanced_means,

        key=
            balanced_means.get,
    )


    fedprox70_vs_37 = overall_tests[

        (
            overall_tests[
                "left_method"
            ]
            ==
            "FedProx70"
        )

        &

        (
            overall_tests[
                "right_method"
            ]
            ==
            "FedProx37"
        )
    ]


    primary_tests = {}


    for metric in (
        PRIMARY_SELECTION_METRICS
    ):

        row = fedprox70_vs_37[

            fedprox70_vs_37[
                "metric"
            ]
            ==
            metric
        ]


        if row.empty:

            continue


        record = row.iloc[
            0
        ]


        primary_tests[
            metric
        ] = {

            "mean_delta":
                float(

                    record[
                        "mean_delta_left_minus_right"
                    ]
                ),

            "wins":
                int(

                    record[
                        "wins_left"
                    ]
                ),

            "losses":
                int(

                    record[
                        "losses_left"
                    ]
                ),

            "raw_p":
                float(

                    record[
                        "p_value"
                    ]
                ),

            "holm_adjusted_p":
                float(

                    record[
                        "holm_adjusted_p"
                    ]
                ),

            "cohen_dz":
                float(

                    record[
                        "cohen_dz"
                    ]
                ),

            "rank_biserial":
                float(

                    record[
                        "rank_biserial"
                    ]
                ),
        }


    portscan = {

        method_key: {

            "mean_recall":
                get_class_summary_value(

                    per_class_summary,

                    method_key,

                    "PortScan",

                    "recall",
                ),

            "mean_f1":
                get_class_summary_value(

                    per_class_summary,

                    method_key,

                    "PortScan",

                    "f1_score",
                ),
        }

        for method_key
        in METHODS
    }


    ftp = {

        method_key: {

            "mean_recall":
                get_class_summary_value(

                    per_class_summary,

                    method_key,

                    "FTP-Patator",

                    "recall",
                ),

            "mean_f1":
                get_class_summary_value(

                    per_class_summary,

                    method_key,

                    "FTP-Patator",

                    "f1_score",
                ),
        }

        for method_key
        in METHODS
    }


    kt_means = {

        method_key:

            get_summary_value(

                knowledge_transfer_summary,

                method_key,

                "mean_knowledge_transfer_gain",
            )

        for method_key
        in METHODS
    }


    return {

        "selection_rule":
            (
                "Primary: highest mean Balanced Accuracy; "
                "secondary: Macro F1, paired significance, "
                "per-class recall, stability, knowledge transfer "
                "and efficiency."
            ),

        "recommended_method_key":
            recommended_key,

        "recommended_method":
            METHODS[
                recommended_key
            ][
                "display_name"
            ],

        "recommended_features":
            METHODS[
                recommended_key
            ][
                "features"
            ],

        "overall_means": {

            method_key: {

                metric:

                    get_summary_value(

                        overall_summary,

                        method_key,

                        metric,
                    )

                for metric in [

                    "accuracy",

                    "balanced_accuracy",

                    "macro_f1",

                    "weighted_f1",

                    "training_time_seconds",

                    "inference_time_seconds",
                ]
            }

            for method_key
            in METHODS
        },

        "fedprox70_vs_fedprox37_primary_tests":
            primary_tests,

        "portscan":
            portscan,

        "ftp_patator":
            ftp,

        "mean_knowledge_transfer_gain":
            kt_means,

        "interpretation_flags": {

            "fedprox70_restores_portscan":
                bool(

                    portscan[
                        "fedprox70"
                    ][
                        "mean_recall"
                    ]

                    >

                    portscan[
                        "fedprox37"
                    ][
                        "mean_recall"
                    ]
                ),

            "fedprox70_improves_balanced_accuracy":
                bool(

                    balanced_means[
                        "fedprox70"
                    ]

                    >

                    balanced_means[
                        "fedprox37"
                    ]
                ),

            "feature_reduction_tradeoff_detected":
                bool(

                    portscan[
                        "fedprox70"
                    ][
                        "mean_recall"
                    ]

                    >

                    portscan[
                        "fedprox37"
                    ][
                        "mean_recall"
                    ]
                ),
        },
    }


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_primary_summary(
    overall_summary,
    per_class_summary,
    overall_tests,
):

    print()

    print(
        "=" * 105
    )

    print(
        "PHASE 10 FINAL OVERALL SUMMARY"
    )

    print(
        "=" * 105
    )


    for metric in [

        "accuracy",

        "balanced_accuracy",

        "macro_f1",

        "weighted_f1",
    ]:

        subset = overall_summary[

            overall_summary[
                "metric"
            ]
            ==
            metric

        ][

            [
                "method",
                "features",
                "mean",
                "std",
                "ci95_lower",
                "ci95_upper",
            ]
        ]


        subset = subset.sort_values(

            "mean",

            ascending=False,
        )


        print()

        print(
            metric.upper()
        )

        print(
            "-" * 105
        )


        print(

            subset.to_string(

                index=False,

                float_format=
                    lambda x:
                        f"{x:.6f}",
            )
        )


    print()

    print(
        "=" * 105
    )

    print(
        "FEDPROX70 VS FEDPROX37 — PRIMARY PAIRED TESTS"
    )

    print(
        "=" * 105
    )


    focus = overall_tests[

        (
            overall_tests[
                "left_method"
            ]
            ==
            "FedProx70"
        )

        &

        (
            overall_tests[
                "right_method"
            ]
            ==
            "FedProx37"
        )

        &

        (
            overall_tests[
                "metric"
            ]
            .isin(
                [
                    "accuracy",
                    "balanced_accuracy",
                    "macro_f1",
                    "weighted_f1",
                ]
            )
        )
    ][

        [
            "metric",
            "mean_delta_left_minus_right",
            "wins_left",
            "losses_left",
            "p_value",
            "holm_adjusted_p",
            "cohen_dz",
            "rank_biserial",
        ]
    ]


    print(

        focus.to_string(

            index=False,

            float_format=
                lambda x:
                    f"{x:.6f}",
        )
    )


    print()

    print(
        "=" * 105
    )

    print(
        "PER-CLASS MEAN RECALL"
    )

    print(
        "=" * 105
    )


    recall = per_class_summary[

        per_class_summary[
            "metric"
        ]
        ==
        "recall"
    ]


    pivot = recall.pivot(

        index=
            "method",

        columns=
            "class",

        values=
            "mean",
    )


    ordered_classes = [

        class_name

        for class_name
        in CLASS_NAMES

        if class_name
        in pivot.columns
    ]


    pivot = pivot[
        ordered_classes
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
            "Phase 10 statistics: "
            "FedAvg70 vs FedProx37 vs FedProx70"
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


    parser.add_argument(

        "--phase9-root",

        type=Path,

        default=
            DEFAULT_PHASE9_ROOT,
    )


    parser.add_argument(

        "--phase10-root",

        type=Path,

        default=
            DEFAULT_PHASE10_ROOT,
    )


    parser.add_argument(

        "--output-dir",

        type=Path,

        default=None,
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

            "At least two paired seeds are required."
        )


    if len(
        set(
            seeds
        )
    ) != len(
        seeds
    ):

        raise ValueError(

            "Duplicate seeds are not allowed."
        )


    phase9_root = (

        args.phase9_root

        .resolve()
    )


    phase10_root = (

        args.phase10_root

        .resolve()
    )


    if args.output_dir is not None:

        output_dir = (

            args.output_dir

            .resolve()
        )


    else:

        output_dir = (

            phase10_root

            / "statistics"

            / args.scenario
        )


    output_dir.mkdir(

        parents=True,

        exist_ok=True,
    )


    print()

    print(
        "=" * 105
    )

    print(
        "PHASE 10 — FEDPROX FEATURE-PRESERVATION STATISTICAL ANALYSIS"
    )

    print(
        "=" * 105
    )


    print(
        f"Scenario      : "
        f"{args.scenario}"
    )


    print(
        f"Seeds         : "
        f"{seeds}"
    )


    print(
        "Methods       : "
        f"{[METHODS[key]['display_name'] for key in METHODS]}"
    )


    print(
        f"Paired runs   : "
        f"{len(METHODS) * len(seeds)}"
    )


    print(
        f"Phase 9 root  : "
        f"{phase9_root}"
    )


    print(
        f"Phase 10 root : "
        f"{phase10_root}"
    )


    # ========================================================
    # VERIFY
    # ========================================================

    verify_runs(

        scenario=
            args.scenario,

        seeds=
            seeds,

        phase9_root=
            phase9_root,

        phase10_root=
            phase10_root,
    )


    # ========================================================
    # OVERALL
    # ========================================================

    overall_df = collect_overall(

        args.scenario,

        seeds,

        phase9_root,

        phase10_root,
    )


    overall_df.to_csv(

        output_dir
        /
        "all_overall_runs.csv",

        index=False,
    )


    overall_summary = (

        build_overall_summary(
            overall_df
        )
    )


    overall_summary.to_csv(

        output_dir
        /
        "overall_summary.csv",

        index=False,
    )


    overall_rankings = (

        build_overall_rankings(
            overall_summary
        )
    )


    overall_rankings.to_csv(

        output_dir
        /
        "overall_rankings.csv",

        index=False,
    )


    overall_tests = (

        build_overall_pairwise_tests(

            overall_df,

            seeds,
        )
    )


    overall_tests.to_csv(

        output_dir
        /
        "overall_pairwise_tests.csv",

        index=False,
    )


    overall_friedman = (

        build_overall_friedman(

            overall_df,

            seeds,
        )
    )


    overall_friedman.to_csv(

        output_dir
        /
        "overall_friedman.csv",

        index=False,
    )


    # ========================================================
    # PER CLASS
    # ========================================================

    per_class_df = collect_per_class(

        args.scenario,

        seeds,

        phase9_root,

        phase10_root,
    )


    verify_class_support(
        per_class_df
    )


    per_class_df.to_csv(

        output_dir
        /
        "per_class_all_runs.csv",

        index=False,
    )


    per_class_summary = (

        build_per_class_summary(
            per_class_df
        )
    )


    per_class_summary.to_csv(

        output_dir
        /
        "per_class_summary.csv",

        index=False,
    )


    per_class_rankings = (

        build_per_class_rankings(
            per_class_summary
        )
    )


    per_class_rankings.to_csv(

        output_dir
        /
        "per_class_rankings.csv",

        index=False,
    )


    per_class_tests = (

        build_per_class_pairwise_tests(

            per_class_df,

            seeds,
        )
    )


    per_class_tests.to_csv(

        output_dir
        /
        "per_class_pairwise_tests.csv",

        index=False,
    )


    per_class_friedman = (

        build_per_class_friedman(

            per_class_df,

            seeds,
        )
    )


    per_class_friedman.to_csv(

        output_dir
        /
        "per_class_friedman.csv",

        index=False,
    )


    # ========================================================
    # PORTSCAN
    # ========================================================

    portscan_seed = (

        build_focus_seed_table(

            per_class_df,

            "PortScan",

            seeds,
        )
    )


    portscan_seed.to_csv(

        output_dir
        /
        "portscan_seed_comparison.csv",

        index=False,
    )


    # ========================================================
    # FTP-PATATOR
    # ========================================================

    ftp_seed = (

        build_focus_seed_table(

            per_class_df,

            "FTP-Patator",

            seeds,
        )
    )


    ftp_seed.to_csv(

        output_dir
        /
        "ftp_seed_comparison.csv",

        index=False,
    )


    # ========================================================
    # PORTSCAN CONFUSION
    # ========================================================

    portscan_confusion = (

        build_portscan_confusion_summary(

            args.scenario,

            seeds,

            phase9_root,

            phase10_root,
        )
    )


    portscan_confusion.to_csv(

        output_dir
        /
        "portscan_confusion_summary.csv",

        index=False,
    )


    # ========================================================
    # KNOWLEDGE TRANSFER
    # ========================================================

    (
        kt_detail,
        kt_run,

    ) = collect_knowledge_transfer(

        args.scenario,

        seeds,

        phase9_root,

        phase10_root,
    )


    kt_detail.to_csv(

        output_dir
        /
        "knowledge_transfer_all_runs.csv",

        index=False,
    )


    kt_run.to_csv(

        output_dir
        /
        "knowledge_transfer_run_means.csv",

        index=False,
    )


    kt_summary = (

        build_knowledge_transfer_summary(
            kt_run
        )
    )


    kt_summary.to_csv(

        output_dir
        /
        "knowledge_transfer_summary.csv",

        index=False,
    )


    kt_tests = (

        build_knowledge_transfer_pairwise_tests(

            kt_run,

            seeds,
        )
    )


    kt_tests.to_csv(

        output_dir
        /
        "knowledge_transfer_pairwise_tests.csv",

        index=False,
    )


    # ========================================================
    # FINAL WINNER
    # ========================================================

    winner_summary = (

        build_winner_summary(

            overall_summary,

            overall_tests,

            per_class_summary,

            kt_summary,
        )
    )


    with open(

        output_dir
        /
        "final_winner_summary.json",

        "w",

        encoding="utf-8",

    ) as file:

        json.dump(

            json_safe(
                winner_summary
            ),

            file,

            indent=4,
        )


    # ========================================================
    # MASTER SUMMARY
    # ========================================================

    master_summary = {

        "phase":
            10,

        "experiment":
            (
                "FedProx37_vs_FedProx70_"
                "feature_preservation_ablation"
            ),

        "scenario":
            args.scenario,

        "seeds":
            seeds,

        "number_of_seeds":
            len(
                seeds
            ),

        "methods": {

            key: {

                "display_name":
                    value[
                        "display_name"
                    ],

                "features":
                    value[
                        "features"
                    ],

                "phase":
                    value[
                        "phase"
                    ],
            }

            for key, value
            in METHODS.items()
        },

        "confidence_interval":
            (
                "95% Student-t confidence interval"
            ),

        "pairwise_test":
            (
                "Two-sided Wilcoxon signed-rank"
            ),

        "multiple_comparison_correction":
            (
                "Holm-Bonferroni within "
                "each metric comparison family"
            ),

        "global_test":
            (
                "Friedman repeated-measures"
            ),

        "effect_sizes": [

            "paired Cohen dz",

            "matched-pairs rank-biserial correlation",

            "Kendall's W",
        ],

        "alpha":
            0.05,

        "winner":
            winner_summary,
    }


    with open(

        output_dir
        /
        "phase10_statistics_summary.json",

        "w",

        encoding="utf-8",

    ) as file:

        json.dump(

            json_safe(
                master_summary
            ),

            file,

            indent=4,
        )


    # ========================================================
    # DISPLAY
    # ========================================================

    print_primary_summary(

        overall_summary,

        per_class_summary,

        overall_tests,
    )


    print()

    print(
        "=" * 105
    )

    print(
        "FINAL RECOMMENDATION"
    )

    print(
        "=" * 105
    )


    print(

        "Recommended method : "
        f"{winner_summary['recommended_method']}"
    )


    print(

        "Features           : "
        f"{winner_summary['recommended_features']}"
    )


    print(
        "Selection basis    : "
        "Balanced Accuracy + Macro F1 + "
        "paired statistics + per-class detection"
    )


    print()

    print(
        "Results saved to:"
    )

    print(
        output_dir
    )


    print()


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()