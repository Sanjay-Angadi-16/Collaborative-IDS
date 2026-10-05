from __future__ import annotations

import json
import sqlite3
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def percentile(
    values: Sequence[float],
    q: float,
) -> float:
    if not values:
        return 0.0

    ordered = sorted(
        float(
            value
        )
        for value
        in values
    )

    if len(
        ordered
    ) == 1:
        return ordered[
            0
        ]

    position = (
        len(
            ordered
        )
        -
        1
    ) * (
        q / 100.0
    )

    lower = int(
        position
    )

    upper = min(
        lower + 1,
        len(
            ordered
        )
        -
        1,
    )

    fraction = (
        position
        -
        lower
    )

    return (
        ordered[
            lower
        ]
        *
        (
            1.0
            -
            fraction
        )
        +
        ordered[
            upper
        ]
        *
        fraction
    )


def summarize_latencies(
    values: Sequence[float],
) -> Dict[str, float]:
    if not values:
        return {
            "count":
                0,

            "mean_ms":
                0.0,

            "median_ms":
                0.0,

            "p95_ms":
                0.0,

            "p99_ms":
                0.0,

            "min_ms":
                0.0,

            "max_ms":
                0.0,
        }

    return {
        "count":
            len(
                values
            ),

        "mean_ms":
            statistics.mean(
                values
            ),

        "median_ms":
            statistics.median(
                values
            ),

        "p95_ms":
            percentile(
                values,
                95,
            ),

        "p99_ms":
            percentile(
                values,
                99,
            ),

        "min_ms":
            min(
                values
            ),

        "max_ms":
            max(
                values
            ),
    }


def timed_sql_query(
    database_path: Path,
    *,
    sql: str,
    params: Sequence[Any] = (),
    repeats: int = 100,
) -> Dict[str, Any]:
    """
    Benchmark one indexed SQLite query repeatedly on the local machine.
    Results are implementation/hardware specific.
    """

    database_path = Path(
        database_path
    )

    latencies = []
    row_counts = []

    connection = sqlite3.connect(
        database_path,
        timeout=10.0,
    )

    connection.row_factory = sqlite3.Row

    try:
        # One untimed warm-up.
        connection.execute(
            sql,
            tuple(
                params
            ),
        ).fetchall()

        for _ in range(
            int(
                repeats
            )
        ):
            start = time.perf_counter()

            rows = connection.execute(
                sql,
                tuple(
                    params
                ),
            ).fetchall()

            elapsed_ms = (
                time.perf_counter()
                -
                start
            ) * 1000.0

            latencies.append(
                elapsed_ms
            )

            row_counts.append(
                len(
                    rows
                )
            )

    finally:
        connection.close()

    summary = summarize_latencies(
        latencies
    )

    summary[
        "rows_returned_last"
    ] = (
        row_counts[
            -1
        ]
        if row_counts
        else
        0
    )

    return summary
