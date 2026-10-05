r"""
Phase 11 - Knowledge Transfer + Local Knowledge Retention
=========================================================

Purpose
-------
Phase 10 selected FedProx70 as the final federated classifier.

Phase 11 answers:

RQ1-A:
    Does federation teach clients attacks that they never observed locally?

RQ1-B:
    Does federation preserve attacks that clients already learned locally?

Metrics
-------
For locally SEEN attacks:

    Retention Delta
        = FedProx70 Recall - LocalOnly Recall

    Retention Ratio
        = FedProx70 Recall / LocalOnly Recall

For locally UNSEEN attacks:

    Transfer Gain
        = FedProx70 Recall - LocalOnly Recall


Experimental fairness
---------------------
The LocalOnly model uses:

- same 70 features
- same scaler
- same MLP
- same global class weights
- same AdamW settings
- same batch size
- same local epochs
- same seed
- same client data
- same total local-data exposure

Example:

10 federated rounds x 1 local epoch

becomes:

10 local-only pseudo-rounds x 1 local epoch

No aggregation happens for LocalOnly.

The locked test is evaluated only AFTER training.


Outputs
-------
results/phase11/knowledge_retention/<scenario>/seed_<seed>/

    client_training_distribution.csv
    local_only_per_class_metrics.csv
    fedprox70_per_class_metrics.csv
    retention_all_clients.csv
    transfer_all_clients.csv
    per_client_summary.csv
    local_only_training_history.csv
    phase11_summary.json


Local models
------------
artifacts/phase11/local_only/<scenario>/seed_<seed>/

    client_1_local_model.pt
    client_2_local_model.pt
    ...


Aggregate outputs
-----------------
results/phase11/knowledge_retention/<scenario>/aggregate/

    retention_all_seeds.csv
    transfer_all_seeds.csv

    retention_summary_across_seeds.csv
    transfer_summary_across_seeds.csv

    retention_paired_tests.csv
    transfer_paired_tests.csv

    aggregate_summary.json


Run one seed
------------
python scripts/phase11_knowledge_retention.py --scenario non_iid --seed 42


Run all 10 seeds - Windows CMD
------------------------------
for %s in (42 52 62 72 82 92 102 112 122 132) do python scripts\phase11_knowledge_retention.py --scenario non_iid --seed %s
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn


# ============================================================
# OPTIONAL SCIPY
# ============================================================

try:

    from scipy.stats import t as student_t
    from scipy.stats import wilcoxon

    SCIPY_AVAILABLE = True

except ImportError:

    student_t = None
    wilcoxon = None

    SCIPY_AVAILABLE = False


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
# REUSE PHASE 9 INFRASTRUCTURE
# ============================================================

try:

    import phase9_fedavg70 as phase4

except ModuleNotFoundError as exc:

    raise ModuleNotFoundError(

        "\nMissing:\n"

        "scripts\\phase9_fedavg70.py\n\n"

        "Keep phase11_knowledge_retention.py "

        "inside the same scripts folder."

    ) from exc


# ============================================================
# CONFIG
# ============================================================

DEFAULT_SEED = phase4.SEED

N_CLIENTS = phase4.N_CLIENTS


EXPECTED_FEATURES = 70


DEFAULT_ROUNDS = (

    phase4.DEFAULT_ROUNDS

)


DEFAULT_LOCAL_EPOCHS = (

    phase4.DEFAULT_LOCAL_EPOCHS

)


DEFAULT_BATCH_SIZE = (

    phase4.DEFAULT_BATCH_SIZE

)


DEFAULT_CHUNK_SIZE = (

    phase4.DEFAULT_CHUNK_SIZE

)


DEFAULT_LEARNING_RATE = (

    phase4.DEFAULT_LEARNING_RATE

)


DEFAULT_WEIGHT_DECAY = (

    phase4.DEFAULT_WEIGHT_DECAY

)


GRADIENT_CLIP = 5.0


# ============================================================
# PATHS
# ============================================================

PHASE10_MODEL_ROOT = (

    PROJECT_ROOT

    / "artifacts"

    / "phase10"

    / "fedprox70"

)


RESULT_ROOT = (

    PROJECT_ROOT

    / "results"

    / "phase11"

    / "knowledge_retention"

)


ARTIFACT_ROOT = (

    PROJECT_ROOT

    / "artifacts"

    / "phase11"

    / "local_only"

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

    "phase11_knowledge_retention"

)


# ============================================================
# SEED
# ============================================================

def set_seed(

    seed: int,

) -> None:


    random.seed(

        seed

    )


    np.random.seed(

        seed

    )


    torch.manual_seed(

        seed

    )


    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(

            seed

        )


    if hasattr(

        torch.backends,

        "cudnn",

    ):

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False


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
# SAVE JSON
# ============================================================

def save_json(

    path: Path,

    payload: dict,

) -> None:


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
# TORCH LOAD
# ============================================================

def load_torch(

    path: Path,

    device,

):


    try:

        return torch.load(

            path,

            map_location=device,

            weights_only=False,

        )


    except TypeError:

        return torch.load(

            path,

            map_location=device,

        )


# ============================================================
# SAFE RATIO
# ============================================================

def safe_ratio(

    numerator: float,

    denominator: float,

):


    if np.isclose(

        denominator,

        0.0,

    ):

        return np.nan


    return float(

        numerator

        /

        denominator

    )


# ============================================================
# FIND BENIGN LABEL
# ============================================================

def detect_benign_label(

    label_mapping,

):


    labels = list(

        label_mapping.keys()

    )


    preferred_labels = [

        "BENIGN",

        "Benign",

        "benign",

        "NORMAL",

        "Normal",

        "normal",

    ]


    for preferred in (

        preferred_labels

    ):


        if preferred in labels:

            return preferred


    for label in labels:


        value = (

            str(

                label

            )

            .strip()

            .lower()

        )


        if (

            "benign"

            in value

            or

            "normal"

            in value

        ):

            return label


    raise ValueError(

        "\nCould not detect benign label.\n"

        f"Available labels:\n{labels}"

    )


# ============================================================
# VERIFY 70 FEATURES
# ============================================================

def verify_70_features(

    active_features,

    active_indices,

    feature_columns,

):


    if (

        len(

            active_features

        )

        !=

        EXPECTED_FEATURES

    ):

        raise ValueError(

            "\nPhase 11 requires "

            "the final 70-feature space.\n"

            f"Found: {len(active_features)}"

        )


    active_indices_array = np.asarray(

        active_indices,

        dtype=np.int64,

    )


    if (

        len(

            active_indices_array

        )

        !=

        EXPECTED_FEATURES

    ):

        raise ValueError(

            "\nExpected exactly "

            "70 active indices.\n"

            f"Found: {len(active_indices_array)}"

        )


    reconstructed = [

        feature_columns[

            int(

                index

            )

        ]

        for index

        in active_indices_array

    ]


    if (

        list(

            reconstructed

        )

        !=

        list(

            active_features

        )

    ):

        raise ValueError(

            "\nFeature/index mapping mismatch."

        )


    return (

        list(

            active_features

        ),

        active_indices_array,

    )


# ============================================================
# CLIENT LABEL COUNTS
# ============================================================

def read_client_label_counts(

    client_file,

    chunk_size,

    label_mapping,

):


    canonical = {

        str(

            label

        )

        .strip()

        .lower():

            str(

                label

            )

            .strip()

        for label

        in label_mapping.keys()

    }


    counts = {

        str(

            label

        ).strip():

            0

        for label

        in label_mapping.keys()

    }


    reader = pd.read_csv(

        client_file,

        usecols=[

            phase4.LABEL_COL

        ],

        chunksize=

            chunk_size,

    )


    unknown = set()


    for chunk in reader:


        values = (

            chunk[

                phase4.LABEL_COL

            ]

            .astype(

                str

            )

            .str.strip()

            .value_counts()

        )


        for (

            raw_label,

            count,

        ) in values.items():


            key = (

                str(

                    raw_label

                )

                .strip()

                .lower()

            )


            if key not in canonical:

                unknown.add(

                    str(

                        raw_label

                    )

                )

                continue


            canonical_label = (

                canonical[

                    key

                ]

            )


            counts[

                canonical_label

            ] += int(

                count

            )


    if unknown:

        raise ValueError(

            "\nUnknown labels found:\n"

            f"{sorted(unknown)}"

        )


    return counts


# ============================================================
# PER CLASS DATAFRAME
# ============================================================

def build_per_class_df(

    report,

    class_names,

    seed,

    model_type,

    client_id=None,

):


    rows = []


    for class_name in (

        class_names

    ):


        metrics = (

            report[

                class_name

            ]

        )


        rows.append({

            "seed":

                seed,

            "model_type":

                model_type,

            "client_id":

                client_id,

            "class":

                class_name,

            "precision":

                float(

                    metrics[

                        "precision"

                    ]

                ),

            "recall":

                float(

                    metrics[

                        "recall"

                    ]

                ),

            "f1_score":

                float(

                    metrics[

                        "f1-score"

                    ]

                ),

            "support":

                int(

                    metrics[

                        "support"

                    ]

                ),

        })


    return pd.DataFrame(

        rows

    )


# ============================================================
# LOAD FEDPROX70
# ============================================================

def load_fedprox70_model(

    scenario,

    seed,

    device,

    input_dim,

    num_classes,

    active_features,

):


    model_file = (

        PHASE10_MODEL_ROOT

        / scenario

        / f"seed_{seed}"

        / "fedprox70_model.pt"

    )


    if not model_file.exists():

        raise FileNotFoundError(

            "\nPhase 10 FedProx70 "

            "model not found:\n"

            f"{model_file}\n\n"

            "Run Phase 10 first."

        )


    checkpoint = (

        load_torch(

            model_file,

            device,

        )

    )


    checkpoint_input_dim = int(

        checkpoint.get(

            "input_dim",

            input_dim,

        )

    )


    if (

        checkpoint_input_dim

        !=

        input_dim

    ):

        raise ValueError(

            "\nFedProx70 input dimension mismatch."

        )


    checkpoint_seed = (

        checkpoint.get(

            "seed"

        )

    )


    if (

        checkpoint_seed

        is not None

        and

        int(

            checkpoint_seed

        )

        !=

        int(

            seed

        )

    ):

        raise ValueError(

            "\nFedProx70 seed mismatch.\n"

            f"Checkpoint: {checkpoint_seed}\n"

            f"Requested : {seed}"

        )


    checkpoint_features = (

        checkpoint.get(

            "features"

        )

    )


    if (

        checkpoint_features

        is not None

    ):


        if (

            list(

                checkpoint_features

            )

            !=

            list(

                active_features

            )

        ):

            raise ValueError(

                "\nFedProx70 feature schema mismatch."

            )


    state_dict = (

        checkpoint.get(

            "model_state_dict"

        )

    )


    if state_dict is None:

        raise KeyError(

            "\nCheckpoint missing model_state_dict."

        )


    model = (

        phase4.IDSMLP(

            input_dim=

                input_dim,

            num_classes=

                num_classes,

        )

        .to(

            device

        )

    )


    model.load_state_dict(

        state_dict

    )


    model.eval()


    return (

        model,

        model_file,

        checkpoint,

    )


# ============================================================
# LOCAL MODEL INIT
# ============================================================

def initialize_local_model(

    input_dim,

    num_classes,

    device,

    seed,

):


    phase4.set_seed(

        seed

    )


    return (

        phase4.IDSMLP(

            input_dim=

                input_dim,

            num_classes=

                num_classes,

        )

        .to(

            device

        )

    )


# ============================================================
# TRAIN LOCAL-ONLY
# ============================================================

def train_local_only(

    client_id,

    client_file,

    scaler,

    feature_columns,

    active_indices,

    label_mapping,

    class_weights,

    input_dim,

    device,

    rounds,

    local_epochs,

    batch_size,

    chunk_size,

    learning_rate,

    weight_decay,

    seed,

):


    training_start = (

        time.perf_counter()

    )


    model = (

        initialize_local_model(

            input_dim=

                input_dim,

            num_classes=

                len(

                    label_mapping

                ),

            device=

                device,

            seed=

                seed,

        )

    )


    class_weight_tensor = (

        torch.tensor(

            class_weights,

            dtype=

                torch.float32,

            device=

                device,

        )

    )


    criterion = (

        nn.CrossEntropyLoss(

            weight=

                class_weight_tensor

        )

    )


    history = []


    # ========================================================
    # PSEUDO ROUNDS
    # ========================================================

    for pseudo_round in range(

        1,

        rounds + 1,

    ):


        round_start = (

            time.perf_counter()

        )


        # Same lifecycle as FL:
        # optimizer recreated each local round.
        optimizer = (

            torch.optim.AdamW(

                model.parameters(),

                lr=

                    learning_rate,

                weight_decay=

                    weight_decay,

            )

        )


        total_loss = 0.0

        total_correct = 0

        total_samples = 0


        model.train()


        for local_epoch in range(

            1,

            local_epochs + 1,

        ):


            # Same seed structure used
            # in Phase 10 FedProx70.
            rng = (

                np.random.default_rng(

                    seed

                    +

                    pseudo_round
                    *
                    1000

                    +

                    client_id
                    *
                    100

                    +

                    local_epoch

                )

            )


            reader = pd.read_csv(

                client_file,

                chunksize=

                    chunk_size,

            )


            for chunk in reader:


                X = (

                    phase4.transform_features(

                        df=

                            chunk,

                        feature_columns=

                            feature_columns,

                        scaler=

                            scaler,

                        active_indices=

                            active_indices,

                    )

                )


                y = (

                    phase4.encode_labels(

                        chunk[

                            phase4.LABEL_COL

                        ],

                        label_mapping,

                    )

                )


                if len(

                    y

                ) == 0:

                    continue


                indices = (

                    rng.permutation(

                        len(

                            y

                        )

                    )

                )


                for start in range(

                    0,

                    len(

                        indices

                    ),

                    batch_size,

                ):


                    batch_indices = (

                        indices[

                            start:

                            start
                            +
                            batch_size

                        ]

                    )


                    X_batch = (

                        torch.from_numpy(

                            X[

                                batch_indices

                            ]

                            .astype(

                                np.float32,

                                copy=False,

                            )

                        )

                        .to(

                            device

                        )

                    )


                    y_batch = (

                        torch.from_numpy(

                            y[

                                batch_indices

                            ]

                            .astype(

                                np.int64,

                                copy=False,

                            )

                        )

                        .to(

                            device

                        )

                    )


                    optimizer.zero_grad(

                        set_to_none=True

                    )


                    logits = model(

                        X_batch

                    )


                    loss = criterion(

                        logits,

                        y_batch,

                    )


                    loss.backward()


                    torch.nn.utils.clip_grad_norm_(

                        model.parameters(),

                        max_norm=

                            GRADIENT_CLIP,

                    )


                    optimizer.step()


                    predictions = (

                        torch.argmax(

                            logits,

                            dim=1,

                        )

                    )


                    batch_samples = int(

                        y_batch.size(

                            0

                        )

                    )


                    batch_correct = int(

                        (

                            predictions

                            ==

                            y_batch

                        )

                        .sum()

                        .item()

                    )


                    total_loss += (

                        float(

                            loss.item()

                        )

                        *

                        batch_samples

                    )


                    total_correct += (

                        batch_correct

                    )


                    total_samples += (

                        batch_samples

                    )


        if total_samples == 0:

            raise RuntimeError(

                f"Client {client_id} "

                "produced zero samples."

            )


        mean_loss = float(

            total_loss

            /

            total_samples

        )


        accuracy = float(

            total_correct

            /

            total_samples

        )


        history.append({

            "seed":

                seed,

            "client_id":

                client_id,

            "pseudo_round":

                pseudo_round,

            "local_epochs":

                local_epochs,

            "loss":

                mean_loss,

            "accuracy":

                accuracy,

            "round_training_seconds":

                float(

                    time.perf_counter()

                    -

                    round_start

                ),

        })


        logger.info(

            "LocalOnly | "

            "seed=%d | "

            "client=%d | "

            "round=%d/%d | "

            "loss=%.6f | "

            "acc=%.6f",

            seed,

            client_id,

            pseudo_round,

            rounds,

            mean_loss,

            accuracy,

        )


    training_seconds = float(

        time.perf_counter()

        -

        training_start

    )


    return (

        model,

        pd.DataFrame(

            history

        ),

        training_seconds,

    )


# ============================================================
# STATISTICS
# ============================================================

def mean_std_ci95(

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


    if len(

        values

    ) == 0:

        return (

            np.nan,

            np.nan,

            np.nan,

            np.nan,

        )


    mean_value = float(

        values.mean()

    )


    if len(

        values

    ) == 1:

        return (

            mean_value,

            0.0,

            mean_value,

            mean_value,

        )


    std_value = float(

        values.std(

            ddof=1

        )

    )


    if SCIPY_AVAILABLE:

        critical = float(

            student_t.ppf(

                0.975,

                df=

                    len(

                        values

                    )

                    -

                    1,

            )

        )


    else:

        critical = 1.96


    margin = (

        critical

        *

        std_value

        /

        math.sqrt(

            len(

                values

            )

        )

    )


    return (

        mean_value,

        std_value,

        mean_value
        -
        margin,

        mean_value
        +
        margin,

    )


# ============================================================
# WILCOXON
# ============================================================

def safe_wilcoxon(

    global_values,

    local_values,

):


    if not SCIPY_AVAILABLE:

        return (

            np.nan,

            np.nan,

        )


    global_values = np.asarray(

        global_values,

        dtype=float,

    )


    local_values = np.asarray(

        local_values,

        dtype=float,

    )


    valid = (

        np.isfinite(

            global_values

        )

        &

        np.isfinite(

            local_values

        )

    )


    global_values = (

        global_values[

            valid

        ]

    )


    local_values = (

        local_values[

            valid

        ]

    )


    if len(

        global_values

    ) == 0:

        return (

            np.nan,

            np.nan,

        )


    differences = (

        global_values

        -

        local_values

    )


    if np.allclose(

        differences,

        0.0,

    ):

        return (

            0.0,

            1.0,

        )


    result = wilcoxon(

        global_values,

        local_values,

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


# ============================================================
# AGGREGATE EXISTING SEEDS
# ============================================================

def aggregate_completed_seeds(

    scenario,

):


    scenario_root = (

        RESULT_ROOT

        / scenario

    )


    aggregate_dir = (

        scenario_root

        / "aggregate"

    )


    aggregate_dir.mkdir(

        parents=True,

        exist_ok=True,

    )


    retention_frames = []

    transfer_frames = []

    completed_seeds = []


    for seed_dir in (

        scenario_root.glob(

            "seed_*"

        )

    ):


        if not seed_dir.is_dir():

            continue


        retention_file = (

            seed_dir

            / "retention_all_clients.csv"

        )


        transfer_file = (

            seed_dir

            / "transfer_all_clients.csv"

        )


        if (

            not retention_file.exists()

            or

            not transfer_file.exists()

        ):

            continue


        seed = int(

            seed_dir.name.split(

                "_"

            )[-1]

        )


        completed_seeds.append(

            seed

        )


        retention_frames.append(

            pd.read_csv(

                retention_file

            )

        )


        transfer_frames.append(

            pd.read_csv(

                transfer_file

            )

        )


    if not completed_seeds:

        return None


    completed_seeds = sorted(

        completed_seeds

    )


    retention_all = (

        pd.concat(

            retention_frames,

            ignore_index=True,

        )

        .sort_values(

            [

                "client_id",

                "attack_class",

                "seed",

            ]

        )

    )


    transfer_all = (

        pd.concat(

            transfer_frames,

            ignore_index=True,

        )

        .sort_values(

            [

                "client_id",

                "attack_class",

                "seed",

            ]

        )

    )


    retention_all.to_csv(

        aggregate_dir

        / "retention_all_seeds.csv",

        index=False,

    )


    transfer_all.to_csv(

        aggregate_dir

        / "transfer_all_seeds.csv",

        index=False,

    )


    # ========================================================
    # RETENTION SUMMARY
    # ========================================================

    retention_summary = []


    for (

        client_id,

        attack_class,

    ), group in retention_all.groupby(

        [

            "client_id",

            "attack_class",

        ]

    ):


        for metric in (

            "local_only_recall",

            "fedprox70_recall",

            "retention_delta",

            "retention_ratio",

        ):


            (

                mean_value,

                std_value,

                ci_lower,

                ci_upper,

            ) = mean_std_ci95(

                group[

                    metric

                ]

                .to_numpy(

                    dtype=float

                )

            )


            retention_summary.append({

                "client_id":

                    int(

                        client_id

                    ),

                "attack_class":

                    attack_class,

                "metric":

                    metric,

                "n":

                    int(

                        group[

                            metric

                        ]

                        .notna()

                        .sum()

                    ),

                "mean":

                    mean_value,

                "std":

                    std_value,

                "ci95_lower":

                    ci_lower,

                "ci95_upper":

                    ci_upper,

            })


    pd.DataFrame(

        retention_summary

    ).to_csv(

        aggregate_dir

        / "retention_summary_across_seeds.csv",

        index=False,

    )


    # ========================================================
    # TRANSFER SUMMARY
    # ========================================================

    transfer_summary = []


    for (

        client_id,

        attack_class,

    ), group in transfer_all.groupby(

        [

            "client_id",

            "attack_class",

        ]

    ):


        for metric in (

            "local_only_recall",

            "fedprox70_recall",

            "transfer_gain",

        ):


            (

                mean_value,

                std_value,

                ci_lower,

                ci_upper,

            ) = mean_std_ci95(

                group[

                    metric

                ]

                .to_numpy(

                    dtype=float

                )

            )


            transfer_summary.append({

                "client_id":

                    int(

                        client_id

                    ),

                "attack_class":

                    attack_class,

                "metric":

                    metric,

                "n":

                    int(

                        group[

                            metric

                        ]

                        .notna()

                        .sum()

                    ),

                "mean":

                    mean_value,

                "std":

                    std_value,

                "ci95_lower":

                    ci_lower,

                "ci95_upper":

                    ci_upper,

            })


    pd.DataFrame(

        transfer_summary

    ).to_csv(

        aggregate_dir

        / "transfer_summary_across_seeds.csv",

        index=False,

    )


    # ========================================================
    # RETENTION PAIRED TESTS
    # ========================================================

    retention_tests = []


    for (

        client_id,

        attack_class,

    ), group in retention_all.groupby(

        [

            "client_id",

            "attack_class",

        ]

    ):


        group = (

            group

            .sort_values(

                "seed"

            )

        )


        local_values = (

            group[

                "local_only_recall"

            ]

            .to_numpy(

                dtype=float

            )

        )


        global_values = (

            group[

                "fedprox70_recall"

            ]

            .to_numpy(

                dtype=float

            )

        )


        delta = (

            global_values

            -

            local_values

        )


        (

            statistic,

            p_value,

        ) = safe_wilcoxon(

            global_values,

            local_values,

        )


        retention_tests.append({

            "client_id":

                int(

                    client_id

                ),

            "attack_class":

                attack_class,

            "n_pairs":

                len(

                    group

                ),

            "mean_local_only_recall":

                float(

                    local_values.mean()

                ),

            "mean_fedprox70_recall":

                float(

                    global_values.mean()

                ),

            "mean_retention_delta":

                float(

                    delta.mean()

                ),

            "global_wins":

                int(

                    (

                        delta

                        >

                        0

                    ).sum()

                ),

            "local_wins":

                int(

                    (

                        delta

                        <

                        0

                    ).sum()

                ),

            "ties":

                int(

                    np.isclose(

                        delta,

                        0.0,

                    ).sum()

                ),

            "wilcoxon_statistic":

                statistic,

            "p_value":

                p_value,

        })


    pd.DataFrame(

        retention_tests

    ).to_csv(

        aggregate_dir

        / "retention_paired_tests.csv",

        index=False,

    )


    # ========================================================
    # TRANSFER PAIRED TESTS
    # ========================================================

    transfer_tests = []


    for (

        client_id,

        attack_class,

    ), group in transfer_all.groupby(

        [

            "client_id",

            "attack_class",

        ]

    ):


        group = (

            group

            .sort_values(

                "seed"

            )

        )


        local_values = (

            group[

                "local_only_recall"

            ]

            .to_numpy(

                dtype=float

            )

        )


        global_values = (

            group[

                "fedprox70_recall"

            ]

            .to_numpy(

                dtype=float

            )

        )


        gain = (

            global_values

            -

            local_values

        )


        (

            statistic,

            p_value,

        ) = safe_wilcoxon(

            global_values,

            local_values,

        )


        transfer_tests.append({

            "client_id":

                int(

                    client_id

                ),

            "attack_class":

                attack_class,

            "n_pairs":

                len(

                    group

                ),

            "mean_local_only_recall":

                float(

                    local_values.mean()

                ),

            "mean_fedprox70_recall":

                float(

                    global_values.mean()

                ),

            "mean_transfer_gain":

                float(

                    gain.mean()

                ),

            "global_wins":

                int(

                    (

                        gain

                        >

                        0

                    ).sum()

                ),

            "local_wins":

                int(

                    (

                        gain

                        <

                        0

                    ).sum()

                ),

            "ties":

                int(

                    np.isclose(

                        gain,

                        0.0,

                    ).sum()

                ),

            "wilcoxon_statistic":

                statistic,

            "p_value":

                p_value,

        })


    pd.DataFrame(

        transfer_tests

    ).to_csv(

        aggregate_dir

        / "transfer_paired_tests.csv",

        index=False,

    )


    # ========================================================
    # MASTER SUMMARY
    # ========================================================

    summary = {

        "scenario":

            scenario,

        "completed_seeds":

            completed_seeds,

        "number_of_completed_seeds":

            len(

                completed_seeds

            ),

        "scipy_available":

            SCIPY_AVAILABLE,

        "retention": {

            "mean_local_seen_recall":

                float(

                    retention_all[

                        "local_only_recall"

                    ].mean()

                ),

            "mean_fedprox70_seen_recall":

                float(

                    retention_all[

                        "fedprox70_recall"

                    ].mean()

                ),

            "mean_retention_delta":

                float(

                    retention_all[

                        "retention_delta"

                    ].mean()

                ),

            "mean_retention_ratio":

                (

                    float(

                        retention_all[

                            "retention_ratio"

                        ]

                        .dropna()

                        .mean()

                    )

                    if

                    retention_all[

                        "retention_ratio"

                    ]

                    .notna()

                    .any()

                    else

                    None

                ),

        },

        "transfer": {

            "mean_local_unseen_recall":

                float(

                    transfer_all[

                        "local_only_recall"

                    ].mean()

                ),

            "mean_fedprox70_unseen_recall":

                float(

                    transfer_all[

                        "fedprox70_recall"

                    ].mean()

                ),

            "mean_transfer_gain":

                float(

                    transfer_all[

                        "transfer_gain"

                    ].mean()

                ),

        },

    }


    save_json(

        aggregate_dir

        / "aggregate_summary.json",

        summary,

    )


    return summary


# ============================================================
# MAIN
# ============================================================

def main():


    parser = (

        argparse.ArgumentParser()

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

        "--seed",

        type=int,

        default=

            DEFAULT_SEED,

    )


    parser.add_argument(

        "--rounds",

        type=int,

        default=

            DEFAULT_ROUNDS,

    )


    parser.add_argument(

        "--local-epochs",

        type=int,

        default=

            DEFAULT_LOCAL_EPOCHS,

    )


    parser.add_argument(

        "--batch-size",

        type=int,

        default=

            DEFAULT_BATCH_SIZE,

    )


    parser.add_argument(

        "--chunk-size",

        type=int,

        default=

            DEFAULT_CHUNK_SIZE,

    )


    parser.add_argument(

        "--learning-rate",

        type=float,

        default=

            DEFAULT_LEARNING_RATE,

    )


    parser.add_argument(

        "--weight-decay",

        type=float,

        default=

            DEFAULT_WEIGHT_DECAY,

    )


    args = parser.parse_args()


    if (

        args.rounds

        <

        1

    ):

        raise ValueError(

            "rounds must be >= 1."

        )


    if (

        args.local_epochs

        <

        1

    ):

        raise ValueError(

            "local-epochs must be >= 1."

        )


    set_seed(

        args.seed

    )


    device = torch.device(

        "cuda"

        if

        torch.cuda.is_available()

        else

        "cpu"

    )


    scenario_dir = (

        phase4.FEDERATED_ROOT

        / args.scenario

    )


    if not scenario_dir.exists():

        raise FileNotFoundError(

            scenario_dir

        )


    result_dir = (

        RESULT_ROOT

        / args.scenario

        / f"seed_{args.seed}"

    )


    artifact_dir = (

        ARTIFACT_ROOT

        / args.scenario

        / f"seed_{args.seed}"

    )


    result_dir.mkdir(

        parents=True,

        exist_ok=True,

    )


    artifact_dir.mkdir(

        parents=True,

        exist_ok=True,

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


    (

        all_features,

        all_active_indices,

    ) = verify_70_features(

        active_features,

        active_indices,

        feature_columns,

    )


    benign_label = (

        detect_benign_label(

            label_mapping

        )

    )


    attack_classes = [

        label

        for label

        in label_mapping.keys()

        if label

        !=

        benign_label

    ]


    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    global_counts = (

        phase4.calculate_global_class_counts(

            label_mapping,

            args.chunk_size,

        )

    )


    global_weights = (

        phase4.calculate_global_class_weights(

            global_counts

        )

    )


    # ========================================================
    # LOAD FEDPROX70
    # ========================================================

    (

        global_model,

        global_model_file,

        global_checkpoint,

    ) = load_fedprox70_model(

        scenario=

            args.scenario,

        seed=

            args.seed,

        device=

            device,

        input_dim=

            len(

                all_features

            ),

        num_classes=

            len(

                label_mapping

            ),

        active_features=

            all_features,

    )


    # ========================================================
    # EVALUATE FEDPROX70
    # ========================================================

    (

        global_y_true,

        global_y_pred,

        global_inference_seconds,

    ) = phase4.evaluate_global_model(

        model=

            global_model,

        scaler=

            scaler,

        feature_columns=

            feature_columns,

        active_indices=

            all_active_indices,

        label_mapping=

            label_mapping,

        device=

            device,

        batch_size=

            args.batch_size,

        chunk_size=

            args.chunk_size,

    )


    (

        global_overall,

        global_report,

        global_cm,

        class_names,

    ) = phase4.calculate_metrics(

        global_y_true,

        global_y_pred,

        label_mapping,

    )


    global_per_class_df = (

        build_per_class_df(

            global_report,

            class_names,

            seed=

                args.seed,

            model_type=

                "FedProx70",

        )

    )


    global_per_class_df.to_csv(

        result_dir

        / "fedprox70_per_class_metrics.csv",

        index=False,

    )


    # ========================================================
    # HEADER
    # ========================================================

    print()

    print(

        "=" * 100

    )


    print(

        "PHASE 11 — "

        "KNOWLEDGE TRANSFER + "

        "LOCAL KNOWLEDGE RETENTION"

    )


    print(

        "=" * 100

    )


    print(

        f"Scenario                 : "

        f"{args.scenario}"

    )


    print(

        f"Seed                     : "

        f"{args.seed}"

    )


    print(

        f"Device                   : "

        f"{device}"

    )


    print(

        f"Clients                  : "

        f"{N_CLIENTS}"

    )


    print(

        f"Features                 : "

        f"{len(all_features)}"

    )


    print(

        f"Pseudo rounds            : "

        f"{args.rounds}"

    )


    print(

        f"Local epochs/round       : "

        f"{args.local_epochs}"

    )


    print(

        f"Total local epochs/client: "

        f"{args.rounds * args.local_epochs}"

    )


    print(

        f"Benign label             : "

        f"{benign_label}"

    )


    print(

        f"FedProx70 Macro F1       : "

        f"{global_overall['macro_f1']:.6f}"

    )


    # ========================================================
    # COLLECTIONS
    # ========================================================

    distribution_rows = []

    local_per_class_frames = []

    local_history_frames = []

    retention_rows = []

    transfer_rows = []

    client_summary_rows = []


    # ========================================================
    # CLIENT LOOP
    # ========================================================

    for client_id in range(

        1,

        N_CLIENTS + 1,

    ):


        client_file = (

            scenario_dir

            / f"client_{client_id}.csv"

        )


        if not client_file.exists():

            raise FileNotFoundError(

                client_file

            )


        label_counts = (

            read_client_label_counts(

                client_file,

                args.chunk_size,

                label_mapping,

            )

        )


        seen_attacks = [

            attack

            for attack

            in attack_classes

            if

            label_counts.get(

                attack,

                0,

            )

            >

            0

        ]


        unseen_attacks = [

            attack

            for attack

            in attack_classes

            if

            label_counts.get(

                attack,

                0,

            )

            ==

            0

        ]


        print()

        print(

            "-" * 100

        )


        print(

            f"CLIENT {client_id}"

        )


        print(

            "-" * 100

        )


        print(

            f"Seen attacks   : "

            f"{seen_attacks}"

        )


        print(

            f"Unseen attacks : "

            f"{unseen_attacks}"

        )


        # ====================================================
        # SAVE TRAINING DISTRIBUTION
        # ====================================================

        for class_name in (

            label_mapping.keys()

        ):


            distribution_rows.append({

                "seed":

                    args.seed,

                "client_id":

                    client_id,

                "class":

                    class_name,

                "count":

                    int(

                        label_counts.get(

                            class_name,

                            0,

                        )

                    ),

                "is_benign":

                    bool(

                        class_name

                        ==

                        benign_label

                    ),

                "is_seen_attack":

                    bool(

                        class_name

                        in

                        seen_attacks

                    ),

                "is_unseen_attack":

                    bool(

                        class_name

                        in

                        unseen_attacks

                    ),

            })


        # ====================================================
        # TRAIN LOCAL-ONLY
        # ====================================================

        (

            local_model,

            local_history_df,

            local_training_seconds,

        ) = train_local_only(

            client_id=

                client_id,

            client_file=

                client_file,

            scaler=

                scaler,

            feature_columns=

                feature_columns,

            active_indices=

                all_active_indices,

            label_mapping=

                label_mapping,

            class_weights=

                global_weights,

            input_dim=

                len(

                    all_features

                ),

            device=

                device,

            rounds=

                args.rounds,

            local_epochs=

                args.local_epochs,

            batch_size=

                args.batch_size,

            chunk_size=

                args.chunk_size,

            learning_rate=

                args.learning_rate,

            weight_decay=

                args.weight_decay,

            seed=

                args.seed,

        )


        local_history_frames.append(

            local_history_df

        )


        # ====================================================
        # EVALUATE LOCAL MODEL
        # ====================================================

        (

            local_y_true,

            local_y_pred,

            local_inference_seconds,

        ) = phase4.evaluate_global_model(

            model=

                local_model,

            scaler=

                scaler,

            feature_columns=

                feature_columns,

            active_indices=

                all_active_indices,

            label_mapping=

                label_mapping,

            device=

                device,

            batch_size=

                args.batch_size,

            chunk_size=

                args.chunk_size,

        )


        (

            local_overall,

            local_report,

            local_cm,

            local_class_names,

        ) = phase4.calculate_metrics(

            local_y_true,

            local_y_pred,

            label_mapping,

        )


        if (

            list(

                local_class_names

            )

            !=

            list(

                class_names

            )

        ):

            raise ValueError(

                "Local/global class order mismatch."

            )


        local_per_class_frames.append(

            build_per_class_df(

                local_report,

                local_class_names,

                seed=

                    args.seed,

                model_type=

                    "LocalOnly",

                client_id=

                    client_id,

            )

        )


        # ====================================================
        # SAVE LOCAL MODEL
        # ====================================================

        torch.save(

            {

                "phase":

                    11,

                "experiment":

                    "local_only_retention_baseline",

                "seed":

                    args.seed,

                "scenario":

                    args.scenario,

                "client_id":

                    client_id,

                "model_state_dict":

                    local_model.state_dict(),

                "input_dim":

                    len(

                        all_features

                    ),

                "num_classes":

                    len(

                        label_mapping

                    ),

                "features":

                    all_features,

                "active_indices":

                    all_active_indices.tolist(),

                "label_mapping":

                    label_mapping,

                "seen_attacks":

                    seen_attacks,

                "unseen_attacks":

                    unseen_attacks,

                "rounds":

                    args.rounds,

                "local_epochs":

                    args.local_epochs,

                "training_time_seconds":

                    local_training_seconds,

                "inference_time_seconds":

                    local_inference_seconds,

                "overall_metrics":

                    local_overall,

            },

            artifact_dir

            / f"client_{client_id}_local_model.pt",

        )


        # ====================================================
        # RETENTION — SEEN ATTACKS
        # ====================================================

        client_retention_rows = []


        for attack in (

            seen_attacks

        ):


            local_recall = float(

                local_report[

                    attack

                ][

                    "recall"

                ]

            )


            global_recall = float(

                global_report[

                    attack

                ][

                    "recall"

                ]

            )


            retention_delta = (

                global_recall

                -

                local_recall

            )


            retention_ratio = (

                safe_ratio(

                    global_recall,

                    local_recall,

                )

            )


            row = {

                "seed":

                    args.seed,

                "client_id":

                    client_id,

                "attack_class":

                    attack,

                "local_only_recall":

                    local_recall,

                "fedprox70_recall":

                    global_recall,

                "retention_delta":

                    retention_delta,

                "retention_ratio":

                    retention_ratio,

                "local_only_f1":

                    float(

                        local_report[

                            attack

                        ][

                            "f1-score"

                        ]

                    ),

                "fedprox70_f1":

                    float(

                        global_report[

                            attack

                        ][

                            "f1-score"

                        ]

                    ),

                "local_training_samples":

                    int(

                        label_counts.get(

                            attack,

                            0,

                        )

                    ),

            }


            retention_rows.append(

                row

            )


            client_retention_rows.append(

                row

            )


        # ====================================================
        # TRANSFER — UNSEEN ATTACKS
        # ====================================================

        client_transfer_rows = []


        for attack in (

            unseen_attacks

        ):


            local_recall = float(

                local_report[

                    attack

                ][

                    "recall"

                ]

            )


            global_recall = float(

                global_report[

                    attack

                ][

                    "recall"

                ]

            )


            transfer_gain = (

                global_recall

                -

                local_recall

            )


            row = {

                "seed":

                    args.seed,

                "client_id":

                    client_id,

                "attack_class":

                    attack,

                "local_only_recall":

                    local_recall,

                "fedprox70_recall":

                    global_recall,

                "transfer_gain":

                    transfer_gain,

                "local_only_f1":

                    float(

                        local_report[

                            attack

                        ][

                            "f1-score"

                        ]

                    ),

                "fedprox70_f1":

                    float(

                        global_report[

                            attack

                        ][

                            "f1-score"

                        ]

                    ),

                "local_training_samples":

                    0,

            }


            transfer_rows.append(

                row

            )


            client_transfer_rows.append(

                row

            )


        # ====================================================
        # CLIENT SUMMARY
        # ====================================================

        retention_deltas = [

            row[

                "retention_delta"

            ]

            for row

            in client_retention_rows

        ]


        retention_ratios = [

            row[

                "retention_ratio"

            ]

            for row

            in client_retention_rows

            if

            np.isfinite(

                row[

                    "retention_ratio"

                ]

            )

        ]


        transfer_gains = [

            row[

                "transfer_gain"

            ]

            for row

            in client_transfer_rows

        ]


        client_summary_rows.append({

            "seed":

                args.seed,

            "client_id":

                client_id,

            "seen_attacks":

                " | ".join(

                    seen_attacks

                ),

            "unseen_attacks":

                " | ".join(

                    unseen_attacks

                ),

            "local_only_accuracy":

                float(

                    local_overall[

                        "accuracy"

                    ]

                ),

            "local_only_balanced_accuracy":

                float(

                    local_overall[

                        "balanced_accuracy"

                    ]

                ),

            "local_only_macro_f1":

                float(

                    local_overall[

                        "macro_f1"

                    ]

                ),

            "mean_retention_delta":

                (

                    float(

                        np.mean(

                            retention_deltas

                        )

                    )

                    if

                    retention_deltas

                    else

                    np.nan

                ),

            "mean_retention_ratio":

                (

                    float(

                        np.mean(

                            retention_ratios

                        )

                    )

                    if

                    retention_ratios

                    else

                    np.nan

                ),

            "mean_transfer_gain":

                (

                    float(

                        np.mean(

                            transfer_gains

                        )

                    )

                    if

                    transfer_gains

                    else

                    np.nan

                ),

            "local_training_seconds":

                local_training_seconds,

            "local_inference_seconds":

                local_inference_seconds,

        })


        print(

            f"Local Macro F1       : "

            f"{local_overall['macro_f1']:.6f}"

        )


        print(

            f"Mean retention delta : "

            f"{client_summary_rows[-1]['mean_retention_delta']:.6f}"

        )


        print(

            f"Mean transfer gain   : "

            f"{client_summary_rows[-1]['mean_transfer_gain']:.6f}"

        )


        del local_model


        if torch.cuda.is_available():

            torch.cuda.empty_cache()


    # ========================================================
    # DATAFRAMES
    # ========================================================

    distribution_df = pd.DataFrame(

        distribution_rows

    )


    local_per_class_df = pd.concat(

        local_per_class_frames,

        ignore_index=True,

    )


    retention_df = pd.DataFrame(

        retention_rows

    )


    transfer_df = pd.DataFrame(

        transfer_rows

    )


    client_summary_df = pd.DataFrame(

        client_summary_rows

    )


    history_df = pd.concat(

        local_history_frames,

        ignore_index=True,

    )


    # ========================================================
    # SAVE
    # ========================================================

    distribution_df.to_csv(

        result_dir

        / "client_training_distribution.csv",

        index=False,

    )


    local_per_class_df.to_csv(

        result_dir

        / "local_only_per_class_metrics.csv",

        index=False,

    )


    retention_df.to_csv(

        result_dir

        / "retention_all_clients.csv",

        index=False,

    )


    transfer_df.to_csv(

        result_dir

        / "transfer_all_clients.csv",

        index=False,

    )


    client_summary_df.to_csv(

        result_dir

        / "per_client_summary.csv",

        index=False,

    )


    history_df.to_csv(

        result_dir

        / "local_only_training_history.csv",

        index=False,

    )


    # ========================================================
    # SEED SUMMARY
    # ========================================================

    seed_summary = {

        "phase":

            11,

        "experiment":

            (

                "FedProx70 knowledge transfer "

                "+ local knowledge retention"

            ),

        "scenario":

            args.scenario,

        "seed":

            args.seed,

        "clients":

            N_CLIENTS,

        "features":

            70,

        "benign_label":

            benign_label,

        "attack_classes":

            attack_classes,

        "fedprox70_model":

            str(

                global_model_file

            ),

        "fedprox70_metrics":

            global_overall,

        "fedprox70_inference_seconds":

            global_inference_seconds,

        "training_protocol": {

            "pseudo_rounds":

                args.rounds,

            "local_epochs_per_round":

                args.local_epochs,

            "total_local_epochs":

                (

                    args.rounds

                    *

                    args.local_epochs

                ),

            "batch_size":

                args.batch_size,

            "learning_rate":

                args.learning_rate,

            "weight_decay":

                args.weight_decay,

            "optimizer":

                "AdamW recreated each pseudo-round",

            "class_weights":

                "same global class weights as Phase 9/10",

        },

        "definitions": {

            "retention_delta":

                (

                    "FedProx70 seen recall "

                    "- LocalOnly seen recall"

                ),

            "retention_ratio":

                (

                    "FedProx70 seen recall "

                    "/ LocalOnly seen recall"

                ),

            "transfer_gain":

                (

                    "FedProx70 unseen recall "

                    "- LocalOnly unseen recall"

                ),

        },

        "locked_test_used_during_training":

            False,

        "locked_test_used_for":

            "post-training evaluation only",

        "mean_retention_delta":

            (

                float(

                    retention_df[

                        "retention_delta"

                    ].mean()

                )

                if

                not retention_df.empty

                else

                None

            ),

        "mean_retention_ratio":

            (

                float(

                    retention_df[

                        "retention_ratio"

                    ]

                    .dropna()

                    .mean()

                )

                if

                (

                    not retention_df.empty

                    and

                    retention_df[

                        "retention_ratio"

                    ]

                    .notna()

                    .any()

                )

                else

                None

            ),

        "mean_transfer_gain":

            (

                float(

                    transfer_df[

                        "transfer_gain"

                    ].mean()

                )

                if

                not transfer_df.empty

                else

                None

            ),

    }


    save_json(

        result_dir

        / "phase11_summary.json",

        seed_summary,

    )


    # ========================================================
    # MULTI-SEED AGGREGATE
    # ========================================================

    aggregate_summary = (

        aggregate_completed_seeds(

            args.scenario

        )

    )


    # ========================================================
    # FINAL DISPLAY
    # ========================================================

    print()

    print(

        "=" * 100

    )


    print(

        "PHASE 11 SEED COMPLETE"

    )


    print(

        "=" * 100

    )


    print()

    print(

        "RETENTION"

    )


    if not retention_df.empty:


        print(

            retention_df[

                [

                    "client_id",

                    "attack_class",

                    "local_only_recall",

                    "fedprox70_recall",

                    "retention_delta",

                    "retention_ratio",

                ]

            ]

            .to_string(

                index=False

            )

        )


    print()

    print(

        "TRANSFER"

    )


    if not transfer_df.empty:


        print(

            transfer_df[

                [

                    "client_id",

                    "attack_class",

                    "local_only_recall",

                    "fedprox70_recall",

                    "transfer_gain",

                ]

            ]

            .to_string(

                index=False

            )

        )


    print()

    print(

        f"Results:\n"

        f"{result_dir}"

    )


    print()

    print(

        f"Local models:\n"

        f"{artifact_dir}"

    )


    if aggregate_summary is not None:


        print()

        print(

            "=" * 100

        )


        print(

            "CURRENT MULTI-SEED AGGREGATE"

        )


        print(

            "=" * 100

        )


        print(

            "Completed seeds      : "

            f"{aggregate_summary['completed_seeds']}"

        )


        print(

            "Mean retention delta : "

            f"{aggregate_summary['retention']['mean_retention_delta']:.6f}"

        )


        ratio = (

            aggregate_summary[

                "retention"

            ][

                "mean_retention_ratio"

            ]

        )


        print(

            "Mean retention ratio : "

            +

            (

                f"{ratio:.6f}"

                if

                ratio is not None

                else

                "N/A"

            )

        )


        print(

            "Mean transfer gain   : "

            f"{aggregate_summary['transfer']['mean_transfer_gain']:.6f}"

        )


    if not SCIPY_AVAILABLE:


        print()

        print(

            "NOTE: SciPy not installed."

        )


        print(

            "Install for Wilcoxon tests:"

        )


        print(

            "pip install scipy"

        )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()