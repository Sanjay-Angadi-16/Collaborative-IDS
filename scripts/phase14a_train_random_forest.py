from __future__ import annotations

import gc
import json
import logging
import random
import sys
import time

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


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
        "Keep this Phase 14 script inside the scripts folder."

    ) from exc


# ============================================================
# PHASE 14A CONFIGURATION
# ============================================================

PHASE_NAME = "PHASE 14A — LOCAL RANDOM FOREST TRAINING"

SCENARIO = "non_iid"

SEED = 42

N_CLIENTS = 5

EXPECTED_FEATURES = 70


# ============================================================
# LOCAL TRAIN / HOLDOUT SPLIT
# ============================================================

HOLDOUT_RATIO = 0.20


# ============================================================
# RANDOM FOREST PARAMETERS
#
# Hardcoded initial production configuration.
#
# IMPORTANT:
# These values are NOT tuned using locked test data.
# ============================================================

N_ESTIMATORS = 200

MAX_DEPTH = 24

MIN_SAMPLES_SPLIT = 2

MIN_SAMPLES_LEAF = 2

MAX_FEATURES = "sqrt"

CLASS_WEIGHT = "balanced_subsample"

N_JOBS = -1


# ============================================================
# OUTPUT PATHS
# ============================================================

ARTIFACT_ROOT = (
    PROJECT_ROOT
    /
    "artifacts"
    /
    "detection"
)

RESULT_ROOT = (
    PROJECT_ROOT
    /
    "results"
    /
    "phase14"
)

RF_METADATA_FILE = (
    ARTIFACT_ROOT
    /
    "random_forest_metadata.json"
)

RF_METRICS_FILE = (
    RESULT_ROOT
    /
    "random_forest_local_metrics.csv"
)

RF_IMPORTANCE_FILE = (
    RESULT_ROOT
    /
    "random_forest_feature_importance.csv"
)

RF_CONSENSUS_IMPORTANCE_FILE = (
    RESULT_ROOT
    /
    "random_forest_mean_feature_importance.csv"
)


# ============================================================
# CLIENT ATTACK MAPPING
# ============================================================

CLIENT_ATTACKS = {

    1:
        "DoS Hulk",

    2:
        "DDoS",

    3:
        "PortScan",

    4:
        "DoS GoldenEye",

    5:
        "FTP-Patator",
}


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
    "phase14a_random_forest"
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
        "=" * 100
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
    path: Path,
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
# FILTER PROJECT CLASSES
# ============================================================

def filter_to_project_classes(
    dataframe: pd.DataFrame,
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


    lower_labels = (
        labels
        .str.lower()
    )


    keep_mask = (

        lower_labels
        .isin(
            canonical.keys()
        )
    )


    filtered = (

        dataframe
        .loc[
            keep_mask
        ]
        .copy()
    )


    filtered[
        phase4.LABEL_COL
    ] = (

        filtered[
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


    return filtered


# ============================================================
# BUILD CLASS NAMES
# ============================================================

def build_class_names(
    label_mapping,
):

    inverse_mapping = {

        int(value):
            str(label)

        for label, value
        in label_mapping.items()
    }


    return [

        inverse_mapping[
            class_id
        ]

        for class_id
        in range(
            len(
                label_mapping
            )
        )
    ]


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

            "\nPhase 14A expects exactly "
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
# LOCAL METRICS
# ============================================================

def calculate_local_metrics(
    y_true,
    y_pred,
    client_id,
    label_mapping,
):

    attack_name = CLIENT_ATTACKS[
        client_id
    ]


    benign_id = int(
        label_mapping[
            "BENIGN"
        ]
    )


    attack_id = int(
        label_mapping[
            attack_name
        ]
    )


    accuracy = accuracy_score(
        y_true,
        y_pred,
    )


    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred,
    )


    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )


    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )


    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )


    benign_mask = (
        y_true
        ==
        benign_id
    )


    attack_mask = (
        y_true
        ==
        attack_id
    )


    benign_recall = (

        np.mean(
            y_pred[
                benign_mask
            ]
            ==
            benign_id
        )

        if np.any(
            benign_mask
        )

        else
        np.nan
    )


    attack_recall = (

        np.mean(
            y_pred[
                attack_mask
            ]
            ==
            attack_id
        )

        if np.any(
            attack_mask
        )

        else
        np.nan
    )


    return {

        "client_id":
            client_id,

        "seen_attack":
            attack_name,

        "accuracy":
            float(
                accuracy
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy
            ),

        "macro_precision":
            float(
                macro_precision
            ),

        "macro_recall":
            float(
                macro_recall
            ),

        "macro_f1":
            float(
                macro_f1
            ),

        "benign_recall":
            float(
                benign_recall
            ),

        "seen_attack_recall":
            float(
                attack_recall
            ),
    }


# ============================================================
# TRAIN ONE CLIENT RANDOM FOREST
# ============================================================

def train_client_random_forest(
    client_id,
    client_file,
    scaler,
    feature_columns,
    active_indices,
    active_features,
    label_mapping,
):

    separator()

    print(
        f"CLIENT {client_id} — RANDOM FOREST"
    )

    print(
        "-" * 100
    )


    attack_name = CLIENT_ATTACKS[
        client_id
    ]


    logger.info(
        "Loading client %d...",
        client_id,
    )


    dataframe = pd.read_csv(
        client_file
    )


    dataframe = filter_to_project_classes(
        dataframe,
        label_mapping,
    )


    print(
        f"Rows loaded       : "
        f"{len(dataframe):,}"
    )


    print(
        f"Expected attack   : "
        f"{attack_name}"
    )


    print()

    print(
        "Class distribution:"
    )

    print(

        dataframe[
            phase4.LABEL_COL
        ]
        .value_counts()
        .to_string()
    )


    # ========================================================
    # APPLY EXISTING PROJECT PREPROCESSING
    # ========================================================

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

            f"\nClient {client_id} generated "
            f"{X.shape[1]} features.\n"

            f"Expected {EXPECTED_FEATURES}."
        )


    del dataframe

    gc.collect()


    # ========================================================
    # STRATIFIED LOCAL TRAIN / HOLDOUT
    # ========================================================

    (
        X_train,
        X_holdout,
        y_train,
        y_holdout,

    ) = train_test_split(

        X,
        y,

        test_size=
            HOLDOUT_RATIO,

        random_state=
            SEED
            +
            client_id,

        stratify=
            y,

        shuffle=
            True,
    )


    print()

    print(
        f"Training rows     : "
        f"{len(y_train):,}"
    )


    print(
        f"Holdout rows      : "
        f"{len(y_holdout):,}"
    )


    print(
        f"Input features    : "
        f"{X_train.shape[1]}"
    )


    # ========================================================
    # CREATE RANDOM FOREST
    # ========================================================

    model = RandomForestClassifier(

        n_estimators=
            N_ESTIMATORS,

        max_depth=
            MAX_DEPTH,

        min_samples_split=
            MIN_SAMPLES_SPLIT,

        min_samples_leaf=
            MIN_SAMPLES_LEAF,

        max_features=
            MAX_FEATURES,

        class_weight=
            CLASS_WEIGHT,

        random_state=
            SEED
            +
            client_id,

        n_jobs=
            N_JOBS,

        bootstrap=
            True,

        oob_score=
            False,

        verbose=
            0,
    )


    # ========================================================
    # TRAIN
    # ========================================================

    print()

    print(
        "Training local Random Forest..."
    )


    training_start = time.perf_counter()


    model.fit(
        X_train,
        y_train,
    )


    training_seconds = (

        time.perf_counter()
        -
        training_start
    )


    print(
        f"Training time     : "
        f"{training_seconds:.2f}s"
    )


    # ========================================================
    # HOLDOUT INFERENCE
    # ========================================================

    inference_start = time.perf_counter()


    y_pred = model.predict(
        X_holdout
    )


    inference_seconds = (

        time.perf_counter()
        -
        inference_start
    )


    # ========================================================
    # METRICS
    # ========================================================

    metrics = calculate_local_metrics(

        y_true=
            y_holdout,

        y_pred=
            y_pred,

        client_id=
            client_id,

        label_mapping=
            label_mapping,
    )


    metrics[
        "training_rows"
    ] = int(
        len(
            y_train
        )
    )


    metrics[
        "holdout_rows"
    ] = int(
        len(
            y_holdout
        )
    )


    metrics[
        "training_seconds"
    ] = float(
        training_seconds
    )


    metrics[
        "inference_seconds"
    ] = float(
        inference_seconds
    )


    metrics[
        "trees"
    ] = N_ESTIMATORS


    metrics[
        "max_depth"
    ] = MAX_DEPTH


    # ========================================================
    # DISPLAY METRICS
    # ========================================================

    print()

    print(
        "LOCAL HOLDOUT RESULTS"
    )

    print(
        "-" * 100
    )


    print(
        f"Accuracy           : "
        f"{metrics['accuracy']:.6f}"
    )


    print(
        f"Balanced Accuracy  : "
        f"{metrics['balanced_accuracy']:.6f}"
    )


    print(
        f"Macro Precision    : "
        f"{metrics['macro_precision']:.6f}"
    )


    print(
        f"Macro Recall       : "
        f"{metrics['macro_recall']:.6f}"
    )


    print(
        f"Macro F1           : "
        f"{metrics['macro_f1']:.6f}"
    )


    print(
        f"BENIGN Recall      : "
        f"{metrics['benign_recall']:.6f}"
    )


    print(
        f"{attack_name} Recall"
        f"{' ' * max(1, 17 - len(attack_name))}: "
        f"{metrics['seen_attack_recall']:.6f}"
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    local_class_ids = sorted(

        set(
            y_holdout.tolist()
        )

        |
        set(
            y_pred.tolist()
        )
    )


    inverse_mapping = {

        int(value):
            str(label)

        for label, value
        in label_mapping.items()
    }


    local_class_names = [

        inverse_mapping[
            int(class_id)
        ]

        for class_id
        in local_class_ids
    ]


    matrix = confusion_matrix(

        y_holdout,
        y_pred,

        labels=
            local_class_ids,
    )


    confusion_dataframe = pd.DataFrame(

        matrix,

        index=
            [
                f"True_{name}"
                for name
                in local_class_names
            ],

        columns=
            [
                f"Pred_{name}"
                for name
                in local_class_names
            ],
    )


    confusion_file = (

        RESULT_ROOT
        /
        (
            f"random_forest_client_"
            f"{client_id}_confusion_matrix.csv"
        )
    )


    confusion_dataframe.to_csv(
        confusion_file
    )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(

        y_holdout,
        y_pred,

        labels=
            local_class_ids,

        target_names=
            local_class_names,

        zero_division=
            0,

        output_dict=
            True,
    )


    report_file = (

        RESULT_ROOT
        /
        (
            f"random_forest_client_"
            f"{client_id}_classification_report.json"
        )
    )


    save_json(
        report_file,
        report,
    )


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    importance_dataframe = pd.DataFrame(

        {

            "client_id":
                client_id,

            "feature":
                list(
                    active_features
                ),

            "importance":
                model.feature_importances_,
        }
    )


    importance_dataframe = (

        importance_dataframe
        .sort_values(

            "importance",

            ascending=
                False,
        )

        .reset_index(
            drop=True
        )
    )


    print()

    print(
        "TOP 10 RANDOM FOREST FEATURES"
    )

    print(
        "-" * 100
    )


    print(

        importance_dataframe[
            [
                "feature",
                "importance",
            ]
        ]

        .head(
            10
        )

        .to_string(
            index=False
        )
    )


    # ========================================================
    # SAVE DEPLOYMENT BUNDLE
    #
    # IMPORTANT:
    # We store model + metadata together.
    # Detection engine can later validate feature order.
    # ========================================================

    model_file = (

        ARTIFACT_ROOT
        /
        (
            f"random_forest_client_"
            f"{client_id}.joblib"
        )
    )


    bundle = {

        "model":
            model,

        "client_id":
            client_id,

        "scenario":
            SCENARIO,

        "seen_attack":
            attack_name,

        "feature_count":
            EXPECTED_FEATURES,

        "feature_names":
            list(
                active_features
            ),

        "label_mapping":
            dict(
                label_mapping
            ),

        "trained_class_ids":
            [
                int(value)
                for value
                in model.classes_
            ],

        "trained_class_names":
            [

                inverse_mapping[
                    int(value)
                ]

                for value
                in model.classes_
            ],

        "rf_parameters": {

            "n_estimators":
                N_ESTIMATORS,

            "max_depth":
                MAX_DEPTH,

            "min_samples_split":
                MIN_SAMPLES_SPLIT,

            "min_samples_leaf":
                MIN_SAMPLES_LEAF,

            "max_features":
                MAX_FEATURES,

            "class_weight":
                CLASS_WEIGHT,

            "seed":
                SEED
                +
                client_id,
        },

        "local_holdout_metrics":
            metrics,
    }


    joblib.dump(

        bundle,

        model_file,

        compress=
            3,
    )


    print()

    print(
        f"Saved model       : "
        f"{model_file}"
    )


    # ========================================================
    # CLEANUP
    # ========================================================

    del X

    del y

    del X_train

    del X_holdout

    del y_train

    del y_holdout

    del y_pred

    gc.collect()


    return (
        metrics,
        importance_dataframe,
        model_file,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(
        SEED
    )


    ARTIFACT_ROOT.mkdir(

        parents=
            True,

        exist_ok=
            True,
    )


    RESULT_ROOT.mkdir(

        parents=
            True,

        exist_ok=
            True,
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
        f"Clients                  : "
        f"{N_CLIENTS}"
    )


    print(
        f"Input features           : "
        f"{EXPECTED_FEATURES}"
    )


    print(
        f"Trees/client             : "
        f"{N_ESTIMATORS}"
    )


    print(
        f"Max depth                : "
        f"{MAX_DEPTH}"
    )


    print(
        f"Class weighting          : "
        f"{CLASS_WEIGHT}"
    )


    print(
        f"Holdout ratio            : "
        f"{HOLDOUT_RATIO * 100:.1f}%"
    )


    print(
        f"Random seed              : "
        f"{SEED}"
    )


    print()

    print(
        "PURPOSE"
    )

    print(
        "-" * 100
    )

    print(
        "Train one LOCAL Random Forest per federated client."
    )

    print(
        "Random Forest is NOT federated."
    )

    print(
        "No RF model parameters are averaged by FedAvg/FedProx."
    )

    print(
        "Each model becomes the fast first-line classifier "
        "inside the adaptive IDS engine."
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


    class_names = build_class_names(
        label_mapping
    )


    print()

    print(
        "CLASS MAPPING"
    )

    print(
        "-" * 100
    )


    for class_id, class_name in enumerate(
        class_names
    ):

        print(
            f"{class_id} -> "
            f"{class_name}"
        )


    # ========================================================
    # FEDERATED CLIENT DIRECTORY
    # ========================================================

    federated_dir = (

        phase4.FEDERATED_ROOT
        /
        SCENARIO
    )


    if not federated_dir.exists():

        raise FileNotFoundError(

            "\nFederated client directory not found:\n"

            f"{federated_dir}"
        )


    # ========================================================
    # TRAIN ALL FIVE CLIENT MODELS
    # ========================================================

    all_metrics = []

    all_importances = []

    model_files = []


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


        if not client_file.exists():

            raise FileNotFoundError(

                "\nMissing client file:\n"

                f"{client_file}"
            )


        (
            metrics,
            importance,
            model_file,

        ) = train_client_random_forest(

            client_id=
                client_id,

            client_file=
                client_file,

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
        )


        all_metrics.append(
            metrics
        )


        all_importances.append(
            importance
        )


        model_files.append(
            model_file
        )


    total_seconds = (

        time.perf_counter()
        -
        total_start
    )


    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics_dataframe = pd.DataFrame(
        all_metrics
    )


    metrics_dataframe.to_csv(

        RF_METRICS_FILE,

        index=
            False,
    )


    # ========================================================
    # SAVE ALL FEATURE IMPORTANCES
    # ========================================================

    importance_dataframe = pd.concat(

        all_importances,

        ignore_index=
            True,
    )


    importance_dataframe.to_csv(

        RF_IMPORTANCE_FILE,

        index=
            False,
    )


    # ========================================================
    # MEAN FEATURE IMPORTANCE ACROSS CLIENTS
    #
    # Reporting only.
    # Does NOT alter Phase 13 feature mask.
    # ========================================================

    mean_importance = (

        importance_dataframe

        .groupby(
            "feature",
            as_index=False,
        )

        [
            "importance"
        ]

        .mean()

        .rename(
            columns={
                "importance":
                    "mean_importance"
            }
        )

        .sort_values(

            "mean_importance",

            ascending=
                False,
        )

        .reset_index(
            drop=True
        )
    )


    mean_importance.to_csv(

        RF_CONSENSUS_IMPORTANCE_FILE,

        index=
            False,
    )


    # ========================================================
    # SAVE METADATA
    # ========================================================

    metadata = {

        "phase":
            "14A",

        "component":
            "Local Random Forest",

        "scenario":
            SCENARIO,

        "seed":
            SEED,

        "clients":
            N_CLIENTS,

        "feature_count":
            EXPECTED_FEATURES,

        "feature_names":
            list(
                active_features
            ),

        "label_mapping":
            dict(
                label_mapping
            ),

        "client_attacks":
            CLIENT_ATTACKS,

        "random_forest_parameters": {

            "n_estimators":
                N_ESTIMATORS,

            "max_depth":
                MAX_DEPTH,

            "min_samples_split":
                MIN_SAMPLES_SPLIT,

            "min_samples_leaf":
                MIN_SAMPLES_LEAF,

            "max_features":
                MAX_FEATURES,

            "class_weight":
                CLASS_WEIGHT,

            "n_jobs":
                N_JOBS,

            "holdout_ratio":
                HOLDOUT_RATIO,
        },

        "models":
            [

                str(
                    file
                )

                for file
                in model_files
            ],

        "metrics_file":
            str(
                RF_METRICS_FILE
            ),

        "feature_importance_file":
            str(
                RF_IMPORTANCE_FILE
            ),

        "mean_feature_importance_file":
            str(
                RF_CONSENSUS_IMPORTANCE_FILE
            ),

        "total_training_seconds":
            float(
                total_seconds
            ),

        "locked_test_used":
            False,

        "random_forest_federated":
            False,
    }


    save_json(

        RF_METADATA_FILE,

        metadata,
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    separator()

    print(
        "PHASE 14A COMPLETE"
    )

    separator()


    print(
        metrics_dataframe[
            [
                "client_id",
                "seen_attack",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "benign_recall",
                "seen_attack_recall",
                "training_seconds",
            ]
        ]

        .to_string(
            index=False
        )
    )


    print()

    print(
        "MEAN RESULTS"
    )

    print(
        "-" * 100
    )


    print(
        f"Mean Accuracy          : "
        f"{metrics_dataframe['accuracy'].mean():.6f}"
    )


    print(
        f"Mean Balanced Accuracy : "
        f"{metrics_dataframe['balanced_accuracy'].mean():.6f}"
    )


    print(
        f"Mean Macro F1          : "
        f"{metrics_dataframe['macro_f1'].mean():.6f}"
    )


    print(
        f"Mean Seen Attack Recall: "
        f"{metrics_dataframe['seen_attack_recall'].mean():.6f}"
    )


    print(
        f"Total RF training time : "
        f"{total_seconds:.2f}s"
    )


    print()

    print(
        "MODELS"
    )

    print(
        "-" * 100
    )


    for model_file in model_files:

        print(
            model_file
        )


    print()

    print(
        "METRICS:"
    )

    print(
        RF_METRICS_FILE
    )


    print()

    print(
        "FEATURE IMPORTANCE:"
    )

    print(
        RF_IMPORTANCE_FILE
    )


    print()

    print(
        "METADATA:"
    )

    print(
        RF_METADATA_FILE
    )


    print()

    print(
        "LOCKED TEST USED : NO"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()