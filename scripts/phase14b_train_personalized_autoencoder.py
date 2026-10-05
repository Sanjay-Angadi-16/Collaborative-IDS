from __future__ import annotations



import copy

import gc

import json

import logging

import random

import sys

import time



from pathlib import Path



import numpy as np

import pandas as pd



import torch

import torch.nn as nn



from sklearn.metrics import (

    accuracy_score,

    average_precision_score,

    balanced_accuracy_score,

    confusion_matrix,

    f1_score,

    precision_score,

    recall_score,

    roc_auc_score,

)



from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, TensorDataset





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

        "scripts\\\phase9_fedavg70.py\n\n"

        "Keep this script inside the scripts folder."

    ) from exc





# ============================================================

# PHASE CONFIGURATION

# ============================================================



PHASE_NAME = (

    "PHASE 14B — PERSONALIZED AUTOENCODER TRAINING"

)



SCENARIO = "non_iid"



SEED = 42



N_CLIENTS = 5



EXPECTED_FEATURES = 70





# ============================================================

# AUTOENCODER CONFIGURATION

# ============================================================



HIDDEN_DIM_1 = 48



HIDDEN_DIM_2 = 24



LATENT_DIM = 12



EPOCHS = 20



BATCH_SIZE = 2048



INFERENCE_BATCH_SIZE = 4096



LEARNING_RATE = 0.001



WEIGHT_DECAY = 0.00001





# ============================================================

# BENIGN TRAIN / VALIDATION

# ============================================================



BENIGN_VALIDATION_RATIO = 0.20





# ============================================================

# EARLY STOPPING

# ============================================================



EARLY_STOPPING_PATIENCE = 4



MIN_DELTA = 1e-7





# ============================================================

# ANOMALY THRESHOLD

#

# IMPORTANT:

#

# Threshold is calibrated ONLY from benign validation

# reconstruction errors.

#

# Locked test is NOT used.

# Attack labels are NOT used to choose the threshold.

# ============================================================



THRESHOLD_PERCENTILE = 99.0





# ============================================================

# RISK-FUSION NORMALIZATION

#

# Reconstruction error equal to the anomaly threshold

# becomes anomaly score = 0.70.

#

# This matches the current DetectionEngine threshold.

# ============================================================



RISK_SCORE_AT_THRESHOLD = 0.70





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

# OUTPUT DIRECTORIES

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





AUTOENCODER_THRESHOLDS_FILE = (

    ARTIFACT_ROOT

    /

    "autoencoder_thresholds.json"

)





AUTOENCODER_METADATA_FILE = (

    ARTIFACT_ROOT

    /

    "autoencoder_metadata.json"

)





AUTOENCODER_METRICS_FILE = (

    RESULT_ROOT

    /

    "autoencoder_local_metrics.csv"

)





AUTOENCODER_HISTORY_FILE = (

    RESULT_ROOT

    /

    "autoencoder_training_history.csv"

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

    "phase14b_personalized_autoencoder"

)





# ============================================================

# RANDOM SEED

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



    torch.manual_seed(

        seed

    )



    if torch.cuda.is_available():



        torch.cuda.manual_seed_all(

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

    path,

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

    dataframe,

    label_mapping,

):



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





    keep_mask = (



        labels

        .str.lower()



        .isin(

            canonical.keys()

        )

    )





    dataframe = (



        dataframe

        .loc[

            keep_mask

        ]

        .copy()

    )





    dataframe[

        phase4.LABEL_COL

    ] = (



        dataframe[

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





    return dataframe





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

            "\nExpected exactly "

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

            "\nFeature ordering mismatch."

        )





    return active_indices





# ============================================================

# AUTOENCODER MODEL

# ============================================================



class PersonalizedAutoencoder(

    nn.Module

):



    def __init__(

        self,

        input_dim=70,

    ):



        super().__init__()





        # ====================================================

        # ENCODER

        # ====================================================



        self.encoder = nn.Sequential(



            nn.Linear(

                input_dim,

                HIDDEN_DIM_1,

            ),



            nn.ReLU(),





            nn.Linear(

                HIDDEN_DIM_1,

                HIDDEN_DIM_2,

            ),



            nn.ReLU(),





            nn.Linear(

                HIDDEN_DIM_2,

                LATENT_DIM,

            ),



            nn.ReLU(),

        )





        # ====================================================

        # DECODER

        # ====================================================



        self.decoder = nn.Sequential(



            nn.Linear(

                LATENT_DIM,

                HIDDEN_DIM_2,

            ),



            nn.ReLU(),





            nn.Linear(

                HIDDEN_DIM_2,

                HIDDEN_DIM_1,

            ),



            nn.ReLU(),





            nn.Linear(

                HIDDEN_DIM_1,

                input_dim,

            ),

        )





    def forward(

        self,

        x,

    ):



        encoded = self.encoder(

            x

        )



        reconstructed = self.decoder(

            encoded

        )



        return reconstructed





# ============================================================

# CREATE DATA LOADER

# ============================================================



def create_loader(

    X,

    batch_size,

    shuffle,

    seed,

):



    tensor = torch.from_numpy(



        X.astype(

            np.float32,

            copy=False,

        )

    )





    dataset = TensorDataset(

        tensor

    )





    generator = torch.Generator()



    generator.manual_seed(

        seed

    )





    return DataLoader(



        dataset,



        batch_size=

            batch_size,



        shuffle=

            shuffle,



        num_workers=

            0,



        pin_memory=

            torch.cuda.is_available(),



        generator=

            generator

            if shuffle

            else None,

    )





# ============================================================

# CALCULATE DATASET LOSS

# ============================================================



@torch.no_grad()

def evaluate_mse(

    model,

    X,

    device,

):



    loader = create_loader(



        X=

            X,



        batch_size=

            INFERENCE_BATCH_SIZE,



        shuffle=

            False,



        seed=

            SEED,

    )





    model.eval()





    total_loss = 0.0



    total_rows = 0





    for (

        batch,



    ) in loader:



        batch = batch.to(

            device

        )





        reconstructed = model(

            batch

        )





        errors = (



            (

                reconstructed

                -

                batch

            )



            ** 2

        )





        row_mse = errors.mean(

            dim=1

        )





        total_loss += float(

            row_mse.sum().item()

        )





        total_rows += len(

            batch

        )





    return (



        total_loss

        /

        max(

            total_rows,

            1,

        )

    )





# ============================================================

# RECONSTRUCTION ERRORS

# ============================================================



@torch.no_grad()

def reconstruction_errors(

    model,

    X,

    device,

):



    loader = create_loader(



        X=

            X,



        batch_size=

            INFERENCE_BATCH_SIZE,



        shuffle=

            False,



        seed=

            SEED,

    )





    model.eval()





    all_errors = []





    for (

        batch,



    ) in loader:



        batch = batch.to(

            device

        )





        reconstructed = model(

            batch

        )





        row_errors = (



            (

                reconstructed

                -

                batch

            )



            ** 2

        ).mean(

            dim=1

        )





        all_errors.append(



            row_errors

            .detach()

            .cpu()

            .numpy()

        )





    return np.concatenate(

        all_errors

    )





# ============================================================

# NORMALIZED ANOMALY SCORE

# ============================================================



def normalize_anomaly_scores(

    reconstruction_error,

    threshold,

):



    safe_threshold = max(

        float(

            threshold

        ),

        1e-12,

    )





    scores = (



        reconstruction_error

        /

        safe_threshold

    ) * RISK_SCORE_AT_THRESHOLD





    return np.clip(



        scores,



        0.0,



        1.0,

    )





# ============================================================

# TRAIN ONE AUTOENCODER

# ============================================================



def train_client_autoencoder(



    client_id,



    client_file,



    scaler,



    feature_columns,



    active_indices,



    active_features,



    label_mapping,



    device,

):



    separator()



    print(

        f"CLIENT {client_id} — PERSONALIZED AUTOENCODER"

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





    # ========================================================

    # BENIGN DATA

    # ========================================================



    benign_dataframe = (



        dataframe[

            dataframe[

                phase4.LABEL_COL

            ]

            ==

            "BENIGN"

        ]



        .copy()

    )





    # ========================================================

    # CLIENT'S LOCAL ATTACK

    # ========================================================



    attack_dataframe = (



        dataframe[

            dataframe[

                phase4.LABEL_COL

            ]

            ==

            attack_name

        ]



        .copy()

    )





    print(

        f"Total client rows       : "

        f"{len(dataframe):,}"

    )





    print(

        f"BENIGN rows             : "

        f"{len(benign_dataframe):,}"

    )





    print(

        f"{attack_name} rows"

        f"{' ' * max(1, 18 - len(attack_name))}: "

        f"{len(attack_dataframe):,}"

    )





    if len(

        benign_dataframe

    ) == 0:



        raise ValueError(

            f"Client {client_id} has no BENIGN rows."

        )





    if len(

        attack_dataframe

    ) == 0:



        raise ValueError(

            f"Client {client_id} has no {attack_name} rows."

        )





    # ========================================================

    # TRANSFORM BENIGN DATA

    # ========================================================



    X_benign = phase4.transform_features(



        df=

            benign_dataframe,



        feature_columns=

            feature_columns,



        scaler=

            scaler,



        active_indices=

            active_indices,

    )





    # ========================================================

    # TRANSFORM ATTACK DATA

    # ========================================================



    X_attack = phase4.transform_features(



        df=

            attack_dataframe,



        feature_columns=

            feature_columns,



        scaler=

            scaler,



        active_indices=

            active_indices,

    )





    X_benign = X_benign.astype(

        np.float32,

        copy=False,

    )





    X_attack = X_attack.astype(

        np.float32,

        copy=False,

    )





    if X_benign.shape[

        1

    ] != EXPECTED_FEATURES:



        raise ValueError(

            "\nUnexpected benign feature count."

        )





    if X_attack.shape[

        1

    ] != EXPECTED_FEATURES:



        raise ValueError(

            "\nUnexpected attack feature count."

        )





    del dataframe



    del benign_dataframe



    del attack_dataframe



    gc.collect()





    # ========================================================

    # BENIGN TRAIN / VALIDATION SPLIT

    #

    # Autoencoder trains ONLY on BENIGN.

    # ========================================================



    (

        X_benign_train,

        X_benign_validation,



    ) = train_test_split(



        X_benign,



        test_size=

            BENIGN_VALIDATION_RATIO,



        random_state=

            SEED

            +

            client_id,



        shuffle=

            True,

    )





    print()



    print(

        f"Benign AE training rows : "

        f"{len(X_benign_train):,}"

    )





    print(

        f"Benign validation rows  : "

        f"{len(X_benign_validation):,}"

    )





    print(

        f"Attack evaluation rows  : "

        f"{len(X_attack):,}"

    )





    # ========================================================
    # AUTOENCODER INPUT DIAGNOSTIC CHECK
    #
    # These values are checked AFTER the project's existing
    # frozen preprocessing and BEFORE Autoencoder training.
    # This helps detect extreme post-scaling values that can
    # make MSE reconstruction loss numerically huge.
    # ========================================================

    print("\n========== AUTOENCODER INPUT CHECK ==========")
    print(f"Client : {client_id}")
    print("Dataset: BENIGN TRAIN (actual AE training input)")
    print("Shape  :", X_benign_train.shape)
    print("Min    :", np.min(X_benign_train))
    print("Max    :", np.max(X_benign_train))
    print("Mean   :", np.mean(X_benign_train))
    print("Std    :", np.std(X_benign_train))
    print("P95    :", np.percentile(X_benign_train, 95))
    print("P99    :", np.percentile(X_benign_train, 99))
    print("P99.9  :", np.percentile(X_benign_train, 99.9))
    print("P99.99 :", np.percentile(X_benign_train, 99.99))
    print("NaN    :", np.isnan(X_benign_train).sum())
    print("Inf    :", np.isinf(X_benign_train).sum())
    print("==============================================")


    # ========================================================

    # MODEL

    # ========================================================



    model_seed = (

        SEED

        +

        client_id

        *

        100

    )





    set_seed(

        model_seed

    )





    model = PersonalizedAutoencoder(



        input_dim=

            EXPECTED_FEATURES



    ).to(

        device

    )





    criterion = nn.MSELoss()





    optimizer = torch.optim.Adam(



        model.parameters(),



        lr=

            LEARNING_RATE,



        weight_decay=

            WEIGHT_DECAY,

    )





    train_loader = create_loader(



        X=

            X_benign_train,



        batch_size=

            BATCH_SIZE,



        shuffle=

            True,



        seed=

            model_seed,

    )





    # ========================================================

    # TRAINING

    # ========================================================



    best_validation_loss = float(

        "inf"

    )





    best_state = None





    epochs_without_improvement = 0





    training_history = []





    training_start = time.perf_counter()





    for epoch in range(

        1,

        EPOCHS + 1,

    ):



        model.train()





        total_train_loss = 0.0



        total_train_rows = 0





        for (

            batch,



        ) in train_loader:



            batch = batch.to(

                device

            )





            optimizer.zero_grad(

                set_to_none=True

            )





            reconstructed = model(

                batch

            )





            loss = criterion(



                reconstructed,



                batch,

            )





            loss.backward()





            optimizer.step()





            batch_rows = len(

                batch

            )





            total_train_loss += (



                float(

                    loss.item()

                )



                *

                batch_rows

            )





            total_train_rows += batch_rows





        train_loss = (



            total_train_loss

            /

            total_train_rows

        )





        validation_loss = evaluate_mse(



            model=

                model,



            X=

                X_benign_validation,



            device=

                device,

        )





        training_history.append(



            {



                "client_id":

                    client_id,



                "epoch":

                    epoch,



                "train_mse":

                    train_loss,



                "validation_mse":

                    validation_loss,

            }

        )





        print(



            f"Epoch "

            f"{epoch:02d}/{EPOCHS} | "



            f"Train MSE="

            f"{train_loss:.8f} | "



            f"Validation MSE="

            f"{validation_loss:.8f}"

        )





        # ====================================================

        # EARLY STOPPING

        # ====================================================



        if (



            validation_loss



            <

            best_validation_loss

            -

            MIN_DELTA



        ):



            best_validation_loss = (

                validation_loss

            )





            best_state = copy.deepcopy(



                model.state_dict()

            )





            epochs_without_improvement = 0





        else:



            epochs_without_improvement += 1





            if (



                epochs_without_improvement



                >=



                EARLY_STOPPING_PATIENCE



            ):



                print(



                    "Early stopping triggered."

                )



                break





    training_seconds = (



        time.perf_counter()

        -

        training_start

    )





    # ========================================================

    # RESTORE BEST MODEL

    # ========================================================



    if best_state is None:



        raise RuntimeError(

            "\nNo valid Autoencoder state was produced."

        )





    model.load_state_dict(

        best_state

    )





    # ========================================================

    # RECONSTRUCTION ERRORS

    # ========================================================



    inference_start = time.perf_counter()





    benign_errors = reconstruction_errors(



        model=

            model,



        X=

            X_benign_validation,



        device=

            device,

    )





    attack_errors = reconstruction_errors(



        model=

            model,



        X=

            X_attack,



        device=

            device,

    )





    inference_seconds = (



        time.perf_counter()

        -

        inference_start

    )





    # ========================================================

    # CALIBRATE THRESHOLD USING BENIGN VALIDATION ONLY

    # ========================================================



    threshold = float(



        np.percentile(



            benign_errors,



            THRESHOLD_PERCENTILE,

        )

    )





    if threshold <= 0:



        raise RuntimeError(

            "\nAutoencoder threshold is non-positive."

        )





    # ========================================================

    # BINARY ANOMALY EVALUATION

    #

    # 0 = BENIGN

    # 1 = ATTACK / ANOMALY

    # ========================================================



    y_true = np.concatenate(



        [



            np.zeros(

                len(

                    benign_errors

                ),

                dtype=np.int64,

            ),



            np.ones(

                len(

                    attack_errors

                ),

                dtype=np.int64,

            ),

        ]

    )





    all_errors = np.concatenate(



        [

            benign_errors,

            attack_errors,

        ]

    )





    y_pred = (



        all_errors



        >

        threshold



    ).astype(

        np.int64

    )





    # ========================================================

    # NORMALIZED ANOMALY SCORES FOR FUTURE RISK FUSION

    # ========================================================



    normalized_scores = normalize_anomaly_scores(



        reconstruction_error=

            all_errors,



        threshold=

            threshold,

    )





    # ========================================================

    # METRICS

    # ========================================================



    accuracy = accuracy_score(



        y_true,

        y_pred,

    )





    balanced_accuracy = balanced_accuracy_score(



        y_true,

        y_pred,

    )





    precision = precision_score(



        y_true,

        y_pred,



        zero_division=

            0,

    )





    recall = recall_score(



        y_true,

        y_pred,



        zero_division=

            0,

    )





    f1 = f1_score(



        y_true,

        y_pred,



        zero_division=

            0,

    )





    roc_auc = roc_auc_score(



        y_true,

        all_errors,

    )





    pr_auc = average_precision_score(



        y_true,

        all_errors,

    )





    benign_false_positive_rate = float(



        np.mean(



            benign_errors



            >

            threshold

        )

    )





    attack_detection_rate = float(



        np.mean(



            attack_errors



            >

            threshold

        )

    )





    metrics = {



        "client_id":

            client_id,



        "seen_attack":

            attack_name,



        "benign_train_rows":

            len(

                X_benign_train

            ),



        "benign_validation_rows":

            len(

                X_benign_validation

            ),



        "attack_evaluation_rows":

            len(

                X_attack

            ),



        "threshold_percentile":

            THRESHOLD_PERCENTILE,



        "reconstruction_threshold":

            threshold,



        "best_validation_mse":

            best_validation_loss,



        "mean_benign_error":

            float(

                benign_errors.mean()

            ),



        "median_benign_error":

            float(

                np.median(

                    benign_errors

                )

            ),



        "mean_attack_error":

            float(

                attack_errors.mean()

            ),



        "median_attack_error":

            float(

                np.median(

                    attack_errors

                )

            ),



        "accuracy":

            float(

                accuracy

            ),



        "balanced_accuracy":

            float(

                balanced_accuracy

            ),



        "precision":

            float(

                precision

            ),



        "recall":

            float(

                recall

            ),



        "f1":

            float(

                f1

            ),



        "roc_auc":

            float(

                roc_auc

            ),



        "pr_auc":

            float(

                pr_auc

            ),



        "benign_false_positive_rate":

            benign_false_positive_rate,



        "attack_detection_rate":

            attack_detection_rate,



        "mean_normalized_anomaly_score":

            float(

                normalized_scores.mean()

            ),



        "training_seconds":

            float(

                training_seconds

            ),



        "inference_seconds":

            float(

                inference_seconds

            ),

    }





    # ========================================================

    # DISPLAY

    # ========================================================



    print()



    print(

        "PERSONALIZED AUTOENCODER RESULTS"

    )



    print(

        "-" * 100

    )





    print(

        f"Threshold              : "

        f"{threshold:.8f}"

    )





    print(

        f"Mean benign error      : "

        f"{benign_errors.mean():.8f}"

    )





    print(

        f"Mean attack error      : "

        f"{attack_errors.mean():.8f}"

    )





    print(

        f"Balanced Accuracy      : "

        f"{balanced_accuracy:.6f}"

    )





    print(

        f"Anomaly Precision      : "

        f"{precision:.6f}"

    )





    print(

        f"Attack Detection Recall: "

        f"{recall:.6f}"

    )





    print(

        f"F1                     : "

        f"{f1:.6f}"

    )





    print(

        f"ROC-AUC                : "

        f"{roc_auc:.6f}"

    )





    print(

        f"PR-AUC                 : "

        f"{pr_auc:.6f}"

    )





    print(

        f"Benign FPR             : "

        f"{benign_false_positive_rate:.6f}"

    )





    print(

        f"Training time          : "

        f"{training_seconds:.2f}s"

    )





    # ========================================================

    # CONFUSION MATRIX

    # ========================================================



    matrix = confusion_matrix(



        y_true,

        y_pred,



        labels=

            [

                0,

                1,

            ],

    )





    matrix_dataframe = pd.DataFrame(



        matrix,



        index=

            [

                "True_BENIGN",

                "True_ANOMALY",

            ],



        columns=

            [

                "Pred_BENIGN",

                "Pred_ANOMALY",

            ],

    )





    confusion_file = (



        RESULT_ROOT

        /

        (

            f"autoencoder_client_"

            f"{client_id}_confusion_matrix.csv"

        )

    )





    matrix_dataframe.to_csv(

        confusion_file

    )





    # ========================================================

    # SAVE AUTOENCODER MODEL

    # ========================================================



    model_file = (



        ARTIFACT_ROOT

        /

        (

            f"autoencoder_client_"

            f"{client_id}.pt"

        )

    )





    checkpoint = {



        "phase":

            "14B",



        "client_id":

            client_id,



        "seen_attack":

            attack_name,



        "input_dim":

            EXPECTED_FEATURES,



        "hidden_dim_1":

            HIDDEN_DIM_1,



        "hidden_dim_2":

            HIDDEN_DIM_2,



        "latent_dim":

            LATENT_DIM,



        "feature_names":

            list(

                active_features

            ),



        "label_mapping":

            dict(

                label_mapping

            ),



        "state_dict":

            best_state,



        "reconstruction_threshold":

            threshold,



        "threshold_percentile":

            THRESHOLD_PERCENTILE,



        "risk_score_at_threshold":

            RISK_SCORE_AT_THRESHOLD,



        "metrics":

            metrics,

    }





    torch.save(



        checkpoint,



        model_file,

    )





    print()



    print(

        f"Saved model             : "

        f"{model_file}"

    )





    # ========================================================

    # CLEANUP

    # ========================================================



    del model



    del best_state



    del X_benign



    del X_attack



    del X_benign_train



    del X_benign_validation



    gc.collect()





    if torch.cuda.is_available():



        torch.cuda.empty_cache()





    return (



        metrics,



        training_history,



        model_file,



        threshold,

    )





# ============================================================

# MAIN

# ============================================================



def main():



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





    device = torch.device(



        "cuda"



        if torch.cuda.is_available()



        else



        "cpu"

    )





    set_seed(

        SEED

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

        f"Device                   : "

        f"{device}"

    )





    print(

        f"Input features           : "

        f"{EXPECTED_FEATURES}"

    )





    print(

        f"Encoder                  : "

        f"70 -> {HIDDEN_DIM_1} -> "

        f"{HIDDEN_DIM_2} -> {LATENT_DIM}"

    )





    print(

        f"Decoder                  : "

        f"{LATENT_DIM} -> {HIDDEN_DIM_2} -> "

        f"{HIDDEN_DIM_1} -> 70"

    )





    print(

        f"Epochs                   : "

        f"{EPOCHS}"

    )





    print(

        f"Batch size               : "

        f"{BATCH_SIZE}"

    )





    print(

        f"Threshold percentile     : "

        f"{THRESHOLD_PERCENTILE}%"

    )





    print()



    print(

        "IMPORTANT TRAINING POLICY"

    )



    print(

        "-" * 100

    )





    print(

        "Each Autoencoder trains ONLY on its client's BENIGN traffic."

    )





    print(

        "Attack rows are used only for internal anomaly evaluation."

    )





    print(

        "Threshold is calibrated ONLY from held-out BENIGN reconstruction errors."

    )





    print(

        "Locked test is NOT used."

    )





    # ========================================================

    # LOAD EXISTING PREPROCESSING

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



            "\nFederated directory missing:\n"



            f"{federated_dir}"

        )





    # ========================================================

    # TRAIN ALL CLIENT AUTOENCODERS

    # ========================================================



    all_metrics = []



    all_history = []



    thresholds = {}



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

            history,

            model_file,

            threshold,



        ) = train_client_autoencoder(



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



            device=

                device,

        )





        all_metrics.append(

            metrics

        )





        all_history.extend(

            history

        )





        model_files.append(

            model_file

        )





        thresholds[

            str(

                client_id

            )

        ] = {



            "client_id":

                client_id,



            "seen_attack":

                CLIENT_ATTACKS[

                    client_id

                ],



            "reconstruction_threshold":

                threshold,



            "threshold_percentile":

                THRESHOLD_PERCENTILE,



            "risk_score_at_threshold":

                RISK_SCORE_AT_THRESHOLD,

        }





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



        AUTOENCODER_METRICS_FILE,



        index=

            False,

    )





    # ========================================================

    # SAVE TRAINING HISTORY

    # ========================================================



    history_dataframe = pd.DataFrame(

        all_history

    )





    history_dataframe.to_csv(



        AUTOENCODER_HISTORY_FILE,



        index=

            False,

    )





    # ========================================================

    # SAVE THRESHOLDS

    # ========================================================



    save_json(



        AUTOENCODER_THRESHOLDS_FILE,



        {



            "phase":

                "14B",



            "scenario":

                SCENARIO,



            "threshold_source":

                "benign_validation_only",



            "threshold_percentile":

                THRESHOLD_PERCENTILE,



            "risk_score_at_threshold":

                RISK_SCORE_AT_THRESHOLD,



            "clients":

                thresholds,



            "locked_test_used":

                False,

        },

    )





    # ========================================================

    # METADATA

    # ========================================================



    metadata = {



        "phase":

            "14B",



        "component":

            "Personalized Autoencoder",



        "scenario":

            SCENARIO,



        "seed":

            SEED,



        "clients":

            N_CLIENTS,



        "device":

            str(

                device

            ),



        "feature_count":

            EXPECTED_FEATURES,



        "feature_names":

            list(

                active_features

            ),



        "architecture": {



            "encoder":

                [

                    EXPECTED_FEATURES,

                    HIDDEN_DIM_1,

                    HIDDEN_DIM_2,

                    LATENT_DIM,

                ],



            "decoder":

                [

                    LATENT_DIM,

                    HIDDEN_DIM_2,

                    HIDDEN_DIM_1,

                    EXPECTED_FEATURES,

                ],

        },



        "training": {



            "epochs":

                EPOCHS,



            "batch_size":

                BATCH_SIZE,



            "learning_rate":

                LEARNING_RATE,



            "weight_decay":

                WEIGHT_DECAY,



            "benign_validation_ratio":

                BENIGN_VALIDATION_RATIO,



            "early_stopping_patience":

                EARLY_STOPPING_PATIENCE,

        },



        "threshold": {



            "percentile":

                THRESHOLD_PERCENTILE,



            "source":

                "held_out_benign_only",



            "risk_score_at_threshold":

                RISK_SCORE_AT_THRESHOLD,

        },



        "model_files":

            [

                str(

                    file

                )

                for file

                in model_files

            ],



        "metrics_file":

            str(

                AUTOENCODER_METRICS_FILE

            ),



        "history_file":

            str(

                AUTOENCODER_HISTORY_FILE

            ),



        "threshold_file":

            str(

                AUTOENCODER_THRESHOLDS_FILE

            ),



        "total_training_seconds":

            total_seconds,



        "locked_test_used":

            False,

    }





    save_json(



        AUTOENCODER_METADATA_FILE,



        metadata,

    )





    # ========================================================

    # FINAL SUMMARY

    # ========================================================



    separator()



    print(

        "PHASE 14B COMPLETE"

    )



    separator()





    print(



        metrics_dataframe[

            [

                "client_id",

                "seen_attack",

                "reconstruction_threshold",

                "balanced_accuracy",

                "precision",

                "recall",

                "f1",

                "roc_auc",

                "pr_auc",

                "benign_false_positive_rate",

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

        f"Mean Balanced Accuracy : "

        f"{metrics_dataframe['balanced_accuracy'].mean():.6f}"

    )





    print(

        f"Mean Attack Recall     : "

        f"{metrics_dataframe['recall'].mean():.6f}"

    )





    print(

        f"Mean F1                : "

        f"{metrics_dataframe['f1'].mean():.6f}"

    )





    print(

        f"Mean ROC-AUC           : "

        f"{metrics_dataframe['roc_auc'].mean():.6f}"

    )





    print(

        f"Mean PR-AUC            : "

        f"{metrics_dataframe['pr_auc'].mean():.6f}"

    )





    print(

        f"Mean Benign FPR        : "

        f"{metrics_dataframe['benign_false_positive_rate'].mean():.6f}"

    )





    print(

        f"Total AE training time : "

        f"{total_seconds:.2f}s"

    )





    print()



    print(

        "AUTOENCODER MODELS:"

    )





    for model_file in model_files:



        print(

            model_file

        )





    print()



    print(

        "THRESHOLDS:"

    )



    print(

        AUTOENCODER_THRESHOLDS_FILE

    )





    print()



    print(

        "METRICS:"

    )



    print(

        AUTOENCODER_METRICS_FILE

    )





    print()



    print(

        "TRAINING HISTORY:"

    )



    print(

        AUTOENCODER_HISTORY_FILE

    )





    print()



    print(

        "LOCKED TEST USED : NO"

    )





# ============================================================

# ENTRY

# ============================================================



if __name__ == "__main__":



    main()