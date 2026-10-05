"""
Phase 2 - Federated Client Partitioning
========================================

Creates:
1. IID clients
2. Attack-exclusive Non-IID clients

FINAL RESEARCH DESIGN

IID:
    All 5 clients receive all 6 classes with approximately
    identical class distributions.

NON-IID:
    Client 1 -> BENIGN + DoS Hulk
    Client 2 -> BENIGN + DDoS
    Client 3 -> BENIGN + PortScan
    Client 4 -> BENIGN + DoS GoldenEye
    Client 5 -> BENIGN + FTP-Patator

Input:
    data/processed/train.csv

Output:
    data/federated/iid/
    data/federated/non_iid/
    data/federated/reports/

Run:
    python scripts/phase2_prepare_clients.py
"""

from pathlib import Path
import json
import logging
import random
import shutil

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

SEED = 42
N_CLIENTS = 5

LABEL_COL = "Label"
BENIGN_LABEL = "BENIGN"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train.csv"
)

FEDERATED_DIR = (
    PROJECT_ROOT
    / "data"
    / "federated"
)

IID_DIR = (
    FEDERATED_DIR
    / "iid"
)

NON_IID_DIR = (
    FEDERATED_DIR
    / "non_iid"
)

REPORT_DIR = (
    FEDERATED_DIR
    / "reports"
)


# ============================================================
# FINAL CLASSES
# ============================================================

FINAL_CLASSES = [

    "BENIGN",
    "DoS Hulk",
    "DDoS",
    "PortScan",
    "DoS GoldenEye",
    "FTP-Patator",
]


# ============================================================
# FIXED NON-IID ASSIGNMENT
# ============================================================

NON_IID_ASSIGNMENT = {

    1: "DoS Hulk",

    2: "DDoS",

    3: "PortScan",

    4: "DoS GoldenEye",

    5: "FTP-Patator",
}


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("phase2")


# ============================================================
# SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# SAFE DATAFRAME SPLITTER
# ============================================================

def split_dataframe(
    df: pd.DataFrame,
    n_splits: int
):

    """
    Split DataFrame into approximately equal DataFrames.

    Avoid np.array_split because some NumPy/Pandas versions
    may return ndarray objects.
    """

    total = len(df)

    base = total // n_splits
    remainder = total % n_splits

    splits = []

    start = 0

    for i in range(n_splits):

        size = (
            base
            + (1 if i < remainder else 0)
        )

        end = start + size

        splits.append(
            df.iloc[start:end].copy()
        )

        start = end

    return splits


# ============================================================
# CLEAN OLD PHASE 2
# ============================================================

def prepare_directories():

    if FEDERATED_DIR.exists():

        logger.info(
            "Removing previous federated output: %s",
            FEDERATED_DIR
        )

        shutil.rmtree(
            FEDERATED_DIR
        )

    IID_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    NON_IID_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# LOAD TRAINING DATA
# ============================================================

def load_training_data():

    if not TRAIN_FILE.exists():

        raise FileNotFoundError(
            f"Training file missing: {TRAIN_FILE}"
        )

    logger.info(
        "Loading training data: %s",
        TRAIN_FILE
    )

    df = pd.read_csv(
        TRAIN_FILE
    )

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    if LABEL_COL not in df.columns:

        raise ValueError(
            f"Missing '{LABEL_COL}' column."
        )

    df[LABEL_COL] = (
        df[LABEL_COL]
        .astype(str)
        .str.strip()
    )

    detected = set(
        df[LABEL_COL].unique()
    )

    expected = set(
        FINAL_CLASSES
    )

    if detected != expected:

        raise RuntimeError(

            "Unexpected classes detected.\n"

            f"Expected:\n{sorted(expected)}\n\n"

            f"Found:\n{sorted(detected)}"
        )

    logger.info(
        "Training rows: %d",
        len(df)
    )

    logger.info(
        "Classes: %s",
        sorted(detected)
    )

    return df


# ============================================================
# IID PARTITION
# ============================================================

def create_iid_clients(df):

    """
    Every class is independently divided between all clients.

    Therefore each client receives approximately the same
    global class distribution.
    """

    logger.info(
        "Creating IID clients..."
    )

    client_parts = {

        i: []

        for i in range(
            1,
            N_CLIENTS + 1
        )
    }

    for class_index, label in enumerate(
        FINAL_CLASSES
    ):

        class_df = df[
            df[LABEL_COL] == label
        ].copy()

        class_df = class_df.sample(
            frac=1,
            random_state=
                SEED + class_index
        ).reset_index(drop=True)

        logger.info(
            "IID class %-20s | samples=%d",
            label,
            len(class_df)
        )

        splits = split_dataframe(
            class_df,
            N_CLIENTS
        )

        for client_id in range(
            1,
            N_CLIENTS + 1
        ):

            client_parts[
                client_id
            ].append(
                splits[
                    client_id - 1
                ]
            )

    clients = {}

    for client_id in range(
        1,
        N_CLIENTS + 1
    ):

        client_df = pd.concat(
            client_parts[
                client_id
            ],
            ignore_index=True
        )

        client_df = client_df.sample(
            frac=1,
            random_state=
                SEED + client_id
        ).reset_index(drop=True)

        clients[
            client_id
        ] = client_df

        logger.info(
            "IID Client %d | rows=%d",
            client_id,
            len(client_df)
        )

    return clients


# ============================================================
# NON-IID PARTITION
# ============================================================

def create_non_iid_clients(df):

    """
    Strict attack-exclusive Non-IID design.

    Each attack class belongs to exactly ONE client.

    BENIGN samples are distributed across all clients.
    """

    logger.info(
        "Creating attack-exclusive Non-IID clients..."
    )

    # --------------------------------------------------------
    # BENIGN
    # --------------------------------------------------------

    benign_df = df[
        df[LABEL_COL]
        == BENIGN_LABEL
    ].copy()

    benign_df = benign_df.sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)

    benign_splits = split_dataframe(
        benign_df,
        N_CLIENTS
    )

    clients = {}

    # --------------------------------------------------------
    # BUILD CLIENTS
    # --------------------------------------------------------

    for client_id in range(
        1,
        N_CLIENTS + 1
    ):

        attack_label = (
            NON_IID_ASSIGNMENT[
                client_id
            ]
        )

        attack_df = df[
            df[LABEL_COL]
            == attack_label
        ].copy()

        logger.info(
            "Client %d attack = %-20s | attack samples=%d",
            client_id,
            attack_label,
            len(attack_df)
        )

        client_df = pd.concat(
            [
                benign_splits[
                    client_id - 1
                ],

                attack_df
            ],
            ignore_index=True
        )

        client_df = client_df.sample(
            frac=1,
            random_state=
                SEED + client_id
        ).reset_index(drop=True)

        clients[
            client_id
        ] = client_df

        logger.info(
            "NON-IID Client %d | total=%d | benign=%d | attack=%d",
            client_id,
            len(client_df),
            len(
                benign_splits[
                    client_id - 1
                ]
            ),
            len(attack_df)
        )

    return clients


# ============================================================
# SAVE CLIENT FILES
# ============================================================

def save_clients(
    clients,
    directory,
    experiment
):

    for client_id, df in clients.items():

        file = (
            directory
            / f"client_{client_id}.csv"
        )

        df.to_csv(
            file,
            index=False
        )

        logger.info(
            "%s client %d saved -> %s",
            experiment,
            client_id,
            file
        )


# ============================================================
# PARTITION VALIDATION
# ============================================================

def validate_partition(
    source_df,
    clients,
    experiment
):

    source_rows = len(
        source_df
    )

    client_rows = sum(
        len(df)
        for df
        in clients.values()
    )

    logger.info(
        "%s validation | source=%d | clients=%d",
        experiment,
        source_rows,
        client_rows
    )

    if source_rows != client_rows:

        raise RuntimeError(

            f"{experiment} partition row mismatch.\n"

            f"Source = {source_rows}\n"

            f"Clients = {client_rows}"
        )

    logger.info(
        "%s partition validation PASSED ✅",
        experiment
    )


# ============================================================
# NON-IID ATTACK EXCLUSIVITY CHECK
# ============================================================

def verify_non_iid_exclusivity(
    clients
):

    logger.info(
        "Verifying Non-IID attack exclusivity..."
    )

    for client_id, df in clients.items():

        attack_labels = set(

            df.loc[
                df[LABEL_COL]
                != BENIGN_LABEL,

                LABEL_COL
            ].unique()
        )

        expected_attack = (
            NON_IID_ASSIGNMENT[
                client_id
            ]
        )

        expected = {
            expected_attack
        }

        if attack_labels != expected:

            raise RuntimeError(

                f"Client {client_id} attack "
                f"assignment incorrect.\n"

                f"Expected: {expected}\n"

                f"Found: {attack_labels}"
            )

        logger.info(
            "Client %d exclusivity verified -> %s ✅",
            client_id,
            expected_attack
        )


# ============================================================
# DISTRIBUTION REPORT
# ============================================================

def create_distribution_report(
    iid_clients,
    non_iid_clients
):

    rows = []

    experiments = {

        "IID":
            iid_clients,

        "NON_IID":
            non_iid_clients
    }

    for experiment, clients in (
        experiments.items()
    ):

        for client_id, df in (
            clients.items()
        ):

            counts = (
                df[LABEL_COL]
                .value_counts()
            )

            total = len(df)

            for label, count in (
                counts.items()
            ):

                rows.append({

                    "experiment":
                        experiment,

                    "client":
                        f"client_{client_id}",

                    "class":
                        label,

                    "samples":
                        int(count),

                    "percentage":
                        round(
                            count
                            / total
                            * 100,
                            4
                        )
                })

    report_df = pd.DataFrame(
        rows
    )

    report_file = (
        REPORT_DIR
        / "client_distributions.csv"
    )

    report_df.to_csv(
        report_file,
        index=False
    )

    return (
        report_df,
        report_file
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    clients,
    title
):

    print()
    print("=" * 85)
    print(title)
    print("=" * 85)

    for client_id, df in (
        clients.items()
    ):

        print()

        print(
            f"CLIENT {client_id}"
        )

        print(
            f"Total = {len(df):,}"
        )

        counts = (
            df[LABEL_COL]
            .value_counts()
        )

        for label, count in (
            counts.items()
        ):

            percentage = (
                count
                / len(df)
                * 100
            )

            print(
                f"{label:25s}"
                f"{count:12,d}"
                f"  {percentage:8.3f}%"
            )


# ============================================================
# MANIFEST
# ============================================================

def save_manifest(
    df,
    iid_clients,
    non_iid_clients
):

    manifest = {

        "phase":
            2,

        "description":
            "6-class federated IDS client partitioning",

        "training_file":
            str(TRAIN_FILE),

        "training_rows":
            len(df),

        "number_of_clients":
            N_CLIENTS,

        "number_of_classes":
            len(FINAL_CLASSES),

        "classes":
            FINAL_CLASSES,

        "seed":
            SEED,

        "iid": {

            "strategy":
                (
                    "Stratified IID partition preserving "
                    "global class proportions"
                ),

            "clients":
                N_CLIENTS
        },

        "non_iid": {

            "strategy":
                (
                    "Attack-exclusive label-skew "
                    "with BENIGN shared"
                ),

            "assignment": {

                f"client_{client_id}":
                    [
                        BENIGN_LABEL,
                        attack
                    ]

                for client_id, attack
                in NON_IID_ASSIGNMENT.items()
            }
        },

        "locked_test_set_used":
            False
    }

    manifest_file = (
        REPORT_DIR
        / "phase2_manifest.json"
    )

    with open(
        manifest_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            manifest,
            f,
            indent=4
        )

    return manifest_file


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 85)
    print("PHASE 2 - FEDERATED CLIENT GENERATION")
    print("=" * 85)

    prepare_directories()

    df = load_training_data()

    # --------------------------------------------------------
    # IID
    # --------------------------------------------------------

    iid_clients = (
        create_iid_clients(
            df
        )
    )

    validate_partition(
        df,
        iid_clients,
        "IID"
    )

    save_clients(
        iid_clients,
        IID_DIR,
        "IID"
    )

    # --------------------------------------------------------
    # NON-IID
    # --------------------------------------------------------

    non_iid_clients = (
        create_non_iid_clients(
            df
        )
    )

    validate_partition(
        df,
        non_iid_clients,
        "NON-IID"
    )

    verify_non_iid_exclusivity(
        non_iid_clients
    )

    save_clients(
        non_iid_clients,
        NON_IID_DIR,
        "NON-IID"
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report_df, report_file = (
        create_distribution_report(
            iid_clients,
            non_iid_clients
        )
    )

    manifest_file = (
        save_manifest(
            df,
            iid_clients,
            non_iid_clients
        )
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print_summary(
        iid_clients,
        "IID CLIENT DISTRIBUTION"
    )

    print_summary(
        non_iid_clients,
        "ATTACK-EXCLUSIVE NON-IID DISTRIBUTION"
    )

    print()
    print("=" * 85)
    print("NON-IID RESEARCH DESIGN")
    print("=" * 85)

    for client_id, attack in (
        NON_IID_ASSIGNMENT.items()
    ):

        print(
            f"Client {client_id} -> "
            f"BENIGN + {attack}"
        )

    print()
    print("=" * 85)
    print("PHASE 2 COMPLETE")
    print("=" * 85)

    print(
        f"IID clients     : {IID_DIR}"
    )

    print(
        f"Non-IID clients : {NON_IID_DIR}"
    )

    print(
        f"Report          : {report_file}"
    )

    print(
        f"Manifest        : {manifest_file}"
    )

    print()

    print(
        "Locked test used for training: NO ✅"
    )


if __name__ == "__main__":
    main()