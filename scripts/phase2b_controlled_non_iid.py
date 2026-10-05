"""
Phase 2B - Controlled Attack-Exclusive Non-IID
===============================================

Creates five equal-sized federated clients.

Each client:
    80% BENIGN
    20% one unique attack class

Client 1 -> BENIGN + DoS Hulk
Client 2 -> BENIGN + DDoS
Client 3 -> BENIGN + PortScan
Client 4 -> BENIGN + DoS GoldenEye
Client 5 -> BENIGN + FTP-Patator

Attack samples/client = 4,168
Benign samples/client = 16,672
Total/client          = 20,840

No oversampling.
No locked-test data used.

Run:
    python scripts/phase2b_controlled_non_iid.py
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

BENIGN_RATIO = 0.80
ATTACK_RATIO = 0.20


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "federated"
    / "non_iid"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "federated"
    / "controlled_non_iid"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "data"
    / "federated"
    / "reports"
)


# ============================================================
# FIXED CLIENT-ATTACK ASSIGNMENT
# ============================================================

CLIENT_ATTACKS = {

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

logger = logging.getLogger("phase2b")


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# PREPARE OUTPUT
# ============================================================

def prepare_output():

    if OUTPUT_DIR.exists():

        logger.info(
            "Removing previous controlled output: %s",
            OUTPUT_DIR
        )

        shutil.rmtree(
            OUTPUT_DIR
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# LOAD NON-IID CLIENTS
# ============================================================

def load_clients():

    clients = {}

    for client_id in range(
        1,
        N_CLIENTS + 1
    ):

        file_path = (
            INPUT_DIR
            / f"client_{client_id}.csv"
        )

        if not file_path.exists():

            raise FileNotFoundError(
                f"Missing Phase-2 client file: {file_path}"
            )

        logger.info(
            "Loading Client %d: %s",
            client_id,
            file_path
        )

        df = pd.read_csv(
            file_path
        )

        df.columns = [
            str(c).strip()
            for c in df.columns
        ]

        df[LABEL_COL] = (
            df[LABEL_COL]
            .astype(str)
            .str.strip()
        )

        clients[
            client_id
        ] = df

    return clients


# ============================================================
# VERIFY PHASE-2 ASSIGNMENT
# ============================================================

def verify_client_attacks(
    clients
):

    logger.info(
        "Verifying attack-exclusive Phase-2 clients..."
    )

    for client_id, df in clients.items():

        detected_attacks = set(

            df.loc[
                df[LABEL_COL] != BENIGN_LABEL,
                LABEL_COL
            ].unique()
        )

        expected_attack = (
            CLIENT_ATTACKS[
                client_id
            ]
        )

        if detected_attacks != {
            expected_attack
        }:

            raise RuntimeError(

                f"Client {client_id} invalid.\n"

                f"Expected attack: "
                f"{expected_attack}\n"

                f"Detected: "
                f"{detected_attacks}"
            )

        logger.info(
            "Client %d verified -> %s ✅",
            client_id,
            expected_attack
        )


# ============================================================
# CALCULATE CONTROLLED BUDGET
# ============================================================

def calculate_budget(
    clients
):

    attack_counts = {}

    print()
    print("=" * 80)
    print("ORIGINAL NON-IID ATTACK COUNTS")
    print("=" * 80)

    for client_id, df in clients.items():

        attack_count = int(

            (
                df[LABEL_COL]
                != BENIGN_LABEL
            ).sum()
        )

        attack_counts[
            client_id
        ] = attack_count

        print(
            f"Client {client_id} "
            f"{CLIENT_ATTACKS[client_id]:20s} "
            f"{attack_count:10,d}"
        )

    # Smallest real attack pool
    attack_budget = min(
        attack_counts.values()
    )

    benign_budget = round(

        attack_budget
        * BENIGN_RATIO
        / ATTACK_RATIO
    )

    return (
        attack_budget,
        benign_budget
    )


# ============================================================
# CREATE CONTROLLED CLIENTS
# ============================================================

def create_controlled_clients(
    clients,
    attack_budget,
    benign_budget
):

    controlled = {}

    for client_id, df in clients.items():

        attack_name = (
            CLIENT_ATTACKS[
                client_id
            ]
        )

        # --------------------------------------------
        # BENIGN
        # --------------------------------------------

        benign_df = df[
            df[LABEL_COL]
            == BENIGN_LABEL
        ].copy()

        # --------------------------------------------
        # ATTACK
        # --------------------------------------------

        attack_df = df[
            df[LABEL_COL]
            == attack_name
        ].copy()

        if len(benign_df) < benign_budget:

            raise RuntimeError(
                f"Client {client_id} has insufficient BENIGN rows."
            )

        if len(attack_df) < attack_budget:

            raise RuntimeError(
                f"Client {client_id} has insufficient attack rows."
            )

        # --------------------------------------------
        # SAMPLE WITHOUT REPLACEMENT
        # --------------------------------------------

        benign_sample = benign_df.sample(

            n=benign_budget,

            replace=False,

            random_state=
                SEED
                + client_id
        )

        attack_sample = attack_df.sample(

            n=attack_budget,

            replace=False,

            random_state=
                SEED
                + 100
                + client_id
        )

        # --------------------------------------------
        # COMBINE
        # --------------------------------------------

        result = pd.concat(
            [
                benign_sample,
                attack_sample
            ],
            ignore_index=True
        )

        result = result.sample(

            frac=1,

            random_state=
                SEED
                + 200
                + client_id

        ).reset_index(
            drop=True
        )

        controlled[
            client_id
        ] = result

        logger.info(
            "Controlled Client %d | "
            "attack=%s | "
            "total=%d | "
            "benign=%d | "
            "attack_rows=%d",
            client_id,
            attack_name,
            len(result),
            len(benign_sample),
            len(attack_sample)
        )

    return controlled


# ============================================================
# VERIFY CONTROLLED DATA
# ============================================================

def verify_controlled(
    clients,
    attack_budget,
    benign_budget
):

    expected_total = (
        attack_budget
        + benign_budget
    )

    for client_id, df in clients.items():

        attack_name = (
            CLIENT_ATTACKS[
                client_id
            ]
        )

        benign_count = int(

            (
                df[LABEL_COL]
                == BENIGN_LABEL
            ).sum()
        )

        attack_count = int(

            (
                df[LABEL_COL]
                == attack_name
            ).sum()
        )

        labels = set(
            df[LABEL_COL].unique()
        )

        expected_labels = {
            BENIGN_LABEL,
            attack_name
        }

        if len(df) != expected_total:

            raise RuntimeError(
                f"Client {client_id} total mismatch."
            )

        if benign_count != benign_budget:

            raise RuntimeError(
                f"Client {client_id} BENIGN mismatch."
            )

        if attack_count != attack_budget:

            raise RuntimeError(
                f"Client {client_id} attack mismatch."
            )

        if labels != expected_labels:

            raise RuntimeError(

                f"Client {client_id} unexpected labels: "
                f"{labels}"
            )

        logger.info(
            "Client %d controlled validation PASSED ✅",
            client_id
        )


# ============================================================
# SAVE CLIENTS
# ============================================================

def save_clients(
    clients
):

    for client_id, df in clients.items():

        output_file = (
            OUTPUT_DIR
            / f"client_{client_id}.csv"
        )

        df.to_csv(
            output_file,
            index=False
        )

        logger.info(
            "Saved Client %d -> %s",
            client_id,
            output_file
        )


# ============================================================
# REPORT
# ============================================================

def create_report(
    clients
):

    rows = []

    for client_id, df in clients.items():

        total = len(df)

        counts = (
            df[LABEL_COL]
            .value_counts()
        )

        for label, count in counts.items():

            rows.append({

                "experiment":
                    "CONTROLLED_NON_IID",

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
        / "controlled_non_iid_distributions.csv"
    )

    report_df.to_csv(
        report_file,
        index=False
    )

    return report_file


# ============================================================
# MANIFEST
# ============================================================

def save_manifest(
    attack_budget,
    benign_budget
):

    manifest = {

        "phase":
            "2B",

        "experiment":
            "Controlled Attack-Exclusive Non-IID",

        "clients":
            N_CLIENTS,

        "seed":
            SEED,

        "benign_ratio":
            BENIGN_RATIO,

        "attack_ratio":
            ATTACK_RATIO,

        "benign_samples_per_client":
            benign_budget,

        "attack_samples_per_client":
            attack_budget,

        "samples_per_client":
            benign_budget
            + attack_budget,

        "total_training_rows":
            (
                benign_budget
                + attack_budget
            )
            * N_CLIENTS,

        "client_assignment": {

            f"client_{client_id}": {

                "benign":
                    BENIGN_LABEL,

                "attack":
                    attack_name
            }

            for client_id, attack_name
            in CLIENT_ATTACKS.items()
        },

        "sampling":
            "Downsampling without replacement",

        "oversampling":
            False,

        "synthetic_data":
            False,

        "locked_test_used":
            False
    }

    manifest_file = (
        REPORT_DIR
        / "phase2b_manifest.json"
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
# PRINT SUMMARY
# ============================================================

def print_summary(
    clients
):

    print()
    print("=" * 85)
    print("CONTROLLED NON-IID CLIENT DISTRIBUTION")
    print("=" * 85)

    for client_id, df in clients.items():

        print()

        print(
            f"CLIENT {client_id}"
        )

        print(
            f"Attack = "
            f"{CLIENT_ATTACKS[client_id]}"
        )

        print(
            f"Total  = "
            f"{len(df):,}"
        )

        counts = (
            df[LABEL_COL]
            .value_counts()
        )

        for label, count in counts.items():

            percentage = (
                count
                / len(df)
                * 100
            )

            print(
                f"{label:25s}"
                f"{count:10,d}"
                f"  {percentage:8.3f}%"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 85)
    print("PHASE 2B - CONTROLLED NON-IID")
    print("=" * 85)

    prepare_output()

    clients = (
        load_clients()
    )

    verify_client_attacks(
        clients
    )

    (
        attack_budget,
        benign_budget

    ) = calculate_budget(
        clients
    )

    logger.info(
        "Attack budget/client = %d",
        attack_budget
    )

    logger.info(
        "Benign budget/client = %d",
        benign_budget
    )

    logger.info(
        "Total/client = %d",
        attack_budget
        + benign_budget
    )

    controlled = (
        create_controlled_clients(
            clients,
            attack_budget,
            benign_budget
        )
    )

    verify_controlled(
        controlled,
        attack_budget,
        benign_budget
    )

    save_clients(
        controlled
    )

    report_file = (
        create_report(
            controlled
        )
    )

    manifest_file = (
        save_manifest(
            attack_budget,
            benign_budget
        )
    )

    print_summary(
        controlled
    )

    print()
    print("=" * 85)
    print("PHASE 2B COMPLETE")
    print("=" * 85)

    print(
        f"Attack/client : "
        f"{attack_budget:,}"
    )

    print(
        f"Benign/client : "
        f"{benign_budget:,}"
    )

    print(
        f"Total/client  : "
        f"{attack_budget + benign_budget:,}"
    )

    print(
        f"All clients   : "
        f"{(attack_budget + benign_budget) * N_CLIENTS:,}"
    )

    print()

    print(
        f"Output:\n{OUTPUT_DIR}"
    )

    print()

    print(
        f"Report:\n{report_file}"
    )

    print()

    print(
        f"Manifest:\n{manifest_file}"
    )

    print()

    print(
        "Oversampling used: NO ✅"
    )

    print(
        "Synthetic data used: NO ✅"
    )

    print(
        "Locked test used: NO ✅"
    )


if __name__ == "__main__":
    main()