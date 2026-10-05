from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.federated.poisoning.phase17e_analysis import (
    build_summary,
)


def main():
    rows = [
        {
            "attack_type": "NONE",
            "magnitude": 1.0,
            "malicious_clients": [],
            "malicious_ratio": 0.0,
            "specialized_attacks": [],
            "macro_f1": 0.80,
            "balanced_accuracy": 0.70,
            "weighted_f1": 0.90,
            "worst_attack_recall": 0.40,
            "target_attack_mean_recall": None,
            "target_attack_worst_recall": None,
            "macro_f1_drop_vs_clean": 0.0,
            "balanced_accuracy_drop_vs_clean": 0.0,
            "attack_audit_pass": True,
        },
        {
            "attack_type": "SCALE",
            "magnitude": 5.0,
            "malicious_clients": [1],
            "malicious_ratio": 0.20,
            "specialized_attacks": ["DoS Hulk"],
            "macro_f1": 0.50,
            "balanced_accuracy": 0.45,
            "weighted_f1": 0.70,
            "worst_attack_recall": 0.0,
            "target_attack_mean_recall": 0.9,
            "target_attack_worst_recall": 0.9,
            "macro_f1_drop_vs_clean": 0.30,
            "balanced_accuracy_drop_vs_clean": 0.25,
            "attack_audit_pass": True,
        },
        {
            "attack_type": "SCALE",
            "magnitude": 5.0,
            "malicious_clients": [1, 2],
            "malicious_ratio": 0.40,
            "specialized_attacks": ["DoS Hulk", "DDoS"],
            "macro_f1": 0.30,
            "balanced_accuracy": 0.30,
            "weighted_f1": 0.50,
            "worst_attack_recall": 0.0,
            "target_attack_mean_recall": 0.8,
            "target_attack_worst_recall": 0.7,
            "macro_f1_drop_vs_clean": 0.50,
            "balanced_accuracy_drop_vs_clean": 0.40,
            "attack_audit_pass": True,
        },
    ]

    summary = build_summary(
        rows=rows
    )

    assert len(
        summary[
            "ratio_rows"
        ]
    ) == 3

    assert len(
        summary[
            "identity_rows"
        ]
    ) == 1

    print("=" * 88)
    print("PHASE 17E ANALYSIS UNIT TEST")
    print("=" * 88)
    print(
        f"Identity rows : {len(summary['identity_rows'])}"
    )
    print(
        f"Ratio rows    : {len(summary['ratio_rows'])}"
    )
    print("STATUS        : PASS")


if __name__ == "__main__":
    main()
