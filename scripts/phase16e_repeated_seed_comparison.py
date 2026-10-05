from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.correlation.phase16e_repeated_seed import (  # noqa: E402
    PHASE16E_VERSION,
    SeedRunConfig,
    select_operational_policy,
    summarize_runs,
    train_single_seed,
)
from src.correlation.rule_temporal_correlator import CorrelationConfig  # noqa: E402


PHASE_NAME = (
    "PHASE 16E — REPEATED-SEED RULE vs GNN vs RULE+GNN HYBRID COMPARISON"
)

RESULT_ROOT = PROJECT_ROOT / "results" / "phase16"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "correlation"

POLICY_FILE = PROJECT_ROOT / "configs" / "phase16e_repeated_seed_policy.json"
PHASE16C_FREEZE_FILE = (
    ARTIFACT_ROOT
    / "phase16c_rule_correlator_freeze_manifest.json"
)
PHASE16D_AUDIT_FILE = (
    RESULT_ROOT
    / "phase16d_audit.json"
)

RUNS_FILE = (
    RESULT_ROOT
    / "phase16e_seed_runs.json"
)
SUMMARY_FILE = (
    RESULT_ROOT
    / "phase16e_statistical_summary.json"
)
DECISION_FILE = (
    RESULT_ROOT
    / "phase16e_policy_decision.json"
)
AUDIT_FILE = (
    RESULT_ROOT
    / "phase16e_audit.json"
)
MANIFEST_FILE = (
    ARTIFACT_ROOT
    / "phase16e_correlation_policy_manifest.json"
)


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=4,
        )


def main():
    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARTIFACT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    policy = load_json(
        POLICY_FILE
    )

    freeze = load_json(
        PHASE16C_FREEZE_FILE
    )

    phase16d = load_json(
        PHASE16D_AUDIT_FILE
    )

    if freeze.get(
        "status"
    ) != "FROZEN":
        raise RuntimeError(
            "Phase16C rule correlator is not frozen."
        )

    if not phase16d.get(
        "all_tests_passed",
        False,
    ):
        raise RuntimeError(
            "Phase16D did not pass."
        )

    if (
        freeze.get(
            "locked_test_used",
            True,
        )
        or
        phase16d.get(
            "locked_test_used",
            True,
        )
    ):
        raise RuntimeError(
            "Locked IDS test usage detected. Phase16E prohibited."
        )

    frozen = freeze[
        "frozen_rule_correlator_contract"
    ]

    rule_config = CorrelationConfig(
        window_seconds=int(
            frozen[
                "window_seconds"
            ]
        ),
        min_cross_client_clients=int(
            frozen[
                "r1_min_clients"
            ]
        ),
        min_multi_class_clients=int(
            frozen[
                "r2_min_clients"
            ]
        ),
        min_multi_class_classes=int(
            frozen[
                "r2_min_classes"
            ]
        ),
        min_destination_clients=int(
            frozen[
                "r3_min_clients"
            ]
        ),
        min_destination_sources=int(
            frozen[
                "r3_min_sources"
            ]
        ),
    ).validate()

    dataset_cfg = policy[
        "dataset_per_seed"
    ]

    gnn_cfg = policy[
        "gnn"
    ]

    hybrid_cfg = policy[
        "hybrid"
    ]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    seeds = [
        int(
            value
        )
        for value
        in policy[
            "seeds"
        ]
    ]

    print()
    print("=" * 136)
    print(PHASE_NAME)
    print("=" * 136)

    print(
        f"Device                         : {device}"
    )
    print(
        f"Repeated seeds                 : {len(seeds)}"
    )
    print(
        f"Seeds                          : {seeds}"
    )
    print(
        f"Graphs/seed train-val-test     : "
        f"{dataset_cfg['train_graphs']} / "
        f"{dataset_cfg['validation_graphs']} / "
        f"{dataset_cfg['test_graphs']}"
    )
    print(
        f"Frozen rule window             : {rule_config.window_seconds} seconds"
    )
    print(
        f"Hybrid rescue threshold        : {hybrid_cfg['gnn_positive_threshold_when_rules_silent']:.2f}"
    )
    print(
        f"Locked IDS test                : NO"
    )

    print()
    print("SCIENTIFIC NOTE:")
    print(
        "This is repeated-seed statistical validation on synthetic privacy-safe "
        "Phase16 correlation graphs."
    )
    print(
        "The generator is structurally aligned with the deterministic correlation "
        "problem, so this phase must NOT be used as a real-world superiority claim."
    )

    runs: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    phase_start = time.perf_counter()

    for index, seed in enumerate(
        seeds,
        start=1,
    ):
        config = SeedRunConfig(
            seed=seed,
            train_graphs=int(
                dataset_cfg[
                    "train_graphs"
                ]
            ),
            validation_graphs=int(
                dataset_cfg[
                    "validation_graphs"
                ]
            ),
            test_graphs=int(
                dataset_cfg[
                    "test_graphs"
                ]
            ),
            epochs=int(
                gnn_cfg[
                    "epochs"
                ]
            ),
            batch_size=int(
                gnn_cfg[
                    "batch_size"
                ]
            ),
            hidden_dim=int(
                gnn_cfg[
                    "hidden_dim"
                ]
            ),
            dropout=float(
                gnn_cfg[
                    "dropout"
                ]
            ),
            learning_rate=float(
                gnn_cfg[
                    "learning_rate"
                ]
            ),
            weight_decay=float(
                gnn_cfg[
                    "weight_decay"
                ]
            ),
            patience=int(
                gnn_cfg[
                    "early_stopping_patience"
                ]
            ),
            hybrid_gnn_threshold=float(
                hybrid_cfg[
                    "gnn_positive_threshold_when_rules_silent"
                ]
            ),
        )

        run_start = time.perf_counter()

        result = train_single_seed(
            config=config,
            rule_config=rule_config,
            device=device,
        )

        result[
            "runtime_seconds"
        ] = (
            time.perf_counter()
            -
            run_start
        )

        runs.append(
            result
        )

        print()
        print(
            f"[{index:02d}/{len(seeds):02d}] seed={seed}"
        )
        print(
            f"  best epoch                   : {result['best_epoch']}"
        )
        print(
            f"  Rule Macro F1                : {result['rule']['macro_f1']:.6f}"
        )
        print(
            f"  GNN Macro F1                 : {result['gnn']['macro_f1']:.6f}"
        )
        print(
            f"  Hybrid Macro F1              : {result['hybrid']['macro_f1']:.6f}"
        )
        print(
            f"  Rule / GNN / Hybrid Precision: "
            f"{result['rule']['precision']:.6f} / "
            f"{result['gnn']['precision']:.6f} / "
            f"{result['hybrid']['precision']:.6f}"
        )

    phase_seconds = (
        time.perf_counter()
        -
        phase_start
    )

    save_json(
        RUNS_FILE,
        {
            "phase": "16E",
            "version": PHASE16E_VERSION,
            "runs": runs,
            "phase_runtime_seconds": phase_seconds,
        },
    )

    summary = summarize_runs(
        runs
    )

    save_json(
        SUMMARY_FILE,
        summary,
    )

    selection_cfg = policy[
        "selection"
    ]

    decision = select_operational_policy(
        summary=summary,
        minimum_material_gain=float(
            selection_cfg[
                "minimum_material_macro_f1_gain"
            ]
        ),
        significance_alpha=float(
            selection_cfg[
                "paired_significance_alpha"
            ]
        ),
        max_precision_drop=float(
            selection_cfg[
                "max_precision_drop"
            ]
        ),
    )

    save_json(
        DECISION_FILE,
        decision,
    )

    print()
    print("REPEATED-SEED SUMMARY")
    print("-" * 136)

    for method, label in (
        ("rule", "Frozen Rule"),
        ("gnn", "GNN"),
        ("hybrid", "Rule+GNN Hybrid"),
    ):
        macro = summary[
            method
        ][
            "macro_f1"
        ]

        precision = summary[
            method
        ][
            "precision"
        ]

        recall = summary[
            method
        ][
            "recall"
        ]

        print(
            f"{label:<26} Macro F1 = "
            f"{macro['mean']:.6f} ± {macro['std']:.6f} | "
            f"Precision = {precision['mean']:.6f} | "
            f"Recall = {recall['mean']:.6f}"
        )

    print()
    print("PAIRED EXACT SIGN TESTS")
    print("-" * 136)

    for name, result in summary[
        "paired_tests"
    ].items():
        print(
            f"{name:<34} "
            f"wins={result['wins']} "
            f"losses={result['losses']} "
            f"ties={result['ties']} "
            f"p={result['two_sided_exact_p']:.6f}"
        )

    print()
    print("PREDECLARED POLICY DECISION")
    print("-" * 136)

    print(
        f"Selected policy                : {decision['selected_policy']}"
    )
    print(
        f"Decision reason                : {decision['reason']}"
    )

    all_finite = all(
        0.0
        <=
        float(
            run[
                method
            ][
                "macro_f1"
            ]
        )
        <=
        1.0
        for run
        in runs
        for method
        in (
            "rule",
            "gnn",
            "hybrid",
        )
    )

    all_passed = (
        len(
            runs
        )
        ==
        len(
            seeds
        )
        and
        all_finite
    )

    audit = {
        "phase": "16E",
        "version": PHASE16E_VERSION,
        "phase_name": PHASE_NAME,
        "seed_count": len(
            seeds
        ),
        "seeds": seeds,
        "frozen_rule_contract": frozen,
        "summary": summary,
        "policy_decision": decision,
        "phase16c_rules_modified": False,
        "phase16d_architecture_modified": False,
        "hybrid_policy": hybrid_cfg,
        "benchmark_is_synthetic_rule_aligned": True,
        "real_world_superiority_claimed": False,
        "locked_test_used": False,
        "all_tests_passed": all_passed,
    }

    save_json(
        AUDIT_FILE,
        audit,
    )

    manifest = {
        "phase": "16E",
        "version": PHASE16E_VERSION,
        "status": (
            "PASS"
            if all_passed
            else
            "FAIL"
        ),
        "selected_phase16_benchmark_policy": decision[
            "selected_policy"
        ],
        "frozen_rule_contract": frozen,
        "hybrid_policy": hybrid_cfg,
        "selection_policy": selection_cfg,
        "important_limitations": [
            "The Phase16 synthetic graph generator contains rule-aligned campaign structures.",
            "A perfect or near-perfect rule result does not prove real-world superiority.",
            "A GNN result from these experiments does not prove real-world generalization.",
            "The selected policy is specific to the Phase16 synthetic benchmark.",
            "Fresh-alert_id semantic replay remains a documented limitation from Phase16C.",
            "No locked CICIDS IDS test data are used.",
        ],
        "recommended_next_step": (
            "If stronger GNN evidence is required, build an independent campaign-ground-truth "
            "benchmark containing correlations that are not defined directly by R1/R2/R3, "
            "then evaluate rule, GNN and hybrid without retuning the frozen rule baseline."
        ),
        "locked_test_used": False,
    }

    save_json(
        MANIFEST_FILE,
        manifest,
    )

    print()
    print("=" * 136)
    print("PHASE 16E COMPLETE")
    print("=" * 136)

    print(
        f"Seeds completed                 : {len(runs)}/{len(seeds)}"
    )
    print(
        f"Frozen Rule mean Macro F1       : "
        f"{summary['rule']['macro_f1']['mean']:.6f} ± "
        f"{summary['rule']['macro_f1']['std']:.6f}"
    )
    print(
        f"GNN mean Macro F1               : "
        f"{summary['gnn']['macro_f1']['mean']:.6f} ± "
        f"{summary['gnn']['macro_f1']['std']:.6f}"
    )
    print(
        f"Hybrid mean Macro F1            : "
        f"{summary['hybrid']['macro_f1']['mean']:.6f} ± "
        f"{summary['hybrid']['macro_f1']['std']:.6f}"
    )
    print(
        f"Selected Phase16 policy          : {decision['selected_policy']}"
    )
    print(
        f"Real-world superiority claim     : NO"
    )
    print(
        f"Locked test used                 : NO"
    )
    print(
        f"Seed runs                        : {RUNS_FILE}"
    )
    print(
        f"Statistical summary              : {SUMMARY_FILE}"
    )
    print(
        f"Policy decision                  : {DECISION_FILE}"
    )
    print(
        f"Audit                            : {AUDIT_FILE}"
    )
    print(
        f"Manifest                         : {MANIFEST_FILE}"
    )

    print()
    print(
        f"STATUS                           : {'PASS' if all_passed else 'FAIL'}"
    )

    print()
    print("IMPORTANT:")
    print(
        "The selected Phase16 policy is benchmark-specific. Because this benchmark is "
        "synthetic and rule-aligned, do not describe it as proof that rules or GNNs are "
        "universally superior in real hybrid-cloud traffic."
    )

    print()
    print("NEXT:")
    print(
        "Review the repeated-seed evidence. If the rule baseline remains perfect, the "
        "scientifically stronger next phase is an independent rule-gap campaign benchmark "
        "before making any production-level GNN superiority claim."
    )

    if not all_passed:
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()
