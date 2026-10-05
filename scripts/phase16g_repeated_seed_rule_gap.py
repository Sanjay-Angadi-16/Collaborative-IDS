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

from src.correlation.phase16g_repeated_rule_gap import (  # noqa: E402
    PHASE16G_VERSION,
    SeedConfig,
    decide_dual_correlation_evidence,
    summarize_runs,
    train_one_seed,
)
from src.correlation.rule_temporal_correlator import CorrelationConfig  # noqa: E402


PHASE_NAME = (
    "PHASE 16G — REPEATED-SEED RULE-GAP CONFIRMATION + VALIDATION-ONLY HYBRID THRESHOLDING"
)

RESULT_ROOT = PROJECT_ROOT / "results" / "phase16"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "correlation"

POLICY_FILE = PROJECT_ROOT / "configs" / "phase16g_repeated_rule_gap_policy.json"
PHASE16C_FREEZE_FILE = (
    ARTIFACT_ROOT
    / "phase16c_rule_correlator_freeze_manifest.json"
)
PHASE16F_AUDIT_FILE = (
    RESULT_ROOT
    / "phase16f_audit.json"
)

RUNS_FILE = RESULT_ROOT / "phase16g_seed_runs.json"
SUMMARY_FILE = RESULT_ROOT / "phase16g_statistical_summary.json"
DECISION_FILE = RESULT_ROOT / "phase16g_research_architecture_decision.json"
AUDIT_FILE = RESULT_ROOT / "phase16g_audit.json"
MANIFEST_FILE = ARTIFACT_ROOT / "phase16g_manifest.json"


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
        return json.load(
            file
        )


def save_json(
    path: Path,
    payload: Any,
) -> None:
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

    phase16f = load_json(
        PHASE16F_AUDIT_FILE
    )

    if freeze.get(
        "status"
    ) != "FROZEN":
        raise RuntimeError(
            "Phase16C rule correlator is not frozen."
        )

    if not phase16f.get(
        "all_tests_passed",
        False,
    ):
        raise RuntimeError(
            "Phase16F did not pass."
        )

    if (
        freeze.get(
            "locked_test_used",
            True,
        )
        or
        phase16f.get(
            "locked_test_used",
            True,
        )
    ):
        raise RuntimeError(
            "Locked-test use detected. Phase16G prohibited."
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

    threshold_cfg = policy[
        "hybrid_threshold_selection"
    ]

    decision_cfg = policy[
        "research_decision"
    ]

    seeds = [
        int(
            seed
        )
        for seed
        in policy[
            "seeds"
        ]
    ]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

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
        f"Hybrid threshold candidates    : {threshold_cfg['candidates']}"
    )
    print(
        f"Hybrid min val precision       : {threshold_cfg['minimum_validation_precision']:.2f}"
    )
    print(
        f"Threshold selection data       : VALIDATION ONLY"
    )
    print(
        f"Locked IDS test                : NO"
    )

    print()
    print("SCIENTIFIC NOTE:")
    print(
        "Phase16G repeats the Phase16F independent rule-gap benchmark across multiple seeds."
    )
    print(
        "Each seed selects the hybrid rescue threshold using validation data only, freezes it, "
        "and then evaluates the held-out test set once."
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
        seed_config = SeedConfig(
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
            rule_positive_fraction_of_positives=float(
                dataset_cfg[
                    "rule_positive_fraction_of_positives"
                ]
            ),
            hybrid_threshold_candidates=tuple(
                float(
                    value
                )
                for value
                in threshold_cfg[
                    "candidates"
                ]
            ),
            hybrid_min_precision=float(
                threshold_cfg[
                    "minimum_validation_precision"
                ]
            ),
        )

        run_start = time.perf_counter()

        result = train_one_seed(
            config=seed_config,
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
            f"  best epoch                    : {result['best_epoch']}"
        )
        print(
            f"  selected hybrid threshold     : "
            f"{result['hybrid_threshold_selection']['selected_threshold']:.2f}"
        )
        print(
            f"  Rule Macro F1 / gap recall    : "
            f"{result['rule']['macro_f1']:.6f} / "
            f"{result['rule']['rule_gap_recall']:.6f}"
        )
        print(
            f"  GNN Macro F1 / gap recall     : "
            f"{result['gnn']['macro_f1']:.6f} / "
            f"{result['gnn']['rule_gap_recall']:.6f}"
        )
        print(
            f"  Hybrid Macro F1 / gap recall  : "
            f"{result['hybrid']['macro_f1']:.6f} / "
            f"{result['hybrid']['rule_gap_recall']:.6f}"
        )
        print(
            f"  Precision Rule/GNN/Hybrid     : "
            f"{result['rule']['precision']:.6f} / "
            f"{result['gnn']['precision']:.6f} / "
            f"{result['hybrid']['precision']:.6f}"
        )

    phase_runtime = (
        time.perf_counter()
        -
        phase_start
    )

    save_json(
        RUNS_FILE,
        {
            "phase": "16G",
            "version": PHASE16G_VERSION,
            "runs": runs,
            "phase_runtime_seconds": phase_runtime,
        },
    )

    summary = summarize_runs(
        runs
    )

    save_json(
        SUMMARY_FILE,
        summary,
    )

    decision = decide_dual_correlation_evidence(
        summary=summary,
        min_macro_f1_gain=float(
            decision_cfg[
                "minimum_macro_f1_gain_vs_rule"
            ]
        ),
        min_rule_gap_recall_gain=float(
            decision_cfg[
                "minimum_rule_gap_recall_gain_vs_rule"
            ]
        ),
        min_precision=float(
            decision_cfg[
                "minimum_gnn_precision"
            ]
        ),
        alpha=float(
            decision_cfg[
                "paired_significance_alpha"
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
        (
            "rule",
            "Frozen Rule",
        ),
        (
            "gnn",
            "GNN",
        ),
        (
            "hybrid",
            "Rule+GNN Hybrid",
        ),
    ):
        macro = summary[
            method
        ][
            "macro_f1"
        ]

        gap = summary[
            method
        ][
            "rule_gap_recall"
        ]

        precision = summary[
            method
        ][
            "precision"
        ]

        print(
            f"{label:<24} "
            f"Macro F1={macro['mean']:.6f} ± {macro['std']:.6f} | "
            f"Gap Recall={gap['mean']:.6f} ± {gap['std']:.6f} | "
            f"Precision={precision['mean']:.6f}"
        )

    print()
    print(
        f"Hybrid selected thresholds     : {summary['hybrid_thresholds']['values']}"
    )

    print()
    print("PAIRED EXACT SIGN TESTS")
    print("-" * 136)

    for name, value in summary[
        "paired_tests"
    ].items():
        print(
            f"{name:<40} "
            f"wins={value['wins']} "
            f"losses={value['losses']} "
            f"ties={value['ties']} "
            f"p={value['two_sided_exact_p']:.6f}"
        )

    print()
    print("PREDECLARED RESEARCH DECISION")
    print("-" * 136)

    print(
        f"Dual-correlation evidence      : "
        f"{'SUPPORTED' if decision['dual_correlation_evidence_supported'] else 'NOT SUPPORTED'}"
    )
    print(
        f"Recommended research architecture: "
        f"{decision['recommended_phase16_research_architecture']}"
    )
    print(
        f"GNN Macro F1 gain vs rule      : {decision['macro_f1_gain']:+.6f}"
    )
    print(
        f"GNN rule-gap recall gain       : {decision['rule_gap_recall_gain']:+.6f}"
    )
    print(
        f"Macro F1 paired p              : {decision['macro_f1_p_value']:.6f}"
    )
    print(
        f"Rule-gap recall paired p       : {decision['rule_gap_recall_p_value']:.6f}"
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
        all(
            not run[
                "hybrid_threshold_selection"
            ][
                "test_data_used_for_selection"
            ]
            for run
            in runs
        )
    )

    audit = {
        "phase": "16G",
        "version": PHASE16G_VERSION,
        "phase_name": PHASE_NAME,
        "seed_count": len(
            seeds
        ),
        "seeds": seeds,
        "frozen_rule_contract": frozen,
        "summary": summary,
        "research_decision": decision,
        "hybrid_threshold_selection": threshold_cfg,
        "test_data_used_for_hybrid_threshold_selection": False,
        "phase16c_rules_modified": False,
        "phase16f_test_results_used_for_tuning": False,
        "benchmark_is_synthetic": True,
        "real_world_superiority_claimed": False,
        "locked_test_used": False,
        "all_tests_passed": all_passed,
    }

    save_json(
        AUDIT_FILE,
        audit,
    )

    manifest = {
        "phase": "16G",
        "version": PHASE16G_VERSION,
        "status": (
            "PASS"
            if all_passed
            else
            "FAIL"
        ),
        "frozen_rule_contract": frozen,
        "research_architecture_decision": decision,
        "hybrid_threshold_policy": threshold_cfg,
        "important_limitations": [
            "The benchmark remains synthetic.",
            "Rule-gap campaigns are engineered multi-stage grammars, not production incident labels.",
            "The result supports modeled correlation capacity only within this benchmark family.",
            "The frozen Phase16C deterministic rule contract is unchanged.",
            "Hybrid thresholds are chosen using validation data only per seed.",
            "No locked CICIDS IDS test data are used.",
            "No real-world superiority claim is permitted from Phase16G alone.",
        ],
        "recommended_next_step": (
            "If dual-correlation evidence is supported, close Phase16 with the frozen rule branch "
            "plus a GNN rule-gap research branch, then move to the next exceptional-condition track: "
            "federated poisoning / malicious-client robustness."
        ),
        "locked_test_used": False,
    }

    save_json(
        MANIFEST_FILE,
        manifest,
    )

    print()
    print("=" * 136)
    print("PHASE 16G COMPLETE")
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
        f"Rule/GNN/Hybrid mean gap recall : "
        f"{summary['rule']['rule_gap_recall']['mean']:.6f} / "
        f"{summary['gnn']['rule_gap_recall']['mean']:.6f} / "
        f"{summary['hybrid']['rule_gap_recall']['mean']:.6f}"
    )
    print(
        f"Dual-correlation evidence        : "
        f"{'SUPPORTED' if decision['dual_correlation_evidence_supported'] else 'NOT SUPPORTED'}"
    )
    print(
        f"Research architecture            : "
        f"{decision['recommended_phase16_research_architecture']}"
    )
    print(
        f"Real-world superiority claim     : NO"
    )
    print(
        f"Locked test used                 : NO"
    )
    print(
        f"Runs                             : {RUNS_FILE}"
    )
    print(
        f"Summary                          : {SUMMARY_FILE}"
    )
    print(
        f"Decision                         : {DECISION_FILE}"
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
    print("NEXT:")
    print(
        "If evidence is supported, Phase16 can be closed scientifically. "
        "Then begin the next exceptional-condition track: federated poisoning / Byzantine-client robustness."
    )

    if not all_passed:
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()
