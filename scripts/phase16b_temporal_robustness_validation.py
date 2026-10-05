from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.correlation.rule_temporal_correlator import CorrelationConfig  # noqa: E402
from src.correlation.phase16b_validation import (  # noqa: E402
    PHASE16B_VERSION,
    build_controlled_validation_scenarios,
    build_exceptional_scenarios,
    evaluate_scenarios,
    sensitivity_grid,
)


PHASE_NAME = (
    "PHASE 16B — TEMPORAL-RULE VALIDATION + EXCEPTIONAL DISTRIBUTED-ALERT ROBUSTNESS"
)

RESULT_ROOT = PROJECT_ROOT / "results" / "phase16"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "correlation"
CONFIG_FILE = PROJECT_ROOT / "configs" / "phase16b_validation_policy.json"

DATASET_MANIFEST_FILE = RESULT_ROOT / "phase16b_validation_dataset_manifest.json"
BASELINE_RESULTS_FILE = RESULT_ROOT / "phase16b_baseline_validation.json"
EXCEPTION_RESULTS_FILE = RESULT_ROOT / "phase16b_exceptional_cases.json"
SENSITIVITY_JSON_FILE = RESULT_ROOT / "phase16b_sensitivity_results.json"
SENSITIVITY_CSV_FILE = RESULT_ROOT / "phase16b_sensitivity_results.csv"
AUDIT_FILE = RESULT_ROOT / "phase16b_audit.json"
MANIFEST_FILE = ARTIFACT_ROOT / "phase16b_manifest.json"


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

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


def load_policy() -> Dict[str, Any]:
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Missing Phase16B config: {CONFIG_FILE}"
        )

    with open(
        CONFIG_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def main():
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    policy = load_policy()
    baseline_cfg = policy["baseline"]

    baseline_config = CorrelationConfig(
        window_seconds=baseline_cfg["window_seconds"],
        min_cross_client_clients=baseline_cfg["min_cross_client_clients"],
        min_multi_class_clients=baseline_cfg["min_multi_class_clients"],
        min_multi_class_classes=baseline_cfg["min_multi_class_classes"],
        min_destination_clients=baseline_cfg["min_destination_clients"],
        min_destination_sources=baseline_cfg["min_destination_sources"],
    ).validate()

    validation_scenarios = build_controlled_validation_scenarios(seed=42)
    exceptional_scenarios = build_exceptional_scenarios(seed=42)

    print()
    print("=" * 132)
    print(PHASE_NAME)
    print("=" * 132)

    print(f"Phase16A baseline window       : {baseline_config.window_seconds} seconds")
    print(f"Labeled validation scenarios  : {len(validation_scenarios)}")
    print(f"Exceptional scenarios         : {len(exceptional_scenarios)}")
    print("Raw identifiers               : NO")
    print("Locked IDS test               : NO")
    print("Phase16A baseline modified     : NO")

    print()
    print("SCIENTIFIC NOTE:")
    print(
        "This phase uses a separately generated privacy-safe labeled correlation-validation dataset."
    )
    print(
        "It evaluates correlation logic, not intrusion-classifier accuracy."
    )

    dataset_manifest = {
        "phase": "16B",
        "version": PHASE16B_VERSION,
        "validation_scenarios": [
            scenario.to_manifest()
            for scenario
            in validation_scenarios
        ],
        "exceptional_scenarios": [
            scenario.to_manifest()
            for scenario
            in exceptional_scenarios
        ],
        "raw_identifiers_used": False,
        "locked_test_used": False,
    }

    save_json(
        DATASET_MANIFEST_FILE,
        dataset_manifest,
    )

    # ============================================================
    # 1. Baseline ground-truth validation
    # ============================================================
    baseline_result = evaluate_scenarios(
        validation_scenarios,
        baseline_config,
    )

    save_json(
        BASELINE_RESULTS_FILE,
        baseline_result,
    )

    print()
    print("GROUND-TRUTH VALIDATION")
    print("-" * 132)

    for row in baseline_result["scenarios"]:
        print(
            f"[{'PASS' if row['passed'] else 'FAIL'}] "
            f"{row['scenario_id']:<36} "
            f"expected={str(row['expected_correlated']):<5} "
            f"predicted={str(row['predicted_correlated']):<5} "
            f"rules={','.join(row['predicted_rules']) or 'NONE'}"
        )

    m = baseline_result["overall"]

    print()
    print(
        f"Precision                     : {m['precision']:.6f}"
    )
    print(
        f"Recall                        : {m['recall']:.6f}"
    )
    print(
        f"F1                            : {m['f1']:.6f}"
    )
    print(
        f"False correlation rate        : {m['false_correlation_rate']:.6f}"
    )

    baseline_pass = (
        m["precision"] == 1.0
        and
        m["recall"] == 1.0
        and
        all(row["passed"] for row in baseline_result["scenarios"])
    )

    # ============================================================
    # 2. Exceptional temporal/distributed cases
    # ============================================================
    exceptional_result = evaluate_scenarios(
        exceptional_scenarios,
        baseline_config,
    )

    save_json(
        EXCEPTION_RESULTS_FILE,
        exceptional_result,
    )

    print()
    print("EXCEPTIONAL CASE VALIDATION")
    print("-" * 132)

    for row in exceptional_result["scenarios"]:
        note = ""

        if row["exceptional_case"] == "DUPLICATE_REPLAY":
            note = f" | duplicates_removed={row['duplicates_removed']}"

        print(
            f"[{'PASS' if row['passed'] else 'FAIL'}] "
            f"{row['scenario_id']:<30} "
            f"{row['exceptional_case']:<18} "
            f"events={row['event_count']}{note}"
        )

    exceptional_pass = all(
        row["passed"]
        for row
        in exceptional_result["scenarios"]
    )

    duplicate_row = next(
        row
        for row
        in exceptional_result["scenarios"]
        if row["exceptional_case"] == "DUPLICATE_REPLAY"
    )

    duplicate_defense_pass = (
        duplicate_row["duplicates_removed"] == 1
    )

    # ============================================================
    # 3. Window + threshold sensitivity
    # ============================================================
    thresholds = policy["threshold_sensitivity"]

    sensitivity_rows = sensitivity_grid(
        validation_scenarios,
        windows=policy["window_sensitivity_seconds"],
        r1_clients=thresholds["r1_min_clients"],
        r2_clients=thresholds["r2_min_clients"],
        r2_classes=thresholds["r2_min_classes"],
        r3_clients=thresholds["r3_min_clients"],
        r3_sources=thresholds["r3_min_sources"],
    )

    save_json(
        SENSITIVITY_JSON_FILE,
        sensitivity_rows,
    )

    if sensitivity_rows:
        with open(
            SENSITIVITY_CSV_FILE,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(sensitivity_rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(sensitivity_rows)

    best = sensitivity_rows[0]

    baseline_grid_match = next(
        row
        for row
        in sensitivity_rows
        if (
            row["window_seconds"] == baseline_config.window_seconds
            and
            row["r1_min_clients"] == baseline_config.min_cross_client_clients
            and
            row["r2_min_clients"] == baseline_config.min_multi_class_clients
            and
            row["r2_min_classes"] == baseline_config.min_multi_class_classes
            and
            row["r3_min_clients"] == baseline_config.min_destination_clients
            and
            row["r3_min_sources"] == baseline_config.min_destination_sources
        )
    )

    print()
    print("WINDOW / THRESHOLD SENSITIVITY")
    print("-" * 132)

    print(f"Configurations evaluated      : {len(sensitivity_rows)}")
    print(
        f"Baseline 60s F1               : {baseline_grid_match['f1']:.6f}"
    )
    print(
        f"Baseline false corr. rate     : {baseline_grid_match['false_correlation_rate']:.6f}"
    )
    print(
        f"Best validation F1            : {best['f1']:.6f}"
    )
    print(
        f"Best validation window        : {best['window_seconds']} seconds"
    )
    print(
        f"Best R1 min clients           : {best['r1_min_clients']}"
    )
    print(
        f"Best R2 clients/classes       : {best['r2_min_clients']} / {best['r2_min_classes']}"
    )
    print(
        f"Best R3 clients/sources       : {best['r3_min_clients']} / {best['r3_min_sources']}"
    )

    # Phase16B evaluates sensitivity only. It does NOT freeze the top row.
    sensitivity_pass = len(sensitivity_rows) > 0

    all_passed = all(
        [
            baseline_pass,
            exceptional_pass,
            duplicate_defense_pass,
            sensitivity_pass,
        ]
    )

    audit = {
        "phase": "16B",
        "phase_name": PHASE_NAME,
        "phase_version": PHASE16B_VERSION,
        "baseline_config": baseline_cfg,
        "baseline_metrics": baseline_result["overall"],
        "per_rule_metrics": baseline_result["per_rule"],
        "exceptional_metrics": exceptional_result["overall"],
        "baseline_ground_truth_validation_passed": baseline_pass,
        "exceptional_case_validation_passed": exceptional_pass,
        "duplicate_replay_defense_passed": duplicate_defense_pass,
        "sensitivity_configuration_count": len(sensitivity_rows),
        "baseline_grid_result": baseline_grid_match,
        "best_validation_grid_result": best,
        "configuration_frozen_in_phase16b": False,
        "configuration_freeze_deferred_to": "Phase16C",
        "phase16a_baseline_modified": False,
        "phase14_risk_semantics_modified": False,
        "raw_identifiers_used": False,
        "locked_test_used": False,
        "all_tests_passed": all_passed,
    }

    save_json(
        AUDIT_FILE,
        audit,
    )

    manifest = {
        "phase": "16B",
        "version": PHASE16B_VERSION,
        "purpose": (
            "Ground-truth rule/temporal correlation validation and exceptional "
            "distributed-alert robustness testing."
        ),
        "exceptional_cases": [
            "delayed alert arrival",
            "out-of-order input",
            "35-second client clock skew",
            "one missing client alert",
            "exact duplicate/replay alert_id",
        ],
        "metrics": [
            "precision",
            "recall",
            "F1",
            "false positive rate",
            "false correlation rate",
            "per-rule precision/recall/F1",
        ],
        "sensitivity": {
            "windows_seconds": policy["window_sensitivity_seconds"],
            "threshold_grid": thresholds,
            "configuration_count": len(sensitivity_rows),
        },
        "important_limitations": [
            "Validation alerts are synthetic privacy-safe correlation scenarios.",
            "Results do not measure IDS attack-classification accuracy.",
            "Arrival-time delay is evaluated while correlation continues to use event_time.",
            "Clock-skew testing is controlled; no production clock synchronization mechanism is claimed.",
            "Exact duplicate alert_id replay is defensively deduplicated before correlation.",
            "The best sensitivity row is not automatically selected or frozen in Phase16B.",
        ],
        "locked_test_used": False,
        "ready_for_phase16c": all_passed,
    }

    save_json(
        MANIFEST_FILE,
        manifest,
    )

    print()
    print("=" * 132)
    print("PHASE 16B COMPLETE")
    print("=" * 132)

    print(
        f"Ground-truth baseline         : {'PASS' if baseline_pass else 'FAIL'}"
    )
    print(
        f"Exceptional cases             : {'PASS' if exceptional_pass else 'FAIL'}"
    )
    print(
        f"Duplicate/replay defense      : {'PASS' if duplicate_defense_pass else 'FAIL'}"
    )
    print(
        f"Baseline Precision / Recall   : {m['precision']:.4f} / {m['recall']:.4f}"
    )
    print(
        f"Baseline F1                   : {m['f1']:.4f}"
    )
    print(
        f"Baseline false corr. rate     : {m['false_correlation_rate']:.4f}"
    )
    print(
        f"Sensitivity configs           : {len(sensitivity_rows)}"
    )
    print(
        f"Locked test used              : NO"
    )
    print(
        f"Audit                         : {AUDIT_FILE}"
    )
    print(
        f"Baseline results              : {BASELINE_RESULTS_FILE}"
    )
    print(
        f"Exceptional results           : {EXCEPTION_RESULTS_FILE}"
    )
    print(
        f"Sensitivity CSV               : {SENSITIVITY_CSV_FILE}"
    )
    print(
        f"Manifest                      : {MANIFEST_FILE}"
    )

    print()
    print(
        f"STATUS                        : {'PASS' if all_passed else 'FAIL'}"
    )

    print()
    print("IMPORTANT:")
    print(
        "Phase16B is a validation/sensitivity phase. Do not freeze the best window/threshold "
        "configuration until Phase16C applies a predeclared selection rule."
    )

    print()
    print("NEXT:")
    print(
        "Phase 16C — predeclared configuration selection + robustness boundary analysis + rule correlator freeze."
    )

    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
