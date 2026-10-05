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

from src.correlation.phase16c_rule_freeze import (  # noqa: E402
    PHASE16C_VERSION,
    SelectionPolicy,
    build_boundary_scenarios,
    choose_configuration,
    config_from_row,
    delayed_arrival_invariance_test,
    missing_evidence_boundary_test,
    permutation_invariance_test,
    replay_characterization_test,
)
from src.correlation.phase16b_validation import evaluate_scenarios  # noqa: E402


PHASE_NAME = (
    "PHASE 16C — PREDECLARED CONFIGURATION SELECTION + ROBUSTNESS BOUNDARY + RULE CORRELATOR FREEZE"
)

RESULT_ROOT = PROJECT_ROOT / "results" / "phase16"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "correlation"

PHASE16B_AUDIT_FILE = RESULT_ROOT / "phase16b_audit.json"
PHASE16B_SENSITIVITY_FILE = RESULT_ROOT / "phase16b_sensitivity_results.csv"
POLICY_FILE = PROJECT_ROOT / "configs" / "phase16c_freeze_policy.json"

SELECTION_FILE = RESULT_ROOT / "phase16c_configuration_selection.json"
BOUNDARY_FILE = RESULT_ROOT / "phase16c_robustness_boundary.json"
AUDIT_FILE = RESULT_ROOT / "phase16c_audit.json"
FREEZE_MANIFEST_FILE = ARTIFACT_ROOT / "phase16c_rule_correlator_freeze_manifest.json"


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


def load_sensitivity_rows() -> List[Dict[str, Any]]:
    if not PHASE16B_SENSITIVITY_FILE.exists():
        raise FileNotFoundError(
            f"Missing Phase16B sensitivity CSV: {PHASE16B_SENSITIVITY_FILE}"
        )

    rows: List[Dict[str, Any]] = []

    with open(
        PHASE16B_SENSITIVITY_FILE,
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            converted = dict(row)

            for key in (
                "window_seconds",
                "r1_min_clients",
                "r2_min_clients",
                "r2_min_classes",
                "r3_min_clients",
                "r3_min_sources",
                "tp",
                "fp",
                "tn",
                "fn",
            ):
                converted[key] = int(
                    float(
                        converted[key]
                    )
                )

            for key in (
                "precision",
                "recall",
                "f1",
                "accuracy",
                "false_positive_rate",
                "false_correlation_rate",
            ):
                converted[key] = float(
                    converted[key]
                )

            rows.append(converted)

    return rows


def main():
    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARTIFACT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    phase16b = load_json(
        PHASE16B_AUDIT_FILE
    )

    policy_payload = load_json(
        POLICY_FILE
    )

    sensitivity_rows = load_sensitivity_rows()

    if not phase16b.get(
        "all_tests_passed",
        False,
    ):
        raise RuntimeError(
            "Phase16B did not pass. Phase16C freeze is prohibited."
        )

    if phase16b.get(
        "locked_test_used",
        True,
    ):
        raise RuntimeError(
            "Locked test usage detected. Phase16C freeze is prohibited."
        )

    selection_cfg = policy_payload[
        "selection_policy"
    ]

    selection_policy = SelectionPolicy(
        material_f1_improvement=float(
            selection_cfg[
                "material_f1_improvement"
            ]
        ),
        material_false_corr_improvement=float(
            selection_cfg[
                "material_false_correlation_improvement"
            ]
        ),
        minimum_recall=float(
            selection_cfg[
                "minimum_recall"
            ]
        ),
    )

    selection = choose_configuration(
        phase16b_audit=phase16b,
        sensitivity_rows=sensitivity_rows,
        policy=selection_policy,
    )

    selected_row = selection["selected"]
    selected_config = config_from_row(
        selected_row
    )

    print()
    print("=" * 136)
    print(PHASE_NAME)
    print("=" * 136)

    print(
        f"Phase16B status               : PASS"
    )
    print(
        f"Locked IDS test               : NO"
    )
    print(
        f"Phase16A baseline window      : 60 seconds"
    )
    print(
        f"Top-ranked sensitivity window : {phase16b['best_validation_grid_result']['window_seconds']} seconds"
    )
    print(
        f"Selected freeze window        : {selected_config.window_seconds} seconds"
    )
    print(
        f"Changed from baseline         : {'YES' if selection['changed_from_phase16a_baseline'] else 'NO'}"
    )

    print()
    print("PREDECLARED SELECTION DECISION")
    print("-" * 136)

    print(selection["selection_reason"])
    print(
        f"Observed best-row F1 gain     : {selection['observed_f1_gain_of_best_row']:.6f}"
    )
    print(
        f"Observed false-corr gain      : {selection['observed_false_correlation_gain_of_best_row']:.6f}"
    )

    save_json(
        SELECTION_FILE,
        selection,
    )

    # ============================================================
    # Robustness boundary characterization
    # ============================================================
    boundary_scenarios = build_boundary_scenarios(
        window_seconds=selected_config.window_seconds,
        seed=42,
    )

    boundary_result = evaluate_scenarios(
        boundary_scenarios,
        selected_config,
    )

    boundary_pass = all(
        row["passed"]
        for row
        in boundary_result["scenarios"]
    )

    print()
    print("TEMPORAL BOUNDARY CHARACTERIZATION")
    print("-" * 136)

    for row in boundary_result["scenarios"]:
        print(
            f"[{'PASS' if row['passed'] else 'FAIL'}] "
            f"{row['scenario_id']:<34} "
            f"expected={str(row['expected_correlated']):<5} "
            f"predicted={str(row['predicted_correlated']):<5}"
        )

    permutation = permutation_invariance_test(
        config=selected_config,
        permutations_to_test=20,
        seed=42,
    )

    delayed = delayed_arrival_invariance_test(
        config=selected_config,
    )

    missing = missing_evidence_boundary_test(
        config=selected_config,
    )

    replay = replay_characterization_test(
        config=selected_config,
    )

    print()
    print("ROBUSTNESS INVARIANTS")
    print("-" * 136)

    print(
        f"[{'PASS' if permutation['passed'] else 'FAIL'}] "
        f"out-of-order permutation invariance ({permutation['permutations_tested']} permutations)"
    )

    print(
        f"[{'PASS' if delayed['passed'] else 'FAIL'}] "
        "delayed-arrival invariance using event_time"
    )

    print(
        f"[{'PASS' if missing['passed'] else 'FAIL'}] "
        "missing-evidence boundary"
    )

    print(
        f"[{'PASS' if replay['exact_alert_id_replay_removed'] else 'FAIL'}] "
        "exact alert_id replay defense"
    )

    print(
        f"[INFO] fresh-alert_id semantic replay removed : "
        f"{'YES' if replay['new_alert_id_semantic_replay_removed'] else 'NO'}"
    )

    if not replay["new_alert_id_semantic_replay_removed"]:
        print(
            "[KNOWN LIMITATION] Fresh-ID semantic replay is not deduplicated by exact alert_id logic."
        )

    robustness = {
        "temporal_boundary": boundary_result,
        "permutation_invariance": permutation,
        "delayed_arrival_invariance": delayed,
        "missing_evidence_boundary": missing,
        "replay_characterization": replay,
    }

    save_json(
        BOUNDARY_FILE,
        robustness,
    )

    freeze_pass = all(
        [
            boundary_pass,
            permutation["passed"],
            delayed["passed"],
            missing["passed"],
            replay["exact_alert_id_replay_removed"],
        ]
    )

    all_passed = (
        freeze_pass
        and
        phase16b.get(
            "all_tests_passed",
            False,
        )
    )

    selected_contract = {
        "window_seconds": selected_config.window_seconds,
        "r1_min_clients": selected_config.min_cross_client_clients,
        "r2_min_clients": selected_config.min_multi_class_clients,
        "r2_min_classes": selected_config.min_multi_class_classes,
        "r3_min_clients": selected_config.min_destination_clients,
        "r3_min_sources": selected_config.min_destination_sources,
    }

    audit = {
        "phase": "16C",
        "phase_version": PHASE16C_VERSION,
        "phase_name": PHASE_NAME,
        "selection": selection,
        "selected_contract": selected_contract,
        "boundary_audit_passed": boundary_pass,
        "permutation_invariance_passed": permutation["passed"],
        "delayed_arrival_invariance_passed": delayed["passed"],
        "missing_evidence_boundary_passed": missing["passed"],
        "exact_duplicate_replay_defense_passed": replay["exact_alert_id_replay_removed"],
        "fresh_id_semantic_replay_defense_present": replay["new_alert_id_semantic_replay_removed"],
        "fresh_id_semantic_replay_known_limitation": (
            not replay["new_alert_id_semantic_replay_removed"]
        ),
        "phase14_risk_semantics_modified": False,
        "correlation_priority_is_probability": False,
        "locked_test_used": False,
        "rule_correlator_frozen": all_passed,
        "all_tests_passed": all_passed,
    }

    save_json(
        AUDIT_FILE,
        audit,
    )

    freeze_manifest = {
        "phase": "16C",
        "version": PHASE16C_VERSION,
        "status": "FROZEN" if all_passed else "NOT_FROZEN",
        "frozen_rule_correlator_contract": selected_contract,
        "selection_policy": policy_payload["selection_policy"],
        "frozen_semantics": {
            "correlation_uses_event_time": True,
            "arrival_order_does_not_define_event_time": True,
            "correlation_priority_is_rule_priority_only": True,
            "correlation_priority_is_not_phase14_risk": True,
            "correlation_priority_is_not_probability": True,
            "exact_alert_id_replay_is_deduplicated": True,
        },
        "known_limitations": [
            "Fresh alert_id semantic replay is not removed by exact-ID deduplication.",
            "Synthetic privacy-safe validation scenarios do not prove production correlation accuracy.",
            "Clock synchronization itself is not implemented; only controlled skew behavior is characterized.",
            "The current deterministic window is boundary-sensitive by construction.",
        ],
        "do_not_change_without_new_version": [
            "window_seconds",
            "R1 minimum client threshold",
            "R2 minimum client threshold",
            "R2 minimum classification threshold",
            "R3 minimum client threshold",
            "R3 minimum source threshold",
            "correlation-priority semantics",
        ],
        "locked_test_used": False,
        "ready_for_phase16d_gnn": all_passed,
    }

    save_json(
        FREEZE_MANIFEST_FILE,
        freeze_manifest,
    )

    print()
    print("=" * 136)
    print("PHASE 16C COMPLETE")
    print("=" * 136)

    print(
        f"Selected temporal window      : {selected_config.window_seconds} seconds"
    )
    print(
        f"R1 minimum clients            : {selected_config.min_cross_client_clients}"
    )
    print(
        f"R2 clients / classes          : "
        f"{selected_config.min_multi_class_clients} / "
        f"{selected_config.min_multi_class_classes}"
    )
    print(
        f"R3 clients / sources          : "
        f"{selected_config.min_destination_clients} / "
        f"{selected_config.min_destination_sources}"
    )
    print(
        f"Temporal boundary audit       : {'PASS' if boundary_pass else 'FAIL'}"
    )
    print(
        f"Out-of-order invariance       : {'PASS' if permutation['passed'] else 'FAIL'}"
    )
    print(
        f"Delayed-arrival invariance    : {'PASS' if delayed['passed'] else 'FAIL'}"
    )
    print(
        f"Missing-evidence boundary     : {'PASS' if missing['passed'] else 'FAIL'}"
    )
    print(
        f"Exact replay defense          : {'PASS' if replay['exact_alert_id_replay_removed'] else 'FAIL'}"
    )
    print(
        f"Fresh-ID replay limitation    : "
        f"{'DOCUMENTED' if not replay['new_alert_id_semantic_replay_removed'] else 'NOT PRESENT'}"
    )
    print(
        f"Rule correlator frozen        : {'YES' if all_passed else 'NO'}"
    )
    print(
        f"Locked test used              : NO"
    )
    print(
        f"Selection                     : {SELECTION_FILE}"
    )
    print(
        f"Boundary audit                : {BOUNDARY_FILE}"
    )
    print(
        f"Audit                         : {AUDIT_FILE}"
    )
    print(
        f"Freeze manifest               : {FREEZE_MANIFEST_FILE}"
    )

    print()
    print(
        f"STATUS                        : {'PASS' if all_passed else 'FAIL'}"
    )

    print()
    print("NEXT:")
    print(
        "Phase 16D — GNN alert-correlation dataset/graph construction and GNN baseline, "
        "using the frozen Phase16C rule correlator as the deterministic comparison baseline."
    )

    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
