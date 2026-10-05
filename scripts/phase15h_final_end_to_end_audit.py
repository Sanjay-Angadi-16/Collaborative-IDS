from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(
    PROJECT_ROOT
) not in sys.path:
    sys.path.insert(
        0,
        str(
            PROJECT_ROOT
        ),
    )


from src.alert_sharing.phase15_final_audit import (  # noqa: E402
    EvidenceItem,
    Phase15FinalAuditError,
    extract_locked_test_flag,
    load_json,
    summarize_phase_status,
)


PHASE_NAME = (
    "PHASE 15H — FINAL END-TO-END AUDIT / REPORT FREEZE"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "phase15"
)

ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "privacy"
)

SUMMARY_FILE = (
    RESULT_ROOT
    / "phase15h_final_summary.json"
)

STATUS_TABLE_FILE = (
    RESULT_ROOT
    / "phase15h_phase_status.csv"
)

REPORT_FILE = (
    RESULT_ROOT
    / "phase15h_final_report.md"
)

FREEZE_FILE = (
    ARTIFACT_ROOT
    / "phase15h_phase15_freeze_manifest.json"
)


EVIDENCE = [
    EvidenceItem(
        phase="15A",
        name="Privacy-aware alert schema",
        path=RESULT_ROOT / "phase15a_schema_audit.json",
    ),
    EvidenceItem(
        phase="15B",
        name="HMAC-SHA256 pseudonymization",
        path=RESULT_ROOT / "phase15b_pseudonymization_audit.json",
    ),
    EvidenceItem(
        phase="15C",
        name="Allowlist privacy filter + leakage audit",
        path=RESULT_ROOT / "phase15c_privacy_filter_audit.json",
    ),
    EvidenceItem(
        phase="15D",
        name="Real Phase14G2 privacy-safe serialization integration",
        path=RESULT_ROOT / "phase15d_integration_audit.json",
    ),
    EvidenceItem(
        phase="15E1",
        name="Local collector + sender HTTP smoke test",
        path=RESULT_ROOT / "phase15e1_smoke_test_audit.json",
    ),
    EvidenceItem(
        phase="15E2",
        name="HTTPS/TLS + mutual-TLS authentication smoke test",
        path=RESULT_ROOT / "phase15e2_mtls_smoke_test_audit.json",
    ),
    EvidenceItem(
        phase="15F",
        name="Central persistence/indexing + multi-client ingestion",
        path=RESULT_ROOT / "phase15f_persistence_audit.json",
    ),
    EvidenceItem(
        phase="15G",
        name="Privacy/retention + central-store performance evaluation",
        path=RESULT_ROOT / "phase15g_privacy_retention_audit.json",
    ),
]


def save_json(
    path: Path,
    payload: Dict[str, Any],
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


def optional_json(
    path: Path,
) -> Dict[str, Any]:
    if not path.exists():
        return {}

    try:
        return load_json(
            path
        )
    except Exception:
        return {}


def extract_phase_metrics() -> Dict[str, Any]:
    phase15c = optional_json(
        RESULT_ROOT / "phase15c_privacy_filter_audit.json"
    )

    phase15d = optional_json(
        RESULT_ROOT / "phase15d_integration_audit.json"
    )

    phase15e1 = optional_json(
        RESULT_ROOT / "phase15e1_smoke_test_audit.json"
    )

    phase15e2 = optional_json(
        RESULT_ROOT / "phase15e2_mtls_smoke_test_audit.json"
    )

    phase15f_metrics = optional_json(
        RESULT_ROOT / "phase15f_ingestion_metrics.json"
    )

    phase15g_perf = optional_json(
        RESULT_ROOT / "phase15g_store_performance.json"
    )

    phase15g_audit = optional_json(
        RESULT_ROOT / "phase15g_privacy_retention_audit.json"
    )

    return {
        "phase15c":
            {
                "tests_total":
                    phase15c.get(
                        "tests_total"
                    ),

                "tests_passed":
                    phase15c.get(
                        "tests_passed"
                    ),

                "fast_size_metrics":
                    (
                        phase15c.get(
                            "size_metrics",
                            [{}],
                        )[
                            0
                        ]
                        if phase15c.get(
                            "size_metrics"
                        )
                        else
                        {}
                    ),

                "deep_size_metrics":
                    (
                        phase15c.get(
                            "size_metrics",
                            [{}, {}],
                        )[
                            1
                        ]
                        if len(
                            phase15c.get(
                                "size_metrics",
                                [],
                            )
                        )
                        >
                        1
                        else
                        {}
                    ),
            },

        "phase15d":
            {
                "classification":
                    phase15d.get(
                        "phase14_alert_classification"
                    ),

                "processing_path":
                    phase15d.get(
                        "phase14_processing_path"
                    ),

                "network_context_mode":
                    phase15d.get(
                        "network_context_mode"
                    ),

                "hmac_key_mode":
                    phase15d.get(
                        "hmac_key_mode"
                    ),

                "metrics":
                    phase15d.get(
                        "metrics",
                        {},
                    ),
            },

        "phase15e1":
            {
                "unique_sends_accepted":
                    phase15e1.get(
                        "accepted_unique_alerts"
                    ),

                "mean_round_trip_ms":
                    phase15e1.get(
                        "mean_sender_round_trip_ms"
                    ),
            },

        "phase15e2":
            {
                "accepted_unique_alerts":
                    phase15e2.get(
                        "accepted_unique_alerts"
                    ),

                "mean_mtls_round_trip_ms":
                    phase15e2.get(
                        "mean_sender_round_trip_ms"
                    ),

                "mutual_tls":
                    phase15e2.get(
                        "mutual_tls"
                    ),
            },

        "phase15f":
            phase15f_metrics,

        "phase15g":
            {
                "performance":
                    phase15g_perf,

                "retention_preview":
                    phase15g_audit.get(
                        "retention_preview",
                        {},
                    ),

                "retention_apply":
                    phase15g_audit.get(
                        "retention_apply",
                        {},
                    ),
            },
    }


def build_markdown_report(
    *,
    rows: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    all_passed: bool,
) -> str:
    status_word = (
        "COMPLETE / FROZEN"
        if all_passed
        else
        "NOT FROZEN"
    )

    lines = []

    lines.append(
        "# Phase 15 — Privacy-Aware Alert Sharing"
    )
    lines.append("")
    lines.append(
        f"**Final status:** {status_word}"
    )
    lines.append("")
    lines.append(
        "Phase 15 implements the privacy-preserving alert-sharing layer "
        "between the local IDS and the future cross-cloud correlation layer."
    )
    lines.append("")

    lines.append(
        "## Final architecture"
    )
    lines.append("")
    lines.extend(
        [
            "```text",
            "Phase14 Local IDS Alert",
            "        ↓",
            "Phase15A Privacy-Aware Schema",
            "        ↓",
            "Phase15B HMAC-SHA256 Pseudonymization",
            "        ↓",
            "Phase15C Allowlist Privacy Filter + Leakage Audit",
            "        ↓",
            "Phase15D Canonical Privacy-Safe Serialization",
            "        ↓",
            "Phase15E1 Local Collector Smoke Test",
            "        ↓",
            "Phase15E2 HTTPS/TLS + Mutual TLS Authentication",
            "        ↓",
            "Phase15F Central SQLite Persistence + Indexing",
            "        ↓",
            "Phase15G Privacy / Retention / Store Performance Audit",
            "        ↓",
            "Phase16 Cross-Cloud Correlation",
            "```",
            "",
        ]
    )

    lines.append(
        "## Phase status"
    )
    lines.append("")
    lines.append(
        "| Phase | Experiment | Result | Locked test |"
    )
    lines.append(
        "|---|---|---:|---:|"
    )

    for row in rows:
        lines.append(
            f"| {row['phase']} | {row['name']} | "
            f"{'PASS' if row['passed'] else 'FAIL'} | "
            f"{'NO' if row['locked_test_used'] is False else row['locked_test_used']} |"
        )

    lines.append("")
    lines.append(
        "## Frozen privacy contract"
    )
    lines.append("")
    lines.extend(
        [
            "- Raw source/destination IPs are not shared.",
            "- Raw packets and packet payloads are not shared.",
            "- Raw flow-feature vectors and CNN sequences are not shared.",
            "- Model parameters/state dictionaries are not shared.",
            "- Stable cross-alert linkage uses keyed HMAC-SHA256 pseudonyms.",
            "- Outbound alerts use an explicit allowlist schema.",
            "- FAST path preserves Phase14G2 semantics: risk_score is null and RF confidence is evidence.",
            "- DEEP path preserves Phase14G2 full-fusion risk semantics.",
            "- Collector transport was validated with HTTPS and mutual TLS in the Phase15E2 local proof-of-function.",
            "- Central storage persists only privacy-safe canonical alerts.",
            "- Retention policy supports a dry-run preview before deletion.",
            "",
        ]
    )

    lines.append(
        "## Selected measured results"
    )
    lines.append("")

    c = metrics.get(
        "phase15c",
        {},
    )

    d = metrics.get(
        "phase15d",
        {},
    )

    e1 = metrics.get(
        "phase15e1",
        {},
    )

    e2 = metrics.get(
        "phase15e2",
        {},
    )

    g = metrics.get(
        "phase15g",
        {},
    )

    perf = g.get(
        "performance",
        {},
    )

    retention = g.get(
        "retention_apply",
        {},
    )

    d_metrics = d.get(
        "metrics",
        {},
    )

    lines.append(
        f"- Phase15C leakage/privacy tests: "
        f"{c.get('tests_passed')} / {c.get('tests_total')} passed."
    )

    if d_metrics:
        lines.append(
            f"- Phase15D privacy-safe integration total: "
            f"{d_metrics.get('integration_total_ms', 'n/a')} ms "
            f"for the measured local run."
        )

        lines.append(
            f"- Phase15D shared alert size: "
            f"{d_metrics.get('privacy_safe_shared_bytes', 'n/a')} bytes."
        )

    lines.append(
        f"- Phase15E1 accepted unique localhost HTTP alerts: "
        f"{e1.get('unique_sends_accepted', 'n/a')}."
    )

    lines.append(
        f"- Phase15E1 mean sender round-trip: "
        f"{e1.get('mean_round_trip_ms', 'n/a')} ms "
        f"(functional smoke-test timing only)."
    )

    lines.append(
        f"- Phase15E2 accepted unique mTLS alerts: "
        f"{e2.get('accepted_unique_alerts', 'n/a')}."
    )

    lines.append(
        f"- Phase15E2 mean mTLS sender round-trip: "
        f"{e2.get('mean_mtls_round_trip_ms', 'n/a')} ms "
        f"(fresh TLS session per request; not a throughput claim)."
    )

    if perf:
        lines.append(
            f"- Phase15G local SQLite sequential ingest throughput: "
            f"{perf.get('sequential_ingest_throughput_alerts_per_second', 'n/a')} alerts/sec."
        )

        insert_latency = perf.get(
            "insert_latency",
            {},
        )

        lines.append(
            f"- Phase15G mean/P95 insert latency: "
            f"{insert_latency.get('mean_ms', 'n/a')} / "
            f"{insert_latency.get('p95_ms', 'n/a')} ms."
        )

    if retention:
        lines.append(
            f"- Phase15G retention test deleted "
            f"{retention.get('deleted_rows', 'n/a')} rows and retained "
            f"{retention.get('remaining_rows', 'n/a')} rows."
        )

    lines.append("")
    lines.append(
        "## Scientific limitations"
    )
    lines.append("")
    lines.extend(
        [
            "- Phase15B/C/D test runs used ephemeral HMAC keys unless an external PHASE15_HMAC_KEY_B64 secret was configured.",
            "- Phase15D used RFC5737 documentation network context when the Phase14 research dataset lacked trustworthy IP identifiers.",
            "- Phase15E2 proves local mTLS behavior with short-lived test certificates; it is not a production PKI deployment.",
            "- Phase15E1/E2 timings are smoke-test timings and must not be reported as production network latency.",
            "- Phase15G throughput is a single-process local SQLite implementation benchmark, not cloud or network throughput.",
            "- Synthetic privacy-safe ingestion messages in Phase15F/G are infrastructure-test data, not IDS detection-performance evidence.",
            "- No locked test set was used in Phase15.",
            "",
        ]
    )

    lines.append(
        "## Freeze decision"
    )
    lines.append("")

    if all_passed:
        lines.append(
            "Phase 15 is frozen. Do not retune privacy schema, HMAC token format, "
            "Phase14G2 risk semantics, or the privacy filter while developing Phase16. "
            "Any future changes should be versioned as a new privacy policy/schema."
        )
    else:
        lines.append(
            "Phase 15 is not frozen because at least one required evidence item failed."
        )

    lines.append("")
    lines.append(
        "## Next phase"
    )
    lines.append("")
    lines.append(
        "**Phase 16 — Cross-Cloud Correlation:** start with rule/temporal correlation "
        "over privacy-safe alerts, then add GNN-based correlation as a separate controlled stage."
    )

    return "\n".join(
        lines
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

    print()
    print(
        "="
        *
        126
    )

    print(
        PHASE_NAME
    )

    print(
        "="
        *
        126
    )

    rows: List[
        Dict[
            str,
            Any,
        ]
    ] = []

    all_passed = True
    any_locked_test = False

    print()
    print(
        "EVIDENCE AUDIT"
    )

    print(
        "-"
        *
        126
    )

    for item in EVIDENCE:
        payload = load_json(
            item.path
        )

        passed, basis = summarize_phase_status(
            phase=
                item.phase,

            payload=
                payload,
        )

        locked_test_used = extract_locked_test_flag(
            payload
        )

        if locked_test_used is True:
            any_locked_test = True

        row = {
            "phase":
                item.phase,

            "name":
                item.name,

            "path":
                str(
                    item.path
                ),

            "passed":
                passed,

            "status_basis":
                basis,

            "locked_test_used":
                locked_test_used,
        }

        rows.append(
            row
        )

        all_passed = (
            all_passed
            and
            passed
        )

        print(
            f"[{'PASS' if passed else 'FAIL'}] "
            f"{item.phase:5s} {item.name}"
        )

    locked_test_pass = (
        not any_locked_test
    )

    print()
    print(
        f"[{'PASS' if locked_test_pass else 'FAIL'}] "
        "Phase15 locked-test isolation"
    )

    all_passed = (
        all_passed
        and
        locked_test_pass
    )

    metrics = extract_phase_metrics()

    with open(
        STATUS_TABLE_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "phase",
            "name",
            "passed",
            "locked_test_used",
            "status_basis",
            "path",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=
                fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    summary = {
        "phase":
            "15H",

        "phase_name":
            PHASE_NAME,

        "required_evidence_count":
            len(
                EVIDENCE
            ),

        "passed_evidence_count":
            sum(
                1
                for row
                in rows
                if row[
                    "passed"
                ]
            ),

        "all_required_phases_passed":
            all(
                row[
                    "passed"
                ]
                for row
                in rows
            ),

        "locked_test_used_anywhere_in_phase15":
            any_locked_test,

        "locked_test_isolation_passed":
            locked_test_pass,

        "phase15_frozen":
            all_passed,

        "phase_status":
            rows,

        "selected_metrics":
            metrics,

        "next_phase":
            "Phase16 cross-cloud correlation",
    }

    save_json(
        SUMMARY_FILE,
        summary,
    )

    report_text = build_markdown_report(
        rows=
            rows,

        metrics=
            metrics,

        all_passed=
            all_passed,
    )

    REPORT_FILE.write_text(
        report_text,
        encoding="utf-8",
    )

    freeze_manifest = {
        "phase":
            "15H",

        "phase15_status":
            (
                "FROZEN"
                if all_passed
                else
                "NOT_FROZEN"
            ),

        "freeze_scope":
            [
                "Phase15A privacy-aware alert schema",
                "Phase15B HMAC-SHA256 pseudonymization format",
                "Phase15C explicit allowlist privacy filter",
                "Phase15D canonical serialization contract",
                "Phase15E1 collector application semantics",
                "Phase15E2 mTLS proof-of-function security contract",
                "Phase15F central privacy-safe persistence/indexes",
                "Phase15G retention-audit methodology",
            ],

        "do_not_change_without_new_version":
            [
                "schema_version",
                "privacy_version",
                "HMAC token namespace/prefix format",
                "Phase14G2 FAST/DEEP risk semantics",
                "outbound allowlist",
                "privacy leakage rules",
            ],

        "production_not_claimed":
            [
                "ephemeral HMAC key runs",
                "RFC5737 demo network context",
                "short-lived local mTLS certificates",
                "Phase15E1/E2 latency",
                "Phase15G local SQLite throughput",
            ],

        "locked_test_used":
            False,

        "ready_for_phase16":
            all_passed,
    }

    save_json(
        FREEZE_FILE,
        freeze_manifest,
    )

    print()
    print(
        "="
        *
        126
    )

    print(
        "PHASE 15H COMPLETE"
    )

    print(
        "="
        *
        126
    )

    print(
        f"Required evidence passed      : "
        f"{summary['passed_evidence_count']}/{summary['required_evidence_count']}"
    )

    print(
        f"Locked test isolation         : "
        f"{'PASS' if locked_test_pass else 'FAIL'}"
    )

    print(
        f"Phase15 freeze                : "
        f"{'YES' if all_passed else 'NO'}"
    )

    print(
        f"Summary                       : {SUMMARY_FILE}"
    )

    print(
        f"Status table                  : {STATUS_TABLE_FILE}"
    )

    print(
        f"Final report                  : {REPORT_FILE}"
    )

    print(
        f"Freeze manifest               : {FREEZE_FILE}"
    )

    print()

    print(
        f"STATUS                        : "
        f"{'PASS' if all_passed else 'FAIL'}"
    )

    print()

    if all_passed:
        print(
            "PHASE 15 STATUS:"
        )

        print(
            "Privacy-Aware Alert Sharing is COMPLETE and FROZEN."
        )

        print()

        print(
            "NEXT:"
        )

        print(
            "Phase 16 — Cross-Cloud Correlation using privacy-safe "
            "alerts: rule/temporal correlation first, GNN second."
        )

    else:
        print(
            "Phase 15 cannot be frozen until all required evidence passes."
        )

        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()
