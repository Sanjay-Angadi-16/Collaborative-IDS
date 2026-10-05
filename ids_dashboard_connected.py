from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent

# Supports either:
#   project_root/ids_dashboard_connected.py
# or:
#   project_root/scripts/ids_dashboard_connected.py
if (PROJECT_ROOT / "results").exists():
    ROOT = PROJECT_ROOT
else:
    ROOT = PROJECT_ROOT.parent

ALERTS_FILE = (
    ROOT
    / "results"
    / "dashboard"
    / "validation_replay_alerts.json"
)

SUMMARY_FILE = (
    ROOT
    / "results"
    / "dashboard"
    / "validation_replay_summary.json"
)

app = FastAPI(
    title="Federated Collaborative IDS Dashboard - Connected Validation Replay"
)

# Manual analyst block decisions for live data.
# Validation replay records are intentionally not blockable because the
# sequence artifact does not contain raw source IP addresses.
BLOCKED_SOURCE_IDS: set[str] = set()


def read_json(path: Path, default: Any):
    if not path.exists():
        return default

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def get_alerts():
    return read_json(
        ALERTS_FILE,
        [],
    )


def get_summary():
    return read_json(
        SUMMARY_FILE,
        {},
    )


@app.get("/api/alerts")
def api_alerts():
    alerts = get_alerts()

    # Newest/highest validation replay row first.
    return list(
        reversed(alerts)
    )


@app.get("/api/summary")
def api_summary():
    return get_summary()


@app.post("/api/reload")
def reload_data():
    return {
        "alerts": len(
            get_alerts()
        ),
        "summary_available": bool(
            get_summary()
        ),
        "source": str(
            ALERTS_FILE
        ),
    }


@app.post("/api/alerts/{alert_id}/block")
def block_source(alert_id: int):
    alerts = get_alerts()

    alert = next(
        (
            a
            for a in alerts
            if int(a["id"])
            == alert_id
        ),
        None,
    )

    if alert is None:
        raise HTTPException(
            status_code=404,
            detail="Alert not found",
        )

    if not alert.get(
        "blockable",
        False,
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "This is a validation replay record. "
                "Raw source IP is not stored in validation_sequences.npz, "
                "so it cannot be firewall-blocked. "
                "Live/local collector alerts can be blockable later."
            ),
        )

    source_id = str(
        alert.get(
            "source_id",
            "",
        )
    )

    BLOCKED_SOURCE_IDS.add(
        source_id
    )

    return {
        "message": "Source marked blocked",
        "source_id": source_id,
    }


HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Federated Collaborative IDS</title>
<style>
* { box-sizing: border-box; }

body {
    margin: 0;
    font-family: Arial, Helvetica, sans-serif;
    background: #08111f;
    color: #e8eef7;
}

header {
    background: #0c1829;
    border-bottom: 1px solid #263752;
    padding: 20px 28px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

h1 { margin: 0; font-size: 23px; }
.subtitle { color: #92a6c2; margin-top: 5px; font-size: 13px; }

.tag {
    border-radius: 999px;
    padding: 8px 12px;
    background: #123b2b;
    color: #93f0ba;
    font-weight: 700;
    font-size: 12px;
}

.container {
    max-width: 1700px;
    margin: auto;
    padding: 22px;
}

.notice {
    background: #132238;
    border: 1px solid #2b4365;
    border-radius: 10px;
    padding: 13px 15px;
    margin-bottom: 18px;
    color: #b8c9df;
    line-height: 1.5;
    font-size: 13px;
}

.cards {
    display: grid;
    grid-template-columns: repeat(6, minmax(130px, 1fr));
    gap: 12px;
    margin-bottom: 18px;
}

.card {
    background: #0e1b2f;
    border: 1px solid #223550;
    border-radius: 11px;
    padding: 15px;
}

.label {
    color: #8fa5c1;
    font-size: 12px;
}

.value {
    font-size: 24px;
    margin-top: 7px;
    font-weight: 700;
}

.panel {
    background: #0e1b2f;
    border: 1px solid #223550;
    border-radius: 11px;
    padding: 16px;
    margin-bottom: 18px;
    overflow-x: auto;
}

.panel h2 {
    margin-top: 0;
    font-size: 17px;
}

.controls {
    display: flex;
    gap: 10px;
    margin-bottom: 12px;
    flex-wrap: wrap;
}

input, select, button {
    border: 1px solid #334b6d;
    background: #101f35;
    color: #e8eef7;
    border-radius: 7px;
    padding: 8px 10px;
}

button { cursor: pointer; }

table {
    width: 100%;
    min-width: 1550px;
    border-collapse: collapse;
}

th, td {
    padding: 10px 8px;
    border-bottom: 1px solid #23364f;
    text-align: left;
    font-size: 12px;
}

th {
    color: #9eb2cb;
    text-transform: uppercase;
    font-size: 11px;
}

.badge {
    border-radius: 6px;
    padding: 5px 8px;
    font-weight: 700;
    display: inline-block;
}

.NORMAL   { background: #113b2c; color: #8de2b1; }
.LOW      { background: #12345a; color: #83c4ff; }
.MEDIUM   { background: #50440c; color: #ffe377; }
.HIGH     { background: #542d0f; color: #ffb26a; }
.CRITICAL { background: #58141b; color: #ff939b; }

.correct { color: #8de2b1; font-weight: 700; }
.wrong { color: #ff939b; font-weight: 700; }

.muted { color: #91a4bd; }
.source { color: #7db9ff; }

.block-disabled {
    opacity: .45;
    cursor: not-allowed;
}

@media (max-width: 1100px) {
    .cards {
        grid-template-columns: repeat(2, 1fr);
    }
}
</style>
</head>

<body>
<header>
    <div>
        <h1>Federated Collaborative IDS</h1>
        <div class="subtitle">
            Validation Replay Dashboard — Real Phase-14 models, no locked-test data
        </div>
    </div>
    <div class="tag">VALIDATION MODE</div>
</header>

<div class="container">

    <div class="notice">
        This dashboard is connected to the real validation-replay output generated from
        the existing Random Forest, Personalized Autoencoder, FedProx70 and CNN-BiLSTM
        pipeline. The locked test is not opened. Raw IP addresses are not present in the
        saved validation sequence artifact, so validation rows show a pseudonymous source
        ID and the Block action is disabled. In live/local collection mode, the raw source
        IP can remain at the local node and become blockable after analyst confirmation.
    </div>

    <div class="cards">
        <div class="card">
            <div class="label">Replay Samples</div>
            <div class="value" id="samples">0</div>
        </div>

        <div class="card">
            <div class="label">Accuracy</div>
            <div class="value" id="accuracy">-</div>
        </div>

        <div class="card">
            <div class="label">Balanced Accuracy</div>
            <div class="value" id="balacc">-</div>
        </div>

        <div class="card">
            <div class="label">Macro F1</div>
            <div class="value" id="macrof1">-</div>
        </div>

        <div class="card">
            <div class="label">FAST Path</div>
            <div class="value" id="fastCount">0</div>
        </div>

        <div class="card">
            <div class="label">DEEP Path</div>
            <div class="value" id="deepCount">0</div>
        </div>
    </div>

    <div class="panel">
        <h2>Validation Replay Alerts / Predictions</h2>

        <div class="controls">
            <input id="search" placeholder="Search attack/source/client">
            <select id="severityFilter">
                <option value="">All severities</option>
                <option>NORMAL</option>
                <option>LOW</option>
                <option>MEDIUM</option>
                <option>HIGH</option>
                <option>CRITICAL</option>
            </select>
            <select id="resultFilter">
                <option value="">All results</option>
                <option value="correct">Correct</option>
                <option value="wrong">Wrong</option>
            </select>
            <button onclick="loadDashboard()">Reload</button>
        </div>

        <table>
            <thead>
            <tr>
                <th>#</th>
                <th>Client</th>
                <th>Source ID</th>
                <th>True Class</th>
                <th>Prediction</th>
                <th>Confidence</th>
                <th>AE Score</th>
                <th>Severity</th>
                <th>Path</th>
                <th>Decision Source</th>
                <th>RF</th>
                <th>FedProx70</th>
                <th>CNN-BiLSTM</th>
                <th>Result</th>
                <th>Action</th>
            </tr>
            </thead>
            <tbody id="rows"></tbody>
        </table>
    </div>
</div>

<script>
let allAlerts = [];

function pct(x) {
    if (x === null || x === undefined || Number.isNaN(Number(x))) return "-";
    return (Number(x) * 100).toFixed(1) + "%";
}

function num(x) {
    return Number(x || 0).toFixed(4);
}

function renderRows() {
    const q = document.getElementById("search").value.toLowerCase();
    const severity = document.getElementById("severityFilter").value;
    const result = document.getElementById("resultFilter").value;

    const rows = allAlerts.filter(a => {
        const haystack = [
            a.attack_type,
            a.true_class,
            a.source_id,
            "client " + a.client_id,
            a.decision_source
        ].join(" ").toLowerCase();

        if (q && !haystack.includes(q)) return false;
        if (severity && a.severity !== severity) return false;
        if (result === "correct" && !a.correct) return false;
        if (result === "wrong" && a.correct) return false;
        return true;
    });

    const body = document.getElementById("rows");
    body.innerHTML = "";

    for (const a of rows) {
        const tr = document.createElement("tr");

        const blockButton = a.blockable
            ? `<button onclick="blockSource(${a.id})">Block Source</button>`
            : `<button class="block-disabled" disabled title="${a.ip_note || ''}">No Raw IP</button>`;

        tr.innerHTML = `
            <td>${a.validation_index}</td>
            <td>Client ${a.client_id}</td>
            <td class="source">${a.source_id}</td>
            <td>${a.true_class}</td>
            <td>${a.attack_type}</td>
            <td>${pct(a.confidence)}</td>
            <td>${pct(a.anomaly_score)}</td>
            <td><span class="badge ${a.severity}">${a.severity}</span></td>
            <td>${a.processing_path}</td>
            <td>${a.decision_source}</td>
            <td>${a.rf_prediction}<br><span class="muted">${pct(a.rf_confidence)}</span></td>
            <td>${a.fedprox70_prediction}<br><span class="muted">${pct(a.fedprox70_confidence)}</span></td>
            <td>${a.cnn_bilstm_prediction}<br><span class="muted">${pct(a.cnn_bilstm_confidence)}</span></td>
            <td class="${a.correct ? 'correct' : 'wrong'}">${a.correct ? 'CORRECT' : 'WRONG'}</td>
            <td>${blockButton}</td>
        `;

        body.appendChild(tr);
    }
}

async function blockSource(alertId) {
    const r = await fetch(`/api/alerts/${alertId}/block`, {
        method: "POST"
    });

    const result = await r.json();

    if (!r.ok) {
        alert(result.detail || "Block failed");
        return;
    }

    alert(result.message);
}

async function loadDashboard() {
    const [alerts, summary] = await Promise.all([
        fetch("/api/alerts").then(r => r.json()),
        fetch("/api/summary").then(r => r.json())
    ]);

    allAlerts = alerts;

    document.getElementById("samples").textContent =
        summary.sample_count || alerts.length;

    const m = summary.metrics || {};

    document.getElementById("accuracy").textContent =
        m.accuracy === undefined ? "-" : pct(m.accuracy);

    document.getElementById("balacc").textContent =
        m.balanced_accuracy === undefined ? "-" : pct(m.balanced_accuracy);

    document.getElementById("macrof1").textContent =
        m.macro_f1 === undefined ? "-" : pct(m.macro_f1);

    document.getElementById("fastCount").textContent =
        summary.fast_path_count || 0;

    document.getElementById("deepCount").textContent =
        summary.deep_path_count || 0;

    renderRows();
}

document.getElementById("search").addEventListener("input", renderRows);
document.getElementById("severityFilter").addEventListener("change", renderRows);
document.getElementById("resultFilter").addEventListener("change", renderRows);

loadDashboard();
setInterval(loadDashboard, 10000);
</script>

</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTML


if __name__ == "__main__":
    uvicorn.run(
        "ids_dashboard_connected:app",
        host="127.0.0.1",
        port=9999,
        reload=True,
    )
