from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import hmac
import ipaddress
import secrets
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

app = FastAPI(title="Federated Collaborative IDS Dashboard")

# -------------------------------------------------------------------
# DEMO / LOCAL DASHBOARD STORAGE
# -------------------------------------------------------------------
# This demo keeps raw IP addresses only in memory.
# For the central privacy-safe store, persist only pseudonymized IDs.
# Replace this list later with Phase-15 alert-store/database queries.
# -------------------------------------------------------------------

HMAC_KEY = b"change-this-key-in-production"

BLOCKED_IPS: set[str] = set()

ALERTS = [
    {
        "id": 1001,
        "event_time": "2026-09-29 10:31:44",
        "client_id": 2,
        "source_ip": "192.168.10.25",
        "destination_ip": "10.0.0.8",
        "attack_type": "DDoS",
        "model": "FedProx70",
        "confidence": 0.982,
        "anomaly_score": 0.72,
        "correlated": True,
        "status": "ACTIVE",
    },
    {
        "id": 1002,
        "event_time": "2026-09-29 10:30:11",
        "client_id": 4,
        "source_ip": "172.16.5.44",
        "destination_ip": "10.0.0.15",
        "attack_type": "DoS GoldenEye",
        "model": "CNN-BiLSTM",
        "confidence": 0.867,
        "anomaly_score": 0.64,
        "correlated": True,
        "status": "ACTIVE",
    },
    {
        "id": 1003,
        "event_time": "2026-09-29 10:28:52",
        "client_id": 3,
        "source_ip": "192.168.20.91",
        "destination_ip": "10.0.0.21",
        "attack_type": "PortScan",
        "model": "RandomForest",
        "confidence": 0.741,
        "anomaly_score": 0.38,
        "correlated": False,
        "status": "ACTIVE",
    },
]


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def pseudonymize_ip(ip: str) -> str:
    digest = hmac.new(
        HMAC_KEY,
        ip.encode("utf-8"),
        sha256,
    ).hexdigest()
    return f"SRC-{digest[:12].upper()}"


def validate_ip(ip: str) -> str:
    try:
        return str(ipaddress.ip_address(ip))
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid IP address: {ip}",
        ) from exc


def severity_from_confidence(confidence: float) -> str:
    if confidence >= 0.95:
        return "CRITICAL"
    if confidence >= 0.80:
        return "HIGH"
    if confidence >= 0.65:
        return "MEDIUM"
    return "LOW"


def enrich(alert: dict) -> dict:
    item = dict(alert)
    item["severity"] = severity_from_confidence(
        float(item["confidence"])
    )
    item["source_id"] = pseudonymize_ip(
        item["source_ip"]
    )
    item["blocked"] = item["source_ip"] in BLOCKED_IPS
    return item


# -------------------------------------------------------------------
# API MODELS
# -------------------------------------------------------------------

class AlertCreate(BaseModel):
    client_id: int = Field(ge=1, le=1000)
    source_ip: str
    destination_ip: str
    attack_type: str
    model: str
    confidence: float = Field(ge=0.0, le=1.0)
    anomaly_score: float = Field(default=0.0, ge=0.0, le=1.0)
    correlated: bool = False
    event_time: Optional[str] = None


class BlockRequest(BaseModel):
    reason: str = "Confirmed by security analyst"


# -------------------------------------------------------------------
# API ENDPOINTS
# -------------------------------------------------------------------

@app.get("/api/alerts")
def get_alerts():
    return [enrich(a) for a in reversed(ALERTS)]


@app.post("/api/alerts")
def create_alert(payload: AlertCreate):
    source_ip = validate_ip(payload.source_ip)
    destination_ip = validate_ip(payload.destination_ip)

    new_alert = {
        "id": max([a["id"] for a in ALERTS], default=1000) + 1,
        "event_time": payload.event_time
        or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "client_id": payload.client_id,
        "source_ip": source_ip,
        "destination_ip": destination_ip,
        "attack_type": payload.attack_type,
        "model": payload.model,
        "confidence": payload.confidence,
        "anomaly_score": payload.anomaly_score,
        "correlated": payload.correlated,
        "status": "ACTIVE",
    }

    ALERTS.append(new_alert)
    return enrich(new_alert)


@app.post("/api/alerts/{alert_id}/block")
def block_source(alert_id: int, payload: BlockRequest):
    alert = next(
        (a for a in ALERTS if a["id"] == alert_id),
        None,
    )

    if alert is None:
        raise HTTPException(
            status_code=404,
            detail="Alert not found",
        )

    source_ip = validate_ip(alert["source_ip"])

    # --------------------------------------------------------------
    # SAFE DEFAULT:
    # We record the block decision but DO NOT modify the OS firewall.
    #
    # Later this function can call:
    # - Windows Defender Firewall
    # - Linux nftables/iptables
    # - AWS Security Group API
    # - Azure NSG API
    #
    # after explicit analyst approval.
    # --------------------------------------------------------------
    BLOCKED_IPS.add(source_ip)

    alert["status"] = "BLOCKED"

    return {
        "message": "Source marked as blocked",
        "alert_id": alert_id,
        "source_ip": source_ip,
        "source_id": pseudonymize_ip(source_ip),
        "reason": payload.reason,
        "firewall_changed": False,
    }


@app.post("/api/alerts/{alert_id}/safe")
def mark_safe(alert_id: int):
    alert = next(
        (a for a in ALERTS if a["id"] == alert_id),
        None,
    )

    if alert is None:
        raise HTTPException(
            status_code=404,
            detail="Alert not found",
        )

    alert["status"] = "MARKED_SAFE"
    return enrich(alert)


@app.post("/api/alerts/{alert_id}/resolve")
def resolve_alert(alert_id: int):
    alert = next(
        (a for a in ALERTS if a["id"] == alert_id),
        None,
    )

    if alert is None:
        raise HTTPException(
            status_code=404,
            detail="Alert not found",
        )

    alert["status"] = "RESOLVED"
    return enrich(alert)


@app.delete("/api/blocked/{ip}")
def unblock_ip(ip: str):
    ip = validate_ip(ip)

    if ip not in BLOCKED_IPS:
        raise HTTPException(
            status_code=404,
            detail="IP is not in block list",
        )

    BLOCKED_IPS.remove(ip)

    for alert in ALERTS:
        if (
            alert["source_ip"] == ip
            and alert["status"] == "BLOCKED"
        ):
            alert["status"] = "ACTIVE"

    return {
        "message": "IP removed from block list",
        "source_ip": ip,
    }


@app.get("/api/blocked")
def blocked_ips():
    return [
        {
            "source_ip": ip,
            "source_id": pseudonymize_ip(ip),
        }
        for ip in sorted(BLOCKED_IPS)
    ]


# -------------------------------------------------------------------
# FRONTEND
# -------------------------------------------------------------------

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
        color: #e7eef9;
    }

    .header {
        padding: 22px 28px;
        border-bottom: 1px solid #223149;
        background: #0b1628;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .header h1 {
        margin: 0;
        font-size: 23px;
    }

    .header small {
        color: #8da2bf;
    }

    .healthy {
        padding: 8px 12px;
        border-radius: 999px;
        background: #123b2b;
        color: #93f0ba;
        font-weight: bold;
    }

    .container {
        padding: 24px;
        max-width: 1600px;
        margin: auto;
    }

    .cards {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 14px;
        margin-bottom: 22px;
    }

    .card {
        background: #0e1b2f;
        border: 1px solid #21324d;
        border-radius: 12px;
        padding: 16px;
    }

    .card .label {
        color: #91a5c2;
        font-size: 13px;
    }

    .card .value {
        font-size: 28px;
        margin-top: 7px;
        font-weight: 700;
    }

    .panel {
        background: #0e1b2f;
        border: 1px solid #21324d;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 20px;
        overflow-x: auto;
    }

    .panel h2 {
        margin-top: 0;
        font-size: 18px;
    }

    table {
        width: 100%;
        border-collapse: collapse;
        min-width: 1100px;
    }

    th, td {
        padding: 11px 10px;
        border-bottom: 1px solid #223149;
        text-align: left;
        font-size: 13px;
    }

    th {
        color: #9db0ca;
        font-size: 12px;
        text-transform: uppercase;
    }

    .badge {
        padding: 5px 9px;
        border-radius: 6px;
        font-weight: bold;
        display: inline-block;
    }

    .CRITICAL { background: #53141b; color: #ff8f99; }
    .HIGH     { background: #532c0e; color: #ffb36a; }
    .MEDIUM   { background: #4b420a; color: #ffe66e; }
    .LOW      { background: #12345a; color: #85c5ff; }

    button {
        border: 0;
        border-radius: 7px;
        padding: 7px 10px;
        cursor: pointer;
        font-weight: bold;
        margin-right: 4px;
    }

    .block-btn {
        background: #b4232c;
        color: white;
    }

    .safe-btn {
        background: #1c7a4b;
        color: white;
    }

    .resolve-btn {
        background: #365c9c;
        color: white;
    }

    .blocked {
        color: #ff7f87;
        font-weight: bold;
    }

    .normal {
        color: #95dfb5;
    }

    .privacy-note {
        color: #92a6c2;
        font-size: 12px;
        line-height: 1.5;
    }

    .source-id {
        color: #7bb6ff;
        font-size: 11px;
    }

    @media (max-width: 1000px) {
        .cards {
            grid-template-columns: repeat(2, 1fr);
        }
    }
</style>
</head>

<body>

<div class="header">
    <div>
        <h1>Federated Collaborative IDS</h1>
        <small>Security Operations Dashboard</small>
    </div>
    <div class="healthy">SYSTEM HEALTHY</div>
</div>

<div class="container">

    <div class="cards">
        <div class="card">
            <div class="label">Clients</div>
            <div class="value">5</div>
        </div>

        <div class="card">
            <div class="label">Total Alerts</div>
            <div class="value" id="totalAlerts">0</div>
        </div>

        <div class="card">
            <div class="label">Critical</div>
            <div class="value" id="criticalAlerts">0</div>
        </div>

        <div class="card">
            <div class="label">Blocked Sources</div>
            <div class="value" id="blockedCount">0</div>
        </div>

        <div class="card">
            <div class="label">Correlation</div>
            <div class="value">ACTIVE</div>
        </div>
    </div>

    <div class="panel">
        <h2>Live Security Alerts</h2>

        <table>
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Client</th>
                    <th>Source IP</th>
                    <th>Destination IP</th>
                    <th>Attack</th>
                    <th>Model</th>
                    <th>Confidence</th>
                    <th>Anomaly</th>
                    <th>Severity</th>
                    <th>Correlation</th>
                    <th>Status</th>
                    <th>Action</th>
                </tr>
            </thead>

            <tbody id="alertsBody"></tbody>
        </table>

        <p class="privacy-note">
            Local analyst view: raw source/destination IPs are visible for investigation.
            For a central privacy-safe store, persist the pseudonymized Source ID instead
            of the raw IP whenever possible.
        </p>
    </div>

    <div class="panel">
        <h2>Blocked Source List</h2>
        <div id="blockedList">No blocked IPs.</div>
    </div>

</div>

<script>

function pct(value) {
    return (value * 100).toFixed(1) + "%";
}

async function loadDashboard() {
    const alerts = await fetch("/api/alerts").then(r => r.json());
    const blocked = await fetch("/api/blocked").then(r => r.json());

    document.getElementById("totalAlerts").textContent =
        alerts.length;

    document.getElementById("criticalAlerts").textContent =
        alerts.filter(a => a.severity === "CRITICAL").length;

    document.getElementById("blockedCount").textContent =
        blocked.length;

    const body = document.getElementById("alertsBody");
    body.innerHTML = "";

    for (const alert of alerts) {
        const row = document.createElement("tr");

        row.innerHTML = `
            <td>${alert.event_time}</td>
            <td>Client ${alert.client_id}</td>

            <td>
                <div>${alert.source_ip}</div>
                <div class="source-id">${alert.source_id}</div>
            </td>

            <td>${alert.destination_ip}</td>
            <td>${alert.attack_type}</td>
            <td>${alert.model}</td>
            <td>${pct(alert.confidence)}</td>
            <td>${pct(alert.anomaly_score)}</td>

            <td>
                <span class="badge ${alert.severity}">
                    ${alert.severity}
                </span>
            </td>

            <td>
                ${alert.correlated ? "YES" : "NO"}
            </td>

            <td class="${alert.blocked ? "blocked" : "normal"}">
                ${alert.status}
            </td>

            <td>
                <button class="block-btn"
                    onclick="blockSource(${alert.id}, '${alert.source_ip}')">
                    Block IP
                </button>

                <button class="safe-btn"
                    onclick="markSafe(${alert.id})">
                    Mark Safe
                </button>

                <button class="resolve-btn"
                    onclick="resolveAlert(${alert.id})">
                    Resolve
                </button>
            </td>
        `;

        body.appendChild(row);
    }

    const blockedList =
        document.getElementById("blockedList");

    if (blocked.length === 0) {
        blockedList.textContent = "No blocked IPs.";
    } else {
        blockedList.innerHTML = blocked
            .map(item =>
                `<div style="margin-bottom:8px;">
                    <b>${item.source_ip}</b>
                    <span class="source-id">${item.source_id}</span>
                    <button onclick="unblock('${item.source_ip}')">
                        Unblock
                    </button>
                </div>`
            )
            .join("");
    }
}

async function blockSource(alertId, ip) {
    const confirmed = confirm(
        "Block source IP " + ip + "?\n\n" +
        "This demo records the analyst decision only. " +
        "It does not modify the operating-system firewall."
    );

    if (!confirmed) return;

    const response = await fetch(
        `/api/alerts/${alertId}/block`,
        {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                reason: "Manual analyst confirmation"
            })
        }
    );

    const result = await response.json();

    if (!response.ok) {
        alert(result.detail || "Block failed");
        return;
    }

    alert(
        "Source marked BLOCKED:\n" +
        result.source_ip +
        "\n\nFirewall changed: " +
        result.firewall_changed
    );

    loadDashboard();
}

async function markSafe(alertId) {
    await fetch(
        `/api/alerts/${alertId}/safe`,
        { method: "POST" }
    );

    loadDashboard();
}

async function resolveAlert(alertId) {
    await fetch(
        `/api/alerts/${alertId}/resolve`,
        { method: "POST" }
    );

    loadDashboard();
}

async function unblock(ip) {
    const response = await fetch(
        `/api/blocked/${encodeURIComponent(ip)}`,
        { method: "DELETE" }
    );

    if (!response.ok) {
        const result = await response.json();
        alert(result.detail || "Unblock failed");
        return;
    }

    loadDashboard();
}

loadDashboard();

setInterval(
    loadDashboard,
    5000
);

</script>

</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTML


if __name__ == "__main__":
    uvicorn.run(
        "ids_dashboard:app",
        host="127.0.0.1",
        port=9999,
        reload=True,
    )
