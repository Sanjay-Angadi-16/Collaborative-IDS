from pathlib import Path
import sys
import json


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# IMPORT DETECTION ENGINE
# ============================================================

from src.detection.detection_engine import DetectionEngine


# ============================================================
# CREATE ENGINE
# ============================================================

engine = DetectionEngine()


print(
    "\n"
    +
    "=" * 90
)

print(
    "PHASE 14 — ADAPTIVE CLIENT-SIDE IDS ENGINE"
)

print(
    "=" * 90
)


# ============================================================
# SIMULATED INPUT
#
# Later this will come from actual preprocessed CICIDS flow.
# ============================================================

client_id = 3


# ============================================================
# 1. RANDOM FOREST OUTPUT
# ============================================================

rf_result = {

    "prediction":
        "PortScan",

    "confidence":
        0.72,
}


print(
    "\n1. RANDOM FOREST"
)

print(
    rf_result
)


# ============================================================
# 2. ADAPTIVE GATE
# ============================================================

gate = engine.adaptive_gate(

    rf_class=
        rf_result[
            "prediction"
        ],

    rf_confidence=
        rf_result[
            "confidence"
        ],
)


print(
    "\n2. ADAPTIVE GATE"
)

print(
    gate
)


# ============================================================
# 3. PERSONALIZED AUTOENCODER
#
# Currently simulated.
#
# Later:
# autoencoder_client_3.pt
# ============================================================

anomaly_score = 0.82


print(
    "\n3. PERSONALIZED AUTOENCODER"
)

print(
    {
        "anomaly_score":
            anomaly_score
    }
)


# ============================================================
# 4. FEDPROX70 MLP
#
# Existing validated global classifier.
# Currently simulated here.
# ============================================================

fedprox_result = {

    "prediction":
        "PortScan",

    "confidence":
        0.88,
}


print(
    "\n4. FEDPROX70 GLOBAL MLP"
)

print(
    fedprox_result
)


# ============================================================
# 5. CNN-BILSTM
#
# Sequence-aware classifier.
#
# Currently simulated.
#
# Later input:
#
# shape:
# (batch, sequence_length, 70)
#
# Example:
# (1, 20, 70)
# ============================================================

cnn_bilstm_result = {

    "prediction":
        "PortScan",

    "confidence":
        0.94,
}


print(
    "\n5. CNN-BILSTM"
)

print(
    cnn_bilstm_result
)


# ============================================================
# 6. RISK FUSION
# ============================================================

final_result = engine.risk_fusion.fuse(

    rf_class=
        rf_result[
            "prediction"
        ],

    rf_confidence=
        rf_result[
            "confidence"
        ],


    fedprox_class=
        fedprox_result[
            "prediction"
        ],

    fedprox_confidence=
        fedprox_result[
            "confidence"
        ],


    cnn_bilstm_class=
        cnn_bilstm_result[
            "prediction"
        ],

    cnn_bilstm_confidence=
        cnn_bilstm_result[
            "confidence"
        ],


    anomaly_score=
        anomaly_score,
)


print(
    "\n6. RISK FUSION RESULT"
)

print(
    final_result
)


# ============================================================
# 7. BUILD LOCAL IDS ALERT
# ============================================================

alert = engine.build_alert(

    client_id=
        client_id,

    final_result=
        final_result,

    rf_result=
        rf_result,

    fedprox_result=
        fedprox_result,

    cnn_bilstm_result=
        cnn_bilstm_result,

    anomaly_score=
        anomaly_score,

    processing_path=
        gate[
            "path"
        ],
)


# ============================================================
# FINAL ALERT
# ============================================================

print(
    "\n"
    +
    "=" * 90
)

print(
    "LOCAL IDS ALERT"
)

print(
    "=" * 90
)


print(
    json.dumps(
        alert,
        indent=4,
    )
)


print(
    "\n"
    +
    "=" * 90
)

print(
    "DETECTION COMPLETE"
)

print(
    "=" * 90
)