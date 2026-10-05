from __future__ import annotations

import asyncio
import io
import json
import sys
import threading
import time
from datetime import datetime, timezone
from hashlib import sha256
import hmac
import ipaddress
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import uvicorn


# ============================================================
# PROJECT ROOT / EXISTING PROJECT MODULES
# ============================================================

THIS_DIR = Path(__file__).resolve().parent

# Supports either:
#   project_root/ids_dashboard_connected_v2.py
# or:
#   project_root/scripts/ids_dashboard_connected_v2.py
if (THIS_DIR / "scripts").exists():
    PROJECT_ROOT = THIS_DIR
    SCRIPT_DIR = PROJECT_ROOT / "scripts"
else:
    PROJECT_ROOT = THIS_DIR.parent
    SCRIPT_DIR = THIS_DIR

for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import phase9_fedavg70 as phase4
    import phase14g1_real_model_wiring as phase14g1
    import phase14h_full_validation_adaptive_gate as phase14h
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "\nCould not import the existing IDS project modules.\n\n"
        "Required files inside scripts\\:\n"
        "  phase9_fedavg70.py\n"
        "  phase14g1_real_model_wiring.py\n"
        "  phase14h_full_validation_adaptive_gate.py\n\n"
        "Place this dashboard file either in the project root or scripts folder."
    ) from exc


# ============================================================
# CONFIG
# ============================================================

APP_TITLE = "Federated Collaborative IDS — Live Security Dashboard"

NUM_CLIENTS = 5
NUM_CLASSES = 6
ANOMALOUS_ID = 6
ANOMALOUS_NAME = "ANOMALOUS_TRAFFIC"

EXPECTED_FEATURES = 70
SEQUENCE_LENGTH = 20

DEFAULT_MAX_ROWS = 1000
DEFAULT_BATCH_SIZE = 2048

RESULT_ROOT = PROJECT_ROOT / "results" / "dashboard"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

VALIDATION_ALERTS_FILE = RESULT_ROOT / "validation_replay_alerts.json"
VALIDATION_SUMMARY_FILE = RESULT_ROOT / "validation_replay_summary.json"

UPLOAD_ALERTS_FILE = RESULT_ROOT / "external_file_alerts.json"
UPLOAD_SUMMARY_FILE = RESULT_ROOT / "external_file_summary.json"
UPLOAD_CSV_FILE = RESULT_ROOT / "external_file_predictions.csv"

STATE_FILE = RESULT_ROOT / "dashboard_state.json"

COMPARISON_FILE = (
    RESULT_ROOT
    / "phase17h1_comparison"
    / "phase17h1_comparison_summary.json"
)

NATURAL_ALERTS_FILE = RESULT_ROOT / "validation_natural_alerts.json"
NATURAL_SUMMARY_FILE = RESULT_ROOT / "validation_natural_summary.json"

BALANCED_ALERTS_FILE = RESULT_ROOT / "validation_balanced_alerts.json"
BALANCED_SUMMARY_FILE = RESULT_ROOT / "validation_balanced_summary.json"

DEFAULT_VALIDATION_SAMPLE_COUNT = 1000

HMAC_KEY = b"phase17h-dashboard-local-ids-v2"

# This dashboard records analyst block decisions only.
# It deliberately does NOT modify the OS/network firewall.
BLOCKED_SOURCES: set[str] = set()

# Prevent two heavy inference jobs from running simultaneously.
INFERENCE_LOCK = threading.Lock()

app = FastAPI(title=APP_TITLE)


# ============================================================
# GENERIC HELPERS
# ============================================================

def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    return value


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def write_json(path: Path, payload: Any) -> None:
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
            json_safe(payload),
            file,
            indent=2,
        )


def current_state() -> dict:
    state = read_json(
        STATE_FILE,
        {},
    )

    source = str(
        state.get(
            "source",
            "validation",
        )
    )

    mode = str(
        state.get(
            "validation_mode",
            "natural",
        )
    )

    if source == "external" and not UPLOAD_ALERTS_FILE.exists():
        source = "validation"

    if mode not in {
        "natural",
        "balanced",
    }:
        mode = "natural"

    return {
        "source": source,
        "validation_mode": mode,
    }


def current_source() -> str:
    return current_state()[
        "source"
    ]


def current_validation_mode() -> str:
    return current_state()[
        "validation_mode"
    ]


def set_dashboard_state(
    *,
    source: Optional[str] = None,
    validation_mode: Optional[str] = None,
) -> None:
    state = current_state()

    if source is not None:
        if source not in {
            "validation",
            "external",
        }:
            raise ValueError(
                "source must be validation or external"
            )

        state[
            "source"
        ] = source

    if validation_mode is not None:
        if validation_mode not in {
            "natural",
            "balanced",
        }:
            raise ValueError(
                "validation_mode must be natural or balanced"
            )

        state[
            "validation_mode"
        ] = validation_mode

    state[
        "updated_utc"
    ] = datetime.now(
        timezone.utc
    ).isoformat()

    write_json(
        STATE_FILE,
        state,
    )


def set_current_source(source: str) -> None:
    set_dashboard_state(
        source=source
    )


def active_files():
    state = current_state()
    source = state[
        "source"
    ]

    if source == "external":
        return (
            source,
            UPLOAD_ALERTS_FILE,
            UPLOAD_SUMMARY_FILE,
        )

    mode = state[
        "validation_mode"
    ]

    if mode == "balanced":
        if BALANCED_ALERTS_FILE.exists() and BALANCED_SUMMARY_FILE.exists():
            return (
                "validation_balanced",
                BALANCED_ALERTS_FILE,
                BALANCED_SUMMARY_FILE,
            )

        return (
            "validation_balanced",
            VALIDATION_ALERTS_FILE,
            VALIDATION_SUMMARY_FILE,
        )

    if NATURAL_ALERTS_FILE.exists() and NATURAL_SUMMARY_FILE.exists():
        return (
            "validation_natural",
            NATURAL_ALERTS_FILE,
            NATURAL_SUMMARY_FILE,
        )

    return (
        "validation_natural",
        VALIDATION_ALERTS_FILE,
        VALIDATION_SUMMARY_FILE,
    )


def get_alerts() -> list[dict]:
    _, path, _ = active_files()
    return read_json(
        path,
        [],
    )


def get_summary() -> dict:
    _, _, path = active_files()
    return read_json(
        path,
        {},
    )


def pseudonymize(value: str, prefix: str = "SRC") -> str:
    digest = hmac.new(
        HMAC_KEY,
        value.encode("utf-8"),
        sha256,
    ).hexdigest()

    return f"{prefix}-{digest[:12].upper()}"


def valid_ip_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()

    if not text or text.lower() in {
        "nan",
        "none",
        "null",
    }:
        return None

    try:
        return str(
            ipaddress.ip_address(
                text
            )
        )
    except ValueError:
        # Do not fail the complete external-file replay merely because
        # one row has a non-IP textual identifier.
        return None


def class_name(
    class_id: int,
    inverse_mapping: dict[int, str],
) -> str:
    class_id = int(
        class_id
    )

    if class_id == ANOMALOUS_ID:
        return ANOMALOUS_NAME

    return str(
        inverse_mapping.get(
            class_id,
            f"CLASS_{class_id}",
        )
    )


def severity_name(
    severity_id: int,
) -> str:
    return str(
        phase14h.SEVERITY_NAMES[
            int(severity_id)
        ]
    )


def find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> Optional[str]:
    lookup = {
        str(column).strip().lower():
            column
        for column in dataframe.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in lookup:
            return lookup[
                key
            ]

    return None


# ============================================================
# EXTERNAL CSV PREPROCESSING
# ============================================================

def normalize_external_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize surrounding whitespace because CICIDS-style CSV files
    often contain column names with leading spaces.
    """
    dataframe = dataframe.copy()

    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    return dataframe


def prepare_external_features(
    dataframe: pd.DataFrame,
):
    """
    Apply the SAME frozen preprocessing used by the project.

    External CSV requirement:
      It must contain the original project feature columns expected by
      the frozen scaler / active-feature selector.

    We do not silently invent missing ML features.
    """

    (
        scaler,
        feature_columns,
        label_mapping,
        active_indices,
        active_features,
    ) = phase4.load_preprocessing()

    feature_columns = [
        str(column).strip()
        for column in feature_columns
    ]

    active_features = [
        str(column).strip()
        for column in active_features
    ]

    dataframe = normalize_external_columns(
        dataframe
    )

    missing = [
        column
        for column in feature_columns
        if column not in dataframe.columns
    ]

    if missing:
        sample = ", ".join(
            missing[:12]
        )

        raise ValueError(
            "External CSV does not contain the complete raw feature schema "
            "required by the frozen preprocessing pipeline. "
            f"Missing {len(missing)} column(s). "
            f"Examples: {sample}"
        )

    X = phase4.transform_features(
        df=dataframe,
        feature_columns=feature_columns,
        scaler=scaler,
        active_indices=np.asarray(
            active_indices,
            dtype=np.int64,
        ),
    ).astype(
        np.float32,
        copy=False,
    )

    if X.ndim != 2 or X.shape[1] != EXPECTED_FEATURES:
        raise RuntimeError(
            "Frozen preprocessing did not produce the expected "
            f"(N,{EXPECTED_FEATURES}) model input. Found {X.shape}."
        )

    return (
        X,
        label_mapping,
        active_features,
    )


def build_row_ordered_sequences(
    flows: np.ndarray,
) -> np.ndarray:
    """
    Creates a 20x70 sequence for each external row.

    First rows are left-padded with the earliest available row.

    Scientific note:
      This is ROW-ORDERED replay for dashboard/integration testing.
      It must not be described as timestamp-confirmed temporal grouping.
    """

    n = len(
        flows
    )

    sequences = np.empty(
        (
            n,
            SEQUENCE_LENGTH,
            EXPECTED_FEATURES,
        ),
        dtype=np.float32,
    )

    for index in range(
        n
    ):
        start = max(
            0,
            index
            -
            SEQUENCE_LENGTH
            +
            1,
        )

        chunk = flows[
            start:index + 1
        ]

        missing = (
            SEQUENCE_LENGTH
            -
            len(chunk)
        )

        if missing > 0:
            pad = np.repeat(
                chunk[
                    0:1
                ],
                repeats=missing,
                axis=0,
            )

            sequence = np.concatenate(
                [
                    pad,
                    chunk,
                ],
                axis=0,
            )
        else:
            sequence = chunk

        sequences[
            index
        ] = sequence

    return sequences


# ============================================================
# TRUE-LABEL EXTRACTION (OPTIONAL)
# ============================================================

def extract_true_labels(
    dataframe: pd.DataFrame,
    label_mapping: dict[str, int],
) -> Optional[np.ndarray]:
    label_col = find_column(
        dataframe,
        [
            "Label",
            "label",
            str(
                getattr(
                    phase4,
                    "LABEL_COL",
                    "Label",
                )
            ),
        ],
    )

    if label_col is None:
        return None

    canonical = {
        str(name).strip().lower():
            int(class_id)
        for name, class_id
        in label_mapping.items()
    }

    values = (
        dataframe[
            label_col
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mapped = values.map(
        canonical
    )

    if mapped.isna().any():
        # External files are allowed to have labels that are outside
        # the project's six frozen classes. We simply skip metrics.
        return None

    return mapped.to_numpy(
        dtype=np.int64
    )


# ============================================================
# EXTERNAL FILE INFERENCE
# ============================================================

def analyze_external_dataframe(
    dataframe: pd.DataFrame,
    filename: str,
    max_rows: int,
    batch_size: int,
) -> tuple[list[dict], dict]:
    """
    Run the existing frozen Phase-14 inference stack on an external CSV.

    Models:
      client-local RF
      client-local personalized AE
      global FedProx70
      global CNN-BiLSTM
      frozen Phase14H vectorized fusion/adaptive gate
    """

    if dataframe.empty:
        raise ValueError(
            "Uploaded CSV is empty."
        )

    max_rows = max(
        1,
        int(max_rows),
    )

    batch_size = max(
        1,
        int(batch_size),
    )

    dataframe = dataframe.head(
        max_rows
    ).copy()

    (
        flows,
        label_mapping,
        active_features,
    ) = prepare_external_features(
        dataframe
    )

    (
        schema_label_mapping,
        inverse_mapping,
        schema_active_features,
    ) = phase14g1.load_project_schema()

    # Sanity check that the loaded schema is aligned.
    if len(
        schema_active_features
    ) != EXPECTED_FEATURES:
        raise RuntimeError(
            "Project schema does not contain exactly 70 active features."
        )

    sequences = build_row_ordered_sequences(
        flows
    )

    y_true = extract_true_labels(
        dataframe,
        label_mapping,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------------
    # Global models
    # --------------------------------------------------------
    global_start = time.perf_counter()

    fedprox70 = phase14g1.RealFedProx70Predictor(
        checkpoint_path=
            phase14g1.FEDPROX70_CHECKPOINT,
        inverse_mapping=
            inverse_mapping,
        device=
            device,
    )

    cnn = phase14g1.build_real_cnn_predictor(
        inverse_mapping=
            inverse_mapping,
        device=
            device,
    )

    (
        fed_pred,
        fed_conf,
    ) = phase14h.predict_fedprox70_batch(
        predictor=
            fedprox70,
        flows=
            flows,
        batch_size=
            batch_size,
        device=
            device,
    )

    (
        cnn_pred,
        cnn_conf,
    ) = phase14h.predict_cnn_batch(
        predictor=
            cnn,
        sequences=
            sequences,
        batch_size=
            batch_size,
        device=
            device,
    )

    global_seconds = (
        time.perf_counter()
        -
        global_start
    )

    # --------------------------------------------------------
    # Client assignment
    # --------------------------------------------------------
    explicit_client_col = find_column(
        dataframe,
        [
            "client_id",
            "client id",
            "client",
        ],
    )

    if explicit_client_col is not None:
        parsed = pd.to_numeric(
            dataframe[
                explicit_client_col
            ],
            errors="coerce",
        )

        valid = (
            parsed.notna()
            &
            parsed.between(
                1,
                NUM_CLIENTS,
            )
        )

        if valid.all():
            assigned_clients = (
                parsed.astype(
                    np.int64
                )
                .to_numpy()
            )

            assignment_policy = (
                "client_id column from uploaded file"
            )
        else:
            assigned_clients = (
                np.arange(
                    len(dataframe)
                )
                %
                NUM_CLIENTS
            ) + 1

            assignment_policy = (
                "round-robin fallback because client_id column "
                "contained invalid/missing values"
            )
    else:
        assigned_clients = (
            np.arange(
                len(dataframe)
            )
            %
            NUM_CLIENTS
        ) + 1

        assignment_policy = (
            "deterministic round-robin; true label not used"
        )

    # --------------------------------------------------------
    # Allocate outputs
    # --------------------------------------------------------
    n = len(
        dataframe
    )

    adaptive_pred = np.empty(
        n,
        dtype=np.int64,
    )

    adaptive_conf = np.empty(
        n,
        dtype=np.float32,
    )

    adaptive_risk = np.empty(
        n,
        dtype=np.float32,
    )

    adaptive_severity = np.empty(
        n,
        dtype=np.int8,
    )

    processing_path = np.empty(
        n,
        dtype=object,
    )

    decision_source = np.empty(
        n,
        dtype=object,
    )

    rf_pred_all = np.empty(
        n,
        dtype=np.int64,
    )

    rf_conf_all = np.empty(
        n,
        dtype=np.float32,
    )

    ae_error_all = np.empty(
        n,
        dtype=np.float32,
    )

    ae_score_all = np.empty(
        n,
        dtype=np.float32,
    )

    ae_anomaly_all = np.empty(
        n,
        dtype=bool,
    )

    local_seconds_total = 0.0

    # --------------------------------------------------------
    # Local models per assigned client
    # --------------------------------------------------------
    for client_id in range(
        1,
        NUM_CLIENTS + 1,
    ):
        positions = np.flatnonzero(
            assigned_clients
            ==
            client_id
        )

        if len(
            positions
        ) == 0:
            continue

        rf = phase14g1.RealRandomForestPredictor(
            client_id=
                client_id,
            label_mapping=
                schema_label_mapping,
            inverse_mapping=
                inverse_mapping,
        )

        ae = phase14g1.RealPersonalizedAutoencoderPredictor(
            client_id=
                client_id,
            device=
                device,
            explicit_model_path=
                None,
        )

        client_flows = flows[
            positions
        ]

        local_start = time.perf_counter()

        (
            rf_pred,
            rf_conf,
        ) = phase14h.predict_rf_batch(
            rf=
                rf,
            flows=
                client_flows,
        )

        (
            ae_error,
            ae_score,
            ae_anomaly,
        ) = phase14h.predict_ae_batch(
            ae=
                ae,
            flows=
                client_flows,
            batch_size=
                batch_size,
            device=
                device,
        )

        local_seconds_total += (
            time.perf_counter()
            -
            local_start
        )

        full_result = phase14h.full_fusion_vectorized(
            rf_pred=
                rf_pred,
            rf_conf=
                rf_conf,
            ae_score=
                ae_score,
            fed_pred=
                fed_pred[
                    positions
                ],
            fed_conf=
                fed_conf[
                    positions
                ],
            cnn_pred=
                cnn_pred[
                    positions
                ],
            cnn_conf=
                cnn_conf[
                    positions
                ],
        )

        adaptive_result = phase14h.adaptive_policy_vectorized(
            rf_pred=
                rf_pred,
            rf_conf=
                rf_conf,
            full_result=
                full_result,
        )

        rf_pred_all[
            positions
        ] = rf_pred

        rf_conf_all[
            positions
        ] = rf_conf

        ae_error_all[
            positions
        ] = ae_error

        ae_score_all[
            positions
        ] = ae_score

        ae_anomaly_all[
            positions
        ] = ae_anomaly

        adaptive_pred[
            positions
        ] = adaptive_result[
            "prediction"
        ]

        adaptive_conf[
            positions
        ] = adaptive_result[
            "confidence"
        ]

        adaptive_risk[
            positions
        ] = adaptive_result[
            "risk"
        ]

        adaptive_severity[
            positions
        ] = adaptive_result[
            "severity_id"
        ]

        fast_mask = adaptive_result[
            "fast_mask"
        ]

        processing_path[
            positions
        ] = np.where(
            fast_mask,
            "FAST_ATTACK_PATH",
            "DEEP_ANALYSIS",
        )

        decision_source[
            positions
        ] = np.where(
            fast_mask,
            "Random Forest",
            full_result[
                "decision_source"
            ],
        )

    # --------------------------------------------------------
    # Optional metadata columns
    # --------------------------------------------------------
    source_ip_col = find_column(
        dataframe,
        [
            "Source IP",
            "Src IP",
            "src_ip",
            "source_ip",
            "SourceIP",
        ],
    )

    destination_ip_col = find_column(
        dataframe,
        [
            "Destination IP",
            "Dst IP",
            "dst_ip",
            "destination_ip",
            "DestinationIP",
        ],
    )

    timestamp_col = find_column(
        dataframe,
        [
            "Timestamp",
            "timestamp",
            "event_time",
            "Event Time",
            "time",
        ],
    )

    # --------------------------------------------------------
    # Build alert/event rows
    # --------------------------------------------------------
    alerts: list[dict] = []

    for i in range(
        n
    ):
        pred_id = int(
            adaptive_pred[
                i
            ]
        )

        risk_value = adaptive_risk[
            i
        ]

        risk_score = (
            None
            if np.isnan(
                risk_value
            )
            else float(
                risk_value
            )
        )

        raw_source = (
            dataframe.iloc[
                i
            ][
                source_ip_col
            ]
            if source_ip_col
            is not None
            else None
        )

        raw_destination = (
            dataframe.iloc[
                i
            ][
                destination_ip_col
            ]
            if destination_ip_col
            is not None
            else None
        )

        source_ip = valid_ip_or_none(
            raw_source
        )

        destination_ip = valid_ip_or_none(
            raw_destination
        )

        if source_ip:
            source_id = pseudonymize(
                source_ip,
                "SRC",
            )
        else:
            source_id = pseudonymize(
                f"{filename}:{i}",
                "ROW",
            )

        if destination_ip:
            destination_id = pseudonymize(
                destination_ip,
                "DST",
            )
        else:
            destination_id = None

        if timestamp_col is not None:
            raw_time = str(
                dataframe.iloc[
                    i
                ][
                    timestamp_col
                ]
            )

            event_time = raw_time
        else:
            event_time = (
                datetime.now(
                    timezone.utc
                )
                .isoformat()
            )

        if y_true is not None:
            true_id = int(
                y_true[
                    i
                ]
            )

            true_class = class_name(
                true_id,
                inverse_mapping,
            )

            correct = bool(
                pred_id
                ==
                true_id
            )
        else:
            true_id = None
            true_class = None
            correct = None

        alerts.append(
            {
                "id":
                    int(
                        i + 1
                    ),

                "replay_type":
                    "EXTERNAL_FILE",

                "source_file":
                    filename,

                "row_index":
                    int(
                        i
                    ),

                "event_time":
                    event_time,

                "client_id":
                    int(
                        assigned_clients[
                            i
                        ]
                    ),

                # Raw IPs remain local to this dashboard process.
                "source_ip":
                    source_ip,

                "destination_ip":
                    destination_ip,

                "source_id":
                    source_id,

                "destination_id":
                    destination_id,

                "true_class_id":
                    true_id,

                "true_class":
                    true_class,

                "prediction_id":
                    pred_id,

                "attack_type":
                    class_name(
                        pred_id,
                        inverse_mapping,
                    ),

                "confidence":
                    float(
                        adaptive_conf[
                            i
                        ]
                    ),

                "risk_score":
                    risk_score,

                "anomaly_score":
                    float(
                        ae_score_all[
                            i
                        ]
                    ),

                "raw_reconstruction_error":
                    float(
                        ae_error_all[
                            i
                        ]
                    ),

                "ae_anomaly":
                    bool(
                        ae_anomaly_all[
                            i
                        ]
                    ),

                "severity":
                    severity_name(
                        int(
                            adaptive_severity[
                                i
                            ]
                        )
                    ),

                "processing_path":
                    str(
                        processing_path[
                            i
                        ]
                    ),

                "decision_source":
                    str(
                        decision_source[
                            i
                        ]
                    ),

                "rf_prediction":
                    class_name(
                        int(
                            rf_pred_all[
                                i
                            ]
                        ),
                        inverse_mapping,
                    ),

                "rf_confidence":
                    float(
                        rf_conf_all[
                            i
                        ]
                    ),

                "fedprox70_prediction":
                    class_name(
                        int(
                            fed_pred[
                                i
                            ]
                        ),
                        inverse_mapping,
                    ),

                "fedprox70_confidence":
                    float(
                        fed_conf[
                            i
                        ]
                    ),

                "cnn_bilstm_prediction":
                    class_name(
                        int(
                            cnn_pred[
                                i
                            ]
                        ),
                        inverse_mapping,
                    ),

                "cnn_bilstm_confidence":
                    float(
                        cnn_conf[
                            i
                        ]
                    ),

                "correct":
                    correct,

                "correlated":
                    False,

                "status":
                    "ACTIVE",

                "blockable":
                    bool(
                        source_ip
                    ),

                "locked_test_used":
                    False,
            }
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    normal_count = int(
        np.sum(
            adaptive_pred
            ==
            0
        )
    )

    anomalous_count = int(
        np.sum(
            adaptive_pred
            ==
            ANOMALOUS_ID
        )
    )

    attack_count = int(
        n
        -
        normal_count
    )

    attack_rate = (
        attack_count
        /
        n
    )

    anomalous_rate = (
        anomalous_count
        /
        n
    )

    fast_count = int(
        np.sum(
            processing_path
            ==
            "FAST_ATTACK_PATH"
        )
    )

    deep_count = int(
        np.sum(
            processing_path
            ==
            "DEEP_ANALYSIS"
        )
    )

    prediction_distribution = {}

    for pred_id in np.unique(
        adaptive_pred
    ):
        prediction_distribution[
            class_name(
                int(
                    pred_id
                ),
                inverse_mapping,
            )
        ] = int(
            np.sum(
                adaptive_pred
                ==
                pred_id
            )
        )

    metrics = None

    if y_true is not None:
        known_labels = [
            0,
            1,
            2,
            3,
            4,
            5,
        ]

        macro_precision = float(
            precision_score(
                y_true,
                adaptive_pred,
                labels=
                    known_labels,
                average=
                    "macro",
                zero_division=
                    0,
            )
        )

        macro_recall = float(
            recall_score(
                y_true,
                adaptive_pred,
                labels=
                    known_labels,
                average=
                    "macro",
                zero_division=
                    0,
            )
        )

        metrics = {
            "accuracy":
                float(
                    accuracy_score(
                        y_true,
                        adaptive_pred,
                    )
                ),

            "balanced_accuracy":
                macro_recall,

            "macro_precision":
                macro_precision,

            "macro_recall":
                macro_recall,

            "macro_f1":
                float(
                    f1_score(
                        y_true,
                        adaptive_pred,
                        labels=
                            known_labels,
                        average=
                            "macro",
                        zero_division=
                            0,
                    )
                ),

            "weighted_f1":
                float(
                    f1_score(
                        y_true,
                        adaptive_pred,
                        labels=
                            known_labels,
                        average=
                            "weighted",
                        zero_division=
                            0,
                    )
                ),
        }

    summary = {
        "source":
            "external",

        "source_file":
            filename,

        "sample_count":
            n,

        "total_flows":
            n,

        "normal_traffic":
            normal_count,

        "attack_alerts":
            attack_count,

        "attack_rate":
            float(
                attack_rate
            ),

        "anomalous_predictions":
            anomalous_count,

        "anomalous_rate":
            float(
                anomalous_rate
            ),

        "fast_path_count":
            fast_count,

        "deep_path_count":
            deep_count,

        "prediction_distribution":
            prediction_distribution,

        "metrics":
            metrics,

        "client_assignment_policy":
            assignment_policy,

        "sequence_policy":
            (
                "20x70 row-ordered rolling sequences with left padding; "
                "integration/live-replay mode, not timestamp-confirmed "
                "temporal grouping"
            ),

        "raw_source_ip_column":
            source_ip_col,

        "raw_destination_ip_column":
            destination_ip_col,

        "timestamp_column":
            timestamp_col,

        "raw_ip_available":
            bool(
                source_ip_col
            ),

        "global_inference_seconds":
            float(
                global_seconds
            ),

        "local_inference_seconds_total":
            float(
                local_seconds_total
            ),

        "locked_test_used":
            False,

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }

    return (
        alerts,
        summary,
    )



# ============================================================
# VALIDATION MODE INFERENCE
# ============================================================

def _validation_indices(
    y: np.ndarray,
    requested: int,
    mode: str,
) -> np.ndarray:
    requested = min(
        max(
            1,
            int(requested),
        ),
        len(y),
    )

    rng = np.random.default_rng(
        42
    )

    if mode == "natural":
        selected = rng.choice(
            np.arange(
                len(y),
                dtype=np.int64,
            ),
            size=requested,
            replace=False,
        )

        return np.asarray(
            sorted(
                selected.tolist()
            ),
            dtype=np.int64,
        )

    if mode != "balanced":
        raise ValueError(
            "mode must be natural or balanced"
        )

    selected: list[int] = []

    per_class = max(
        1,
        requested
        //
        NUM_CLASSES,
    )

    for class_id in range(
        NUM_CLASSES
    ):
        class_indices = np.flatnonzero(
            y
            ==
            class_id
        )

        take = min(
            per_class,
            len(
                class_indices
            ),
        )

        if take > 0:
            sample = rng.choice(
                class_indices,
                size=take,
                replace=False,
            )

            selected.extend(
                sample.tolist()
            )

    if len(
        selected
    ) < requested:
        already = set(
            selected
        )

        remaining = np.asarray(
            [
                index
                for index
                in range(
                    len(y)
                )
                if index not in already
            ],
            dtype=np.int64,
        )

        extra_count = min(
            requested
            -
            len(
                selected
            ),
            len(
                remaining
            ),
        )

        if extra_count > 0:
            extra = rng.choice(
                remaining,
                size=extra_count,
                replace=False,
            )

            selected.extend(
                extra.tolist()
            )

    return np.asarray(
        sorted(
            selected[
                :requested
            ]
        ),
        dtype=np.int64,
    )


def build_validation_mode_results(
    mode: str,
    sample_count: int = DEFAULT_VALIDATION_SAMPLE_COUNT,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[list[dict], dict]:
    """
    Generate one validation dashboard mode using the frozen Phase-14 stack.

    NATURAL:
      uniform random validation sample, preserving the natural distribution.

    BALANCED:
      approximately equal sample counts from all six known classes.

    Models are INFERENCE ONLY. No model is retrained or updated.
    """

    validation_file = phase14h.VALIDATION_SEQUENCE_FILE

    if not validation_file.exists():
        raise FileNotFoundError(
            f"Missing validation sequence file: {validation_file}"
        )

    with np.load(
        validation_file,
        allow_pickle=False,
    ) as data:
        X_all = data[
            "X"
        ].astype(
            np.float32,
            copy=False,
        )

        y_all = data[
            "y"
        ].astype(
            np.int64,
            copy=False,
        )

    indices = _validation_indices(
        y=y_all,
        requested=sample_count,
        mode=mode,
    )

    sequences = X_all[
        indices
    ]

    y_true = y_all[
        indices
    ]

    flows = np.asarray(
        sequences[
            :,
            -1,
            :,
        ],
        dtype=np.float32,
    )

    (
        label_mapping,
        inverse_mapping,
        active_features,
    ) = phase14g1.load_project_schema()

    if len(
        active_features
    ) != EXPECTED_FEATURES:
        raise RuntimeError(
            "Project schema does not contain exactly 70 active features."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    start = time.perf_counter()

    fedprox70 = phase14g1.RealFedProx70Predictor(
        checkpoint_path=
            phase14g1.FEDPROX70_CHECKPOINT,
        inverse_mapping=
            inverse_mapping,
        device=
            device,
    )

    cnn = phase14g1.build_real_cnn_predictor(
        inverse_mapping=
            inverse_mapping,
        device=
            device,
    )

    (
        fed_pred,
        fed_conf,
    ) = phase14h.predict_fedprox70_batch(
        predictor=
            fedprox70,
        flows=
            flows,
        batch_size=
            batch_size,
        device=
            device,
    )

    (
        cnn_pred,
        cnn_conf,
    ) = phase14h.predict_cnn_batch(
        predictor=
            cnn,
        sequences=
            sequences,
        batch_size=
            batch_size,
        device=
            device,
    )

    n = len(
        y_true
    )

    assigned_clients = (
        np.arange(
            n
        )
        %
        NUM_CLIENTS
    ) + 1

    adaptive_pred = np.empty(
        n,
        dtype=np.int64,
    )

    adaptive_conf = np.empty(
        n,
        dtype=np.float32,
    )

    adaptive_risk = np.empty(
        n,
        dtype=np.float32,
    )

    adaptive_severity = np.empty(
        n,
        dtype=np.int8,
    )

    processing_path = np.empty(
        n,
        dtype=object,
    )

    decision_source = np.empty(
        n,
        dtype=object,
    )

    rf_pred_all = np.empty(
        n,
        dtype=np.int64,
    )

    rf_conf_all = np.empty(
        n,
        dtype=np.float32,
    )

    ae_error_all = np.empty(
        n,
        dtype=np.float32,
    )

    ae_score_all = np.empty(
        n,
        dtype=np.float32,
    )

    ae_anomaly_all = np.empty(
        n,
        dtype=bool,
    )

    for client_id in range(
        1,
        NUM_CLIENTS + 1,
    ):
        positions = np.flatnonzero(
            assigned_clients
            ==
            client_id
        )

        if len(
            positions
        ) == 0:
            continue

        rf = phase14g1.RealRandomForestPredictor(
            client_id=
                client_id,
            label_mapping=
                label_mapping,
            inverse_mapping=
                inverse_mapping,
        )

        ae = phase14g1.RealPersonalizedAutoencoderPredictor(
            client_id=
                client_id,
            device=
                device,
            explicit_model_path=
                None,
        )

        (
            rf_pred,
            rf_conf,
        ) = phase14h.predict_rf_batch(
            rf=
                rf,
            flows=
                flows[
                    positions
                ],
        )

        (
            ae_error,
            ae_score,
            ae_anomaly,
        ) = phase14h.predict_ae_batch(
            ae=
                ae,
            flows=
                flows[
                    positions
                ],
            batch_size=
                batch_size,
            device=
                device,
        )

        full_result = phase14h.full_fusion_vectorized(
            rf_pred=
                rf_pred,
            rf_conf=
                rf_conf,
            ae_score=
                ae_score,
            fed_pred=
                fed_pred[
                    positions
                ],
            fed_conf=
                fed_conf[
                    positions
                ],
            cnn_pred=
                cnn_pred[
                    positions
                ],
            cnn_conf=
                cnn_conf[
                    positions
                ],
        )

        adaptive_result = phase14h.adaptive_policy_vectorized(
            rf_pred=
                rf_pred,
            rf_conf=
                rf_conf,
            full_result=
                full_result,
        )

        rf_pred_all[
            positions
        ] = rf_pred

        rf_conf_all[
            positions
        ] = rf_conf

        ae_error_all[
            positions
        ] = ae_error

        ae_score_all[
            positions
        ] = ae_score

        ae_anomaly_all[
            positions
        ] = ae_anomaly

        adaptive_pred[
            positions
        ] = adaptive_result[
            "prediction"
        ]

        adaptive_conf[
            positions
        ] = adaptive_result[
            "confidence"
        ]

        adaptive_risk[
            positions
        ] = adaptive_result[
            "risk"
        ]

        adaptive_severity[
            positions
        ] = adaptive_result[
            "severity_id"
        ]

        fast_mask = adaptive_result[
            "fast_mask"
        ]

        processing_path[
            positions
        ] = np.where(
            fast_mask,
            "FAST_ATTACK_PATH",
            "DEEP_ANALYSIS",
        )

        decision_source[
            positions
        ] = np.where(
            fast_mask,
            "Random Forest",
            full_result[
                "decision_source"
            ],
        )

    known_labels = [
        0,
        1,
        2,
        3,
        4,
        5,
    ]

    macro_precision = float(
        precision_score(
            y_true,
            adaptive_pred,
            labels=
                known_labels,
            average=
                "macro",
            zero_division=
                0,
        )
    )

    macro_recall = float(
        recall_score(
            y_true,
            adaptive_pred,
            labels=
                known_labels,
            average=
                "macro",
            zero_division=
                0,
        )
    )

    overall_metrics = {
        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    adaptive_pred,
                )
            ),

        "balanced_accuracy":
            macro_recall,

        "macro_precision":
            macro_precision,

        "macro_recall":
            macro_recall,

        "macro_f1":
            float(
                f1_score(
                    y_true,
                    adaptive_pred,
                    labels=
                        known_labels,
                    average=
                        "macro",
                    zero_division=
                        0,
                )
            ),

        "weighted_f1":
            float(
                f1_score(
                    y_true,
                    adaptive_pred,
                    labels=
                        known_labels,
                    average=
                        "weighted",
                    zero_division=
                        0,
                )
            ),
    }

    def classifier_metrics(
        prediction: np.ndarray,
    ) -> dict:
        recall = float(
            recall_score(
                y_true,
                prediction,
                labels=
                    known_labels,
                average=
                    "macro",
                zero_division=
                    0,
            )
        )

        return {
            "accuracy":
                float(
                    accuracy_score(
                        y_true,
                        prediction,
                    )
                ),

            "balanced_accuracy":
                recall,

            "macro_f1":
                float(
                    f1_score(
                        y_true,
                        prediction,
                        labels=
                            known_labels,
                        average=
                            "macro",
                        zero_division=
                            0,
                    )
                ),
        }

    model_metrics = {
        "random_forest":
            classifier_metrics(
                rf_pred_all
            ),

        "fedprox70":
            classifier_metrics(
                fed_pred
            ),

        "cnn_bilstm":
            classifier_metrics(
                cnn_pred
            ),

        "autoencoder":
            {
                "anomaly_count":
                    int(
                        np.sum(
                            ae_anomaly_all
                        )
                    ),

                "anomaly_rate":
                    float(
                        np.mean(
                            ae_anomaly_all
                        )
                    ),

                "mean_anomaly_score":
                    float(
                        np.mean(
                            ae_score_all
                        )
                    ),
            },
    }

    alerts: list[dict] = []

    now = datetime.now(
        timezone.utc
    ).isoformat()

    for i in range(
        n
    ):
        pred_id = int(
            adaptive_pred[
                i
            ]
        )

        true_id = int(
            y_true[
                i
            ]
        )

        risk_value = adaptive_risk[
            i
        ]

        risk_score = (
            None
            if np.isnan(
                risk_value
            )
            else float(
                risk_value
            )
        )

        source_id = pseudonymize(
            f"{mode}:{int(indices[i])}",
            "VAL",
        )

        alerts.append(
            {
                "id":
                    int(
                        i + 1
                    ),

                "replay_type":
                    (
                        "VALIDATION_NATURAL"
                        if mode
                        ==
                        "natural"
                        else
                        "VALIDATION_BALANCED"
                    ),

                "validation_mode":
                    mode,

                "validation_index":
                    int(
                        indices[
                            i
                        ]
                    ),

                "event_time":
                    now,

                "client_id":
                    int(
                        assigned_clients[
                            i
                        ]
                    ),

                "source_id":
                    source_id,

                "source_ip":
                    None,

                "destination_ip":
                    None,

                "true_class_id":
                    true_id,

                "true_class":
                    class_name(
                        true_id,
                        inverse_mapping,
                    ),

                "prediction_id":
                    pred_id,

                "attack_type":
                    class_name(
                        pred_id,
                        inverse_mapping,
                    ),

                "confidence":
                    float(
                        adaptive_conf[
                            i
                        ]
                    ),

                "risk_score":
                    risk_score,

                "anomaly_score":
                    float(
                        ae_score_all[
                            i
                        ]
                    ),

                "raw_reconstruction_error":
                    float(
                        ae_error_all[
                            i
                        ]
                    ),

                "ae_anomaly":
                    bool(
                        ae_anomaly_all[
                            i
                        ]
                    ),

                "severity":
                    severity_name(
                        int(
                            adaptive_severity[
                                i
                            ]
                        )
                    ),

                "processing_path":
                    str(
                        processing_path[
                            i
                        ]
                    ),

                "decision_source":
                    str(
                        decision_source[
                            i
                        ]
                    ),

                "rf_prediction":
                    class_name(
                        int(
                            rf_pred_all[
                                i
                            ]
                        ),
                        inverse_mapping,
                    ),

                "rf_confidence":
                    float(
                        rf_conf_all[
                            i
                        ]
                    ),

                "fedprox70_prediction":
                    class_name(
                        int(
                            fed_pred[
                                i
                            ]
                        ),
                        inverse_mapping,
                    ),

                "fedprox70_confidence":
                    float(
                        fed_conf[
                            i
                        ]
                    ),

                "cnn_bilstm_prediction":
                    class_name(
                        int(
                            cnn_pred[
                                i
                            ]
                        ),
                        inverse_mapping,
                    ),

                "cnn_bilstm_confidence":
                    float(
                        cnn_conf[
                            i
                        ]
                    ),

                "correct":
                    bool(
                        pred_id
                        ==
                        true_id
                    ),

                "correlated":
                    False,

                "status":
                    "ACTIVE",

                "blockable":
                    False,

                "locked_test_used":
                    False,
            }
        )

    anomalous_count = int(
        np.sum(
            adaptive_pred
            ==
            ANOMALOUS_ID
        )
    )

    normal_count = int(
        np.sum(
            adaptive_pred
            ==
            0
        )
    )

    attack_count = int(
        n
        -
        normal_count
    )

    summary = {
        "source":
            "validation",

        "validation_mode":
            mode,

        "sample_count":
            n,

        "total_flows":
            n,

        "normal_traffic":
            normal_count,

        "attack_alerts":
            attack_count,

        "attack_rate":
            float(
                attack_count
                /
                max(
                    n,
                    1,
                )
            ),

        "anomalous_predictions":
            anomalous_count,

        "anomalous_rate":
            float(
                anomalous_count
                /
                max(
                    n,
                    1,
                )
            ),

        "fast_path_count":
            int(
                np.sum(
                    processing_path
                    ==
                    "FAST_ATTACK_PATH"
                )
            ),

        "deep_path_count":
            int(
                np.sum(
                    processing_path
                    ==
                    "DEEP_ANALYSIS"
                )
            ),

        "metrics":
            overall_metrics,

        "model_metrics":
            model_metrics,

        "sampling_policy":
            (
                "uniform random validation sampling"
                if mode
                ==
                "natural"
                else
                "approximately equal six-class validation sampling"
            ),

        "client_assignment_policy":
            (
                "deterministic round-robin; true label not used"
            ),

        "inference_seconds":
            float(
                time.perf_counter()
                -
                start
            ),

        "locked_test_used":
            False,

        "models_retrained":
            False,

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }

    return (
        alerts,
        summary,
    )


def ensure_validation_mode_cache(
    mode: str,
    sample_count: int = DEFAULT_VALIDATION_SAMPLE_COUNT,
) -> tuple[Path, Path]:
    if mode == "natural":
        alert_file = NATURAL_ALERTS_FILE
        summary_file = NATURAL_SUMMARY_FILE
    elif mode == "balanced":
        alert_file = BALANCED_ALERTS_FILE
        summary_file = BALANCED_SUMMARY_FILE
    else:
        raise ValueError(
            "mode must be natural or balanced"
        )

    regenerate = True

    if alert_file.exists() and summary_file.exists():
        cached_summary = read_json(
            summary_file,
            {},
        )

        regenerate = (
            int(
                cached_summary.get(
                    "sample_count",
                    -1,
                )
            )
            !=
            int(
                sample_count
            )
        )

    if regenerate:
        alerts, summary = build_validation_mode_results(
            mode=mode,
            sample_count=sample_count,
            batch_size=DEFAULT_BATCH_SIZE,
        )

        write_json(
            alert_file,
            alerts,
        )

        write_json(
            summary_file,
            summary,
        )

    return (
        alert_file,
        summary_file,
    )


# ============================================================
# DERIVED GRAPH DATA
# ============================================================

def graph_data(
    alerts: list[dict],
) -> dict:
    class_counts: dict[str, int] = {}

    client_attack_counts = {
        str(i): 0
        for i in range(
            1,
            NUM_CLIENTS + 1,
        )
    }

    for alert in alerts:
        name = str(
            alert.get(
                "attack_type",
                "UNKNOWN",
            )
        )

        class_counts[
            name
        ] = (
            class_counts.get(
                name,
                0,
            )
            +
            1
        )

        if name != "BENIGN":
            client = str(
                alert.get(
                    "client_id",
                    "?",
                )
            )

            if client in client_attack_counts:
                client_attack_counts[
                    client
                ] += 1

    # Attack timeline in fixed row buckets.
    n = len(
        alerts
    )

    bucket_size = max(
        1,
        int(
            np.ceil(
                max(
                    n,
                    1,
                )
                /
                30.0
            )
        ),
    )

    timeline = []

    for start in range(
        0,
        n,
        bucket_size,
    ):
        chunk = alerts[
            start:
            start
            +
            bucket_size
        ]

        attacks = sum(
            1
            for alert
            in chunk
            if str(
                alert.get(
                    "attack_type",
                    "BENIGN",
                )
            )
            !=
            "BENIGN"
        )

        anomalous = sum(
            1
            for alert
            in chunk
            if str(
                alert.get(
                    "attack_type",
                    "",
                )
            )
            ==
            ANOMALOUS_NAME
        )

        timeline.append(
            {
                "start":
                    start,

                "end":
                    min(
                        start
                        +
                        bucket_size,
                        n,
                    ),

                "attacks":
                    attacks,

                "anomalous":
                    anomalous,
            }
        )

    return {
        "class_distribution":
            class_counts,

        "client_attack_distribution":
            client_attack_counts,

        "attack_timeline":
            timeline,

        "bucket_size":
            bucket_size,
    }


# ============================================================
# API
# ============================================================

@app.get(
    "/api/alerts"
)
def api_alerts(
    limit: int = Query(
        5000,
        ge=1,
        le=100000,
    )
):
    alerts = get_alerts()

    # newest/highest row first for the table
    return list(
        reversed(
            alerts[
                -limit:
            ]
        )
    )


@app.get(
    "/api/summary"
)
def api_summary():
    state = current_state()
    summary = get_summary()

    return {
        **summary,

        "dashboard_source":
            state[
                "source"
            ],

        "validation_mode":
            state[
                "validation_mode"
            ],
    }


@app.get(
    "/api/graphs"
)
def api_graphs():
    alerts = get_alerts()
    return graph_data(
        alerts
    )


@app.get(
    "/api/comparison"
)
def api_comparison():
    """
    Phase 17H.1 diagnostic comparison.

    The file is produced by:
      scripts/phase17h1_sampling_client_assignment_comparison.py

    Natural-validation metrics are shown as the representative
    dashboard-validation reference, while balanced sampling is
    explicitly labelled as a class-balanced stress test.
    """
    if not COMPARISON_FILE.exists():
        return {
            "available": False,
            "comparison_file": str(COMPARISON_FILE),
            "phase17h_frozen": False,
            "message": (
                "Run phase17h1_sampling_client_assignment_comparison.py "
                "to populate Natural vs Balanced comparison metrics."
            ),
        }

    payload = read_json(
        COMPARISON_FILE,
        {},
    )

    experiments = payload.get(
        "experiments",
        [],
    )

    by_name = {
        str(item.get("experiment")): item
        for item in experiments
    }

    return {
        "available": True,
        "phase17h_frozen": True,
        "comparison_file": str(COMPARISON_FILE),
        "locked_test_used": False,
        "recommended_reference": "B_natural_round_robin",
        "balanced_stress_test": by_name.get(
            "A_balanced_round_robin"
        ),
        "natural_validation": by_name.get(
            "B_natural_round_robin"
        ),
        "phase14h_style": by_name.get(
            "C_natural_phase14h_all_clients"
        ),
        "raw": payload,
    }


@app.post(
    "/api/validation-mode/{mode}"
)
def choose_validation_mode(
    mode: str,
    samples: int = Query(
        DEFAULT_VALIDATION_SAMPLE_COUNT,
        ge=1,
        le=50000,
    ),
):
    if mode not in {
        "natural",
        "balanced",
    }:
        raise HTTPException(
            status_code=400,
            detail=(
                "mode must be natural or balanced"
            ),
        )

    if not INFERENCE_LOCK.acquire(
        blocking=False
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "Another inference job is currently running."
            ),
        )

    try:
        try:
            alert_file, summary_file = ensure_validation_mode_cache(
                mode=
                    mode,
                sample_count=
                    samples,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=str(
                    exc
                ),
            ) from exc

        set_dashboard_state(
            source=
                "validation",
            validation_mode=
                mode,
        )

        summary = read_json(
            summary_file,
            {},
        )

        return {
            "message":
                (
                    "Natural/random validation mode selected."
                    if mode
                    ==
                    "natural"
                    else
                    "Class-balanced validation stress-test mode selected."
                ),

            "mode":
                mode,

            "sample_count":
                summary.get(
                    "sample_count"
                ),

            "metrics":
                summary.get(
                    "metrics"
                ),

            "model_metrics":
                summary.get(
                    "model_metrics"
                ),

            "alerts_file":
                str(
                    alert_file
                ),

            "models_retrained":
                False,

            "locked_test_used":
                False,
        }

    finally:
        INFERENCE_LOCK.release()


@app.post(
    "/api/source/{source}"
)
def choose_source(
    source: str,
):
    if source not in {
        "validation",
        "external",
    }:
        raise HTTPException(
            status_code=400,
            detail=(
                "source must be validation or external"
            ),
        )

    if (
        source == "external"
        and
        not UPLOAD_ALERTS_FILE.exists()
    ):
        raise HTTPException(
            status_code=404,
            detail=(
                "No external-file analysis exists yet."
            ),
        )

    set_dashboard_state(
        source=source
    )

    return {
        "source":
            source,
    }


@app.post(
    "/api/upload-analyze"
)
async def upload_analyze(
    file: UploadFile = File(...),
    max_rows: int = Query(
        DEFAULT_MAX_ROWS,
        ge=1,
        le=100000,
    ),
    batch_size: int = Query(
        DEFAULT_BATCH_SIZE,
        ge=1,
        le=65536,
    ),
):
    filename = (
        file.filename
        or
        "uploaded.csv"
    )

    if not filename.lower().endswith(
        ".csv"
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file.",
        )

    content = await file.read()

    try:
        dataframe = pd.read_csv(
            io.BytesIO(
                content
            )
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not read CSV: {exc}"
            ),
        ) from exc

    if not INFERENCE_LOCK.acquire(
        blocking=False
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "Another inference job is currently running."
            ),
        )

    try:
        try:
            alerts, summary = await asyncio.to_thread(
                analyze_external_dataframe,
                dataframe,
                filename,
                max_rows,
                batch_size,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=str(
                    exc
                ),
            ) from exc

        write_json(
            UPLOAD_ALERTS_FILE,
            alerts,
        )

        write_json(
            UPLOAD_SUMMARY_FILE,
            summary,
        )

        pd.DataFrame(
            alerts
        ).to_csv(
            UPLOAD_CSV_FILE,
            index=False,
        )

        set_dashboard_state(
            source="external"
        )

        return {
            "message":
                "External CSV inference completed.",

            "source_file":
                filename,

            "rows":
                len(
                    alerts
                ),

            "attack_alerts":
                summary[
                    "attack_alerts"
                ],

            "anomalous_predictions":
                summary[
                    "anomalous_predictions"
                ],

            "locked_test_used":
                False,
        }

    finally:
        INFERENCE_LOCK.release()


@app.post(
    "/api/alerts/{alert_id}/block"
)
def block_source(
    alert_id: int,
):
    alerts = get_alerts()

    alert = next(
        (
            item
            for item
            in alerts
            if int(
                item.get(
                    "id",
                    -1,
                )
            )
            ==
            int(
                alert_id
            )
        ),
        None,
    )

    if alert is None:
        raise HTTPException(
            status_code=404,
            detail="Alert not found.",
        )

    if not alert.get(
        "blockable",
        False,
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "This record has no locally available raw source IP. "
                "It cannot be firewall-blocked from the dashboard."
            ),
        )

    source_ip = alert.get(
        "source_ip"
    )

    if not source_ip:
        raise HTTPException(
            status_code=409,
            detail="Raw source IP is unavailable.",
        )

    BLOCKED_SOURCES.add(
        str(
            source_ip
        )
    )

    return {
        "message":
            (
                "Source added to the dashboard block list. "
                "The OS/network firewall was NOT modified."
            ),

        "source_ip":
            source_ip,

        "firewall_changed":
            False,
    }


@app.get(
    "/api/blocked"
)
def blocked_sources():
    return sorted(
        BLOCKED_SOURCES
    )


# ============================================================
# FRONTEND
# ============================================================

HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Federated Collaborative IDS</title>

<style>
* {
    box-sizing: border-box;
}

:root {
    --bg: #08111f;
    --panel: #0e1b2f;
    --panel2: #101f35;
    --line: #243853;
    --text: #e8eef7;
    --muted: #91a6c2;
    --good: #8ce0b0;
    --bad: #ff929b;
    --accent: #83bfff;
}

body {
    margin: 0;
    font-family: Inter, Arial, Helvetica, sans-serif;
    background: var(--bg);
    color: var(--text);
}

header {
    padding: 20px 26px;
    background: #0b1728;
    border-bottom: 1px solid var(--line);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

header h1 {
    margin: 0;
    font-size: 23px;
}

.subtitle {
    color: var(--muted);
    margin-top: 5px;
    font-size: 12px;
}

.mode {
    border: 1px solid #315579;
    padding: 7px 11px;
    border-radius: 999px;
    color: #9bd0ff;
    font-size: 12px;
    font-weight: 700;
}

.layout {
    display: grid;
    grid-template-columns: 300px 1fr;
    min-height: calc(100vh - 73px);
}

.sidebar {
    padding: 20px;
    background: #0b1627;
    border-right: 1px solid var(--line);
}

.main {
    padding: 20px;
    overflow: hidden;
}

.box {
    border: 1px solid var(--line);
    border-radius: 11px;
    padding: 14px;
    background: var(--panel);
    margin-bottom: 15px;
}

.box h3 {
    margin: 0 0 12px;
    font-size: 15px;
}

label {
    display: block;
    color: var(--muted);
    font-size: 12px;
    margin: 10px 0 5px;
}

input,
select,
button {
    width: 100%;
    border: 1px solid #345070;
    background: #102139;
    color: var(--text);
    border-radius: 7px;
    padding: 9px 10px;
}

button {
    cursor: pointer;
    font-weight: 700;
}

button:hover {
    background: #183052;
}

button.primary {
    background: #155fa7;
    border-color: #2378c8;
}

button.danger {
    background: #8a2028;
    border-color: #b03943;
}

button:disabled {
    opacity: .45;
    cursor: not-allowed;
}

.small {
    color: var(--muted);
    font-size: 11px;
    line-height: 1.5;
}

.status {
    padding: 11px 13px;
    border-radius: 9px;
    margin-bottom: 15px;
    background: #103829;
    color: #8fe0b1;
    border: 1px solid #225b45;
    font-size: 13px;
}

.status.error {
    background: #491820;
    color: #ff9ba4;
    border-color: #76303a;
}

.cards {
    display: grid;
    grid-template-columns: repeat(8, minmax(110px, 1fr));
    gap: 11px;
    margin-bottom: 16px;
}

.card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 13px;
}

.card .label {
    color: var(--muted);
    font-size: 11px;
}

.card .value {
    font-size: 23px;
    font-weight: 750;
    margin-top: 7px;
}

.grid2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 15px;
    margin-bottom: 15px;
}

.chart-panel {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 11px;
    padding: 14px;
}

.chart-panel h2 {
    font-size: 16px;
    margin: 0 0 12px;
}

canvas {
    width: 100%;
    height: 280px;
    background: #0b1728;
    border-radius: 8px;
}

.table-panel {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 11px;
    padding: 14px;
    overflow-x: auto;
}

.table-panel h2 {
    margin: 0 0 12px;
    font-size: 17px;
}

.controls {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 10px;
}

.controls input,
.controls select,
.controls button {
    width: auto;
    min-width: 130px;
}

table {
    width: 100%;
    min-width: 1750px;
    border-collapse: collapse;
}

th,
td {
    text-align: left;
    padding: 9px 8px;
    border-bottom: 1px solid #22344e;
    font-size: 11px;
}

th {
    color: #9fb4ce;
    font-size: 10px;
    text-transform: uppercase;
}

.badge {
    display: inline-block;
    padding: 4px 7px;
    border-radius: 6px;
    font-weight: 700;
}

.NORMAL {
    background: #113b2c;
    color: #8de2b1;
}

.LOW {
    background: #12345a;
    color: #84c6ff;
}

.MEDIUM {
    background: #50440c;
    color: #ffe578;
}

.HIGH {
    background: #542d0f;
    color: #ffb36d;
}

.CRITICAL {
    background: #58141b;
    color: #ff929b;
}

.correct {
    color: var(--good);
    font-weight: 700;
}

.wrong {
    color: var(--bad);
    font-weight: 700;
}

.source {
    color: var(--accent);
}

.metric {
    color: #b7c9de;
}

.replay-banner {
    display: none;
    padding: 9px 11px;
    background: #473c0a;
    color: #ffe778;
    border: 1px solid #796915;
    border-radius: 8px;
    margin-bottom: 12px;
}

.header-badges {
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
    justify-content: flex-end;
}

.freeze {
    border: 1px solid #2f6e4c;
    padding: 7px 11px;
    border-radius: 999px;
    color: #9ce5b8;
    background: #103325;
    font-size: 12px;
    font-weight: 700;
}

.freeze.pending {
    border-color: #735f1f;
    color: #ffe27b;
    background: #3a310e;
}

.comparison-wrap {
    margin-bottom: 16px;
}

.comparison-head {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    align-items: flex-start;
    margin-bottom: 12px;
}

.comparison-head h2 {
    margin: 0;
    font-size: 17px;
}

.comparison-note {
    color: var(--muted);
    font-size: 12px;
    line-height: 1.5;
    max-width: 950px;
}

.comparison-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
}

.comparison-card {
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 14px;
    background: #0b1728;
}

.comparison-card.reference {
    border-color: #2e77b8;
    box-shadow: inset 0 0 0 1px #2e77b8;
}

.comparison-card.stress {
    border-color: #725f20;
}

.comparison-card h3 {
    margin: 0 0 7px;
    font-size: 14px;
}

.comparison-card .sub {
    color: var(--muted);
    font-size: 11px;
    margin-bottom: 12px;
    line-height: 1.45;
}

.comparison-metrics {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
}

.comparison-metric {
    border: 1px solid #20344e;
    border-radius: 7px;
    padding: 8px;
    background: #0d1b2e;
}

.comparison-metric .k {
    color: var(--muted);
    font-size: 10px;
}

.comparison-metric .v {
    font-size: 17px;
    font-weight: 750;
    margin-top: 4px;
}

.reference-label {
    display: inline-block;
    margin-top: 10px;
    border-radius: 999px;
    padding: 5px 8px;
    background: #123d60;
    color: #9bd3ff;
    font-size: 10px;
    font-weight: 700;
}

.stress-label {
    display: inline-block;
    margin-top: 10px;
    border-radius: 999px;
    padding: 5px 8px;
    background: #473c0a;
    color: #ffe27a;
    font-size: 10px;
    font-weight: 700;
}

@media (max-width: 1350px) {
    .cards {
        grid-template-columns: repeat(4, 1fr);
    }

    .grid2 {
        grid-template-columns: 1fr;
    }

    .comparison-grid {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 900px) {
    .layout {
        grid-template-columns: 1fr;
    }

    .sidebar {
        border-right: 0;
        border-bottom: 1px solid var(--line);
    }

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
            Phase 17H dashboard integration — validation/external replay only; locked test is untouched
        </div>
    </div>

    <div class="header-badges">
        <div class="freeze pending" id="freezeBadge">
            PHASE 17H CHECK
        </div>

        <div class="mode" id="modeBadge">
            VALIDATION
        </div>
    </div>
</header>

<div class="layout">

    <aside class="sidebar">

        <div class="box">
            <h3>External Flow CSV</h3>

            <div class="small">
                Upload a CICIDS-style network-flow CSV containing the same raw feature
                columns used by the frozen project preprocessing pipeline.
            </div>

            <label>CSV file</label>
            <input
                id="csvFile"
                type="file"
                accept=".csv,text/csv"
            >

            <label>Maximum rows</label>
            <input
                id="maxRows"
                type="number"
                min="1"
                max="100000"
                value="1000"
            >

            <label>Inference batch size</label>
            <input
                id="batchSize"
                type="number"
                min="1"
                value="2048"
            >

            <button
                class="primary"
                style="margin-top:12px"
                onclick="analyzeUpload()"
            >
                Run IDS Detection
            </button>

            <div
                id="uploadInfo"
                class="small"
                style="margin-top:10px"
            ></div>
        </div>

        <div class="box">
            <h3>Evaluation Mode</h3>

            <div class="small">
                Only the selected validation mode updates the main cards,
                graphs, table and independent model outputs.
            </div>

            <label>Validation sampling</label>

            <select id="evaluationMode">
                <option value="natural">
                    Natural / Random
                </option>

                <option value="balanced">
                    Class-Balanced Stress Test
                </option>
            </select>

            <label>Validation samples</label>

            <input
                id="validationSamples"
                type="number"
                min="1"
                max="50000"
                value="1000"
            >

            <button
                class="primary"
                style="margin-top:10px"
                onclick="applyEvaluationMode()"
            >
                Apply Validation Mode
            </button>

            <button
                style="margin-top:8px"
                onclick="chooseSource('external')"
            >
                Latest External File
            </button>

            <div
                id="evaluationModeInfo"
                class="small"
                style="margin-top:10px"
            ></div>
        </div>

        <div class="box">
            <h3>Live Attack Replay</h3>

            <div class="small">
                Replays the already-computed predictions progressively so you can
                watch alerts and graphs update in real time. It does not capture
                packets from the network.
            </div>

            <label>Events / second</label>
            <input
                id="replaySpeed"
                type="number"
                min="1"
                max="100"
                value="10"
            >

            <button
                style="margin-top:10px"
                onclick="startLiveReplay()"
            >
                Start Live Replay
            </button>

            <button
                style="margin-top:8px"
                onclick="stopLiveReplay()"
            >
                Stop
            </button>

            <button
                style="margin-top:8px"
                onclick="showAll()"
            >
                Show All
            </button>
        </div>

        <div class="box">
            <h3>Model Contract</h3>

            <div class="small">
                Random Forest<br>
                Personalized Autoencoder<br>
                FedProx70 MLP<br>
                CNN-BiLSTM<br>
                Adaptive FAST / DEEP policy<br><br>

                The dashboard performs inference only.
                It does not retrain the frozen models.
            </div>
        </div>

    </aside>

    <main class="main">

        <div id="status" class="status">
            Loading dashboard data...
        </div>

        <div id="replayBanner" class="replay-banner">
            LIVE REPLAY ACTIVE
        </div>

        <div class="cards">

            <div class="card">
                <div class="label">Total Flows</div>
                <div class="value" id="totalFlows">0</div>
            </div>

            <div class="card">
                <div class="label">Normal Traffic</div>
                <div class="value" id="normalTraffic">0</div>
            </div>

            <div class="card">
                <div class="label">Attack Alerts</div>
                <div class="value" id="attackAlerts">0</div>
            </div>

            <div class="card">
                <div class="label">Attack Rate</div>
                <div class="value" id="attackRate">0%</div>
            </div>

            <div class="card">
                <div class="label">Anomalous Predictions</div>
                <div class="value" id="anomalousPredictions">0</div>
            </div>

            <div class="card">
                <div class="label">Anomalous Rate</div>
                <div class="value" id="anomalousRate">0%</div>
            </div>

            <div class="card">
                <div class="label">FAST Path</div>
                <div class="value" id="fastPath">0</div>
            </div>

            <div class="card">
                <div class="label">DEEP Path</div>
                <div class="value" id="deepPath">0</div>
            </div>

        </div>

        <section class="chart-panel comparison-wrap">

            <div class="comparison-head">
                <div>
                    <h2>Active Evaluation Mode</h2>

                    <div class="comparison-note" id="activeModeDescription">
                        Natural / Random validation is the default reference.
                        Change the mode from the left panel to run the class-balanced stress test.
                        Only the selected mode updates this dashboard.
                    </div>
                </div>
            </div>

            <div class="comparison-grid">

                <div class="comparison-card reference">
                    <h3 id="activeModeTitle">
                        Natural / Random Validation
                    </h3>

                    <div class="sub" id="activeModeSub">
                        Validation-only inference. Frozen models; no retraining.
                    </div>

                    <div class="comparison-metrics">

                        <div class="comparison-metric">
                            <div class="k">Accuracy</div>
                            <div class="v" id="activeAccuracy">-</div>
                        </div>

                        <div class="comparison-metric">
                            <div class="k">Balanced Accuracy</div>
                            <div class="v" id="activeBalAcc">-</div>
                        </div>

                        <div class="comparison-metric">
                            <div class="k">Macro F1</div>
                            <div class="v" id="activeMacroF1">-</div>
                        </div>

                        <div class="comparison-metric">
                            <div class="k">Anomalous Rate</div>
                            <div class="v" id="activeAnomalyRate">-</div>
                        </div>

                        <div class="comparison-metric">
                            <div class="k">FAST Path</div>
                            <div class="v" id="activeFastRate">-</div>
                        </div>

                        <div class="comparison-metric">
                            <div class="k">Samples</div>
                            <div class="v" id="activeSamples">-</div>
                        </div>

                    </div>

                    <div class="reference-label" id="activeModeLabel">
                        DEFAULT VALIDATION REFERENCE
                    </div>
                </div>

                <div class="comparison-card">
                    <h3>Independent Model Status</h3>

                    <div class="sub">
                        Each frozen model runs its own inference. The fusion/adaptive gate
                        combines those outputs only after the individual results are produced.
                    </div>

                    <div class="comparison-metrics">

                        <div class="comparison-metric">
                            <div class="k">Random Forest</div>
                            <div class="v" id="rfModelMetric">-</div>
                        </div>

                        <div class="comparison-metric">
                            <div class="k">Autoencoder</div>
                            <div class="v" id="aeModelMetric">-</div>
                        </div>

                        <div class="comparison-metric">
                            <div class="k">FedProx70</div>
                            <div class="v" id="fedModelMetric">-</div>
                        </div>

                        <div class="comparison-metric">
                            <div class="k">CNN-BiLSTM</div>
                            <div class="v" id="cnnModelMetric">-</div>
                        </div>

                    </div>

                    <div class="reference-label">
                        INFERENCE ONLY — MODEL WEIGHTS FROZEN
                    </div>
                </div>

                <div class="comparison-card">
                    <h3>Current / Live Event</h3>

                    <div class="sub">
                        During live replay, these values update independently for the
                        most recently visible event.
                    </div>

                    <div class="comparison-metrics">

                        <div class="comparison-metric">
                            <div class="k">RF Output</div>
                            <div class="v" id="liveRF">-</div>
                        </div>

                        <div class="comparison-metric">
                            <div class="k">AE Score</div>
                            <div class="v" id="liveAE">-</div>
                        </div>

                        <div class="comparison-metric">
                            <div class="k">FedProx70 Output</div>
                            <div class="v" id="liveFed">-</div>
                        </div>

                        <div class="comparison-metric">
                            <div class="k">CNN Output</div>
                            <div class="v" id="liveCNN">-</div>
                        </div>

                        <div class="comparison-metric">
                            <div class="k">Final Prediction</div>
                            <div class="v" id="liveFinal">-</div>
                        </div>

                        <div class="comparison-metric">
                            <div class="k">Severity</div>
                            <div class="v" id="liveSeverity">-</div>
                        </div>

                    </div>
                </div>

            </div>

        </section>


        <div class="grid2">

            <section class="chart-panel">
                <h2>Predicted Class Distribution</h2>
                <canvas id="classChart"></canvas>
            </section>

            <section class="chart-panel">
                <h2>Live Attack Timeline</h2>
                <canvas id="timelineChart"></canvas>
            </section>

        </div>

        <div class="grid2">

            <section class="chart-panel">
                <h2>Client Attack Load</h2>
                <canvas id="clientChart"></canvas>
            </section>

            <section class="chart-panel">
                <h2>Model / Evaluation Snapshot</h2>

                <div
                    id="metricsBox"
                    class="small"
                    style="font-size:13px; line-height:2"
                >
                    Metrics are shown when ground-truth labels are available.
                </div>
            </section>

        </div>

        <section class="table-panel">

            <h2>Live IDS Events</h2>

            <div class="controls">

                <input
                    id="search"
                    placeholder="Search class/source/client"
                >

                <select id="severityFilter">
                    <option value="">All severities</option>
                    <option>NORMAL</option>
                    <option>LOW</option>
                    <option>MEDIUM</option>
                    <option>HIGH</option>
                    <option>CRITICAL</option>
                </select>

                <select id="classFilter">
                    <option value="">All classes</option>
                </select>

                <button onclick="loadDashboard()">
                    Refresh
                </button>

            </div>

            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Time</th>
                        <th>Client</th>
                        <th>Source</th>
                        <th>Destination</th>
                        <th>Prediction</th>
                        <th>Confidence</th>
                        <th>AE Score</th>
                        <th>Severity</th>
                        <th>Path</th>
                        <th>Decision Source</th>
                        <th>RF</th>
                        <th>FedProx70</th>
                        <th>CNN-BiLSTM</th>
                        <th>Ground Truth</th>
                        <th>Result</th>
                        <th>Action</th>
                    </tr>
                </thead>

                <tbody id="alertsBody"></tbody>
            </table>

        </section>

    </main>

</div>


<script>
let allAlerts = [];
let summary = {};
let comparisonData = {};
let replayTimer = null;
let replayCursor = null;


// ============================================================
// BASIC FORMATTERS
// ============================================================

function pct(value) {
    if (
        value === null ||
        value === undefined ||
        Number.isNaN(Number(value))
    ) {
        return "-";
    }

    return (
        Number(value) * 100
    ).toFixed(2) + "%";
}


function displaySource(alert) {
    if (alert.source_ip) {
        return `
            <div>${escapeHtml(alert.source_ip)}</div>
            <div class="source">${escapeHtml(alert.source_id || "")}</div>
        `;
    }

    return `
        <div class="source">${escapeHtml(alert.source_id || "-")}</div>
        <div class="metric">No raw IP</div>
    `;
}


function displayDestination(alert) {
    if (alert.destination_ip) {
        return `
            <div>${escapeHtml(alert.destination_ip)}</div>
            <div class="source">${escapeHtml(alert.destination_id || "")}</div>
        `;
    }

    return `<span class="metric">-</span>`;
}


function escapeHtml(value) {
    return String(
        value ?? ""
    )
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}


// ============================================================
// CANVAS CHARTS
// ============================================================

function resizeCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    canvas.width = Math.max(
        400,
        Math.floor(rect.width * dpr)
    );

    canvas.height = Math.floor(
        280 * dpr
    );

    const ctx = canvas.getContext("2d");

    ctx.setTransform(
        dpr,
        0,
        0,
        dpr,
        0,
        0
    );

    return {
        ctx,
        width: rect.width,
        height: 280
    };
}


function drawEmpty(
    canvas,
    message
) {
    const {
        ctx,
        width,
        height
    } = resizeCanvas(canvas);

    ctx.clearRect(
        0,
        0,
        width,
        height
    );

    ctx.fillStyle = "#91a6c2";
    ctx.font = "13px Arial";
    ctx.fillText(
        message,
        20,
        35
    );
}


function drawBarChart(
    canvas,
    labels,
    values
) {
    if (
        !labels.length
    ) {
        drawEmpty(
            canvas,
            "No data"
        );
        return;
    }

    const {
        ctx,
        width,
        height
    } = resizeCanvas(canvas);

    ctx.clearRect(
        0,
        0,
        width,
        height
    );

    const left = 48;
    const right = 18;
    const top = 18;
    const bottom = 70;

    const chartWidth =
        width
        -
        left
        -
        right;

    const chartHeight =
        height
        -
        top
        -
        bottom;

    const maxValue = Math.max(
        1,
        ...values
    );

    // grid
    ctx.strokeStyle = "#263a55";
    ctx.fillStyle = "#8fa5bf";
    ctx.font = "10px Arial";

    for (
        let i = 0;
        i <= 4;
        i++
    ) {
        const y =
            top
            +
            chartHeight
            -
            (
                chartHeight
                *
                i
                /
                4
            );

        ctx.beginPath();
        ctx.moveTo(
            left,
            y
        );
        ctx.lineTo(
            left
            +
            chartWidth,
            y
        );
        ctx.stroke();

        const value = Math.round(
            maxValue
            *
            i
            /
            4
        );

        ctx.fillText(
            String(value),
            7,
            y + 3
        );
    }

    const slot =
        chartWidth
        /
        labels.length;

    const barWidth = Math.max(
        8,
        slot * 0.55
    );

    labels.forEach(
        (
            label,
            index
        ) => {
            const value =
                Number(
                    values[
                        index
                    ]
                )
                ||
                0;

            const barHeight =
                chartHeight
                *
                value
                /
                maxValue;

            const x =
                left
                +
                slot
                *
                index
                +
                (
                    slot
                    -
                    barWidth
                )
                /
                2;

            const y =
                top
                +
                chartHeight
                -
                barHeight;

            ctx.fillStyle = "#2785d8";

            ctx.fillRect(
                x,
                y,
                barWidth,
                barHeight
            );

            ctx.fillStyle = "#dbe8f7";
            ctx.font = "10px Arial";
            ctx.textAlign = "center";

            ctx.fillText(
                String(value),
                x
                +
                barWidth
                /
                2,
                Math.max(
                    12,
                    y - 5
                )
            );

            ctx.save();

            ctx.translate(
                x
                +
                barWidth
                /
                2,
                top
                +
                chartHeight
                +
                10
            );

            ctx.rotate(
                -Math.PI
                /
                4
            );

            ctx.fillStyle = "#9db1ca";

            ctx.fillText(
                String(
                    label
                ).slice(
                    0,
                    20
                ),
                0,
                0
            );

            ctx.restore();
        }
    );

    ctx.textAlign = "left";
}


function drawLineChart(
    canvas,
    points
) {
    if (
        !points.length
    ) {
        drawEmpty(
            canvas,
            "No live attack data"
        );
        return;
    }

    const {
        ctx,
        width,
        height
    } = resizeCanvas(canvas);

    ctx.clearRect(
        0,
        0,
        width,
        height
    );

    const left = 45;
    const right = 15;
    const top = 20;
    const bottom = 35;

    const chartWidth =
        width
        -
        left
        -
        right;

    const chartHeight =
        height
        -
        top
        -
        bottom;

    const maxValue = Math.max(
        1,
        ...points.map(
            p => p.attacks
        )
    );

    ctx.strokeStyle = "#263a55";

    for (
        let i = 0;
        i <= 4;
        i++
    ) {
        const y =
            top
            +
            chartHeight
            -
            (
                chartHeight
                *
                i
                /
                4
            );

        ctx.beginPath();
        ctx.moveTo(
            left,
            y
        );
        ctx.lineTo(
            left
            +
            chartWidth,
            y
        );
        ctx.stroke();

        ctx.fillStyle = "#8fa5bf";
        ctx.font = "10px Arial";
        ctx.fillText(
            String(
                Math.round(
                    maxValue
                    *
                    i
                    /
                    4
                )
            ),
            8,
            y + 3
        );
    }

    const count =
        points.length;

    ctx.beginPath();

    points.forEach(
        (
            point,
            index
        ) => {
            const x =
                left
                +
                (
                    count === 1
                    ?
                    chartWidth
                    /
                    2
                    :
                    chartWidth
                    *
                    index
                    /
                    (
                        count
                        -
                        1
                    )
                );

            const y =
                top
                +
                chartHeight
                -
                (
                    chartHeight
                    *
                    point.attacks
                    /
                    maxValue
                );

            if (
                index === 0
            ) {
                ctx.moveTo(
                    x,
                    y
                );
            } else {
                ctx.lineTo(
                    x,
                    y
                );
            }
        }
    );

    ctx.strokeStyle = "#ff7f88";
    ctx.lineWidth = 2;
    ctx.stroke();

    points.forEach(
        (
            point,
            index
        ) => {
            const x =
                left
                +
                (
                    count === 1
                    ?
                    chartWidth
                    /
                    2
                    :
                    chartWidth
                    *
                    index
                    /
                    (
                        count
                        -
                        1
                    )
                );

            const y =
                top
                +
                chartHeight
                -
                (
                    chartHeight
                    *
                    point.attacks
                    /
                    maxValue
                );

            ctx.fillStyle = "#ff7f88";

            ctx.beginPath();
            ctx.arc(
                x,
                y,
                3,
                0,
                Math.PI * 2
            );
            ctx.fill();
        }
    );

    ctx.fillStyle = "#9db1ca";
    ctx.font = "10px Arial";

    ctx.fillText(
        "Replay order →",
        left,
        height - 10
    );
}


// ============================================================
// DATA DERIVATION FOR LIVE REPLAY
// ============================================================

function visibleAlerts() {
    if (
        replayCursor === null
    ) {
        return allAlerts;
    }

    // allAlerts is newest first from API.
    // Reverse temporarily so replay starts from earliest row.
    const chronological =
        [...allAlerts]
        .reverse();

    return chronological.slice(
        0,
        replayCursor
    );
}


function deriveGraphData(
    alerts
) {
    const classCounts = {};
    const clientCounts = {
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0
    };

    alerts.forEach(
        alert => {
            const name =
                alert.attack_type
                ||
                "UNKNOWN";

            classCounts[
                name
            ] = (
                classCounts[
                    name
                ]
                ||
                0
            ) + 1;

            if (
                name !== "BENIGN"
            ) {
                const client =
                    String(
                        alert.client_id
                    );

                if (
                    clientCounts[
                        client
                    ]
                    !==
                    undefined
                ) {
                    clientCounts[
                        client
                    ] += 1;
                }
            }
        }
    );

    const bucketSize =
        Math.max(
            1,
            Math.ceil(
                Math.max(
                    alerts.length,
                    1
                )
                /
                30
            )
        );

    const timeline = [];

    for (
        let start = 0;
        start < alerts.length;
        start += bucketSize
    ) {
        const chunk =
            alerts.slice(
                start,
                start
                +
                bucketSize
            );

        const attacks =
            chunk.filter(
                alert =>
                    alert.attack_type
                    !==
                    "BENIGN"
            ).length;

        timeline.push(
            {
                start,
                end:
                    Math.min(
                        start
                        +
                        bucketSize,
                        alerts.length
                    ),
                attacks
            }
        );
    }

    return {
        classCounts,
        clientCounts,
        timeline
    };
}


// ============================================================
// RENDER
// ============================================================

function renderDashboard() {
    const alerts =
        visibleAlerts();

    const total =
        alerts.length;

    const normal =
        alerts.filter(
            a =>
                a.attack_type
                ===
                "BENIGN"
        ).length;

    const attackCount =
        total
        -
        normal;

    const anomalousCount =
        alerts.filter(
            a =>
                a.attack_type
                ===
                "ANOMALOUS_TRAFFIC"
        ).length;

    const fast =
        alerts.filter(
            a =>
                a.processing_path
                ===
                "FAST_ATTACK_PATH"
        ).length;

    const deep =
        alerts.filter(
            a =>
                a.processing_path
                ===
                "DEEP_ANALYSIS"
        ).length;

    document.getElementById(
        "totalFlows"
    ).textContent =
        total;

    document.getElementById(
        "normalTraffic"
    ).textContent =
        normal;

    document.getElementById(
        "attackAlerts"
    ).textContent =
        attackCount;

    document.getElementById(
        "attackRate"
    ).textContent =
        total
        ?
        (
            attackCount
            *
            100
            /
            total
        ).toFixed(2)
        +
        "%"
        :
        "0%";

    document.getElementById(
        "anomalousPredictions"
    ).textContent =
        anomalousCount;

    document.getElementById(
        "anomalousRate"
    ).textContent =
        total
        ?
        (
            anomalousCount
            *
            100
            /
            total
        ).toFixed(2)
        +
        "%"
        :
        "0%";

    document.getElementById(
        "fastPath"
    ).textContent =
        fast;

    document.getElementById(
        "deepPath"
    ).textContent =
        deep;


    const graphs =
        deriveGraphData(
            alerts
        );

    const classLabels =
        Object.keys(
            graphs.classCounts
        );

    drawBarChart(
        document.getElementById(
            "classChart"
        ),
        classLabels,
        classLabels.map(
            key =>
                graphs.classCounts[
                    key
                ]
        )
    );

    drawLineChart(
        document.getElementById(
            "timelineChart"
        ),
        graphs.timeline
    );

    const clientLabels = [
        "Client 1",
        "Client 2",
        "Client 3",
        "Client 4",
        "Client 5"
    ];

    drawBarChart(
        document.getElementById(
            "clientChart"
        ),
        clientLabels,
        [
            graphs.clientCounts["1"],
            graphs.clientCounts["2"],
            graphs.clientCounts["3"],
            graphs.clientCounts["4"],
            graphs.clientCounts["5"]
        ]
    );

    renderTable(
        alerts
    );

    populateClassFilter(
        classLabels
    );

    renderCurrentModelEvent();
}


function renderActiveMode() {
    const mode =
        summary.validation_mode
        ||
        "natural";

    const metrics =
        summary.metrics
        ||
        {};

    const total =
        Number(
            summary.sample_count
            ??
            summary.total_flows
            ??
            allAlerts.length
            ??
            0
        );

    const fast =
        Number(
            summary.fast_path_count
            ??
            0
        );

    const fastRate =
        total
        ?
        fast
        /
        total
        :
        0;

    document.getElementById(
        "evaluationMode"
    ).value =
        mode;

    document.getElementById(
        "activeAccuracy"
    ).textContent =
        pct(
            metrics.accuracy
        );

    document.getElementById(
        "activeBalAcc"
    ).textContent =
        pct(
            metrics.balanced_accuracy
        );

    document.getElementById(
        "activeMacroF1"
    ).textContent =
        pct(
            metrics.macro_f1
        );

    document.getElementById(
        "activeAnomalyRate"
    ).textContent =
        pct(
            summary.anomalous_rate
        );

    document.getElementById(
        "activeFastRate"
    ).textContent =
        pct(
            fastRate
        );

    document.getElementById(
        "activeSamples"
    ).textContent =
        total;

    const title =
        document.getElementById(
            "activeModeTitle"
        );

    const label =
        document.getElementById(
            "activeModeLabel"
        );

    const description =
        document.getElementById(
            "activeModeDescription"
        );

    const sub =
        document.getElementById(
            "activeModeSub"
        );

    if (
        summary.dashboard_source
        ===
        "external"
    ) {
        title.textContent =
            "External File Inference";

        label.textContent =
            "EXTERNAL INPUT";

        description.textContent =
            "The dashboard is currently showing the latest uploaded external CSV.";

        sub.textContent =
            "Frozen model inference on external flow records.";

    } else if (
        mode
        ===
        "balanced"
    ) {
        title.textContent =
            "Class-Balanced Stress Test";

        label.textContent =
            "STRESS TEST — NOT OVERALL MODEL ACCURACY";

        description.textContent =
            "The dashboard is showing approximately equal numbers from all six known classes.";

        sub.textContent =
            "Balanced validation sampling + round-robin client assignment.";

    } else {
        title.textContent =
            "Natural / Random Validation";

        label.textContent =
            "DEFAULT VALIDATION REFERENCE";

        description.textContent =
            "The dashboard is showing a uniform random validation sample that preserves the natural validation distribution.";

        sub.textContent =
            "Natural/random validation sampling + round-robin client assignment.";
    }

    const modelMetrics =
        summary.model_metrics
        ||
        {};

    const rf =
        modelMetrics.random_forest;

    const ae =
        modelMetrics.autoencoder;

    const fed =
        modelMetrics.fedprox70;

    const cnn =
        modelMetrics.cnn_bilstm;

    document.getElementById(
        "rfModelMetric"
    ).textContent =
        rf
        ?
        "Acc "
        +
        pct(
            rf.accuracy
        )
        :
        "Active";

    document.getElementById(
        "aeModelMetric"
    ).textContent =
        ae
        ?
        "Anom "
        +
        pct(
            ae.anomaly_rate
        )
        :
        "Active";

    document.getElementById(
        "fedModelMetric"
    ).textContent =
        fed
        ?
        "Acc "
        +
        pct(
            fed.accuracy
        )
        :
        "Active";

    document.getElementById(
        "cnnModelMetric"
    ).textContent =
        cnn
        ?
        "Acc "
        +
        pct(
            cnn.accuracy
        )
        :
        "Active";
}


function renderCurrentModelEvent() {
    const alerts =
        visibleAlerts();

    if (
        !alerts.length
    ) {
        [
            "liveRF",
            "liveAE",
            "liveFed",
            "liveCNN",
            "liveFinal",
            "liveSeverity"
        ]
        .forEach(
            id => {
                document.getElementById(
                    id
                ).textContent =
                    "-";
            }
        );

        return;
    }

    // visibleAlerts() is chronological when replay is active.
    // When showing all, API memory is newest-first, so use index 0.
    const event =
        replayCursor
        ===
        null
        ?
        alerts[
            0
        ]
        :
        alerts[
            alerts.length
            -
            1
        ];

    document.getElementById(
        "liveRF"
    ).textContent =
        (
            event.rf_prediction
            ||
            "-"
        )
        +
        " "
        +
        pct(
            event.rf_confidence
        );

    document.getElementById(
        "liveAE"
    ).textContent =
        pct(
            event.anomaly_score
        );

    document.getElementById(
        "liveFed"
    ).textContent =
        (
            event.fedprox70_prediction
            ||
            "-"
        )
        +
        " "
        +
        pct(
            event.fedprox70_confidence
        );

    document.getElementById(
        "liveCNN"
    ).textContent =
        (
            event.cnn_bilstm_prediction
            ||
            "-"
        )
        +
        " "
        +
        pct(
            event.cnn_bilstm_confidence
        );

    document.getElementById(
        "liveFinal"
    ).textContent =
        (
            event.attack_type
            ||
            "-"
        )
        +
        " "
        +
        pct(
            event.confidence
        );

    document.getElementById(
        "liveSeverity"
    ).textContent =
        event.severity
        ||
        "-";
}


function renderMetrics() {
    const box =
        document.getElementById(
            "metricsBox"
        );

    const metrics =
        summary.metrics;

    if (
        !metrics
    ) {
        box.innerHTML = `
            Ground-truth labels are not available for this input.<br>
            Dashboard is showing inference results only.<br><br>
            Locked test used: <b>NO</b><br>
            Models retrained: <b>NO</b>
        `;

        return;
    }

    const modeLabel =
        summary.dashboard_source
        ===
        "external"
        ?
        "External File"
        :
        (
            summary.validation_mode
            ===
            "balanced"
        )
        ?
        "Class-Balanced Stress Test"
        :
        "Natural / Random Validation";

    box.innerHTML = `
        <b>${modeLabel}</b><br>
        Accuracy: <b>${pct(metrics.accuracy)}</b><br>
        Balanced Accuracy: <b>${pct(metrics.balanced_accuracy)}</b><br>
        Macro Precision: <b>${pct(metrics.macro_precision)}</b><br>
        Macro Recall: <b>${pct(metrics.macro_recall)}</b><br>
        Macro F1: <b>${pct(metrics.macro_f1)}</b><br>
        Weighted F1: <b>${pct(metrics.weighted_f1)}</b><br><br>
        Locked test used: <b>NO</b><br>
        Models retrained: <b>NO</b>
    `;
}


function populateClassFilter(
    labels
) {
    const select =
        document.getElementById(
            "classFilter"
        );

    const existing =
        new Set(
            Array.from(
                select.options
            ).map(
                option =>
                    option.value
            )
        );

    labels
    .sort()
    .forEach(
        label => {
            if (
                existing.has(
                    label
                )
            ) {
                return;
            }

            const option =
                document.createElement(
                    "option"
                );

            option.value =
                label;

            option.textContent =
                label;

            select.appendChild(
                option
            );
        }
    );
}


function renderTable(
    sourceAlerts
) {
    const q =
        document.getElementById(
            "search"
        )
        .value
        .toLowerCase();

    const severity =
        document.getElementById(
            "severityFilter"
        )
        .value;

    const classFilter =
        document.getElementById(
            "classFilter"
        )
        .value;

    let rows =
        [...sourceAlerts];

    // Latest event at the top.
    rows.reverse();

    rows = rows.filter(
        alert => {
            const haystack = [
                alert.attack_type,
                alert.source_id,
                alert.source_ip,
                alert.destination_ip,
                alert.true_class,
                alert.decision_source,
                "client "
                +
                alert.client_id
            ]
            .join(
                " "
            )
            .toLowerCase();

            if (
                q
                &&
                !haystack.includes(
                    q
                )
            ) {
                return false;
            }

            if (
                severity
                &&
                alert.severity
                !==
                severity
            ) {
                return false;
            }

            if (
                classFilter
                &&
                alert.attack_type
                !==
                classFilter
            ) {
                return false;
            }

            return true;
        }
    );

    const body =
        document.getElementById(
            "alertsBody"
        );

    body.innerHTML =
        "";

    // Browser table remains responsive while charts may contain
    // all replay rows.
    rows
    .slice(
        0,
        2000
    )
    .forEach(
        alert => {
            const tr =
                document.createElement(
                    "tr"
                );

            const truth =
                alert.true_class
                ??
                "-";

            let resultText =
                "-";

            let resultClass =
                "";

            if (
                alert.correct
                ===
                true
            ) {
                resultText =
                    "CORRECT";

                resultClass =
                    "correct";
            } else if (
                alert.correct
                ===
                false
            ) {
                resultText =
                    "WRONG";

                resultClass =
                    "wrong";
            }

            const blockButton =
                alert.blockable
                ?
                `
                <button
                    class="danger"
                    onclick="blockSource(${Number(alert.id)})"
                >
                    Block
                </button>
                `
                :
                `
                <button disabled>
                    No Raw IP
                </button>
                `;

            tr.innerHTML = `
                <td>${escapeHtml(alert.row_index ?? alert.validation_index ?? alert.id)}</td>
                <td>${escapeHtml(alert.event_time || "-")}</td>
                <td>Client ${escapeHtml(alert.client_id)}</td>
                <td>${displaySource(alert)}</td>
                <td>${displayDestination(alert)}</td>
                <td><b>${escapeHtml(alert.attack_type)}</b></td>
                <td>${pct(alert.confidence)}</td>
                <td>${pct(alert.anomaly_score)}</td>
                <td><span class="badge ${escapeHtml(alert.severity)}">${escapeHtml(alert.severity)}</span></td>
                <td>${escapeHtml(alert.processing_path)}</td>
                <td>${escapeHtml(alert.decision_source)}</td>
                <td>${escapeHtml(alert.rf_prediction || "-")}<br><span class="metric">${pct(alert.rf_confidence)}</span></td>
                <td>${escapeHtml(alert.fedprox70_prediction || "-")}<br><span class="metric">${pct(alert.fedprox70_confidence)}</span></td>
                <td>${escapeHtml(alert.cnn_bilstm_prediction || "-")}<br><span class="metric">${pct(alert.cnn_bilstm_confidence)}</span></td>
                <td>${escapeHtml(truth)}</td>
                <td class="${resultClass}">${resultText}</td>
                <td>${blockButton}</td>
            `;

            body.appendChild(
                tr
            );
        }
    );
}


// ============================================================
// API ACTIONS
// ============================================================

async function loadDashboard() {
    try {
        const [
            alertsResponse,
            summaryResponse
        ] = await Promise.all(
            [
                fetch(
                    "/api/alerts?limit=100000"
                ),
                fetch(
                    "/api/summary"
                )
            ]
        );

        if (
            !alertsResponse.ok
            ||
            !summaryResponse.ok
        ) {
            throw new Error(
                "Could not load dashboard API data."
            );
        }

        allAlerts =
            await alertsResponse.json();

        // API sends newest first. Keep that in memory;
        // render functions handle replay ordering.
        summary =
            await summaryResponse.json();

        document.getElementById(
            "modeBadge"
        ).textContent =
            (
                summary.dashboard_source
                ===
                "external"
            )
            ?
            "EXTERNAL"
            :
            (
                summary.validation_mode
                ===
                "balanced"
            )
            ?
            "BALANCED"
            :
            "NATURAL";

        document.getElementById(
            "status"
        ).className =
            "status";

        document.getElementById(
            "status"
        ).textContent =
            (
                summary.dashboard_source
                ===
                "external"
            )
            ?
            `External IDS inference ready: ${summary.source_file || "uploaded CSV"}`
            :
            (
                summary.validation_mode
                ===
                "balanced"
            )
            ?
            "Class-Balanced validation stress test loaded successfully."
            :
            "Natural / Random validation reference loaded successfully.";

        const freezeBadge =
            document.getElementById(
                "freezeBadge"
            );

        freezeBadge.className =
            "freeze";

        freezeBadge.textContent =
            "PHASE 17H FROZEN";

        renderMetrics();
        renderActiveMode();

        if (
            replayCursor
            ===
            null
        ) {
            renderDashboard();
        }

    } catch (
        error
    ) {
        const status =
            document.getElementById(
                "status"
            );

        status.className =
            "status error";

        status.textContent =
            String(
                error
            );
    }
}


async function analyzeUpload() {
    const input =
        document.getElementById(
            "csvFile"
        );

    const info =
        document.getElementById(
            "uploadInfo"
        );

    if (
        !input.files.length
    ) {
        info.textContent =
            "Choose a CSV file first.";
        return;
    }

    const maxRows =
        Number(
            document.getElementById(
                "maxRows"
            ).value
        )
        ||
        1000;

    const batchSize =
        Number(
            document.getElementById(
                "batchSize"
            ).value
        )
        ||
        2048;

    const form =
        new FormData();

    form.append(
        "file",
        input.files[
            0
        ]
    );

    info.textContent =
        "Running frozen IDS models...";

    const status =
        document.getElementById(
            "status"
        );

    status.className =
        "status";

    status.textContent =
        "External file inference is running. On CPU this may take some time.";

    try {
        const response =
            await fetch(
                `/api/upload-analyze?max_rows=${encodeURIComponent(maxRows)}&batch_size=${encodeURIComponent(batchSize)}`,
                {
                    method:
                        "POST",
                    body:
                        form
                }
            );

        const result =
            await response.json();

        if (
            !response.ok
        ) {
            throw new Error(
                result.detail
                ||
                "Upload inference failed."
            );
        }

        info.textContent =
            `Completed: ${result.rows} rows, ${result.attack_alerts} attack alerts, ${result.anomalous_predictions} anomalous predictions.`;

        replayCursor =
            null;

        await loadDashboard();

    } catch (
        error
    ) {
        info.textContent =
            String(
                error
            );

        status.className =
            "status error";

        status.textContent =
            String(
                error
            );
    }
}


async function applyEvaluationMode() {
    const mode =
        document.getElementById(
            "evaluationMode"
        ).value;

    const samples =
        Math.max(
            1,
            Number(
                document.getElementById(
                    "validationSamples"
                ).value
            )
            ||
            1000
        );

    const info =
        document.getElementById(
            "evaluationModeInfo"
        );

    const status =
        document.getElementById(
            "status"
        );

    info.textContent =
        "Running frozen validation inference...";

    status.className =
        "status";

    status.textContent =
        (
            mode
            ===
            "natural"
        )
        ?
        "Building Natural / Random validation dashboard..."
        :
        "Building Class-Balanced validation stress-test dashboard...";

    try {
        const response =
            await fetch(
                `/api/validation-mode/${encodeURIComponent(mode)}?samples=${encodeURIComponent(samples)}`,
                {
                    method:
                        "POST"
                }
            );

        const result =
            await response.json();

        if (
            !response.ok
        ) {
            throw new Error(
                result.detail
                ||
                "Validation mode inference failed."
            );
        }

        stopLiveReplay();

        replayCursor =
            null;

        info.textContent =
            `${result.message} Samples: ${result.sample_count}. Models retrained: NO.`;

        await loadDashboard();

    } catch (
        error
    ) {
        info.textContent =
            String(
                error
            );

        status.className =
            "status error";

        status.textContent =
            String(
                error
            );
    }
}


async function chooseSource(
    source
) {
    try {
        const response =
            await fetch(
                `/api/source/${encodeURIComponent(source)}`,
                {
                    method:
                        "POST"
                }
            );

        const result =
            await response.json();

        if (
            !response.ok
        ) {
            throw new Error(
                result.detail
                ||
                "Could not change source."
            );
        }

        stopLiveReplay();
        replayCursor =
            null;

        await loadDashboard();

    } catch (
        error
    ) {
        const status =
            document.getElementById(
                "status"
            );

        status.className =
            "status error";

        status.textContent =
            String(
                error
            );
    }
}


async function blockSource(
    alertId
) {
    const confirmed =
        confirm(
            "Add this raw source IP to the dashboard block list?\n\n"
            +
            "This does NOT change the operating-system/network firewall."
        );

    if (
        !confirmed
    ) {
        return;
    }

    const response =
        await fetch(
            `/api/alerts/${encodeURIComponent(alertId)}/block`,
            {
                method:
                    "POST"
            }
        );

    const result =
        await response.json();

    if (
        !response.ok
    ) {
        alert(
            result.detail
            ||
            "Block failed."
        );

        return;
    }

    alert(
        result.message
        +
        "\nSource: "
        +
        result.source_ip
    );
}


// ============================================================
// LIVE REPLAY
// ============================================================

function startLiveReplay() {
    stopLiveReplay();

    if (
        !allAlerts.length
    ) {
        return;
    }

    replayCursor =
        0;

    document.getElementById(
        "replayBanner"
    ).style.display =
        "block";

    const speed =
        Math.max(
            1,
            Number(
                document.getElementById(
                    "replaySpeed"
                ).value
            )
            ||
            10
        );

    const interval =
        Math.max(
            50,
            1000
            /
            speed
        );

    replayTimer =
        setInterval(
            () => {
                replayCursor += 1;

                if (
                    replayCursor
                    >=
                    allAlerts.length
                ) {
                    replayCursor =
                        allAlerts.length;

                    renderDashboard();
                    stopLiveReplay(
                        false
                    );
                    return;
                }

                renderDashboard();

            },
            interval
        );

    renderDashboard();
}


function stopLiveReplay(
    keepCursor = true
) {
    if (
        replayTimer
    ) {
        clearInterval(
            replayTimer
        );

        replayTimer =
            null;
    }

    document.getElementById(
        "replayBanner"
    ).style.display =
        "none";

    if (
        !keepCursor
    ) {
        // Preserve the fully replayed dataset.
        replayCursor =
            allAlerts.length;
    }
}


function showAll() {
    stopLiveReplay();

    replayCursor =
        null;

    renderDashboard();
}


// ============================================================
// FILTER EVENTS / RESIZE
// ============================================================

document.getElementById(
    "search"
).addEventListener(
    "input",
    renderDashboard
);

document.getElementById(
    "severityFilter"
).addEventListener(
    "change",
    renderDashboard
);

document.getElementById(
    "classFilter"
).addEventListener(
    "change",
    renderDashboard
);

window.addEventListener(
    "resize",
    () => {
        renderDashboard();
    }
);


// ============================================================
// START
// ============================================================

loadDashboard();

// Poll summary/alerts for external updates.
// When live replay is active, avoid resetting the visible cursor.
setInterval(
    async () => {
        if (
            replayTimer
        ) {
            return;
        }

        await loadDashboard();
    },
    10000
);
</script>

</body>
</html>
"""


@app.get(
    "/",
    response_class=HTMLResponse,
)
def dashboard():
    return HTML


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        "ids_dashboard_connected_v5_mode_switch:app",
        host="127.0.0.1",
        port=9999,
        reload=True,
    )
