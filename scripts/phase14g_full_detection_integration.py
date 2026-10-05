from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# ============================================================
# EXISTING PROJECT INFRASTRUCTURE
# ============================================================

try:
    import phase9_fedavg70 as phase4
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "\nCould not import scripts\\phase9_fedavg70.py\n"
        "Keep this script inside the scripts folder."
    ) from exc


# ============================================================
# PHASE
# ============================================================

PHASE_NAME = (
    "PHASE 14G — FULL DETECTION INTEGRATION"
)


# ============================================================
# FROZEN DETECTION ARCHITECTURE
# ============================================================

EXPECTED_FEATURES = 70
SEQUENCE_LENGTH = 20
NUM_CLASSES = 6

RF_FAST_ATTACK_THRESHOLD = 0.95
AE_ANOMALY_THRESHOLD = 0.70
DEEP_CLASSIFIER_THRESHOLD = 0.90

RISK_WEIGHTS = {
    "rf": 0.20,
    "fedprox70": 0.30,
    "cnn_bilstm": 0.35,
    "autoencoder": 0.15,
}

SEVERITY_THRESHOLDS = {
    "normal_max": 0.20,
    "low_max": 0.40,
    "medium_max": 0.60,
    "high_max": 0.80,
}

CLIENT_ATTACKS = {
    1: "DoS Hulk",
    2: "DDoS",
    3: "PortScan",
    4: "DoS GoldenEye",
    5: "FTP-Patator",
}


# ============================================================
# FINAL CNN-BILSTM CHECKPOINT
# ============================================================

CNN_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "detection"
    / "phase14f2"
    / "cnn_bilstm_fedprox_mu001_best_round_1_to_30.pt"
)


# ============================================================
# OUTPUT PATHS
# ============================================================

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "phase14"
    / "phase14g"
)

ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "detection"
    / "phase14g"
)

INTEGRATION_CONFIG_FILE = (
    ARTIFACT_ROOT
    / "phase14g_integration_config.json"
)

DEMO_ALERT_FILE = (
    RESULT_ROOT
    / "phase14g_demo_alert.json"
)


# ============================================================
# CNN-BILSTM ARCHITECTURE
# EXACTLY SAME AS PHASE14E / 14F / 14F1 / 14F2
# ============================================================

CONV1_CHANNELS = 128
CONV2_CHANNELS = 64
LSTM_HIDDEN_SIZE = 64
LSTM_LAYERS = 1
BIDIRECTIONAL = True
FC_HIDDEN_SIZE = 64
DROPOUT = 0.25
EXPECTED_CNN_PARAMETERS = 126_982


class CNNBiLSTM(nn.Module):

    def __init__(
        self,
        feature_count: int = EXPECTED_FEATURES,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=feature_count,
            out_channels=CONV1_CHANNELS,
            kernel_size=3,
            padding=1,
        )

        self.conv2 = nn.Conv1d(
            in_channels=CONV1_CHANNELS,
            out_channels=CONV2_CHANNELS,
            kernel_size=3,
            padding=1,
        )

        self.relu = nn.ReLU()

        self.layer_norm = nn.LayerNorm(
            CONV2_CHANNELS
        )

        self.bilstm = nn.LSTM(
            input_size=CONV2_CHANNELS,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            bidirectional=BIDIRECTIONAL,
        )

        representation_size = (
            LSTM_HIDDEN_SIZE
            *
            (
                2
                if BIDIRECTIONAL
                else 1
            )
        )

        self.dropout = nn.Dropout(
            DROPOUT
        )

        self.fc1 = nn.Linear(
            representation_size,
            FC_HIDDEN_SIZE,
        )

        self.fc2 = nn.Linear(
            FC_HIDDEN_SIZE,
            num_classes,
        )

    def forward(self, x):
        # (B,20,70) -> (B,70,20)
        x = x.transpose(1, 2)

        x = self.relu(
            self.conv1(x)
        )

        x = self.relu(
            self.conv2(x)
        )

        # (B,64,20) -> (B,20,64)
        x = x.transpose(1, 2)

        x = self.layer_norm(
            x
        )

        lstm_output, (
            h_n,
            c_n,
        ) = self.bilstm(
            x
        )

        if BIDIRECTIONAL:
            representation = torch.cat(
                (
                    h_n[-2],
                    h_n[-1],
                ),
                dim=1,
            )
        else:
            representation = h_n[-1]

        representation = self.dropout(
            representation
        )

        representation = self.relu(
            self.fc1(
                representation
            )
        )

        representation = self.dropout(
            representation
        )

        return self.fc2(
            representation
        )


# ============================================================
# DATA CONTRACTS
# ============================================================

@dataclass
class ClassificationResult:
    model: str
    predicted_class: str
    class_id: int
    confidence: float


@dataclass
class AutoencoderResult:
    anomaly_score: float
    is_anomaly: bool


@dataclass
class RiskResult:
    risk_score: float
    severity: str
    final_class: str
    confidence: float
    decision_source: str


@dataclass
class DetectionAlert:
    timestamp_utc: str
    client_id: int
    classification: str
    severity: str
    risk_score: float
    confidence: float
    decision_source: str
    processing_path: str
    rf: Optional[Dict[str, Any]]
    autoencoder: Optional[Dict[str, Any]]
    fedprox70: Optional[Dict[str, Any]]
    cnn_bilstm: Optional[Dict[str, Any]]


# ============================================================
# LABEL MAPPING
# ============================================================

def load_label_mapping():
    (
        _scaler,
        _feature_columns,
        label_mapping,
        _active_indices,
        active_features,
    ) = phase4.load_preprocessing()

    if len(active_features) != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} active features; "
            f"found {len(active_features)}."
        )

    inverse_mapping = {
        int(class_id):
            str(label)
        for label, class_id
        in label_mapping.items()
    }

    if len(inverse_mapping) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes; "
            f"found {len(inverse_mapping)}."
        )

    return (
        label_mapping,
        inverse_mapping,
        list(active_features),
    )


# ============================================================
# FINAL CNN-BILSTM LOADER
# ============================================================

class FrozenCNNBiLSTMPredictor:

    def __init__(
        self,
        checkpoint_path: Path,
        inverse_mapping: Dict[int, str],
        device: torch.device,
    ):
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "\nFrozen Round-20 CNN-BiLSTM checkpoint missing:\n"
                f"{checkpoint_path}"
            )

        self.device = device
        self.inverse_mapping = inverse_mapping

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

        checkpoint_round = int(
            checkpoint.get(
                "round",
                -1,
            )
        )

        checkpoint_mu = float(
            checkpoint.get(
                "mu",
                0.01,
            )
        )

        if checkpoint_round != 20:
            raise RuntimeError(
                "\nPhase14G requires the frozen best Round-20 "
                "CNN-BiLSTM checkpoint.\n"
                f"Checkpoint reports round {checkpoint_round}."
            )

        if abs(checkpoint_mu - 0.01) > 1e-12:
            raise RuntimeError(
                "\nPhase14G requires FedProx mu=0.01 CNN-BiLSTM."
            )

        self.model = CNNBiLSTM().to(
            device
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        parameter_count = sum(
            parameter.numel()
            for parameter
            in self.model.parameters()
        )

        if parameter_count != EXPECTED_CNN_PARAMETERS:
            raise RuntimeError(
                "\nCNN-BiLSTM architecture drift detected.\n"
                f"Expected {EXPECTED_CNN_PARAMETERS:,}; "
                f"found {parameter_count:,}."
            )

        self.model.eval()

        self.checkpoint_round = checkpoint_round
        self.checkpoint_mu = checkpoint_mu
        self.checkpoint_path = checkpoint_path

    @torch.no_grad()
    def predict(
        self,
        sequence_20x70: np.ndarray,
    ) -> ClassificationResult:
        sequence = np.asarray(
            sequence_20x70,
            dtype=np.float32,
        )

        if sequence.shape != (
            SEQUENCE_LENGTH,
            EXPECTED_FEATURES,
        ):
            raise ValueError(
                "\nCNN-BiLSTM sequence must have shape "
                f"({SEQUENCE_LENGTH}, {EXPECTED_FEATURES}); "
                f"found {sequence.shape}."
            )

        tensor = torch.from_numpy(
            sequence
        ).unsqueeze(
            0
        ).to(
            self.device
        )

        logits = self.model(
            tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

        class_id = int(
            torch.argmax(
                probabilities
            ).item()
        )

        confidence = float(
            probabilities[
                class_id
            ].item()
        )

        return ClassificationResult(
            model="CNN-BiLSTM",
            predicted_class=
                self.inverse_mapping[
                    class_id
                ],
            class_id=class_id,
            confidence=confidence,
        )


# ============================================================
# GENERIC CLASSIFIER ADAPTERS
# ============================================================

class CallableClassifierPredictor:
    """
    Adapter for an existing classifier inference function.

    The callable must accept a 70-feature vector and return either:

        ClassificationResult

    or:

        {
            "class_id": 0..5,
            "confidence": 0..1
        }

    This keeps Phase14G independent of the exact RF/FedProx70
    checkpoint serialization format while still enforcing a
    strict integration contract.
    """

    def __init__(
        self,
        model_name: str,
        predictor: Callable[[np.ndarray], Any],
        inverse_mapping: Dict[int, str],
    ):
        self.model_name = model_name
        self.predictor = predictor
        self.inverse_mapping = inverse_mapping

    def predict(
        self,
        flow_70: np.ndarray,
    ) -> ClassificationResult:
        flow = np.asarray(
            flow_70,
            dtype=np.float32,
        ).reshape(-1)

        if flow.shape != (
            EXPECTED_FEATURES,
        ):
            raise ValueError(
                f"\n{self.model_name} input must contain "
                f"{EXPECTED_FEATURES} features; "
                f"found {flow.shape}."
            )

        raw = self.predictor(
            flow
        )

        if isinstance(
            raw,
            ClassificationResult,
        ):
            return raw

        if not isinstance(
            raw,
            dict,
        ):
            raise TypeError(
                f"\n{self.model_name} predictor must return "
                "ClassificationResult or dict."
            )

        class_id = int(
            raw[
                "class_id"
            ]
        )

        confidence = float(
            raw[
                "confidence"
            ]
        )

        if not (
            0.0
            <=
            confidence
            <=
            1.0
        ):
            raise ValueError(
                f"{self.model_name} confidence outside [0,1]."
            )

        return ClassificationResult(
            model=self.model_name,
            predicted_class=
                self.inverse_mapping[
                    class_id
                ],
            class_id=class_id,
            confidence=confidence,
        )


class CallableAutoencoderPredictor:
    """
    Adapter for the personalized autoencoder.

    Callable contract:
        score_fn(flow_70) -> anomaly_score in [0,1]

    Phase14B already established that AE is a supporting novelty
    signal, not the primary classifier.
    """

    def __init__(
        self,
        score_fn: Callable[[np.ndarray], float],
        threshold: float = AE_ANOMALY_THRESHOLD,
    ):
        self.score_fn = score_fn
        self.threshold = float(
            threshold
        )

    def predict(
        self,
        flow_70: np.ndarray,
    ) -> AutoencoderResult:
        flow = np.asarray(
            flow_70,
            dtype=np.float32,
        ).reshape(-1)

        if flow.shape != (
            EXPECTED_FEATURES,
        ):
            raise ValueError(
                "\nAutoencoder input must contain "
                f"{EXPECTED_FEATURES} features."
            )

        anomaly_score = float(
            self.score_fn(
                flow
            )
        )

        anomaly_score = float(
            np.clip(
                anomaly_score,
                0.0,
                1.0,
            )
        )

        return AutoencoderResult(
            anomaly_score=anomaly_score,
            is_anomaly=(
                anomaly_score
                >=
                self.threshold
            ),
        )


# ============================================================
# RISK FUSION
# ============================================================

def classification_risk(
    result: ClassificationResult,
) -> float:
    if (
        result.predicted_class
        .strip()
        .upper()
        ==
        "BENIGN"
    ):
        return float(
            1.0
            -
            result.confidence
        )

    return float(
        result.confidence
    )


def severity_from_score(
    risk_score: float,
) -> str:
    if risk_score < (
        SEVERITY_THRESHOLDS[
            "normal_max"
        ]
    ):
        return "NORMAL"

    if risk_score < (
        SEVERITY_THRESHOLDS[
            "low_max"
        ]
    ):
        return "LOW"

    if risk_score < (
        SEVERITY_THRESHOLDS[
            "medium_max"
        ]
    ):
        return "MEDIUM"

    if risk_score < (
        SEVERITY_THRESHOLDS[
            "high_max"
        ]
    ):
        return "HIGH"

    return "CRITICAL"


def choose_final_class(
    classifiers,
    anomaly_score: float,
) -> Tuple[
    str,
    float,
    str,
]:
    non_benign_results = [
        result
        for result
        in classifiers
        if (
            result is not None
            and
            result.predicted_class
            .strip()
            .upper()
            !=
            "BENIGN"
        )
    ]

    if non_benign_results:
        best = max(
            non_benign_results,
            key=lambda result:
                result.confidence,
        )

        return (
            best.predicted_class,
            float(
                best.confidence
            ),
            best.model,
        )

    if anomaly_score >= AE_ANOMALY_THRESHOLD:
        return (
            "ANOMALOUS_TRAFFIC",
            float(
                anomaly_score
            ),
            "Personalized Autoencoder",
        )

    benign_results = [
        result
        for result
        in classifiers
        if (
            result is not None
            and
            result.predicted_class
            .strip()
            .upper()
            ==
            "BENIGN"
        )
    ]

    if benign_results:
        best = max(
            benign_results,
            key=lambda result:
                result.confidence,
        )

        return (
            "BENIGN",
            float(
                best.confidence
            ),
            best.model,
        )

    return (
        "BENIGN",
        float(
            1.0
            -
            anomaly_score
        ),
        "Risk Fusion",
    )


def fuse_risk(
    rf_result: Optional[ClassificationResult],
    fedprox_result: Optional[ClassificationResult],
    cnn_result: Optional[ClassificationResult],
    ae_result: Optional[AutoencoderResult],
) -> RiskResult:
    rf_risk = (
        classification_risk(
            rf_result
        )
        if rf_result is not None
        else 0.0
    )

    fedprox_risk = (
        classification_risk(
            fedprox_result
        )
        if fedprox_result is not None
        else 0.0
    )

    cnn_risk = (
        classification_risk(
            cnn_result
        )
        if cnn_result is not None
        else 0.0
    )

    ae_risk = (
        ae_result.anomaly_score
        if ae_result is not None
        else 0.0
    )

    # Renormalize if a fast path intentionally skipped deep models.
    available_weights = 0.0
    weighted_sum = 0.0

    if rf_result is not None:
        available_weights += RISK_WEIGHTS[
            "rf"
        ]

        weighted_sum += (
            RISK_WEIGHTS[
                "rf"
            ]
            *
            rf_risk
        )

    if fedprox_result is not None:
        available_weights += RISK_WEIGHTS[
            "fedprox70"
        ]

        weighted_sum += (
            RISK_WEIGHTS[
                "fedprox70"
            ]
            *
            fedprox_risk
        )

    if cnn_result is not None:
        available_weights += RISK_WEIGHTS[
            "cnn_bilstm"
        ]

        weighted_sum += (
            RISK_WEIGHTS[
                "cnn_bilstm"
            ]
            *
            cnn_risk
        )

    if ae_result is not None:
        available_weights += RISK_WEIGHTS[
            "autoencoder"
        ]

        weighted_sum += (
            RISK_WEIGHTS[
                "autoencoder"
            ]
            *
            ae_risk
        )

    if available_weights <= 0.0:
        raise RuntimeError(
            "Risk fusion received no model outputs."
        )

    risk_score = float(
        np.clip(
            weighted_sum
            /
            available_weights,
            0.0,
            1.0,
        )
    )

    final_class, confidence, source = (
        choose_final_class(
            classifiers=[
                rf_result,
                fedprox_result,
                cnn_result,
            ],
            anomaly_score=ae_risk,
        )
    )

    return RiskResult(
        risk_score=risk_score,
        severity=
            severity_from_score(
                risk_score
            ),
        final_class=final_class,
        confidence=confidence,
        decision_source=source,
    )


# ============================================================
# ADAPTIVE GATE
# ============================================================

def should_take_fast_rf_path(
    rf_result: ClassificationResult,
) -> bool:
    return (
        rf_result.predicted_class
        .strip()
        .upper()
        !=
        "BENIGN"
        and
        rf_result.confidence
        >=
        RF_FAST_ATTACK_THRESHOLD
    )


# ============================================================
# FULL DETECTION ENGINE
# ============================================================

class FullDetectionEngine:

    def __init__(
        self,
        client_id: int,
        rf_predictor: CallableClassifierPredictor,
        autoencoder_predictor: CallableAutoencoderPredictor,
        fedprox70_predictor: CallableClassifierPredictor,
        cnn_predictor: FrozenCNNBiLSTMPredictor,
    ):
        if client_id not in CLIENT_ATTACKS:
            raise ValueError(
                f"client_id must be one of "
                f"{sorted(CLIENT_ATTACKS.keys())}."
            )

        self.client_id = client_id
        self.rf_predictor = (
            rf_predictor
        )
        self.autoencoder_predictor = (
            autoencoder_predictor
        )
        self.fedprox70_predictor = (
            fedprox70_predictor
        )
        self.cnn_predictor = (
            cnn_predictor
        )

    def detect(
        self,
        current_flow_70: np.ndarray,
        sequence_20x70: np.ndarray,
    ) -> DetectionAlert:
        # ----------------------------------------------------
        # 1. RANDOM FOREST — FAST FIRST LINE
        # ----------------------------------------------------

        rf_result = (
            self.rf_predictor.predict(
                current_flow_70
            )
        )

        # ----------------------------------------------------
        # 2. ADAPTIVE GATE
        # ----------------------------------------------------

        if should_take_fast_rf_path(
            rf_result
        ):
            risk_result = fuse_risk(
                rf_result=
                    rf_result,
                fedprox_result=
                    None,
                cnn_result=
                    None,
                ae_result=
                    None,
            )

            return self._build_alert(
                processing_path=
                    "FAST_ATTACK_PATH",
                risk_result=
                    risk_result,
                rf_result=
                    rf_result,
                ae_result=
                    None,
                fedprox_result=
                    None,
                cnn_result=
                    None,
            )

        # ----------------------------------------------------
        # 3. DEEP ANALYSIS
        # ----------------------------------------------------

        ae_result = (
            self.autoencoder_predictor.predict(
                current_flow_70
            )
        )

        fedprox_result = (
            self.fedprox70_predictor.predict(
                current_flow_70
            )
        )

        cnn_result = (
            self.cnn_predictor.predict(
                sequence_20x70
            )
        )

        # ----------------------------------------------------
        # 4. RISK FUSION
        # ----------------------------------------------------

        risk_result = fuse_risk(
            rf_result=
                rf_result,
            fedprox_result=
                fedprox_result,
            cnn_result=
                cnn_result,
            ae_result=
                ae_result,
        )

        # ----------------------------------------------------
        # 5. LOCAL IDS ALERT
        # ----------------------------------------------------

        return self._build_alert(
            processing_path=
                "DEEP_ANALYSIS",
            risk_result=
                risk_result,
            rf_result=
                rf_result,
            ae_result=
                ae_result,
            fedprox_result=
                fedprox_result,
            cnn_result=
                cnn_result,
        )

    def _build_alert(
        self,
        processing_path: str,
        risk_result: RiskResult,
        rf_result,
        ae_result,
        fedprox_result,
        cnn_result,
    ) -> DetectionAlert:
        return DetectionAlert(
            timestamp_utc=
                datetime.now(
                    timezone.utc
                ).isoformat(),
            client_id=
                self.client_id,
            classification=
                risk_result.final_class,
            severity=
                risk_result.severity,
            risk_score=
                risk_result.risk_score,
            confidence=
                risk_result.confidence,
            decision_source=
                risk_result.decision_source,
            processing_path=
                processing_path,
            rf=(
                asdict(
                    rf_result
                )
                if rf_result is not None
                else None
            ),
            autoencoder=(
                asdict(
                    ae_result
                )
                if ae_result is not None
                else None
            ),
            fedprox70=(
                asdict(
                    fedprox_result
                )
                if fedprox_result is not None
                else None
            ),
            cnn_bilstm=(
                asdict(
                    cnn_result
                )
                if cnn_result is not None
                else None
            ),
        )


# ============================================================
# DEMONSTRATION ADAPTERS
#
# These are ONLY for integration smoke testing.
# They do NOT replace Phase14A / Phase14B / FedProx70 models.
# ============================================================

def build_demo_engine(
    client_id: int,
    inverse_mapping,
    cnn_predictor,
) -> FullDetectionEngine:
    # Moderate RF PortScan-like signal:
    def demo_rf(flow):
        return {
            "class_id": 5,
            "confidence": 0.72,
        }

    # Supporting novelty signal.
    def demo_ae(flow):
        return 0.82

    # Global MLP signal.
    def demo_fedprox(flow):
        return {
            "class_id": 5,
            "confidence": 0.88,
        }

    return FullDetectionEngine(
        client_id=client_id,
        rf_predictor=
            CallableClassifierPredictor(
                model_name=
                    "Random Forest",
                predictor=
                    demo_rf,
                inverse_mapping=
                    inverse_mapping,
            ),
        autoencoder_predictor=
            CallableAutoencoderPredictor(
                score_fn=
                    demo_ae,
                threshold=
                    AE_ANOMALY_THRESHOLD,
            ),
        fedprox70_predictor=
            CallableClassifierPredictor(
                model_name=
                    "FedProx70 MLP",
                predictor=
                    demo_fedprox,
                inverse_mapping=
                    inverse_mapping,
            ),
        cnn_predictor=
            cnn_predictor,
    )


# ============================================================
# CONFIG / REPORTING
# ============================================================

def save_integration_config(
    active_features,
):
    payload = {
        "phase":
            "14G",

        "architecture":
            [
                "Random Forest",
                "Adaptive Gate",
                "Personalized Autoencoder",
                "FedProx70 MLP",
                "Frozen Round-20 CNN-BiLSTM",
                "Risk Fusion",
                "Severity Mapping",
                "Local IDS Alert",
            ],

        "cnn_checkpoint":
            str(
                CNN_CHECKPOINT
            ),

        "cnn_round":
            20,

        "cnn_fedprox_mu":
            0.01,

        "cnn_sequence_length":
            SEQUENCE_LENGTH,

        "feature_count":
            EXPECTED_FEATURES,

        "feature_names":
            active_features,

        "risk_weights":
            RISK_WEIGHTS,

        "rf_fast_attack_threshold":
            RF_FAST_ATTACK_THRESHOLD,

        "ae_anomaly_threshold":
            AE_ANOMALY_THRESHOLD,

        "deep_classifier_threshold":
            DEEP_CLASSIFIER_THRESHOLD,

        "severity_thresholds":
            SEVERITY_THRESHOLDS,

        "locked_test_used":
            False,

        "scientific_label":
            (
                "SOURCE-ORDERED CNN-BiLSTM sequences; "
                "NOT timestamp-confirmed temporal sequences."
            ),
    }

    with open(
        INTEGRATION_CONFIG_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=4,
        )


# ============================================================
# DEMO
# ============================================================

def run_demo(
    client_id: int,
    device: torch.device,
):
    (
        label_mapping,
        inverse_mapping,
        active_features,
    ) = load_label_mapping()

    cnn_predictor = (
        FrozenCNNBiLSTMPredictor(
            checkpoint_path=
                CNN_CHECKPOINT,
            inverse_mapping=
                inverse_mapping,
            device=
                device,
        )
    )

    save_integration_config(
        active_features
    )

    # --------------------------------------------------------
    # Demo only:
    # use a zero sequence so the final CNN checkpoint is
    # actually invoked and the entire pipeline executes.
    #
    # For real deployment, supply the latest transformed
    # 70-feature flow and its corresponding 20x70 sequence.
    # --------------------------------------------------------

    current_flow = np.zeros(
        EXPECTED_FEATURES,
        dtype=np.float32,
    )

    sequence = np.zeros(
        (
            SEQUENCE_LENGTH,
            EXPECTED_FEATURES,
        ),
        dtype=np.float32,
    )

    engine = build_demo_engine(
        client_id=
            client_id,
        inverse_mapping=
            inverse_mapping,
        cnn_predictor=
            cnn_predictor,
    )

    alert = engine.detect(
        current_flow_70=
            current_flow,
        sequence_20x70=
            sequence,
    )

    alert_dict = asdict(
        alert
    )

    with open(
        DEMO_ALERT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            alert_dict,
            file,
            indent=4,
        )

    print()
    print("=" * 118)

    print(
        PHASE_NAME
    )

    print("=" * 118)

    print(
        f"Device                    : {device}"
    )

    print(
        f"Client                    : {client_id}"
    )

    print(
        f"CNN checkpoint            : {CNN_CHECKPOINT}"
    )

    print(
        f"CNN frozen round          : "
        f"{cnn_predictor.checkpoint_round}"
    )

    print(
        f"CNN FedProx mu            : "
        f"{cnn_predictor.checkpoint_mu}"
    )

    print(
        f"Risk weights              : {RISK_WEIGHTS}"
    )

    print()

    print(
        "PIPELINE"
    )

    print("-" * 118)

    print(
        "Incoming flow / source-ordered sequence"
    )

    print(
        "        ↓"
    )

    print(
        "Random Forest"
    )

    print(
        "        ↓"
    )

    print(
        "Adaptive Gate"
    )

    print(
        "        ↓"
    )

    print(
        "Personalized Autoencoder + "
        "FedProx70 MLP + Frozen Round-20 CNN-BiLSTM"
    )

    print(
        "        ↓"
    )

    print(
        "Risk Fusion"
    )

    print(
        "        ↓"
    )

    print(
        "NORMAL / LOW / MEDIUM / HIGH / CRITICAL"
    )

    print(
        "        ↓"
    )

    print(
        "Local IDS Alert"
    )

    print()

    print(
        "DEMO ALERT"
    )

    print("-" * 118)

    print(
        json.dumps(
            alert_dict,
            indent=4,
        )
    )

    print()

    print(
        f"Demo alert saved          : "
        f"{DEMO_ALERT_FILE}"
    )

    print(
        f"Integration config        : "
        f"{INTEGRATION_CONFIG_FILE}"
    )

    print()

    print(
        "LOCKED TEST USED : NO"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Demo RF / AE / FedProx70 callables are placeholders "
        "for smoke testing only."
    )

    print(
        "The CNN-BiLSTM is the REAL frozen Phase14F2 Round-20 model."
    )

    print(
        "Wire the existing Phase14A RF, Phase14B AE and "
        "FedProx70 inference functions into the same adapter contracts "
        "for production/full evaluation."
    )


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=PHASE_NAME
    )

    parser.add_argument(
        "--client-id",
        type=int,
        default=3,
        choices=[
            1,
            2,
            3,
            4,
            5,
        ],
        help=(
            "Client whose local RF/AE adapters are being integrated."
        ),
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Run a complete integration smoke test. "
            "RF/AE/FedProx70 use deterministic demo adapters; "
            "CNN-BiLSTM uses the real frozen Round-20 checkpoint."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARTIFACT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if not args.demo:
        print()
        print("=" * 118)

        print(
            PHASE_NAME
        )

        print("=" * 118)

        print(
            "Integration engine is ready."
        )

        print()

        print(
            "Run the smoke test with:"
        )

        print()

        print(
            "python scripts\\phase14g_full_detection_integration.py "
            "--demo --client-id 3"
        )

        print()

        print(
            "For real traffic, instantiate FullDetectionEngine "
            "and wire your existing Phase14A RF, Phase14B AE, "
            "and FedProx70 inference functions through the provided "
            "adapter contracts."
        )

        print()

        print(
            f"Frozen CNN checkpoint:"
        )

        print(
            CNN_CHECKPOINT
        )

        return

    run_demo(
        client_id=
            args.client_id,
        device=
            device,
    )


if __name__ == "__main__":
    main()
