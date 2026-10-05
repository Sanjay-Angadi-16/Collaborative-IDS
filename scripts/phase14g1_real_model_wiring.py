from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# PROJECT ROOT / SCRIPT IMPORTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import phase9_fedavg70 as phase4
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "\nCould not import scripts\\phase9_fedavg70.py\n"
        "Keep this file inside the scripts folder."
    ) from exc

try:
    import phase14g_full_detection_integration as phase14g
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "\nCould not import scripts\\phase14g_full_detection_integration.py\n"
        "Phase14G1 intentionally reuses the frozen Phase14G "
        "adaptive-gate / risk-fusion engine."
    ) from exc


# ============================================================
# PHASE
# ============================================================

PHASE_NAME = (
    "PHASE 14G1 — REAL MODEL WIRING"
)


# ============================================================
# FROZEN MODEL / INFERENCE CONTRACT
# ============================================================

EXPECTED_FEATURES = 70
SEQUENCE_LENGTH = 20
NUM_CLASSES = 6

RF_FAST_ATTACK_THRESHOLD = 0.95

# Phase14G risk fusion expects AE score in [0,1] and uses 0.70
# as the anomaly decision point. Phase14B thresholds are raw
# reconstruction errors, therefore we map:
#
#   raw_error == personalized_threshold  -> normalized score 0.70
#
# This is an engineering normalization for fusion, not a
# calibrated probability.
AE_FUSION_DECISION_SCORE = 0.70

RISK_WEIGHTS = {
    "rf": 0.20,
    "fedprox70": 0.30,
    "cnn_bilstm": 0.35,
    "autoencoder": 0.15,
}

CLIENT_ATTACKS = {
    1: "DoS Hulk",
    2: "DDoS",
    3: "PortScan",
    4: "DoS GoldenEye",
    5: "FTP-Patator",
}

# Frozen Phase14B validation-derived raw reconstruction thresholds.
AE_FROZEN_THRESHOLDS = {
    1: 0.645766,
    2: 0.458977,
    3: 0.417044,
    4: 0.602649,
    5: 0.787584,
}


# ============================================================
# REAL ARTIFACT PATHS
# ============================================================

RF_MODEL_TEMPLATE = (
    PROJECT_ROOT
    / "artifacts"
    / "detection"
    / "random_forest_client_{client_id}.joblib"
)

FEDPROX70_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "phase10"
    / "fedprox70"
    / "non_iid"
    / "seed_42"
    / "fedprox70_model.pt"
)

CNN_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "detection"
    / "phase14f2"
    / "cnn_bilstm_fedprox_mu001_best_round_1_to_30.pt"
)

VALIDATION_SEQUENCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "sequences"
    / "validation_sequences.npz"
)


# ============================================================
# OUTPUTS
# ============================================================

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "phase14"
    / "phase14g1"
)

ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "detection"
    / "phase14g1"
)

REAL_ALERT_FILE = (
    RESULT_ROOT
    / "phase14g1_real_alert.json"
)

WIRING_AUDIT_FILE = (
    RESULT_ROOT
    / "phase14g1_real_model_wiring_audit.json"
)

INTEGRATION_MANIFEST_FILE = (
    ARTIFACT_ROOT
    / "phase14g1_real_model_manifest.json"
)


# ============================================================
# HELPERS
# ============================================================

def separator(width: int = 118) -> None:
    print()
    print("=" * width)


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
            json_safe(payload),
            file,
            indent=4,
        )


def safe_torch_load(
    path: Path,
    device: torch.device,
):
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=device,
        )


def load_project_schema():
    (
        _scaler,
        _feature_columns,
        label_mapping,
        _active_indices,
        active_features,
    ) = phase4.load_preprocessing()

    if len(active_features) != EXPECTED_FEATURES:
        raise RuntimeError(
            f"\nExpected {EXPECTED_FEATURES} active features; "
            f"found {len(active_features)}."
        )

    inverse_mapping = {
        int(class_id): str(label)
        for label, class_id
        in label_mapping.items()
    }

    if len(inverse_mapping) != NUM_CLASSES:
        raise RuntimeError(
            f"\nExpected {NUM_CLASSES} classes; "
            f"found {len(inverse_mapping)}."
        )

    return (
        label_mapping,
        inverse_mapping,
        list(active_features),
    )


# ============================================================
# REAL PHASE14A LOCAL RANDOM FOREST
# ============================================================

class RealRandomForestPredictor:

    def __init__(
        self,
        client_id: int,
        label_mapping: Dict[str, int],
        inverse_mapping: Dict[int, str],
    ):
        self.client_id = int(
            client_id
        )

        self.label_mapping = (
            label_mapping
        )

        self.inverse_mapping = (
            inverse_mapping
        )

        self.seen_attack = (
            CLIENT_ATTACKS[
                self.client_id
            ]
        )

        self.seen_attack_global_id = int(
            label_mapping[
                self.seen_attack
            ]
        )

        self.path = Path(
            str(
                RF_MODEL_TEMPLATE
            ).format(
                client_id=
                    self.client_id
            )
        )

        if not self.path.exists():
            raise FileNotFoundError(
                "\nPhase14A Random Forest model missing:\n"
                f"{self.path}"
            )

        loaded_artifact = joblib.load(
            self.path
        )

        self.artifact_container_type = (
            type(
                loaded_artifact
            ).__name__
        )

        self.artifact_keys = None

        if isinstance(
            loaded_artifact,
            dict,
        ):
            self.artifact_keys = list(
                loaded_artifact.keys()
            )

        self.model = (
            self._extract_predict_proba_model(
                loaded_artifact
            )
        )

        if self.model is None:
            details = (
                f"container_type={self.artifact_container_type}"
            )

            if self.artifact_keys is not None:
                details += (
                    f", keys={self.artifact_keys}"
                )

            raise TypeError(
                "\nPhase14A Random Forest artifact was loaded, "
                "but no estimator implementing predict_proba() "
                "could be found inside it.\n"
                f"Artifact: {self.path}\n"
                f"Details : {details}"
            )

        if not hasattr(
            self.model,
            "classes_",
        ):
            raise TypeError(
                "\nExtracted Random Forest estimator does not "
                "expose classes_."
            )

        self.estimator_type = (
            type(
                self.model
            ).__name__
        )

    @staticmethod
    def _extract_predict_proba_model(
        artifact: Any,
        depth: int = 0,
    ):
        """
        Phase14A may persist either the estimator directly or a
        metadata dictionary containing the estimator. Recover the
        first object that provides the sklearn predict_proba contract.

        Search is intentionally bounded to avoid walking arbitrary
        object graphs.
        """
        if artifact is None:
            return None

        if hasattr(
            artifact,
            "predict_proba",
        ):
            return artifact

        if depth >= 3:
            return None

        if isinstance(
            artifact,
            dict,
        ):
            # Prefer conventional model keys before scanning metadata.
            preferred_keys = (
                "model",
                "classifier",
                "rf",
                "random_forest",
                "random_forest_model",
                "estimator",
                "pipeline",
            )

            for key in preferred_keys:
                if key not in artifact:
                    continue

                candidate = (
                    RealRandomForestPredictor
                    ._extract_predict_proba_model(
                        artifact[
                            key
                        ],
                        depth=
                            depth
                            +
                            1,
                    )
                )

                if candidate is not None:
                    return candidate

            for value in artifact.values():
                candidate = (
                    RealRandomForestPredictor
                    ._extract_predict_proba_model(
                        value,
                        depth=
                            depth
                            +
                            1,
                    )
                )

                if candidate is not None:
                    return candidate

            return None

        if isinstance(
            artifact,
            (list, tuple),
        ):
            for value in artifact:
                candidate = (
                    RealRandomForestPredictor
                    ._extract_predict_proba_model(
                        value,
                        depth=
                            depth
                            +
                            1,
                    )
                )

                if candidate is not None:
                    return candidate

        return None

    def _map_model_class_to_global_id(
        self,
        model_class: Any,
        model_classes: np.ndarray,
    ) -> int:
        # String labels can be mapped directly.
        if isinstance(
            model_class,
            (str, np.str_),
        ):
            label = str(
                model_class
            ).strip()

            if label not in self.label_mapping:
                raise KeyError(
                    f"\nRF emitted unknown string class: {label}"
                )

            return int(
                self.label_mapping[
                    label
                ]
            )

        numeric_class = int(
            model_class
        )

        numeric_classes = {
            int(item)
            for item
            in model_classes
            if not isinstance(
                item,
                (str, np.str_),
            )
        }

        # Phase14A local RF may be stored with binary labels
        # {0,1}. In that representation:
        # 0 -> BENIGN
        # 1 -> this client's own seen attack.
        if numeric_classes.issubset(
            {0, 1}
        ):
            if numeric_class == 0:
                return int(
                    self.label_mapping[
                        "BENIGN"
                    ]
                )

            return self.seen_attack_global_id

        # Otherwise treat the value as the global six-class id.
        if numeric_class not in self.inverse_mapping:
            raise KeyError(
                "\nRF emitted numeric class outside global mapping: "
                f"{numeric_class}"
            )

        return numeric_class

    def predict(
        self,
        flow_70: np.ndarray,
    ) -> phase14g.ClassificationResult:
        flow = np.asarray(
            flow_70,
            dtype=np.float32,
        ).reshape(
            1,
            -1,
        )

        if flow.shape != (
            1,
            EXPECTED_FEATURES,
        ):
            raise ValueError(
                f"\nRF expected (1,{EXPECTED_FEATURES}); "
                f"found {flow.shape}."
            )

        probabilities = np.asarray(
            self.model.predict_proba(
                flow
            )
        )[0]

        model_classes = np.asarray(
            self.model.classes_
        )

        best_position = int(
            np.argmax(
                probabilities
            )
        )

        global_class_id = (
            self._map_model_class_to_global_id(
                model_classes[
                    best_position
                ],
                model_classes,
            )
        )

        return phase14g.ClassificationResult(
            model="Random Forest",
            predicted_class=
                self.inverse_mapping[
                    global_class_id
                ],
            class_id=
                global_class_id,
            confidence=
                float(
                    probabilities[
                        best_position
                    ]
                ),
        )


# ============================================================
# REAL PHASE10 FEDPROX70 MLP
# ============================================================

class RealFedProx70Predictor:

    def __init__(
        self,
        checkpoint_path: Path,
        inverse_mapping: Dict[int, str],
        device: torch.device,
    ):
        self.path = Path(
            checkpoint_path
        )

        self.device = (
            device
        )

        self.inverse_mapping = (
            inverse_mapping
        )

        if not self.path.exists():
            raise FileNotFoundError(
                "\nFedProx70 checkpoint missing:\n"
                f"{self.path}"
            )

        checkpoint = safe_torch_load(
            self.path,
            device,
        )

        if not isinstance(
            checkpoint,
            dict,
        ):
            raise TypeError(
                "\nFedProx70 checkpoint must be a dictionary."
            )

        if (
            checkpoint.get(
                "algorithm"
            )
            !=
            "FedProx70"
        ):
            raise RuntimeError(
                "\nCheckpoint is not labelled FedProx70."
            )

        input_dim = int(
            checkpoint.get(
                "input_dim",
                EXPECTED_FEATURES,
            )
        )

        num_classes = int(
            checkpoint.get(
                "num_classes",
                NUM_CLASSES,
            )
        )

        if input_dim != EXPECTED_FEATURES:
            raise RuntimeError(
                f"\nFedProx70 expected 70 features; "
                f"checkpoint reports {input_dim}."
            )

        if num_classes != NUM_CLASSES:
            raise RuntimeError(
                f"\nFedProx70 expected 6 classes; "
                f"checkpoint reports {num_classes}."
            )

        self.model = phase4.IDSMLP(
            input_dim=
                input_dim,
            num_classes=
                num_classes,
        ).to(
            device
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.model.eval()

        self.seed = checkpoint.get(
            "seed"
        )

        self.mu = checkpoint.get(
            "fedprox_mu"
        )

    @torch.no_grad()
    def predict(
        self,
        flow_70: np.ndarray,
    ) -> phase14g.ClassificationResult:
        flow = np.asarray(
            flow_70,
            dtype=np.float32,
        ).reshape(
            1,
            -1,
        )

        if flow.shape != (
            1,
            EXPECTED_FEATURES,
        ):
            raise ValueError(
                "\nFedProx70 input shape mismatch."
            )

        tensor = torch.from_numpy(
            flow
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

        return phase14g.ClassificationResult(
            model="FedProx70 MLP",
            predicted_class=
                self.inverse_mapping[
                    class_id
                ],
            class_id=
                class_id,
            confidence=
                confidence,
        )


# ============================================================
# PHASE14B AUTOENCODER ARTIFACT DISCOVERY
# ============================================================

def autoencoder_candidate_paths(
    client_id: int,
) -> Iterable[Path]:
    detection_root = (
        PROJECT_ROOT
        / "artifacts"
        / "detection"
    )

    explicit_names = [
        f"autoencoder_client_{client_id}.pt",
        f"personalized_autoencoder_client_{client_id}.pt",
        f"personalized_ae_client_{client_id}.pt",
        f"ae_client_{client_id}.pt",
        f"autoencoder_client_{client_id}.pth",
        f"personalized_autoencoder_client_{client_id}.pth",
        f"personalized_ae_client_{client_id}.pth",
        f"ae_client_{client_id}.pth",
        f"autoencoder_client_{client_id}.joblib",
        f"personalized_autoencoder_client_{client_id}.joblib",
        f"autoencoder_client_{client_id}.pkl",
        f"personalized_autoencoder_client_{client_id}.pkl",
    ]

    seen = set()

    for name in explicit_names:
        path = (
            detection_root
            / name
        )

        if path not in seen:
            seen.add(
                path
            )

            yield path

    if detection_root.exists():
        for path in sorted(
            detection_root.rglob(
                "*"
            )
        ):
            if not path.is_file():
                continue

            lower = path.name.lower()

            if (
                "autoencoder" not in lower
                and
                not lower.startswith(
                    "ae_"
                )
            ):
                continue

            if (
                f"client_{client_id}"
                not in lower
                and
                f"client{client_id}"
                not in lower
            ):
                continue

            if path.suffix.lower() not in {
                ".pt",
                ".pth",
                ".joblib",
                ".pkl",
            }:
                continue

            if path not in seen:
                seen.add(
                    path
                )

                yield path


def discover_autoencoder_path(
    client_id: int,
    explicit_path: Optional[str],
) -> Tuple[
    Path,
    list[str],
]:
    searched = []

    if explicit_path:
        path = Path(
            explicit_path
        )

        if not path.is_absolute():
            path = (
                PROJECT_ROOT
                / path
            )

        searched.append(
            str(
                path
            )
        )

        if not path.exists():
            raise FileNotFoundError(
                "\nExplicit Autoencoder artifact does not exist:\n"
                f"{path}"
            )

        return (
            path,
            searched,
        )

    for candidate in autoencoder_candidate_paths(
        client_id
    ):
        searched.append(
            str(
                candidate
            )
        )

        if candidate.exists():
            return (
                candidate,
                searched,
            )

    raise FileNotFoundError(
        "\nCould not automatically locate the Phase14B "
        f"Autoencoder model for client {client_id}.\n\n"
        "Searched candidate paths including:\n"
        +
        "\n".join(
            f"  {path}"
            for path in searched
        )
        +
        "\n\nRun:\n"
        "  dir artifacts\\detection\\*auto* /s /b\n\n"
        "Then pass the exact result with:\n"
        "  --ae-model <path>"
    )


# ============================================================
# AUTOENCODER DYNAMIC PYTORCH ARCHITECTURE RECOVERY
# ============================================================

def candidate_phase14b_scripts() -> list[Path]:
    candidates = []

    patterns = [
        "phase14b*.py",
        "*autoencoder*.py",
        "*auto_encoder*.py",
    ]

    for pattern in patterns:
        candidates.extend(
            SCRIPT_DIR.glob(
                pattern
            )
        )

    unique = []

    for path in candidates:
        if path.resolve() == Path(
            __file__
        ).resolve():
            continue

        if path not in unique:
            unique.append(
                path
            )

    return sorted(
        unique
    )


def import_module_from_path(
    path: Path,
):
    module_name = (
        "_phase14g1_dynamic_"
        +
        path.stem
        .replace(
            "-",
            "_",
        )
    )

    spec = (
        importlib.util
        .spec_from_file_location(
            module_name,
            path,
        )
    )

    if (
        spec is None
        or
        spec.loader is None
    ):
        raise ImportError(
            f"Cannot import {path}"
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


def instantiate_autoencoder_class(
    cls,
) -> Optional[nn.Module]:
    # Try common constructors without inventing hidden/latent
    # dimensions. We only supply the known 70-feature input.
    attempts = [
        {},
        {
            "input_dim":
                EXPECTED_FEATURES
        },
        {
            "input_size":
                EXPECTED_FEATURES
        },
        {
            "feature_count":
                EXPECTED_FEATURES
        },
        {
            "n_features":
                EXPECTED_FEATURES
        },
        {
            "num_features":
                EXPECTED_FEATURES
        },
    ]

    for kwargs in attempts:
        try:
            model = cls(
                **kwargs
            )

            if isinstance(
                model,
                nn.Module,
            ):
                return model
        except Exception:
            continue

    return None


def reconstruct_autoencoder_from_phase14b_script(
    state_dict: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[
    nn.Module,
    str,
]:
    errors = []

    for script_path in candidate_phase14b_scripts():
        try:
            module = import_module_from_path(
                script_path
            )
        except Exception as exc:
            errors.append(
                f"{script_path.name}: import failed: {exc}"
            )
            continue

        candidate_classes = []

        for name, obj in vars(
            module
        ).items():
            if not inspect.isclass(
                obj
            ):
                continue

            if not issubclass(
                obj,
                nn.Module,
            ):
                continue

            if obj is nn.Module:
                continue

            if (
                "autoencoder"
                in name.lower()
                or
                "auto_encoder"
                in name.lower()
            ):
                candidate_classes.append(
                    (
                        name,
                        obj,
                    )
                )

        for class_name, cls in candidate_classes:
            model = instantiate_autoencoder_class(
                cls
            )

            if model is None:
                continue

            try:
                model.load_state_dict(
                    state_dict,
                    strict=True,
                )

                model = model.to(
                    device
                )

                model.eval()

                return (
                    model,
                    (
                        f"{script_path.name}::"
                        f"{class_name}"
                    ),
                )
            except Exception as exc:
                errors.append(
                    f"{script_path.name}::{class_name}: {exc}"
                )

    raise RuntimeError(
        "\nThe Autoencoder artifact contains a state_dict, "
        "but its Phase14B architecture could not be reconstructed "
        "automatically.\n\n"
        "The loader searched Phase14B/autoencoder scripts and tried "
        "their nn.Module Autoencoder classes.\n\n"
        "Diagnostic attempts:\n"
        +
        "\n".join(
            f"  {item}"
            for item in errors[
                :20
            ]
        )
    )


# ============================================================
# REAL PHASE14B PERSONALIZED AUTOENCODER
# ============================================================

class RealPersonalizedAutoencoderPredictor:

    def __init__(
        self,
        client_id: int,
        device: torch.device,
        explicit_model_path: Optional[str] = None,
    ):
        self.client_id = int(
            client_id
        )

        self.device = (
            device
        )

        (
            self.path,
            self.searched_paths,
        ) = discover_autoencoder_path(
            client_id=
                self.client_id,
            explicit_path=
                explicit_model_path,
        )

        self.threshold = float(
            AE_FROZEN_THRESHOLDS[
                self.client_id
            ]
        )

        self.backend = None
        self.model = None
        self.architecture_source = None

        suffix = (
            self.path
            .suffix
            .lower()
        )

        if suffix in {
            ".joblib",
            ".pkl",
        }:
            self.model = joblib.load(
                self.path
            )

            if not hasattr(
                self.model,
                "predict",
            ):
                raise TypeError(
                    "\nJoblib Autoencoder artifact must "
                    "implement predict()."
                )

            self.backend = (
                "joblib_predict_reconstruction"
            )

            self.architecture_source = (
                type(
                    self.model
                ).__name__
            )

            return

        if suffix not in {
            ".pt",
            ".pth",
        }:
            raise ValueError(
                f"\nUnsupported Autoencoder extension: {suffix}"
            )

        payload = safe_torch_load(
            self.path,
            device,
        )

        # Full serialized nn.Module.
        if isinstance(
            payload,
            nn.Module,
        ):
            self.model = (
                payload
                .to(
                    device
                )
            )

            self.model.eval()

            self.backend = (
                "torch_serialized_module"
            )

            self.architecture_source = (
                type(
                    self.model
                ).__name__
            )

            return

        if not isinstance(
            payload,
            dict,
        ):
            raise TypeError(
                "\nUnsupported Autoencoder torch payload."
            )

        # Prefer threshold persisted with the model if available.
        for key in (
            "threshold",
            "anomaly_threshold",
            "reconstruction_threshold",
        ):
            if key in payload:
                try:
                    self.threshold = float(
                        payload[
                            key
                        ]
                    )
                except Exception:
                    pass

        for model_key in (
            "model",
            "autoencoder",
            "ae_model",
        ):
            if (
                model_key in payload
                and
                isinstance(
                    payload[
                        model_key
                    ],
                    nn.Module,
                )
            ):
                self.model = (
                    payload[
                        model_key
                    ]
                    .to(
                        device
                    )
                )

                self.model.eval()

                self.backend = (
                    f"torch_payload_{model_key}"
                )

                self.architecture_source = (
                    type(
                        self.model
                    ).__name__
                )

                return

        state_dict = None

        for key in (
            "model_state_dict",
            "state_dict",
            "autoencoder_state_dict",
        ):
            if key in payload:
                candidate = (
                    payload[
                        key
                    ]
                )

                if isinstance(
                    candidate,
                    dict,
                ):
                    state_dict = (
                        candidate
                    )

                    break

        if state_dict is None:
            # Some training scripts save a raw state_dict directly.
            if (
                payload
                and
                all(
                    isinstance(
                        value,
                        torch.Tensor,
                    )
                    for value in payload.values()
                )
            ):
                state_dict = payload

        if state_dict is None:
            raise RuntimeError(
                "\nCould not find an Autoencoder model or state_dict "
                f"inside:\n{self.path}"
            )

        (
            self.model,
            self.architecture_source,
        ) = (
            reconstruct_autoencoder_from_phase14b_script(
                state_dict=
                    state_dict,
                device=
                    device,
            )
        )

        self.backend = (
            "torch_state_dict_reconstructed"
        )

    def _torch_reconstruct(
        self,
        flow: np.ndarray,
    ) -> np.ndarray:
        tensor = (
            torch.from_numpy(
                flow.reshape(
                    1,
                    -1,
                )
            )
            .to(
                self.device
            )
        )

        self.model.eval()

        with torch.no_grad():
            output = self.model(
                tensor
            )

        if isinstance(
            output,
            (tuple, list),
        ):
            output = output[
                0
            ]

        if isinstance(
            output,
            dict,
        ):
            for key in (
                "reconstruction",
                "output",
                "decoded",
            ):
                if key in output:
                    output = output[
                        key
                    ]
                    break

        if not isinstance(
            output,
            torch.Tensor,
        ):
            raise TypeError(
                "\nAutoencoder forward() did not return a tensor "
                "reconstruction."
            )

        return (
            output
            .detach()
            .cpu()
            .numpy()
            .reshape(
                -1
            )
            .astype(
                np.float32,
                copy=False,
            )
        )

    def raw_reconstruction_error(
        self,
        flow_70: np.ndarray,
    ) -> float:
        flow = np.asarray(
            flow_70,
            dtype=np.float32,
        ).reshape(
            -1
        )

        if flow.shape != (
            EXPECTED_FEATURES,
        ):
            raise ValueError(
                "\nAutoencoder input must be a 70-feature flow."
            )

        if self.backend.startswith(
            "joblib"
        ):
            reconstructed = np.asarray(
                self.model.predict(
                    flow.reshape(
                        1,
                        -1,
                    )
                )
            ).reshape(
                -1
            )
        else:
            reconstructed = (
                self._torch_reconstruct(
                    flow
                )
            )

        if reconstructed.shape != (
            EXPECTED_FEATURES,
        ):
            raise RuntimeError(
                "\nAutoencoder reconstruction shape mismatch:\n"
                f"Expected ({EXPECTED_FEATURES},), "
                f"found {reconstructed.shape}."
            )

        mse = float(
            np.mean(
                (
                    flow
                    -
                    reconstructed
                )
                ** 2
            )
        )

        return mse

    def normalized_anomaly_score(
        self,
        flow_70: np.ndarray,
    ) -> float:
        raw_error = (
            self.raw_reconstruction_error(
                flow_70
            )
        )

        if self.threshold <= 0.0:
            raise RuntimeError(
                "\nAutoencoder threshold must be > 0."
            )

        # Personalized threshold maps exactly to the Phase14G
        # anomaly decision score of 0.70.
        normalized = (
            AE_FUSION_DECISION_SCORE
            *
            raw_error
            /
            self.threshold
        )

        return float(
            np.clip(
                normalized,
                0.0,
                1.0,
            )
        )

    def details(
        self,
        flow_70: np.ndarray,
    ) -> Dict[str, Any]:
        raw_error = (
            self.raw_reconstruction_error(
                flow_70
            )
        )

        normalized = float(
            np.clip(
                AE_FUSION_DECISION_SCORE
                *
                raw_error
                /
                self.threshold,
                0.0,
                1.0,
            )
        )

        return {
            "artifact":
                str(
                    self.path
                ),

            "backend":
                self.backend,

            "architecture_source":
                self.architecture_source,

            "raw_reconstruction_error":
                raw_error,

            "raw_threshold":
                self.threshold,

            "raw_is_anomaly":
                bool(
                    raw_error
                    >=
                    self.threshold
                ),

            "normalized_anomaly_score":
                normalized,

            "normalization":
                (
                    "clip(0.70 * raw_error / "
                    "personalized_threshold, 0, 1)"
                ),

            "normalized_score_is_probability":
                False,
        }


# ============================================================
# REAL CNN-BILSTM
# ============================================================

def build_real_cnn_predictor(
    inverse_mapping: Dict[int, str],
    device: torch.device,
):
    # Reuse Phase14G's already-verified loader. It checks that
    # the checkpoint is Round 20 and FedProx mu=0.01.
    return phase14g.FrozenCNNBiLSTMPredictor(
        checkpoint_path=
            CNN_CHECKPOINT,
        inverse_mapping=
            inverse_mapping,
        device=
            device,
    )


# ============================================================
# VALIDATION SAMPLE
# ============================================================

def load_real_validation_sample(
    client_id: int,
    label_mapping: Dict[str, int],
    inverse_mapping: Dict[int, str],
    sample_index: Optional[int],
    target_class: str,
):
    if not VALIDATION_SEQUENCE_FILE.exists():
        raise FileNotFoundError(
            "\nValidation sequence file missing:\n"
            f"{VALIDATION_SEQUENCE_FILE}"
        )

    with np.load(
        VALIDATION_SEQUENCE_FILE,
        allow_pickle=False,
    ) as data:
        X = data[
            "X"
        ].astype(
            np.float32,
            copy=False,
        )

        y = data[
            "y"
        ].astype(
            np.int64,
            copy=False,
        )

    if X.ndim != 3:
        raise RuntimeError(
            f"\nValidation X must be 3D; found {X.shape}."
        )

    if tuple(
        X.shape[
            1:
        ]
    ) != (
        SEQUENCE_LENGTH,
        EXPECTED_FEATURES,
    ):
        raise RuntimeError(
            "\nValidation sequence shape must be "
            f"(*,{SEQUENCE_LENGTH},{EXPECTED_FEATURES}); "
            f"found {X.shape}."
        )

    if sample_index is not None:
        index = int(
            sample_index
        )

        if (
            index < 0
            or
            index >= len(
                X
            )
        ):
            raise IndexError(
                f"\n--sample-index must be 0..{len(X)-1}."
            )
    else:
        if target_class.lower() == "auto":
            class_name = (
                CLIENT_ATTACKS[
                    client_id
                ]
            )
        else:
            class_name = (
                target_class
            )

        if class_name not in label_mapping:
            raise KeyError(
                "\nUnknown target class. Available labels:\n"
                +
                "\n".join(
                    sorted(
                        label_mapping.keys()
                    )
                )
            )

        target_id = int(
            label_mapping[
                class_name
            ]
        )

        matches = np.flatnonzero(
            y
            ==
            target_id
        )

        if len(
            matches
        ) == 0:
            raise RuntimeError(
                f"\nValidation contains no {class_name} sequence."
            )

        index = int(
            matches[
                0
            ]
        )

    sequence = np.asarray(
        X[
            index
        ],
        dtype=np.float32,
    )

    true_class_id = int(
        y[
            index
        ]
    )

    # The final row is the exact current flow whose context is
    # represented by the 20-flow CNN-BiLSTM sequence.
    current_flow = np.asarray(
        sequence[
            -1
        ],
        dtype=np.float32,
    )

    return {
        "sample_index":
            index,

        "true_class_id":
            true_class_id,

        "true_class":
            inverse_mapping[
                true_class_id
            ],

        "current_flow":
            current_flow,

        "sequence":
            sequence,
    }


# ============================================================
# REAL MODEL WIRING
# ============================================================

def build_real_models(
    client_id: int,
    label_mapping,
    inverse_mapping,
    device,
    ae_model_path: Optional[str],
):
    rf = RealRandomForestPredictor(
        client_id=
            client_id,
        label_mapping=
            label_mapping,
        inverse_mapping=
            inverse_mapping,
    )

    ae = RealPersonalizedAutoencoderPredictor(
        client_id=
            client_id,
        device=
            device,
        explicit_model_path=
            ae_model_path,
    )

    fedprox70 = RealFedProx70Predictor(
        checkpoint_path=
            FEDPROX70_CHECKPOINT,
        inverse_mapping=
            inverse_mapping,
        device=
            device,
    )

    cnn = build_real_cnn_predictor(
        inverse_mapping=
            inverse_mapping,
        device=
            device,
    )

    return (
        rf,
        ae,
        fedprox70,
        cnn,
    )


def build_adaptive_engine(
    client_id: int,
    inverse_mapping,
    rf,
    ae,
    fedprox70,
    cnn,
):
    return phase14g.FullDetectionEngine(
        client_id=
            client_id,

        rf_predictor=
            phase14g.CallableClassifierPredictor(
                model_name=
                    "Random Forest",
                predictor=
                    lambda flow: {
                        "class_id":
                            rf.predict(
                                flow
                            ).class_id,

                        "confidence":
                            rf.predict(
                                flow
                            ).confidence,
                    },
                inverse_mapping=
                    inverse_mapping,
            ),

        autoencoder_predictor=
            phase14g.CallableAutoencoderPredictor(
                score_fn=
                    lambda flow:
                        ae.normalized_anomaly_score(
                            flow
                        ),
                threshold=
                    AE_FUSION_DECISION_SCORE,
            ),

        fedprox70_predictor=
            phase14g.CallableClassifierPredictor(
                model_name=
                    "FedProx70 MLP",
                predictor=
                    lambda flow: {
                        "class_id":
                            fedprox70.predict(
                                flow
                            ).class_id,

                        "confidence":
                            fedprox70.predict(
                                flow
                            ).confidence,
                    },
                inverse_mapping=
                    inverse_mapping,
            ),

        cnn_predictor=
            cnn,
    )


# ============================================================
# ALWAYS-RUN-ALL AUDIT
# ============================================================

def timed_call(
    fn,
):
    start = (
        time.perf_counter()
    )

    result = fn()

    elapsed_ms = (
        time.perf_counter()
        -
        start
    ) * 1000.0

    return (
        result,
        float(
            elapsed_ms
        ),
    )


def run_all_models_once(
    flow,
    sequence,
    rf,
    ae,
    fedprox70,
    cnn,
):
    (
        rf_result,
        rf_ms,
    ) = timed_call(
        lambda:
            rf.predict(
                flow
            )
    )

    (
        ae_details,
        ae_ms,
    ) = timed_call(
        lambda:
            ae.details(
                flow
            )
    )

    ae_result = (
        phase14g.AutoencoderResult(
            anomaly_score=
                float(
                    ae_details[
                        "normalized_anomaly_score"
                    ]
                ),
            is_anomaly=
                bool(
                    ae_details[
                        "raw_is_anomaly"
                    ]
                ),
        )
    )

    (
        fedprox_result,
        fedprox_ms,
    ) = timed_call(
        lambda:
            fedprox70.predict(
                flow
            )
    )

    (
        cnn_result,
        cnn_ms,
    ) = timed_call(
        lambda:
            cnn.predict(
                sequence
            )
    )

    fusion_start = (
        time.perf_counter()
    )

    risk_result = (
        phase14g.fuse_risk(
            rf_result=
                rf_result,
            fedprox_result=
                fedprox_result,
            cnn_result=
                cnn_result,
            ae_result=
                ae_result,
        )
    )

    fusion_ms = (
        time.perf_counter()
        -
        fusion_start
    ) * 1000.0

    return {
        "rf_result":
            rf_result,

        "ae_details":
            ae_details,

        "ae_result":
            ae_result,

        "fedprox_result":
            fedprox_result,

        "cnn_result":
            cnn_result,

        "risk_result":
            risk_result,

        "timings_ms":
            {
                "random_forest":
                    rf_ms,

                "autoencoder":
                    ae_ms,

                "fedprox70":
                    fedprox_ms,

                "cnn_bilstm":
                    cnn_ms,

                "risk_fusion":
                    float(
                        fusion_ms
                    ),

                "all_models_plus_fusion":
                    float(
                        rf_ms
                        +
                        ae_ms
                        +
                        fedprox_ms
                        +
                        cnn_ms
                        +
                        fusion_ms
                    ),
            },
    }


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
            "Client whose personalized RF and AE are loaded."
        ),
    )

    parser.add_argument(
        "--target-class",
        type=str,
        default="auto",
        help=(
            "Validation class to select. "
            "'auto' chooses the client's own seen attack."
        ),
    )

    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help=(
            "Optional explicit validation sequence index. "
            "Overrides --target-class."
        ),
    )

    parser.add_argument(
        "--ae-model",
        type=str,
        default=None,
        help=(
            "Optional explicit Phase14B AE artifact path. "
            "If omitted, artifacts/detection is searched automatically."
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

    (
        label_mapping,
        inverse_mapping,
        active_features,
    ) = load_project_schema()

    separator()

    print(
        PHASE_NAME
    )

    separator()

    print(
        f"Device                    : {device}"
    )

    print(
        f"Client                    : {args.client_id}"
    )

    print(
        f"Client seen attack        : "
        f"{CLIENT_ATTACKS[args.client_id]}"
    )

    print(
        "Locked test               : NO"
    )

    print()
    print("REAL ARTIFACT WIRING")
    print("-" * 118)

    (
        rf,
        ae,
        fedprox70,
        cnn,
    ) = build_real_models(
        client_id=
            args.client_id,
        label_mapping=
            label_mapping,
        inverse_mapping=
            inverse_mapping,
        device=
            device,
        ae_model_path=
            args.ae_model,
    )

    print(
        f"Random Forest             : {rf.path}"
    )

    print(
        f"RF artifact container     : "
        f"{rf.artifact_container_type}"
    )

    print(
        f"RF estimator              : "
        f"{rf.estimator_type}"
    )

    if rf.artifact_keys is not None:
        print(
            f"RF artifact keys          : "
            f"{rf.artifact_keys}"
        )

    print(
        f"Personalized AE           : {ae.path}"
    )

    print(
        f"AE backend                : {ae.backend}"
    )

    print(
        f"AE architecture source    : {ae.architecture_source}"
    )

    print(
        f"AE raw threshold          : {ae.threshold:.6f}"
    )

    print(
        f"FedProx70 MLP             : {fedprox70.path}"
    )

    print(
        f"FedProx70 seed            : {fedprox70.seed}"
    )

    print(
        f"FedProx70 mu              : {fedprox70.mu}"
    )

    print(
        f"CNN-BiLSTM                : {CNN_CHECKPOINT}"
    )

    print(
        f"CNN frozen round          : {cnn.checkpoint_round}"
    )

    print(
        f"CNN FedProx mu            : {cnn.checkpoint_mu}"
    )

    # ========================================================
    # REAL VALIDATION SAMPLE
    # ========================================================

    sample = load_real_validation_sample(
        client_id=
            args.client_id,
        label_mapping=
            label_mapping,
        inverse_mapping=
            inverse_mapping,
        sample_index=
            args.sample_index,
        target_class=
            args.target_class,
    )

    flow = sample[
        "current_flow"
    ]

    sequence = sample[
        "sequence"
    ]

    print()
    print("REAL VALIDATION INPUT")
    print("-" * 118)

    print(
        f"Validation file           : "
        f"{VALIDATION_SEQUENCE_FILE}"
    )

    print(
        f"Sequence index            : "
        f"{sample['sample_index']}"
    )

    print(
        f"True class                : "
        f"{sample['true_class']}"
    )

    print(
        f"Current flow shape        : "
        f"{flow.shape}"
    )

    print(
        f"Sequence shape            : "
        f"{sequence.shape}"
    )

    print(
        "Current flow alignment    : sequence[-1]"
    )

    # ========================================================
    # ALWAYS-RUN-ALL REAL MODEL AUDIT
    # ========================================================

    all_run = run_all_models_once(
        flow=
            flow,
        sequence=
            sequence,
        rf=
            rf,
        ae=
            ae,
        fedprox70=
            fedprox70,
        cnn=
            cnn,
    )

    rf_result = all_run[
        "rf_result"
    ]

    ae_details = all_run[
        "ae_details"
    ]

    fedprox_result = all_run[
        "fedprox_result"
    ]

    cnn_result = all_run[
        "cnn_result"
    ]

    risk_result = all_run[
        "risk_result"
    ]

    print()
    print("ALWAYS-RUN-ALL — REAL MODEL OUTPUTS")
    print("-" * 118)

    print(
        f"RF                        : "
        f"{rf_result.predicted_class} "
        f"({rf_result.confidence:.6f})"
    )

    print(
        f"AE raw error              : "
        f"{ae_details['raw_reconstruction_error']:.6f}"
    )

    print(
        f"AE raw threshold          : "
        f"{ae_details['raw_threshold']:.6f}"
    )

    print(
        f"AE anomaly                : "
        f"{ae_details['raw_is_anomaly']}"
    )

    print(
        f"AE normalized risk        : "
        f"{ae_details['normalized_anomaly_score']:.6f}"
    )

    print(
        f"FedProx70                 : "
        f"{fedprox_result.predicted_class} "
        f"({fedprox_result.confidence:.6f})"
    )

    print(
        f"CNN-BiLSTM                : "
        f"{cnn_result.predicted_class} "
        f"({cnn_result.confidence:.6f})"
    )

    print(
        f"Fused class               : "
        f"{risk_result.final_class}"
    )

    print(
        f"Fused severity            : "
        f"{risk_result.severity}"
    )

    print(
        f"Fused risk                : "
        f"{risk_result.risk_score:.6f}"
    )

    print(
        f"Decision source           : "
        f"{risk_result.decision_source}"
    )

    print()
    print("MODEL LATENCY")
    print("-" * 118)

    for name, value in all_run[
        "timings_ms"
    ].items():
        print(
            f"{name:26s}: "
            f"{value:.3f} ms"
        )

    # ========================================================
    # ADAPTIVE-GATE REAL PIPELINE
    # ========================================================

    # Avoid evaluating RF / FedProx twice inside a lambda by
    # defining one-call helper functions.
    def rf_callable(
        current_flow,
    ):
        result = rf.predict(
            current_flow
        )

        return {
            "class_id":
                result.class_id,

            "confidence":
                result.confidence,
        }

    def fedprox_callable(
        current_flow,
    ):
        result = fedprox70.predict(
            current_flow
        )

        return {
            "class_id":
                result.class_id,

            "confidence":
                result.confidence,
        }

    adaptive_engine = (
        phase14g.FullDetectionEngine(
            client_id=
                args.client_id,

            rf_predictor=
                phase14g.CallableClassifierPredictor(
                    model_name=
                        "Random Forest",
                    predictor=
                        rf_callable,
                    inverse_mapping=
                        inverse_mapping,
                ),

            autoencoder_predictor=
                phase14g.CallableAutoencoderPredictor(
                    score_fn=
                        lambda current_flow:
                            ae.normalized_anomaly_score(
                                current_flow
                            ),
                    threshold=
                        AE_FUSION_DECISION_SCORE,
                ),

            fedprox70_predictor=
                phase14g.CallableClassifierPredictor(
                    model_name=
                        "FedProx70 MLP",
                    predictor=
                        fedprox_callable,
                    inverse_mapping=
                        inverse_mapping,
                ),

            cnn_predictor=
                cnn,
        )
    )

    adaptive_start = (
        time.perf_counter()
    )

    adaptive_alert = (
        adaptive_engine.detect(
            current_flow_70=
                flow,
            sequence_20x70=
                sequence,
        )
    )

    adaptive_ms = (
        time.perf_counter()
        -
        adaptive_start
    ) * 1000.0

    adaptive_alert_dict = (
        vars(
            adaptive_alert
        )
    )

    print()
    print("ADAPTIVE-GATE REAL ALERT")
    print("-" * 118)

    print(
        json.dumps(
            json_safe(
                adaptive_alert_dict
            ),
            indent=4,
        )
    )

    print(
        f"\nAdaptive pipeline latency : "
        f"{adaptive_ms:.3f} ms"
    )

    # ========================================================
    # SAVE EVIDENCE
    # ========================================================

    audit_payload = {
        "phase":
            "14G1",

        "client_id":
            args.client_id,

        "true_class":
            sample[
                "true_class"
            ],

        "validation_sample_index":
            sample[
                "sample_index"
            ],

        "locked_test_used":
            False,

        "model_artifacts":
            {
                "random_forest":
                    rf.path,

                "personalized_autoencoder":
                    ae.path,

                "fedprox70":
                    fedprox70.path,

                "cnn_bilstm":
                    CNN_CHECKPOINT,
            },

        "autoencoder":
            ae_details,

        "always_run_all":
            {
                "rf":
                    vars(
                        rf_result
                    ),

                "fedprox70":
                    vars(
                        fedprox_result
                    ),

                "cnn_bilstm":
                    vars(
                        cnn_result
                    ),

                "risk_fusion":
                    vars(
                        risk_result
                    ),

                "timings_ms":
                    all_run[
                        "timings_ms"
                    ],
            },

        "adaptive_alert":
            adaptive_alert_dict,

        "adaptive_pipeline_latency_ms":
            adaptive_ms,

        "scientific_note":
            (
                "Validation-only real-model wiring. "
                "No locked-test evaluation. "
                "AE normalized anomaly score is an engineering "
                "threshold-relative fusion score, not a calibrated "
                "probability."
            ),

        "scientific_label":
            (
                "SOURCE-ORDERED CNN-BiLSTM sequences; "
                "NOT timestamp-confirmed temporal sequences."
            ),
    }

    save_json(
        WIRING_AUDIT_FILE,
        audit_payload,
    )

    save_json(
        REAL_ALERT_FILE,
        {
            "true_class":
                sample[
                    "true_class"
                ],

            "alert":
                adaptive_alert_dict,

            "locked_test_used":
                False,
        },
    )

    manifest = {
        "phase":
            "14G1",

        "architecture":
            [
                "Phase14A client-local Random Forest",
                "Adaptive Gate",
                "Phase14B personalized Autoencoder",
                "Phase10 FedProx70 MLP seed 42",
                "Phase14F2 frozen Round-20 CNN-BiLSTM",
                "Phase14G Risk Fusion",
                "Local IDS Alert",
            ],

        "feature_count":
            EXPECTED_FEATURES,

        "feature_names":
            active_features,

        "sequence_length":
            SEQUENCE_LENGTH,

        "risk_weights":
            RISK_WEIGHTS,

        "rf_fast_attack_threshold":
            RF_FAST_ATTACK_THRESHOLD,

        "ae_frozen_raw_thresholds":
            AE_FROZEN_THRESHOLDS,

        "ae_fusion_normalization":
            (
                "clip(0.70 * raw_reconstruction_error / "
                "personalized_threshold, 0, 1)"
            ),

        "ae_fusion_score_is_calibrated_probability":
            False,

        "fedprox70_checkpoint":
            FEDPROX70_CHECKPOINT,

        "cnn_checkpoint":
            CNN_CHECKPOINT,

        "cnn_round":
            20,

        "cnn_mu":
            0.01,

        "validation_file":
            VALIDATION_SEQUENCE_FILE,

        "locked_test_used":
            False,
    }

    save_json(
        INTEGRATION_MANIFEST_FILE,
        manifest,
    )

    separator()

    print(
        "PHASE 14G1 COMPLETE"
    )

    separator()

    print(
        f"True class                : "
        f"{sample['true_class']}"
    )

    print(
        f"Adaptive classification   : "
        f"{adaptive_alert.classification}"
    )

    print(
        f"Severity                  : "
        f"{adaptive_alert.severity}"
    )

    print(
        f"Risk score                : "
        f"{adaptive_alert.risk_score:.6f}"
    )

    print(
        f"Processing path           : "
        f"{adaptive_alert.processing_path}"
    )

    print(
        f"Decision source           : "
        f"{adaptive_alert.decision_source}"
    )

    print()

    print(
        f"Wiring audit              : "
        f"{WIRING_AUDIT_FILE}"
    )

    print(
        f"Real alert                : "
        f"{REAL_ALERT_FILE}"
    )

    print(
        f"Manifest                  : "
        f"{INTEGRATION_MANIFEST_FILE}"
    )

    print()

    print(
        "LOCKED TEST USED : NO"
    )

    print()

    print(
        "SCIENTIFIC NOTE:"
    )

    print(
        "All four model branches are real trained artifacts."
    )

    print(
        "The always-run-all section verifies every model on the "
        "same validation flow/sequence."
    )

    print(
        "The adaptive section then executes the actual RF gate policy."
    )

    print(
        "AE fusion score is threshold-relative engineering risk, "
        "not a calibrated probability."
    )


if __name__ == "__main__":
    main()
