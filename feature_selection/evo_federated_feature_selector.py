# ============================================================
# file: feature_selection/evo_federated_feature_selector.py
# ============================================================
#
# FEDERATED-AWARE EVO FEATURE SELECTOR
#
# Purpose:
#
#   continuous EVO particle
#           ↓
#   binary feature mask
#           ↓
#   selected features
#           ↓
#   small federated training proxy
#           ↓
#   FedAvg across heterogeneous clients
#           ↓
#   validation of aggregated global model
#           ↓
#   Macro F1
#   Balanced Accuracy
#   Attack Macro Recall
#   Worst Attack Recall
#   Feature Reduction
#   Zero-Attack-Recall Penalty
#           ↓
#   scalar fitness returned to evo.py
#
#
# IMPORTANT:
#
#   This selector NEVER reads test_locked.csv.
#
#   It expects train-derived client splits prepared by the
#   calling script:
#
#       client_1:
#           X_train
#           y_train
#           X_val
#           y_val
#
#       ...
#
#       client_5
#
#
# Recommended fitness:
#
# Utility =
#
#   0.35 * Macro F1
# + 0.30 * Balanced Accuracy
# + 0.20 * Attack Macro Recall
# + 0.10 * Worst Attack Recall
# + 0.05 * Feature Reduction
#
#
# Fitness for minimization:
#
#   fitness =
#       1 - utility
#       + zero_attack_penalty
#
#
# This prevents EVO from selecting tiny feature subsets that
# look efficient but collapse all attack classes.
#
# ============================================================


from __future__ import annotations

import copy
import hashlib
import random

from dataclasses import dataclass, asdict

from typing import (
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# RESULT OBJECT
# ============================================================

@dataclass(frozen=True)
class FederatedFeatureSelectionResult:

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    fitness: float
    utility: float

    # --------------------------------------------------------
    # Global validation metrics
    # --------------------------------------------------------

    accuracy: float

    macro_precision: float

    macro_recall: float

    macro_f1: float

    balanced_accuracy: float

    attack_macro_recall: float

    worst_attack_recall: float

    # --------------------------------------------------------
    # Attack collapse diagnostics
    # --------------------------------------------------------

    zero_attack_count: int

    zero_attack_fraction: float

    zero_attack_penalty: float

    # --------------------------------------------------------
    # Feature selection
    # --------------------------------------------------------

    feature_reduction: float

    selected_count: int

    total_features: int

    selected_indices: Tuple[int, ...]

    selected_features: Tuple[str, ...]

    # --------------------------------------------------------
    # Per-class recalls
    # --------------------------------------------------------

    class_recalls: Tuple[float, ...]

    # --------------------------------------------------------
    # Federated diagnostics
    # --------------------------------------------------------

    rounds: int

    total_client_training_samples: int

    def to_dict(self):

        return asdict(self)


# ============================================================
# MLP
#
# Same basic architecture as Phase 4:
#
# input
#   ↓
# 256
#   ↓
# 128
#   ↓
# 64
#   ↓
# classes
#
# ============================================================

class FederatedProxyMLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                256,
            ),

            nn.LayerNorm(
                256
            ),

            nn.ReLU(),

            nn.Dropout(
                0.20
            ),

            nn.Linear(
                256,
                128,
            ),

            nn.LayerNorm(
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                0.20
            ),

            nn.Linear(
                128,
                64,
            ),

            nn.LayerNorm(
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.10
            ),

            nn.Linear(
                64,
                num_classes,
            ),
        )

    def forward(
        self,
        x,
    ):

        return self.network(
            x
        )


# ============================================================
# FEDERATED EVO FEATURE SELECTOR
# ============================================================

class EVOFederatedFeatureSelector:

    """
    Federated-aware objective evaluator for Energy Valley
    Optimization.

    Expected client_splits format:

        [
            {
                "client_id": 1,

                "X_train": np.ndarray,
                "y_train": np.ndarray,

                "X_val": np.ndarray,
                "y_val": np.ndarray,
            },

            ...

            {
                "client_id": 5,
                ...
            }
        ]


    All X matrices must contain the SAME starting feature
    space, for example the 70 active Phase-4 features.


    EVO sees only:

        position -> scalar fitness


    The selector internally performs:

        feature mask
             ↓
        client local training
             ↓
        sample-weighted FedAvg
             ↓
        aggregated validation
             ↓
        federated-aware fitness
    """

    def __init__(
        self,
        client_splits: Sequence[Dict],
        feature_names: Sequence[str],
        num_classes: int,
        benign_class_id: int,

        # ----------------------------------------------------
        # Federated proxy
        # ----------------------------------------------------

        federated_rounds: int = 2,

        local_epochs: int = 1,

        batch_size: int = 1024,

        learning_rate: float = 0.001,

        weight_decay: float = 0.0001,

        gradient_clip: float = 5.0,

        # ----------------------------------------------------
        # EVO binary conversion
        # ----------------------------------------------------

        transfer_function: str = "sigmoid",

        threshold: float = 0.5,

        min_features: int = 2,

        # ----------------------------------------------------
        # Fitness weights
        # ----------------------------------------------------

        macro_f1_weight: float = 0.35,

        balanced_accuracy_weight: float = 0.30,

        attack_macro_recall_weight: float = 0.20,

        worst_attack_recall_weight: float = 0.10,

        feature_reduction_weight: float = 0.05,

        # ----------------------------------------------------
        # Attack-collapse penalty
        # ----------------------------------------------------

        zero_attack_penalty_weight: float = 0.15,

        zero_recall_threshold: float = 1e-12,

        # ----------------------------------------------------
        # Optimization
        # ----------------------------------------------------

        objective: str = "minimize",

        random_state: int = 42,

        device: Optional[str] = None,

        enable_cache: bool = True,

        verbose: bool = True,
    ):

        # ====================================================
        # BASIC CONFIGURATION
        # ====================================================

        self.client_splits = list(
            client_splits
        )

        self.feature_names = tuple(
            feature_names
        )

        self.n_features = len(
            self.feature_names
        )

        self.num_classes = int(
            num_classes
        )

        self.benign_class_id = int(
            benign_class_id
        )

        self.attack_class_ids = tuple(

            class_id

            for class_id
            in range(
                self.num_classes
            )

            if class_id
            !=
            self.benign_class_id
        )


        # ====================================================
        # FEDERATED CONFIG
        # ====================================================

        self.federated_rounds = int(
            federated_rounds
        )

        self.local_epochs = int(
            local_epochs
        )

        self.batch_size = int(
            batch_size
        )

        self.learning_rate = float(
            learning_rate
        )

        self.weight_decay = float(
            weight_decay
        )

        self.gradient_clip = float(
            gradient_clip
        )


        # ====================================================
        # MASK CONFIG
        # ====================================================

        self.transfer_function = (
            transfer_function
        )

        self.threshold = float(
            threshold
        )

        self.min_features = int(
            min_features
        )


        # ====================================================
        # FITNESS CONFIG
        # ====================================================

        self.macro_f1_weight = float(
            macro_f1_weight
        )

        self.balanced_accuracy_weight = float(
            balanced_accuracy_weight
        )

        self.attack_macro_recall_weight = float(
            attack_macro_recall_weight
        )

        self.worst_attack_recall_weight = float(
            worst_attack_recall_weight
        )

        self.feature_reduction_weight = float(
            feature_reduction_weight
        )

        self.zero_attack_penalty_weight = float(
            zero_attack_penalty_weight
        )

        self.zero_recall_threshold = float(
            zero_recall_threshold
        )


        # ====================================================
        # VALIDATE FITNESS WEIGHTS
        # ====================================================

        weight_sum = (

            self.macro_f1_weight

            +

            self.balanced_accuracy_weight

            +

            self.attack_macro_recall_weight

            +

            self.worst_attack_recall_weight

            +

            self.feature_reduction_weight
        )

        if not np.isclose(
            weight_sum,
            1.0,
        ):

            raise ValueError(
                "\nFitness metric weights must "
                "sum to 1.0.\n"
                f"Current sum = {weight_sum}"
            )


        # ====================================================
        # OBJECTIVE
        # ====================================================

        if objective not in (
            "minimize",
            "maximize",
        ):

            raise ValueError(
                "objective must be "
                "'minimize' or 'maximize'."
            )

        self.objective = objective


        # ====================================================
        # REPRODUCIBILITY
        # ====================================================

        self.random_state = int(
            random_state
        )


        # ====================================================
        # DEVICE
        # ====================================================

        if device is None:

            self.device = torch.device(

                "cuda"

                if torch.cuda.is_available()

                else "cpu"
            )

        else:

            self.device = torch.device(
                device
            )


        # ====================================================
        # CACHE
        # ====================================================

        self.enable_cache = bool(
            enable_cache
        )

        self.cache = {}

        self.evaluation_count = 0

        self.cache_hits = 0


        # ====================================================
        # BEST RESULT
        # ====================================================

        self.best_result = None

        self.best_mask = None


        # ====================================================
        # VERBOSE
        # ====================================================

        self.verbose = bool(
            verbose
        )


        # ====================================================
        # VALIDATE INPUT DATA
        # ====================================================

        self._validate_client_splits()


        # ====================================================
        # GLOBAL CLASS WEIGHTS
        # ====================================================

        self.class_weights = (
            self._calculate_global_class_weights()
        )


        # ====================================================
        # TOTAL TRAIN SAMPLES
        # ====================================================

        self.total_client_training_samples = int(

            sum(

                len(
                    client[
                        "y_train"
                    ]
                )

                for client
                in self.client_splits
            )
        )


        if self.verbose:

            print()

            print(
                "=" * 70
            )

            print(
                "FEDERATED-AWARE EVO FEATURE SELECTOR"
            )

            print(
                "=" * 70
            )

            print(
                f"Clients              : "
                f"{len(self.client_splits)}"
            )

            print(
                f"Starting features    : "
                f"{self.n_features}"
            )

            print(
                f"Classes              : "
                f"{self.num_classes}"
            )

            print(
                f"Attack classes       : "
                f"{len(self.attack_class_ids)}"
            )

            print(
                f"Proxy FedAvg rounds  : "
                f"{self.federated_rounds}"
            )

            print(
                f"Local epochs         : "
                f"{self.local_epochs}"
            )

            print(
                f"Device               : "
                f"{self.device}"
            )

            print(
                "=" * 70
            )


    # ========================================================
    # SET RANDOM SEED
    # ========================================================

    def _set_seed(
        self,
        seed: int,
    ):

        random.seed(
            seed
        )

        np.random.seed(
            seed
        )

        torch.manual_seed(
            seed
        )

        if torch.cuda.is_available():

            torch.cuda.manual_seed_all(
                seed
            )

        if hasattr(
            torch.backends,
            "cudnn",
        ):

            torch.backends.cudnn.deterministic = True

            torch.backends.cudnn.benchmark = False


    # ========================================================
    # VALIDATE CLIENT SPLITS
    # ========================================================

    def _validate_client_splits(
        self,
    ):

        if len(
            self.client_splits
        ) < 2:

            raise ValueError(
                "At least two federated clients "
                "are required."
            )

        required = {

            "X_train",
            "y_train",

            "X_val",
            "y_val",
        }

        for index, client in enumerate(
            self.client_splits,
            start=1,
        ):

            missing = (

                required

                -
                set(
                    client.keys()
                )
            )

            if missing:

                raise ValueError(
                    f"Client {index} missing keys: "
                    f"{missing}"
                )

            X_train = np.asarray(
                client[
                    "X_train"
                ]
            )

            y_train = np.asarray(
                client[
                    "y_train"
                ]
            )

            X_val = np.asarray(
                client[
                    "X_val"
                ]
            )

            y_val = np.asarray(
                client[
                    "y_val"
                ]
            )

            if X_train.ndim != 2:

                raise ValueError(
                    f"Client {index} X_train "
                    "must be 2-D."
                )

            if X_val.ndim != 2:

                raise ValueError(
                    f"Client {index} X_val "
                    "must be 2-D."
                )

            if (
                X_train.shape[1]
                !=
                self.n_features
            ):

                raise ValueError(
                    f"Client {index} X_train "
                    f"contains {X_train.shape[1]} "
                    f"features, expected "
                    f"{self.n_features}."
                )

            if (
                X_val.shape[1]
                !=
                self.n_features
            ):

                raise ValueError(
                    f"Client {index} X_val "
                    f"contains {X_val.shape[1]} "
                    f"features, expected "
                    f"{self.n_features}."
                )

            if len(
                X_train
            ) != len(
                y_train
            ):

                raise ValueError(
                    f"Client {index} training "
                    "length mismatch."
                )

            if len(
                X_val
            ) != len(
                y_val
            ):

                raise ValueError(
                    f"Client {index} validation "
                    "length mismatch."
                )

            if not np.isfinite(
                X_train
            ).all():

                raise ValueError(
                    f"Client {index} X_train "
                    "contains NaN/inf."
                )

            if not np.isfinite(
                X_val
            ).all():

                raise ValueError(
                    f"Client {index} X_val "
                    "contains NaN/inf."
                )

            # Store normalized arrays back.

            client[
                "X_train"
            ] = X_train.astype(
                np.float32,
                copy=False,
            )

            client[
                "X_val"
            ] = X_val.astype(
                np.float32,
                copy=False,
            )

            client[
                "y_train"
            ] = y_train.astype(
                np.int64,
                copy=False,
            )

            client[
                "y_val"
            ] = y_val.astype(
                np.int64,
                copy=False,
            )


    # ========================================================
    # GLOBAL CLASS WEIGHTS
    #
    # Same moderated inverse-frequency idea as Phase 4:
    #
    # sqrt(max_count / count)
    # cap = 20
    # normalize mean = 1
    # ========================================================

    def _calculate_global_class_weights(
        self,
    ):

        counts = np.zeros(
            self.num_classes,
            dtype=np.int64,
        )

        for client in (
            self.client_splits
        ):

            counts += np.bincount(

                client[
                    "y_train"
                ],

                minlength=
                    self.num_classes,
            )

        safe_counts = np.maximum(

            counts.astype(
                np.float64
            ),

            1.0,
        )

        maximum = (
            safe_counts.max()
        )

        weights = np.sqrt(

            maximum

            /
            safe_counts
        )

        weights = np.minimum(
            weights,
            20.0,
        )

        weights = (

            weights

            /
            weights.mean()
        )

        return weights.astype(
            np.float32
        )


    # ========================================================
    # SIGMOID
    # ========================================================

    @staticmethod
    def sigmoid(
        x,
    ):

        x = np.asarray(
            x,
            dtype=float,
        )

        x = np.clip(
            x,
            -60,
            60,
        )

        return (

            1.0

            /

            (
                1.0
                +
                np.exp(
                    -x
                )
            )
        )


    # ========================================================
    # POSITION -> FEATURE MASK
    # ========================================================

    def position_to_mask(
        self,
        position: Iterable[float],
    ):

        position = np.asarray(
            position,
            dtype=float,
        ).reshape(-1)

        if len(
            position
        ) != self.n_features:

            raise ValueError(
                f"EVO dimension "
                f"{len(position)} "
                f"does not match "
                f"{self.n_features} features."
            )

        if not np.all(
            np.isfinite(
                position
            )
        ):

            raise ValueError(
                "EVO position contains "
                "NaN or infinite values."
            )

        if (
            self.transfer_function
            ==
            "sigmoid"
        ):

            scores = self.sigmoid(
                position
            )

        elif (
            self.transfer_function
            ==
            "threshold"
        ):

            scores = position

        else:

            raise ValueError(
                "Unknown transfer function."
            )

        mask = (

            scores

            >=

            self.threshold
        )

        # Prevent empty / trivially tiny feature sets.

        if (
            np.sum(
                mask
            )
            <
            self.min_features
        ):

            best_indices = np.argsort(
                scores
            )[
                -self.min_features:
            ]

            mask[:] = False

            mask[
                best_indices
            ] = True

        return mask.astype(
            bool
        )


    # ========================================================
    # MASK HASH
    # ========================================================

    @staticmethod
    def _mask_hash(
        mask,
    ):

        return hashlib.sha256(
            mask.tobytes()
        ).hexdigest()


    # ========================================================
    # BUILD DETERMINISTIC PROJECTED MODEL
    #
    # Important:
    #
    # Build a full-input reference model with the same seed.
    # Then keep only selected first-layer columns.
    #
    # This reduces candidate-to-candidate initialization noise.
    # ========================================================

    def _build_projected_model(
        self,
        selected_indices,
    ):

        self._set_seed(
            self.random_state
        )

        reference_model = (
            FederatedProxyMLP(

                input_dim=
                    self.n_features,

                num_classes=
                    self.num_classes,

            ).to(
                self.device
            )
        )

        reference_state = (
            reference_model
            .state_dict()
        )

        selected_model = (
            FederatedProxyMLP(

                input_dim=
                    len(
                        selected_indices
                    ),

                num_classes=
                    self.num_classes,

            ).to(
                self.device
            )
        )

        selected_state = (
            selected_model
            .state_dict()
        )

        first_layer_key = (
            "network.0.weight"
        )

        selected_list = [

            int(i)

            for i
            in selected_indices
        ]

        for key in (
            selected_state.keys()
        ):

            if (
                key
                ==
                first_layer_key
            ):

                selected_state[
                    key
                ] = (

                    reference_state[
                        key
                    ][
                        :,
                        selected_list
                    ]
                    .clone()
                )

            else:

                selected_state[
                    key
                ] = (

                    reference_state[
                        key
                    ]
                    .clone()
                )

        selected_model.load_state_dict(
            selected_state
        )

        del reference_model

        return selected_model


    # ========================================================
    # TRAIN ONE CLIENT
    # ========================================================

    def _train_client(
        self,
        global_state,
        X_train,
        y_train,
        selected_indices,
        client_id,
        round_number,
    ):

        local_model = (
            FederatedProxyMLP(

                input_dim=
                    len(
                        selected_indices
                    ),

                num_classes=
                    self.num_classes,

            ).to(
                self.device
            )
        )

        local_model.load_state_dict(
            global_state
        )

        weight_tensor = torch.tensor(

            self.class_weights,

            dtype=torch.float32,

            device=
                self.device,
        )

        criterion = (
            nn.CrossEntropyLoss(
                weight=
                    weight_tensor
            )
        )

        optimizer = (
            torch.optim.AdamW(

                local_model.parameters(),

                lr=
                    self.learning_rate,

                weight_decay=
                    self.weight_decay,
            )
        )

        local_model.train()

        selected_X = (
            X_train[
                :,
                selected_indices
            ]
        )

        # Same deterministic shuffle for every candidate,
        # client and round.

        rng = np.random.default_rng(

            self.random_state

            +
            round_number
            * 1000

            +
            client_id
            * 100
        )

        for local_epoch in range(
            1,
            self.local_epochs + 1,
        ):

            indices = rng.permutation(
                len(
                    selected_X
                )
            )

            for start in range(
                0,
                len(
                    selected_X
                ),
                self.batch_size,
            ):

                batch_indices = indices[

                    start:

                    min(
                        start
                        +
                        self.batch_size,

                        len(
                            indices
                        )
                    )
                ]

                xb = torch.from_numpy(

                    selected_X[
                        batch_indices
                    ]

                ).to(
                    self.device
                )

                yb = torch.from_numpy(

                    y_train[
                        batch_indices
                    ]

                ).to(
                    self.device
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                logits = local_model(
                    xb
                )

                loss = criterion(
                    logits,
                    yb
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(

                    local_model.parameters(),

                    max_norm=
                        self.gradient_clip,
                )

                optimizer.step()

        state = {

            key:
                value
                .detach()
                .cpu()
                .clone()

            for key, value
            in local_model
            .state_dict()
            .items()
        }

        del local_model

        return state


    # ========================================================
    # SAMPLE-WEIGHTED FEDAVG
    # ========================================================

    @staticmethod
    def _federated_average(
        client_states,
        client_sample_counts,
    ):

        total_samples = float(
            sum(
                client_sample_counts
            )
        )

        global_state = {}

        for key in (
            client_states[0]
            .keys()
        ):

            aggregated = (
                torch.zeros_like(

                    client_states[0][
                        key
                    ],

                    dtype=
                        torch.float32,
                )
            )

            for state, samples in zip(
                client_states,
                client_sample_counts,
            ):

                aggregated += (

                    state[
                        key
                    ]
                    .float()

                    *

                    (
                        samples
                        /
                        total_samples
                    )
                )

            global_state[
                key
            ] = aggregated

        return global_state


    # ========================================================
    # RUN FEDERATED PROXY
    # ========================================================

    def _run_federated_proxy(
        self,
        selected_indices,
    ):

        global_model = (
            self._build_projected_model(
                selected_indices
            )
        )

        for round_number in range(
            1,
            self.federated_rounds + 1,
        ):

            global_state = {

                key:
                    value
                    .detach()
                    .cpu()
                    .clone()

                for key, value
                in global_model
                .state_dict()
                .items()
            }

            client_states = []

            client_sample_counts = []

            for client_number, client in enumerate(
                self.client_splits,
                start=1,
            ):

                client_id = int(
                    client.get(
                        "client_id",
                        client_number,
                    )
                )

                local_state = (
                    self._train_client(

                        global_state=
                            global_state,

                        X_train=
                            client[
                                "X_train"
                            ],

                        y_train=
                            client[
                                "y_train"
                            ],

                        selected_indices=
                            selected_indices,

                        client_id=
                            client_id,

                        round_number=
                            round_number,
                    )
                )

                client_states.append(
                    local_state
                )

                client_sample_counts.append(

                    len(
                        client[
                            "y_train"
                        ]
                    )
                )

            aggregated_state = (
                self._federated_average(

                    client_states,

                    client_sample_counts,
                )
            )

            global_model.load_state_dict(
                aggregated_state
            )

        return global_model


    # ========================================================
    # GLOBAL FEDERATED VALIDATION
    # ========================================================

    def _evaluate_global_model(
        self,
        model,
        selected_indices,
    ):

        model.eval()

        all_true = []

        all_pred = []

        with torch.no_grad():

            for client in (
                self.client_splits
            ):

                X_val = (
                    client[
                        "X_val"
                    ][
                        :,
                        selected_indices
                    ]
                )

                y_val = (
                    client[
                        "y_val"
                    ]
                )

                client_predictions = []

                for start in range(
                    0,
                    len(
                        X_val
                    ),
                    self.batch_size,
                ):

                    end = min(

                        start
                        +
                        self.batch_size,

                        len(
                            X_val
                        ),
                    )

                    xb = torch.from_numpy(

                        X_val[
                            start:end
                        ]

                    ).to(
                        self.device
                    )

                    logits = model(
                        xb
                    )

                    predictions = (

                        logits
                        .argmax(
                            dim=1
                        )
                        .cpu()
                        .numpy()
                    )

                    client_predictions.append(
                        predictions
                    )

                all_true.append(
                    y_val
                )

                all_pred.append(

                    np.concatenate(
                        client_predictions
                    )
                )

        y_true = np.concatenate(
            all_true
        )

        y_pred = np.concatenate(
            all_pred
        )

        return (
            y_true,
            y_pred,
        )


    # ========================================================
    # CALCULATE FITNESS
    # ========================================================

    def _calculate_result(
        self,
        mask,
        y_true,
        y_pred,
    ):

        selected_indices = (
            np.flatnonzero(
                mask
            )
        )

        selected_count = len(
            selected_indices
        )


        # ----------------------------------------------------
        # Overall metrics
        # ----------------------------------------------------

        accuracy = float(

            accuracy_score(
                y_true,
                y_pred,
            )
        )


        macro_precision = float(

            precision_score(

                y_true,
                y_pred,

                labels=
                    list(
                        range(
                            self.num_classes
                        )
                    ),

                average=
                    "macro",

                zero_division=
                    0,
            )
        )


        macro_recall = float(

            recall_score(

                y_true,
                y_pred,

                labels=
                    list(
                        range(
                            self.num_classes
                        )
                    ),

                average=
                    "macro",

                zero_division=
                    0,
            )
        )


        macro_f1 = float(

            f1_score(

                y_true,
                y_pred,

                labels=
                    list(
                        range(
                            self.num_classes
                        )
                    ),

                average=
                    "macro",

                zero_division=
                    0,
            )
        )


        # ----------------------------------------------------
        # Balanced accuracy
        #
        # Explicit class-wise recall mean is used so that all
        # six classes contribute even if a class is predicted
        # zero times.
        # ----------------------------------------------------

        class_recalls = (
            recall_score(

                y_true,
                y_pred,

                labels=
                    list(
                        range(
                            self.num_classes
                        )
                    ),

                average=None,

                zero_division=0,
            )
        )

        balanced_accuracy = float(
            np.mean(
                class_recalls
            )
        )


        # ----------------------------------------------------
        # Attack recalls
        # ----------------------------------------------------

        attack_recalls = np.asarray(

            [

                class_recalls[
                    class_id
                ]

                for class_id
                in self.attack_class_ids
            ],

            dtype=float,
        )


        if len(
            attack_recalls
        ) > 0:

            attack_macro_recall = float(
                np.mean(
                    attack_recalls
                )
            )

            worst_attack_recall = float(
                np.min(
                    attack_recalls
                )
            )

        else:

            attack_macro_recall = 0.0

            worst_attack_recall = 0.0


        # ----------------------------------------------------
        # Zero-recall attacks
        # ----------------------------------------------------

        zero_attack_count = int(

            np.sum(

                attack_recalls

                <=

                self.zero_recall_threshold
            )
        )

        zero_attack_fraction = float(

            zero_attack_count

            /

            max(
                len(
                    attack_recalls
                ),
                1,
            )
        )


        zero_attack_penalty = float(

            self.zero_attack_penalty_weight

            *

            zero_attack_fraction
        )


        # ----------------------------------------------------
        # Feature reduction
        # ----------------------------------------------------

        feature_reduction = float(

            1.0

            -

            selected_count
            /
            self.n_features
        )


        # ----------------------------------------------------
        # Utility
        # ----------------------------------------------------

        utility = float(

            self.macro_f1_weight
            *
            macro_f1

            +

            self.balanced_accuracy_weight
            *
            balanced_accuracy

            +

            self.attack_macro_recall_weight
            *
            attack_macro_recall

            +

            self.worst_attack_recall_weight
            *
            worst_attack_recall

            +

            self.feature_reduction_weight
            *
            feature_reduction
        )


        # ----------------------------------------------------
        # Fitness
        # ----------------------------------------------------

        if self.objective == "minimize":

            fitness = float(

                1.0

                -

                utility

                +

                zero_attack_penalty
            )

        else:

            fitness = float(

                utility

                -

                zero_attack_penalty
            )


        selected_features = tuple(

            self.feature_names[
                index
            ]

            for index
            in selected_indices
        )


        return FederatedFeatureSelectionResult(

            fitness=
                fitness,

            utility=
                utility,

            accuracy=
                accuracy,

            macro_precision=
                macro_precision,

            macro_recall=
                macro_recall,

            macro_f1=
                macro_f1,

            balanced_accuracy=
                balanced_accuracy,

            attack_macro_recall=
                attack_macro_recall,

            worst_attack_recall=
                worst_attack_recall,

            zero_attack_count=
                zero_attack_count,

            zero_attack_fraction=
                zero_attack_fraction,

            zero_attack_penalty=
                zero_attack_penalty,

            feature_reduction=
                feature_reduction,

            selected_count=
                int(
                    selected_count
                ),

            total_features=
                int(
                    self.n_features
                ),

            selected_indices=
                tuple(

                    int(
                        index
                    )

                    for index
                    in selected_indices
                ),

            selected_features=
                selected_features,

            class_recalls=
                tuple(

                    float(
                        value
                    )

                    for value
                    in class_recalls
                ),

            rounds=
                int(
                    self.federated_rounds
                ),

            total_client_training_samples=
                int(
                    self.total_client_training_samples
                ),
        )


    # ========================================================
    # CHECK BETTER
    # ========================================================

    def _is_better(
        self,
        new_fitness,
        old_fitness,
    ):

        if (
            self.objective
            ==
            "minimize"
        ):

            return (
                new_fitness
                <
                old_fitness
            )

        return (
            new_fitness
            >
            old_fitness
        )


    # ========================================================
    # EVALUATE ONE EVO PARTICLE
    # ========================================================

    def evaluate(
        self,
        position,
    ) -> FederatedFeatureSelectionResult:

        mask = (
            self.position_to_mask(
                position
            )
        )

        cache_key = (
            self._mask_hash(
                mask
            )
        )


        # ----------------------------------------------------
        # CACHE
        # ----------------------------------------------------

        if (
            self.enable_cache
            and
            cache_key
            in self.cache
        ):

            self.cache_hits += 1

            result = self.cache[
                cache_key
            ]

            self._update_best(
                mask,
                result,
            )

            return result


        # ----------------------------------------------------
        # FEATURE INDICES
        # ----------------------------------------------------

        selected_indices = (
            np.flatnonzero(
                mask
            )
        )


        # ----------------------------------------------------
        # RUN FEDERATED PROXY
        # ----------------------------------------------------

        model = (
            self._run_federated_proxy(
                selected_indices
            )
        )


        # ----------------------------------------------------
        # VALIDATE GLOBAL FEDERATED MODEL
        # ----------------------------------------------------

        (
            y_true,
            y_pred,

        ) = self._evaluate_global_model(

            model=
                model,

            selected_indices=
                selected_indices,
        )


        # ----------------------------------------------------
        # METRICS / FITNESS
        # ----------------------------------------------------

        result = (
            self._calculate_result(

                mask=
                    mask,

                y_true=
                    y_true,

                y_pred=
                    y_pred,
            )
        )


        # ----------------------------------------------------
        # REAL EVALUATION COUNT
        # ----------------------------------------------------

        self.evaluation_count += 1


        # ----------------------------------------------------
        # CACHE RESULT
        # ----------------------------------------------------

        if self.enable_cache:

            self.cache[
                cache_key
            ] = result


        # ----------------------------------------------------
        # BEST
        # ----------------------------------------------------

        self._update_best(
            mask,
            result,
        )


        # ----------------------------------------------------
        # LOG
        # ----------------------------------------------------

        if self.verbose:

            print(

                f"[Federated EVO Eval "
                f"{self.evaluation_count:04d}] "

                f"features="
                f"{result.selected_count:02d}/"
                f"{result.total_features} | "

                f"MacroF1="
                f"{result.macro_f1:.4f} | "

                f"BalAcc="
                f"{result.balanced_accuracy:.4f} | "

                f"AttackRecall="
                f"{result.attack_macro_recall:.4f} | "

                f"WorstAttack="
                f"{result.worst_attack_recall:.4f} | "

                f"ZeroAttacks="
                f"{result.zero_attack_count} | "

                f"Reduction="
                f"{result.feature_reduction * 100:.1f}% | "

                f"fitness="
                f"{result.fitness:.6f}"
            )


        del model

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        return result


    # ========================================================
    # EVO CALLBACK
    # ========================================================

    def fitness(
        self,
        position,
    ) -> float:

        """

        optimization/evo.py calls:

            selector.fitness(position)

        and receives only:

            scalar fitness

        """

        return float(

            self.evaluate(
                position
            ).fitness
        )


    # ========================================================
    # UPDATE GLOBAL BEST
    # ========================================================

    def _update_best(
        self,
        mask,
        result,
    ):

        if (
            self.best_result
            is None
        ):

            self.best_result = result

            self.best_mask = (
                mask.copy()
            )

            return

        if self._is_better(

            result.fitness,

            self.best_result.fitness,

        ):

            self.best_result = result

            self.best_mask = (
                mask.copy()
            )


    # ========================================================
    # BEST RESULT
    # ========================================================

    def get_best_result(
        self,
    ):

        if (
            self.best_result
            is None
        ):

            raise RuntimeError(
                "No feature masks have "
                "been evaluated."
            )

        return self.best_result


    # ========================================================
    # BEST MASK
    # ========================================================

    def get_best_mask(
        self,
    ):

        if (
            self.best_mask
            is None
        ):

            raise RuntimeError(
                "No feature masks have "
                "been evaluated."
            )

        return (
            self.best_mask.copy()
        )


    # ========================================================
    # CLEAR CACHE
    # ========================================================

    def clear_cache(
        self,
    ):

        self.cache.clear()

        self.cache_hits = 0


    # ========================================================
    # RESET FULL SELECTOR STATE
    # ========================================================

    def reset(
        self,
    ):

        self.cache.clear()

        self.cache_hits = 0

        self.evaluation_count = 0

        self.best_result = None

        self.best_mask = None