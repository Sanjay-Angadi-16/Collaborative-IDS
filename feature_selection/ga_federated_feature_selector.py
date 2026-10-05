"""
Federated-Aware GA Feature Selector
===================================

Purpose
-------

This selector converts a real-valued GA chromosome into a
binary feature mask, evaluates that feature subset using a
small federated-learning proxy, and returns one scalar fitness
value to optimization/ga.py.

The design mirrors:

    feature_selection/evo_federated_feature_selector.py

so that EVO and GA use the SAME:

    - federated client structure
    - proxy MLP
    - local optimizer
    - FedAvg aggregation
    - class weights
    - fitness metrics
    - attack-collapse penalty
    - feature-reduction objective

The only optimization algorithm that changes is:

    EVO  -> optimization/evo.py
    GA   -> optimization/ga.py


Expected use
------------

    from feature_selection.ga_federated_feature_selector import (
        GAFederatedFeatureSelector
    )

    selector = GAFederatedFeatureSelector(
        client_splits=client_splits,
        feature_names=active_features,
        num_classes=6,
        benign_class_id=0,
    )

    optimizer = GeneticAlgorithmOptimizer(...)

    result = optimizer.optimize(
        objective_function=selector.fitness,
        dimension=70,
        lower_bound=-6.0,
        upper_bound=6.0,
    )


IMPORTANT
---------

This class never loads:
    - validation.csv
    - test_locked.csv

The calling script must provide train-derived federated
proxy splits.

Fitness
-------

Utility =

    0.35 * Macro F1
  + 0.30 * Balanced Accuracy
  + 0.20 * Attack Macro Recall
  + 0.10 * Worst Attack Recall
  + 0.05 * Feature Reduction

Fitness for minimization =

    1 - Utility + Zero-Attack Penalty
"""

from __future__ import annotations

import hashlib
import random

from dataclasses import (
    asdict,
    dataclass,
)

from typing import (
    Dict,
    Iterable,
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
# RESULT
# ============================================================

@dataclass(frozen=True)
class GAFederatedFeatureSelectionResult:

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
    # Attack collapse
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
    # Federated proxy
    # --------------------------------------------------------

    rounds: int

    total_client_training_samples: int

    def to_dict(self):

        return asdict(
            self
        )


# ============================================================
# PROXY MODEL
#
# Same architecture used for the federated-aware EVO proxy.
# ============================================================

class FederatedGAProxyMLP(nn.Module):

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
# GA FEDERATED FEATURE SELECTOR
# ============================================================

class GAFederatedFeatureSelector:

    """
    Federated-aware GA feature-selection objective.

    client_splits format:

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

    All feature matrices must contain the same initial
    feature space, normally the 70 active Phase-4 features.
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
        # GA chromosome -> mask
        # ----------------------------------------------------

        transfer_function: str = "sigmoid",

        threshold: float = 0.5,

        min_features: int = 5,

        # ----------------------------------------------------
        # Fitness
        # ----------------------------------------------------

        macro_f1_weight: float = 0.35,

        balanced_accuracy_weight: float = 0.30,

        attack_macro_recall_weight: float = 0.20,

        worst_attack_recall_weight: float = 0.10,

        feature_reduction_weight: float = 0.05,

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
        # CLIENT DATA
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

        self.transfer_function = str(
            transfer_function
        )

        self.threshold = float(
            threshold
        )

        self.min_features = int(
            min_features
        )

        if self.min_features < 1:

            raise ValueError(
                "min_features must be >= 1."
            )

        if (
            self.min_features
            >
            self.n_features
        ):

            raise ValueError(
                "min_features cannot exceed "
                "the total feature count."
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
        # FITNESS WEIGHT CHECK
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

        if objective not in {
            "minimize",
            "maximize",
        }:

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
        # CACHE / COUNTERS
        # ====================================================

        self.enable_cache = bool(
            enable_cache
        )

        self.cache = {}

        self.evaluation_count = 0

        self.cache_hits = 0


        # ====================================================
        # BEST
        # ====================================================

        self.best_result = None

        self.best_mask = None


        # ====================================================
        # DISPLAY
        # ====================================================

        self.verbose = bool(
            verbose
        )


        # ====================================================
        # INPUT VALIDATION
        # ====================================================

        self._validate_client_splits()


        # ====================================================
        # GLOBAL CLASS WEIGHTS
        # ====================================================

        self.class_weights = (
            self._calculate_global_class_weights()
        )


        # ====================================================
        # TRAINING SAMPLE COUNT
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
                "=" * 72
            )

            print(
                "FEDERATED-AWARE GA FEATURE SELECTOR"
            )

            print(
                "=" * 72
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
                f"Minimum features     : "
                f"{self.min_features}"
            )

            print(
                f"Device               : "
                f"{self.device}"
            )

            print(
                "=" * 72
            )


    # ========================================================
    # SEED
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
    # VALIDATE DATA
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

        for client_number, client in enumerate(
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
                    f"Client {client_number} "
                    f"missing keys: {missing}"
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
                    f"Client {client_number} "
                    "X_train must be 2-D."
                )

            if X_val.ndim != 2:

                raise ValueError(
                    f"Client {client_number} "
                    "X_val must be 2-D."
                )

            if (
                X_train.shape[1]
                !=
                self.n_features
            ):

                raise ValueError(
                    f"Client {client_number} "
                    f"X_train has "
                    f"{X_train.shape[1]} features; "
                    f"expected {self.n_features}."
                )

            if (
                X_val.shape[1]
                !=
                self.n_features
            ):

                raise ValueError(
                    f"Client {client_number} "
                    f"X_val has "
                    f"{X_val.shape[1]} features; "
                    f"expected {self.n_features}."
                )

            if (
                len(X_train)
                !=
                len(y_train)
            ):

                raise ValueError(
                    f"Client {client_number} "
                    "training row mismatch."
                )

            if (
                len(X_val)
                !=
                len(y_val)
            ):

                raise ValueError(
                    f"Client {client_number} "
                    "validation row mismatch."
                )

            if not np.isfinite(
                X_train
            ).all():

                raise ValueError(
                    f"Client {client_number} "
                    "X_train contains NaN/inf."
                )

            if not np.isfinite(
                X_val
            ).all():

                raise ValueError(
                    f"Client {client_number} "
                    "X_val contains NaN/inf."
                )

            if len(
                y_train
            ) == 0:

                raise ValueError(
                    f"Client {client_number} "
                    "has no training samples."
                )

            if len(
                y_val
            ) == 0:

                raise ValueError(
                    f"Client {client_number} "
                    "has no validation samples."
                )

            if (
                np.min(
                    y_train
                )
                <
                0
            ):

                raise ValueError(
                    "Negative class id detected."
                )

            if (
                np.max(
                    y_train
                )
                >=
                self.num_classes
            ):

                raise ValueError(
                    "Training class id exceeds "
                    "num_classes."
                )

            if (
                np.min(
                    y_val
                )
                <
                0
            ):

                raise ValueError(
                    "Negative validation class id detected."
                )

            if (
                np.max(
                    y_val
                )
                >=
                self.num_classes
            ):

                raise ValueError(
                    "Validation class id exceeds "
                    "num_classes."
                )

            # Normalize representations.

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
    # Same strategy as Phase 4 / federated EVO.
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

        maximum = float(
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
        values,
    ):

        values = np.asarray(
            values,
            dtype=np.float64,
        )

        values = np.clip(
            values,
            -60.0,
            60.0,
        )

        return (

            1.0

            /

            (
                1.0

                +

                np.exp(
                    -values
                )
            )
        )


    # ========================================================
    # CHROMOSOME -> MASK
    # ========================================================

    def position_to_mask(
        self,
        position: Iterable[float],
    ) -> np.ndarray:

        position = np.asarray(
            position,
            dtype=np.float64,
        ).reshape(-1)

        if (
            len(position)
            !=
            self.n_features
        ):

            raise ValueError(
                f"GA chromosome dimension "
                f"{len(position)} does not match "
                f"{self.n_features} features."
            )

        if not np.isfinite(
            position
        ).all():

            raise ValueError(
                "GA chromosome contains "
                "NaN/inf."
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

            scores = position.copy()

        else:

            raise ValueError(
                "transfer_function must be "
                "'sigmoid' or 'threshold'."
            )

        mask = (

            scores

            >=

            self.threshold
        )

        # ----------------------------------------------------
        # Minimum-feature protection
        # ----------------------------------------------------

        selected_count = int(
            np.sum(
                mask
            )
        )

        if (
            selected_count
            <
            self.min_features
        ):

            strongest_indices = np.argsort(
                scores
            )[
                -self.min_features:
            ]

            mask[:] = False

            mask[
                strongest_indices
            ] = True

        return mask.astype(
            bool
        )


    # ========================================================
    # CACHE KEY
    # ========================================================

    @staticmethod
    def _mask_hash(
        mask,
    ):

        return hashlib.sha256(
            mask.tobytes()
        ).hexdigest()


    # ========================================================
    # PROJECTED DETERMINISTIC MODEL
    #
    # The complete 70-feature reference network is always
    # initialized with the same seed, then the selected
    # first-layer columns are retained.
    #
    # This reduces random initialization noise across masks.
    # ========================================================

    def _build_projected_model(
        self,
        selected_indices,
    ):

        self._set_seed(
            self.random_state
        )

        reference_model = (
            FederatedGAProxyMLP(

                input_dim=
                    self.n_features,

                num_classes=
                    self.num_classes,

            ).to(
                self.device
            )
        )

        reference_state = {

            key:
                value
                .detach()
                .clone()

            for key, value
            in reference_model
            .state_dict()
            .items()
        }

        selected_model = (
            FederatedGAProxyMLP(

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
            selected_model.state_dict()
        )

        first_layer_key = (
            "network.0.weight"
        )

        positions = [

            int(index)

            for index
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

                projected = (

                    reference_state[
                        key
                    ][
                        :,
                        positions
                    ]
                    .clone()
                )

                if (
                    projected.shape
                    !=
                    selected_state[
                        key
                    ].shape
                ):

                    raise ValueError(
                        "\nProjected first layer "
                        "has unexpected shape.\n"
                        f"Projected: {projected.shape}\n"
                        f"Expected : "
                        f"{selected_state[key].shape}"
                    )

                selected_state[
                    key
                ] = projected

            else:

                if (
                    reference_state[
                        key
                    ].shape
                    !=
                    selected_state[
                        key
                    ].shape
                ):

                    raise ValueError(
                        "Unexpected shared-layer "
                        f"shape mismatch: {key}"
                    )

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
    # LOCAL CLIENT TRAINING
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
            FederatedGAProxyMLP(

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

        criterion = nn.CrossEntropyLoss(
            weight=
                weight_tensor
        )

        optimizer = torch.optim.AdamW(

            local_model.parameters(),

            lr=
                self.learning_rate,

            weight_decay=
                self.weight_decay,
        )

        local_model.train()

        selected_X = (

            X_train[
                :,
                selected_indices
            ]
        )

        for local_epoch in range(
            1,
            self.local_epochs + 1,
        ):

            # Same deterministic ordering rule as the
            # federated-aware EVO selector.

            rng = np.random.default_rng(

                self.random_state

                +
                round_number
                * 1000

                +
                client_id
                * 100

                +
                local_epoch
            )

            indices = rng.permutation(
                len(
                    selected_X
                )
            )

            for start in range(
                0,
                len(indices),
                self.batch_size,
            ):

                end = min(

                    start
                    +
                    self.batch_size,

                    len(
                        indices
                    ),
                )

                batch_indices = (
                    indices[
                        start:end
                    ]
                )

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

        local_state = {

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

        return local_state


    # ========================================================
    # FEDAVG
    # ========================================================

    @staticmethod
    def _federated_average(
        client_states,
        client_sample_counts,
    ):

        if not client_states:

            raise ValueError(
                "No client states supplied to FedAvg."
            )

        total_samples = float(
            sum(
                client_sample_counts
            )
        )

        if total_samples <= 0:

            raise ValueError(
                "FedAvg total sample count "
                "must be > 0."
            )

        global_state = {}

        for key in (
            client_states[0]
            .keys()
        ):

            aggregated = torch.zeros_like(

                client_states[0][
                    key
                ],

                dtype=torch.float32,
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
    # FEDERATED PROXY
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

                    client_states=
                        client_states,

                    client_sample_counts=
                        client_sample_counts,
                )
            )

            global_model.load_state_dict(
                aggregated_state
            )

        return global_model


    # ========================================================
    # GLOBAL VALIDATION
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

                    pred = (

                        logits
                        .argmax(
                            dim=1
                        )
                        .cpu()
                        .numpy()
                    )

                    client_predictions.append(
                        pred
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
    # FITNESS METRICS
    # ========================================================

    def _calculate_result(
        self,
        mask,
        y_true,
        y_pred,
    ):

        selected_indices = np.flatnonzero(
            mask
        )

        selected_count = int(
            len(
                selected_indices
            )
        )


        # ----------------------------------------------------
        # Accuracy
        # ----------------------------------------------------

        accuracy = float(
            accuracy_score(
                y_true,
                y_pred,
            )
        )


        # ----------------------------------------------------
        # Macro precision
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Per-class recalls
        # ----------------------------------------------------

        class_recalls = recall_score(

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


        # ----------------------------------------------------
        # Macro recall / balanced accuracy
        # ----------------------------------------------------

        macro_recall = float(
            np.mean(
                class_recalls
            )
        )

        balanced_accuracy = float(
            np.mean(
                class_recalls
            )
        )


        # ----------------------------------------------------
        # Macro F1
        # ----------------------------------------------------

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
        # Attack-only recall
        # ----------------------------------------------------

        attack_recalls = np.asarray(

            [

                class_recalls[
                    class_id
                ]

                for class_id
                in self.attack_class_ids
            ],

            dtype=np.float64,
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
        # Zero recall attacks
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

        if (
            self.objective
            ==
            "minimize"
        ):

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


        # ----------------------------------------------------
        # Selected names
        # ----------------------------------------------------

        selected_features = tuple(

            self.feature_names[
                int(index)
            ]

            for index
            in selected_indices
        )


        return GAFederatedFeatureSelectionResult(

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
                selected_count,

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
                        recall
                    )

                    for recall
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
    # FITNESS COMPARISON
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
    # UPDATE BEST
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
    # EVALUATE CHROMOSOME
    # ========================================================

    def evaluate(
        self,
        position,
    ) -> GAFederatedFeatureSelectionResult:

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
        # Selected positions
        # ----------------------------------------------------

        selected_indices = np.flatnonzero(
            mask
        )


        # ----------------------------------------------------
        # Federated training
        # ----------------------------------------------------

        model = (
            self._run_federated_proxy(
                selected_indices
            )
        )


        # ----------------------------------------------------
        # Federated validation
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
        # Result
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
        # Evaluation counter
        # ----------------------------------------------------

        self.evaluation_count += 1


        # ----------------------------------------------------
        # Cache
        # ----------------------------------------------------

        if self.enable_cache:

            self.cache[
                cache_key
            ] = result


        # ----------------------------------------------------
        # Best
        # ----------------------------------------------------

        self._update_best(
            mask,
            result,
        )


        # ----------------------------------------------------
        # Log
        # ----------------------------------------------------

        if self.verbose:

            print(

                f"[Federated GA Eval "
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
    # OBJECTIVE CALLBACK FOR optimization/ga.py
    # ========================================================

    def fitness(
        self,
        position,
    ) -> float:

        return float(

            self.evaluate(
                position
            ).fitness
        )


    # ========================================================
    # GET BEST RESULT
    # ========================================================

    def get_best_result(
        self,
    ):

        if (
            self.best_result
            is None
        ):

            raise RuntimeError(
                "No GA feature masks "
                "have been evaluated."
            )

        return self.best_result


    # ========================================================
    # GET BEST MASK
    # ========================================================

    def get_best_mask(
        self,
    ):

        if (
            self.best_mask
            is None
        ):

            raise RuntimeError(
                "No GA feature masks "
                "have been evaluated."
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
    # RESET
    # ========================================================

    def reset(
        self,
    ):

        self.cache.clear()

        self.cache_hits = 0

        self.evaluation_count = 0

        self.best_result = None

        self.best_mask = None