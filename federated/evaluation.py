# ============================================================
# file: feature_selection/evo_feature_selector.py
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
)


# ============================================================
# RESULT OBJECT
# ============================================================

@dataclass(frozen=True)
class FeatureSelectionResult:
    """
    Stores the evaluation result of one EVO particle.
    """

    fitness: float
    utility: float

    macro_f1: float
    balanced_accuracy: float
    feature_reduction: float

    selected_count: int
    total_features: int

    selected_indices: Tuple[int, ...]
    selected_features: Tuple[str, ...]

    def to_dict(self):
        return asdict(self)


# ============================================================
# EVO FEATURE SELECTOR
# ============================================================

class EVOFeatureSelector:
    """
    Independent feature-selection evaluator for EVO.

    Pipeline:

        continuous EVO particle
                ↓
        binary feature mask
                ↓
        selected feature subset
                ↓
        train classifier
                ↓
        validation prediction
                ↓
        Macro F1
        Balanced Accuracy
        Feature Reduction
                ↓
        fitness
                ↓
        returned to evo.py


    IMPORTANT:
    ----------
    This class knows NOTHING about FedAvg.

    It can therefore be tested independently before
    integrating EVO with federated learning.
    """

    def __init__(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        feature_names: Optional[Sequence[str]] = None,

        # binary conversion
        transfer_function: str = "sigmoid",
        threshold: float = 0.5,
        min_features: int = 1,

        # fitness configuration
        macro_f1_weight: float = 0.50,
        balanced_accuracy_weight: float = 0.35,
        feature_reduction_weight: float = 0.15,

        # EVO optimization direction
        objective: str = "minimize",

        # reproducibility
        random_state: int = 42,

        # optional external classifier
        classifier=None,

        # cache duplicate masks
        enable_cache: bool = True,
    ):

        # ----------------------------------------------------
        # DATA
        # ----------------------------------------------------

        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)

        self.X_val = np.asarray(X_val)
        self.y_val = np.asarray(y_val)

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if self.X_train.ndim != 2:
            raise ValueError("X_train must be a 2-D matrix.")

        if self.X_val.ndim != 2:
            raise ValueError("X_val must be a 2-D matrix.")

        if self.X_train.shape[1] != self.X_val.shape[1]:
            raise ValueError(
                "X_train and X_val must contain the same "
                "number of features."
            )

        if len(self.X_train) != len(self.y_train):
            raise ValueError(
                "X_train and y_train length mismatch."
            )

        if len(self.X_val) != len(self.y_val):
            raise ValueError(
                "X_val and y_val length mismatch."
            )

        # ----------------------------------------------------
        # FEATURE INFORMATION
        # ----------------------------------------------------

        self.n_features = self.X_train.shape[1]

        if feature_names is None:
            self.feature_names = tuple(
                f"feature_{i}"
                for i in range(self.n_features)
            )
        else:

            if len(feature_names) != self.n_features:
                raise ValueError(
                    "Number of feature names must match "
                    "number of input features."
                )

            self.feature_names = tuple(feature_names)

        # ----------------------------------------------------
        # BINARY CONVERSION
        # ----------------------------------------------------

        if transfer_function not in [
            "sigmoid",
            "threshold",
        ]:
            raise ValueError(
                "transfer_function must be "
                "'sigmoid' or 'threshold'."
            )

        self.transfer_function = transfer_function
        self.threshold = threshold

        if min_features < 1:
            raise ValueError(
                "At least one feature must be selected."
            )

        if min_features > self.n_features:
            raise ValueError(
                "min_features cannot exceed "
                "the total number of features."
            )

        self.min_features = min_features

        # ----------------------------------------------------
        # FITNESS WEIGHTS
        # ----------------------------------------------------

        total_weight = (
            macro_f1_weight
            + balanced_accuracy_weight
            + feature_reduction_weight
        )

        if not np.isclose(total_weight, 1.0):
            raise ValueError(
                "Fitness weights must sum to 1.0"
            )

        self.macro_f1_weight = macro_f1_weight

        self.balanced_accuracy_weight = (
            balanced_accuracy_weight
        )

        self.feature_reduction_weight = (
            feature_reduction_weight
        )

        # ----------------------------------------------------
        # OBJECTIVE
        # ----------------------------------------------------

        if objective not in [
            "minimize",
            "maximize",
        ]:
            raise ValueError(
                "objective must be "
                "'minimize' or 'maximize'."
            )

        self.objective = objective

        # ----------------------------------------------------
        # CLASSIFIER
        # ----------------------------------------------------

        self.random_state = random_state

        if classifier is None:

            # Lightweight proxy classifier.
            #
            # We don't want feature-selection fitness
            # evaluation to become more expensive than
            # federated training itself.

            self.classifier = ExtraTreesClassifier(
                n_estimators=80,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            )

        else:
            self.classifier = classifier

        # ----------------------------------------------------
        # CACHE
        # ----------------------------------------------------

        self.enable_cache = enable_cache
        self.cache = {}

        self.evaluation_count = 0
        self.cache_hits = 0

        # ----------------------------------------------------
        # BEST RESULT
        # ----------------------------------------------------

        self.best_result = None
        self.best_mask = None


    # ========================================================
    # SIGMOID TRANSFER FUNCTION
    # ========================================================

    @staticmethod
    def sigmoid(x):

        x = np.asarray(
            x,
            dtype=float
        )

        # avoid exp overflow
        x = np.clip(
            x,
            -60,
            60
        )

        return 1.0 / (
            1.0 + np.exp(-x)
        )


    # ========================================================
    # CONTINUOUS POSITION → BINARY MASK
    # ========================================================

    def position_to_mask(
        self,
        position: Iterable[float]
    ):

        position = np.asarray(
            position,
            dtype=float
        ).reshape(-1)

        # ----------------------------------------------------
        # DIMENSION CHECK
        # ----------------------------------------------------

        if len(position) != self.n_features:

            raise ValueError(
                f"EVO particle dimension "
                f"{len(position)} does not match "
                f"number of features "
                f"{self.n_features}."
            )

        if not np.all(
            np.isfinite(position)
        ):

            raise ValueError(
                "EVO particle contains "
                "NaN or infinite values."
            )

        # ----------------------------------------------------
        # TRANSFER FUNCTION
        # ----------------------------------------------------

        if self.transfer_function == "sigmoid":

            scores = self.sigmoid(
                position
            )

        else:

            # useful when EVO values are
            # already constrained to [0, 1]

            scores = position

        # ----------------------------------------------------
        # BINARY MASK
        # ----------------------------------------------------

        mask = (
            scores >= self.threshold
        )

        # ----------------------------------------------------
        # PREVENT EMPTY FEATURE SUBSET
        # ----------------------------------------------------

        selected = np.sum(mask)

        if selected < self.min_features:

            best_indices = np.argsort(
                scores
            )[-self.min_features:]

            mask[:] = False

            mask[
                best_indices
            ] = True

        return mask.astype(bool)


    # ========================================================
    # FITNESS CALCULATION
    # ========================================================

    def calculate_fitness(
        self,
        macro_f1,
        balanced_accuracy,
        feature_reduction,
    ):

        """
        Publication-oriented multi-objective utility.

        Primary objective:
            Macro F1

        Secondary:
            Balanced Accuracy

        Optimization pressure:
            Feature Reduction
        """

        utility = (

            self.macro_f1_weight
            * macro_f1

            +

            self.balanced_accuracy_weight
            * balanced_accuracy

            +

            self.feature_reduction_weight
            * feature_reduction

        )

        # ----------------------------------------------------
        # MINIMIZATION
        # ----------------------------------------------------

        if self.objective == "minimize":

            fitness = (
                1.0 - utility
            )

        # ----------------------------------------------------
        # MAXIMIZATION
        # ----------------------------------------------------

        else:

            fitness = utility

        return (
            float(fitness),
            float(utility),
        )


    # ========================================================
    # CHECK BETTER FITNESS
    # ========================================================

    def is_better(
        self,
        new_fitness,
        old_fitness,
    ):

        if self.objective == "minimize":

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
    # FULL PARTICLE EVALUATION
    # ========================================================

    def evaluate(
        self,
        position: Iterable[float],
    ) -> FeatureSelectionResult:

        # ----------------------------------------------------
        # POSITION → MASK
        # ----------------------------------------------------

        mask = self.position_to_mask(
            position
        )

        # ----------------------------------------------------
        # CACHE KEY
        # ----------------------------------------------------

        cache_key = mask.tobytes()

        if (
            self.enable_cache
            and cache_key in self.cache
        ):

            self.cache_hits += 1

            result = self.cache[
                cache_key
            ]

            self._update_best(
                mask,
                result
            )

            return result

        # ----------------------------------------------------
        # FEATURE INDICES
        # ----------------------------------------------------

        selected_indices = np.flatnonzero(
            mask
        )

        selected_count = len(
            selected_indices
        )

        # ----------------------------------------------------
        # SELECT FEATURES
        # ----------------------------------------------------

        X_train_selected = (
            self.X_train[:, mask]
        )

        X_val_selected = (
            self.X_val[:, mask]
        )

        # ----------------------------------------------------
        # TRAIN CLASSIFIER
        # ----------------------------------------------------

        model = clone(
            self.classifier
        )

        model.fit(
            X_train_selected,
            self.y_train
        )

        # ----------------------------------------------------
        # VALIDATION PREDICTION
        # ----------------------------------------------------

        predictions = model.predict(
            X_val_selected
        )

        # ----------------------------------------------------
        # MACRO F1
        # ----------------------------------------------------

        macro_f1 = f1_score(
            self.y_val,
            predictions,
            average="macro",
            zero_division=0,
        )

        # ----------------------------------------------------
        # BALANCED ACCURACY
        # ----------------------------------------------------

        balanced_accuracy = (
            balanced_accuracy_score(
                self.y_val,
                predictions,
            )
        )

        # ----------------------------------------------------
        # FEATURE REDUCTION
        # ----------------------------------------------------

        feature_reduction = (

            1.0

            -

            (
                selected_count
                /
                self.n_features
            )
        )

        # ----------------------------------------------------
        # FITNESS
        # ----------------------------------------------------

        fitness, utility = (
            self.calculate_fitness(

                macro_f1=
                    macro_f1,

                balanced_accuracy=
                    balanced_accuracy,

                feature_reduction=
                    feature_reduction,
            )
        )

        # ----------------------------------------------------
        # SELECTED FEATURE NAMES
        # ----------------------------------------------------

        selected_features = tuple(

            self.feature_names[i]

            for i
            in selected_indices
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        result = FeatureSelectionResult(

            fitness=float(
                fitness
            ),

            utility=float(
                utility
            ),

            macro_f1=float(
                macro_f1
            ),

            balanced_accuracy=float(
                balanced_accuracy
            ),

            feature_reduction=float(
                feature_reduction
            ),

            selected_count=int(
                selected_count
            ),

            total_features=int(
                self.n_features
            ),

            selected_indices=tuple(
                int(i)
                for i
                in selected_indices
            ),

            selected_features=
                selected_features,
        )

        # ----------------------------------------------------
        # COUNT REAL MODEL EVALUATIONS
        # ----------------------------------------------------

        self.evaluation_count += 1

        # ----------------------------------------------------
        # SAVE CACHE
        # ----------------------------------------------------

        if self.enable_cache:

            self.cache[
                cache_key
            ] = result

        # ----------------------------------------------------
        # UPDATE GLOBAL BEST
        # ----------------------------------------------------

        self._update_best(
            mask,
            result
        )

        return result


    # ========================================================
    # FITNESS CALLBACK FOR evo.py
    # ========================================================

    def fitness(
        self,
        position: Iterable[float]
    ) -> float:

        """
        This is the function that evo.py should call.

        EVO does NOT need to know anything about:
            classifier
            Macro F1
            balanced accuracy
            feature reduction
            train/validation data

        EVO only sees:

            position → scalar fitness
        """

        result = self.evaluate(
            position
        )

        return result.fitness


    # ========================================================
    # UPDATE BEST PARTICLE
    # ========================================================

    def _update_best(
        self,
        mask,
        result,
    ):

        if self.best_result is None:

            self.best_result = result
            self.best_mask = mask.copy()

            return

        if self.is_better(
            result.fitness,
            self.best_result.fitness,
        ):

            self.best_result = result
            self.best_mask = mask.copy()


    # ========================================================
    # GET BEST RESULT
    # ========================================================

    def get_best_result(
        self
    ) -> FeatureSelectionResult:

        if self.best_result is None:

            raise RuntimeError(
                "No EVO particles have "
                "been evaluated yet."
            )

        return self.best_result


    # ========================================================
    # GET BEST BINARY MASK
    # ========================================================

    def get_best_mask(
        self
    ):

        if self.best_mask is None:

            raise RuntimeError(
                "No EVO particles have "
                "been evaluated yet."
            )

        return self.best_mask.copy()


    # ========================================================
    # CLEAR CACHE
    # ========================================================

    def clear_cache(
        self
    ):

        self.cache.clear()

        self.cache_hits = 0