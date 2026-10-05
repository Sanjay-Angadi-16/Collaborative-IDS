# ============================================================
# feature_selection/evo_feature_selector.py
# ============================================================

from dataclasses import dataclass, asdict

import numpy as np

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
)


# ============================================================
# RESULT
# ============================================================

@dataclass(frozen=True)
class FeatureSelectionResult:

    fitness: float
    utility: float

    macro_f1: float
    balanced_accuracy: float
    feature_reduction: float

    selected_count: int
    total_features: int

    selected_indices: tuple
    selected_features: tuple

    def to_dict(self):
        return asdict(self)


# ============================================================
# EVO FEATURE SELECTOR
# ============================================================

class EVOFeatureSelector:

    def __init__(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        feature_names=None,
        transfer_function="sigmoid",
        threshold=0.5,
        min_features=1,
        macro_f1_weight=0.50,
        balanced_accuracy_weight=0.35,
        feature_reduction_weight=0.15,
        objective="minimize",
        random_state=42,
        classifier=None,
        enable_cache=True,
    ):

        # ----------------------------------------------------
        # DATA
        # ----------------------------------------------------

        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)

        self.X_val = np.asarray(X_val)
        self.y_val = np.asarray(y_val)

        if self.X_train.ndim != 2:
            raise ValueError(
                "X_train must be 2-dimensional."
            )

        if self.X_val.ndim != 2:
            raise ValueError(
                "X_val must be 2-dimensional."
            )

        if (
            self.X_train.shape[1]
            !=
            self.X_val.shape[1]
        ):
            raise ValueError(
                "Train/validation feature counts do not match."
            )

        self.n_features = (
            self.X_train.shape[1]
        )

        # ----------------------------------------------------
        # FEATURE NAMES
        # ----------------------------------------------------

        if feature_names is None:

            self.feature_names = tuple(
                f"feature_{i}"
                for i in range(
                    self.n_features
                )
            )

        else:

            if (
                len(feature_names)
                !=
                self.n_features
            ):
                raise ValueError(
                    "feature_names count does not "
                    "match X_train."
                )

            self.feature_names = tuple(
                feature_names
            )

        # ----------------------------------------------------
        # TRANSFER
        # ----------------------------------------------------

        self.transfer_function = (
            transfer_function
        )

        self.threshold = threshold
        self.min_features = min_features

        # ----------------------------------------------------
        # FITNESS WEIGHTS
        # ----------------------------------------------------

        self.macro_f1_weight = (
            macro_f1_weight
        )

        self.balanced_accuracy_weight = (
            balanced_accuracy_weight
        )

        self.feature_reduction_weight = (
            feature_reduction_weight
        )

        total_weight = (

            macro_f1_weight

            +

            balanced_accuracy_weight

            +

            feature_reduction_weight
        )

        if not np.isclose(
            total_weight,
            1.0
        ):
            raise ValueError(
                "Fitness weights must sum to 1."
            )

        # ----------------------------------------------------
        # OBJECTIVE
        # ----------------------------------------------------

        if objective not in (
            "minimize",
            "maximize",
        ):
            raise ValueError(
                "objective must be minimize or maximize"
            )

        self.objective = objective

        self.random_state = (
            random_state
        )

        # ----------------------------------------------------
        # CLASSIFIER
        # ----------------------------------------------------

        if classifier is None:

            self.classifier = (
                ExtraTreesClassifier(

                    n_estimators=80,

                    class_weight="balanced",

                    random_state=
                        random_state,

                    n_jobs=-1,
                )
            )

        else:

            self.classifier = classifier

        # ----------------------------------------------------
        # CACHE
        # ----------------------------------------------------

        self.enable_cache = (
            enable_cache
        )

        self.cache = {}

        self.evaluation_count = 0
        self.cache_hits = 0

        # ----------------------------------------------------
        # BEST RESULT
        # ----------------------------------------------------

        self.best_result = None
        self.best_mask = None


    # ========================================================
    # SIGMOID
    # ========================================================

    @staticmethod
    def sigmoid(x):

        x = np.asarray(
            x,
            dtype=float
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
                np.exp(-x)
            )
        )


    # ========================================================
    # CONTINUOUS PARTICLE -> BINARY MASK
    # ========================================================

    def position_to_mask(
        self,
        position,
    ):

        position = np.asarray(
            position,
            dtype=float
        ).reshape(-1)

        if (
            len(position)
            !=
            self.n_features
        ):

            raise ValueError(

                f"EVO particle contains "
                f"{len(position)} dimensions, "
                f"but dataset contains "
                f"{self.n_features} features."
            )

        # ----------------------------------------------------
        # transfer
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # binary feature mask
        # ----------------------------------------------------

        mask = (
            scores
            >=
            self.threshold
        )

        # ----------------------------------------------------
        # prevent zero features
        # ----------------------------------------------------

        if (
            np.sum(mask)
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

        return mask.astype(bool)


    # ========================================================
    # FITNESS
    # ========================================================

    def calculate_fitness(
        self,
        macro_f1,
        balanced_accuracy,
        feature_reduction,
    ):

        utility = (

            self.macro_f1_weight
            *
            macro_f1

            +

            self.balanced_accuracy_weight
            *
            balanced_accuracy

            +

            self.feature_reduction_weight
            *
            feature_reduction
        )

        if (
            self.objective
            ==
            "minimize"
        ):

            fitness = (
                1.0
                -
                utility
            )

        else:

            fitness = utility

        return (
            float(fitness),
            float(utility),
        )


    # ========================================================
    # EVALUATE PARTICLE
    # ========================================================

    def evaluate(
        self,
        position,
    ):

        mask = self.position_to_mask(
            position
        )

        # ----------------------------------------------------
        # duplicate mask cache
        # ----------------------------------------------------

        key = mask.tobytes()

        if (
            self.enable_cache
            and
            key in self.cache
        ):

            self.cache_hits += 1

            return self.cache[key]

        # ----------------------------------------------------
        # selected feature indices
        # ----------------------------------------------------

        selected_indices = (
            np.flatnonzero(mask)
        )

        selected_count = len(
            selected_indices
        )

        # ----------------------------------------------------
        # subset
        # ----------------------------------------------------

        X_train_selected = (
            self.X_train[:, mask]
        )

        X_val_selected = (
            self.X_val[:, mask]
        )

        # ----------------------------------------------------
        # classifier
        # ----------------------------------------------------

        model = clone(
            self.classifier
        )

        model.fit(
            X_train_selected,
            self.y_train,
        )

        predictions = model.predict(
            X_val_selected
        )

        # ----------------------------------------------------
        # metrics
        # ----------------------------------------------------

        macro_f1 = f1_score(

            self.y_val,
            predictions,

            average="macro",
            zero_division=0,
        )

        balanced_accuracy = (
            balanced_accuracy_score(

                self.y_val,
                predictions,
            )
        )

        # ----------------------------------------------------
        # feature reduction
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
        # fitness
        # ----------------------------------------------------

        (
            fitness,
            utility,

        ) = self.calculate_fitness(

            macro_f1,

            balanced_accuracy,

            feature_reduction,
        )

        # ----------------------------------------------------
        # feature names
        # ----------------------------------------------------

        selected_features = tuple(

            self.feature_names[i]

            for i
            in selected_indices
        )

        # ----------------------------------------------------
        # result object
        # ----------------------------------------------------

        result = FeatureSelectionResult(

            fitness=
                fitness,

            utility=
                utility,

            macro_f1=
                float(macro_f1),

            balanced_accuracy=
                float(
                    balanced_accuracy
                ),

            feature_reduction=
                float(
                    feature_reduction
                ),

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
                    int(i)
                    for i
                    in selected_indices
                ),

            selected_features=
                selected_features,
        )

        self.evaluation_count += 1

        # ----------------------------------------------------
        # cache
        # ----------------------------------------------------

        if self.enable_cache:
            self.cache[key] = result

        # ----------------------------------------------------
        # best solution
        # ----------------------------------------------------

        if self.best_result is None:

            self.best_result = result
            self.best_mask = mask.copy()

        else:

            if self.objective == "minimize":

                better = (
                    result.fitness
                    <
                    self.best_result.fitness
                )

            else:

                better = (
                    result.fitness
                    >
                    self.best_result.fitness
                )

            if better:

                self.best_result = result
                self.best_mask = (
                    mask.copy()
                )

        return result


    # ========================================================
    # CALLBACK USED BY evo.py
    # ========================================================

    def fitness(
        self,
        position,
    ):

        return self.evaluate(
            position
        ).fitness


    # ========================================================
    # BEST RESULT
    # ========================================================

    def get_best_result(self):

        if self.best_result is None:

            raise RuntimeError(
                "EVO has not evaluated "
                "any particles yet."
            )

        return self.best_result


    def get_best_mask(self):

        if self.best_mask is None:

            raise RuntimeError(
                "EVO has not evaluated "
                "any particles yet."
            )

        return self.best_mask.copy()


    # ========================================================
    # CLEAR CACHE
    # ========================================================

    def clear_cache(self):

        self.cache.clear()

        self.cache_hits = 0