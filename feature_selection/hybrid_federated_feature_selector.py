"""
Hybrid EVO-GA Federated-Aware Feature Selector
==============================================

Purpose
-------

This module provides the federated-aware feature-selection
objective used by:

    optimization/hybrid_evo_ga.py

The Hybrid EVO-GA optimizer should NOT use a different
classification model, fitness function, class weighting,
FedAvg implementation, or validation procedure.

For a scientifically fair comparison:

    Federated-aware EVO
    Federated-aware GA
    Hybrid EVO-GA

must all optimize the SAME federated proxy objective.

Therefore this implementation deliberately reuses:

    feature_selection/
        ga_federated_feature_selector.py

which already contains the validated federated-aware fitness
pipeline.

The ONLY thing that changes during Phase 7 is the optimizer:

    EVO
        vs

    GA
        vs

    Hybrid EVO -> GA refinement


Fitness
-------

Same as Phase 5B / Phase 6:

    Utility =

        0.35 * Macro F1
      + 0.30 * Balanced Accuracy
      + 0.20 * Attack Macro Recall
      + 0.10 * Worst Attack Recall
      + 0.05 * Feature Reduction

    Fitness (minimize) =

        1 - Utility + Zero-Attack Penalty


Data integrity
--------------

This class does NOT load:

    validation.csv

or:

    test_locked.csv

The calling script must supply train-derived federated proxy
splits of the form:

    [
        {
            "client_id": 1,
            "X_train": ...,
            "y_train": ...,
            "X_val": ...,
            "y_val": ...
        },
        ...
    ]


Typical usage
-------------

    from optimization.hybrid_evo_ga import (
        HybridEVOGAOptimizer
    )

    from feature_selection.hybrid_federated_feature_selector import (
        HybridFederatedFeatureSelector
    )

    selector = HybridFederatedFeatureSelector(
        client_splits=client_splits,
        feature_names=active_features,
        num_classes=6,
        benign_class_id=0,
        federated_rounds=2,
        local_epochs=1,
        batch_size=1024,
        random_state=42,
    )

    optimizer = HybridEVOGAOptimizer(
        evo_population_size=10,
        evo_iterations=10,
        ga_population_size=10,
        ga_generations=10,
        random_state=42,
    )

    hybrid_result = optimizer.optimize(
        objective_function=selector.fitness,
        dimension=70,
        lower_bound=-6.0,
        upper_bound=6.0,
    )

    final_result = selector.evaluate(
        hybrid_result.best_position
    )
"""

from __future__ import annotations

from typing import (
    Dict,
    Optional,
    Sequence,
)

from feature_selection.ga_federated_feature_selector import (
    GAFederatedFeatureSelector,
    GAFederatedFeatureSelectionResult,
)


# ============================================================
# RESULT ALIAS
#
# The underlying metric/result structure is deliberately
# identical to GA/EVO because the fitness definition must
# remain unchanged.
# ============================================================

HybridFederatedFeatureSelectionResult = (
    GAFederatedFeatureSelectionResult
)


# ============================================================
# HYBRID FEDERATED FEATURE SELECTOR
# ============================================================

class HybridFederatedFeatureSelector(
    GAFederatedFeatureSelector
):

    """
    Federated-aware feature selector for Hybrid EVO-GA.

    This class intentionally inherits all feature-mask,
    federated training, validation, metric, caching, and
    fitness behavior from GAFederatedFeatureSelector.

    The purpose of the subclass is:

        1. Give Phase 7 a semantically clear component.
        2. Avoid duplicating federated-learning code.
        3. Guarantee EVO/GA/Hybrid objective consistency.
        4. Keep future Hybrid-specific extensions isolated.

    No optimizer-specific logic belongs here.
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
        # Continuous position -> binary mask
        # ----------------------------------------------------

        transfer_function: str = "sigmoid",

        threshold: float = 0.5,

        min_features: int = 5,

        # ----------------------------------------------------
        # SAME fitness weights as EVO and GA
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
        # REUSE THE EXACT FEDERATED FITNESS IMPLEMENTATION
        # ====================================================

        super().__init__(

            client_splits=
                client_splits,

            feature_names=
                feature_names,

            num_classes=
                num_classes,

            benign_class_id=
                benign_class_id,

            federated_rounds=
                federated_rounds,

            local_epochs=
                local_epochs,

            batch_size=
                batch_size,

            learning_rate=
                learning_rate,

            weight_decay=
                weight_decay,

            gradient_clip=
                gradient_clip,

            transfer_function=
                transfer_function,

            threshold=
                threshold,

            min_features=
                min_features,

            macro_f1_weight=
                macro_f1_weight,

            balanced_accuracy_weight=
                balanced_accuracy_weight,

            attack_macro_recall_weight=
                attack_macro_recall_weight,

            worst_attack_recall_weight=
                worst_attack_recall_weight,

            feature_reduction_weight=
                feature_reduction_weight,

            zero_attack_penalty_weight=
                zero_attack_penalty_weight,

            zero_recall_threshold=
                zero_recall_threshold,

            objective=
                objective,

            random_state=
                random_state,

            device=
                device,

            enable_cache=
                enable_cache,

            verbose=
                False,
        )


        # ====================================================
        # HYBRID-SPECIFIC METADATA
        # ====================================================

        self.selector_name = (
            "Hybrid Federated-Aware Feature Selector"
        )

        self.optimizer_family = (
            "Hybrid EVO-GA"
        )

        self.verbose = bool(
            verbose
        )


        # ====================================================
        # DISPLAY
        # ====================================================

        if self.verbose:

            print()

            print(
                "=" * 78
            )

            print(
                "HYBRID EVO-GA FEDERATED-AWARE "
                "FEATURE SELECTOR"
            )

            print(
                "=" * 78
            )

            print(
                f"Clients                 : "
                f"{len(self.client_splits)}"
            )

            print(
                f"Starting features       : "
                f"{self.n_features}"
            )

            print(
                f"Classes                 : "
                f"{self.num_classes}"
            )

            print(
                f"Attack classes          : "
                f"{len(self.attack_class_ids)}"
            )

            print(
                f"Proxy FedAvg rounds     : "
                f"{self.federated_rounds}"
            )

            print(
                f"Local epochs            : "
                f"{self.local_epochs}"
            )

            print(
                f"Batch size              : "
                f"{self.batch_size}"
            )

            print(
                f"Minimum features        : "
                f"{self.min_features}"
            )

            print(
                f"Objective               : "
                f"{self.objective}"
            )

            print(
                f"Device                  : "
                f"{self.device}"
            )

            print()

            print(
                "FITNESS"
            )

            print(
                f"  Macro F1              : "
                f"{self.macro_f1_weight:.2f}"
            )

            print(
                f"  Balanced Accuracy     : "
                f"{self.balanced_accuracy_weight:.2f}"
            )

            print(
                f"  Attack Macro Recall   : "
                f"{self.attack_macro_recall_weight:.2f}"
            )

            print(
                f"  Worst Attack Recall   : "
                f"{self.worst_attack_recall_weight:.2f}"
            )

            print(
                f"  Feature Reduction     : "
                f"{self.feature_reduction_weight:.2f}"
            )

            print(
                f"  Zero-Attack Penalty   : "
                f"{self.zero_attack_penalty_weight:.2f}"
            )

            print(
                "=" * 78
            )


    # ========================================================
    # OPTIONAL METADATA
    # ========================================================

    def get_configuration(
        self,
    ) -> Dict:

        """
        Return the complete Phase-7 federated proxy
        configuration for experiment manifests.
        """

        return {

            "selector":
                self.selector_name,

            "optimizer_family":
                self.optimizer_family,

            "clients":
                int(
                    len(
                        self.client_splits
                    )
                ),

            "total_features":
                int(
                    self.n_features
                ),

            "num_classes":
                int(
                    self.num_classes
                ),

            "benign_class_id":
                int(
                    self.benign_class_id
                ),

            "attack_class_ids":
                [
                    int(x)
                    for x
                    in self.attack_class_ids
                ],

            "federated_rounds":
                int(
                    self.federated_rounds
                ),

            "local_epochs":
                int(
                    self.local_epochs
                ),

            "batch_size":
                int(
                    self.batch_size
                ),

            "learning_rate":
                float(
                    self.learning_rate
                ),

            "weight_decay":
                float(
                    self.weight_decay
                ),

            "gradient_clip":
                float(
                    self.gradient_clip
                ),

            "transfer_function":
                self.transfer_function,

            "threshold":
                float(
                    self.threshold
                ),

            "min_features":
                int(
                    self.min_features
                ),

            "fitness_weights": {

                "macro_f1":
                    float(
                        self.macro_f1_weight
                    ),

                "balanced_accuracy":
                    float(
                        self.balanced_accuracy_weight
                    ),

                "attack_macro_recall":
                    float(
                        self.attack_macro_recall_weight
                    ),

                "worst_attack_recall":
                    float(
                        self.worst_attack_recall_weight
                    ),

                "feature_reduction":
                    float(
                        self.feature_reduction_weight
                    ),
            },

            "zero_attack_penalty_weight":
                float(
                    self.zero_attack_penalty_weight
                ),

            "zero_recall_threshold":
                float(
                    self.zero_recall_threshold
                ),

            "objective":
                self.objective,

            "random_state":
                int(
                    self.random_state
                ),

            "device":
                str(
                    self.device
                ),

            "cache_enabled":
                bool(
                    self.enable_cache
                ),
        }


    # ========================================================
    # PHASE-7 STATUS
    # ========================================================

    def get_search_statistics(
        self,
    ) -> Dict:

        """
        Return runtime search statistics useful for the
        Phase-7 summary JSON.
        """

        best_fitness = None
        best_feature_count = None

        if self.best_result is not None:

            best_fitness = float(
                self.best_result.fitness
            )

            best_feature_count = int(
                self.best_result.selected_count
            )

        return {

            "real_federated_evaluations":
                int(
                    self.evaluation_count
                ),

            "cache_hits":
                int(
                    self.cache_hits
                ),

            "cached_masks":
                int(
                    len(
                        self.cache
                    )
                ),

            "best_fitness":
                best_fitness,

            "best_feature_count":
                best_feature_count,
        }