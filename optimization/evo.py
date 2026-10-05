"""
Energy Valley Optimizer (EVO)
=============================

Reusable implementation of the Energy Valley Optimizer.

Important:
    This module contains ONLY the optimization algorithm.

It does NOT contain:
    - feature-selection logic
    - ML models
    - FedAvg
    - CICIDS data loading
    - fitness-function design
    - binary feature masks

Therefore the same EVO implementation can later be reused for:
    - feature selection
    - hyperparameter optimization
    - neural-network parameter optimization
    - hybrid EVO-GA
    - other research experiments

Reference:
    Azizi et al.
    "Energy valley optimizer: a novel metaheuristic algorithm
    for global and engineering optimization"
    Scientific Reports, 2023.

Optimization convention:
    MINIMIZATION

The supplied objective function must therefore return a lower score
for a better solution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np


ArrayLike = Union[float, int, List[float], np.ndarray]


# ============================================================
# RESULT OBJECT
# ============================================================

@dataclass
class EVOResult:
    """
    Stores the result returned by the EVO optimizer.
    """

    best_position: np.ndarray
    best_fitness: float
    convergence_history: List[float]
    population: np.ndarray
    population_fitness: np.ndarray
    iterations: int
    evaluations: int


# ============================================================
# ENERGY VALLEY OPTIMIZER
# ============================================================

class EnergyValleyOptimizer:
    """
    Energy Valley Optimizer (EVO).

    The optimizer is intentionally problem-independent.

    Example
    -------
    >>> def sphere(x):
    ...     return np.sum(x ** 2)
    ...
    >>> optimizer = EnergyValleyOptimizer(
    ...     population_size=20,
    ...     max_iterations=50,
    ...     random_state=42
    ... )
    ...
    >>> result = optimizer.optimize(
    ...     objective_function=sphere,
    ...     dimension=10,
    ...     lower_bound=-10,
    ...     upper_bound=10
    ... )
    ...
    >>> print(result.best_fitness)
    """

    def __init__(
        self,
        population_size: int = 20,
        max_iterations: int = 25,
        random_state: Optional[int] = 42,
        verbose: bool = True,
        epsilon: float = 1e-12,
    ):
        if population_size < 2:
            raise ValueError(
                "population_size must be at least 2."
            )

        if max_iterations < 1:
            raise ValueError(
                "max_iterations must be at least 1."
            )

        self.population_size = population_size
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.verbose = verbose
        self.epsilon = epsilon

        self.rng = np.random.default_rng(random_state)

        self.evaluations = 0

    # ========================================================
    # PUBLIC OPTIMIZATION METHOD
    # ========================================================

    def optimize(
        self,
        objective_function: Callable[[np.ndarray], float],
        dimension: int,
        lower_bound: ArrayLike = 0.0,
        upper_bound: ArrayLike = 1.0,
    ) -> EVOResult:
        """
        Run Energy Valley Optimization.

        Parameters
        ----------
        objective_function:
            Function receiving one candidate vector and returning
            a scalar fitness.

            IMPORTANT:
                EVO minimizes this value.

        dimension:
            Number of decision variables.

            For our CICIDS feature-selection problem this will
            eventually be:

                dimension = 70

        lower_bound:
            Scalar or vector containing lower bounds.

        upper_bound:
            Scalar or vector containing upper bounds.

        Returns
        -------
        EVOResult
            Complete optimization result.
        """

        if dimension < 1:
            raise ValueError(
                "dimension must be greater than zero."
            )

        lower = self._prepare_bound(
            lower_bound,
            dimension
        )

        upper = self._prepare_bound(
            upper_bound,
            dimension
        )

        if np.any(lower >= upper):
            raise ValueError(
                "Every lower bound must be smaller "
                "than its corresponding upper bound."
            )

        self.evaluations = 0

        # ----------------------------------------------------
        # STEP 1
        # Initialize particles
        # ----------------------------------------------------

        population = self.rng.uniform(
            low=lower,
            high=upper,
            size=(self.population_size, dimension)
        )

        fitness = self._evaluate_population(
            population,
            objective_function
        )

        convergence_history: List[float] = []

        # ====================================================
        # MAIN EVO LOOP
        # ====================================================

        for iteration in range(1, self.max_iterations + 1):

            # ------------------------------------------------
            # Best and worst particles
            # EVO is implemented as MINIMIZATION.
            # ------------------------------------------------

            best_index = int(np.argmin(fitness))
            worst_index = int(np.argmax(fitness))

            best_fitness = float(fitness[best_index])
            worst_fitness = float(fitness[worst_index])

            best_particle = population[best_index].copy()

            # ------------------------------------------------
            # STEP 2
            # Enrichment Bound (EB)
            #
            # EB = mean(NEL)
            #
            # Here:
            #     NEL = objective fitness
            # ------------------------------------------------

            enrichment_bound = float(
                np.mean(fitness)
            )

            # ------------------------------------------------
            # STEP 3
            # Stability Level (SL)
            #
            # SL_i =
            # (NEL_i - BS) / (WS - BS)
            #
            # BS = best objective value
            # WS = worst objective value
            # ------------------------------------------------

            denominator = (
                worst_fitness
                - best_fitness
            )

            if abs(denominator) < self.epsilon:

                stability_levels = np.zeros(
                    self.population_size,
                    dtype=float
                )

            else:

                stability_levels = (
                    fitness - best_fitness
                ) / denominator

            # Particle center
            particle_center = np.mean(
                population,
                axis=0
            )

            # ------------------------------------------------
            # Generate new EVO candidates
            # ------------------------------------------------

            new_candidates = []

            for i in range(self.population_size):

                current = population[i].copy()

                current_fitness = float(
                    fitness[i]
                )

                stability = float(
                    stability_levels[i]
                )

                # Find nearest neighboring particle
                neighbor = self._nearest_neighbor(
                    population,
                    i
                )

                # ============================================
                # CASE 1
                # NEL_i > EB
                # ============================================

                if current_fitness > enrichment_bound:

                    stability_bound = float(
                        self.rng.random()
                    )

                    # ========================================
                    # Alpha + Gamma decay
                    # ========================================

                    if stability > stability_bound:

                        # ------------------------------------
                        # ALPHA DECAY
                        #
                        # Some decision variables from the
                        # current particle are replaced by
                        # values from the best particle.
                        # ------------------------------------

                        alpha_candidate = (
                            self._alpha_decay(
                                current=current,
                                best=best_particle
                            )
                        )

                        alpha_candidate = np.clip(
                            alpha_candidate,
                            lower,
                            upper
                        )

                        new_candidates.append(
                            alpha_candidate
                        )

                        # ------------------------------------
                        # GAMMA DECAY
                        #
                        # Some decision variables are replaced
                        # with variables from the nearest
                        # neighboring particle.
                        # ------------------------------------

                        gamma_candidate = (
                            self._gamma_decay(
                                current=current,
                                neighbor=neighbor
                            )
                        )

                        gamma_candidate = np.clip(
                            gamma_candidate,
                            lower,
                            upper
                        )

                        new_candidates.append(
                            gamma_candidate
                        )

                    # ========================================
                    # Beta decay
                    # ========================================

                    else:

                        safe_stability = max(
                            stability,
                            self.epsilon
                        )

                        r1 = float(
                            self.rng.random()
                        )

                        r2 = float(
                            self.rng.random()
                        )

                        # Equation:
                        #
                        # Xnew1 =
                        # Xi +
                        # (r1 * Xbest - r2 * Xcenter)
                        # / SL_i

                        beta_candidate_1 = (
                            current
                            + (
                                r1 * best_particle
                                - r2 * particle_center
                            )
                            / safe_stability
                        )

                        beta_candidate_1 = np.clip(
                            beta_candidate_1,
                            lower,
                            upper
                        )

                        new_candidates.append(
                            beta_candidate_1
                        )

                        # ------------------------------------
                        # Second beta-decay movement
                        # ------------------------------------

                        r3 = float(
                            self.rng.random()
                        )

                        r4 = float(
                            self.rng.random()
                        )

                        # Equation:
                        #
                        # Xnew2 =
                        # Xi +
                        # (r3*Xbest - r4*Xneighbor)

                        beta_candidate_2 = (
                            current
                            + (
                                r3 * best_particle
                                - r4 * neighbor
                            )
                        )

                        beta_candidate_2 = np.clip(
                            beta_candidate_2,
                            lower,
                            upper
                        )

                        new_candidates.append(
                            beta_candidate_2
                        )

                # ============================================
                # CASE 2
                # NEL_i <= EB
                #
                # Positron emission / electron capture
                # ============================================

                else:

                    random_movement = float(
                        self.rng.random()
                    )

                    movement_candidate = (
                        current
                        + random_movement
                    )

                    movement_candidate = np.clip(
                        movement_candidate,
                        lower,
                        upper
                    )

                    new_candidates.append(
                        movement_candidate
                    )

            # ------------------------------------------------
            # Convert candidates to array
            # ------------------------------------------------

            if len(new_candidates) == 0:
                continue

            new_candidates = np.asarray(
                new_candidates,
                dtype=float
            )

            # ------------------------------------------------
            # Evaluate new particles
            # ------------------------------------------------

            new_fitness = self._evaluate_population(
                new_candidates,
                objective_function
            )

            # ------------------------------------------------
            # Merge:
            #
            # existing population
            #        +
            # newly generated population
            # ------------------------------------------------

            combined_population = np.vstack(
                (
                    population,
                    new_candidates
                )
            )

            combined_fitness = np.concatenate(
                (
                    fitness,
                    new_fitness
                )
            )

            # ------------------------------------------------
            # Sort by objective fitness
            #
            # Lower fitness is better.
            # ------------------------------------------------

            sorted_indices = np.argsort(
                combined_fitness
            )

            # ------------------------------------------------
            # Survival of best N particles
            # ------------------------------------------------

            selected_indices = sorted_indices[
                :self.population_size
            ]

            population = combined_population[
                selected_indices
            ].copy()

            fitness = combined_fitness[
                selected_indices
            ].copy()

            # ------------------------------------------------
            # Record best fitness
            # ------------------------------------------------

            current_best = float(
                np.min(fitness)
            )

            convergence_history.append(
                current_best
            )

            if self.verbose:

                print(
                    f"EVO Iteration "
                    f"{iteration:03d}/"
                    f"{self.max_iterations:03d}"
                    f" | Best Fitness: "
                    f"{current_best:.8f}"
                )

        # ====================================================
        # FINAL RESULT
        # ====================================================

        best_index = int(
            np.argmin(fitness)
        )

        best_position = population[
            best_index
        ].copy()

        best_fitness = float(
            fitness[best_index]
        )

        return EVOResult(
            best_position=best_position,
            best_fitness=best_fitness,
            convergence_history=convergence_history,
            population=population.copy(),
            population_fitness=fitness.copy(),
            iterations=self.max_iterations,
            evaluations=self.evaluations,
        )

    # ========================================================
    # ALPHA DECAY
    # ========================================================

    def _alpha_decay(
        self,
        current: np.ndarray,
        best: np.ndarray,
    ) -> np.ndarray:
        """
        Alpha-decay position update.

        Random decision variables in the current particle
        are replaced by values from the current best particle.
        """

        dimension = len(current)

        candidate = current.copy()

        # Number of emitted/replaced variables
        number_of_variables = int(
            self.rng.integers(
                low=1,
                high=dimension + 1
            )
        )

        indices = self.rng.choice(
            dimension,
            size=number_of_variables,
            replace=False
        )

        candidate[indices] = best[indices]

        return candidate

    # ========================================================
    # GAMMA DECAY
    # ========================================================

    def _gamma_decay(
        self,
        current: np.ndarray,
        neighbor: np.ndarray,
    ) -> np.ndarray:
        """
        Gamma-decay position update.

        Random decision variables are replaced by values
        from the nearest neighboring particle.
        """

        dimension = len(current)

        candidate = current.copy()

        number_of_variables = int(
            self.rng.integers(
                low=1,
                high=dimension + 1
            )
        )

        indices = self.rng.choice(
            dimension,
            size=number_of_variables,
            replace=False
        )

        candidate[indices] = neighbor[indices]

        return candidate

    # ========================================================
    # NEAREST NEIGHBOR
    # ========================================================

    def _nearest_neighbor(
        self,
        population: np.ndarray,
        particle_index: int,
    ) -> np.ndarray:
        """
        Return the nearest particle using Euclidean distance.
        """

        current = population[
            particle_index
        ]

        distances = np.linalg.norm(
            population - current,
            axis=1
        )

        # Prevent selecting itself
        distances[particle_index] = np.inf

        nearest_index = int(
            np.argmin(distances)
        )

        return population[
            nearest_index
        ].copy()

    # ========================================================
    # FITNESS EVALUATION
    # ========================================================

    def _evaluate_population(
        self,
        population: np.ndarray,
        objective_function: Callable[
            [np.ndarray],
            float
        ],
    ) -> np.ndarray:
        """
        Evaluate every particle in the population.
        """

        fitness_values = np.empty(
            len(population),
            dtype=float
        )

        for i, particle in enumerate(population):

            value = objective_function(
                particle.copy()
            )

            value = float(value)

            if not np.isfinite(value):

                value = np.inf

            fitness_values[i] = value

            self.evaluations += 1

        return fitness_values

    # ========================================================
    # BOUND PREPARATION
    # ========================================================

    @staticmethod
    def _prepare_bound(
        bound: ArrayLike,
        dimension: int,
    ) -> np.ndarray:
        """
        Convert scalar/vector bounds into dimension-sized array.
        """

        array = np.asarray(
            bound,
            dtype=float
        )

        if array.ndim == 0:

            return np.full(
                dimension,
                float(array),
                dtype=float
            )

        if len(array) != dimension:

            raise ValueError(
                f"Bound length {len(array)} "
                f"does not match dimension "
                f"{dimension}."
            )

        return array.copy()


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("ENERGY VALLEY OPTIMIZER — TEST")
    print("=" * 60)

    # Sphere benchmark function
    #
    # Global optimum:
    #
    #     x = [0, 0, ..., 0]
    #     fitness = 0

    def sphere_function(x: np.ndarray) -> float:
        return float(
            np.sum(x ** 2)
        )

    evo = EnergyValleyOptimizer(
        population_size=20,
        max_iterations=25,
        random_state=42,
        verbose=True,
    )

    result = evo.optimize(
        objective_function=sphere_function,
        dimension=10,
        lower_bound=-10.0,
        upper_bound=10.0,
    )

    print()
    print("=" * 60)
    print("EVO COMPLETE")
    print("=" * 60)

    print(
        f"Best fitness : "
        f"{result.best_fitness:.10f}"
    )

    print(
        f"Best position: "
        f"{result.best_position}"
    )

    print(
        f"Evaluations  : "
        f"{result.evaluations}"
    )