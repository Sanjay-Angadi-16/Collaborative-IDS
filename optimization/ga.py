"""
Genetic Algorithm Optimizer
===========================

Generic real-valued Genetic Algorithm for continuous search spaces.

Designed to work with feature-selection wrappers such as:

    feature_selection/ga_federated_feature_selector.py

The optimizer itself knows nothing about:
    - datasets
    - feature masks
    - federated learning
    - IDS
    - classifiers

It only solves:

    minimize or maximize objective_function(position)

Example:

    from optimization.ga import GeneticAlgorithmOptimizer

    optimizer = GeneticAlgorithmOptimizer(
        population_size=20,
        max_generations=30,
        crossover_rate=0.9,
        mutation_rate=0.1,
        random_state=42,
        verbose=True,
    )

    result = optimizer.optimize(
        objective_function=selector.fitness,
        dimension=70,
        lower_bound=-6.0,
        upper_bound=6.0,
    )
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import (
    Callable,
    List,
    Optional,
    Union,
)

import numpy as np


ArrayLike = Union[
    float,
    int,
    List[float],
    np.ndarray,
]


# ============================================================
# RESULT
# ============================================================

@dataclass(frozen=True)
class GAResult:

    best_position: np.ndarray

    best_fitness: float

    convergence_history: List[float]

    population: np.ndarray

    population_fitness: np.ndarray

    generations: int

    evaluations: int


# ============================================================
# GENETIC ALGORITHM
# ============================================================

class GeneticAlgorithmOptimizer:

    """
    Generic Genetic Algorithm.

    Default behavior:
        objective = minimization

    Main operations:
        1. Population initialization
        2. Fitness evaluation
        3. Tournament selection
        4. Arithmetic crossover
        5. Gaussian mutation
        6. Elitism
        7. Bound clipping
    """

    def __init__(
        self,
        population_size: int = 20,
        max_generations: int = 30,
        crossover_rate: float = 0.90,
        mutation_rate: float = 0.10,
        mutation_scale: float = 0.15,
        tournament_size: int = 3,
        elitism_count: int = 1,
        random_state: Optional[int] = 42,
        verbose: bool = True,
        objective: str = "minimize",
    ):

        # ====================================================
        # VALIDATION
        # ====================================================

        if population_size < 2:

            raise ValueError(
                "population_size must be >= 2."
            )

        if max_generations < 1:

            raise ValueError(
                "max_generations must be >= 1."
            )

        if not 0.0 <= crossover_rate <= 1.0:

            raise ValueError(
                "crossover_rate must be between 0 and 1."
            )

        if not 0.0 <= mutation_rate <= 1.0:

            raise ValueError(
                "mutation_rate must be between 0 and 1."
            )

        if mutation_scale < 0.0:

            raise ValueError(
                "mutation_scale must be >= 0."
            )

        if tournament_size < 2:

            raise ValueError(
                "tournament_size must be >= 2."
            )

        if elitism_count < 0:

            raise ValueError(
                "elitism_count must be >= 0."
            )

        if elitism_count >= population_size:

            raise ValueError(
                "elitism_count must be smaller than population_size."
            )

        if objective not in {
            "minimize",
            "maximize",
        }:

            raise ValueError(
                "objective must be 'minimize' or 'maximize'."
            )


        # ====================================================
        # CONFIG
        # ====================================================

        self.population_size = int(
            population_size
        )

        self.max_generations = int(
            max_generations
        )

        self.crossover_rate = float(
            crossover_rate
        )

        self.mutation_rate = float(
            mutation_rate
        )

        self.mutation_scale = float(
            mutation_scale
        )

        self.tournament_size = int(
            tournament_size
        )

        self.elitism_count = int(
            elitism_count
        )

        self.random_state = (
            None
            if random_state is None
            else int(random_state)
        )

        self.verbose = bool(
            verbose
        )

        self.objective = objective


    # ========================================================
    # BOUNDS
    # ========================================================

    @staticmethod
    def _prepare_bounds(
        value: ArrayLike,
        dimension: int,
        name: str,
    ) -> np.ndarray:

        if np.isscalar(
            value
        ):

            array = np.full(
                dimension,
                float(value),
                dtype=np.float64,
            )

        else:

            array = np.asarray(
                value,
                dtype=np.float64,
            ).reshape(-1)

            if len(
                array
            ) != dimension:

                raise ValueError(
                    f"{name} must contain "
                    f"{dimension} values."
                )

        if not np.all(
            np.isfinite(
                array
            )
        ):

            raise ValueError(
                f"{name} contains NaN/inf."
            )

        return array


    # ========================================================
    # FITNESS COMPARISON
    # ========================================================

    def _better(
        self,
        a: float,
        b: float,
    ) -> bool:

        if self.objective == "minimize":

            return a < b

        return a > b


    def _best_index(
        self,
        fitness: np.ndarray,
    ) -> int:

        if self.objective == "minimize":

            return int(
                np.argmin(
                    fitness
                )
            )

        return int(
            np.argmax(
                fitness
            )
        )


    def _sorted_indices(
        self,
        fitness: np.ndarray,
    ) -> np.ndarray:

        if self.objective == "minimize":

            return np.argsort(
                fitness
            )

        return np.argsort(
            -fitness
        )


    # ========================================================
    # INITIAL POPULATION
    # ========================================================

    def _initialize_population(
        self,
        rng,
        lower_bound,
        upper_bound,
        dimension,
    ):

        return rng.uniform(

            low=
                lower_bound,

            high=
                upper_bound,

            size=(
                self.population_size,
                dimension,
            ),
        )


    # ========================================================
    # EVALUATE POPULATION
    # ========================================================

    @staticmethod
    def _evaluate_population(
        population,
        objective_function,
    ):

        fitness = np.empty(
            len(
                population
            ),
            dtype=np.float64,
        )

        for index, individual in enumerate(
            population
        ):

            value = objective_function(
                individual
            )

            value = float(
                value
            )

            if not np.isfinite(
                value
            ):

                raise ValueError(
                    "Objective function returned "
                    "NaN or infinite fitness."
                )

            fitness[
                index
            ] = value

        return fitness


    # ========================================================
    # TOURNAMENT SELECTION
    # ========================================================

    def _tournament_select(
        self,
        population,
        fitness,
        rng,
    ):

        tournament_size = min(
            self.tournament_size,
            len(
                population
            ),
        )

        candidate_indices = (
            rng.choice(

                len(
                    population
                ),

                size=
                    tournament_size,

                replace=False,
            )
        )

        candidate_fitness = (
            fitness[
                candidate_indices
            ]
        )

        local_best_index = (
            self._best_index(
                candidate_fitness
            )
        )

        winner_index = int(
            candidate_indices[
                local_best_index
            ]
        )

        return population[
            winner_index
        ].copy()


    # ========================================================
    # CROSSOVER
    # ========================================================

    def _crossover(
        self,
        parent1,
        parent2,
        rng,
    ):

        if (
            rng.random()
            >
            self.crossover_rate
        ):

            return (
                parent1.copy(),
                parent2.copy(),
            )

        # ----------------------------------------------------
        # Blend/arithmetic crossover
        #
        # Per-gene alpha gives more diversity than using one
        # single alpha for the full chromosome.
        # ----------------------------------------------------

        alpha = rng.random(
            size=
                parent1.shape
        )

        child1 = (

            alpha
            *
            parent1

            +

            (
                1.0
                -
                alpha
            )
            *
            parent2
        )

        child2 = (

            alpha
            *
            parent2

            +

            (
                1.0
                -
                alpha
            )
            *
            parent1
        )

        return (
            child1,
            child2,
        )


    # ========================================================
    # MUTATION
    # ========================================================

    def _mutate(
        self,
        individual,
        lower_bound,
        upper_bound,
        rng,
    ):

        child = individual.copy()

        mutation_mask = (

            rng.random(
                size=
                    child.shape
            )

            <
            self.mutation_rate
        )

        if np.any(
            mutation_mask
        ):

            search_range = (

                upper_bound

                -
                lower_bound
            )

            standard_deviation = (

                self.mutation_scale

                *
                search_range
            )

            noise = rng.normal(

                loc=
                    0.0,

                scale=
                    standard_deviation,

                size=
                    child.shape,
            )

            child[
                mutation_mask
            ] += noise[
                mutation_mask
            ]

        child = np.clip(

            child,

            lower_bound,

            upper_bound,
        )

        return child


    # ========================================================
    # OPTIMIZE
    # ========================================================

    def optimize(
        self,
        objective_function: Callable[
            [np.ndarray],
            float,
        ],
        dimension: int,
        lower_bound: ArrayLike = 0.0,
        upper_bound: ArrayLike = 1.0,
    ) -> GAResult:

        # ====================================================
        # VALIDATE
        # ====================================================

        if dimension < 1:

            raise ValueError(
                "dimension must be >= 1."
            )

        lower = self._prepare_bounds(

            value=
                lower_bound,

            dimension=
                dimension,

            name=
                "lower_bound",
        )

        upper = self._prepare_bounds(

            value=
                upper_bound,

            dimension=
                dimension,

            name=
                "upper_bound",
        )

        if np.any(
            upper <= lower
        ):

            raise ValueError(
                "Every upper_bound value must "
                "be greater than lower_bound."
            )


        # ====================================================
        # RANDOM GENERATOR
        # ====================================================

        rng = np.random.default_rng(
            self.random_state
        )


        # ====================================================
        # INITIAL POPULATION
        # ====================================================

        population = (
            self._initialize_population(

                rng=
                    rng,

                lower_bound=
                    lower,

                upper_bound=
                    upper,

                dimension=
                    dimension,
            )
        )


        # ====================================================
        # INITIAL FITNESS
        # ====================================================

        population_fitness = (
            self._evaluate_population(

                population=
                    population,

                objective_function=
                    objective_function,
            )
        )

        evaluations = int(
            len(
                population
            )
        )


        # ====================================================
        # GLOBAL BEST
        # ====================================================

        best_index = (
            self._best_index(
                population_fitness
            )
        )

        best_position = (
            population[
                best_index
            ]
            .copy()
        )

        best_fitness = float(
            population_fitness[
                best_index
            ]
        )

        convergence_history = []


        # ====================================================
        # DISPLAY INITIAL
        # ====================================================

        if self.verbose:

            print()

            print(
                "=" * 70
            )

            print(
                "GENETIC ALGORITHM OPTIMIZER"
            )

            print(
                "=" * 70
            )

            print(
                f"Population       : "
                f"{self.population_size}"
            )

            print(
                f"Generations      : "
                f"{self.max_generations}"
            )

            print(
                f"Dimension        : "
                f"{dimension}"
            )

            print(
                f"Crossover rate   : "
                f"{self.crossover_rate}"
            )

            print(
                f"Mutation rate    : "
                f"{self.mutation_rate}"
            )

            print(
                f"Mutation scale   : "
                f"{self.mutation_scale}"
            )

            print(
                f"Tournament size  : "
                f"{self.tournament_size}"
            )

            print(
                f"Elites           : "
                f"{self.elitism_count}"
            )

            print(
                f"Objective        : "
                f"{self.objective}"
            )

            print(
                "=" * 70
            )


        # ====================================================
        # GENERATIONS
        # ====================================================

        for generation in range(
            1,
            self.max_generations + 1,
        ):

            # ------------------------------------------------
            # ELITISM
            # ------------------------------------------------

            sorted_indices = (
                self._sorted_indices(
                    population_fitness
                )
            )

            next_population = []

            if (
                self.elitism_count
                >
                0
            ):

                elite_indices = (
                    sorted_indices[
                        :self.elitism_count
                    ]
                )

                for elite_index in (
                    elite_indices
                ):

                    next_population.append(

                        population[
                            elite_index
                        ]
                        .copy()
                    )


            # ------------------------------------------------
            # CREATE CHILDREN
            # ------------------------------------------------

            while (
                len(
                    next_population
                )
                <
                self.population_size
            ):

                parent1 = (
                    self._tournament_select(

                        population=
                            population,

                        fitness=
                            population_fitness,

                        rng=
                            rng,
                    )
                )

                parent2 = (
                    self._tournament_select(

                        population=
                            population,

                        fitness=
                            population_fitness,

                        rng=
                            rng,
                    )
                )

                (
                    child1,
                    child2,

                ) = self._crossover(

                    parent1=
                        parent1,

                    parent2=
                        parent2,

                    rng=
                        rng,
                )

                child1 = self._mutate(

                    individual=
                        child1,

                    lower_bound=
                        lower,

                    upper_bound=
                        upper,

                    rng=
                        rng,
                )

                child2 = self._mutate(

                    individual=
                        child2,

                    lower_bound=
                        lower,

                    upper_bound=
                        upper,

                    rng=
                        rng,
                )

                next_population.append(
                    child1
                )

                if (
                    len(
                        next_population
                    )
                    <
                    self.population_size
                ):

                    next_population.append(
                        child2
                    )


            # ------------------------------------------------
            # REPLACE POPULATION
            # ------------------------------------------------

            population = np.asarray(

                next_population[
                    :self.population_size
                ],

                dtype=np.float64,
            )


            # ------------------------------------------------
            # EVALUATE
            # ------------------------------------------------

            population_fitness = (
                self._evaluate_population(

                    population=
                        population,

                    objective_function=
                        objective_function,
                )
            )

            evaluations += int(
                len(
                    population
                )
            )


            # ------------------------------------------------
            # GENERATION BEST
            # ------------------------------------------------

            generation_best_index = (
                self._best_index(
                    population_fitness
                )
            )

            generation_best_fitness = float(

                population_fitness[
                    generation_best_index
                ]
            )


            # ------------------------------------------------
            # GLOBAL BEST
            # ------------------------------------------------

            if self._better(

                generation_best_fitness,

                best_fitness,

            ):

                best_fitness = (
                    generation_best_fitness
                )

                best_position = (

                    population[
                        generation_best_index
                    ]
                    .copy()
                )


            # ------------------------------------------------
            # CONVERGENCE
            # ------------------------------------------------

            convergence_history.append(
                float(
                    best_fitness
                )
            )


            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            if self.verbose:

                mean_fitness = float(
                    np.mean(
                        population_fitness
                    )
                )

                print(

                    f"[GA Generation "
                    f"{generation:03d}/"
                    f"{self.max_generations:03d}] "

                    f"best="
                    f"{best_fitness:.6f} | "

                    f"generation_best="
                    f"{generation_best_fitness:.6f} | "

                    f"mean="
                    f"{mean_fitness:.6f} | "

                    f"evaluations="
                    f"{evaluations}"
                )


        # ====================================================
        # RESULT
        # ====================================================

        return GAResult(

            best_position=
                best_position,

            best_fitness=
                float(
                    best_fitness
                ),

            convergence_history=
                [
                    float(
                        value
                    )

                    for value
                    in convergence_history
                ],

            population=
                population.copy(),

            population_fitness=
                population_fitness.copy(),

            generations=
                int(
                    self.max_generations
                ),

            evaluations=
                int(
                    evaluations
                ),
        )