"""
Hybrid EVO-GA Optimizer
=======================

Hybrid optimization strategy:

    Stage 1:
        Energy Valley Optimizer (EVO)
        performs global exploration.

    Stage 2:
        Genetic Algorithm (GA)
        refines the strongest EVO solutions.

The optimizer is generic and knows nothing about:

    - IDS
    - Federated Learning
    - datasets
    - feature selection
    - neural networks

It only optimizes:

    objective_function(position)

Typical feature-selection usage:

    from optimization.hybrid_evo_ga import (
        HybridEVOGAOptimizer
    )

    optimizer = HybridEVOGAOptimizer(
        evo_population_size=10,
        evo_iterations=10,
        ga_population_size=10,
        ga_generations=10,
        random_state=42,
        verbose=True,
    )

    result = optimizer.optimize(
        objective_function=selector.fitness,
        dimension=70,
        lower_bound=-6.0,
        upper_bound=6.0,
    )

Hybrid philosophy
-----------------

EVO:
    Broad/global exploration.

GA:
    Local/evolutionary refinement around strong EVO solutions.

This is preferable to simply intersecting EVO and GA feature
masks because the two methods may identify complementary
feature combinations.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import (
    Callable,
    List,
    Optional,
    Sequence,
    Union,
)

import numpy as np

from optimization.evo import (
    EnergyValleyOptimizer,
)


# ============================================================
# TYPES
# ============================================================

ArrayLike = Union[
    float,
    int,
    Sequence[float],
    np.ndarray,
]


# ============================================================
# RESULT
# ============================================================

@dataclass(frozen=True)
class HybridEVOGAResult:

    # --------------------------------------------------------
    # Final hybrid result
    # --------------------------------------------------------

    best_position: np.ndarray

    best_fitness: float

    convergence_history: List[float]

    # --------------------------------------------------------
    # Final GA population
    # --------------------------------------------------------

    population: np.ndarray

    population_fitness: np.ndarray

    # --------------------------------------------------------
    # EVO stage
    # --------------------------------------------------------

    evo_best_position: np.ndarray

    evo_best_fitness: float

    evo_convergence_history: List[float]

    evo_iterations: int

    evo_evaluations: int

    # --------------------------------------------------------
    # GA stage
    # --------------------------------------------------------

    ga_best_position: np.ndarray

    ga_best_fitness: float

    ga_convergence_history: List[float]

    ga_generations: int

    ga_evaluations: int

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    total_evaluations: int


# ============================================================
# HYBRID EVO-GA
# ============================================================

class HybridEVOGAOptimizer:

    """
    Generic hybrid EVO-GA optimizer.

    Stage 1:
        Energy Valley Optimizer.

    Stage 2:
        GA initialized using strong EVO solutions.

    Default objective:
        minimization
    """

    def __init__(
        self,

        # ----------------------------------------------------
        # EVO
        # ----------------------------------------------------

        evo_population_size: int = 10,

        evo_iterations: int = 10,

        # ----------------------------------------------------
        # GA
        # ----------------------------------------------------

        ga_population_size: int = 10,

        ga_generations: int = 10,

        crossover_rate: float = 0.90,

        mutation_rate: float = 0.10,

        mutation_scale: float = 0.15,

        tournament_size: int = 3,

        elitism_count: int = 1,

        # ----------------------------------------------------
        # EVO -> GA transfer
        # ----------------------------------------------------

        evo_elite_fraction: float = 0.50,

        best_solution_copies: int = 1,

        seeded_mutation_scale: float = 0.05,

        random_injection_fraction: float = 0.10,

        # ----------------------------------------------------
        # General
        # ----------------------------------------------------

        objective: str = "minimize",

        random_state: Optional[int] = 42,

        verbose: bool = True,
    ):

        # ====================================================
        # VALIDATION
        # ====================================================

        if evo_population_size < 2:

            raise ValueError(
                "evo_population_size must be >= 2."
            )

        if evo_iterations < 1:

            raise ValueError(
                "evo_iterations must be >= 1."
            )

        if ga_population_size < 2:

            raise ValueError(
                "ga_population_size must be >= 2."
            )

        if ga_generations < 1:

            raise ValueError(
                "ga_generations must be >= 1."
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

        if elitism_count >= ga_population_size:

            raise ValueError(
                "elitism_count must be smaller "
                "than ga_population_size."
            )

        if not 0.0 <= evo_elite_fraction <= 1.0:

            raise ValueError(
                "evo_elite_fraction must be "
                "between 0 and 1."
            )

        if best_solution_copies < 0:

            raise ValueError(
                "best_solution_copies must be >= 0."
            )

        if seeded_mutation_scale < 0.0:

            raise ValueError(
                "seeded_mutation_scale must be >= 0."
            )

        if not 0.0 <= random_injection_fraction <= 1.0:

            raise ValueError(
                "random_injection_fraction must be "
                "between 0 and 1."
            )

        if objective not in {
            "minimize",
            "maximize",
        }:

            raise ValueError(
                "objective must be "
                "'minimize' or 'maximize'."
            )


        # ====================================================
        # STORE CONFIG
        # ====================================================

        self.evo_population_size = int(
            evo_population_size
        )

        self.evo_iterations = int(
            evo_iterations
        )

        self.ga_population_size = int(
            ga_population_size
        )

        self.ga_generations = int(
            ga_generations
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

        self.evo_elite_fraction = float(
            evo_elite_fraction
        )

        self.best_solution_copies = int(
            best_solution_copies
        )

        self.seeded_mutation_scale = float(
            seeded_mutation_scale
        )

        self.random_injection_fraction = float(
            random_injection_fraction
        )

        self.objective = objective

        self.random_state = (
            None
            if random_state is None
            else int(random_state)
        )

        self.verbose = bool(
            verbose
        )


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

            result = np.full(
                dimension,
                float(value),
                dtype=np.float64,
            )

        else:

            result = np.asarray(
                value,
                dtype=np.float64,
            ).reshape(-1)

            if len(
                result
            ) != dimension:

                raise ValueError(
                    f"{name} must contain exactly "
                    f"{dimension} values."
                )

        if not np.all(
            np.isfinite(
                result
            )
        ):

            raise ValueError(
                f"{name} contains NaN/inf."
            )

        return result


    # ========================================================
    # COMPARISON
    # ========================================================

    def _better(
        self,
        candidate: float,
        reference: float,
    ) -> bool:

        if self.objective == "minimize":

            return candidate < reference

        return candidate > reference


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

            value = float(
                objective_function(
                    individual
                )
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
    # GA TOURNAMENT
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

        candidate_indices = rng.choice(

            len(
                population
            ),

            size=
                tournament_size,

            replace=False,
        )

        candidate_fitness = fitness[
            candidate_indices
        ]

        winner_local_index = (
            self._best_index(
                candidate_fitness
            )
        )

        winner_global_index = int(

            candidate_indices[
                winner_local_index
            ]
        )

        return population[
            winner_global_index
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

        # Per-gene arithmetic/blend crossover.

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
    # GA MUTATION
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

            sigma = (

                self.mutation_scale

                *
                search_range
            )

            noise = rng.normal(

                loc=
                    0.0,

                scale=
                    sigma,

                size=
                    child.shape,
            )

            child[
                mutation_mask
            ] += noise[
                mutation_mask
            ]

        return np.clip(

            child,

            lower_bound,

            upper_bound,
        )


    # ========================================================
    # SEEDED MUTATION
    #
    # Used only when constructing the initial GA population
    # from EVO solutions.
    # ========================================================

    def _seeded_mutation(
        self,
        individual,
        lower_bound,
        upper_bound,
        rng,
    ):

        search_range = (

            upper_bound

            -
            lower_bound
        )

        sigma = (

            self.seeded_mutation_scale

            *
            search_range
        )

        noise = rng.normal(

            loc=
                0.0,

            scale=
                sigma,

            size=
                individual.shape,
        )

        child = (

            individual
            +
            noise
        )

        return np.clip(

            child,

            lower_bound,

            upper_bound,
        )


    # ========================================================
    # BUILD GA INITIAL POPULATION FROM EVO
    # ========================================================

    def _build_ga_population(
        self,
        evo_population,
        evo_fitness,
        evo_best_position,
        lower_bound,
        upper_bound,
        rng,
    ):

        """
        Construct GA's starting population using:

            1. Exact EVO global best.
            2. Strong EVO elites.
            3. Mutated copies around strong EVO candidates.
            4. Small random injection for diversity.
        """

        evo_population = np.asarray(
            evo_population,
            dtype=np.float64,
        )

        evo_fitness = np.asarray(
            evo_fitness,
            dtype=np.float64,
        )

        sorted_indices = (
            self._sorted_indices(
                evo_fitness
            )
        )


        # ----------------------------------------------------
        # RANDOM INJECTION COUNT
        # ----------------------------------------------------

        random_count = int(
            round(

                self.ga_population_size

                *
                self.random_injection_fraction
            )
        )

        random_count = min(
            random_count,
            self.ga_population_size,
        )


        # ----------------------------------------------------
        # AVAILABLE NON-RANDOM POSITIONS
        # ----------------------------------------------------

        seeded_target = (

            self.ga_population_size

            -
            random_count
        )


        # ----------------------------------------------------
        # EVO ELITE COUNT
        # ----------------------------------------------------

        elite_count = int(
            round(

                seeded_target

                *
                self.evo_elite_fraction
            )
        )

        elite_count = max(
            1,
            elite_count,
        )

        elite_count = min(
            elite_count,
            len(
                evo_population
            ),
            seeded_target,
        )


        # ----------------------------------------------------
        # INITIAL POPULATION
        # ----------------------------------------------------

        initial_population = []


        # ----------------------------------------------------
        # EXACT EVO BEST COPIES
        # ----------------------------------------------------

        copies = min(

            self.best_solution_copies,

            seeded_target,
        )

        for _ in range(
            copies
        ):

            initial_population.append(

                np.asarray(
                    evo_best_position,
                    dtype=np.float64,
                ).copy()
            )


        # ----------------------------------------------------
        # STRONG EVO ELITES
        # ----------------------------------------------------

        elite_indices = sorted_indices[
            :elite_count
        ]

        for elite_index in (
            elite_indices
        ):

            if (
                len(
                    initial_population
                )
                >=
                seeded_target
            ):

                break

            candidate = (

                evo_population[
                    elite_index
                ]
                .copy()
            )

            # Avoid unnecessary exact duplicate of best
            # when possible.

            if (
                initial_population
                and
                np.array_equal(
                    candidate,
                    initial_population[0],
                )
            ):

                continue

            initial_population.append(
                candidate
            )


        # ----------------------------------------------------
        # MUTATED EVO CANDIDATES
        # ----------------------------------------------------

        source_indices = sorted_indices[
            :max(
                1,
                elite_count,
            )
        ]

        source_pointer = 0

        while (
            len(
                initial_population
            )
            <
            seeded_target
        ):

            source_index = (

                source_indices[
                    source_pointer
                    %
                    len(
                        source_indices
                    )
                ]
            )

            source = (

                evo_population[
                    source_index
                ]
                .copy()
            )

            mutated = (
                self._seeded_mutation(

                    individual=
                        source,

                    lower_bound=
                        lower_bound,

                    upper_bound=
                        upper_bound,

                    rng=
                        rng,
                )
            )

            initial_population.append(
                mutated
            )

            source_pointer += 1


        # ----------------------------------------------------
        # RANDOM DIVERSITY INJECTION
        # ----------------------------------------------------

        for _ in range(
            random_count
        ):

            random_individual = rng.uniform(

                low=
                    lower_bound,

                high=
                    upper_bound,
            )

            initial_population.append(
                random_individual
            )


        # ----------------------------------------------------
        # SIZE SAFETY
        # ----------------------------------------------------

        initial_population = np.asarray(

            initial_population[
                :self.ga_population_size
            ],

            dtype=np.float64,
        )

        if (
            len(
                initial_population
            )
            !=
            self.ga_population_size
        ):

            raise RuntimeError(
                "Failed to build requested GA "
                "population size."
            )

        return initial_population


    # ========================================================
    # GA REFINEMENT
    # ========================================================

    def _run_ga_refinement(
        self,
        objective_function,
        initial_population,
        lower_bound,
        upper_bound,
        rng,
    ):

        population = np.asarray(
            initial_population,
            dtype=np.float64,
        ).copy()


        # ----------------------------------------------------
        # INITIAL GA FITNESS
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # INITIAL BEST
        # ----------------------------------------------------

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
        # GENERATIONS
        # ====================================================

        for generation in range(
            1,
            self.ga_generations + 1,
        ):

            sorted_indices = (
                self._sorted_indices(
                    population_fitness
                )
            )

            next_population = []


            # ------------------------------------------------
            # ELITISM
            # ------------------------------------------------

            if (
                self.elitism_count
                >
                0
            ):

                elite_indices = sorted_indices[
                    :self.elitism_count
                ]

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
            # CROSSOVER + MUTATION
            # ------------------------------------------------

            while (
                len(
                    next_population
                )
                <
                self.ga_population_size
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

                child1 = (
                    self._mutate(

                        individual=
                            child1,

                        lower_bound=
                            lower_bound,

                        upper_bound=
                            upper_bound,

                        rng=
                            rng,
                    )
                )

                child2 = (
                    self._mutate(

                        individual=
                            child2,

                        lower_bound=
                            lower_bound,

                        upper_bound=
                            upper_bound,

                        rng=
                            rng,
                    )
                )

                next_population.append(
                    child1
                )

                if (
                    len(
                        next_population
                    )
                    <
                    self.ga_population_size
                ):

                    next_population.append(
                        child2
                    )


            # ------------------------------------------------
            # NEW POPULATION
            # ------------------------------------------------

            population = np.asarray(

                next_population[
                    :self.ga_population_size
                ],

                dtype=np.float64,
            )


            # ------------------------------------------------
            # FITNESS
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
            # GLOBAL GA BEST
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

                print(

                    f"[Hybrid GA Generation "
                    f"{generation:03d}/"
                    f"{self.ga_generations:03d}] "

                    f"best="
                    f"{best_fitness:.6f} | "

                    f"generation_best="
                    f"{generation_best_fitness:.6f} | "

                    f"mean="
                    f"{np.mean(population_fitness):.6f} | "

                    f"evaluations="
                    f"{evaluations}"
                )


        return (
            best_position,
            best_fitness,
            convergence_history,
            population,
            population_fitness,
            evaluations,
        )


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
    ) -> HybridEVOGAResult:

        # ====================================================
        # VALIDATE DIMENSION / BOUNDS
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
                "Every upper_bound value must be "
                "greater than lower_bound."
            )


        # ====================================================
        # RNG
        # ====================================================

        rng = np.random.default_rng(
            self.random_state
        )


        # ====================================================
        # DISPLAY
        # ====================================================

        if self.verbose:

            print()
            print("=" * 80)

            print(
                "HYBRID EVO-GA OPTIMIZER"
            )

            print("=" * 80)

            print(
                f"Dimension              : "
                f"{dimension}"
            )

            print(
                f"Objective              : "
                f"{self.objective}"
            )

            print()

            print(
                "EVO STAGE"
            )

            print(
                f"  Population           : "
                f"{self.evo_population_size}"
            )

            print(
                f"  Iterations           : "
                f"{self.evo_iterations}"
            )

            print()

            print(
                "GA REFINEMENT STAGE"
            )

            print(
                f"  Population           : "
                f"{self.ga_population_size}"
            )

            print(
                f"  Generations          : "
                f"{self.ga_generations}"
            )

            print(
                f"  Crossover rate       : "
                f"{self.crossover_rate}"
            )

            print(
                f"  Mutation rate        : "
                f"{self.mutation_rate}"
            )

            print(
                f"  EVO elite fraction   : "
                f"{self.evo_elite_fraction}"
            )

            print(
                f"  Random injection     : "
                f"{self.random_injection_fraction}"
            )

            print("=" * 80)


        # ====================================================
        # STAGE 1 — EVO
        # ====================================================

        if self.verbose:

            print()
            print("-" * 80)

            print(
                "STAGE 1 — EVO GLOBAL EXPLORATION"
            )

            print("-" * 80)


        evo_optimizer = (
            EnergyValleyOptimizer(

                population_size=
                    self.evo_population_size,

                max_iterations=
                    self.evo_iterations,

                random_state=
                    self.random_state,

                verbose=
                    self.verbose,
            )
        )


        # ----------------------------------------------------
        # Important:
        # Current project EVO is a minimization optimizer.
        # Feature-selection fitness is also minimization.
        # ----------------------------------------------------

        if self.objective != "minimize":

            raise NotImplementedError(
                "Current project EnergyValleyOptimizer "
                "integration is configured for minimization. "
                "Use objective='minimize' for the hybrid."
            )


        evo_result = (
            evo_optimizer.optimize(

                objective_function=
                    objective_function,

                dimension=
                    dimension,

                lower_bound=
                    lower,

                upper_bound=
                    upper,
            )
        )


        # ====================================================
        # STAGE 2 — BUILD GA POPULATION
        # ====================================================

        if self.verbose:

            print()
            print("-" * 80)

            print(
                "STAGE 2 — INITIALIZE GA FROM EVO"
            )

            print("-" * 80)

            print(
                f"EVO best fitness       : "
                f"{evo_result.best_fitness:.6f}"
            )


        ga_initial_population = (
            self._build_ga_population(

                evo_population=
                    evo_result.population,

                evo_fitness=
                    evo_result.population_fitness,

                evo_best_position=
                    evo_result.best_position,

                lower_bound=
                    lower,

                upper_bound=
                    upper,

                rng=
                    rng,
            )
        )


        # ====================================================
        # STAGE 3 — GA REFINEMENT
        # ====================================================

        if self.verbose:

            print()
            print("-" * 80)

            print(
                "STAGE 3 — GA REFINEMENT"
            )

            print("-" * 80)


        (
            ga_best_position,
            ga_best_fitness,
            ga_convergence,
            final_population,
            final_population_fitness,
            ga_evaluations,

        ) = self._run_ga_refinement(

            objective_function=
                objective_function,

            initial_population=
                ga_initial_population,

            lower_bound=
                lower,

            upper_bound=
                upper,

            rng=
                rng,
        )


        # ====================================================
        # FINAL HYBRID BEST
        #
        # Usually GA should preserve/improve the EVO best
        # because EVO best is seeded into GA and elitism is
        # enabled. Still compare explicitly.
        # ====================================================

        if self._better(

            ga_best_fitness,

            float(
                evo_result.best_fitness
            ),

        ):

            final_best_position = (
                ga_best_position.copy()
            )

            final_best_fitness = float(
                ga_best_fitness
            )

        else:

            final_best_position = (

                np.asarray(
                    evo_result.best_position,
                    dtype=np.float64,
                )
                .copy()
            )

            final_best_fitness = float(
                evo_result.best_fitness
            )


        # ====================================================
        # COMBINED CONVERGENCE
        # ====================================================

        combined_convergence = [

            float(value)

            for value
            in evo_result.convergence_history
        ]

        running_best = float(
            evo_result.best_fitness
        )

        for value in (
            ga_convergence
        ):

            value = float(
                value
            )

            if self._better(
                value,
                running_best,
            ):

                running_best = value

            combined_convergence.append(
                float(
                    running_best
                )
            )


        # ====================================================
        # TOTAL EVALUATIONS
        # ====================================================

        total_evaluations = int(

            evo_result.evaluations

            +
            ga_evaluations
        )


        # ====================================================
        # FINAL DISPLAY
        # ====================================================

        if self.verbose:

            print()
            print("=" * 80)

            print(
                "HYBRID EVO-GA COMPLETE"
            )

            print("=" * 80)

            print(
                f"EVO best fitness       : "
                f"{evo_result.best_fitness:.6f}"
            )

            print(
                f"GA best fitness        : "
                f"{ga_best_fitness:.6f}"
            )

            print(
                f"Hybrid best fitness    : "
                f"{final_best_fitness:.6f}"
            )

            improvement = (

                float(
                    evo_result.best_fitness
                )

                -
                float(
                    final_best_fitness
                )
            )

            print(
                f"GA improvement over EVO: "
                f"{improvement:+.6f}"
            )

            print(
                f"EVO evaluations        : "
                f"{evo_result.evaluations}"
            )

            print(
                f"GA evaluations         : "
                f"{ga_evaluations}"
            )

            print(
                f"Total evaluations      : "
                f"{total_evaluations}"
            )

            print("=" * 80)


        # ====================================================
        # RESULT
        # ====================================================

        return HybridEVOGAResult(

            best_position=
                final_best_position,

            best_fitness=
                final_best_fitness,

            convergence_history=
                combined_convergence,

            population=
                final_population.copy(),

            population_fitness=
                final_population_fitness.copy(),

            evo_best_position=
                np.asarray(
                    evo_result.best_position,
                    dtype=np.float64,
                ).copy(),

            evo_best_fitness=
                float(
                    evo_result.best_fitness
                ),

            evo_convergence_history=
                [
                    float(value)

                    for value
                    in evo_result.convergence_history
                ],

            evo_iterations=
                int(
                    evo_result.iterations
                ),

            evo_evaluations=
                int(
                    evo_result.evaluations
                ),

            ga_best_position=
                ga_best_position.copy(),

            ga_best_fitness=
                float(
                    ga_best_fitness
                ),

            ga_convergence_history=
                [
                    float(value)

                    for value
                    in ga_convergence
                ],

            ga_generations=
                int(
                    self.ga_generations
                ),

            ga_evaluations=
                int(
                    ga_evaluations
                ),

            total_evaluations=
                total_evaluations,
        )