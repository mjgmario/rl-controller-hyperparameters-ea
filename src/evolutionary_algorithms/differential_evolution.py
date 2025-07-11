import random
from typing import List, Tuple, Union

import numpy as np

import utils.constants as constants
from benchmarking.optimization_problem import OptimizationProblem
from evolutionary_algorithms.abstract_evolutionary_algorithm import (
    AbstractEvolutionaryAlgorithm,
)


class DifferentialEvolutionOptimizer(AbstractEvolutionaryAlgorithm):
    def __init__(
        self,
        problem: OptimizationProblem,
        population_size: int,
        differential_weights: Union[float, np.ndarray],
        crossover_probability: float,
        base_index_strategy: str,
        differential_number: int,
        recombination_strategy: str,
        differentials_weights_strategy: str,
        max_number_generation: int = 1000,
    ):
        """
        Initializes the Differential Evolution algorithm.

        Parameters:
        bounds (List[Tuple[float, float]]): The bounds for the variables in the optimization problem.
        num_generations (int): The number of generations to run the algorithm.
        fitness_function (Callable[[np.ndarray], float]): The function to optimize.
        population_size (int): The size of the population.
        differential_weights (Union[float, np.ndarray]): The weights for the differential evolution.
            Can be a single float or a NumPy array of floats.
        crossover_probability (float): The recombination rate.
        success_threshold (float): The threshold to determine if the optimization is successful.
        base_index_strategy (str): Strategy for selecting the base index.
        differential_number (int): Number of differentials to use.
        recombination_strategy (str): Strategy for recombination.
        differentials_weights_strategy (str): Strategy for differential weights.
        """
        self.problem = problem
        self.bounds = problem.bounds
        self.fitness_function = problem.evaluate
        self.population_size = population_size
        self.differential_weights = (
            np.full(differential_number, differential_weights)
            if isinstance(differential_weights, float)
            else np.array(differential_weights)
            if len(differential_weights) == differential_number
            else np.full(differential_number, differential_weights[0])
        )
        self.crossover_probability = crossover_probability
        self.best_individuals: List[
            Tuple[int, float, np.ndarray]
        ] = []  # List of (generation, fitness, individual)
        self.base_index_strategy = base_index_strategy
        self.differential_number = differential_number
        self.recombination_strategy = recombination_strategy
        self.differentials_weights_strategy = differentials_weights_strategy
        self.num_function_calls: int = 0
        self.best_individual_value: float = np.inf
        self.best_individual_execution: np.ndarray = None
        self.population, self.fitness_values = self.initialize_population()
        self.generation = 0
        self.max_number_generation = max_number_generation

    def run_generation(self, num_generations):
        """
        :return:
        success or not (boolean value)
        """
        num_generations = 1 if num_generations is None else num_generations
        for _ in range(num_generations):
            best_index = np.argmin(self.fitness_values)
            children, children_fitness = (
                np.empty_like(self.population),
                np.empty(self.population_size),
            )
            for i in range(self.population_size):
                mutation_proposal = self.mutation(
                    self.population, self.fitness_values, best_index, i
                )
                recombination_value = self.recombine(
                    self.population[i], mutation_proposal
                )
                repaired_recombination_value = self.repair_recombination_value(
                    self.population[i], recombination_value
                )
                self.update_children_values(
                    children, children_fitness, i, repaired_recombination_value
                )
            self.num_function_calls += len(self.population)
            self.population, self.fitness_values = self.select_survivors(
                self.population,
                self.fitness_values,
                children,
                children_fitness,
            )
            success = self.update_parameters_end_iteration(
                self.population, self.fitness_values, self.generation
            )
            self.generation += 1
            if success:
                return True
            if self.generation >= self.max_number_generation:
                return False
        return False

    def initialize_population(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initializes the population and computes the fitness for each individual.

        Returns:
        Tuple[np.ndarray, np.ndarray]:
        The initialized population of individuals and the corresponding fitness values.
        """
        lower_bounds, upper_bounds = self.bounds.T
        population = np.random.uniform(
            low=lower_bounds,
            high=upper_bounds,
            size=(self.population_size, len(self.bounds)),
        )
        fitness_values = np.array(
            [self.fitness_function(individual) for individual in population]
        )
        self.num_function_calls += len(population)
        return population, np.where(np.isnan(fitness_values), np.inf, fitness_values)

    def reset_population(self):
        self.best_individuals: List[Tuple[int, float, np.ndarray]] = []
        self.num_function_calls: int = 0
        self.best_individual_value: float = np.inf
        self.best_individual_execution: np.ndarray = None
        self.population, self.fitness_values = self.initialize_population()
        self.generation = 0

    def update_children_values(
        self,
        children: np.ndarray,
        children_fitness: np.ndarray,
        index: int,
        repaired_recombination_value: np.ndarray,
    ) -> None:
        """
        Updates the values of the children array and their corresponding fitness values.

        Args:
            children (np.ndarray): The array representing the current generation of children.
            children_fitness (np.ndarray): The array holding the fitness values of the children.
            index (int): The index at which to update the child's value and fitness.
            repaired_recombination_value (np.ndarray): The value obtained after recombination and repair, to replace the child's current value.

        Returns:
            None: This method updates the children and their fitness in place.
        """
        children[index] = repaired_recombination_value
        children_fitness[index] = self.fitness_function(repaired_recombination_value)
        children_fitness[index] = (
            np.inf if np.isnan(children_fitness[index]) else children_fitness[index]
        )

    def mutation(
        self,
        population: np.ndarray,
        fitness_values: np.ndarray,
        best_index: int,
        target_index: int,
    ) -> np.ndarray:
        base_index, base_value = self.select_base_value(
            population, fitness_values, best_index, target_index
        )
        diff1_array, diff2_array = self.select_differentials(population, base_index)
        differential_value = self.generate_differential_value(diff1_array, diff2_array)
        return base_value + differential_value

    def select_base_value(
        self,
        population: np.ndarray,
        fitness_values: np.ndarray,
        best_index: int,
        target_index: int,
    ) -> Tuple[int, np.ndarray]:
        """
        Selects the base value from the population according to the specified strategy.

        Returns:
        Tuple[int, np.ndarray]: The index and the selected base value.
        """
        if self.base_index_strategy == constants.DE_BASE_INDEX_STRATEGY_RAND:
            base_index = random.choice(
                [i for i in range(len(population)) if i != target_index]
            )
            return base_index, population[base_index]
        elif self.base_index_strategy == constants.DE_BASE_INDEX_STRATEGY_BETTER:
            better_candidates = [
                i
                for i in range(len(population))
                if i != target_index
                and fitness_values[i] <= fitness_values[target_index]
            ]
            base_index = (
                random.choice(better_candidates) if better_candidates else target_index
            )
            return base_index, population[base_index]
        elif (
            self.base_index_strategy == constants.DE_BASE_INDEX_STRATEGY_TARGET_TO_BEST
        ):
            return best_index, population[
                target_index
            ] + constants.DE_BASE_INDEX_STRATEGY_TARGET_TO_BEST_K_VALUE * (
                population[best_index] - population[target_index]
            )
        elif self.base_index_strategy == constants.DE_BASE_INDEX_STRATEGY_BEST:
            return best_index, population[best_index]
        else:
            raise ValueError("Base index vector strategy not available")

    def select_differentials(
        self, population: np.ndarray, base_index: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Selects a list of differential pairs from the population.

        Returns:
        Tuple[np.ndarray, np.ndarray]: Two arrays of differentials.
        """
        indices = list(set(range(self.population_size)) - {base_index})
        selected_indices = random.sample(indices, self.differential_number * 2)
        return (
            population[selected_indices[: self.differential_number]],
            population[selected_indices[self.differential_number :]],
        )

    def generate_differential_value(
        self, diff1_array: np.ndarray, diff2_array: np.ndarray
    ) -> np.ndarray:
        """
        Generates the differential value based on the selected differential weights strategy.

        Returns:
        np.ndarray: The generated differential value.
        """
        diff = diff2_array - diff1_array
        if (
            self.differentials_weights_strategy
            == constants.DE_DIFFERENTIAL_WEIGHTS_STRATEGY_GIVEN
        ):
            return np.dot(self.differential_weights, diff)
        elif (
            self.differentials_weights_strategy
            == constants.DE_DIFFERENTIAL_WEIGHTS_STRATEGY_DITHER
        ):
            random_factor = np.random.uniform(0.5, 1.0)
            return random_factor * np.sum(diff, axis=0)
        elif (
            self.differentials_weights_strategy
            == constants.DE_DIFFERENTIAL_WEIGHTS_STRATEGY_JITTER
        ):
            random_factors = np.random.uniform(0.5, 1.0, diff.shape[0])
            return np.dot(random_factors, diff)
        else:
            raise ValueError("Differential weights strategy not available")

    def recombine(
        self, origin_value: np.ndarray, mutation_proposal: np.ndarray
    ) -> np.ndarray:
        """
        Recombines the origin value and mutation proposal based on the selected recombination strategy.

        Returns:
        np.ndarray: The recombined value.
        """
        if self.recombination_strategy == constants.DE_RECOMBINATION_STRATEGY_BINOMIAL:
            return self.recombination_binomial(origin_value, mutation_proposal)
        elif (
            self.recombination_strategy
            == constants.DE_RECOMBINATION_STRATEGY_EXPONENTIAL
        ):
            return self.recombination_exponential(origin_value, mutation_proposal)
        else:
            raise ValueError("Recombination strategy not available")

    def recombination_binomial(
        self, origin_value: np.ndarray, mutation_proposal: np.ndarray
    ) -> np.ndarray:
        """
        Performs binomial recombination between the origin value and mutation proposal.

        Returns:
        np.ndarray: The recombined value.
        """
        random_index = np.random.randint(len(origin_value))
        random_values = np.random.rand(len(origin_value))
        mask = (random_values < self.crossover_probability) | (
            np.arange(len(origin_value)) == random_index
        )
        return np.where(mask, mutation_proposal, origin_value)

    def recombination_exponential(
        self, origin_value: np.ndarray, mutation_proposal: np.ndarray
    ) -> np.ndarray:
        """
        Performs exponential recombination between the origin value and mutation proposal.

        Returns:
        np.ndarray: The recombined value.
        """
        n = len(origin_value)
        random_index = np.random.randint(n)
        recombined_vector = np.copy(origin_value)
        i = random_index
        while True:
            recombined_vector[i] = mutation_proposal[i]
            i = (i + 1) % n
            if np.random.rand() >= self.crossover_probability and i != random_index:
                break
        return recombined_vector

    def repair_recombination_value(
        self, base_value: np.ndarray, recombination_value: np.ndarray
    ) -> np.ndarray:
        """
        Repairs the recombination value to ensure it is within the bounds.

        Returns:
        np.ndarray: The repaired recombination value.
        """
        lower_bounds, upper_bounds = np.array(self.bounds).T
        below_lower = recombination_value < lower_bounds
        above_upper = recombination_value > upper_bounds
        recombination_value[below_lower] = base_value[below_lower] + (
            lower_bounds[below_lower] - base_value[below_lower]
        ) * np.random.uniform(size=np.sum(below_lower))
        recombination_value[above_upper] = base_value[above_upper] + (
            upper_bounds[above_upper] - base_value[above_upper]
        ) * np.random.uniform(size=np.sum(above_upper))
        return recombination_value

    def select_survivors(
        self,
        population: np.ndarray,
        fitness_values: np.ndarray,
        children: np.ndarray,
        children_fitness: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Selects the survivors for the next generation by comparing the fitness of the current population
        with the fitness of the children.

        Parameters:
        population (np.ndarray): The current population of solutions.
        fitness_values (np.ndarray): The fitness values of the current population.
        children (np.ndarray): The children generated in the current generation.
        children_fitness (np.ndarray): The fitness values of the children.

        Returns:
        np.ndarray, np.ndarray: The new population and their corresponding fitness values.
        """
        better_mask = children_fitness < fitness_values
        population[better_mask] = children[better_mask]
        fitness_values[better_mask] = children_fitness[better_mask]
        return population, fitness_values

    def update_parameters_end_iteration(
        self,
        population: np.ndarray,
        fitness_values: np.ndarray,
        generation: int,
    ) -> bool:
        """
        Updates the parameters at the end of each iteration of an optimization algorithm.

        This method identifies the best individual in the current population based on fitness values,
        updates the best individual value if a better one is found, appends the best individual data
        to the history, and checks if the success threshold has been met.

        Args:
            population (np.ndarray): An array representing the current population of individuals.
            fitness_values (np.ndarray): An array of fitness values corresponding to each individual in the population.
            generation (int): The current generation or iteration number.

        Returns:
            bool: True if the minimum child value is less than or equal to the success threshold, False otherwise.
        """
        best_index = np.argmin(fitness_values)
        min_child_value = fitness_values[best_index]
        best_update = min_child_value <= self.best_individual_value
        self.best_individual_value = (
            min_child_value if best_update else self.best_individual_value
        )
        self.best_individual_execution = (
            population[best_index].copy()
            if best_update
            else self.best_individual_execution
        )
        self.best_individuals.append(
            (generation, min_child_value, population[best_index].copy())
        )
        return self.problem.is_success(min_child_value)
