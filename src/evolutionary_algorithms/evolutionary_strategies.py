from typing import Tuple, Union

import numpy as np

from benchmarking.optimization_problem import OptimizationProblem
from utils import constants


class EvolutionaryStrategies:
    def __init__(
        self,
        problem: OptimizationProblem,
        phro: int,
        epsilon0: float,
        tau: float,
        tau_prime: float,
        recombination_individuals_strategy: str,
        recombination_strategy_strategy: str,
        mutation_steps: int,
        survivor_strategy: str,
        mu: int = 30,
        lamda: int = 200,
        max_number_generation: int = 1000,
        use_rechenberg: bool = True,
        c_up: float = 1.5,
        c_down: float = 1.5,
        target_success_rate: float = 0.2,
        window_r: int = 10,
        external_sigma_control: bool = False,
    ) -> None:
        """
        Initialize the Evolutionary Strategies optimizer.

        Sets up the optimization problem, hyperparameters, and initial population.

        Args:
            problem (OptimizationProblem):
                The problem to optimize; must provide `bounds`, `evaluate()`, and `is_success()`.
            phro (int):
                Number of parents selected for recombination.
            epsilon0 (float):
                Minimum allowed mutation step size.
            tau (float):
                Global learning rate for strategy parameter mutation.
            tau_prime (float):
                Individual learning rate for strategy parameter mutation.
            recombination_individuals_strategy (str):
                Strategy for recombining parent individuals.
            recombination_strategy_strategy (str):
                Strategy for recombining strategy parameters.
            mutation_steps (int):
                Number of uncorrelated mutation step sizes.
            survivor_strategy (str):
                Survivor selection method ('mu,lambda' or 'mu+lambda').
            mu (int, optional):
                Parent population size. Defaults to 30.
            lamda (int, optional):
                Number of offspring per generation. Defaults to 200.
            max_number_generation (int, optional):
                Maximum number of generations to run. Defaults to 1000.
            use_rechenberg (bool, optional):
                Enable Rechenberg sigma adaptation when single-step mutation is used.
            c_up (float, optional):
                Factor to increase sigma under Rechenberg rule. Defaults to 1.5.
            c_down (float, optional):
                Factor to decrease sigma under Rechenberg rule. Defaults to 1.5.
            target_success_rate (float, optional):
                Desired success fraction in each Rechenberg window. Defaults to 0.2.
            window_r (int, optional):
                Number of generations per Rechenberg adaptation window. Defaults to 10.
            external_sigma_control (bool, optional):
                If True, skip internal sigma adaptation. Defaults to False.

        Raises:
            ValueError:
                If any of the provided hyperparameters are out of valid ranges.
        """
        self.problem = problem
        self.bounds = problem.bounds
        self.fitness_function = problem.evaluate
        self.phro = phro
        self.epsilon0 = float(epsilon0)
        self.tau = float(tau)
        self.tau_prime = float(tau_prime)
        self.recombination_individuals_strategy = recombination_individuals_strategy
        self.recombination_strategy_strategy = recombination_strategy_strategy
        self.mutation_steps = mutation_steps
        self.survivor_strategy = survivor_strategy
        self.mu = mu
        self.lamda = lamda

        self.best_individuals = []
        self.best_individual_value = np.inf
        self.best_individual_execution = None
        self.num_function_calls = 0
        (
            self.population,
            self.strategy_parameters,
            self.fitness_values,
        ) = self.initialize_population()
        self.num_function_calls = len(self.population)
        self.generation = 0
        self.max_number_generation = max_number_generation

        self.use_rechenberg = use_rechenberg and mutation_steps == 1
        self.c_up = float(c_up)
        self.c_down = float(c_down)
        self.target_success_rate = float(target_success_rate)
        self.window_r = int(window_r)
        self.success_counter = 0
        self.external_sigma_control = external_sigma_control

    def run_generation(self, num_generations):
        """
        :return:
        success or not (boolean value)
        """
        num_generations = 1 if num_generations is None else num_generations
        for _ in range(num_generations):
            children = np.empty((self.lamda, self.population.shape[1]))
            children_strategy_params = (
                np.empty((self.lamda, 1))
                if self.mutation_steps == 1
                else np.empty((self.lamda, len(self.bounds)))
            )
            parent_fit = self.fitness_values.copy()
            for i in range(self.lamda):
                random_indexes = np.random.choice(
                    self.mu, size=self.phro, replace=False
                )
                parents, strategy_parents = (
                    self.population[random_indexes],
                    self.strategy_parameters[random_indexes],
                )
                strategy_parents = self.recombine(
                    strategy_parents, self.recombination_individuals_strategy
                )
                recombined_individuals = self.recombine(
                    parents, strategy=self.recombination_strategy_strategy
                )
                children_strategy_params[i] = self.mutate_strategy_parameters(
                    strategy_parents
                )
                children[i] = self.mutation_uncorrelated_individual(
                    recombined_individuals, children_strategy_params[i]
                )
            (
                self.population,
                self.strategy_parameters,
                self.fitness_values,
                min_child_value,
                children_fitness_values,
            ) = self.select_survivors(
                self.population,
                self.fitness_values,
                children,
                self.strategy_parameters,
                children_strategy_params,
                self.survivor_strategy,
            )
            m = min(len(parent_fit), len(children_fitness_values))
            success_flags = children_fitness_values[:m] < parent_fit[:m]
            if self.use_rechenberg:
                self._adapt_sigma_rechenberg(success_flags)
            self.num_function_calls += len(self.population)
            success = self.update_parameters_end_iteration(
                self.generation, self.population, min_child_value
            )
            if success:
                return True
            self.generation += 1
        return False

    def initialize_population(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Initializes the population, strategy parameters, and fitness values for the evolutionary algorithm.

        :return: A tuple containing:
            - individuals: A NumPy array representing the initial population, with each row corresponding to an individual.
            - strategy_params: A NumPy array representing the strategy parameters for each individual.
            - fitness_values: A NumPy array containing the fitness values of the initial population.
        """
        lower_bounds, upper_bounds = np.array(self.bounds).T
        individuals = np.random.uniform(
            low=lower_bounds,
            high=upper_bounds,
            size=(self.mu, len(self.bounds)),
        )
        strategy_params = np.ones(
            (self.mu, 1) if self.mutation_steps == 1 else (self.mu, len(self.bounds))
        )
        fitness_values = np.array(
            [self.safe_fitness(individual) for individual in individuals]
        )
        return individuals, strategy_params, fitness_values

    def _adapt_sigma_rechenberg(self, success_flags: np.ndarray) -> None:
        """
        Adapt the mutation step-size (sigma) according to Rechenberg's 1/5 success rule.

        Over each window of `window_r` generations, computes the ratio of successful
        offspring (those that improved over their parents). If the achieved success rate
        exceeds the `target_success_rate`, sigma is increased by dividing by `c_up`;
        otherwise, it is decreased by multiplying by `c_down`. Sigma values are floored
        at `epsilon0`. If `external_sigma_control` is True, no adaptation is performed.

        Args:
            success_flags (np.ndarray): Boolean array indicating which offspring
                improved on their parent fitness.
        """
        if self.external_sigma_control:
            return
        self.success_counter += int(np.count_nonzero(success_flags))
        if (self.generation + 1) % self.window_r == 0:
            achieved_rate = self.success_counter / (len(success_flags) * self.window_r)
            if achieved_rate > self.target_success_rate:
                self.strategy_parameters /= self.c_up
            else:
                self.strategy_parameters *= self.c_down

            self.strategy_parameters = np.maximum(
                self.strategy_parameters, self.epsilon0
            )
            self.success_counter = 0

    def reset_population(self):
        self.best_individuals = []
        self.best_individual_value = np.inf
        self.best_individual_execution = None
        self.num_function_calls = 0
        (
            self.population,
            self.strategy_parameters,
            self.fitness_values,
        ) = self.initialize_population()
        self.num_function_calls = len(self.population)
        self.generation = 0

    def recombine(self, parents: np.ndarray, strategy: str) -> np.ndarray:
        """
        Applies recombination to the parents using the specified strategy.

        :param parents: List of parents.
        :param strategy: Recombination strategy ('discrete', 'averaged_intermediate', 'generalized_intermediate', 'averaged').
        :return: Offspring generated by recombination.
        """
        if strategy == constants.ES_RECOMBINATION_STRATEGY_DISCRETE:
            return self.recombination_discrete(parents)
        elif strategy == constants.ES_RECOMBINATION_STRATEGY_AVERAGE_INTERMEDIATE:
            return self.recombination_averaged_intermediate(parents)
        elif strategy == constants.ES_RECOMBINATION_STRATEGY_GENERALIZED_INTERMEDIATE:
            return self.recombination_generalized_intermediate(parents)
        elif strategy == constants.ES_RECOMBINATION_STRATEGY_INTERMEDIATE_AVERAGED:
            return self.recombination_averaged(parents)
        else:
            raise ValueError("Invalid recombination strategy specified.")

    def recombination_discrete(self, parents: np.ndarray) -> np.ndarray:
        """
        Performs discrete recombination among the parents.

        :param parents: Array of parents.
        :return: Offspring generated by discrete recombination.
        """
        if parents.ndim > 1:
            indices = np.random.randint(parents.shape[0], size=parents.shape[1])
            return parents[indices, np.arange(parents.shape[1])]
        else:
            return np.random.choice(parents)

    def recombination_averaged_intermediate(self, parents: np.ndarray) -> np.ndarray:
        """
        Performs averaged intermediate recombination among the parents.

        :param parents: Array of parents.
        :return: Offspring generated by averaged intermediate recombination.
        """
        ri = 0.5
        num_parents, num_components = parents.shape
        fixed_parent_index = np.random.randint(num_parents)
        fixed_parent = parents[fixed_parent_index]
        random_parent_indices = np.random.choice(
            np.delete(np.arange(num_parents), fixed_parent_index),
            num_components,
        )
        random_parents = parents[random_parent_indices, np.arange(num_components)]
        return ri * fixed_parent + (1 - ri) * random_parents

    def recombination_generalized_intermediate(
        self, parents: np.ndarray
    ) -> Union[float, np.ndarray]:
        """
        Performs generalized intermediate recombination among the parents.

        :param parents: Array of parents.
        :param alpha: Mixing parameter between parents.
        :return: Offspring generated by generalized intermediate recombination.
        """
        num_parents, num_components = parents.shape
        fixed_parent_index = np.random.randint(num_parents)
        fixed_parent = parents[fixed_parent_index]
        random_parent_indices = np.random.choice(
            np.delete(np.arange(num_parents), fixed_parent_index),
            num_components,
        )
        random_parents = parents[random_parent_indices, np.arange(num_components)]
        ri = np.random.rand(num_components)
        return ri * fixed_parent + (1 - ri) * random_parents

    def recombination_averaged(self, parents: np.ndarray) -> np.ndarray:
        """
        Performs averaged recombination among the parents.

        :param parents: List of parents.
        :return: Offspring generated by averaged recombination.
        """
        return np.mean(parents, axis=0)

    def mutate_strategy_parameters(self, strategy_parents: np.ndarray) -> np.ndarray:
        """
        Mutates the strategy parameters of the parents using uncorrelated mutation.

        Depending on the number of mutation steps (`self.mutation_steps`), it applies either
        the uncorrelated mutation with a single step size or with multiple step sizes.

        :param strategy_parents: The strategy parameters of the parents to be mutated as a numpy array.
        :return: The mutated strategy parameters as a numpy array.
        """
        if self.external_sigma_control:
            return strategy_parents.copy()
        if self.use_rechenberg and self.mutation_steps == 1:
            return strategy_parents.copy()
        if self.mutation_steps > 1:
            return self.mutation_uncorrelated_step_size(
                strategy_parents, self.tau, self.tau_prime, self.epsilon0
            )
        else:
            return self.mutation_uncorrelated_one_step_size(
                strategy_parents, self.tau, self.epsilon0
            )

    def mutation_uncorrelated_step_size(
        self,
        step_size: Union[float, np.ndarray],
        tau: float,
        tau_prime: float,
        epsilon: float,
    ) -> np.ndarray:
        """
        Performs uncorrelated mutation on the strategy parameters.

        :param step_size: Current step size.
        :param tau: Mutation parameter (tau).
        :param tau_prime: Mutation parameter (tau').
        :param epsilon: Minimum allowed value for strategy parameters.
        :return: Mutated step size as a numpy array.
        """
        step_size = np.array(step_size) if isinstance(step_size, float) else step_size
        random_value_tau = np.random.normal(0, 1)
        random_values_tau_prime = np.random.normal(0, 1, size=len(step_size))
        return np.array(
            np.maximum(
                step_size
                * np.exp(tau * random_value_tau + tau_prime * random_values_tau_prime),
                epsilon,
            )
        )

    def mutation_uncorrelated_one_step_size(
        self, step_size: Union[float, np.ndarray], tau: float, epsilon: float
    ) -> np.ndarray:
        """
        Performs uncorrelated mutation with a single step size.

        :param step_size: Current step size.
        :param tau: Mutation parameter (tau).
        :param epsilon: Minimum allowed value for strategy parameters.
        :return: Mutated step size as a numpy array.
        """
        step_size = np.array(step_size) if isinstance(step_size, float) else step_size
        noise = (
            np.random.normal(0, 1, size=len(step_size))
            if self.mutation_steps > 1
            else np.array([np.random.normal(0, 1)])
        )
        return np.array(np.maximum(step_size * np.exp(tau * noise), epsilon))

    def mutation_uncorrelated_individual(
        self, individual: np.ndarray, strategy_params: np.ndarray
    ) -> np.ndarray:
        """
        Performs uncorrelated mutation on the individual.

        :param individual: Individual to mutate as a numpy array.
        :param strategy_params: Strategy parameters for mutation as a numpy array.
        :return: Mutated individual as a numpy array.
        """
        n = len(individual)
        norm = np.random.normal(0, 1, n)
        return individual + strategy_params * norm

    def safe_fitness(self, individual):
        try:
            value = self.fitness_function(individual)

            if (
                value is None
                or not isinstance(value, (float, int))
                or not np.isfinite(value)
            ):
                print(
                    f"[WARNING] Invalid fitness value for individual {individual}: {value}"
                )
                return np.inf

            return value

        except Exception as e:
            print(f"[WARNING] Fitness function error for individual {individual}: {e}")
            return np.inf

    def select_survivors(
        self,
        population: np.ndarray,
        fitness_values: np.ndarray,
        children: np.ndarray,
        strategy_parameters: np.ndarray,
        children_strategy_params: np.ndarray,
        strategy: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
        """
        Selects survivors for the next generation.

        :param population: Current population.
        :param children: Generated offspring.
        :param fitness_values: Pre-calculated fitness values of the current population.
        :param strategy_parameters: Strategy parameters of the current population.
        :param children_strategy_params: Strategy parameters of the offspring.
        :param strategy: Survivor selection strategy ('mu, lambda' or 'mu+lambda').
        :return: A tuple containing:
            - New population: Selected individuals for the next generation.
            - New strategy parameters: Corresponding strategy parameters for the selected individuals.
            - Selected fitness values: Fitness values of the selected individuals.
            - Minimum fitness value: The minimum fitness value in the new population.
        """
        if strategy == constants.ES_SURVIVOR_STRATEGY_MU_LAMBDA:
            combined_population = children
            combined_strategy_params = children_strategy_params
            combined_fitness_values = np.array(
                [self.safe_fitness(child) for child in children]
            )
        elif strategy == constants.ES_SURVIVOR_STRATEGY_MU_PLUS_LAMBDA:
            combined_population = np.vstack((population, children))
            combined_strategy_params = np.vstack(
                (strategy_parameters, children_strategy_params)
            )
            children_fitness_values = np.array(
                [self.safe_fitness(child) for child in children]
            )
            combined_fitness_values = np.hstack(
                (fitness_values, children_fitness_values)
            )
        else:
            raise ValueError("Invalid survivor selection strategy specified.")

        sorted_indices = np.argsort(combined_fitness_values)
        selected_indices = sorted_indices[: self.mu]
        new_population = combined_population[selected_indices]
        new_strategy_params = combined_strategy_params[selected_indices]
        selected_fitness_values = combined_fitness_values[selected_indices]
        return (
            new_population,
            new_strategy_params,
            selected_fitness_values,
            selected_fitness_values[0],
            children_fitness_values,
        )

    def update_parameters_end_iteration(
        self, generation: int, population: np.ndarray, min_child_value: float
    ) -> bool:
        """
        Update parameters at the end of each generation. Track the best individual found and
        determine if the success threshold has been met.

        :param generation: The current generation number.
        :param population: The current population of individuals.
        :param min_child_value: The fitness value of the best individual in the current population.
        :return: A boolean indicating whether the success threshold has been met.
        """
        if min_child_value <= self.best_individual_value:
            self.best_individual_value = min_child_value
            self.best_individual_execution = population[0].copy()
        self.best_individuals.append(
            (generation, min_child_value, population[0].copy())
        )
        return self.problem.is_success(min_child_value)
