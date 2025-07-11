from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from benchmarking.optimization_problem import OptimizationProblem


class AbstractEvolutionaryAlgorithm(ABC):
    """
    Abstract base class for evolutionary algorithms.

    Defines core properties and interface that all evolutionary optimizers must implement.
    """

    def __init__(self, problem: OptimizationProblem) -> None:
        """
        Initialize shared attributes for evolutionary algorithms.

        Parameters:
        - problem: OptimizationProblem instance providing bounds and evaluation function.
        """
        self.problem = problem
        self.bounds = problem.bounds
        self.fitness_function = problem.evaluate
        # Common state members
        self.population: np.ndarray
        self.fitness_values: np.ndarray
        self.generation: int
        self.num_function_calls: int

    @abstractmethod
    def initialize_population(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize the population and compute initial fitness values.

        Returns:
        - population: NumPy array of shape (pop_size, dim).
        - fitness_values: NumPy array of shape (pop_size,).
        """
        pass

    @abstractmethod
    def run_generation(self, num_generations: int) -> bool:
        """
        Execute one or more generations of the evolutionary process.

        Parameters:
        - num_generations: Number of generations to run in this call.

        Returns:
        - success: True if termination criteria met, False otherwise.
        """
        pass

    @abstractmethod
    def reset_population(self) -> None:
        """
        Reset the population to its initial state and clear counters.
        """
        pass
