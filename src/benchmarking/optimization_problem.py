from typing import Callable, Optional
import numpy as np


class OptimizationProblem:
    def __init__(
        self,
        objective_function: Callable[[np.ndarray], float],
        bounds: np.ndarray,
        maximize: bool = False,
        target_value: Optional[float] = None,
        epsilon: float = 1e-6,
        x_opt=None,
    ):
        """
        Represents an optimization problem for use in evolutionary algorithms.

        Args:
            objective_function (Callable[[np.ndarray], float]):
                The objective function to optimize. It should take a 1D NumPy array and return a float.
            bounds (np.ndarray):
                A NumPy array of shape (n_dimensions, 2), where each row represents (lower_bound, upper_bound)
                for one variable.
            maximize (bool, optional):
                If True, the problem will be transformed internally for maximization. Defaults to False (minimization).
            target_value (float, optional):
                Optional target objective function value to consider the problem solved if reached. Defaults to None.
            epsilon (float, optional):
                Acceptable absolute difference from the target_value to consider the solution successful. Defaults to 1e-6.
        """
        self.objective_function = objective_function
        self.bounds = np.array(bounds)
        self.maximize = maximize
        self.target_value = target_value
        self.epsilon = epsilon
        self._multiplier = -1.0 if maximize else 1.0
        self.x_opt = x_opt

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluates the adapted objective function, taking into account whether the problem is minimization or maximization.

        Args:
            x (np.ndarray): A 1D array representing a candidate solution.

        Returns:
            float: The adapted fitness value (minimized or maximized).
        """
        raw_value = self.objective_function(x)
        return self._multiplier * raw_value

    def is_success(self, value: float) -> bool:
        """
        Checks if the provided fitness value is within epsilon of the target value.

        Args:
            value (float): The evaluated fitness value.

        Returns:
            bool: True if the value is close enough to the target value, False otherwise.
        """
        if self.target_value is None:
            return False
        return abs((self._multiplier * value) - self.target_value) <= self.epsilon
