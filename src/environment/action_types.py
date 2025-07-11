from typing import Any, List, Tuple, Union

import numpy as np

import utils.constants as constants
from configurations.operations import Operation


class ActionType:
    """
    Base class for all action types.
    Defines the parameter range and enforces implementation of sample_value.
    """

    def __init__(self, param_range: Union[List[float], None] = None) -> None:
        """
        Initialize the action type.

        Args:
            param_range (List[float] | None): Range [min, max] for the parameter.
                                              Defaults to [0.0, DEFAULT_UPPER_LIMIT].
        """
        self.param_range = (
            param_range if param_range else [0.0, constants.DEFAULT_UPPER_LIMIT]
        )

    def sample_value(self, action_index: int) -> float:
        """
        To be implemented by subclasses.

        Args:
            action_index (int): Action chosen by the agent.

        Returns:
            float: Resulting parameter value.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


class DirectSelectionAction(ActionType):
    """
    Discrete selection from a range divided into fixed-size subintervals.
    """

    def __init__(self, param_range: List[float], epsilon: float) -> None:
        super().__init__(param_range)
        self.epsilon = epsilon
        self.subintervals = self.calculate_subintervals()

    def calculate_subintervals(self) -> List[Tuple[float, float]]:
        """
        Divide the range into equal-sized subintervals based on epsilon.

        Returns:
            List of tuples representing (low, high) bounds of each subinterval.
        """
        num_intervals = max(
            1, int((self.param_range[1] - self.param_range[0]) / self.epsilon)
        )
        interval_range = (self.param_range[1] - self.param_range[0]) / num_intervals
        return [
            (
                self.param_range[0] + i * interval_range,
                self.param_range[0] + (i + 1) * interval_range,
            )
            for i in range(num_intervals)
        ]

    def sample_value(self, action_index: int) -> float:
        """
        Sample a value uniformly from the selected subinterval.

        Args:
            action_index (int): Index of the subinterval.

        Returns:
            float: A randomly sampled value within the subinterval.
        """
        low, high = self.subintervals[action_index]
        return np.random.uniform(low, high)


class OperationBasedAction(ActionType):
    """
    Applies one of several predefined arithmetic operations to the current value.
    """

    def __init__(
        self,
        param_range: Union[List[float], None],
        operations: List[Operation],
    ) -> None:
        super().__init__(param_range)
        self.operations = operations

    def sample_value(self, action_index: int, current_value: float) -> float:
        """
        Apply the selected operation to the current value.

        Args:
            action_index (int): Index of the operation to apply.
            current_value (float): Current value of the parameter.

        Returns:
            float: Result after applying the operation, clipped to the range.
        """
        op = self.operations[action_index]
        new_val = op.apply(current_value)
        return np.clip(new_val, self.param_range[0], self.param_range[1])


class CategoricalAction(ActionType):
    """
    Selects a value from a predefined set of categorical options.
    """

    def __init__(self, values: List[str]) -> None:
        """
        Initialize with a list of possible values.

        Args:
            values (List[str]): List of categorical choices.
        """
        super().__init__()
        self.values = values

    def sample_value(self, action_index: int) -> Any:
        """
        Return the categorical value corresponding to the given index.

        Args:
            action_index (int): Index of the selected value.

        Returns:
            Any: Selected categorical value.
        """
        return self.values[action_index]


class ContinuousAction(ActionType):
    """
    Accepts a continuous value directly from the agent within a specified range.
    """

    def __init__(self, param_range: List[float]) -> None:
        """
        Initialize a continuous action with its valid range.

        Args:
            param_range (List[float]): Range [min, max] for valid values.
        """
        super().__init__(param_range)

    def sample_value(self, action_index: float) -> float:
        """
        Clip the agent's output to the valid range and return it.

        Args:
            action_index (float): Raw action output from the agent.

        Returns:
            float: Clipped continuous value.
        """
        return np.clip(action_index, self.param_range[0], self.param_range[1])
