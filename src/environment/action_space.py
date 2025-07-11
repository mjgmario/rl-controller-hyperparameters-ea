from typing import Any, Dict, List, Union

from configurations.operation_parser import OperationParser
from environment.action_types import (
    ActionType,
    CategoricalAction,
    ContinuousAction,
    DirectSelectionAction,
    OperationBasedAction,
)


class ActionSpace:
    def __init__(
        self,
        hyperparameters: Dict[str, Dict[str, Any]],
        controllable_parameters: List[str],
        epsilon: float = 0.01,
    ) -> None:
        """
        Initializes the ActionSpace using hyperparameter definitions and a list
        of which parameters the agent can control.

        Args:
            hyperparameters (Dict[str, Dict[str, Any]]): Dictionary containing action specs for each parameter.
            controllable_parameters (List[str]): List of parameters to be controlled by the agent.
            epsilon (float): Step size used for discretization in DirectSelectionAction.
        """
        self.epsilon = epsilon
        self.actions = self.initialize_actions(hyperparameters, controllable_parameters)

    def initialize_actions(
        self,
        hyperparameters: Dict[str, Dict[str, Any]],
        controllable_parameters: List[str],
    ) -> Dict[str, ActionType]:
        """
        Instantiates action objects for all controllable parameters based on their configuration.

        Args:
            hyperparameters (Dict): Hyperparameter action definitions.
            controllable_parameters (List): Subset of parameters to include.

        Returns:
            Dict[str, ActionType]: Dictionary mapping each parameter to its action type instance.
        """
        actions = {}
        for param, settings in hyperparameters.items():
            if param not in controllable_parameters:
                continue

            if "operations" in settings:
                ops_list = OperationParser.parse_operations(settings["operations"])
                actions[param] = OperationBasedAction(
                    param_range=settings.get("range", None),
                    operations=ops_list,
                )
            elif "values" in settings:
                actions[param] = CategoricalAction(values=settings["values"])
            elif settings.get("continuous", False):
                actions[param] = ContinuousAction(param_range=settings["range"])
            else:
                actions[param] = DirectSelectionAction(
                    param_range=settings["range"], epsilon=self.epsilon
                )

        return actions

    def sample_values(
        self,
        action_indices: Dict[str, Union[int, float]],
        current_values: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Converts the action indices selected by the agent into actual hyperparameter values.

        Args:
            action_indices (Dict[str, int | float]): Indices returned by the agent for each parameter.
            current_values (Dict[str, float]): Current values of the hyperparameters.

        Returns:
            Dict[str, float]: New hyperparameter values to apply.
        """
        action_values = {}
        for param, action_index in action_indices.items():
            action_type = self.actions[param]
            if isinstance(action_type, OperationBasedAction):
                action_values[param] = action_type.sample_value(
                    action_index, current_value=current_values.get(param)
                )
            else:
                action_values[param] = action_type.sample_value(action_index)
        return action_values
