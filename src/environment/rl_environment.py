import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from environment.action_types import (
    CategoricalAction,
    ContinuousAction,
    DirectSelectionAction,
    OperationBasedAction,
)
from environment.observable import Observable
from environment.observable_factory import ObservableFactory
from environment.rewards import (
    binary_reward,
    combined_increment_with_binary,
    parent_offspring_reward,
    population_reward,
    weighted_improvement_reward,
)
from tensorforce.environments import Environment
from utils import constants
from utils.utils import format_differential_weights, format_strategy_parameters

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("experiment.log", mode="a"),
    ],
)

logger = logging.getLogger(__name__)


class RLEnvironment(Environment):
    """
    Reinforcement Learning Environment for tuning evolutionary algorithm hyperparameters.

    Attributes:
        algorithm (Any): The evolutionary algorithm instance.
        action_space (Any): ActionSpace object that defines possible hyperparameter adjustments.
        num_generations_per_iteration (int): Number of generations per environment step.
        max_generations (Optional[int]): Max total generations before termination.
        observables (List[Observable]): List of observable instances.
        reward_type (str): Strategy for computing reward.
        omega (float): Weighting factor for population reward.
        ref_window (int): Reference window size for weighted improvement reward.
        time_scale (float): Scaling factor for time-sensitive rewards.
    """

    def __init__(
        self,
        algorithm: Any,
        action_space: Any,
        num_generations_per_iteration: int = 1,
        max_generations: Optional[int] = constants.MAX_GENERATIONS,
        observables: Optional[List[Union[str, Observable]]] = None,
        observables_config: Optional[Dict[str, Dict[str, Any]]] = None,
        reward_type: str = "parent_offspring",
        omega: float = 0.5,
        ref_window: int = 5,
        time_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.algorithm = algorithm
        self.action_space = action_space
        self.num_generations_per_iteration = num_generations_per_iteration
        self.max_generations = max_generations

        # Histories
        self.best_history: List[float] = [float(np.min(self.algorithm.fitness_values))]
        self.delta_history: List[float] = []

        # Reward config
        self.reward_type = reward_type
        self.omega = omega
        self.ref_window = ref_window
        self.time_scale = time_scale

        # Observables setup
        default_names = list(ObservableFactory._registry.keys())
        entries = observables or default_names
        self.observables = ObservableFactory.build(entries, observables_config)

        # Initialize algorithm
        self.algorithm.reset_population()

    def states(self) -> Dict[str, Any]:
        """
        Define the observable state space.

        Returns:
            Dict[str, Any]: Dictionary mapping observable names to space specs.
        """
        spec: Dict[str, Any] = {}
        for obs in self.observables:
            # if Observable instance
            minv = getattr(obs, "min_val", 0.0)
            maxv = getattr(obs, "max_val", 1.0)
            if getattr(obs, "bins", None) is not None:
                spec[obs.name] = {
                    "type": "int",
                    "shape": (),
                    "num_values": obs.bins,
                }
            else:
                spec[obs.name] = {
                    "type": "float",
                    "shape": (),
                    "min_value": minv,
                    "max_value": maxv,
                }
        return spec

    def actions(self) -> Dict[str, Any]:
        """
        Define the action space for Tensorforce, handling different types of actions:
        - CategoricalAction: discrete, based on predefined values.
        - OperationBasedAction: discrete, based on number of operations.
        - DirectSelectionAction: discrete, based on subintervals.
        - ContinuousAction: continuous float values.

        Returns:
            Dict[str, Any]: Action space specification for Tensorforce.
        """
        action_space = {}

        for param_name, action_type in self.action_space.actions.items():
            if isinstance(action_type, CategoricalAction):
                action_space[param_name] = {
                    "type": "int",
                    "num_values": len(action_type.values),
                }

            elif isinstance(action_type, OperationBasedAction):
                # Discrete action based on number of operations
                action_space[param_name] = {
                    "type": "int",
                    "num_values": len(action_type.operations),
                }

            elif isinstance(action_type, DirectSelectionAction):
                # Discrete action based on subintervals
                action_space[param_name] = {
                    "type": "int",
                    "num_values": len(action_type.subintervals),
                }

            elif isinstance(action_type, ContinuousAction):
                # Continuous action in a given range
                action_space[param_name] = {
                    "type": "float",
                    "shape": (),
                    "min_value": action_type.param_range[0],
                    "max_value": action_type.param_range[1],
                }

            else:
                raise ValueError(
                    f"Unknown action type for parameter '{param_name}': {type(action_type)}"
                )

        return action_space

    def get_state(self) -> Dict[str, Any]:
        """
        Build the current state representation based on observables.

        Returns:
            Dict[str, Any]: Dictionary of observable values.
        """
        pop = self.algorithm.population
        fitness = self.algorithm.fitness_values

        state: Dict[str, Any] = {}
        gen = self.algorithm.generation
        for obs in self.observables:
            raw = obs.compute(
                pop,
                fitness,
                self.best_history,
                self.algorithm.bounds,
                gen,
                self.max_generations,
            )
            # discretize or clamp
            state[obs.name] = obs.discretize(raw)
        return state

    def execute(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, float]:
        """
        Execute an environment step by applying the action and running generations.

        Args:
            action (Dict[str, Any]): Action to be applied.

        Returns:
            Tuple[Dict[str, Any], bool, float]: (next_state, done_flag, reward)
        """
        current_values: Dict[str, float] = {}
        for p in self.action_space.actions:
            if hasattr(self.algorithm, p):
                current_values[p] = getattr(self.algorithm, p)
            else:
                current_values[p] = None
        params = self.action_space.sample_values(action, current_values=current_values)
        self.apply_hyperparameters(params)

        # Record fitness before generation
        pre = np.copy(self.algorithm.fitness_values)

        # Run evolution step(s)
        done_flag = self.algorithm.run_generation(self.num_generations_per_iteration)

        # Record fitness after generation
        post = np.copy(self.algorithm.fitness_values)
        best_post = float(np.min(post))
        self.best_history.append(best_post)

        delta = parent_offspring_reward(pre, post, time_scale=self.time_scale)
        self.delta_history.append(delta)

        # Compute reward according to selected scheme
        if self.reward_type == "parent_offspring":
            reward = parent_offspring_reward(pre, post, time_scale=self.time_scale)
        elif self.reward_type == "population_mean":
            reward = population_reward(pre, post, omega=self.omega, mode="mean")
        elif self.reward_type == "population_max":
            reward = population_reward(pre, post, omega=self.omega, mode="max")
        elif self.reward_type == "binary":
            reward = binary_reward(self.best_history)
        elif self.reward_type == "weighted_improvement":
            reward = weighted_improvement_reward(
                self.delta_history, ref_window=self.ref_window
            )
        elif self.reward_type == "combined_increment_with_binary":
            reward = combined_increment_with_binary(
                pre, post, self.best_history, self.time_scale
            )
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")

        done = done_flag or (
            self.algorithm.generation >= self.algorithm.max_number_generation
        )
        next_state = self.get_state()
        return next_state, done, reward

    def apply_hyperparameters(self, new_hyperparameters: Dict[str, Any]) -> None:
        """
        Apply the specified hyperparameter values to the algorithm.

        Args:
            new_hyperparameters (Dict[str, Any]): Hyperparameters to be applied.
        """
        for key, value in new_hyperparameters.items():
            if key == "strategy_parameters":
                self.algorithm.external_sigma_control = True
                self.algorithm.strategy_parameters[:] = format_strategy_parameters(
                    value, self.algorithm.strategy_parameters
                )
                continue
            elif key == "differential_weights":
                diff_num = getattr(self.algorithm, "differential_number", None)
                if diff_num is None:
                    logger.info(
                        "Warning: algorithm has no 'differential_number'; skipping differential_weights."
                    )
                    continue
                formatted = format_differential_weights(value, diff_num)
                setattr(self.algorithm, "differential_weights", formatted)
            else:
                if not hasattr(self.algorithm, key):
                    logger.info(
                        f"Warning: '{key}' is not an attribute of the algorithm."
                    )
                    continue
                setattr(self.algorithm, key, value)

    def decode_action(self, action: Dict[str, int]) -> Dict[str, float]:
        """
        Convert discrete action indices to their corresponding parameter values.

        Args:
            action (Dict[str, int]): Discrete action indices.

        Returns:
            Dict[str, float]: Decoded continuous values.
        """
        return self.action_space.sample_values(action)

    def reset(self) -> Dict[str, Any]:
        """
        Reset the environment to its initial state.

        Returns:
            Dict[str, Any]: Initial observation after reset.
        """
        self.algorithm.reset_population()
        initial_state = self.get_state()
        return initial_state
