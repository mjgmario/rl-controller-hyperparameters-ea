from unittest.mock import MagicMock

import numpy as np
import pytest

from environment.action_types import CategoricalAction, ContinuousAction
from environment.observable import Entropy
from environment.rl_environment import RLEnvironment

# --------- Fixtures ---------


@pytest.fixture
def mock_algorithm():
    algo = MagicMock()
    algo.fitness_values = np.array([3.0, 2.0, 1.0])
    algo.population = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    algo.bounds = [(0.0, 1.0), (0.0, 1.0)]
    algo.run_generation.return_value = False
    algo.generation = 0
    algo.max_number_generation = 100
    algo.reset_population.return_value = None
    algo.lr = 0.01  # for controllable param
    return algo


@pytest.fixture
def mock_action_space():
    class DummyActionSpace:
        def __init__(self):
            self.actions = {
                "lr": ContinuousAction([0.001, 0.1]),
                "opt": CategoricalAction(["adam", "sgd"]),
            }

        def sample_values(self, action, current_values=None):
            return {"lr": 0.05, "opt": "adam"}

    return DummyActionSpace()


# --------- Tests ---------


def test_initialize_environment(mock_algorithm, mock_action_space):
    env = RLEnvironment(
        algorithm=mock_algorithm,
        action_space=mock_action_space,
        observables=["entropy"],
        reward_type="parent_offspring",
    )

    assert isinstance(env.observables[0], Entropy)
    assert env.best_history == [1.0]  # min of initial fitness_values


def test_state_space_specification(mock_algorithm, mock_action_space):
    env = RLEnvironment(algorithm=mock_algorithm, action_space=mock_action_space)
    state_space = env.states()
    assert isinstance(state_space, dict)
    for name, spec in state_space.items():
        assert "type" in spec


def test_action_space_specification(mock_algorithm, mock_action_space):
    env = RLEnvironment(algorithm=mock_algorithm, action_space=mock_action_space)
    actions = env.actions()
    assert "lr" in actions and actions["lr"]["type"] == "float"
    assert "opt" in actions and actions["opt"]["type"] == "int"


def test_execute_returns_expected_output(mock_algorithm, mock_action_space):
    env = RLEnvironment(algorithm=mock_algorithm, action_space=mock_action_space)
    action = {"lr": 0, "opt": 0}
    next_state, done, reward = env.execute(action)
    assert isinstance(next_state, dict)
    assert done is False
    assert isinstance(reward, float)


def test_execute_handles_reward_types(mock_algorithm, mock_action_space):
    reward_types = [
        "parent_offspring",
        "population_mean",
        "population_max",
        "binary",
        "weighted_improvement",
        "combined_increment_with_binary",
    ]
    for reward_type in reward_types:
        env = RLEnvironment(
            algorithm=mock_algorithm,
            action_space=mock_action_space,
            reward_type=reward_type,
        )
        action = {"lr": 0, "opt": 0}
        next_state, done, reward = env.execute(action)
        assert isinstance(reward, float)


def test_execute_raises_on_invalid_reward_type(mock_algorithm, mock_action_space):
    env = RLEnvironment(
        algorithm=mock_algorithm,
        action_space=mock_action_space,
        reward_type="invalid_reward",
    )
    with pytest.raises(ValueError, match="Unknown reward type"):
        env.execute({"lr": 0, "opt": 0})


def test_apply_hyperparameters_sets_attributes(mock_algorithm, mock_action_space):
    env = RLEnvironment(algorithm=mock_algorithm, action_space=mock_action_space)
    env.apply_hyperparameters({"lr": 0.05})
    assert mock_algorithm.lr == 0.05


def test_reset_calls_algorithm_reset(mock_algorithm, mock_action_space):
    _ = RLEnvironment(algorithm=mock_algorithm, action_space=mock_action_space)
    mock_algorithm.reset_population.assert_called_once()
