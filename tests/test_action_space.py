from unittest.mock import MagicMock, patch

import pytest

from environment.action_space import ActionSpace
from environment.action_types import (
    CategoricalAction,
    ContinuousAction,
    DirectSelectionAction,
    OperationBasedAction,
)

# ----------- Fixtures -----------


@pytest.fixture
def basic_hyperparameter_config():
    return {
        "learning_rate": {"range": [0.001, 0.1], "continuous": True},
        "optimizer": {"values": ["adam", "sgd", "rmsprop"]},
        "momentum": {"range": [0.0, 1.0]},
        "batch_size": {
            "operations": [
                {"type": "multiply", "value": 2.0},
                {"type": "add", "value": 16},
            ],
            "range": [16, 256],
        },
    }


@pytest.fixture
def controllable():
    return ["learning_rate", "optimizer", "momentum", "batch_size"]


# ----------- Tests -----------


@patch("environment.action_space.OperationParser")
def test_actionspace_initializes_all_types(
    mock_parser, basic_hyperparameter_config, controllable
):
    mock_parser.parse_operations.return_value = [MagicMock(), MagicMock()]
    space = ActionSpace(basic_hyperparameter_config, controllable)

    actions = space.actions
    assert isinstance(actions["learning_rate"], ContinuousAction)
    assert isinstance(actions["optimizer"], CategoricalAction)
    assert isinstance(actions["momentum"], DirectSelectionAction)
    assert isinstance(actions["batch_size"], OperationBasedAction)
    assert len(actions) == 4
