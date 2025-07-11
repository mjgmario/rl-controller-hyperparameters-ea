import pickle
from unittest.mock import MagicMock

import numpy as np
import pytest

from controllers.base_controller import BaseController

# ----------- Fixtures -----------


class DummyAlgorithm:
    def __init__(self):
        self.lr = np.array([0.01])
        self.momentum = [0.9]
        self.dropout = (0.5,)
        self.weight_decay = 0.0001
        self.schedule = [0.1, 0.1]
        self.unused = "non-numeric"


class DummyActionSpace:
    def __init__(self):
        self.actions = [
            "lr",
            "momentum",
            "dropout",
            "weight_decay",
            "schedule",
            "unused",
        ]


class DummyEnv:
    def __init__(self):
        self.action_space = DummyActionSpace()
        self.algorithm = DummyAlgorithm()

    def get_state(self):
        return {"obs": [1, 2, 3], "info": {"x": 42}}


# ----------- Tests -----------


def test_get_state_returns_env_state():
    env = DummyEnv()
    controller = BaseController(env)
    state = controller.get_state()
    assert isinstance(state, dict)
    assert "obs" in state


def test_get_state_returns_none_if_unavailable():
    env = MagicMock()
    del env.get_state
    controller = BaseController(env)
    assert controller.get_state() is None


def test_get_hyperparameters_parses_correctly():
    controller = BaseController(DummyEnv())
    hp = controller.get_hyperparameters()
    assert isinstance(hp, dict)
    assert hp["lr"] == 0.01
    assert hp["momentum"] == 0.9
    assert hp["dropout"] == 0.5
    assert hp["weight_decay"] == 0.0001
    assert hp["schedule"] == 0.1
    assert hp["unused"] == "non-numeric"


def test_record_step_appends_history():
    controller = BaseController(DummyEnv())
    controller.record_step(step=1, reward=0.5)
    assert len(controller.history) == 1
    assert controller.history[0]["step"] == 1


def test_save_history_creates_file_and_saves(tmp_path):
    controller = BaseController(DummyEnv())
    controller.record_step(episode=1, reward=1.0)
    save_path = tmp_path / "history.pkl"
    controller.save_history(str(save_path))

    assert save_path.exists()

    with open(save_path, "rb") as f:
        loaded = pickle.load(f)
    assert isinstance(loaded, list)
    assert loaded[0]["reward"] == 1.0


def test_set_environment_replaces_env():
    old_env = DummyEnv()
    new_env = MagicMock()
    controller = BaseController(old_env)
    controller.set_environment(new_env)
    assert controller.env == new_env


def test_train_not_implemented():
    controller = BaseController(DummyEnv())
    with pytest.raises(NotImplementedError, match="train"):
        controller.train()


def test_infer_not_implemented():
    controller = BaseController(DummyEnv())
    with pytest.raises(NotImplementedError, match="infer"):
        controller.infer()
