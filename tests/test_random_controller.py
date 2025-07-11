import random
from unittest.mock import MagicMock

from controllers.random_controller import RandomController

# ---------- Mock Environment ----------


class MockRLEnvironment:
    def __init__(self, max_steps=3):
        self._step = 0
        self._max_steps = max_steps
        self.action_space = MagicMock()
        self.algorithm = MagicMock()
        self.action_space.actions = ["lr", "momentum"]
        self.num_generations_per_iteration = 1

    def reset(self):
        self._step = 0
        return {"init": True}

    def get_state(self):
        return {"mock_state": True}

    def actions(self):
        return {
            "lr": {"type": "float", "min_value": 0.001, "max_value": 0.1},
            "momentum": {"type": "int", "num_values": 3},
        }

    def execute(self, action):
        self._step += 1
        done = self._step >= self._max_steps
        reward = random.uniform(0, 1)
        next_state = {"step": self._step}
        return next_state, done, reward


# ---------- Tests ----------


def test_randomcontroller_train_records_steps():
    env = MockRLEnvironment(max_steps=4)
    controller = RandomController(env)
    history = controller.train(episodes=1)

    assert isinstance(history, list)
    assert len(history) == 4
    for entry in history:
        assert isinstance(entry["action"], dict)
        assert "lr" in entry["action"]
        assert "momentum" in entry["action"]
        assert 0.001 <= entry["action"]["lr"] <= 0.1
        assert entry["action"]["momentum"] in [0, 1, 2]
        assert "reward" in entry


def test_randomcontroller_train_multiple_episodes():
    env = MockRLEnvironment(max_steps=2)
    controller = RandomController(env)
    history = controller.train(episodes=3)

    assert len(history) == 6
    episodes = set(entry["episode"] for entry in history)
    assert episodes == {0, 1, 2}


def test_randomcontroller_infer_until_done():
    env = MockRLEnvironment(max_steps=5)
    controller = RandomController(env)
    history = controller.infer()

    assert len(history) == 5
    assert history[-1]["done"] is True
