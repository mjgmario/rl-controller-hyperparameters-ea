from unittest.mock import MagicMock

import numpy as np

from controllers.no_action_controlller import NoOpController

# ----------- Mocks -----------


class MockAlgorithm:
    def __init__(self, stop_after=3):
        self.call_count = 0
        self.stop_after = stop_after
        self.fitness_values = np.array([1.0, 2.0, 3.0])
        self.generation = 1

    def reset_population(self):
        self.call_count = 0

    def run_generation(self, step_size):
        self.call_count += 1
        self.fitness_values = np.array([self.call_count])  # simulate improving fitness
        return self.call_count >= self.stop_after


class MockEnv:
    def __init__(self):
        self.algorithm = MockAlgorithm()
        self.num_generations_per_iteration = 1
        self.action_space = MagicMock()
        self.action_space.actions = []
        self.num_generations_per_iteration = 1

    def get_state(self):
        return {"mock_state": True}


# ----------- Tests -----------


def test_noopcontroller_train_executes_steps():
    env = MockEnv()
    controller = NoOpController(env, num_generations_per_iteration=1)
    history = controller.train(episodes=1)

    assert isinstance(history, list)
    assert len(history) == env.algorithm.stop_after
    assert all("action" in entry and entry["action"] == {} for entry in history)
    assert all("reward" in entry for entry in history)
    assert history[-1]["done"] is True


def test_noopcontroller_train_multiple_episodes():
    env = MockEnv()
    controller = NoOpController(env, num_generations_per_iteration=1)
    history = controller.train(episodes=2)

    steps_per_episode = env.algorithm.stop_after
    assert len(history) == 2 * steps_per_episode
    assert set(entry["episode"] for entry in history) == {0, 1}


def test_noopcontroller_infer_until_done():
    env = MockEnv()
    controller = NoOpController(env, num_generations_per_iteration=1)
    history = controller.infer()

    assert isinstance(history, list)
    assert len(history) == env.algorithm.stop_after
    assert history[-1]["done"] is True
