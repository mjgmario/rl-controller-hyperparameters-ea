from unittest.mock import MagicMock, patch

import pytest

from controllers.rl_controller import RLController

# ---------- Mocked Components ----------


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.act.side_effect = lambda states, independent: {"mock_action": 1}
    agent.observe.return_value = None
    agent.close.return_value = None
    agent.save.return_value = None
    return agent


@pytest.fixture
def mock_env():
    env = MagicMock()
    env.reset.return_value = {"state": 0}
    env.execute.side_effect = [({"state": i + 1}, False, 0.5) for i in range(2)] + [
        ({"state": 3}, True, 1.0)
    ]
    env.algorithm.generation = 3
    env.num_generations_per_iteration = 1
    env.action_space.actions = []
    env.actions.return_value = {}
    return env


# ---------- Tests ----------


def test_rlcontroller_train_runs_episode(mock_env, mock_agent):
    controller = RLController(
        environment=mock_env,
        agent=mock_agent,
        mode="train",
        max_number_generations_per_episode=3,
    )
    history = controller.train(episodes=1)

    assert len(history) == 3
    for entry in history:
        assert "state" in entry
        assert "action" in entry
        assert "reward" in entry
        assert "done" in entry
    assert history[-1]["done"] is True
    mock_agent.observe.assert_called()
    mock_agent.act.assert_called()
    mock_agent.close.assert_called()


def test_rlcontroller_train_raises_in_inference_mode(mock_env, mock_agent):
    controller = RLController(environment=mock_env, agent=mock_agent, mode="inference")
    with pytest.raises(RuntimeError, match="Cannot train in 'inference' mode."):
        controller.train(episodes=1)


def test_rlcontroller_infer_runs_until_done(mock_env, mock_agent):
    controller = RLController(
        environment=mock_env,
        agent=mock_agent,
        mode="inference",
        max_number_generations_per_episode=3,
    )
    history = controller.infer()

    assert isinstance(history, list)
    assert len(history) == 3
    assert history[-1]["done"] is True
    mock_agent.act.assert_called()
    mock_agent.close.assert_called()


def test_rlcontroller_save_agent(tmp_path, mock_env, mock_agent):
    controller = RLController(environment=mock_env, agent=mock_agent)
    save_dir = tmp_path / "saved_agent"
    controller.save_agent(str(save_dir))

    mock_agent.save.assert_called_with(directory=str(save_dir))
    assert save_dir.exists()


@patch("controllers.rl_controller.Agent")
def test_rlcontroller_set_environment_recreates_agent(
    mock_agent_class, mock_env, mock_agent
):
    mock_agent_class.create.return_value = mock_agent
    controller = RLController(environment=mock_env, agent=mock_agent)
    new_env = MagicMock()
    new_env.num_generations_per_iteration = 1
    new_env.action_space.actions = []
    new_env.actions.return_value = {}
    controller.set_environment(new_env)

    mock_agent_class.create.assert_called()
    assert controller.env == new_env
