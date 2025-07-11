import numpy as np
import pytest

from controllers.pop_de_controller import PopDEController
import controllers.pop_de_controller as pc

# Dummy algorithm and environment for train/infer tests
after = None


class DummyAlg:
    def __init__(self):
        self.generation = 0


class DummyEnv:
    num_generations_per_iteration = 1

    def __init__(self):
        self.algorithm = DummyAlg()

    def reset(self):
        return "init"

    def execute(self, action):
        return ("next", True, 0.1)


# Stub BaseController to capture record_step calls
class StubBaseController:
    def __init__(self, environment):
        self.env = environment
        self.history = []

    def record_step(self, episode, step, state, action, hyperparameters, reward, done):
        self.history.append(
            {
                "episode": episode,
                "step": step,
                "state": state,
                "action": action,
                "hyperparameters": hyperparameters,
                "reward": reward,
                "done": done,
            }
        )


@pytest.fixture(autouse=True)
def stub_base(monkeypatch):
    # Monkeypatch BaseController in the module so PopDEController uses our stub
    monkeypatch.setattr(pc, "BaseController", StubBaseController)
    return


# ---------------------- Initialization ----------------------


def test_init_valid_methods_and_parameters():
    env = DummyEnv()
    ctrl_jade = PopDEController(env, method="JADE", c=0.1)
    assert ctrl_jade.method == "jade"
    assert ctrl_jade.c == 0.1

    ctrl_shade = PopDEController(env, method="shade", c=0.5, history_size=3)
    assert ctrl_shade.method == "shade"
    assert ctrl_shade.c == 0.5
    assert ctrl_shade.H == 3


def test_init_invalid_method_raises():
    with pytest.raises(ValueError):
        PopDEController(DummyEnv(), method="foo", c=0.1)


def test_init_invalid_c_raises():
    with pytest.raises(ValueError):
        PopDEController(DummyEnv(), method="jade", c=0)
    with pytest.raises(ValueError):
        PopDEController(DummyEnv(), method="shade", c=1.1)


# ---------------------- _cauchy_positive ----------------------


def test_cauchy_positive_retries_until_positive(monkeypatch):
    calls = {"count": 0}

    def fake_cauchy():
        calls["count"] += 1
        # First return a value that makes v <= 0, then a zero
        return -6 if calls["count"] == 1 else 0

    monkeypatch.setattr(np.random, "standard_cauchy", fake_cauchy)

    result = PopDEController(DummyEnv(), method="jade", c=0.1)._cauchy_positive(
        0.5, 0.1
    )
    assert result == pytest.approx(0.5)
    assert calls["count"] == 2


# ---------------------- _sample_params ----------------------


def test_sample_params_jade(monkeypatch):
    # Make random draws deterministic
    monkeypatch.setattr(np.random, "standard_cauchy", lambda: 0.0)
    monkeypatch.setattr(np.random, "normal", lambda loc, scale: 0.0)

    ctrl = PopDEController(DummyEnv(), method="jade", c=0.1)
    ctrl.mu_F = 0.5
    ctrl.mu_CR = 0.5
    F, CR = ctrl._sample_params()
    assert F == pytest.approx(0.5)
    assert CR == pytest.approx(0.0)


def test_sample_params_shade(monkeypatch):
    monkeypatch.setattr(np.random, "randint", lambda low, high=None: 0)
    monkeypatch.setattr(np.random, "standard_cauchy", lambda: 0.0)
    monkeypatch.setattr(np.random, "normal", lambda loc, scale: 0.0)

    ctrl = PopDEController(DummyEnv(), method="shade", c=0.1, history_size=4)
    # All memory entries are default 0.5
    F, CR = ctrl._sample_params()
    assert F == pytest.approx(0.5)
    assert CR == pytest.approx(0.0)


# ---------------------- Statistics reset and accumulate ----------------------


def test_reset_statistics_jade():
    ctrl = PopDEController(DummyEnv(), method="jade", c=0.1)
    # Change mu values
    ctrl.mu_F = 0.8
    ctrl.mu_CR = 0.2
    # Add dummy successes
    ctrl._success_F = [1]
    ctrl._success_CR = [1]
    ctrl._success_df = [1]
    ctrl._reset_statistics()
    assert ctrl.mu_F == pytest.approx(0.5)
    assert ctrl.mu_CR == pytest.approx(0.5)
    assert ctrl._success_F == []
    assert ctrl._success_CR == []
    assert ctrl._success_df == []


def test_reset_statistics_shade():
    ctrl = PopDEController(DummyEnv(), method="shade", c=0.1, history_size=3)
    ctrl.MF[:] = [0.1, 0.2, 0.3]
    ctrl.MCR[:] = [0.4, 0.5, 0.6]
    ctrl.k = 2
    ctrl._success_F = [1]
    ctrl._success_CR = [1]
    ctrl._success_df = [1]
    ctrl._reset_statistics()
    assert np.all(ctrl.MF == 0.5)
    assert np.all(ctrl.MCR == 0.5)
    assert ctrl.k == 0
    assert ctrl._success_F == []
    assert ctrl._success_CR == []
    assert ctrl._success_df == []


def test_accumulate_generation_stats():
    ctrl = PopDEController(DummyEnv(), method="jade", c=0.1)
    # Non-positive reward should not accumulate
    ctrl._accumulate_generation_stats(0.3, 0.4, 0.0)
    assert ctrl._success_F == []
    # Positive reward accumulates
    ctrl._accumulate_generation_stats(0.3, 0.4, 0.5)
    assert ctrl._success_F == [0.3]
    assert ctrl._success_CR == [0.4]
    assert ctrl._success_df == [0.5]


# ---------------------- Flush generation statistics ----------------------


def test_flush_generation_statistics_jade():
    ctrl = PopDEController(DummyEnv(), method="jade", c=0.5)
    # Provide known successes
    ctrl._success_F = [1.0, 2.0]
    ctrl._success_CR = [0.2, 0.8]
    ctrl._success_df = [0.1, 0.2]  # unused for JADE
    # Initial mu
    ctrl.mu_F = 0.5
    ctrl.mu_CR = 0.5
    ctrl._flush_generation_statistics()
    # Lehmer mean = (1^2+2^2)/(1+2)=5/3≈1.6667
    expected_F = (1 - 0.5) * 0.5 + 0.5 * (5 / 3)
    expected_CR = (1 - 0.5) * 0.5 + 0.5 * 0.5
    assert ctrl.mu_F == pytest.approx(expected_F)
    assert ctrl.mu_CR == pytest.approx(expected_CR)
    assert ctrl._success_F == []


def test_flush_generation_statistics_shade():
    ctrl = PopDEController(DummyEnv(), method="shade", c=0.1, history_size=3)
    ctrl.MF[:] = [0.1, 0.2, 0.3]
    ctrl.MCR[:] = [0.4, 0.5, 0.6]
    ctrl.k = 1
    # Successes
    ctrl._success_F = [0.5, 1.0]
    ctrl._success_CR = [0.6, 0.8]
    ctrl._success_df = [4.0, 1.0]
    ctrl._flush_generation_statistics()
    # Compute expected
    improvements = np.array([4.0, 1.0])
    scaled = np.log1p(improvements)
    weights = scaled / scaled.sum()
    F_arr = np.array([0.5, 1.0])
    CR_arr = np.array([0.6, 0.8])
    lehmer_F = float(np.sum(weights * np.square(F_arr)) / np.sum(weights * F_arr))
    mean_CR = float(np.sum(weights * CR_arr))
    # Check that the entry at index 1 was updated
    assert ctrl.MF[1] == pytest.approx(lehmer_F)
    assert ctrl.MCR[1] == pytest.approx(mean_CR)
    # k should have advanced
    assert ctrl.k == 2
    assert ctrl._success_F == []


# ---------------------- Checkpoint helpers ----------------------


def test_save_and_load_state_jade(tmp_path):
    env = DummyEnv()
    ctrl = PopDEController(env, method="jade", c=0.1)
    ctrl.mu_F = 0.8
    ctrl.mu_CR = 0.3
    state_file = tmp_path / "state_jade.npz"
    ctrl.save_state(str(state_file))

    new_ctrl = PopDEController(env, method="jade", c=0.1)
    new_ctrl.load_state(str(state_file))
    assert new_ctrl.mu_F == pytest.approx(0.8)
    assert new_ctrl.mu_CR == pytest.approx(0.3)


def test_save_and_load_state_shade(tmp_path):
    env = DummyEnv()
    ctrl = PopDEController(env, method="shade", c=0.1, history_size=4)
    # Modify memories
    ctrl.MF[:] = [0.2, 0.3, 0.4, 0.5]
    ctrl.MCR[:] = [0.6, 0.7, 0.8, 0.9]
    ctrl.k = 2
    state_file = tmp_path / "state_shade.npz"
    ctrl.save_state(str(state_file))

    new_ctrl = PopDEController(env, method="shade", c=0.1, history_size=4)
    new_ctrl.load_state(str(state_file))
    assert np.all(new_ctrl.MF == [0.2, 0.3, 0.4, 0.5])
    assert np.all(new_ctrl.MCR == [0.6, 0.7, 0.8, 0.9])
    assert new_ctrl.k == 2


def test_load_state_method_mismatch(tmp_path):
    env = DummyEnv()
    jade_ctrl = PopDEController(env, method="jade", c=0.1)
    state_file = tmp_path / "state_mismatch.npz"
    jade_ctrl.save_state(str(state_file))

    shade_ctrl = PopDEController(env, method="shade", c=0.1, history_size=5)
    with pytest.raises(ValueError):
        shade_ctrl.load_state(str(state_file))


# ---------------------- Train and Infer ----------------------


def test_train_and_infer(monkeypatch):
    env = DummyEnv()
    # Ensure stub BaseController is used
    ctrl = PopDEController(
        env, method="jade", c=0.1, max_number_generations_per_episode=1
    )
    history = ctrl.train(episodes=1)
    # Only one step per episode
    assert isinstance(history, list)
    assert len(history) == 1
    entry = history[0]
    # Check required keys
    for key in (
        "episode",
        "step",
        "state",
        "action",
        "hyperparameters",
        "reward",
        "done",
    ):
        assert key in entry

    history_inf = ctrl.infer()
    assert isinstance(history_inf, list)
    assert len(history_inf) == 1
