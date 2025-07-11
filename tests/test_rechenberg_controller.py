import pytest
import numpy as np

from controllers.rechenberg_controller import RechenbergController

# Helper to create dummy algorithm and environment


def make_dummy(env_gen_sequence=None, sigma_vals=None, fitness_vals=None):
    class DummyAlg:
        def __init__(self):
            self.generation = 0
            self.strategy_parameters = (
                sigma_vals if sigma_vals is not None else [0.2, 0.8]
            )
            self.fitness_values = (
                fitness_vals if fitness_vals is not None else [1.0, 2.0]
            )
            self._gen_calls = (
                iter(env_gen_sequence) if env_gen_sequence is not None else iter([True])
            )

        def reset_population(self):
            self.generation = 0

        def run_generation(self, num):
            self.generation += 1
            try:
                return next(self._gen_calls)
            except StopIteration:
                return True

    class DummyEnv:
        num_generations_per_iteration = 1

        def __init__(self, alg):
            self.algorithm = alg

    alg = DummyAlg()
    env = DummyEnv(alg)
    return alg, env


# ---------------------- _inject_sigma ----------------------


def test_inject_sigma_computes_mean_and_preserves_input():
    alg, env = make_dummy(sigma_vals=[0.2, 0.4, 0.6])
    # Instantiate controller with dummy generation-per-iteration
    ctrl = RechenbergController(env, 1)
    # Prepare hyperparams and call
    base_hyp = {"foo": "bar"}
    out = ctrl._inject_sigma(base_hyp)
    # sigma is mean of [0.2,0.4,0.6] = 0.4
    assert out["sigma"] == pytest.approx(0.4)
    # original dict unchanged
    assert "sigma" not in base_hyp


def test_inject_sigma_handles_exception(monkeypatch):
    alg, env = make_dummy(sigma_vals=[0.1, 0.2])
    # Monkey-patch np.mean to raise
    monkeypatch.setattr(np, "mean", lambda x: (_ for _ in ()).throw(ValueError()))
    ctrl = RechenbergController(env, 1)
    out = ctrl._inject_sigma({"a": 1})
    assert np.isnan(out["sigma"])


# ---------------------- train ----------------------


def test_train_records_steps_and_sigma():
    # Sequence: False, False, True -> 3 steps
    alg, env = make_dummy(
        env_gen_sequence=[False, False, True], fitness_vals=[3.0, 1.0]
    )
    ctrl = RechenbergController(env, 1)
    # Override record and state/hyperparameters getters
    ctrl.history = []
    ctrl.get_state = lambda: {"dummy_state": True}
    ctrl.get_hyperparameters = lambda: {"alpha": 0.5}
    ctrl.record_step = lambda **kwargs: ctrl.history.append(kwargs)
    history = ctrl.train(episodes=1)
    # Verify record count and contents
    assert len(history) == 3
    for i, entry in enumerate(history):
        assert entry["episode"] == 0
        assert entry["step"] == i
        assert entry["state"] == {"dummy_state": True}
        assert "sigma" in entry["hyperparameters"]
        # reward is min of fitness values = 1.0
        assert entry["reward"] == pytest.approx(1.0)
        # done flag on last step
        assert entry["done"] == (i == 2)


# ---------------------- infer ----------------------


def test_infer_stops_at_max_steps():
    # Always returns False -> stops by max_steps_per_episode
    alg, env = make_dummy(
        env_gen_sequence=[False, False, False], fitness_vals=[0.5, 0.2]
    )
    ctrl = RechenbergController(env, 1)
    ctrl.max_steps_per_episode = 2
    ctrl.history = []
    ctrl.get_state = lambda: {"s": 1}
    ctrl.get_hyperparameters = lambda: {"b": 2}
    ctrl.record_step = lambda **kwargs: ctrl.history.append(kwargs)
    history = ctrl.infer()
    assert len(history) == 2
    for entry in history:
        assert entry["episode"] == 0
        assert "sigma" in entry["hyperparameters"]
        assert entry["step"] < 2


def test_infer_stops_when_done():
    # Done on first generation
    alg, env = make_dummy(env_gen_sequence=[True], fitness_vals=[0.9, 0.3])
    ctrl = RechenbergController(env, 1)
    ctrl.history = []
    ctrl.get_state = lambda: {"x": 0}
    ctrl.get_hyperparameters = lambda: {}
    ctrl.record_step = lambda **kwargs: ctrl.history.append(kwargs)
    history = ctrl.infer()
    assert len(history) == 1
    assert history[0]["done"] is True
