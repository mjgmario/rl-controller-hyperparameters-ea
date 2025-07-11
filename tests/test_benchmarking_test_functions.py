import numpy as np
import pytest

from benchmarking.functions import _basic_configs, _bbob_2013_configs, create_problems
from benchmarking.optimization_problem import OptimizationProblem


@pytest.mark.parametrize("name", list(_basic_configs.keys()))
def test_basic_problem_evaluation_at_optimum(name):
    config = _basic_configs[name]
    dim = config["default_dim"]
    func = config["func"]
    x_opt = config["opt_point"](dim)
    expected = config["opt"]
    value = func(x_opt)

    assert isinstance(value, float), f"{name} did not return a float"

    if name == "schwefel":
        tolerance = 1e-1
    else:
        tolerance = 1e-6

    assert np.isclose(
        value, expected, atol=tolerance
    ), f"{name} did not evaluate to expected optimal value (expected={expected}, got={value})"


@pytest.mark.parametrize("name", list(_bbob_2013_configs.keys()))
def test_bbob_problem_evaluation(name):
    config = _bbob_2013_configs[name]
    func = config["func"]
    x_opt = config["x_opt"]
    f_opt = config["opt"]
    value = func(x_opt)
    assert isinstance(value, float)
    assert np.isclose(value, f_opt, atol=1e-4), f"{name} did not evaluate to f_opt"


@pytest.mark.parametrize(
    "name", list(_basic_configs.keys()) + list(_bbob_2013_configs.keys())
)
def test_create_problems_structure(name):
    problems = create_problems([name])
    assert name in problems
    prob = problems[name]
    assert isinstance(prob, OptimizationProblem)
    assert prob.bounds.shape == (prob.bounds.shape[0], 2)
    assert prob.objective_function is not None


def test_override_dims_and_bounds():
    name = "sphere"
    custom_dim = 5
    custom_bounds = (-1.0, 1.0)
    problems = create_problems(
        [name],
        override_dims={name: custom_dim},
        override_bounds={name: custom_bounds},
    )
    prob = problems[name]
    assert prob.bounds.shape == (custom_dim, 2)
    assert np.all(prob.bounds[:, 0] == custom_bounds[0])
    assert np.all(prob.bounds[:, 1] == custom_bounds[1])
