import numpy as np

from benchmarking.optimization_problem import OptimizationProblem


def dummy_func(x):
    return np.sum(x**2)


def test_minimization_evaluation():
    prob = OptimizationProblem(
        objective_function=dummy_func, bounds=np.array([[0, 1]] * 3)
    )
    x = np.array([1.0, 0.0, -1.0])
    result = prob.evaluate(x)
    assert np.isclose(result, 2.0)


def test_maximization_evaluation():
    prob = OptimizationProblem(
        objective_function=dummy_func,
        bounds=np.array([[0, 1]] * 3),
        maximize=True,
    )
    x = np.array([1.0, 0.0, -1.0])
    result = prob.evaluate(x)
    assert np.isclose(result, -2.0)
