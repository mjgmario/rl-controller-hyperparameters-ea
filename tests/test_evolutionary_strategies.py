import pytest
import numpy as np

from benchmarking.optimization_problem import OptimizationProblem
from evolutionary_algorithms.evolutionary_strategies import EvolutionaryStrategies
from utils import constants


# Dummy problem for testing
def make_problem(bounds, func, success_threshold):
    class DummyProblem(OptimizationProblem):
        def __init__(self):
            self.bounds = bounds

        def evaluate(self, individual):
            return func(individual)

        def is_success(self, value):
            return value <= success_threshold

    return DummyProblem()


# ---------------------- Initialization ----------------------


def test_initialization_sets_attributes_and_population_shape():
    bounds = [(0, 1), (2, 3)]
    prob = make_problem(bounds, lambda x: 0.5, success_threshold=0.1)
    es = EvolutionaryStrategies(
        problem=prob,
        phro=2,
        epsilon0=0.01,
        tau=0.5,
        tau_prime=0.1,
        recombination_individuals_strategy=constants.ES_RECOMBINATION_STRATEGY_DISCRETE,
        recombination_strategy_strategy=constants.ES_RECOMBINATION_STRATEGY_INTERMEDIATE_AVERAGED,
        mutation_steps=1,
        survivor_strategy=constants.ES_SURVIVOR_STRATEGY_MU_LAMBDA,
        mu=10,
        lamda=20,
        max_number_generation=5,
        use_rechenberg=False,
    )
    # population shape
    assert es.population.shape == (10, 2)
    # strategy parameters shape
    assert es.strategy_parameters.shape == (10, 1)
    # fitness values length and num_function_calls
    assert len(es.fitness_values) == 10
    assert es.num_function_calls == 10
    assert es.generation == 0


# ---------------------- safe_fitness ----------------------

# ---------------------- recombine ----------------------


def test_recombine_invalid_strategy_raises():
    prob = make_problem([(0, 1)], lambda x: 0, 0)
    es = EvolutionaryStrategies(
        problem=prob,
        phro=1,
        epsilon0=0.1,
        tau=0.1,
        tau_prime=0.1,
        recombination_individuals_strategy="",
        recombination_strategy_strategy="",
        mutation_steps=1,
        survivor_strategy=constants.ES_SURVIVOR_STRATEGY_MU_LAMBDA,
    )
    with pytest.raises(ValueError):
        es.recombine(np.array([[1]]), "invalid")


# ---------------------- mutation_uncorrelated_one_step_size ----------------------


def test_mutation_uncorrelated_one_step_size_min_epsilon(monkeypatch):
    prob = make_problem([(0, 1)], lambda x: 0, 0)
    es = EvolutionaryStrategies(
        problem=prob,
        phro=1,
        epsilon0=0.5,
        tau=0.0,
        tau_prime=0.1,
        recombination_individuals_strategy="",
        recombination_strategy_strategy="",
        mutation_steps=1,
        survivor_strategy=constants.ES_SURVIVOR_STRATEGY_MU_LAMBDA,
    )
    # patch normal to return negative to force below epsilon
    monkeypatch.setattr(np.random, "normal", lambda *args, **kwargs: np.array([-10.0]))
    out = es.mutation_uncorrelated_one_step_size(np.array([0.4]), es.tau, es.epsilon0)
    # should be at least epsilon0
    assert out[0] == pytest.approx(0.5)


# ---------------------- select_survivors mu+lambda ----------------------


def test_select_survivors_mu_plus_lambda():
    prob = make_problem([(0, 1)], lambda x: x[0], success_threshold=0)
    es = EvolutionaryStrategies(
        problem=prob,
        phro=1,
        epsilon0=0.1,
        tau=0.1,
        tau_prime=0.1,
        recombination_individuals_strategy="",
        recombination_strategy_strategy="",
        mutation_steps=1,
        survivor_strategy=constants.ES_SURVIVOR_STRATEGY_MU_PLUS_LAMBDA,
        mu=2,
        lamda=3,
    )
    pop = np.array([[0.2], [0.8]])
    strat = np.array([[0.1], [0.2]])
    fit = np.array([0.2, 0.8])
    children = np.array([[0.5], [0.1], [0.9]])
    child_strat = np.array([[0.3], [0.4], [0.5]])
    new_pop, new_strat, new_fit, min_val, children_fit = es.select_survivors(
        pop,
        fit,
        children,
        strat,
        child_strat,
        constants.ES_SURVIVOR_STRATEGY_MU_PLUS_LAMBDA,
    )
    # mu=2 smallest values: 0.1 and 0.2
    assert new_fit.tolist() == [0.1, 0.2]
    assert min_val == 0.1
    assert children_fit.shape == (3,)


# ---------------------- update_parameters_end_iteration ----------------------


def test_update_parameters_end_iteration_and_best():
    # problem success threshold 0.5
    prob = make_problem([(0, 1)], lambda x: 0.3, success_threshold=0.5)
    es = EvolutionaryStrategies(
        problem=prob,
        phro=1,
        epsilon0=0.1,
        tau=0.1,
        tau_prime=0.1,
        recombination_individuals_strategy="",
        recombination_strategy_strategy="",
        mutation_steps=1,
        survivor_strategy=constants.ES_SURVIVOR_STRATEGY_MU_LAMBDA,
    )
    pop = np.array([[0.3], [0.7]])
    # first call should track best and return True (is_success)
    success = es.update_parameters_end_iteration(0, pop, min_child_value=0.3)
    assert success is True
    assert es.best_individual_value == 0.3
    assert es.best_individual_execution.tolist() == [0.3]
    assert es.best_individuals[-1][1] == pytest.approx(0.3)
