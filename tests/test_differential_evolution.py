import numpy as np
import pytest

import utils.constants as constants
from benchmarking.optimization_problem import OptimizationProblem
from evolutionary_algorithms.differential_evolution import (
    DifferentialEvolutionOptimizer,
)


def simple_fitness(x: np.ndarray) -> float:
    return float(np.sum(x**2))


@pytest.fixture
def simple_problem():
    bounds = np.array([[-5, 5], [-5, 5]])
    return OptimizationProblem(objective_function=simple_fitness, bounds=bounds)


def create_optimizer(
    problem,
    strategy,
    recombination,
    weights_strategy,
    weights,
    crossover,
    diff_num,
):
    return DifferentialEvolutionOptimizer(
        problem=problem,
        population_size=10,
        differential_weights=weights,
        crossover_probability=crossover,
        base_index_strategy=strategy,
        differential_number=diff_num,
        recombination_strategy=recombination,
        differentials_weights_strategy=weights_strategy,
        max_number_generation=100,
    )


@pytest.fixture
def optimizer_rand_binomial(simple_problem):
    return create_optimizer(
        simple_problem,
        constants.DE_BASE_INDEX_STRATEGY_RAND,
        constants.DE_RECOMBINATION_STRATEGY_BINOMIAL,
        constants.DE_DIFFERENTIAL_WEIGHTS_STRATEGY_GIVEN,
        0.5,
        0.7,
        2,
    )


@pytest.fixture
def optimizer_better_exponential(simple_problem):
    return create_optimizer(
        simple_problem,
        constants.DE_BASE_INDEX_STRATEGY_BETTER,
        constants.DE_RECOMBINATION_STRATEGY_EXPONENTIAL,
        constants.DE_DIFFERENTIAL_WEIGHTS_STRATEGY_DITHER,
        0.5,
        0.8,
        3,
    )


@pytest.fixture
def optimizer_best_jitter(simple_problem):
    return create_optimizer(
        simple_problem,
        constants.DE_BASE_INDEX_STRATEGY_BEST,
        constants.DE_RECOMBINATION_STRATEGY_BINOMIAL,
        constants.DE_DIFFERENTIAL_WEIGHTS_STRATEGY_JITTER,
        0.7,
        0.9,
        2,
    )


@pytest.fixture
def optimizer_target_to_best(simple_problem):
    return create_optimizer(
        simple_problem,
        constants.DE_BASE_INDEX_STRATEGY_TARGET_TO_BEST,
        constants.DE_RECOMBINATION_STRATEGY_EXPONENTIAL,
        constants.DE_DIFFERENTIAL_WEIGHTS_STRATEGY_GIVEN,
        0.6,
        0.6,
        3,
    )


def test_initialize_population(optimizer_rand_binomial):
    pop, fit = (
        optimizer_rand_binomial.population,
        optimizer_rand_binomial.fitness_values,
    )
    assert pop.shape == (optimizer_rand_binomial.population_size, 2)
    assert fit.shape[0] == optimizer_rand_binomial.population_size
    assert np.all(fit >= 0)


def test_mutation_shape(optimizer_best_jitter):
    pop, fit = (
        optimizer_best_jitter.population,
        optimizer_best_jitter.fitness_values,
    )
    best_idx = np.argmin(fit)
    val = optimizer_best_jitter.mutation(pop, fit, best_idx, 0)
    assert val.shape == pop[0].shape


def test_binomial_recombination_valid(optimizer_rand_binomial):
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    rec = optimizer_rand_binomial.recombination_binomial(a, b)
    assert rec.shape == a.shape
    assert np.all((rec == a) | (rec == b))


def test_exponential_recombination_valid(optimizer_better_exponential):
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    rec = optimizer_better_exponential.recombination_exponential(a, b)
    assert rec.shape == a.shape
    assert np.all((rec == a) | (rec == b))


def test_repair_recombination_value(optimizer_best_jitter):
    base = np.array([2.0, 2.0])
    recomb = np.array([6.0, -6.0])
    repaired = optimizer_best_jitter.repair_recombination_value(base, recomb)
    lower, upper = np.array(optimizer_best_jitter.bounds).T
    assert np.all(repaired >= lower)
    assert np.all(repaired <= upper)


def test_select_survivors_shape(optimizer_target_to_best):
    pop, fit = (
        optimizer_target_to_best.population,
        optimizer_target_to_best.fitness_values,
    )
    children = np.random.uniform(-5, 5, pop.shape)
    children_fit = np.array([simple_fitness(x) for x in children])
    new_pop, new_fit = optimizer_target_to_best.select_survivors(
        pop, fit, children, children_fit
    )
    assert new_pop.shape == pop.shape
    assert new_fit.shape == fit.shape


def test_generate_differential_value_shape(optimizer_best_jitter):
    pop = optimizer_best_jitter.population
    base_idx = 0
    d1, d2 = optimizer_best_jitter.select_differentials(pop, base_idx)
    diff = optimizer_best_jitter.generate_differential_value(d1, d2)
    assert diff.shape == (2,)  # dimension of problem


def test_update_children_values_correct(optimizer_rand_binomial):
    pop = optimizer_rand_binomial.population
    children = np.zeros_like(pop)
    children_fit = np.full(pop.shape[0], np.inf)
    recomb = pop[0] * 0.5
    optimizer_rand_binomial.update_children_values(children, children_fit, 0, recomb)
    assert np.allclose(children[0], recomb)
    assert children_fit[0] == simple_fitness(recomb)


def test_run_generation_behavior(optimizer_rand_binomial):
    old_gen = optimizer_rand_binomial.generation
    success = optimizer_rand_binomial.run_generation(1)
    assert isinstance(success, bool)
    assert optimizer_rand_binomial.generation == old_gen + 1
