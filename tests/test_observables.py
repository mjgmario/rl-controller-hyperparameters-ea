import numpy as np
import pytest

from environment import observable as observables


@pytest.fixture
def simple_inputs():
    pop = np.array([[0, 0], [1, 1], [2, 2]])
    fitness = np.array([3.0, 2.0, 1.0])
    best_history = [3.0, 2.5, 2.0, 1.5]
    bounds = [(0.0, 2.0), (0.0, 2.0)]
    generation = 2
    max_gen = 10
    return pop, fitness, best_history, bounds, generation, max_gen


def test_generation_log(simple_inputs):
    pop, fitness, hist, bounds, gen, max_gen = simple_inputs
    o = observables.GenerationLog()
    result = o.compute(pop, fitness, hist, bounds, gen, max_gen)
    assert 0 <= result <= 1
    assert result == pytest.approx(np.log(3) / np.log(max_gen + 1))


def test_average_pop_fitness(simple_inputs):
    pop, fitness, hist, bounds, gen, max_gen = simple_inputs
    o = observables.AveragePopFitness()
    val1 = o.compute(pop, fitness, hist, bounds, gen, max_gen)
    val2 = o.compute(pop, fitness, hist, bounds, gen + 1, max_gen)
    assert val1 == pytest.approx(1.0, rel=1e-6)
    assert val2 == pytest.approx(1.0, rel=1e-6)


def test_entropy_normalized(simple_inputs):
    pop, fitness, hist, bounds, gen, max_gen = simple_inputs
    o = observables.Entropy()
    result = o.compute(pop, fitness, hist, bounds, gen, max_gen)
    assert 0 <= result <= 1


def test_std_fitness(simple_inputs):
    pop, fitness, hist, bounds, gen, max_gen = simple_inputs
    o = observables.StdFitness()
    result = o.compute(pop, fitness, hist, bounds, gen, max_gen)
    assert isinstance(result, float)
    assert 0 <= result <= 1


def test_fitness_improvement(simple_inputs):
    pop, fitness, hist, bounds, gen, max_gen = simple_inputs
    o = observables.FitnessImprovement(window=2)
    result = o.compute(pop, fitness, hist, bounds, gen, max_gen)
    assert result > 0


def test_stagnation(simple_inputs):
    pop, fitness, hist, bounds, gen, max_gen = simple_inputs
    stagnating_hist = [3.0, 3.0, 3.0]
    o = observables.Stagnation(max_stag=5)
    result = o.compute(pop, fitness, stagnating_hist, bounds, gen, max_gen)
    assert result == pytest.approx(2 / 5)


def test_genotypic_diversity(simple_inputs):
    pop, fitness, hist, bounds, gen, max_gen = simple_inputs
    o = observables.GenotypicDiversity()
    result = o.compute(pop, fitness, hist, bounds, gen, max_gen)
    assert 0 <= result <= 1


def test_best_fitness_norm(simple_inputs):
    pop, fitness, hist, bounds, gen, max_gen = simple_inputs
    o = observables.BestFitnessNorm()
    val1 = o.compute(pop, fitness, hist, bounds, gen, max_gen)
    assert val1 == 1.0
    val2 = o.compute(pop, np.array([0.5, 0.5, 0.5]), hist, bounds, gen + 1, max_gen)
    assert val2 < 1.0


def test_avg_distance_from_best(simple_inputs):
    pop, fitness, hist, bounds, gen, max_gen = simple_inputs
    o = observables.AvgDistanceFromBest()
    result = o.compute(pop, fitness, hist, bounds, gen, max_gen)
    assert 0 <= result <= 1


def test_discretize_behavior():
    class Dummy(observables.Observable):
        name = "dummy"
        min_val = 0.0
        max_val = 1.0
        bins = 5

        def compute(*args, **kwargs):
            return 0.0

    d = Dummy()
    val = d.discretize(0.63)
    assert isinstance(val, int)
    assert 0 <= val <= 4
