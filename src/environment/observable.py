import math
from itertools import combinations
from typing import List, Tuple

import numpy as np

__all__ = [
    "GenerationLog",
    "AveragePopFitness",
    "Entropy",
    "StdFitness",
    "FitnessImprovement",
    "Stagnation",
    "GenotypicDiversity",
    "BestFitnessNorm",
    "AvgDistanceFromBest",
]


class Observable:
    """
    Base class for observable metrics. Subclasses must implement compute(...).

    Configurable attributes (via YAML or constructor):
      - name: Unique identifier (string)
      - min_val, max_val: Range for clamping/discretization
      - bins: Number of bins for discretization (optional)
    """

    name: str
    min_val: float
    max_val: float
    bins: int = None

    def compute(
        self,
        pop: np.ndarray,
        fitness: np.ndarray,
        best_history: List[float],
        bounds: List[Tuple[float, float]],
        generation: int,
        max_gen: int,
    ) -> float:
        raise NotImplementedError

    def discretize(self, value: float) -> float:
        """
        Clamp the value to [min_val, max_val] and discretize if bins are defined.
        """
        v = min(max(value, self.min_val), self.max_val)
        if self.bins is None:
            return v
        edges = np.linspace(self.min_val, self.max_val, self.bins + 1)
        idx = int(np.digitize(v, edges) - 1)
        return max(0, min(idx, self.bins - 1))


class GenerationLog(Observable):
    """
    Log-scaled generation number:
      compute = log(generation + 1) / log(max_gen + 1)
    """

    name = "generation_log"
    min_val = 0.0
    max_val = 1.0
    bins = None

    def compute(self, pop, fitness, best_history, bounds, generation, max_gen):
        return math.log(generation + 1, max_gen + 1)


class AveragePopFitness(Observable):
    """
    Average fitness ratio:
      f̄^t = (sum of current fitness) / (initial sum of fitness)
    Lower is better in minimization: f̄ ≤ 1.
    """

    name = "average_pop_fitness"
    min_val = 0.0
    max_val = 1.0
    bins = None

    def __init__(self):
        super().__init__()
        self.initial_sum = None

    def compute(self, pop, fitness, best_history, bounds, generation, max_gen):
        total = float(np.sum(fitness))
        if self.initial_sum is None:
            self.initial_sum = total + 1e-12
            return 1.0
        return total / self.initial_sum


class Entropy(Observable):
    """
    Entropy of the fitness distribution:
      p_m = f(x_m) / sum(f(x_m))
      H = -∑ p_m * log2(p_m) / log2(M)
    Guaranteed to be within [0, 1].
    """

    name = "entropy"
    min_val = 0.0
    max_val = 1.0
    bins = None

    def compute(self, pop, fitness, best_history, bounds, generation, max_gen):
        p = fitness.astype(float)
        total = p.sum()
        if total <= 0:
            return 0.0
        q = p / total
        q = q[q > 0]
        H = -(q * np.log2(q)).sum()
        return H / math.log2(len(fitness))


class StdFitness(Observable):
    """
    Normalized fitness standard deviation:
      σ_f^norm = σ_f / (f_max - f_min + ε)
    """

    name = "std_fitness"
    min_val = 0.0
    max_val = 1.0
    bins = None

    def compute(self, pop, fitness, best_history, bounds, generation, max_gen):
        sigma = float(np.std(fitness))
        f_max = float(np.max(fitness))
        f_min = float(np.min(fitness))
        denom = (f_max - f_min) + 1e-12
        return sigma / denom


class FitnessImprovement(Observable):
    """
    Normalized fitness improvement over a window:
      Δf^norm = (f_best(t-k) - f_best(t)) / (|f_best(t-k)| + ε)
    Positive value indicates improvement (in minimization).
    """

    name = "fitness_improvement"
    min_val = -1.0
    max_val = 1.0
    bins = None

    def __init__(self, window: int = 1):
        super().__init__()
        self.window = window

    def compute(self, pop, fitness, best_history, bounds, generation, max_gen):
        k = min(self.window, len(best_history) - 1)
        if k <= 0:
            return 0.0
        prev = best_history[-1 - k]
        curr = best_history[-1]
        delta = prev - curr
        return delta / (abs(prev) + 1e-12)


class Stagnation(Observable):
    """
    Stagnation counter:
      Number of consecutive generations without improvement in best fitness.
      S^norm = min(S / max_stag, 1.0)
    """

    name = "stagnation"
    min_val = 0.0
    max_val = 1.0
    bins = None

    def __init__(self, max_stag: int = 50):
        super().__init__()
        self.max_stag = max_stag

    def compute(self, pop, fitness, best_history, bounds, generation, max_gen):
        count = 0
        N = len(best_history)
        for i in range(1, N):
            curr = best_history[-i]
            prev = best_history[-i - 1]
            if curr >= prev:  # no improvement
                count += 1
            else:
                break
        return min(count / self.max_stag, 1.0)


class GenotypicDiversity(Observable):
    """
    Genotypic diversity:
      Average pairwise distance between individuals, normalized by max theoretical distance.
    """

    name = "genotypic_diversity"
    min_val = 0.0
    max_val = 1.0
    bins = None

    def compute(self, pop, fitness, best_history, bounds, generation, max_gen):
        n, d = pop.shape
        if n < 2:
            return 0.0
        ranges = np.array([b[1] - b[0] for b in bounds])
        max_dist = np.linalg.norm(ranges) * math.sqrt(d)
        dists = [np.linalg.norm(pop[i] - pop[j]) for i, j in combinations(range(n), 2)]
        return float(np.mean(dists)) / (max_dist + 1e-12)


class BestFitnessNorm(Observable):
    """
    Normalized best fitness:
      f_best^norm = (f_best(0) - f_best(t)) / (f_best(0) + ε)
    Starts at 1.0 and decreases as fitness improves (minimization).
    """

    name = "best_fitness_norm"
    min_val = 0.0
    max_val = 1.0
    bins = None

    def __init__(self):
        super().__init__()
        self.initial_best = None

    def compute(self, pop, fitness, best_history, bounds, generation, max_gen):
        curr = float(np.min(fitness))
        if self.initial_best is None:
            self.initial_best = curr + 1e-12
            return 1.0
        return (self.initial_best - curr) / (self.initial_best + 1e-12)


class AvgDistanceFromBest(Observable):
    """
    Average distance from the best individual:
      D_avg = (1/M) ∑ ||x_m - x_best||, normalized by maximum theoretical distance.
    """

    name = "avg_distance_from_best"
    min_val = 0.0
    max_val = 1.0
    bins = None

    def compute(self, pop, fitness, best_history, bounds, generation, max_gen):
        best_idx = int(np.argmin(fitness))
        best_ind = pop[best_idx]
        dists = np.linalg.norm(pop - best_ind, axis=1)
        mean_dist = float(np.mean(dists))
        ranges = np.array([b[1] - b[0] for b in bounds])
        max_dist = np.linalg.norm(ranges) * math.sqrt(pop.shape[1])
        return mean_dist / (max_dist + 1e-12)
