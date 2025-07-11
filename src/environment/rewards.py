from typing import List

import numpy as np

__all__ = [
    "parent_offspring_reward",
    "population_reward",
    "binary_reward",
    "weighted_improvement_reward",
    "combined_increment_with_binary",
]


def combined_increment_with_binary(
    pre: np.ndarray, post: np.ndarray, best_history, time_scale: float = 1.0
) -> float:
    """
    Compute a combined reward by summing a continuous parent-offspring improvement
    and a binary success indicator based on historical best performance.

    This function adds:
      1. The incremental reward from parent to offspring, scaled by `time_scale`.
      2. A binary reward indicating whether the current state represents a new best.

    Args:
        pre (np.ndarray): Parent population (or fitness) before mutation/recombination.
        post (np.ndarray): Offspring population (or fitness) after mutation/recombination.
        best_history: Data structure tracking best-ever performance; used by `binary_reward`.
        time_scale (float): Scaling factor applied to the continuous parent-offspring reward.

    Returns:
        float: Sum of the parent-offspring incremental reward and the binary best-history reward.
    """
    return parent_offspring_reward(pre, post, time_scale) + binary_reward(best_history)


def parent_offspring_reward(
    pre: np.ndarray, post: np.ndarray, time_scale: float = 1.0
) -> float:
    """
    Reward based on the improvement of the best individual (minimization).

    Args:
        pre (np.ndarray): Fitness values before evolution.
        post (np.ndarray): Fitness values after evolution.
        time_scale (float): Scaling factor for normalization.

    Returns:
        float: Normalized improvement of best fitness.
    """
    min_pre, min_post = float(np.min(pre)), float(np.min(post))
    return (min_pre - min_post) / (abs(min_pre) * time_scale + 1e-12)


def population_reward(
    pre: np.ndarray, post: np.ndarray, omega: float = 0.5, mode: str = "mean"
) -> float:
    """
    Reward based on overall population improvement.

    Args:
        pre (np.ndarray): Fitness values before evolution.
        post (np.ndarray): Fitness values after evolution.
        omega (float): Weighting factor between max and average reward.
        mode (str): 'mean' to average improvements, 'max' to take the best.

    Returns:
        float: Combined reward from population-wide improvements.
    """
    diffs = pre - post  # improvement: positive if pre > post
    diffs = np.where(diffs > 0, diffs, 0.0)  # only positive improvements
    r_m = float(np.max(diffs))
    r_pop = float(np.mean(diffs)) if mode == "mean" else r_m
    return (1.0 - omega) * r_m + omega * r_pop


def binary_reward(best_history: List[float]) -> float:
    """
    Binary reward indicating whether best fitness improved.

    Args:
        best_history (List[float]): History of best fitness values.

    Returns:
        float: 1.0 if last value improved over the previous, else 0.0.
    """
    if len(best_history) < 2:
        return 0.0
    return 1.0 if best_history[-1] < best_history[-2] else 0.0


def weighted_improvement_reward(
    delta_history: List[float], ref_window: int = 5
) -> float:
    """
    Weighted improvement reward based on recent performance history.

    Computes:
        W(Δf) = Δf - Ref(N), where:
        - Δf is the most recent improvement value.
        - Ref(N) is the average of the last N non-zero improvement values.

    Args:
        delta_history (List[float]): List of past normalized Δf values.
        ref_window (int): Number of recent non-zero deltas to use for reference.

    Returns:
        float: Difference between the current delta and its reference value.
    """
    if not delta_history:
        return 0.0

    curr_delta = delta_history[-1]
    nonzero_deltas = [d for d in delta_history if abs(d) > 0.0]

    if not nonzero_deltas:
        ref = 0.0
    else:
        ref = float(np.mean(nonzero_deltas[-ref_window:]))

    return curr_delta - ref
