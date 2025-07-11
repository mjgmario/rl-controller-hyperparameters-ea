import numpy as np
import pytest

from environment import rewards as rf


def test_parent_offspring_reward_improves():
    pre = np.array([5.0, 4.0, 6.0])
    post = np.array([3.0, 2.0, 4.0])
    reward = rf.parent_offspring_reward(pre, post)
    assert reward > 0


def test_parent_offspring_reward_no_improvement():
    pre = np.array([1.0, 1.0])
    post = np.array([1.0, 1.0])
    reward = rf.parent_offspring_reward(pre, post)
    assert reward == 0.0


def test_population_reward_mean_mode():
    pre = np.array([5.0, 5.0, 5.0])
    post = np.array([4.0, 3.0, 5.0])
    reward = rf.population_reward(pre, post, omega=0.5, mode="mean")
    assert reward > 0


def test_population_reward_max_mode():
    pre = np.array([4.0, 4.0])
    post = np.array([3.5, 4.0])
    reward = rf.population_reward(pre, post, omega=0.5, mode="max")
    assert reward > 0


def test_population_reward_no_improvement():
    pre = np.array([2.0, 2.0])
    post = np.array([2.0, 2.0])
    reward = rf.population_reward(pre, post)
    assert reward == 0.0


def test_binary_reward_improved():
    hist = [5.0, 4.0]
    assert rf.binary_reward(hist) == 1.0


def test_binary_reward_not_improved():
    hist = [4.0, 4.5]
    assert rf.binary_reward(hist) == 0.0


def test_binary_reward_insufficient_history():
    hist = [1.0]
    assert rf.binary_reward(hist) == 0.0


def test_weighted_improvement_reward_basic():
    deltas = [0.1, 0.2, 0.3, 0.4]
    result = rf.weighted_improvement_reward(deltas, ref_window=2)
    expected = 0.4 - np.mean([0.3, 0.4])
    assert result == pytest.approx(expected)


def test_weighted_improvement_reward_empty():
    assert rf.weighted_improvement_reward([]) == 0.0


def test_weighted_improvement_reward_all_zero():
    result = rf.weighted_improvement_reward([0.0, 0.0, 0.0])
    assert result == 0.0


def test_combined_increment_with_binary():
    pre = np.array([5.0, 6.0])
    post = np.array([4.0, 5.0])
    hist = [6.0, 5.5]
    result = rf.combined_increment_with_binary(pre, post, hist)
    assert result > 0.0
