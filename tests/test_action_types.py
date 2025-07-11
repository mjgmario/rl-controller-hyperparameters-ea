from unittest.mock import MagicMock

import pytest

from environment.action_types import (
    ActionType,
    CategoricalAction,
    ContinuousAction,
    DirectSelectionAction,
    OperationBasedAction,
)

# ---------- Tests for ActionType (base) ----------


def test_actiontype_raises_not_implemented():
    base = ActionType([0, 1])
    with pytest.raises(NotImplementedError):
        base.sample_value(0)


# ---------- Tests for DirectSelectionAction ----------


def test_direct_selection_creates_correct_subintervals():
    action = DirectSelectionAction([0.0, 1.0], epsilon=0.2)
    subintervals = action.subintervals
    assert len(subintervals) == 5
    assert subintervals[0] == (0.0, 0.2)
    assert subintervals[-1][1] == pytest.approx(1.0)


def test_direct_selection_samples_within_range():
    action = DirectSelectionAction([0.0, 1.0], epsilon=0.1)
    value = action.sample_value(3)
    low, high = action.subintervals[3]
    assert low <= value <= high


# ---------- Tests for OperationBasedAction ----------


def test_operation_based_applies_and_clips():
    op_mock = MagicMock()
    op_mock.apply.return_value = 1.2
    action = OperationBasedAction([0.0, 1.0], operations=[op_mock])
    result = action.sample_value(0, current_value=0.5)
    assert result == 1.0  # Clipped to upper bound


# ---------- Tests for CategoricalAction ----------


def test_categorical_action_returns_correct_value():
    values = ["adam", "sgd", "rmsprop"]
    action = CategoricalAction(values)
    assert action.sample_value(1) == "sgd"
    assert action.sample_value(0) == "adam"


# ---------- Tests for ContinuousAction ----------


def test_continuous_action_clips_properly():
    action = ContinuousAction([0.0, 1.0])
    assert action.sample_value(0.5) == 0.5
    assert action.sample_value(-1.0) == 0.0
    assert action.sample_value(2.0) == 1.0
