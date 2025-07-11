import pytest

from environment.observable import Entropy, FitnessImprovement
from environment.observable_factory import ObservableFactory

# ----------- Tests -----------


def test_build_from_name_basic():
    result = ObservableFactory.build(["entropy"])
    assert len(result) == 1
    assert isinstance(result[0], Entropy)
    assert result[0].name == "entropy"


def test_build_with_constructor_args():
    result = ObservableFactory.build(
        ["fitness_improvement"], config={"fitness_improvement": {"window": 3}}
    )
    assert isinstance(result[0], FitnessImprovement)
    assert result[0].window == 3


def test_build_with_attr_override():
    result = ObservableFactory.build(["entropy"], config={"entropy": {"bins": 20}})
    entropy = result[0]
    assert entropy.bins == 20


def test_build_from_existing_instance():
    inst = Entropy()
    result = ObservableFactory.build([inst])
    assert result[0] is inst


def test_raises_on_unknown_name():
    with pytest.raises(ValueError, match="Unknown observable 'unknown_metric'"):
        ObservableFactory.build(["unknown_metric"])


def test_raises_on_invalid_type():
    with pytest.raises(ValueError, match="must be name or Observable"):
        ObservableFactory.build([123])
