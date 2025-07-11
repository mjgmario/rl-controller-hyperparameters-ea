import pytest

from configurations.operations import Operation


@pytest.mark.parametrize(
    "op_type, value, x, expected",
    [
        ("add", 5.0, 10.0, 15.0),
        ("subtract", 3.0, 10.0, 7.0),
        ("multiply", 2.0, 4.0, 8.0),
        ("divide", 2.0, 10.0, 5.0),
    ],
)
def test_operation_apply_valid(op_type, value, x, expected):
    op = Operation(op_type, value)
    result = op.apply(x)
    assert result == expected


def test_operation_divide_by_zero_returns_input():
    op = Operation("divide", 0.0)
    result = op.apply(10.0)
    assert result == 10.0


def test_operation_unsupported_type_raises():
    op = Operation("modulus", 3.0)
    with pytest.raises(ValueError, match="Unsupported operation type: modulus"):
        op.apply(10.0)
