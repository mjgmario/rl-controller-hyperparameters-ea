import pytest

from configurations.operation_parser import OperationParser
from configurations.operations import Operation


def test_parse_operations_valid():
    config = [
        {"type": "add", "value": 5.0},
        {"type": "multiply", "value": 2.0},
    ]
    operations = OperationParser.parse_operations(config)
    assert len(operations) == 2
    assert isinstance(operations[0], Operation)
    assert operations[0].value == 5.0
    assert operations[1].value == 2.0


def test_parse_operations_missing_type():
    config = [{"value": 3.0}]
    with pytest.raises(ValueError, match="Each operation requires 'type' and 'value'"):
        OperationParser.parse_operations(config)


def test_parse_operations_missing_value():
    config = [{"type": "add"}]
    with pytest.raises(ValueError, match="Each operation requires 'type' and 'value'"):
        OperationParser.parse_operations(config)


def test_parse_operations_empty_list():
    config = []
    operations = OperationParser.parse_operations(config)
    assert operations == []
