from typing import Dict, List
from configurations.operations import Operation


class OperationParser:
    @staticmethod
    def parse_operations(operations_config: List[Dict[str, float]]) -> List[Operation]:
        """
        Parse a list of operation configurations and return a list of Operation objects.

        Parameters:
        operations_config (List[Dict[str, float]]):
            List of dictionaries with {'type': <"multiply"/"add"/...>, 'value': <float>}.
        Returns:
        List[Operation]: Operation objects ready to use.
        """
        operations: List[Operation] = []
        for op in operations_config:
            op_type = op.get("type")
            val = op.get("value")
            if op_type is None or val is None:
                raise ValueError("Each operation requires 'type' and 'value'")
            operations.append(Operation(op_type, val))
        return operations
