class Operation:
    """
    Represents an arithmetic operation used to adjust a hyperparameter.
    """

    def __init__(self, operation_type: str, value: float) -> None:
        """
        Initializes the operation.

        Args:
            operation_type (str): Type of operation. Must be one of:
                'multiply', 'divide', 'add', or 'subtract'.
            value (float): The value used in the operation.
        """
        self.operation_type = operation_type
        self.value = value

    def apply(self, x: float) -> float:
        """
        Applies the operation to the input value x.

        Args:
            x (float): The current value of the hyperparameter.

        Returns:
            float: Result after applying the operation.
        """
        if self.operation_type == "multiply":
            return x * self.value
        elif self.operation_type == "divide":
            return x / self.value if self.value != 0 else x
        elif self.operation_type == "add":
            return x + self.value
        elif self.operation_type == "subtract":
            return x - self.value
        else:
            raise ValueError(f"Unsupported operation type: {self.operation_type}")
