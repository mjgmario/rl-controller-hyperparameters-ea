import numpy as np


def format_strategy_parameters(value, template: np.ndarray) -> np.ndarray:
    """
    Normalize and broadcast a strategy parameter value to match a template array.

    This function ensures the returned array has the same shape as `template`:
      1. If `value` is a scalar (size 1), returns an array filled with that scalar.
      2. If `value` has the same number of elements as `template`, reshapes it to `template.shape`.
      3. Otherwise, fills an array with the first element of `value`.

    Args:
        value: A scalar or iterable of floats/ints representing strategy parameters.
        template (np.ndarray): Array whose shape is used for output formatting.

    Returns:
        np.ndarray: An array of the same shape as `template` with formatted strategy parameters.
    """
    arr = np.asarray(value, dtype=float)
    if arr.size == 1:
        return np.full_like(template, float(arr))
    if arr.size == template.size:
        return arr.reshape(template.shape)
    return np.full_like(template, float(arr.flat[0]))


def format_differential_weights(value, differential_number):
    """
    Convert a differential weight specification into a 1-D array of length `differential_number`.

    Behavior:
      - If `value` is a float or int, returns an array filled with that value.
      - If `value` is an iterable of length exactly `differential_number`,
        returns it as a NumPy array.
      - Otherwise, fills the output array with the first element of `value`.

    Args:
        value: A float, int, or iterable of floats/ints specifying weights.
        differential_number (int): Desired length of the output array.

    Returns:
        np.ndarray: 1-D array of length `differential_number` with differential weights.
    """
    if isinstance(value, (float, int)):
        return np.full(differential_number, float(value))
