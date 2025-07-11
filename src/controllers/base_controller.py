import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np


class BaseController:
    """
    General controller base class. Provides:
      - `env` assignment and `history` list
      - common methods: `get_state()`, `get_hyperparameters()`, `record_step(...)`, `save_history()`, `set_environment()`
    Subclasses implement `train(...)` and `infer(...)`, using these shared utilities.
    """

    def __init__(self, environment: Any) -> None:
        """
        Args:
            environment: The environment instance (e.g., RLEnvironment).
        """
        self.env = environment
        self.history: List[Dict[str, Any]] = []

    def get_state(self) -> Optional[Dict[str, Any]]:
        """
        Return the current observable state from the environment, if available.
        """
        if hasattr(self.env, "get_state"):
            return self.env.get_state()
        return None

    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Capture current values of all controllable hyperparameters from self.env.algorithm.
        Converts numpy arrays to float or list of floats, single-element lists to float, etc.
        Returns an empty dict if no action_space or algorithm is present.
        """
        snapshot: Dict[str, Any] = {}
        if not hasattr(self.env, "action_space") or not hasattr(self.env, "algorithm"):
            return snapshot

        for param in self.env.action_space.actions:
            val = getattr(self.env.algorithm, param, None)
            if val is None:
                snapshot[param] = None
            if isinstance(val, np.ndarray):
                flat = val.astype(float).ravel()
                if flat.size == 1:
                    snapshot[param] = float(flat[0])
                elif np.all(flat == flat[0]):
                    snapshot[param] = float(flat[0])
                else:
                    snapshot[param] = flat.tolist()
            elif isinstance(val, (list, tuple)):
                converted: List[Any] = []
                for elem in val:
                    try:
                        converted.append(float(elem))
                    except Exception:
                        converted.append(elem)
                if len(converted) == 1:
                    snapshot[param] = converted[0]
                elif all(elem == converted[0] for elem in converted):
                    snapshot[param] = converted[0]
                else:
                    snapshot[param] = converted
            elif isinstance(val, (int, float)):
                snapshot[param] = float(val)
            else:
                snapshot[param] = val
        return snapshot

    def record_step(self, **kwargs) -> None:
        """
        Append a dictionary of step-related data into self.history.
        Example kwargs keys: 'episode', 'step', 'state', 'action', 'hyperparameters', 'reward', 'done'
        """
        self.history.append(kwargs)

    def save_history(self, save_path: str) -> None:
        """
        Serialize `self.history` to a pickle file at `save_path`.

        Args:
            save_path (str): Full file path (including filename) where history is saved.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(self.history, f)

    def set_environment(self, new_env: Any) -> None:
        """
        Default environment swap: reassigns self.env. Subclasses may override for extra logic.
        """
        self.env = new_env

    def train(self, *args, **kwargs) -> List[Any]:
        raise NotImplementedError("Subclasses must implement train().")

    def infer(self, *args, **kwargs) -> List[Any]:
        raise NotImplementedError("Subclasses must implement infer().")
