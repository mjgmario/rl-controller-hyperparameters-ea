import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import os

from controllers.base_controller import BaseController
from utils import constants

logger = logging.getLogger(__name__)


class PopDEController(BaseController):
    """
    Population‑based controller for self‑adaptive Differential Evolution.

    Supports two adaptive DE variants:

    * **JADE** – Global DE with optional external archive (Zhang & Sanderson, 2009)
    * **SHADE** – Success‑History‑based Adaptive DE (Tanabe & Fukunaga, 2013)

    ----------------
    Implementation highlights
    ----------------
    • Collects success tuples (F, CR, Δf) per generation and updates memories
      once at each generation boundary.
    • Uses the Lehmer mean for *F* and arithmetic mean for *CR*.
    • In SHADE, means are **weighted** by the (log‑scaled) fitness improvement
      to prevent early large gains from dominating later fine‑tuning.
    • Provides hooks (`save_state`, `load_state`) so a run can be paused and
      resumed without losing the adaptive memories.
    """

    # ←––––––––––––––––– Constructor ––––––––––––––––→
    def __init__(
        self,
        environment,
        *,
        method: str = "jade",
        c: float = 0.1,
        history_size: int = 5,
        max_number_generations_per_episode: int = constants.MAX_GENERATIONS,
    ) -> None:
        """
        Initialize the PopDEController.

        Args:
            environment: The environment implementing the DE algorithm interface.
            method (str): 'jade' or 'shade' variant of adaptive DE.
            c (float): Learning factor for updating means (0 < c <= 1).
            history_size (int): Memory size for SHADE variant.
            max_number_generations_per_episode (int): Generation budget per episode.
        Raises:
            ValueError: If `method` is not 'jade'/'shade' or c is out of (0,1].
        """

        super().__init__(environment)

        self.method = method.lower()
        if self.method not in {"jade", "shade"}:
            raise ValueError("`method` must be either 'jade' or 'shade'")

        if not 0 < c <= 1:
            raise ValueError("`c` must be within (0, 1]")
        self.c = c

        # Generation budget per episode
        self.max_number_generations_per_episode = max_number_generations_per_episode
        self.max_steps_per_episode = int(
            self.max_number_generations_per_episode
            / self.env.num_generations_per_iteration
        )

        # JADE statistics
        self.mu_F: float = 0.5
        self.mu_CR: float = 0.5

        # SHADE statistics
        self.H: int = history_size
        self.MF: np.ndarray = np.full(self.H, 0.5)
        self.MCR: np.ndarray = np.full(self.H, 0.5)
        self.k: int = 0  # circular pointer within H

        # Temporary success buffers (cleared each generation)
        self._success_F: List[float] = []
        self._success_CR: List[float] = []
        self._success_df: List[float] = []

        self._current_gen: int = 0  # generation tracker

    # ←––––––––––––––––– Public API ––––––––––––––––→
    def train(
        self,
        episodes: Optional[int] = 1,
        save_path: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Run one or more training episodes, adapting parameters.

        Args:
            episodes (Optional[int]): Number of episodes to run (defaults to 1).
            save_path (Optional[str]): Directory to save history.pkl if provided.
            **kwargs: Ignored.

        Returns:
            List[Dict[str, Any]]: A list of step records collected across episodes.
        """
        self.history = []
        episodes = episodes or 1
        for ep in range(episodes):
            logger.info("[PopDE] Episode %d/%d", ep + 1, episodes)
            self._run_episode(ep, learn=True)
        if save_path:
            self.save_history(os.path.join(save_path, "history.pkl"))
        return self.history

    def infer(
        self,
        max_steps: Optional[int] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Run a single inference episode without adapting parameters.

        Args:
            max_steps (Optional[int]): Override the step limit for this run.
            **kwargs: Ignored.

        Returns:
            List[Dict[str, Any]]: A list of step records for the inference run.
        """
        self.history = []
        self._run_episode(0, max_steps=max_steps, learn=False)
        return self.history

    # ←––––––––––––––––– Core episode loop ––––––––––––––––→
    def _run_episode(
        self,
        episode: int,
        *,
        max_steps: Optional[int] = None,
        learn: bool = True,
    ) -> None:
        """
        Execute one training or inference episode, stepping through the environment
        until either the episode terminates or the maximum number of steps is reached.

        During training (`learn=True`), this method:
          1. Resets internal statistics at start.
          2. Flushes and updates adaptive parameters at each new generation.
          3. Samples DE parameters (F, CR) and applies them as an action.
          4. Records successes to update adaptation memories.
          5. Logs completion at the end.

        Args:
            episode (int): Index of the current episode (0–based).
            max_steps (Optional[int]): Override the default step limit for this episode.
                If `None`, uses `self.max_steps_per_episode`.
            learn (bool): Whether to adapt parameters from rewards (True) or
                run purely in inference mode (False).

        Returns:
            None: History is recorded via `self.record_step`; no return value.
        """
        # Reset statistics/memories at the start of each episode
        self._reset_statistics()
        state = self.env.reset()
        done = False
        step = 0
        limit = max_steps if max_steps is not None else self.max_steps_per_episode

        self._current_gen = getattr(self.env.algorithm, "generation", 0)

        while not done and step < limit:
            # — Flush generation buffers when env advances to next generation
            gen = getattr(self.env.algorithm, "generation", self._current_gen)
            if learn and gen != self._current_gen:
                self._flush_generation_statistics()
                self._current_gen = gen

            # — Sample F and CR from memories
            F, CR = self._sample_params()
            action = {"differential_weights": F, "crossover_probability": CR}

            # — Interact with environment
            next_state, done, reward = self.env.execute(action)

            # — Accumulate successes
            if learn:
                self._accumulate_generation_stats(F, CR, reward)

            # — Record step
            self.record_step(
                episode=episode,
                step=step,
                state=state,
                action=action,
                hyperparameters=action,
                reward=reward,
                done=done,
            )

            state = next_state
            step += 1

        # Flush remaining successes (if we exited mid‑generation)
        if learn:
            self._flush_generation_statistics()

        logger.info(
            "[PopDE] Episode %d finished (steps=%d) • generations=%d",
            episode + 1,
            step,
            self.env.algorithm.generation,
        )

    # ←––––––––––––––––– Parameter sampling ––––––––––––––––→
    def _sample_params(self) -> Tuple[float, float]:
        """
        Sample F (differential weight) and CR (crossover probability)
        from the current adaptive memories.

        Returns:
            Tuple[float, float]: Clipped values of (F, CR) ∈ [0,1].
        """
        if self.method == "jade":
            F = self._cauchy_positive(self.mu_F, 0.1)
            CR = np.clip(np.random.normal(self.mu_CR, 0.1), 0.0, 1.0)
        else:
            r = np.random.randint(self.H)
            F = self._cauchy_positive(self.MF[r], 0.1)
            CR = np.clip(np.random.normal(self.MCR[r], 0.1), 0.0, 1.0)
        return float(np.clip(F, 0.0, 1.0)), float(CR)

    @staticmethod
    def _cauchy_positive(loc: float, scale: float) -> float:
        """
        Draw from a positive half-Cauchy distribution.

        Args:
            loc (float): Location (median) parameter.
            scale (float): Scale parameter.

        Returns:
            float: A strictly positive sample from Cauchy(loc, scale).
        """
        v = loc + scale * np.random.standard_cauchy()
        while v <= 0.0:
            v = loc + scale * np.random.standard_cauchy()
        return v

    # ←––––––––––––––––– Generation bookkeeping ––––––––––––––––→
    def _reset_statistics(self) -> None:
        """
        Clear and/or reset adaptive statistics at the start of an episode.
        """
        if self.method == "jade":
            self.mu_F, self.mu_CR = 0.5, 0.5
        else:
            self.MF.fill(0.5)
            self.MCR.fill(0.5)
            self.k = 0
        self._success_F.clear()
        self._success_CR.clear()
        self._success_df.clear()

    def _accumulate_generation_stats(self, F: float, CR: float, reward: float) -> None:
        """
        Record a successful trial if reward > 0, buffering F, CR, and Δf.

        Args:
            F (float): Differential weight used.
            CR (float): Crossover probability used.
            reward (float): Observed fitness improvement (Δf).
        """
        if reward > 0:
            self._success_F.append(F)
            self._success_CR.append(CR)
            self._success_df.append(reward)

    def _flush_generation_statistics(self) -> None:
        """
        Update adaptive memories (means or arrays) at generation boundary,
        then clear the per-generation success buffers.
        """
        if not self._success_F:
            return
        if self.method == "jade":
            lehmer_F = float(
                np.sum(np.square(self._success_F)) / np.sum(self._success_F)
            )
            mean_CR = float(np.mean(self._success_CR))
            self.mu_F = (1 - self.c) * self.mu_F + self.c * lehmer_F
            self.mu_CR = (1 - self.c) * self.mu_CR + self.c * mean_CR
        else:
            improvements = np.array(self._success_df, dtype=float)
            scaled = np.log1p(improvements)
            weights = scaled / scaled.sum()

            F_arr = np.array(self._success_F, dtype=float)
            CR_arr = np.array(self._success_CR, dtype=float)

            lehmer_F = float(
                np.sum(weights * np.square(F_arr)) / np.sum(weights * F_arr)
            )
            mean_CR = float(np.sum(weights * CR_arr))

            self.MF[self.k] = np.clip(lehmer_F, 0.0, 1.0)
            self.MCR[self.k] = mean_CR
            self.k = (self.k + 1) % self.H
        self._success_F.clear()
        self._success_CR.clear()
        self._success_df.clear()

    # ←––––––––––––––––– Checkpoint helpers ––––––––––––––––→
    def save_state(self, fname: str) -> None:
        """Save adaptive memories to disk (numpy npz)."""
        data = {
            "method": self.method,
            "mu_F": self.mu_F,
            "mu_CR": self.mu_CR,
            "MF": self.MF,
            "MCR": self.MCR,
            "k": self.k,
        }
        np.savez(fname, **data)

    def load_state(self, fname: str) -> None:
        """Load adaptive memories from disk (created with `save_state`)."""
        data = np.load(fname, allow_pickle=True)
        if data["method"].item() != self.method:
            raise ValueError("Checkpoint was created with a different method.")
        self.mu_F = float(data["mu_F"])
        self.mu_CR = float(data["mu_CR"])
        self.MF[:] = data["MF"]
        self.MCR[:] = data["MCR"]
        self.k = int(data["k"])

    # ←––––––––––––––––– Legacy instant update (optional) ––––––––––––––––→
    def _legacy_update_memory(
        self, F: float, CR: float, reward: float
    ) -> None:  # pragma: no cover
        """
        Fallback single-step update of μ_F/μ_CR (not used in core loop).

        Args:
            F (float): Differential weight.
            CR (float): Crossover probability.
            reward (float): Observed improvement.
        """
        if reward <= 0:
            return
        if self.method == "jade":
            self.mu_F = (1 - self.c) * self.mu_F + self.c * F
            self.mu_CR = (1 - self.c) * self.mu
