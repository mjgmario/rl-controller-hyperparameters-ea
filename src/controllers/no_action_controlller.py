import logging
from typing import Any, Dict, List

import numpy as np

from controllers.base_controller import BaseController
from environment.rl_environment import RLEnvironment
from utils import constants

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("experiment.log", mode="a"),
    ],
)

logger = logging.getLogger(__name__)


class NoOpController(BaseController):
    """
    Controller that does not adjust hyperparameters:
      - Resets the population at the start of each episode.
      - Runs the evolutionary algorithm step by step until completion.
      - Records, at each generation:
          * 'episode', 'step', 'state', 'action', 'hyperparameters', 'reward', 'done'
        where 'action' is {} (no action taken).
    """

    def __init__(
        self,
        environment: RLEnvironment,
        num_generations_per_iteration,
        max_number_generations_per_episode=constants.MAX_GENERATIONS,
    ) -> None:
        """
        Args:
            environment (RLEnvironment): The RL environment containing the algorithm.
        """
        super().__init__(environment)
        self.alg = environment.algorithm
        self.num_generations_per_iteration = num_generations_per_iteration
        self.max_number_generations_per_episode = max_number_generations_per_episode
        self.max_steps_per_episode = (
            self.max_number_generations_per_episode
            / environment.num_generations_per_iteration
        )

    def train(self, episodes: int = 1, save_path: str = None) -> List[Dict[str, Any]]:
        """
        Run training without hyperparameter tuning.

        Args:
            episodes (int): Number of independent runs.
            save_path (str): Ignored in this controller.

        Returns:
            List[Dict]: A list of step-dictionaries.
        """
        self.history = []
        for ep in range(episodes):
            self.alg.reset_population()
            done = False
            step = 0
            while not done:
                state = self.get_state()
                hyperparams = self.get_hyperparameters()

                done = self.alg.run_generation(self.num_generations_per_iteration)
                best_fitness = float(np.min(self.alg.fitness_values))

                entry = {
                    "episode": ep,
                    "step": step,
                    "state": state,
                    "action": {},
                    "hyperparameters": hyperparams,
                    "reward": best_fitness,
                    "done": done,
                }
                self.record_step(**entry)

                step += 1
            logger.info(
                f"--- Run completed (steps: {step}) --- generations: {self.env.algorithm.generation}"
            )
        return self.history

    def infer(self, max_steps: int = None) -> List[Dict[str, Any]]:
        """
        Run inference with no hyperparameter changes.

        Args:
            max_steps (int, optional): Max number of steps to run. If None, run until `done`.

        Returns:
            List[Dict]: A list of step-dictionaries.
        """
        self.history = []
        self.alg.reset_population()
        done = False
        step = 0
        logger.info("======== DEBUG: Starting inference ========")
        logger.info(f"Max Steps: {self.max_steps_per_episode}")
        while not done and step < self.max_steps_per_episode:
            state = self.get_state()
            hyperparams = self.get_hyperparameters()

            done = self.alg.run_generation(self.num_generations_per_iteration)
            best_fitness = float(np.min(self.alg.fitness_values))

            entry = {
                "episode": 0,
                "step": step,
                "state": state,
                "action": {},
                "hyperparameters": hyperparams,
                "reward": best_fitness,
                "done": done,
            }
            self.record_step(**entry)

            step += 1
        logger.info(
            f"--- Run completed (steps: {step}) --- generations: {self.env.algorithm.generation}"
        )
        return self.history
