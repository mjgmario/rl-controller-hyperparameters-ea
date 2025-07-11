import logging
import random
from typing import Any, Dict, List, Union

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


class RandomController(BaseController):
    """
    Controller that selects random actions in the environment (no learning).
    Records, at each step:
      - 'episode', 'step', 'state', 'action', 'hyperparameters', 'reward', 'done'
    """

    def __init__(
        self,
        env: RLEnvironment,
        max_number_generations_per_episode=constants.MAX_GENERATIONS,
    ) -> None:
        """
        Args:
            env (RLEnvironment): The environment to interact with.
        """
        super().__init__(env)
        self.action_space = env.action_space
        self.max_number_generations_per_episode = max_number_generations_per_episode
        self.max_steps_per_episode = (
            self.max_number_generations_per_episode / env.num_generations_per_iteration
        )

    def train(self, episodes: int = 1, save_path: str = None) -> List[Dict[str, Any]]:
        """
        Run random interaction episodes (no learning).

        Args:
            episodes (int): Number of episodes to run.
            save_path (str): Ignored; included for interface compatibility.

        Returns:
            List[Dict]: A list of step-dictionaries.
        """
        self.history = []
        for ep in range(episodes):
            self.env.reset()
            done = False
            step = 0
            while not done:
                obs = self.get_state()
                hyperparams = self.get_hyperparameters()

                action: Dict[str, Union[int, float]] = {}
                for param, spec in self.env.actions().items():
                    if spec["type"] == "int":
                        action[param] = random.randrange(spec["num_values"])
                    else:
                        action[param] = random.uniform(
                            spec["min_value"], spec["max_value"]
                        )

                next_state, done, reward = self.env.execute(action)
                entry = {
                    "episode": ep,
                    "step": step,
                    "state": obs,
                    "action": action,
                    "hyperparameters": hyperparams,
                    "reward": reward,
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
        Perform inference using random actions.

        Args:
            max_steps (int, optional): Maximum number of steps to run. If None, run until `done`.

        Returns:
            List[Dict]: A list of step-dictionaries.
        """
        self.history = []
        self.env.reset()
        done = False
        step = 0
        logger.info("======== DEBUG: Starting inference ========")
        logger.info(f"Max Steps: {self.max_steps_per_episode}")
        while not done and step < self.max_steps_per_episode:
            obs = self.get_state()
            hyperparams = self.get_hyperparameters()

            action: Dict[str, Union[int, float]] = {}
            for param, spec in self.env.actions().items():
                if spec["type"] == "int":
                    action[param] = random.randrange(spec["num_values"])
                else:
                    action[param] = random.uniform(spec["min_value"], spec["max_value"])

            next_state, done, reward = self.env.execute(action)
            entry = {
                "episode": 0,
                "step": step,
                "state": obs,
                "action": action,
                "hyperparameters": hyperparams,
                "reward": reward,
                "done": done,
            }
            self.record_step(**entry)

            step += 1
        logger.info(
            f"--- Run completed (steps: {step}) --- generations: {self.env.algorithm.generation}"
        )
        return self.history
