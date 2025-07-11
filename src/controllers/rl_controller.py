import logging
import os
from typing import Any, Dict, List, Optional

from controllers.base_controller import BaseController
from environment.rl_environment import RLEnvironment
from tensorforce.agents import Agent
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


class RLController(BaseController):
    """
    Reinforcement Learning Controller:
      - Uses a TensorForce Agent.
      - At each step records: episode, step, state, action, hyperparameters, reward, done.
    """

    def __init__(
        self,
        environment: RLEnvironment,
        agent: Optional[Agent] = None,
        agent_type: str = "ppo",
        agent_config: Optional[Dict[str, Any]] = None,
        mode: str = "train",
        pretrained_dir: Optional[str] = None,
        max_number_generations_per_episode=constants.MAX_GENERATIONS,
    ) -> None:
        """
        Args:
            environment (RLEnvironment): The RL environment.
            agent (Optional[Agent]): If provided, use this agent; otherwise create/load one.
            agent_type (str): TensorForce agent type (e.g., 'ppo', 'dqn').
            agent_config (Optional[Dict[str, Any]]): Config kwargs for Agent.create().
            mode (str): "train" or "inference"; determines whether learning occurs.
            pretrained_dir (Optional[str]): Directory to load a pretrained agent from.
        """
        super().__init__(environment)
        self.mode = mode
        self.agent_type = agent_type
        self.agent_config = agent_config or {
            "batch_size": 10,
            "learning_rate": 1e-3,
        }
        self.pretrained_dir = pretrained_dir
        self.max_number_generations_per_episode = max_number_generations_per_episode
        self.max_steps_per_episode = (
            self.max_number_generations_per_episode
            / self.env.num_generations_per_iteration
        )
        if agent is not None:
            self.agent = agent
        else:
            if self.pretrained_dir and os.path.isdir(self.pretrained_dir):
                self.agent = Agent.load(directory=self.pretrained_dir)
            else:
                self.agent = Agent.create(
                    agent=self.agent_type,
                    environment=self.env,
                    **self.agent_config,
                )

    def train(
        self, episodes: Optional[int] = None, save_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Train the agent in the environment, recording each step.

        Args:
            episodes (Optional[int]): Number of episodes to train. If None, continues until `done`.
            save_path (Optional[str]): If provided, saves the trained agent here afterward.

        Returns:
            List[Dict]: A list of dictionaries, each containing step data.
        """
        if self.mode == "inference":
            raise RuntimeError("Cannot train in 'inference' mode.")

        self.history = []
        episode = 0
        logger.info("======== DEBUG: Starting training ========")
        logger.info(f"Mode: {self.mode}, Episodes: {episodes}, Save Path: {save_path}")
        while True:
            state = self.env.reset()
            done = False
            step = 0

            while not done and step < self.max_steps_per_episode:
                action = self.agent.act(states=state, independent=False)
                next_state, done, reward = self.env.execute(action)
                self.agent.observe(terminal=done, reward=reward)

                entry = {
                    "episode": episode,
                    "step": step,
                    "state": state,
                    "action": action,
                    "hyperparameters": self.get_hyperparameters(),
                    "reward": reward,
                    "done": done,
                }
                self.record_step(**entry)

                state = next_state
                step += 1

            logger.info(
                f"--- Episode {episode + 1} completed (steps: {step}) --- generations: {self.env.algorithm.generation}"
            )
            episode += 1
            if episodes is not None and episode >= episodes:
                break
            if episodes is None and done:
                break

        if save_path:
            self.save_agent(save_path)
        try:
            self.agent.close()
        except Exception:
            pass
        logger.info("======== DEBUG: Training finished ========")
        return self.history

    def infer(self, max_steps: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Run the agent in inference mode (no learning), recording each step.

        Args:
            max_steps (Optional[int]): Max steps to run. If None, runs until `done`.

        Returns:
            List[Dict]: A list of dictionaries, each containing step data.
        """
        self.history = []
        state = self.env.reset()
        done = False
        step = 0
        logger.info("======== DEBUG: Starting inference ========")
        logger.info(f"Mode: {self.mode}, Max Steps: {max_steps}")

        while not done and step < self.max_steps_per_episode:
            action = self.agent.act(states=state, independent=True)
            next_state, done, reward = self.env.execute(action)

            entry = {
                "episode": 0,
                "step": step,
                "state": state,
                "action": action,
                "hyperparameters": self.get_hyperparameters(),
                "reward": reward,
                "done": done,
            }
            self.record_step(**entry)

            state = next_state
            step += 1
        logger.info(
            f"Completed (steps: {step}) --- generations: {self.env.algorithm.generation}"
        )
        try:
            self.agent.close()
        except Exception:
            pass
        logger.info("======== DEBUG: Inference finished ========")

        return self.history

    def set_environment(self, new_env: RLEnvironment) -> None:
        """
        Swap out the environment and re-create the TensorForce agent.

        Args:
            new_env (RLEnvironment): The new environment instance.
        """
        try:
            self.agent.close()
        except Exception:
            pass

        self.env = new_env
        self.agent = Agent.create(
            agent=self.agent_type, environment=self.env, **self.agent_config
        )
        if self.pretrained_dir and os.path.isdir(self.pretrained_dir):
            self.agent.load(directory=self.pretrained_dir)

    def save_agent(self, save_path: str) -> None:
        """
        Save the agent's state to disk.

        Args:
            save_path (str): Directory where the agent will be saved.
        """
        os.makedirs(save_path, exist_ok=True)
        self.agent.save(directory=save_path)
