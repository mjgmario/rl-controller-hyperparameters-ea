import logging
from typing import Any, Dict, List

import numpy as np

from controllers.no_action_controlller import NoOpController

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


class RechenbergController(NoOpController):
    """
    Extends NoOpController by injecting a sigma parameter into the hyperparameters
    recorded in history. Sigma is computed as the mean of the strategy parameters
    from the underlying algorithm.
    """

    def _inject_sigma(self, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inject the current average sigma value into a copy of the provided hyperparameters.

        Attempts to compute the mean of `self.alg.strategy_parameters`. If an error
        occurs, sets sigma to NaN.

        Args:
            hyperparams (Dict[str, Any]): Base hyperparameters to augment.

        Returns:
            Dict[str, Any]: A new dict containing all original keys plus a 'sigma' key.
        """
        try:
            sigma_val = float(np.mean(self.alg.strategy_parameters))
        except Exception:  # por seguridad
            sigma_val = float("nan")
        hyperparams = hyperparams.copy()  # evita aliasing
        hyperparams["sigma"] = sigma_val  # ✔ aquí el nuevo campo
        return hyperparams

    def train(self, episodes: int = 1, save_path: str = None) -> List[Dict[str, Any]]:
        """
        Run the training loop for a specified number of episodes.

        Each episode resets the population and runs generations until completion.
        After each generation, records a step in history including the injected sigma.

        Args:
            episodes (int): Number of episodes to execute (default 1).
            save_path (str, optional): Path to save additional outputs (unused).

        Returns:
            List[Dict[str, Any]]: The history records of all steps across episodes.
        """
        self.history = []
        for ep in range(episodes):
            self.alg.reset_population()
            done, step = False, 0
            while not done:
                state = self.get_state()
                hyperparams = self._inject_sigma(self.get_hyperparameters())

                done = self.alg.run_generation(self.num_generations_per_iteration)
                best_fitness = float(np.min(self.alg.fitness_values))

                self.record_step(
                    episode=ep,
                    step=step,
                    state=state,
                    action={},
                    hyperparameters=hyperparams,
                    reward=best_fitness,
                    done=done,
                )
                step += 1
            logger.info(
                f"--- Run completed (steps: {step}) --- generations: {self.env.algorithm.generation}"
            )
        return self.history

    def infer(self, max_steps: int = None) -> List[Dict[str, Any]]:
        """
        Run a single inference session without adapting strategy parameters.

        Resets the population, then runs generations until either completion or
        the maximum number of steps (`self.max_steps_per_episode`) is reached.
        Records each step including the injected sigma value.

        Args:
            max_steps (int, optional): Unused override for maximum steps.

        Returns:
            List[Dict[str, Any]]: The history records of all inference steps.
        """
        self.history = []
        self.alg.reset_population()
        done, step = False, 0
        logger.info("======== DEBUG: Starting inference ========")
        logger.info(f"Max Steps: {self.max_steps_per_episode}")
        while not done and step < self.max_steps_per_episode:
            state = self.get_state()
            hyperparams = self._inject_sigma(self.get_hyperparameters())

            done = self.alg.run_generation(self.num_generations_per_iteration)
            best_fitness = float(np.min(self.alg.fitness_values))

            self.record_step(
                episode=0,
                step=step,
                state=state,
                action={},
                hyperparameters=hyperparams,
                reward=best_fitness,
                done=done,
            )
            step += 1
        logger.info(
            f"--- Run completed (steps: {step}) --- generations: {self.env.algorithm.generation}"
        )
        return self.history
