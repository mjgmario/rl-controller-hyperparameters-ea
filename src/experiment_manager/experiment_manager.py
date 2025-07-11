import logging
import os
import pickle
import random
import sys
from datetime import datetime
from shutil import copyfile

import mlflow
import numpy as np
import tensorflow as tf
import yaml

from benchmarking.functions import create_problems
from benchmarking.optimization_problem import OptimizationProblem
from controllers.no_action_controlller import NoOpController
from controllers.pop_de_controller import PopDEController
from controllers.random_controller import RandomController
from controllers.rechenberg_controller import RechenbergController
from controllers.rl_controller import RLController
from environment.action_space import ActionSpace
from environment.rl_environment import RLEnvironment
from evolutionary_algorithms.differential_evolution import (
    DifferentialEvolutionOptimizer,
)
from evolutionary_algorithms.evolutionary_strategies import EvolutionaryStrategies
from utils import constants
from utils.utils_results import (
    collect_kpis_and_results,
    log_problem_metadata,
    log_single_results_to_mlflow,
    plot_group_history,
    plot_single_fitness,
    plot_single_history,
)

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


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception


class ExperimentManager:
    """
    Manages end-to-end experiments combining DE/ES algorithms with RL controllers.

    Reads configuration, sets up MLflow tracking, seeds, action spaces,
    builds algorithms and environments, selects controllers, executes runs,
    and logs results and KPIs.
    """

    def __init__(self, config_path: str):
        """
        Load experiment configuration and initialize core components.

        - Parses YAML config.
        - Configures MLflow experiment and tracking URI.
        - Sets random seeds for reproducibility.
        - Builds the ActionSpace.
        - Reads algorithm-specific and RL settings.
        - Instantiates optimization problems.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        self.config_path = config_path

        # MLflow setup
        mlf = self.cfg.get("mlflow", {})
        mlflow.set_experiment(mlf.get("experiment_name", "default"))
        if mlf.get("tracking_uri"):
            mlflow.set_tracking_uri(mlf["tracking_uri"])
        self.mlflow_tags = mlf.get("tags", {})

        # Random seed
        seed = self.cfg["experiment"].get("random_seed")
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Action space and algorithm/env configs
        as_cfg = self.cfg["action_space"]
        self.action_space = ActionSpace(
            hyperparameters=as_cfg["hyperparameters"],
            controllable_parameters=as_cfg["controllable_parameters"],
            epsilon=as_cfg.get("epsilon", 0.01),
        )
        self.algo_type = self.cfg.get("algorithm", "de").lower()
        if self.algo_type == "de":
            self.de_cfg = self.cfg["differential_evolution"]
        else:
            self.es_cfg = self.cfg["evolutionary_strategies"]

        self.rl_cfg = self.cfg["rl_environment"]
        self.agent_cfg = self.cfg["agent"]

        self.controller_type = self.cfg.get("controller_type", "rl")
        self.runs_per_problem = self.cfg["experiment"].get("runs_per_problem", 1)
        self.mode = self.cfg.get("mode", "train")
        self.results_dir = self.cfg["experiment"].get("results_dir", "../results")
        self.execution_tag = self.cfg["experiment"].get(
            "execution_tag"
        )  # e.g. 'experimentacion_1'

        self.agent_cfg.get("config", {})["max_episode_timesteps"] = self.rl_cfg.get(
            "max_number_generations_per_episode", constants.MAX_GENERATIONS
        ) / self.rl_cfg.get("num_generations_per_iteration", 1)

        self.pretrained_checkpoint = self.agent_cfg.get("pretrained_dir")
        self.problem_prefix = self.cfg["experiment"].get("problem_tag_prefix", "")
        # Problems to run (single-phase)
        prob_cfg = self.cfg["problem"]
        self.problems = create_problems(
            names=prob_cfg.get("names", []),
            epsilon=prob_cfg.get("epsilon", 1e-4),
            override_dims=prob_cfg.get("override_dims", {}),
            override_bounds=prob_cfg.get("override_bounds", {}),
        )
        self.history_all = {}

    def run(self):
        """
        Execute the experiment across all problems, in the configured mode.

        - Starts a global MLflow run.
        - Logs problem metadata.
        - Calls _run_phase for training or inference.
        - Saves the combined history to disk.
        """
        with mlflow.start_run(run_name="global_run"):
            for k, v in self.mlflow_tags.items():
                mlflow.set_tag(k, v)
            for pname, problem in self.problems.items():
                log_problem_metadata(pname, problem)
            self._run_phase(self.problems, self.mode)

        # Save combined history
        outpath = self.cfg["experiment"].get("history_path", "history_all.pkl")
        with open(outpath, "wb") as f:
            pickle.dump(self.history_all, f)
        logger.info(f"Saved full history to {outpath}")

    def _snapshot_config(self, output_dir: str):
        """
        Copy the config file into the results directory and log it as an MLflow artifact.

        Args:
            output_dir (str): Directory where the config snapshot will be saved.
        """
        os.makedirs(output_dir, exist_ok=True)
        cfg_name = os.path.basename(self.config_path)
        dest = os.path.join(output_dir, cfg_name)
        copyfile(self.config_path, dest)
        mlflow.log_artifact(dest, artifact_path=os.path.basename(output_dir))

    def _run_phase(self, problems: dict, mode: str):
        """
        Iterates over each problem and run, builds environments/controllers,
        executes experiments, and logs results.

        Args:
            problems (dict): Mapping from problem name to OptimizationProblem.
            mode (str): Either 'train' or 'infer'.
        """
        for raw_label, problem in problems.items():
            mlflow.set_tag("problem", raw_label)

            base_name = (
                f"{self.execution_tag + '_' if self.execution_tag else ''}{raw_label}"
            )

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            problem_dir = os.path.join(self.results_dir, f"{base_name}_{stamp}")
            os.makedirs(problem_dir, exist_ok=True)
            self._snapshot_config(problem_dir)
            log_problem_metadata(raw_label, problem, problem_dir)
            results, hist_list = [], []
            with mlflow.start_run(run_name=f"{mode}_{base_name}", nested=True):
                for run_id in range(1, self.runs_per_problem + 1):
                    run_name = f"{mode}_{base_name}_run{run_id}"
                    run_dir = os.path.join(problem_dir, f"run_{run_id}")
                    os.makedirs(run_dir, exist_ok=True)
                    logger.info(f"Starting {run_name} -> {run_dir}")
                    with mlflow.start_run(run_name=run_name, nested=True):
                        de, env = self._build_algo_and_env(problem)
                        tf.keras.backend.clear_session()
                        controller = self._select_controller(
                            env, run_id, problem_dir, mode
                        )
                        history = self._execute(controller, mode, run_dir)
                        result = self._log_single(
                            env, history, run_dir, run_id, problem
                        )
                        results.append(result)
                        hist_list.append(history)
                        # Update pretrained for next run if training
                        if mode == "train" and isinstance(controller, RLController):
                            self.pretrained_checkpoint = run_dir
                    # end nested run

                collect_kpis_and_results(
                    base_dir=problem_dir,
                    results=results,
                    args_dict={
                        **{"problem": problem},
                        **(self.de_cfg if self.algo_type == "de" else self.es_cfg),
                    },
                )
                plot_group_history(hist_list, problem_dir)
                self.history_all[raw_label] = hist_list

    def _build_algo_and_env(self, problem: OptimizationProblem):
        """
        Instantiate the optimization algorithm (DE or ES) and wrap it in an RL environment.

        Args:
            problem (OptimizationProblem): The optimization problem to solve.

        Returns:
            tuple: (algorithm instance, RLEnvironment instance)
        """
        if self.algo_type == "es":
            es_cfg = self.cfg["evolutionary_strategies"]
            algo = EvolutionaryStrategies(
                problem=problem,
                phro=es_cfg["phro"],
                epsilon0=es_cfg["epsilon0"],
                tau=es_cfg["tau"],
                tau_prime=es_cfg["tau_prime"],
                recombination_individuals_strategy=es_cfg[
                    "recombination_individuals_strategy"
                ],
                recombination_strategy_strategy=es_cfg[
                    "recombination_strategy_strategy"
                ],
                mutation_steps=es_cfg["mutation_steps"],
                survivor_strategy=es_cfg["survivor_strategy"],
                mu=es_cfg["mu"],
                lamda=es_cfg["lamda"],
                max_number_generation=self.rl_cfg.get(
                    "max_generations", constants.MAX_GENERATIONS
                ),
                use_rechenberg=es_cfg.get("use_rechenberg", False),
                c_up=es_cfg.get("c_up", 1.5),
                c_down=es_cfg.get("c_down", 1.5),
                target_success_rate=es_cfg.get("target_success_rate", 0.2),
                window_r=es_cfg.get("window_r", 10),
                external_sigma_control=es_cfg.get("external_sigma_control", False),
            )
        else:
            de_cfg = self.de_cfg
            algo = DifferentialEvolutionOptimizer(
                problem=problem,
                population_size=de_cfg["population_size"],
                differential_weights=de_cfg["differential_weights"],
                crossover_probability=de_cfg["crossover_probability"],
                base_index_strategy=de_cfg["base_index_strategy"],
                differential_number=de_cfg["differential_number"],
                recombination_strategy=de_cfg["recombination_strategy"],
                differentials_weights_strategy=de_cfg["differentials_weights_strategy"],
                max_number_generation=self.rl_cfg.get(
                    "max_generations", constants.MAX_GENERATIONS
                ),
            )
        env = RLEnvironment(
            algorithm=algo,
            action_space=self.action_space,
            num_generations_per_iteration=self.rl_cfg.get(
                "num_generations_per_iteration", 1
            ),
            max_generations=self.rl_cfg.get("max_generations", None),
            observables=self.rl_cfg.get("observables"),
            observables_config=self.rl_cfg.get("observables_config"),
            reward_type=self.rl_cfg.get("reward_type", "parent_offspring"),
            omega=self.rl_cfg.get("omega", 0.5),
            ref_window=self.rl_cfg.get("ref_window", 5),
            time_scale=self.rl_cfg.get("time_scale", 1.0),
        )
        return algo, env

    def _select_controller(self, env, run_id: int, problem_dir: str, mode: str):
        """
        Choose and instantiate the appropriate controller based on config.

        Args:
            env: The RL environment wrapping the optimizer.
            run_id (int): The current run index for logging.
            problem_dir (str): Directory to save run artifacts.
            mode (str): 'train' or 'infer'.

        Returns:
            BaseController: One of RandomController, PopDEController, NoOpController,
                            RechenbergController, or RLController.
        """
        if self.controller_type == "random":
            return RandomController(
                env,
                max_number_generations_per_episode=self.rl_cfg.get(
                    "max_number_generations_per_episode",
                    constants.MAX_GENERATIONS,
                ),
            )
        if self.controller_type in {"jade", "shade"}:
            return PopDEController(
                env,
                method=self.controller_type,  # jade|shade
                c=self.cfg["popde"].get("c", 0.1),
                history_size=self.cfg["popde"].get("history_size", 5),
                max_number_generations_per_episode=self.rl_cfg.get(
                    "max_number_generations_per_episode",
                    constants.MAX_GENERATIONS,
                ),
            )
        if self.controller_type == "noop":
            self.rl_cfg.get("num_generations_per_iteration", 1)
            return NoOpController(
                env,
                num_generations_per_iteration=self.rl_cfg.get(
                    "num_generations_per_iteration", 1
                ),
                max_number_generations_per_episode=self.rl_cfg.get(
                    "max_number_generations_per_episode",
                    constants.MAX_GENERATIONS,
                ),
            )
        if self.controller_type == "rechenberg":
            self.rl_cfg.get("num_generations_per_iteration", 1)
            return RechenbergController(
                env,
                num_generations_per_iteration=self.rl_cfg.get(
                    "num_generations_per_iteration", 1
                ),
                max_number_generations_per_episode=self.rl_cfg.get(
                    "max_number_generations_per_episode",
                    constants.MAX_GENERATIONS,
                ),
            )
        # RL controller
        return RLController(
            environment=env,
            agent=None,
            agent_type=self.agent_cfg.get("type", "ppo"),
            agent_config=self.agent_cfg.get("config", {}),
            mode=mode,
            pretrained_dir=self.pretrained_checkpoint,
            max_number_generations_per_episode=self.rl_cfg.get(
                "max_number_generations_per_episode", constants.MAX_GENERATIONS
            ),
        )

    def _execute(self, controller, mode: str, save_path: str):
        """
        Run either training or inference on the provided controller.

        Args:
            controller: The controller instance to run.
            mode (str): 'train' to call `.train()`, otherwise `.infer()`.
            save_path (str): Directory for saving any output artifacts.

        Returns:
            List[dict]: The history produced by the controller.
        """
        if mode == "train" and isinstance(controller, RLController):
            return controller.train(
                episodes=self.cfg.get("train", {}).get("episodes_per_problem"),
                save_path=save_path,
            )
        return controller.infer(max_steps=self.cfg.get("infer", {}).get("max_steps"))

    def _log_single(
        self,
        env,
        history,
        run_dir: str,
        run_id: int,
        problem: OptimizationProblem,
    ):
        """
        Log the results of a single run to disk, MLflow, and generate plots.

        Args:
            env: The RL environment used for this run.
            history (list): Step-by-step history from the controller.
            run_dir (str): Directory for run artifacts.
            run_id (int): Index of the current run.
            problem (OptimizationProblem): The problem instance.

        Returns:
            dict: A summary of results including success flag, calls, best fitness, etc.
        """
        # save history
        hist_path = os.path.join(run_dir, "history.pkl")
        with open(hist_path, "wb") as f:
            pickle.dump(history, f)
        mlflow.log_artifact(hist_path, artifact_path="history_pickle")
        plot_single_history(history, run_dir)

        best_fit = float(np.min(env.algorithm.fitness_values))
        res = {
            "success": problem.is_success(best_fit),
            "num_function_calls": getattr(
                env.algorithm, "num_function_calls", len(history)
            ),
            "best_fitness": best_fit,
            "best_individual": getattr(
                env.algorithm, "best_individual_execution", None
            ),
            "individuals_history": getattr(
                env.algorithm, "best_individuals", len(history)
            ),
        }
        best_fitness_per_generation = np.array(
            [gen[1] for gen in res["individuals_history"]]
        )

        target = None
        if problem is not None and hasattr(problem, "target_value"):
            target = problem.target_value

        log_single_results_to_mlflow(res, run_id, run_dir)
        plot_single_fitness(run_dir, best_fitness_per_generation, target)
        return res
