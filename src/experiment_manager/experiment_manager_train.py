import logging

import mlflow

from benchmarking.functions import create_problems
from experiment_manager.experiment_manager import ExperimentManager

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


class ExperimentManagerTrain(ExperimentManager):
    """
    Extends ExperimentManager to run experiments in two phases: TRAIN and TEST.

    This manager first executes the training phase over a set of problems and
    dimensions, then runs the inference (test) phase on a separate set.
    """

    def __init__(self, config_path: str):
        """
        Initialize training/test experiment manager.

        Reads training and testing problem specifications from the configuration
        and sets up lists of (problem_name, dimension) tuples for each phase.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        super().__init__(config_path)
        train = self.cfg.get("train_problems", {}).get("problems", [])
        test = self.cfg.get("test_problems", {}).get("problems", [])
        self.train_specs = [(p["name"], d) for p in train for d in p["dims"]]
        self.test_specs = [(p["name"], d) for p in test for d in p["dims"]]

    def _train_phase(self):
        """
        Execute the training phase of the experiment.

        Builds problem instances for each training spec and invokes
        the base `_run_phase` method in 'train' mode.
        """
        self.mode = "train"
        eps = self.cfg["problem"]["epsilon"]
        probs = {}
        for pname, dim in self.train_specs:
            raw = f"{pname}_dim{dim}"
            probs[raw] = create_problems([pname], eps, {pname: dim})[pname]
        super()._run_phase(probs, "train")

    def _test_phase(self):
        """
        Execute the testing (inference) phase of the experiment.

        Builds problem instances for each test spec and invokes
        the base `_run_phase` method in 'inference' mode.
        """
        self.mode = "inference"
        eps = self.cfg["problem"]["epsilon"]
        probs = {}
        for pname, dim in self.test_specs:
            raw = f"{pname}_dim{dim}"
            probs[raw] = create_problems([pname], eps, {pname: dim})[pname]
        super()._run_phase(probs, "inference")

    def run(self):
        """
        Orchestrate the full experiment run: training phase followed by testing phase.

        Starts a top-level MLflow run tagged by `self.execution_tag`, logs tags,
        and sequentially calls `_train_phase` and `_test_phase`.
        """
        with mlflow.start_run(run_name=self.execution_tag):
            for k, v in self.mlflow_tags.items():
                mlflow.set_tag(k, v)
            logger.info(">>> Starting TRAIN phase <<<")
            self._train_phase()
            logger.info(">>> Starting TEST phase <<<")
            self._test_phase()
            logger.info(">>> Experiment complete <<<")
