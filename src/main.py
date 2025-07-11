import argparse
import logging
import os

from experiment_manager.experiment_manager_train import ExperimentManagerTrain

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


def main():
    parser = argparse.ArgumentParser(description="Run DE+RL tuning experiment")

    parser.add_argument(
        "--config",
        type=str,
        default="configurations/config_train.yaml",
        help="Path to the YAML configuration file (default: configurations/config.yaml)",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    logger.info(f"Using configuration: {args.config}")
    manager = ExperimentManagerTrain(config_path=args.config)
    manager.run()


if __name__ == "__main__":
    main()
