import json
import logging
import os
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
from matplotlib import pyplot as plt

from benchmarking.functions import _basic_configs, _bbob_2013_configs
from benchmarking.optimization_problem import OptimizationProblem

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


def log_model_params(
    execution_number: int,
    problem_type: str,
    args_dict: Dict[str, Any],
    controller_type: Optional[str] = None,
) -> None:
    """
    Logs experiment parameters to MLflow.

    Includes:
    - Global experiment identifiers (execution number, experiment type).
    - Controller type (if provided).
    - Optimization problem metadata (bounds, maximize flag, target, epsilon).
    - All dynamic parameters from the input dictionary (e.g., EA, RL, agent settings).

    Args:
        execution_number (int): Execution index for the current run.
        problem_type (str): Label for the experiment type (e.g., 'DE_RL').
        args_dict (Dict[str, Any]): Dictionary containing parameters to log.
        controller_type (Optional[str]): Type of controller used ('rl', 'random', etc.).
    """
    mlflow.log_param("execution_number", execution_number)
    mlflow.log_param("experiment_type", problem_type)
    if controller_type is not None:
        mlflow.log_param("controller_type", controller_type)

    problem = args_dict.get("problem")
    if problem is not None:
        mlflow.log_param("problem_bounds", getattr(problem, "bounds", None).tolist())
        mlflow.log_param("problem_maximize", getattr(problem, "maximize", False))
        mlflow.log_param("problem_target_value", getattr(problem, "target_value", None))
        mlflow.log_param("problem_epsilon", getattr(problem, "epsilon", None))

    for key, val in args_dict.items():
        if key == "problem":
            continue
        try:
            if isinstance(val, np.ndarray):
                mlflow.log_param(key, val.tolist())
            else:
                mlflow.log_param(key, val)
        except Exception:
            mlflow.log_param(key, str(val))


def log_single_results_to_mlflow(
    result: Dict[str, Any], execution_number: int, output_dir: str
) -> None:
    """
    Log the outcome of a single optimization run to MLflow and filesystem.

    - Logs metrics: success flag, function call count, best fitness, evaluations to success.
    - Saves the best individual as a JSON artifact.
    - Prepares a subdirectory for further artifacts of this execution.

    Args:
        result (Dict[str, Any]): Contains keys 'success', 'num_function_calls', 'best_fitness', 'best_individual'.
        execution_number (int): Index of this execution (used in filenames).
        output_dir (str): Directory where artifacts and subfolders are created.
    """
    mlflow.log_metric("success", int(result["success"]))
    mlflow.log_metric("num_function_calls", result["num_function_calls"])
    mlflow.log_metric("best_fitness", result["best_fitness"])
    if result["success"]:
        mlflow.log_metric("evaluations_to_success", result["num_function_calls"])

    best_individual = result["best_individual"]
    best_indiv_filename = f"best_individual_exec_{execution_number}.json"
    best_indiv_path = os.path.join(output_dir, best_indiv_filename)

    with open(best_indiv_path, "w") as f:
        json.dump(best_individual.tolist(), f)
    mlflow.log_artifact(best_indiv_path)

    execution_dir = os.path.join(output_dir, f"execution_{execution_number}")
    os.makedirs(execution_dir, exist_ok=True)


def plot_single_fitness(
    base_dir: str,
    fitness_per_generation: np.ndarray,
    target_value: Optional[float] = None,
    run_name: Optional[str] = "run",
) -> None:
    """
    Generate and save a line plot of fitness values over generations for a single run.

    - X-axis: generation index (1-based).
    - Y-axis: fitness value.
    - Optionally draws a horizontal line for `target_value`.

    Saves the figure as 'fitness_single_<run_name>.png' under '<base_dir>/plots/' and logs it to MLflow.

    Args:
        base_dir (str): Root directory for saving the plot.
        fitness_per_generation (np.ndarray): 1D array of fitness values.
        target_value (Optional[float]): If provided, draws and labels a target line.
        run_name (Optional[str]): Identifier used in the plot title and filename.
    """
    generations = np.arange(1, len(fitness_per_generation) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(generations, fitness_per_generation, marker="o", label="Fitness")

    if target_value is not None:
        plt.axhline(
            y=target_value,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Target = {target_value:.2f}",
        )
        plt.text(
            generations[0],
            target_value,
            f"Target: {target_value:.2f}",
            color="red",
            va="bottom",
        )

        all_values = np.append(fitness_per_generation, target_value)
        ymin = np.min(all_values) - abs(np.min(all_values)) * 0.1
        ymax = np.max(all_values) + abs(np.max(all_values)) * 0.05
        plt.ylim(ymin, ymax)

    plt.title(f"Fitness per Generation ({run_name})")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plot_dir = os.path.join(base_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"fitness_single_{run_name}.png")
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    plt.close()


def collect_kpis_and_results(
    base_dir: str, results: List[Dict[str, Any]], args_dict: Dict[str, Any]
) -> None:
    """
    Collect KPIs and aggregate results from multiple executions, and generate global plots.

    :param results: A list of dictionaries containing the results of individual executions.
    """
    # Convert list of results to numpy arrays for efficient processing
    successes = np.array([result["success"] for result in results])
    num_function_calls = np.array([result["num_function_calls"] for result in results])
    best_fitnesses = np.array([result["best_fitness"] for result in results])
    best_individuals = np.array([result["best_individual"] for result in results])

    # Log global metrics
    log_global_metrics(
        successes,
        num_function_calls,
        best_fitnesses,
        best_individuals,
        output_dir=base_dir,
    )

    # Extract the fitness history for each execution
    fitness_history = [
        np.array([gen[1] for gen in result["individuals_history"]])
        for result in results
    ]

    # Plot mean fitness across generations using the new function
    problem = args_dict.get("problem", None)
    target = None
    if problem is not None and hasattr(problem, "target_value"):
        target = problem.target_value

    plot_mean_fitness(
        base_dir=base_dir, fitness_history=fitness_history, target_value=target
    )


def plot_mean_fitness(
    base_dir: str,
    fitness_history: List[np.ndarray],
    target_value: Optional[float] = None,
    zoom_start_gen: int = 50,
) -> None:
    """
    Plot the mean fitness per generation over a group of runs.

    - Pads shorter runs with NaN to align lengths.
    - First saves a full-range plot; then, if enough generations, a zoomed plot starting at `zoom_start_gen`.
    - Logs both as artifacts to MLflow.

    Args:
        base_dir (str): Directory where 'plots/' resides.
        fitness_history (List[np.ndarray]): List of per-run fitness arrays.
        target_value (Optional[float]): Horizontal reference line for target fitness.
        zoom_start_gen (int): Generation index at which to start the zoomed-in plot.
    """
    max_generations = max(fh.shape[0] for fh in fitness_history)
    padded_fitness_history = np.array(
        [
            np.pad(
                fh,
                (0, max_generations - fh.shape[0]),
                "constant",
                constant_values=np.nan,
            )
            for fh in fitness_history
        ]
    )
    mean_fitness_per_generation = np.nanmean(padded_fitness_history, axis=0)

    def plot_with_limits(gen_range: slice, filename: str, title_suffix: str):
        gens = range(1, max_generations + 1)
        gens = gens[gen_range]
        values = mean_fitness_per_generation[gen_range]

        plt.figure(figsize=(10, 6))
        plt.plot(gens, values, marker="o", label="Mean Fitness")

        if target_value is not None:
            plt.axhline(
                y=target_value,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Target = {target_value:.2f}",
            )
            plt.text(
                gens[0],
                target_value,
                f"Target: {target_value:.2f}",
                color="red",
                va="bottom",
            )

            all_values = np.append(values, target_value)
            ymin = np.min(all_values) - abs(np.min(all_values)) * 0.1
            ymax = np.max(all_values) + abs(np.max(all_values)) * 0.05
            plt.ylim(ymin, ymax)

        plt.title(f"Mean Fitness per Generation {title_suffix}")
        plt.xlabel("Generation")
        plt.ylabel("Mean Fitness")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plot_dir = os.path.join(base_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, filename)
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()

    plot_with_limits(slice(None), "mean_fitness_full.png", "(Full Range)")

    if max_generations >= zoom_start_gen:
        plot_with_limits(
            slice(zoom_start_gen - 1, None),
            "mean_fitness_zoom.png",
            f"(From Gen {zoom_start_gen})",
        )
    else:
        logger.info(
            f"[Warning] Zoom plot not generated: total generations = {max_generations} < zoom_start_gen = {zoom_start_gen}"
        )


def log_global_metrics(
    successes: np.ndarray,
    num_function_calls: np.ndarray,
    best_fitnesses: np.ndarray,
    best_individuals: np.ndarray,
    output_dir: str,
) -> None:
    """
    Compute and log global experiment metrics to MLflow and save the best individual.

    - Success rate, average calls, average best fitness, last-run best, overall best.
    - Average evaluations to success (if any successes).
    - Saves the best overall individual as JSON.

    Args:
        successes (np.ndarray): Boolean array indicating run successes.
        num_function_calls (np.ndarray): Number of function calls per run.
        best_fitnesses (np.ndarray): Best fitness values per run.
        best_individuals (np.ndarray): Corresponding best individual arrays per run.
        output_dir (str): Directory where the 'best_overall_individual.json' is saved.
    """

    # Calculate KPIs
    success_rate = np.mean(successes)
    avg_num_function_calls = np.mean(num_function_calls)
    avg_best_fitness = np.mean(best_fitnesses)
    last_execution_best_fitness = best_fitnesses[-1]

    # Determine the best execution
    best_execution_index = np.argmin(best_fitnesses)
    best_overall_fitness = best_fitnesses[best_execution_index]
    best_overall_individual = best_individuals[best_execution_index]

    # Calculate average evaluations to success, considering only successful runs
    evaluations_for_success = num_function_calls[successes == 1]
    avg_evaluations_to_success: Optional[float] = (
        np.mean(evaluations_for_success) if len(evaluations_for_success) > 0 else None
    )

    # Log KPIs to MLflow
    mlflow.log_metric("success_rate", success_rate)
    mlflow.log_metric("avg_num_function_calls", avg_num_function_calls)
    mlflow.log_metric("avg_best_fitness", avg_best_fitness)
    mlflow.log_metric("last_execution_best_fitness", last_execution_best_fitness)
    mlflow.log_metric("best_overall_fitness", best_overall_fitness)

    if avg_evaluations_to_success is not None:
        mlflow.log_metric("avg_evaluations_to_success", avg_evaluations_to_success)

    # Log the best execution details
    mlflow.log_metric("best_execution_index", int(best_execution_index))

    mlflow.log_metric("best_fitness", float(best_overall_fitness))

    artifact_path = os.path.join(output_dir, "best_overall_individual.json")
    with open(artifact_path, "w") as f:
        import json

        json.dump(best_overall_individual.tolist(), f)

    mlflow.log_artifact(artifact_path)

    mlflow.log_metric(
        "num_function_calls", int(num_function_calls[best_execution_index])
    )

    logger.info(f"Success Rate: {success_rate:.2f}")
    logger.info(f"Average Number of Function Calls: {avg_num_function_calls:.2f}")
    logger.info(f"Average Best Fitness: {avg_best_fitness:.4f}")
    logger.info(f"Best Overall Fitness: {best_overall_fitness:.4f}")
    logger.info(f"Best Individual: {best_overall_individual.tolist()}")
    if avg_evaluations_to_success is not None:
        logger.info(f"Average Evaluations to Success: {avg_evaluations_to_success:.2f}")


def plot_single_history(history: List[Dict[str, Any]], run_dir: str) -> None:
    """
    Generate time-series plots for a single run’s history, broken into four categories:
      1. Observables (from `state` dict)
      2. Hyperparameters (from `hyperparameters` dict)
      3. Actions (from `action` dict)
      4. Reward

    Each category is placed in its own subfolder under '<run_dir>/plots/'.

    Args:
        history (List[Dict[str, Any]]): Sequence of step records containing 'state', 'hyperparameters', 'action', 'reward'.
        run_dir (str): Base directory for saving plots and MLflow artifacts.
    """

    if not history:
        return

    # Ensure directory structure
    base_plots_dir = os.path.join(run_dir, "plots")
    obs_dir = os.path.join(base_plots_dir, "observables")
    hp_dir = os.path.join(base_plots_dir, "hyperparameters")
    action_dir = os.path.join(base_plots_dir, "actions")
    reward_dir = os.path.join(base_plots_dir, "reward")

    os.makedirs(obs_dir, exist_ok=True)
    os.makedirs(hp_dir, exist_ok=True)
    os.makedirs(action_dir, exist_ok=True)
    os.makedirs(reward_dir, exist_ok=True)

    # Build time index
    steps = [entry["step"] for entry in history]

    # 1) Plot each observable
    first_state = history[0].get("state") or {}
    if isinstance(first_state, dict) and first_state:
        observable_keys = list(first_state.keys())
        for obs_key in observable_keys:
            values = []
            for entry in history:
                state = entry.get("state") or {}
                values.append(state.get(obs_key, np.nan))
            plt.figure()
            plt.plot(steps, values, label=obs_key)
            plt.xlabel("Step")
            plt.ylabel(obs_key)
            plt.title(f"{obs_key} over Time")
            plt.legend()
            plt.tight_layout()
            filename = os.path.join(obs_dir, f"{obs_key}.png")
            plt.savefig(filename)
            plt.close()
            mlflow.log_artifact(
                filename, artifact_path=os.path.relpath(obs_dir, run_dir)
            )

    # 2) Plot each hyperparameter
    first_hp = history[0].get("hyperparameters") or {}
    if isinstance(first_hp, dict) and first_hp:
        hyperparam_keys = list(first_hp.keys())
        for hp_key in hyperparam_keys:
            vals = []
            labels = {}
            label_counter = 0
            is_categorical = False

            for entry in history:
                raw = (entry.get("hyperparameters") or {}).get(hp_key)
                if isinstance(raw, list) and raw:
                    raw = raw[0]

                if isinstance(raw, (int, float)):
                    vals.append(float(raw))
                elif isinstance(raw, str):
                    is_categorical = True
                    if raw not in labels:
                        labels[raw] = label_counter
                        label_counter += 1
                    vals.append(labels[raw])
                elif raw is not None:
                    try:
                        vals.append(float(raw))
                    except Exception:
                        vals.append(np.nan)
                else:
                    vals.append(np.nan)

            plt.figure()
            plt.plot(steps, vals, label=hp_key, marker="o")
            plt.xlabel("Step")
            plt.ylabel(hp_key)
            plt.title(f"{hp_key} over Time")
            plt.legend()

            if is_categorical and labels:
                cat_legend = ", ".join([f"{v}={k}" for k, v in labels.items()])
                plt.figtext(
                    0.5,
                    -0.1,
                    f"Categories: {cat_legend}",
                    wrap=True,
                    ha="center",
                    fontsize=8,
                )

            plt.tight_layout()
            filename = os.path.join(hp_dir, f"{hp_key}.png")
            plt.savefig(filename, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(
                filename, artifact_path=os.path.relpath(hp_dir, run_dir)
            )

    # 3) Plot each action component
    first_action = history[0].get("action") or {}
    if isinstance(first_action, dict) and first_action:
        action_keys = list(first_action.keys())
        for act_key in action_keys:
            vals = []
            labels = {}
            label_counter = 0
            is_categorical = False

            for entry in history:
                raw = (entry.get("action") or {}).get(act_key)

                if isinstance(raw, (int, float)):
                    vals.append(float(raw))
                elif isinstance(raw, str):
                    is_categorical = True
                    if raw not in labels:
                        labels[raw] = label_counter
                        label_counter += 1
                    vals.append(labels[raw])
                elif raw is not None:
                    try:
                        vals.append(float(raw))
                    except Exception:
                        vals.append(np.nan)
                else:
                    vals.append(np.nan)

            plt.figure()
            plt.plot(steps, vals, label=act_key, marker="o")
            plt.xlabel("Step")
            plt.ylabel(act_key)
            plt.title(f"{act_key} over Time")
            plt.legend()

            if is_categorical and labels:
                cat_legend = ", ".join([f"{v}={k}" for k, v in labels.items()])
                plt.figtext(
                    0.5,
                    -0.1,
                    f"Categories: {cat_legend}",
                    wrap=True,
                    ha="center",
                    fontsize=8,
                )

            plt.tight_layout()
            filename = os.path.join(action_dir, f"{act_key}.png")
            plt.savefig(filename, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(
                filename, artifact_path=os.path.relpath(action_dir, run_dir)
            )

    # 4) Plot reward
    if any("reward" in entry for entry in history):
        reward_vals = [entry.get("reward", np.nan) for entry in history]
        plt.figure()
        plt.plot(steps, reward_vals, label="reward")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.title("Reward over Time")
        plt.legend()
        plt.tight_layout()
        filename = os.path.join(reward_dir, "reward.png")
        plt.savefig(filename)
        plt.close()
        mlflow.log_artifact(
            filename, artifact_path=os.path.relpath(reward_dir, run_dir)
        )


def plot_group_history(histories: List[List[Dict[str, Any]]], problem_dir: str) -> None:
    """
    Create mean time-series plots across multiple runs for:
      - Observables
      - Hyperparameters
      - Actions
      - Reward (if present)

    Pads each run’s data to the same length, then computes and plots the mean.

    Args:
        histories (List[List[Dict[str, Any]]]): List of run histories.
        problem_dir (str): Directory under which 'plots_group/' is created.
    """
    if not histories:
        return

    # Create directory structure
    group_plots_dir = os.path.join(problem_dir, "plots_group")
    obs_dir = os.path.join(group_plots_dir, "observables")
    hp_dir = os.path.join(group_plots_dir, "hyperparameters")
    action_dir = os.path.join(group_plots_dir, "actions")
    reward_dir = os.path.join(group_plots_dir, "reward")

    os.makedirs(obs_dir, exist_ok=True)
    os.makedirs(hp_dir, exist_ok=True)
    os.makedirs(action_dir, exist_ok=True)
    os.makedirs(reward_dir, exist_ok=True)

    max_steps = max(len(h) for h in histories)
    time_axis = list(range(max_steps))

    first_entry = histories[0][0]
    observable_keys = list((first_entry.get("state") or {}).keys())
    hyperparam_keys = list((first_entry.get("hyperparameters") or {}).keys())
    action_keys = list((first_entry.get("action") or {}).keys())
    has_reward = "reward" in first_entry

    obs_aggregate = {k: [] for k in observable_keys}
    hp_aggregate = {k: [] for k in hyperparam_keys}
    action_aggregate = {k: [] for k in action_keys}
    reward_aggregate: List[List[float]] = []

    hp_label_maps = {k: {} for k in hyperparam_keys}
    action_label_maps = {k: {} for k in action_keys}

    for hist in histories:
        length = len(hist)

        for k in observable_keys:
            series = [(entry.get("state") or {}).get(k, np.nan) for entry in hist]
            padded = series + [np.nan] * (max_steps - length)
            obs_aggregate[k].append(padded)

        for k in hyperparam_keys:
            label_map = hp_label_maps[k]
            label_counter = len(label_map)
            series = []
            for entry in hist:
                raw = (entry.get("hyperparameters") or {}).get(k)
                if isinstance(raw, list) and raw:
                    raw = raw[0]
                if isinstance(raw, (int, float)):
                    series.append(float(raw))
                elif isinstance(raw, str):
                    if raw not in label_map:
                        label_map[raw] = label_counter
                        label_counter += 1
                    series.append(label_map[raw])
                elif raw is not None:
                    try:
                        series.append(float(raw))
                    except Exception:
                        series.append(np.nan)
                else:
                    series.append(np.nan)
            padded = series + [np.nan] * (max_steps - length)
            hp_aggregate[k].append(padded)

        for k in action_keys:
            label_map = action_label_maps[k]
            label_counter = len(label_map)
            series = []
            for entry in hist:
                raw = (entry.get("action") or {}).get(k)
                if isinstance(raw, (int, float)):
                    series.append(float(raw))
                elif isinstance(raw, str):
                    if raw not in label_map:
                        label_map[raw] = label_counter
                        label_counter += 1
                    series.append(label_map[raw])
                elif raw is not None:
                    try:
                        series.append(float(raw))
                    except Exception:
                        series.append(np.nan)
                else:
                    series.append(np.nan)
            padded = series + [np.nan] * (max_steps - length)
            action_aggregate[k].append(padded)

        if has_reward:
            series = [entry.get("reward", np.nan) for entry in hist]
            padded = series + [np.nan] * (max_steps - length)
            reward_aggregate.append(padded)

    # Plot Observables
    for k, runs in obs_aggregate.items():
        data = np.vstack(runs)
        mean_series = np.nanmean(data, axis=0)
        plt.figure()
        plt.plot(time_axis, mean_series, label=f"mean {k}")
        plt.xlabel("Step")
        plt.ylabel(k)
        plt.title(f"Mean {k} over Time (across runs)")
        plt.legend()
        plt.tight_layout()
        filename = os.path.join(obs_dir, f"mean_{k}.png")
        plt.savefig(filename)
        plt.close()
        mlflow.log_artifact(
            filename, artifact_path=os.path.relpath(obs_dir, problem_dir)
        )

    # Plot Hyperparameters
    for k, runs in hp_aggregate.items():
        data = np.vstack(runs)
        mean_series = np.nanmean(data, axis=0)
        plt.figure()
        plt.plot(time_axis, mean_series, label=f"mean {k}")
        plt.xlabel("Step")
        plt.ylabel(k)
        plt.title(f"Mean {k} over Time (across runs)")
        plt.legend()

        label_map = hp_label_maps.get(k)
        if label_map:
            inv_map = {v: k for k, v in label_map.items()}
            cat_legend = ", ".join([f"{v}={inv_map[v]}" for v in sorted(inv_map)])
            plt.figtext(
                0.5,
                -0.1,
                f"Categories: {cat_legend}",
                wrap=True,
                ha="center",
                fontsize=8,
            )

        plt.tight_layout()
        filename = os.path.join(hp_dir, f"mean_{k}.png")
        plt.savefig(filename, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(
            filename, artifact_path=os.path.relpath(hp_dir, problem_dir)
        )

    # Plot Actions
    for k, runs in action_aggregate.items():
        data = np.vstack(runs)
        mean_series = np.nanmean(data, axis=0)
        plt.figure()
        plt.plot(time_axis, mean_series, label=f"mean {k}")
        plt.xlabel("Step")
        plt.ylabel(k)
        plt.title(f"Mean {k} over Time (across runs)")
        plt.legend()

        label_map = action_label_maps.get(k)
        if label_map:
            inv_map = {v: k for k, v in label_map.items()}
            cat_legend = ", ".join([f"{v}={inv_map[v]}" for v in sorted(inv_map)])
            plt.figtext(
                0.5,
                -0.1,
                f"Categories: {cat_legend}",
                wrap=True,
                ha="center",
                fontsize=8,
            )

        plt.tight_layout()
        filename = os.path.join(action_dir, f"mean_{k}.png")
        plt.savefig(filename, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(
            filename, artifact_path=os.path.relpath(action_dir, problem_dir)
        )

    # Plot Reward
    if has_reward and reward_aggregate:
        data = np.vstack(reward_aggregate)
        mean_reward = np.nanmean(data, axis=0)
        plt.figure()
        plt.plot(time_axis, mean_reward, label="mean reward")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.title("Mean Reward over Time (across runs)")
        plt.legend()
        plt.tight_layout()
        filename = os.path.join(reward_dir, "mean_reward.png")
        plt.savefig(filename)
        plt.close()
        mlflow.log_artifact(
            filename, artifact_path=os.path.relpath(reward_dir, problem_dir)
        )


def log_problem_metadata(
    pname: str, problem: OptimizationProblem, output_dir: str = None
):
    """
    Log metadata for a given optimization problem to MLflow and save a JSON summary.

    Supports:
      - Classical benchmark functions from `_basic_configs`.
      - BBOB-2013 functions from `_bbob_2013_configs`.

    For each problem:
      - Logs dimension, bounds, target value, and additional config (seed or opt point).
      - Writes `<pname>_info.json` with summary fields.

    Args:
        pname (str): Problem identifier (may include '_dim<d>').
        problem (OptimizationProblem): Instance containing `bounds`, `target_value`, `x_opt`, etc.
        output_dir (str, optional): Directory to save the JSON artifact; defaults to working directory.

    Raises:
        ValueError: If `pname` is unrecognized by the known configurations.
    """
    # Extract base problem name if dimension suffix present
    base_name = pname
    if "_dim" in pname:
        base_name = pname.rsplit("_dim", 1)[0]

    dim = problem.bounds.shape[0]
    lb, ub = problem.bounds[0]
    f_opt = problem.target_value

    # Classical problem
    if base_name in _basic_configs:
        entry = _basic_configs[base_name]
        opt_pt = entry["opt_point"](dim)
        logger.info(
            f"• Classical problem '{pname}': dim={dim}, bounds=[{lb},{ub}], f_opt={f_opt}"
        )
        mlflow.log_param(f"{pname}_dim", dim)
        mlflow.log_param(f"{pname}_bounds", f"[{lb},{ub}]")
        mlflow.log_param(f"{pname}_fopt", f_opt)

        # Save metadata artifact
        artifact_dir = output_dir or os.getcwd()
        os.makedirs(artifact_dir, exist_ok=True)
        info = {
            "name": str(pname),
            "type": "classical",
            "dim": int(dim),
            "bounds": [float(lb), float(ub)],
            "f_opt": float(f_opt),
            "opt_point_first_5": [float(x) for x in opt_pt[: min(5, dim)]],
        }
        info_path = os.path.join(artifact_dir, f"{pname}_info.json")
        with open(info_path, "w") as jf:
            json.dump(info, jf, indent=2)
        mlflow.log_artifact(info_path, artifact_path=os.path.basename(artifact_dir))
        return

    # Implemented BBOB-2013
    if base_name in _bbob_2013_configs:
        entry = _bbob_2013_configs[base_name]
        seed = entry.get("seed", 0)
        x_opt_preview = problem.x_opt[: min(5, dim)]
        logger.info(
            f"• BBOB problem '{pname}' (implemented): dim={dim}, bounds=[{lb},{ub}], seed={seed}, f_opt={f_opt}"
        )
        mlflow.log_param(f"{pname}_dim", dim)
        mlflow.log_param(f"{pname}_bounds", f"[{lb},{ub}]")
        mlflow.log_param(f"{pname}_seed", seed)
        mlflow.log_param(f"{pname}_fopt", f_opt)
        mlflow.log_param(f"{pname}_xopt_0_to_4", str(list(x_opt_preview)))

        artifact_dir = output_dir or os.getcwd()
        os.makedirs(artifact_dir, exist_ok=True)
        info = {
            "name": pname,
            "type": "bbob_2013",
            "dim": dim,
            "bounds": [lb, ub],
            "seed": seed,
            "x_opt_first_5": list(x_opt_preview),
            "f_opt": f_opt,
        }
        info_path = os.path.join(artifact_dir, f"{pname}_info.json")
        with open(info_path, "w") as jf:
            json.dump(info, jf, indent=2)
        artifact_subdir = os.path.basename(output_dir)
        mlflow.log_artifact(info_path, artifact_path=artifact_subdir)
        return

    # Unexpected name
    raise ValueError(f"Unknown problem name '{pname}' in log_problem_metadata().")
