# Mario Jiménez's RL Controller hyperparameters Evolutionary Algorithm
A reinforcement learning (RL)-based controller optimizes the hyperparameters of evolutionary algorithms in real time. The RL agent dynamically tunes user-configurable parameters—such as mutation rate and selection strategies—to improve performance across benchmark optimization problems.
## Setup and Execution

### Setup code
1. Install Conda and Python 3.9.
   2. Setup the environment:
      ```sh
      conda create --name myenv python=3.9
      conda init
      conda activate myenv
      git clone https://github.com/tensorforce/tensorforce.git
      pip3 install -e tensorforce
      pip install -r requirements.txt
   3. Code execution:
      ```sh
      cd src
      python main.py
      ```

### Running Tests

1. Set the PYTHONPATH environment variable:
```sh
$env:PYTHONPATH="./src"
```
2. Run the tests using pytest:
```sh
pytest
```

### Pre-commit Configuration and Usage

This project uses `pre-commit` to enforce code quality on every commit. The configured hooks include:

- **pre-commit-hooks**  
  - `trailing-whitespace`  
  - `end-of-file-fixer`  
  - `check-yaml`  
  - `check-added-large-files`  
- **Ruff**  
  - `ruff` (via `ruff check`)

_Hooks run only on files in `src/` and `tests/`, excluding Jupyter notebooks (`*.ipynb`)._

- `trailing-whitespace`, `end-of-file-fixer`, `check-added-large-files`  
  – all file types under `src/` and `tests/`  
- `check-yaml`  
  – only `.yaml` and `.yml` under `src/` and `tests/`  
- `ruff`  
  – only `.py` under `src/` and `tests/`


#### Installing and Configuring pre-commit

1. **Install `pre-commit`**  
   ```sh
   pip install pre-commit

#### 2. Install the pre-commit hooks

Once pre-commit is installed, run the following command to install the hooks defined in `.pre-commit-config.yaml`:
  ```sh
  pre-commit install
  ```

This will set up the hooks to automatically run every time you attempt a `git commit`.

#### 3. Run pre-commit manually on all files

If you want to run the pre-commit hooks on all files manually (e.g., after changing configuration or adding new files), use the following command:
  ```sh
  pre-commit run --all-files
  ```

#### 4. Clean up previous pre-commit (optional)

Before running `pre-commit`, you may want to clean up previous attempts. You can do so using:
  ```sh
  pre-commit clean 
  ```


## Code documentation


## Code Structure

    .
    ├── src/
    │   ├── benchmarking/              # Classic + BBOB-2013 benchmark functions
    │   ├── configurations/            # YAML + parsers that define experiments
    │   ├── controllers/               # Control logics (No-Op, Random, RL…)
    │   ├── environment/               # Environment that connects evolutionary algorithm with RL
    │   ├── evolutionary_algorithms/   # DE implementation
    │   ├── experiment_manager/        # Orchestrates runs and logs to MLflow
    │   ├── utils/                     # Constants, helpers, visualisation
    │   ├── main.py                    # Entry point – single experiment
    │   └── main_train_test.py         # Two-phase: training + inference
    ├── tests/                         # PyTest: unit + integration


### `benchmarking/`

| File / Object              | Brief description                                                                                                                                   |
|----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`functions.py`**         | Implements the *benchmark functions*.<br>• Classic: *Sphere*, *Ackley*, *Rastrigin*, …<br>• BBOB-2013: *Ellipsoidal*, *Discus*, *Bent Cigar*, etc.<br>Helper → `create_problems()` returns a `dict[str, OptimizationProblem]`. |
| **`optimization_problem.py`** | Thin wrapper that stores bounds, minimise/maximise flag, and exposes `is_success(target_value ± ε)`.                                               |


### `configurations/`

| Resource                 | Purpose                                                                                                             |
|--------------------------|---------------------------------------------------------------------------------------------------------------------|
| **`config_train.yaml`**  | *Single source of truth* – problems, DE hyper-params, RL settings, and agent config.                                |
| **`operation_parser.py`**| Converts YAML “operations” blocks → list of `Operation`.                                                            |
| **`operations.py`**      | Four arithmetic ops (*multiply, divide, add, subtract*) used by the agent to tweak HPs.                             |


### `controllers/`

| Class                  | Key methods | Role                                                                                                |
|------------------------|-------------|-----------------------------------------------------------------------------------------------------|
| `BaseController`       | train, infer (abstract) | Defines the common interface for all controllers.                                                   |
| `PopDEController`      | idem        | Adaptive DE (JADE/SHADE) with self-adaptive F/CR; collects generation successes and updates memory. |
| `RechenbergController` | idem        | Baseline DE with sigma monitoring: injects average strategy parameter (`sigma`) into history.       |
| `NoOpController`       | idem        | Baseline DE without any hyperparameter changes.                                                     |
| `RandomController`     | idem        | Executes purely random actions each step.                                                           |
| `RLController`         | idem        | Wraps a Tensorforce agent; handles save/load, trains & infer with agent.                            |

### `environment/`

| Module / Class                       | Responsibility                                                                                                          |
|--------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| `action_space.py → ActionSpace`      | Turns YAML definitions into action objects and maps agent outputs to real values.                                       |
| `action_types.py`                    | Four kinds of action: 1) `DirectSelectionAction` 2) `OperationBasedAction` 3) `CategoricalAction` 4) `ContinuousAction` |
| `observable.py`                      | Nine metrics – entropy, diversity, improvement, stagnation, …                                                           |
| `observable_factory.py`              | Builds observables by name + config and applies overrides.                                                              |
| `rewards.py`                         | Five reward schemes; default is `combined_increment_with_binary`.                                                       |
| `rl_environment.py → RLEnvironment`  | Glue layer DE ↔ RL: runs generations, builds state, computes reward, checks `done`.                                     |


### `evolutionary_algorithms/`

| File                         | Purpose                                                                                                                |
|------------------------------|------------------------------------------------------------------------------------------------------------------------|
| `differential_evolution.py`  | Full Differential Evolution implementation whose attributes are tweakable online by the agent (`F`, `CR`, strategy …). |
| `evolutionary_strategies.py` | Evolutionary strategies implementation with attributes that can be adjusted by controllers.                            |


### `experiment_manager/`

| File                           | Function                                                                                                                  |
|--------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| `experiment_manager.py`        | Base class: reads YAML, seeds, creates action-space, launches runs, saves histories, logs to MLflow, draws plots.     |
| `experiment_manager_train.py`  | Two-phase subclass: first **train**, then **test** (re-uses the trained agent).                                           |


### `utils/`

| File             | Contents                                                                                              |
|------------------|-------------------------------------------------------------------------------------------------------|
| `constants.py`   | Default limits and string tokens for DE/ES strategies.                                                |
| `utils.py`       | Misc helpers (e.g. `format_differential_weights`).                                                     |
| `utils_results.py`| All MLflow logging, global metrics, single-run and group plots.                                       |

## Configuration Parameters (`configurations/config_train.yaml`)

Below is a complete reference of all parameters in `config_train.yaml`, grouped by section. For each parameter you’ll find its type, default value, description and (where applicable) valid options.

---

### 1. `mlflow`

| Parameter          | Type               | Default                            | Description                                        | Valid options                          |
|--------------------|--------------------|------------------------------------|----------------------------------------------------|----------------------------------------|
| `experiment_name`  | _string_           | `"DE_RL_Tuning_Simple"`            | Name of the MLflow experiment.                     | —                                      |
| `tracking_uri`     | _string_ or `null` | `null`                             | URI where MLflow stores run data.                  | URL (e.g. `"http://localhost:5000"`) or `null` |
| `tags`             | _dict_             | `{ project: "EA_RL", team: "opt-rl" }` | Key–value tags to attach to every run.             | Any key/value pairs                   |

---

### 2. `experiment`

| Parameter             | Type     | Default                      | Description                                                     |
|-----------------------|----------|------------------------------|-----------------------------------------------------------------|
| `random_seed`         | _int_    | `42`                         | Seed for all random number generators.                          |
| `history_path`        | _string_ | `"history_all_simple.pkl"`   | Path to pickle file where training history is saved.            |
| `results_dir`         | _string_ | `"./results_training"`       | Directory where final results are written.                      |
| `runs_per_problem`    | _int_    | `20`                         | Number of runs per optimization problem.            |
| `execution_tag`       | _string_ | `"exp_rl"`                   | Tag to append to log filenames and artifacts.                   |

---

### 3. `problem`

| Parameter   | Type    | Default | Description                                                       |
|-------------|---------|---------|-------------------------------------------------------------------|
| `epsilon`   | _float_ | `1e-3`  | Stopping criterion tolerance (e.g. when Δfitness < ε).            |

---

### 4. `train_problems` & `test_problems`

Each entry in `problems` must have:
- `name` (_string_): Benchmark or objective function identifier.
- `dims` (_array of ints_): Dimensionalities to test.

```yaml
train_problems:
  problems:
    - name: "bbob_2013_attractive_sector"
      dims: [10]
    - name: "schwefel"
      dims: [5, 10, 15]
    # …
test_problems:
  problems:
    - name: "bbob_2013_buche_rastrigin"
      dims: [5, 10]
    - name: "ackley"
      dims: [15]
    # …
```
### 5. `differential_evolution`

| Parameter                         | Type     | Default        | Description                                                 | Valid options                                      |
|-----------------------------------|----------|----------------|-------------------------------------------------------------|----------------------------------------------------|
| `population_size`                 | _int_    | `100`          | Number of candidates in each generation.                    | —                                                  |
| `differential_weights`            | _float_  | `0.8`          | Scale factor for mutation.                                  | any float, typically in the range [0.0, 1.0]       |
| `crossover_probability`           | _float_  | `0.9`          | Probability of recombination.                               | any float, typically in the range [0.0, 1.0]                               |
| `base_index_strategy`             | _string_ | `"rand"`       | How to select the base vector for mutation.                 | `"rand"`, `"better"`, `"target-to-best"`, `"best"` |
| `differential_number`             | _int_    | `2`            | Number of difference vectors to sum.                        | —                                                  |
| `recombination_strategy`          | _string_ | `"binomial"`   | Crossover scheme.                                           | `"binomial"`, `"exponential"`                      |
| `differentials_weights_strategy`  | _string_ | `"given"`      | How weights are assigned (static vs. adaptive).             | `"given"`, `"exponential"`                       |

---

### 6. `action_space`

Defines which DE hyperparameters the RL agent can control, and their action types:

| Field                        | Description                                                                                                                                                                                                                       |
|------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `hyperparameters`            | Map of each tunable hyperparameter with its action space definition.                                                                                                                                                              |
| &nbsp;&nbsp;`<param>.range`  | (for continuous) `[min, max]` interval.                                                                                                                                                                                           |
| &nbsp;&nbsp;`<param>.values` | (for discrete) list of allowed values.                                                                                                                                                                                            |
| `continuous`                 | If this flag is activated action is considered as continuous (ContinuousAction) and not discretized (by default if range is given DirectSelectionAction is considered). |
| `controllable_parameters`    | List of hyperparameter keys the agent is allowed to modify.                                                                                                                                                                       |
| `epsilon`                    | Minimum step size for continuous actions (discretization granularity).                                                                                                                                                            |

#### Example

```yaml
action_space:
  hyperparameters:
    # continuous action: pick F ∈ [0.0, 1.0]
    differential_weights:
      range: [0.0, 1.0]
      continuous: true

    # continuous action: pick CR ∈ [0.0, 1.0]
    crossover_probability:
      range: [0.0, 1.0]
      continuous: true

    # discrete action: choose base index strategy
    base_index_strategy:
      values:
        - "rand"
        - "target-to-best"
        - "best"
      continuous: false

    # discrete action: choose recombination scheme
    recombination_strategy:
      values:
        - "binomial"
        - "exponential"
      continuous: false

  controllable_parameters:
    - differential_weights
    - crossover_probability
    - base_index_strategy
    - recombination_strategy

  epsilon: 0.01
```

### 7. `rl_environment`

| Parameter                            | Type            | Default                             | Description                                                      |
|--------------------------------------|-----------------|-------------------------------------|------------------------------------------------------------------|
| `num_generations_per_iteration`      | _int_           | `10`                                | DE generations per RL decision step.                             |
| `max_generations`                    | _int_           | `5000`                              | Global cap on total generations per problem.                     |
| `max_number_generations_per_episode` | _int_           | `3000`                              | Max generations (steps) per RL episode.                          |
| `reward_type`                        | _string_        | `"combined_increment_with_binary"`  | Reward calculation scheme.                                       |
| `omega`                              | _float_         | `0.5`                               | Weighting factor between max and average reward in population_reward.             |
| `ref_window`                         | _int_           | `5`                                 | Reference window for weighted_improvement_reward.                |
| `time_scale`                         | _float_         | `1.0`                               | Time normalization factor for rewards.                           |
| `observables`                        | _array[string]  | see YAML                            | Metrics the agent observes (e.g. `entropy`, `stagnation`, etc.). |
| `observables_config`                 | _dict_          | see YAML                            | Extra config per observable (e.g. `stagnation.max_stag`).        |

---

### 8. `agent`

| Parameter              | Type               | Default  | Description                                    | Valid options                          |
|------------------------|--------------------|----------|------------------------------------------------|----------------------------------------|
| `type`                 | _string_           | `"ppo"`  | RL algorithm to train.                         | `"ppo"`, `"dqn"`, `"a2c"`, `"ddpg"`, … |
| `config.batch_size`    | _int_              | `32`     | Mini-batch size for policy updates.            | —                                      |
| `config.learning_rate` | _float_            | `0.001`  | Learning rate for optimizer.                   | —                                      |
| `pretrained_dir`       | _string_ or `null` | `null`   | Path to pretrained model directory (if any).   | path or `null`                         |

---

### 9. `controller_type`, `train` & `popde`

| Parameter                        | Type     | Default | Description                                                                                                               |
|----------------------------------|----------|---------|---------------------------------------------------------------------------------------------------------------------------|
| `controller_type`                | _string_ | `"rl"`  | Control strategy:                                                                                                         |
|                                  |          |         | • `"jade"` (JADE)  • `"shade"` (SHADE)  • `"rl"`  • `"random"`  • `"noop"`                                                |
|                                  |          |         | For • `"jade"` (JADE)  • `"shade"` (SHADE)  and `"rechenberg"` need to specify corresponding parameter(s) in action_space |
| `train.episodes_per_problem`     | _int_    | `1`     | Number of RL episodes per training problem.                                                                               |
| `popde.c`                        | _float_  | `0.1`   | Learning-rate factor for mean update (SHADE/JADE only).                                                                   |
| `popde.history_size`             | _int_    | `5`     | Archive size for SHADE; ignored by JADE.                                                                                  |

---

### 10. `algorithm` & `evolutionary_strategies`

| Parameter                                                   | Type      | Default       | Description                                                         | Valid options                                           |
|-------------------------------------------------------------|-----------|---------------|---------------------------------------------------------------------|---------------------------------------------------------|
| `algorithm`                                                 | _string_  | `"de"`        | Evolutionary algorithm to use.                                      | `"de"`, `"es"`                                          |
| `evolutionary_strategies.mu`                                | _int_     | `30`          | Parent population size (μ) for ES.                                  | —                                                       |
| `evolutionary_strategies.lamda`                             | _int_     | `200`         | Number of offspring (λ) per generation.                             | —                                                       |
| `evolutionary_strategies.phro`                              | _int_     | `2`           | Step-size selection parameter (if applicable).                      | —                                                       |
| `evolutionary_strategies.epsilon0`                          | _float_   | `1e-4`        | Initial mutation strength (σ₀).                                      | —                                                       |
| `evolutionary_strategies.tau`                               | _float_   | `0.3`         | Individual sigma adaptation rate.                                   | —                                                       |
| `evolutionary_strategies.tau_prime`                         | _float_   | `0.2`         | Global sigma adaptation rate.                                       | —                                                       |
| `evolutionary_strategies.mutation_steps`                    | _int_     | `1`           | Number of mutation steps per generation.                            | —                                                       |
| `evolutionary_strategies.recombination_individuals_strategy`| _string_  | `"discrete"`  | How parent parameters are combined.                                 | `"discrete"`, `"intermediate"`, `"blended"`             |
| `evolutionary_strategies.recombination_strategy_strategy`   | _string_  | `"averaged_intermediate"` | ES crossover method.                               | `"discrete"`, `"averaged_intermediate"`, …              |
| `evolutionary_strategies.survivor_strategy`                 | _string_  | `"mu+lambda"` | Survivor selection scheme.                                          | `"mu+lambda"`, `"mu,lambda"`                            |
| `evolutionary_strategies.external_sigma_control`            | _bool_    | `true`        | If true, sigma control occurs outside recombination.                | `true`, `false`                                         |


