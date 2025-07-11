import math
from typing import Callable, Dict, List, Tuple

import numpy as np
from numpy.linalg import qr

from benchmarking.optimization_problem import OptimizationProblem

# ————— Classical problems —————
_basic_configs = {
    "sphere": {
        "func": lambda x: np.sum(x**2),
        "default_dim": 10,
        "default_bounds": (-5.12, 5.12),
        "opt": 0.0,
        "opt_point": lambda dim: np.zeros(dim),
    },
    "noisy_sphere": {
        "func": lambda x: np.sum(x**2 + 1e-3 * np.sin(10 * x)),
        "default_dim": 10,
        "default_bounds": (-5, 5),
        "opt": 0.0,
        "opt_point": lambda dim: np.zeros(dim),
    },
    "ackley": {
        "func": lambda x: (
            -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / len(x)))
            - np.exp(np.sum(np.cos(2 * np.pi * x)) / len(x))
            + 20.0
            + np.e
        ),
        "default_dim": 10,
        "default_bounds": (-5, 5),
        "opt": 0.0,
        "opt_point": lambda dim: np.zeros(dim),
    },
    "rosenbrock": {
        "func": lambda x: np.sum(
            100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2
        ),
        "default_dim": 10,
        "default_bounds": (-2.048, 2.048),
        "opt": 0.0,
        "opt_point": lambda dim: np.ones(dim),
    },
    "rastrigin": {
        "func": lambda x: 10.0 * len(x) + np.sum(x**2 - 10.0 * np.cos(2 * np.pi * x)),
        "default_dim": 10,
        "default_bounds": (-5.12, 5.12),
        "opt": 0.0,
        "opt_point": lambda dim: np.zeros(dim),
    },
    "schaffer": {
        "func": lambda x: np.sum(
            (x[:-1] ** 2 + x[1:] ** 2) ** 0.25
            * (np.sin(50 * (x[:-1] ** 2 + x[1:] ** 2) ** 0.1) ** 2 + 1)
        ),
        "default_dim": 10,
        "default_bounds": (-100, 100),
        "opt": 0.0,
        "opt_point": lambda dim: np.zeros(dim),
    },
    "schwefel": {
        "func": lambda x: 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x)))),
        "default_dim": 10,
        "default_bounds": (-500, 500),
        "opt": 0.0,
        "opt_point": lambda dim: np.full(dim, 420.9687),
    },
    "griewank": {
        "func": lambda x: (
            np.sum(x**2) / 4000.0
            - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
            + 1.0
        ),
        "default_dim": 10,
        "default_bounds": (-600, 600),
        "opt": 0.0,
        "opt_point": lambda dim: np.zeros(dim),
    },
}

# ————— BBOB 2013 problems —————


def generate_bbob_instance(dim: int, seed: int) -> dict:
    """
    Generate a BBOB instance containing an optimal solution and its function value.

    Args:
        dim (int): The dimensionality of the problem.
        seed (int): Seed for the random number generator.

    Returns:
        dict: A dictionary with keys:
            - "x_opt" (np.ndarray): The optimal vector sampled uniformly from [-4, 4]^dim.
            - "f_opt" (float): The optimal function value, drawn from a Cauchy distribution
              (rounded to two decimals and clipped to ±1000).
    """
    rng = np.random.RandomState(seed)
    x_opt = rng.uniform(low=-4.0, high=4.0, size=(dim,))
    raw_cauchy = rng.standard_cauchy()
    f_opt = float(np.round(raw_cauchy, 2))
    if abs(f_opt) > 1000.0:
        f_opt = np.sign(f_opt) * 1000.0
    return {"x_opt": x_opt, "f_opt": f_opt}


def f_penalty(x: np.ndarray) -> float:
    """
    Compute a quadratic penalty for components of x outside the range [-5, 5].

    Args:
        x (np.ndarray): Input vector.

    Returns:
        float: Sum of squared violations max(0, |x_i| - 5)^2 for all components.
    """
    return np.sum(np.maximum(0.0, np.abs(x) - 5.0) ** 2)


def generate_rotation_matrix(dim: int, seed: int = None) -> np.ndarray:
    """
    Create a random orthonormal rotation matrix of size (dim x dim).

    Args:
        dim (int): Dimension of the square matrix.
        seed (int, optional): Seed for the random number generator. Defaults to None.

    Returns:
        np.ndarray: A dim-by-dim orthonormal rotation matrix.
    """
    rng = np.random.RandomState(seed)
    A = rng.normal(size=(dim, dim))
    Q, _ = qr(A)
    return Q


def T_osz(x: np.ndarray) -> np.ndarray:
    """
    Apply the oscillatory transformation T_osz to each component of x.

    The transform introduces nonlinearity via sine perturbations on the log scale.

    Args:
        x (np.ndarray): Input vector.

    Returns:
        np.ndarray: Transformed vector of the same shape as x.
    """
    z = np.zeros_like(x, dtype=float)
    for i, xi in enumerate(x):
        if xi != 0.0:
            x_hat = math.log(abs(xi))
        else:
            x_hat = 0.0
        if xi > 0.0:
            c1, c2, sgn = 10.0, 7.9, 1.0
        elif xi < 0.0:
            c1, c2, sgn = 5.5, 3.1, -1.0
        else:
            c1 = c2 = 0.0
            sgn = 0.0
        z[i] = sgn * math.exp(
            x_hat + 0.049 * (math.sin(c1 * x_hat) + math.sin(c2 * x_hat))
        )
    return z


def T_asy(x: np.ndarray, beta: float = 0.2) -> np.ndarray:
    """
    Apply the asymmetric transformation T_asy to the input vector.

    Positive components are raised to a power that increases linearly with their index.

    Args:
        x (np.ndarray): Input vector.
        beta (float, optional): Asymmetry parameter. Defaults to 0.2.

    Returns:
        np.ndarray: Transformed vector with the same shape as x.
    """
    D = len(x)
    z = np.zeros_like(x, dtype=float)
    for i, xi in enumerate(x):
        if xi > 0.0:
            exponent = 1.0 + beta * (i / (D - 1)) * math.sqrt(xi)
            z[i] = xi**exponent
        else:
            z[i] = xi
    return z


def bbob2013_ellipsoidal(
    x: np.ndarray, x_opt: np.ndarray, f_opt: float, seed: int
) -> float:
    """
    Compute the BBOB 2013 ellipsoidal (rotated) function value.

    Steps:
      1. Shift x by the optimal vector x_opt.
      2. Apply a random rotation.
      3. Apply the oscillatory transform T_osz.
      4. Weight squared components by an exponential scale.
      5. Add a boundary penalty and the base optimum value.

    Args:
        x (np.ndarray): Input vector.
        x_opt (np.ndarray): Optimal shift vector.
        f_opt (float): Base optimal function value.
        seed (int): Seed for the rotation matrix.

    Returns:
        float: The function value f(x) = sum(weights * z_i^2) + 1e6 * penalty(x) + f_opt.
    """
    D = len(x)
    x_shifted = x - x_opt
    R = generate_rotation_matrix(D, seed=seed)
    y = R.dot(x_shifted)
    z = T_osz(y)
    exponents = np.linspace(0, 1, D)
    weights = 10 ** (6 * exponents)
    obj = np.sum(weights * (z**2))
    return obj + 1e6 * f_penalty(x) + f_opt


def bbob2013_buche_rastrigin(
    x: np.ndarray, x_opt: np.ndarray, f_opt: float, seed: int
) -> float:
    """
    Büche-Rastrigin (BBOB f4).

    z_i = s_i * T_osz(x_i - x_opt_i)
    f(x) = 10 * (D − Σ cos(2π z_i)) + Σ z_i² + 100 * f_penalty(x) + f_opt
    """
    D = len(x)

    x_shifted = x - x_opt

    y = T_osz(x_shifted)

    exponents = 0.5 * np.arange(D) / (D - 1)
    s = 10**exponents
    odd_idx = np.arange(D) % 2 == 0
    pos_idx = y > 0
    s[odd_idx & pos_idx] *= 10

    z = s * y

    obj = 10.0 * (D - np.cos(2.0 * math.pi * z).sum()) + np.square(z).sum()
    obj += 100.0 * f_penalty(x)

    return obj + f_opt


def bbob2013_attractive_sector(
    x: np.ndarray, x_opt: np.ndarray, f_opt: float, seed: int
) -> float:
    """
    Corrected BBOB 2013 Attractive Sector (F6).

      f6(x) = T_osz( [ Σ_{i=1}^D ( s_i * z_i )^2 ]^0.9 )  +  f_opt

    where
      • z = Q · Λ^{10} · R · (x – x_opt)
           – R, Q are two independent orthonormal matrices
             (seeded by `seed` and `seed+1`, respectively).
           – Λ^{10} is diagonal with entries α = 10:
                Λ^{10}_{ii} = 10^{\, (i-1)/(D–1) }, for i = 1…D.
      • s_i = 100 if (z_i * x_opt_i) > 0, else 1.
      • T_osz is applied to the single‐value u = [ Σ (s_i · z_i)^2 ]^0.9
        by wrapping u into a length‐1 array, then unwrapping.

    No boundary‐penalty term is added (the official definition does not include f_penalty
    for the Attractive Sector function).

    Args:
      x (np.ndarray): Input vector of length D.
      x_opt (np.ndarray): Shift (optimum location), also length D.
      f_opt (float): Known optimal function value.
      seed (int): Seed used to generate R (with `seed`) and Q (with `seed + 1`).

    Returns:
      float: Value of the BBOB Attractive Sector function at x.
    """
    D = len(x)

    x_shifted = x - x_opt

    R = generate_rotation_matrix(D, seed=seed)
    Q = generate_rotation_matrix(D, seed=seed + 1)

    exponents = np.linspace(0.0, 1.0, D)  # [0, 1/(D–1), 2/(D–1), …, 1]
    lam10_diag = 10.0**exponents  # 10^{(i–1)/(D–1)}
    Lambda10 = np.diag(lam10_diag)  # shape (D, D)

    y1 = R.dot(x_shifted)  # apply R
    y2 = Lambda10.dot(y1)  # apply Λ^{10}
    z = Q.dot(y2)  # apply Q

    s = np.where(z * x_opt > 0.0, 100.0, 1.0)  # length D
    inner = float(np.sum((s * z) ** 2))
    u = inner**0.9
    t_val = float(T_osz(np.array([u]))[0])
    return t_val + f_opt


def bbob2013_discus(x: np.ndarray, x_opt: np.ndarray, f_opt: float, seed: int) -> float:
    """
    Corrected BBOB 2013 Discus Function (F11):

      f(x) = 10^6 * z_1^2 + ∑_{i=2}^D z_i^2 + f_opt,

    where:
      - z = T_osz( R @ (x - x_opt) )
      - R is a random orthonormal matrix (seeded by `seed`)

    Args:
        x (np.ndarray): Input vector of length D.
        x_opt (np.ndarray): Shift (optimum location), length D.
        f_opt (float): Known optimum function value.
        seed (int): Seed for generating the orthonormal matrix R.

    Returns:
        float: Function value of the Discus function at x.
    """
    D = len(x)
    x_shifted = x - x_opt
    R = generate_rotation_matrix(D, seed=seed)
    y = R.dot(x_shifted)
    z = T_osz(y)
    obj = 1e6 * (z[0] ** 2) + np.sum(z[1:] ** 2)
    return obj + f_opt


def bbob2013_bent_cigar(
    x: np.ndarray, x_opt: np.ndarray, f_opt: float, seed: int
) -> float:
    """
    Corrected BBOB 2013 Bent Cigar Function (F12):

      f(x) = z_1^2 + 10^6 * ∑_{i=2}^D z_i^2 + f_opt,

    where:
      - z = R2 @ T_asy^{(β=0.5)}( R1 @ (x - x_opt) )
      - R1, R2 are two independent random orthonormal matrices
        (seeded by `seed` and `seed+1` respectively)
      - T_asy uses β = 0.5

    Args:
        x (np.ndarray): Input vector of length D.
        x_opt (np.ndarray): Shift (optimum location), length D.
        f_opt (float): Known optimum function value.
        seed (int): Seed for generating R1 and R2.

    Returns:
        float: Function value of the Bent Cigar function at x.
    """
    D = len(x)

    x_shifted = x - x_opt
    R1 = generate_rotation_matrix(D, seed=seed)
    R2 = generate_rotation_matrix(D, seed=seed + 1)
    y = R1.dot(x_shifted)
    y2 = T_asy(y, beta=0.5)
    z = R2.dot(y2)
    obj = (z[0] ** 2) + 1e6 * np.sum(z[1:] ** 2)

    return obj + f_opt


def bbob2013_sharp_ridge(
    x: np.ndarray, x_opt: np.ndarray, f_opt: float, seed: int
) -> float:
    """
    Corrected BBOB 2013 Sharp Ridge Function (F13).

      f13(x) = z₁² + 100 * sqrt( ∑_{i=2}^D zᵢ² ) + f_opt

    where
      • z = Q · Λ^{10} · R · (x – x_opt)
        – R, Q are two independent orthonormal matrices
          (seeded by `seed` and `seed + 1`, respectively)
        – Λ^{10} is diagonal with entries
            Λ^{10}_{ii} = 10^((i–1)/(D–1)), for i = 1…D

    Args:
        x (np.ndarray): Input vector of length D.
        x_opt (np.ndarray): Shift (optimum location), also length D.
        f_opt (float): Known optimum function value.
        seed (int): Seed used to generate R (with `seed`) and Q (with `seed + 1`).

    Returns:
        float: The Sharp Ridge function value at x.
    """
    D = len(x)

    # 1) shift
    x_shifted = x - x_opt

    # 2) generate two independent rotations R and Q
    R = generate_rotation_matrix(D, seed=seed)
    Q = generate_rotation_matrix(D, seed=seed + 1)

    # 3) build diagonal matrix Λ^{10} with entries 10^((i-1)/(D-1))
    exponents = np.linspace(0.0, 1.0, D)  # [0, 1/(D-1), 2/(D-1), …, 1]
    lam10_diag = 10.0**exponents  # shape (D,), lam10_diag[i] = 10^((i)/(D-1))
    Lambda10 = np.diag(lam10_diag)  # shape (D, D)

    # 4) compute z = Q @ Λ^{10} @ R @ (x - x_opt)
    y1 = R.dot(x_shifted)  # apply R
    y2 = Lambda10.dot(y1)  # apply Λ^{10}
    z = Q.dot(y2)  # apply Q

    # 5) compute objective: z[0]^2 + 100 * sqrt(sum_{i=2}^D z[i]^2)
    tail_norm = math.sqrt(np.sum(z[1:] ** 2))
    obj = (z[0] ** 2) + 100.0 * tail_norm

    # 6) add f_opt
    return obj + f_opt


def bbob2013_different_powers(
    x: np.ndarray, x_opt: np.ndarray, f_opt: float, seed: int
) -> float:
    """
    Corrected BBOB 2013 Different Powers Function (F14).

      f14(x) = sqrt( Σ_{i=1}^D | z_i |^{ 2 + 4 * (i-1)/(D-1) } )  +  f_opt

    where
      • z = R · (x – x_opt),
        and R is a random orthonormal matrix (seeded by `seed`).
      • The exponent for component i is 2 + 4·(i–1)/(D–1), for i=1…D.

    No boundary‐penalty term is included (per official BBOB definition).

    Args:
        x (np.ndarray): Input vector of length D.
        x_opt (np.ndarray): Shift (optimum location), length D.
        f_opt (float): Known optimum function value.
        seed (int): Seed for generating the rotation matrix R.

    Returns:
        float: Value of the Different Powers function at x.
    """
    D = len(x)

    x_shifted = x - x_opt

    R = generate_rotation_matrix(D, seed=seed)

    z = R.dot(x_shifted)

    exponents = 2.0 + 4.0 * np.linspace(
        0.0, 1.0, D
    )  # yields 2, 2+4/(D-1), 2+8/(D-1), …, 6

    powered = np.abs(z) ** exponents
    sum_val = float(np.sum(powered))

    obj = math.sqrt(sum_val)

    return obj + f_opt


_fn_name_map = {
    "bbob_2013_ellipsoidal": bbob2013_ellipsoidal,
    "bbob_2013_buche_rastrigin": bbob2013_buche_rastrigin,
    "bbob_2013_attractive_sector": bbob2013_attractive_sector,
    "bbob_2013_discus": bbob2013_discus,
    "bbob_2013_bent_cigar": bbob2013_bent_cigar,
    "bbob_2013_sharp_ridge": bbob2013_sharp_ridge,
    "bbob_2013_different_powers": bbob2013_different_powers,
}

_seeds: Dict[str, int] = {
    "bbob_2013_ellipsoidal": 100,
    "bbob_2013_buche_rastrigin": 101,
    "bbob_2013_attractive_sector": 102,
    "bbob_2013_discus": 103,
    "bbob_2013_bent_cigar": 104,
    "bbob_2013_sharp_ridge": 105,
    "bbob_2013_different_powers": 106,
}

_bounds_overrides: Dict[str, Tuple[float, float]] = {}
_default_dim_bbob = 10

_bbob_2013_configs: Dict[str, dict] = {}

for name, fn in _fn_name_map.items():
    seed = _seeds[name]
    inst = generate_bbob_instance(_default_dim_bbob, seed)
    x_opt = inst["x_opt"]
    f_opt = inst["f_opt"]
    lb, ub = _bounds_overrides.get(name, (-5.0, 5.0))

    def make_objective(func, x_opt_vec, f_opt_val, s):
        def objective(x: np.ndarray) -> float:
            return func(x, x_opt=x_opt_vec, f_opt=f_opt_val, seed=s)

        return objective

    bbob_callable = make_objective(fn, x_opt, f_opt, seed)

    _bbob_2013_configs[name] = {
        "func": bbob_callable,
        "default_dim": _default_dim_bbob,
        "default_bounds": (lb, ub),
        "opt": f_opt,
        "opt_point": (lambda xo=x_opt: (lambda d: xo.copy())),
        "x_opt": x_opt,
        "seed": seed,
    }


_ALL_PROBLEM_CONFIGS = {**_basic_configs, **_bbob_2013_configs}


def create_problems(
    names: List[str],
    epsilon: float = 1e-4,
    override_dims: Dict[str, int] = None,
    override_bounds: Dict[str, Tuple[float, float]] = None,
) -> Dict[str, OptimizationProblem]:
    """
    Build a dictionary of OptimizationProblem instances from a list of names.

    If a name starts with "bbob_2013_", we
    regenerate x_opt and f_opt at exactly the requested dimension.
    """
    override_dims = override_dims or {}
    override_bounds = override_bounds or {}
    result: Dict[str, OptimizationProblem] = {}

    for pname in names:
        # ——— Classical problems ———
        if pname in _basic_configs:
            entry = _basic_configs[pname]
            dim = override_dims.get(pname, entry["default_dim"])
            lb, ub = override_bounds.get(pname, entry["default_bounds"])
            func: Callable[[np.ndarray], float] = entry["func"]
            f_opt = float(entry["opt"])
            bounds_array = np.tile((lb, ub), (dim, 1))

            opt = OptimizationProblem(
                objective_function=func,
                bounds=bounds_array,
                maximize=False,
                target_value=f_opt,
                epsilon=float(epsilon),
            )
            result[pname] = opt
            continue

        # ——— BBOB‐2013 problems ———
        if pname.startswith("bbob_2013_"):
            # decide final dimension
            dim = override_dims.get(pname, _default_dim_bbob)
            lb, ub = override_bounds.get(
                pname, _bounds_overrides.get(pname, (-5.0, 5.0))
            )

            # pick a seed
            seed = _seeds.get(pname, 0)

            # regenerate x_opt, f_opt at this dimension
            inst = generate_bbob_instance(dim, seed)
            x_opt = inst["x_opt"]
            f_opt = inst["f_opt"]

            # pick correct base function
            base_fn = _fn_name_map.get(pname, None)

            if base_fn is not None:
                # wrap it so that x_opt/f_opt/seed are baked in
                def make_objective(func, x_opt_vec, f_opt_val, s):
                    def objective(x: np.ndarray) -> float:
                        return func(x, x_opt=x_opt_vec, f_opt=f_opt_val, seed=s)

                    return objective

                bbob_callable = make_objective(base_fn, x_opt, f_opt, seed)
            else:
                # If unimplemented (e.g. weierstrass was None in map), use a dummy stub:
                def bbob_callable(x: np.ndarray) -> float:
                    return 0.0

            bounds_array = np.tile((lb, ub), (dim, 1))
            opt = OptimizationProblem(
                objective_function=bbob_callable,
                bounds=bounds_array,
                maximize=False,
                target_value=float(f_opt),
                epsilon=float(epsilon),
                x_opt=x_opt,
            )
            result[pname] = opt
            continue

        # ——— Unknown problem name ———
        raise ValueError(f"Unknown problem name '{pname}' in create_problems().")

    return result
