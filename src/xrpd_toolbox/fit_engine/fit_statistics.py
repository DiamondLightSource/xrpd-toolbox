import numpy as np


def _as_arrays(
    y_calc: np.ndarray, y_obs: np.ndarray, y_obs_error: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    y_obs = np.asarray(y_obs, dtype=float)
    y_calc = np.asarray(y_calc, dtype=float)

    if y_obs.shape != y_calc.shape:
        raise ValueError("y_obs and y_calc must have same shape")

    if y_obs_error is not None:
        y_obs_error = np.asarray(y_obs_error, dtype=float)

        if y_obs_error.shape != y_obs.shape:
            raise ValueError("y_obs_error must have same shape")

    return y_calc, y_obs, y_obs_error


def calculate_weights(
    y_obs: np.ndarray, y_obs_error: np.ndarray | None = None
) -> np.ndarray:
    """
    Rietveld weights.

    If y_obs_error is None:
        w_i = 1 / y_obs_i

    Otherwise:
        w_i = 1 / sigma_i^2
    """

    y_obs = np.asarray(y_obs, dtype=float)

    if y_obs_error is None:
        variance = np.clip(y_obs, 1e-12, None)
    else:
        sigma = np.asarray(y_obs_error, dtype=float)
        variance = np.clip(sigma**2, 1e-12, None)

    return 1.0 / variance


def calculate_chi_squared(
    y_calc: np.ndarray, y_obs: np.ndarray, y_obs_error: np.ndarray | None = None
) -> float:
    """
    Weighted chi-squared:

        χ² = Σ w_i (y_obs_i - y_calc_i)^2

    calculates chi_squared (the minimisation cost function) that is familiar to
    those who do retiveld refinements

    """

    y_calc, y_obs, y_obs_error = _as_arrays(
        y_calc=y_calc,
        y_obs=y_obs,
        y_obs_error=y_obs_error,
    )

    w = calculate_weights(y_obs=y_obs, y_obs_error=y_obs_error)

    residual = y_obs - y_calc

    return float(np.sum(w * residual**2))


def calculate_reduced_chi_squared(
    y_calc: np.ndarray,
    y_obs: np.ndarray,
    n_parameters: int,
    y_obs_error: np.ndarray | None = None,
) -> float:
    """
    Reduced chi-squared:

        χ²_red = χ² / (N - P)

    where:
        N = number of observations
        P = number of refined parameters
    """

    n_obs = len(y_obs)
    dof = n_obs - n_parameters

    if dof <= 0:
        raise ValueError("Degrees of freedom must be positive")

    return (
        calculate_chi_squared(
            y_calc=y_calc,
            y_obs=y_obs,
            y_obs_error=y_obs_error,
        )
        / dof
    )


def calculate_rwp(
    y_calc: np.ndarray, y_obs: np.ndarray, y_obs_error: np.ndarray | None = None
) -> float:
    """
    Weighted profile R-factor:

                    Σ w_i (y_obs_i - y_calc_i)^2
        Rwp = sqrt( -------------------------------- )
                         Σ w_i y_obs_i^2
    """

    y_calc, y_obs, y_obs_error = _as_arrays(
        y_calc=y_calc,
        y_obs=y_obs,
        y_obs_error=y_obs_error,
    )

    w = calculate_weights(y_obs=y_obs, y_obs_error=y_obs_error)

    numerator = np.sum(w * (y_obs - y_calc) ** 2)
    denominator = np.sum(w * y_obs**2)

    return float(np.sqrt(numerator / denominator))


def calculate_rp(
    y_calc: np.ndarray,
    y_obs: np.ndarray,
) -> float:
    """
    Unweighted profile R-factor:

              Σ |y_obs_i - y_calc_i|
        Rp = ------------------------
                  Σ |y_obs_i|
    """

    y_calc, y_obs, _ = _as_arrays(y_calc=y_calc, y_obs=y_obs)

    numerator = np.sum(np.abs(y_obs - y_calc))
    denominator = np.sum(np.abs(y_obs))

    return float(numerator / denominator)


def calculate_expected_r_factor(
    y_obs: np.ndarray,
    n_parameters: int,
    y_obs_error=None,
) -> float:
    """
    Expected R-factor (Rexp):

                     N - P
        Rexp = sqrt(-------)
                    Σ w_i y_obs_i^2
    """

    y_obs = np.asarray(y_obs, dtype=float)

    w = calculate_weights(y_obs=y_obs, y_obs_error=y_obs_error)

    n_obs = len(y_obs)
    dof = n_obs - n_parameters

    if dof <= 0:
        raise ValueError("Degrees of freedom must be positive")

    denominator = np.sum(w * y_obs**2)

    return float(np.sqrt(dof / denominator))


def calculate_goodness_of_fit(
    y_calc: np.ndarray,
    y_obs: np.ndarray,
    n_parameters: int,
    y_obs_error: np.ndarray | None = None,
) -> float:
    """
    Goodness-of-fit (GoF), also called chi:

               Rwp
        GoF = -----
               Rexp

        equivalently:

        GoF = sqrt(reduced chi-squared)
    """

    return calculate_rwp(
        y_calc=y_calc, y_obs=y_obs, y_obs_error=y_obs_error
    ) / calculate_expected_r_factor(
        y_obs=y_obs,
        n_parameters=n_parameters,
        y_obs_error=y_obs_error,
    )
