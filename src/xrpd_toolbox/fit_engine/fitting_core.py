"""Refinement utilities for XRPD model fitting.

This module provides an interface for refining parameterised models
against observed data using SciPy optimizers. It supports both
least-squares and general minimization methods, with optional live plotting.
"""

from __future__ import annotations

import time
from abc import abstractmethod
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    Literal,
    Self,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy import optimize

from xrpd_toolbox.core import (
    Parameter,
    ParameterArray,
    ScatteringData,
    XRPDBaseModel,
    XYEData,
)
from xrpd_toolbox.utils.utils import calculate_chi_squared

# TODO: Expand this to other methods: simulated annealing etc, AI stuff?
RefineMethod = Literal[
    "least_squares",
    "trf",
    "dogbox",
    "lm",
    "nelder-mead",
    "powell",
    "cg",
    "bfgs",
    "l-bfgs-b",
    "tnc",
    "cobyla",
    "slsqp",
    "trust-constr",
    "dogleg",
    "trust-ncg",
    "trust-exact",
    "trust-krylov",
]


def is_parameter_like(annotation):
    return Parameter in get_args(annotation)


def is_number_like(annotation) -> bool:
    origin = get_origin(annotation)

    if annotation in (float, int):
        return True

    if origin is Union:
        return any(is_number_like(arg) for arg in get_args(annotation))

    return False


# class RefinementBaseModel(XRPDBaseModel, extra="allow"):
# ^ dont do this unless you want serualisation to get messy
class RefinementBaseModel(XRPDBaseModel):
    """
    This class is use to contain Parameters, on it's own this class cannot be refined.

    In the RefinementBaseModel ANYTHING that is a Parameter can be refined.
    eg. Therefore if you set cubic lattice angles to refine, you will break symmetry.
    This requires the user to know what they're doing.
    With great power comes great reposibility"""

    def get_bounds_from_metadata(self, metadata):
        lower = None
        upper = None

        for meta in metadata:
            ge = getattr(meta, "ge", None)
            gt = getattr(meta, "gt", None)
            le = getattr(meta, "le", None)
            lt = getattr(meta, "lt", None)

            if gt is not None:
                lower = float(gt)
            elif ge is not None and lower is None:
                lower = float(ge)

            if lt is not None:
                upper = float(lt)
            elif le is not None and upper is None:
                upper = float(le)

        if lower is None:
            lower = -np.inf
        if upper is None:
            upper = np.inf

        bounds = [float(lower), float(upper)]
        return bounds

    def parameterise_all(self, refine: bool = False):
        for name, field_info in type(self).model_fields.items():
            if is_parameter_like(annotation=field_info.annotation):
                field = getattr(self, name)

                if not isinstance(field, Parameter):
                    bounds = self.get_bounds_from_metadata(field_info.metadata)

                    setattr(
                        self,
                        name,
                        Parameter(value=float(field), refine=refine, bounds=bounds),
                    )

    def deparameterise(self, value, parent, key):
        print(key, value, type(value))

        if isinstance(value, Parameter):
            setattr(parent, key, float(value.value))
            return

        if isinstance(value, RefinementBaseModel):
            for sub_name in type(value).model_fields:
                self.deparameterise(
                    getattr(value, sub_name),
                    value,
                    sub_name,
                )
            return

        if isinstance(value, (tuple, list)):
            for i, obj in enumerate(value):
                self.deparameterise(obj, value, i)
            return

    def deparameterise_all(self):
        """Turns things that maybe a Parmaeter into a float,
        such that it can be more easily serialised"""

        for name in type(self).model_fields:
            self.deparameterise(getattr(self, name), self, name)

    def path_to_string(self, path) -> str:
        out = []
        for p in path:
            if isinstance(p, int):
                out[-1] = f"{out[-1]}[{p}]"
            else:
                out.append(p)
        return ".".join(out)

    def iter_parameters(self, prefix=()):
        fields = type(self).model_fields

        for name in fields:
            value = getattr(self, name)
            path = prefix + (name,)

            if isinstance(value, Parameter):
                yield path, value
                continue

            elif isinstance(value, ParameterArray):
                for i, p in enumerate(value.parameter_array):
                    subpath = path + (i,)
                    yield subpath, p

            elif isinstance(value, RefinementBaseModel):
                yield from value.iter_parameters(path)

            elif isinstance(value, (list, tuple)):
                for i, v in enumerate(value):
                    subpath = path + (i,)

                    if isinstance(v, Parameter):
                        yield subpath, v
                    elif isinstance(v, RefinementBaseModel):
                        yield from v.iter_parameters(subpath)

            elif isinstance(value, dict):
                for k, v in value.items():
                    subpath = path + (k,)

                    if isinstance(v, Parameter):
                        yield subpath, v
                    elif isinstance(v, RefinementBaseModel):
                        yield from v.iter_parameters(subpath)

    def model_post_init(self, __context: Any):
        params = self._collect_parameters()
        self._bind_parameters(params)

    def _collect_parameters(self) -> dict[str, Parameter]:
        params: dict[str, Parameter] = {}

        for name, value in self.__dict__.items():
            if isinstance(value, Parameter):
                params[name] = value

        return params

    def _bind_parameters(self, params: dict) -> None:
        for name, param in params.items():
            param._name = name  # noqa
            param._model = self  # noqa

    # def iter_parameters(self, prefix=()):
    #     """this version is better - but breaks if Paramater aren't fixed"""

    #     stack = deque([(prefix, self)])

    #     while stack:
    #         path, value = stack.pop()

    #         # --- Parameter (fast path, most important) ---
    #         if isinstance(value, Parameter):
    #             yield path, value
    #             continue

    #         # --- Model ---
    #         if isinstance(value, RefinementBaseModel):
    #             for name in value.model_fields:
    #                 stack.append((path + (name,), getattr(value, name)))
    #             continue

    #         # --- Mapping ---
    #         if isinstance(value, dict):
    #             for k, v in value.items():  # type: ignore - this is impossible to not have items # noqa
    #                 stack.append((path + (k,), v))
    #             continue

    #         # --- Iterable containers (lists, tuples, ParameterArray, etc.) ---
    #         if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
    #             # special-case: ParameterArray exposes internal list
    #             if hasattr(value, "parameter_array"):
    #                 value = value.parameter_array

    #             for i, v in enumerate(value):
    #                 stack.append((path + (i,), v))

    def get_refinement_parameters(self) -> dict:
        result = {}
        seen = set()

        for path, p in self.iter_parameters():
            if not p.refine:
                continue

            pid = id(p)

            # this bit can be removed after I have fixed how parameter stores equiv vals
            if pid in seen:
                continue

            seen.add(pid)

            # Only build string here (once)
            key = self.path_to_string(path)
            result[key] = p.value

        return result

    def get_param_by_path(self, path: str) -> Parameter:
        obj = self
        tokens = path.split(".")

        for token in tokens:
            if "[" in token:
                name, rest = token.split("[", 1)
                obj = getattr(obj, name)
                obj = obj[int(rest[:-1])]  # strip trailing ']'
            else:
                obj = getattr(obj, token)

        if not isinstance(obj, Parameter):
            raise TypeError(f"{path} does not resolve to Parameter")

        return obj

    def set_refinement_parameters(self, values: dict[str, float]):
        # Build mapping once
        path_map = {}
        seen = set()

        add_seen = seen.add

        for path, p in self.iter_parameters():
            if not p.refine:
                continue

            pid = id(p)
            if pid in seen:
                continue

            add_seen(pid)
            path_map[self.path_to_string(path)] = p

        # Apply updates
        for key, val in values.items():
            path_map[key].value = float(val)

    def refine_none(self, keep_refined: list[str] | None = None):
        if keep_refined is None:
            keep_refined = []

        for name, param in self.iter_parameters():
            if name[-1] in keep_refined:
                continue
            else:
                param.refine = False

    def refine_all(self, keep_fixed: list[str] | None = None):
        if keep_fixed is None:
            keep_fixed = []

        for name, param in self.iter_parameters():
            if name[-1] in keep_fixed:
                continue
            else:
                param.refine = True


ModelDataVar = TypeVar("ModelDataVar", XYEData, ScatteringData)


class Model(RefinementBaseModel, Generic[ModelDataVar]):
    """A model can be refined by the refiner. It must contain:
    data and a way to calculate_profile

    This class should contain RefinementBaseModel classes or Parameters/Parameter arrays
    """

    data: ModelDataVar

    @abstractmethod
    def calculate_profile(self):
        raise NotImplementedError("Must implmenet calculate_profile for Model subclass")

    def refine(
        self,
        method: RefineMethod = "least_squares",
        *,
        copy_model: bool = False,
        bounds: Sequence[tuple[float, float]] | None = None,
        plot: bool = False,
        plot_every: int = 10,
        step_time: float | None = None,
        max_nfev: int = 1000,
        maxiter: int | None = None,
        callback: Callable[[np.ndarray], None] | None = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> tuple[dict[str, float], Self, optimize.OptimizeResult]:
        """This optimises the model"""

        return refine_model(
            self,
            method=method,
            copy_model=copy_model,
            bounds=bounds,
            plot=plot,
            plot_every=plot_every,
            step_time=step_time,
            max_nfev=max_nfev,
            maxiter=maxiter,
            callback=callback,
            verbose=verbose,
            **kwargs,
        )


ModelType = TypeVar("ModelType", bound=Model)


@dataclass
class PlotState:
    fig: Figure
    ax: Axes
    lines: tuple[Line2D, Line2D, Line2D]
    canvas: FigureCanvasAgg
    background: Any
    offset: float


def setup_plot(model: Model) -> PlotState:
    plt.ion()
    fig, ax = plt.subplots()

    x = model.data.x
    y_obs = model.data.y
    y_calc = model.calculate_profile()
    offset = -0.1 * np.max(y_obs)

    (line_obs,) = ax.plot(
        x,
        y_obs,
        linestyle="none",
        marker=".",
        color="black",
        label="Observed",
    )
    (line_calc,) = ax.plot(x, y_calc, color="red", label="Calculated")
    (line_diff,) = ax.plot(
        x,
        y_obs - y_calc + offset,
        color="blue",
        label="Obs - Calc",
    )

    ax.legend()

    x_unit = getattr(model.data, "x_unit", "index")
    ax.set_xlabel(x_unit)
    ax.set_ylabel("Intensity")

    ax.relim()

    canvas = cast(FigureCanvasAgg, fig.canvas)
    canvas.draw()
    background = canvas.copy_from_bbox(ax.bbox)

    return PlotState(
        fig, ax, (line_obs, line_calc, line_diff), canvas, background, offset
    )


def update_plot(model: Model, plot_state: PlotState) -> None:
    line_obs, line_calc, line_diff = plot_state.lines

    y_obs = model.data.y
    y_calc = model.calculate_profile()

    line_calc.set_ydata(y_calc)
    line_diff.set_ydata(y_obs - y_calc + plot_state.offset)

    plot_state.ax.relim()
    plot_state.ax.autoscale_view()

    # Draw axes frame after rescaling
    plot_state.canvas.draw()

    # Explicitly draw animated artists (lines)
    plot_state.ax.draw_artist(line_obs)
    plot_state.ax.draw_artist(line_calc)
    plot_state.ax.draw_artist(line_diff)
    plot_state.canvas.blit(plot_state.ax.bbox)

    plot_state.canvas.flush_events()


def _get_refinable_parameters(model: Model) -> list[tuple[str, Parameter]]:
    seen_ids: set[int] = set()
    refinable: list[tuple[str, Parameter]] = []

    for path, parameter in model.iter_parameters():
        if not parameter.refine:
            continue

        if id(parameter) in seen_ids:
            continue

        if not isinstance(parameter.value, (int, float, np.integer, np.floating)):
            continue

        seen_ids.add(id(parameter))
        refinable.append((".".join(str(token) for token in path), parameter))

    return refinable


def _build_parameter_vectors(
    parameters: Sequence[tuple[str, Parameter]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x0 = np.empty(len(parameters), dtype=float)
    lower = np.empty(len(parameters), dtype=float)
    upper = np.empty(len(parameters), dtype=float)

    for index, (_, parameter) in enumerate(parameters):
        x0[index] = float(parameter.value)
        lower[index], upper[index] = parameter.bounds

    return x0, lower, upper


def _update_parameters(
    parameters: Sequence[tuple[str, Parameter]], values: np.ndarray
) -> None:
    for index, (_, parameter) in enumerate(parameters):
        parameter.value = float(values[index])


def _default_plot_callback(plot_state: PlotState, model: Model, interval: int):
    counter = {"step": 0}

    def callback() -> None:
        counter["step"] += 1
        if counter["step"] % interval == 0:
            update_plot(model, plot_state)

    return callback


def refine_model(
    model: ModelType,
    method: RefineMethod = "least_squares",
    *,
    copy_model: bool = False,
    bounds: Sequence[tuple[float, float]] | None = None,
    plot: bool = False,
    plot_every: int = 10,
    step_time: float | None = None,
    max_nfev: int = 1000,
    maxiter: int | None = None,
    callback: Callable[[np.ndarray], None] | None = None,
    verbose: bool = True,
    **kwargs: Any,
) -> tuple[dict[str, float], ModelType, optimize.OptimizeResult]:
    """Refine a Model using SciPy optimization routines.

    Parameters
    ----------
    model : Model
        Model containing `data`, `calculate_profile`, and refinable parameters.
    method : RefineMethod, optional
        Optimization method. ``least_squares`` is the default.
    copy_model : bool, optional
        If True, the input model is deep-copied before refinement.
    bounds : Optional[Sequence[Tuple[float, float]]], optional
        Optional parameter bounds for ``minimize``. If not provided, parameter
        bounds are inferred from the model's Parameter objects.
    plot : bool, optional
        If True, show a live refinement plot.
    plot_every : int, optional
        Update the plot every N iterations.
    step_time : Optional[float], optional
        Time in seconds to pause during each optimization evaluation.
    max_nfev : int, optional
        Maximum function evaluations for least-squares optimization.
    maxiter : Optional[int], optional
        Maximum iterations for general minimization.
    callback : Optional[Callable[[np.ndarray], None]], optional
        Optional callback invoked with the current parameter vector.
    verbose : bool, optional
        If True (default), print chi-squared and parameter values after each evaluation.
    **kwargs : Any
        Additional optimizer-specific keyword arguments.

    Returns
    -------
    tuple[dict[str, float], ModelType, optimize.OptimizeResult]
        A tuple containing updated parameter values, the refined model,
        and the SciPy optimization result.
    """
    optimized_model = deepcopy(model) if copy_model else model
    parameters = _get_refinable_parameters(optimized_model)

    if not parameters:
        raise ValueError("No refinable parameters found")

    x0, lower, upper = _build_parameter_vectors(parameters)
    y_obs = np.asarray(optimized_model.data.y, dtype=float)

    plot_state: PlotState | None = None
    plot_callback: Callable[[], None]

    if plot:
        plot_state = setup_plot(optimized_model)
        plot_callback = _default_plot_callback(plot_state, optimized_model, plot_every)
    else:

        def _no_plot() -> None:
            pass

        plot_callback = _no_plot

    def _evaluate(x: np.ndarray) -> np.ndarray:
        _update_parameters(parameters, x)
        if step_time is not None and step_time > 0:
            time.sleep(step_time)

        if callback is not None:
            callback(x)

        plot_callback()
        y_calc = optimized_model.calculate_profile()
        residuals = y_obs - np.asarray(y_calc, dtype=float)

        chi_squared = calculate_chi_squared(y_calc, model.data.y, model.data.e)
        print(f"Chi-squared: {chi_squared}")

        if verbose:
            for path, param in parameters:
                value = param.value

                if abs(float(value)) >= 10000:
                    value_str = f"{value:12.3e}"
                else:
                    value_str = f"{value:12.3f}"

                print(f"{path:<40} : {value_str}")

            print()

        return residuals

    if method in {"least_squares", "trf", "dogbox", "lm"}:
        solver_kwargs: dict[str, Any] = {
            "x0": x0,
            "bounds": (lower, upper),
            "max_nfev": max_nfev,
            **kwargs,
        }

        if method != "least_squares":
            solver_kwargs["method"] = method

        result = optimize.least_squares(_evaluate, **solver_kwargs)
    else:

        def _objective(x: np.ndarray) -> float:
            residual = _evaluate(x)
            return float(np.dot(residual, residual))

        minimize_bounds = (
            bounds if bounds is not None else list(zip(lower, upper, strict=True))
        )
        minimize_options = kwargs.pop("options", {})
        if maxiter is not None:
            minimize_options["maxiter"] = maxiter

        result = optimize.minimize(
            _objective,
            x0,
            method=method,
            bounds=minimize_bounds,
            options=minimize_options,
            **kwargs,
        )

    _update_parameters(parameters, np.asarray(result.x, dtype=float))
    updated = {
        path: float(result.x[index]) for index, (path, _) in enumerate(parameters)
    }

    if plot and plot_state is not None:
        plt.close(plot_state.fig)

    return updated, optimized_model, result


if __name__ == "__main__":
    rf = RefinementBaseModel()
    rf["a"] = 1

    print(rf.model_dump_json())

    print(rf)
