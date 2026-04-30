from copy import deepcopy
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy import optimize

# from xrpd_toolbox.fit_engine.profile_calculation import ReitveldRefinement
from xrpd_toolbox.utils.utils import timeit


def setup_plot(model):
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
        animated=True,
        label="Observed",
    )

    (line_calc,) = ax.plot(x, y_calc, color="red", animated=True, label="Calculated")

    (line_diff,) = ax.plot(
        x, y_obs - y_calc + offset, color="blue", animated=True, label="Obs - Calc"
    )

    ax.legend()
    ax.set_xlabel(model.data.x_unit)
    ax.set_ylabel("Intensity")

    canvas = cast(FigureCanvasAgg, fig.canvas)

    canvas.draw()
    background = canvas.copy_from_bbox(ax.bbox)

    return fig, ax, (line_obs, line_calc, line_diff, offset), canvas, background


def update_plot(model, lines, ax, canvas, background):
    line_obs, line_calc, line_diff, offset = lines

    y_obs = model.data.y
    y_calc = model.calculate_profile()

    line_calc.set_ydata(y_calc)
    line_diff.set_ydata(y_obs - y_calc + offset)

    canvas.restore_region(background)

    ax.draw_artist(line_obs)
    ax.draw_artist(line_calc)
    ax.draw_artist(line_diff)

    canvas.blit(ax.bbox)
    canvas.flush_events()


# refinement algorithm
def refine_model(
    model,
    method="least_squares",
    bounds=None,
    plot: bool = False,
    plot_every: int = 5,
    **kwargs,
):
    params = []
    # seen = set()

    new_model = deepcopy(model)

    for _, p in new_model.iter_parameters():
        if (not p.refine) or (not isinstance(p.value, (int | float))):
            continue

        params.append(p)

    if not params:
        raise ValueError("No refinable parameters")

    n = len(params)

    x0 = np.empty(n, dtype=float)
    lower = np.empty(n, dtype=float)
    upper = np.empty(n, dtype=float)

    for i, p in enumerate(params):
        x0[i] = float(p.value)
        lower[i], upper[i] = p.bounds

    def update(x):
        for i in range(n):
            params[i].value = float(x[i])

    if plot:
        counter = {"i": 0}

        fig, ax, lines, canvas, background = setup_plot(new_model)

        def maybe_update_plot():
            counter["i"] += 1
            if counter["i"] % plot_every == 0:
                update_plot(new_model, lines, ax, canvas, background)

    else:

        def maybe_update_plot():
            pass

    @timeit
    def residual(x):
        update(x)

        print(new_model.get_refinement_parameters())
        y_calc = new_model.calculate_profile()
        _ = new_model.chi_squared
        r = new_model.data.y - y_calc

        maybe_update_plot()

        return np.asarray(r, dtype=float)

    if method == "least_squares":
        result = optimize.least_squares(
            residual,
            x0,
            bounds=(lower, upper),
            **kwargs,
        )

    else:

        def objective(x):
            r = residual(x)
            return float(r @ r)

        result = optimize.minimize(
            objective,
            x0,
            bounds=list(zip(lower, upper, strict=True)) if bounds is None else bounds,
            **kwargs,
        )

    update(result.x)

    updated = {i: float(result.x[i]) for i in range(n)}

    if plot:
        plt.close()

    return updated, new_model, result
