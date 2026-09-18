from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import requests
from pydantic import Field

from xrpd_toolbox.core import (
    SerialisableNDArray,
    XYEData,
)
from xrpd_toolbox.utils.files import (
    get_filenumber_from_filepath,
    get_instrument_session_from_filepath,
)

PLOT_TYPES = Literal["scatter", "line", "line+markers"]


BEAMLINE_TO_URL = {"i15-1": "https://i15-1-datavis.diamond.ac.uk/plot"}


class DataPlot(XYEData):
    """An :class:`XYEData` trace plus everything needed to plot and locate it.

    This is what actually gets drawn - as opposed to ``XYEData``, which is
    just the numbers.
    """

    filepath: str | None = None
    # explicit override for get_filenumber() - usually left unset and derived
    # from filepath instead
    filenumber: int | None = None
    # explicit override for get_instrument_session() - usually left unset and
    # derived from filepath instead
    instrument_session: str | None = None
    # axis labels for the frontend, e.g. "2θ / °" and "Intensity / counts"
    x_label: str = "index"
    y_label: str = "Intensity (Arb. Units)"
    data_type: str | None = None
    plot_type: PLOT_TYPES = Field(default="line")
    # replace an existing plot that has the same title rather than adding a
    # new one - useful for live scans that push repeated updates of one trace
    upsert: bool = False

    def get_instrument_session(self) -> str | None:
        """Return instrument session if its not none, otherwise
        try and determine it from the filepath"""

        if self.instrument_session is not None:
            return self.instrument_session
        elif self.filepath is not None:
            try:
                return get_instrument_session_from_filepath(self.filepath)
            except ValueError:
                # filepath doesn't match the expected /dls/BEAMLINE/data/YEAR/SESSION
                # shape - not every plot has a Diamond-style path, so this is
                # expected rather than exceptional
                return None
        else:
            return None

    def get_filenumber(self) -> int | None:
        if self.filenumber is not None:
            return self.filenumber
        elif self.filepath is not None:
            try:
                return get_filenumber_from_filepath(self.filepath)
            except ValueError:
                # there is no filenumber in the filepath or no
                return None
        else:
            return None

    def plot(self):
        if self.title is not None:
            plt.title(self.title)
        plt.scatter(self.x, self.y, label="Obs", color="black", s=5)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.legend()
        plt.show()

    def publish(self, beamline: str | None = None, url: str | None = None):
        if beamline is not None:
            datavis_url = BEAMLINE_TO_URL.get(beamline)
            if datavis_url is None:
                raise ValueError("Beamline doesn't have a defined datavis url")
        elif url is not None:
            datavis_url = url
        else:
            raise ValueError("Either beamline or url must be given")

        response = requests.post(
            url=datavis_url,
            data=self.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()


class FittedDataPlot(DataPlot):
    """A :class:`DataPlot` with a fit: a calculated curve sharing ``x`` with
    the observed data, plus the difference, background and reflection-marker
    curves conventionally drawn alongside it (as in a Rietveld refinement
    plot)."""

    calc: SerialisableNDArray | list[float]
    diff: SerialisableNDArray | list[float] | None = None
    background: SerialisableNDArray | list[float] | float | None = None
    markers: SerialisableNDArray | list[float] | None = None

    @property
    def obs(self):
        return self.y

    @property
    def difference(self) -> list[float] | SerialisableNDArray:
        """returns obs-calc"""
        if self.diff is not None:
            return self.diff
        else:
            difference = np.array(self.y) - np.array(self.calc)
            return difference.tolist()

    def plot(self, save_to: str | None = None):
        if isinstance(self.background, float):
            background = [self.background] * len(self.x)
        elif isinstance(self.background, np.ndarray):
            background = self.background
        else:
            background = self.background

        offset = -0.1 * np.amax(self.y)
        if self.title is not None:
            plt.title(self.title)
        plt.scatter(self.x, self.y, label="Obs", color="black", s=5)
        plt.plot(self.x, self.calc, label="Calc", color="red")
        if background is not None:
            plt.plot(self.x, background, label="Background")
        plt.plot(
            self.x,
            self.y - self.calc + offset,
            label="Obs-Calc",
            color="blue",
        )

        if self.markers is not None:
            plt.vlines(
                self.markers,
                0,
                np.amax(self.y) / 10,
                color="magenta",
                label="Marker",
            )

        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.legend()
        if save_to is not None:
            plt.savefig(save_to)
        else:
            plt.show()
