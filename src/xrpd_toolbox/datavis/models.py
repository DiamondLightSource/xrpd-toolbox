from typing import Literal

import numpy as np
from pydantic import Field

from xrpd_toolbox.core import XYEData
from xrpd_toolbox.utils.files import (
    get_filenumber_from_filepath,
    get_instrument_session_from_filepath,
)

PLOT_TYPES = Literal["scatter", "line", "line+markers"]


class DataPlot(XYEData):
    """An XYEData trace plus everything needed to plot and locate it.

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
    x_label: str | None = None
    y_label: str | None = "Intensity (Arb. Units)"
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


class FittedDataPlot(DataPlot):
    """A :class:`DataPlot` with a fit: a calculated curve sharing ``x`` with
    the observed data, plus the difference, background and reflection-marker
    curves conventionally drawn alongside it (as in a Rietveld refinement
    plot)."""

    calc: list[float]
    diff: list[float] | None = None
    background: list[float] | float | None = None
    markers: list[float] | None = None

    @property
    def obs(self):
        return self.y

    @property
    def difference(self) -> list[float]:
        """returns obs-calc"""
        if self.diff is not None:
            return self.diff
        else:
            difference = np.array(self.y) - np.array(self.calc)
            return difference.tolist()
