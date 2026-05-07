import matplotlib.pyplot as plt
import numpy as np

from xrpd_toolbox.core import (
    SerialisableNDArray,
    XRPDBaseModel,
    XYEData,
)
from xrpd_toolbox.utils.messenger import Messenger


class PlotData(XRPDBaseModel):
    data: XYEData
    calc: SerialisableNDArray
    diff: SerialisableNDArray
    background: SerialisableNDArray | float | None = None
    markers: SerialisableNDArray | None = None
    title: str | None = None

    def plot(self):
        if isinstance(self.background, float):
            background = [self.background] * len(self.data.x)
        elif isinstance(self.background, np.ndarray):
            background = self.background
        else:
            background = self.background

        offset = -0.1 * self.data.y.max()
        if self.title is not None:
            plt.title(self.title)
        plt.scatter(self.data.x, self.data.y, label="Obs", color="black", s=5)
        plt.plot(self.data.x, self.calc, label="Calc", color="red")
        if background is not None:
            plt.plot(self.data.x, background, label="Background")
        plt.plot(
            self.data.x,
            self.data.y - self.calc + offset,
            label="Obs-Calc",
            color="blue",
        )

        if self.markers is not None:
            plt.vlines(
                self.markers,
                0,
                self.data.y.max() / 10,
                color="magenta",
                label="Marker",
            )

        plt.xlabel(self.data.x_unit)
        plt.ylabel(self.data.y_unit)
        plt.legend()
        plt.show()

    def publish(self, beamline: str):
        messenger = Messenger(beamline)
        messenger.send_plot_data(self)
