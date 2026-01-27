from _collections_abc import Iterable

import numpy as np


def normalise_to(data: Iterable[float | int], minval: float | int = 0) -> np.ndarray:
    """
    normalises an array
    minval is  the minimum value that the
    processed array is scaled to.
    """

    data_array = np.array(data, dtype=float)

    return (data_array - minval) / (np.amax(data_array) - minval)
