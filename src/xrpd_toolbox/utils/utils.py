import os
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


def load_int_array_from_file(filepath: str) -> np.ndarray:
    """
    File format is just a list of integers in a text file, one integer per line.

    If no file, will return no empty array.

    If empty file, will return no empty array.

    """

    if not os.path.exists(filepath):
        return np.array([])
    elif os.path.getsize(filepath) == 0:
        return np.array([])
    else:
        return np.loadtxt(filepath, dtype=np.int64, comments="#", usecols=0, ndmin=1)
