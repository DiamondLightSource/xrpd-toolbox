"""Top level API.

.. data:: __version__
    :type: str

    Version number as calculated by https://github.com/pypa/setuptools_scm
"""

from pathlib import Path

from ._version import __version__

__all__ = ["__version__"]


BASE_PATH = Path(__file__).parent
