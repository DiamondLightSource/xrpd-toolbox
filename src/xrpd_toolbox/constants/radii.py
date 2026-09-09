from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from xrpd_toolbox.constants.constants import ATOMIC_RADII
from xrpd_toolbox.core import XRPDBaseModel

radii_filepath = Path(__file__).parent / "radii.csv"


class ElementRadii(XRPDBaseModel):
    """Class to hold atomic radii data."""

    _elements, _charges, _coordinations, _crystal_radii, _ionic_radii = np.genfromtxt(
        radii_filepath,
        delimiter=",",
        unpack=True,
        dtype=None,
        skip_header=1,
        encoding=None,
    )

    elements: NDArray[np.str_] = np.asarray(_elements, dtype=str)
    charges: NDArray[np.int_] = np.asarray(_charges, dtype=int)
    coordinations: NDArray[np.int_] = np.asarray(_coordinations, dtype=int)
    crystal_radii: NDArray[np.floating] = np.asarray(_crystal_radii, dtype=float)
    ionic_radii: NDArray[np.floating] = np.asarray(_ionic_radii, dtype=float)
    atomic_radii: dict[str, float | None] = ATOMIC_RADII

    def get_crystal_radius(
        self, element: str, charge: int | None = None
    ) -> NDArray[np.floating] | None:
        """Get the crystal radius for a given element."""

        if charge is not None:
            index = np.where((self.elements == element) & (self.charges == charge))[0]
        else:
            index = np.where(self.elements == element)[0]

        if len(index) > 0:
            return self.crystal_radii[index]
        else:
            return None

    def get_atomic_radius(self, element: str) -> float | None:
        """Get the atomic radius for a given element."""
        return self.atomic_radii.get(element)

    def get_interatomic_distance(
        self,
        element1: str,
        element2: str,
        charge1: int | None = None,
        charge2: int | None = None,
    ) -> float | None:
        """Get the interatomic distance between two elements."""
        radius1 = self.get_crystal_radius(element1, charge1)
        radius2 = self.get_crystal_radius(element2, charge2)
        if radius1 is not None and radius2 is not None:
            return float(min(radius1) + min(radius2))
        else:
            return None


if __name__ == "__main__":
    radii = ElementRadii()
    print(radii.get_atomic_radius("Si"))
    print(radii.get_interatomic_distance("Si", "Si"))

    mask = (radii.elements == "Fe") & (radii.charges == 2) & (radii.coordinations == 6)
    print("Fe2+ CN=6 crystal radii:", radii.crystal_radii[mask])
    print("Fe2+ CN=6 ionic radii:  ", radii.ionic_radii[mask])
    print("Si atomic radius:", radii.atomic_radii["Si"])
