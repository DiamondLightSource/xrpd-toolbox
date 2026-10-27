from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from xrpd_toolbox.constants.constants import (
    ATOMIC_RADII,
    COVALENT_RADII,
    COVALENT_RADII_DOUBLE,
    COVALENT_RADII_TRIPLE,
)
from xrpd_toolbox.core import XRPDBaseModel

radii_filepath = Path(__file__).parent / "radii.csv"

BOND_ORDER = Literal[1, 2, 3]


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
    covalent_radii: dict[BOND_ORDER, dict[str, float]] = {
        1: COVALENT_RADII,
        2: COVALENT_RADII_DOUBLE,
        3: COVALENT_RADII_TRIPLE,
    }

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
            raise ValueError(f"No crystal radius for {element=} with charge {charge=}")

    def get_atomic_radius(self, element: str) -> float | None:
        """Get the atomic radius for a given element."""
        return self.atomic_radii.get(element)

    def get_covalent_radius(self, element: str, bond_order: BOND_ORDER = 1) -> float:
        """Get the covalent radius for a given element in a single (1), double (2)
        or triple (3) bond."""

        radius = self.covalent_radii[bond_order].get(element)

        if radius is None:
            raise ValueError(f"No covalent radius for {element=} with {bond_order=}")

        return radius

    def get_interatomic_distance(
        self,
        element1: str,
        element2: str,
        charge1: int | None = None,
        charge2: int | None = None,
        bond_order: BOND_ORDER | None = None,
    ) -> float | None:
        """Get the interatomic distance between two elements.

        If a bond_order is given the bond is covalent (eg. C=O), so it's the sum of
        the covalent radii for that bond order. A bond between the same element
        (eg. Si-Si) is a covalent single bond, unless a charge is given. Otherwise
        the ions are taken to touch, so it's the sum of their largest crystal radii.
        """

        if bond_order is not None:
            if charge1 is not None or charge2 is not None:
                raise ValueError("Give either a bond_order or charges, not both")
            return self.get_covalent_radius(
                element1, bond_order
            ) + self.get_covalent_radius(element2, bond_order)

        if element1 == element2 and charge1 is None and charge2 is None:
            return 2 * self.get_covalent_radius(element1)

        radius1 = self.get_crystal_radius(element1, charge1)
        radius2 = self.get_crystal_radius(element2, charge2)

        if radius1 is not None and radius2 is not None:
            return float(max(radius1) + max(radius2))
        else:
            raise ValueError(f"No interatomic distance for {element1=} and {element2=}")


if __name__ == "__main__":
    radii = ElementRadii()
    print(radii.get_atomic_radius("Si"))
    print(radii.get_interatomic_distance("Si", "Si"))
    print(radii.get_interatomic_distance("K", "Pd"))
    print(radii.get_interatomic_distance("Pd", "Cl"))

#     mask = (radii.elements == "Fe") & (radii.charges == 2) & (radii.coordinations==6)
#     print("Fe2+ CN=6 crystal radii:", radii.crystal_radii[mask])
#     print("Fe2+ CN=6 ionic radii:  ", radii.ionic_radii[mask])
#     print("Si atomic radius:", radii.atomic_radii["Si"])
