import numpy as np
import numpy.typing as npt
from pydantic import computed_field

from xrpd_toolbox.utils.settings import XRPDBaseModel


class Atom(XRPDBaseModel):
    """This describes an atom"""

    label: str  # elemnt label ie Si1 Si2
    element: str  # element name eg Si
    xyz: npt.NDArray[np.float64]  # fractional coorindates of xyz #type: ignore
    b_iso: float
    occupancy: float = 1.0

    @computed_field
    @property
    def x(self) -> float:
        return float(self.xyz[0])

    @computed_field
    @property
    def y(self) -> float:
        return float(self.xyz[1])

    @computed_field
    @property
    def z(self) -> float:
        return float(self.xyz[2])


class Atoms(XRPDBaseModel):
    """This is the array version of atoms"""

    model_config = {"arbitrary_types_allowed": True}

    labels: npt.NDArray[np.str_]  # element label ie Si1 Si2
    elements: npt.NDArray[np.str_]  # element name eg Si
    xyz: npt.NDArray[np.float64]  # fractional coordinates of xyz
    b_iso: npt.NDArray[np.float64]
    occupancies: npt.NDArray[np.float64]

    def __getitem__(self, label_or_index: str | int) -> Atom:
        if isinstance(label_or_index, int):
            index = label_or_index
        elif isinstance(label_or_index, str):
            index = np.where(self.labels == label_or_index)[0][0]
        else:
            raise ValueError("label_or_index must be a string or an integer")

        label = self.labels[index]
        element = self.elements[index]
        xyz = self.xyz[index]
        b_iso = self.b_iso[index]
        occupancy = self.occupancies[index]

        return Atom(
            label=label,
            element=element,
            xyz=xyz,
            b_iso=b_iso,
            occupancy=occupancy,
        )


if __name__ == "__main__":
    # Example usage
    atom = Atom(
        label="Si1",
        element="Si",
        xyz=np.array([0.0, 0.0, 0.0]),
        b_iso=1.0,
        occupancy=1.0,
    )

    atoms = [atom, atom]

    positions = np.stack([a.xyz for a in atoms])

    print(positions)
