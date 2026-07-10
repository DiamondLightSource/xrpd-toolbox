from __future__ import annotations

import re
from functools import cached_property
from itertools import combinations

from pydantic import Field, computed_field, model_validator
from scipy.constants import Avogadro

from xrpd_toolbox.constants.constants import ATOMIC_MASSES, get_atomic_mass
from xrpd_toolbox.constants.radii import ElementRadii
from xrpd_toolbox.core import XRPDBaseModel


def _parse(s: str, i: int, stop_at: str | None) -> tuple[dict[str, float], int]:
    """
    Parse a formula portion (no leading coefficient) until `stop_at` is encountered
    or end of string.
    """
    counts: dict[str, float | int] = {}
    # Pre-compiled number pattern: integer, decimal, or scientific
    num_pat = re.compile(r"(\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")

    while i < len(s):
        ch = s[i]

        # Check for closing bracket (stop condition)
        if ch in ")]}":
            if stop_at is not None and ch == stop_at:
                return counts, i  # Do not consume the closing bracket
            else:
                raise ValueError(f"Unexpected closing bracket '{ch}' at position {i}")

        # Element: uppercase followed by optional lowercase letters
        if ch.isupper():
            j = i + 1
            while j < len(s) and s[j].islower():
                j += 1
            elem = s[i:j]
            i = j

            # Optional subscript number
            num = 1.0
            m = num_pat.match(s, i)
            if m:
                num = float(m.group())
                i = m.end()
            counts[elem] = counts.get(elem, 0.0) + num
            continue

        # Opening bracket: parse inner group recursively
        if ch in "([{":
            bracket_map = {"(": ")", "[": "]", "{": "}"}
            close_b = bracket_map[ch]
            inner_counts, i = _parse(s, i + 1, stop_at=close_b)

            # Expect the matching closing bracket
            if i >= len(s) or s[i] != close_b:
                raise ValueError(f"Unmatched opening bracket '{ch}'")
            i += 1  # consume closing bracket

            # Optional subscript after the group
            num = 1.0
            m = num_pat.match(s, i)
            if m:
                num = float(m.group())
                i = m.end()

            for elem, cnt in inner_counts.items():
                counts[elem] = counts.get(elem, 0.0) + cnt * num
            continue

        # Skip any remaining whitespace (should be none after initial strip)
        if ch.isspace():
            i += 1
            continue

        # Any other character is invalid
        raise ValueError(f"Unexpected character '{ch}' at position {i}")

    # End of string reached
    if stop_at is not None:
        raise ValueError(f"Unmatched opening bracket, expected closing '{stop_at}'")
    return counts, i


def parse_chemical_formula(formula: str) -> dict[str, float | int]:
    """
    Convert a chemical formula string into a dictionary of element counts.
    """
    # Remove all whitespace
    formula = re.sub(r"\s+", "", formula)

    # Split on hydrate separator (middle dot or asterisk)
    parts = re.split(r"[·*]", formula)

    total_counts: dict[str, float] = {}

    for part in parts:
        if not part:
            continue

        # Parse an optional leading coefficient for this part
        coeff = 1.0
        match = re.match(r"(\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", part)
        if match:
            coeff = float(match.group())
            i = match.end()
        else:
            i = 0

        # Recursive parser (no internal leading coefficients)
        counts, end = _parse(part, i, stop_at=None)
        if end != len(part):
            raise ValueError(f"Unexpected character at position {end} in '{part}'")

        # Merge multiplied counts into total
        for elem, cnt in counts.items():
            total_counts[elem] = total_counts.get(elem, 0.0) + cnt * coeff

    # Convert counts that are integer to int for cleaner output
    result = {}
    for elem, cnt in total_counts.items():
        result[elem] = int(cnt) if cnt.is_integer() else cnt
    return result


def molecular_weight(chemical_formula: dict[str, float | int]):
    """dict must be in the form: {"Si" : 1} or "{"Na": 0.96, "K": 0.04, "Cl": 1}
    returns molecular weight in g/mol
    """

    molecular_weight = 0

    for element, count in chemical_formula.items():
        atomic_mass = get_atomic_mass(element)
        molecular_weight = molecular_weight + (atomic_mass * count)

    return molecular_weight


def density_to_number_density(
    density_g_cm3: float,
    molar_mass_g_mol: float,
    atoms_per_formula_unit: int | float = 1,  # can be float because doping
):
    """
    Convert mass density (g/cm^3) to atomic number density (atoms/Å^3).

    Returns Number density in atoms/Å^3.
    """
    rho = density_g_cm3 * Avogadro * atoms_per_formula_unit / (molar_mass_g_mol * 1e24)
    return rho


class ChemicalFormula(XRPDBaseModel):
    formula: str = Field(frozen=True)

    @model_validator(mode="after")
    def element_must_be_known(self) -> ChemicalFormula:

        for element in self.chemical_formula.keys():
            if element not in ATOMIC_MASSES:
                raise ValueError(
                    f"Element '{element}' not in ATOMIC_MASSES table. "
                    f"Available: {sorted(ATOMIC_MASSES.keys())}"
                )
        return self

    @computed_field
    @cached_property
    def chemical_formula(self) -> dict:
        return parse_chemical_formula(self.formula)

    @cached_property
    def total_atoms(self) -> int | float:
        """Total number of atoms in one formula unit."""
        total_atoms = sum(count for count in self.chemical_formula.values())
        return total_atoms

    @cached_property
    def elements(self) -> list[str]:

        return list(self.chemical_formula.keys())

    @cached_property
    def estimated_min_distance(self) -> float | None:

        element_pairs = combinations(set(self.elements), 2)

        radii = ElementRadii()

        min_distance = None

        for elem1, elem2 in element_pairs:
            if elem1 == elem2:
                distance = radii.get_crystal_radius(elem1)
            else:
                distance = radii.get_interatomic_distance(elem1, elem2)

            if distance is not None:
                if min_distance is None or distance < min_distance:
                    min_distance = distance

        return min_distance

    @cached_property
    def atomic_fraction(self) -> list[float | int]:

        atomic_fraction = [
            count / self.total_atoms for count in self.chemical_formula.values()
        ]
        return atomic_fraction

    @computed_field
    @cached_property
    def molecular_weight(self) -> float:
        """return molecular weight in g/mol"""

        return molecular_weight(self.chemical_formula)

    def rho(self, density: float) -> float:
        """Convert mass density (g/cm^3) to atomic number density (atoms/Å^3)."""

        return density_to_number_density(
            density_g_cm3=density,
            molar_mass_g_mol=self.molecular_weight,
            atoms_per_formula_unit=self.total_atoms,
        )

    @classmethod
    def load_from_composition(cls, composition: dict):
        """Loads from a dict in the form: {"Si": 1} or {"Si": 1, "O": 2}"""

        string_formula = ""

        for element, count in composition.items():
            string_formula = string_formula + (element + str(count))

        assert isinstance(string_formula, str)

        return cls(formula=string_formula)


if __name__ == "__main__":
    # Examples from the problem statement
    print(parse_chemical_formula("Si"))  # {'Si': 1}
    print(parse_chemical_formula("Tb(HCO2)3"))  # {'Tb': 1, 'H': 3, 'C': 3, 'O': 6}
    print(parse_chemical_formula("SiO2"))  # {'Si': 1, 'O': 2}
    print(parse_chemical_formula("Na0.96K0.04Cl"))  # {'Na': 0.96, 'K': 0.04, 'Cl': 1}

    # Additional tests
    print(parse_chemical_formula("K4[Fe(CN)6]"))  # {'K': 4, 'Fe': 1, 'C': 6, 'N': 6}
    print(parse_chemical_formula("CuSO4·5H2O"))  # {'Cu': 1, 'S': 1, 'O': 9, 'H': 10}
    print(parse_chemical_formula("Fe3O4"))  # {'Fe': 3, 'O': 4}
    print(
        parse_chemical_formula("{Pb2(C2O4)2}·3H2O")
    )  # {'Pb': 2, 'C': 4, 'O': 11, 'H': 6}

    si_density_g_cm3 = 2.33
    si_molar_mass_g_mol = 28.0855

    print(
        density_to_number_density(
            density_g_cm3=si_density_g_cm3, molar_mass_g_mol=si_molar_mass_g_mol
        )
    )

    print(ChemicalFormula(formula="SiO2").estimated_min_distance)
