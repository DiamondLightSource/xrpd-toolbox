import re

from scipy.constants import Avogadro

# Atomic masses (g/mol)
ATOMIC_MASSES = {
    "H": 1.008,
    "He": 4.0026,
    "Li": 6.94,
    "Be": 9.0122,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Ne": 20.180,
    "Na": 22.990,
    "Mg": 24.305,
    "Al": 26.982,
    "Si": 28.0855,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
    "Ar": 39.948,
    "K": 39.0983,
    "Ca": 40.078,
    "Ti": 47.867,
    "V": 50.9415,
    "Cr": 51.9961,
    "Mn": 54.938,
    "Fe": 55.845,
    "Co": 58.933,
    "Ni": 58.693,
    "Cu": 63.546,
    "Zn": 65.38,
    "Ga": 69.723,
    "Ge": 72.630,
    "As": 74.922,
    "Se": 78.971,
    "Br": 79.904,
    "Ag": 107.8682,
    "Cd": 112.414,
    "Sn": 118.710,
    "I": 126.904,
    "Au": 196.967,
    "Hg": 200.592,
    "Pb": 207.2,
}


def molar_mass(formula):
    """
    Calculate the molar mass (g/mol) of a chemical formula.

    Examples
    --------
    >>> molar_mass("SiO2")
    60.0835
    >>> molar_mass("Fe2O3")
    159.687
    >>> molar_mass("Cu")
    63.546
    """
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)

    mass = 0.0
    for element, count in tokens:
        if element not in ATOMIC_MASSES:
            raise ValueError(f"Unknown element: {element}")
        count = int(count) if count else 1
        mass += ATOMIC_MASSES[element] * count

    return mass


def density_to_number_density(
    density_g_cm3: float, molar_mass_g_mol: float, atoms_per_formula_unit: int = 1
):
    """
    Convert mass density (g/cm^3) to atomic number density (atoms/Å^3).

    Parameters
    ----------
    density_g_cm3 : float
        Density in g/cm^3.
    molar_mass_g_mol : float
        Molar mass of the formula unit in g/mol.
    atoms_per_formula_unit : int, optional
        Number of atoms in one formula unit (default = 1).

    Returns
    -------
    float
        Number density in atoms/Å^3.
    """
    rho = density_g_cm3 * Avogadro * atoms_per_formula_unit / (molar_mass_g_mol * 1e24)
    return rho


def density_and_formula_to_number_density(density_g_cm3, formula):
    molar_mass_g_mol = molar_mass(formula)

    # Count total atoms in the formula
    atoms_per_formula_unit = sum(
        int(c) if c else 1 for _, c in re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    )

    return density_to_number_density(
        density_g_cm3=density_g_cm3,
        molar_mass_g_mol=molar_mass_g_mol,
        atoms_per_formula_unit=atoms_per_formula_unit,
    )


if __name__ == "__main__":
    density_g_cm3 = 2.33
    molar_mass_g_mol = 28.0855

    print(
        density_to_number_density(
            density_g_cm3=density_g_cm3, molar_mass_g_mol=molar_mass_g_mol
        )
    )
