import numpy as np
import pytest

from xrpd_toolbox.constants.radii import ElementRadii
from xrpd_toolbox.utils.chemical_formula import (
    ChemicalFormula,
    density_to_number_density,
    molecular_weight,
    parse_chemical_formula,
)

# ---------------------------------------------------------------------------
# parse_chemical_formula
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "formula, expected",
    [
        ("Si", {"Si": 1}),
        ("SiO2", {"Si": 1, "O": 2}),
        ("Tb(HCO2)3", {"Tb": 1, "H": 3, "C": 3, "O": 6}),
        ("Na0.96K0.04Cl", {"Na": 0.96, "K": 0.04, "Cl": 1}),
        ("K4[Fe(CN)6]", {"K": 4, "Fe": 1, "C": 6, "N": 6}),
        ("CuSO4·5H2O", {"Cu": 1, "S": 1, "O": 9, "H": 10}),
        ("CuSO4*5H2O", {"Cu": 1, "S": 1, "O": 9, "H": 10}),
        ("{Pb2(C2O4)2}·3H2O", {"Pb": 2, "C": 4, "O": 11, "H": 6}),
        ("K2 Pd Cl6", {"K": 2, "Pd": 1, "Cl": 6}),
    ],
)
def test_parse_chemical_formula(formula, expected):
    assert parse_chemical_formula(formula) == pytest.approx(expected)


def test_parse_chemical_formula_whole_counts_are_ints():
    counts = parse_chemical_formula("SiO2")

    assert all(isinstance(count, int) for count in counts.values())


@pytest.mark.parametrize("formula", ["Si)", "(SiO2", "[SiO2)", "Si-O2"])
def test_parse_chemical_formula_invalid_raises(formula):
    with pytest.raises(ValueError):
        parse_chemical_formula(formula)


# ---------------------------------------------------------------------------
# molecular_weight / density_to_number_density
# ---------------------------------------------------------------------------


def test_molecular_weight():
    assert molecular_weight({"Si": 1, "O": 2}) == pytest.approx(60.08, abs=0.01)


def test_density_to_number_density_of_silicon():
    # crystalline Si: 8 atoms in a 5.431 Å cubic cell
    expected = 8 / 5.431**3

    rho = density_to_number_density(density_g_cm3=2.329, molar_mass_g_mol=28.085)

    assert rho == pytest.approx(expected, rel=1e-3)


def test_density_to_number_density_scales_with_atoms_per_formula_unit():
    one = density_to_number_density(density_g_cm3=2.0, molar_mass_g_mol=60.0)
    three = density_to_number_density(
        density_g_cm3=2.0, molar_mass_g_mol=60.0, atoms_per_formula_unit=3
    )

    assert three == pytest.approx(3 * one)


# ---------------------------------------------------------------------------
# ChemicalFormula
# ---------------------------------------------------------------------------


def test_chemical_formula_properties():
    chem = ChemicalFormula(formula="K2PdCl6")

    assert chem.chemical_formula == {"K": 2, "Pd": 1, "Cl": 6}
    assert chem.elements == ["K", "Pd", "Cl"]
    assert chem.total_atoms == 9
    assert chem.atomic_fraction == pytest.approx([2 / 9, 1 / 9, 6 / 9])
    assert sum(chem.atomic_fraction) == pytest.approx(1)


def test_chemical_formula_molecular_weight_matches_molecular_weight():
    chem = ChemicalFormula(formula="K2PdCl6")

    assert chem.molecular_weight == pytest.approx(
        molecular_weight({"K": 2, "Pd": 1, "Cl": 6})
    )


def test_chemical_formula_unknown_element_raises():
    with pytest.raises(ValueError, match="Xx"):
        ChemicalFormula(formula="Xx2O")


def test_chemical_formula_is_frozen():
    chem = ChemicalFormula(formula="SiO2")

    with pytest.raises(ValueError):
        chem.formula = "Si"


def test_rho_uses_molecular_weight_and_atoms_per_formula_unit():
    chem = ChemicalFormula(formula="SiO2")

    rho = chem.rho(density_g_cm3=2.2)

    assert rho == pytest.approx(
        density_to_number_density(
            density_g_cm3=2.2,
            molar_mass_g_mol=chem.molecular_weight,
            atoms_per_formula_unit=3,
        )
    )


def test_load_from_composition():
    chem = ChemicalFormula.load_from_composition({"Si": 1, "O": 2})

    assert chem.chemical_formula == {"Si": 1, "O": 2}


def test_element_pair_distances_has_every_pair_of_different_elements():
    distances = ChemicalFormula(formula="K2PdCl6").get_element_pair_distances()

    pairs = {frozenset(pair.split(" -> ")) for pair in distances}

    assert pairs == {
        frozenset({"K", "Pd"}),
        frozenset({"K", "Cl"}),
        frozenset({"Pd", "Cl"}),
    }
    assert all(distance > 0 for distance in distances.values())


def test_element_pair_distance_is_the_sum_of_the_largest_crystal_radii():
    distances = ChemicalFormula(formula="PdCl6").get_element_pair_distances()

    (distance,) = distances.values()
    # Pd2+ CN6 = 1.0 Å, Cl- CN6 = 1.67 Å
    assert distance == pytest.approx(2.67)


def test_estimated_min_distance_is_the_shortest_pair_distance():
    chem = ChemicalFormula(formula="K2PdCl6")

    distances = chem.get_element_pair_distances()

    assert chem.estimated_min_distance == min(distances.values())


def test_estimated_min_distance_of_a_single_element():
    chem = ChemicalFormula(formula="Si")

    assert chem.estimated_min_distance > 0


# ---------------------------------------------------------------------------
# ElementRadii
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def radii() -> ElementRadii:
    return ElementRadii()


def test_element_radii_columns_are_the_same_length(radii):
    n = len(radii.elements)

    assert n > 0
    assert len(radii.charges) == n
    assert len(radii.coordinations) == n
    assert len(radii.crystal_radii) == n
    assert len(radii.ionic_radii) == n


def test_get_crystal_radius_returns_every_charge_and_coordination(radii):
    si_radii = radii.get_crystal_radius("Si")

    assert si_radii is not None
    np.testing.assert_allclose(si_radii, [0.4, 0.54])


def test_get_crystal_radius_for_a_charge(radii):
    pd_radii = radii.get_crystal_radius("Pd", charge=2)

    assert pd_radii is not None
    np.testing.assert_allclose(pd_radii, [0.78, 1.0])


def test_get_crystal_radius_unknown_is_none(radii):
    assert radii.get_crystal_radius("Xx") is None
    assert radii.get_crystal_radius("Si", charge=-4) is None


def test_get_atomic_radius(radii):
    assert radii.get_atomic_radius("Si") == pytest.approx(1.1)
    assert radii.get_atomic_radius("Xx") is None


def test_get_interatomic_distance_sums_the_largest_crystal_radii(radii):
    # K+ CN12 = 1.78 Å, Pd2+ CN6 = 1.0 Å
    assert radii.get_interatomic_distance("K", "Pd") == pytest.approx(2.78)


def test_get_interatomic_distance_with_charges(radii):
    # Pd4+ CN6 = 0.755 Å, Cl- CN6 = 1.67 Å
    distance = radii.get_interatomic_distance("Pd", "Cl", charge1=4, charge2=-1)

    assert distance == pytest.approx(2.425)


def test_get_interatomic_distance_is_symmetric(radii):
    assert radii.get_interatomic_distance("Pd", "Cl") == radii.get_interatomic_distance(
        "Cl", "Pd"
    )


def test_get_interatomic_distance_unknown_is_none(radii):
    assert radii.get_interatomic_distance("Xx", "Cl") is None
