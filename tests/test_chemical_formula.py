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

    # Si-Si bond in crystalline silicon is 2.35 Å
    assert chem.estimated_min_distance == pytest.approx(2.35, abs=0.05)


# ---------------------------------------------------------------------------
# ElementRadii
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def radii() -> ElementRadii:
    return ElementRadii()


def test_element_radii_columns_are_the_same_length(radii: ElementRadii):
    n = len(radii.elements)

    assert n > 0
    assert len(radii.charges) == n
    assert len(radii.coordinations) == n
    assert len(radii.crystal_radii) == n
    assert len(radii.ionic_radii) == n


def test_get_crystal_radius_returns_every_charge_and_coordination(radii: ElementRadii):
    si_radii = radii.get_crystal_radius("Si")

    assert si_radii is not None
    np.testing.assert_allclose(si_radii, [0.4, 0.54])


def test_get_crystal_radius_for_a_charge(radii):
    pd_radii = radii.get_crystal_radius("Pd", charge=2)

    assert pd_radii is not None
    np.testing.assert_allclose(pd_radii, [0.78, 1.0])


def test_get_atomic_radius(radii: ElementRadii):
    assert radii.get_atomic_radius("Si") == pytest.approx(1.1)
    assert radii.get_atomic_radius("Xx") is None


def test_get_interatomic_distance_sums_the_largest_crystal_radii(radii: ElementRadii):
    # K+ CN12 = 1.78 Å, Pd2+ CN6 = 1.0 Å
    assert radii.get_interatomic_distance("K", "Pd") == pytest.approx(2.78)


def test_get_interatomic_distance_with_charges(radii: ElementRadii):
    # Pd4+ CN6 = 0.755 Å, Cl- CN6 = 1.67 Å
    distance = radii.get_interatomic_distance("Pd", "Cl", charge1=4, charge2=-1)

    assert distance == pytest.approx(2.425)


def test_get_interatomic_distance_is_symmetric(radii: ElementRadii):
    assert radii.get_interatomic_distance("Pd", "Cl") == radii.get_interatomic_distance(
        "Cl", "Pd"
    )


def test_get_crystal_radius_unknown_raises(radii: ElementRadii):
    with pytest.raises(ValueError):
        radii.get_crystal_radius("Xx")
    with pytest.raises(ValueError):
        radii.get_crystal_radius("Si", charge=-4)


def test_get_interatomic_distance_unknown_raises(radii: ElementRadii):
    with pytest.raises(ValueError):
        radii.get_interatomic_distance("Xx", "Cl")


def test_get_covalent_radius(radii: ElementRadii):
    assert radii.get_covalent_radius("Si") == pytest.approx(1.16)
    assert radii.get_covalent_radius("C") == pytest.approx(0.75)


def test_get_covalent_radius_unknown_raises(radii: ElementRadii):
    with pytest.raises(ValueError):
        radii.get_covalent_radius("Xx")


@pytest.mark.parametrize(
    "element, bond_length",
    [
        ("Si", 2.35),  # crystalline silicon
        ("C", 1.54),  # diamond
        ("Ge", 2.45),  # crystalline germanium
    ],
)
def test_get_interatomic_distance_of_the_same_element_is_the_covalent_bond(
    radii: ElementRadii, element: str, bond_length: float
):
    distance = radii.get_interatomic_distance(element, element)

    assert distance == pytest.approx(2 * radii.get_covalent_radius(element))
    assert distance == pytest.approx(bond_length, abs=0.05)


def test_get_interatomic_distance_of_the_same_element_with_charges_uses_crystal_radii(
    radii: ElementRadii,
):
    # Si4+ CN6 = 0.54 Å
    distance = radii.get_interatomic_distance("Si", "Si", charge1=4, charge2=4)

    assert distance == pytest.approx(1.08)


@pytest.mark.parametrize(
    "bond_order, radius",
    [(1, 0.75), (2, 0.67), (3, 0.60)],
)
def test_get_covalent_radius_for_a_bond_order(
    radii: ElementRadii, bond_order, radius: float
):
    assert radii.get_covalent_radius("C", bond_order=bond_order) == pytest.approx(
        radius
    )


def test_get_covalent_radius_without_that_bond_order_raises(radii: ElementRadii):
    # Pyykkö gives no double or triple bond radius for H, or triple for Li
    with pytest.raises(ValueError, match="bond_order=2"):
        radii.get_covalent_radius("H", bond_order=2)
    with pytest.raises(ValueError, match="bond_order=3"):
        radii.get_covalent_radius("Li", bond_order=3)


@pytest.mark.parametrize("element", ["C", "N", "O", "Si", "P", "Fe", "Mo", "W"])
def test_covalent_radii_get_shorter_with_bond_order(radii: ElementRadii, element):
    single, double, triple = (
        radii.get_covalent_radius(element, bond_order=order) for order in (1, 2, 3)
    )

    assert single > double > triple


@pytest.mark.parametrize(
    "element1, element2, bond_order, bond_length",
    [
        ("C", "C", 1, 1.54),  # ethane
        ("C", "C", 2, 1.34),  # ethene
        ("C", "C", 3, 1.20),  # ethyne
        ("C", "O", 2, 1.21),  # formaldehyde
        ("N", "N", 3, 1.10),  # N2
        ("C", "N", 3, 1.16),  # HCN
    ],
)
def test_get_interatomic_distance_with_a_bond_order(
    radii: ElementRadii, element1, element2, bond_order, bond_length: float
):
    distance = radii.get_interatomic_distance(element1, element2, bond_order=bond_order)

    # the radii are fitted to many molecules, so are good to a few pm
    assert distance == pytest.approx(bond_length, abs=0.05)


def test_get_interatomic_distance_single_bond_order_is_the_same_element_default(
    radii: ElementRadii,
):
    assert radii.get_interatomic_distance(
        "Si", "Si", bond_order=1
    ) == radii.get_interatomic_distance("Si", "Si")


def test_get_interatomic_distance_with_bond_order_and_charges_raises(
    radii: ElementRadii,
):
    with pytest.raises(ValueError, match="bond_order"):
        radii.get_interatomic_distance("Si", "O", charge1=4, bond_order=2)
