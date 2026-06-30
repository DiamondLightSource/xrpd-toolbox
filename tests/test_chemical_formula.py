import pytest

from xrpd_toolbox.utils.chemical_formula import ChemicalFormula, parse_chemical_formula


def test_various_formulas():
    # Examples
    assert parse_chemical_formula("Si") == {"Si": 1}
    assert parse_chemical_formula("Tb(HCO2)3") == {"Tb": 1, "H": 3, "C": 3, "O": 6}
    assert parse_chemical_formula("SiO2") == {"Si": 1, "O": 2}
    assert parse_chemical_formula("Na0.96K0.04Cl") == {"Na": 0.96, "K": 0.04, "Cl": 1}

    assert parse_chemical_formula("K4[Fe(CN)6]") == {"K": 4, "Fe": 1, "C": 6, "N": 6}
    assert parse_chemical_formula("CuSO4·5H2O") == {"Cu": 1, "S": 1, "O": 9, "H": 10}
    assert parse_chemical_formula("Fe3O4") == {"Fe": 3, "O": 4}

    assert parse_chemical_formula("{Pb2(C2O4)2}·3H2O") == {
        "Pb": 2,
        "C": 4,
        "O": 11,
        "H": 6,
    }


def test_chemical_formula_class():

    sio2 = ChemicalFormula(formula="SiO2")
    assert sio2.chemical_formula == {"Si": 1, "O": 2}
    assert sio2.molecular_weight == pytest.approx(60.084, abs=0.1)
    assert sio2.total_atoms == 3


def test_number_density():

    si = ChemicalFormula(formula="Si")
    assert si.chemical_formula == {"Si": 1}
    assert si.molecular_weight == pytest.approx(28, abs=0.1)
    assert si.total_atoms == 1

    si_density_g_cm3 = 2.33

    assert si.rho(si_density_g_cm3) == pytest.approx(0.0499, abs=0.1)


def test_invalid_chemical_formula():

    with pytest.raises(ValueError):
        _ = ChemicalFormula(formula="GrO2")


def test_chemical_formula_class_with_spaces():

    sio2 = ChemicalFormula(formula="Si O2")
    assert sio2.chemical_formula == {"Si": 1, "O": 2}
    assert sio2.molecular_weight == pytest.approx(60.084, abs=0.1)
    assert sio2.total_atoms == 3


def test_chemical_formula_load_from_dict_composition():

    composition = {"Si": 1, "O": 2}

    sio2 = ChemicalFormula.load_from_composition(composition)

    assert sio2.chemical_formula == {"Si": 1, "O": 2}
    assert sio2.molecular_weight == pytest.approx(60.084, abs=0.1)
    assert sio2.total_atoms == 3


def test_chemical_formula_with_big_numbers():

    sio20 = ChemicalFormula(formula="SiO20")
    assert sio20.chemical_formula == {"Si": 1, "O": 20}
    assert sio20.total_atoms == 21

    si22o20 = ChemicalFormula(formula="Si22O20")
    assert si22o20.chemical_formula == {"Si": 22, "O": 20}
    assert si22o20.total_atoms == 42

    mof_like = ChemicalFormula(formula="Mn40O123P45H234")
    assert mof_like.chemical_formula == {"Mn": 40, "O": 123, "P": 45, "H": 234}
