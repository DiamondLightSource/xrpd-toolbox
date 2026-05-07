from pathlib import Path

import numpy as np
import pytest

from xrpd_toolbox.core import (
    Parameter,
    ParameterArray,
    ScatteringData,
    XRPDBaseModel,
    XYEData,
    evaluate_expression,
    safe_exp,
    safe_pow,
    to_ndarray,
)
from xrpd_toolbox.fit_engine.fitting_core import RefinementBaseModel, is_parameter_like


def test_to_ndarray_converts_lists_and_preserves_ndarrays():
    source = [1, 2, 3]
    result = to_ndarray(source)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([1, 2, 3]))

    arr = np.array([4.0, 5.0])
    assert to_ndarray(arr) is arr
    assert to_ndarray(7) == 7


def test_safe_pow_and_safe_exp_valid_and_invalid():
    assert safe_pow(2, 3) == 8
    assert safe_pow(5, 0) == 1

    with pytest.raises(ValueError, match="too large"):
        safe_pow(1e7, 1)

    assert safe_exp(2) == pytest.approx(np.exp(2.0))
    with pytest.raises(ValueError, match="exp too large"):
        safe_exp(701)


def test_evaluate_expression_uses_safe_globals_and_rejects_bad_input():
    assert evaluate_expression("x * 2 + sin(y)", {"x": 3.0, "y": 0.0}) == 6.0
    assert evaluate_expression("pow(2, 3)", {}) == 8.0

    with pytest.raises(ValueError):
        evaluate_expression("__import__('os')", {})

    with pytest.raises(ValueError):
        evaluate_expression("x.__class__", {"x": 1})

    with pytest.raises(ValueError):
        evaluate_expression("a" * 201, {})


class ParameterModel(RefinementBaseModel):
    a: Parameter = Parameter(value=3)
    b: Parameter = Parameter(value="a + 2")


def test_parameter_string_expression_with_model_context():
    model = ParameterModel()
    model.b._ctx = lambda: {"a": float(model.a)}
    model.b._name = "b"

    assert float(model.a) == 3.0
    assert float(model.b) == 5.0

    assert model.a == 3
    assert model.a < 4
    assert model.a <= 3
    assert model.a > 2


def test_parameter_arithmetic_returns_raw_numbers_when_possible():
    a = Parameter(value=3)
    assert a + 2 == 5


def test_is_parameter_like_detects_parameter_in_union_annotations():
    assert is_parameter_like(Parameter | float)
    assert not is_parameter_like(float)


def test_parameter_array_serialisation_and_indexing():
    array = ParameterArray.from_array([1, 2, 3], refine=False)
    assert isinstance(array, ParameterArray)
    assert np.array_equal(array.__array__(), np.array([1.0, 2.0, 3.0]))
    first_item = array[1]
    assert isinstance(first_item, Parameter)
    assert first_item.value == 2
    assert isinstance(array[1:2], ParameterArray)
    serialised = array.serialize()
    assert serialised["value"] == [1, 2, 3]
    assert serialised["refine"] == [False, False, False]

    loaded = ParameterArray.model_validate(
        {
            "value": [4, 5],
            "refine": [True, False],
            "lower_bounds": [-np.inf, -np.inf],
            "upper_bounds": [np.inf, np.inf],
        }
    )
    assert isinstance(loaded, ParameterArray)
    first_item = loaded[0]
    assert isinstance(first_item, Parameter)
    assert first_item.value == 4
    second_item = loaded[1]
    assert isinstance(second_item, Parameter)
    assert second_item.refine is False


class SimpleXRPDModel(XRPDBaseModel):
    x: int
    y: str


def test_xrpd_base_model_save_load(tmp_path: Path):
    model = SimpleXRPDModel(x=1, y="hello")

    json_path = tmp_path / "model.json"
    model.save_to_json(json_path)
    loaded_json = SimpleXRPDModel.load_from_json(json_path)
    assert loaded_json == model

    toml_path = tmp_path / "model.toml"
    model.save_to_toml(toml_path)
    loaded_toml = SimpleXRPDModel.load_from_toml(toml_path)
    assert loaded_toml == model

    yaml_path = tmp_path / "model.yaml"
    model.save_to_yaml(yaml_path)
    loaded_yaml = SimpleXRPDModel.load_from_yaml(yaml_path)
    assert loaded_yaml == model

    with pytest.raises(ValueError):
        model.save_to_json(tmp_path / "bad.txt")

    with pytest.raises(ValueError):
        SimpleXRPDModel.load(tmp_path / "bad.txt")


def test_xrpd_base_model_get_set_item():
    model = SimpleXRPDModel(x=2, y="world")
    assert model["x"] == 2
    model["x"] = 5
    assert model.x == 5
    with pytest.raises(ValueError):
        _ = model["missing"]


class NestedRefinementModel(RefinementBaseModel):
    c: Parameter = Parameter(value=3)


class RefinementModelUnderTest(RefinementBaseModel):
    a: Parameter | float = 1.5
    b: Parameter = Parameter(value=2)
    nested: NestedRefinementModel = NestedRefinementModel()


def test_refinement_base_model_parameterisation_and_iteration():
    model = RefinementModelUnderTest()
    model.parameterise_all(refine=True)

    assert isinstance(model.a, Parameter)
    assert model.a.refine is True
    assert isinstance(model.b, Parameter)
    assert isinstance(model.nested.c, Parameter)

    paths = {model.path_to_string(path) for path, _ in model.iter_parameters()}
    assert "a" in paths
    assert "b" in paths
    assert "nested.c" in paths

    model.parameterise_all(refine=True)
    model.refine_none()
    assert model.a.refine is False
    assert model.b.refine is False
    assert model.nested.c.refine is False

    model.refine_all(keep_fixed=["b"])
    assert model.a.refine is True
    assert model.b.refine is False
    assert model.nested.c.refine is True

    params = model.get_refinement_parameters()
    assert "a" in params
    assert "nested.c" in params

    model.set_refinement_parameters({"a": 4.0, "nested.c": 6.0})
    assert float(model.a) == 4.0
    assert float(model.nested.c) == 6.0

    assert model.get_param_by_path("a") is model.a
    assert model.get_param_by_path("nested.c") is model.nested.c


def test_xyedata_from_csv(tmp_path: Path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("1 2 0.1\n2 3 0.2\n")

    data = XYEData.from_csv(str(csv_file))
    assert data.e is not None
    assert isinstance(data, XYEData)
    assert np.array_equal(data.x, np.array([1.0, 2.0]))
    assert np.array_equal(data.y, np.array([2.0, 3.0]))
    assert np.array_equal(data.e, np.array([0.1, 0.2]))
    assert data.source == str(csv_file)


def test_scattering_data_from_xye_and_fullprof(tmp_path: Path):
    xye_file = tmp_path / "data.xye"
    xye_file.write_text("1 2 0.1\n2 3 0.2\n")

    scattering = ScatteringData.from_xye(
        str(xye_file), x_unit="tth", data_type="xray", wavelength=1.54
    )
    assert isinstance(scattering, ScatteringData)
    assert scattering.x_unit == "tth"
    assert scattering.data_type == "xray"
    assert float(scattering.wavelength) == pytest.approx(1.54)

    fullprof_file = tmp_path / "data.dat"
    fullprof_file.write_text("header\n1 2 0.1\n2 3 0.2\n")

    fullprof_data = ScatteringData.from_fullprof(
        str(fullprof_file), x_unit="tth", data_type="xray", wavelength=2.0
    )
    assert isinstance(fullprof_data, ScatteringData)
    assert fullprof_data.e is not None
    assert np.array_equal(fullprof_data.x, np.array([1.0, 2.0]))
    assert np.array_equal(fullprof_data.y, np.array([2.0, 3.0]))
    assert np.array_equal(fullprof_data.e, np.array([0.1, 0.2]))
