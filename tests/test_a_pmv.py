import pytest

from pythermalcomfort.clothing import clo_dynamic_iso
from pythermalcomfort.environment import v_relative
from pythermalcomfort.models import pmv_a
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_a_pmv(get_test_url, retrieve_data) -> None:
    """Test that the function calculates the aPMV correctly for various inputs."""
    reference_table = retrieve_reference_table(
        get_test_url,
        retrieve_data,
        Urls.A_PMV.name,
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = pmv_a(**inputs)

        validate_result(result, outputs, tolerance)


def test_a_pmv_wrong_input_type() -> None:
    """Test that the function raises a TypeError for wrong input types."""
    with pytest.raises(TypeError):
        pmv_a("25", 25, 0.1, 50, 1.2, 0.5, 7)
    with pytest.raises(ValueError):
        pmv_a(25, 25, 0.1, 50, 1.2, 0.5, 7, units="celsius")


def test_not_valid_units() -> None:
    """Test that the function raises a ValueError for invalid units."""
    with pytest.raises(ValueError):
        pmv_a(25, 25, 0.1, 50, 1.2, 0.5, 7, units="wrong")


def test_pmv_a_docstring_example() -> None:
    """The published example must match the class and value the function returns."""
    v = 0.1
    met = 1.4
    clo = 0.5
    v_r = v_relative(v=v, met=met)
    clo_d = clo_dynamic_iso(clo=clo, met=met, v=v)
    results = pmv_a(
        tdb=28,
        tr=28,
        vr=v_r,
        rh=50,
        met=met,
        clo=clo_d,
        a_coefficient=0.293,
    )
    assert type(results).__name__ == "APMV"
    assert results.a_pmv == pytest.approx(0.71)

    example = pmv_a.__doc__.split("Examples", 1)[1]
    assert "0.71" in example
    assert "0.74" not in example
    assert "AdaptivePMV" not in pmv_a.__doc__
    assert ":py:class:`~pythermalcomfort.classes_return.APMV`" in pmv_a.__doc__
