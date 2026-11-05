import numpy as np
import pytest

from pythermalcomfort.models import cooling_effect
from tests.conftest import Urls, retrieve_reference_table, validate_result


@pytest.mark.parametrize("vr", [0.05, 0.3, 1.0])
def test_cooling_effect_ip_is_a_temperature_difference(vr: float) -> None:
    """Equivalent SI and IP inputs differ only by the temperature scale."""
    si = cooling_effect(tdb=30, tr=30, vr=vr, rh=50, met=1.2, clo=0.5).ce
    ip = cooling_effect(
        tdb=86, tr=86, vr=vr * 3.281, rh=50, met=1.2, clo=0.5, units="IP"
    ).ce

    # Both public results are rounded to two decimals independently.
    assert ip == pytest.approx(si * 1.8, abs=0.014)


def test_cooling_effect_ip_array_temperature_differences() -> None:
    """Unit conversion also preserves broadcast array results."""
    tdb = np.array([25.0, 30.0, 35.0])
    si = cooling_effect(tdb=tdb, tr=tdb, vr=1.0, rh=50, met=1.2, clo=0.5).ce
    ip = cooling_effect(
        tdb=tdb * 1.8 + 32,
        tr=tdb * 1.8 + 32,
        vr=3.281,
        rh=50,
        met=1.2,
        clo=0.5,
        units="IP",
    ).ce

    np.testing.assert_allclose(ip, si * 1.8, atol=0.014, rtol=0)


def test_cooling_effect_docstring_example() -> None:
    """The published example must match the values the function returns.

    77 °F at 1.64 fps, 1 met and 0.6 clo is about 2.17 °C of cooling. Scaled to
    Fahrenheit and rounded to the public two decimals, that is 3.9, not 3.95.
    """
    scalar = cooling_effect(tdb=25, tr=25, vr=0.3, rh=50, met=1.2, clo=0.5)
    assert scalar.ce == pytest.approx(1.68)

    array = cooling_effect(
        tdb=[25, 77],
        tr=[25, 77],
        vr=[0.3, 1.64],
        rh=[50, 50],
        met=[1.2, 1],
        clo=[0.5, 0.6],
        units="IP",
    )
    assert array.ce == pytest.approx([0.0, 3.9])

    example = cooling_effect.__doc__.split("Examples", 1)[1]
    assert "1.68" in example
    assert "[0, 3.9]" in example
    assert "3.95" not in example


def test_cooling_effect_regression_values() -> None:
    """Pin ce against the reference (pre-refactor) implementation.

    The previous implementation already called the numba-jitted Gagge kernel
    (via set_tmp()/two_nodes_gagge()) - it wasn't slow because it lacked
    numba, but because every root-finding iteration re-ran the full public
    set_tmp() API (input validation, dataclass construction, etc.) on top of
    that kernel. Values were captured before the inner SET calculation was
    rewritten to call the jitted kernel directly, skipping that overhead.
    Confirmed to match bit-for-bit across a wide random sweep plus edge cases
    (vr <= 0.1, vr == 0.1 boundary).
    """
    cases = [
        (25, 25, 0.05, 50, 1.2, 0.5, 0.0),  # vr <= 0.1 -> 0
        (25, 25, 0.1, 50, 1.2, 0.5, 0.0),  # boundary
        (25, 25, 0.3, 50, 1.2, 0.5, 1.68),
        (35, 35, 1.5, 90, 2.5, 0.2, 5.11),  # 5.14 before #489
        (15, 15, 0.5, 10, 1.0, 1.2, 2.61),
    ]
    for tdb, tr, vr, rh, met, clo, expected in cases:
        ce = cooling_effect(tdb=tdb, tr=tr, vr=vr, rh=rh, met=met, clo=clo).ce
        assert np.isclose(ce, expected, atol=0.01), (
            tdb,
            tr,
            vr,
            rh,
            met,
            clo,
            ce,
            expected,
        )


def test_cooling_effect(get_test_url, retrieve_data) -> None:
    """Test that the function calculates the cooling effect correctly for various inputs."""
    reference_table = retrieve_reference_table(
        get_test_url,
        retrieve_data,
        Urls.COOLING_EFFECT.name,
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = cooling_effect(**inputs)

        validate_result(result, outputs, tolerance)
