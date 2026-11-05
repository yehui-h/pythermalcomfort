import numpy as np
import pytest

from pythermalcomfort.models import set_tmp, two_nodes_gagge
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_two_nodes(get_test_url, retrieve_data) -> None:
    """Test that the function calculates the two nodes Gagge model correctly for various inputs."""
    reference_table = retrieve_reference_table(
        get_test_url,
        retrieve_data,
        Urls.TWO_NODES.name,
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = two_nodes_gagge(**inputs)

        validate_result(result, outputs, tolerance)


@pytest.mark.parametrize(
    "w_max", [0.01, 0.06, 1.2, [0.5, 0.05], np.nan, [0.5, np.nan], True, [True, 0.5]]
)
def test_two_nodes_rejects_w_max_out_of_range(w_max) -> None:
    """Reject w_max that would break the wettedness cap (#489).

    Below 0.06 p_rsw is negative at the cap, at exactly 0.06 disc divides by zero
    when met <= 1, NaN disables the cap, and True would silently mean a cap of 1.
    """
    with pytest.raises(ValueError, match="w_max"):
        two_nodes_gagge(tdb=40, tr=40, v=0.1, rh=50, met=1, clo=0.5, w_max=w_max)


@pytest.mark.parametrize("w_max", [False, 0, 0.0601, 0.5, 1])
def test_two_nodes_accepts_w_max_in_range(w_max) -> None:
    """False and 0 still mean that w_max is computed from v and clo."""
    two_nodes_gagge(tdb=40, tr=40, v=0.1, rh=50, met=2, clo=0.5, w_max=w_max)


@pytest.mark.parametrize("name", ["tdb", "tr", "v", "rh"])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_two_nodes_returns_nan_for_non_finite_inputs(name, bad) -> None:
    """A non-finite value gives NaN for that point only, instead of StopIteration."""
    inputs = {"tdb": [30, 30], "tr": [30, 30], "v": [0.2, 0.2], "rh": [50, 50]}
    inputs[name] = [inputs[name][0], bad]
    result = two_nodes_gagge(met=1.2, clo=0.5, **inputs)
    reference = two_nodes_gagge(tdb=30, tr=30, v=0.2, rh=50, met=1.2, clo=0.5)

    assert result.disc[0] == reference.disc
    assert np.isnan(result.disc[1])
    assert np.isnan(result.t_core[1])
    assert np.isnan(result.set[1])


def test_two_nodes_external_work_is_in_met() -> None:
    """Subtract external work once, in W/m2, from a value given in met.

    The core balance used to subtract ``wme`` as though it were already W/m2, and
    the standard-clothing formula divided that met value by 58.2 again. For this
    case that raised SET from 23.8 °C to 24.4 °C. ``wme`` of 0 is unchanged, and
    ``set_tmp`` passes ``wme`` through without converting it a second time.
    """
    common = {
        "tdb": 25,
        "tr": 25,
        "v": 0.1,
        "rh": 50,
        "met": 1.2,
        "clo": 0.5,
        "round_output": False,
    }
    at_rest = two_nodes_gagge(**common, wme=0)
    working = two_nodes_gagge(**common, wme=0.2)

    assert at_rest.set == pytest.approx(24.314907736364308)
    assert working.set == pytest.approx(23.812103016488845)
    assert working.t_core == pytest.approx(36.829365440649205)
    assert working.pmv_gagge == pytest.approx(-0.0045946706553809)
    assert working.set < at_rest.set
    assert working.t_core < at_rest.t_core
    assert working.e_skin < at_rest.e_skin

    paired = two_nodes_gagge(
        tdb=[25, 25],
        tr=25,
        v=0.1,
        rh=50,
        met=1.2,
        clo=0.5,
        wme=[0, 0.2],
        round_output=False,
    )
    assert paired.set[0] == pytest.approx(at_rest.set)
    assert paired.set[1] == pytest.approx(working.set)
    assert set_tmp(**common, wme=0.2, limit_inputs=False).set == pytest.approx(
        working.set
    )
