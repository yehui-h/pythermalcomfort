import json

import numpy as np
import pytest

from pythermalcomfort.models import adaptive_ashrae
from pythermalcomfort.utilities import Units
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_adaptive_ashrae(get_test_url, retrieve_data) -> None:
    """Test that the Adaptive ASHRAE function calculates correctly for various inputs."""
    reference_table = retrieve_reference_table(
        get_test_url,
        retrieve_data,
        Urls.ADAPTIVE_ASHRAE.name,
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        units = inputs.get("units", Units.SI.value)
        result = adaptive_ashrae(
            inputs["tdb"],
            inputs["tr"],
            inputs["t_running_mean"],
            inputs["v"],
            units,
        )

        validate_result(result, outputs, tolerance)


def test_ashrae_inputs_invalid_units() -> None:
    """Test that the function raises a ValueError for invalid units."""
    with pytest.raises(ValueError):
        adaptive_ashrae(tdb=25, tr=25, t_running_mean=20, v=0.1, units="INVALID")


def test_ashrae_inputs_invalid_tdb_type() -> None:
    """Test that the function raises a TypeError for invalid tdb type."""
    with pytest.raises(TypeError):
        adaptive_ashrae(tdb="invalid", tr=25, t_running_mean=20, v=0.1)


def test_ashrae_inputs_invalid_tr_type() -> None:
    """Test that the function raises a TypeError for invalid tr type."""
    with pytest.raises(TypeError):
        adaptive_ashrae(tdb=25, tr="invalid", t_running_mean=20, v=0.1)


def test_ashrae_inputs_invalid_t_running_mean_type() -> None:
    """Test that the function raises a TypeError for invalid t_running_mean type."""
    with pytest.raises(TypeError):
        adaptive_ashrae(tdb=25, tr=25, t_running_mean="invalid", v=0.1)


def test_ashrae_inputs_invalid_v_type() -> None:
    """Test that the function raises a TypeError for invalid v type."""
    with pytest.raises(TypeError):
        adaptive_ashrae(tdb=25, tr=25, t_running_mean=20, v="invalid")

    # Return nan values when limit_inputs=True and inputs are invalid


def test_nan_values_for_invalid_inputs() -> None:
    """Test that the function returns nan values for invalid inputs when limit_inputs=True."""
    # Test with invalid inputs where limit_inputs=True
    result = adaptive_ashrae(
        tdb=5.0,
        tr=5.0,
        t_running_mean=5.0,
        v=3.0,
        limit_inputs=True,
    )

    # Check that the comfort temperature is nan
    assert np.isnan(result.tmp_cmf)

    # Check that the acceptability flags are False
    assert result.acceptability_80 == False
    assert result.acceptability_90 == False


def test_round_output_default_preserves_behaviour() -> None:
    """Default call must match an explicit round_output=True call."""
    default_result = adaptive_ashrae(tdb=25, tr=25, t_running_mean=21.5, v=0.1)
    explicit_result = adaptive_ashrae(
        tdb=25,
        tr=25,
        t_running_mean=21.5,
        v=0.1,
        round_output=True,
    )

    assert default_result.tmp_cmf == explicit_result.tmp_cmf
    assert default_result.tmp_cmf_80_low == explicit_result.tmp_cmf_80_low
    assert default_result.tmp_cmf_80_up == explicit_result.tmp_cmf_80_up
    assert default_result.tmp_cmf_90_low == explicit_result.tmp_cmf_90_low
    assert default_result.tmp_cmf_90_up == explicit_result.tmp_cmf_90_up


def test_round_output_false_returns_unrounded() -> None:
    """round_output=False must return unrounded t_cmf at full precision."""
    # 0.31 * 21.5 + 17.8 = 24.465; rounded to 1 dp this is 24.5.
    unrounded = adaptive_ashrae(
        tdb=25,
        tr=25,
        t_running_mean=21.5,
        v=0.1,
        round_output=False,
    )
    rounded = adaptive_ashrae(
        tdb=25,
        tr=25,
        t_running_mean=21.5,
        v=0.1,
        round_output=True,
    )

    assert np.isclose(unrounded.tmp_cmf, 24.465)
    assert np.isclose(rounded.tmp_cmf, 24.5)
    assert unrounded.tmp_cmf != rounded.tmp_cmf


def test_round_output_false_propagates_to_bounds() -> None:
    """Unrounded t_cmf must propagate into the derived comfort bounds."""
    unrounded = adaptive_ashrae(
        tdb=25,
        tr=25,
        t_running_mean=21.5,
        v=0.1,
        round_output=False,
    )

    # tmp_cmf_80_low = t_cmf - 3.5; with unrounded t_cmf=24.465 this is 20.965.
    assert np.isclose(unrounded.tmp_cmf_80_low, 24.465 - 3.5)
    assert np.isclose(unrounded.tmp_cmf_90_low, 24.465 - 2.5)
    assert np.isclose(unrounded.tmp_cmf_80_up, 24.465 + 3.5)
    assert np.isclose(unrounded.tmp_cmf_90_up, 24.465 + 2.5)


def test_scalar_acceptability_is_python_bool() -> None:
    """Scalar acceptability flags are builtin bools that json can serialize."""
    comfortable = adaptive_ashrae(tdb=25, tr=25, t_running_mean=20, v=0.1)
    assert comfortable.acceptability_80 is True
    assert comfortable.acceptability_90 is True

    # 80% band is 20.5-27.5 °C and 90% is 21.5-26.5 °C at this running mean.
    warm = adaptive_ashrae(tdb=27, tr=27, t_running_mean=20, v=0.1)
    assert warm.acceptability_80 is True
    assert warm.acceptability_90 is False

    hot = adaptive_ashrae(tdb=30, tr=30, t_running_mean=20, v=0.1)
    assert hot.acceptability_80 is False
    assert hot.acceptability_90 is False

    ip_result = adaptive_ashrae(tdb=77, tr=77, t_running_mean=68, v=0.3, units="IP")
    assert type(ip_result.acceptability_80) is bool
    assert type(ip_result.acceptability_90) is bool
    json.dumps(
        {
            "acceptability_80": comfortable.acceptability_80,
            "acceptability_90": warm.acceptability_90,
            "acceptability_80_ip": ip_result.acceptability_80,
        }
    )

    # A length-1 list is an array, not a scalar bool.
    one = adaptive_ashrae(tdb=[25], tr=25, t_running_mean=20, v=0.1)
    assert isinstance(one.acceptability_80, np.ndarray)
    assert one.acceptability_80.dtype == bool
    assert one.acceptability_80.shape == (1,)

    array_result = adaptive_ashrae(tdb=[25, 30], tr=[25, 30], t_running_mean=20, v=0.1)
    assert array_result.acceptability_80.dtype == bool
    assert array_result.acceptability_90.dtype == bool
    np.testing.assert_array_equal(array_result.acceptability_80, [True, False])
    np.testing.assert_array_equal(array_result.acceptability_90, [True, False])

    with pytest.warns(UserWarning, match="t_running_mean"):
        out_of_range = adaptive_ashrae(tdb=25, tr=25, t_running_mean=9, v=0.1)
    assert out_of_range.acceptability_80 is False
    assert out_of_range.acceptability_90 is False


def test_round_output_invalid_type_raises() -> None:
    """A non-boolean round_output must raise TypeError."""
    with pytest.raises(TypeError):
        adaptive_ashrae(
            tdb=25,
            tr=25,
            t_running_mean=20,
            v=0.1,
            round_output="yes",
        )
