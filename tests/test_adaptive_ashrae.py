import numpy as np
import pytest

from pythermalcomfort.models import adaptive_ashrae
from pythermalcomfort.utilities import Units
from tests.conftest import Urls, retrieve_reference_table, validate_result


@pytest.mark.parametrize("round_output", [True, False])
def test_ip_temperature_rounding_preserves_acceptability(round_output: bool) -> None:
    """Round reported IP temperatures without changing SI comfort decisions."""
    running_mean = np.array([21.5, 22.0, 25.0])
    temperatures = np.array([20.9, 27.14, 31.0])
    si = adaptive_ashrae(
        tdb=temperatures,
        tr=temperatures,
        t_running_mean=running_mean,
        v=0.1,
        round_output=False,
    )
    ip = adaptive_ashrae(
        tdb=temperatures * 1.8 + 32,
        tr=temperatures * 1.8 + 32,
        t_running_mean=running_mean * 1.8 + 32,
        v=0.3281,
        units="IP",
        round_output=round_output,
    )

    for field in (
        "tmp_cmf",
        "tmp_cmf_80_low",
        "tmp_cmf_80_up",
        "tmp_cmf_90_low",
        "tmp_cmf_90_up",
    ):
        expected = getattr(si, field) * 1.8 + 32
        if round_output:
            # rounded once, after the conversion to °F
            expected = np.around(expected, 1)
        np.testing.assert_allclose(getattr(ip, field), expected)
    np.testing.assert_array_equal(ip.acceptability_80, si.acceptability_80)
    np.testing.assert_array_equal(ip.acceptability_90, si.acceptability_90)


@pytest.mark.parametrize("units", ["SI", "IP"])
def test_round_output_never_changes_acceptability(units: str) -> None:
    """round_output only rounds the reported temperatures (#500)."""
    # the unrounded 80 % lower limit at t_running_mean=22 is 21.12 °C
    to = np.array([21.11, 21.12, 21.13, 24.5, 27.5])
    t_rm = 22.0
    if units == "IP":
        to, t_rm = to * 1.8 + 32, t_rm * 1.8 + 32
    kwargs = dict(tdb=to, tr=to, t_running_mean=t_rm, v=0.1, units=units)
    rounded = adaptive_ashrae(**kwargs, round_output=True)
    unrounded = adaptive_ashrae(**kwargs, round_output=False)
    np.testing.assert_array_equal(rounded.acceptability_80, unrounded.acceptability_80)
    np.testing.assert_array_equal(rounded.acceptability_90, unrounded.acceptability_90)


def test_issue_500_example_is_not_acceptable_for_either_rounding() -> None:
    for round_output in (True, False):
        result = adaptive_ashrae(
            tdb=21.11, tr=21.11, t_running_mean=22.0, v=0.1, round_output=round_output
        )
        assert not result.acceptability_80


def test_ip_scalar_comfort_temperatures_round_to_one_decimal() -> None:
    result = adaptive_ashrae(tdb=77, tr=77, t_running_mean=71.6, v=0.3, units="IP")
    assert result.tmp_cmf == 76.3
    assert result.tmp_cmf_80_low == 70.0
    assert result.tmp_cmf_80_up == 82.6
    assert result.tmp_cmf_90_low == 71.8
    assert result.tmp_cmf_90_up == 80.8


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

    # outside the applicability limits the acceptability flags are NaN, not 0.0 (#484)
    assert np.isnan(result.acceptability_80)
    assert np.isnan(result.acceptability_90)


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


@pytest.mark.parametrize("nan_input", ["tdb", "tr", "v"])
def test_acceptability_nan_for_nan_input(nan_input: str) -> None:
    """With limit_inputs=False, a NaN input gives NaN acceptability, not 0.0 or 1.0."""
    inputs = {"tdb": 25.0, "tr": 25.0, "t_running_mean": 20.0, "v": 0.1}
    inputs[nan_input] = [inputs[nan_input], np.nan]
    result = adaptive_ashrae(**inputs, limit_inputs=False)
    np.testing.assert_array_equal(result.acceptability_80, [1.0, np.nan])
    np.testing.assert_array_equal(result.acceptability_90, [1.0, np.nan])


def test_cooling_effect_follows_each_bands_own_25_c_limit() -> None:
    """A warm room does not raise a limit that is still below 25 °C."""
    # 0.31 * 11 + 17.8 = 21.21, so the unraised limits are 24.71 °C and 23.71 °C.
    t_cmf = 0.31 * 11 + 17.8
    for room in (24.9, 25.5):
        result = adaptive_ashrae(
            tdb=room,
            tr=room,
            t_running_mean=11,
            v=0.6,
            round_output=False,
        )
        assert result.tmp_cmf_80_up == pytest.approx(t_cmf + 3.5)
        assert result.tmp_cmf_90_up == pytest.approx(t_cmf + 2.5)
        np.testing.assert_array_equal(result.acceptability_80, 0.0)
        np.testing.assert_array_equal(result.acceptability_90, 0.0)

    rounded = adaptive_ashrae(tdb=25.5, tr=25.5, t_running_mean=11, v=0.6)
    assert rounded.tmp_cmf_80_up == pytest.approx(24.7)
    assert rounded.tmp_cmf_90_up == pytest.approx(23.7)


def test_cooling_effect_can_raise_one_band_and_not_the_other() -> None:
    """At 13 °C outdoors only the 80 % limit has reached 25 °C."""
    t_cmf = 0.31 * 13 + 17.8
    result = adaptive_ashrae(
        tdb=25,
        tr=25,
        t_running_mean=13,
        v=0.6,
        round_output=False,
    )
    assert result.tmp_cmf_80_up == pytest.approx(t_cmf + 3.5 + 1.2)
    assert result.tmp_cmf_90_up == pytest.approx(t_cmf + 2.5)
    np.testing.assert_array_equal(result.acceptability_80, 1.0)
    np.testing.assert_array_equal(result.acceptability_90, 0.0)

    rounded = adaptive_ashrae(tdb=25, tr=25, t_running_mean=13, v=0.6)
    np.testing.assert_array_equal(rounded.acceptability_80, result.acceptability_80)
    np.testing.assert_array_equal(rounded.acceptability_90, result.acceptability_90)


def test_cooling_effect_still_requires_a_room_of_at_least_25_c() -> None:
    """A cool room keeps the unraised limits when the bands are already above 25 °C."""
    t_cmf = 0.31 * 20 + 17.8
    cool = adaptive_ashrae(tdb=24, tr=24, t_running_mean=20, v=1.2, round_output=False)
    warm = adaptive_ashrae(tdb=25, tr=25, t_running_mean=20, v=1.2, round_output=False)
    assert cool.tmp_cmf_80_up == pytest.approx(t_cmf + 3.5)
    assert cool.tmp_cmf_90_up == pytest.approx(t_cmf + 2.5)
    assert warm.tmp_cmf_80_up == pytest.approx(t_cmf + 3.5 + 2.2)
    assert warm.tmp_cmf_90_up == pytest.approx(t_cmf + 2.5 + 2.2)


def test_eighty_percent_band_steps_at_its_own_25_c_crossing() -> None:
    """The 80 % limit starts rising where its unraised value reaches 25 °C."""
    t_rm_step = (25.0 - 17.8 - 3.5) / 0.31
    below = adaptive_ashrae(
        tdb=26,
        tr=26,
        t_running_mean=t_rm_step - 0.05,
        v=0.9,
        round_output=False,
    )
    at_step = adaptive_ashrae(
        tdb=26,
        tr=26,
        t_running_mean=t_rm_step,
        v=0.9,
        round_output=False,
    )
    assert below.tmp_cmf_80_up < 25
    assert at_step.tmp_cmf_80_up == pytest.approx(25 + 1.8)
    assert at_step.tmp_cmf_90_up < 25
    np.testing.assert_array_equal(below.acceptability_80, 0.0)
    np.testing.assert_array_equal(at_step.acceptability_80, 1.0)


def test_band_gate_is_applied_in_celsius_before_ip_conversion() -> None:
    """IP results use the same 25 °C gate as SI."""
    si = adaptive_ashrae(tdb=25, tr=25, t_running_mean=13, v=0.6, round_output=False)
    ip = adaptive_ashrae(
        tdb=77,
        tr=77,
        t_running_mean=13 * 1.8 + 32,
        v=0.6 * 3.281,
        units="IP",
        round_output=False,
    )
    assert ip.tmp_cmf_80_up == pytest.approx(float(si.tmp_cmf_80_up) * 1.8 + 32)
    assert ip.tmp_cmf_90_up == pytest.approx(float(si.tmp_cmf_90_up) * 1.8 + 32)
    np.testing.assert_array_equal(ip.acceptability_80, si.acceptability_80)
    np.testing.assert_array_equal(ip.acceptability_90, si.acceptability_90)


def test_array_rows_use_each_rows_own_band_limit() -> None:
    """Each row is raised from its own unraised limit, not from its neighbour's."""
    result = adaptive_ashrae(
        tdb=[25.5, 25.0],
        tr=[25.5, 25.0],
        t_running_mean=[11.0, 13.0],
        v=0.6,
        round_output=False,
    )
    np.testing.assert_allclose(
        result.tmp_cmf_80_up,
        [0.31 * 11 + 17.8 + 3.5, 0.31 * 13 + 17.8 + 3.5 + 1.2],
    )
    np.testing.assert_allclose(
        result.tmp_cmf_90_up,
        [0.31 * 11 + 17.8 + 2.5, 0.31 * 13 + 17.8 + 2.5],
    )
    np.testing.assert_array_equal(result.acceptability_80, [0.0, 1.0])
    np.testing.assert_array_equal(result.acceptability_90, [0.0, 0.0])
