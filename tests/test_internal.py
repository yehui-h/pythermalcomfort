import warnings

import numpy as np
import pytest

import pythermalcomfort._internal.validation as validation
from pythermalcomfort._internal.ashrae55 import _check_ashrae55_compliance
from pythermalcomfort._internal.validation import (
    _as_flag,
    _mapping,
    _valid_range,
    validate_type,
)
from pythermalcomfort.classes_input import BaseInputs


class TestValidRange:
    """Tests for _valid_range warning behaviour."""

    def test_scalar_out_of_range_warns(self) -> None:
        """Scalar value outside range triggers UserWarning with the value."""
        with pytest.warns(
            UserWarning, match=r"'tdb' has value 50\.0.*\[10\.0, 40\.0\]"
        ):
            result = _valid_range(50.0, (10.0, 40.0), "tdb")
        assert np.isnan(result)

    def test_array_out_of_range_warns(self) -> None:
        """Array with out-of-range values triggers UserWarning with count, values, and indices."""
        with pytest.warns(
            UserWarning,
            match=r"'tdb' has 2 values \[50\.0, 45\.0\] at indices \[1, 3\].*\[10\.0, 40\.0\]",
        ):
            result = _valid_range([20.0, 50.0, 30.0, 45.0], (10.0, 40.0), "tdb")
        assert np.isnan(result[1]) and np.isnan(result[3])
        assert result[0] == 20.0 and result[2] == 30.0

    def test_in_range_no_warning(self, recwarn) -> None:
        """Values within range produce no warning."""
        result = _valid_range(25.0, (10.0, 40.0), "tdb")
        assert len(recwarn) == 0
        assert result == 25.0

    def test_no_param_name_auto_extracts_from_caller(self) -> None:
        """Without ``param_name``, the caller's variable name is auto-extracted."""
        tdb = 50.0
        with pytest.warns(
            UserWarning, match=r"'tdb' has value 50\.0.*\[10\.0, 40\.0\]"
        ):
            result = _valid_range(tdb, (10.0, 40.0))
        assert np.isnan(result)

    def test_no_param_name_literal_falls_back_to_unknown(self) -> None:
        """Literal first arg cannot be auto-named; warning falls back to ``<unknown>``."""
        with pytest.warns(
            UserWarning, match=r"'<unknown>' has value 50\.0.*\[10\.0, 40\.0\]"
        ):
            result = _valid_range(50.0, (10.0, 40.0))
        assert np.isnan(result)


def test_internal_renames_have_no_compatibility_aliases() -> None:
    """Internal helpers expose only their new private names."""
    assert not hasattr(validation, "valid_range")
    assert not hasattr(validation, "mapping")
    assert _mapping([10, 20, 30], {15: "low", 25: "medium", 35: "high"}).tolist() == [
        "low",
        "medium",
        "high",
    ]


class TestCheckAshrae55Compliance:
    """Tests for _check_ashrae55_compliance warning behaviour (airspeed_control=False)."""

    def test_airspeed_control_cond1_warns(self) -> None:
        """Above 25.5 °C, air speed over 0.8 m/s triggers one UserWarning.

        tdb=tr=30 → to=30, so only the 0.8 m/s cap applies.
        """
        with pytest.warns(UserWarning, match=r"exceeding 0\.8 m/s") as record:
            _check_ashrae55_compliance(
                tdb=np.float64(30),
                tr=np.float64(30),
                v=np.float64(1.0),
                met=np.float64(1.2),
                clo=np.float64(0.5),
                airspeed_control=False,
            )
        assert len(record) == 1

    def test_airspeed_control_cond2_warns(self) -> None:
        """Air speed over the equal-SET curve triggers one UserWarning.

        With tdb=tr=24, to=24; the curve limit is about 0.32 m/s and v=0.5 exceeds it.
        """
        with pytest.warns(UserWarning, match=r"from 23°C through 25\.5°C"):
            _check_ashrae55_compliance(
                tdb=np.float64(24),
                tr=np.float64(24),
                v=np.float64(0.5),
                met=np.float64(1.2),
                clo=np.float64(0.5),
                airspeed_control=False,
            )

    def test_airspeed_control_cond3_warns(self) -> None:
        """Below 23 °C, air speed over 0.2 m/s triggers one UserWarning.

        With tdb=tr=22, to=22; v=0.3 exceeds 0.2 m/s at 0.5 clo and 1.2 met.
        """
        with pytest.warns(UserWarning, match=r"below an operative temperature of 23°C"):
            _check_ashrae55_compliance(
                tdb=np.float64(22),
                tr=np.float64(22),
                v=np.float64(0.3),
                met=np.float64(1.2),
                clo=np.float64(0.5),
                airspeed_control=False,
            )

    def test_airspeed_control_true_no_condition_warning(self, recwarn) -> None:
        """airspeed_control=True skips cond1/cond2/cond3 checks even when v=1.0."""
        _check_ashrae55_compliance(
            tdb=np.float64(25),
            tr=np.float64(25),
            v=np.float64(1.0),
            met=np.float64(1.2),
            clo=np.float64(0.5),
            airspeed_control=True,
        )
        assert len(recwarn) == 0

    @pytest.mark.parametrize("missing", ["v", "met", "clo"])
    def test_airspeed_control_false_requires_v_met_clo(self, missing: str) -> None:
        """airspeed_control=False without v, met or clo raises a clear error (#391)."""
        kwargs = {
            "tdb": np.float64(24),
            "tr": np.float64(24),
            "v": np.float64(0.1),
            "met": np.float64(1.2),
            "clo": np.float64(0.5),
        }
        del kwargs[missing]
        with pytest.raises(ValueError, match=f"requires {missing}"):
            _check_ashrae55_compliance(**kwargs, airspeed_control=False)


def _air_speed_after_no_control_check(tdb, tr, v, met, clo):
    """Return the air speed left after the no-control limits, NaN where rejected."""
    _tdb, _tr, checked, _met, _clo = _check_ashrae55_compliance(
        tdb=tdb,
        tr=tr,
        v=v,
        met=met,
        clo=clo,
        airspeed_control=False,
    )
    return checked


class TestNoControlAirSpeedLimits:
    """Section 5.3.2.4 bands, exception, and ASHRAE operative temperature."""

    @pytest.mark.parametrize(
        ("tdb", "tr", "v", "met", "clo", "rejected"),
        [
            pytest.param(26, 26, 0.9, 1.1, 0.5, True, id="above-25.5-over-0.8"),
            pytest.param(26, 26, 0.8, 1.1, 0.5, False, id="above-25.5-at-0.8"),
            pytest.param(26, 26, 0.9, 1.3, 0.8, True, id="0.8-cap-ignores-clo-and-met"),
            pytest.param(25.5, 25.5, 0.85, 1.1, 0.5, False, id="curve-includes-25.5"),
            pytest.param(
                25.6, 25.6, 0.85, 1.1, 0.5, True, id="just-above-25.5-uses-0.8"
            ),
            pytest.param(24, 24, 0.5, 1.1, 0.8, True, id="curve-ignores-clo"),
            pytest.param(23, 23, 0.195, 1.1, 0.5, True, id="curve-includes-23"),
            pytest.param(23, 23, 0.19, 1.1, 0.5, False, id="under-curve-at-23"),
            pytest.param(
                22, 22, 0.3, 1.3, 0.7, True, id="0.2-cap-at-exact-0.7-clo-and-1.3-met"
            ),
            pytest.param(22, 22, 0.3, 1.1, 0.71, False, id="0.2-waived-above-0.7-clo"),
            pytest.param(22, 22, 0.3, 1.31, 0.5, False, id="0.2-waived-above-1.3-met"),
            # Air 20 °C, radiant 30 °C, 0.3 m/s. ASHRAE operative temperature is
            # 24 °C (weight 0.6) and the curve limit is 0.318 m/s. ISO 7726 would
            # give 23.66 °C, where the curve is 0.251 m/s and 0.3 m/s is over it.
            pytest.param(
                20, 30, 0.3, 1.1, 0.5, False, id="ashrae-operative-temperature"
            ),
            pytest.param(
                20,
                30,
                0.35,
                1.1,
                0.5,
                True,
                id="ashrae-operative-temperature-over-curve",
            ),
        ],
    )
    def test_band_exception_and_operative_temperature(
        self,
        tdb: float,
        tr: float,
        v: float,
        met: float,
        clo: float,
        rejected: bool,
    ) -> None:
        """Each band uses its own cap, and the 0.2 m/s waiver is only below 23 °C."""
        if rejected:
            with pytest.warns(UserWarning):
                checked = _air_speed_after_no_control_check(tdb, tr, v, met, clo)
            assert np.isnan(checked)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                checked = _air_speed_after_no_control_check(tdb, tr, v, met, clo)
            assert checked == pytest.approx(v)

    def test_array_rows_use_each_rows_own_band(self) -> None:
        """One call can reject one row and keep the next."""
        with pytest.warns(UserWarning):
            checked = _air_speed_after_no_control_check(
                tdb=np.array([26.0, 25.5, 22.0, 20.0]),
                tr=np.array([26.0, 25.5, 22.0, 30.0]),
                v=np.array([0.9, 0.85, 0.3, 0.3]),
                met=np.array([1.3, 1.1, 1.3, 1.1]),
                clo=np.array([0.7, 0.5, 0.7, 0.5]),
            )
        assert np.isnan(checked[0])  # 26 °C, 0.9 m/s: over 0.8, clo does not waive it
        assert checked[1] == pytest.approx(0.85)  # 25.5 °C is still on the curve
        assert np.isnan(checked[2])  # exactly 0.7 clo and 1.3 met stays limited
        assert checked[3] == pytest.approx(0.3)  # ASHRAE operative temperature 24 °C


def test_validate_type() -> None:
    """Test the validate_type function for type validation."""
    allowed = (float, int, list, np.ndarray)

    # valid cases
    # native Python types
    validate_type(1, "int_value", allowed)
    validate_type(3.1415, "float_value", allowed)
    validate_type([1, 2, 3], "list_value", allowed)
    validate_type(np.asarray([1, 2, 3]), "array_value", allowed)

    # np scalars should be converted to native types via .item()
    normalized_float = validate_type(np.float32(40.0), "np_float32", allowed)
    normalized_int32 = validate_type(np.int32(100), "np_int32", allowed)
    normalized_int64 = validate_type(np.int64(200), "np_int64", allowed)
    assert normalized_float == 40.0
    assert normalized_int32 == 100
    assert normalized_int64 == 200
    assert not isinstance(normalized_float, np.generic)
    assert not isinstance(normalized_int32, np.generic)
    assert not isinstance(normalized_int64, np.generic)

    # np array of floats and ints should be allowed
    arr_numeric = np.asarray([np.float32(1.0), np.int32(2), 3, 3.52])
    validate_type(arr_numeric, "arr_numeric", allowed)

    # empty NumPy array should also pass
    validate_type(np.asarray([]), "empty_array", allowed)

    # empty list should pass
    validate_type([], "empty_list", allowed)

    # --- Invalid cases ---

    with pytest.raises(TypeError) as exc_info:
        validate_type({"a": 1}, "dict_type", allowed)
    assert "dict_type must be one of the following types:" in str(exc_info.value)

    with pytest.raises(TypeError) as exc_info:
        validate_type("hello", "str_value", allowed)
    assert "str_value must be one of the following types:" in str(exc_info.value)

    with pytest.raises(TypeError) as exc_info:
        validate_type(np.str_("hello"), "np_str", allowed)
    assert "np_str must be one of the following types:" in str(exc_info.value)


def test_base_inputs_store_normalized_numpy_scalar() -> None:
    """NumPy scalar inputs are stored as their native Python equivalents."""
    inputs = BaseInputs(tdb=np.float32(25.0))

    assert inputs.tdb == 25.0
    assert not isinstance(inputs.tdb, np.generic)


def test_as_flag() -> None:
    """Yes/no outputs are 1.0 or 0.0, and NaN where the inputs are invalid (#484)."""
    np.testing.assert_array_equal(_as_flag([True, False]), [1.0, 0.0])
    np.testing.assert_array_equal(
        _as_flag([True, False, True], valid=[True, True, False]), [1.0, 0.0, np.nan]
    )
    assert _as_flag(True).dtype == float
