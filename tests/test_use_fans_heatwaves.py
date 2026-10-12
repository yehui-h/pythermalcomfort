import warnings

import numpy as np
import pytest

from pythermalcomfort.models import use_fans_heatwaves
from tests.conftest import Urls, retrieve_reference_table, validate_result

_FLAGS = (
    "heat_strain",
    "heat_strain_blood_flow",
    "heat_strain_w",
    "heat_strain_sweating",
)

# Inside the documented applicability limits.
_COMFORTABLE = {
    "tdb": 35,
    "tr": 35,
    "v": 1.0,
    "rh": 50,
    "met": 1.2,
    "clo": 0.5,
}

# Mild conditions: none of the physiological caps are reached.
_MILD = {
    "tdb": 25,
    "tr": 25,
    "v": 0.2,
    "rh": 40,
    "met": 1.0,
    "clo": 0.5,
}

# Hot conditions: skin blood flow and wettedness reach their caps.
_HOT = {
    "tdb": 45,
    "tr": 45,
    "v": 0.2,
    "rh": 80,
    "met": 1.8,
    "clo": 0.9,
}


def _call(**overrides):
    """Call use_fans_heatwaves with comfortable defaults."""
    return use_fans_heatwaves(**{**_COMFORTABLE, **overrides})


def _assert_bool_entries(value) -> None:
    """Assert value is a Python bool or an object array of Python bools."""
    if isinstance(value, np.ndarray):
        assert value.dtype == object
        items = list(value.flat)
        assert items
        assert all(type(item) is bool for item in items)
        return
    assert type(value) is bool


def test_use_fans_heatwaves(get_test_url, retrieve_data) -> None:
    """Test that the function calculates the use of fans during heatwaves correctly for various inputs."""
    reference_table = retrieve_reference_table(
        get_test_url,
        retrieve_data,
        Urls.USE_FANS_HEATWAVES.name,
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = use_fans_heatwaves(**inputs)

        validate_result(result, outputs, tolerance)


def test_scalar_heat_strain_flags_are_bool() -> None:
    """Valid scalar inputs return True or False, not 0.0 or 1.0."""
    comfortable = _call()
    assert comfortable.heat_strain_blood_flow is False
    assert comfortable.heat_strain_w is False
    assert comfortable.heat_strain_sweating is False
    assert comfortable.heat_strain is False

    hot = use_fans_heatwaves(**_HOT)
    assert hot.heat_strain_blood_flow is True
    assert hot.heat_strain_w is True
    assert hot.heat_strain_sweating is False
    assert hot.heat_strain is True

    # Regulatory sweating is capped at max_sweating, which trips only that flag.
    sweating = _call(max_sweating=80)
    assert sweating.heat_strain_sweating is True
    assert sweating.heat_strain_blood_flow is False
    assert sweating.heat_strain_w is False
    assert sweating.heat_strain is True

    for result in (comfortable, hot, sweating):
        assert result.heat_strain is (
            result.heat_strain_blood_flow
            or result.heat_strain_w
            or result.heat_strain_sweating
        )


def test_array_heat_strain_flags_are_bool() -> None:
    """Array inputs return an object array whose entries are True or False."""
    result = use_fans_heatwaves(
        tdb=[_MILD["tdb"], _HOT["tdb"]],
        tr=[_MILD["tr"], _HOT["tr"]],
        v=0.2,
        rh=[_MILD["rh"], _HOT["rh"]],
        met=[_MILD["met"], _HOT["met"]],
        clo=[_MILD["clo"], _HOT["clo"]],
    )
    np.testing.assert_array_equal(result.heat_strain_blood_flow, [False, True])
    np.testing.assert_array_equal(result.heat_strain_w, [False, True])
    np.testing.assert_array_equal(result.heat_strain_sweating, [False, False])
    np.testing.assert_array_equal(result.heat_strain, [False, True])
    for name in _FLAGS:
        _assert_bool_entries(getattr(result, name))

    # A length-1 list stays an array; it is not unwrapped to a scalar.
    single = use_fans_heatwaves(
        tdb=[35],
        tr=[35],
        v=[1.0],
        rh=[50],
        met=[1.2],
        clo=[0.5],
    )
    assert isinstance(single.heat_strain, np.ndarray)
    assert single.heat_strain.shape == (1,)
    assert single.heat_strain[0] is False


def test_limit_inputs_false_round_output_true_keeps_bool_flags() -> None:
    """Rounding must not be applied to bool flags, including out-of-range inputs."""
    scalar = _call(limit_inputs=False, round_output=True)
    assert scalar.heat_strain is False
    assert scalar.e_skin == pytest.approx(63.0)

    hot = use_fans_heatwaves(**_HOT, limit_inputs=False, round_output=True)
    assert hot.heat_strain_blood_flow is True
    assert hot.heat_strain_w is True
    assert hot.heat_strain_sweating is False
    assert hot.heat_strain is True

    array = use_fans_heatwaves(
        tdb=[_MILD["tdb"], _HOT["tdb"]],
        tr=[_MILD["tr"], _HOT["tr"]],
        v=0.2,
        rh=[_MILD["rh"], _HOT["rh"]],
        met=[_MILD["met"], _HOT["met"]],
        clo=[_MILD["clo"], _HOT["clo"]],
        limit_inputs=False,
        round_output=True,
    )
    _assert_bool_entries(array.heat_strain)
    np.testing.assert_array_equal(array.heat_strain, [False, True])

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        outside = _call(tdb=10, limit_inputs=False, round_output=True)
    _assert_bool_entries(outside.heat_strain)
    assert np.isfinite(outside.e_skin)


def test_round_output_rounds_numbers_and_leaves_flags() -> None:
    """round_output changes numeric fields only."""
    raw = _call(round_output=False)
    rounded = _call(round_output=True)
    for name in ("e_skin", "e_rsw", "t_core", "t_skin", "m_bl", "m_rsw", "w", "w_max"):
        assert getattr(rounded, name) == pytest.approx(
            np.around(np.asarray(getattr(raw, name)), 1),
        )
    assert rounded.e_skin == pytest.approx(63.0)
    for name in _FLAGS:
        assert type(getattr(raw, name)) is bool
        assert getattr(raw, name) is getattr(rounded, name)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("tdb", 19.9),
        ("tdb", 50.1),
        ("tr", 19.9),
        ("tr", 50.1),
        ("v", 0.09),
        ("v", 4.51),
        ("met", 0.69),
        ("met", 2.01),
        ("clo", -0.01),
        ("clo", 1.01),
    ],
)
def test_out_of_range_scalar_flags_are_nan(name, value) -> None:
    """Inputs outside the applicability limits set every heat-strain flag to nan."""
    with pytest.warns(UserWarning, match=rf"'{name}'"):
        result = _call(**{name: value})
    for flag in _FLAGS:
        assert np.isnan(getattr(result, flag))
    assert np.isnan(result.e_skin)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("tdb", 20.0),
        ("tdb", 50.0),
        ("tr", 20.0),
        ("tr", 50.0),
        ("v", 0.1),
        ("v", 4.5),
        ("met", 0.7),
        ("met", 2.0),
        ("clo", 0.0),
        ("clo", 1.0),
    ],
)
def test_applicability_limits_are_inclusive(name, value) -> None:
    """Values on the inclusive limits stay boolean rather than becoming nan."""
    result = _call(**{name: value})
    for flag in _FLAGS:
        assert type(getattr(result, flag)) is bool
    assert np.isfinite(result.e_skin)


def test_rh_outside_0_100_does_not_blank_flags() -> None:
    """Relative humidity is warned about and is not part of the validity mask."""
    with pytest.warns(UserWarning, match="'rh'"):
        result = _call(rh=110)
    for flag in _FLAGS:
        assert type(getattr(result, flag)) is bool
    assert np.isfinite(result.e_skin)
    assert result.heat_strain is (
        result.heat_strain_blood_flow
        or result.heat_strain_w
        or result.heat_strain_sweating
    )


def test_mixed_array_keeps_bools_and_nan() -> None:
    """Only the out-of-range row becomes nan; valid rows stay True or False."""
    with pytest.warns(UserWarning, match="'tdb'"):
        result = use_fans_heatwaves(
            tdb=[_MILD["tdb"], _HOT["tdb"], 10],
            tr=[_MILD["tr"], _HOT["tr"], _MILD["tr"]],
            v=0.2,
            rh=[_MILD["rh"], _HOT["rh"], _MILD["rh"]],
            met=[_MILD["met"], _HOT["met"], _MILD["met"]],
            clo=[_MILD["clo"], _HOT["clo"], _MILD["clo"]],
        )

    np.testing.assert_array_equal(result.heat_strain_blood_flow[:2], [False, True])
    np.testing.assert_array_equal(result.heat_strain_w[:2], [False, True])
    np.testing.assert_array_equal(result.heat_strain_sweating[:2], [False, False])
    np.testing.assert_array_equal(result.heat_strain[:2], [False, True])
    for name in _FLAGS:
        column = getattr(result, name)
        assert column.dtype == object
        assert type(column[0]) is bool
        assert type(column[1]) is bool
        assert np.isnan(column[2])
    assert np.isfinite(result.e_skin[0])
    assert np.isfinite(result.e_skin[1])
    assert np.isnan(result.e_skin[2])
