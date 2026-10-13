import warnings

import numpy as np
import pytest

from pythermalcomfort.models.ankle_draft import AnkleDraft, ankle_draft
from pythermalcomfort.utilities import Units
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_ankle_draft(get_test_url, retrieve_data) -> None:
    """Test that the function calculates the ankle draft correctly for various inputs."""
    reference_table = retrieve_reference_table(
        get_test_url,
        retrieve_data,
        Urls.ANKLE_DRAFT.name,
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = ankle_draft(**inputs)

        validate_result(result, outputs, tolerance)


def test_ankle_draft_invalid_input_range() -> None:
    """Test for ankle_draft with invalid input ranges."""
    # Test for ValueError
    with pytest.raises(ValueError):
        ankle_draft(25, 25, 0.3, 50, 1.2, 0.5, 7)
    # Test for ValueError
    with pytest.raises(ValueError):
        ankle_draft(25, 25, 0.3, 50, 1.2, 0.5, [0.1, 0.2, 0.3, 0.4])


def test_ankle_draft_outside_ashrae_range() -> None:
    """Test for ankle_draft with inputs outside the ASHRAE range."""
    r = ankle_draft(50, 25, 0.1, 50, 1.2, 0.5, 0.1)
    assert np.isnan(r.ppd_ad)

    r = ankle_draft([50, 45], 25, 0.1, 50, 1.2, 0.5, 0.1)
    assert np.all(np.isnan(r.ppd_ad))


def test_ankle_draft_list_inputs() -> None:
    """Test for ankle_draft with list inputs."""
    results = ankle_draft(
        tdb=[25, 26, 27],
        tr=[25, 26, 27],
        vr=[0.1, 0.1, 0.1],
        rh=[50, 50, 50],
        met=[1.2, 1.2, 1.2],
        clo=[0.5, 0.5, 0.5],
        v_ankle=[0.1, 0.1, 0.1],
        units=Units.SI.value,
    )
    assert isinstance(results, AnkleDraft)
    assert len(results.ppd_ad) == 3
    assert len(results.acceptability) == 3


def test_ankle_draft_invalid_list_inputs() -> None:
    """Test for TypeError with invalid types in list inputs."""
    with pytest.raises(TypeError):
        ankle_draft(
            tdb=[25, "invalid", 27],
            tr=[25, 26, 27],
            vr=[0.1, 0.1, 0.1],
            rh=[50, 50, 50],
            met=[1.2, 1.2, 1.2],
            clo=[0.5, 0.5, 0.5],
            v_ankle=[0.1, 0.1, 0.1],
            units=Units.SI.value,
        )


def test_ankle_draft_limit_inputs_false_suppresses_warning_and_nan() -> None:
    """limit_inputs=False bypasses range checks: no UserWarning, no NaN mask."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = ankle_draft(50, 25, 0.1, 50, 1.2, 0.5, 0.1, limit_inputs=False)
    assert not np.isnan(result.ppd_ad)


def test_ankle_draft_limit_inputs_false_skips_valueerror() -> None:
    """limit_inputs=False bypasses the vr < 0.2 equation-applicability check."""
    # vr=0.5 normally raises ValueError; with limit_inputs=False it should not
    result = ankle_draft(25, 25, 0.5, 50, 1.2, 0.5, 0.1, limit_inputs=False)
    assert not np.isnan(result.ppd_ad)


def test_ankle_draft_limit_inputs_true_still_warns_and_nans() -> None:
    """limit_inputs=True (default) keeps current behavior: warning + NaN mask."""
    with pytest.warns(UserWarning):
        result = ankle_draft(50, 25, 0.1, 50, 1.2, 0.5, 0.1)
    assert np.isnan(result.ppd_ad)


def test_ankle_draft_scalar_acceptability_is_bool() -> None:
    """Valid scalar inputs return Python True or False."""
    # ppd_ad is 18.5 at v_ankle=0.3 and 24.2 at v_ankle=0.5 (threshold is 20).
    acceptable = ankle_draft(
        tdb=25, tr=25, vr=0.2, rh=50, met=1.2, clo=0.5, v_ankle=0.3
    )
    assert acceptable.acceptability is True

    unacceptable = ankle_draft(
        tdb=25, tr=25, vr=0.1, rh=50, met=1.2, clo=0.5, v_ankle=0.5
    )
    assert unacceptable.acceptability is False

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        bypass = ankle_draft(
            tdb=25,
            tr=25,
            vr=0.1,
            rh=50,
            met=1.2,
            clo=0.5,
            v_ankle=0.1,
            limit_inputs=False,
        )
    assert bypass.acceptability is True


def test_ankle_draft_array_acceptability_is_bool() -> None:
    """Array inputs return an object array whose entries are True or False."""
    result = ankle_draft(
        tdb=25,
        tr=25,
        vr=0.1,
        rh=50,
        met=1.2,
        clo=0.5,
        v_ankle=[0.1, 0.5],
    )
    assert isinstance(result.acceptability, np.ndarray)
    assert result.acceptability.dtype == object
    assert result.acceptability[0] is True
    assert result.acceptability[1] is False

    # A length-1 list stays an array; it is not unwrapped to a scalar.
    single = ankle_draft(
        tdb=[25],
        tr=[25],
        vr=[0.1],
        rh=[50],
        met=[1.2],
        clo=[0.5],
        v_ankle=[0.1],
    )
    assert isinstance(single.acceptability, np.ndarray)
    assert single.acceptability.shape == (1,)
    assert single.acceptability.dtype == object
    assert single.acceptability[0] is True


def test_ankle_draft_mixed_acceptability_keeps_bool_and_nan() -> None:
    """Only the out-of-range row becomes nan; valid rows stay True or False."""
    with pytest.warns(UserWarning, match="'tdb'"):
        result = ankle_draft(
            tdb=[25, 25, 50],
            tr=25,
            vr=0.1,
            rh=50,
            met=1.2,
            clo=0.5,
            v_ankle=[0.1, 0.5, 0.1],
        )

    assert result.acceptability.dtype == object
    assert result.acceptability[0] is True
    assert result.acceptability[1] is False
    assert np.isnan(result.acceptability[2])
    assert np.isfinite(result.ppd_ad[0])
    assert np.isfinite(result.ppd_ad[1])
    assert np.isnan(result.ppd_ad[2])

    with pytest.warns(UserWarning, match="'tdb'"):
        outside = ankle_draft(
            tdb=50, tr=25, vr=0.1, rh=50, met=1.2, clo=0.5, v_ankle=0.1
        )
    assert outside.acceptability is np.nan
