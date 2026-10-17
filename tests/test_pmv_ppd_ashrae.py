import json
import math

import numpy as np
import pytest

from pythermalcomfort.classes_return import PMVPPDAshrae
from pythermalcomfort.models import pmv_ppd_ashrae
from pythermalcomfort.utilities import Models
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_pmv_ppd(get_test_url, retrieve_data) -> None:
    """Test that the function calculates the PMV and PPD values correctly for various inputs."""
    reference_table = retrieve_reference_table(
        get_test_url,
        retrieve_data,
        Urls.PMV_PPD.name,
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        # TODO change the validation table code and removed the following
        if "standard" not in inputs:
            inputs["standard"] = "iso"
        if inputs["standard"].lower() == "ashrae":
            inputs["model"] = Models.ashrae_55_2023.value
            del inputs["standard"]
            result = pmv_ppd_ashrae(**inputs)

            validate_result(result, outputs, tolerance)


class TestPmvPpd:
    """Test cases for the PMV and PPD model."""

    def test_thermal_sensation(self) -> None:
        """Test that the function returns the correct thermal sensation values."""
        np.testing.assert_equal(
            pmv_ppd_ashrae(
                [16, 21, 24, 26, 29, 32, 34, 33.47, 33.46],
                [16, 21, 24, 26, 29, 32, 34, 33.47, 33.46],
                0.2,
                50,
                1,
                0.5,
                model=Models.ashrae_55_2023.value,
            ).tsv,
            [
                "Cold",
                "Cool",
                "Slightly Cool",
                "Neutral",
                "Slightly Warm",
                "Warm",
                "Hot",
                "Warm",
                "Warm",
            ],
        )

    #  Returns NaN for invalid input values
    def test_returns_nan_for_invalid_input_values(self) -> None:
        """Test that the function returns NaN for invalid input values."""
        # test airspeed limits
        np.testing.assert_equal(
            pmv_ppd_ashrae(
                [26, 24, 22, 26, 24, 22],
                [26, 24, 22, 26, 24, 22],
                [0.9, 0.6, 0.3, 0.9, 0.6, 0.3],
                50,
                [1.1, 1.1, 1.1, 1.3, 1.3, 1.3],
                [0.5, 0.5, 0.5, 0.7, 0.7, 0.7],
                model=Models.ashrae_55_2023.value,
                airspeed_control=False,
            ).pmv,
            [np.nan, np.nan, np.nan, -0.14, -0.43, -0.57],
        )

        np.testing.assert_equal(
            pmv_ppd_ashrae(
                [41, 20, 20, 20, 20, 39],
                [20, 41, 20, 20, 20, 39],
                [0.1, 0.1, 2.1, 0.1, 0.1, 0.1],
                50,
                [1.1, 1.1, 1.1, 0.7, 1.1, 3.9],
                [0.5, 0.5, 0.5, 0.5, 2.1, 1.9],
                model=Models.ashrae_55_2023.value,
            ).pmv,
            np.asarray([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
        )

        result = pmv_ppd_ashrae(
            41,
            41,
            2,
            50,
            0.7,
            2.1,
            model=Models.ashrae_55_2023.value,
            limit_inputs=False,
        )
        np.testing.assert_equal(
            result,
            PMVPPDAshrae(
                pmv=np.float64(4.48),
                ppd=np.float64(100.0),
                tsv=np.str_("Hot"),
                compliance=False,
            ),
        )
        # assert_equal treats np.bool_(False) as equal to False.
        assert result.compliance is False
        assert type(result.compliance) is bool

    def test_compliance(self) -> None:
        """Test that the function returns the correct compliance values."""
        # Test scalar values - compliant case (PMV within -0.5 to 0.5)
        result = pmv_ppd_ashrae(
            tdb=25,
            tr=25,
            vr=0.1,
            rh=50,
            met=1.0,
            clo=0.5,
            model=Models.ashrae_55_2023.value,
        )
        assert result.compliance is True

        # Test scalar values - non-compliant case (PMV outside -0.5 to 0.5)
        result = pmv_ppd_ashrae(
            tdb=30,
            tr=30,
            vr=0.1,
            rh=50,
            met=1.0,
            clo=0.5,
            model=Models.ashrae_55_2023.value,
        )
        assert result.compliance is False

        # Test array values with known PMV results
        result = pmv_ppd_ashrae(
            tdb=[22, 25, 30],
            tr=[22, 25, 30],
            vr=0.1,
            rh=50,
            met=1.0,
            clo=0.5,
            model=Models.ashrae_55_2023.value,
        )
        # PMV around -0.77, -0.03, 1.17 approximately
        # First is not compliant (< -0.5), second is compliant, third is not compliant (> 0.5)
        expected_compliance = np.array([False, True, False])
        np.testing.assert_array_equal(result.compliance, expected_compliance)

        result = pmv_ppd_ashrae(
            tdb=22.5,
            tr=22.5,
            vr=0.1,
            rh=50,
            met=1.0,
            clo=0.5,
            model=Models.ashrae_55_2023.value,
        )
        # For these inputs, PMV is outside (-0.5, 0.5), so should be non-compliant
        assert result.pmv < -0.5 or result.pmv > 0.5
        assert result.compliance is False

        # Also test inputs that produce PMV close to upper bound
        result = pmv_ppd_ashrae(
            tdb=27.45,
            tr=27.45,
            vr=0.1,
            rh=50,
            met=1.0,
            clo=0.5,
            model=Models.ashrae_55_2023.value,
        )
        assert result.pmv < -0.5 or result.pmv > 0.5
        assert result.compliance is False

    def test_compliance_scalar_is_python_bool_without_input_limits(self) -> None:
        """Test that scalar compliance is a Python bool when limit_inputs is False."""
        compliant = pmv_ppd_ashrae(
            tdb=25,
            tr=25,
            vr=0.1,
            rh=50,
            met=1.2,
            clo=0.5,
            model=Models.ashrae_55_2023.value,
            limit_inputs=False,
        )
        assert compliant.compliance is True
        assert type(compliant.compliance) is bool
        assert json.dumps(compliant.compliance) == "true"
        assert "np.True_" not in str(compliant)

        noncompliant = pmv_ppd_ashrae(
            tdb=30,
            tr=30,
            vr=0.1,
            rh=50,
            met=1.2,
            clo=0.5,
            model=Models.ashrae_55_2023.value,
            limit_inputs=False,
        )
        assert noncompliant.compliance is False
        assert type(noncompliant.compliance) is bool
        assert json.dumps(noncompliant.compliance) == "false"

        with pytest.warns(UserWarning, match=r"'tdb'"):
            masked = pmv_ppd_ashrae(
                tdb=5,
                tr=25,
                vr=0.1,
                rh=50,
                met=1.2,
                clo=0.5,
                model=Models.ashrae_55_2023.value,
                limit_inputs=True,
            )
        assert isinstance(masked.compliance, float)
        assert math.isnan(masked.compliance)

        with pytest.warns(UserWarning, match=r"'tdb'"):
            array_result = pmv_ppd_ashrae(
                tdb=[25, 5],
                tr=25,
                vr=0.1,
                rh=50,
                met=1.2,
                clo=0.5,
                model=Models.ashrae_55_2023.value,
                limit_inputs=True,
            )
        assert isinstance(array_result.compliance, np.ndarray)
        assert array_result.compliance.dtype == object
        assert array_result.compliance[0] is True
        assert math.isnan(array_result.compliance[1])

        array_unmasked = pmv_ppd_ashrae(
            tdb=[25, 30],
            tr=25,
            vr=0.1,
            rh=50,
            met=1.2,
            clo=0.5,
            model=Models.ashrae_55_2023.value,
            limit_inputs=False,
        )
        assert isinstance(array_unmasked.compliance, np.ndarray)
        assert array_unmasked.compliance.dtype == object
        assert array_unmasked.compliance[0] is True
        assert array_unmasked.compliance[1] is False

    def test_wrong_standard(self) -> None:
        """Test that the function raises a ValueError for an unsupported model."""
        with pytest.raises(ValueError):
            pmv_ppd_ashrae(25, 25, 0.1, 50, 1.1, 0.5, model="random")

    def test_out_of_range_warns(self) -> None:
        """Inputs outside ASHRAE 55 limits trigger a UserWarning naming the parameter."""
        with pytest.warns(UserWarning, match=r"'tdb'"):
            result = pmv_ppd_ashrae(
                tdb=41,
                tr=25,
                vr=0.1,
                rh=50,
                met=1.1,
                clo=0.5,
                model=Models.ashrae_55_2023.value,
            )
        assert np.isnan(result.pmv)

    def test_limit_inputs_false_no_warning(self, recwarn) -> None:
        """With limit_inputs=False, no UserWarning is raised for out-of-range inputs."""
        pmv_ppd_ashrae(
            tdb=41,
            tr=41,
            vr=2,
            rh=50,
            met=0.7,
            clo=2.1,
            model=Models.ashrae_55_2023.value,
            limit_inputs=False,
        )
        assert len(recwarn) == 0

    def test_airspeed_control_false_cond1_warns(self) -> None:
        """airspeed_control=False: v > 0.8 with low clo and met triggers UserWarning."""
        # to ≈ 26 > 25.5, so only cond1 applies: v=0.9 > 0.8, clo=0.5 < 0.7, met=1.1 < 1.3
        with pytest.warns(
            UserWarning, match=r"airspeed_control=False.*exceeding 0\.8 m/s"
        ):
            pmv_ppd_ashrae(
                tdb=26,
                tr=26,
                vr=0.9,
                rh=50,
                met=1.1,
                clo=0.5,
                model=Models.ashrae_55_2023.value,
                airspeed_control=False,
            )

    def test_airspeed_control_false_cond2_warns(self) -> None:
        """airspeed_control=False: v exceeds comfort-zone limit triggers UserWarning."""
        # to ≈ 24, so 23 < to < 25.5; v=0.6 exceeds computed v_limit ≈ 0.32
        with pytest.warns(UserWarning, match=r"airspeed_control=False.*comfort zone"):
            pmv_ppd_ashrae(
                tdb=24,
                tr=24,
                vr=0.6,
                rh=50,
                met=1.1,
                clo=0.5,
                model=Models.ashrae_55_2023.value,
                airspeed_control=False,
            )

    def test_airspeed_control_false_cond3_warns(self) -> None:
        """airspeed_control=False: v > 0.2 when to <= 23°C triggers UserWarning."""
        # to ≈ 22 <= 23; v=0.3 > 0.2, clo=0.5 < 0.7, met=1.1 < 1.3
        with pytest.warns(
            UserWarning, match=r"airspeed_control=False.*exceeding 0\.2 m/s"
        ):
            pmv_ppd_ashrae(
                tdb=22,
                tr=22,
                vr=0.3,
                rh=50,
                met=1.1,
                clo=0.5,
                model=Models.ashrae_55_2023.value,
                airspeed_control=False,
            )
