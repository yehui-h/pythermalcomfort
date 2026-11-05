from __future__ import annotations

import warnings

import numpy as np

from pythermalcomfort._internal.validation import _format_violation_detail, _valid_range
from pythermalcomfort.environment import operative_tmp


def _check_ashrae55_compliance(**kwargs):
    """ASHRAE 55-2023 input applicability check.

    Implements the Table 7.3.4 single-variable limits. With ``airspeed_control=False``
    it also applies the no-occupant-control air-speed limits (Addendum d to 55-2017,
    Section 5.3.2.4): 0.8 m/s above 25.5 °C, the equal-SET curve from 23 °C through 25.5
    °C, and 0.2 m/s below 23 °C. The clothing and metabolic-rate exception applies only
    below 23 °C, and only above 0.7 clo or above 1.3 met. Operative temperature uses the
    ASHRAE weighting. Single-use compliance checks for other standards (ISO 7730, ISO
    7933, fan_heatwaves) live inline in the respective model files.

    ``v_param_name`` (default ``"v"``) controls how the airspeed variable is rendered in
    UserWarnings emitted by this function. Callers whose signature names the airspeed
    argument ``vr`` should pass ``v_param_name="vr"`` so the warning text matches the
    caller's variable name. It does not affect which value is checked.
    """
    default_kwargs = {"airspeed_control": True, "v_param_name": "v"}
    params = {**default_kwargs, **kwargs}
    v_param_name = params["v_param_name"]
    values_to_return = {}

    tdb_valid = _valid_range(params["tdb"], (10.0, 40.0), param_name="tdb")
    tr_valid = _valid_range(params["tr"], (10.0, 40.0), param_name="tr")

    values_to_return["tdb"] = tdb_valid
    values_to_return["tr"] = tr_valid

    if "v" in params:
        v_valid = _valid_range(params["v"], (0.0, 2.0), param_name=v_param_name)
        values_to_return["v"] = v_valid

    if not params["airspeed_control"]:
        missing = [k for k in ("v", "met", "clo") if k not in params]
        if missing:
            msg = f"airspeed_control=False requires {', '.join(missing)} to check the limits."
            raise ValueError(msg)
        v = np.asarray(params["v"], dtype=np.float64)
        clo = np.asarray(params["clo"], dtype=np.float64)
        met = np.asarray(params["met"], dtype=np.float64)
        # Addendum d to ASHRAE 55-2017, Section 5.3.2.4. Operative temperature
        # is the ASHRAE weighting (Appendix A), not the ISO 7726 formula.
        to = operative_tmp(tdb=params["tdb"], tr=params["tr"], v=v, standard="ashrae")

        # (a) above 25.5 °C the cap is 0.8 m/s, with no clothing or activity exception.
        above_cap = (to > 25.5) & (v > 0.8)
        if np.any(above_cap):
            detail = _format_violation_detail(v, above_cap)
            warnings.warn(
                f"airspeed_control=False: '{v_param_name}' has {detail} exceeding "
                "0.8 m/s above an operative temperature of 25.5°C and will be "
                "set to NaN.",
                UserWarning,
                stacklevel=2,
            )
        v_valid = np.where(above_cap, np.nan, v_valid)

        # (b) from 23 °C through 25.5 °C the cap is the equal-SET curve. (a) is
        # strictly above 25.5 °C and (c) is strictly below 23 °C, so both
        # endpoints belong to this clause.
        v_limit = 50.49 - 4.4047 * to + 0.096425 * to * to
        on_curve = (to >= 23.0) & (to <= 25.5) & (v > v_limit)
        if np.any(on_curve):
            detail = _format_violation_detail(v, on_curve)
            warnings.warn(
                f"airspeed_control=False: '{v_param_name}' has {detail} exceeding "
                "the ASHRAE 55 air-speed limit from 23°C through 25.5°C and "
                "will be set to NaN.",
                UserWarning,
                stacklevel=2,
            )
        v_valid = np.where(on_curve, np.nan, v_valid)

        # (c) below 23 °C the cap is 0.2 m/s. The exception is only this clause,
        # and only strictly above 0.7 clo or strictly above 1.3 met.
        below_cap = (to < 23.0) & (v > 0.2) & (clo <= 0.7) & (met <= 1.3)
        if np.any(below_cap):
            detail = _format_violation_detail(v, below_cap)
            warnings.warn(
                f"airspeed_control=False: '{v_param_name}' has {detail} exceeding "
                "0.2 m/s below an operative temperature of 23°C and will be "
                "set to NaN.",
                UserWarning,
                stacklevel=2,
            )
        v_valid = np.where(below_cap, np.nan, v_valid)

        values_to_return["v"] = v_valid

    if "met" in params:
        met_valid = _valid_range(params["met"], (1.0, 4.0), param_name="met")
        clo_valid = _valid_range(params["clo"], (0.0, 1.5), param_name="clo")

        values_to_return["met"] = met_valid
        values_to_return["clo"] = clo_valid

    if "v_limited" in params:
        valid = _valid_range(params["v_limited"], (0.0, 0.2), param_name="v_limited")
        values_to_return["v_limited"] = valid

    return values_to_return.values()
