"""Definitions related to HIRS FCDR
"""

from typhon.physics.units.common import radiance_units as rad_u
from typhon.datasets._tovs_defs import (_coding, _count_coding, _temp_coding)

# FIXME: uncertainty does NOT have the same dimensions as quantity it
# belongs to...

FCDR_data_vars_props = dict(
    T_IWCT_calib_mean = (
        "T_IWCT_calib_mean",
        ("calibration_cycle",),
        {"long_name": "Mean Temperature Internal Warm Calibration Target "
                      "as used for calibration (IWCT) ",
         "units": "K"},
        _temp_coding),
    R_IWCT = (
        "R_IWCT",
        ("calibration_cycle", "channel"),
        {"long_name": "Radiance for Internal Warm Calibration Target "
                      "(IWCT) "
                      "calculated from SRF with T_IWCT and Plack function",
         "units": "{:~}".format(rad_u["ir"])},
        _coding),
    C_space = (
        "C_space",
        ("calibration_cycle", "calibration_position", "channel"),
        {"long_name": "Counts for space view as used for calibration ",
         "units": "count"},
        _count_coding),
    C_IWCT = (
        "C_IWCT",
        ("calibration_cycle", "calibration_position", "channel"),
        {"long_name": "Counts for IWCT view as used for calibration ",
         "units": "count"},
        _count_coding),
    C_Earth = (
        "C_Earth",
        ("scanline_earth", "scanpos", "channel"),
        {"long_name": "Counts for Earth views ",
         "units": "count"},
        _count_coding),
    offset = (
        "offset",
        ("calibration_cycle", "calibration_position", "channel"),
        {"long_name": "Offset (a_0)"}, # units set in _quantity_to_xarray
        _coding),
    slope = (
        "slope",
        ("calibration_cycle", "calibration_position", "channel"),
        {"long_name": "Slope (a_1) or 1/gain"}, # units set in _quantity_to_xarray
        _coding),
    a_2 = (
        "a_2",
        ("channel",),
        {"long_name": "Non-linearity (a_2)"},
        _coding),
    Rself = (
        "Rself",
        ("scanline_earth", "scanpos", "calibrated_channel"),
        {"long_name": "Correction to Earth radiance due to self-emission "
                      "change since last calibration"},
        _coding),
)
# FIXME: needs more uncertainties, or do I calculate this directly from
# the effects and is this one unused and should it be removed?
FCDR_data_vars_props.update(
    u_C_IWCT = (
        "u_C_IWCT",
        FCDR_data_vars_props["C_IWCT"][1],
        {"long_name": "Uncertainty on C_IWCT"},
        _coding))

FCDR_uncertainty_encodings = dict(
    O_TPRT = _temp_coding)
