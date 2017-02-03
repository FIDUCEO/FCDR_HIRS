"""Definitions related to HIRS FCDR
"""

from typhon.physics.units.common import radiance_units as rad_u
from typhon.datasets._tovs_defs import (_coding, _count_coding, _temp_coding)

FCDR_data_vars_props = dict(
    T_IWCT_calib_mean = (
        "T_IWCT_calib_mean",
        ("calibration_cycle",),
        {"long_name": "Mean Temperature Internal Warm Calibration Target "
                      "as used for calibration (IWCT) "},
         "units": "K"},
        _temp_coding),
    R_IWCT = (
        "R_IWCT",
        ("calibration_cycle",),
        {"long_name": "Radiance for Internal Warm Calibration Target "
                      "(IWCT) "
                      "calculated from SRF with T_IWCT and Plack function"},
         "units": "{:~}".format(rad_u["ir"])},
        _coding),
    C_space = (
        "C_space",
        ("calibration_cycle", "calibration_position"),
        {"long_name": "Counts for space view as used for calibration ",
         "units": "count"},
        _count_coding),
    C_IWCT = (
        "C_IWCT",
        ("calibration_cycle", "calibration_position"),
        {"long_name": "Counts for IWCT view as used for calibration ",
         "units": "count"},
        _count_coding),
    offset = (
        "offset",
        ("calibration_cycle",),
        {"long_name": "Offset (a_0)"}, # units set in _quantity_to_xarray
        _coding),
    slope = (
        "slope",
        ("calibration_cycle",),
        {"long_name": "Slope (a_1) or 1/gain"}, # units set in _quantity_to_xarray
        _coding),
    a2 = (
        "a2",
        (),
        {"long_name": "Non-linearity (a_2)"},
        _coding),
    Rself = (
        "Rself",
        ("time", "scanpos", "calibrated_channel"),
        {"long_name": "Correction to Earth radiance due to self-emission "
                      "change since last calibration"},
        _coding)
)
