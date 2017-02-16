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
    RselfIWCT = (
        "RselfIWCT",
        ("calibration_cycle", "calibration_position", "calibrated_channel"),
        {"long_name": "Self-emission during IWCT view"},
        _coding),
    Rselfspace = (
        "Rselfspace",
        ("calibration_cycle", "calibration_position", "calibrated_channel"),
        {"long_name": "Self-emission during space view"},
        _coding),
    R_Earth = (
        "R_Earth",
        ("scanline_earth", "scanpos", "channel"),
        {"long_name": "Radiance for Earth views",
         "units": rad_u["ir"]},
        _coding),
    R_refl = (
        "R_refl",
        ("calibration_cycle", "calibration_position", "channel"),
        {"long_name": "Earthshine during calibration",
         "units": rad_u["si"]},
        _coding),
    ε = (
        "ε",
        (),
        {"long_name": "emissivity"},
        _coding),
    a_3 = (
        "a_3",
        (),
        {"long_name": "emissivity correctio"},
        _coding),
    α = (
        "α",
        ("calibrated_channel",),
        {"long_name": "Offset for effective temperature correction",
         "units": "K"},
        _coding),
    β = (
        "β",
        ("calibrated_channel",),
        {"long_name": "Slope for effective temperature correction",
         "units": "1"},
        _coding),
    λ_eff = (
        "λ_eff",
        ("calibrated_channel",),
        {"long_name": "Effective wavelength for channel",
         "units": "µm"},
        _coding),
)
# FIXME: needs more uncertainties, or do I calculate this directly from
# the effects and is this one unused and should it be removed?
# In principle, no uncertainty needs more than one byte as we just express
# uncertainty on the last two significant digits...
# 12.345(12)
FCDR_data_vars_props.update(
    u_C_IWCT = (
        "u_C_IWCT",
        FCDR_data_vars_props["C_IWCT"][1],
        {"long_name": "Uncertainty on C_IWCT"},
        _coding))

FCDR_uncertainty_encodings = dict(
    O_TPRT = _temp_coding)
