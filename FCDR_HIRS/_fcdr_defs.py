"""Definitions related to HIRS FCDR
"""

from typhon.physics.units.common import ureg, radiance_units as rad_u
from typhon.datasets._tovs_defs import (_u1_coding, _coding, _count_coding, _temp_coding)

_u_count_coding = _count_coding.copy()
_u_count_coding["dtype"] = "u2"
_u_count_coding["scale_factor"] = 0.005

# FIXME: uncertainty does NOT always have the same dimensions as quantity
# it belongs to...

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
        ("calibration_cycle", "calibrated_channel"),
        {"long_name": "Radiance for Internal Warm Calibration Target "
                      "(IWCT) "
                      "calculated from SRF with T_IWCT and Plack function",
         "units": "{:~}".format(rad_u["si"])},
        _coding),
    C_space = (
        "C_space",
        ("calibration_cycle", "calibration_position", "calibrated_channel"),
        {"long_name": "Counts for space view as used for calibration ",
         "units": "count"},
        _count_coding),
    C_IWCT = (
        "C_IWCT",
        ("calibration_cycle", "calibration_position", "calibrated_channel"),
        {"long_name": "Counts for IWCT view as used for calibration ",
         "units": "count"},
        _count_coding),
    C_Earth = (
        "C_Earth",
        ("scanline_earth", "scanpos", "calibrated_channel"),
        {"long_name": "Counts for Earth views ",
         "units": "count"},
        _count_coding),
    offset = (
        "offset",
        ("calibration_cycle", "calibration_position", "calibrated_channel"),
        {"long_name": "Offset (a_0)",
         "units": rad_u["si"]},
        _coding),
    slope = (
        "slope",
        ("calibration_cycle", "calibration_position", "calibrated_channel"),
        {"long_name": "Slope (a_1) or 1/gain",
         "units": (rad_u["si"]/ureg.count)},
        _coding),
    a_2 = (
        "a_2",
        ("calibrated_channel",),
        {"long_name": "Non-linearity (a_2)",
         "units": (rad_u["si"] / (ureg.count**2))},
        _coding),
    Rself = (
        "Rself",
        ("scanline_earth", "scanpos", "calibrated_channel"),
        {"long_name": "Correction to Earth radiance due to self-emission "
                      "change since last calibration",
         "units": rad_u["si"]},
        _coding),
    RselfIWCT = (
        "RselfIWCT",
        ("calibration_cycle", "calibration_position", "calibrated_channel"),
        {"long_name": "Self-emission during IWCT view",
         "units": rad_u["si"]},
        _coding),
    Rselfspace = (
        "Rselfspace",
        ("calibration_cycle", "calibration_position", "calibrated_channel"),
        {"long_name": "Self-emission during space view",
         "units": rad_u["si"]},
        _coding),
    R_Earth = (
        "R_Earth",
        ("scanline_earth", "scanpos", "calibrated_channel"),
        {"long_name": "Radiance for Earth views",
         "units": rad_u["si"]},
        _coding),
    R_refl = (
        "R_refl",
        ("calibration_cycle", "calibration_position", "calibrated_channel"),
        {"long_name": "Earthshine during calibration",
         "units": rad_u["si"]},
        _coding),
    ε = (
        "ε",
        (),
        {"long_name": "IWCT emissivity assumed in calibration",
         "units": "1"},
        _coding),
    a_3 = (
        "a_3",
        (),
        {"long_name": "IWCT emissivity correction assumed in calibration",
         "units": "1"},
        _coding),
    α = (
        "α",
        ("calibrated_channel",),
        {"long_name": "Offset for effective temperature correction",
         "units": "K"},
        _coding),
    Tstar = (
        "Tstar", # also has channel because α and β are involved
        ("calibration_cycle", "calibrated_channel"),
        {"long_name": "Corrected temperature for explicitly formulated "
            "estimate for SRF integration and reverse",
         "units": "K"},
        _temp_coding),
    β = (
        "β",
        ("calibrated_channel",),
        {"long_name": "Slope for effective temperature correction",
         "units": "1"},
        _coding),
#    λ_eff = (
#        "λ_eff",
#        ("calibrated_channel",),
#        {"long_name": "Effective wavelength for channel",
#         "units": "µm"},
#        _coding),
    f_eff = (
        "f_eff",
        ("calibrated_channel",),
        {"long_name": "Effective frequency for channel",
         "units": "PHz"},
        _coding),
#    ν_eff = (
#        "ν_eff",
#        ("calibrated_channel",),
#        {"long_name": "Effective wavenumber for channel",
#         "units": "1/cm"},
#        _coding),
    prt_number_iwt = (
        "prt_number_iwt",
        (),
        {"long_name": "Number of PRTs for IWCT temperature measurement",
         "units": "1"},
        _u1_coding),
    B = (
        "planck_radiance_IWCT",
        ("calibration_cycle", "calibrated_channel"),
        {"long_name": "Planck black body radiance",
         "note": "Valid at effective wavelength and temperature, see "
                 "λ_eff, Tstar, α, β.  Since effective wavelength is "
                 "a channel property this has a channel dimension.",
         "units": rad_u["si"]},
         _coding),
)
p = FCDR_data_vars_props
for (var, corr) in {("R_Earth", "O_Re"),
                    ("T_IWCT_calib_mean", "O_TIWCT"),
                    ("R_IWCT", "O_RIWCT")}:
    FCDR_data_vars_props[corr] = (
        "corr_" + p[var][0],
        p[var][1],
        {"long_name": "correction to " + p[var][2]["long_name"],
         "units": p[var][2]["units"]},
        _coding) # actually, always zero…
# FIXME: needs more uncertainties, or do I calculate this directly from
# the effects and is this one unused and should it be removed?
# In principle, no uncertainty needs more than one byte as we just express
# uncertainty on the last two significant digits...
# For example, 12.345(12)
        
FCDR_uncertainty_encodings = {}

FCDR_uncertainty_encodings["O_TPRT"] = _temp_coding
for (k, v) in FCDR_data_vars_props.items():
    if v[2].get("units") == "count":
        FCDR_uncertainty_encodings[k] = _u_count_coding
