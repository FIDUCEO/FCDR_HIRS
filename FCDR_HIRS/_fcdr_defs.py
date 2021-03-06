"""Definitions related to HIRS FCDR

This module contains various definitions related to the FCDR.  They extend
definitions in ``typhon.datasets._tovs_defs``, which are more related to the
reading routine.  Important definitions here:

:attr:`FCDR_data_vars_props`
    Defines how data are stored in the debug FCDR,
:attr:`FCDR_uncertainty_encodings`
    Defines encodings for uncertainties, which may differ from the
    quantity the uncertainty belongs to.

However, for the purposes of the easy FCDR, a lot of this module is
superseded by Tom Blocks routines.
"""

import itertools
import enum
from typhon.physics.units.common import ureg, radiance_units as rad_u
from typhon.datasets._tovs_defs import (_u1_coding, _u2_coding, _coding,
            _count_coding, _temp_coding, _cal_coding, _latlon_coding,
            _ang_coding, _u4_coding)

# as per https://github.com/FIDUCEO/FCDR_HIRS/issues/69

_debug_bt_coding = _temp_coding.copy()
_debug_bt_coding["dtype"] = "u4"
#_debug_bt_coding["least_significant_digit"] = 4
_debug_bt_coding["scale_factor"] /= 10

_debug_Re_coding = _coding.copy()
_debug_Re_coding["dtype"] = "u8"
_debug_Re_coding["scale_factor"] = 1e-21 # mW units...
_debug_Re_coding["_FillValue"] = _u2_coding["_FillValue"]

_u_count_coding = _count_coding.copy()
_u_count_coding["dtype"] = "u2"
_u_count_coding["scale_factor"] = 0.005

_corr_coding = _coding.copy()
_corr_coding["dtype"] = "u2"
_corr_coding["scale_factor"] = 0.001
_corr_coding["_FillValue"] = _u2_coding["_FillValue"]

_str_coding = _coding.copy()
del _str_coding["dtype"]
del _str_coding["_FillValue"]

# FIXME: uncertainty does NOT always have the same dimensions as quantity
# it belongs to...

#: Dictionary storing information for debug FCDR fields, as far as not inherited from typhon
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
        ("scanline_earth", "calibrated_channel"),
        {"long_name": "Correction to Earth radiance due to self-emission "
                      "change since last calibration",
         "units": rad_u["si"]},
        _coding),
    RselfIWCT = (
        "RselfIWCT",
        ("calibration_cycle", "calibrated_channel"),
        {"long_name": "Self-emission during IWCT view",
         "units": rad_u["si"]},
        _coding),
    Rselfspace = (
        "Rselfspace",
        ("calibration_cycle", "calibrated_channel"),
        {"long_name": "Self-emission during space view",
         "units": rad_u["si"]},
        _coding),
    R_Earth = (
        "R_Earth",
        ("scanline_earth", "scanpos", "calibrated_channel"),
        {"long_name": "Radiance for Earth views",
         "units": rad_u["si"]},
        _debug_Re_coding),
    T_b = (
        "T_b",
        ("scanline_earth", "scanpos", "calibrated_channel"),
        {"long_name": "Brightness temperature for Earth views",
         "units": "K"},
         _debug_bt_coding),
    T_bstar = (
        "T_bstar",
        ("scanline_earth", "scanpos", "calibrated_channel"),
        {"long_name": "Uncorrected monochromatic brightness temperature for Earth views",
         "units": "K"},
         _debug_bt_coding),
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
         "units": ureg.Unit("1")},
        _coding),
    a_3 = (
        "a_3",
        (),
        {"long_name": "IWCT emissivity correction assumed in calibration",
         "units": ureg.Unit("1")},
        _coding),
    a_4 = (
        "a_4",
        (),
        {"long_name": "Bias harmonisation term",
         "note": ("Self-emission model can only be trained with subsequent "
                  "space views at different thermal environments, but "
                  "in reality the geometry during Earth or IWCT views is "
                  "somewhat different, so the self-emission is, too.  The "
                  "bias term contains a harmonisation estimate of this "
                  "difference."),
         "units": rad_u["si"]},
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
        _debug_bt_coding),
    β = (
        "β",
        ("calibrated_channel",),
        {"long_name": "Slope for effective temperature correction",
         "units": ureg.Unit("1")},
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
         "units": ureg.Unit("1")},
        _u1_coding),
    prt_reading = (
        "prt_reading",
        (),
        {"long_name": "Number of PRT views per PRT for IWCT temperature measurement",
         "units": ureg.Unit("1")},
        _u1_coding),
    prt_iwct_polynomial_order = (
        "prt_iwct_polynomial_order",
        (),
        {"long_name": "Order of polynomial for each IWCT PRT calibration",
         "units": ureg.Unit("1")},
        _u1_coding),
    prt_iwct_counts = (
        "prt_iwct_counts",
        ("time", "prt_number", "prt_reading"),
        {"long_name": "IWCT PRT counts",
         "units": ureg.count},
        _count_coding),
    prt_iwct_coefficients = (
        "prt_iwct_coefficients",
        ("prt_number", "prt_iwct_polynomial_order"),
        {"long_name": "IWCT PRT coefficients",
         "units": [ureg.K, ureg.K/ureg.count, ureg.K/ureg.count**2,
                   ureg.K/ureg.count**3, ureg.K/ureg.count**4,
                   ureg.K/ureg.count**4]},
        _coding),
    prt_iwct_temperature = (
        "prt_iwct_temperature",
        ("time", "prt_number", "prt_reading"),
        {"long_name": "IWCT PRT temperature",
         "units": ureg.K},
        _temp_coding),
    B = (
        "planck_radiance_IWCT",
        ("calibration_cycle", "calibrated_channel"),
        {"long_name": "Planck black body radiance",
         "note": "Valid at effective wavelength and temperature, see "
                 "λ_eff, Tstar, α, β.  Since effective wavelength is "
                 "a channel property this has a channel dimension.",
         "units": rad_u["si"]},
         _coding),
    planck_constant = (
        "planck_constant",
        (),
        {"long_name": "Max Planck constant",
         "units": "J s"},
        _coding),
    speed_of_light = (
        "speed_of_light",
        (),
        {"long_name": "Speed of light in vacuum",
         "units": "m/s"},
        _coding),
    boltzmann_constant = (
        "boltzmann_constant",
        (),
        {"long_name": "Boltzmann constant",
         "units": "J/K"},
        _coding),
    # … and the coordinates not earlier defined …
    calibration_cycle = (
        "calibration_cycle",
        ("calibration_cycle",),
        {"long_name": "Time for calibration cycle"},
        _cal_coding),
    calibrated_channel = (
        "calibrated_channel",
        ("calibrated_channel",),
        {"long_name": "Calibrated channel"},
        _u1_coding),
    rself_update_time = (
        "rself_update_time",
        ("rself_update_time",),
        {"long_name": "Time of last update of self-emission params"},
        _cal_coding),
    channel = (
        "channel",
        ("channel",),
        {"long_name": "Channel"},
        _u1_coding),
    scanpos = (
        "scanpos",
        ("scanpos",),
        {"scanpos": "Scan position"},
        _u1_coding),
    scanline_earth = (
        "scanline_earth",
        ("scanline_earth",),
        {"long_name": "Time for Earth viewing scanlines"},
        _cal_coding),
    scanline = (
        "scanline",
        ("scanline",),
        {"long_name": "Scanline number (in original data)"},
        _u2_coding),
    channel_correlation_matrix = (
        "channel_correlation_matrix",
        ("channel", "channel"),
        {"long_name": "Channel error correlation matrix",
         "units": "dimensionless",
         "valid_range": [0, 1]},
         _corr_coding),
    lookup_table_BT = (
        "lookup_table_BT",
        ("lut_size", "calibrated_channel"),
        {"long_name": "Brightness temperatures for look-up table",
         "units": "K"},
        _temp_coding),
    lookup_table_radiance = (
        "lookup_table_radiance",
        ("lut_size", "calibrated_channel"),
        {"long_name": "Radiance for look-up table",
         "units": "cm*mW/m**2/sr"},
        _coding),
    quality_scanline_bitmask = (
        "quality_scanline_bitmask",
        ("scanline",),
        {"long_name": "Bitmask for quality per scanline"},
        _u1_coding),
    quality_channel_bitmask = (
        "quality_channel_bitmask",
        ("scanline", "channel"),
        {"long_name": "Bitmask for quality per channel"},
        _u1_coding),
    quality_minorframe_bitmask = (
        "quality_minorframe_bitmask",
        ("scanline", "minor_frame"),
        {"long_name": "Bitmask for quality per minor frame"},
        _u1_coding),
    quality_pixel_bitmask = (
        "quality_pixel_bitmask",
        ("scanline_earth", "scanpos", "calibrated_channel"),
        {"long_name": "Bitmask for quality per pixel"},
        _u1_coding),
    cross_line_radiance_error_correlation_length_scale_structured_effects = (
        "cross_line_radiance_error_correlation_length_scale_structured_effects",
        ("calibrated_channel",),
        {"long_name": "Cross line radiance error correlation length scale for structured effects",
         "units": "scanlines"},
        _u2_coding),
    cross_element_radiance_error_correlation_length_scale_structured_effects = (
        "cross_element_radiance_error_correlation_length_scale_structured_effects",
        ("calibrated_channel",),
        {"long_name": "Cross element radiance error correlation length scale for structured effocts",
         "units": "scanlines"},
        _u2_coding),
    cross_channel_error_correlation_matrix_independent_effects = (
        "cross_channel_error_correlation_matrix_independent_effects",
        ("calibrated_channel", "calibrated_channel"),
        {"long_name": "Channel error correlation matrix for independent effects",
         "units": "scanlines"},
        _corr_coding),
    cross_channel_error_correlation_matrix_structured_effects = (
        "cross_channel_error_correlation_matrix_structured_effects",
        ("calibrated_channel", "calibrated_channel"),
        {"long_name": "Channel error correlation matrix for structured effects",
         "units": "scanlines"},
        _corr_coding),
    filename = (
        "filename",
        ("time",),
        {"long_name": "Original filename in L1B"},
        _str_coding),
    platform_zenith_angle = (
        "platform_zenith_angle",
        ("scanline_earth", "scanpos"),
        {"long_name": "Zenith angle of satellite as seen from the ground",
         "units": "degrees"},
        _ang_coding),
    platform_azimuth_angle = (
        "platform_azimuth_angle",
        ("scanline_earth", "scanpos"),
        {"long_name": "Azimuth angle (from the north) of satellite as seen from the ground",
         "units": "degrees"},
        _ang_coding),
    solar_zenith_angle = (
        "solar_zenith_angle",
        ("scanline_earth", "scanpos"),
        {"long_name": "Zenith angle of the Sun as seen from the ground",
         "units": "degrees"},
        _ang_coding),
    solar_azimuth_angle = (
        "solar_azimuth_angle",
        ("scanline_earth", "scanpos"),
        {"long_name": "Azimuth angle (from the north) of the Sun as seen from the ground",
         "units": "degrees"},
        _ang_coding),
    R_e_alt_meq_full = (
        "R_e_alt_meq_full",
        ("calibrated_channel", "scanline_earth", "scanpos"),
        {"long_name": ("Earth radiance calculated directly with the full "
                    "symbolic measurement equation and band "
                    "coefficients, as opposed to in-code and SRF "
                    "integration"),
         "units": rad_u["si"]},
        _coding),
    R_e_alt_meq_simple = (
        "R_e_alt_meq_simple",
        ("calibrated_channel", "scanline_earth", "scanpos"),
        {"long_name": ("Earth radiance calculated directly with the "
                    "simplified symbolic measurement equation and band "
                    "coefficients, as opposed to in-code and SRF "
                    "integration"),
         "units": rad_u["si"]},
        _coding)
)
for allskips in itertools.product(
        (0, 1), repeat=4):
    if not any(allskips):
        continue
    lab = (("linear" if allskips[0] else "") +
           ("norself" if allskips[1] else "") +
           ("nooffset" if allskips[2] else "") +
           ("noεcorr" if allskips[3] else ""))
    FCDR_data_vars_props[f"rad_wn_{lab:s}"]= (
        f"rad_wn_{lab:s}",
        ("scanline_earth", "scanpos", "calibrated_channel"),
        {"units": rad_u["si"]},
        _coding)

p = FCDR_data_vars_props
for (var, corr) in {("R_Earth", "O_Re"),
                    ("T_IWCT_calib_mean", "O_TIWCT"),
                    ("R_IWCT", "O_RIWCT"),
                    ("prt_iwct_temperature", "O_TPRT")}:
    FCDR_data_vars_props[corr] = (
        "corr_" + p[var][0],
        (),
        {"long_name": "correction to " + p[var][2]["long_name"],
         "units": p[var][2]["units"]},
        _coding) # actually, always zero…
# FIXME: needs more uncertainties, or do I calculate this directly from
# the effects and is this one unused and should it be removed?
# In principle, no uncertainty needs more than one byte as we just express
# uncertainty on the last two significant digits...
# For example, 12.345(12)
        
#: Dictionary with encodings for uncertainties, where those diverge from quantities
FCDR_uncertainty_encodings = {}

FCDR_uncertainty_encodings["O_TPRT"] = _temp_coding
for (k, v) in FCDR_data_vars_props.items():
    if v[2].get("units") == "count":
        FCDR_uncertainty_encodings[k] = _u_count_coding

#: Encodings for easy FCDR.  Should no longer be used, as this is in ``FCDRTools``.
FCDR_easy_encodings = dict(
    latitude = _latlon_coding,
    longitude = _latlon_coding,
    bt = _temp_coding,
    satellite_zenith_angle = _ang_coding,
    satellite_azimuth_angle = _ang_coding,
    solar_zenith_angle = _ang_coding,
    solar_azimuth_angle = _ang_coding,
    scanline = _u2_coding,
    time = _cal_coding, # identical to y? See #94
#    qualind = _u4_coding,
#    linqualflags = _u4_coding,
#    chqualflags = _u2_coding,
#    mnfrqualflags = _u1_coding,
    quality_scanline_bitmask = _u1_coding,
    quality_channel_bitmask = _u1_coding,
    u_independent = _temp_coding.copy(),
    u_structured = _temp_coding.copy(),
    # and the dimensions/coordinates
    channel = _u1_coding,
    #rad_channel = _u1_coding,
    x = _u1_coding,
    y = _u2_coding,
    channel_correlation_matrix = _corr_coding,
    lookup_table_BT = _temp_coding,
    lookup_table_radiance = _coding,
    cross_line_radiance_error_correlation_length_scale_structured_effects = _u2_coding,
    cross_element_radiance_error_correlation_length_scale_structured_effects = _u2_coding,
    cross_channel_error_correlation_matrix_independent_effects = _corr_coding,
    cross_channel_error_correlation_matrix_structured_effects = _corr_coding,
)

# this should ensure #70 as long as u>0.01K
for v in ("u_independent", "u_structured"):
    FCDR_easy_encodings[v]["dtype"] = "u4"
    FCDR_easy_encodings[v]["scale_factor"] = 0.001

#: attributes not defined elsewhere for whatever reason, such as only being coordinates or only occurring temporarily or in easy or being otherwise calculated later
FCDR_extra_attrs = dict(
    x = {"long_name": "scan position",
         "units": "dimensionless",
         "valid_range": [1, 56]},
    y = {"long_name": "scanline number",
         "units": "dimensionless"},
    channel =
        {"long_name": "channel number",
         "units": "dimensionless",
         "valid_range": [1, 20],
         "note": "channel 20 not calibrated by FIDUCEO"},
    u_independent =
        {"long_name": "uncertainty from independent errors"},
    u_structured =
        {"long_name": "uncertainty from structured errors",
         "note": ("contains uncertainties from fully systematic, "
                  "and structured random effects.  For a more complete "
                  "treatment, please use full FCDR.")}
    )


# TB writer 2018-08-21 (1.1.5):
#
# In : easy["quality_pixel_bitmask"].flag_meanings
# 'invalid use_with_caution invalid_input invalid_geoloc invalid_time
# sensor_error padded_data incomplete_channel_data'
#
# In : easy["data_quality_bitmask"].flag_meanings
# 'suspect_mirror suspect_geo suspect_time outlier_nos
# uncertainty_too_large'
#
# In : easy["quality_scanline_bitmask"].flag_meanings
# 'do_not_use_scan time_sequence_error data_gap_preceding_scan
# no_calibration no_earth_location clock_update status_changed
# line_incomplete, time_field_bad time_field_bad_not_inf
# inconsistent_sequence scan_time_repeat uncalib_bad_time calib_few_scans
# uncalib_bad_prt calib_marginal_prt uncalib_channels uncalib_inst_mode
# quest_ant_black_body zero_loc bad_loc_time bad_loc_marginal bad_loc_reason
# bad_loc_ant reduced_context bad_temp_no_rself'
#
# In : easy["quality_channel_bitmask"].flag_meanings
# 'do_not_use uncertainty_suspicious self_emission_fails
# calibration_impossible calibration_suspect'

@enum.unique
class FlagsScanline(enum.IntFlag):
    """Enumerator for scanline-based flags.
    """
    DO_NOT_USE = enum.auto()
    SUSPECT_GEO = enum.auto()
    SUSPECT_TIME = enum.auto()
    SUSPECT_CALIB = enum.auto()
    SUSPECT_MIRROR_ANY = enum.auto()
    REDUCED_CONTEXT = enum.auto()
    UNCERTAINTY_SUSPICIOUS = enum.auto()
    BAD_TEMP_NO_RSELF = enum.auto()

@enum.unique
class FlagsChannel(enum.IntFlag):
    """Enumerator for channel-based flags.
    """
    DO_NOT_USE = enum.auto()
    UNCERTAINTY_SUSPICIOUS = enum.auto()
    SELF_EMISSION_FAILS = enum.auto()
    CALIBRATION_IMPOSSIBLE = enum.auto()

@enum.unique
class FlagsMinorFrame(enum.IntFlag):
    """Enumerator for minorframe-based flags.
    """
    SUSPECT_MIRROR = enum.auto()

@enum.unique
class FlagsPixel(enum.IntFlag):
    """Enumerator for pixel-based flags.
    """
    DO_NOT_USE = enum.auto()
    OUTLIER_NOS = enum.auto()
    UNCERTAINTY_TOO_LARGE = enum.auto()

#FCDR_data_vars_props["quality_scanline_bitmask"][2]["flag_masks"] = ", ".join(str(int(v)) for v in FlagsScanline.__members__.values())
FCDR_data_vars_props["quality_scanline_bitmask"][2]["flag_masks"] = [
    int(v) for v in FlagsScanline.__members__.values()]
FCDR_data_vars_props["quality_scanline_bitmask"][2]["flag_meanings"] = ", ".join(FlagsScanline.__members__.keys()) 

#FCDR_data_vars_props["quality_channel_bitmask"][2]["flag_masks"] = ", ".join(str(int(v)) for v in FlagsChannel.__members__.values())
FCDR_data_vars_props["quality_channel_bitmask"][2]["flag_masks"] = [
    int(v) for v in FlagsChannel.__members__.values()]
FCDR_data_vars_props["quality_channel_bitmask"][2]["flag_meanings"] = ", ".join(FlagsChannel.__members__.keys())

#FCDR_data_vars_props["quality_minorframe_bitmask"][2]["flag_masks"] = ", ".join(str(int(v)) for v in FlagsMinorFrame.__members__.values())
FCDR_data_vars_props["quality_minorframe_bitmask"][2]["flag_masks"] = [
    int(v) for v in FlagsMinorFrame.__members__.values()]
FCDR_data_vars_props["quality_minorframe_bitmask"][2]["flag_meanings"] = ", ".join(FlagsMinorFrame.__members__.keys())

FCDR_data_vars_props["quality_pixel_bitmask"][2]["flag_masks"] = [
    int(v) for v in FlagsPixel.__members__.values()]
FCDR_data_vars_props["quality_pixel_bitmask"][2]["flag_meanings"] = ", ".join(FlagsPixel.__members__.keys())



