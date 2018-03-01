"""Any code related to processing or analysing matchups
"""

import abc
import warnings

import numpy

import typhon.datasets.tovs
import typhon.datasets.filters
import itertools

from . import fcdr

from typhon.physics.units.common import ureg, radiance_units as rad_u
from typhon.physics.units.tools import UnitsAwareDataArray as UADA
from typhon.physics.units.em import SRF

class HHMatchupCountFilter(typhon.datasets.filters.OrbitFilter):
    def __init__(self, prim, sec):
        self.prim = prim
        self.sec = sec

    def filter(self, ds, **extra):
        return ds[{"matchup_count":
            (ds[f"hirs-{self.prim:s}_lza"][:, 3, 3] < 10) &
            (ds[f"hirs-{self.sec:s}_lza"][:, 3, 3] < 10) &
            (ds["matchup_spherical_distance"]<20)}]
    
    def finalise(self, ds):
        idx = numpy.argsort(ds[f"time_{self.prim:s}"])
        return ds[{"matchup_count": idx}]

class HIMatchupCountFilter(typhon.datasets.filters.OrbitFilter):
    def filter(self, ds, **extra):
        bad = (ds["ref_radiance"]<=0).any("ch_ref")
        return ds[{"line": ~bad}]
    
    def finalise(self, ds):
        # Another round of sorting, not sure why needed
        idx = numpy.argsort(ds["ref_time"])
        return ds[{"line": idx}]

class CalibrationCountDimensionReducer(typhon.datasets.filters.OrbitFilter):
    """Reduce the size of calibration counts.

    When we want to add calibraiton counts to matchup, we want to add only
    one (median) to each matchup, not the whole batch of 48, that becomes
    too large...
    """

    def finalise(self, arr):
        hcd = [k for (k, v) in arr.data_vars.items() if "calibration_position" in v.dims]
        for k in hcd:
            arr[k] = arr[k].median("calibration_position")
        return arr

# inspect_hirs_matchups, work again
hh = typhon.datasets.tovs.HIRSHIRS(read_returns="xarray")
hi = typhon.datasets.tovs.HIASI(read_returns="xarray") # metopa only

class HIRSMatchupCombiner:
    fcdr_info = {"data_version": "0.8pre", "fcdr_type": "debug"}
    fields_from_each = [
         'B',
         'C_E',
         'C_IWCT',
         'C_s',
         'LUT_BT',
         'LUT_radiance',
         'N',
         'R_IWCT',
         'R_e',
         'R_refl',
         'R_selfE',
         'R_selfIWCT',
         'R_selfs',
         'T_IWCT',
         'T_b',
         'Tstar',
         'a_0',
         'a_1',
         'a_2',
         'a_3',
         'a_4',
         'c',
         'channel_correlation_matrix',
         #'counts',
         #'earth_location',
         #'elements',
         'fstar',
         'h',
         'k_b',
         'latitude',
         'longitude',
         #'original_calibration_coefficients',
         #'original_calibration_coefficients_sorted',
         'platform_altitude',
         'platform_zenith_angle',
         'quality_channel_bitmask',
         'quality_flags_bitfield',
         'quality_minorframe_bitmask',
         'quality_pixel_bitmask',
         'quality_scanline_bitmask',
         'scanline_number',
         'scantype',
         'scnlintime',
         'temperature_baseplate',
         'temperature_cooler_housing',
         'temperature_electronics',
         'temperature_filter_wheel_housing',
         'temperature_filter_wheel_motor',
         'temperature_first_stage_radiator',
         'temperature_internal_warm_calibration_target',
         'temperature_patch_expanded',
         'temperature_patch_full',
         'temperature_primary_telescope',
         'temperature_scanmirror',
         'temperature_scanmotor',
         'temperature_secondary_telescope',
         #'toa_brightness_temperature',
         'toa_outgoing_radiance_per_unit_frequency',
         'u_C_Earth',
         'u_C_IWCT',
         'u_C_PRT',
         'u_C_space',
         'u_Earthshine',
         'u_O_Re',
         'u_O_TIWCT',
         'u_O_TPRT',
         'u_R_Earth',
         'u_R_Earth_nonrandom',
         'u_R_Earth_random',
         'u_Rself',
         'u_Rselfparams',
         'u_SRF_calib',
         'u_T_b_nonrandom',
         'u_T_b_random',
         'u_d_PRT',
         'u_electronics',
         'u_extraneous_periodic',
         'u_f_eff',
#         'u_from_B',
#         'u_from_C_E',
#         'u_from_C_IWCT',
#         'u_from_C_s',
#         'u_from_O_TIWCT',
#         'u_from_R_IWCT',
#         'u_from_R_refl',
#         'u_from_R_selfE',
#         'u_from_T_IWCT',
#         'u_from_Tstar',
#         'u_from_a_0',
#         'u_from_a_1',
#         'u_from_a_2',
#         'u_from_fstar',
#         'u_from_α',
#         'u_from_β',
         'u_nonlinearity',
         'u_α',
         'u_β',
         'α',
         'β',
         'ε']

    mode = None
    def __init__(self, start_date, end_date, prim_name, sec_name):
        #self.ds = netCDF4.Dataset(str(sf), "r")
        # acquire original brightness temperatures here for the purposes
        # of estimating Kr.  Of course this should come from my own
        # brightness temperatures, but the context of those is not
        # readibly available in the matchups from BC, so it would take
        # more effort to gather the necessary context information.  See
        # #117.
        if prim_name.lower() == "iasi":
            if sec_name.lower() not in ("metopa", "ma"):
                raise ValueError(f"When primary is IASI, secondary "
                    f"must be metopa, not {sec_name!s}")
            self.mode = "reference"
            ds = hi.read_period(start_date, end_date,
                orbit_filters=hi.default_orbit_filters+[HIMatchupCountFilter()])
            self.prim_hirs = "iasi"
            self.hiasi = hi
        else:
            self.mode = "hirs"
            ds = hh.read_period(start_date, end_date,
                locator_args={"prim": prim_name, "sec": sec_name},
                fields={"hirs-{:s}_{:s}".format(s, field)
                    for field in ("x", "y", "time", "lza", "file_name",
                                  "acquisition_time", "scanpos") + tuple(
                                    "bt_ch{:02d}".format(ch) for ch in
                                    range(1, 20))
                    for s in (prim_name, sec_name)}|{"matchup_spherical_distance"},
                pseudo_fields={
                    "time_{:s}".format(prim_name):
                        lambda ds: ds["hirs-{:s}_time".format(prim_name)][:, 3, 3].astype("M8[s]"),
                    "time_{:s}".format(sec_name):
                        lambda ds: ds["hirs-{:s}_time".format(sec_name)][:, 3, 3].astype("M8[s]")},
                orbit_filters=hh.default_orbit_filters+[HHMatchupCountFilter(prim_name,sec_name)])
            self.prim_hirs = fcdr.which_hirs_fcdr(prim_name, read="L1C")

        self.sec_hirs = fcdr.which_hirs_fcdr(sec_name, read="L1C")

        if self.mode == "reference":
            # There is no Mcp, for the primary (reference) is IASI
            Mcp = None
            Mcs = hi.combine(ds,
                self.sec_hirs,
                trans={"mon_time": "time"},
                timetol=numpy.timedelta64(4, 's'),
                other_args={"locator_args": self.fcdr_info,
                            "fields": self.fields_from_each}).drop(
                    ("lat_earth", "lon_earth"))
        elif self.mode == "hirs":
            Mcp = hh.combine(ds, self.prim_hirs, trans={"time_{:s}".format(prim_name): "time"},
                             timetol=numpy.timedelta64(4, 's'),
                             col_field="hirs-{:s}_x".format(prim_name),
                             col_dim_name="scanpos",
                             other_args={"locator_args": self.fcdr_info,
                                         "fields": self.fields_from_each,
                                         "orbit_filters": [CalibrationCountDimensionReducer()],
                                         "NO_CACHE": True},
                             time_name="time_"+prim_name).drop(
                    ("lat_earth", "lon_earth"))
            Mcs = hh.combine(ds, self.sec_hirs, trans={"time_{:s}".format(sec_name): "time"},
                             timetol=numpy.timedelta64(4, 's'),
                             col_field="hirs-{:s}_x".format(sec_name),
                             col_dim_name="scanpos",
                             other_args={"locator_args": self.fcdr_info,
                                         "orbit_filters": [CalibrationCountDimensionReducer()],
                                         "fields": self.fields_from_each},
                             time_name="time_"+sec_name).drop(
                    ("lat_earth", "lon_earth"))
        else:
            raise RuntimeError(f"Mode can't possibly be {self.mode:s}!")

        self.start_date = start_date
        self.end_date = end_date
        self.ds = ds
        self.Mcp = Mcp
        self.Mcs = Mcs
        self.prim_name = prim_name
        self.sec_name = sec_name


class KModel(metaclass=abc.ABCMeta):
    """Model to estimate K and Ks (Kr is seperate)

    There is currently an ad-hoc implementation, estimating K based on BB
    assumption only on L→BT conversions.  There will later be an
    implementation based on BTs simulated with a forward model.  In
    practice, those are calculated ``offline'' and looked up.

    For definitions, see document
    20171205-FIDUCEO-SH-Harmonisation_Input_File_Format_Definition-v6.pdf
    from Sam Hunt available on the FIDUCEO wiki:

    > Uncertainties are also required for the estimates of the match-up
    > adjustment factor, K. It currently is assumed that all errors in K are
    > independent, though are separated into two categories. Uncertainties
    > due to the SRF differences should be also be stored in a vector K r
    > and the uncertainties due to the match-up process should be stored
    > in a vector K s .
    
    (Note that Kr and Ks are swapped in this guide.)

    """

    def __init__(self, ds, ds_orig, prim_name, prim_hirs, sec_name, sec_hirs):
        self.ds = ds
        self.ds_orig = ds_orig
        self.prim_name = prim_name
        self.prim_hirs = prim_hirs
        self.sec_name = sec_name
        self.sec_hirs = sec_hirs

    @abc.abstractmethod
    def calc_K(self, channel):
        """K is the expected value of the matchup difference:

        K = E(L_2 - L_1)
        """
        ...

    @abc.abstractmethod
    def calc_Ks(self, channel):
        ...

    filtered = False
    def limit(self, ok, mdim):
        """Reduce dataset to those values
        """
        self.ds = self.ds[{mdim:ok}]
        self.ds_orig = self.ds_orig[{mdim:ok}]
        self.filtered = True

class KrModel(metaclass=abc.ABCMeta):
    """Implementations of models to estimate the matchup uncertainty
    """

    def __init__(self, ds, ds_orig, prim_name, prim_hirs, sec_name, sec_hirs):
        self.ds = ds
        self.ds_orig = ds_orig
        self.prim_name = prim_name
        self.prim_hirs = prim_hirs
        self.sec_name = sec_name
        self.sec_hirs = sec_hirs

    @abc.abstractmethod
    def calc_Kr(self, channel):
        ...

    filtered = False
    def limit(self, ok, mdim):
        """Reduce dataset to those values
        """
        self.ds = self.ds[{mdim:ok}]
        self.ds_orig = self.ds_orig[{mdim:ok}]
        self.filtered = True

class KModelPlanck(KModel):
    """Simplified implementation.

    This is a temporary implementation for estimating values of K
    (expected difference in L).
    """

    def calc_K(self, channel):
        warnings.warn(
            f"Using ad-hoc/Planck approximation for K, not accurate.",
            UserWarning)
        L1 = self.prim_hirs.srfs[channel-1].channel_radiance2bt(
            ureg.Quantity(
                self.ds[f"{self.prim_name:s}_R_e"].sel(
                    calibrated_channel=channel).values,
                rad_u["si"]))

        # the "slave" BT should be the radiance of the REFERENCE using the
        # SRF of the SECONDARY, so that all other effects are zero here.
        # But that's not quite accurate, because the calibration
        # coefficients are still computed using the primary... need to
        # redo the entire measurement equation?  See #181 or rather #183.
        # Need to use forward modelling to do this properly.
        L2 = self.sec_hirs.srfs[channel-1].channel_radiance2bt(
            ureg.Quantity(
                self.ds[f"{self.prim_name:s}_R_e"].sel(
                    calibrated_channel=channel).values,
                rad_u["si"]))

        return (
            L2.to(rad_u["si"], "radiance", srf=self.sec_hirs.srfs[channel-1])
          - L1.to(rad_u["si"], "radiance", srf=self.sec_hirs.srfs[channel-1]))

        return L2 - L1

    def calc_Ks(self, channel):
        # propagate from band correction
        Δ = self.sec_hirs.srfs[channel-1].estimate_band_coefficients(
            self.sec_hirs.satname, "fcdr_hirs", channel)[-1]
        Δ = ureg.Quantity(Δ.values, Δ.units)
        slave_bt = self.sec_hirs.srfs[channel-1].channel_radiance2bt(
            ureg.Quantity(
                self.ds[f"{self.prim_name:s}_R_e"].sel(
                    calibrated_channel=channel).values,
                rad_u["si"]))
        slave_bt_perturbed = self.sec_hirs.srfs[channel-1].shift(
            Δ).channel_radiance2bt(ureg.Quantity(
                self.ds["{:s}_R_e".format(self.prim_name)].sel(
                    calibrated_channel=channel).values,
                rad_u["si"]))
        slave_bt_perturbed_2 = self.sec_hirs.srfs[channel-1].shift(
            Δ).channel_radiance2bt(ureg.Quantity(
                self.ds["{:s}_R_e".format(self.prim_name)].sel(
                    calibrated_channel=channel).values,
                rad_u["si"]))
        Δslave_bt = (abs(slave_bt_perturbed - slave_bt)
                   + abs(slave_bt_perturbed_2 - slave_bt))/2

        srf = self.sec_hirs.srfs[channel-1]
        Δslave_rad = (abs(slave_bt_perturbed.to(
                            rad_u["si"], "radiance", srf=srf)
                        - slave_bt.to(
                            rad_u["si"], "radiance", srf=srf))
                    + abs(slave_bt_perturbed_2.to(
                            rad_u["si"], "radiance", srf=srf)
                       - slave_bt.to(
                            rad_u["si"], "radiance", srf=srf)))/2

        return Δslave_rad

class KModelIASIRef(KModel):
    """Estimate K and Ks in case of IASI reference
    """
    def calc_K(self, channel):
        return numpy.zeros(shape=self.ds.dims["line"])

    def calc_Ks(self, channel):
        return numpy.zeros(shape=self.ds.dims["line"])

class KrModelLSD(KrModel):

    def calc_Kr(self, channel):
        btlocal = self.ds_orig["hirs-{:s}_bt_ch{:02d}".format(self.prim_name, channel)]
        btlocal.values.reshape((-1,))[btlocal.values.ravel()>400] = numpy.nan # not all are flagged correctly
        btlocal = btlocal.loc[{"hirs-{:s}_ny".format(self.prim_name): slice(1, 6),
                           "hirs-{:s}_nx".format(self.prim_name): slice(1, 6)}].stack(
                    z=("hirs-{:s}_ny".format(self.prim_name),
                       "hirs-{:s}_nx".format(self.prim_name)))
        btlocal = UADA(btlocal)
        srf = SRF.fromArtsXML(
            typhon.datasets.tovs.norm_tovs_name(self.prim_name).upper(),
            "hirs", channel)
        radlocal = btlocal.to(rad_u["si"], "radiance", srf=srf)
        lsd = radlocal.std("z")
        #lsd.attrs["units"] = "K"
        # Convert from K to radiance
        #lsd = UADA(lsd).to(rad_u["si"], "radiance", srf=srf)
        return lsd

class KrModelIASIRef(KrModel):
    def calc_Kr(self, channel):
        srf = SRF.fromArtsXML(
            typhon.datasets.tovs.norm_tovs_name(self.sec_name).upper(),
            "hirs", channel)
        return abs(
            ureg.Quantity(
                self.ds["metopa_T_b"].sel(calibrated_channel=channel).values, "K"
                    ).to(rad_u["si"], "radiance", srf=srf) -
            ureg.Quantity(
                self.ds["metopa_T_b"].sel(calibrated_channel=channel).values+0.1, "K"
                    ).to(rad_u["si"], "radiance", srf=srf))
