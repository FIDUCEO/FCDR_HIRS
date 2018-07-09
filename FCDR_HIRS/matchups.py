"""Any code related to processing or analysing matchups
"""

import abc
import warnings
import datetime
import itertools
import functools
import operator
import unicodedata
import re

import numpy
import scipy
import xarray

import typhon.datasets.tovs
import typhon.datasets.filters

import sklearn.linear_model

from . import fcdr

from typhon.physics.units.common import ureg, radiance_units as rad_u
from typhon.physics.units.tools import UnitsAwareDataArray as UADA
from typhon.physics.units.em import SRF

class ODRFitError:
    pass

class NoDataError:
    pass

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
         'u_a_2',
         'u_a_3',
         'u_a_4',
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
                    for field in ("x", "y", "time", "lza", "file_name", "lat", "lon",
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
            try:
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
            except ValueError as e:
                if e.args[0] == "array of sample points is empty":
                    raise NoDataError("Not enough matching data found, can't interpolate")
                else:
                    raise
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

    ds = None
    ds_filt = None
    ds_orig = None
    ds_filt_orig = None
    prim_name = None
    prim_hirs = None
    sec_name = None
    sec_hirs = None

    def __init__(self, **kwargs):
        for (k, v) in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise AttributeError(f"Unknown keyword argument/attribute: {k:s}")

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
        self.ds_filt = self.ds[{mdim:ok}]
        self.ds_filt_orig = self.ds_orig[{mdim:ok}]
        self.filtered = True

    def filter(self, mdim):
        """Extra filtering imposed by this model
        """
        return numpy.ones(self.ds.dims[mdim], "?")

    def extra(self):
        """Return Dataset with extra information
        """

        return xarray.Dataset()

class KrModel(metaclass=abc.ABCMeta):
    """Implementations of models to estimate the matchup uncertainty
    """

    def __init__(self, ds, ds_orig, prim_name, prim_hirs, sec_name, sec_hirs):
        self.ds = ds
        self.ds_filt = None
        self.ds_orig = ds_orig
        self.ds_filt_orig = None
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
        self.ds_filt = self.ds[{mdim:ok}]
        self.ds_filt_orig = self.ds_orig[{mdim:ok}]
        self.filtered = True

    def filter(self, mdim):
        """Extra filtering imposed by this model
        """
        return numpy.ones(self.ds.dims[mdim], "?")

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
                self.ds_filt[f"{self.prim_name:s}_R_e"].sel(
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
                self.ds_filt[f"{self.prim_name:s}_R_e"].sel(
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
                self.ds_filt[f"{self.prim_name:s}_R_e"].sel(
                    calibrated_channel=channel).values,
                rad_u["si"]))
        slave_bt_perturbed = self.sec_hirs.srfs[channel-1].shift(
            Δ).channel_radiance2bt(ureg.Quantity(
                self.ds_filt["{:s}_R_e".format(self.prim_name)].sel(
                    calibrated_channel=channel).values,
                rad_u["si"]))
        slave_bt_perturbed_2 = self.sec_hirs.srfs[channel-1].shift(
            Δ).channel_radiance2bt(ureg.Quantity(
                self.ds_filt["{:s}_R_e".format(self.prim_name)].sel(
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
        return numpy.zeros(shape=self.ds_filt.dims["line"])

    def calc_Ks(self, channel):
        return numpy.zeros(shape=self.ds_filt.dims["line"])

class KrModelLSD(KrModel):

    def calc_Kr(self, channel):
        btlocal = self.ds_filt_orig["hirs-{:s}_bt_ch{:02d}".format(self.prim_name, channel)]
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

    def filter(self, mdim):
        ok = super().filter(mdim)
        return functools.reduce(
            operator.and_,
            ((self.ds_orig[f"hirs-{s:s}_bt_ch{c:02d}"].notnull()
                    .sum(f"hirs-{s:s}_nx").sum(f"hirs-{s:s}_ny")>25).values
                for c in range(1, 20)
                for s in (self.prim_name, self.sec_name)),
            ok)

class KrModelIASIRef(KrModel):
    def calc_Kr(self, channel):
        srf = SRF.fromArtsXML(
            typhon.datasets.tovs.norm_tovs_name(self.sec_name).upper(),
            "hirs", channel)
        return abs(
            ureg.Quantity(
                self.ds_filt["metopa_T_b"].sel(calibrated_channel=channel).values, "K"
                    ).to(rad_u["si"], "radiance", srf=srf) -
            ureg.Quantity(
                self.ds_filt["metopa_T_b"].sel(calibrated_channel=channel).values+0.1, "K"
                    ).to(rad_u["si"], "radiance", srf=srf))



class KModelSRFIASIDB(KModel):
    """Estimate K using IASI spectral database.

    Based on the SRF recovery method described in PD4.4
    """

    iasi_start = datetime.datetime(2011, 1, 1)
    iasi_end = datetime.datetime(2011, 2, 1)
    iasi = typhon.datasets.tovs.IASISub(name="iasisub")
    M_iasi = None
    Ldb_iasi_full = None
    Ldb_hirs_simul = None
    srfs = None
    fitter = None

    regression = "LR"
    chan_pairs = None
    mode = "delta"
    units = rad_u["si"]
    debug = False
    others = None
    #regression = "ODR"
    # FIXME: Use ODR with uncertainties
#    regression_model = sklearn.linear_model.LinearRegression
#    regression_args = {"fit_intercept": True}



    def __init__(self, chan_pairs="all", *args, **kwargs):
        # FIXME: add an "optimal", channel-specific mapping
        if chan_pairs == "all":
            chan_pairs = dict.fromkeys(numpy.arange(1, 20), numpy.arange(1, 20))
        elif chan_pairs == "single":
            chan_pairs = {c:[c] for c in range(1, 20)}
        elif chan_pairs == "neighbours":
            chan_pairs = {ch: numpy.arange(max(ch-1, 1), min(ch+2, 20)) for ch in range(1, 20)}
        else:
            raise ValueError(f"Unknown value for chan_pairs: {chan_pairs!s}")
        self.chan_pairs = chan_pairs
        super().__init__(*args, **kwargs)
        if self.debug:
            self.others = {}
            for (cp, md, u) in itertools.product(
                    ["all", "single", "neighbours"],
                    ["delta", "standard"],
                    [rad_u["si"], rad_u["ir"], ureg.Unit("K")]):
                unitlab = re.sub(r'/([A-Za-z]*)', r' \1^-1',
                "{:~P}".format(u)).replace("-1²", "-2")
                lab = unicodedata.normalize("NFKC", f"{cp:s}_{md:s}_{unitlab:s}")
                self.others[lab] = self.__class__(
                        chan_pairs=cp, 
                        mode=md,
                        units=u,
                        debug=False,
                        ds=self.ds,
                        ds_orig=self.ds_orig,
                        prim_name=self.prim_name,
                        prim_hirs=self.prim_hirs,
                        sec_name=self.sec_name,
                        sec_hirs=self.sec_hirs)

    def limit(self, ok, mdim):
        super().limit(ok, mdim)
        if self.debug:
            for v in self.others.values():
                v.ds_filt = self.ds_filt
                v.ds_filt_orig = self.ds_filt_orig

    def read_iasi(self):
        """Read spectral database from IASI.

        Details on what is read are stored in self attributes iasi_start,
        iasi_end, passed on on object creation.
        """

        # this is a structured array with fields
        # ('time', 'lat', 'lon', 'satellite_zenith_angle',
        # 'satellite_azimuth_angle', 'solar_zenith_angle',
        # 'solar_azimuth_angle', 'spectral_radiance')
        self.M_iasi = self.iasi.read_period(self.iasi_start, self.iasi_end)
        # with 'spectral_radiance' in units of [W m^-2 sr^-1 m]
        if self.debug:
            for v in self.others.values():
                v.M_iasi = self.M_iasi

    def init_iasi(self):
        """Initialise IASI data in right format
        """
        if self.M_iasi is None:
            self.read_iasi()
        L_wavenum = ureg.Quantity(self.M_iasi["spectral_radiance"][2::5, 2, :8461], rad_u["ir"])
        L_freq = L_wavenum.to(rad_u["si"], "radiance")
        self.Ldb_iasi_full = L_freq
        if self.debug:
            for v in self.others.values():
                v.init_iasi() # various units

    def init_srfs(self):
        """Initialise SRFs, reading from ArtsXML format
        """
        self.srfs = {}
        for sat in (self.prim_name, self.sec_name):
            self.srfs[sat] = [
                typhon.physics.units.em.SRF.fromArtsXML(
                    typhon.datasets.tovs.norm_tovs_name(sat).upper(),
                    "hirs", ch)
                for ch in range(1, 20)]
        if self.debug:
            for v in self.others.values():
                v.srfs = self.srfs

    def init_Ldb(self):
        """Calculate IASI-simulated HIRS

        For all channels for either pair.
        """

        if self.Ldb_iasi_full is None:
            self.init_iasi()
        if self.srfs is None:
            self.init_srfs()
        ndb = self.Ldb_iasi_full.shape[0]
        Ldb = xarray.Dataset(
            {self.prim_name: (["chan", "pixel"], numpy.zeros((19, ndb))),
             self.sec_name: (["chan", "pixel"], numpy.zeros((19, ndb)))},
            coords={"chan": numpy.arange(1, 20)})
        for sat in (self.prim_name, self.sec_name):
            for ch in range(1,20):
                Ldb[sat].loc[{"chan": ch}] = self.srfs[sat][ch-1].integrate_radiances(
                        self.iasi.frequency,
                        self.Ldb_iasi_full).to(
                            self.units, "radiance",
                            srf=self.srfs[sat][ch-1])
        self.Ldb_hirs_simul = Ldb
        if self.debug:
            for v in self.others.values():
                v.init_Ldb()

    def init_regression(self):
        if self.Ldb_hirs_simul is None:
            self.init_Ldb()
        # make one fitter for each target channel, in both directions
        fitter = {}
        for (from_sat, to_sat) in [(self.prim_name, self.sec_name),
                                   (self.sec_name, self.prim_name)]:
            fitter[f"{from_sat:s}-{to_sat:s}"] = {}
            for chan in range(1, 20):
                y_ref_ch = self.chan_pairs[chan]
                y_ref = self.Ldb_hirs_simul[from_sat].sel(chan=y_ref_ch).values
                y_target = self.Ldb_hirs_simul[to_sat].sel(chan=chan).values
                if self.mode == "delta":
                    y_target -= self.Ldb_hirs_simul[from_sat].sel(chan=chan)
                # for training with sklearn, dimensions should be n_p × n_c
                if self.regression == "LR":
                    clf = sklearn.linear_model.LinearRegression(fit_intercept=True)
                    clf.fit(y_ref.T, y_target)
                elif self.regression == "ODR":
                    raise NotImplementedError("ODR not implemented yet. "
                            "I'm not sure how useful ODR would be.  "
                            "Training data is from IASI.  "
                            "I don't have real IASI uncertainties.  "
                            "I could add noise and then use those  "
                            "uncertainties.  "
                            "Would not be weighted unless I make it so, "
                            "but according to what rules?")
                    odr_data = scipy.odr.RealData(y_ref.T, y_target, sx=sx, sy=sy)
                    clf = scipy.odr.ODR(mydata, scipy.odr.multilinear, beta0=β0) 
                    clf = myodr.run()
                    if not any(x in clf.stopreason for x in
                        {"Sum of squares convergence",
                         "Iteration limit reached",
                         "Parameter convergence",
                         "Both sum of squares and parameter convergence"}):
                        raise ODRFitError("ODR fitting did not converge.  "
                            "Stop reason: {:s}".format(clf.stopreason[0]))

                else:
                    raise ValueError(f"Unknown regression: {self.regression:s}")
                fitter[f"{from_sat:s}-{to_sat:s}"][chan] = clf
        self.fitter = fitter
        if self.debug:
            for v in self.others.values():
                v.init_regression()

    K = dict.fromkeys(range(1, 20), None)
    def calc_K(self, channel):
        if self.fitter is None:
            self.init_regression()
        # Use regression to predict Δy for channel from set of reference
        # channels.
        K = []
        for (from_sat, to_sat) in [(self.prim_name, self.sec_name),
                                   (self.sec_name, self.prim_name)]:
            clf = self.fitter[f"{from_sat:s}-{to_sat:s}"][channel]
            y_source = self.ds_filt[f"{from_sat:s}_R_e"].sel(calibrated_channel=self.chan_pairs[channel]).values.T
            y_ref = self.ds_filt[f"{to_sat:s}_R_e"].sel(calibrated_channel=channel).values
            model = self.fitter[f"{from_sat:s}-{to_sat:s}"][channel]
            Δy_pred = model.predict(y_source)
            #K.append(y_pred - y_source[:, channel-1]) # value for K?
            K.append(Δy_pred)
            # derive Ks from spread of predictions?  Or Δ with y_ref?  Or
            # from uncertainty provided by regression?  Isn't that part of
            # Kr instead?
        self.K[channel] = K
        if self.debug:
            for v in self.others.values():
                v.calc_K(channel)
        return numpy.array(K).mean(0)

    def calc_Ks(self, channel):
        # spread of db will inform us of uncertainty in predicted
        # radiance?  Or take double direction approach?
        if self.K[channel] is None:
            self.calc_K(channel)
        # FIXME: This is not a good estimate, this Δ tends to be a
        # Gaussian which is not centred at zero, it can have either sign.
        # I don't know what this Δ does indicate, but a good estimate for
        # Ks it is not.
        # NB: self.K is Dict[List[ndarray, ndarray]]
        return abs(self.K[channel][0] - self.K[channel][1])

    def extra(self, channel):
        ds = super().extra()
        ds["K_forward"] = (("M",), self.K[channel][0])
        ds["K_backward"] = (("M",), self.K[channel][1])
        if self.debug:
            for (k, v) in self.others.items():
                ds[f"K_other_{k:s}_forward"] = (("M",), v.K[channel][0])
                ds[f"K_other_{k:s}_backward"] = (("M",), v.K[channel][1])
                for direction in ("forward", "backward"):
                    ds[f"K_other_{k:s}_{direction:s}"].attrs.update(
                        units=f"{v.units:~}",
                        direction=direction,
                        mode=v.mode,
                        channels_prediction=v.chan_pairs[channel],
                        regression=v.regression,
                        )
                    
        for (from_sat, to_sat, k) in [
                (self.prim_name, self.sec_name, "K_forward"),
                (self.sec_name, self.prim_name, "K_backward")]:
            ds[k].attrs["units"] = f"{self.units:~}"
            ds[k].attrs["description"] = ("Prediction of "
                    f"delta {to_sat:s} from {from_sat:s} ({self.regression:s})")
            ds[k].attrs["channels_used"] = self.chan_pairs[channel]
        ds.attrs.update(
            mode=self.mode,
            channels_prediction=self.chan_pairs[channel],
            regression=self.regression,
            )
        return ds

    def filter(self, mdim):
        # go through self.ds_filt R_e values.  All channels must be not nan.
        ok = super().filter(mdim)
        for sat in self.prim_name, self.sec_name:
            ok &= self.ds[f"{sat:s}_R_e"].notnull().all("calibrated_channel").values
        return ok
