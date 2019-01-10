"""Module to collect code related to processing or analysing matchups

This module defines base classes related to the processing of matchups, in
particular in preparation for the creation of harmonisation input data.
The `matchups` module, and in particular the `HIRSMatchupCombiner` class,
serve as the basis for the `processing.combine_matchups` module and its
:class:`processing.combine_matchups.HIRSMatchupCombiner` class, but is
also used by scripts in the `analysis` package.

The estimation of K and Ks is done by implementations of the `KModel`
class.  For HIRS-HIRS matchups, the currently applied implementation is
described by the `KModelSRFIASIDB` class.  For HIRS-IASI matchups, the
implementation is `KModelIASIRef`, but note that this estimate is always
zero.  The estimate of Kr is performed by implementations of the `KrModel`
class.  For HIRS-HIRS matchups, the implementation currently used is
`KrModelJointLSD`.  For HIRS-IASI matchups, it's `KrModelIASIRef`.  All
other classes in the `KModel` and `KrModel` hierarchy are either
intermediate classes or classes that have been used before or
experimentally but not currently.

The module also defines a number of filters that go with `KModel` and
`KrModel`, such as `KFilter` and its implementations.  Some of those
filters are used when preparing harmoniation matchups.
"""

from __future__ import annotations

import abc
import warnings
import datetime
import itertools
import functools
import operator
import unicodedata
import re
import pathlib
from dataclasses import dataclass

import numpy
import scipy
import scipy.odr
import xarray

import typhon.datasets.tovs
import typhon.datasets.filters

import sklearn.linear_model

from . import fcdr

from typhon.physics.units.common import ureg, radiance_units as rad_u
from typhon.physics.units.tools import UnitsAwareDataArray as UADA
from typhon.physics.units.em import SRF

unit_iasi = ureg.W / (ureg.m**2 * ureg.sr * (1/ureg.m))

class ODRFitError(Exception):
    """Exception to raise if ODR fitting fails to converge
    """
    pass

class NoDataError(Exception):
    """Exception to raise if not enough matchup data could be found
    """
    pass

class HHMatchupCountFilter(typhon.datasets.filters.OrbitFilter):
    """HIRS-HIRS matchup filter

    Filter on HIRS-HIRS matchups, selecting only matchups with a spherical
    distance of less than 20 km and local zenith angles of less than 10°.

    The methods on this class should not be used directly, rather objects
    of this class should be passed to
    :meth:`typhon.datasets.Dataset.read_period`.
    """
    msd_field = None
    def __init__(self, prim, sec, msd_field="matchup_spherical_distance"):
        """Initialise HIRS-HIRS matchup filter

        Parameters
        ----------

        prim : str
            Name of primary for matchups to which the filter is applied
        sec : str
            Name of secondary for matchups to which the filter is applied
        msd_field : str, optional
            Name of field containing the distance.  Default is
            "matchup_spherical_distance" but for some pairs it's
            different, for example,
            "hirs-n15_hirs-n14_matchup_spherical_distance".
        """
        self.prim = prim
        self.sec = sec
        self.msd_field = msd_field

    def filter(self, ds, **extra):
        """Apply HIRS-HIRS matchup filter

        This method is called by typhon after reading every file.

        Parameters
        ----------

        ds : xarray.Dataset
            xarray Dataset such as returned by the HIRSHIRS reader.

        Returns
        -------

        xarray.Dataset
            Subset of ``ds`` containing only those matchups which meet the
            criteria outlined in the class documentation.

        """
        return ds[{"matchup_count":
            (ds[f"hirs-{self.prim:s}_lza"][:, 3, 3] < 10) &
            (ds[f"hirs-{self.sec:s}_lza"][:, 3, 3] < 10) &
            (ds[self.msd_field]<20)}]

    def finalise(self, ds):
        """Finalise HIRS-HIRS matchups

        This method is called by typhon at the end, when all matchup files
        have been read.  It sorts the matchups by primary time.

        Parameters
        ----------

        ds : xarray.Dataset
            xarray Dataset such as returned by the HIRSHIRS reader as
            defined in `typhon.datasets.tovs.HIRSHIRS`.

        Returns
        -------

        xarray.Dataset
            Subset of ``ds``, sorted.
        """
        idx = numpy.argsort(ds[f"time_{self.prim:s}"])
        return ds[{"matchup_count": idx}]

class HIMatchupCountFilter(typhon.datasets.filters.OrbitFilter):
    """HIRS-IASI matchup filter

    Filter on HIRS-IASI matchups.  Currently, the only filter this applies
    is that it rejects any IASI where any radiance is less than zero, and
    it sorts the matchups by IASI time at the very end.

    The methods on this class should not be used directly, rather objects
    of this class should be passed to
    `typhon.datasets.Dataset.read_period`.
    """

    def filter(self, ds, **extra):
        """Apply HIRS-IASI matchup filter

        This method is called by `typhon` after reading every file.

        Parameters
        ----------

        ds : xarray.Dataset
            xarray Dataset such as returned by the HIRSIASI reader as
            defined in `typhon.datasets.tovs.HIASI`.

        Returns
        -------

        xarray.Dataset
            Subset of ``ds`` containing only those matchups which meet the
            criteria outlined in the class documentation.

        """
        bad = (ds["ref_radiance"]<=0).any("ch_ref")
        return ds[{"line": ~bad}]

    def finalise(self, ds):
        """Finalise HIRS-IASI matchups

        This method is called by typhon at the end, when all matchup files
        have been read.  It sorts the matchups by primary time.

        Parameters
        ----------

        ds : xarray.Dataset
            xarray Dataset such as returned by the HIASI reader as
            defined in `typhon.datasets.tovs.HIASI`.

        Returns
        -------

        xarray.Dataset
            Subset of ``ds``, sorted.
        """
        # Another round of sorting, not sure why needed
        idx = numpy.argsort(ds["ref_time"])
        return ds[{"line": idx}]

class CalibrationCountDimensionReducer(typhon.datasets.filters.OrbitFilter):
    """Reduce the size of calibration counts.

    When we want to add calibraiton counts to matchup, we want to add only
    one (median) to each matchup, not the whole batch of 48, that becomes
    too large...

    The methods on this class should not be used directly, rather objects
    of this class should be passed to
    :meth:`typhon.datasets.Dataset.read_period`.
    """

    def finalise(self, arr):
        hcd = [k for (k, v) in arr.data_vars.items() if "calibration_position" in v.dims]
        for k in hcd:
            arr[k] = arr[k].median("calibration_position")
        return arr

# extra filters plugged into K or Kr filtering
@dataclass
class KFilter:
    """Superclass for K or Kr filters

    Takes one argument, which is the KModel or KRmodel in question.
    Through this we can access ds, ds_orig, etc.
    """

    #: `KModel` or `KRModel` to which this filter applies
    model: (KModel, KRModel)
    #: label/parameter associated with `KModel` or `KRModel` instance
    lab: str

    def filter(self, mdim, channel):
        """Apply filter

        Parameters
        ----------

        mdim : str
            Name of the matchup dimension, such as "matchup_count"
        channel : int
            Channel number to consider

        Returns
        -------

        ndarray
            Boolean array, true for matchups that are kept, false for
            matchups that need to be thrown out.
        """
        return numpy.ones(self.model.ds.dims[mdim], "?")

@dataclass
class KrFilterHomogeneousScenes(KFilter):
    """Filter on homogeneous scenes

    Select scenes where Kr is at most ``max_ratio`` times the joint
    uncertainty.
    """

    max_ratio = 5

    def filter(self, mdim, channel):
        ok = super().filter(mdim, channel)
        srf1 = self.model.prim_hirs.srfs[channel-1]
        srf2 = self.model.sec_hirs.srfs[channel-1]
        ds = self.model.ds.sel(calibrated_channel=channel)
        y1 = UADA(ds[f"{self.model.prim_name}_R_e"])
        y2 = UADA(ds[f"{self.model.sec_name}_R_e"])
        u1 = UADA(ds[f"{self.model.prim_name}_u_R_Earth_random"])
        u2 = UADA(ds[f"{self.model.sec_name}_u_R_Earth_random"])

        # Kr is always in SI units

        uc = numpy.sqrt(u1**2+u2**2)
        Kr = self.model.calc_Kr(channel, ds_to_use=self.model.ds_orig)
        ok &= ((Kr/uc)<self.max_ratio).values
        return ok

@dataclass
class KFilterFromFile(KFilter):
    """Class for any KFilter or KrFilter using a file

    Intermediate superclass for any `KFilter` (including KrFilter) that obtains
    parameters from a file.
    """
    def get_harm_filter_path(self, channel, which="K_min_dL"):
        """Get path to harmonisation parameter file

        This requires that the field ``harmfilterparams`` is defined in
        the typhon configuration, in the ``main`` section.  The same
        section is used when writing the filter, which happens in
        :func:`FCDR_HIRS.analysis.inspect_hirs_harm_matchups.plot_hist_with_medmad_and_fitted_normal`.

        Parameters
        ----------

        channel : int
            Channel number for channel of consideration
        which : str, optional
            Which filter we are considering.  Optional, defaults to
            "K_min_dL".

        Returns
        -------

        pathlib.Path
            :class:`pathlib.Path` object pointing to the location of the filter
        """
        p = pathlib.Path(typhon.config.conf["main"]["harmfilterparams"])
        p /= f"{self.model.prim_name:s}_{self.model.sec_name:s}"
        p /= f"ch{channel:d}" 
        #p /= "other_neighbours_standard_LR_K_"
        p /= f"other_{self.lab:s}_"
        p /= f"{which:s}.nc" 
        return p

@dataclass
class KrFilterDeltaLKr(KFilterFromFile):
    """Filter on ΔL/Kr ratio

    Use filter parameters derived by
    :func:`~FCDR_HIRS.analysis.inspect_hirs_harm_matchups.plot_hist_with_medmad_and_fitted_normal`
    to filter out values where ``ΔL/Kr`` is too large.

    See Vijus email 2018-09-27.
    """

    def get_DeltaL(self, channel):
        """Extract ΔL from model

        Extract ΔL from self.model (the `KrModel`).

        Parameters
        ----------

        channel : int
            Channel of interest

        Returns
        -------

        `typhon.physics.units.tools.UnitsAwareDataArray`
            A `UnitsAwareDataArray` with the radiance differences
            secondary - primary.
        """
        # always in SI units
        y1 = UADA(self.model.ds[f"{self.model.prim_name}_R_e"]).sel(
            calibrated_channel=channel)
        y2 = UADA(self.model.ds[f"{self.model.sec_name}_R_e"]).sel(
            calibrated_channel=channel)
        return y2 - y1

    def filter(self, mdim, channel):
        ok = super().filter(mdim, channel)
        ds = xarray.open_dataset(self.get_harm_filter_path(channel, "dL_over_Kr"))
        Kr = self.model.calc_Kr(channel, ds_to_use=self.model.ds_orig)
        srf1 = self.model.prim_hirs.srfs[channel-1]
        y1 = UADA(self.model.ds[f"{self.model.prim_name}_R_e"]).sel(calibrated_channel=channel)
        # Kr is always in SI units
        ΔL = self.get_DeltaL(channel)
        rat = ΔL/Kr
        fnc = scipy.interpolate.interp1d(ds["x"], ds["y"], kind="linear",
            fill_value=0, bounds_error=False)
        P_keep = fnc(rat)
        ok &= numpy.random.random(ok.size) < P_keep
        return ok

@dataclass
class KFilterKDeltaL(KFilterFromFile):
    """Filter on K-ΔL from file

    Use filter parameters derived by
    :func:`FCDR_HIRS.analysis.inspect_hirs_harm_matchups.plot_hist_with_medmad_and_fitted_normal`
    to filter out values where ``K-ΔL`` is too large.
    """
    
    def get_DeltaL(self, channel):
        y1 = UADA(self.model.ds[f"{self.model.prim_name}_R_e"]).sel(
            calibrated_channel=channel).to(
            self.model.units, "radiance", srf=self.model.prim_hirs.srfs[channel-1])
        y2 = UADA(self.model.ds[f"{self.model.sec_name}_R_e"]).sel(
            calibrated_channel=channel).to(
            self.model.units, "radiance", srf=self.model.sec_hirs.srfs[channel-1])
        return y2 - y1
    get_DeltaL.__doc__ = KrFilterDeltaLKr.get_DeltaL.__doc__

    def filter(self, mdim, channel):
        ok = super().filter(mdim, channel)
        ds = xarray.open_dataset(self.get_harm_filter_path(channel, "K_min_dL"))
        K = self.model.calc_K(channel, ds_to_use=self.model.ds, debug=False)
        ΔL = self.get_DeltaL(channel)
        K = UADA(K, dims=(mdim,), attrs={"units": rad_u["si"]}, coords={mdim: ΔL[mdim]})
        srf1 = self.model.prim_hirs.srfs[channel-1]
        y1 = UADA(self.model.ds[f"{self.model.prim_name}_R_e"]).sel(calibrated_channel=channel)
        K_K = ((y1+K).to(self.model.units, "radiance", srf=srf1)
              -y1.to(self.model.units, "radiance", srf=srf1))
        if str(ΔL.units) != str(self.model.units):
            ΔL_K = ((y1+ΔL).to(self.model.units, "radiance", srf=srf1)
                  -y1.to(self.model.units, "radiance", srfsrf1))
        else:
            ΔL_K = ΔL
        Δ = K_K-ΔL_K
        fnc = scipy.interpolate.interp1d(ds["x"], ds["y"], kind="linear",
            fill_value=0, bounds_error=False)
        P_keep = fnc(Δ)
        ok &= numpy.random.random(ok.size) < P_keep
        return ok

# inspect_hirs_matchups, work again
hh = typhon.datasets.tovs.HIRSHIRS(read_returns="xarray")
hi = typhon.datasets.tovs.HIASI(read_returns="xarray") # metopa only

class HIRSMatchupCombiner:
    """Class to combine HIRS matchups with additional data

    This is a small class to add measurements from the FIDUCEO FCDR, debug
    version, to the matchups.  Most of the heavy work is being done within
    `typhon.datasets.tovs.HIRSHIRS.combine` and its superclasses.  The
    functionality in
    :class:`FCDR_HIRS.processing.combine_matchups.HIRSMatchupCombiner` and
    :class:`FCDR_HIRS.analysis.inspect_hirs_matchups.HIRSMatchupInspector` is
    built on top of the `HIRSMatchupCombiner` class defined here.
    """

    #: information on the type of FCDR that will be read
    fcdr_info = {"data_version": "0.8pre", "fcdr_type": "debug"}
    #: fields that will be read from each of the FCDR (primary, secondary)
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


    msd_field = "matchup_spherical_distance" # default
    """Name of the field containing the matchup spherical distance

    TBs files contain either matchup_spherical_distance or
    hirs-n15_hirs-n14_matchup_spherical_distance
    """

    #: attribute to hold the mode (reference or hirs)
    mode = None
    def __init__(self, start_date, end_date, prim_name, sec_name):
        """Create HIRSMatchupCombiner object

        Parameters
        ----------

        start_date : datetime.datetime
            Beginning of period for which to combine matchups
        end_date : datetime.datetime
            End of period for which to combine matchups
        prim_name : str
            Name of primary
        sec_name : str
        `   Name of secondary
        """
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
            if ("n15" in (prim_name.lower(), sec_name.lower()) or
                prim_name.lower() == "n14" and sec_name.lower() == "n12"):
                self.msd_field = f"hirs-{prim_name}_hirs-{sec_name}_matchup_spherical_distance"
            else:
                self.msd_field = "matchup_spherical_distance"
            self.mode = "hirs"
            ds = hh.read_period(start_date, end_date,
                locator_args={"prim": prim_name, "sec": sec_name},
                fields={"hirs-{:s}_{:s}".format(s, field)
                    for field in ("x", "y", "time", "lza", "file_name", "lat", "lon",
                                  "acquisition_time", "scanpos") + tuple(
                                    "bt_ch{:02d}".format(ch) for ch in
                                    range(1, 20))
                    for s in (prim_name, sec_name)}|{self.msd_field},
                pseudo_fields={
                    "time_{:s}".format(prim_name):
                        lambda ds, D=None, H=None, fn=None: ds["hirs-{:s}_time".format(prim_name)][:, 3, 3].astype("M8[s]"),
                    "time_{:s}".format(sec_name):
                        lambda ds, D=None, H=None, fn=None: ds["hirs-{:s}_time".format(sec_name)][:, 3, 3].astype("M8[s]")},
                orbit_filters=hh.default_orbit_filters+[HHMatchupCountFilter(prim_name,sec_name,msd_field=self.msd_field)])
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

    There are currently several implementations, the most advanced one
    that is currently being used is `KModelSRFIASIDB`.  There are also
    simpler implementations, `KModelPlanck` which is based on ad-hoc BB
    L→BT conversions.  There should also be an implementation based on BTs
    simulated with a forward model, but this does not currently exist.

    For definitions, see document
    20171205-FIDUCEO-SH-Harmonisation_Input_File_Format_Definition-v6.pdf
    from Sam Hunt available on the FIDUCEO wiki::

        > Uncertainties are also required for the estimates of the match-up
        > adjustment factor, K. It currently is assumed that all errors in K are
        > independent, though are separated into two categories. Uncertainties
        > due to the SRF differences should be also be stored in a vector K r
        > and the uncertainties due to the match-up process should be stored
        > in a vector K s .

    (Note that Kr and Ks are swapped in this guide.)

    """

    #: xarray.Dataset with combined measurement obtained from FCDR files
    ds = None
    #: like `ds`, but after applying the filters
    ds_filt = None
    #: xarray.Dataset with original matchups as obtained from BC
    ds_orig = None
    #: like `ds_orig`, but filtered
    ds_filt_orig = None
    #: name of primary
    prim_name = None
    #: `fcdr.HIRSFCDR` object for primary
    prim_hirs = None
    #: name of secondary
    sec_name = None
    #: `fcdr.HIRSFCDR` object for secondary
    sec_hirs = None
    #: extra filters to be applied
    extra_filters = None

    def __init__(self, **kwargs):
        """Initialise KModel
        
        All keyword arguments correspond to attributes being set upon
        creation.
        """

        self.extra_filters = []
        for (k, v) in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise AttributeError(f"Unknown keyword argument/attribute: {k:s}")

    @abc.abstractmethod
    def calc_K(self, channel):
        """Calculate K.
        
        K is the expected value of the matchup difference.

        .. math::

            K = E(L_2 - L_1)

        """
        ...

    @abc.abstractmethod
    def calc_Ks(self, channel):
        """Calculate Ks.

        Ks is the structured uncertainty associated with K.

        Note that Kr, the independent uncertainty associated with K, has
        its own calculation method.
        """
        ...

    filtered = False
    def limit(self, ok, mdim):
        """Reduce dataset to those values

        Define :attr:`ds_filt` and :attr:`ds_filt_orig` based on the boolean array
        ``ok`` associated with dimension ``mdim``.  In practice, ``ok`` is
        obtained by filtering.

        This method does not return anything, but after calling it,
        ``self.ds_filt`` and ``self.ds_filt_orig`` will be defined.

        Parameters
        ----------

        ok : ndarray
            Boolean array as returned by filters, used to index the
            datasets.
        mdim : str
            Name of dimension along which the filtering is applied, for
            example, "matchup_spherical_distance".
        """
        self.ds_filt = self.ds[{mdim:ok}]
        self.ds_filt_orig = self.ds_orig[{mdim:ok}]
        self.filtered = True

    def filter(self, mdim, channel):
        """Extra filtering imposed by this model

        Apart from the usual filtering going on, what filtering does this
        model need to impose on the entire process in order to function?

        Parameters
        ----------

        mdim : str
            Name of dimension along which the filtering is applied, for
            example, "matchup_spherical_distance".
        channel : int
            Channel along which the filtering is applied.

        Returns
        -------

        ndarray
            Boolean array, True for matchups to be keps, False for
            matchups to be filtered out.
        """
        rv = numpy.ones(self.ds.dims[mdim], "?")
        for ef in self.extra_filters:
            rv &= ef.filter(mdim, channel)
        return rv

    def extra(self, channel, ok):
        """Return Dataset with extra information

        Some models may want to return additional information, that may be
        included with the resulting harmonisation files for debugging
        purposes.  Those models can do so by implemening this method.

        Parameters
        ----------

        channel : int
            Channel for which the extras are obtained
        ok : ndarray
            Boolean array indicating valid matchups.


        Returns
        -------

        xarray.Dataset
            Dataset containing extra information.  Possibly empty.
        """

        return xarray.Dataset()

class KrModel(metaclass=abc.ABCMeta):
    """Implementations of models to estimate the matchup uncertainty

    Abstract base class defining an interface for all models implementing
    an estimate of independent matchup uncertainty (``Kr``).  There are
    currently two implementations: `KrModelLSD` and `KrModelJointLSD`.
    The latter is currently used for the estimation of ``Kr``.
    """

    def __init__(self, ds, ds_orig, prim_name, prim_hirs, sec_name,
                 sec_hirs, extra_filters=None):
        """Initialise Kr model

        Parameters
        ----------

        ds : xarray.Dataset
        ds_orig : xarray.Dataset
        prim_name : str
            Name of primary
        prim_hirs : fcdr.HIRSFCDR
            HIRS FCDR object corresponding to the primary
        sec_name : str
            Name of secondary
        sec_hirs : fcdr.HIRSFCDR
            HIRS FCDR object corresponding to the secondary
        extra_filters : List[`KFilter`], optional
            List of extra filters, which must all be instances of
            subclasses of `KFilter`, that this model imposes upon the
            universe.
        """
        self.ds = ds
        self.ds_filt = None
        self.ds_orig = ds_orig
        self.ds_filt_orig = None
        self.prim_name = prim_name
        self.prim_hirs = prim_hirs
        self.sec_name = sec_name
        self.sec_hirs = sec_hirs
        if extra_filters is None:
            self.extra_filters = []
        else:
            self.extra_filters = extra_filters

    @abc.abstractmethod
    def calc_Kr(self, channel):
        """Calculate Kr

        Parameters
        ----------

        channel : int
            Channel for which Kr is being calculated

        Returns
        -------

        xarray.DataArray
            Estimate of Kr in radiance units
        """
        ...

    filtered = False
    def limit(self, ok, mdim):
        self.ds_filt = self.ds[{mdim:ok}]
        self.ds_filt_orig = self.ds_orig[{mdim:ok}]
        self.filtered = True
    limit.__doc__ = KModel.limit.__doc__

    def filter(self, mdim, channel):
        rv = numpy.ones(self.ds.dims[mdim], "?")
        for ef in self.extra_filters:
            rv &= ef.filter(mdim, channel)
        return rv
    filter.__doc__ = KModel.filter.__doc__

class KModelPlanck(KModel):
    """Simplified implementation.

    This is a temporary implementation for estimating values of K
    (expected difference in L).

    This implemtation is incorrect and no longer in use.
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

    In case of a IASI reference, the estimates for both K and Ks are zero.
    """
    def calc_K(self, channel):
        return numpy.zeros(shape=self.ds_filt.dims["line"])

    def calc_Ks(self, channel):
        return numpy.zeros(shape=self.ds_filt.dims["line"])

class KrModelLSD(KrModel):
    """Estimate Kr using the local standard deviation of the primary

    This is an implementation of the KrModel using the local standard
    deviation of the primary only.

    For an implementation that uses the combined local standard deviation
    between primary and secondary, use `KrModelJointLSD`.
    """
    def calc_Kr_for(self, channel, which, ds_to_use=None):
        """Helper function to calculate Kr for primary or secondary

        Parameters
        ----------

        channel : int
            Channel to use
        which : str
            Name of either primary or secondary
        ds_to_use : xarray.Dataset
            ds_to_use should refer to either self.ds_orig or self.ds_filt_orig

        Returns
        -------

        xarray.DataArray
            Kr estimated as LSD
        """
        if ds_to_use is None:
            ds_to_use = self.ds_filt_orig
        btlocal = ds_to_use["hirs-{:s}_bt_ch{:02d}".format(which, channel)]
        # 2018-08-14: disabling this, it's the wrong place and leads to
        # trouble with nans propagating into K.
        #btlocal.values.reshape((-1,))[btlocal.values.ravel()>400] = numpy.nan # not all are flagged correctly
        btlocal = btlocal.loc[{"hirs-{:s}_ny".format(which): slice(1, 6),
                           "hirs-{:s}_nx".format(which): slice(1, 6)}].stack(
                    z=("hirs-{:s}_ny".format(which),
                       "hirs-{:s}_nx".format(which)))
        btlocal = UADA(btlocal)
        srf = SRF.fromArtsXML(
            typhon.datasets.tovs.norm_tovs_name(which).upper(),
            "hirs", channel)
        radlocal = btlocal.to(rad_u["si"], "radiance", srf=srf)
        lsd = radlocal.std("z")
        #lsd.attrs["units"] = "K"
        # Convert from K to radiance
        #lsd = UADA(lsd).to(rad_u["si"], "radiance", srf=srf)
        return lsd

    def calc_Kr(self, channel, ds_to_use=None):
        return self.calc_Kr_for(self.prim_name)

    def filter(self, mdim, channel):
        ok = super().filter(mdim, channel)
        ok &= functools.reduce(
            operator.and_,
            ((self.ds_orig[f"hirs-{s:s}_bt_ch{channel:02d}"].notnull()
                    .sum(f"hirs-{s:s}_nx").sum(f"hirs-{s:s}_ny")>25).values
                for s in (self.prim_name, self.sec_name)),
            ok)
        return ok

class KrModelJointLSD(KrModelLSD):
    """Estimate Kr using the joint local standard deviation

    This is an implementation of the `KrModel` using the joint local
    standard deviation between primary and secondary.  It builds on top of
    `KrModelLSD` which uses the LSD for the primary only.
    """
    def calc_Kr(self, channel, ds_to_use=None):
        lsd_prim = self.calc_Kr_for(channel, self.prim_name, ds_to_use=ds_to_use)
        lsd_sec = self.calc_Kr_for(channel, self.sec_name, ds_to_use=ds_to_use)
        return numpy.sqrt(lsd_prim**2+lsd_sec**2)

class KrModelIASIRef(KrModel):
    """Estimate Kr for IASI-HIRS matchups
    
    I don't have a good model estimating Kr for IASI-HIRS matchups.  This
    is currently the only implemetation.  It assumes a hardcoded 0.1 K
    uncertainty converted back to radiance space.
    """

    def calc_Kr(self, channel, ds_to_use=None):
        """Calculate Kr for channel

        ds_to_use is by default self.ds_filt.  But sometimes we need to
        call calc_Kr before we've done filtering, for example, when we use
        it to do filtering, then you will want to pass self.ds.
        """
        if ds_to_use is None:
            ds_to_use = self.ds_filt
        srf = SRF.fromArtsXML(
            typhon.datasets.tovs.norm_tovs_name(self.sec_name).upper(),
            "hirs", channel)
        return abs(
            ureg.Quantity(
                ds_to_use["metopa_T_b"].sel(calibrated_channel=channel).values, "K"
                    ).to(rad_u["si"], "radiance", srf=srf) -
            ureg.Quantity(
                ds_to_use["metopa_T_b"].sel(calibrated_channel=channel).values+0.1, "K"
                    ).to(rad_u["si"], "radiance", srf=srf))



class KModelSRFIASIDB(KModel):
    """Estimate K for HIRS-HIRS matchups, using a IASI spectral database.

    Based on the SRF recovery method described in PD4.4.  IASI training
    data obtained from the period between `KModelSRFIASIDB.iasi_start` and
    `KModelSRFIASIDB.iasi_end` are used to simulated HIRS using SRFs
    shifted such as reported by Paul Menzel and included with RTTOV, such
    that we get HIASI for all channels for the primary and the secondary,
    based on the same IASI spectra, such that any differences are
    exclusively due to reported SRF differences.  Using the HIASI database, I
    develop a regression to predict channel n on the secondary from one
    or more channels on the primary.  Applying this regression to the real
    matchups, I the difference between the estimate for secondary channel
    n and the real primary channel n is the estimate for K.

    This estimate is performed bidirectionally.  The difference is as
    entimate of Ks.
    
    There are a few ways in which this can be varied:

    * What channels do I use as the predictor?  Only channel n, all
      channels, or something in-between?  This can be configured with the
      ``chan_pairs`` argument.

    * What type of regression to use (``regression``)?  This can be
      ``"LR"`` or ``"ODR"``.

    * Should the regression predict the value for the secondary, or
      should it predict the Δ between the secondary and the primary?  This
      is configured with the ``mode`` argument.

    * Should all of this happen in radiance space or brightness
      temperature space?  See the ``units`` argument.

    For details, see documentation on the attributes and on the
    constructor.
    """

    iasi_start = datetime.datetime(2011, 1, 1)
    """datetime.datetime: start period for IASI training database

    Beginning date for which IASI data are read in order to produce IASI
    training database used to simulate HIASI.  Note that due to data
    volume limitations, I do not read the original IASI data but rather a
    subset via the typhon `typhon.datasets.tovs.IASISub` class.
    """
    iasi_end = datetime.datetime(2011, 2, 1)
    """datetime.datetime: end period for IASI training database

    End date for which IASI data are read, or rather,
    `typhon.datasets.tovs.IASISub` data.
    """
    iasi = typhon.datasets.tovs.IASISub(name="iasisub")
    """typhon.datasets.tovs.IASISub: instance use do read IASI data

    This attribute holds the `typhon.datasets.tovs.IASISub` object used to
    read IASI data.  It is defined by default and there should normally be
    no reason to redefine it.
    """
    #: ndarray: IASI data (internal use)
    M_iasi = None
    #: ndarray: IASI training database (internal use)
    Ldb_iasi_full = None
    #: ndarray: HIASI training database (internal use)
    Ldb_hirs_simul = None
    #: List[SRF]: SRFs
    srfs = None
    #: Mapping[str, Mapping[int, regression_model]]:  various fitters
    fitter = None

    regression = "LR"
    """str: type of regression to use

    String describing what type of regression to use.  Valid options are
    ``"LR"``, which is the default, and ``"ODR"``, for orthogonal
    distance regression.
    """
    chan_pairs = None
    """Mapping[int, array_like]: what primary channels predict what secondary

    Dictionary that describes, for each channel on the secondary, what
    channels from the primary are used to predict it. 
    """
    #: str: label describing the type of channel pairing
    chan_pairs_label = None
    mode = "standard"
    """str: String describing whether regression should target BT or ΔBT

    The mode can be either ``"standard"``, which means the regression will
    target brightness temperatures, or ``"delta"``, which means the
    regression will target the ΔBT between the primary and the secondary,
    for the channel being targeted.
    """
    units = rad_u["si"]
    """pint Unit: what units to use

    A pint Unit describing in what unit radiances or brightness
    temperatures are predicted.  The module
    `typhon.physics.units.common` defines a ``radiance_units`` dictionary
    that contains units for radiance in SI or typical IR units, or one can
    use kelvin.  It's also valid to pass a string that can be converted to
    a unit.
    """
    #: bool: if True, generate all possible mode/unit/chan/regression combinations
    debug = False
    #: Mapping: dictionary filled in case of debug mode (internal use)
    others = None
    #: Mapping: dictionary to hold estimates for K in both directions
    K = None
    #regression = "ODR"
    # FIXME: Use ODR with uncertainties
#    regression_model = sklearn.linear_model.LinearRegression
#    regression_args = {"fit_intercept": True}



    def __init__(self, chan_pairs="all", *args, **kwargs):
        """Initialise KModelSRFIASIDB class

        Parameters
        ----------

        chan_pairs : str
            A string describing how the channel pairs are distributed.
            Valid values are ``"all"``, which means all channels from the
            primary are used to predict any channel from the secondary,
            ``"single"``, which means only channel n on the primary is
            used to predict channel n on the secondary, and
            ``"neighbours"``, which uses channels ``[n-1, n, n+1]`` on the
            primary to predict channel n on the secondary.  A further
            case, ``"optimal"``, which has an optimal choice for each
            channel, is not yet implemented
        *args
            Passed to superclass, which does not use it, do not pass.
        **kwargs
            Passed to superclass, which results in the setting of already
            defined attributes.  See attribute documentation, in
            particular for `regression`, `mode`, `units`, and `debug`.
        """

        self.chan_pairs_label = chan_pairs
        # FIXME: add an "optimal", channel-specific mapping
        if chan_pairs == "all":
            chan_pairs = dict.fromkeys(numpy.arange(1, 20), numpy.arange(1, 20))
        elif chan_pairs == "single":
            chan_pairs = {c:[c] for c in range(1, 20)}
        elif chan_pairs == "neighbours":
            chan_pairs = {ch: numpy.arange(max(ch-1, 1), min(ch+2, 20)) for ch in range(1, 20)}
        elif chan_pairs == "optimal":
            chain_pairs = {
                1: [1],
                3: [3],
                4: [3, 4, 5],
                }
            raise NotImplementedError("To be implemented")
        else:
            raise ValueError(f"Unknown value for chan_pairs: {chan_pairs!s}")
        self.chan_pairs = chan_pairs
        super().__init__(*args, **kwargs)
        if self.debug:
            self.others = {}
            for (cp, regr, md, u) in itertools.product(
                    ["single", "neighbours"],
                    ["LR", "ODR"],
                    ["standard"], # "delta
                    [rad_u["si"], rad_u["ir"], ureg.Unit("K")]):
                unitlab = re.sub(r'/([A-Za-z]*)', r' \1^-1', "{:~P}".format(u)).replace("-1²", "-2")
                lab = unicodedata.normalize("NFKC", f"{cp:s}_{md:s}_{regr:s}_{unitlab:s}")
                self.others[lab] = self.__class__(
                        chan_pairs=cp, 
                        mode=md,
                        units=u,
                        regression=regr,
                        debug=False,
                        ds=self.ds,
                        ds_orig=self.ds_orig,
                        prim_name=self.prim_name,
                        prim_hirs=self.prim_hirs,
                        sec_name=self.sec_name,
                        sec_hirs=self.sec_hirs)
        self.K = dict.fromkeys(range(1, 20), None)

    def get_lab(self):
        """Return label for self settings

        Get a label that describes the settings for this instance, a
        string describing the channel pairing, the mode, the regression
        type, and the unit.

        Returns
        -------

        str
            String describing the instance configuration.
        """
        unitlab = re.sub(r'/([A-Za-z]*)', r' \1^-1', "{:~P}".format(self.units)).replace("-1²", "-2")
        lab = unicodedata.normalize("NFKC", f"{self.chan_pairs_label:s}_{self.mode:s}_{self.regression:s}_{unitlab:s}")
        return lab


    def limit(self, ok, mdim):
        super().limit(ok, mdim)
        if self.debug:
            for v in self.others.values():
                v.ds_filt = self.ds_filt
                v.ds_filt_orig = self.ds_filt_orig

    def read_iasi(self):
        """Read spectral database from IASI.

        Details on what is read are stored in self attributes `iasi_start`,
        `iasi_end`, passed on on object creation.  The result is stored in
        the attribute `M_iasi`.  This method does not take any input
        arguments nor does it return anything.  In debug mode, it also
        sets the same object for the other instances held in `debug`.
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

        Make sure IASI data have been read, calling `read_iasi` if
        necessary, and construct the full IASI spectral database in SI
        units, storing it in `Ldb_iasi_full`.  Do the same for other
        instances if in debug mode.  No inputs or outputs.
        """
        if self.M_iasi is None:
            self.read_iasi()
        L_wavenum = ureg.Quantity(self.M_iasi["spectral_radiance"][2::5, 2, :8461], unit_iasi)
        L_freq = L_wavenum.to(rad_u["si"], "radiance")
        self.Ldb_iasi_full = L_freq
        if self.debug:
            for v in self.others.values():
                v.Ldb_iasi_full = self.Ldb_iasi_full # always in freq

    def init_srfs(self):
        """Initialise SRFs, reading from RTTOV format
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

        Calculate IASI-simulated HIRS for all channels for both
        satellites in the matchup pair.  Ensure that IASI data and SRFs
        are ready to read.  In debug mode, do the same for all pairs in
        self.debug.
        
        Does not input or output anything but sets attributes needed
        elsewhere.
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
        """Set up regression objects

        Populate the self.fitter dictionary with regression objects in
        both directions for all channels.  Initiates Ldb and reads IASI data
        if necessary.
        """
        if self.Ldb_hirs_simul is None:
            self.init_Ldb()
        # make one fitter for each target channel, in both directions
        fitter = {}
        for (from_sat, to_sat) in [(self.prim_name, self.sec_name),
                                   (self.sec_name, self.prim_name)]:
            fitter[f"{from_sat:s}-{to_sat:s}"] = {}
            for chan in range(1, 20):
                y_ref_ch = self.chan_pairs[chan]
                y_ref = self.Ldb_hirs_simul[from_sat].sel(chan=y_ref_ch).values.copy()
                y_target = self.Ldb_hirs_simul[to_sat].sel(chan=chan).values.copy()
                if self.mode == "delta":
                    y_target -= self.Ldb_hirs_simul[from_sat].sel(chan=chan)
                # for training with sklearn, dimensions should be n_p × n_c
                if self.regression == "LR":
                    clf = sklearn.linear_model.LinearRegression(fit_intercept=True)
                    clf.fit(y_ref.T, y_target)
                elif self.regression == "ODR":
                    odr_data = scipy.odr.RealData(y_ref, y_target)
                    # set β0 to 0 offset and 0 for all channels, but 1 for
                    # same channel.  But not really zero as ODRpack guide
                    # section 1.E says it should not be exactly zero.
                    β0 = numpy.zeros(len(y_ref_ch)+1)+0.001
                    β0[numpy.atleast_1d(self.chan_pairs[chan]).tolist().index(chan)+1] = 1
                    myodr = scipy.odr.ODR(odr_data, scipy.odr.multilinear, beta0=β0) 
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

    _y_pred = None
    def calc_K(self, channel, ds_to_use=None, debug=None):
        """Calculate K for channel.

        Calculates K in both directions: from primary to secondary, and
        from secondary to primary.  In an ideal case, the two results
        differ only by sign.  This method puts both in the self.K
        dictionary in units of self.units (within ``self.K[channel]``,
        and returns the average of -(secondary from primary) and (primary
        from secondary) converted to units of kelvin.

        Parameters
        ----------

        channel : int
            Channel number for which to estimate K
        ds_to_use: xarray.Dataset, optional
            Should be set either to ``self.ds`` or to ``self.ds_filt``.
            The latter is default.  You'll want to put it to ``self.ds``
            if calling this before the filtering has been performed, which
            may be needed for filters that are a function of K.
        debug : bool, optional  
            Whether or not we are running in debug mode.  Overrides
            self.debug.

        Returns
        -------

        ndarray
            Estimate for K

        """
        if self.fitter is None:
            self.init_regression()
        if ds_to_use is None:
            ds_to_use = self.ds_filt
        if debug is None:
            debug = self.debug
        # Use regression to predict Δy for channel from set of reference
        # channels.
        K = []
        for (from_sat, to_sat) in [(self.prim_name, self.sec_name),
                                   (self.sec_name, self.prim_name)]:
            clf = self.fitter[f"{from_sat:s}-{to_sat:s}"][channel]
            y_source = ds_to_use[f"{from_sat:s}_R_e"].sel(calibrated_channel=self.chan_pairs[channel])
            y_source = numpy.vstack(
                [UADA(y_source.sel(calibrated_channel=c)).to(
                    self.units, "radiance", srf=self.srfs[from_sat][c-1]).values
                    for c in range(1, 20)
                    if c in y_source.calibrated_channel.values]).T
#            y_ref = self.ds_filt[f"{to_sat:s}_R_e"].sel(calibrated_channel=channel)
#            y_ref = UADA(y_ref).to(
#                    self.units, "radiance",
#                    srf=self.srfs[to_sat][channel-1]).values
            model = self.fitter[f"{from_sat:s}-{to_sat:s}"][channel]
            if isinstance(model, scipy.odr.odrpack.Output):
                y_pred = (model.beta[0] + model.beta[numpy.newaxis, 1:] * y_source).sum(1)
            else:
                y_pred = model.predict(y_source)
            if self.mode == "delta":
                K.append(y_pred)
            else:
                K.append(y_pred - y_source[:, list(self.chan_pairs[channel]).index(channel)])
            # derive Ks from spread of predictions?  Or Δ with y_ref?  Or
            # from uncertainty provided by regression?  Isn't that part of
            # Kr instead?
        self._y_pred = y_pred
        self.K[channel] = K
        if self.debug and not debug:
            for v in self.others.values():
                v.calc_K(channel, ds_to_use=ds_to_use)
        # convert back to si units for conversion, but cannot convert K
        # (if ΔBT) directly via SRF, as this conversion is
        # only valid for actual BTs, not ΔBTs
        K = numpy.array([-K[0], K[1]]).mean(0)
        K = (ureg.Quantity(y_pred+K, self.units).to(
             rad_u["si"], "radiance", srf=self.srfs[from_sat][channel-1]) -
             ureg.Quantity(y_pred, self.units).to(
             rad_u["si"], "radiance", srf=self.srfs[from_sat][channel-1]))
        return K.m

    def calc_Ks(self, channel):
        """Estimate Ks

        Calculate Ks, after K has already been calculated.  This is
        estimated by taking the absolute difference of the two different
        ways of estimating K (forward and backward).  I don't think this
        is a good estimate but it's what I've implemented at the moment.

        Parameters
        ----------

        channel : int
            Channel number

        Returns
        -------

        ndarray
            Estimate for Ks
        """
        # spread of db will inform us of uncertainty in predicted
        # radiance?  Or take double direction approach?
        if self.K[channel] is None:
            self.calc_K(channel)
        # FIXME: This is not a good estimate, this Δ tends to be a
        # Gaussian which is not centred at zero, it can have either sign.
        # I don't know what this Δ does indicate, but a good estimate for
        # Ks it is not.
        # NB: self.K is Dict[List[ndarray, ndarray]]
        Ks = abs(self.K[channel][0] - self.K[channel][1])
        if self._y_pred is None:
            raise RuntimeError("Must call calc_K first")
        Ks = (ureg.Quantity(self._y_pred+Ks, self.units).to(
              rad_u["si"], "radiance", srf=self.srfs[self.prim_name][channel-1]) -
              ureg.Quantity(self._y_pred, self.units).to(
              rad_u["si"], "radiance", srf=self.srfs[self.prim_name][channel-1]))
        return Ks.m

    def extra(self, channel, ok):
        """Get dictinoary with some extra information to add to matchups

        Implementation of method that returns some extra information, that
        will be added to the harmonisation matchups for debugging
        purposes.  This method returns a dictionary with the fields
        ``K_forward`` and ``K_backward``, which are the two
        different estimates of ``K``, identicaly in a perfect world, not
        so in reality.  If self.debug is set, it also defined a bunch of
        ``"K_other_..."`` fields for all the debug versions.

        Parameters
        ----------

        channel : int
            Channel for which to return extras
        ok : indexer
            Indexer for filtering

        Returns
        -------

        xarray.Dataset
            Dataset with extra fields to be merged with matchups.
        """
        ds = super().extra(channel, ok)
        ds["K_forward"] = (("M",), self.K[channel][0])
        ds["K_backward"] = (("M",), self.K[channel][1])
        if self.debug:
            for (k, v) in self.others.items():
                ds[f"K_other_{k:s}_forward"] = (("M",), v.K[channel][0][ok])
                ds[f"K_other_{k:s}_backward"] = (("M",), v.K[channel][1][ok])
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

    def filter(self, mdim, channel):
        # go through self.ds_filt R_e values.
        ok = super().filter(mdim, channel)
        for sat in self.prim_name, self.sec_name:
            ok &= self.ds[f"{sat:s}_R_e"].sel(calibrated_channel=self.chan_pairs[channel]).notnull().all("calibrated_channel").values
        return ok
