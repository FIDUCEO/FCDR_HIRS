"""Generate FCDR for satellite and period

Generate HIRS FCDR for a particular satellite and period.

"""

DATA_CHANGELOG="""
0.1

First version with seperate random and systematic uncertainty.

0.2

Added brightness temperatures (preliminary).
Fixed some bugs.

0.3

Added Metop-B
Changed propagation uncertainty to BT from analytical to numerical
Higher precision in debug version
Removed some fields we will not use (easy)

0.4

Fixed bug in too large uncertainty on SRFs, in particular for
small-wavelength channels
Renamed systematic to nonrandom
Improved storage for coordinates and other values (debug+easy)
Applied encodings and dtypes for easy

0.5

Variable attributes were missing in easy FCDR.
Added support for HIRS/2.
Added starting time to global attributes.
Less verbose global attributes.
Added LUT betwen BT and L.
Added channel correlation matrix.
Ensure enough significant digits for uncertainties.
Added typical nonrandom correlation scale.
Improved coordinates.
Renamed angles.
Changed filename structure to follow FIDUCEO standard.

0.6

Change estimate for ε=1 to ε=0.98 (or rather, a_3=0 to a_3=-0.02)
Added bias term a_4 (debug only)
Added handling of flags

0.7

Correct time axis for uncertainties per calibration cycle, preventing half
the values being nans (debug version only)

Fix bug which caused random uncertainty on Earth counts to be estimated a
factor √48 too low.

Fix bug which caused both random and nonrandom uncertainty to be
incorrectly propagated from radiance to brightness temperature space, see
#134

Handle many more forms of problematic data, leading to a more complete
dataset.

Changed approach to self-emission:
- use more temperatures for prediction, to be precise, use all
  temperatures that are available for all HIRS
- use ordinary least squares rather than partial least squares
- detect and flag cases of gain change, and use old model in this case
- keep track of times used to train self-emission using a 2-D (channel,
  time) coordinate for two fields (start, end).  Has to be 2-D because
  self-emission might fail for some channels but not others.

Changed approach to flags:
- Still copy over flags, but do not set corresponding data fields to nan
- Copy over/consolidate mirror flag.  Since the easy FCDR does not contain
  a flag per minor frame, the entire scanline is flagged if there is any
  mirror flag for any minor frame anywhere on the scanline.
- Added more flags to both easy and debug versions

0.8

Overlap between subsequent granules are now selected based on "best
scanline" criterion, rather than always the oldest (GH#7).

Add missing months January-February 1985 for NOAA-7 (reported by MS).

Fix encoding for radiances in debug version.

Added correlation length scales and channel error correlations, according
to recipes developed by Merchant et al (GH #212, #228).

Use first results from harmonisation (NOAA-16 onwards).
"""

VERSION_HISTORY_EASY="""Generated from L1B data using FCDR_HIRS.  See
release notes for details on versions used."""


import sys
import pathlib
#pathlib.Path("/dev/shm/gerrit/cache").mkdir(parents=True, exist_ok=True)
from .. import common
import argparse
import subprocess
import warnings
import functools
import operator
import enum
import types

def parse_cmdline():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser = common.add_to_argparse(parser,
        include_period=True,
        include_sat=1,
        include_channels=False,
        include_temperatures=False)

    parser.add_argument("modes", action="store", type=str,
        nargs="+", choices=["easy", "debug", "none"],
        help="What FCDR(s) to write?")

    parser.add_argument("--days", action="store", type=int,
        default=0,
        metavar="N",
        help=("If non-zero, generate only first N days of each month. "
            "Use this to reduce data volume for debug version but "
            "still have data throughout lifetime."))

    return parser.parse_args()
p = parse_cmdline()

import logging
logging.basicConfig(
    format=("%(levelname)-8s %(asctime)s %(module)s.%(funcName)s:"
            "%(lineno)s: %(message)s"),
    filename=p.log,
    level=logging.DEBUG if p.verbose else logging.INFO)

import pkg_resources
import datetime

import numpy
import pandas
import xarray
import typhon.datasets.dataset
import typhon.datasets.filters
from typhon.physics.units.common import radiance_units as rad_u
from typhon.physics.units.tools import UnitsAwareDataArray as UADA
from .. import fcdr
from .. import models
from .. import effects
from .. import measurement_equation as me
from .. import _fcdr_defs
from .. import metrology

import fiduceo.fcdr.writer.fcdr_writer

class FCDRGenerator:
    # for now, step_size should be smaller than segment_size and I will
    # only store whole orbits within each segment
    epoch = datetime.datetime(1970, 1, 1, 0, 0, 0)
    window_size = datetime.timedelta(hours=24)
    segment_size = datetime.timedelta(hours=6)
    step_size = datetime.timedelta(hours=4)
    skip_problem_step = datetime.timedelta(seconds=900)
    data_version = "0.8pre"
    # see comment in models.Rself
    rself_temperatures = ["baseplate", "internal_warm_calibration_target",
        "scanmirror", "scanmotor", "secondary_telescope"]
    # 2017-07-14 GH: Use LR again, seems to work better than PDR although
    # I don't know why it should.
    rself_regr = ("LR", {"fit_intercept": True})
    orbit_filters = None # set in __init__
    pseudo_fields = {
        "filename":
            lambda M, D, H, fn: numpy.full(M.shape[0], pathlib.Path(fn).stem)}

    # maximum number of correlation length to store in single FCDR debug
    # file.  For the easy, it's N//2 where N is length of orbit.
    max_debug_corr_length = 1000

    # FIXME: use filename convention through FCDRTools, 
    def __init__(self, sat, start_date, end_date, modes):
        logging.info("Preparing to generate FCDR for {sat:s} HIRS, "
            "{start:%Y-%m-%d %H:%M:%S} – {end_time:%Y-%m-%d %H:%M:%S}. "
            "Software:".format(
            sat=sat, start=start_date, end_time=end_date))
        pr = subprocess.run(["pip", "freeze"], stdout=subprocess.PIPE)
        info = pr.stdout.decode("utf-8")
        logging.info(info)
        self.info = info
        self.satname = sat
        self.fcdr = fcdr.which_hirs_fcdr(sat, read="L1B")
        self.fcdr.my_pseudo_fields.clear() # suppress pseudo fields radiance_fid, bt_fid here
        self.start_date = start_date
        self.end_date = end_date

        orbit_filters=[
#                typhon.datasets.filters.FirstlineDBFilter(
#                    self.fcdr,
#                    self.fcdr.granules_firstline_file),
                typhon.datasets.filters.HIRSBestLineFilter(self.fcdr),
                typhon.datasets.filters.TimeMaskFilter(self.fcdr),
                typhon.datasets.filters.HIRSTimeSequenceDuplicateFilter()]
        self.orbit_filters = orbit_filters
        self.dd = typhon.datasets.dataset.DatasetDeque(
            self.fcdr, self.window_size, start_date,
            orbit_filters=orbit_filters,
            pseudo_fields=self.pseudo_fields)

        self.rself = models.RSelf(self.fcdr,
            temperatures=self.rself_temperatures,
            regr=self.rself_regr)
        self.modes = modes

    def process(self, start=None, end_time=None):
        """Generate FCDR for indicated period
        """
        start = start or self.start_date
        end_time = end_time or self.end_date
        logging.info("Now processing FCDR for {self.satname:s} HIRS, "
            "{start:%Y-%m-%d %H:%M:%S} – {end_time:%Y-%m-%d %H:%M:%S}. ".format(
            self=self, start=start, end_time=end_time))
        anyok = False
        try:
            self.dd.reset(start,
                orbit_filters=self.orbit_filters,
                pseudo_fields=self.pseudo_fields)
        except typhon.datasets.dataset.DataFileError as e:
            warnings.warn("Unable to generate FCDR: {:s}".format(e.args[0]))
        while self.dd.center_time < end_time:
            try:
                self.dd.move(self.step_size,
                    orbit_filters=self.orbit_filters,
                    pseudo_fields=self.pseudo_fields)
                self.make_and_store_piece(self.dd.center_time - self.segment_size,
                    self.dd.center_time)
            except (fcdr.FCDRError, typhon.datasets.dataset.DataFileError) as e:
                warnings.warn("Unable to generate FCDR: {:s}".format(e.args[0]))
            else:
                anyok = True
        if anyok:
            logging.info("Successfully completed, completed successfully.")
            logging.info("Everything seems fine.")
        else:
            raise fcdr.FCDRError("All has failed")
    
    def fragmentate(self, piece):
        """Yield fragments per orbit
        """
        ssp = piece["lat"].sel(scanpos=28)
        crossing = xarray.DataArray(
            numpy.r_[
                True,
                ((ssp.values[1:] > 0) & (ssp.values[:-1] < 0))],
            coords=ssp.coords)
        segments = numpy.r_[
            crossing.values.nonzero()[0],
            -1]
        time_segments = [piece["lat"]["time"][s] for s in segments]
        time_segments[-1] = numpy.datetime64("99999-12-31") # always in the future
#            piece.coords["time"].size]

        # make sure I split by all time-coordinates… 
        time_coords = {k for (k, v) in piece.coords.items()
                           if v.dtype.kind=="M"}
        # don't write partial orbits; skip first and last piece within
        # each segment.  Compensated by stepping 4 hours after processing
        # each 6 hour segment.  This is a suboptimal solution.
        for (s, e) in list(zip(segments[:-1], segments[1:]))[1:-1]:
            p = piece
            # only loop through time-coords that are also dimensions,
            # other coords we don't want to subselect on; for example,
            # rself period coordinates SHOULD refer to a period outside of
            # the coverage time for the granule
            for tc in time_coords & piece.dims.keys():
                p = p[{tc: 
                        (p[tc] >= piece["lat"]["time"][s]) &
                        (p[tc] < piece["lat"]["time"][e])}]
            yield p
            #yield piece.isel(time=slice(s, e))

    def make_and_store_piece(self, from_, to):
        """Generate and store one “piece” of FCDR

        This generates one “piece” of FCDR, i.e. in a single block.  For
        longer periods, use the higher level method `process`.
        """

        (piece, sensRe) = self.get_piece(from_, to, return_more=True)
#        self.store_piece(piece)
        for piece in self.fragmentate(piece):
            piece = self.add_orbit_info_to_piece(piece)


            self.store_piece(piece)

    def get_piece(self, from_, to, return_more=False):
        """Get FCDR piece for period.

        Returns a single xarray.Dataset
        """
        subset = self.dd.data.sel(time=slice(from_, to))
        # This is not supposed to happen consistently, but may happen if
        # for some 24-hour period the first >6 hours were not available
        # but the next <18 hours were
        if subset.dims["time"] == 0:
            raise fcdr.FCDRError("Could not find any L1B data in subset "
                f"{from_:%Y-%m-%d %H:%M}--{to:%Y-%m-%d %H:%M}.  NB: "
                "if this error message happens consistently there is a "
                "bug!")
        elif subset.dims["time"] < 3:
            raise fcdr.FCDRError(f"Found only {subset.dims['time']:d} "
                "scanlines in period "
                f"{from_:%Y-%m-%d %H:%M}--{to:%Y-%m-%d %H:%M}.  NB: "
                "if this error message happens consistently there is a "
                "bug!")
        context = self.dd.data
#        if not (context["time"].values[0] < subset["time"].values[0] <
#                subset["time"].values[-1] < context["time"].values[-1]):
#            warnings.warn("Cannot generate FCDR for "
#                "{0:%Y-%m-%d %H:%M:%S} – {1:%Y-%m-%d %H:%M:%S}.  Context is "
#                "needed for interpolation of calibration and self-emission "
#                "model (among others), but context only available "
#                "between {2:%Y-%m-%d %H:%M:%S} – {3:%Y-%m-%d %H:%M:%S}.  I "
#                "will skip {4:.0f} seconds and hope for the best.".format(
#                    subset["time"].values[0].astype("M8[ms]").astype(datetime.datetime),
#                    subset["time"].values[-1].astype("M8[ms]").astype(datetime.datetime),
#                    context["time"].values[0].astype("M8[ms]").astype(datetime.datetime),
#                    context["time"].values[-1].astype("M8[ms]").astype(datetime.datetime),
#                    self.skip_problem_step.total_seconds()),
#                fcdr.FCDRWarning)
#            from_ = subset["time"].values[0]
#            to = subset["time"].values[-1]
#            while context["time"].values[0] >= subset["time"].values[0]:
#                from_ += numpy.timedelta64(self.skip_problem_step)
#                subset = subset.sel(time=slice(
#                    from_.astype("M8[ms]").astype(datetime.datetime),
#                    to.astype("M8[ms]").astype(datetime.datetime)))
#            while context["time"].values[-1] <= subset["time"].values[1]:
#                to -= numpy.timedelta64(self.skip_problem_step)
#                subset = subset.sel(time=slice(
#                    from_.astype("M8[ms]").astype(datetime.datetime),
#                    to.astype("M8[ms]").astype(datetime.datetime)))
        # NB: by calibrating the entire subset at once, I get a single set
        # of parameters for the  Rself model.  That may be undesirable.
        # Worse, it means that in most of my files, the dimension for
        # rself_update_time is zero, which is not only undesirable, but
        # also currently triggers the bug at
        # https://github.com/pydata/xarray/issues/1329
        # either I calibrate smaller pieces or I decouple the
        # self-emission model+context evaluation therefrom!
        # OR repeat the same self-emission model parameters but pretend,
        # a.k.a. lie, that they are updated, as a placeholder until I
        # really update them frequently enough such that the aforementioned
        # xarray bug is not triggered
        R_E = self.fcdr.calculate_radiance_all(subset,
            context=context, Rself_model=self.rself)
        cu = {} # should NOT be an expressiondict, I don't want T[1]==T[2] here!
        # FIXME: also receive covariant components
        (uRe, sensRe, compRe, covcomps) = self.fcdr.calc_u_for_variable(
            "R_e", self.fcdr._quantities, self.fcdr._effects, cu,
            return_more=True)
        unc_components = dict(self.fcdr.propagate_uncertainty_components(uRe,
            sensRe, compRe))
        harm_covcomp_component2 = functools.reduce(
            operator.add,
            ((v[0]*v[1]).to(rad_u["si"]**2) for v in covcomps.values()))

#        u_from = xarray.Dataset(dict([(f"u_from_{k!s}", v) for (k, v) in
#                    unc_components.items()]))
        S = self.fcdr.estimate_channel_correlation_matrix(context)
        (lookup_table_BT, LUT_radiance) = self.fcdr.get_BT_to_L_LUT()

        (flags_scanline, flags_channel, flags_minorframe, flags_pixel) = self.fcdr.get_flags(
            subset, context, R_E)
        if ((self.dd.data["time"][-1] - self.dd.data["time"][0]).values.astype("m8[s]").astype(datetime.timedelta) / self.window_size) < 0.9:
            logging.warn("Reduced context available, flagging data")
            flags_scanline |= _fcdr_defs.FlagsScanline.REDUCED_CONTEXT
        
        # "sum" doesn't work because it's initialised with 0 and then the
        # units don't match!  Use reduce with operator.add instead.
        uRe_syst = numpy.sqrt(functools.reduce(operator.add,
            (v[0]**2 for (k, v) in compRe.items() if k is not me.symbols["C_E"])))
        uRe_rand = compRe[me.symbols["C_E"]][0]
        # uncertainty from harmonisation parameters only...
        uRe_harm2 = (functools.reduce(operator.add,
            (v**2
             for (k, v) in unc_components.items()
             if str(k) in ("a_2", "a_3", "a_4"))))
        with numpy.errstate(invalid="raise"):
            try:
                uRe_harm = numpy.sqrt(uRe_harm2 + harm_covcomp_component2)
            except FloatingPointError as e:
                raise fcdr.FCDRError(f"When processing the harmonisation "
                    "parameters, I found that the total contribution of "
                    "harmonisation variance + covariance is negative. "
                    f"Getting FloatingPointError({e.args:s}). Can "
                    "the covariant part really be more negative than the "
                    "variance is positive, or is this a bug?") from e
        # ...which needs to be subtracted from systematic!
        uRe_syst = numpy.sqrt(uRe_syst**2 - uRe_harm**2)

        # Proper propagation goes wrong.  I believe this is due to
        # https://github.com/FIDUCEO/FCDR_HIRS/issues/78
        # Estimate numerically instead.
#        (uTb, sensTb, compTb) = self.fcdr.calc_u_for_variable("T_b",
#            self.fcdr._quantities, self.fcdr._effects, cu,
#            return_more=True)
        tempcode = _fcdr_defs._temp_coding.copy()
        tempcode["scale_factor"] /= 10
        if uRe.dims == ():
            logging.error("Scalar uncertainty?!  Hopefully the lines "
                "immediately above give some hint of what's going on "
                "here!")
            uTb = UADA(0, dims=uRe.dims, coords=uRe.coords, attrs=dict(units="K"))
            u_from = xarray.Dataset(
                {f"u_from_{k!s}": UADA(0, dims=uRe.dims, coords=uRe.coords,
                                       attrs=dict(units="K"),
                                       encoding=tempcode)
                    for (k, v) in unc_components.items()
                    if v.size>1})
            uTb_syst = uTb.copy()
            uTb_rand = uTb.copy()
            uRe_harm = uTb.copy()
        else:
            uTb = self.fcdr.numerically_propagate_ΔL(R_E, uRe)
            u_from = xarray.Dataset(
                {f"u_from_{k!s}": self.fcdr.numerically_propagate_ΔL(R_E, v)
                    for (k, v) in unc_components.items()
                    if v.size>1})
            uTb_syst = self.fcdr.numerically_propagate_ΔL(R_E, uRe_syst)
            uTb_rand = self.fcdr.numerically_propagate_ΔL(R_E, uRe_rand)
            uTb_harm = self.fcdr.numerically_propagate_ΔL(R_E, uRe_harm)
        uTb.name = "u_T_b"

        uRe_rand.encoding = uRe_syst.encoding = uRe_harm.encoding = uRe.encoding = R_E.encoding
        uTb_rand.encoding = uTb_syst.encoding = uTb_harm.encoding = uTb.encoding = self.fcdr._quantities[me.symbols["T_b"]].encoding
        uRe_rand.name = uRe.name + "_random"
        uTb_rand.name = uTb.name + "_random"
        uRe_syst.name = uRe.name + "_nonrandom"
        uTb_syst.name = uTb.name + "_nonrandom"
        uRe_harm.name = uRe.name + "_harm"
        uTb_harm.name = uTb.name + "_harm"
        uc = xarray.Dataset({k: v.magnitude for (k, v) in self.fcdr._effects_by_name.items()})
        qc = xarray.Dataset(self.fcdr._quantities)
        qc = xarray.Dataset(
            {str(k): v for (k, v) in self.fcdr._quantities.items()})
        (SRF_weights, SRF_frequencies) = self.get_srfs()

        (sat_za, sat_aa, sun_za, sun_aa) = self.fcdr.calc_angles(
            subset.sel(time=qc["scanline_earth"]))
        # rename original ones

        # uncertainty scanline coordinate conflicts with subset scanline
        # coordinate, drop the former
        stuff_to_merge = [uc.rename({k: "u_"+k for k in uc.data_vars.keys()}),
                            qc,
                          subset.rename({k.name: "original_"+k.name
                                for k in (sat_za, sat_aa, sun_za, sun_aa)
                                if k.name in subset.data_vars.keys()}),
                            uRe,
                            sat_za, sat_aa, sun_za, sun_aa,
                            uRe_syst, uRe_rand, uRe_harm,
                            uTb_syst, uTb_rand, uTb_harm,
                            S, lookup_table_BT, LUT_radiance,
                            flags_scanline, flags_channel,
                            flags_minorframe, flags_pixel,
                            u_from, SRF_weights, SRF_frequencies]
        ds = xarray.merge(
            [da.drop("scanline").rename(
                {"lat": "lat_earth", "lon": "lon_earth"})
                    if "scanline_earth" in da.coords
                    and "scanline" in da.coords
                    else da for da in stuff_to_merge])
        # NB: when quantities are gathered, offset and slope and others
        # per calibration_cycle are calculated for the entire context
        # period rather than the core dataset period.  I don't want to
        # store the entire context period.  I do this after the merger
        # because it affects both qc and uc.
        ds = ds.isel(
            calibration_cycle=
                (ds["calibration_cycle"] >= subset["time"][0]) &
                (ds["calibration_cycle"] <= subset["time"][-1]))
        # make sure encoding set on coordinates
        for cn in ds.coords.keys():
            if cn in self.fcdr._data_vars_props.keys():
                ds[cn].encoding.update(self.fcdr._data_vars_props[cn][3])
        ds = self.add_attributes(ds)
        ds = common.time_epoch_to(ds, self.epoch)

        # set uncertainty flag when extended uncertainty larger than value
        ds["quality_pixel_bitmask"].values[((2*ds["u_R_Earth"]) > ds["R_e"]).transpose(*ds["quality_pixel_bitmask"].dims).values] |= _fcdr_defs.FlagsChannel.UNCERTAINTY_SUSPICIOUS

        try:
            (Δ_l, Δ_e, R_ci, R_cs, Δ_l_full, Δ_e_full) = metrology.calc_corr_scale_channel(
                self.fcdr._effects, sensRe, ds, flags=self.fcdr._flags,
                robust=True, return_vectors=True, interpolate_lengths=True)
        except fcdr.FCDRError as e:
            logging.error("Failed to calculate correlation length scales: "
                          f"{e.args[0]}")
        else:
            # add those to the ds
            ds["cross_line_radiance_error_correlation_length_scale_structured_effects"] = (("calibrated_channel",), Δ_l.sel(val="popt").values)
            ds["cross_element_radiance_error_correlation_length_scale_structured_effects"] = (("calibrated_channel",), Δ_e.sel(val="popt").values)
            ds["cross_channel_error_correlation_matrix_independent_effects"] = (
                ("calibrated_channel", "calibrated_channel"), R_ci)
            ds["cross_channel_error_correlation_matrix_structured_effects"] = (
                ("calibrated_channel", "calibrated_channel"), R_cs)
            ds["cross_line_radiance_error_correlation_length_average"] = (
                ("delta_scanline_earth", "calibrated_channel"),
                Δ_l_full.values[:self.max_debug_corr_length])
            ds["cross_element_radiance_error_correlation_length_average"] = (
                ("delta_scanpos", "calibrated_channel"), Δ_e_full.values)


        if return_more:
            return (ds, sensRe)
        else:
            return ds

    def get_srfs(self):
        """Return xarray dataset with SRF info
        """

        SRF_weights = xarray.DataArray(
            numpy.full((19, 2751), numpy.nan),
            dims=("channel", "n_frequencies"),
            coords={"channel": numpy.arange(1, 20)},
            name="SRF_weights")

        SRF_frequencies = xarray.full_like(SRF_weights, numpy.nan)
        SRF_frequencies.name = "SRF_frequencies"
        SRF_frequencies.attrs["units"] = "Hz"

        for ch in range(1, 20):
            srf = self.fcdr.srfs[ch-1]
            f = srf.frequency
            W = srf.W
            SRF_frequencies.loc[{"channel": ch}][:f.size] = f
            SRF_weights.loc[{"channel": ch}][:f.size] = W

        return (SRF_weights, SRF_frequencies)

    def add_attributes(self, ds):
        """Add attributes to piece.

        Some attributes must only be added later, in
        self.add_orbit_info_to_piece, because information may become
        incorrect if the piece is split (such as start time or granules
        covered).
        """
        #pr = subprocess.run(["pip", "freeze"], stdout=subprocess.PIPE)
        ds.attrs.update(
            author="Gerrit Holl and the FIDUCEO team",
            email="fiduceo-coordinator@lists.reading.ac.uk",
            title="HIRS FCDR",
            satellite=self.satname,
            url="http://www.fiduceo.eu/",
            #verbose_version_info=pr.stdout.decode("utf-8"),
            fcdr_software_version=pkg_resources.get_distribution("FCDR_HIRS").version,
            institution="University of Reading",
            data_version=self.data_version,
            WARNING=effects.WARNING,
            history = "Produced from L1B on {:%Y-%m-%dT%H:%M:%SZ}".format(
                datetime.datetime.utcnow())
            )
        return ds

    def add_orbit_info_to_piece(self, piece):
        """Add orbital information to piece

        This should be done after calling self.fragmentate because it
        tells about the time coverage.
        """
        at_start = self.fcdr.find_most_recent_granule_before(
            piece["time"][0].values.astype("M8[s]").astype(datetime.datetime)).stem
        at_end = self.fcdr.find_most_recent_granule_before(
            piece["time"][-1].values.astype("M8[s]").astype(datetime.datetime)).stem
        piece.attrs.update(
            orbit_start_time=piece["time"][0].values.astype("M8[ms]").astype(datetime.datetime).isoformat(),
            orbit_end_time=piece["time"][-1].values.astype("M8[ms]").astype(datetime.datetime).isoformat(),
            orbit_start_granule=at_start,
            orbit_end_granule=at_end,
        )
        return piece

    def store_piece(self, piece):
        # FIXME: concatenate when appropriate
        for mode in self.modes:
            fn = self.get_filename_for_piece(piece, fcdr_type=mode)
            fn.parent.mkdir(exist_ok=True, parents=True)
            logging.info("Storing to {!s}".format(fn))
            getattr(self, "store_piece_{:s}".format(mode))(piece, fn)

    def store_piece_debug(self, piece, fn):
        piece.attrs["full_info"] = self.info
        piece.to_netcdf(str(fn))

    def store_piece_easy(self, piece, fn):
        piece_easy = self.debug2easy(piece)
        piece_easy.attrs["institution"] = "University of Reading"
        piece_easy.attrs["title"] = "HIRS Easy FCDR"
        # already included with add_attributes
#        piece_easy.attrs["warning"] = ("TRIAL VERSION, DO NOT USE UNDER "
#            "ANY CIRCUMSTANCES FOR ANY PURPOSE EVER")
        piece_easy.attrs["source"] = ("Produced with HIRS_FCDR code, "
            "version {!s}".format(
                pkg_resources.get_distribution("FCDR_HIRS").version))
        piece_easy.attrs["history"] = "Produced on {:%Y-%m-%dT%H:%M:%SZ}.".format(
            datetime.datetime.utcnow()) + "\n" + VERSION_HISTORY_EASY
        piece_easy.attrs["references"] = "In preparation"
        piece_easy.attrs["url"] = "http://www.fiduceo.eu"
        piece_easy.attrs["author"] = "Gerrit Holl and the FIDUCEO team"
        piece_easy.attrs["contact"] = "fiduceo-coordinator@lists.reading.ac.uk"
        piece_easy.attrs["comment"] = "Early version.  Please note warning."
        piece_easy.attrs["typical_structured_correlation_scale"] = "40 scanlines"
        try:
            fiduceo.fcdr.writer.fcdr_writer.FCDRWriter.write(
                piece_easy,
                fn,
                overwrite=True)
        except FileExistsError as e:
            logging.info("Already exists: {!s}".format(e.args[0]))
        except ValueError as e:
            if "chunksize" in e.args[0]:
                logging.error(f"ERROR! Cannot store due to ValueError: {e.args!s} "
                    "See https://github.com/FIDUCEO/FCDRTools/issues/15")
            else:
                raise

    def store_piece_none(self, piece, fn):
        """Do not store anything!"""
        logging.info("You told me to write nothing.  I will not write "
            f"anything to {fn!s} nor to anywhere else (but I have "
            "accidentally created a directory already, oops).")

    map_dims_debug_to_easy = {
        "scanline_earth": "y",
        "time": "y",
        "scanpos": "x",
        "channel": "rad_channel",
        "calibrated_channel": "channel",
        "delta_scanline_earth": "delta_y",
        "delta_scanpos": "delta_x",
        }

    map_names_debug_to_easy = {
        "latitude": "lat",
        "longitude": "lon",
        "bt": "T_b",
        "satellite_zenith_angle": "platform_zenith_angle",
        "satellite_azimuth_angle": "local_azimuth_angle",
        }
    def debug2easy(self, piece):
        """Convert debug FCDR to easy FCDR

        Follows Tom Blocks format
        """

        N = piece["scanline_earth"].size
        easy = fiduceo.fcdr.writer.fcdr_writer.FCDRWriter.createTemplateEasy(
            f"HIRS{self.fcdr.version:d}", N,
            srf_size=piece.dims["n_frequencies"],
            lut_size=piece.dims["lut_size"],
            corr_dx=self.fcdr.n_perline,
            corr_dy=N//2)
        t_earth = piece["scanline_earth"]
        t_earth_i = piece.get_index("scanline_earth")
        mpd = self.map_dims_debug_to_easy

        newcont = dict(
            time=t_earth,
            latitude=piece["lat"].sel(time=t_earth),
            longitude=piece["lon"].sel(time=t_earth),
            bt=UADA(piece["T_b"]),
            satellite_zenith_angle=piece["platform_zenith_angle"],
            satellite_azimuth_angle=piece["platform_azimuth_angle"],
            solar_zenith_angle=piece["solar_zenith_angle"],
            solar_azimuth_angle=piece["solar_azimuth_angle"],
            scanline=piece["scanline"].sel(time=t_earth),
#            quality_scanline_bitmask = piece["quality_scanline_bitmask"],
#            quality_channel_bitmask = piece["quality_channel_bitmask"],
            u_independent=piece["u_T_b_random"],
            u_structured=piece["u_T_b_nonrandom"],
            u_common=piece["u_T_b_harm"], # NB 2018-08-02: not in TBs writer yet!
            lookup_table_BT=piece["lookup_table_BT"],
            lookup_table_radiance=piece["lookup_table_radiance"],
            scanline_origl1b=piece["scanline_number"].sel(time=t_earth))
        try:
            newcont.update(**dict(
#                cross_line_radiance_error_correlation_length_scale_structured_effects=piece["cross_line_radiance_error_correlation_length_scale_structured_effects"],
#                cross_element_radiance_error_correlation_length_scale_structured_effects=piece["cross_element_radiance_error_correlation_length_scale_structured_effects"],
                channel_correlation_matrix_independent=piece["cross_channel_error_correlation_matrix_independent_effects"],
                channel_correlation_matrix_structured=piece["cross_channel_error_correlation_matrix_structured_effects"],
                cross_element_correlation_coefficients=piece["cross_element_radiance_error_correlation_length_average"],
                cross_line_correlation_coefficients=piece["cross_line_radiance_error_correlation_length_average"].isel(delta_scanline_earth=slice(easy.dims["delta_y"])),
                    ))
        except KeyError as e:
            # assuming they're missing because their calculation failed
            logging.warning("Correlation length scales missing in debug FCDR." 
                "See above, I guess their calculation failed. "
                f"For the record: {e.args[0]}")

#        if self.fcdr.version >= 3:
#            newcont.update(
#                linqualflags=piece["line_quality_flags"].sel(time=t_earth),
#                chqualflags=piece["channel_quality_flags"].sel(
#                    time=t_earth,
#                    channel=slice(19)).rename(
#                    {"channel": "calibrated_channel"}), # TODO: #75
#                mnfrqualflags=piece["minorframe_quality_flags"].sel(
#                    time=t_earth,
#                    minor_frame=slice(56)).rename(
#                    {"minor_frame": "scanpos"}), # see #73 (TODO: #74, #97)
#                satellite_azimuth_angle=piece["local_azimuth_angle"].sel(time=t_earth),
#                solar_zenith_angle=piece["solar_zenith_angle"].sel(time=t_earth),
#                )
#        else:
#            easy = easy.drop(easy.data_vars.keys() &
#                {"solar_zenith_angle", "solar_azimuth_angle",
#                 "satellite_azimuth_angle"})
            #newcont["local_azimuth_angle"] = None
            #newcont["solar_zenith_angle"] = None
        # if we are going to respect Toms template this should contain
        # v.astype(easy[k].dtype)
        transfer = {k: ([mpd.get(d,d) for d in v.dims],
                    v)
                for (k, v) in newcont.items()}
        easy = easy.assign(**transfer)
        self.debug2easy_flags(easy, piece)
        
        # add orig_l1b
        src_filenames = pandas.unique(piece["filename"].sel(time=t_earth))
        easy["scanline_map_to_origl1bfile"][:] = [src_filenames.tolist().index(fn) for fn in piece["filename"].sel(time=t_earth)]
        easy.attrs["source"] = src_filenames

        easy = easy.assign_coords(
            x=numpy.arange(1, 57),
            #y=easy["scanline"],
            y=numpy.arange(easy.dims["y"]),
            channel=numpy.arange(1, 20))

        for k in easy.variables.keys():
            # see
            # https://github.com/FIDUCEO/FCDR_HIRS/issues/215#issuecomment-393879944
            # for why the following line is deactivated (commented out)
            #easy[k].encoding = _fcdr_defs.FCDR_easy_encodings[k]
            # when any of those keys is set in
            # both in attrs and encoding, writing to disk fails with:
            # ValueError: Failed hard to prevent overwriting key '_FillValue'
            for var in {"_FillValue", "add_offset", "encoding", "scale_factor"}:
                if (var in easy[k].encoding.keys() and
                    var in easy[k].attrs.keys()):
                    if easy[k].encoding[var] != easy[k].attrs[var]:
                        warnings.warn("Easy FCDR {:s} attribute {:s} has value {!s} "
                        "in attributes.  Using {!s} from encoding instead!".format(
                            k, var, easy[k].attrs[var],
                            easy[k].encoding[var]))
                    del easy[k].attrs[var]

        # .assign does not copy variable attributes or encoding...
        for (k, v) in transfer.items():
            # but we don't want to overwrite attributes already there
            for (kk, vv) in v[1].attrs.items():
                if kk not in easy[k].attrs:
                    easy[k].attrs[kk] = vv
            for (kk, vv) in v[1].encoding.items():
                if kk not in easy[k].encoding:
                    easy[k].encoding[kk] = vv

        easy["SRF_weights"][...] = piece["SRF_weights"].sel(channel=range(1, 20), n_frequencies=range(easy.dims["n_wavelengths"]))
        easy["SRF_wavelengths"][...] = UADA(
            piece["SRF_frequencies"].sel(channel=range(1, 20), n_frequencies=range(easy.dims["n_wavelengths"]))).to(
                "um", "sp")
        easy.attrs.update(piece.attrs)

        for (k, v) in _fcdr_defs.FCDR_extra_attrs.items():
            easy[k].attrs.update(v)
            
        easy = easy.drop(("scanline",)) # see #94
        return easy

    def debug2easy_flags(self, easy, piece):
        """Copy over flags from piece to easy

        Arguments:
            easy
                as returned by TBs code
            piece
                as processed in this code too (debug)

        No return value.
        """

        # prepare enum.IntFlag for easier access
        ef = {}
        for f in ("quality_pixel_bitmask", "data_quality_bitmask",
                  "quality_scanline_bitmask", "quality_channel_bitmask"):
            ef[f] = enum.IntFlag(
                    f,
                    dict(zip(easy[f].flag_meanings.split(),
                             easy[f].flag_masks.split(', '))))

        dfs = _fcdr_defs.FlagsScanline
        dfc = _fcdr_defs.FlagsChannel
        dfmf = _fcdr_defs.FlagsMinorFrame
        dfp = _fcdr_defs.FlagsPixel
        
        efqpb = ef["quality_pixel_bitmask"]
        efdqb = ef["data_quality_bitmask"]
        efqsb = ef["quality_scanline_bitmask"]
        efqcb = ef["quality_channel_bitmask"]

        dpb = piece["quality_pixel_bitmask"]
        dsb = piece["quality_scanline_bitmask"]
        dcb = piece["quality_channel_bitmask"]
        dmfb = piece["quality_minorframe_bitmask"]
        
        eqpb = easy["quality_pixel_bitmask"]
        edqb = easy["data_quality_bitmask"]
        eqsb = easy["quality_scanline_bitmask"]
        eqcb = easy["quality_channel_bitmask"]


        # placeholders with noidx to be filled in
        noidx = numpy.zeros(0, dtype="u2")

        # efqpb (this is shared between sensors)
        eqpb.values[(dpb&dfp.DO_NOT_USE).all("calibrated_channel").values] \
            |= efqpb.invalid
        eqpb.values[(dpb&dfp.DO_NOT_USE).any("calibrated_channel").values] \
            |= efqpb.incomplete_channel_data
        eqpb[noidx] |= efqpb.invalid_geoloc # FIXME: expand from dsb&dfs.SUSPECT_GEO 
        eqpb[noidx] |= efqpb.invalid_input # FIXME: ?
        eqpb[noidx] |= efqpb.invalid_time # FIXME: expand from dsb&dfs.SUSPECT_TIME
        eqpb[noidx] |= efqpb.padded_data # FIXME: ?
        eqpb[noidx] |= efqpb.sensor_error # FIXME: ?
        eqpb[noidx] |= efqpb.use_with_caution # set by writer

        # efdqb
        edqb.values[(dpb&dfp.OUTLIER_NOS).any("calibrated_channel").values] \
            |= efdqb.outlier_nos
        edqb.values[((dmfb.sel(minor_frame=slice(56)).rename(minor_frame='x')&dfmf.SUSPECT_MIRROR)!=0).values] |= efdqb.suspect_mirror
        edqb.values[(dpb&dfp.UNCERTAINTY_TOO_LARGE).any("calibrated_channel").values] |= efdqb.uncertainty_too_large

        # efqsb
        eqsb.values[((dsb&dfs.DO_NOT_USE)!=0).values] |= efqsb.do_not_use_scan
        eqsb.values[((dsb&dfs.BAD_TEMP_NO_RSELF)!=0).values] |= efqsb.bad_temp_no_rself
        eqsb.values[((dsb&dfs.REDUCED_CONTEXT)!=0).values] |= efqsb.reduced_context
        eqsb.values[((dsb&dfs.SUSPECT_GEO)!=0).values] |= efqsb.suspect_geo
        eqsb.values[((dsb&dfs.SUSPECT_TIME)!=0).values] |= efqsb.suspect_time
        
        # MISSING:
        #
        # SUSPECT_MIRROR_ANY (but I have suspect_mirror)
        # UNCERTAINTY_SUSPICIOUS

        # efqcb
        eqcb.values[((dcb&dfc.DO_NOT_USE)!=0).values] |= efqcb.do_not_use
        eqcb.values[((dcb&dfc.CALIBRATION_IMPOSSIBLE)!=0).values] |= efqcb.calibration_impossible
        # from quality_scanline_bitmask into quality_channel_bitmask
        eqcb.loc[{"y":(dsb.rename(scanline_earth="y")&dfs.SUSPECT_CALIB)!=0}] |= efqcb.calibration_suspect
        eqcb.values[((dcb&dfc.SELF_EMISSION_FAILS)!=0).values] |= efqcb.self_emission_fails
        eqcb.values[((dcb&dfc.UNCERTAINTY_SUSPICIOUS)!=0).values] |= efqcb.uncertainty_suspicious

        # summarise per line
        eqcb.values[(dpb&dfp.UNCERTAINTY_TOO_LARGE).any("scanpos").values] |=  \
            efqcb.uncertainty_suspicious

#        easy["quality_channel_bitmask"].values[(piece["quality_pixel_bitmask"] &
#            _fcdr_defs.FlagsPixel.UNCERTAINTY_TOO_LARGE).any("scanpos").values] |= \
#            _fcdr_defs.FlagsChannel.UNCERTAINTY_SUSPICIOUS
        
#        raise NotImplementedError("Not implemented yet!")

    _i = 0
    def get_filename_for_piece(self, piece, fcdr_type):
        # instead of using datetime-formatting codes directly, pass all
        # seperately so that the same format can be more easily used in
        # reading mode as well
        from_time=piece["time"][0].values.astype("M8[s]").astype(datetime.datetime)
        to_time=piece["time"][-1].values.astype("M8[s]").astype(datetime.datetime)
        fn = self.fcdr.find_granule_for_time(
#        fn = "/".join((self.basedir, self.subdir, self.filename)).format(
            satname=self.satname,
            year=from_time.year,
            month=from_time.month,
            day=from_time.day,
            hour=from_time.hour,
            minute=from_time.minute,
            second=from_time.second,
            year_end=to_time.year,
            month_end=to_time.month,
            day_end=to_time.day,
            hour_end=to_time.hour,
            minute_end=to_time.minute,
            second_end=to_time.second,
            data_version=self.data_version,
            fcdr_type=fcdr_type,
            mode="write")
        return pathlib.Path(fn)
#        raise NotImplementedError()
            

def main():
    warnings.filterwarnings("error", category=numpy.VisibleDeprecationWarning)
    warnings.filterwarnings("error", 
        message="iteration over an xarray.Dataset will change",
        category=FutureWarning)
#    warnings.filterwarnings("error",
#        message="invalid value encountered in log", category=RuntimeWarning)
    if p.days == 0:
        fgen = FCDRGenerator(p.satname,
            datetime.datetime.strptime(p.from_date, p.datefmt),
            datetime.datetime.strptime(p.to_date, p.datefmt),
            p.modes)
        fgen.process()
    else:
        dates = pandas.date_range(p.from_date, p.to_date, freq="MS")
        for d in dates:
            fgen = FCDRGenerator(p.satname,
                d.to_pydatetime(),
                d.to_pydatetime() + datetime.timedelta(days=p.days),
                p.modes)
            fgen.process()

