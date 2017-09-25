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
from typhon.physics.units.common import radiance_units as rad_u
from typhon.physics.units.tools import UnitsAwareDataArray as UADA
from .. import fcdr
from .. import models
from .. import effects
from .. import measurement_equation as me
from .. import _fcdr_defs

import fiduceo.fcdr.writer.fcdr_writer

class FCDRGenerator:
    # for now, step_size should be smaller than segment_size and I will
    # only store whole orbits within each segment
    epoch = datetime.datetime(1970, 1, 1, 0, 0, 0)
    window_size = datetime.timedelta(hours=24)
    segment_size = datetime.timedelta(hours=6)
    step_size = datetime.timedelta(hours=4)
    skip_problem_step = datetime.timedelta(seconds=900)
    data_version = "0.7"
    # see comment in models.Rself
    rself_temperatures = ["baseplate", "internal_warm_calibration_target",
        "scanmirror", "scanmotor", "secondary_telescope"]
    # 2017-07-14 GH: Use LR again, seems to work better than PDR although
    # I don't know why it should.
    rself_regr = ("LR", {"fit_intercept": True})
    reader_args = {"apply_flags": False, "apply_filter": False}

    # FIXME: use filename convention through FCDRTools, 
    def __init__(self, sat, start_date, end_date, modes):
        logging.info("Preparing to generate FCDR for {sat:s} HIRS, "
            "{start:%Y-%m-%d %H:%M:%S} – {end_time:%Y-%m-%d %H:%M:%S}. "
            "Software:".format(
            sat=sat, start=start_date, end_time=end_date))
        pr = subprocess.run(["pip", "freeze"], stdout=subprocess.PIPE)
        info = pr.stdout.decode("utf-8")
        logging.info(info)
        self.satname = sat
        self.fcdr = fcdr.which_hirs_fcdr(sat, read="L1B")
        self.fcdr.my_pseudo_fields.clear() # suppress pseudo fields radiance_fid, bt_fid here
        self.start_date = start_date
        self.end_date = end_date
        self.dd = typhon.datasets.dataset.DatasetDeque(
            self.fcdr, self.window_size, start_date,
            reader_args=self.reader_args)

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
            self.dd.reset(start, reader_args=self.reader_args)
        except typhon.datasets.dataset.DataFileError as e:
            warnings.warn("Unable to generate FCDR: {:s}".format(e.args[0]))
        while self.dd.center_time < end_time:
            try:
                self.dd.move(self.step_size,
                    reader_args=self.reader_args)
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

        piece = self.get_piece(from_, to)
#        self.store_piece(piece)
        for piece in self.fragmentate(piece):
            piece = self.add_orbit_info_to_piece(piece)
            self.store_piece(piece)

    def get_piece(self, from_, to):
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
        cu = {}
        (uRe, sensRe, compRe) = self.fcdr.calc_u_for_variable(
            "R_e", self.fcdr._quantities, self.fcdr._effects, cu,
            return_more=True)
        unc_components = dict(self.fcdr.propagate_uncertainty_components(uRe,
            sensRe, compRe))
#        u_from = xarray.Dataset(dict([(f"u_from_{k!s}", v) for (k, v) in
#                    unc_components.items()]))
        S = self.fcdr.estimate_channel_correlation_matrix(context)
        (LUT_BT, LUT_L) = self.fcdr.get_BT_to_L_LUT()

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

        # Proper propagation goes wrong.  I believe this is due to
        # https://github.com/FIDUCEO/FCDR_HIRS/issues/78
        # Estimate numerically instead.
#        (uTb, sensTb, compTb) = self.fcdr.calc_u_for_variable("T_b",
#            self.fcdr._quantities, self.fcdr._effects, cu,
#            return_more=True)
        if uRe.dims == ():
            logging.error("Scalar uncertainty?!  Hopefully the lines "
                "immediately above give some hint of what's going on "
                "here!")
            uTb = UADA(0, dims=uRe.dims, coords=uRe.coords, attrs=dict(units="K"))
            u_from = xarray.Dataset(
                {f"u_from_{k!s}": UADA(0, dims=uRe.dims, coords=uRe.coords,
                                       attrs=dict(units="K"))
                    for (k, v) in unc_components.items()
                    if v.size>1})
            uTb_syst = uTb_rand = uTb
        else:
            uTb = self.fcdr.numerically_propagate_ΔL(R_E, uRe)
            u_from = xarray.Dataset(
                {f"u_from_{k!s}": self.fcdr.numerically_propagate_ΔL(R_E, v).astype("f4")
                    for (k, v) in unc_components.items()
                    if v.size>1})
            uTb_syst = self.fcdr.numerically_propagate_ΔL(R_E, uRe_syst)
            uTb_rand = self.fcdr.numerically_propagate_ΔL(R_E, uRe_rand)
        uTb.name = "u_T_b"
        

        uRe_rand.encoding = uRe_syst.encoding = uRe.encoding = R_E.encoding
        uTb_rand.encoding = uTb_syst.encoding = uTb.encoding = self.fcdr._quantities[me.symbols["T_b"]].encoding
        uRe_rand.name = uRe.name + "_random"
        uTb_rand.name = uTb.name + "_random"
        uRe_syst.name = uRe.name + "_nonrandom"
        uTb_syst.name = uTb.name + "_nonrandom"
        uc = xarray.Dataset({k: v.magnitude for (k, v) in self.fcdr._effects_by_name.items()})
        qc = xarray.Dataset(self.fcdr._quantities)
        qc = xarray.Dataset(
            {str(k): v for (k, v) in self.fcdr._quantities.items()})
        # uncertainty scanline coordinate conflicts with subset scanline
        # coordinate, drop the former
        stuff_to_merge = [uc.rename({k: "u_"+k for k in uc.data_vars.keys()}),
                            qc, subset, uRe,
                            uRe_syst, uRe_rand,
                            uTb_syst, uTb_rand,
                            S, LUT_BT, LUT_L,
                            flags_scanline, flags_channel,
                            flags_minorframe, flags_pixel,
                            u_from]
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

        return ds

    def add_attributes(self, ds):
        """Add attributes to piece.

        Some attributes must only be added later, in
        self.add_orbit_info_to_piece, because information may become
        incorrect if the piece is split (such as start time or granules
        covered).
        """
        #pr = subprocess.run(["pip", "freeze"], stdout=subprocess.PIPE)
        ds.attrs.update(
            author="Gerrit Holl",
            email="g.holl@reading.ac.uk",
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
        piece.to_netcdf(str(fn))

    def store_piece_easy(self, piece, fn):
        piece_easy = self.debug2easy(piece)
        piece_easy.attrs["institution"] = "Reading University"
        piece_easy.attrs["title"] = "HIRS Easy FCDR"
        # already included with add_attributes
#        piece_easy.attrs["warning"] = ("TRIAL VERSION, DO NOT USE UNDER "
#            "ANY CIRCUMSTANCES FOR ANY PURPOSE EVER")
        piece_easy.attrs["source"] = ("Produced with HIRS_FCDR code, "
            "version {!s}".format(
                pkg_resources.get_distribution("FCDR_HIRS").version))
        piece_easy.attrs["history"] = "Produced on {:%Y-%m-%dT%H:%M:%SZ}".format(
            datetime.datetime.utcnow()) + VERSION_HISTORY_EASY
        piece_easy.attrs["references"] = "In preparation"
        piece_easy.attrs["url"] = "http://www.fiduceo.eu"
        piece_easy.attrs["author"] = "Gerrit Holl <g.holl@reading.ac.uk>"
        piece_easy.attrs["comment"] = "Early version.  Please note warning."
        piece_easy.attrs["typical_nonrandom_correlation_scale"] = "40 scanlines"
        try:
            # Don't use this one for now, because it doesn't apply scaling
            # and ofsets and such
            #writer.fcdr_writer.FCDRWriter.write(piece_easy, str(fn))
            piece_easy.to_netcdf(str(fn))
        except FileExistsError as e:
            logging.info("Already exists: {!s}".format(e.args[0]))

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
            f"HIRS{self.fcdr.version:d}", N)
        # Remove following line as soon as Toms writer no longer includes
        # them in the template
        easy = easy.drop(easy.data_vars.keys() &
            {"scnlintime", "scnlinf", "scantype", "qualind",
             "linqualflags", "chqualflags", "mnfrqualflags"})
        t_earth = piece["scanline_earth"]
        t_earth_i = piece.get_index("scanline_earth")
        mpd = self.map_dims_debug_to_easy

        newcont = dict(
            time=t_earth,
            latitude=piece["lat"].sel(time=t_earth),
            longitude=piece["lon"].sel(time=t_earth),
            #c_earth=piece["counts"].sel(time=t_earth),
            bt=UADA(piece["T_b"]),
            satellite_zenith_angle=piece["platform_zenith_angle"].sel(time=t_earth),
            scanline=piece["scanline"].sel(time=t_earth),
##            scnlintime=UADA((
#                t_earth_i.hour*24*60 +
#                t_earth_i.minute+60+t_earth_i.second +
#                t_earth_i.microsecond/1e6)*1e3,
#                dims=("time",),
#                coords={"time": t_earth.values}),
            quality_scanline_bitmask = piece["quality_scanline_bitmask"],
            quality_channel_bitmask = piece["quality_channel_bitmask"],
#            qualind=piece["quality_flags"].sel(time=t_earth),
            u_random=piece["u_T_b_random"],
            u_non_random=piece["u_T_b_nonrandom"],
            channel_correlation_matrix=piece["channel_correlation_matrix"].sel(
                channel=slice(19)).rename({"channel": "calibrated_channel"}),
            LUT_BT=piece["LUT_BT"],
            LUT_radiance=piece["LUT_radiance"]
                )
#            u_random=UADA(piece["u_R_Earth_random"]).to(rad_u["ir"], "radiance"),
#            u_non_random=UADA(piece["u_R_Earth_nonrandom"]).to(rad_u["ir"], "radiance"))

        if self.fcdr.version >= 3:
            newcont.update(
#                linqualflags=piece["line_quality_flags"].sel(time=t_earth),
#                chqualflags=piece["channel_quality_flags"].sel(
#                    time=t_earth,
#                    channel=slice(19)).rename(
#                    {"channel": "calibrated_channel"}), # TODO: #75
#                mnfrqualflags=piece["minorframe_quality_flags"].sel(
#                    time=t_earth,
#                    minor_frame=slice(56)).rename(
#                    {"minor_frame": "scanpos"}), # see #73 (TODO: #74, #97)
                satellite_azimuth_angle=piece["local_azimuth_angle"].sel(time=t_earth),
                solar_zenith_angle=piece["solar_zenith_angle"].sel(time=t_earth),
                )
        else:
            easy = easy.drop(easy.data_vars.keys() &
                {"solar_zenith_angle", "solar_azimuth_angle",
                 "satellite_azimuth_angle"})
            #newcont["local_azimuth_angle"] = None
            #newcont["solar_zenith_angle"] = None
        # if we are going to respect Toms template this should contain
        # v.astype(easy[k].dtype)
        transfer = {k: ([mpd.get(d,d) for d in v.dims],
                    v)
                for (k, v) in newcont.items()}
        easy = easy.assign(**transfer)
        easy = easy.assign_coords(
            x=numpy.arange(1, 57),
            y=easy["scanline"],
            channel=numpy.arange(1, 20))

        for k in easy.keys():
            easy[k].encoding = _fcdr_defs.FCDR_easy_encodings[k]
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

        easy["quality_channel_bitmask"].values[(piece["quality_pixel_bitmask"] &
            _fcdr_defs.FlagsPixel.UNCERTAINTY_TOO_LARGE).any("scanpos")] |= \
            _fcdr_defs.FlagsChannel.UNCERTAINTY_SUSPICIOUS
                
        easy.attrs.update(piece.attrs)

        for (k, v) in _fcdr_defs.FCDR_extra_attrs.items():
            easy[k].attrs.update(v)
            
        easy = easy.drop(("scanline",)) # see #94
        return easy

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
            second=to_time.second,
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

