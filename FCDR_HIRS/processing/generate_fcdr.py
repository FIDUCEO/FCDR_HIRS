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
"""

VERSION_HISTORY_EASY="""Generated from L1B data using FCDR_HIRS.  See
release notes for details on versions used."""


import sys
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
        nargs="+", choices=["easy", "debug"],
        help="What FCDR(s) to write?")

    return parser.parse_args()
p = parse_cmdline()

import logging
logging.basicConfig(
    format=("%(levelname)-8s %(asctime)s %(module)s.%(funcName)s:"
            "%(lineno)s: %(message)s"),
    filename=p.log,
    level=logging.DEBUG if p.verbose else logging.INFO)

import pathlib
import datetime

import numpy
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
    data_version = "0.5pre"
    # FIXME: do we have a filename convention?
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
        self.start_date = start_date
        self.end_date = end_date
        self.dd = typhon.datasets.dataset.DatasetDeque(
            self.fcdr, self.window_size, start_date)

        self.rself = models.RSelf(self.fcdr)
        self.modes = modes

    def process(self, start=None, end_time=None):
        """Generate FCDR for indicated period
        """
        start = start or self.start_date
        end_time = end_time or self.end_date
        logging.info("Now processing FCDR for {self.satname:s} HIRS, "
            "{start:%Y-%m-%d %H:%M:%S} – {end_time:%Y-%m-%d %H:%M:%S}. ".format(
            self=self, start=start, end_time=end_time))
        self.dd.reset(start)
        while self.dd.center_time < end_time:
            self.dd.move(self.step_size)
            try:
                self.make_and_store_piece(self.dd.center_time - self.segment_size,
                    self.dd.center_time)
            except fcdr.FCDRError as e:
                warnings.warn("Unable to generate FCDR: {:s}".format(e.args[0]))
        logging.info("Successfully completed, completed successfully.")
        logging.info("Everything seems fine.")
    
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
            for tc in time_coords:
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
            context=self.dd.data, Rself_model=self.rself)
        cu = {}
        (uRe, sensRe, compRe) = self.fcdr.calc_u_for_variable("R_e", self.fcdr._quantities,
            self.fcdr._effects, cu, return_more=True)
        S = self.fcdr.estimate_channel_correlation_matrix(self.dd.data)
        (LUT_BT, LUT_L) = self.fcdr.get_BT_to_L_LUT()
        # "sum" doesn't work because it's initialised with 0 and then the
        # units don't match!  Use reduce with operator.add instead.
        uRe_syst = numpy.sqrt(functools.reduce(operator.add,
            (v[0]**2 for (k, v) in compRe.items() if k is not me.symbols["C_E"])))
        uRe_rand = compRe[me.symbols["C_E"]][0]

        # This goes wrong.  I believe this is due to
        # https://github.com/FIDUCEO/FCDR_HIRS/issues/78
        # Estimate numerically instead.
#        (uTb, sensTb, compTb) = self.fcdr.calc_u_for_variable("T_b",
#            self.fcdr._quantities, self.fcdr._effects, cu,
#            return_more=True)
        uTb = self.fcdr.numerically_propagate_ΔL(R_E, uRe)
        uTb.name = "u_T_b"
        
        # this is approximate, not accurate, but will do for now
        uTb_syst = uRe_syst/(uRe_syst+uRe_rand) * uTb
        uTb_rand = uRe_rand/(uRe_syst+uRe_rand) * uTb

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
        ds = xarray.merge(
            [uc.rename({k: "u_"+k for k in uc.data_vars.keys()}
                            ).drop("scanline"), qc, subset, uRe,
                            uRe_syst, uRe_rand,
                            uTb_syst, uTb_rand,
                            S, LUT_BT, LUT_L])
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
            ds[cn].encoding.update(self.fcdr._data_vars_props[cn][3])
        ds = self.add_attributes(ds)
        for k in [k for (k, v) in ds.items() if v.dtype.kind.startswith("M")]:
            ds[k].encoding["units"] = "seconds since {:%Y-%m-%d %H:%M:%S}".format(self.epoch)
            ds[k].encoding["add_offset"] = (
                ds["time"][0].values.astype("M8[ms]").astype(datetime.datetime)
                - self.epoch).total_seconds()
        return ds

    def add_attributes(self, ds):
        """Add attributes to piece.

        Some attributes must only be added later, in
        self.add_orbit_info_to_piece, because information may become
        incorrect if the piece is split (such as start time or granules
        covered).
        """
        pr = subprocess.run(["pip", "freeze"], stdout=subprocess.PIPE)
        ds.attrs.update(
            author="Gerrit Holl",
            email="g.holl@reading.ac.uk",
            title="HIRS FCDR",
            satellite=self.satname,
            url="http://www.fiduceo.eu/",
            verbose_version_info=pr.stdout.decode("utf-8"),
            institution="University of Reading",
            data_version=self.data_version,
            WARNING=effects.WARNING,
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
        piece_easy.attrs["warning"] = ("TRIAL VERSION, DO NOT USE UNDER "
            "ANY CIRCUMSTANCES FOR ANY PURPOSE EVER")
        piece_easy.attrs["source"] = "Produced with HIRS_FCDR code"
        piece_easy.attrs["history"] = "Produced on {:%Y-%m-%dT%H:%M:%SZ}".format(
            datetime.datetime.utcnow()) + VERSION_HISTORY_EASY
        piece_easy.attrs["references"] = "In preparation"
        piece_easy.attrs["url"] = "http://www.fiduceo.eu"
        piece_easy.attrs["author"] = "Gerrit Holl <g.holl@reading.ac.uk>"
        piece_easy.attrs["comment"] = "Not for the faint of heart.  See warning!"
        piece_easy.attrs["typical_nonrandom_correlation_scale"] = "40 scanlines"
        try:
            # Don't use this one for now, because it doesn't apply scaling
            # and ofsets and such
            #writer.fcdr_writer.FCDRWriter.write(piece_easy, str(fn))
            piece_easy.to_netcdf(str(fn))
        except FileExistsError as e:
            logging.info("Already exists: {!s}".format(e.args[0]))

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
            "HIRS", N)
        # Remove following line as soon as Toms writer no longer includes
        # them in the template
        easy = easy.drop(("scnlintime", "scnlinf", "scantype"))
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
            qualind=piece["quality_flags"].sel(time=t_earth),
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
                linqualflags=piece["line_quality_flags"].sel(time=t_earth),
                chqualflags=piece["channel_quality_flags"].sel(
                    time=t_earth,
                    channel=slice(19)).rename(
                    {"channel": "calibrated_channel"}), # TODO: #75
                mnfrqualflags=piece["minorframe_quality_flags"].sel(
                    time=t_earth,
                    minor_frame=slice(56)).rename(
                    {"minor_frame": "scanpos"}), # see #73 (TODO: #74, #97)
                satellite_azimuth_angle=piece["local_azimuth_angle"].sel(time=t_earth),
                solar_zenith_angle=piece["solar_zenith_angle"].sel(time=t_earth),
                )
        else:
            easy = easy.drop(("linqualflags",
                              "chqualflags",
                              "mnfrqualflags",
                              "solar_zenith_angle",
                              "solar_azimuth_angle",
                              "satellite_azimuth_angle"))
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
    fgen = FCDRGenerator(p.satname,
        datetime.datetime.strptime(p.from_date, p.datefmt),
        datetime.datetime.strptime(p.to_date, p.datefmt),
        p.modes)
    fgen.process()
