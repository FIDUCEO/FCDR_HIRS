"""Generate FCDR for satellite and period

Generate HIRS FCDR for a particular satellite and period.

"""

import sys
from .. import common
import argparse
import subprocess
import warnings

def parse_cmdline():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser = common.add_to_argparse(parser,
        include_period=True,
        include_sat=1,
        include_channels=False,
        include_temperatures=False)

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
from .. import fcdr
from .. import models
from .. import effects


class FCDRGenerator:
    # for now, step_size should be smaller than segment_size and I will
    # only store whole orbits within each segment
    window_size = datetime.timedelta(hours=24)
    segment_size = datetime.timedelta(hours=6)
    step_size = datetime.timedelta(hours=4)
    data_version = "0.1"
    # FIXME: do we have a filename convention?
    # FIXME: this should be incorporated in the general HomemadeDataset
    # class
#    basedir = "/group_workspaces/cems2/fiduceo/Data/FCDR/HIRS/pre-β/testing"
#    subdir = "{from_time:%Y/%m/%d}"
#    filename = "HIRS_FCDR_sketch_{satname:s}_{from_time:%Y%m%d%H%M}_{to_time:%H%M}.nc"
    def __init__(self, sat, start_date, end_date):
        self.satname = sat
        self.fcdr = fcdr.which_hirs_fcdr(sat, read="L1B")
        self.start_date = start_date
        self.end_date = end_date
        self.dd = typhon.datasets.dataset.DatasetDeque(
            self.fcdr, self.window_size, start_date)

        self.rself = models.RSelf(self.fcdr)

    def process(self, start=None, end_time=None):
        """Generate FCDR for indicated period
        """
        start = start or self.start_date
        end_time = end_time or self.end_date
        self.dd.reset(start)
        while self.dd.center_time < end_time:
            self.dd.move(self.step_size)
            try:
                self.make_and_store_piece(self.dd.center_time - self.segment_size,
                    self.dd.center_time)
            except fcdr.FCDRError as e:
                warnings.warn("Unable to generate FCDR: {:s}".format(e.args[0]))
    
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
        (u, sens, comp) = self.fcdr.calc_u_for_variable("R_e", self.fcdr._quantities,
            self.fcdr._effects, cu, return_more=True)
        u.encoding = R_E.encoding
        uc = xarray.Dataset({k: v.magnitude for (k, v) in self.fcdr._effects_by_name.items()})
        qc = xarray.Dataset(self.fcdr._quantities)
        qc = xarray.Dataset(
            {str(k): v for (k, v) in self.fcdr._quantities.items()})
        # uncertainty scanline coordinate conflicts with subset scanline
        # coordinate, drop the former
        ds = xarray.merge(
            [uc.rename({k: "u_"+k for k in uc.data_vars.keys()}
                            ).drop("scanline"), qc, subset, u])
        # NB: when quantities are gathered, offset and slope and others
        # per calibration_cycle are calculated for the entire context
        # period rather than the core dataset period.  I don't want to
        # store the entire context period.  I do this after the merger
        # because it affects both qc and uc.
        ds = ds.isel(
            calibration_cycle=
                (ds["calibration_cycle"] >= subset["time"][0]) &
                (ds["calibration_cycle"] <= subset["time"][-1]))
        ds = self.add_attributes(ds)
        return ds

    def add_attributes(self, ds):
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

    def store_piece(self, piece):
        # FIXME: concatenate when appropriate
        fn = self.get_filename_for_piece(piece)
        fn.parent.mkdir(exist_ok=True, parents=True)
        logging.info("Storing to {!s}".format(fn))
        piece.to_netcdf(str(fn))

    _i = 0
    def get_filename_for_piece(self, piece):
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
            year_end=to_time.year,
            month_end=to_time.month,
            day_end=to_time.day,
            hour_end=to_time.hour,
            minute_end=to_time.minute,
            fcdr_version=self.data_version,
            mode="write")
        return pathlib.Path(fn)
#        raise NotImplementedError()
            

def _no_exit(code):
    raise RuntimeError("Who is calling sys.exit() and why?")

def main():
    sys.exit = _no_exit
    warnings.filterwarnings("error", category=numpy.VisibleDeprecationWarning)
    fgen = FCDRGenerator(p.satname,
        datetime.datetime.strptime(p.from_date, p.datefmt),
        datetime.datetime.strptime(p.to_date, p.datefmt))
    fgen.process()
