"""Generate FCDR for satellite and period

Generate HIRS FCDR for a particular satellite and period.

"""

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
import xarray
import typhon.datasets.dataset
from .. import fcdr


class FCDRGenerator:
    window_size = datetime.timedelta(hours=24)
    step_size = datetime.timedelta(hours=6)
    # FIXME: do we have a filename convention?
    # FIXME: this should be incorporated in the general HomemadeDataset
    # class
    basedir = "/group_workspaces/cems2/fiduceo/Data/FCDR/HIRS/pre-β/testing"
    subdir = "{from_time:%Y/%m/%d}"
    filename = "HIRS_FCDR_sketch_{satname:s}_{from_time:%Y%m%d%H%M}_{to_time:%H%M}.nc"
    def __init__(self, sat, start_date, end_date):
        self.satname = sat
        self.fcdr = fcdr.which_hirs_fcdr(sat)
        self.start_date = start_date
        self.end_date = end_date
        self.dd = typhon.datasets.dataset.DatasetDeque(
            self.fcdr, self.window_size, start_date)

    def process(self, start=None, end_time=None):
        """Generate FCDR for indicated period
        """
        start = start or self.start_date
        end_time = end_time or self.end_date
        self.dd.reset(start)
        while self.dd.center_time < end_time:
            self.make_and_store_piece(self.dd.center_time,
                self.dd.center_time + self.step_size)
            self.dd.move(self.step_size)

    def make_and_store_piece(self, from_, to):
        """Generate and store one “piece” of FCDR

        This generates one “piece” of FCDR, i.e. in a single block.  For
        longer periods, use the higher level method `process`.
        """

        piece = self.get_piece(from_, to)
        self.store_piece(piece)

    def get_piece(self, from_, to):
        """Get FCDR piece for period.

        Returns a single xarray.Dataset
        """
        subset = self.dd.data.sel(time=slice(from_, to))
        R_E = self.fcdr.calculate_radiance_all(subset, context=self.dd.data)  
        cu = {}
        u = self.fcdr.calc_u_for_variable("R_e", self.fcdr._quantities,
            self.fcdr._effects, cu)
        uc = xarray.Dataset({k: v.magnitude for (k, v) in self.fcdr._effects_by_name.items()})
        qc = xarray.Dataset(self.fcdr._quantities)
        qc = xarray.Dataset(
            {str(k): v for (k, v) in self.fcdr._quantities.items()})
        ds = xarray.merge([uc.rename({k: "u_"+k for k in uc.data_vars.keys()}),
                      qc, subset, u])
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
            )
        return ds

    def store_piece(self, piece):
        fn = self.get_filename_for_piece(piece)
        fn.parent.mkdir(exist_ok=True, parents=True)
        logging.info("Storing to {!s}".format(fn))
        piece.to_netcdf(str(fn))

    _i = 0
    def get_filename_for_piece(self, piece):
        fn = "/".join((self.basedir, self.subdir, self.filename))
        fn = fn.format(satname=self.satname,
            from_time=piece["time"][0].values.astype("M8[s]").astype(datetime.datetime),
            to_time=piece["time"][-1].values.astype("M8[s]").astype(datetime.datetime))
        return pathlib.Path(fn)
#        raise NotImplementedError()
            

def main():
    fgen = FCDRGenerator(p.satname,
        datetime.datetime.strptime(p.from_date, p.datefmt),
        datetime.datetime.strptime(p.to_date, p.datefmt))
    fgen.process()
