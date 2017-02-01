"""Generate FCDR for satellite and period

Generate HIRS FCDR for a particular satellite and period.

"""

# need some kind of deque for processing

from .. import common
import argparse

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
    window_size = datetime.timedelta(hours=48)
    step_size = datetime.timedelta(hours=6)
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
        logging.warning("Not implemented, making empty")
        return xarray.Dataset()

    def store_piece(self, piece):
        fn = self.get_filename_for_piece(piece)
        fn.parent.mkdir(exist_ok=True, parents=True)
        piece.to_netcdf(str(fn))

    _i = 0
    def get_filename_for_piece(self, piece):
        logging.warning("Not implemented, inventing phony filename")
        d = pathlib.Path("/work/scratch/gholl/test_fcdr_generation")
        self._i += 1
        return d / "{:s}_{:d}/{:d}.nc".format(self.satname, *divmod(self._i, 100))
#        raise NotImplementedError()
            

def main():
    fgen = FCDRGenerator(p.satname,
        datetime.datetime.strptime(p.from_date, p.datefmt),
        datetime.datetime.strptime(p.to_date, p.datefmt))
    fgen.process()
