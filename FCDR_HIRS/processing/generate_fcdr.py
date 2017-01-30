"""Generate FCDR for satellite and period

Generate HIRS FCDR for a particular satellite and period.

"""

# need some kind of deque for processing

from .. import common
import argparse

def parse_cmdline():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentsDefaultHelpFormatter)

    common.add_to_argparse(
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
    filename=parsed_cmdline.log,
    level=logging.DEBUG if parsed_cmdline.verbose else logging.INFO)

from .. import fcdr

class FCDRGenerator:
    def __init__(self, sat, start_date, end_date):
        self.fcdr = fcdr.which_hirs_fcdr(sat)
        self.start_date = start_date
        self.end_date = end_date

    def process(self):
        pass

def main():
    fgen = FCDRGenerator(p.satname,
        datetime.datetime.strptime(p.from_date, p.datefmt),
        datetime.datetime.strptime(p.to_date, p.datefmt))
