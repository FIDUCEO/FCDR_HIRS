"""Convert HIRS-HIRS matchups for harmonisation

Take HIRS-HIRS matchups and add telemetry and other information as needed
for the harmonisation effort.
"""



from .. import common
import argparse

def parse_cmdline():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser = common.add_to_argparse(parser,
        include_period=True,
        include_sat=2,
        include_channels=False,
        include_temperatures=False)

    return parser.parse_args()
p = parse_cmdline()

import logging
logging.basicConfig(
    format=("%(levelname)-8s %(asctime)s %(module)s.%(funcName)s:"
            "%(lineno)s: %(message)s"),
    level=logging.DEBUG if p.verbose else logging.INFO)

import datetime

import xarray
from .. import matchups

class HIRSMatchupCombiner(matchups.HIRSMatchupCombiner):
    def write(self, outfile):
        raise NotImplementedError

def main():
    hmc = HIRSMatchupCombiner(
        datetime.datetime.strptime(p.from_date, p.datefmt),
        datetime.datetime.strptime(p.to_date, p.datefmt),
        p.satname1, p.satname2)

    hmc.write("test")
