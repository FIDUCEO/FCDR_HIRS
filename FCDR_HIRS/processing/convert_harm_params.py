"""Get harmonisation parameters and write out Python code
"""

import argparse
import itertools

import xarray
import pprint

def parse_cmdline():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("files", action="store", type=str,
        nargs=19,
        help="List of files (one per channel) containing output from "
             "RQs harmonisation process")

    return parser.parse_args()
p = parse_cmdline()

# from attachment from email RQ 2018-03-26, sent to SH, JM, EW, GH

scaling = [1e-15, 1e-21, 1e-3]

def convert_all(files):
    """Convert all.
    """

    if len(files) != 19:
        raise ValueError("Must pass 19 paths")

    D = {}
    harms = {}
    for (ch, fn) in enumerate(files, 1):
        with xarray.open_dataset(fn) as ds:
            for i in range(ds.dims["n"]):
                sat = ds["parameter_sensors"][i].item().decode("ascii").strip()
                if not sat in D.keys():
                    D[sat] = {ch: itertools.count() for ch in range(1, 20)}
                    harms[sat] = {ch: {} for ch in range(1, 20)}
                c = next(D[sat][ch])
                harms[sat][ch][c] = ds["parameter"][i].item() * scaling[c]
    print("harmparms = ", pprint.pformat(harms))

def main():
    convert_all(p.files)
