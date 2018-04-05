"""Get harmonisation parameters and write out Python code

Need to pass files from which to read, in order of channel.  So, for
example, one might call it as

convert_hirs_harmonisation_parameters /group_workspaces/cems2/fiduceo/Users/rquast/processing/harmonisation/3.0-3cab9f5/*/*19.nc
"""

import sys
import argparse
import itertools

import xarray
import pprint

import typhon.datasets.tovs

preamble='''"""Harmonisation definitions.

Definitions relating to the output of harmonisation.

In particular, the `harmonisation_parameters` dictionary contains the
relevant parameters for each satellite and channel for which harmonisation
has so far been applied in the form of:

`Dict[str, Dict[int, Dict[int, float]]]`

For example, to get a₀ for noaa18:

`harmonisation_parameters["noaa18"][12][0]`

Corresponding uncertainties are contained in
`harmonisation_parameters_uncertainty`.  Covariance is not yet supported.
Harmonisation parameters are derived using software developed by Ralf Quast.
"""

###############################################
###                                         ###
### AUTOMATICALLY GENERATED — DO NOT EDIT!! ###
###                                         ###
###############################################

'''

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

def get_harm_dict(files):
    """Convert all.

    Returns (harms, u_harms)
    """

    if len(files) != 19:
        raise ValueError("Must pass 19 paths")

    D = {}
    harms = {}
    u_harms = {}
    for (ch, fn) in enumerate(files, 1):
        with xarray.open_dataset(fn) as ds:
            for i in range(ds.dims["n"]):
                sat = ds["parameter_sensors"][i].item().decode("ascii").strip()
                sat = typhon.datasets.tovs.norm_tovs_name(sat)
                if not sat in D.keys():
                    D[sat] = {ch: itertools.count() for ch in range(1, 20)}
                    harms[sat] = {ch: {} for ch in range(1, 20)}
                    u_harms[sat] = {ch: {} for ch in range(1, 20)}
                c = next(D[sat][ch])
                harms[sat][ch][c] = ds["parameter"][i].item() * scaling[c]
                u_harms[sat][ch][c] = ds["parameter_uncertainty"][i].item() * scaling[c]
    return (harms, u_harms)

def write_harm_dict(fp, harms, write_preamble=True):
    if write_preamble:
        fp.write(preamble)

    print("harmonisation_parameters = ", pprint.pformat(harms[0]),
            file=fp)

    print("harmonisation_parameter_uncertainties = ", pprint.pformat(harms[1]),
            file=fp)

def main():
    write_harm_dict(sys.stdout, get_harm_dict(p.files), True)
