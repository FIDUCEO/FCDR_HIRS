"""Get harmonisation parameters and write out Python code

Need to pass files from which to read, in order of channel.  So, for
example, one might call it as

convert_hirs_harmonisation_parameters /group_workspaces/cems2/fiduceo/Users/rquast/processing/harmonisation/3.0-3cab9f5/*/*19.nc
"""

import sys
import copy
import argparse
import itertools

import numpy
import xarray
import pprint

import typhon.datasets.tovs
from .. import fcdr

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
###                                         ###
### This file is generated using the script ###
###                                         ###
### convert_hirs_harmonisation_parameters   ###
###                                         ###
### see convert_harm_params.py              ###
###                                         ###
###############################################
###                                         ###
### AUTOMATICALLY GENERATED — DO NOT EDIT!! ###
###                                         ###
###############################################

from numpy import (nan, array)

'''

def parse_cmdline():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--chans", action="store", type=int,
        nargs="+",
        help="List of channels containing output from harmonisation")

    parser.add_argument("--files", action="store", type=str,
        nargs="+",
        help="List of files (one per channel) containing output from "
             "RQs harmonisation process")

    return parser.parse_args()
p = parse_cmdline()

# from attachment from email RQ 2018-03-26, sent to SH, JM, EW, GH

scaling = numpy.array([1e-15, 1e-21, 1e-3])

def get_harm_dict(chans, files):
    """Convert all.

    Returns (harms, u_harms)
    """

    all_sats = {typhon.datasets.tovs.norm_tovs_name(sat) for sat in
                fcdr.list_all_satellites()}
    D = {}
    harms = {sat: {ch: {} for ch in chans} for sat in all_sats}
    u_harms = copy.deepcopy(harms)
    s_harms = copy.deepcopy(harms)
    sats_found = set()
    for (ch, fn) in zip(chans, files):
        with xarray.open_dataset(fn) as ds:
            for i in range(ds.dims["n"]):
                sat = ds["parameter_sensors"][i].item().decode("ascii").strip()
                sat = typhon.datasets.tovs.norm_tovs_name(sat)
                sats_found.add(sat)
                if not sat in D.keys():
                    D[sat] = {ch: itertools.count() for ch in chans}
                c = next(D[sat][ch])
                harms[sat][ch][c] = ds["parameter"][i].item() * scaling[c]
                u_harms[sat][ch][c] = ds["parameter_uncertainty"][i].item() * scaling[c]
                if c==0:
                    s_harms[sat][ch] = (ds["parameter_covariance_matrix"][i:(i+3), i:(i+3)]
                        * scaling[:, numpy.newaxis] * scaling[numpy.newaxis, :]).values
    
    # set to zero for satellites not found
    for sat in all_sats - sats_found:
        for ch in chans:
            for i in range(3):
                harms[sat][ch][i] = 0
                u_harms[sat][ch][i] = numpy.array(
                    [u_harms[sat][ch][i] for sat in sats_found]
                        ).mean()
                s_harms[sat][ch] = numpy.zeros((3, 3))
    return (harms, u_harms, s_harms)

def write_harm_dict(fp, harms, write_preamble=True):
    if write_preamble:
        fp.write(preamble)

    print("harmonisation_parameters = ", pprint.pformat(harms[0]),
            file=fp)

    print("harmonisation_parameter_uncertainties = ", pprint.pformat(harms[1]),
            file=fp)

    print("harmonisation_parameter_covariances = ", pprint.pformat(harms[2]),
            file=fp)

def main():
    with numpy.errstate(all="raise"):
        write_harm_dict(sys.stdout, get_harm_dict(p.chans, p.files), True)
