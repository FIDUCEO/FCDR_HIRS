"""Get harmonisation parameters and write out Python code

Need to pass files from which to read, in order of channel.  So, for
example, one might call it as

``convert_hirs_harmonisation_parameters /group_workspaces/cems2/fiduceo/Users/rquast/processing/harmonisation/3.0-3cab9f5/\*/\*19.nc``

This will write to stdout a Python module that should be written to
``_harm_defs.py`` inside the ``FCDR_HIRS/`` package.
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

**AUTOMATICALLY GENERATED — DO NOT EDIT!!**

Definitions relating to the output of harmonisation.

In particular, the `harmonisation_parameters` dictionary contains the
relevant parameters for each satellite and channel for which harmonisation
has so far been applied in the form of:

``Dict[str, Dict[int, Dict[int, float]]]``

For example, to get a₀ for noaa18, channel 12:

``harmonisation_parameters["noaa18"][12][0]``

Corresponding uncertainties are contained in
`harmonisation_parameters_uncertainty`.  
Covariances are contained in
`harmonisation_parameter_covariances`
Harmonisation parameters are derived using software developed by Ralf Quast.

**AUTOMATICALLY GENERATED — DO NOT EDIT!!**
"""

###############################################
###                                         ###
### AUTOMATICALLY GENERATED — DO NOT EDIT!! ###
###                                         ###
###############################################
###                                         ###
### This file is generated using:           ###
###                                         ###
### {cmdline:s}
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

def get_parser():
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
    parser.add_argument("--params-included", action="store", type=int,
        nargs="+", default=[0, 1, 2],
        help=("Numbers of harmonisation parameters included, set rest to 0. "
              "For example, without emissivity, pass 1 3"))

    return parser
def parse_cmdline():
    return get_parser().parse_args()

# from attachment from email RQ 2018-03-26, sent to SH, JM, EW, GH

#scaling = numpy.array([1e-15, 1e-21, 1e-3])
scaling = numpy.array([1, 1, 1]) # RQ e-mail 2018-10-11, scaling already applied

def get_harm_dict(chans, files, params_included=[0, 1, 2]):
    """Convert all.

    Returns (harms, u_harms)
    """

    all_sats = {typhon.datasets.tovs.norm_tovs_name(sat) for sat in
                fcdr.list_all_satellites()}
    D = {}
    harms = {sat: {ch: numpy.zeros(3) for ch in chans} for sat in all_sats}
    u_harms = copy.deepcopy(harms)
    s_harms = {sat: {ch: numpy.zeros((3,3)) for ch in chans} for sat in all_sats}
    ni = len(params_included)
    sats_found = set()
    for (ch, fn) in zip(chans, files):
        with xarray.open_dataset(fn) as ds:
            for i in range(ds.dims["n"]):
                sat = ds["parameter_name"][i].item().decode("ascii").strip()
                sat = typhon.datasets.tovs.norm_tovs_name(sat)
                sats_found.add(sat)
                if not sat in D.keys():
                    D[sat] = {ch: iter(params_included.copy()) for ch in chans}
                c = next(D[sat][ch])
                harms[sat][ch][c] = ds["parameter"][i].item() * scaling[c]
                u_harms[sat][ch][c] = ds["parameter_uncertainty"][i].item() * scaling[c]
                if c==params_included[0]:
                    s_harms[sat][ch][
                        numpy.ix_(params_included, params_included)] = (
                        ds["parameter_covariance_matrix"][i:(i+ni), i:(i+ni)]
                        * scaling[params_included, numpy.newaxis]
                        * scaling[numpy.newaxis, params_included]).values
    
    return (harms, u_harms, s_harms)

def write_harm_dict(fp, harms, write_preamble=True):
    if write_preamble:
        fp.write(preamble.format(cmdline=" ".join(sys.argv)))

    print("#: harmonisation parameters")
    print("harmonisation_parameters = ", pprint.pformat(harms[0]),
            file=fp)

    print("#: harmonisation uncertainties")
    print("harmonisation_parameter_uncertainties = ", pprint.pformat(harms[1]),
            file=fp)

    print("#: harmonisation covariances")
    print("harmonisation_parameter_covariances = ", pprint.pformat(harms[2]),
            file=fp)

def main():
    p = parse_cmdline()
    with numpy.errstate(all="raise"):
        write_harm_dict(
            sys.stdout,
            get_harm_dict(
                p.chans,
                p.files,
                params_included=p.params_included),
            True)
