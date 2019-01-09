"""Determine optimal uncertainty format.

As per issue :issue:`88` and as discussed during the FCDR design review
2017-04-27, we need to check whether uncertainties are stored more
efficiently in brightness temperatures or as percentages, and then stick
with that.  This script makes such a comparison.
"""

import datetime
import tempfile
import pathlib

from .. import fcdr

codings = {
    "as_bt":
        {"dtype": "int32",
         "scale_factor": 1e-3,
         "complevel": 4,
         "_FillValue": -1,
         "zlib": True},
    "as_perc":
        {"dtype": "int32",
         "scale_factor": 1e-7,
         "complevel": 4,
         "_FillValue": -1,
         "zlib": True},
}

def main():
    hirs = fcdr.which_hirs_fcdr("noaa15", read="L1C")
    dsref = hirs.read_period(
        datetime.datetime(2004, 4, 1),
        datetime.datetime(2004, 4, 2),
        locator_args={"data_version": "0.8pre", "fcdr_type": "debug"},
        fields=("u_T_b_nonrandom", "u_T_b_random", "T_b"))
    
    with tempfile.TemporaryDirectory() as td:
        ds = dsref.copy()
        out_as_bt = pathlib.Path(td + "/u_as_bt.nc")
        out_as_perc = pathlib.Path(td + "/u_as_perc.nc")
        ds["u_T_b_nonrandom"].encoding = codings["as_bt"]
        ds["u_T_b_random"].encoding = codings["as_bt"]
        ds.to_netcdf(str(out_as_bt))
        ds = dsref.copy()
        ds["u_T_b_nonrandom"] /= ds["T_b"]
        ds["u_T_b_random"] /= ds["T_b"]
        ds["u_T_b_nonrandom"].encoding = codings["as_perc"]
        ds["u_T_b_random"].encoding = codings["as_perc"]
        ds.to_netcdf(str(out_as_perc))
        print("as bt", out_as_bt.stat().st_size)
        print("as perc", out_as_perc.stat().st_size)
