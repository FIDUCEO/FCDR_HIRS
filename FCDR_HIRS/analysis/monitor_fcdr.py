"""Plot some monitoring info on FCDR
"""

from .. import common
import argparse

def parse_cmdline():
    parser = argpars.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser = common.add_to_argparse(parser,
        include_period=True,
        include_sat=True,
        include_channels=True,
        include_temperatures=False)

    p = parser.parse_args()
    return p

import datetime
import xarray
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot
import FCDR_HIRS.fcdr
import pyatmlab.graphics

def plot():
#    hn17c = FCDR_HIRS.fcdr.HIRS3FCDR(satname="noaa17", read="L1C")
#    ds = hn17c.read_period(
#        datetime.datetime(2008, 1, 1, 0),
#        datetime.datetime(2008, 1, 3, 0),
#        locator_args={"fcdr_version": "0.1", "fcdr_type": "debug"})
    ds1 = xarray.open_dataset("/group_workspaces/cems2/fiduceo/Data/FCDR/HIRS/pre-β/v0.2/debug/metopa/2016/04/07/FCDR_HIRS_metopa_v0.2_debug_201604070021_201604070202.nc")

    fig, axes = matplotlib.pyplot.subplots(ncols=2, nrows=2)
    da1 = ds1["u_T_b_random"]
    da2 = ds1["u_T_b_systematic"]

    sp = 28; cc = 3
    da1.sel(scanpos=sp, calibrated_channel=cc).plot(ax=axes[0,0])
    da1.sel(scanpos=sp, calibrated_channel=cc).plot.hist(ax=axes[0,1])
    da2.sel(scanpos=sp, calibrated_channel=cc).plot(ax=axes[1,1])
    da2.sel(scanpos=sp, calibrated_channel=cc).plot.hist(ax=axes[1,1])
    pyatmlab.graphics.print_or_show(fig, False,
        "first_fcdr_uncertainties.pdf")
    

def main():
    plot()
