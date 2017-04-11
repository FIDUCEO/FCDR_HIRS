"""Convert HIRS SRFs
"""
import sys
import pathlib

import numpy
import xarray
import datetime
now = datetime.datetime.now

from typhon.physics.units.em import SRF
from typhon.physics.units.common import ureg
from typhon.datasets.tovs import (HIRS3, HIRS4)

outdir = pathlib.Path("/group_workspaces/cems2/fiduceo/scratch/HIRS_SRF")
satname = sys.argv[1].upper()


def main():
    std_centroid_per_ch_nm = {
        no+1: numpy.std(
            [SRF.fromArtsXML(satname.upper(), "hirs", no+1
                            ).centroid().to("nm", "sp").m
                for satname in HIRS3.satellites.keys() | HIRS4.satellites.keys()])
            for no in range(19)}
    (outdir / satname.lower()).mkdir(parents=True, exist_ok=True)
    for ch in range(1, 20):
        srf = SRF.fromArtsXML(satname, "hirs", ch)

        std = std_centroid_per_ch_nm[ch]
        for shift in ureg.Quantity(numpy.linspace(-std, std, 7), ureg.nm):
            print(now(), "channel", ch, "shift", shift)
            da = srf.shift(shift).as_dataarray("wavenumber")
            da.attrs["shift"] = shift.to("nm").m
            da.attrs["shift_units"] = "nm"
            da.to_netcdf(
                str(outdir / satname.lower() / satname.lower())
                    + "_ch{:d}_shift{:+d}pm.nc".format(ch, int(shift.to("pm").m)))
