"""Convert HIRS SRFs
"""
import pathlib

import numpy
import xarray
import datetime
now = datetime.datetime.now

from typhon.physics.units.em import SRF
from typhon.physics.units.common import ureg

outdir = pathlib.Path("/group_workspaces/cems2/fiduceo/scratch/HIRS_SRF")

def main():
    outdir.mkdir(parents=True, exist_ok=True)
    for ch in range(1, 20):
        srf = SRF.fromArtsXML("METOPA", "hirs", ch)

        for shift in ureg.Quantity(numpy.arange(-20, 21, 10), ureg.nm):
            print(now(), "channel", ch, "shift", shift)
            da = srf.shift(shift).as_dataarray("wavenumber")
            da.to_netcdf(str(outdir / "METOPA_ch{:d}_shift{:+d}nm.nc").format(
                ch, shift.m))
