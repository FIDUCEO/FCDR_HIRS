"""Convert HIRS SRFs

Convert HIRS SRFs, read from RTTOV, to NetCDF.

Those NetCDF files can be used by a script that Jon Mittaz wrote to
calculate band coefficients.

Call as

convert_hirs_srfs satname

after installing the FCDR_HIRS package.
"""

import sys
import pathlib

from typhon.physics.units.em import SRF
from typhon.datasets.tovs import norm_tovs_name

outdir = pathlib.Path("/group_workspaces/cems2/fiduceo/scratch/HIRS_SRF")

satname = norm_tovs_name(sys.argv[1], mode="default")

# NB: an earlier version of this script contained code for shifts based on
# ARTS SRFs
def main():
    (outdir / satname).mkdir(parents=True, exist_ok=True)
    for ch in range(1, 20):
        srf = SRF.fromRTTOV(norm_tovs_name(satname, mode="RTTOV"), "hirs", ch)
        da = srf.as_dataarray("wavenumber")
        outfile = str(outdir / satname / satname) + f"_ch{ch:d}_rttov.nc"
        print("Writing to", outfile)
        da.to_netcdf(outfile)
