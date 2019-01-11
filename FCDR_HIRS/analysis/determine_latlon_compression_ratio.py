"""Determine latlon compression ratio

Read some HIRS FCDR data, take the latitude and longitude.  Store to disk
only those using various combinations of parameters.  In particular,
determine whether scaled int has significant gain over float including
compression.

Commandline script with no command-line arguments, no input files, and no
output files.  See also :ref:`determine-hirs-latlon-compression-ratio`.
"""

import datetime
import tempfile
import pathlib

from .. import fcdr

codings = {
    "float":
        {"dtype": "float32",
         "complevel": 4,
         "_FillValue": 0,
         "zlib": True},
}
codings.update(**{
    "scaled_int32_{n:d}dp".format(n=n):
        {"dtype": "int32",
         "scale_factor": 1/10**n,
         "complevel": 4,
         "_FillValue": 0,
         "zlib": True}
        for n in range(7)})

def main():
    """Main function.

    See module docstring: :mod:`determine_latlon_compression_ratio`.
    """
    hirs = fcdr.which_hirs_fcdr("metopa", read="L1C")
    dsref = hirs.read_period(
        datetime.datetime(2010, 5, 1),
        datetime.datetime(2010, 5, 2),
        locator_args={"fcdr_version": "0.4", "fcdr_type": "debug"},
        fields=("lat", "lon"))
    
    with tempfile.TemporaryDirectory() as td:
        for (k, v) in codings.items():
            ds = dsref.copy()
            for var in ("lat", "lon"):
                ds[var].encoding = v
            ds = ds.drop(("scanline", "scanpos", "time"))
            outfile = pathlib.Path(td + "/latlon_as_{:s}.nc".format(k))
            print("Writing", outfile)
            ds.to_netcdf(str(outfile))
        for (k, v) in sorted(codings.items()):
            reffile = pathlib.Path(td + "/latlon_as_float.nc".format(k))
            outfile = pathlib.Path(td + "/latlon_as_{:s}.nc".format(k))
            print(k, outfile.stat().st_size,
                "{:%}".format(1-outfile.stat().st_size/reffile.stat().st_size),
                "smaller")
