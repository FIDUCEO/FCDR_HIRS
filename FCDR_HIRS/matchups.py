"""Any code related to processing or analysing matchups
"""

import numpy

import typhon.datasets.tovs
import itertools

from . import fcdr

# change "xarray" to "ndarray" to make old scripts, such as
# inspect_hirs_matchups, work again
hh = typhon.datasets.tovs.HIRSHIRS(read_returns="xarray")

class HIRSMatchupCombiner:
    fcdr_info = {"data_version": "0.5", "fcdr_type": "debug"}
    def __init__(self, start_date, end_date, prim, sec):
        #self.ds = netCDF4.Dataset(str(sf), "r")
        ds = hh.read_period(start_date, end_date,
            locator_args={"prim": prim, "sec": sec},
            fields={"hirs-{:s}_{:s}".format(s, field)
                for field in ("x", "y", "time", "lza", "file_name",
                              "acquisition_time", "scanpos")
                for s in (prim, sec)}|{"matchup_spherical_distance"},
            pseudo_fields={
                "time_{:s}".format(prim):
                    lambda ds: ds["hirs-{:s}_time".format(prim)][:, 3, 3].astype("M8[s]"),
                "time_{:s}".format(sec):
                    lambda ds: ds["hirs-{:s}_time".format(sec)][:, 3, 3].astype("M8[s]")})
        self.hirs_prim = fcdr.which_hirs_fcdr(prim, read="L1C")
        self.hirs_sec = fcdr.which_hirs_fcdr(sec, read="L1C")
        Mcp = hh.combine(ds, self.hirs_prim, trans={"time_{:s}".format(prim): "time"},
                         timetol=numpy.timedelta64(3, 's'),
                         col_field="hirs-{:s}_x".format(prim),
                         col_dim_name="scanpos",
                         other_args={"locator_args": self.fcdr_info})
        Mcs = hh.combine(ds, self.hirs_sec, trans={"time_{:s}".format(sec): "time"},
                         timetol=numpy.timedelta64(3, 's'),
                         col_field="hirs-{:s}_x".format(sec),
                         col_dim_name="scanpos",
                         other_args={"locator_args": self.fcdr_info})
        self.start_date = start_date
        self.end_date = end_date
        self.ds = ds
        self.Mcp = Mcp
        self.Mcs = Mcs
        self.prim = prim
        self.sec = sec

