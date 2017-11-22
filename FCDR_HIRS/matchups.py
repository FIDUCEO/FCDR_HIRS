"""Any code related to processing or analysing matchups
"""

import numpy

import typhon.datasets.tovs
import typhon.datasets.filters
import itertools

from . import fcdr

class HHMatchupCountFilter(typhon.datasets.filters.OrbitFilter):
    def __init__(self, prim, sec):
        self.prim = prim
        self.sec = sec

    def filter(self, ds, **extra):
        return ds[{"matchup_count": \
                        (abs(ds[f"hirs-{self.prim:s}_lza"][:, 3, 3] - \
                             ds[f"hirs-{self.sec:s}_lza"][:, 3, 3]) < 5)}]

# inspect_hirs_matchups, work again
hh = typhon.datasets.tovs.HIRSHIRS(read_returns="xarray")

class HIRSMatchupCombiner:
    fcdr_info = {"data_version": "0.8pre", "fcdr_type": "debug"}
    fields_from_each = {"u_α", "u_Rself", "u_f_eff", "u_C_Earth",
    "u_C_space", "u_β", "u_C_IWCT", "α", "β", "fstar", "C_E", "C_IWCT",
    "C_s", "T_IWCT", "R_selfE", "R_e", "T_b"}

    def __init__(self, start_date, end_date, prim, sec):
        #self.ds = netCDF4.Dataset(str(sf), "r")
        # acquire original brightness temperatures here for the purposes
        # of estimating Kr.  Of course this should come from my own
        # brightness temperatures, but the context of those is not
        # readibly available in the matchups from BC, so it would take
        # more effort to gather the necessary context information.  See
        # #117.
        ds = hh.read_period(start_date, end_date,
            locator_args={"prim": prim, "sec": sec},
            fields={"hirs-{:s}_{:s}".format(s, field)
                for field in ("x", "y", "time", "lza", "file_name",
                              "acquisition_time", "scanpos") + tuple(
                                "bt_ch{:02d}".format(ch) for ch in
                                range(1, 20))
                for s in (prim, sec)}|{"matchup_spherical_distance"},
            pseudo_fields={
                "time_{:s}".format(prim):
                    lambda ds: ds["hirs-{:s}_time".format(prim)][:, 3, 3].astype("M8[s]"),
                "time_{:s}".format(sec):
                    lambda ds: ds["hirs-{:s}_time".format(sec)][:, 3, 3].astype("M8[s]")},
            orbit_filters=hh.default_orbit_filters+[HHMatchupCountFilter(prim,sec)])
        self.hirs_prim = fcdr.which_hirs_fcdr(prim, read="L1C")
        self.hirs_sec = fcdr.which_hirs_fcdr(sec, read="L1C")
        Mcp = hh.combine(ds, self.hirs_prim, trans={"time_{:s}".format(prim): "time"},
                         timetol=numpy.timedelta64(4, 's'),
                         col_field="hirs-{:s}_x".format(prim),
                         col_dim_name="scanpos",
                         other_args={"locator_args": self.fcdr_info,
                                     "fields": self.fields_from_each},
                         time_name="time_"+prim)
        Mcs = hh.combine(ds, self.hirs_sec, trans={"time_{:s}".format(sec): "time"},
                         timetol=numpy.timedelta64(4, 's'),
                         col_field="hirs-{:s}_x".format(sec),
                         col_dim_name="scanpos",
                         other_args={"locator_args": self.fcdr_info,
                                     "fields": self.fields_from_each},
                         time_name="time_"+sec)
        self.start_date = start_date
        self.end_date = end_date
        self.ds = ds
        self.Mcp = Mcp
        self.Mcs = Mcs
        self.prim = prim
        self.sec = sec
