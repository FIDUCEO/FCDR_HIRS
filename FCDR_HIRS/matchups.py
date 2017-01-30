"""Any code related to processing or analysing matchups
"""

import numpy

import typhon.datasets.tovs

from . import fcdr

hh = typhon.datasets.tovs.HIRSHIRS()

class HIRSMatchupCombiner:
    def __init__(self, start_date, end_date, prim, sec):
        #self.ds = netCDF4.Dataset(str(sf), "r")
        M = hh.read_period(start_date, end_date,
            locator_args={"prim": prim, "sec": sec},
            pseudo_fields={
                "time_{:s}".format(prim):
                    lambda M: M["hirs-{:s}_time".format(prim)][:, 3, 3].astype("M8[s]"),
                "time_{:s}".format(sec):
                    lambda M: M["hirs-{:s}_time".format(sec)][:, 3, 3].astype("M8[s]")})
        self.hirs_prim = fcdr.which_hirs_fcdr(prim)
        self.hirs_sec = fcdr.which_hirs_fcdr(sec)
        Mcp = hh.combine(M, self.hirs_prim, trans={"time_{:s}".format(prim): "time"},
                         timetol=numpy.timedelta64(3, 's'),
                         col_field="hirs-{:s}_scanpos".format(prim))
        Mcs = hh.combine(M, self.hirs_sec, trans={"time_{:s}".format(sec): "time"},
                         timetol=numpy.timedelta64(3, 's'),
                         col_field="hirs-{:s}_scanpos".format(sec))
        self.start_date = start_date
        self.end_date = end_date
        self.M = M
        self.Mcp = Mcp
        self.Mcs = Mcs
        self.prim = prim
        self.sec = sec

