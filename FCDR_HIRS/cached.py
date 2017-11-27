"""For utilities with disk-memoisation

Many of those are rather hard-coded, and optimised for using with
disk-memoisation through joblib
"""

import numpy
import tempfile
import joblib
from typhon.datasets import tovs
from typhon.datasets import filters
from typhon import config

memory = joblib.Memory(
    cachedir=config.conf["main"]["cachedir"],
    verbose=1,
    )

@memory.cache
def read_tovs_hirs_period(satname, start_date, to_date, fields):
    h = tovs.which_hirs(satname)
    return h.read_period(
        start_date,
        to_date,
        fields=fields,
        orbit_filters=[
            filters.HIRSBestLineFilter(h),
            filters.TimeMaskFilter(h),
            filters.HIRSTimeSequenceDuplicateFilter(),
            filters.HIRSFlagger(h),
            filters.HIRSCalibCountFilter(h, h.filter_calibcounts),
            filters.HIRSPRTTempFilter(h, h.filter_prttemps)])

# FIXME: put something like fieldmat._SatPlotFFT._extract_counts
# as this method takes a lot of time when plotting
