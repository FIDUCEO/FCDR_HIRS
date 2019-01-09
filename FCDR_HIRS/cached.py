"""For utilities with disk-memoisation

This module contains a number of helper functions that employ disk
memoisation using `joblib`.  Essentially, those functions are identical to
counterparts elsewhere in the `FCDR_HIRS` package, but including disk
memoisation.
"""

import numpy
import tempfile
import joblib
from typhon.datasets import tovs
from typhon.datasets import filters
from typhon import config

#: `joblib.Memory` object used for disk cache for functions in this module
memory = joblib.Memory(
    cachedir=config.conf["main"]["cachedir"],
    verbose=1,
    )

@memory.cache
def read_tovs_hirs_period(satname, start_date, to_date, fields):
    """Read L1B for HIRS satellite using standard orbit filters

    Read HIRS L1B using a set of recommended defined orbit filters.

    Parameters
    ----------

    satname : str
        Name of the satellite for which to read HIRS data.
    start_date : datetime.datetime
        Starting datetime for which to read.
    to_date : datetime.datetime
        Ending date.
    fields : List[str] or str
        List of strings, fields of which to read.  The special case "all"
        will result in all fields being read from the dataset.

    Returns
    -------

    xarray.Dataset
        Dataset containing the requested fields along with associated
        coordinates for the entire period.

    See Also
    --------

    This function calls
    :meth:`typhon.datasets.dataset.Dataset.read_period` for
    HIRS objects, for example
    :meth:`typhon.datasets.tovs.HIRS4.read_period` in case of a HIRS4
    satellite.

    """
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
