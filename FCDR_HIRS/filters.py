"""Classes related to filtering
"""

import abc

import numpy
import scipy.stats

from typhon.datasets.filters import OutlierFilter, MEDMAD
from typhon.math.stats import adev

class CalibrationMirrorFilter(metaclass=abc.ABCMeta):
    """Filter for cases where first N space views do not really view space

    Happens for IWCT too.  See
    https://github.com/FIDUCEO/FCDR_HIRS/issues/12
    """
    
    @abc.abstractmethod
    def filter_calibcounts(self, counts):
        """Expects xarray DataArray
        """
        ...


class IQRCalibFilter(CalibrationMirrorFilter):
    def __init__(self, rng=(10, 90), cutoff=3.3):

        self.rng = rng
        self.cutoff = cutoff

    def filter_calibcounts(self, counts, dim="scanpos"):
        """Expects xarray DataArray.  Scanning over dimension 'dim',
        defaults to "scanpos".
        """

        # NB: need to encapsulate this or it will fail with ValueError,
        # see https://github.com/scipy/scipy/issues/7178
        return counts.reduce(
            scipy.stats.iqr, dim=dim,
            rng=self.rng) > 3.3 * adev(counts, dim=dim)


class ImSoColdFilter:
    """Intended for Earth Counts, reject when colder than space
    """

    def filter(self, C_Earth, C_space):
        # depending on the sign of the gain, either they should be all
        # larger or all smaller... in either case, equal to space is bad
        # enough!

        csm = numpy.ma.median(C_space)
        # to circumvent https://github.com/pydata/xarray/issues/1792
        if isinstance(C_space, numpy.ma.masked_array):
            C_space = C_space.data
        if C_Earth.median() > csm: # should never equal
            return C_Earth <= C_space[:, numpy.newaxis]
        else:
            return C_Earth >= C_space[:, numpy.newaxis]
