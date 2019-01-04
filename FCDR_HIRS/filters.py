"""Classes related to filtering

This module contains classes related to filtering for the purposes of FCDR
processing.

Filtering on the reading routine is implemented in
`typhon.datasets.filters`.
"""

import abc

import numpy
import scipy.stats

from typhon.datasets.filters import OutlierFilter, MEDMAD
from typhon.math.stats import adev

class CalibrationMirrorFilter(metaclass=abc.ABCMeta):
    """Filter for cases where first N space views do not really view space

    Happens for IWCT too.  See `issue`:12 at
    https://github.com/FIDUCEO/FCDR_HIRS/issues/12
    """
    
    @abc.abstractmethod
    def filter_calibcounts(self, counts):
        """Filter calibration counts

        Considering a set of calibration counts as an `xarray.DataArray`,
        any implementations of this abstract method must return another
        array of the same dimensions, which is True for any calibration
        line that must be filtered out.

        There is currently a single implementation of this class,
        `IQRCalibFilter`.
        """
        pass


class IQRCalibFilter(CalibrationMirrorFilter):
    """Filter calibration lines based on the inter quartile range

    Filter out calibration lines where the calibration mirror is slow to
    swing into place, by rejecting cases where the inter quartile range is
    much larger than the Allan deviation, as determined by the cutoff,
    which defaults to 3.3.  The inter "quartile" range is taken between
    the percentiles given by ``rng``, which defaults to (10, 90).

    This is the sole implementation of the `CalibrationMirrorFilter`.
    """

    def __init__(self, rng=(10, 90), cutoff=3.3):
        """Initialise the IQRCalibFilter

        Parameters
        ----------

        rng : Tuple[Number, Number], optional
            Range of percentiles between which to calculate the range.
        cutoff : Number
            Maximum factor to accept.  If the IQR is more than ``cutoff``
            times the Allan deviation, the line will be rejected.
        """
        self.rng = rng
        self.cutoff = cutoff

    def filter_calibcounts(self, counts, dim="scanpos"):
        # NB: need to encapsulate this or it will fail with ValueError,
        # see https://github.com/scipy/scipy/issues/7178
        return counts.reduce(
            scipy.stats.iqr, dim=dim,
            rng=self.rng) > self.cutoff * adev(counts, dim=dim)


class ImSoColdFilter:
    """Filter to reject Earth counts that are colder than space
    """

    def filter(self, C_Earth, C_space):
        """Filter Earth counts colder than space

        Parameters
        ----------
        C_Earth : array_like
            Earth counts
        C_space : array_like
            Space counts

        Returns
        -------
        ndarray
            True where Earth counts are colder than space counts.
        """
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
