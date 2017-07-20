"""Classes related to filtering
"""

import abc

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
