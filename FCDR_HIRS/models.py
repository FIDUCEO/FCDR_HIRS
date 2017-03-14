"""Classes for models included with measurement equation.

To include models for self-emission, reflected radiation, and others.

Work in progress!
"""

import scipy.stats
import sklearn.cross_decomposition
import numpy
import xarray

from typhon.physics.units.common import ureg
from typhon.physics.units.tools import UnitsAwareDataArray as UADA


class RSelf:
    temperatures = ["scanmirror", "fwh", "iwct",
                    "secondary_telescope", "baseplate", "electronics"]

    X_ref = None
    Y_ref = None

    def __init__(self, hirs, temperatures=None):
        self.hirs = hirs
        self.model = sklearn.cross_decomposition.PLSRegression(
            n_components=2)
        if temperatures is not None:
            self.temperatures = temperatures
    
    def get_predictor(self, ds, ch):
        L = []
        for t_fld in self.temperatures:
            x = ds["temperature_{:s}".format(t_fld)]
            for dim in set(x.dims) - {"time"}:
                x = x.mean(dim=dim, keep_attrs=True)
            L.append(x.astype("f8")) # prevent X⁴ precision loss
        #X = numpy.concatenate(tuple(L), 1)
        X = xarray.merge(L)
        # fit in terms of X⁴ because that's closer to radiance than
        # temperature is.
        Xn = ((X**4)-(X**4).mean("time"))/(X**4).std("time")

        return Xn

    @staticmethod
    def _dsOK(X):
        OK = X.notnull()
        OK = xarray.concat(OK.data_vars.values(), dim="dummy").all("dummy")
        return OK


    def _ds2ndarray(self, X, Y=None):
        OK = self._dsOK(X)
        if Y is not None:
            OK = OK & Y.notnull()
        # self.model needs ndarray not xarray
        Xx = numpy.concatenate([x.values[OK, numpy.newaxis]
                for x in X.data_vars.values()], 1)
        if Y is not None:
            Yy = Y.values[OK]
        return (Xx, Yy) if Y is not None else Xx

    def get_predictand(self, M, ch):
        offset = self.hirs.calculate_offset_and_slope(M, ch)[1]
        # Note: median may not be an optimal solution, see plots produced
        # by plot_hirs_calibcounts_per_scanpos
#        Y = ureg.Quantity(numpy.ma.median(offset.m, 1),
#                          offset.u)
        Y = offset.median("scanpos", keep_attrs=True)
        return Y

    def fit(self, ds, ch):
        """Fit model
        """
        #ds = self._subsel_calib(ds, ch)
        ix = self.hirs.extract_calibcounts_and_temp(
            ds, ch, return_ix=True)[-1]
        X = self.get_predictor(ds.isel(time=ix), ch)
        Y = self.get_predictand(ds, ch)
#        #OK = ~(X.mask.any(1)) & ~(Y.mask)
#        OK = X.notnull()
#        OK = xarray.concat(OK.data_vars.values(), dim="dummy").all("dummy")
#        OK = OK & Y.notnull()
#
#        # self.model needs ndarray not xarray
#        Xx = numpy.concatenate([x.values[OK, numpy.newaxis]
#                for x in X.data_vars.values()], 1)
#        Yy = Y.values[OK]
        (Xx, Yy) = self._ds2ndarray(X, Y)
        self.model.fit(Xx, Yy)
        self.X_ref = X
        self.Y_ref = Y

    def evaluate(self, ds, ch):
        X = self.get_predictor(ds, ch)
#        Y_ref = self.get_predictand(M, ch)
#        (Xx, Yy) = self._ds2ndarray(X, Y_ref)
        Xx = self._ds2ndarray(X)
        Yy_pred = self.model.predict(Xx).squeeze()
        Y_pred = UADA(Yy_pred,
            coords=X.isel(time=self._dsOK(X)).coords, attrs=self.Y_ref.attrs)
        return (X, Y_pred)

    def test(self, ds, ch):
        """Test model for reference data.

        Use this to estimate uncertainties!

        FIXME expand
        """
        Y_ref = self.get_predictand(ds, ch)
        ix = self.hirs.extract_calibcounts_and_temp(
            ds, ch, return_ix=True)[-1]
        (X, Y_pred) = self.evaluate(ds.isel(time=ix), ch)
        return (X, Y_ref.squeeze(), Y_pred.squeeze())

    def __str__(self):
        return (
            "Self-emission model for {self.hirs!s}.\n"
            "Model type: {self.model!s}.\n"
            "Temperatures used: {self.temperatures!s}.\n"
            ) 


class RRefl:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Not implemented yet!")
