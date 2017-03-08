"""Classes for models included with measurement equation.

To include models for self-emission, reflected radiation, and others.

Work in progress!
"""

import scipy.stats
import sklearn.cross_decomposition
import numpy
import xarray

from typhon.physics.units.common import ureg


class RSelf:
    temperatures = ["scanmirror", "fwh", "iwct",
                    "secondary_telescope", "baseplate", "electronics"]

    def __init__(self, hirs, temperatures=None):
        self.hirs = hirs
        self.model = sklearn.cross_decomposition.PLSRegression(
            n_components=2)
        if temperatures is not None:
            self.temperatures = temperatures
    
    def get_predictor(self, ds, ch):
        # figure out what are the calibration positions
        ix = self.hirs.extract_calibcounts_and_temp(
            ds, ch, return_ix=True)[-1]
        ds = ds.isel(time=ix)

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
        X = self.get_predictor(ds, ch)
        Y = self.get_predictand(ds, ch)
        #OK = ~(X.mask.any(1)) & ~(Y.mask)
        OK = X.notnull()
        OK = xarray.concat(OK.data_vars.values(), dim="dummy").all("dummy")
        OK = OK & Y.notnull()

        self.model.fit(X.isel(time=OK), Y.isel(time=OK))

    def evaluate(self, M, ch):
        X = self.get_predictor(M, ch)
        Y_ref = self.get_predictand(M, ch)
        Y_pred = ureg.Quantity(self.model.predict(X),
            Y_ref.u)
        return (X, Y_ref, Y_pred)

    def test(self, M, ch):
        """Test model for reference data.

        Use this to estimate uncertainties!

        FIXME expand
        """
        (X, Y_ref, Y_pred) = self.evaluate(M, ch)
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
