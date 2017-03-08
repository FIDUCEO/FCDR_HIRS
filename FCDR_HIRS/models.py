"""Classes for models included with measurement equation.

To include models for self-emission, reflected radiation, and others.

Work in progress!
"""

import scipy.stats
import sklearn.cross_decomposition
import numpy

from typhon.physics.units.common import ureg

class RSelf:
    temperatures = ["scanmirror", "fwh", "iwt",
                    "sectlscp", "baseplate", "elec"]

    def __init__(self, hirs, temperatures=None):
        self.hirs = hirs
        self.model = sklearn.cross_decomposition.PLSRegression(
            n_components=2)
        if temperatures is not None:
            self.temperatures = temperatures
    
    def get_predictor(self, M, ch):
        # figure out what are the calibration positions
        ix = self.hirs.extract_calibcounts_and_temp(
            M, ch, return_ix=True)[-1]
        M = M[ix]

        L = []
        for t_fld in self.temperatures:
            x = M["temp_{:s}".format(t_fld)]
            while x.ndim > 1:
                x = x.mean(-1)
            L.append(x[:, numpy.newaxis])
        X = numpy.concatenate(tuple(L), 1)
        # fit in terms of X⁴ because that's closer to radiance than
        # temperature is
        Xn = ((X**4)-(X**4).mean(0))/(X**4).std(0)

        return Xn

    def get_predictand(self, M, ch):
        offset = self.hirs.calculate_offset_and_slope(M, ch)[1]
        # Note: median may not be an optimal solution, see plots produced
        # by plot_hirs_calibcounts_per_scanpos
        Y = ureg.Quantity(numpy.ma.median(offset.m, 1),
                          offset.u)
        return Y

    def fit(self, M, ch):
        """Fit model
        """
        X = self.get_predictor(M, ch)
        Y = self.get_predictand(M, ch)
        OK = ~(X.mask.any(1)) & ~(Y.mask)
        self.model.fit(X[OK, :], Y[OK])

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
