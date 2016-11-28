"""Classes for models included with measurement equation.

To include models for self-emission, reflected radiation, and others.

Work in progress!
"""

import scipy.stats
import sklearn.cross_decomposition.PLSRegression

class RSelf:

    temperatures = ["scanmirror", "fwh", "iwt",
                    "sectlscp", "baseplate", "elec"]

    def __init__(self, temperatures=None):
        self.model = sklearn.cross_decomposition.PLSRegression(
            n_components=2)
        if temperatures is not None:
            self.temperatures = temperatures
    
    def get_predictor(self, M, ch):
        L = []
        for t_fld in self.temperatures:
            x = M["temp_{:s}".format(t_fld)]
            while x.ndim > 1:
                x = x.mean(-1)
            L.append(x[:, numpy.newaxis])
        X = numpy.concatenate(tuple(L))
        # Note: median may not be an optimal solution, see plots produced
        # by plot_hirs_calibcounts_per_scanpos
        Y = numpy.ma.median(M["counts"][:, 8:, ch-1], 1)
        # fit in terms of X⁴ because that's closer to radiance than
        # temperature is
        Xn = ((X**4)-(X**4).mean(0))/(X**4).std(0)
        return Xn

    def fit(self, M, ch):
        """Fit model
        """
        (X, Y) = self.get_predictor_predictand(M, ch)
        self.model.fit(X, Y)

    def evaluate(self, M, ch):
        (X, Y_ref) = self.get_predictor_predictand(M, ch)
        Y_pred = self.model.predict(X)
        return (X, Y_ref, Y_pred)

    def test(self, M, ch):
        """Test model for reference data.

        Use this to estimate uncertainties!

        FIXME expand
        """
        (X, Y_ref, Y_pred) = self.evaluate(M, ch)
        return (X, Y_ref.squeeze(), Y_pred.squeeze())
