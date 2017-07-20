"""Classes for models included with measurement equation.

To include models for self-emission, reflected radiation, and others.

Work in progress!
"""
import datetime
import copy
import logging

import scipy.stats
import sklearn.cross_decomposition
import sklearn.linear_model
import numpy
import xarray

from typhon.physics.units.common import ureg
from typhon.physics.units.tools import UnitsAwareDataArray as UADA
from typhon.datasets import _tovs_defs

regression_types = {
    "PLSR": sklearn.cross_decomposition.PLSRegression,
    "LR": sklearn.linear_model.LinearRegression}

class RSelf:
    # default set of temperatures: all that consistently exist across
    # series
    #
    # in theory, one should want to use temperatures for the components
    # dominating the self-emission, but in practice, sufficiently
    # desirable properties will be:
    #
    # - temperature should vary with slope
    # - temperature should not vary with not-slope
    #
    # noted:
    #
    # - filter_wheel_housing and filter_wheel_motor may be heated up which
    # may be associated with
    # a gain change.  Even with mild overfitting that will kill the
    # self-emission model.  See, for example, MetOp-A 2016-03-01.  All
    # other temperatures behave nominally, but MetOp-A
    # filter_wheel_housing has a state change, and back.  No way I can
    # keep using the old self-emission model that way.
    #
    # - smaller but significant changes in channel_housing, primtlscp,
    # elec
    #
    # - minor changes in scanmirror, scanmotor, baseplate
    #
    # - no noticeable changes in sectlscp, iwt, 
    #
    # - patch_full is useless, doesn't change
    #
    # - should still do a more formal analysis for an optimal seletion
    temperatures = ['baseplate', 
        'internal_warm_calibration_target',
        'scanmirror', 'scanmotor',
        'secondary_telescope']
    # not included: patch_full, filter_wheel_housing, cooler_housing,
    # primary_telescope, filter_wheel_motor, electronics,
    X_ref = None
    Y_ref = None
    fit_time = None

    def __init__(self, hirs, temperatures=None,
            regr=("LR", {"fit_intercept": True})):
        self.hirs = hirs
        self.core_model = regression_types[regr[0]](**regr[1])#sklearn.cross_decomposition.PLSRegression(
        self.models = {}
            #n_components=2)
        if temperatures is not None:
            self.temperatures = temperatures
    
    def get_predictor(self, ds, ch):
        L = []
        for t_fld in self.temperatures:
            t_fld = _tovs_defs.temperature_names.get(t_fld, t_fld)
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


    def _ds2ndarray(self, X, Y=None, dropna=False):
        OK = self._dsOK(X)
        if Y is not None:
            OK = OK & Y.notnull()
        # model needs ndarray not xarray
        # FIXME: when I fit, I may want to REMOVE invalid entries
        # completely as I won't want to use them in the training.  But
        # when I evaluate, I want to keep them or the dimensions of the
        # self-emission estimate data array will be inconsistent with
        # others, i.e. I need to fill the rest up somehow, probably best
        # to do that here.
        Xx = numpy.zeros(shape=(X.coords["time"].size, len(X.data_vars.keys())),
            dtype="f4")
        Xx.fill(numpy.nan)
        OK = OK.values # needed for NumPy < 1.13
        Xx[OK, :] = numpy.concatenate([x.values[OK, numpy.newaxis]
                for x in X.data_vars.values()], 1)
        if dropna:
            Xx = Xx[OK, :]
        if Y is not None:
            Yy = numpy.zeros(shape=(X.coords["time"].size,), dtype="f4")
            Yy.fill(numpy.nan)
            Yy[OK] = Y.values[OK]
            if dropna:
                Yy = Yy[OK]
        return (Xx, Yy) if Y is not None else Xx

    def get_predictand(self, M, ch):
        offset = self.hirs.calculate_offset_and_slope(M, ch)[1]
        # Note: median may not be an optimal solution, see plots produced
        # by plot_hirs_calibcounts_per_scanpos
#        Y = ureg.Quantity(numpy.ma.median(offset.m, 1),
#                          offset.u)
        Y = offset.median("scanpos", keep_attrs=True)
        return Y

    def fit(self, ds, ch, force=False):
        """Fit model for channel
        """
        #ds = self._subsel_calib(ds, ch)
        ix = self.hirs.extract_calibcounts_and_temp(
            ds, ch, return_ix=True)[-1]
        X = self.get_predictor(ds.isel(time=ix), ch)
        Y = self.get_predictand(ds, ch)
        # filtering on counts here… should not be in reading routine as I
        # don't necessarily want to mask them all out as nan (and they're
        # ints!).  Doesn't really fit in get_predictand either for Y and X
        # will be out of shape.  Can't fit it anywhere else.  Believe me,
        # I've tried.  — GH, 2017-07-19
        OK = (~self.hirs.filterer.filter_outliers(ds["counts"].isel(time=ix).sel(
            channel=ch, scanpos=slice(8, None)).values).any(1))
        OK &= (ds.isel(time=ix).sel(channel=ch)["channel_quality_flags_bitfield"].values==0)
        OK &= (ds.isel(time=ix).sel(channel=ch)["channel_quality_flags_bitfield"].values==0)
        OK &= (ds.isel(time=ix)["quality_flags_bitfield"].values==0)
#        #OK = ~(X.mask.any(1)) & ~(Y.mask)
#        OK = X.notnull()
#        OK = xarray.concat(OK.data_vars.values(), dim="dummy").all("dummy")
#        OK = OK & Y.notnull()
#
#        # self.models[ch] needs ndarray not xarray
#        Xx = numpy.concatenate([x.values[OK, numpy.newaxis]
#                for x in X.data_vars.values()], 1)
#        Yy = Y.values[OK]
        (Xx, Yy) = self._ds2ndarray(X, Y, dropna=True)
        if OK.sum() < .5*OK.size:
            logging.warning("When trying to fit self-emission model in "
                "period {:%Y-%m-%d %H:%M}--{:%Y-%m-%d %H:%M} for channel "
                "{:d}, only "
                "{:d}/{:d} space calibration lines pass tests.  Proceeding "
                "self-emisison training with caution (and calibration "
                "should also take note of this).".format(
                    ds.isel(time=ix)["time"][0].values.astype("M8[ms]").astype(datetime.datetime),
                    ds.isel(time=ix)["time"][-1].values.astype("M8[ms]").astype(datetime.datetime),
                    ch,
                    int(OK.sum()), int(OK.size)))
        elif OK.sum() < 20:
            raise ValueError("All space views in fitting period "
                "contain outliers.  Perhaps space views are not really "
                "space views, see github issue #12.  For now I can't "
                "proceed.  I'm sorry.")
        Xx = Xx[OK, :]
        Yy = Yy[OK]
        # if slopes fail normality test, we may be in a regime where we
        # the gain is changing more than usually; for example, see MetOp-A
        # 2016-03-01 – 03-03.  The assumptions behind the self-emission
        # model will fail.  Instead, we should decline to update it and
        # flag that the self-emission model is old.
        tr = scipy.stats.normaltest(Yy)
        if tr.statistic > 10 and tr.pvalue < 0.05 and not force:
            raise ValueError("Space views fail normality test: "
                f"test statistic {tr.statistic:.3f}, p-value "
                f"{tr.pvalue:9.3e}. Is the gain changing?")
        if ch not in self.models.keys():
            self.models[ch] = copy.copy(self.core_model)
        self.models[ch].fit(Xx, Yy)
        self.X_ref = X
        self.Y_ref = Y
        self.fit_time = ds["time"].values[[0,-1]].astype("M8[ms]")

    def evaluate(self, ds, ch):
        X = self.get_predictor(ds, ch)
#        Y_ref = self.get_predictand(M, ch)
#        (Xx, Yy) = self._ds2ndarray(X, Y_ref)
        Xx = self._ds2ndarray(X, dropna=False)
        Yy_pred = numpy.zeros(shape=(X.coords["time"].size,), dtype="f4")
        Yy_pred.fill(numpy.nan)
        OK = numpy.isfinite(Xx).all(1)
        Yy_pred[OK] = self.models[ch].predict(Xx[OK, :]).squeeze()
        Y_pred = UADA(Yy_pred,
            coords=X["time"].coords, attrs=self.Y_ref.attrs)
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
        # str(self.models[ch]) triggers http://bugs.python.org/issue29672 due
        # to a catch_warnings call in
        # sklearn.base.BaseEstimator.get_params, which gets called by
        # __repr__.  Call self.core_model.__class__ instead until the Python
        # bug is fixed.
        return (
            f"Self-emission model for {self.hirs!s}.\n"
            f"Model type: {self.core_model.__class__!s}.\n"
            f"Temperatures used: {self.temperatures!s}.\n"
            ) 


class RRefl:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Not implemented yet!")
