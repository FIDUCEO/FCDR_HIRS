"""Classes for models needed to evaluate measurement equation.

This module gathers classes implementing models that are needed to
evaluate the measurement equation.  The measurement equation currently
contains two models:

- `RSelf`, which is the sole implementation of the self-emission model.
   In HIRS, modelling self-emission is essential, because it only views
   the calibration target and deep space every 40 scanlines / 256 seconds,
   yet it is not cryogenically cooled.  See the documentation for the
   `RSelf` class for details on the current approach.

   To create an alternative implementation with the same interface, the
   simplest way would be to create another class with the same interface,
   although ideally one would want to create an abstract class defining
   the interface from which the old and the new implementation both
   inherit.  To create an alternative implementation with a different
   interface, which is likely scenario, one would also have to adapt the
   location where the self-emission model is being used:

   - The most important location is in fcdr.HIRSFCDR.calculate_radiance,
     which calls `RSelf.fit`, `RSelf.test`, and `RSelf.evaluate`.  The
     model is first defined within
     `processing.generate_fcdr.FCDRGenerator.__init__`.  If the interface
     is simliar enough, only the latter would need to be changed.
   - There is also the `analysis.test_rself`, which is an entire module
     (associated with the script :ref:`plot-hirs-test-rself`), which you
     would probably want to adapt to test and compare both self-emission
     models.
  
  In addition to a new implementation of `RSelf`, one would also need to
  create a new implementation of `effects.RModelRSelf`, which is not
  currently directly derived from `RSelf`.  The model in
  `effects.RModelRSelf` describes the correlation between scanlines and
  channels.  However, the assumptions in `effects.RModelRSelf` are
  currently so poor that they might not be any poorer in a new
  implementation.

  The simplest alternative self-emission model, that is not currently
  implemented, would be one that does a simply linear interpolation
  between the nearest calibration lines.  I think this could be
  implemented by a new `RSelf` in which the `RSelf.fit` method is a noop,
  the `RSelf.test` method would develop a model of how large an error
  compared to the IASI reference such a linear interpolation may get as a
  function of the distance in scanlines to the nearest calibration cycle,
  and th `RSelf.evaluate` equivalent would perform interpolation.  I think
  this could leave `fcdr.HIRSFCDR.calculate_radiance` largely untouched,
  although one way want to add some notes to the debug FCDR on whach
  self-emission model has been used.  I don't know what a good model for
  `effects.RModelRSelf` would be in this case.

  Finally, there is the generation of the W-Matrix, which does not yet use
  the CURUC recipes which use `effects.RModelRSelf`.  Until it does, the
  hardcoded W-Matrix in :mod:`FCDR_HIRS.processing.combine_matchups` (linked to
  the scripts :ref:`combine-hirs-hirs-matchups` and
  :ref:`combine-hirs-iasi-matchups`) would need to be adapted as well, but
  ideally the generation of the W-Matrix needs to be integrated with the
  CURUC recipes.  See :issue:`224`.

- `RRefl`, which is not implemented, but which is a placeholder for the
   evaluation of Earthshine or any other radiance that reflects on the IWCT
   into the detector during IWCT views, i.e., whatever contributes (1-ε) to
   the measured IWCT radiance.  Implementing this would require
   substantial work.

"""
import datetime
import copy
import logging
import functools
import operator

import scipy.stats
import sklearn.cross_decomposition
import sklearn.linear_model
import numpy
import xarray
import abc

from typhon.physics.units.common import ureg
from typhon.physics.units.tools import UnitsAwareDataArray as UADA
from typhon.datasets import _tovs_defs

logger = logging.getLogger(__name__)

#: Mapping : mapping of supported regression types
regression_types = {
    "PLSR": sklearn.cross_decomposition.PLSRegression,
    "LR": sklearn.linear_model.LinearRegression}

class RSelf:

    @abc.abstractmethod
    def fit(self, context, ch):
        ...


    @abc.abstractmethod
    def test(self, context, ch):
        ...

    @abc.abstractmethod
    def evaluate(self, lines, ch):
        ...

class RSelfTemperature(RSelf):
    """Implementation of self emission model using temperatures

    This class models the HIRS self emission as a function of a set of
    instrument temperatures, given by the attribute `RSelf.temperatures`.  
    Training is performed based on fitting the space counts (which is
    essentially the offset) as a function of those temperatures normalised 
    to the 4th power.  To keep independence between training and testing,
    the training data is split in two parts, one is used for fitting using
    `RSelf.fit`, the other for testing using `RSelf.test`.

    To use the model, there are three important external methods to use:

    - After initialising (see `RSelf.__init__`) the model, the model needs
      to be fit using `RSelf.fit`, passing any number of scanlines
      including calibration cycles (elsewhere I call this *context*).
      Fitting will define the fit parameters.  This can be done as often
      as one likes, but needs to be done at least once.  If `RSelf.fit` is
      called again, all parameters are redefined and all previous
      parameters are lost.
      
      Ideally, it should also define their uncertainties, but currently
      uncertainties are evaluated in the next step.
    - To estimate the performance of the model, use `RSelf.test`.
      You should pass the same data to `RSelf.test` as to `RSelf.fit`, as
      the model uses indices to divide the data into a part for
      fitting/training and a part for testing.  I currently use the return
      value of `RSelf.test` to estimate uncertainties statistically, but
      there is much room for improvement here.
    - To subsequently use the model, use `RSelf.evaluate`.

    Those are the most important methods, and those are the methods that
    any other implementation should have externally, and that any abstract
    class would define in case this class was restructured.

    There are some scripts tesing self-emission stuff, but they are old
    and rusty.  The script :ref:`plot-hirs-field-timeseries` includes a panel on
    self-emission as well.

    See also
    --------

    :class:`FCDR_HIRS.effects.RModelRSelf`
        Implementation of self emission model error correlation.  This is
        not yet integrated into the present class, although the
        uncertainties themselves are derived from the `RSelf.test` method.
    """
    temperatures = ['baseplate', 
        'internal_warm_calibration_target',
        'scanmirror', 'scanmotor',
        'secondary_telescope']
    """List[str] : Default set of temperatures to use for self-emission

    In theory, one should want to use temperatures for the components
    dominating the self-emission, but in practice, sufficiently
    desirable properties will be:
    
    - temperature should vary with slope
    - temperature should not vary with not-slope
   
    Some other thoughts:
    
    - ``filter_wheel_housing`` and ``filter_wheel_motor`` may be heated up
      which may be associated with a gain change.  Even with mild
      overfitting that will kill the self-emission model.  See, for
      example, MetOp-A 2016-03-01.  All other temperatures behave
      nominally, but MetOp-A ``filter_wheel_housing`` has a state change,
      and back.  No way I can keep using the old self-emission model that way.
    - smaller but significant changes in ``channel_housing``,
      ``primtlscp``, ``elec``
    - minor changes in ``scanmirror``, ``scanmotor``, ``baseplate``
    - no noticeable changes in ``sectlscp``, ``iwt``, 
    - ``patch_full`` is useless, doesn't change
    - should still do a more formal analysis for an optimal seletion
      not included: ``patch_full``, ``filter_wheel_housing``,
      ``cooler_housing``, ``primary_telescope``, ``filter_wheel_motor``,
      ``electronics``.
    """

    X_ref = None
    """ndarray : Reference measurements, independent variable, multivariate

    Attribute to hold the multivariate independent variable for reference
    measurements, used to train the regression.
    """

    Y_ref = None
    """ndarray : Reference measurement, dependent variable

    Attribute to hold the univariate dependent variable for reference
    measurements, used to train the regression.
    """
    fit_time = None
    """List[datetime.datetime] : Time period for reference data

    After fitting, this attribute is set to the beginning and end time of
    the time period for the training data, such as stored in `X_ref` and
    `Y_ref`.
    """

    def __init__(self, hirs, temperatures=None,
            regr=("LR", {"fit_intercept": True})):
        """Initialise `RSelf` object

        Create an `RSelf` object.  Usually, you'll just want to pass as the
        first argument the `HIRS2FCDR`, `HIRS3FCDR`, or `HIRS4FCDR` object
        that you are using to develop the HIRS FCDR.  For the other
        arguments, the default values are usually fine.

        Parameters
        ----------

        hirs : `fcdr.HIRSFCDR`
            'fcdr.HIRSFCDR` object that the self-emission model relates
            to.
        temperatures : List[str], optional
            List of temperatures to use.  Overrides the `temperatures`
            attribute.
        regr : Tuple[str, Mapping], optional
            Regression type to use.  Should be a 2-tuple with a string and
            a mapping.  The string can be either "LR" or "PLSR", according
            to the module attribute :attr:`regression_types`.  The second
            argument is a dictionary that will be passed on to the class
            on construction.
        """
        self.hirs = hirs
        self.core_model = regression_types[regr[0]](**regr[1])#sklearn.cross_decomposition.PLSRegression(
        self.models = {}
            #n_components=2)
        if temperatures is not None:
            self.temperatures = temperatures
    
    def get_predictor(self, ds, ch, recalculate_norm=False):
        """Get predictor (temperatures)

        Get the normalised predictor, i.e. the multivariate independent
        variable, normalised.  This will consider all lines passed in.  If
        you are training or testing, please preselect the input before
        passing it on.
    
        This method uses self.temperatures to determine what temperatures
        to use, averages the temperatures over any non-time dimension (for
        example, if there are multiple PRTs or multiple measurements for a
        PRT), then filters outliers before normalising the 4th power to
        its norm and standard deviation.

        This method is specific to the temperature-based implementation of
        the self-emission model, is not necessary in other
        implementations, and is not intended to be used externally.

        Parameters
        ----------

        ds : xarray.Dataset
            Dataset for the period for which to get the predictor.
        ch : int
            Channel.  Not used.  I don't know why I have it here.  This
            does not depend on the channel.  I suppose I have it here in
            order to be consistent with `get_predictand`, which does have
            a channel dependency, and because other implementations might
            in principle have a channel dependency, except that this
            method is not externaly interfaced anyway so it doesn't matter
            for that reason.
        recalculate_norm : bool, optional
            Recalculate the standard deviation and mean that are used to
            normalise the predictor.  You don't want to do this every
            time, because you want to make sure you use the same
            normalisation coefficients when training, testing, and when
            actually using the self-emission model!  Defaults to False.
        """
        L = []
        for t_fld in self.temperatures:
            t_fld = _tovs_defs.temperature_names.get(t_fld, t_fld)
            x = ds["temperature_{:s}".format(t_fld)]
            for dim in set(x.dims) - {"time"}:
                x = x.mean(dim=dim, keep_attrs=True)
            L.append(x.astype("f8")) # prevent X⁴ precision loss
        #X = numpy.concatenate(tuple(L), 1)
        X = xarray.merge(L)
        # std is spectacularly sensitive to outliers, so we need to make
        # sure we normalise this while removing those
        OK = ~functools.reduce(
            operator.or_, 
            [self.hirs.filter_prttemps.filter_outliers(v.values) for v in X.data_vars.values()])
        # fit in terms of X⁴ because that's closer to radiance than
        # temperature is.
        # if times constant then std dev should be zero (except it isn't:
        # https://github.com/numpy/numpy/issues/9631), account for this
        if recalculate_norm:
            stdoffs = (X**4).isel(time=OK).mean("time")
            stdnorm = (X**4).isel(time=OK).std("time")
            # not sure if I can skip the loop on this one...
            for k in X.data_vars.keys():
                if X.isel(time=OK)[k].values.ptp() == 0:
                    # norm doesn't matter much as the values should be all
                    # (close to) zero anyway; there is no information in
                    # this predictor.  The best thing would be to throw it
                    # out but that's a bit beyond the responsibility of a
                    # normalisation routine.  See #147
                    stdnorm[k] = 1.0
            self.norm_offset = stdoffs
            self.norm_factor = stdnorm
        Xn = ((X**4)-self.norm_offset)/self.norm_factor

        return Xn

    @staticmethod
    def _dsOK(X):
        """Static helper method to get an index of OK predictands

        Parameters
        ----------

        X : xarray.DataArray
            Array of predictands

        Returns
        -------

        xarray.DataArray
            Boolean array, True for OK values in X
        """
        OK = X.notnull()
        OK = xarray.concat(OK.data_vars.values(), dim="dummy").all("dummy")
        return OK


    def _ds2ndarray(self, X, Y=None, dropna=False):
        """Helper method to convert DataArray to ndarray

        The fitting model needs an ndarray, but I'm internally using
        DataArray.  This helper method selects OK data points and converts
        the result to ndarray.

        Parameters
        ----------

        X : xarray.DataArray
            Independent variable, predictor.
        Y : xarray.DataArray, optional
            If applicable (because we're training or testing), dependent
            variable, predictand.
        dropna : bool, optional
            If true, drop bad data points completely rather than masking
            them.  Defaults to False, but you'll want to put this to True
            when training.

        Returns
        -------

        X : ndarray
            X with bad data masked or removed, as an ndarray
        Y : ndarray
            Only returned if Y passed as input.  Y with bad data masked or
            removed, as an ndarray.
        """
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
        """Get predictand (space counts)

        Get the predictant, here implemented as the median offset for all
        space views.  This is of course only relevant for the training and
        testing phase.

        This method is specific to the temperature-based implementation of
        the self-emission model, is not necessary in other
        implementations, and is not intended to be used externally.
        
        Parameters
        ----------

        M : xarray.Dataset
            Full dataset segment / context of the format coming from the
            HIRS reading routines ``as_xarray_dataset`` method.  Should
            not be preselected, this method does the selection.
        ch : int
            Channel number for which to obtain this.

        Returns
        -------

        Y : xarray.DataArray
            Predictand (space counts).
        """
        offset = self.hirs.calculate_offset_and_slope(M, ch)[1]
        # Note: median may not be an optimal solution, see plots produced
        # by plot_hirs_calibcounts_per_scanpos
#        Y = ureg.Quantity(numpy.ma.median(offset.m, 1),
#                          offset.u)
        Y = offset.median("scanpos", keep_attrs=True)
        return Y

    def _OK_traintest(self, Y):
        """Filter the predictand for training and testing purposes.
        
        Filter out outliers in the predictand for training and testing purposes.

        Parameters
        ----------

        Y : xarray.DataArray
            Predictand

        Returns
        -------

        ndarray
            True for OK values, false for outliers
        """
        return ~self.hirs.filter_calibcounts.filter_outliers(Y.values)
#        return (~self.hirs.filter_calibcounts.filter_outliers(ds["counts"].sel(
#            scanpos=slice(8, None)).values).any(1))

    _OKfields = {
        2: ("quality_flags_bitfield",),
        3: ("channel_quality_flags_bitfield", "quality_flags_bitfield"),
        4: ("channel_quality_flags_bitfield", "quality_flags_bitfield")}

    _badflags = {
        2: {"quality_flags_bitfield":
            ["do_not_use"]},
        3: {"channel_quality_flags_bitfield":
            ["bad_prt_ch", "bad_space_ch", "bad_iwct_ch"],
            "quality_flags_bitfield":
            ["do_not_use", "insufficient_calib", "time_seq_error"]},
        4: {"channel_quality_flags_bitfield":
            ["space_failed_nedc_ch", "iwct_failed_nedc_ch",
            "anom_iwct_or_space_ch", "calib_failed_ch"],
            "quality_flags_bitfield":
            ["do_not_use", "insufficient_calib", "time_seq_error"]}}
#    _badflags[4] = _badflags[3] # no need to copy

    def _OK_eval(self, ds):
        """Filter the dataset for model evaluation purposes
        
        Consider whether lines are OK for evaluation by looking at flags.

        Parameters
        ----------

        ds : xarray.Dataset
            Dataset containing scanlines to be checked prior to
            evaluation.

        Returns
        -------

        ndarray
            True for OK lines, False for bad lines.
        """
        # ds["quality_flags_bitfield"] & functools.reduce(operator.or_,
        # ds["quality_flags_bitfield"].flag_masks)
        #
        # For each flagfield, do an OR between all flag meanings I have
        # listed as bad and verify those are all zero
        return functools.reduce(operator.and_, 
               (((ds[flagfield] &
                  functools.reduce(
                    operator.or_,
                    (flag_mask
                        for (flag_meaning, flag_mask)
                        in zip(ds[flagfield].flag_meanings.split(),
                               ds[flagfield].flag_masks)
                        if flag_meaning in
                               self._badflags[self.hirs.version][flagfield])
                  )).values==0) for flagfield in
                self._OKfields[self.hirs.version]))

#        return ((ds["channel_quality_flags_bitfield"].values==0)
#            & (ds["quality_flags_bitfield"].values==0))

    def _ensure_enough_OK(self, ds, OK):
        """Helper method ensuring there's enough OK lines for fitting

        If more than half the inputs are bad, raise a warning.

        If less than 20 good values, raise an exception.

        Parameters
        ----------

        ds : xarray.Dataset
            Dataset with scanlines to be checked prior to evaluation.
        OK : ndarray
            Boolean array such as returned by `RSelf._OK_eval`.

        Raises
        ------

        ValueError
            Raised if less than 20 lines are happy
        """
        if OK.sum() < .5*OK.size:
            logger.warning("When trying to fit or test self-emission model in "
                "period {:%Y-%m-%d %H:%M}--{:%Y-%m-%d %H:%M} for channel "
                "{:d}, only "
                "{:d}/{:d} space calibration lines pass tests.  Proceeding "
                "self-emission training with caution (and calibration "
                "should also take note of this).".format(
                    ds["time"][0].values.astype("M8[ms]").astype(datetime.datetime),
                    ds["time"][-1].values.astype("M8[ms]").astype(datetime.datetime),
                    int(ds.channel.values),
                    int(OK.sum()), int(OK.size)))
        if OK.sum() < 20:
            raise ValueError("All space views in fitting period "
                "contain outliers.  Perhaps space views are not really "
                "space views, see github issue #12.  For now I can't "
                "proceed.  I'm sorry.")

    def fit(self, ds, ch, force=False):
        """Fit model for channel

        Using training data that will be obtained from the dataset passed
        in, fit the self-emission model for a particular channel.  This
        method:

        - extracts the predictor and predictand from the dataset
        - selects for both the lines that are OK
        - applies normalisation
        - verifies the distribution is sufficiently normal
        - does the actual fitting
        - stores the result in the attributes `models`, `X_ref`, `Y_ref`,
          and `fit_time`

        Parameters
        ----------

        ds : xarray.Dataset
            Dataset containing segment of L1B data based on which the
            training will occur.  I often choose 24 hour of data for this,
            but I don't know how optimal this is.
        ch: int
            Channel to train.
        force: bool, optional
            If True, proceed with fitting even if the training data fails
            the normality test.  Defaults to False.
        """
        #ds = self._subsel_calib(ds, ch)
        ix = self.hirs.extract_calibcounts_and_temp(
            ds, ch, return_ix=True)[-1]
        X = self.get_predictor(ds.isel(time=ix), ch,
            recalculate_norm=True)
        Y = self.get_predictand(ds, ch)
        if numpy.isinf(Y).any():
            raise ValueError("Some offsets are infinite.  That probably "
                "means calibration counts (space and IWCT) are all equal. "
                "There is no hope of doing anything meaningful here. "
                "Sorry.")
        # filtering on counts here… should not be in reading routine as I
        # don't necessarily want to mask them all out as nan (and they're
        # ints!).  Doesn't really fit in get_predictand either for Y and X
        # will be out of shape.  It's different between training/testing
        # or evaluation, because this is for space views only.
        OK = self._OK_traintest(Y)
#        OK = self._OK_traintest(ds.isel(time=ix).sel(channel=ch))
#        OK = (~self.hirs.filterer.filter_outliers(ds["counts"].isel(time=ix).sel(
#            channel=ch, scanpos=slice(8, None)).values).any(1))
        OK &= self._OK_eval(ds.isel(time=ix).sel(channel=ch))
#        OK &= (ds.isel(time=ix).sel(channel=ch)["channel_quality_flags_bitfield"].values==0)
#        OK &= (ds.isel(time=ix).sel(channel=ch)["channel_quality_flags_bitfield"].values==0)
#        OK &= (ds.isel(time=ix)["quality_flags_bitfield"].values==0)

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
        # remove outliers due to PRT temperature problems from training
        OK &= ~self.hirs.filter_prttemps.filter_outliers(Xx).any(1)
        self._ensure_enough_OK(ds.isel(time=ix).sel(channel=ch), OK)
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
        if ch not in self.models:
            self.models[ch] = copy.copy(self.core_model)
        self.models[ch].fit(Xx, Yy)
        self.X_ref = X
        self.Y_ref = Y
        self.fit_time = ds["time"].values[[0,-1]].astype("M8[ms]")

    def evaluate(self, ds, ch):
        """Apply self-emission model to data

        When fitting (and preferably testing) has been completed, apply
        self-emission model to real data.  This method:

        - extracts the predictor from the source scanlines dataset
        - converts this to the right format, including masking bad lines
        - estimate the predictand (space counts) for all lines

        Parameters
        ----------

        ds : xarray.Dataset
            Dataset containing the L1B scanlines for which the
            self-emission is to be estimated
        ch : int
            Channel for which to estimate self-emission.

        Returns
        -------

        X : xarray.Dataset
            Predictor that was used to evaluate
        Y : `typhon.physics.units.tools.UnitsAwareDataArray`
            Estimates of self-emission for all scanlines in ds
        """
        X = self.get_predictor(ds, ch,
            recalculate_norm=False)
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

        Test the self-emission model using reference data (calibration
        lines) for which both predictor (temperatures) and predictand
        (space counts) are available.  This is currently the only source
        from which to estimate uncertainties on the self-emission model.
        
        This method:

        - extracts the predictand (space counts) from the calibration
          lines within the dataset,
        - extracts the predictor (normalised temperatures) from the same,
        - applies the model to the same, getting predicted space counts

        This method does not deal with error correlations, those are
        rather treated in the :class:`effects.RModelRSelf` class.

        Parameters
        ----------

        ds : xarray.Dataset
            Dataset containing the scanlines for which to test the model.
        ch : int
            Channel for which to test the model.

        Returns
        -------

        X : xarray.DataArray
            Predictors, normalised temperatures used to test the model.
        Yref: xarray.DataArray
            Predictand, reference values corresponding to X.
        Ypred: xarray.DataArray
            Actual outcome of model, evaluated for X.

        """
        ix = self.hirs.extract_calibcounts_and_temp(
            ds, ch, return_ix=True)[-1]
        Y_ref = self.get_predictand(ds, ch)
        OK = (self._OK_traintest(Y_ref) &
              self._OK_eval(ds.isel(time=ix).sel(channel=ch)))
#              self._OK_traintest(ds.isel(time=ix).sel(channel=ch))
        self._ensure_enough_OK(ds.isel(time=ix).sel(channel=ch), OK)
        (X, Y_pred) = self.evaluate(ds.isel(time=ix).isel(time=OK), ch)
        return (X, Y_ref.isel(time=OK).squeeze(), Y_pred.squeeze())

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
    """Placeholder for Earthshine model

    Nothing here yet.  Not used.  Unconditional error.
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Not implemented yet!")
