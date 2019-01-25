Overview of FCDR\_HIRS code
===========================

The FCDR\_HIRS code processes L1B measurements from the NOAA CLASS
archive, analyses the properties of those measurements, and derive the
FIDUECO HIRS FCDR.  Some of the improvements compared to pre-FIDUCEO work:

- Metrologically traceable uncertainties per datum
- Error correlation estimates per orbit
- Consistent measurement equation for entire FCDR
- Consistent, modern fileformat for entire FCDR
- Physics-based self-emission model
- Attempts at harmonisation
- Fully open-source processing code

Limitations and problems
------------------------

Despite years of hard work, there are currently some serious limitations
and problems with the FIDUCEO HIRS FCDR and the code used to generate it.

Harmonisation
^^^^^^^^^^^^^

The harmonisation is currently not working well.  The process through
which the harmonisation is supposed to work is described under
:ref:`how-to-generate`.  In practice:

- Only channels 2--12 have been harmonised.
- Only 2 parameters are harmonised.
- The harmonised FCDR has, for some channels, larger differences than the
  unharmonised FCDR.
- Despite filtering (see the :mod:`matchups` module and the classes
  :class:`~matchups.KrFilterHomogeneousScenes`,
  :class:`~matchups.KrFilterDeltaLKr`, and :class:`~matchups.KFilterKDeltaL`),
  there remain severe outliers in some of the HIRS-HIRS matchup pairs.
  There are also pairs/channels with a large skew in the distribution of
  the residuals.  Those observations are probably related to the previous
  points.  See issues :issue:`112`, :issue:`191`, :issue:`280`,
  :issue:`287`, :issue:`332`, :issue:`365`, and
  `other harmonisation issues
  <https://github.com/FIDUCEO/FCDR_HIRS/issues?q=is%3Aissue+is%3Aopen+label%3Aharmonisation>`_.
- The filtering appears to be overdoing the stripping of the tails,
  leading to the tails being too narrow.  The aim was to shorten the
  tails.  Now the tails become too short.  See the presentation Gerrit
  gave at the science meeting on 21 January 2019.  This happens because I
  take the ratio between a fitted normal distribution and the real
  distribution, probabilistically reject matchups when this ratio is
  smaller than 1, but unconditionally accept when this ratio is 1 or
  lorger.  See the :class:`~matchups.KrFilterDeltaLKr` and
  :class:`~matchups.KFilterKDeltaL` classes for details on the current
  implementation.
- Ralf Giering considers that the currently implemented filters, in
  particuler in the classes :class:`~matchups.KrFilterDeltaLKr`,
  :class:`~matchups.KFilterKDeltaL`, is inappropriate.

To resolve those issues, we would need an improved method of outlier
detection.  The matchup combination scripts
(:ref:`combine-hirs-hirs-matchups` and :ref:`combine-hirs-iasi-matchups`)
take HIRS data from the generated debug HIRS FCDR.  Flagged data are
discarded, so if all bad HIRS data are correctly flagged (see
:ref:`outliers`), there should be no outliers in the HIRS-HIRS matchups,
and outliers in the HIRS-IASI matchups should be due to IASI only.
Ideally, this outlier detection should be done at the source.  It needs in
any case to be done before the W-matrix is calculated in
:meth:`~FCDR_HIRS.processing.HIRSMatchupCombiner.get_w_matrix`.

This would not resolve the skewed distribution or the too-narrow
distribution.  I don't know how big an impact on harmonisation either have
or how to resolve it.


Self-emission
^^^^^^^^^^^^^

Because HIRS is warm and calibration only occurs every 40 scanlines, a
self-emission model is essential; see the :mod:`models` module and the
:class:`~models.RSelfTemperature` class.  The standard implementation of
FCDR\_HIRS updates the self-emission parameters every 24 hours.  In practice,
there are several problems with the self-emission model:

- It appears to lead to larger day-to-day instabilities than the previous,
  L1B HIRS.  Viju John has details on this.
- The uncertainty model with the self-emission model is too simple.
  Currently, this uncertainty is derived from the RMSE between the
  model and validation data.  See issues :issue:`36` and :issue:`64`.
  In reality, this uncertainty should be split into uncertainty on the
  various measurement equation components for the self-emission model:
  on the temperatures used for the model, on the parameters derived on
  the model, and the model uncertainty itself.  Currently all uncertainty
  is the model uncertainty.
- The error correlations associated with the self-emission model uncertainty
  are not realistic.  See issue :issue:`226`.
- The temperature-based self-emission model has room for improvement.
  For example, it needs to be more robust and resistant (see issues
  :issue:`1`, :issue:`105`, :issue:`132`, :issue:`144`, :issue:`164`, and
  :issue:`243`).
- Several smaller problems, more related to bookkeeping.  For a complete
  overview, see https://github.com/FIDUCEO/FCDR_HIRS/labels/self-emission

A relatively simple replacement for the temperature-based self-emission
model would be a basic interpolation self-emission model.  This would
likely be more robust and more stable, but I'm not sure how to estimate
the uncertainties, and it wouldn't help at all for estimating the error
correlations.

.. _outliers:
Undetected outliers
^^^^^^^^^^^^^^^^^^^

Although there is considerable code for filtering out outliers (for
example, :attr:`~fcdr.HIRSFCDR.filter_earthcounts`,
:attr:`~fcdr.HIRSFCDR.filter_calibcounts`, and
:attr:`~fcdr.HIRSFCDR.filter_prtcounts`, which are all implementations of
:class:`~typhon.datasets.filters.MEDMAD`), there are still significant
problems with undetected outliers in Earth counts, calibration counts, PRT
counts, and time.  These propagate into the Earth radiances, into
uncertainties, into the self-emission model, into the harmonisation, and
probably elsewhere as well.  See issues :issue:`15`, :issue:`144`,
:issue:`163`, :issue:`167`, :issue:`194`, :issue:`287`, and :issue:`365`.

Documentation
^^^^^^^^^^^^^

The documentation is incomplete.  Although all modules have some
docstrings, my aim in my final weeks was to improve the docstrings
throughout.  I have not completed this.  If you browse through the
documentation, you will find that some modules, including all the modules
directly in the :mod:`FCDR_HIRS` package, are rather well documented,
whereas other modules, including all in the :mod:`FCDR_HIRS.processing`
and some in the :mod:`FCDR_HIRS.analysis` packages, have a lower level of
documentation with some functions and classes lacking any docstrings.

Validation
^^^^^^^^^^

Neither the HIRS FCDR brightness temperatures nor its uncertainties have
currently been validated.  A validated CDR could potentially be used
to validate the FCDR including both brightness temperatures and
uncertainties.  Viju John has code that can be used for stability testing.

Causes of delay
^^^^^^^^^^^^^^^

The HIRS FCDR was originally meant to be delivered in January 2017, but
remains unfinished in January 2019.  Some of the causes of the delay
include:

- Initial learning curve
- Lack of PyGAC equivalent
- A lot of bad L1B data
- Difficult harmonisation
- Difficult self-emission
- Availability of project partners to contribute to HIRS work
