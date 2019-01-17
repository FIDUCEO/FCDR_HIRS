.. _cmdline:
Command-line scripts
====================

This is an overview of command-line utilities that are available when
:doc:`FCDR_HIRS` has been successfully installed.
This page describes a summary of the behaviour of each of the utilities,
as well as an overview of mandatory and optional command-line flags.
For each script, you will get the same interview by passing the
``--help`` flag.
All utilities are non-interactive and therefore suitable to use on the
cluster using the load scheduler.

FCDR generation
---------------

.. _generate-fcdr:

generate\_fcdr
^^^^^^^^^^^^^^

Implemented in :mod:`FCDR_HIRS.processing.generate_fcdr`.

.. argparse::
    :module: FCDR_HIRS.processing.generate_fcdr
    :func: get_parser
    :prog: generate_fcdr

Matchup / harmonisation processing
----------------------------------

.. _combine-hirs-hirs-matchups:

combine_hirs_hirs_matchups
^^^^^^^^^^^^^^^^^^^^^^^^^^

Implemented in :mod:`FCDR_HIRS.processing.combine_matchups`.

.. argparse::
    :module: FCDR_HIRS.processing.combine_matchups
    :func: get_parser_hirs
    :prog: combine_hirs_hirs_matchups


.. _combine-hirs-iasi-matchups:

combine_hirs_iasi_matchups
^^^^^^^^^^^^^^^^^^^^^^^^^^

Implemented in :mod:`FCDR_HIRS.processing.combine_matchups`.

.. argparse::
    :module: FCDR_HIRS.processing.combine_matchups
    :func: get_parser_iasi
    :prog: combine_hirs_iasi_matchups


write_hirs_harm_meta
^^^^^^^^^^^^^^^^^^^^

Basic script without command-line arguments, implemented in
:mod:`FCDR_HIRS.analysis.write_harm_meta`.

.. automodule:: FCDR_HIRS.analysis.write_harm_meta

merge_hirs_harmonisation
^^^^^^^^^^^^^^^^^^^^^^^^

Implemented in :mod:`FCDR_HIRS.processing.combine_matchups`.

.. argparse::
    :module: FCDR_HIRS.processing.combine_matchups
    :func: get_parser_merge
    :prog: merge_hirs_harmonisation

.. _convert-hirs-harmonisation-parameters:

convert_hirs_harmonisation_parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Implemented in :mod:`FCDR_HIRS.processing.convert_harm_params`.

.. argparse::
    :module: FCDR_HIRS.processing.convert_harm_params
    :func: get_parser
    :prog: convert_hirs_harmonisation_parameters

L1B analysis
------------

.. _plot-hirs-field-timeseries:

plot_hirs_field_timeseries
^^^^^^^^^^^^^^^^^^^^^^^^^^

Implemented in :mod:`FCDR_HIRS.analysis.timeseries`.

.. argparse::
    :module: FCDR_HIRS.analysis.timeseries
    :func: get_parser
    :prog: plot_hirs_field_timeseries

.. _plot-hirs-field-matrix:

plot_hirs_field_matrix
^^^^^^^^^^^^^^^^^^^^^^

Implemented in :mod:`FCDR_HIRS.analysis.fieldmat`.

.. argparse::
    :module: FCDR_HIRS.analysis.fieldmat
    :func: get_parser
    :prog: plot_hirs_field_matrix

plot_hirs_calibcounts_per_scanpos
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Implemented in :mod:`FCDR_HIRS.analysis.calibcounts_stats_per_scanpos`.

.. argparse::
    :module: FCDR_HIRS.analysis.calibcounts_stats_per_scanpos
    :func: get_parser
    :prog: plot_hirs_calibcounts_per_scanpos

plot_hirs_flags
^^^^^^^^^^^^^^^

Implemented in :mod:`FCDR_HIRS.analysis.plot_flags`.

.. argparse::
    :module: FCDR_HIRS.analysis.plot_flags
    :func: get_parser
    :prog: plot_hirs_flags

.. _hirs-info-content:

hirs_info_content
^^^^^^^^^^^^^^^^^

Basic script without command-line arguments.  Implemented in
:mod:`FCDR_HIRS.analysis.corrmat_info_content`.

.. automodule:: FCDR_HIRS.analysis.corrmat_info_content

.. _hirs-convert-l1b-to-l1c:

hirs_convert_l1b_to_l1c
^^^^^^^^^^^^^^^^^^^^^^^

Implemented in :mod:`FCDR_HIRS.analysis.convert_hirs_l1b_to_nc`.

.. argparse::
    :module: FCDR_HIRS.analysis.convert_hirs_l1b_to_nc
    :func: get_parser
    :prog: hirs_convert_l1b_to_l1c

.. _srf-recovery:

hirs_iasi_srf_recovery
^^^^^^^^^^^^^^^^^^^^^^

Implemented in :mod:`FCDR_HIRS.analysis..hirs_iasi_srf_estimation`.

.. argparse::
    :module: FCDR_HIRS.analysis.hirs_iasi_srf_estimation
    :func: get_parser
    :prog: hirs_iasi_srf_recovery

FCDR analysis
-------------

map_hirs_field
^^^^^^^^^^^^^^

Implemented in :mod:`FCDR_HIRS.analysis.map`.

.. argparse::
    :module: FCDR_HIRS.analysis.map
    :func: get_parser
    :prog: map_hirs_field

plot_hirs_fcdr
^^^^^^^^^^^^^^

Implemented in :mod:`FCDR_HIRS.analysis.monitor_fcdr`.

.. argparse::
    :module: FCDR_HIRS.analysis.monitor_fcdr
    :func: get_parser
    :prog: plot_hirs_fcdr

.. _summarise-fcdr:
summarise_hirs_fcdr
^^^^^^^^^^^^^^^^^^^

Implemented in :mod:`FCDR_HIRS.analysis.summarise_fcdr`.

.. argparse::
    :module: FCDR_HIRS.analysis.summarise_fcdr
    :func: get_parser
    :prog: summarise_hirs_fcdr

hirs_orbit_map
^^^^^^^^^^^^^^

Implemented in :mod:`FCDR_HIRS.analysis.map_single_orbit`.

.. argparse::
    :module: FCDR_HIRS.analysis.map_single_orbit
    :func: get_parser
    :prog: hirs_orbit_map

.. _hirs-logfile-analysis:

hirs_logfile_analysis
^^^^^^^^^^^^^^^^^^^^^

Basic script without commandline arguments.  Implemented in
:mod:`FCDR_HIRS.analysis.logfile_analysis`.

.. automodule:: FCDR_HIRS.analysis.logfile_analysis

Matchups analysis
-----------------

inspect_hirs_matchups
^^^^^^^^^^^^^^^^^^^^^

Implemented in :mod:`FCDR_HIRS.analysis.inspect_hirs_matchups`.

.. argparse::
    :module: FCDR_HIRS.analysis.inspect_hirs_matchups
    :func: get_parser
    :prog: inspect_hirs_matchups

hirs_inspect_harm_matchups
^^^^^^^^^^^^^^^^^^^^^^^^^^

Implemented in :mod:`FCDR_HIRS.analysis.inspect_hirs_harm_matchups`.

.. argparse::
    :module: FCDR_HIRS.analysis.inspect_hirs_harm_matchups
    :func: get_parser
    :prog: hirs_inspect_harm_matchups

Testing algorithms
------------------

.. _plot-hirs-test-rself:

plot_hirs_test_rself
^^^^^^^^^^^^^^^^^^^^

Implemented in :mod:`FCDR_HIRS.analysis.test_rself`.

.. argparse::
    :module: FCDR_HIRS.analysis.test_rself
    :func: get_parser
    :prog: plot_hirs_test_rself

hirs_curuc_checker
^^^^^^^^^^^^^^^^^^

Implemented in :mod:`FCDR_HIRS.analysis.inspect_orbit_curuc`.

.. argparse::
    :module: FCDR_HIRS.analysis.inspect_orbit_curuc
    :func: get_parser
    :prog: hirs_curuc_checker

Other
-----

calc_sensitivity_params
^^^^^^^^^^^^^^^^^^^^^^^

Implemented in :mod:`FCDR_HIRS.analysis.sensitivities`.

.. argparse::
    :module: FCDR_HIRS.analysis.sensitivities
    :func: get_parser
    :prog: calc_sensitivity_params

.. _determine-hirs-latlon-compression-ratio:

determine_hirs_latlon_compression_ratio
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Basic script without commandline arguments.  Implemented in
:mod:`FCDR_HIRS.analysis.determine_latlon_compression_ratio`.

.. automodule:: FCDR_HIRS.analysis.determine_latlon_compression_ratio

.. _determine-hirs-unc-storage:

determine_hirs_unc_storage
^^^^^^^^^^^^^^^^^^^^^^^^^^

Basic script without commandline arguments.  Implemented in
:mod:`FCDR_HIRS.analysis.determine_optimal_uncertainty_format`.

.. automodule:: FCDR_HIRS.analysis.determine_optimal_uncertainty_format
