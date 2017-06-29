Development code for HIRS FCDR uncertainties.

Upon installation with pip, all the Python-based dependencies should be
installed automatically.  In addition, you will need:

- HIRS L1B data in NOAA format, obtainable from the NOAA CLASS archive.
- spectral response functions that come with
  `ARTS <http://www.radiativetransfer.org>`_.  Not that a current version
  temporarily uses band correction factors that are not included with
  ARTS.  Contact Gerrit Holl <g.holl@reading.ac.uk> to get those.
- A configuration file indicating where different datasets and SRFs are located.
  Set the environment variable TYPHONRC to its path.  See 
  `typhon documentation <http://www.radiativetransfer.org/misc/typhon/doc/>`
  for details.

Before the first run you may have to update the firstline db::

    import FCDR_HIRS.fcdr
    hma = FCDR_HIRS.fcdr.HIRS4FCDR(name="metopa", satname="metopa")
    hma.update_firstline_db()

After installation, command-line utilities include:

    plot_hirs_field_timeseries
    inspect_hirs_matchups
    map_hirs_field
    plot_hirs_temp_matrix
    plot_hirs_calibcounts_per_scanpos
    plot_hirs_test_rself
    calc_sensitivity_params
    combine_matchups
    generate_fcdr
    plot_hirs_fcdr
    summarise_hirs_fcdr
    determine_hirs_latlon_compression_ratio
    determine_hirs_unc_storage

Most of those have an online help, i.e. plot_hirs_field_timeseries --help,
listing all the options and capabilities.

To get started with FCDR generation, analysis, or anything else, you need
to tell it where the files are.  This happens through the typhon
dependency.  You will need to create a .typhonrc file and tell typhon
where it is through the environment variable TYPHONRC.
