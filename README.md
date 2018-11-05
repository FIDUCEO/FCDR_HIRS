Development code for HIRS FCDR uncertainties.

Upon installation with pip, all the Python-based dependencies should be
installed automatically, except that in some cases you may need the latest
master.  Not everything is in conda sa you may have to install some things
from git.  FCDRTools needs to be installed from git master branch,
possibly typhon too.  pyatmlab is on pypi but not on conda.  In addition, you will need:

- HIRS L1B data in NOAA format, obtainable from the NOAA CLASS archive.
- spectral response functions that come with RTTOV
  Note that a current version
  temporarily uses band correction factors that are not included with
  ARTS.  Contact Gerrit Holl <g.holl@reading.ac.uk> or Jon Mittaz
  <j.mittaz@reading.ac.uk> to get those.
- A configuration file indicating where different datasets and SRFs are located.
  Set the environment variable TYPHONRC to its path.  See 
  `typhon documentation <http://www.radiativetransfer.org/misc/typhon/doc/>`
  for details.

After installation, command-line utilities include:

    plot_hirs_field_timeseries
    inspect_hirs_matchups
    map_hirs_field
    plot_hirs_temp_matrix
    plot_hirs_calibcounts_per_scanpos
    plot_hirs_test_rself
    calc_sensitivity_params
    combine_hirs_hirs_matchups
    combine_hirs_iasi_matchups
    generate_fcdr
    plot_hirs_fcdr
    summarise_hirs_fcdr
    determine_hirs_latlon_compression_ratio
    determine_hirs_unc_storage
    plot_hirs_flags
    hirs_info_content
    hirs_logfile_analysis
    hirs_orbit_map
    write_hirs_harm_meta
    merge_hirs_harmonisation
    convert_hirs_harmonisation_parameters
    hirs_curuc_checker
    hirs_convert_l1b_to_l1c
    hirs_iasi_srf_recovery
    hirs_inspect_harm_matchups

Most of those have an online help, i.e. plot_hirs_field_timeseries --help,
listing all the options and capabilities.

To get started with FCDR generation, analysis, or anything else, you need
to tell it where the files are.  This happens through the typhon
dependency.  You will need to create a .typhonrc file and tell typhon
where it is through the environment variable TYPHONRC.

The steps to generate a new version of the FCDR:

- (devel) Switch to correct branch, rebase on master, and install

- Generate an unharmonised FCDR by passing the --no-harm flag to
  generate_fcdr.

- (devel) Switch to correct branch, rebase on master, and install

- Use hirs_logfile_analysis script to inspect outcome

- (devel) Switch to correct branch, rebase on master, and install

- From the unharmonised FCDR, create an unfiltered harmonisation database,
  by calling combine_hirs_hirs_matchups with the --without-filters flag.

- Combine the small harmonisatin files into single big ones per pair and
  channel, using merge_hirs_harmonisation

- (devel) Switch to correct branch, rebase on master, and install

- From the unfiltered harmonisation database, derive filter parameters by
  calling hirs_inspect_harm_matchups with the --write-filters flag

- (devel) Switch to correct branch, rebase on master, and install

- Generate a new harmonisation database, now --with-filters.

- Run merge_hirs_harmonisation on the same

- Run the harmonisation.  Ralf Quast <ralf.quast@fastopt.de> can do this.

- Convert the harmonisation parameters to the Python file _harm_defs.py
  using convert_hirs_harmonisation_parameters

- (devel) Switch to correct branch, rebase on master, and install

- Run the FCDR with harmonisation

- (devel) Switch to correct branch, rebase on master, and install

- Use hirs_logfile_analysis script to inspect outcome

- (devel) Switch to correct branch, rebase on master, and install

- Use summarise_hirs_fcdr with 'summarise' mode to generate summarising
  statistic

- Use summarise_hirs_fcdr with 'plot' mode to visualise summarising
  statistic

- For shorter periods of plotting, use plot_hirs_fcdr (short time series)
  and hirs_orbit_map (maps)
