Development code for HIRS FCDR uncertainties
============================================

Installation
------------

To install FCDR\_HIRS, you need to install the dependencies first.
Almost all dependencies are installable through pip or conda. The only
exception is (currently) FCDRTools, which you need to obtain from the
[FIDUCEO Github](https://github.com/FIDUCEO/FCDRTools). As long as
FCDR\_HIRS is under active development, you may also need to install the
latest master for the [typhon
Github](https://github.com/atmtools/typhon/). Eventually, we should make
both FCDR\_HIRS and FCDRTools installable under its own conda channel so
that the FCDR\_HIRS installation is as easy as
`conda install FCDR_HIRS`, but we are not there yet.

Currently, you also need to manually install:

-   HIRS L1B data in NOAA format, obtainable from the NOAA CLASS
    archive.
-   spectral response functions that come with RTTOV Note that a current
    version temporarily uses band correction factors that are not
    included with ARTS. Contact Gerrit Holl <g.holl@reading.ac.uk> or
    Jon Mittaz <j.mittaz@reading.ac.uk> to get those.
-   A configuration file indicating where different datasets and SRFs
    are located. Set the environment variable TYPHONRC to its path. See
    `typhon documentation <http://www.radiativetransfer.org/misc/typhon/doc/>`
    for details.

Later, all of those will be included with FCDR\_HIRS.

Command-line scripts
--------------------

After installation, command-line utilities include:

-   plot\_hirs\_field\_timeseries
-   inspect\_hirs\_matchups
-   map\_hirs\_field
-   plot\_hirs\_temp\_matrix
-   plot\_hirs\_calibcounts\_per\_scanpos
-   plot\_hirs\_test\_rself
-   calc\_sensitivity\_params
-   combine\_hirs\_hirs\_matchups
-   combine\_hirs\_iasi\_matchups
-   generate\_fcdr
-   plot\_hirs\_fcdr
-   summarise\_hirs\_fcdr
-   determine\_hirs\_latlon\_compression\_ratio
-   determine\_hirs\_unc\_storage
-   plot\_hirs\_flags
-   hirs\_info\_content
-   hirs\_logfile\_analysis
-   hirs\_orbit\_map
-   write\_hirs\_harm\_meta
-   merge\_hirs\_harmonisation
-   convert\_hirs\_harmonisation\_parameters
-   hirs\_curuc\_checker
-   hirs\_convert\_l1b\_to\_l1c
-   hirs\_iasi\_srf\_recovery
-   hirs\_inspect\_harm\_matchups

All of those have an online help, for example, `generate_fcdr --help`
will show the options and capabilities for this script.

Configuration
-------------

To get started with FCDR generation, analysis, or anything else, you
need to tell it where the files are. This happens through the typhon
dependency. You will need to create a .typhonrc file and tell typhon
where it is through the environment variable TYPHONRC.

Generating an FCDR
------------------

Generating a new version of the FCDR contains a number of steps.
Usually, when a new version is generated, this means the code and/or its
dependencies are under active development. Therofer, the listing below
also includes steps marked with "(devel)", which are only relevant when
the code is under active development.

-   (devel) Switch to correct branch, rebase on master, and install.
    Perhaps need to create a dedicated branch and merge multiple
    development branches (verify with git branch --merged). You may need
    to do the same for typhon and FCDRTools if those have been changed.
-   Generate an unharmonised FCDR by running `generate_fcdr --no-harm`
    along with other flags. You must generate both the easy and the
    debug FCDR at this stage, because the harmonisation input file
    generator needs information from the debug FCDR. You probably want
    to pass the --abridged flag to `generate_fcdr` to save space. This
    will skip the `u_from` and `rad_wn` debug fields, saving about 88%
    in size per FCDR file. If running on Jasmin/CEMS, the shell script
    `submit_all_generate_fcdr.sh` can be used to submit jobs for all
    satellites and months to LOTUS. The script is available from Gerrit,
    on CEMS, or on bitbucket. You will need to edit it first to update
    paths and such. It may be added to the FCDR\_HIRS distribution
    later.
-   (devel) Switch to correct branch, rebase on master, and install.
-   Use the `hirs_logfile_analysis` script to study the output of the
    logfiles. This assumes logfiles of a structure such as generated on
    CEMS using Gerrits shell scripts.
-   (devel) Switch to correct branch, rebase on master, and install.
-   From the unharmonised FCDR, create an unfiltered harmonisation
    database, by calling `combine_hirs_hirs_matchups` with the
    `--without-filters` flag. This step needs the debug FCDR. An
    abridged debug FCDR is sufficient. When working on CEMS, the shell
    script `submit_all_combine_hirs_matchups.sh` can be edited to submit
    jobs to LOTUS.
-   Combine the small harmonisation files into single big ones per pair
    and channel, using `merge_hirs_harmonisation`. On CEMS, the shell
    script `submit_all_merge_harmonisation_files.sh` can be edited for
    job submission.
-   (devel) Switch to correct branch, rebase on master, and install.
-   From the unfiltered harmonisation database, derive filter parameters
    by calling `hirs_inspect_harm_matchups` with the `--write-filters`
    flag. You may want to inspect the resulting plots. On CEMS, the the
    script `submit_all_plot_harm_matchups.sh` can be edited for job
    submission.
-   (devel) Switch to correct branch, rebase on master, and install.
-   Generate a new harmonisation database, now passing `--with-filters`
    to `combine_hirs_hirs_matchups`. On CEMS you can again use an edited
    version of `submit_all_combine_hirs_matchups.sh`.
-   Run `merge_hirs_harmonisation` again to generate one file of
    filtered harmonisation input files per sensor pair. You can again
    use `submit_all_merge_harmonisation_files.sh`.
-   Inspect the results with `hirs_inspect_harm_matchups`, this time
    *without* passing `--write-filters` or the filters will be
    incorrectly overwritten, as filter derivation assumes unfiltered
    input. You can again use `submit_all_plot_harm_matchups.sh` or just
    look at selected pairs only.
-   Run the harmonisation. [Ralf Quast](ralf.quast@fastopt.de) can do
    this, but he has also left instructions to others on how to do so.
-   Write the harmonisation parameters to the Python file
    `_harm_defs.py` in the git source tree, using the script
    `convert_hirs_harmonisation_parameters`.
-   (devel) Switch to correct branch, rebase on master, and install
-   Generate the FCDR with harmonisation using `generate_fcdr`, this
    time *without* passing the `--no-harm` flag. You can again edit
    `submit_all_generate_fcdr.sh` for your needs on CEMS.
-   (devel) Switch to correct branch, rebase on master, and install.
-   Use the `hirs_logfile_analysis` script again to inspect the logfiles
    for major problems.
-   (devel) Switch to correct branch, rebase on master, and install.
-   If needed, rerun (partially) failed jobs.
-   (devel) Switch to correct branch, rebase on master, and install.
-   Run `summarise_hirs_fcdr --mode summarise` and appropriate flags to
    generate summary statistics. On CEMS, you can edit
    `sumbit_all_summarise_fcdr.sh` for job submission.
-   Run `summarise_hirs_fcdr --mode plot` and appropriate flags to
    generate plots. to generate with 'plot' mode to visualise
    summarising statistics.
-   For shorter periods of plotting, you can use `plot_hirs_fcdr` for
    short time series and `hirs_orbit_map` for orbit maps. You can edit
    the job submission scripts `submit_all_plot_fcdr_random_orbits.sh`
    and `submit_all_plot_fcdr.sh` for your needs on CEMS.
-   Possibly generate yet another set of enhanced matchups, such that we
    can look at matchup-derived statistics on the resulting FCDR.

