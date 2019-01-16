Configuration
-------------

To get started with FCDR generation, analysis, or anything else, you
need to tell it where the files are. This happens through the typhon
dependency. You will need to create a ``.typhonrc`` file and tell typhon
where it is through the environment variable ``TYPHONRC``.

The ``.typhonrc`` file follows a :mod:`configparser` syntax and describes
the location of where input data are read, where output data are written,
where temporary files are written, where plots are written, and various
other items.  It consists of a ``[main]`` part with general locations,
and then one part for various different datasets.  Within each of the
dataset-specific sections, the variables defined correspond to attributes
defined within the classes.  For example, the section on ``[fcdr_hirs]``
corresponds to variables defined in the :class:`~FCDR_HIRS.fcdr.HIRSFCDR`
classes, and within this section, the variable ``write_basedir``
corresponds to the :attr:`~FCDR_HIRS.fcdr.HIRSFCDR.write_basedir`
attribute, which defines where the data are written.  For details on all
the other dataset-specific variables, refer to the attributes documented
within each class.

As an example, my ``.typhonrc`` file looks like this::

    [main]
    homedir = /home/users/gholl
    plotdir = ${main:homedir}/plots/%Y/%m/%d
    plotdatadir = ${main:homedir}/plotdata/%Y/%m
    fiduceo = /gws/nopw/j04/fiduceo
    myfiduceodir = ${main:fiduceo}/Users/gholl
    fiddatadir = ${main:fiduceo}/Data
    mydatadir = ${main:myfiduceodir}/Data
    artsdir = ${main:homedir}/checkouts/arts
    lookup_table_dir = ${main:fiduceo}/Users/gholl/hirs_lookup_table
    simuldir = ${main:myfiduceodir}/simulations
    tmpdir = /tmp
    tmpdirb = /var/tmp
    myscratchdir = /tmp/scratch
    cachedir = /tmp/cache
    harmfilterparams = ${main:fiddatadir}/Harmonisation_matchups/HIRS_filter_params

    [hirs]
    basedir = ${main:fiddatadir}/HIRS
    subdir = {satname}_hirs_{year:04d}/{month:02d}/{day:02d}
    srf_dir = ${main:artsdir}/controlfiles/instruments/hirs
    srf_backend_response = ${hirs:srf_dir}/{sat}_HIRS.backend_channel_response.xml
    srf_backend_f = ${hirs:srf_dir}/{sat}_HIRS.f_backend.xml
    format_definition_file = ${main:homedir}/checkouts/HIRStoHDF/docs/NWPSAF-MF-UD-003_Formats.pdf
    granules_firstline_file = firstline.db
    srf_rttov = ${main:fiduceo}/Data/HIRS/SRF/rtcoef_{sat:s}_hirs-shifted_srf_ch{ch:>02d}.txt

    [fcdr_hirs]
    write_basedir = ${main:fiduceo}/Data/FCDR/HIRS/v{data_version:s}
    basedir = ${hirs:basedir}
    subdir = ${hirs:subdir}
    srf_backend_response = ${hirs:srf_backend_response}
    srf_backend_f = ${hirs:srf_backend_f}
    band_dir = ${main:fiduceo}/scratch/HIRS_SRF
    band_file = ${fcdr_hirs:band_dir}/{sat:s}_band_coefs.dat
    granules_firstline_file = ${hirs:granules_firstline_file}
    srf_rttov = ${main:fiduceo}/Data/HIRS/SRF/rtcoef_{sat:s}_hirs-shifted_srf_ch{ch:>02d}.txt

    [fcdr_hirs_summary]
    basedir = ${main:myfiduceodir}/scratch/HIRS_summaries
    subdir = {satname:s}/{year:04d}

    [iasinc]
    basedir = ${main:fiddatadir}/IASI
    subdir =
    re = W_XX-EUMETSAT-Darmstadt,HYPERSPECT\+SOUNDING,MetOpA\+IASI_C_EUMP_(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})_\d{5}_eps_o_l1.nc

    # Warning: IASI in the CEMS archive has an extra subdirectory on metop-b,
    # how to handle this?
    [iasi]
    basedir = /neodc/
    subdir = iasi_{satname}/data/l1c/{year:04d}/{month:02d}
    re = IASI_xxx_1C_M(?P<satno>\d{2})_(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})Z_(?P<year_end>\d{4})(?P<month_end>\d{2})(?P<day_end>\d{2})(?P<hour_end>\d{2})(?P<minute_end>\d{2})(?P<second_end>\d{2})Z_N_O_\d{14}Z\.nat\.gz

    [iasisub]
    basedir = ${main:fiduceo}/Data/IASI_subset
    freqfile = ${iasinc:basedir}/frequency.txt

    [hiasi]
    basedir = ${main:fiduceo}/Data/Matchup_Data/HIRS_IASI
    freqfile = ${iasinc:basedir}/frequency.txt

    [mhs_l1c]
    basedir = ${main:fiduceo}/Data/AMSUB_MHS_L1C

    [hirshirs]
    basedir = ${main:fiduceo}/Users/gholl/data/hirshirs

    [era_interim]
    basedir = /badc/ecmwf-era-interim/

To process the FCDR, you will need the sections ``[main]``, ``[[hirs]]``,
and ``[[fcdr_hirs]]``, and you will only need some of the definitions
within ``[[main]]``.  To analyse summaries (see :ref:`summarise-fcdr`),
you will also need ``[fcdr_hirs_summary]``.  For some of the other
analysis scripts, you will also need ``[iasinc]``, ``[iasi]``,
``[[iasisub]]``, ``[hiasi]``, and ``[hirshirs]]``.
