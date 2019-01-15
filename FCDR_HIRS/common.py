"""Common utility functions between scripts

This module contains various small utility functions that are used in
various other methods and scripts.

Some of the functions in here may be sorted into their own modules in the
future.
"""

import sys
import logging
import datetime
import warnings
import traceback
import io
import pprint
import inspect
import numpy
import xarray
import progressbar

#: progressbar widget used to display progress for various reading tasks
my_pb_widget = [progressbar.Bar("=", "[", "]"), " ",
                progressbar.Percentage(), " (",
                progressbar.AdaptiveETA(), " -> ",
                progressbar.AbsoluteETA(), ') ']

def add_to_argparse(parser,
        include_period=True,
        include_sat=0,
        include_channels=True,
        include_temperatures=False,
        include_debug=False):
    """Add commoners to argparse object.

    Helper function to add flags to :class:`argparse.ArgumentParser`
    objects, where the exact same flags are occurring in multiple scripts.
    The flags ``--verbose`` (taking a `bool`) and ``--log`` are always added as optional
    flags.  Other flags are only added if the corresponding parameter to
    this function is `True`.  You should call this function at most once
    for any one parser.

    Parameters
    ----------

    parser : :class:`~argparse.ArgumentParser` object
        The :class:`~argparse.ArgumentParser` object to which the arguments
        shall be added.
    include_period : bool, optional
        Include the mandatory arguments ``from_date`` and ``to_date``, as well
        as the optional argument ``--datefmt``.  Use this for a script
        operating over a certain time period, such as FCDR generation or
        plotting scripts.  Defaults to `True`.
    include_sat : {0, 1, 2}, optional
        Include 0, 1, or 2 satellite name arguments, which will be
        mandatory.  The valid options as satellite names will be taken
        from :func:`list_all_satellites`.  For a single satellite,
        the option will be called ``satname``.  If there are two, they
        will be called ``satname1`` and ``satname2``.  Defaults to 0.
    include_channels : bool, optional
        Include an ``--channels`` flag, defaulting to a list of channels 1--19.
        This flag will be optional, unless ``include_temperatures`` is also true, in
        which case both ``include_channels`` and ``include_temperatures`` will
        be mandatory.
        Defaults to `True`.
    include_temperatures : bool, optional
        Include a ``--include_temperatures`` flag, that will accept a list
        of strings on what temperatures to be considered.  Defaults to
        `False`.
    include_debug : bool, optional
        Include an optional ``--debug`` parameter taking a bool, that will
        default to `False`.  Defaults to `False`.

    Returns
    -------

    ~argparse.ArgumentParser
        The same parser that went in, but with arguments added.
    """

    if include_sat == 1:
        parser.add_argument("satname", action="store", type=str.lower,
            help="Satellite name",
            metavar="satname",
            choices=sorted(list_all_satellites())+["all"])
    elif include_sat == 2:
        parser.add_argument("satname1", action="store", type=str,
            help="Satellite name, primary",
            metavar="SAT1",
            choices=sorted(list_all_satellites()))

        parser.add_argument("satname2", action="store", type=str,
            help="Satellite name, secondary",
            metavar="SAT2",
            choices=sorted(list_all_satellites()))
    elif include_sat!=0:
        raise ValueError("include_sat should be False, 0, 1, True, or 2. "
            "Got {!s}.".format(include_sat))

    if include_period:
        parser.add_argument("from_date", action="store", type=str,
            help="Start date/time")

        parser.add_argument("to_date", action="store", type=str,
            help="End date/time.  This uses the standards of "
                 "Python date/time intervals, so the end is "
                 "not included and when only a date is given, "
                 "time is assumed to be 00:00:00.  For example, "
                 "to process only 2010-01-01, one should give "
                 "as a start date 2010-01-01 and as an end "
                 "date 2010-01-02, which means end at 2010-01-02T00:00:00 ",
                 "such that zero data on 2010-01-02 are included.")

        parser.add_argument("--datefmt", action="store", type=str,
            help="Date format for start/end dates",
            default="%Y-%m-%d")

    hasboth = include_channels and include_temperatures
    if hasboth:
        required_named = parser.add_argument_group(
            "required named arguments")
        ch_tm = required_named
        regarg = dict(required=True)
    else:
        ch_tm = parser
        regarg = {}

    # http://stackoverflow.com/a/24181138/974555
    if include_channels:
        ch_tm.add_argument(("--" if hasboth else "") + "channels", action="store", type=int,
            nargs="+", choices=list(range(1, 21)),
            default=list(range(1, 20)),
            help="Channels to consider",
            **regarg)

    if include_temperatures:
        ch_tm.add_argument(("--" if hasboth else "") + "temperatures", action="store", type=str,
            nargs="+",
            choices=['an_el', 'patch_exp', 'an_scnm', 'fwm', 'scanmotor',
                'iwt', 'sectlscp', 'primtlscp', 'elec', 'baseplate',
                'an_rd', 'an_baseplate', 'ch', 'an_fwm', 'ict', 'an_pch',
                'scanmirror', 'fwh', 'patch_full', 'fsr'],
            help="Temperature fields to use",
            **regarg)

    parser.add_argument("--verbose", action="store_true",
        help="Be verbose", default=False)

    parser.add_argument("--log", action="store", type=str,
        help="Logfile to write to.  Leave out for stdout.")

    if include_debug:
        parser.add_argument("--debug", action="store_true",
            help="Add extra debugging information", default=False)

    return parser


def time_epoch_to(ds: xarray.Dataset,
                  epoch: datetime.datetime):
    """Convert all time variables/coordinates to count from epoch

    For the :class:`xarray.Dataset` object ``ds``, change the encoding for
    all variables (data or coordinate variables) with a
    ``numpy.datetime64`` dtype to one that counds seconds since the
    `datetime` expressed by `epoch`.
    This function does not take care that this actually fits, so you may
    want to adapt ``ds.encoding["dtype"]`` and
    ``ds.encoding["scale_factor"]`` yourself.

    Parameters
    ----------

    ds : xarray.Dataset
        Dataset for which to adapt the time-based fields.
    epoch : ~datetime.datetime
        Datetime to set as the new epoch.

    Returns
    -------

    xarray.Dataset
        Dataset with zero or more time-based fields encodings adapted.
    """

    for k in [k for (k, v) in ds.variables.items() if v.dtype.kind.startswith("M")]:
        ds[k].encoding["units"] = "seconds since {:%Y-%m-%d %H:%M:%S}".format(epoch)
        if ds[k].size > 0:
            ds[k].encoding["add_offset"] = (
                ds[k].min().values.astype("M8[ms]").astype(datetime.datetime)
                - epoch).total_seconds()
        else:
            ds[k].encoding["add_offset"] = 0
    return ds


def sample_flags(da, period="1H", dim="time"):
    """Calculate frequency of flags on/off per time period

    For a data variable that is a CF-compliant
    `flag field`_, calculate the frequency per unit time of its occurrence.

    Must have .flag_masks and .flag_meanings following CF-compliant
    `flag field`_.

    .. _flag field: <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/build/ch03s05.html>

    Parameters
    ----------

    da : ~xarray.DataArray
        DataArray which is a CF-compliant flag field with the ``flag_masks``
        and ``flag_meanings``, for which the frequency of occurrence per
        unit time is calculated.  The DataArray must have a time
        dimension.
    period : str, optional
        Temporal resolution over which the flags are resampled.  Valid
        strings are as for `pandas`_.  Defaults to "1H".

        .. _pandas: http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
    dim : str, optional
        Name of the time dimension.  Defaults to "time".

    Returns
    -------

    ~xarray.DataArray
        DataArray with a dimensions of "flag" and time, where the time
        dimension is a regular grid with an interval of ``period``, which
        indicates for each flag and time the percentage of this time
        period for which the flag is set.
    List[`str`]
        Meanings for each of the flags, taken directly from
        ``da.flag_meanings``.

    Example
    -------

    .. todo::

        Write example section.
    """

    flags = da & xarray.DataArray(numpy.atleast_1d(da.flag_masks), dims=("flag",))
    perc = (100*(flags!=0)).resample(period, dim=dim, how="mean")
    # deeper dimensions
    for d in set(perc.dims) - {dim, "flag"}:
        perc = perc.mean(dim=d)
    
    return (perc, da.flag_meanings.split())


_loggers_set = set()
def set_logger(level, filename=None, loggers=None):
    """Set properties of package-level loggers

    Set handlers for the package-level loggers, such as for :py:mod:`FCDR_HIRS`,
    typhon, or others.  Those handlers enable verbose or regular logging
    with a defined string including the level name, the name, time,
    module, function name, line number, and of course the logging message.

    Parameters
    ----------

    level : int
        What loglevel to use.  Although these are numeric values, you will
        want to use the constants from the `logging` module, such as
        `logging.DEBUG <https://docs.python.org/3/library/logging.html#logging-levels>`_ 
        or logging.INFO.
    filename : str or None, optional
        What file to log to.  `None` will log to `sys.stderr`.  Defaults to `None`.
    loggers : List[str] or List[logging.Logger]
        To what loggers shall the handlers be added?  By default, it only
        adds the handler to the "FCDR_HIRS" logger, but if you want
        debugging from other packages as well, you might want to set those
        as well, for example,["FCDR_HIRS", "typhon"].

    Example
    -------

    .. todo::

        Write example.

    """
    global _root_logger_set
    if loggers is None:
        loggers = {logging.getLogger(__name__).parent}
    loggers = {logging.getLogger(s) if isinstance(s, str) else s
                for s in loggers}
    if filename:
        handler = logging.FileHandler(filename, mode="a", encoding="utf-8")
    else:
        handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(levelname)-8s %(name)s %(asctime)s %(module)s.%(funcName)s:%(lineno)s: %(message)s"))
    for logger in loggers:
        if logger in _loggers_set:
            warnings.warn(f"Logger {logger!s} already configured")
            continue
        logger.setLevel(level)
        logger.addHandler(handler)

def get_verbose_stack_description(first=2, last=-1, include_source=True,
                                    include_locals=True, include_globals=False):
    """Provide a big description of the stack, for debugging purposes

    Provide a big, verbose description of the stack, writing the source of
    the calling function and all subsequent calling functions, along with
    optionally all locals and all globals at each stack level.  Writing
    this out may be useful for debugging purposes in case of a serious
    crash, or perhaps whenever writing a plot so one can always retrieve
    the parameters and some of the code used for plotting.

    Parameters
    ----------

    first : int, optional
        First stacklevel for which to report the source code.  Including
        `inspect.stack` and `get_verbose_stack_description`, the number 2
        corresponds to the function calling this function.  Defaults to 2.
    last: int, optional
        Last stacklevel for which to report the source code.  Defaults to
        -1, which means it's turtles all the way down.
    include_source: bool, optional
        Include the source code for all functions/methods in the stack.
        Defaults to True.
    include_locals: bool, optional
        Include the locals for all functions/methods in the stack.
        Defaults to True.
    include_globals: bool, optional
        Include the globals for all functions/methods in the stack.
        Defaults to False.

    Returns
    -------

    str
        Verbose description of the stack.

    Example
    -------

    .. todo::

        Write example.
    """
    f = io.StringIO()
    f.write("".join(traceback.format_stack()))
    for fminfo in inspect.stack()[first:last]:
        frame = fminfo.frame
        try:
            f.write("-" * 60 + "\n")
            if include_source:
                try:
                    f.write(inspect.getsource(frame) + "\n")
                except OSError:
                    f.write(str(inspect.getframeinfo(frame)) + 
                         "\n(no source code)\n")
            if include_locals:
                f.write(pprint.pformat(frame.f_locals) + "\n")
            if include_globals:
                f.write(pprint.pformat(frame.f_globals) + "\n")
        finally:
            try:
                frame.clear()
            except RuntimeError:
                pass
    return f.getvalue()

def savetxt_3d(fname, data, *args, **kwargs):
    """Write 3D-array to file that pgfplots likes

    For a 3d ndarray, write text to a file separated by empty lines.  This
    can be interpreted by `pgfplots <http://pgfplots.sourceforge.net/>`_
    for plotting directly in a LaTeX document.

    Parameters
    ----------

    fname : str
        Filename to write the plotdata to.  You might want to get this
        from `plotdatadir`.
    data : ~numpy.ndarray
        3-dimensional `numpy.ndarray` containing the data to be written to
        a file.
    *args
        Remaining arguments passed to :func:`~numpy.savetxt`
    **kwargs
        Remaining arguments passed to :func:`~numpy.savetxt`.

    Example
    -------

    .. todo::

        Write example.
    """
    with open(fname, 'wb') as outfile:
        for data_slice in data:
            numpy.savetxt(outfile, data_slice, *args, **kwargs)
            outfile.write(b"\n")

def plotdatadir() -> str:
    """Returns todays plotdatadir.

    Configuration ``plotdatadir`` in section ``main`` must be set.
    The value is expanded with `datetime.datetime.strftime`.

    Returns
    -------

    str
        Plot data directory corresponding to today.

    Example
    -------

    .. todo::

        Write example.
    """
    return datetime.date.today().strftime(
        config.conf["main"]["plotdatadir"])

def list_all_satellites() -> set:
    """Return a set of all satellites

    Returns
    -------

    set
        Set of all accepted variations of names of all satellites
        considered in the FIDUCEO-FCDR.
    """
    return {'ma', 'mb', 'metop_1', 'metop_2', 'metopa', 'metopb', 'n-05',
            'n-06', 'n-07', 'n-08', 'n-09', 'n-10', 'n-11', 'n-12',
            'n-13', 'n-14', 'n-15', 'n-16', 'n-17', 'n-18', 'n-19', 'n-5',
            'n-6', 'n-7', 'n-8', 'n-9', 'n05', 'n06', 'n07', 'n08', 'n09',
            'n10', 'n11', 'n12', 'n13', 'n14', 'n15', 'n16', 'n17', 'n18',
            'n19', 'n5', 'n6', 'n7', 'n8', 'n9', 'n_05', 'n_06', 'n_07',
            'n_08', 'n_09', 'n_10', 'n_11', 'n_12', 'n_13', 'n_14',
            'n_15', 'n_16', 'n_17', 'n_18', 'n_19', 'n_5', 'n_6', 'n_7',
            'n_8', 'n_9', 'noaa-05', 'noaa-06', 'noaa-07', 'noaa-08',
            'noaa-09', 'noaa-10', 'noaa-11', 'noaa-12', 'noaa-13',
            'noaa-14', 'noaa-15', 'noaa-16', 'noaa-17', 'noaa-18',
            'noaa-19', 'noaa-5', 'noaa-6', 'noaa-7', 'noaa-8', 'noaa-9',
            'noaa05', 'noaa06', 'noaa07', 'noaa08', 'noaa09', 'noaa10',
            'noaa11', 'noaa12', 'noaa13', 'noaa14', 'noaa15', 'noaa16',
            'noaa17', 'noaa18', 'noaa19', 'noaa5', 'noaa6', 'noaa7',
            'noaa8', 'noaa9', 'noaa_05', 'noaa_06', 'noaa_07', 'noaa_08',
            'noaa_09', 'noaa_10', 'noaa_11', 'noaa_12', 'noaa_13',
            'noaa_14', 'noaa_15', 'noaa_16', 'noaa_17', 'noaa_18',
            'noaa_19', 'noaa_5', 'noaa_6', 'noaa_7', 'noaa_8', 'noaa_9',
            'tirosn', 'tn'}
