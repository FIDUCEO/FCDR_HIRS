"""Common utility functions between scripts

Should probably be sorted into smaller modules.
"""

import sys
import logging
import datetime
import warnings
import numpy
import xarray
from .fcdr import list_all_satellites

def add_to_argparse(parser,
        include_period=True,
        include_sat=0,
        include_channels=True,
        include_temperatures=False,
        include_debug=False):
    """Add commoners to argparse object
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
            help="End date/time")

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


def time_epoch_to(ds, epoch):
    """Convert all time variables/coordinates to count from epoch
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
    """Sample flags

    For a flag field, estimate percentage during which flag is set each
    period (default 1H)

    Must have .flag_masks and .flag_meanings following
    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/build/ch03s05.html

    Arguments:

        da

            must have flag_masks and flag_meanings attributes

        period

        dim

    Returns:

        (percs, labels)
    """

    flags = da & xarray.DataArray(numpy.atleast_1d(da.flag_masks), dims=("flag",))
    perc = (100*(flags!=0)).resample(period, dim=dim, how="mean")
    # deeper dimensions
    for d in set(perc.dims) - {dim, "flag"}:
        perc = perc.mean(dim=d)
    
    return (perc, da.flag_meanings.split())


_loggers_set = set()
def set_logger(level, filename=None, loggers=None):
    """Set propertios of FIDUCEO root logger

    Arguments:

        level

            What loglevel to use.

        filename

            What file to log to.  None for sys.stderr.

        loggers

            What loggers to set.  Default only sets the "FCDR_HIRS"
            logger, but you may want to set others like "typhon".
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
