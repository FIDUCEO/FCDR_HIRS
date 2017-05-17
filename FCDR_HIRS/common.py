"""Common utilities between scripts
"""

from .fcdr import list_all_satellites

import datetime

def add_to_argparse(parser,
        include_period=True,
        include_sat=0,
        include_channels=True,
        include_temperatures=False):
    """Add commoners to argparse object
    """

    if include_sat == 1:
        parser.add_argument("satname", action="store", type=str,
            help="Satellite name",
            metavar="satname",
            choices=sorted(list_all_satellites()))
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

    return parser


def time_epoch_to(ds, epoch):
    """Convert all time variables/coordinates to count from epoch
    """

    for k in [k for (k, v) in ds.items() if v.dtype.kind.startswith("M")]:
        ds[k].encoding["units"] = "seconds since {:%Y-%m-%d %H:%M:%S}".format(epoch)
        ds[k].encoding["add_offset"] = (
            ds[k][0].values.astype("M8[ms]").astype(datetime.datetime)
            - epoch).total_seconds()
    return ds
