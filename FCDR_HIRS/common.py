"""Common utilities between scripts
"""

def add_to_argparse(parser,
        include_period=True,
        include_sat=True,
        include_channels=True,
        include_temperatures=False):
    """Add commoners to argparse object
    """

    if include_sat:
        parser.add_argument("satname", action="store", type=str,
            help="Satellite name",
            choices=["tirosn"]
                + ["noaa{:d}".format(n) for n in range(6, 20)]
                + ["metop{:s}".format(s) for s in "ab"])

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
    else:
        ch_tm = parser

    # http://stackoverflow.com/a/24181138/974555
    if include_channels:
        ch_tm.add_argument(("--" if hasboth else "") + "channels", action="store", type=int,
            nargs="+", choices=list(range(1, 21)),
            default=list(range(1, 20)),
            help="Channels to consider",
            required=True)

    if include_temperatures:
        ch_tm.add_argument(("--" if hasboth else "") + "temperatures", action="store", type=str,
            nargs="+",
            choices=['an_el', 'patch_exp', 'an_scnm', 'fwm', 'scanmotor',
                'iwt', 'sectlscp', 'primtlscp', 'elec', 'baseplate',
                'an_rd', 'an_baseplate', 'ch', 'an_fwm', 'ict', 'an_pch',
                'scanmirror', 'fwh', 'patch_full', 'fsr'],
            help="Temperature fields to use",
            required=True)

    parser.add_argument("--verbose", action="store_true",
        help="Be verbose", default=False)

    parser.add_argument("--log", action="store", type=str,
        help="Logfile to write to.  Leave out for stdout.")

    return parser
