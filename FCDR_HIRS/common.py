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
                + ["noaa{:d}".format(n) for n in range(6, 19)]
                + ["metop{:s}".format(s) for s in "ab"])

    if include_period:
        parser.add_argument("from_date", action="store", type=str,
            help="Start date/time")

        parser.add_argument("to_date", action="store", type=str,
            help="End date/time")

        parser.add_argument("--datefmt", action="store", type=str,
            help="Date format for start/end dates",
            default="%Y-%m-%d")

    if include_channels:
        parser.add_argument("channels", action="store", type=int,
            nargs="+", choices=list(range(1, 21)),
            default=list(range(1, 20)),
            help="Channels to consider")

    if include_temperatures:
        parser.add_argument("temp_fields", action="store", type=str,
            nargs="+",
            choices=['an_el', 'patch_exp', 'an_scnm', 'fwm', 'scanmotor',
                'iwt', 'sectlscp', 'primtlscp', 'elec', 'baseplate',
                'an_rd', 'an_baseplate', 'ch', 'an_fwm', 'ict', 'an_pch',
                'scanmirror', 'fwh', 'patch_full', 'fsr'],
            help="Temperature fields to use")

    parser.add_argument("--verbose", action="store_true",
        help="Be verbose", default=False)

    parser.add_argument("--log", action="store", type=str,
        help="Logfile to write to.  Leave out for stdout.")

    return parser
