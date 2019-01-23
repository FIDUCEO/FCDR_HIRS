"""Show field on map

Note that this could be more generic than just for HIRS FCDR
"""

import argparse

import logging

import datetime
import matplotlib.pyplot
import mpl_toolkits.basemap

from .. import fcdr
from .. import common
from .. import graphics

logger = logging.getLogger(__name__)
def get_parser():
    parser = argparse.ArgumentParser(
        description="Show field on map",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("satname", action="store", type=str,
        help="Satellite to plot")

    parser.add_argument("field", action="store", type=str,
        help="Field to plot.  If this has channels, also pass --channels")

    parser.add_argument("start_time", action="store", type=str,
        help="Starting time in %%Y-%%m-%%dT%%H:%%M")

    parser.add_argument("duration", action="store", type=str,
        help="duration in %%H:%%M")

    parser.add_argument("--channels", action="store", type=int,
        nargs="+", help="Channels to consider.  Only used/needed "
        "for some fields.")

    parser.add_argument("--vmin", action="store", type=float,
        nargs="+",
        help="As pcolors vmin, one per channel")

    parser.add_argument("--vmax", action="store", type=float,
        nargs="+",
        help="As vmin")

    parser.add_argument("--label", action="store", type=str,
        help="Label to put on colourbar",
        default="UNLABELED!")

    parser.add_argument("--verbose", action="store_true", default=False)

    return parser
def parse_cmdline():
    return get_parser().parse_args()

def plot_field(lon, lat, fld, filename, tit, cblabel, **kwargs):
    """Plot a field on a world map using basemap

    Plot a single field on a world map, using pcolor and basemap.  Write
    the result to ``filename``.

    Parameters
    ----------

    lon : ndarray
        Longitude
    lat : ndarray
        Latitude
    fld : ndarray
        Field to be visualised, such as BTs
    filename : str
        Filename to write result to
    tit : str
        Figure title
    cblabel : str
        Colorbar label
    **kwargs
        Remaining arguments passed on to :func:`graphics.pcolor_on_map`, 
    """
    (f, a) = matplotlib.pyplot.subplots(figsize=(14, 8))
    m = mpl_toolkits.basemap.Basemap(projection="moll", resolution="c",
        lon_0=0, ax=a)
    c = graphics.pcolor_on_map(
        m, lon, lat, fld, cmap="viridis", **kwargs)
    m.drawcoastlines()
    cb = m.colorbar(c)
    cb.set_label(cblabel)
    a.set_title(tit)
    graphics.print_or_show(
        f, False, filename)

def read_and_plot_field(satname, field, start_time, duration, channels=[],
        vmin=None, vmax=None, label="",
        **kwargs):
    """Read field and plot on map

    Read a field from HIRS and plot it on a map using :func:`plot_field`.

    .. image: /images/hirs-on-map.png

    Parameters
    ----------

    satname : str
        Satellite name
    field : str
        Name of field to plot
    start_time : datetime.datetime
        Starting period to plot
    duration : datetime.timedelta
        Length of time to plot
    channels : array_like
        List of channenls to plot
    vmin : list[float]
        Lower range per channel
    vmax : list[float]
        Upper range per channel
    label : str
        Colorbar label, one for all
    **kwargs
        Remaining arguments get passed on to :func:`plot_field`.
    """
    if duration > datetime.timedelta(days=1):
        raise ValueError("Duration must not exceed 24 hours, found "
                         "{!s}".format(duration))
    h = fcdr.which_hirs_fcdr(satname)
    M = h.read_period(start_time, duration,
        fields=("lat", "lon", "time", field))

    tit = ("{satname:s} {field:s} {start_time:%Y-%m-%d %H:%M} -- "
           "{end_time:%H:%M}".format(end_time=start_time+duration,
           **locals()))

    if M[field].ndim > 2:
        for (ch, mn, mx) in zip(channels, vmin, vmax):
            plot_field(
                M["lon"], M["lat"], M[field][..., ch-1],
                    vmin=mn, vmax=mx, 
                    cblabel=label,
                    tit=tit + ", ch. {:d}".format(ch),
                    filename=
                    "HIRS_{satname:s}_{field:s}_{ch:d}_{start_time:%Y%m%d%H%M}.png".format(
                        **locals()),
                    **kwargs)
    else:
        plot_field(
                M["lon"], M["lat"], M[field],
                    vmin=mn[0], vmax=mx[0], 
                    cblabel=label,
                    tit=tit,
                    filename=
                    "HIRS_{satname:s}_{field:s}_{start_time:%Y%m%d%H%M}.png".format(**locals()),
                    **kwargs)

def main():
    """Main function, expects commandline input

    See module documentation and :ref:`map-field`.
    """
    p = parse_cmdline()
    common.set_logger(
        logging.DEBUG if p.verbose else logging.INFO,
        p.log,
        loggers={"FCDR_HIRS", "typhon"})

    start_time = datetime.datetime.strptime(p.start_time,
        "%Y-%m-%dT%H:%M")
    (hours, minutes) = p.duration.split(":")
    duration = datetime.timedelta(hours=int(hours), minutes=int(minutes))
    vmin = p.vmin or [None] * len(p.channels)
    vmax = p.vmax or [None] * len(p.channels)
    read_and_plot_field(p.satname, p.field, start_time, duration, p.channels,
        vmin=vmin, vmax=vmax, label=p.label)
