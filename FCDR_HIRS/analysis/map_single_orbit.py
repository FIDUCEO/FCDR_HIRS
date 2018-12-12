"""Plot (segment of) single easy FCDR orbit file on set of maps

This module and script plots a single orbit or a segment of an orbit on a
projected map.  It always plots the brightness temperature as well as two
or three uncertainty components, depending on whether the FCDR it is
plotted from is harmonised or not.  It also optionally plots bitfields to
show what parts of the orbit may be masked.

By default it plots channels 1–12, but it can plot as few as one or as
many as 19 channels.

It can either plot the full orbit (default), or a section of the orbit.
To plot only a section of the orbit, use the --range option.

There is the option to mark one or more pixels.
"""

import logging
import argparse
import sys

import itertools
import math
import datetime
import copy
import pathlib
import string

import scipy
import numpy
import matplotlib.pyplot
import matplotlib.colors
import matplotlib.ticker
import xarray
import cartopy
import cartopy.crs
import typhon.plots.plots
from .. import math as fcm
from .. import common
from . import inspect_orbit_curuc

import pyatmlab.graphics

logger = logging.getLogger(__name__)

#from .. import fcdr
def parse_cmdline():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser = common.add_to_argparse(parser,
        include_period=False,
        include_sat=0,
        include_channels=False,
        include_temperatures=False,
        include_debug=False)

    parser.add_argument("arg1", action="store", type=str,
        help="Either path to an orbit file, or satellite name.  In latter "
             "case also need to give timestamp")

    parser.add_argument("--time", action="store", type=str,
        help="Time in %%Y-%%m-%%dT%%H:%%M")

    parser.add_argument("--channels", action="store", type=int,
        nargs="+", help="Channels to consider.  Only used/needed "
        "for some fields.",
        default=list(range(1, 13)))

    parser.add_argument("--range", action="store", type=int,
        nargs=2, help="What fraction of orbit to plot, in %%.  Normally 0-100.",
        default=[0, 100])

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--with-bitmasks", action="store_true")
    group.add_argument("--without-bitmasks", action="store_false")

    parser.add_argument("--mark-pixels", action="store", type=float,
        nargs="*", default=[],
        help="Mark 0 or more pixels.  The numbers refer to percentile "
             "values in brightness temperature.  For example, 50 marks "
             "the median BT pixel, 10 the pixel corresponding to the 10th "
             "percentile, etc. ")

    parser.add_argument("--do-curuc", action="store_true", default=False,
        help="Follow orbit plot by calling hirs_curuc_checker")

    parser.add_argument("--btrange", action="store", type=float,
        nargs=2,
        help="Use this range for BTs.  If not given, base range on content.")

    parser.add_argument("--urange", action="store", type=float,
        nargs=2,
        help="Use this range for uncertainties.  If not given, base range on content.")

    parser.add_argument("--split", action="store_true",
        help="Split each channel over two lines.  Only works if "
            "--without-bitmasks.")

    p = parser.parse_args()
    return p

class OrbitPlotter:
    def __init__(self, f, channels, range=(0, 100),
                 plot_bitmasks=True,
                 mark_pixels=[],
                 btrange=None,
                 urange=None,
                 split=False):
        self.path = pathlib.Path(f)
        self.ds = xarray.open_dataset(f)
        self.channels = channels
        self.range = range
        self.start = self.range[0]*self.ds.dims["y"]//100
        self.end = self.range[1]*self.ds.dims["y"]//100
        self.plot_bitmasks = plot_bitmasks
        self.btrange = btrange
        self.urange = urange
        self.split = split
        (fig, ax_all, cax_all) = self.prepare_figure_and_axes(channels)
        self.fig = fig
        self.ax_all = ax_all
        self.selections = {}
        for ch in channels:
            self.selections[ch] = self.plot_channel(
                ch, ax_all[ch], cax_all[ch],
                mark_pixels=mark_pixels)
#        pyatmlab.graphics.print_or_show(
#            f, False, filename)

    def prepare_figure_and_axes(self, channels):
        ncol = 7 if self.plot_bitmasks else 2 if self.split else 4
        #ncol = int(math.ceil(math.sqrt(len(channels))))
        nrow = len(channels) * (2 if self.split else 1)
        #nrow = int(math.floor(math.sqrt(len(channels))))
        f = matplotlib.pyplot.figure(
            figsize=((3 if self.split else 4)*(ncol+1),(3 if self.split else 2.5)*(nrow+1)))
        gs = matplotlib.gridspec.GridSpec(10*nrow-2, 16*ncol+1)
        #proj = cartopy.crs.Mollweide(central_longitude=90)
        central_longitude=int(self.ds["latitude"].isel(y=0).sel(x=28)+0)
        if self.range[1]-self.range[0] > 30:
            proj = cartopy.crs.Mollweide(central_longitude=central_longitude)
        else:
            proj = cartopy.crs.PlateCarree(central_longitude=central_longitude)

        ax_all = {ch: [] for ch in channels}
        cax_all = copy.deepcopy(ax_all)
        if self.split:
            # [((0, 1), 0), ((0, 1), 1),
            #  ((1, 1), 0), ((1, 1), 1),
            #  ((2, 2), 0), ((2, 2), 1),
            #  ((3, 2), 0), ((3, 2), 1)]
            it = itertools.product(
                enumerate(
                    itertools.chain.from_iterable(
                        itertools.repeat(ch, 2) for ch in channels)),
                range(ncol))
        else:
            # [((0, 1), 0), ((0, 1), 1), ..., ((0, 1), ncol),
            #  ((1, 2), 0), ((1, 2), 1), ..., ((1, 2), ncol)]
            it = itertools.product(enumerate(channels), range(ncol))
        for ((r, ch), c) in it:
#        for ((r, c), ch) in zip(
#                itertools.product(range(nrow), range(ncol)),
#                channels):
            ax = f.add_subplot(
                gs[(r*10):(r+1)*10-2, c*16:(c+1)*16],
                projection=proj) # passing the projection makes it a GeoAxes
            ax.coastlines()
            try:
                gl = ax.gridlines(draw_labels=True)
                gl.xlabels_top = False
                gl.ylabels_right = False
                # see https://stackoverflow.com/a/35483665/974555
                ax.text(-0.15, 0.55, 'latitude [degrees]', va='bottom', ha='center',
                        rotation='vertical', rotation_mode='anchor',
                        transform=ax.transAxes)
                ax.text(0.5, -0.15, 'longitude [degrees]', va='bottom', ha='center',
                        rotation='horizontal', rotation_mode='anchor',
                        transform=ax.transAxes)

            except TypeError: # no labels
                ax.gridlines()
            cax = f.add_subplot(gs[(r*10):(r+1)*10-2, (c+1)*16-2])
            ax_all[ch].append(ax)
            cax_all[ch].append(cax)
            if c==0 and len(channels)>1:
                ax.text(-0.2, 0.5, f"Ch. {ch:d}", transform=ax.transAxes)
        ax_all[channels[0]][0].set_title("Brightness temperature")
        ax_all[channels[0]][1].set_title("Independent")
        ax_all[channels[0]][2].set_title("Structured")
        ax_all[channels[0]][3].set_title("Common")
        if self.plot_bitmasks:
            ax_all[channels[0]][4].set_title("Quality channel bitmask")
            ax_all[channels[0]][5].set_title("Quality scanline bitmask")
            ax_all[channels[0]][6].set_title("Quality pixel bitmask")
        f.suptitle("FIDUCEO HIRS FCDR " + self.ds.attrs["satellite"] +
            " {start:%Y-%m-%d %H:%M:%S}–{end:%H:%M:%S}".format(
                start=self.ds.isel(y=self.start)["time"].values.astype("datetime64[ms]").item(),
                end=self.ds.isel(y=self.end)["time"].values.astype("datetime64[ms]").item()) +
                f", channel {channels[0]:d}" if len(channels)==1 else "",
                y=0.96
            )

        f.subplots_adjust(wspace=0, left=0, right=0.95)
        return (f, ax_all, cax_all)

    def plot_channel(self, ch, ax_all, cax_all,
                     mark_pixels=[]):
        ds = self.ds
        ok = (((ds["quality_channel_bitmask"].astype("uint8")&1)==0) &
              ((ds["quality_scanline_bitmask"].astype("uint8")&1)==0))
        dsx = ds.sel(channel=ch).isel(y=ok.sel(channel=ch))
        if ok.sel(channel=ch).sum() < 20:
            sys.exit("Less than 20 lines left after filtering, no plot")
        start = self.start
        end = self.end
        dsx = dsx.isel(y=slice(start, end))
        print(f"Channel {ch:d}:")
        linerange = [dsx.isel(y=0)["y"].item(),
                     dsx.isel(y=-1)["y"].item()]
        print("Selecting lines", dsx.isel(y=0)["y"].item(), "to", dsx.isel(y=-1)["y"].item(), "inclusive")
        print("Covering {start:%Y-%m-%d %H:%M:%S}–{end:%H:%M:%S}".format(
                start=dsx.isel(y=0)["time"].values.astype("datetime64[ms]").item(),
                end=dsx.isel(y=-1)["time"].values.astype("datetime64[ms]").item()))

        dsx = fcm.gap_fill(dsx, "y", "time", numpy.timedelta64(6400, 'ms'))
        if dsx.dims["y"] < 5:
            logger.warning(f"Skipping channel {ch:d}, only {dsx.dims['y']:d} valid scanlines")
            ax_all[0].clear()
            ax_all[1].clear()
            ax_all[2].clear()
        else:
            lons = dsx["longitude"].values
            lats = dsx["latitude"].values
            trans = ax_all[0].projection.transform_points(cartopy.crs.Geodetic(), lons, lats)
            t0 = trans[:, :, 0]
            t1 = trans[:, :, 1]
            self._plot_to(ax_all[0], cax_all[0], t0, t1, dsx["bt"].values,
                "Brightness temperature / K")
            self._plot_to(ax_all[1], cax_all[1], t0, t1, dsx["u_independent"].values,
                "Uncertainty / K",
                is_uncertainty=True)
            self._plot_to(ax_all[2], cax_all[2], t0, t1, dsx["u_structured"].values,
                "Uncertainty / K",
                is_uncertainty=True)
            self._plot_to(ax_all[3], cax_all[3], t0, t1, dsx["u_common"].values,
                "Uncertainty / K",
                is_uncertainty=True)

        if self.plot_bitmasks:
            # flags are plotted for all cases, flagged or not
            dsx = ds.sel(channel=ch).isel(y=slice(start, end))
            lons = dsx["longitude"].values
            lats = dsx["latitude"].values
            trans = ax_all[0].projection.transform_points(cartopy.crs.Geodetic(), lons, lats)
            t0 = trans[:, :, 0]
            t1 = trans[:, :, 1]
            self.plot_bitfield(ax_all[4], cax_all[4], t0, t1,
                dsx["quality_channel_bitmask"],
                "Quality channel bitmask")
            self.plot_bitfield(ax_all[5], cax_all[5], t0, t1,
                dsx["quality_scanline_bitmask"],
                "Quality scanline bitmask")
            self.plot_bitfield(ax_all[6], cax_all[6], t0, t1,
                dsx["quality_pixel_bitmask"],
                "Quality pixel bitmask")

        pixels = []
        if mark_pixels:
            p_vals = numpy.nanpercentile(
                dsx["bt"].values.ravel(), mark_pixels,
                interpolation="lower")
            for (lab, p_val) in zip(string.ascii_uppercase, p_vals):
                (ycoor, xcoor) = [c[0].item() for c in
                    (dsx["bt"]==p_val).values.nonzero()]
                dsp = dsx.isel(y=ycoor, x=xcoor)
                lat = dsp["latitude"].item()
                lon = dsp["longitude"].item()
                print(lab, "x", dsp["x"].item(), "y", dsp["y"].item(),
                    "bt", dsp["bt"].item(), "lat", lat, "lon", lon)
                for ax in ax_all:
                    ax.plot(lon, lat, marker='o', markersize=5, color="red")
                    ax.text(lon, lat, lab, fontsize=20, color="red")
                    ax.plot(
                        dsx.isel(y=ycoor)["longitude"],
                        dsx.isel(y=ycoor)["latitude"],
                        linestyle=":",
                        color="red")
                    ax.plot(
                        dsx.isel(x=xcoor)["longitude"],
                        dsx.isel(x=xcoor)["latitude"],
                        linestyle=":",
                        color="red")
                    dsx.isel(x=xcoor)["latitude"]
                pixels.append((int(dsp["x"].item()),
                               int(dsp["y"].item())))
        return (linerange, pixels)


    def _plot_to(self, ax, cax, t0, t1, val, clab,
            is_uncertainty=False,
            is_bitmask=False):
        # plot in two parts to prevent spurious striping across the edge
        # of the map.  See also
        # https://stackoverflow.com/q/46527456/974555,
        # https://stackoverflow.com/q/46547310/974555, and
        # https://stackoverflow.com/q/46548044/974555.

        if (val==0).all():
            logger.warning("Only zeroes for "
                f"{clab:s}, skipping because splitting plot "
                "seems to cause problems")
            return
        val = val.copy()
        inval = numpy.isnan(val)
        val[inval] = 0
        if self.urange and is_uncertainty:
            loest, hiest = self.urange
        elif self.btrange and not is_uncertainty:
            loest, hiest = self.btrange
        else:
            loest = val[~inval].min()
            hiest = val[~inval].max()
        for mask in (t0>1e6, t0<1e-6):
            mask |= inval
            if not mask.all():
                img = ax.pcolor(numpy.ma.masked_where(mask, t0),
                          numpy.ma.masked_where(mask, t1),
                          numpy.ma.masked_where(mask, val),
                          transform=ax.projection,
                          cmap="viridis",
                          vmin=loest,
                          vmax=hiest,
                          norm=matplotlib.colors.Normalize(
                            vmin=loest,
                            vmax=hiest))
        cb = ax.figure.colorbar(img, cax=cax, orientation="vertical")
        cb.set_label(clab)

    def plot_bitfield(self, ax, cax, t0, t1, da, tit):
        # https://gist.github.com/jakevdp/8a992f606899ac24b711
        flagdefs = dict(zip(
            (int(x.strip()) for x in da.flag_masks.split(",")),
            da.flag_meanings.split()))

#        if not da.any():
#            logger.warning("Current unable to plot "
#                "bitfields where no data are flagged")
#            return

        # FIXME: this suffers once again from spurious horizontal lines
        for mask in (t0>1e5, t0<1e-5):
            if not mask.all():
                typhon.plots.plots.plot_bitfield(
                    ax,
                    numpy.ma.masked_where(mask, t0),
                    numpy.ma.masked_where(mask, t1),
                    numpy.ma.masked_where(mask,
                        da.astype("uint8").values
                            if "x" in da.dims
                            else numpy.tile(da.astype("uint8").values,
                        [56, 1]).T),
                    flagdefs,
                    cmap="Set3",
                    cax=cax,
                    pcolor_args=dict(transform=ax.projection), 
                    colorbar_args=dict(orientation="vertical"),
                    joiner=",\n")

    def write(self):
        p = self.path.absolute()
        pyatmlab.graphics.print_or_show(
            self.fig, False,
            "orbitplots/"
            + str((p.relative_to(p.parents[3]).parent / p.stem))
            + "_ch" + ",".join(str(ch) for ch in self.channels)
            + f"_{self.range[0]:d}-{self.range[1]:d}"
            + ".")

def main():
    p = parse_cmdline()
    common.set_logger(
        logging.DEBUG if p.verbose else logging.INFO,
        p.log)

    op = OrbitPlotter(p.arg1, p.channels, range=p.range,
        plot_bitmasks=p.with_bitmasks,
        mark_pixels=p.mark_pixels,
        btrange=p.btrange,
        urange=p.urange,
        split=p.split)
    op.write()

    if p.do_curuc:
        for ch in p.channels:
            (lines, pixels) = op.selections[ch]
            (xpix, ypix) = zip(*pixels)
            inspect_orbit_curuc.plot_curuc_for_pixels(
                xarray.open_dataset(p.arg1),
                lines=lines,
                x_all=xpix,
                y_all=ypix,
                channel=ch)
#    p = parsed_cmdline 
#    start_time = datetime.datetime.strptime(p.start_time,
#        "%Y-%m-%dT%H:%M")
#    (hours, minutes) = p.duration.split(":")
#    duration = datetime.timedelta(hours=int(hours), minutes=int(minutes))
#    vmin = p.vmin or [None] * len(p.channels)
#    vmax = p.vmax or [None] * len(p.channels)
#    read_and_plot_field(p.satname, p.field, start_time, duration, p.channels,
#        vmin=vmin, vmax=vmax, label=p.label)
