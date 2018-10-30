"""Plot single easy FCDR orbit file on set of maps

"""

import logging
import argparse

def parse_cmdline():
    parser = argparse.ArgumentParser(
        description="Show orbit on map",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

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

    parser.add_argument("--verbose", action="store_true", default=False)

    p = parser.parse_args()
    return p
#parsed_cmdline = parse_cmdline()

import itertools
import math
import datetime
import copy
import pathlib

import numpy
import matplotlib.pyplot
import matplotlib.colors
import matplotlib.ticker
import xarray
import cartopy
import cartopy.crs
import typhon.plots.plots

import pyatmlab.graphics
#from .. import fcdr

class OrbitPlotter:
    def __init__(self, f, channels, range=(0, 100)):
        self.path = pathlib.Path(f)
        self.ds = xarray.open_dataset(f)
        (fig, ax_all, cax_all) = self.prepare_figure_and_axes(channels)
        self.fig = fig
        self.ax_all = ax_all
        self.channels = channels
        self.range = range
        for ch in channels:
            self.plot_channel(ch, ax_all[ch], cax_all[ch])
#        pyatmlab.graphics.print_or_show(
#            f, False, filename)

    def prepare_figure_and_axes(self, channels):
        ncol = 6
        #ncol = int(math.ceil(math.sqrt(len(channels))))
        nrow = len(channels)
        #nrow = int(math.floor(math.sqrt(len(channels))))
        f = matplotlib.pyplot.figure(
            figsize=(5*(ncol+1),2.5*(nrow+1)))
        gs = matplotlib.gridspec.GridSpec(10*nrow, 16*ncol+1)
        #proj = cartopy.crs.Mollweide(central_longitude=90)
        proj = cartopy.crs.Mollweide(central_longitude=int(self.ds["latitude"].isel(y=0).sel(x=28)+0))

        ax_all = {ch: [] for ch in channels}
        cax_all = copy.deepcopy(ax_all)
        for ((r, ch), c) in itertools.product(
                enumerate(channels), range(ncol)):
#        for ((r, c), ch) in zip(
#                itertools.product(range(nrow), range(ncol)),
#                channels):
            ax = f.add_subplot(
                gs[(r*10):(r+1)*10, c*16:(c+1)*16],
                projection=proj) # passing the projection makes it a GeoAxes
            ax.coastlines()
            cax = f.add_subplot(gs[(r*10):(r+1)*10, (c+1)*16-1])
            ax_all[ch].append(ax)
            cax_all[ch].append(cax)
            if c==0:
                ax.text(-0.2, 0.5, f"Ch. {ch:d}", transform=ax.transAxes)
        ax_all[channels[0]][0].set_title("BT")
        ax_all[channels[0]][1].set_title("Independent uncertainty")
        ax_all[channels[0]][2].set_title("Structured uncertainty")
        ax_all[channels[0]][3].set_title("Common uncertainty")
        ax_all[channels[0]][4].set_title("Quality channel bitmask")
        ax_all[channels[0]][5].set_title("Quality scanline bitmask")
        f.suptitle(self.path.stem)

        return (f, ax_all, cax_all)

    def plot_channel(self, ch, ax_all, cax_all):
        ds = self.ds
        ok = (((ds["quality_channel_bitmask"].astype("uint8")&1)==0) &
              ((ds["quality_scanline_bitmask"].astype("uint8")&1)==0))
        dsx = ds.sel(channel=ch).isel(y=ok.sel(channel=ch))
        start = self.range[0]/100*dsx.dims["y"]
        end = self.range[1]/100*dsx.dims["y"]
        if dsx.dims["y"] < 5:
            logging.warning(f"Skipping channel {ch:d}, only {dsx.dims['y']:d} valid scanlines")
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
                "BT [K]")
            self._plot_to(ax_all[1], cax_all[1], t0, t1, dsx["u_independent"].values,
                "Independent uncertainty [K]",
                is_uncertainty=True)
            self._plot_to(ax_all[2], cax_all[2], t0, t1, dsx["u_structured"].values,
                "Structured uncertainty [K]",
                is_uncertainty=True)
            self._plot_to(ax_all[3], cax_all[3], t0, t1, dsx["u_common"].values,
                "Common uncertainty [K]",
                is_uncertainty=True)
        # flags are plotted for all cases, flagged or not
        dsx = ds.sel(channel=ch)
        lons = dsx["longitude"].values
        lats = dsx["latitude"].values
        trans = ax_all[0].projection.transform_points(cartopy.crs.Geodetic(), lons, lats)
        t0 = trans[:, :, 0]
        t1 = trans[:, :, 1]
        self.plot_bitfield(ax_all[4], cax_all[4], t0, t1,
            ds["quality_channel_bitmask"].sel(channel=ch),
            "Quality channel bitmask")
        self.plot_bitfield(ax_all[5], cax_all[5], t0, t1,
            ds["quality_scanline_bitmask"],
            "Quality scanline bitmask")

    def _plot_to(self, ax, cax, t0, t1, val, clab,
            is_uncertainty=False,
            is_bitmask=False):
        # plot in two parts to prevent spurious striping across the edge
        # of the map.  See also
        # https://stackoverflow.com/q/46527456/974555,
        # https://stackoverflow.com/q/46547310/974555, and
        # https://stackoverflow.com/q/46548044/974555.

        if (val==0).all():
            logging.warning("Only zeroes for "
                f"{clab:s}, skipping because splitting plot "
                "seems to cause problems")
            return
        for mask in (t0>1e6, t0<1e-6):
            img = ax.pcolor(numpy.ma.masked_where(mask, t0),
                      numpy.ma.masked_where(mask, t1),
                      numpy.ma.masked_where(mask, val),
                      transform=ax.projection,
                      cmap="viridis",
                      vmin=0 if is_uncertainty else val.min(),
                      vmax=val.max(),
                      norm=matplotlib.colors.Normalize(
                        vmin=0 if is_uncertainty else val.min(),
                        vmax=val.max()),
                      )
        cb = ax.figure.colorbar(img, cax=cax, orientation="vertical")
        cb.set_label(clab)

    def plot_bitfield(self, ax, cax, t0, t1, da, tit):
        # https://gist.github.com/jakevdp/8a992f606899ac24b711
        flagdefs = dict(zip(
            (int(x.strip()) for x in da.flag_masks.split(",")),
            da.flag_meanings.split()))

#        if not da.any():
#            logging.warning("Current unable to plot "
#                "bitfields where no data are flagged")
#            return

        # FIXME: this suffers once again from spurious horizontal lines
        for mask in (t0>1e5, t0<1e-5):
            typhon.plots.plots.plot_bitfield(
                ax,
                numpy.ma.masked_where(mask, t0),
                numpy.ma.masked_where(mask, t1),
                numpy.ma.masked_where(mask,
                    numpy.tile(da.astype("uint8").values, [56, 1]).T),
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
            + ".png")

def main():
    p = parse_cmdline()
    logging.basicConfig(
        format=("%(levelname)-8s %(asctime)s %(module)s.%(funcName)s:"
                "%(lineno)s: %(message)s"),
        level=logging.DEBUG if p.verbose else logging.INFO)
    op = OrbitPlotter(p.arg1, p.channels, range=p.range)
    op.write()
#    p = parsed_cmdline 
#    start_time = datetime.datetime.strptime(p.start_time,
#        "%Y-%m-%dT%H:%M")
#    (hours, minutes) = p.duration.split(":")
#    duration = datetime.timedelta(hours=int(hours), minutes=int(minutes))
#    vmin = p.vmin or [None] * len(p.channels)
#    vmax = p.vmax or [None] * len(p.channels)
#    read_and_plot_field(p.satname, p.field, start_time, duration, p.channels,
#        vmin=vmin, vmax=vmax, label=p.label)
