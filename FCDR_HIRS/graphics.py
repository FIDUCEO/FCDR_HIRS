"""Helper functions to interact with matplotlib and others

This module collects various small helper functions to assist with
plotting.
"""

import datetime
import logging
import subprocess
import sys
import pickle # FIXME: replace by dill
import lzma
import pathlib

import numpy
import matplotlib
import matplotlib.cbook
import matplotlib.pyplot
from typhon import config
from . import common

logger = logging.getLogger(__name__)

def plotdir():
    """Returns todays plotdir.

    Configuration 'plotdir' must be set.  Value is expanded with strftime.

    Returns
    -------

    str
        Path to a directory where plots shall be stored today.
    """
    return datetime.date.today().strftime(config.conf["main"]['plotdir'])

def print_or_show(fig, show, outfile, tikz=None,
                  data=None, close=True,
                  dump_pickle=True):
    """Either print or save figure, or both, depending on arguments.

    Taking a figure, show and/or save figure in the default directory,
    obtained with :func:plotdir.  Creates plot directory if needed.

    Parameters
    ----------

    fig : matplotlib.Figure
        Figure to store
    show : bool
        Show figure or not
    outfile : str
        Basename of file to write to.  If the string ends in a ``"."``, four
        files will be written: figure files to ``"{outfile}.pdf"`` and
        ``"{outfile}.png"``, a pickle object to ``"{outfile}.pkl.xz"``,
        and an information file with verbose stack info to
        ``"{outfile}.info"``.  The outfile is always written to the
        directory returned by `plotdir`.
    tikz : str, optional
        If given, try to write tikz code to the indicated file (again
        within ``plotdir``, using the ``matplotlib2tikz`` module.
    data : array_like, optional
        If given, write out data used to store the plot to the directory
        given by `common.plotdatadir`.  May be a list of ndarrays, which
        results in multiple numbered datafiles.
    close : bool, optional
        Close figure after showing and writing.
    dump_pickle : bool, optional
        Write figure to pickle object.
    """

    if outfile is not None:
        outfiles = [outfile] if isinstance(outfile, str) else outfile
        
        bs = pathlib.Path(plotdir())
        if isinstance(outfile, str):
            if outfile.endswith("."):
                outfiles = [bs / pathlib.Path(outfile+ext) for ext in ("png", "pdf")]
                infofile = bs / pathlib.Path(outfile + "info")
                figfile = bs / pathlib.Path(outfile + "pkl.xz")
            else:
                outfiles = [bs / pathlib.Path(outfile)]
                infofile = None
                figfile = None

        if infofile is not None:
            infofile.parent.mkdir(parents=True, exist_ok=True)

            logger.debug("Obtaining verbose stack info")
            pr = subprocess.run(["pip", "freeze"], stdout=subprocess.PIPE) 
            info = " ".join(sys.argv) + "\n" + pr.stdout.decode("utf-8") + "\n"
            info += common.get_verbose_stack_description()

#        if infofile is not None and info:
            logger.info("Writing info to {!s}".format(infofile))
            with infofile.open("w", encoding="utf-8") as fp:
                fp.write(info)
        if dump_pickle and figfile is not None:
            logger.info("Writing figure object to {!s}".format(figfile))
            with lzma.open(str(figfile), "wb", preset=lzma.PRESET_DEFAULT) as fp:
                pickle.dump(fig, fp, protocol=4)
        # interpret as sequence
        for outf in outfiles:
            logger.info("Writing to file: {!s}".format(outf))
            outf.parent.mkdir(parents=True, exist_ok=True)
            i = 0
            while True:
                i += 1
                try:
                    fig.canvas.print_figure(str(outf))
                except matplotlib.cbook.Locked.TimeoutError:
                    logger.warning("Failed attempt no. {:d}".format(i))
                    if i > 100:
                        raise
                else:
                    break
    if show:
        matplotlib.pyplot.show()

    if close:
        matplotlib.pyplot.close(fig)

    if tikz is not None:
        import matplotlib2tikz
        logger.info("Writing also to: " + os.path.join(plotdir(), tikz))
        matplotlib2tikz.save(os.path.join(plotdir(), tikz))
    if data is not None:
        if not os.path.exists(common.plotdatadir()):
            os.makedirs(common.plotdatadir())
        if isinstance(data, numpy.ndarray):
            data = (data,)
        # now take it as a loop
        for (i, dat) in enumerate(data):
            outf = os.path.join(common.plotdatadir(),
                "{:s}{:d}.dat".format(
                    os.path.splitext(outfiles[0])[0], i))
            fmt = ("%d" if issubclass(dat.dtype.type, numpy.integer) else
                    '%.18e')
            if len(dat.shape) < 3:
                numpy.savetxt(outf, dat, fmt=fmt)
            elif len(dat.shape) == 3:
                common.savetxt_3d(outf, dat, fmt=fmt)
            else:
                raise ValueError("Cannot write {:d}-dim ndarray to textfile".format(
                    len(dat.shape)))

def pcolor_on_map(m, lon, lat, C, **kwargs):
    """Wrapper around pcolor on a map, in case we cross the IDL

    Preventing spurious lines.

    This has a bad implementation.  **DO NOT USE**.
    """

    warnings.warn("Do not use this function, it is bad.",
                  DeprecationWarning)

    # Need to investigate why and how to solve:
    # -175 - minor polar problems (5° missing)
    # -178 - polar problems (2° missing)
    # -179 - some problems (1° missing)
    # -179.9 - many problems
    # perhaps I need to mask neighbours or so?
    C1 = numpy.ma.masked_where((lon<-175)|(lon>175), C, copy=True)
    p1 = m.pcolor(lon, lat, C1, latlon=True, **kwargs)
    return p1
