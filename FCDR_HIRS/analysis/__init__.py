"""High-end functionality for producing files and plots.

Essentially scripts.
"""

from . import calibcounts_stats_per_scanpos
from . import convert_hirs_l1b_to_nc
from . import convert_srfs_with_shift
from . import corrmat_info_content
from . import determine_latlon_compression_ratio
from . import determine_optimal_uncertainty_format
from . import fieldmat
from . import hirs_iasi_srf_estimation
from . import inspect_hirs_harm_matchups
from . import inspect_hirs_matchups
from . import inspect_orbit_curuc
from . import logfile_analysis
from . import map
from . import map_single_orbit
from . import monitor_fcdr
from . import plot_flags
from . import sensitivities
from . import summarise_fcdr
from . import test_rself
from . import timeseries
from . import write_harm_meta

__all__ = [ "calibcounts_stats_per_scanpos", "convert_hirs_l1b_to_nc",
    "convert_srfs_with_shift", "corrmat_info_content",
    "determine_latlon_compression_ratio",
    "determine_optimal_uncertainty_format", "fieldmat",
    "hirs_iasi_srf_estimation", "inspect_hirs_harm_matchups",
    "inspect_hirs_matchups", "inspect_orbit_curuc", "logfile_analysis",
    "map", "map_single_orbit", "monitor_fcdr", "plot_flags",
    "sensitivities", "summarise_fcdr", "test_rself", "timeseries",
    "write_harm_meta"]
