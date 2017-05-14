"""Convert HIRS-HIRS matchups for harmonisation

Take HIRS-HIRS matchups and add telemetry and other information as needed
for the harmonisation effort.

See issue #22
"""



from .. import common
import argparse

def parse_cmdline():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser = common.add_to_argparse(parser,
        include_period=True,
        include_sat=2,
        include_channels=False,
        include_temperatures=False)

    return parser.parse_args()
p = parse_cmdline()

import logging
logging.basicConfig(
    format=("%(levelname)-8s %(asctime)s %(module)s.%(funcName)s:"
            "%(lineno)s: %(message)s"),
    level=logging.DEBUG if p.verbose else logging.INFO)

import itertools
import datetime

import numpy
import xarray
from .. import matchups

class HIRSMatchupCombiner(matchups.HIRSMatchupCombiner):
    def as_xarray_dataset(self):
        """Returns SINGLE xarray dataset for matchups
        """

        is_xarray = isinstance(self.Mcp, xarray.Dataset)
        is_ndarray = not is_xarray
        if is_ndarray:
            (p_ds, s_ds) = (tp.as_xarray_dataset(src,
                skip_dimensions=["scanpos"],
                rename_dimensions={"scanline": "collocation"})
                    for (tp, src) in ((self.hirs_prim, self.Mcp),
                                      (self.hirs_sec, self.Mcs)))
        elif is_xarray:
            p_ds = self.Mcp
            s_ds = self.Mcs
        else:
            raise RuntimeError("Onmogelĳk.  Impossible.  Unmöglich.")
        #
        keep = {"collocation", "channel", "calibrated_channel",
                "matchup_count"}
        p_ds.rename(
            {nm: "{:s}_{:s}".format(self.prim, nm)
                for nm in p_ds.keys()
                if nm not in keep},
            inplace=True)
        s_ds.rename(
            {nm: "{:s}_{:s}".format(self.sec, nm)
                for nm in s_ds.keys()
                if nm not in keep},
            inplace=True)
        # dimension prt_number_iwt may differ
        if p_ds["prt_number_iwt"].shape != s_ds["prt_number_iwt"].shape:
            p_ds.rename(
                {"prt_number_iwt": self.prim + "_prt_number_iwt"},
                inplace=True)
            s_ds.rename(
                {"prt_number_iwt": self.sec + "_prt_number_iwt"},
                inplace=True)
        ds = xarray.merge([p_ds, s_ds,
            xarray.DataArray(
                self.ds["matchup_spherical_distance"], 
                dims=["collocation"],
                name="matchup_spherical_distance")
            ])
        return ds

    def ds2harm(self, ds, channel):
        """Convert matchup dataset to harmonisation matchup dataset

        Taking the resulf of self.as_xarray_dataset, convert it to a
        dataset of the format wanted by the harmonisation.

        Modelled on files that Jon created for AVHRR, such as on CEMS at
        /group_workspaces/cems2/fiduceo/Data/Matchup_Simulated/Data/Harm_RealData/n17_n16.nc

        See also Arta's document on the FIDUCEO wiki.

        Note that one would want to merge a collection of those.
        """

        take_for_each = ("C_E", "C_IWCT", "C_s", "T_IWCT", "R_selfE")
        take_total = list('_'.join(x) for x in itertools.product(
                (self.prim, self.sec),
                take_for_each))
        da_all = [ds.sel(calibrated_channel=channel)[v]
                    for v in take_total]
        for (i, da) in enumerate(da_all):
            if "calibration_position" in da.dims:
                da_all[i] = da.median(dim="calibration_position")
        H = xarray.concat(da_all, dim="H_labels")
        H.name = "H_matrix"
        H = H.assign_coords(H_labels=take_total)

        raise NotImplementedError("Tot hier")

    def write(self, outfile):
        ds = self.as_xarray_dataset()
        logging.info("Storing to {:s}".format(
            outfile,
            mode='w',
            format="NETCDF4"))
        ds.to_netcdf(outfile)

def main():
    hmc = HIRSMatchupCombiner(
        datetime.datetime.strptime(p.from_date, p.datefmt),
        datetime.datetime.strptime(p.to_date, p.datefmt),
        p.satname1, p.satname2)

    ds = hmc.as_xarray_dataset()
    harm = hmc.ds2harm(ds, 12)
    #hmc.write("/work/scratch/gholl/test.nc")
