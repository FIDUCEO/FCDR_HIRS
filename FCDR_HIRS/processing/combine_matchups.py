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

# workaround for #174
tl = dict(C_E="C_Earth",
          fstar="f_eff",
          R_selfE="Rself")

import logging
logging.basicConfig(
    format=("%(levelname)-8s %(asctime)s %(module)s.%(funcName)s:"
            "%(lineno)s: %(message)s"),
    level=logging.DEBUG if p.verbose else logging.INFO)

import itertools
import datetime
import warnings

import numpy
import xarray
from .. import matchups

import typhon.datasets._tovs_defs

from typhon.physics.units.common import ureg, radiance_units as rad_u

class HIRSMatchupCombiner(matchups.HIRSMatchupCombiner):
    # FIXME: go through NetCDFDataset functionality
    basedir = "/group_workspaces/cems2/fiduceo/Data/Harmonisation_matchups/HIRS/"

    kmodel = None
    def __init__(self, start_date, end_date, prim, sec,
                 kmodel=None,
                 krmodel=None):
        super().__init__(start_date, end_date, prim, sec)
        if kmodel is None:
            kmodel = matchups.KModelPlanck(
                self.as_xarray_dataset(),
                self.ds,
                self.prim_name,
                self.prim_hirs,
                self.sec_name,
                self.sec_hirs)
        if krmodel is None:
            krmodel = matchups.KrModelLSD(
                self.as_xarray_dataset(),
                self.ds,
                self.prim_name,
                self.prim_hirs,
                self.sec_name,
                self.sec_hirs)
        self.kmodel = kmodel
        self.krmodel = krmodel

    def as_xarray_dataset(self):
        """Returns SINGLE xarray dataset for matchups
        """

        is_xarray = isinstance(self.Mcp, xarray.Dataset)
        is_ndarray = not is_xarray
        if is_ndarray:
            (p_ds, s_ds) = (tp.as_xarray_dataset(src,
                skip_dimensions=["scanpos"],
                rename_dimensions={"scanline": "collocation"})
                    for (tp, src) in ((self.prim_hirs, self.Mcp),
                                      (self.sec_hirs, self.Mcs)))
        elif is_xarray:
            p_ds = self.Mcp.copy()
            s_ds = self.Mcs.copy()
        else:
            raise RuntimeError("Onmogelĳk.  Impossible.  Unmöglich.")
        #
        keep = {"collocation", "channel", "calibrated_channel",
                "matchup_count", "calibration_position", "scanpos"}
        p_ds.rename(
            {nm: "{:s}_{:s}".format(self.prim_name, nm)
                for nm in p_ds.variables.keys()
                if nm not in keep|set(p_ds.dims)},
            inplace=True)
        s_ds.rename(
            {nm: "{:s}_{:s}".format(self.sec_name, nm)
                for nm in s_ds.variables.keys()
                if nm not in keep|set(s_ds.dims)},
            inplace=True)
        # dimension prt_number_iwt may differ
        if ("prt_number_iwt" in p_ds and
            "prt_number_iwt" in s_ds and
            p_ds["prt_number_iwt"].shape != s_ds["prt_number_iwt"].shape):

            p_ds.rename(
                {"prt_number_iwt": self.prim_name + "_prt_number_iwt"},
                inplace=True)
            s_ds.rename(
                {"prt_number_iwt": self.sec_name + "_prt_number_iwt"},
                inplace=True)
        ds = xarray.merge([p_ds, s_ds,
            xarray.DataArray(
                self.ds["matchup_spherical_distance"], 
                dims=["matchup_count"],
                name="matchup_spherical_distance")
            ])
        return ds

    def ds2harm(self, ds, channel):
        """Convert matchup dataset to harmonisation matchup dataset

        Taking the resulf of self.as_xarray_dataset, convert it to a
        dataset of the format wanted by the harmonisation.

        As described by Sam's document on the FIDUCEO wiki:

        20171205-FIDUCEO-SH-Harmonisation_Input_File_Format_Definition-v6.pdf

        and example files at

        /group_workspaces/cems2/fiduceo/Users/shunt/public/harmonisation/data
        
        Note that one would want to merge a collection of those.
        """

        take_for_each = ["C_s", "C_IWCT", "C_E", "T_IWCT", "α", "β", "fstar", "R_selfE"]

        independent = {"C_E"}
        u_common = {"α", "β", "fstar"}
        structured = {"C_s", "C_IWCT", "T_IWCT", "R_selfE"}
        wmats = {**dict.fromkeys({"C_s", "C_IWCT", "T_IWCT"}, 0),
                 **dict.fromkeys({"R_selfE"}, 1)}

        take_total = list('_'.join(x) for x in itertools.product(
                (self.prim_name, self.sec_name),
                take_for_each) if not x[1].startswith("u_"))
        da_all = [ds.sel(calibrated_channel=channel)[v]
                    for v in take_total]
        for (i, da) in enumerate(da_all):
            if "calibration_position" in da.dims:
                da_all[i] = da.median(dim="calibration_position")
            if "matchup_count" not in da.dims:
                da_all[i] = xarray.concat(
                    [da_all[i]]*ds.dims["matchup_count"],
                    "matchup_count").assign_coords(
                        **next(d.coords for d in da_all if (self.prim_name+"_scanline") in d.coords))
        H = xarray.concat(da_all, dim="m").transpose("matchup_count", "m")
#        H.name = "H_matrix"
#        H = H.assign_coords(H_labels=take_total)

        # Dimensions: (M1, m1, m2, w_matrix_count, w_matrix_num_row,
        #              w_matrix_sum_nnz, uncertainty_vector_count,
        #              uncertainty_vector_sum_row)
        harm = xarray.Dataset()

        {"X1": (("M", "m1"),
                ),
         "X2": (("M", "m2"),
                ),
         "Ur1": (("M", "m1"),
                ),
         "Ur2": (("M", "m2"),
                ),
         "Us1": (("M", "m1"),
                ),
         "Us2": (("M", "m2"),
                ),
         "uncertainty_type1": (("m1",),
                ),
         "uncertainty_type2": (("m2",),
                ),
         "K": (("M",),
                ),
         "Kr": (("M",),
                ),
         "Ks": (("M",),
                ),
         "time1": (("M",),
                ),
         "time2": (("M",),
                ),
         "w_matrix_nnz": (("w_matrix_count",),
                ),
         "w_matrix_row": (("w_matrix_count", "w_matrix_num_row"),
                ),
         "w_matrix_col": (("w_matrix_sum_nnz",),
                ),
         "w_matrix_val": (("w_matrix_val",),
                ),
         "w_matrix_use1": (("m1",),
                ),
         "w_matrix_use2": (("m2",),
                ),
         "u_matrix_row_count": (("u_matrix_count",),
                ),
         "u_matrix_val": (("u_matrix_row_count_sum",),
                ),
         "u_matrix_use1": (("m1",),
                ),
         "u_matrix_use2": (("m2",),
                )}

        daa = xarray.merge(da_all)

        # counter for uncertainties that have w-matrices.  Some but not
        # all share the same w-matrix so the counters are different, i.e.
        # there are more uncertainty vectors than w-matrices.
        cc = itertools.count(1)
        for (sat, i) in ((self.prim_name, 1), (self.sec_name, 2)):
            # fill X1, X2

            harm[f"X{i:d}"] = (
                ("M", f"m{i:d}"),
                numpy.concatenate(
                    [daa[f"{sat:s}_{x:s}"].values[:, numpy.newaxis]
                     for x in take_for_each], 1))

            # fill Ur1, Ur2

            harm[f"Ur{i:d}"] = (
                ("M", f"m{i:d}"),
                numpy.concatenate(
                    [(ds.sel(calibrated_channel=channel)[f"{sat:s}_u_{tl.get(x,x):s}"].values if x in independent
                      else numpy.zeros(ds.dims["matchup_count"])
                      )[:, numpy.newaxis]
                      for x in take_for_each], 1))

            # fill Us1, Us2

            # NB: need to tile scalar (constant) common uncertainties
            L = []
            for x in take_for_each:
                if x in u_common:
                    da = ds.sel(calibrated_channel=channel)[f"{sat:s}_u_{tl.get(x,x):s}"]
                    L.append(
                        numpy.tile(
                            da.values,
                            [1 if da.ndim>0 else ds.dims["matchup_count"],
                             1]))
                else:
                    L.append(numpy.zeros((ds.dims["matchup_count"], 1)))
            harm[f"Us{i:d}"] = (
                ("M", f"m{i:d}"),
                numpy.concatenate(L, 1))

            # fill uncertainty_type1, uncertainty_type2

            harm[f"uncertainty_type{i:d}"] = (
                (f"m{i:d}",),
                [1 if x in independent else
                 2 if x in structured else
                 3 if x in u_common else 0
                 for x in take_for_each]
                )

            # fill time1, time2

            harm[f"time{i:d}"] = ((("M",), ds[f"{sat:s}_time"]))

            # fill w_matrix_use1, w_matrix_use2
            harm[f"w_matrix_use{i:d}"] = (
                (f"m{i:d}",),
                [2*i-1+wmats[x] if x in structured else 0
                    for x in take_for_each])

            # fill u_matrix_use1, u_matrix_use2
            harm[f"u_matrix_use{i:d}"] = (
                (f"m{i:d}",),
                [next(cc) if x in structured else 0
                    for x in take_for_each])

        # dimension matchup only:
        #
        # K, Kr, Ks

        harm["K"] = (("M",), self.kmodel.calc_K(channel))
        harm["Ks"] = (("M",), self.kmodel.calc_Ks(channel))
        harm["Kr"] = (("M",), self.krmodel.calc_Kr(channel))

        # W-matrix for C_S, C_IWCT, T_IWCT.  This should have a 1 where
        # the matchup shares the same correlation information, and a 0
        # otherwise.  There is one for the primary and one for the
        # secondary.  I only have ones so w_matrix_val contains only ones.
        # The work is in bookkeeping the locations in the sparse matrix

        # the w-matrix is a block of ones for each unique calibration
        # cycle occurring in the matchups.  That means the number of ones
        # is the sum of the squares of the number of unique calibration
        # cycles.

        w_matrix_count = 4
        # according to the previous iteration, we have first the primary
        # for C_S, C_IWCT, T_IWCT, then the primary for RselfE, then the
        # secondary for the same two.

        w_matrix_nnz = []
        w_matrix_col = []
        w_matrix_row = []
        u_matrix_val = []
        u_matrix_row_count = []
        for sat in (self.prim_name, self.sec_name):
            for dim in ("calibration_cycle", "rself_update_time"):
                counts = numpy.unique(ds[f"{sat:s}_{dim:s}"],
                            return_counts=True)[1]
                w_matrix_nnz.append((counts**2).sum())

                # each square follows diagonally below the previous one, such that
                # w_matrix_col gets a form like [0 1 0 1 2 3 4 2 3 4 2 3 4 5 6 5 6]

                c = itertools.count(0)
                w_matrix_col.extend(numpy.concatenate([numpy.tile([next(c) for _ in range(cnt)], cnt) for cnt in counts]))

                # and w_matrix_row will have a form like
                # [2 2 3 3 3 2 2]
                # or is it
                # [0 0 3 3 3 5 5] 
                # or is it
                # [0 2 4 7 10 13 15]
                # ?
                # I think it's the latter!
                nonzero_count_per_row = numpy.concatenate([numpy.tile(cnt, cnt) for cnt in counts])
                w_matrix_row.append(nonzero_count_per_row.cumsum()-nonzero_count_per_row[0])
            for v in structured:
                u = ds.sel(calibrated_channel=channel)[f"{sat:s}_u_{tl.get(x,x):s}"].values
                u_matrix_val.extend(u)
                u_matrix_row_count.append(len(u))
            

        harm["w_matrix_nnz"] = (("w_matrix_count",),
            numpy.array(w_matrix_nnz))
        harm["w_matrix_col"] = (("w_matrix_nnz_sum",),
            numpy.array(w_matrix_col))
        harm["w_matrix_row"] = (("w_matrix_count", "w_matrix_num_row"),
            numpy.array(w_matrix_row))
        harm["w_matrix_val"] = (("w_matrix_nnz_sum",),
            numpy.ones(numpy.array(w_matrix_nnz).sum()))

        harm["u_matrix_val"] = (("u_matrix_row_count_sum",),
            numpy.array(u_matrix_val))
        harm["u_matrix_row_count"] = (("u_matrix_count",),
            numpy.array(u_matrix_row_count))

            # fill uncertainty_vector_use1, uncertainty_vector_use2

        # and the rest: K, Kr, Ks, w_matrix_nnz, w_matrix_row,
        # w_matrix_col, w_matrix_val, uncertainty_vector_row_count,
        # uncertainty_vector
        raise NotImplementedError("Tot hier hernieuwd!")

        harm = xarray.Dataset(
            {"H": H.rename({"matchup_count": "M"})},
            coords={"m": take_total},
            attrs={"Channel": 
                "Ch. {:d}: {:.6~} (primary, {:s}), {:.6~} (secondary, {:s})".format(
                    channel,
                    self.hirs_prim.srfs[channel-1].centroid().to("um", "sp"),
                    self.prim,
                    self.hirs_sec.srfs[channel-1].centroid().to("um", "sp"),
                    self.sec)})
        harm["H"].attrs["description"] = "Inputs for harmonisation functions"
        harm["H"].attrs["units"] = "Various"

        harm["lm"] = (("L", "nl"),
            numpy.array([[
                typhon.datasets._tovs_defs.NOAA_numbers[self.hirs_prim.satname],
                 typhon.datasets._tovs_defs.NOAA_numbers[self.hirs_sec.satname],
                 harm.dims["M"]]]))
        harm["lm"].attrs["description"] = ("Harmonisation satellite "
            "numbers and number of entries")

        # this just to fill it up so I can write to it more specifically
        # later
        harm["Ur"] = (harm["H"].dims, numpy.zeros_like(harm["H"].values))
        harm["Us"] = (harm["H"].dims, numpy.zeros_like(harm["H"].values))
        u_trans = dict(C_s="C_space", R_selfE="Rself", fstar="f_eff",
                       C_E="C_Earth")

        for sat in (self.prim, self.sec):
            for v in take_for_each:
                dest = "Ur" if v == "C_E" else "Us"
                try:
                    harm[dest].loc[{"m": "{:s}_{:s}".format(sat, v)}] = ds[
                        "{:s}_u_{:s}".format(sat, u_trans.get(v,v))].sel(
                            calibrated_channel=channel)
                except KeyError as e:
                    warnings.warn("No uncertainty defined for "
                        "{:s}: {:s}".format(v, e.args[0]),
                        UserWarning)

        harm["Ur"].attrs["description"] = ("Random error in elements "
            "of H")

        harm["Us"].attrs["description"] = ("Nonrandom error in elements "
            "of H")

        harm["CorrIndexArray"] = (("M",), harm["{:s}_calibration_cycle".format(self.prim)])
        harm["CorrIndexArray"].attrs["description"] = ("time at which "
            "calibration for this measurement was performed (slave)")
        harm["ref_CorrIndexArray"] = (("M",), harm["{:s}_calibration_cycle".format(self.sec)])
        harm["CorrIndexArray"].attrs["description"] = ("time at which "
            "calibration for this measurement was performed (reference)")

        ref_bt = self.hirs_prim.srfs[channel-1].channel_radiance2bt(
            ureg.Quantity(
                ds["{:s}_R_e".format(self.prim)].sel(
                    calibrated_channel=channel).values,
                rad_u["si"]))

        # the "slave" BT should be the radiance of the REFERENCE using the
        # SRF of the SECONDARY, so that all other effects are nada.
        slave_bt = self.hirs_sec.srfs[channel-1].channel_radiance2bt(
            ureg.Quantity(
                ds["{:s}_R_e".format(self.prim)].sel(
                    calibrated_channel=channel).values,
                rad_u["si"]))

        harm["K_InputData"] = (("M",), ref_bt)
        harm["K_InputData"].attrs["description"] = "Reference BT"
        harm["K_InputData"].attrs["units"] = "K"
        harm["K_Auxilliary"] = (("nsrfAux",),
            [ds["{:s}_α".format(self.prim)].sel(calibrated_channel=channel),
             ds["{:s}_β".format(self.prim)].sel(calibrated_channel=channel),
             ds["{:s}_fstar".format(self.prim)].sel(calibrated_channel=channel)])
        harm["K_Auxilliary"].attrs["description"] = ("Band correction "
            "parameters, reference")

        harm = harm.assign_coords(nsrfAux=["α", "β", "fstar"])

        harm["corrData"] = (("ncorr",), [40])
        harm["corrData"].attrs["description"] = "length of normal calibration cycle"
        harm["corrData"].attrs["units"] = "Scanlines"

        # to estimate K, use BT with both SRFs for ΔL
        harm["K"] = (("M",), slave_bt - ref_bt)
        harm["K"].attrs["description"] = ("Expected ΔBT due to nominal "
            "SRF (slave - reference)")

        harm["Kr"].attrs["description"] = ("Local standard deviation "
            "in 5×5 square of HIRS around collocation")
        harm["Kr"].attrs["units"] = "K"

        harm["Ks"].attrs["description"] = ("Propagated systematic "
            "uncertainty due to band correction factors.")

        harm.attrs["time_coverage"] = "{:%Y-%m-%d} -- {:%Y-%m-%d}".format(
            harm[self.prim_name + "_time"].values[0].astype("M8[s]").astype(datetime.datetime),
            harm[self.prim_name + "_time"].values[-1].astype("M8[s]").astype(datetime.datetime))

        harm.attrs["reference_satellite"] = self.prim_name
        harm.attrs["slave_satellite"] = self.sec_name

        harm = common.time_epoch_to(
            harm,
            datetime.datetime(1970, 1, 1, 0, 0, 0))

        return harm

    def write(self, outfile):
        ds = self.as_xarray_dataset()
        logging.info("Storing to {:s}".format(
            outfile,
            mode='w',
            format="NETCDF4"))
        ds.to_netcdf(outfile)

    def write_harm(self, harm):
        out = (self.basedir + 
               "{:s}_{:s}_ch{:d}_{:%Y%m%d}-{:%Y%m%d}.nc".format(
                    self.prim_name,
                    self.sec_name,
                    int(harm["calibrated_channel"]),
                    harm["{:s}_time".format(self.prim_name)].values[0].astype("M8[s]").astype(datetime.datetime),
                    harm["{:s}_time".format(self.prim_name)].values[-1].astype("M8[s]").astype(datetime.datetime),
                    ))
        logging.info("Writing {:s}".format(out))
        harm.to_netcdf(out)

def main():
    warnings.filterwarnings("error",
        message="iteration over an xarray.Dataset will change",
        category=FutureWarning)
    hmc = HIRSMatchupCombiner(
        datetime.datetime.strptime(p.from_date, p.datefmt),
        datetime.datetime.strptime(p.to_date, p.datefmt),
        p.satname1, p.satname2)

    ds = hmc.as_xarray_dataset()
    harm = hmc.ds2harm(ds, 12)
    hmc.write_harm(harm)
    #hmc.write("/work/scratch/gholl/test.nc")
