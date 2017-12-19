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
          C_s="C_space",
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

    # fallback for simplified only, because I don't store the
    # intermediate value and 
    u_fallback = {"T_IWCT": 0.1}

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
        if ("prt_number_iwt" in p_ds.dims and
            "prt_number_iwt" in s_ds.dims and
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

        #take_for_each = ["C_s", "C_IWCT", "C_E", "T_IWCT", "α", "β", "fstar", "R_selfE"]
        take_for_each = ["C_s", "C_IWCT", "C_E", "T_IWCT", "R_selfE"]

        independent = {"C_E"}
        #u_common = {"α", "β", "fstar"}
        structured = {"C_s", "C_IWCT", "T_IWCT", "R_selfE"}
        wmats = {**dict.fromkeys({"C_s", "C_IWCT", "T_IWCT"}, 0),
                 **dict.fromkeys({"R_selfE"}, 1)}
        # for W-matrices, what corresponds to what
        dim_var = {"calibration_cycle": {"C_s", "C_IWCT", "T_IWCT"},
                   "rself_update_time": {"R_selfE"}}

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

        # Dimensions: (M1, m1, m2, w_matrix_count, w_matrix_row_count,
        #              w_matrix_sum_nnz, uncertainty_vector_count,
        #              uncertainty_vector_sum_row)
        harm = xarray.Dataset()

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
                    [daa[f"{sat:s}_{x:s}"].values[:, numpy.newaxis].astype("f4")
                     for x in take_for_each], 1))

            # fill Ur1, Ur2

            harm[f"Ur{i:d}"] = (
                ("M", f"m{i:d}"),
                numpy.concatenate(
                    [(ds.sel(calibrated_channel=channel)[f"{sat:s}_u_{tl.get(x,x):s}"].values.astype("f4") if x in independent
                      else numpy.zeros(ds.dims["matchup_count"])
                      )[:, numpy.newaxis]
                      for x in take_for_each], 1))

            # fill Us1, Us2

            # NB: not used!  Only my constants have common uncertainties,
            # and those should be corrected for in the harmonisation.
            L = []
            for x in take_for_each:
                L.append(numpy.zeros((ds.dims["matchup_count"], 1)))
            harm[f"Us{i:d}"] = (
                ("M", f"m{i:d}"),
                numpy.concatenate(L, 1))

            # fill uncertainty_type1, uncertainty_type2

            harm[f"uncertainty_type{i:d}"] = (
                (f"m{i:d}",),
                numpy.array([1 if x in independent else
                 3 if x in structured else 0
                 for x in take_for_each], dtype="i4")
                )

            # fill time1, time2

            harm[f"time{i:d}"] = ((("M",), ds[f"{sat:s}_time"]))

            # fill w_matrix_use1, w_matrix_use2
            harm[f"w_matrix_use{i:d}"] = (
                (f"m{i:d}",),
                numpy.array([2*i-1+wmats[x] if x in structured else 0
                    for x in take_for_each], dtype="i4"))

            # fill u_matrix_use1, u_matrix_use2
            harm[f"u_matrix_use{i:d}"] = (
                (f"m{i:d}",),
                numpy.array([next(cc) if x in structured else 0
                    for x in take_for_each], dtype="i4"))

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

        # the w-matrix has a form like:
        # [[1 0 0 0 ...]
        #  [1 0 0 0 ...]
        #  [0 1 0 0 ...]
        #  [0 1 0 0 ...]
        #  [0 1 0 0 ...]
        #  [0 0 1 0 ...]
        #  [0 0 1 0 ...]
        #  [...        ]]
        # That means the number of ones is equal to 
        # w_matrix_row_count = M+1

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
                (w_matrix_nnz_i, w_matrix_col_i, w_matrix_row_i, idx) = \
                    self.get_w_matrix(ds, sat, dim)
                w_matrix_nnz.append(w_matrix_nnz_i)
                w_matrix_col.extend(w_matrix_col_i)
                w_matrix_row.append(w_matrix_row_i)
                for var in dim_var[dim]:
                    (u_matrix_val_i, u_matrix_row_count_i) = \
                        self.get_u_matrix(ds, sat, channel, var, idx)
                    u_matrix_val.extend(u_matrix_val_i)
                    u_matrix_row_count.append(u_matrix_row_count_i)

#                (_, idx, counts) = numpy.unique(ds[f"{sat:s}_{dim:s}"],
#                        return_index=True, return_counts=True)
#                uidx.append(idx)
#                w_matrix_nnz.append(counts.sum())
#
#                # w_matrix_col gets a form like [0 0 1 1 1 2 2 ...]
#
#                c = itertools.count(0)
#                w_matrix_col.extend(
#                    numpy.concatenate([numpy.tile(next(c), cnt) for cnt in counts]))
#
#                # and w_matrix_row will simply be
#                # [0 1 2 3 ...]
#                w_matrix_row.append(numpy.arange(counts.sum()+1))
#                # construct u-matrices for those parameters that use this
#                # w-matrix, information contained in wmats
#            for v in structured:
#                u = ds.sel(calibrated_channel=channel)[f"{sat:s}_u_{tl.get(x,x):s}"].values
#                u_matrix_val.extend(u[uidx[wmats[v]]])
#                u_matrix_row_count.append(len(idx))
                

        harm["w_matrix_nnz"] = (("w_matrix_count",),
            numpy.array(w_matrix_nnz, dtype="i4"))
        harm["w_matrix_col"] = (("w_matrix_nnz_sum",),
            numpy.array(w_matrix_col, dtype="i4"))
        harm["w_matrix_row"] = (("w_matrix_count", "w_matrix_row_count"),
            numpy.array(w_matrix_row, dtype="i4"))
        harm["w_matrix_val"] = (("w_matrix_nnz_sum",),
            numpy.ones(numpy.array(w_matrix_nnz).sum(), dtype="f4"))

        harm["u_matrix_val"] = (("u_matrix_row_count_sum",),
            numpy.array(u_matrix_val, dtype="f4"))
        harm["u_matrix_row_count"] = (("u_matrix_count",),
            numpy.array(u_matrix_row_count, dtype="i4"))

        harm = harm.assign_coords(
            m1=take_for_each,
            m2=take_for_each,
            channel=channel)
        # need to recover other coordinates too

        harm["K"].attrs["description"] = ("Expected ΔBT due to nominal "
            "SRF (slave - reference)")

        harm["Kr"].attrs["description"] = ("Local standard deviation "
            "in 5×5 square of HIRS around collocation")
        harm["Kr"].attrs["units"] = "K"

        harm["Ks"].attrs["description"] = ("Propagated systematic "
            "uncertainty due to band correction factors.")

        harm.attrs["time_coverage"] = "{:%Y-%m-%d} -- {:%Y-%m-%d}".format(
            harm["time1"].values[0].astype("M8[s]").astype(datetime.datetime),
            harm["time1"].values[-1].astype("M8[s]").astype(datetime.datetime))

        harm.attrs["sensor_1_name"] = self.prim_name
        harm.attrs["sensor_2_name"] = self.sec_name

        harm = common.time_epoch_to(
            harm,
            datetime.datetime(1970, 1, 1, 0, 0, 0))

        return (harm, ds)

    def get_w_matrix(self, ds, sat, dim):
        """Get W matrix from ds for dimension

        Returns w_matrix_nnz, w_matrix_col,
        """

        (_, idx, counts) = numpy.unique(ds[f"{sat:s}_{dim:s}"],
                return_index=True, return_counts=True)
        w_matrix_nnz = counts.sum()

        # w_matrix_col gets a form like [0 0 1 1 1 2 2 ...]

        c = itertools.count(0)
        w_matrix_col = numpy.concatenate([numpy.tile(next(c), cnt) for cnt in counts])

        # and w_matrix_row will simply be
        # [0 1 2 3 ...]
        w_matrix_row = numpy.arange(counts.sum()+1)
        # construct u-matrices for those parameters that use this
        # w-matrix, information contained in wmats

        return (w_matrix_nnz, w_matrix_col, w_matrix_row, idx)

    def get_u_matrix(self, ds, sat, channel, var, idx):
        try:
            u = ds.sel(calibrated_channel=channel)[f"{sat:s}_u_{tl.get(var,var):s}"].values
        except KeyError:
            warnings.warn(
                f"Could not get uncertainty for {var:s}, using fallback.",
                UserWarning)
            u_matrix_val = numpy.tile(self.u_fallback[var],
                                      idx.size)
        else:
            u_matrix_val = u[idx]
        u_matrix_row_count = len(idx)
        return (u_matrix_val, u_matrix_row_count)

    def write(self, outfile):
        ds = self.as_xarray_dataset()
        logging.info("Storing to {:s}".format(
            outfile,
            mode='w',
            format="NETCDF4"))
        ds.to_netcdf(outfile)

    def write_harm(self, harm, ds_new):
        out = (self.basedir + f"{self.prim_name:s}_{self.sec_name:s}/" +
               "{:s}_{:s}_ch{:d}_{:%Y%m%d}-{:%Y%m%d}.nc".format(
                    self.prim_name,
                    self.sec_name,
                    int(harm["channel"]),
                    harm["time1".format(self.prim_name)].values[0].astype("M8[s]").astype(datetime.datetime),
                    harm["time1".format(self.prim_name)].values[-1].astype("M8[s]").astype(datetime.datetime),
                    ))
        logging.info("Writing {:s}".format(out))
        harm.to_netcdf(out)
        if int(harm["channel"]) == 1:
            ds_out = out[:-3] + "_ds.nc"
            logging.info("Writing {:s}".format(ds_out))
            ds_new.to_netcdf(ds_out.replace("_ch1", ""))

def main():
    warnings.filterwarnings("error",
        message="iteration over an xarray.Dataset will change",
        category=FutureWarning)
    hmc = HIRSMatchupCombiner(
        datetime.datetime.strptime(p.from_date, p.datefmt),
        datetime.datetime.strptime(p.to_date, p.datefmt),
        p.satname1, p.satname2)

    ds = hmc.as_xarray_dataset()
    for channel in range(1, 20):
        (harm, ds_new) = hmc.ds2harm(ds, channel)
        hmc.write_harm(harm, ds_new)
    #hmc.write("/work/scratch/gholl/test.nc")
