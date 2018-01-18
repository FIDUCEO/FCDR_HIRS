


from .. import common
import argparse
import tempfile
import subprocess
import time
import sys

def parse_cmdline_hirs():
    parser = argparse.ArgumentParser(
        description="""
Convert HIRS-HIRS matchups for harmonisation

Take HIRS-HIRS matchups and add telemetry and other information as needed
for the harmonisation effort.

See issue #22
""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser = common.add_to_argparse(parser,
        include_period=True,
        include_sat=2,
        include_channels=False,
        include_temperatures=False)

    return parser.parse_args()

def parse_cmdline_iasi():
    parser = argparse.ArgumentParser(
        description="""
Convert HIRS-IASI matchups for harmonisation
""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser = common.add_to_argparse(parser,
        include_period=True,
        include_sat=0,
        include_channels=False,
        include_temperatures=False)

    return parser.parse_args()

def parse_cmdline_merge():
    parser = argparse.ArgumentParser(
        description="Merge multiple harmonisation files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("files", type=str, nargs="+",
        help="Files to concatenate")

    parser.add_argument("--out", type=str,
        default="out.nc",
        help="File to write output to")

    return parser.parse_args()
    
# workaround for #174
tl = dict(C_E="C_Earth",
          C_s="C_space",
          fstar="f_eff",
          R_selfE="Rself")

import logging
logging.basicConfig(
    format=("%(levelname)-8s %(asctime)s %(module)s.%(funcName)s:"
            "%(lineno)s: %(message)s"),
    level=logging.INFO)

import itertools
import datetime
import warnings

import numpy
import xarray
import pathlib
from .. import matchups

import typhon.datasets._tovs_defs

from typhon.physics.units.common import ureg, radiance_units as rad_u
from typhon.physics.units.tools import UnitsAwareDataArray as UADA

class HIRSMatchupCombiner(matchups.HIRSMatchupCombiner):
    # FIXME: go through NetCDFDataset functionality
    # Experiencing problems with extreme slowness and hanging writing to
    # the GWS.  Experiment if this is any better writing to scratch2 (and
    # can then rsync those over later).
    #basedir = "/group_workspaces/cems2/fiduceo/Data/Harmonisation_matchups/HIRS/"
    basedir = "/work/scratch2/gholl/Harmonisation_matchups/HIRS/"

    # fallback for simplified only, because I don't store the
    # intermediate value and 
    u_fallback = {"T_IWCT": 0.1}

    kmodel = None
    def __init__(self, start_date, end_date, prim, sec,
                 kmodel=None,
                 krmodel=None):
        super().__init__(start_date, end_date, prim, sec)
        # parent has set self.mode to either "hirs" or "reference"
        if self.mode not in ("hirs", "reference"):
            raise RuntimeError("My father has been bad.")
        if kmodel is None:
            if self.mode == "hirs":
                kmodel = matchups.KModelPlanck(
                    self.as_xarray_dataset(),
                    self.ds,
                    self.prim_name,
                    self.prim_hirs,
                    self.sec_name,
                    self.sec_hirs)
            else:
                kmodel = matchups.KModelIASIRef(
                    self.as_xarray_dataset(),
                    self.ds,
                    "iasi",
                    None,
                    self.sec_name,
                    self.sec_hirs)
        if krmodel is None:
            if self.mode == "hirs":
                krmodel = matchups.KrModelLSD(
                    self.as_xarray_dataset(),
                    self.ds,
                    self.prim_name,
                    self.prim_hirs,
                    self.sec_name,
                    self.sec_hirs)
            else:
                krmodel = matchups.KrModelIASIRef(
                    self.as_xarray_dataset(),
                    self.ds,
                    "iasi",
                    None,
                    self.sec_name,
                    self.sec_hirs)
        self.kmodel = kmodel
        self.krmodel = krmodel

    def as_xarray_dataset(self):
        """Returns SINGLE xarray dataset for matchups
        """

        is_xarray = isinstance(self.Mcs, xarray.Dataset)
        is_ndarray = not is_xarray
        if is_ndarray:
            if self.mode == "reference":
                raise NotImplementedError("ndarray / reference not implemented")
            (p_ds, s_ds) = (tp.as_xarray_dataset(src,
                skip_dimensions=["scanpos"],
                rename_dimensions={"scanline": "collocation"})
                    for (tp, src) in ((self.prim_hirs, self.Mcp),
                                      (self.sec_hirs, self.Mcs)))
        elif is_xarray:
            p_ds = self.Mcp.copy() if self.mode=="hirs" else None
            s_ds = self.Mcs.copy()
        else:
            raise RuntimeError("Onmogelĳk.  Impossible.  Unmöglich.")
        #
        keep = {"collocation", "channel", "calibrated_channel",
                "matchup_count", "calibration_position", "scanpos"}
        if self.mode == "hirs":
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
        if (self.mode == "hirs" and
            "prt_number_iwt" in p_ds.dims and
            "prt_number_iwt" in s_ds.dims and
            p_ds["prt_number_iwt"].shape != s_ds["prt_number_iwt"].shape):

            p_ds.rename(
                {"prt_number_iwt": self.prim_name + "_prt_number_iwt"},
                inplace=True)
            s_ds.rename(
                {"prt_number_iwt": self.sec_name + "_prt_number_iwt"},
                inplace=True)
        to_merge = ([p_ds] if self.mode == "hirs" else []) + [s_ds,
            xarray.DataArray(
                self.ds["matchup_spherical_distance"] if self.mode=="hirs"
                    else numpy.zeros(self.ds.dims["line"]), 
                dims=["matchup_count"],
                name="matchup_spherical_distance")
            ]
        ds = xarray.merge(to_merge)
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
                take_for_each) if not x[1].startswith("u_")
                              and not x[0] == "iasi")
        da_all = [ds.sel(calibrated_channel=channel)[v]
                    for v in take_total]
        mdim = "matchup_count" if self.mode=="hirs" else "line"
        for (i, da) in enumerate(da_all):
            if "calibration_position" in da.dims:
                da_all[i] = da.median(dim="calibration_position")
            if mdim not in da.dims:
                da_all[i] = xarray.concat(
                    [da_all[i]]*ds.dims[mdim], mdim).assign_coords(
                        **next(d.coords for d in da_all if (self.sec_name+"_scanline") in d.coords))
        H = xarray.concat(da_all, dim="m").transpose(mdim, "m")
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

        # add to harmonisation the variables that exist for each 1, 2
        if self.mode == "reference":
            self._add_harm_for_iasi(harm, channel)
        else:
            self._add_harm_for_hirs(harm, channel, self.prim_name, 1, ds, daa,
                                    take_for_each, wmats, independent,
                                    structured, cc)
        self._add_harm_for_hirs(harm, channel, self.sec_name, 2, ds, daa,
                                take_for_each, wmats, independent,
                                structured, cc)

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

        w_matrix_count = 4 if self.mode == "hirs" else 2
        # according to the previous iteration, we have first the primary
        # for C_S, C_IWCT, T_IWCT, then the primary for RselfE, then the
        # secondary for the same two.

        w_matrix_nnz = []
        w_matrix_col = []
        w_matrix_row = []

        u_matrix_val = []
        u_matrix_row_count = []

        for sat in (self.prim_name, self.sec_name):
            if sat == "iasi": continue
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
            m1=take_for_each if self.mode == "hirs" else ["Lref"],
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

        # time should be double and without add-offset
        for i in (1, 2):
            v = f"time{i:d}"
            harm[v].encoding.pop("add_offset", None)
            harm[v].encoding["dtype"] = "f8"
            harm[v].encoding["units"] = "seconds since 1970-01-01"

        return (harm, ds)

    def get_w_matrix(self, ds, sat, dim):
        """Get W matrix from ds for dimension

        Returns w_matrix_nnz, w_matrix_col,
        """

        if sat == "iasi":
            raise ValueError("IASI has no W-Matrix!")

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

    def _add_harm_for_hirs(self, harm, channel, sat, i, ds, daa, take_for_each,
                           wmats, independent, structured, cc):
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
            numpy.array([2*(i if self.mode=="hirs" else i-1)-1+wmats[x] if x in structured else 0
                for x in take_for_each], dtype="i4"))

        # fill u_matrix_use1, u_matrix_use2
        harm[f"u_matrix_use{i:d}"] = (
            (f"m{i:d}",),
            numpy.array([next(cc) if x in structured else 0
                for x in take_for_each], dtype="i4"))

    def _add_harm_for_iasi(self, harm, channel):
        # fill X1

        # self.ds["ref_radiance"] contains IASI radiances; need to use
        # this to simulate HIRS radiance for MetOp-A
        freq = ureg.Quantity(numpy.loadtxt(self.hiasi.freqfile), ureg.Hz)
        specrad_wn = UADA(self.ds["ref_radiance"])
        specrad_f = specrad_wn.to(rad_u["si"], "radiance")
        srf = typhon.physics.units.em.SRF.fromArtsXML(
                "METOPA", "hirs", channel)
        L = srf.integrate_radiances(freq,
            ureg.Quantity(specrad_f.values, specrad_f.attrs["units"]))
        harm["X1"] = (
            ("M", "m1"),
            L[:, numpy.newaxis])
 
        # fill Ur1.  Uncertainties in refeence not considered.
        # Arbitrarily put at 1‰.

        harm["Ur1"] = (
            ("M", "m1"),
            harm["X1"]*0.001)

        # fill Us1.

        harm["Us1"] = (
            ("M", "m1"),
            numpy.zeros((harm.dims["M"], 1)))

        # fill uncertainty_type1

        harm["uncertainty_type1"] = (
            ("m1",),
            numpy.array([1], dtype="i4"))

        # fill time1

        harm["time1"] = (("M",), self.ds["mon_time"])

        # and w-matrix stuff

        harm["w_matrix_use1"] = (("m1",), numpy.array([0], dtype="i4"))
        harm["u_matrix_use1"] = (("m1",), numpy.array([0], dtype="i4"))

    def write(self, outfile):
        ds = self.as_xarray_dataset()
        logging.info("Storing to {:s}".format(
            outfile,
            mode='w',
            format="NETCDF4"))
        p = pathlib.Path(outfile)
        p.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(outfile)

    def write_harm(self, harm, ds_new, basedir=None):
        if basedir is None:
            basedir = self.basedir
        out = (basedir + f"/{self.prim_name:s}_{self.sec_name:s}/" +
               "{st:%Y-%m-%d}/{pn:s}_{sn:s}_ch{ch:d}_{st:%Y%m%d}-{en:%Y%m%d}.nc".format(
                    pn=self.prim_name,
                    sn=self.sec_name,
                    ch=int(harm["channel"]),
                    st=harm["time1".format(self.prim_name)].values[0].astype("M8[s]").astype(datetime.datetime),
                    en=harm["time1".format(self.prim_name)].values[-1].astype("M8[s]").astype(datetime.datetime),
                    ))
        pathlib.Path(out).parent.mkdir(exist_ok=True, parents=True)
        logging.info("Writing {:s}".format(out))
        harm.to_netcdf(out, unlimited_dims=["M"])
        if int(harm["channel"]) == 1:
            ds_out = out[:-3] + "_ds.nc"
            logging.info("Writing {:s}".format(ds_out))
            ds_new.to_netcdf(ds_out.replace("_ch1", ""), unlimited_dims=["line"])

def combine_hirs():
    p = parse_cmdline_hirs()
    warnings.filterwarnings("error",
        message="iteration over an xarray.Dataset will change",
        category=FutureWarning)
    hmc = HIRSMatchupCombiner(
        datetime.datetime.strptime(p.from_date, p.datefmt),
        datetime.datetime.strptime(p.to_date, p.datefmt),
        p.satname1, p.satname2)

    ds = hmc.as_xarray_dataset()
    with tempfile.TemporaryDirectory() as tmpdir:
        for channel in range(1, 20):
            (harm, ds_new) = hmc.ds2harm(ds, channel)
            hmc.write_harm(harm, ds_new, basedir=tmpdir)
        # copy over
        for i in range(50):
            logging.info(f"Copying files from {tmpdir:s} to {hmc.basedir:s} (attempt {i+1:d})")
            try:
                subprocess.run(["rsync", "-av", tmpdir + "/", hmc.basedir + "/"], 
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True)
            except subprocess.CalledProcessError as cpe:
                wait = i**2/5
                print(f"Copying failed with code {cpe.returncode!s}.\n"
                    f"Command:\n{cpe.cmd!s}\n"
                    f"STDOUT:\n{cpe.stdout.decode('utf8'):s}\n"
                    f"STDERR:\n{cpe.stderr.decode('utf8'):s}.\n"
                    f"Waiting {wait:.3f} seconds.",
                    file=sys.stderr)
                time.sleep(wait)
            else:
                break
        else:
            raise IOError("Failed 50 copying attempts, see above.")

def combine_iasi():
    p = parse_cmdline_iasi()
    warnings.filterwarnings("error",
        message="iteration over an xarray.Dataset will change",
        category=FutureWarning)
    hmc = HIRSMatchupCombiner(
        datetime.datetime.strptime(p.from_date, p.datefmt),
        datetime.datetime.strptime(p.to_date, p.datefmt),
        "iasi", "metopa")

    ds = hmc.as_xarray_dataset()
    for channel in range(1, 20):
        (harm, ds_new) = hmc.ds2harm(ds, channel)
        hmc.write_harm(harm, ds_new)

def merge_all(*files):
    """Merge one or more matchup harmonisation files

    This ignores the fact that there may be correlations between matchups
    in different files.
    """

    ds_all = [xarray.open_dataset(f) for f in files]
    # dimensions
    dims = {}
    ds_new = xarray.Dataset()

    # concatenated interleaving.  Length determined by w_matrix_nnz.  Each
    # W-matrix concatenated individually.
        
    w_mat_val_all = []
    u_mat_val_all = []

    w_mat_col_all = []
    w_mat_col_num = numpy.zeros(ds_all[0].dims["w_matrix_count"], "u4")

    w_mat_row_all = [[] for _ in range(ds_all[0].dims["w_matrix_count"])]
    w_mat_row_num = numpy.zeros(ds_all[0].dims["w_matrix_count"], "u4")

    for i in range(ds_all[0].dims["w_matrix_count"]):
        for (j, dsl) in enumerate(ds_all):
            wmatcumsum = dsl["w_matrix_nnz"].cumsum("w_matrix_count")
            w_mat_val_all.append(dsl["w_matrix_val"][int(wmatcumsum[i-1] if i>0 else 0):int(wmatcumsum[i])])
            # and w_matrix_col, but need to add the corresponding i from
            # previous dsl, if any
            w_mat_col = dsl["w_matrix_col"][int(wmatcumsum[i-1] if i>0 else 0):int(wmatcumsum[i])]
            w_mat_col_all.append(w_mat_col + w_mat_col_num[i])
            w_mat_col_num[i] += w_mat_col[-1] + 1
            # same for w_matrix_row
            w_mat_row = dsl["w_matrix_row"].sel(w_matrix_count=i)
            w_mat_row_all[i].append(w_mat_row[(j>0):] + w_mat_row_num[i])
            w_mat_row_num[i] += w_mat_row[-1]

    for i in range(ds_all[0].dims["u_matrix_count"]):
        for dsl in ds_all:
            umatcumsum = dsl["u_matrix_row_count"].cumsum("u_matrix_count")
            u_mat_val_all.append(dsl["u_matrix_val"][int(umatcumsum[i-1] if i>0 else 0):int(umatcumsum[i])])

    ds_new["w_matrix_val"] = xarray.concat(w_mat_val_all, dim="w_matrix_nnz_sum")
    ds_new["w_matrix_col"] = xarray.concat(w_mat_col_all, dim="w_matrix_nnz_sum")
    ds_new["u_matrix_val"] = xarray.concat(u_mat_val_all, dim="u_matrix_row_count_sum")

    ds_new["w_matrix_row"] = xarray.concat([
        xarray.concat(w_mat_row_all[i], dim="w_matrix_row_count")
        for i in range(ds_all[0].dims["w_matrix_count"])],
        dim="w_matrix_count")

    # additions
    #
    # extra .astype needed due to https://github.com/pydata/xarray/issues/1838

    ds_new["w_matrix_nnz"] = xarray.concat([da["w_matrix_nnz"] for da in ds_all], dim="dummy").sum("dummy", dtype="i4").astype("i4")
    ds_new["u_matrix_row_count"] = xarray.concat([da["u_matrix_row_count"] for da in ds_all], dim="dummy").sum("dummy", dtype="i4").astype("i4")

    identicals = ["w_matrix_use1", "w_matrix_use2", "u_matrix_use1", "u_matrix_use2"]
    for k in ds_all[0].data_vars.keys()-ds_new.keys():
        if "M" in ds_all[0][k].dims:
            ds_new[k] = xarray.concat([da[k] for da in ds_all], dim="M")
        else:
            if not ("m1" in ds_all[0][k].dims or 
                    "m2" in ds_all[0][k].dims):
                raise RuntimeError(f"I forgot about {k:s}?") 
            identicals.append(k)

    for (i, ds) in enumerate(ds_all):
        for var in identicals:
            if not (ds[var]==ds_all[0][var]).all():
                raise ValueError(f"Contents of {var:s} in {files[i]:s} "
                    f"must be equal to that in {files[0]:s} but is not.")
            ds_new[var] = ds_all[0][var]
        if not (ds.sensor_1_name == ds_all[0].sensor_1_name and
                ds.sensor_2_name == ds_all[0].sensor_2_name):
            raise ValueError(f"Attributes of {files[i]:s} inconsistent "
                "with the ones of {files[0]:s}.")

    ds_new.attrs["sensor_1_name"] = ds_all[0].sensor_1_name
    ds_new.attrs["sensor_2_name"] = ds_all[0].sensor_2_name
    ds_new.attrs["time_coverage"] = ds_all[0].time_coverage.split()[0] + "--" + ds_all[-1].time_coverage.split()[-1]

    return ds_new

def merge_files():
    p = parse_cmdline_merge()
    logging.info(f"Merging {len(p.files):d} files")
    new = merge_all(*p.files)
    logging.info(f"Writing to {p.out:s}")
    new.to_netcdf(p.out)
