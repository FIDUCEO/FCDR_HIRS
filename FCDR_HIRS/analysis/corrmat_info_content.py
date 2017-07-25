"""Functionality related to correlation matrix information content

Rodgers (2000), Equation (2.56)

d_s = tr([K^T S_ε^⁻¹ K + S_a^⁻¹]^⁻¹ K^T S_ε^⁻¹ K

where:

- K is the Jacobian matrix, obtained from ARTS
- S_ε is the measurement error covariance matrix.  This is the one I
  change for different HIRSes
- S_a is the a priori covariance matrix, such as for water vapour.
  This can be obtained from reanalysis, or the Chevallier data, or
  NWP, or otherwise.
"""

import pathlib

import numpy
import scipy
# scipy.linalg.inv does not broadcast?
from numpy.linalg import inv
import matplotlib.pyplot

import xarray
import typhon.arts.xml
import typhon.atmosphere
import typhon.config

import pyatmlab.graphics

hirs_simul_dir = pathlib.Path(typhon.config.conf["main"]["simuldir"]) / "hirs"
jacob_dir = hirs_simul_dir / "Jacobians"
S_a_loc = pathlib.Path(typhon.config.conf["main"]["myscratchdir"]) / "HIRS" / "SA_H2O_VMR.nc"
K_loc = pathlib.Path(typhon.config.conf["main"]["myscratchdir"]) / "HIRS" / "K.nc"
dofloc = pathlib.Path(typhon.config.conf["main"]["myscratchdir"]) / "HIRS" / "dofs.npz"
#/H2O_VMR

p_newgrid = numpy.logspace(numpy.log10(1e5), 0, 92)

def get_S_a_from_Chevallier_ArtsXML():
    #p_newgrid = numpy.logspace(5, 3, 40)
    L = typhon.arts.xml.load("/home/users/gholl/checkouts/arts-xml-data/planets/Earth/ECMWF/IFS/Chevallier_91L/chevallierl91_all_q.xml.gz")
    #p_newgrid = L[n].grids[1]
    da_all = []
    for l in L:
        da = l.to_xarray().squeeze()
        vmr = da.sel(dim_0="abs_species-H2O")
        rh = typhon.atmosphere.relative_humidity(
            vmr=da.sel(dim_0="abs_species-H2O"),
            p=da.dim_1,
            T=da.sel(dim_0="T"))
        da = xarray.DataArray(
            scipy.interpolate.interp1d(
                numpy.log10(da.dim_1),
                vmr,
                bounds_error=False,
                fill_value=numpy.nan)(numpy.log10(p_newgrid)),
            dims=("p",),
            coords={"p": p_newgrid})
        da_all.append(da)

    da = xarray.concat(da_all, dim="profile")

    da = xarray.DataArray(
        numpy.ma.cov(numpy.ma.masked_invalid(da).T),
        dims=("p", "p"),
        coords={"p": p_newgrid},
        name="S_a")

    da.to_netcdf(S_a_loc)
    return da

def get_K():
    # need to read original p-grids from original Chevallier data
    L = typhon.arts.xml.load("/home/users/gholl/checkouts/arts-xml-data/planets/Earth/ECMWF/IFS/Chevallier_91L/chevallierl91_all_q.xml.gz")
    #K_all = []
    da_all = []
    for n in range(0, 5000, 1):
        K = typhon.arts.xml.load(str(jacob_dir / "H2O_VMR" / f"HIRS_Chevallier_RH_Jacobians.jacobian.{n:d}.xml.gz"))
        da = xarray.DataArray(
            scipy.interpolate.interp1d(
                numpy.log10(L[n].grids[1]),
                K,
                axis=1,
                bounds_error=False,
                fill_value=0)(numpy.log10(p_newgrid)),
            dims=("channel", "p"),
            coords={"channel": range(1, 13), "p": p_newgrid})
        da_all.append(da)

    da = xarray.concat(da_all, dim="profile")
    da.name = "K"
    da.to_netcdf(K_loc)

    return da

def get_S_ε(corrmat=None):
    ds = xarray.open_dataset("/group_workspaces/cems2/fiduceo/Data/FCDR/HIRS/v0.7pre/debug/metopa/2016/03/02/FIDUCEO_FCDR_L1C_HIRS4_metopa_20160302010629_20160302024729_debug_v0.7pre_fv0.3.nc")
    u_R = ds["u_R_Earth_random"].mean("scanpos").mean("scanline_earth").sel(calibrated_channel=slice(12)).values
    corrmat = corrmat if corrmat is not None else ds["channel_correlation_matrix"].sel(channel=slice(12)).values
    covmat = corrmat * (u_R[:, numpy.newaxis] * u_R[numpy.newaxis, :])
    return covmat

def dofs(S_a, K, S_ε):
    """
    Rodgers (2000), Equation (2.56)

    d_s = tr([K^T S_ε^⁻¹ K + S_a^⁻¹]^⁻¹ K^T S_ε^⁻¹ K

    Can be broadcasted, last two dimensions will be matrix dimensions
    """

    DOFS = (inv(K.swapaxes(-1, -2) @ inv(S_ε) @ K
                + inv(S_a)) @ K.swapaxes(-1, -2) @ inv(S_ε) @ K).trace(
                    axis1=-2, axis2=-1)
    return DOFS

def dofn(S_a, K, S_ε):
    """
    Rodgers (2000), Equation (2.57)

    d_n = tr(S_ε[K S_a K^T + S_ε]^⁻¹)

    Can be broadcasted, last two dimensions will be matrix dimensions
    """

    return (S_ε @ inv(K @ S_a @ K.swapaxes(-1, -2) + S_ε)).trace(
        axis1=-2, axis2=-1)

def get_all_dofs():
    try:
        S_a = xarray.open_dataset(S_a_loc)["S_a"].values
    except FileNotFoundError:
        S_a = get_S_a_from_Chevallier_ArtsXML().values

    try:
        K_all = xarray.open_dataset(K_loc)["K"].values
    except FileNotFoundError:
        K_all = get_K().values

    # not sure why needed with fill_value=0 in get_K()
    K_all[numpy.isnan(K_all)] = 0

    S_ε = get_S_ε()
    S_ε_diag = numpy.diag(numpy.diag(S_ε))
    S_ε_extreme = get_S_ε(numpy.ones((12,12)))

    # does this fix the negative DOFS bug?
    # no :(
    #OK = (~(K_all==0).any(-1).any(-1)) 
    #K_all = K_all[OK, :, :]
    # does THIS?
    S_a = S_a[:10, :10]
    K_all = K_all[:, :, :10]
    dofs_actual = dofs(S_a, K_all, S_ε)
    dofs_diag = dofs(S_a, K_all, S_ε_diag)
    dofs_full = dofs(S_a, K_all, S_ε_extreme)

    numpy.savez(dofloc,
        dofs_actual=dofs_actual,
        dofs_diag=dofs_diag,
        dofs_full=dofs_full)

    return (dofs_actual, dofs_diag, dofs_full)


def plot_dofs_hists():
    try:
        content = numpy.load(dofloc)
    except FileNotFoundError:
        (dofs_actual, dofs_diag, dofs_full) = get_all_dofs()
    else:
        dofs_actual = content["dofs_actual"]
        dofs_diag = content["dofs_diag"]
        dofs_full = content["dofs_full"]

    Δ_diag = dofs_actual - dofs_diag

    (f, a) = matplotlib.pyplot.subplots()

    bins = numpy.linspace(0, 4.5, 40)
    a.hist(dofs_actual, bins=bins, histtype="step", label="actual")
    a.hist(dofs_diag, bins=bins, histtype="step", label="assuming diagonal")
    #a.hist(dofs_full, bins=bins, histtype="step", label="full correlation")
    a.hist(Δ_diag, bins=bins, histtype="step", label="actual - diag")

    a.legend()

    f.suptitle("DOFS implication (MetOpA 2016-03-02)")

    pyatmlab.graphics.print_or_show(f, False, "DOFS.")
#    print("Actual:",
#          "DOFS", dofs(S_a, K_all, S_ε),
#          "DOFN", dofn(S_a, K_all, S_ε))
#    print("Diagonal:",
#          "DOFS", dofs(S_a, K_all, S_ε_diag),
#          "DOFN", dofn(S_a, K_all, S_ε_diag))
#    print("Extreme (100% correlation):",
#          "DOFS", dofs(S_a, K_all, S_ε_extreme),
#          "DOFN", dofn(S_a, K_all, S_ε_extreme))

def main():
    plot_dofs_hists()
#    print("Zero (nearly no noise in measurement)",
#          "DOFS", dofs(S_a, K, numpy.diag([1e-200]*12)),
#          "DOFN", dofn(S_a, K, numpy.diag([1e-200]*12)))
