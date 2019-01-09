"""Functionality related to correlation matrix information content

Rodgers (2000), Equation (2.56)

d_s = tr([K^T S_eps^⁻¹ K + S_a^⁻¹]^⁻¹ K^T S_eps^⁻¹ K

where:

- K is the Jacobian matrix, obtained from ARTS
- S_eps is the measurement error covariance matrix.  This is the one I
  change for different HIRSes
- S_a is the a priori covariance matrix, such as for water vapour.
  This can be obtained from reanalysis, or the Chevallier data, or
  NWP, or otherwise.

This module is incomplete.  The results are incorrect.  The development is
inconclusive.  **DO NOT USE**

I said **DO NOT USE**.  What are you still doing here?

Go away!
"""

import itertools
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

from .. import graphics

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
        T = da.sel(dim_0="T")
#        vmr = da.sel(dim_0="abs_species-H2O")
#        rh = typhon.atmosphere.relative_humidity(
#            vmr=da.sel(dim_0="abs_species-H2O"),
#            p=da.dim_1,
#            T=da.sel(dim_0="T"))
        da = xarray.DataArray(
            scipy.interpolate.interp1d(
                numpy.log10(da.dim_1),
                T,
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
        K = typhon.arts.xml.load(str(jacob_dir / "temp_and_H2O_VMR" / f"HIRS_Chevallier_Jacobians.jacobian.{n:d}.xml.gz"))
        da = xarray.DataArray(
            scipy.interpolate.interp1d(
                numpy.log10(L[n].grids[1]),
                K[:, :92], # first half relates to temperature
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

def get_S_eps(corrmat=None):
    ds = xarray.open_dataset("/group_workspaces/cems2/fiduceo/Data/FCDR/HIRS/v0.7pre/debug/metopa/2016/03/02/FIDUCEO_FCDR_L1C_HIRS4_metopa_20160302010629_20160302024729_debug_v0.7pre_fv0.4.nc")
    u_R = ds["u_R_Earth_random"].mean("scanpos").mean("scanline_earth").sel(calibrated_channel=slice(12)).values
    corrmat = corrmat if corrmat is not None else ds["channel_correlation_matrix"].sel(channel=slice(12)).values
    covmat = corrmat * (u_R[:, numpy.newaxis] * u_R[numpy.newaxis, :])
    return covmat

def dofs(S_a, K, S_eps):
    """
    Rodgers (2000), Equation (2.56)

    d_s = tr([K^T S_eps^⁻¹ K + S_a^⁻¹]^⁻¹ K^T S_eps^⁻¹ K

    Can be broadcasted, last two dimensions will be matrix dimensions
    """

    DOFS = (inv(K.swapaxes(-1, -2) @ inv(S_eps) @ K
                + inv(S_a)) @ K.swapaxes(-1, -2) @ inv(S_eps) @ K).trace(
                    axis1=-2, axis2=-1)
    return DOFS

def dofn(S_a, K, S_eps):
    """
    Rodgers (2000), Equation (2.57)

    d_n = tr(S_eps[K S_a K^T + S_eps]^⁻¹)

    Can be broadcasted, last two dimensions will be matrix dimensions
    """

    return (S_eps @ inv(K @ S_a @ K.swapaxes(-1, -2) + S_eps)).trace(
        axis1=-2, axis2=-1)

def gain(S_a, K, S_eps):
    """Gain matrix, broadcasting version
    """
    return inv(K.swapaxes(-1, -2) @ inv(S_eps) @ K
                  + inv(S_a)) @ K.swapaxes(-1, -2) @ inv(S_eps)
    

def S_degradation(S_a, K, S_eps):
    """What is the error covariance from use of the wrong observation error covarionce?

    Chris Merchant, personal communication, 2017:

    S_{\hat{x}'-\hat{x}} = (G'-G)(y-F)(y-F)^T(G'-G)^T

    where

    G' = (K^T S_eps'^-1 K + S_a^-1)^-1 K^T S_eps'^-1

    G = (K^T S_eps^-1 K + S_a^-1)^-1 K^T S_eps^-1

    (y-F)(y-F)^T = S_y = K S_a K^T + S_eps

    Parameters
    ----------

    - S_a
    - K
    - S_eps

    """

    S_eps_reg = S_eps
    S_eps_diag = numpy.diag(numpy.diag(S_eps))
    G_prime = gain(S_a, K, S_eps_diag)
    G_reg = gain(S_a, K, S_eps_reg)
    ΔG = G_prime - G_reg

    S_y = K @ S_a @ K.swapaxes(-1, -2) + S_eps_reg

    return ΔG @ S_y @ ΔG.swapaxes(-1, -2)

def get_S_a_K_S_eps():
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

    S_eps = get_S_eps()

    return (S_a, K_all, S_eps)

def get_all_dofs():
    (S_a, K_all, S_eps) = get_S_a_K_S_eps()

    S_eps_diag = numpy.diag(numpy.diag(S_eps))
    S_eps_extreme = get_S_eps(numpy.ones((12,12)))

    # does this fix the negative DOFS bug?
    # no :(
    #OK = (~(K_all==0).any(-1).any(-1)) 
    #K_all = K_all[OK, :, :]
    # does THIS? (yes)
    S_a = S_a[:10, :10]
    K_all = K_all[:, :, :10]
    dofs_actual = dofs(S_a, K_all, S_eps)
    dofs_diag = dofs(S_a, K_all, S_eps_diag)
    dofs_full = dofs(S_a, K_all, S_eps_extreme)

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

    bins = numpy.linspace(0, 8.5, 40)
    a.hist(dofs_actual, bins=bins, histtype="step", label="actual")
    a.hist(dofs_diag, bins=bins, histtype="step", label="assuming diagonal")
    #a.hist(dofs_full, bins=bins, histtype="step", label="full correlation")
    a.hist(Δ_diag, bins=bins, histtype="step", label="actual - diag")

    a.legend()

    f.suptitle("DOFS implication (MetOpA 2016-03-02)")

    graphics.print_or_show(f, False, "DOFS.")
#    print("Actual:",
#          "DOFS", dofs(S_a, K_all, S_eps),
#          "DOFN", dofn(S_a, K_all, S_eps))
#    print("Diagonal:",
#          "DOFS", dofs(S_a, K_all, S_eps_diag),
#          "DOFN", dofn(S_a, K_all, S_eps_diag))
#    print("Extreme (100% correlation):",
#          "DOFS", dofs(S_a, K_all, S_eps_extreme),
#          "DOFN", dofn(S_a, K_all, S_eps_extreme))

def plot_S_degradation():
    N = 15
    (S_a, K_all, S_eps) = get_S_a_K_S_eps()
    K_da = xarray.open_dataset(K_loc) # for pressures
    p = K_da["p"][1:N]

    OK = ~(K_all[:, 0, 1:] == 0).any(1)
    S_degr = S_degradation(S_a[1:N, :][:, 1:N], K_all[:, :, 1:N][OK, :, :], S_eps)
    OK = numpy.isfinite(S_degr).all(1).all(1)
    S_degr = S_degr[OK, :, :]

    f = matplotlib.pyplot.figure(figsize=(16, 9))
    gs = matplotlib.gridspec.GridSpec(3, 6)
    a_all = []
    cm_all = []
    tot = S_degr.shape[0]
    for i in range(9):
        a = f.add_subplot(gs[i//3, i%3])
        idx = tot//9*i
        cm = a.pcolor(p, p, S_degr[idx, :, :], cmap="PuOr")
        a_all.append(a)
        cm_all.append(cm)
        a.set_title(f"No. {idx:d}")
        if (i%3) != 0:
            #a.set_yticklabels([])
            #a.set_xticks([])
            pass
        if (i//3) != 2:
            #a.set_xticklabels([])
            #a.set_yticks([])
            pass

    a_avg = f.add_subplot(gs[:, 3:6])
    cm = a_avg.pcolor(p, p, numpy.median(S_degr, 0), cmap="PuOr")
    a_avg.set_title("Median error covariance\n(is this meaningful?)")
    a_all.append(a_avg)
    cm_all.append(cm)

#    lim = max(abs(
#        numpy.array(
#            list(itertools.chain.from_iterable(cm.get_clim() for cm in cm_all))
#        )))
    lim = 4
    for (a, cm) in zip(a_all, cm_all):
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_aspect("equal")
        a.invert_xaxis()
        a.invert_yaxis()
        cm.set_clim([-lim, +lim])
        if not (a.is_first_col() or a is a_avg):
            a.set_yticklabels([])
            a.set_yticklabels([], minor=True)
        else:
            a.set_ylabel("Pressure [Pa]")
        if not (a.is_last_row() or a is a_avg):
            a.set_xticklabels([])
            a.set_xticklabels([], minor=True)
        else:
            a.set_xlabel("Pressure [Pa]")
        if (not a is a_avg) and a.is_last_row():
            for lab in a.get_xticklabels(which="both"):
                lab.set_rotation(30)
            
    cb = f.colorbar(cm, ax=a)
    cb.set_label("Error covariance [K^2]")
    #cb.set_label("Error covariance [(cm mW^-1 m^-2 sr^-1)^2]")

    f.suptitle("What is the error covariance from use of the wrong "
               "observation error covariance?")
    graphics.print_or_show(f, False, "S_degr.")

def main():
    plot_S_degradation()
    plot_dofs_hists()
#    print("Zero (nearly no noise in measurement)",
#          "DOFS", dofs(S_a, K, numpy.diag([1e-200]*12)),
#          "DOFN", dofn(S_a, K, numpy.diag([1e-200]*12)))
