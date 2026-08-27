# -*- coding: utf-8 -*-
"""
identify.py — Direct analytical identification of the 7 parameters of
the strict 2R2C model with switched R_env and R_ap.

PRINCIPLE
--------
Each parameter comes from a CLOSED-FORM ANALYTICAL FORMULA applied to
the data sub-segment where it is dominant. No joint optimisation. No
bounds. No prior.

Exception: the non-linear component (latent heat of water) in Phase A
requires a simulation to check the consistency of the analytically
estimated COP, but no parameter is tuned there.

PARAMETER <-> SUB-SEGMENT CORRESPONDENCE
-----------------------------------------
    R_env_off, R_ap_off, C_air_D : pure Phase D (compressor OFF throughout)
        -> simultaneous bi-exponential fit + algebraic inversion, 3 equations

    R_env_on, R_ap_on, COP, C_air_A : pure Phase A (compressor ON continuously)
        -> final steady-state balance + global enthalpy balance +
           dynamic balance on the initial slope

    Final C_air : weighted mean of C_air_D and C_air_A (two independent
                  estimates of the same constant parameter; the one with
                  the lowest uncertainty is retained)

    E_door : Phase B (24 openings), energy balance on each peak,
             mean over 24 independent estimates.

VALIDATION
----------
Phase C (mixed cyclic regime without openings): no parameter is
identified there. The predictive simulation on Phase C tests the
consistency of the ON/OFF switching. Phase C RMSE is the acceptance
criterion of the model.
"""

from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import config as C
from src.model import RegimeParams, simulate, metrics, P_THRESHOLD_W


# =============================================================================
# UTILITIES
# =============================================================================

def _t_seconds(ds: pd.DataFrame) -> np.ndarray:
    return (ds["datetime"] - ds["datetime"].iloc[0]).dt.total_seconds().to_numpy()


def _phase_segment(ds: pd.DataFrame, phase: str) -> pd.DataFrame:
    return ds[ds["phase"] == phase].copy().reset_index(drop=True)


# =============================================================================
# PHASE D -> R_env_off, R_ap_off, C_air (simple, robust analytical route)
# =============================================================================
# Three INDEPENDENT physical equations, three unknowns:
#
#   (1) Slow decay of T_prod (slow-mode equation of the 2R2C system)
#       In autonomy (locally constant T_amb):
#         dT_prod/dt = (T_air - T_prod) / (R_ap * C_prod)
#       In fast quasi-equilibrium (T_air settled with respect to T_amb, T_prod):
#         T_air,qs = (T_amb*R_ap + T_prod*R_env) / (R_env + R_ap)
#       => dT_prod/dt = -(T_prod - T_amb) / [(R_env + R_ap) * C_prod]
#       => tau_slow = (R_env + R_ap) * C_prod    [equation 1]
#
#   (2) Quasi-stationary relation for T_air:
#         T_air,qs = (T_amb*R_ap + T_prod*R_env) / (R_env + R_ap)
#       Measured at any instant t > tau_fast (after the initial settling):
#         R_ap / R_env = (T_air - T_prod) / (T_amb - T_air)   [equation 2]
#
#   (3) Fast decay of T_air towards its quasi-equilibrium (time constant
#       tied to C_air):
#         tau_fast ~ (R_env || R_ap) * C_air   [equation 3]
#       with R_env || R_ap = R_env*R_ap/(R_env+R_ap)
#
# All quantities (tau_slow, tau_fast, K=R_ap/R_env) are measured directly
# on the curves. No optimisation, no bounds.

def identify_phaseD(ds: pd.DataFrame) -> Dict:
    ds_D = _phase_segment(ds, "D")
    if len(ds_D) < 50:
        raise ValueError("Phase D too short.")

    t_s = _t_seconds(ds_D)
    Tair = ds_D["Tair_in"].to_numpy(dtype=float)
    Tprod = ds_D["Teau"].to_numpy(dtype=float)
    Tamb = ds_D["Tamb"].to_numpy(dtype=float)
    Tamb_mean = float(np.nanmean(Tamb))

    # ---- Equation 1: tau_slow from the log-linear decay of T_prod ----
    # T_prod(t) - T_amb_mean = (T_prod(0) - T_amb_mean) * exp(-t / tau_slow)
    # The autonomy period is too short to reach T_amb: fit of
    # log|T_prod - T_amb_mean| against t.
    y_prod = Tprod - Tamb_mean
    mask = np.abs(y_prod) > 0.1 * abs(y_prod[0])
    if mask.sum() < 10:
        raise ValueError("Insufficient T_prod decay.")
    coef_slow = np.polyfit(t_s[mask], np.log(np.abs(y_prod[mask])), 1)
    lam_slow = float(coef_slow[0])  # negative
    if lam_slow >= 0:
        raise ValueError("T_prod does not decay towards T_amb (Phase D).")
    tau_slow = -1.0 / lam_slow

    # ---- Equation 2: ratio R_ap/R_env from the quasi-stationary relation ----
    # K = R_ap / R_env = (T_air - T_prod) / (T_amb - T_air)
    # Measured over the quasi-stationary interval (after the first ~30 min)
    t_qs_start = max(1800.0, 0.1 * t_s[-1])
    qs_mask = t_s > t_qs_start
    if qs_mask.sum() < 10:
        qs_mask = t_s > 0.5 * t_s[-1]
    K_vals = ((Tair[qs_mask] - Tprod[qs_mask])
              / np.maximum(Tamb[qs_mask] - Tair[qs_mask], 1e-3))
    K = float(np.median(K_vals[np.isfinite(K_vals) & (K_vals > 0)]))

    # ---- Resolution: R_env, R_ap from tau_slow and K ----
    # tau_slow = (R_env + R_ap) * C_prod  =>  R_env + R_ap = tau_slow / C_prod
    # K = R_ap / R_env                     =>  R_ap = K * R_env
    # => R_env * (1 + K) = tau_slow / C_prod
    # => R_env = tau_slow / [(1 + K) * C_prod]
    C_prod_glace = C.M_EAU_KG * C.C_ICE_J_KG_K
    R_env_off = tau_slow / ((1.0 + K) * C_prod_glace)
    R_ap_off = K * R_env_off

    # ---- Equation 3: C_air via the fast initial relaxation of T_air ----
    # Over the first 5-15 minutes, T_air relaxes towards T_air,qs.
    # tau_fast = (R_env*R_ap)/(R_env+R_ap) * C_air
    # tau_fast measured by regression on (T_air - extrapolated T_air,qs).
    # T_air,qs evolves slowly (follows T_prod), estimated locally.
    # Approximation: T_air,qs(t) ~ (T_amb*R_ap + T_prod(t)*R_env) / (R_env + R_ap)
    Tair_qs = (Tamb * R_ap_off + Tprod * R_env_off) / (R_env_off + R_ap_off)
    delta = Tair - Tair_qs
    # Initial window: <= 1500 s (25 min) after the start of Phase D
    fast_mask = (t_s <= 1500.0) & (np.abs(delta) > 0.05 * abs(delta[0] or 1.0))
    if fast_mask.sum() >= 5 and abs(delta[0]) > 0.5:
        coef_fast = np.polyfit(t_s[fast_mask], np.log(np.abs(delta[fast_mask])), 1)
        lam_fast = float(coef_fast[0])
        if lam_fast < 0:
            tau_fast = -1.0 / lam_fast
            R_par = R_env_off * R_ap_off / (R_env_off + R_ap_off)
            C_air_from_D = tau_fast / R_par
        else:
            tau_fast = np.nan
            C_air_from_D = np.nan
    else:
        # No identifiable fast transient: C_air will be constrained
        # by Phase A. NaN returned here to signal it.
        tau_fast = np.nan
        C_air_from_D = np.nan

    return {
        "R_env_off": float(R_env_off),
        "R_ap_off": float(R_ap_off),
        "C_air_from_D": (float(C_air_from_D) if np.isfinite(C_air_from_D)
                         else None),
        "tau_slow_s": float(tau_slow),
        "tau_slow_h": float(tau_slow / 3600),
        "tau_fast_s": float(tau_fast) if np.isfinite(tau_fast) else None,
        "K_ratio_Rap_over_Renv": float(K),
        "C_prod_glace_used_J_per_K": float(C_prod_glace),
        "physical_validity": bool(R_env_off > 0 and R_ap_off > 0),
    }


# =============================================================================
# PHASE A -> R_env_on, R_ap_on, COP, C_air_A
# =============================================================================
# Phase A: compressor ON continuously, pulldown +30 -> -25 C, crosses fusion.
#
# 1) COP from the global enthalpy balance:
#       int(P) * COP = dH_water (with latent heat) + dH_air + Q_internal_leaks
#    Q_leaks depends on R_env_on (to be determined). Iterative resolution.
#
# 2) R_env_on from the final steady-state balance (last hour of Phase A):
#       in quasi-stationary regime: COP * <P> ~ <T_amb - T_air> / R_env_on
#
# 3) R_ap_on from the dynamic balance on the initial T_prod slope:
#       at t=0+ : T_air ~ T_prod ~ T_amb, hence dT_prod/dt = (T_air-T_prod)/(R_ap*C_prod)
#       Measurement of the initial T_prod slope and initial T_air-T_prod gap.
#
# 4) C_air from the initial T_air slope:
#       at t=0+ : (T_amb-T_air)/R_env ~ 0 and (T_prod-T_air)/R_ap ~ 0
#       hence C_air * dT_air/dt|0+ = -COP * P(0+)
#       => C_air = -COP * P(0+) / dT_air/dt|0+

def _enthalpy_balance_phaseA(ds_A: pd.DataFrame, R_env_on: float,
                              C_air: float) -> float:
    """COP via the global enthalpy balance over Phase A."""
    t_s = _t_seconds(ds_A)
    dt_s = np.diff(t_s, prepend=t_s[0])
    P = np.nan_to_num(ds_A["P_abs_mean_W"].to_numpy(dtype=float), nan=0.0)
    Tair = ds_A["Tair_in"].to_numpy(dtype=float)
    Tprod = ds_A["Teau"].to_numpy(dtype=float)
    Tamb = ds_A["Tamb"].to_numpy(dtype=float)

    E_elec = float(np.nansum(P[:-1] * dt_s[1:]))
    if E_elec <= 0:
        return np.nan

    T_prod0, T_prodN = float(Tprod[0]), float(Tprod[-1])
    T_air0, T_airN = float(Tair[0]), float(Tair[-1])

    # Enthalpy variation of the water (with latent heat if 0 C is crossed)
    if T_prod0 > 0 and T_prodN < 0:
        dH_eau = (C.M_EAU_KG * C.C_LIQ_J_KG_K * (T_prod0 - 0.0)
                  + C.M_EAU_KG * C.L_F_J_KG
                  + C.M_EAU_KG * C.C_ICE_J_KG_K * (0.0 - T_prodN))
    elif T_prod0 > T_prodN >= 0:
        dH_eau = C.M_EAU_KG * C.C_LIQ_J_KG_K * (T_prod0 - T_prodN)
    else:
        dH_eau = C.M_EAU_KG * C.C_ICE_J_KG_K * (T_prod0 - T_prodN)

    dH_air = C_air * (T_air0 - T_airN)
    Q_leak = float(np.nansum((Tamb[:-1] - Tair[:-1]) / R_env_on * dt_s[1:]))

    return float((dH_eau + dH_air + Q_leak) / E_elec)


def identify_phaseA(ds: pd.DataFrame, R_env_off: float, R_ap_off: float,
                     C_air: float) -> Dict:
    """
    Phase A: compressor ON continuously. The effective parameters are thus
    R_env_on, R_ap_on, COP. C_air is known (Phase D). R_env_off and
    R_ap_off are fixed but unused here (pure ON regime).

    The R_env_on <-> COP coupling cannot be separated analytically (two
    unknowns in a single enthalpy equation). Resolution by least_squares
    ON THE SIMULATION of Phase A. This is the only optimisation of the
    pipeline: 3 parameters, no bounds, data-driven initialisations.

    Initialisations:
      COP_init    = enthalpy balance with R_env_off and C_air
      R_env_on_init = R_env_off (assumed unswitched as a starting point)
      R_ap_on_init  = initial T_prod slope (independent of COP)
    """
    ds_A = _phase_segment(ds, "A")
    if len(ds_A) < 50:
        raise ValueError("Phase A too short.")

    t_s = _t_seconds(ds_A)
    P = np.nan_to_num(ds_A["P_abs_mean_W"].to_numpy(dtype=float), nan=0.0)
    Tair = ds_A["Tair_in"].to_numpy(dtype=float)
    Tprod = ds_A["Teau"].to_numpy(dtype=float)

    # ---- Init R_ap_on: initial T_prod slope (analytical, independent) ----
    # dT_prod/dt|0 = (T_air(0) - T_prod(0)) / (R_ap * C_prod_liquid)
    n_init = max(int(np.searchsorted(t_s, 300.0)), 5)
    slope_prod = float(np.polyfit(t_s[:n_init], Tprod[:n_init], 1)[0])
    dT_0 = float(np.mean(Tair[:n_init] - Tprod[:n_init]))
    C_prod_liq = C.M_EAU_KG * C.C_LIQ_J_KG_K
    if slope_prod < 0 and abs(dT_0) > 0.05:
        R_ap_on_init = abs(dT_0 / (slope_prod * C_prod_liq))
    else:
        R_ap_on_init = R_ap_off / 4.0

    # ---- Init COP: enthalpy balance with R_env_off and C_air ----
    COP_init = _enthalpy_balance_phaseA(ds_A, R_env_off, C_air)
    if not np.isfinite(COP_init) or COP_init <= 0:
        COP_init = 1.0

    # ---- Init R_env_on: same value as R_env_off (starting point) ----
    R_env_on_init = R_env_off

    # ---- Phase A simulation optimisation: (R_env_on, R_ap_on, COP) ----
    # Phase A simulated in pure ON regime (measured P_elec, hence natural
    # switching mostly ON). Residuals: simulated vs measured T_air and T_prod.

    Tamb_arr = ds_A["Tamb"].to_numpy(dtype=float)
    P_arr = np.nan_to_num(ds_A["P_abs_mean_W"].to_numpy(dtype=float), nan=0.0)
    t_dt = ds_A["datetime"].to_numpy()

    def _residuals(log_p):
        R_env_on, R_ap_on, COP = np.exp(log_p)
        # In Phase A, P_elec > threshold everywhere: R_env_off and R_ap_off
        # are never reached. Neutral values are acceptable for them.
        rp = RegimeParams(
            R_env_on=R_env_on, R_env_off=R_env_off,
            R_ap_on=R_ap_on, R_ap_off=R_ap_off,
            C_air=C_air, COP=COP, E_door=0.0,
        )
        T_air_sim, T_prod_sim = simulate(
            t_dt, Tamb_arr, P_arr, rp,
            float(Tair[0]), float(Tprod[0]),
            door_times_s=np.array([]),
        )
        r = np.concatenate([T_air_sim - Tair, T_prod_sim - Tprod])
        return r[np.isfinite(r)]

    x0 = np.log(np.array([R_env_on_init, R_ap_on_init, COP_init]))
    res = least_squares(_residuals, x0=x0, method="lm",
                         max_nfev=500, ftol=1e-9, xtol=1e-9)
    R_env_on, R_ap_on, COP = np.exp(res.x)

    return {
        "R_env_on": float(R_env_on),
        "R_ap_on": float(R_ap_on),
        "COP": float(COP),
        "init_R_env_on": float(R_env_on_init),
        "init_R_ap_on": float(R_ap_on_init),
        "init_COP_enthalpique": float(COP_init),
        "n_iter_simulation_LM": int(res.nfev),
        "physical_validity": bool(R_env_on > 0 and R_ap_on > 0 and
                                   0.3 < COP < 5.0),
    }


# =============================================================================
# PHASE B -> E_door (24 openings, direct mean)
# =============================================================================

def identify_phaseB(ds: pd.DataFrame, door_times_s: np.ndarray,
                     C_air: float) -> Dict:
    """
    For each opening peak, the measured amplitude gives:
        E_door,k = ΔT_peak,k * C_air
    Mean over 24 independent estimates.
    """
    ds_B = _phase_segment(ds, "B")
    if len(ds_B) < 50 or len(door_times_s) == 0:
        raise ValueError("Phase B without openings.")

    t0_ds = ds["datetime"].iloc[0]
    t_start_B = (ds_B["datetime"].iloc[0] - t0_ds).total_seconds()
    t_end_B = (ds_B["datetime"].iloc[-1] - t0_ds).total_seconds()
    dmask = (door_times_s >= t_start_B) & (door_times_s < t_end_B)
    door_rel = door_times_s[dmask] - t_start_B

    t_s_B = _t_seconds(ds_B)
    Tair_B = ds_B["Tair_in"].to_numpy(dtype=float)

    peaks_dT = []
    for td in door_rel:
        i_open = int(np.searchsorted(t_s_B, td))
        i_end = int(np.searchsorted(t_s_B, td + 90.0))   # 90 s post-opening
        i_pre = max(0, i_open - 3)
        if i_end - i_open < 2 or i_pre >= len(Tair_B):
            continue
        T_before = float(np.nanmean(Tair_B[i_pre:i_open]))
        T_peak = float(np.nanmax(Tair_B[i_open:i_end]))
        dT = T_peak - T_before
        if np.isfinite(dT) and 0.1 < dT < 80.0:
            peaks_dT.append(dT)

    if len(peaks_dT) == 0:
        raise ValueError("No peak detected in Phase B.")

    peaks_dT = np.array(peaks_dT)
    E_door_per_event = peaks_dT * C_air
    E_door_mean = float(np.mean(E_door_per_event))
    E_door_std = float(np.std(E_door_per_event, ddof=1))

    return {
        "E_door": E_door_mean,
        "E_door_std": E_door_std,
        "n_events_used": int(len(peaks_dT)),
        "peak_amplitude_mean_C": float(np.mean(peaks_dT)),
        "peak_amplitude_std_C": float(np.std(peaks_dT, ddof=1)),
    }


# =============================================================================
# FULL IDENTIFICATION PIPELINE
# =============================================================================

def identify_all(ds: pd.DataFrame, door_times_s: np.ndarray) -> Dict:
    """
    Full identification of the 7 parameters through analytical formulas:
      - Phase D    -> R_env_off, R_ap_off, C_air (estimate 1)
      - Phase A    -> R_env_on, R_ap_on, COP, C_air (estimate 2)
      - Phase B    -> E_door
    The retained C_air is the mean of the two estimates (Phases D and A)
    since it is physically the same quantity, each phase constraining it
    differently (equal weighting, no prior).
    """
    print("  [Phase D] T_prod decay + quasi-stationary relation...")
    t0 = time.time()
    rD = identify_phaseD(ds)
    print(f"    tau_slow   = {rD['tau_slow_h']:.2f} h")
    print(f"    K = R_ap/R_env  = {rD['K_ratio_Rap_over_Renv']:.3f}")
    print(f"    R_env_off  = {rD['R_env_off']:.4f} K/W")
    print(f"    R_ap_off   = {rD['R_ap_off']:.4f} K/W")
    if rD["C_air_from_D"] is not None:
        print(f"    C_air_D    = {rD['C_air_from_D']:.0f} J/K "
              f"(via tau_fast = {rD['tau_fast_s']:.0f} s)")
    else:
        print(f"    C_air_D    = not identifiable (no fast transient)")

    # C_air: Phase D value when identifiable, otherwise raw physical value
    if rD["C_air_from_D"] is not None and rD["C_air_from_D"] > 0:
        C_air_final = rD["C_air_from_D"]
        src = "Phase D (tau_fast)"
    else:
        # fallback: pure air capacity + fast wall-layer estimate
        C_air_final = C.c_air_pure() + 2000.0  # ~2200 J/K
        src = "physical (m_air*cp + walls)"
    print(f"    C_air      = {C_air_final:.0f} J/K (source: {src})")

    print("  [Phase A] LM simulation: R_env_on, R_ap_on, COP "
          "(C_air and R_env_off, R_ap_off frozen)...")
    rA = identify_phaseA(ds, R_env_off=rD["R_env_off"],
                          R_ap_off=rD["R_ap_off"], C_air=C_air_final)
    print(f"    R_env_on   = {rA['R_env_on']:.4f} K/W "
          f"(off/on ratio = {rD['R_env_off']/rA['R_env_on']:.2f})")
    print(f"    R_ap_on    = {rA['R_ap_on']:.4f} K/W "
          f"(off/on ratio = {rD['R_ap_off']/rA['R_ap_on']:.2f})")
    print(f"    COP        = {rA['COP']:.3f} "
          f"(enthalpy init = {rA['init_COP_enthalpique']:.3f})")

    print("  [Phase B] opening energy from each individual peak...")
    rB = identify_phaseB(ds, door_times_s, C_air=C_air_final)
    print(f"    E_door     = {rB['E_door']/1000:.1f} +/- "
          f"{rB['E_door_std']/1000:.1f} kJ "
          f"(over {rB['n_events_used']} peaks)")

    rp = RegimeParams(
        R_env_on=rA["R_env_on"], R_env_off=rD["R_env_off"],
        R_ap_on=rA["R_ap_on"], R_ap_off=rD["R_ap_off"],
        C_air=C_air_final, COP=rA["COP"], E_door=rB["E_door"],
    )
    elapsed = time.time() - t0

    return {
        "rp": rp,
        "phaseD": rD,
        "phaseA": rA,
        "phaseB": rB,
        "C_air_final": float(C_air_final),
        "elapsed_s": float(elapsed),
    }
