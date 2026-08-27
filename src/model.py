# -*- coding: utf-8 -*-
"""
model.py — Strict 2R2C model with parameters switched by the compressor
regime (ON/OFF). 2R2C topology preserved: 2 resistances in the network,
2 capacities. Values switch on the physical signal P_elec(t) > threshold.

PHYSICAL JUSTIFICATION OF THE SWITCHING
----------------------------------------
R_env is a composite, series sum of three transfers:
    R_env = R_ext + R_paroi + R_int

    R_ext   : natural convection ambient -> outer wall (constant)
    R_paroi : conduction through the PUR foam (material constant)
    R_int   : convection inner wall -> internal air (VARIABLE)

The cold-active evaporator (compressor ON) reaches -30 to -35 C, i.e.
10-15 C below the internal air. This gradient drives vigorous natural
convection inside (no fan):
    h_int_ON  ~ 5-15 W/m2.K   -> low R_int_ON
    h_int_OFF ~ 1-3 W/m2.K    -> high R_int_OFF

R_ap (internal air <-> product resistance) is dominated by 1/(h_int * A_ap),
hence varies through the same mechanism.

STATE EQUATIONS
----------------
Consistent with the first law and Newton/Fourier constitutive laws:

    C_air * dT_air/dt = (T_amb - T_air)/R_env(t) + (T_prod - T_air)/R_ap(t)
                       - COP * P_elec(t) + Q_door(t)

    C_prod(T) * dT_prod/dt = (T_air - T_prod)/R_ap(t)

with:
    R_env(t) = R_env_on  if P_elec(t) > P_THRESHOLD, else R_env_off
    R_ap(t)  = R_ap_on   if P_elec(t) > P_THRESHOLD, else R_ap_off
    Q_door(t) = E_door / DOOR_OPEN_S   during an opening, else 0
    C_prod(T) = m_water * c_apparent(T)  (regularised latent heat, Bonacina method)

IDENTIFIED PARAMETERS (7)
-------------------------
    R_env_on, R_env_off  (K/W)
    R_ap_on,  R_ap_off   (K/W)
    C_air                (J/K)
    COP                  (-)
    E_door               (J)

C_prod : NOT a parameter. Physical function with the measured mass.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import config as C


P_THRESHOLD_W = 30.0   # compressor ON/OFF threshold (measured: typical P ON ~150 W,
                       # idle P < 5 W). 30 W discriminates without ambiguity.


# =============================================================================
# MODEL PARAMETERS
# =============================================================================

@dataclass
class RegimeParams:
    """The 7 parameters of the strict 2R2C model with switched R_env and R_ap."""
    R_env_on: float       # K/W : envelope in active regime (low R_int)
    R_env_off: float      # K/W : envelope in passive regime (high R_int)
    R_ap_on: float        # K/W : air-product coupling, active regime
    R_ap_off: float       # K/W : air-product coupling, passive regime
    C_air: float          # J/K : internal air capacity + fast wall layers
    COP: float            # -   : refrigeration efficiency
    E_door: float         # J   : energy of one door opening

    def to_dict(self):
        return {
            "R_env_on": self.R_env_on, "R_env_off": self.R_env_off,
            "R_ap_on": self.R_ap_on, "R_ap_off": self.R_ap_off,
            "C_air": self.C_air, "COP": self.COP, "E_door": self.E_door,
        }


# =============================================================================
# APPARENT CAPACITY OF THE PRODUCT (physical function, NOT a parameter)
# =============================================================================

def c_prod_apparent(T_prod: float) -> float:
    """
    Apparent thermal capacity of water: sensible part + regularised latent
    peak over +/- 0.5 C around 0 C (Bonacina 1973, Voller-Swaminathan
    1990 method). Preserves the integral int(C_prod dT) = m * L_f over
    the plateau.
    """
    eps = C.EPSILON_FREEZE_C
    T0 = C.T_FREEZE_C
    C_liq = C.M_EAU_KG * C.C_LIQ_J_KG_K
    C_ice = C.M_EAU_KG * C.C_ICE_J_KG_K
    if T_prod > T0 + eps:
        return C_liq
    if T_prod < T0 - eps:
        return C_ice
    frac_solid = (T0 + eps - T_prod) / (2.0 * eps)
    C_sens = C_liq * (1.0 - frac_solid) + C_ice * frac_solid
    C_lat = C.M_EAU_KG * C.L_F_J_KG / (2.0 * eps)
    return C_sens + C_lat


# =============================================================================
# DOOR-OPENING INDICATOR
# =============================================================================

def door_open(t_s: float, door_times_s: np.ndarray) -> bool:
    """True if t_s falls inside a window [t_open, t_open + DOOR_OPEN_S]."""
    if len(door_times_s) == 0:
        return False
    idx = np.searchsorted(door_times_s, t_s, side="right") - 1
    if idx < 0:
        return False
    t0 = door_times_s[idx]
    return (t_s >= t0) and (t_s - t0 < C.DOOR_OPEN_S)


def compressor_on(P_elec: np.ndarray) -> np.ndarray:
    """Binary ON/OFF signal deduced from the measured power."""
    return np.nan_to_num(P_elec, nan=0.0) > P_THRESHOLD_W


# =============================================================================
# CONTINUOUS SIMULATION OVER THE FULL CAMPAIGN
# =============================================================================

def simulate(t_datetime: np.ndarray,
             Tamb: np.ndarray,
             P_elec: np.ndarray,
             rp: RegimeParams,
             T_air_init: float,
             T_prod_init: float,
             door_times_s: Optional[np.ndarray] = None,
             dt_sub_s: Optional[float] = None) -> tuple:
    """
    Continuous 2R2C simulation over the whole supplied time vector.
    R_env and R_ap switch on the signal P_elec(t) > P_THRESHOLD_W.
    No state reinitialisation: physical continuity from start to end.

    Integration: Euler with substep dt_sub_s (default: config.INTEGRATION_SUBSTEP_S).
    """
    if door_times_s is None:
        door_times_s = np.array([], dtype=float)
    if dt_sub_s is None:
        dt_sub_s = C.INTEGRATION_SUBSTEP_S

    n = len(t_datetime)
    t0 = t_datetime[0]
    t_s = ((t_datetime - t0).astype("timedelta64[ms]")
           .astype(np.int64) / 1000.0)

    T_air_out = np.empty(n, dtype=float)
    T_prod_out = np.empty(n, dtype=float)
    T_air_out[0] = T_air_init
    T_prod_out[0] = T_prod_init

    Ta = T_air_init
    Tp = T_prod_init

    inv_R_env_on = 1.0 / rp.R_env_on
    inv_R_env_off = 1.0 / rp.R_env_off
    inv_R_ap_on = 1.0 / rp.R_ap_on
    inv_R_ap_off = 1.0 / rp.R_ap_off
    inv_C_air = 1.0 / rp.C_air
    Edoor_rate = rp.E_door / C.DOOR_OPEN_S

    for k in range(n - 1):
        t_start = t_s[k]
        t_end = t_s[k + 1]
        n_sub = max(1, int(np.ceil((t_end - t_start) / dt_sub_s)))
        dt_loc = (t_end - t_start) / n_sub

        Tamb_a, Tamb_b = float(Tamb[k]), float(Tamb[k + 1])
        P_a, P_b = float(P_elec[k]), float(P_elec[k + 1])
        if not np.isfinite(P_a):
            P_a = 0.0
        if not np.isfinite(P_b):
            P_b = 0.0

        for j in range(n_sub):
            alpha = (j + 0.5) / n_sub
            Tamb_j = Tamb_a + alpha * (Tamb_b - Tamb_a)
            P_j = P_a + alpha * (P_b - P_a)

            t_now = t_start + (j + 0.5) * dt_loc
            is_open = door_open(t_now, door_times_s)
            on = P_j > P_THRESHOLD_W

            inv_Re = inv_R_env_on if on else inv_R_env_off
            inv_Ra = inv_R_ap_on if on else inv_R_ap_off

            Qc = rp.COP * max(P_j, 0.0)
            Qdoor = Edoor_rate if is_open else 0.0
            Cp_eff = c_prod_apparent(Tp)

            Qenv = (Tamb_j - Ta) * inv_Re
            Qap = (Tp - Ta) * inv_Ra

            dTa = (Qenv + Qap - Qc + Qdoor) * inv_C_air
            dTp = (Ta - Tp) * inv_Ra / Cp_eff

            Ta = Ta + dt_loc * dTa
            Tp = Tp + dt_loc * dTp

        T_air_out[k + 1] = Ta
        T_prod_out[k + 1] = Tp

    return T_air_out, T_prod_out


# =============================================================================
# METRICS
# =============================================================================

def metrics(sim: np.ndarray, meas: np.ndarray) -> dict:
    valid = np.isfinite(sim) & np.isfinite(meas)
    s = sim[valid]
    m = meas[valid]
    if len(s) == 0:
        return {"rmse": np.nan, "r2": np.nan, "bias": np.nan, "mae": np.nan,
                "n": 0, "cv_rmse": np.nan, "nmbe": np.nan}
    e = s - m
    rmse_ = float(np.sqrt(np.mean(e ** 2)))
    ss_res = float(np.sum(e ** 2))
    m_mean = float(np.mean(m))
    ss_tot = float(np.sum((m - m_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    abs_mean = abs(m_mean) if abs(m_mean) > 1e-3 else 1.0
    return {
        "rmse": rmse_,
        "r2": float(r2),
        "bias": float(np.mean(e)),
        "mae": float(np.mean(np.abs(e))),
        "n": int(len(s)),
        "cv_rmse_pct": 100.0 * rmse_ / abs_mean,    # ASHRAE Guideline 14
        "nmbe_pct": 100.0 * np.mean(e) / abs_mean,  # ASHRAE Guideline 14
    }


def metrics_by_phase(ds, T_air_sim, T_prod_sim) -> dict:
    out = {}
    Tair_m = ds["Tair_in"].to_numpy(dtype=float)
    Tprod_m = ds["Teau"].to_numpy(dtype=float)
    for ph in ("A", "B", "C", "D"):
        mask = (ds["phase"] == ph).to_numpy()
        if mask.sum() == 0:
            continue
        out[ph] = {
            "Tair": metrics(T_air_sim[mask], Tair_m[mask]),
            "Tprod": metrics(T_prod_sim[mask], Tprod_m[mask]),
        }
    return out
