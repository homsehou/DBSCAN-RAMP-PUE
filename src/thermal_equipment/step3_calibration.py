# -*- coding: utf-8 -*-
"""step3_calibration.py - Cold chain step 3: RAMP calibration per season.

Quantile-based calibration. Per season, the days of the dominant cluster as the
source of analytical initial values and day-to-day stds for every RAMP
parameter: segmentation of the representative profile into 3 power levels, then
derivation of the duty cycle, ON power p1, basal power p2 and the stochastic
variabilities from the level statistics. Refinement of the 12 parameters inside
data-driven bounds by a global Latin-Hypercube screening on real RAMP, then
Nelder-Mead restarts; re-ranking of the finalists at full fidelity and a
long-horizon polish of the winner, hence every reported figure from a
full-length RAMP simulation. The three seasons across parallel processes.

Two robustness guards for every calibrated season:
- extension of the cycle windows into a full tiling of the day before the pass
  to RAMP, against endless RAMP recursion on an event drawn inside a gap;
- search bounds always inclusive of the observed seasonal peak, against the
  unreachable correct powers of bounds locked below it.

Outputs: calibration_export.json, ramp_calibrated_params.csv,
validation_metrics.csv and ramp_simulated_profiles.csv (used by step 4).
The client code comes from the PUE_CLIENT environment variable (config.py).
"""
import datetime
import io
import json
import random
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import qmc

import config as C

warnings.filterwarnings("ignore")

# Search budget. The search runs on cheap common-seed evaluations (the seeds
# are fixed, so candidates stay comparable); the final evaluation and the
# long-horizon polish use the full seed count and the real season length.
SEARCH_SEEDS = 8
SEARCH_DAYS = 14
LHS_POINTS = 500
NM_STARTS = 2
NM_MAXITER = 350
POLISH_SEEDS = 8
POLISH_DAYS = 90
POLISH_MAXITER = 80


# --- 1. RAMP compatibility patch ----------------------------------------------
def patch_ramp_windows() -> None:
    """RAMP 0.5.0 + numpy >= 2: Appliance.windows calls int(np.diff(...)) on
    scalars, which breaks. Same logic rewritten with plain integer arithmetic.
    """
    from ramp import Appliance

    def safe_windows(self, window_1=None, window_2=None, random_var_w=0,
                     window_3=None):
        self.window_1 = np.asarray(window_1 if window_1 is not None
                                   else [0, 1440], dtype=int)
        self.window_2 = np.asarray(window_2 if window_2 is not None
                                   else [0, 0], dtype=int)
        self.window_3 = np.asarray(window_3 if window_3 is not None
                                   else [0, 0], dtype=int)
        self.random_var_w = float(random_var_w)
        d1 = int(self.window_1[1] - self.window_1[0])
        d2 = int(self.window_2[1] - self.window_2[0])
        d3 = int(self.window_3[1] - self.window_3[0])
        if d1 + d2 + d3 < self.func_time:
            from ramp.core.core import InvalidWindow
            raise InvalidWindow(f"Sum of windows {d1 + d2 + d3} "
                                f"< func_time {self.func_time}")
        self.daily_use = np.zeros(1440)
        for dur, win in ((d1, self.window_1), (d2, self.window_2),
                         (d3, self.window_3)):
            if dur > 0:
                self.daily_use[win[0]:win[1]] = 0.001
        self.random_var_1 = int(round(random_var_w * d1))
        self.random_var_2 = int(round(random_var_w * d2))
        self.random_var_3 = int(round(random_var_w * d3))
        # Register the appliance in the user's list, otherwise the UseCase
        # does not see it at simulation time
        if self not in self.user.App_list:
            self.user.App_list.append(self)
        # Default cycle windows, as in the original implementation
        if self.fixed_cycle == 1:
            self.cw11 = self.window_1
            self.cw12 = self.window_2

    Appliance.windows = safe_windows


patch_ramp_windows()


# --- 2. Validation metrics -----------------------------------------------------
def nrmse(real: np.ndarray, sim: np.ndarray) -> float:
    den = max(real.max() - real.min(), 1e-9)
    return float(np.sqrt(np.mean((real - sim) ** 2)) / den)


def ldc_err(real: np.ndarray, sim: np.ndarray) -> float:
    """NRMSE between the two load duration curves (sorted profiles)."""
    rl = np.sort(real)[::-1]
    sl = np.sort(sim)[::-1]
    den = max(rl.max() - rl.min(), 1e-9)
    return float(np.sqrt(np.mean((rl - sl) ** 2)) / den)


def fft_err(real: np.ndarray, sim: np.ndarray) -> float:
    """NRMSE between the log-magnitudes of the first FFT_N_KEEP harmonics."""
    R = np.abs(np.fft.rfft(real))
    S = np.abs(np.fft.rfft(sim))
    if C.FFT_REMOVE_DC:
        R, S = R[1:], S[1:]
    R, S = R[:C.FFT_N_KEEP], S[:C.FFT_N_KEEP]
    if C.FFT_LOG_SCALE:
        R, S = np.log(R + 1.0), np.log(S + 1.0)
    den = max(R.max() - R.min(), 1e-9)
    return float(np.sqrt(np.mean((R - S) ** 2)) / den)


def all_metrics(real: np.ndarray, sim: np.ndarray) -> dict:
    """The 6 validation errors between the real and simulated 96-slot days:
    NRMSE, LDC error, FFT error, energy error err_E_pct, peak error err_P_pct
    and load-factor error err_LF (LF = load factor, mean power over peak
    power)."""
    e_r = real.sum() * (C.DT_MIN / 60.0) / 1000.0
    e_s = sim.sum() * (C.DT_MIN / 60.0) / 1000.0
    lf_r = real.mean() / max(real.max(), 1e-9)
    lf_s = sim.mean() / max(sim.max(), 1e-9)
    return {"NRMSE": nrmse(real, sim),
            "LDC_err": ldc_err(real, sim),
            "FFT_err": fft_err(real, sim),
            "err_E_pct": float(100.0 * (e_s - e_r) / e_r) if e_r > 0 else 100.0,
            "err_P_pct": float(100.0 * (sim.max() - real.max()) / real.max())
                         if real.max() > 0 else 100.0,
            "err_LF": float(lf_s - lf_r)}


def composite_score(metrics: dict) -> float:
    """Weighted sum of the 6 errors, each normalised by its threshold, then
    scaled by a fixed 1/12.0 divisor. The divisor (near the ~13 total of the six
    W_* weights) as a constant, neutral for candidate ranking and only a
    convenience to keep the composite score close to the unit scale."""
    return float((C.W_NRMSE * metrics["NRMSE"] / C.MAX_NRMSE
                  + C.W_LDC * metrics["LDC_err"] / C.MAX_LDC_ERR
                  + C.W_FFT * metrics["FFT_err"] / C.MAX_FFT_ERR
                  + C.W_E * abs(metrics["err_E_pct"]) / C.MAX_ERR_E_PCT
                  + C.W_P * abs(metrics["err_P_pct"]) / C.MAX_ERR_P_PCT
                  + C.W_LF * abs(metrics["err_LF"]) / C.MAX_ERR_LF) / 12.0)


def count_passed(metrics: dict) -> int:
    """Number of validation criteria satisfied (out of 6)."""
    if metrics is None:
        return 0
    return int(sum([metrics["NRMSE"] <= C.MAX_NRMSE,
                    metrics["LDC_err"] <= C.MAX_LDC_ERR,
                    metrics["FFT_err"] <= C.MAX_FFT_ERR,
                    abs(metrics["err_E_pct"]) <= C.MAX_ERR_E_PCT,
                    abs(metrics["err_P_pct"]) <= C.MAX_ERR_P_PCT,
                    abs(metrics["err_LF"]) <= C.MAX_ERR_LF]))


# --- 3. Analytical inits from the measurements ---------------------------------
def compute_inits(profile_W: np.ndarray, daily_seas: np.ndarray,
                  rated_power: float, n_levels: int = 3) -> dict:
    """Initial value and day-to-day std of every RAMP parameter, derived from
    the days of the dominant cluster (daily_seas, one row per day)."""
    n_slots = len(profile_W)
    n_days = daily_seas.shape[0]

    # p1: typical compressor peak (P75 of the daily maxima). RAMP `power` is
    # the RATED power of the appliance, not the inrush peak, hence the cap.
    daily_max = daily_seas.max(axis=1)
    rated = float(rated_power)
    p1_init = min(float(np.percentile(daily_max, 75)), rated)
    p1_std = float(min(np.std(daily_max), 0.15 * rated))

    # p2: P5 of the positive values (typical basal load)
    pos_vals = daily_seas[daily_seas > 0]
    p2_init = float(np.percentile(pos_vals, 5)) if len(pos_vals) > 0 else 1.0
    p2_std = max(0.5, p2_init * 0.5)

    # Quantile segmentation of the profile into n_levels power levels
    bounds = np.percentile(profile_W, np.linspace(0, 100, n_levels + 1))
    bounds[0], bounds[-1] = -1e9, 1e9
    slot_levels = np.digitize(profile_W, bounds[1:-1])  # 0..n_levels-1

    # Per level: mean power, std, expected duty cycle and its support
    # (1 - coefficient of variation: high = repeatable, low = variable)
    L_means, L_stds, duty_inits, duty_stds, supports = [], [], [], [], []
    denom = max(1e-3, p1_init - p2_init)
    for level in range(n_levels):
        in_level = slot_levels == level
        if in_level.sum() == 0:
            L_means.append(0.0)
            L_stds.append(0.0)
            duty_inits.append(0.5)
            duty_stds.append(0.1)
            supports.append(0.0)
            continue
        per_day_L = daily_seas[:, in_level].mean(axis=1)
        L_mean, L_std = float(per_day_L.mean()), float(per_day_L.std())
        L_means.append(L_mean)
        L_stds.append(L_std)
        duty_inits.append(float(np.clip((L_mean - p2_init) / denom,
                                        0.05, 0.95)))
        duty_stds.append(float(np.clip(L_std / denom, 0.01, 0.40)))
        cv = L_std / max(1e-3, L_mean)
        supports.append(float(np.clip(1.0 - min(1.0, cv), 0.0, 1.0)))

    # Cycle windows: the 2 largest contiguous slot regions of each level,
    # in minutes, sorted by start time
    cw_assignments = []
    for level in range(n_levels):
        regions, start = [], None
        for i in range(n_slots):
            if slot_levels[i] == level:
                if start is None:
                    start = i
            elif start is not None:
                regions.append((int(start * C.DT_MIN), int(i * C.DT_MIN)))
                start = None
        if start is not None:
            regions.append((int(start * C.DT_MIN), int(n_slots * C.DT_MIN)))
        regions.sort(key=lambda r: -(r[1] - r[0]))
        regions = sorted(regions[:2], key=lambda r: r[0])
        cw_assignments.append([list(regions[0]) if len(regions) >= 1 else [0, 0],
                               list(regions[1]) if len(regions) >= 2 else [0, 0]])

    # random_var_w: day-to-day std of the first active slot / half a day
    threshold = 0.5 * profile_W.max() if profile_W.max() > 0 else 1.0
    starts = [int(np.argmax(daily_seas[d] > threshold))
              for d in range(n_days) if (daily_seas[d] > threshold).any()]
    if len(starts) >= 2:
        random_var_w_init = float(np.std(starts) / max(1.0, n_slots / 2))
    else:
        random_var_w_init = 0.02
    random_var_w_init = float(np.clip(random_var_w_init, 0.0, 0.15))

    # r_c per level: day-to-day coefficient of variation of the level power
    r_c_inits, r_c_stds = [0.10] * n_levels, [0.05] * n_levels
    for level in range(n_levels):
        in_level = slot_levels == level
        if in_level.sum() < 2:
            continue
        per_day = daily_seas[:, in_level].mean(axis=1)
        if per_day.mean() > 0:
            r_c_inits[level] = float(np.clip(per_day.std() / per_day.mean(),
                                             0.0, 0.30))
            r_c_stds[level] = float(np.clip(r_c_inits[level] * 0.5, 0.02, 0.10))

    # time_frac_var / thermal_p_var: day-to-day variability of energy and peak
    e_per_day = daily_seas.sum(axis=1) * C.DT_MIN / 60.0
    time_frac_var_init = float(np.clip(e_per_day.std() / e_per_day.mean(),
                                       0.0, 0.30)) if e_per_day.mean() > 0 \
        else 0.05
    thermal_p_var_init = float(np.clip(daily_max.std() / daily_max.mean(),
                                       0.0, 0.30)) if daily_max.mean() > 0 \
        else 0.05

    return {"n_levels": n_levels,
            "p1_init": p1_init, "p1_std": p1_std, "p1_max_physical": rated,
            "target_peak": float(profile_W.max()),
            "p2_init": p2_init, "p2_std": p2_std,
            "duty_inits": duty_inits, "duty_stds": duty_stds,
            "supports": supports, "cw_assignments": cw_assignments,
            "random_var_w_init": random_var_w_init,
            "random_var_w_std": float(random_var_w_init * 0.5 + 0.005),
            "r_c_inits": r_c_inits, "r_c_stds": r_c_stds,
            "time_frac_var_init": time_frac_var_init,
            "time_frac_var_std": float(time_frac_var_init * 0.4 + 0.005),
            "thermal_p_var_init": thermal_p_var_init,
            "thermal_p_var_std": float(thermal_p_var_init * 0.4 + 0.005),
            "func_cycle_init": 30.0, "func_cycle_std": 10.0}


# --- 4. Search space (bounds = init +/- std, with physical safeguards) --------
def build_search_space(inits: dict):
    """Bounds, starting point x0 and parameter names of the 12-dim search."""
    lows, highs, x0, names = [], [], [], []
    n_levels = inits["n_levels"]

    def add(lo, hi, init, name):
        lows.append(float(lo))
        highs.append(float(hi))
        x0.append(float(np.clip(init, lo, hi)))
        names.append(name)

    # p1 (ON power): capped at the rated power, and the upper bound always
    # contains the observed seasonal peak (bounds locked below it made the
    # correct powers unreachable on some seasons)
    p1_i, p1_s = inits["p1_init"], inits["p1_std"]
    p1_phys = inits["p1_max_physical"]
    p1_lo = max(20.0, p1_i - p1_s, p1_i * 0.85)
    p1_hi = min(p1_phys, max(p1_i + p1_s, p1_i * 1.15,
                             1.15 * inits["target_peak"]))
    p1_lo = min(p1_lo, p1_phys * 0.95)
    if p1_hi <= p1_lo:
        p1_hi = min(p1_phys, p1_lo + 5.0)
    add(p1_lo, p1_hi, p1_i, "p1")

    # p2 (basal power): kept well below p1
    p2_i, p2_s = inits["p2_init"], inits["p2_std"]
    p2_lo = max(0.0, p2_i - p2_s, p2_i * 0.5)
    p2_hi = min(p1_lo * 0.8, max(p2_i + p2_s, p2_i * 2.0 + 5.0))
    if p2_hi <= p2_lo:
        p2_hi = p2_lo + 1.0
    add(p2_lo, p2_hi, p2_i, "p2")

    # duty per level, widened rather than pinched (the search handles it)
    for level in range(n_levels):
        d_i, d_s = inits["duty_inits"][level], inits["duty_stds"][level]
        support = inits["supports"][level]
        margin = 0.15 if support >= 0.75 else (0.25 if support >= 0.50
                                               else 0.35)
        d_lo = max(0.05, min(d_i - d_s, d_i * (1 - margin)))
        d_hi = min(0.95, max(d_i + d_s, d_i * (1 + margin)))
        if d_hi <= d_lo:
            d_hi = d_lo + 0.02
        add(d_lo, d_hi, d_i, f"duty_{level}")

    # func_cycle (full compressor cycle duration, 10-60 min)
    fc_i, fc_s = inits["func_cycle_init"], inits["func_cycle_std"]
    fc_lo = max(10, int(fc_i - fc_s))
    fc_hi = min(60, int(fc_i + fc_s + 1))
    if fc_hi <= fc_lo:
        fc_hi = fc_lo + 5
    add(fc_lo, fc_hi, fc_i, "func_cycle")

    # random_var_w (window boundary variability), safeguarded at 0.15
    rv_i, rv_s = inits["random_var_w_init"], inits["random_var_w_std"]
    rv_lo = max(0.0, rv_i - rv_s, rv_i * 0.4)
    rv_hi = min(0.15, max(rv_i + rv_s, rv_i * 1.5 + 0.02))
    if rv_hi <= rv_lo:
        rv_hi = rv_lo + 0.005
    add(rv_lo, rv_hi, rv_i, "random_var_w")

    # r_c per level (cycle duration variability)
    for level in range(n_levels):
        rc_i, rc_s = inits["r_c_inits"][level], inits["r_c_stds"][level]
        rc_lo = max(0.0, rc_i - rc_s, rc_i * 0.5)
        rc_hi = min(0.35, max(rc_i + rc_s, rc_i * 1.5 + 0.02))
        if rc_hi <= rc_lo:
            rc_hi = rc_lo + 0.02
        add(rc_lo, rc_hi, rc_i, f"r_c_{level}")

    # time_fraction_random_variability and thermal_p_var, both capped at 0.10
    for name in ("time_frac_var", "thermal_p_var"):
        v_i, v_s = inits[f"{name}_init"], inits[f"{name}_std"]
        v_lo = max(0.0, v_i - v_s, v_i * 0.5)
        v_hi = min(0.10, max(v_i + v_s, v_i * 1.5 + 0.01))
        if v_hi <= v_lo:
            v_hi = v_lo + 0.01
        add(v_lo, v_hi, v_i, name)

    return np.array(lows), np.array(highs), x0, names


# --- 5. Decode x and simulate with RAMP ----------------------------------------
def decode_x(x: list, names: list, inits: dict) -> dict:
    """Vector x -> RAMP configuration (shared p1/p2, one cycle per level)."""
    d = dict(zip(names, x))
    p1, p2 = float(d["p1"]), float(d["p2"])
    cycles = []
    for level in range(inits["n_levels"]):
        duty, fc = float(d[f"duty_{level}"]), float(d["func_cycle"])
        cycles.append({"p1": p1, "t1": max(1, int(round(duty * fc))),
                       "p2": p2, "t2": max(1, int(round((1 - duty) * fc))),
                       "r_c": float(d[f"r_c_{level}"]), "duty": duty})
    return {"p1": p1, "p2": p2, "cycles": cycles,
            "func_cycle": float(d["func_cycle"]),
            "random_var_w": float(d["random_var_w"]),
            "time_frac_var": float(d["time_frac_var"]),
            "thermal_p_var": float(d["thermal_p_var"])}


def close_cw_gaps(cw_assignments: list) -> list:
    """Duty-cycle windows extended into a full tiling of the day [0, 1440].
    Rationale: a switch-on event drawn inside a gap between windows as a cause
    of endless RAMP redraw (no window to play for a gap-drawn event); under a
    full tiling, every event inside a window. Split of each gap at its midpoint; already-tiled seasons unchanged."""
    wins = []
    for level, pair in enumerate(cw_assignments):
        for k, cw in enumerate(pair):
            if cw[1] > cw[0]:
                wins.append([int(cw[0]), int(cw[1]), level, k])
    if not wins:
        return cw_assignments
    wins.sort(key=lambda w: w[0])
    wins[0][0] = 0
    wins[-1][1] = 1440
    for prev, nxt in zip(wins, wins[1:]):
        if nxt[0] > prev[1]:
            mid = (prev[1] + nxt[0]) // 2
            prev[1], nxt[0] = mid, mid
    closed = [[[0, 0], [0, 0]] for _ in cw_assignments]
    for start, end, level, k in wins:
        closed[level][k] = [start, end]
    return closed


def simulate(decoded: dict, cw_assignments: list, n_seeds: int,
             n_days: int) -> np.ndarray:
    """Mean 96-slot day over n_seeds RAMP runs of n_days each."""
    from ramp import User, UseCase

    cycles = decoded["cycles"]
    # func_time = total coverage of the cycle windows, clamped to [60, 1440],
    # and kept below the jitter-shrunk window so RAMP can place the cycles
    cw_coverage = sum(max(0, cw[1] - cw[0])
                      for pair in cw_assignments for cw in pair)
    func_time = float(min(1440, max(60, cw_coverage)))
    jitter = int(decoded["random_var_w"] * 1440)
    func_time = min(func_time, 1440 - jitter - 20)

    profiles = []
    for seed in range(n_seeds):
        # Seed both generators: RAMP as a consumer of both numpy AND the stdlib
        # random module, hence irreproducible runs under numpy-only seeding
        np.random.seed(C.RANDOM_SEED + seed)
        random.seed(C.RANDOM_SEED + seed)
        start = datetime.date(2024, 1, 1)
        end = start + datetime.timedelta(days=int(n_days) - 1)
        try:
            # Silence RAMP's per-run "You will simulate ..." banner; thousands
            # of runs would otherwise flood the notebook log.
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                use_case = UseCase(name="thermal_equipment",
                                   date_start=start.strftime("%Y-%m-%d"),
                                   date_end=end.strftime("%Y-%m-%d"))
            user = User(user_name="household")
            use_case.add_user(user)
            app = user.add_appliance(
                name="Cold_app", number=1,
                power=float(decoded["p1"]),
                num_windows=1,  # single 24h window: compressor always plugged
                func_time=int(round(func_time)),
                time_fraction_random_variability=float(np.clip(
                    decoded["time_frac_var"], 0.0, 0.5)),
                func_cycle=int(round(decoded["func_cycle"])),
                fixed="yes",
                fixed_cycle=len(cycles),
                occasional_use=1.0,
                flat="no",
                thermal_p_var=float(np.clip(decoded["thermal_p_var"], 0.0,
                                            C.THERMAL_P_VAR_MAX)))
            app.windows(window_1=[0, 1440], window_2=[0, 0], window_3=[0, 0],
                        random_var_w=float(np.clip(decoded["random_var_w"],
                                                   0.0, 0.40)))
            for idx, cyc in enumerate(cycles[:3], start=1):
                getattr(app, f"specific_cycle_{idx}")(
                    float(cyc["p1"]), int(round(cyc["t1"])),
                    float(cyc["p2"]), int(round(cyc["t2"])),
                    float(np.clip(cyc.get("r_c", 0.10), 0.0, 0.5)))
            # Each cycle is active on the 2 sub-windows of its power level,
            # extended into a gap-free tiling of the day
            cb_args = [list(cw) for pair in close_cw_gaps(cw_assignments)[:3]
                       for cw in pair]
            while len(cb_args) < 6:
                cb_args.append([0, 0])
            app.cycle_behaviour(*cb_args)

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                arr = use_case.generate_daily_load_profiles(flat=False)
            arr = np.asarray(arr).flatten()
            if arr.size >= 1440:
                k = arr.size // 1440
                day = arr[:k * 1440].reshape(k, 1440).mean(axis=0)
            else:
                day = np.zeros(1440)
            # Aggregate the 1440-min day to 96 slots of 15 min
            profiles.append(day.reshape(C.N_SLOTS, int(C.DT_MIN)).mean(axis=1))
        except Exception:
            continue

    if not profiles:
        return np.zeros(C.N_SLOTS)
    return np.mean(profiles, axis=0)


# --- 6. Optimisation (LHS screening + Nelder-Mead + full-fidelity polish) ------
def evaluate(x: list, names: list, inits: dict, target: np.ndarray,
             cw_assignments: list, n_seeds: int, n_days: int) -> dict:
    decoded = decode_x(x, names, inits)
    sim = simulate(decoded, cw_assignments, n_seeds, n_days)
    if sim.sum() <= 0:
        return {"score": 10.0, "metrics": None, "decoded": decoded, "sim": None}
    metrics = all_metrics(target, sim)
    return {"score": composite_score(metrics), "metrics": metrics,
            "decoded": decoded, "sim": sim}


def optimize_season(args):
    """Full search for one season, run in its own process: analytical inits,
    Latin-Hypercube screening on real RAMP, Nelder-Mead from the best starts,
    full-fidelity re-ranking of the finalists, then a long-horizon polish."""
    season, target, daily_seas, rated, final_days = args
    t0 = time.time()
    inits = compute_inits(target, daily_seas, rated)
    cw = inits["cw_assignments"]
    lo, hi, x0, names = build_search_space(inits)

    def f_search(x):
        xc = [max(l, min(h, v)) for v, l, h in zip(x, lo, hi)]
        return evaluate(xc, names, inits, target, cw,
                        SEARCH_SEEDS, SEARCH_DAYS)["score"]

    # Global screening: the analytical start plus LHS_POINTS box samples
    sampler = qmc.LatinHypercube(d=len(lo), seed=C.RANDOM_SEED)
    X = qmc.scale(sampler.random(LHS_POINTS), lo, hi)
    X = np.vstack([np.asarray(x0, dtype=float), X])
    scores = np.array([f_search(x) for x in X])

    # Local refinement from the best screening starts
    finalists = []
    for idx in np.argsort(scores)[:NM_STARTS]:
        try:
            res = minimize(f_search, X[idx], method="Nelder-Mead",
                           options=dict(maxiter=NM_MAXITER, xatol=1e-3,
                                        fatol=1e-3, adaptive=True))
            finalists.append(list(res.x))
        except Exception:
            continue
    finalists.append(list(X[int(np.argmin(scores))]))

    # Full-fidelity re-ranking of the finalists: the reported optimum is the
    # best real one, not the best cheap estimate
    best, best_x = None, None
    for x in finalists:
        xc = [max(l, min(h, v)) for v, l, h in zip(x, lo, hi)]
        cand = evaluate(xc, names, inits, target, cw,
                        C.FINAL_N_SEEDS, final_days)
        if cand["metrics"] and (best is None or cand["score"] < best["score"]):
            best, best_x = cand, list(xc)
    if best is None:
        return season, {"score": 10.0, "metrics": None, "decoded": None,
                        "sim": None, "x": None, "names": names,
                        "cw": cw}, time.time() - t0

    # Long-horizon polish: a short Nelder-Mead at near-final fidelity removes
    # the drift between the short search horizon and the full-length
    # validation (energy and peak can drift over ~200 days)
    pol_days = int(min(POLISH_DAYS, final_days))

    def f_long(x):
        xc = [max(l, min(h, v)) for v, l, h in zip(x, lo, hi)]
        return evaluate(xc, names, inits, target, cw,
                        POLISH_SEEDS, pol_days)["score"]

    try:
        res = minimize(f_long, np.asarray(best_x, dtype=float),
                       method="Nelder-Mead",
                       options=dict(maxiter=POLISH_MAXITER, xatol=1e-3,
                                    fatol=1e-3, adaptive=True))
        xp = [max(l, min(h, v)) for v, l, h in zip(res.x, lo, hi)]
        cand = evaluate(xp, names, inits, target, cw,
                        C.FINAL_N_SEEDS, final_days)
        if cand["metrics"] and cand["score"] < best["score"]:
            best, best_x = cand, xp
    except Exception:
        pass

    best["x"] = best_x
    best["names"] = names
    best["cw"] = cw
    return season, best, time.time() - t0


# --- 7. Full step 3 for one client ---------------------------------------------
def calibrate_client(client_code: str) -> dict:
    print(f"\n=== STEP 3 - CALIBRATION : {client_code} ===", flush=True)
    rdir = C.client_results_dir(client_code)
    daily = pd.read_csv(rdir / C.DAILY_MATRIX_CSV, index_col=0,
                        parse_dates=True)
    feats_clust = pd.read_csv(rdir / C.CLUSTERED_CSV, index_col=0,
                              parse_dates=True)
    seas_repr = pd.read_csv(rdir / C.SEASONAL_PROFILES_CSV)
    slot_cols = [c for c in seas_repr.columns if c.startswith("slot_")]
    day_seasons = pd.Series(daily.index).dt.month.apply(
        C.month_to_season).values
    p1_global = float(np.percentile(daily.values.max(axis=1), 75))

    # One job per season; the days of the dominant cluster feed the inits
    jobs = []
    for _, srow in seas_repr.iterrows():
        season = str(srow["season"])
        profile = srow[slot_cols].values.astype(float)
        if profile.sum() <= 0:
            print(f"  [{season}] empty profile, skipped", flush=True)
            continue
        in_season = day_seasons == season
        if in_season.any():
            daily_seas = daily.values[in_season]
            p1_season = float(np.percentile(daily_seas.max(axis=1), 75))
        else:
            daily_seas = daily.values
            p1_season = p1_global
        cluster_dom = int(srow["cluster_dominant"])
        if cluster_dom != -1:
            mask = in_season & (feats_clust["cluster"] == cluster_dom).values
            if mask.any():
                daily_seas = daily.values[mask]
        rated = C.RATED_POWER_W or p1_season
        jobs.append((season, profile, daily_seas, rated,
                     int(daily_seas.shape[0])))
        print(f"  [{season}] LHS {LHS_POINTS} + NM {NM_STARTS}x{NM_MAXITER} "
              f"({SEARCH_SEEDS} seeds x {SEARCH_DAYS} d), final "
              f"{C.FINAL_N_SEEDS} seeds x {daily_seas.shape[0]} d", flush=True)

    export = {"client": client_code, "seasons": {}}
    rows, sim_rows = [], []
    with ProcessPoolExecutor(max_workers=3) as pool:
        for season, result, elapsed in pool.map(optimize_season, jobs):
            if result["metrics"] is None:
                print(f"  [{season}] empty simulation, season skipped",
                      flush=True)
                continue
            decoded = result["decoded"]
            metrics = result["metrics"]
            n_pass = count_passed(metrics)
            print(f"  [{season}] {elapsed:.0f}s  score={result['score']:.4f}  "
                  f"n_pass={n_pass}/6  NRMSE={metrics['NRMSE']:.3f}  "
                  f"LDC={metrics['LDC_err']:.3f}  FFT={metrics['FFT_err']:.3f}  "
                  f"E={metrics['err_E_pct']:+.1f}%  "
                  f"P={metrics['err_P_pct']:+.1f}%  "
                  f"LF={metrics['err_LF']:+.3f}", flush=True)

            # Flat list of the effective cycle windows, for the export and step 4
            windows = []
            for cyc_idx, pair in enumerate(result["cw"]):
                cyc = decoded["cycles"][cyc_idx]
                for sub_idx, cw in enumerate(pair):
                    if cw[1] <= cw[0]:
                        continue
                    windows.append({"window_id": len(windows),
                                    "cycle_idx": cyc_idx,
                                    "cw_sub_idx": sub_idx + 1,
                                    "start_h": cw[0] / 60.0,
                                    "end_h": cw[1] / 60.0,
                                    "p1_W": float(cyc["p1"]),
                                    "p2_W": float(cyc["p2"]),
                                    "t1_min": float(cyc["t1"]),
                                    "t2_min": float(cyc["t2"]),
                                    "duty": float(cyc["duty"])})
            func_time = min(1440, sum(cw[1] - cw[0] for pair in result["cw"]
                                      for cw in pair if cw[1] > cw[0]))
            export["seasons"][season] = {
                "validation": {k: float(v) for k, v in metrics.items()},
                "n_pass": n_pass,
                "converged": n_pass == 6,
                "global_params": {
                    "func_time_min": float(func_time),
                    "func_cycle_min": float(int(decoded["func_cycle"])),
                    "random_var_w": float(decoded["random_var_w"]),
                    "occasional_use": 1.0,
                    "time_frac_var": float(decoded["time_frac_var"]),
                    "thermal_p_var": float(decoded["thermal_p_var"])},
                "windows": windows,
            }
            rows.append({"season": season, "n_dimensions": len(result["names"]),
                         "score": result["score"], "n_pass": n_pass,
                         **{k: metrics[k] for k in ("NRMSE", "LDC_err",
                                                    "FFT_err", "err_E_pct",
                                                    "err_P_pct", "err_LF")},
                         "x_final_json": json.dumps([float(v) for v in
                                                     result["x"]]),
                         "param_names_json": json.dumps(result["names"]),
                         "decoded_json": json.dumps(decoded)})
            sim_rows.append({"season": season,
                             **{f"slot_{j}": float(v)
                                for j, v in enumerate(result["sim"])}})

    # Exports: full parameter table, validation extract, simulated profiles
    # (step 4 figures) and the JSON consumed by step 5
    val_cols = ["season", "score", "n_pass", "NRMSE", "LDC_err", "FFT_err",
                "err_E_pct", "err_P_pct", "err_LF"]
    df_out = pd.DataFrame(rows, columns=["season", "n_dimensions"] + val_cols[1:]
                          + ["x_final_json", "param_names_json", "decoded_json"])
    df_out.to_csv(rdir / C.RAMP_PARAMS_CSV, index=False)
    df_out[val_cols].to_csv(rdir / C.VALIDATION_CSV, index=False)
    pd.DataFrame(sim_rows, columns=["season"] + [f"slot_{j}" for j in
                                                 range(C.N_SLOTS)]
                 ).to_csv(rdir / C.SIM_PROFILES_CSV, index=False)
    with open(rdir / C.CALIB_EXPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)
    print(f"  -> {C.CALIB_EXPORT_JSON}, {C.RAMP_PARAMS_CSV}, "
          f"{C.VALIDATION_CSV}, {C.SIM_PROFILES_CSV} in {rdir}", flush=True)

    return {"client": client_code, "n_seasons": len(rows),
            "mean_NRMSE": float(df_out["NRMSE"].mean()) if rows else np.nan}


if __name__ == "__main__":
    calibrate_client(C.PUE_CLIENT)
