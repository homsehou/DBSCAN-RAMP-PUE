# -*- coding: utf-8 -*-
"""Settings common to every client: thresholds, seeds, seasons, colours.

Client-specific inputs (rated power, study period) in the notebooks.

Import from a step script:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import config as C
"""
import os

import numpy as np

# --- Measurement grid and day cleaning (step 1) --------------------------------
UTC_OFFSET_H = 1        # Benin local time = UTC + 1
VOLTAGE_MIN = 200.0     # readings below this voltage = grid outage, not usage
SLOTS_PER_DAY = 96      # 24 hours in steps of 15 minutes
MIN_SLOTS_KEPT = 90     # a day with fewer valid slots is dropped
GAP_INTERP_SLOTS = 2    # interpolation limited to 30 min of missing data
GAP_REJECT_SLOTS = 3    # a longer hole inside a day rejects the whole day
DORMANT_SHARE = 0.25    # a day below 25 % of the active reference is inactive
STOP_BLOCK_DAYS = 30    # 30+ inactive days in a row = operation stop

# Unit conversions and numerical floors kept in the formulas themselves

# --- Day clustering (step 2 and its figures) -----------------------------------
FEATURES = ["E_kWh", "P_peak_W", "Load_factor", "peak_sin", "peak_cos",
            "ratio_day_night"]
PCA_MAX_COMPONENTS = 10     # shape summary kept below 10 dimensions
PCA_VARIANCE_KEPT = 0.90    # enough components to explain 90 % of variance
K_NEIGHBORS = 5             # k of the k-distance rule and DBSCAN min_samples
EPS_PERCENTILE = 90         # radius = high percentile of the k-distances

# --- Calibration seasons: May and October get their own model each -------------
SEASONS = {11: "Dry season", 12: "Dry season", 1: "Dry season",
           2: "Dry season", 3: "Dry season", 4: "Dry season",
           5: "May", 10: "October",
           6: "Rainy season", 7: "Rainy season", 8: "Rainy season",
           9: "Rainy season"}
SEASON_ORDER = ["Dry season", "May", "Rainy season", "October"]

# --- Figure style, identical across every script -------------------------------
HOURS = np.arange(SLOTS_PER_DAY) * 0.25     # slot number -> decimal hour
SEASON_COLORS = {"Dry season": "#E74C3C", "May": "#F39C12",
                 "October": "#9B59B6", "Rainy season": "#3498DB"}
PALETTE = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
           "#e377c2", "#17becf", "#bcbd22", "#1a9850", "#7f7f7f"]
# Orange reserved for DBSCAN noise, never a cluster colour

# --- Calibration (thermal step 4, milling step 5) ------------------------------
SEED, EVAL_SEED = 42, 1042  # tuning / reporting random streams (disjoint)
SLOT_MIN = 15               # one slot lasts 15 minutes
THRESH = {"NRMSE": 0.10, "LDC_err": 0.10, "FFT_err": 0.10,
          "err_E_pct": 10.0, "err_P_pct": 10.0, "err_LF": 0.05}
AMP_LOW, AMP_HIGH = 0.9, 1.1    # accepted band for the amplitude ratio
# Size of the two RAMP simulation stages, in seeds x days
TUNE_SEEDS, TUNE_DAYS = 12, 60      # cheap runs, used while correcting
EVAL_SEEDS, EVAL_DAYS = 20, 120     # careful runs, used for the verdict
SIM_START_DATE = "2024-01-01"       # arbitrary calendar start of a RAMP run
RAMP_VARIABILITY = 0.10             # RAMP jitter on burst power and duration
# Damped correction loop against the untouched engine
FIXED_POINT_ROUNDS = 6              # cheap correction rounds before evaluating
POLISH_SELECT = 4                   # polish rounds when selecting the model
POLISH_DELIVERED = 8                # ... and when fitting the delivered one
POLISH_NRMSE = 0.03                 # polishing stops below this shape error
POLISH_PEAK_TOL = 0.05              # ... and within 5 % of the target peak
DAMP_FIRST, DAMP_LATE = 0.8, 0.6    # correction damping, early then late
DAMP_SWITCH = 2                     # round at which damping drops
CORRECT_LOW, CORRECT_HIGH = 0.5, 2.0    # a correction never exceeds x0.5..x2
TFRV_CAP = 0.95                     # ceiling of the native time variability
# Measured anchors, taken from the day population
P1_PCTL = 90            # burst power percentile of the daily maxima; the
                        # step-12 sweep (P90/95/99, 0017SAM and 0016GBO)
                        # picks P90: best n_pass and dP on both families
BURST_PCTL = 75         # burst duration anchored on this percentile
P2_PCTL = 5             # standby = this percentile of the positive readings
P2_SAFETY = 0.9         # ... and at most 90 % of the target minimum
MIN_SPLIT_DAYS = 5      # each half of the even/odd split needs this many days
MIN_ACF_DAYS = 10       # day-to-day persistence estimated only above this
# Acceptance of the good-day / bad-day factor
ACTIVITY_CV_MIN = 0.05          # below this missing spread, do not bother
ACTIVITY_NRMSE_MARGIN = 0.005   # accepted shape loss for keeping the factor
ACTIVITY_NRMSE_CAP = 0.05       # ... or any NRMSE under this value
CI_BLOCKS = 5                   # seed blocks used for the confidence margin
# Surrogate candidate search (thermal step 5, milling step 6)
SEARCH_SEED = 7042      # random stream of the search, disjoint from the others
SEARCH_DAYS = 40        # replica days behind every candidate score
RESCORE_DAYS = 160      # ... and behind the second look at the short list
RESCORE_KEEP = 30       # candidates kept for the second look, per branch
SEARCH_TOPK = 8         # candidates verified by the real engine, per branch
SEARCH_CENTERS = 2      # exact-inversion candidates per screened pair
SEARCH_PER_STRUCT = 12  # refinement draws per surviving structure and power
SEARCH_KEEP_PAIRS = 60  # structure x power pairs surviving the screening
# Powers offered to the search, as percentiles of the measured population.
# The anchors P1_PCTL / P2_PCTL stay the delivered values of the classic
# arm; the screened arm may prefer another percentile, and the engine has
# the last word. A 15-minute meter cannot tell a short strong burst from
# a long weak one, so the criteria settle what the measurement cannot.
SEARCH_P1_PCTLS = (60, 70, 75, 80, 85, 90, 95, 99)
SEARCH_P2_PCTLS = (1, 5, 10, 20)
SEARCH_P_VARS = (0.0, 0.05, 0.10, 0.20, 0.30)   # engine dispersion tried
HOLDOUT_VETO = True     # a screened model losing out of sample is refused
SHAPE_CAP = 1.5         # NRMSE cap factor over the split floor (shape guard)
# Export keys, common to both families
# Season entries of calibration_export.json (steps 4-6 fill them in turn)
EXPORT_KEYS = ("p1_W", "p2_W", "t1_amp_min", "fit", "holdout", "n_pass",
               "method_arm", "p1_anchor_W", "p2_anchor_W", "thermal_p_var",
               "holdout_veto", "n_pass_holdout", "NRMSE_split_floor",
               "ratio_daily_max",
               "activity_cv", "tfrv", "occasional_use", "cv_E_meas",
               "cv_E_sim", "NRMSE_ci95", "FFT_ci95", "amp_note", "n_rescue",
               "holdout_valid", "all_criteria_met", "native")
# Scalar columns of validation_metrics.csv (metrics travel separately)
SCALAR_KEYS = tuple(k for k in EXPORT_KEYS
                    if k not in ("fit", "holdout", "native"))

# --- Daily descriptors and activity timeline (step 1) --------------------------
DAY_START_SLOT, DAY_END_SLOT = 6 * 4, 18 * 4   # "daytime" = 06:00 to 18:00
RATIO_DAY_NIGHT_CAP = 10.0      # a night close to zero would blow the ratio up
ACTIVE_REF_PCTL = 0.75          # typical active day = this quantile of energy
TIMELINE_WINDOW_D = 30          # rolling median width of the activity regime
TIMELINE_MIN_D = 10             # ... and the minimum count to draw a point

# --- Milling activity windows (milling step 4) ---------------------------------
ACTIVE_FRACTION = 0.10   # a slot is active above 10 % of the profile peak
ACTIVE_MIN_W = 50.0      # ... and above 50 W (below = sensor noise)
WINDOW_GAP_SLOTS = 2     # windows separated by < 30 min are merged
MIN_LEN_SLOTS = 2        # windows shorter than 30 min are dropped
MAX_WINDOWS = 3          # RAMP accepts at most three windows per appliance
PEAK_PROMINENCE = 0.2    # a real peak stands out by 20 % of the local maximum
PEAK_DISTANCE_SLOTS = 8  # ... and two peaks are at least 2 hours apart
VALLEY_SPLIT = 0.55      # split a window only if the dip falls below 55 %
DISPERSION_SLOTS = 8     # widen a window by 2 hours to catch early/late starts


def tfrv_from(cv_a):
    """Native time variability reproducing a daily-energy spread cv_a.

    The engine draws rand_time uniformly in [(1-f)T, 0.99T] (top capped),
    so cv = (f - 0.01)/(sqrt(3)(1.99 - f)); inverted for f."""
    k = np.sqrt(3.0) * max(cv_a, 0.0)
    return float(min(TFRV_CAP, (0.01 + 1.99 * k) / (1.0 + k)))


def metrics(real, sim):
    """Six validation criteria between two 96-slot profiles.

    Single definition for both chains, for a common yardstick.
    """
    span = max(real.max() - real.min(), 1e-9)
    r_s, s_s = np.sort(real)[::-1], np.sort(sim)[::-1]
    R = np.log(np.abs(np.fft.rfft(real))[1:11] + 1.0)
    S = np.log(np.abs(np.fft.rfft(sim))[1:11] + 1.0)
    return {"NRMSE": float(np.sqrt(np.mean((real - sim) ** 2)) / span),
            "LDC_err": float(np.sqrt(np.mean((r_s - s_s) ** 2)) / span),
            "FFT_err": float(np.sqrt(np.mean((R - S) ** 2))
                             / max(R.max() - R.min(), 1e-9)),
            "err_E_pct": float(100 * (sim.sum() - real.sum()) / real.sum()),
            "err_P_pct": float(100 * (sim.max() - real.max()) / real.max()),
            "err_LF": float(sim.mean()/sim.max() - real.mean()/real.max())}


def n_passed(m):
    """Number of criteria under their threshold."""
    return sum(abs(m[k]) <= v for k, v in THRESH.items())


def exceedance(m):
    """Total normalised distance to full conformity, zero when all pass.

    Every criterion counts its overshoot in units of its own threshold,
    so candidates passing the same number of criteria are compared on
    how far the failing ones still are."""
    return float(sum(max(0.0, abs(m[k]) / v - 1.0)
                     for k, v in THRESH.items()))


def confidence_margin(target, seed_profiles):
    """95 % margin on NRMSE and FFT, over independent blocks of seeds.

    Spread of the block scores, as the uncertainty of a reported score.
    """
    blocks = np.asarray(seed_profiles)
    n = len(blocks) // CI_BLOCKS
    if n < 1:
        return float("nan"), float("nan")
    means = blocks[:n * CI_BLOCKS].reshape(CI_BLOCKS, n, -1).mean(axis=1)
    scores = [metrics(target, b) for b in means]
    return (float(1.96 * np.std([s["NRMSE"] for s in scores])
                  / np.sqrt(CI_BLOCKS)),
            float(1.96 * np.std([s["FFT_err"] for s in scores])
                  / np.sqrt(CI_BLOCKS)))


def amp_note(ratio, t1_amp, rated_capped):
    """Note on the identifiability of the burst power, empty when sound.

    Averaging over 15 min confuses a short strong burst with a long weak
    one. No judgement on the nameplate here: the share actually reached
    (rated_share_max) travels with the results as a plain number.
    """
    if not np.isfinite(ratio) or AMP_LOW <= ratio <= AMP_HIGH:
        return ""
    if ratio < AMP_LOW and rated_capped:
        return "measured maxima exceed the nameplate cap"
    if ratio > AMP_HIGH and t1_amp <= 2:
        return "sub-15-min bursts, amplitude not identifiable from the meter"
    if ratio < AMP_LOW and t1_amp >= SLOT_MIN:
        return "burst spans the whole band, ceiling reached"
    return "amplitude partially identifiable at 15-min metering"


# --- Sensitivity overrides (refactoring protocol) ------------------------------
# PUE_OVERRIDE_<NAME>=<value> in the environment replaces the setting for
# one run, e.g. PUE_OVERRIDE_EPS_PERCENTILE=85. Numbers only.
for _key, _value in os.environ.items():
    if _key.startswith("PUE_OVERRIDE_") and _key[13:] in globals():
        globals()[_key[13:]] = type(globals()[_key[13:]])(float(_value))
