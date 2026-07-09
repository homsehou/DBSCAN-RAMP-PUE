#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuration of the grain-milling chain: paths, time grid, clustering and
calibration constants. The client under study is selected through the PUE_TYPE
and PUE_CLIENT environment variables."""

import os
import sys

# Resolution of the saved figures, set once for the whole chain. A moderate value
# keeps the PNG files light and has no effect on the calibration results.
import matplotlib
matplotlib.rcParams["savefig.dpi"] = 110

# Folder layout: inputs in data/<type>/<client>/, outputs in resultats/<type>/<client>/.
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SRC_DIR))
PUE_TYPE = os.environ.get("PUE_TYPE", "grain_milling")
PUE_CLIENT = os.environ.get("PUE_CLIENT", "0016GBO")

# Client identity (CSV path and study period) served by the dynamic registry in src/.
if os.path.dirname(SRC_DIR) not in sys.path:
    sys.path.insert(0, os.path.dirname(SRC_DIR))
from registry import client_info

INFO = client_info(PUE_TYPE, PUE_CLIENT)
RAW_CSV = str(INFO["csv"])
# Study period as plain dates "YYYY-MM-DD"; both bound days inclusive.
STUDY_PERIOD_START = os.environ.get("PUE_PERIOD_START") or INFO["period_start"]
STUDY_PERIOD_END = os.environ.get("PUE_PERIOD_END") or INFO["period_end"]

RESULTS_DIR = os.path.join(REPO_ROOT, "resultats", PUE_TYPE, PUE_CLIENT)
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Output files of the five steps.
CLEAN_TS_CSV = os.path.join(RESULTS_DIR, "timeseries_clean.csv")
DAILY_MATRIX_CSV = os.path.join(RESULTS_DIR, "daily_matrix_W.csv")
FEATURES_CSV = os.path.join(RESULTS_DIR, "features_by_day.csv")
ENTROPY_CSV = os.path.join(RESULTS_DIR, "entropy_by_day.csv")
FEATURES_WITH_CLUSTERS_CSV = os.path.join(RESULTS_DIR, "features_with_clusters.csv")
CLUSTERED_CSV = os.path.join(RESULTS_DIR, "clustered_features.csv")
CLUSTER_PROFILES_CSV = os.path.join(RESULTS_DIR, "cluster_profiles_W.csv")
CLUSTER_FREQ_GLOBAL_CSV = os.path.join(RESULTS_DIR, "cluster_freq_global.csv")
CLUSTER_FREQ_BY_SEASON_CSV = os.path.join(RESULTS_DIR, "cluster_freq_by_season_local.csv")
SEASONAL_PROFILES_CSV = os.path.join(RESULTS_DIR, "seasonal_representative_profiles.csv")
SEASONAL_PROFILES_MEAN_CSV = os.path.join(RESULTS_DIR, "seasonal_representative_profiles_mean.csv")
ANALYTICAL_PARAMS_CSV = os.path.join(RESULTS_DIR, "analytical_params.csv")
RAMP_PARAMS_CSV = os.path.join(RESULTS_DIR, "ramp_calibrated_params.csv")
VALIDATION_CSV = os.path.join(RESULTS_DIR, "validation_metrics.csv")
SIM_PROFILES_CSV = os.path.join(RESULTS_DIR, "ramp_simulated_profiles.csv")
CALIB_EXPORT_JSON = os.path.join(RESULTS_DIR, "calibration_export.json")

# Raw CSV columns.
RAW_DATETIME_COL = "Date"
RAW_POWER_COL = "Average_Active_power"

# Time grid: 96 slots of 15 minutes per day; daytime block 06:00-18:00.
TIME_FREQ = "15min"
DT_MIN = 15.0
N_SLOTS = 96
INTERP_MAX_GAP_H = 1.0
DAY_START_H = 6
DAY_END_H = 18

# Entropy screen plus hybrid DBSCAN: distance from the PCA-reduced daily
# profiles (time-series term) plus the standardized daily features,
# per-block weights DBSCAN_W_TS and DBSCAN_W_FEAT.
# Entropy threshold at 0.04: lower bound on daily entropy; days below it
# as degenerate flat/idle days to drop, days at or above genuine activity to keep.
ENTROPY_THRESHOLD = 0.04
CLUSTERING_FEATURES = ["p_peak", "p_mean", "lf", "hour_of_peak_min",
                       "ratio_day_night", "entropy"]
# PCA components from the 96-point daily profiles for the time-series
# distance term: 10 components as a compact basis for the dominant shape variance
# while dropping slot-level noise.
DBSCAN_N_COMPONENTS = 10
DBSCAN_W_TS = 1.0
DBSCAN_W_FEAT = 1.0
DBSCAN_K_NEIGHBORS = 5
# eps at the 90th percentile of the k-nearest-neighbour distances (k-distance
# heuristic): a high percentile above the standard knee, to absorb sparse outliers
# into noise instead of forcing a cluster around them.
DBSCAN_EPS_PERCENTILE = 90

# Benin seasons (center region): dry November-April, transition May and October,
# rainy June-September. The same three seasons apply to every client.
SEASON_BY_MONTH = {
    11: "Dry season", 12: "Dry season", 1: "Dry season", 2: "Dry season",
    3: "Dry season", 4: "Dry season",
    5: "Transition", 10: "Transition",
    6: "Rainy season", 7: "Rainy season", 8: "Rainy season",
    9: "Rainy season",
}
SEASON_ORDER = ["Dry season", "Transition", "Rainy season"]

# Local season labels (DRY/TRANSITION/WET) and their project equivalents.
SEASON_MAP_LOCAL_TO_PROJECT = {"DRY": "Dry season", "TRANSITION": "Transition",
                               "WET": "Rainy season"}


def month_to_season(month):
    """Project season label of a calendar month."""
    return SEASON_BY_MONTH.get(int(month), "Unknown")


def local_season_of_date(timestamp):
    """Local season label (DRY/TRANSITION/WET) of a date."""
    month = int(timestamp.month)
    if month in (11, 12, 1, 2, 3, 4):
        return "DRY"
    if month in (6, 7, 8, 9):
        return "WET"
    return "TRANSITION"


# Statistic kept on the representative profiles for steps 3 and 4.
PROFILE_STATISTIC_FOR_CALIBRATION = "mean"

# Validation thresholds (step 4).
MAX_ERR_E_PCT = 10.0
MAX_ERR_P_PCT = 10.0
MAX_ERR_LF = 0.05
# NRMSE (Normalized Root-Mean-Square Error): RMSE between simulated and reference
# profiles, relative to the reference scale.
MAX_NRMSE = 0.15
# LDC (Load Duration Curve): the daily power values sorted in decreasing order,
# for the time-independent view of the load; error as the gap between the two curves.
MAX_LDC_ERR = 0.15
# FFT (Fast Fourier Transform): frequency-domain comparison of the two profiles.
MAX_FFT_ERR = 0.15

# FFT comparison settings.
FFT_N_KEEP = 10
FFT_REMOVE_DC = True
FFT_LOG_SCALE = True

# Reproducibility and validation diagnostics.
RANDOM_SEED = 42
VALID_N_DAILY_DATES = 5

# Step 3 window-detection constants.
# Activity threshold as a fraction of the daily peak: slots above 5% of the peak
# as on-slots, low enough to catch light milling yet above the sensor baseline.
ACTIVITY_THRESHOLD_FRAC = 0.05
# Minimum gap in minutes between two activity windows to keep them separate;
# below 30 min a single merged window instead.
MIN_GAP_BETWEEN_WINDOWS = 30
# Minimum window duration in minutes, one grid slot, to drop single-slot spikes.
MIN_WINDOW_DURATION = 15

# Step 4 optimization constants. Phase A: search-space screening with a Latin
# Hypercube (LHS) sampled directly on the real RAMP engine. Phase B: refinement
# of the best point with Nelder-Mead, refinement passes arbitrated at full
# fidelity. K_SIGMA for the width of the data-driven bounds around the step-3
# analytical estimates (about a 95% interval). Search over OPT_N_SEEDS cheap
# common-seed evaluations; final evaluations at FINAL_N_SEEDS full length.
LHS_POINTS = 400
K_SIGMA = 2.0
NM_MAXITER = 600
NM_XATOL = 1e-6
NM_FATOL = 1e-7
OPT_N_SEEDS = 4
OPT_SIM_DAYS = 10
FINAL_N_SEEDS = 20
FINAL_SIM_DAYS = 90

# Objective-score weights (step 4): coefficients of the normalized-metric terms
# in the composite score, normalized by their sum. NRMSE and FFT highest for
# point-wise and spectral shape fidelity as the primary target; E and P next for
# energy and peak levels; LDC moderate; LF lowest as a coarse aggregate already
# carried by the energy and peak terms.
W_NRMSE = 3.5
W_LDC = 1.5
W_FFT = 3.0
W_E = 2.0
W_P = 2.5
W_LF = 0.5
