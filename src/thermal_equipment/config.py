# -*- coding: utf-8 -*-
"""Configuration of the cold appliance calibration pipeline, shared with the
poultry incubator (same thermal_equipment chain, its own folders under data/).

Chain driven by this file: step1_preprocess.py, step2_clustering.py,
step3_calibration.py, step4_figures.py and step5_summary.py.
"""
import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent.parent
sys.path.insert(0, str(SRC_DIR.parent))  # src/ on the path for the registry
from registry import client_info, clients  # noqa: E402

# Resolution of the saved figures, set once for the whole chain. A moderate value
# for light PNG files, without any effect on the calibration results.
import matplotlib
matplotlib.rcParams["savefig.dpi"] = 110

# --- 1. Client selection and paths -------------------------------------------
# PUE type and client code from environment variables. Client-list discovery from
# the data/<type>/<client>/<client>.csv tree (see registry.py), for a new client
# without any edit to shared files.
PUE_TYPE = os.environ.get("PUE_TYPE", "cold_chain")
PUE_CLIENT = os.environ.get("PUE_CLIENT") or clients(PUE_TYPE)[0]
CLIENTS = {code: client_info(PUE_TYPE, code) for code in clients(PUE_TYPE)}
INFO = CLIENTS[PUE_CLIENT]
RAW_CSV = str(INFO["csv"])
# Study period and rated power from the notebook (env variables set by run.run);
# absent both, fall back to the full CSV span with no power ceiling.
PERIOD_START = os.environ.get("PUE_PERIOD_START") or INFO["period_start"]
PERIOD_END = os.environ.get("PUE_PERIOD_END") or INFO["period_end"]
_rated = os.environ.get("PUE_RATED_POWER_W")
RATED_POWER_W = float(_rated) if _rated else None

RESULTS_DIR = REPO_ROOT / "resultats" / PUE_TYPE
FIG_DIR = RESULTS_DIR
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def client_results_dir(client_code: str) -> Path:
    d = RESULTS_DIR / client_code
    d.mkdir(parents=True, exist_ok=True)
    return d


def client_fig_dir(client_code: str) -> Path:
    d = RESULTS_DIR / client_code / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --- 2. Output file names -----------------------------------------------------
CLEAN_TS_CSV = "timeseries_clean.csv"
DAILY_MATRIX_CSV = "daily_matrix_W.csv"
FEATURES_CSV = "features_by_day.csv"
ENTROPY_CSV = "entropy_by_day.csv"
FEATURES_WITH_CLUSTERS_CSV = "features_with_clusters.csv"
CLUSTERED_CSV = "clustered_features.csv"
CLUSTER_PROFILES_CSV = "cluster_profiles_W.csv"
CLUSTER_FREQ_GLOBAL_CSV = "cluster_freq_global.csv"
CLUSTER_FREQ_BY_SEASON_CSV = "cluster_freq_by_season.csv"
SEASONAL_PROFILES_CSV = "seasonal_representative_profiles.csv"
ANALYTICAL_PARAMS_CSV = "analytical_params.csv"
RAMP_PARAMS_CSV = "ramp_calibrated_params.csv"
VALIDATION_CSV = "validation_metrics.csv"
SIM_PROFILES_CSV = "ramp_simulated_profiles.csv"
CALIB_EXPORT_JSON = "calibration_export.json"

# --- 3. Cleaning and time grid (15 min step) ----------------------------------
LOCAL_TZ_OFFSET_H = 1.0     # Benin = UTC+1
VOLTAGE_MIN_VALID = 200.0   # mini-grid outage filter
TIME_FREQ = "15min"
DT_MIN = 15.0
N_SLOTS = 96                # number of 15-min slots per day
MIN_POINTS_PER_DAY = 90     # day rejected below 90 valid slots
INTERP_MAX_GAP_H = 1.0
DAY_START_H = 6
DAY_END_H = 18

# --- 4. Entropy filter and clustering features --------------------------------
NORMALIZE_METHOD = "peak"   # "peak" (= /max) or "energy" (= /sum)
ENTROPY_THRESHOLD = 0.04  # Shannon entropy of the peak-normalised daily profile; floor at 0.04 to drop only degenerate flat days (recorder stuck or appliance fully off), never a genuine low-activity day.
CLUSTERING_FEATURES = ["E_kWh", "P_peak_W", "Load_factor", "peak_sin",
                       "peak_cos", "ratio_day_night"]  # no collinearity

# --- 5. Hybrid DBSCAN (step 2) ------------------------------------------------
DBSCAN_N_COMPONENTS = 10    # max PCA components for shape
DBSCAN_W_TS = 1.0           # weight of the shape block (PCA)
DBSCAN_W_FEAT = 1.0         # weight of the feature block
DBSCAN_K_NEIGHBORS = 5      # k for the k-distance heuristic
DBSCAN_EPS_PERCENTILE = 90  # eps at the 90th percentile of the k-distances rather than the hand-read knee: a high, reproducible percentile absorbing sparse days into noise instead of forcing them into a cluster.

# --- 6. Seasons of central Benin (3 cold chain seasons) -----------------------
SEASON_MAP_CENTER = {11: "Dry season", 12: "Dry season", 1: "Dry season",
                     2: "Dry season", 3: "Dry season", 4: "Dry season",
                     5: "Transition", 10: "Transition",
                     6: "Rainy season", 7: "Rainy season",
                     8: "Rainy season", 9: "Rainy season"}
SEASON_ORDER = ["Dry season", "Transition", "Rainy season"]


def month_to_season(month: int) -> str:
    return SEASON_MAP_CENTER.get(month, "Unknown")


# --- 7. Step 3 calibration -----------------------------------------------------
# The search itself (Latin-Hypercube screening + Nelder-Mead restarts on cheap
# common-seed evaluations) is configured at the top of step3_calibration.py;
# only the full-fidelity final evaluation and the score weights live here.
FINAL_N_SEEDS = 20   # seeds of the final evaluation (full season length)
# Composite score weights over the 6 normalised errors, relative magnitudes by
# importance: point-wise shape (NRMSE) and spectral content (FFT) weighted highest,
# energy and peak next as physical anchors, load factor and load-duration curve
# lowest as derived, partly redundant views.
W_NRMSE = 3.5        # NRMSE (Normalized Root-Mean-Square Error): point-wise fidelity between measured and simulated 96-slot days
W_LDC = 1.5          # LDC (Load Duration Curve): the daily power values sorted in decreasing order, time-independent view of the load
W_FFT = 3.0          # FFT (Fast Fourier Transform) magnitude match of the daily profile, for the periodic structure
W_E = 2.0            # daily energy error
W_P = 2.5            # daily peak-power error
W_LF = 0.5           # lf (load factor, mean power over peak power) error

# --- 8. Validation thresholds (ASHRAE Guideline 14), FFT, reproducibility -----
MAX_ERR_E_PCT = 10.0
MAX_ERR_P_PCT = 10.0
MAX_ERR_LF = 0.05
MAX_NRMSE = 0.15
MAX_LDC_ERR = 0.15
MAX_FFT_ERR = 0.15
FFT_N_KEEP = 10       # number of low-order harmonics kept for the spectral error, enough for the daily shape without high-frequency measurement noise
FFT_REMOVE_DC = True  # removal of the DC (slot-0 mean) component to restrict the spectral error to the shape only, not the already-scored daily energy
FFT_LOG_SCALE = True  # log-scale comparison of the magnitudes to keep small harmonics in play, not only the dominant one
RANDOM_SEED = 42
THERMAL_P_VAR_MAX = 0.10  # upper cap on thermal_p_var (RAMP relative power variability, fraction of nominal): at most +/-10 % swing, plausible for a central Benin cold-chain compressor and the ceiling for still-physical power.
