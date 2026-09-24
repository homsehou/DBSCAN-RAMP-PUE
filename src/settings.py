"""Settings shared by every step of the chain: clients, nameplates, dates, thresholds, metrics.

Season names, month names and result file names keep their original French spelling
("Saison seche", "Mai", "cibles_saison.csv"): they are data labels written in the published
results, and renaming them would break the link between this code and those results.
"""
import json
import os
import re
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

# Folder layout of the repository
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "resultats"
TARGETS = RESULTS / "cibles_v9"

# Time grid: one slot lasts 15 minutes, 96 slots per day
SLOT_MIN = 15
SLOTS_PER_DAY = 96

# The nine clients of the study and the family of their appliance
CLIENTS = {"0017SAM": "cold_chain", "0018SAM": "cold_chain",
           "0035SAM": "cold_chain", "0151GBO": "cold_chain",
           "0016GBO": "grain_milling", "0043SAM": "grain_milling",
           "0097SAM": "grain_milling", "0154GBO": "grain_milling",
           "0152GBO": "poultry_incubation"}

# Official commissioning date of the two mini-grids
COMMISSIONING = {"SAM": "2024-02-01", "GBO": "2024-10-01"}

# Nameplate power entered in RAMP (power = p_i1), never the power measured at the meter.
# Sources: February 2025 survey, One Power O&M daily reports, February 2026 survey photos.
NAMEPLATE_W = {"0017SAM": 276.0, "0018SAM": 276.0, "0035SAM": 276.0,
               "0151GBO": 220.0, "0152GBO": 200.0,
               "0016GBO": 7500.0, "0043SAM": 7500.0,
               "0097SAM": 7350.0, "0154GBO": 7500.0}

# Two-appliance fleets since mid-September 2025: nameplate of each appliance, the original one first
TWO_APPLIANCE_NAMEPLATES = {"0017SAM": (276.0, 276.0), "0018SAM": (276.0, 220.0),
                            "0151GBO": (220.0, 220.0)}

# Status of each nameplate against the sources and the meter, reported with the results
NAMEPLATE_STATUS = {
    "0017SAM": "sourced",
    "0018SAM": "conflicting sources (276 or 220 W)",
    "0035SAM": "measured appliance unknown (64 W), 276 W nameplate never observed at the meter",
    "0151GBO": "no dated source",
    "0152GBO": "assumption, no source",
    "0016GBO": "sourced", "0043SAM": "sourced", "0097SAM": "sourced", "0154GBO": "sourced"}
TWO_APPLIANCE_STATUS = ("assumption partly sourced: the meter shows a single running level "
                        "(326 to 370 W)")

# Client-specific start of the study, when the appliance arrived after commissioning
CLIENT_START = {"0017SAM": "2024-03-19", "0018SAM": "2024-02-01", "0151GBO": "2025-03-13"}
# Date of the fleet change, as seen at the meter (the O&M register gives a few days earlier)
FLEET_CHANGE = {"0017SAM": "2025-09-25", "0018SAM": "2025-10-02", "0151GBO": "2025-09-17"}
# Period during which the incubator 0152GBO was in operation
INCUBATOR_ACTIVE = ("2024-12-30", "2025-06-13")

# Running day: daily 15-minute peak at least equal to this threshold
RUNNING_THRESHOLD_W = {"cold_chain": 60.0, "grain_milling": 1000.0, "poultry_incubation": 60.0}
# 0035SAM runs at about 56 W: 30 W means at least seven minutes of running in the slot,
# a value read on the histogram of its daily peaks
CLIENT_RUNNING_THRESHOLD_W = {"0035SAM": 30.0}
OUT_OF_SERVICE_DAYS = 21     # at least 21 consecutive days without running: out of service
MIN_TARGET_DAYS = 10         # fewer retained days than this: target not calibrated
MIN_DAYS_FLAG = 10           # fewer clean days in a month than this: month flagged, kept

# Cleaning of the meter series
VOLTAGE_OUT = 200.0          # below this voltage the grid is off, the power is unknown
GAP_MAX_MIN = 30             # gaps up to 30 minutes are filled by linear interpolation
MIN_VALID_SLOTS = 90         # at least 90 valid slots out of 96 for a day to be kept
PERIOD_MAX_S = 900           # integration periods longer than 15 minutes are discarded

# The four Benin seasons, with May and October kept apart as transition months
SEASONS = {11: "Saison seche", 12: "Saison seche", 1: "Saison seche", 2: "Saison seche",
           3: "Saison seche", 4: "Saison seche", 5: "Mai",
           6: "Saison des pluies", 7: "Saison des pluies", 8: "Saison des pluies",
           9: "Saison des pluies", 10: "Octobre"}
MONTHS = ["Janvier", "Fevrier", "Mars", "Avril", "Mai", "Juin", "Juillet", "Aout",
          "Septembre", "Octobre", "Novembre", "Decembre"]

# DBSCAN: minimum number of neighbours of a core day
DBSCAN_MIN_SAMPLES = 5

# First simulated day of every RAMP run
SIM_START_DATE = "2024-01-01"

# Nominal thresholds of the validation contract
THRESH_10 = {"NRMSE": 0.10, "LDC_err": 0.10, "FFT_err": 0.10,
             "err_E_pct": 10.0, "err_P_pct": 10.0, "err_LF": 0.10}
SHAPE_THRESHOLDS = {"CORR_h": 0.05, "TVD_h": 0.10, "EXT_prof": 0.25}


def season_of_month(month):
    """Season of a month written as 'YYYY-MM', for instance '2025-03' gives 'Saison seche'."""
    return SEASONS[int(str(month)[5:7])]


def metrics(real, sim):
    """Six criteria between two 96-slot profiles, with the same definitions as the earlier chains."""
    real, sim = np.asarray(real, float), np.asarray(sim, float)
    span = max(real.max() - real.min(), 1e-9)
    r_s, s_s = np.sort(real)[::-1], np.sort(sim)[::-1]
    R = np.log(np.abs(np.fft.rfft(real))[1:11] + 1.0)
    S = np.log(np.abs(np.fft.rfft(sim))[1:11] + 1.0)
    return {"NRMSE": float(np.sqrt(np.mean((real - sim) ** 2)) / span),
            "LDC_err": float(np.sqrt(np.mean((r_s - s_s) ** 2)) / span),
            "FFT_err": float(np.sqrt(np.mean((R - S) ** 2)) / max(R.max() - R.min(), 1e-9)),
            "err_E_pct": float(100 * (sim.sum() - real.sum()) / real.sum()),
            "err_P_pct": float(100 * (sim.max() - real.max()) / real.max()),
            "err_LF": float(sim.mean() / sim.max() - real.mean() / real.max())}


def smooth(p, n):
    """Centred circular moving average over n slots."""
    p = np.asarray(p, float)
    ext = np.concatenate([p[-n:], p, p[:n]])
    return np.convolve(ext, np.ones(n) / n, mode="same")[n:-n]


def shape(real, sim):
    """Three shape criteria between two 96-slot profiles.

    CORR_h   : one minus the correlation of the 24 hourly means (phase and transitions)
    TVD_h    : share of the daily energy placed at the wrong hour
    EXT_prof : worst gap of the model at the peaks and troughs of the one-hour smoothed target
               (prominence of at least 20 % of the amplitude), relative to that amplitude
    """
    r, s = np.asarray(real, float), np.asarray(sim, float)
    rh, sh = r.reshape(24, 4).mean(axis=1), s.reshape(24, 4).mean(axis=1)
    a, b = rh - rh.mean(), sh - sh.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    corr = float((a * b).sum() / den) if den > 0 else 0.0
    tvd = 0.5 * np.abs(rh / max(rh.sum(), 1e-9) - sh / max(sh.sum(), 1e-9)).sum()
    gr, gs = smooth(r, 4), smooth(s, 4)
    amp = max(float(np.ptp(gr)), 1e-9)
    ext = np.concatenate([gr[-8:], gr, gr[:8]])
    idx = []
    for sign in (1, -1):
        k, _ = find_peaks(sign * ext, prominence=0.2 * amp)
        idx += sorted({int(i) - 8 for i in k if 8 <= i < 104})
    prof = float(np.abs(gs[idx] - gr[idx]).max() / amp) if idx else 0.0
    return {"CORR_h": 1.0 - corr, "TVD_h": float(tvd), "EXT_prof": prof}


def write_json(path, data):
    """Safe writing: a temporary file first, then a rename, so that a stop never truncates it."""
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def file_name(client, target):
    """File-safe name of a target, for instance '0151GBO__Saison_seche_avant_2025-09-17'."""
    return client + "__" + re.sub(r"[^A-Za-z0-9-]+", "_", str(target)).strip("_")
