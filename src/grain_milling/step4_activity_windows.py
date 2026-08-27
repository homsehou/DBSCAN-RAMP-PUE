# -*- coding: utf-8 -*-
"""Step 4 - Activity windows of the mill and their day-to-day drift.

Detection, per season, of up to three recurring activity windows on the
mean profile, then measurement on every single day of the drift of the
session start and end around them. The measured dispersion widens each
window, so a session starting at 10:00 one day and 14:00 another stays
inside it. The windows feed the native RAMP appliance of step 5.

Output: analytical_params.csv    one row per season (windows + dispersions)
        figures/30_windows_<season>.png

Self-contained script. Run it alone with:
    PUE_TYPE=grain_milling PUE_CLIENT=0016GBO python step4_activity_windows.py
"""
import json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C   # shared settings of the repository

PUE_TYPE = os.environ.get("PUE_TYPE", "grain_milling")
CLIENT = os.environ.get("PUE_CLIENT", "0016GBO")
REPO = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO / "resultats" / PUE_TYPE / CLIENT



def find_windows(mean_profile):
    """Active regions of the seasonal mean profile -> up to 3 windows.

    Returns a list of (first_slot, last_slot) pairs, both included.
    """
    threshold = max(C.ACTIVE_FRACTION * mean_profile.max(), C.ACTIVE_MIN_W)
    active = mean_profile >= threshold
    windows, start = [], None
    for j in range(C.SLOTS_PER_DAY):
        if active[j] and start is None:
            start = j
        if start is not None and (not active[j] or j == C.SLOTS_PER_DAY - 1):
            end = j if active[j] else j - 1
            windows.append([start, end])
            start = None
    # Merge windows separated by a short gap (one milling session paused)
    merged = []
    for w in windows:
        if merged and w[0] - merged[-1][1] - 1 <= C.WINDOW_GAP_SLOTS:
            merged[-1][1] = w[1]
        else:
            merged.append(w)
    merged = [w for w in merged if w[1] - w[0] + 1 >= C.MIN_LEN_SLOTS]
    # Split a wide region at a pronounced valley between two peaks: two
    # milling sessions (morning / afternoon) separated by a real lull
    split = []
    for w in merged:
        seg = mean_profile[w[0]:w[1] + 1]
        peaks, _ = find_peaks(seg, prominence=C.PEAK_PROMINENCE * seg.max(),
                              distance=C.PEAK_DISTANCE_SLOTS)
        cut = None
        if len(peaks) >= 2:
            valley = w[0] + peaks[0] + int(np.argmin(
                seg[peaks[0]:peaks[-1]]))
            if mean_profile[valley] < C.VALLEY_SPLIT * min(seg[peaks[0]],
                                                 seg[peaks[-1]]):
                cut = valley
        if cut and cut - w[0] >= C.MIN_LEN_SLOTS and w[1] - cut >= C.MIN_LEN_SLOTS:
            split += [[w[0], cut - 1], [cut + 1, w[1]]]
        else:
            split.append(w)
    # Keep the three windows carrying the most energy
    split.sort(key=lambda w: -mean_profile[w[0]:w[1] + 1].sum())
    return sorted(split[:C.MAX_WINDOWS])


def daily_dispersion(days, window):
    """Observed day-to-day drift of the session inside one window.

    For each day, the session start is the first active slot in the
    window (extended by 2 h on both sides) and the end the last one.
    Returns starts, ends (minutes), and the share of days with activity.
    """
    lo = max(0, window[0] - C.DISPERSION_SLOTS)
    hi = min(C.SLOTS_PER_DAY - 1, window[1] + C.DISPERSION_SLOTS)
    starts, ends = [], []
    for day in days:
        threshold = max(C.ACTIVE_FRACTION * day.max(), C.ACTIVE_MIN_W)
        active = np.nonzero(day[lo:hi + 1] >= threshold)[0]
        if len(active):
            starts.append((lo + active[0]) * float(C.SLOT_MIN))
            ends.append((lo + active[-1] + 1) * float(C.SLOT_MIN))
    support = len(starts) / max(len(days), 1)
    return np.array(starts), np.array(ends), support


def window_figure(mean_profile, windows, season):
    """Mean profile with the detected windows shaded."""
    hours = np.arange(C.SLOTS_PER_DAY) * 0.25 + 0.25 / 2
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(hours, mean_profile, color="#0072B2", alpha=0.4)
    for k, w in enumerate(windows):
        ax.axvspan(w[0] * 0.25, (w[1] + 1) * 0.25, color="#E69F00",
                   alpha=0.25, label="window" if k == 0 else None)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Power [W]")
    ax.set_title(f"{CLIENT} - {season} - activity windows")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figures"
                / f"30_windows_{season.replace(' ', '_')}.png", dpi=110)
    plt.close(fig)


def main():
    (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(OUT_DIR / "daily_matrix_W.csv", index_col=0)
    feats = pd.read_csv(OUT_DIR / "clustered_features.csv", index_col=0)
    targets = pd.read_csv(OUT_DIR / "seasonal_representative_profiles.csv")
    rows = []
    for _, srow in targets.iterrows():
        season, dom = srow["season"], int(srow["cluster_dominant"])
        mean_profile = srow[[f"slot_{j}" for j in range(C.SLOTS_PER_DAY)]].to_numpy(float)
        pick = (feats["season"] == season).values
        if dom != -1:
            pick &= (feats["cluster"] == dom).values
        days = daily.to_numpy(float)[pick]
        windows = find_windows(mean_profile)
        window_figure(mean_profile, windows, season)
        # Per window: bounds, energy share, day-to-day start/end dispersion
        bounds, weights, s_stds, e_stds, supports = [], [], [], [], []
        total_energy = max(mean_profile.sum(), 1e-9)
        for w in windows:
            starts, ends, support = daily_dispersion(days, w)
            bounds.append([int(w[0]) * C.SLOT_MIN,
                       (int(w[1]) + 1) * C.SLOT_MIN])
            weights.append(float(mean_profile[w[0]:w[1] + 1].sum()
                                 / total_energy))
            s_stds.append(float(np.std(starts)) if len(starts) else 0.0)
            e_stds.append(float(np.std(ends)) if len(ends) else 0.0)
            supports.append(float(support))
        # Share of days with any milling at all (for the summary tables)
        peaks = days.max(axis=1)
        active_days = float(np.mean(peaks >= np.maximum(
            C.ACTIVE_FRACTION * peaks.max(), C.ACTIVE_MIN_W)))
        rows.append({
            "season": season, "n_windows": len(windows),
            "windows_json": json.dumps(bounds),
            "window_weights_json": json.dumps(weights),
            "window_start_stds_json": json.dumps(s_stds),
            "window_end_stds_json": json.dumps(e_stds),
            "window_supports_json": json.dumps(supports),
            "occasional_use": active_days})
        spans = ", ".join(f"{b[0] // 60:02d}:{b[0] % 60:02d}-"
                          f"{b[1] // 60:02d}:{b[1] % 60:02d}"
                          for b in bounds)
        print(f"[step4] {season}: {len(windows)} windows [{spans}]  "
              f"start std {np.mean(s_stds):.0f} min")
    pd.DataFrame(rows).to_csv(OUT_DIR / "analytical_params.csv", index=False)


if __name__ == "__main__":
    main()
