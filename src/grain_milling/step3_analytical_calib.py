#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 3: analytical calibration of the milling machine, one parameter set per season.

The daily profiles of the dominant regular cluster are decomposed into operating
events (peak detection plus valley cuts). Recurring events are grouped into at most
three activity windows -- the native RAMP limit -- through successive merge/split
passes guided by the day-to-day evidence and by the seasonal mean profile. Each
window carries a two-level duty cycle (high grinding power p1 / low idle power p2,
1-D k-means on the active samples). Global RAMP parameters and data-driven
optimization bounds (center +/- K_SIGMA x day-to-day std) complete the export
consumed by step 4.

Inputs : seasonal_representative_profiles.csv, daily_matrix_W.csv,
         clustered_features.csv
Output : analytical_params.csv (one row per season)
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from config import (
    ANALYTICAL_PARAMS_CSV,
    CLUSTERED_CSV,
    DAILY_MATRIX_CSV,
    DT_MIN,
    K_SIGMA,
    N_SLOTS,
    SEASONAL_PROFILES_CSV,
)

DT_H = DT_MIN / 60.0


# ===================================================================
# 1. DATA LOADING
# ===================================================================


def load_seasonal_profiles():
    """Representative mean profile (96 slots, W) and dominant cluster per season."""
    if not os.path.exists(SEASONAL_PROFILES_CSV):
        sys.exit(f"[ERROR] {SEASONAL_PROFILES_CSV} not found. Run step 2 first.")
    df = pd.read_csv(SEASONAL_PROFILES_CSV)
    slot_cols = [c for c in df.columns if c.startswith("slot_")]
    profiles = {}
    for _, row in df.iterrows():
        profiles[row["season"]] = {
            "dominant_cluster": int(row["dominant_cluster"]),
            "n_days": int(row["n_days"]),
            "mean_profile_W": row[slot_cols].values.astype(float),
        }
    return profiles


def load_daily_matrix():
    """Daily power matrix (one row per day, N_SLOTS columns, W)."""
    dm = pd.read_csv(DAILY_MATRIX_CSV, index_col=0, parse_dates=True)
    dm.columns = dm.columns.astype(int)
    return dm.reindex(columns=np.arange(N_SLOTS))


def load_clustered_features():
    """Per-day cluster assignments and season labels produced by step 2."""
    return pd.read_csv(CLUSTERED_CSV, index_col=0, parse_dates=True)


# ===================================================================
# 2. SIGNAL HELPERS AND DUTY-CYCLE EXTRACTION
# ===================================================================


def _smooth_signal(x, kernel_size=3):
    """Apply a short moving average so event detection reacts to structure, not slot noise."""
    arr = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    kernel_size = max(1, min(int(kernel_size), len(arr)))
    if kernel_size <= 1:
        return arr
    kernel = np.ones(kernel_size, dtype=float) / kernel_size
    return np.convolve(arr, kernel, mode="same")


def _contiguous_true_regions(mask):
    """Convert a Boolean mask into a list of contiguous active regions."""
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0:
        return []

    regions = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            regions.append((start, i))
            start = None
    if start is not None:
        regions.append((start, len(mask)))
    return regions


def _state_run_lengths(mask, target_value):
    """Measure consecutive run lengths for one Boolean state inside a mask."""
    runs = []
    count = 0
    for val in np.asarray(mask, dtype=bool):
        if bool(val) == bool(target_value):
            count += 1
        elif count > 0:
            runs.append(count)
            count = 0
    if count > 0:
        runs.append(count)
    return runs


def _mask_from_regions(regions, size):
    """Rebuild a Boolean mask from a list of inclusive-exclusive regions."""
    mask = np.zeros(size, dtype=bool)
    for start, end in regions:
        mask[start:end] = True
    return mask


def _fill_small_gaps(mask, max_gap=1):
    """Close very short inactive gaps so one event is not split by a single noisy slot."""
    mask = np.asarray(mask, dtype=bool)
    false_regions = _contiguous_true_regions(~mask)
    filled = mask.copy()
    for start, end in false_regions:
        if start > 0 and end < len(mask) and (end - start) <= max_gap:
            filled[start:end] = True
    return filled


def _remove_short_regions(mask, min_len=2):
    """Remove isolated active fragments that are too short to represent a real event."""
    regions = [rg for rg in _contiguous_true_regions(mask) if (rg[1] - rg[0]) >= min_len]
    return _mask_from_regions(regions, len(mask))


def _classify_segment_states(segment):
    """Split one event segment into low-state and high-state operating levels."""
    seg = _smooth_signal(np.asarray(segment, dtype=float), kernel_size=3)
    if len(seg) == 0 or np.nanmax(seg) <= 0:
        return {"smoothed": seg, "active_mask": np.zeros(len(seg), dtype=bool),
                "low_mask": np.zeros(len(seg), dtype=bool), "high_mask": np.zeros(len(seg), dtype=bool),
                "low_level": 0.0, "high_level": 0.0, "threshold": 0.0}

    seg_max = float(np.nanmax(seg))
    active_thr = max(seg_max * 0.10, 30.0)
    active_mask = seg >= active_thr
    active_mask = _fill_small_gaps(active_mask, max_gap=1)
    active_mask = _remove_short_regions(active_mask, min_len=1)
    active_vals = seg[active_mask]
    if len(active_vals) == 0:
        return {"smoothed": seg, "active_mask": active_mask,
                "low_mask": np.zeros(len(seg), dtype=bool), "high_mask": np.zeros(len(seg), dtype=bool),
                "low_level": 0.0, "high_level": seg_max, "threshold": seg_max}

    # Two-level k-means seeds from the 30th/80th percentiles of the active samples: a robust low/high
    # starting split, widened to the 20th/90th percentiles on a collision of the two seeds.                                                                                                                     
    
    low_center = float(np.percentile(active_vals, 30))
    high_center = float(np.percentile(active_vals, 80))
    if high_center <= low_center:
        low_center = float(np.percentile(active_vals, 20))
        high_center = float(np.percentile(active_vals, 90))

    centers = np.array([low_center, high_center], dtype=float)
    # 1-D k-means on the active samples, capped at 20 iterations: enough for the two centers to
    # settle, with an early exit on convergence below the 1e-3 tolerance.
    for _ in range(20):
        dist = np.abs(active_vals[:, None] - centers[None, :])
        labels = np.argmin(dist, axis=1)
        if len(np.unique(labels)) < 2:
            break
        new_centers = np.array(
            [np.mean(active_vals[labels == 0]), np.mean(active_vals[labels == 1])], dtype=float
        )
        if np.allclose(new_centers, centers, atol=1e-3):
            centers = new_centers
            break
        centers = new_centers

    centers = np.sort(centers)
    low_center = float(centers[0])
    high_center = float(max(centers[1], low_center + 1.0))
    contrast = high_center - low_center
    if contrast < max(60.0, seg_max * 0.10):
        low_center = float(np.percentile(active_vals, 25))
        high_center = float(np.percentile(active_vals, 75))
        contrast = max(high_center - low_center, 1.0)

    # Mid-contrast threshold at 0.50: the equidistant cut between the two duty levels, splitting the
    # active samples into low and high states.
    threshold = low_center + 0.50 * contrast
    high_mask = active_mask & (seg >= threshold)
    low_mask = active_mask & ~high_mask

    high_mask = _fill_small_gaps(high_mask, max_gap=1)
    high_mask = _remove_short_regions(high_mask, min_len=1)
    low_mask = active_mask & ~high_mask

    low_vals = seg[low_mask]
    high_vals = seg[high_mask]
    # Representative levels: median for the low state, 60th percentile for the high state to favor the
    # sustained grinding power over transient peaks.
    low_level = float(np.median(low_vals)) if len(low_vals) > 0 else low_center
    high_level = float(np.percentile(high_vals, 60)) if len(high_vals) > 0 else high_center
    if high_level <= low_level:
        high_level = float(max(seg_max, low_level + 1.0))

    return {"smoothed": seg, "active_mask": active_mask, "low_mask": low_mask,
            "high_mask": high_mask, "low_level": low_level, "high_level": high_level,
            "threshold": threshold}


def extract_cycle_for_window(profile_15min, w_start_min, w_end_min):
    """
    Extract one duty cycle (p1, t1, p2, t2) for a given window.

    - p1 (ON power) = P90 of the active values inside the window
    - t1 (ON duration) = number of slots above 50% of the peak x DT_MIN
    - p2 (OFF power) = P10 of the active values (standby)
    - t2 (OFF duration) = fraction of the low-state duration
    """
    s_start = int(w_start_min / DT_MIN)
    s_end = min(N_SLOTS, int(w_end_min / DT_MIN))
    window_profile = profile_15min[s_start:s_end]

    # Neutral fallback duty cycle for an empty or flat window: default 1000 W / 5 min ON and
    # 100 W / 5 min OFF.
    # r_c (cycle-time random coefficient): relative jitter applied by RAMP to the ON durations,
    # here a mild 0.10 for a near-fixed cycle length.
    if len(window_profile) == 0 or np.nanmax(window_profile) <= 0:
        return {"p1": 1000, "t1": 5, "p2": 100, "t2": 5, "r_c": 0.10}

    p_peak_w = float(np.nanmax(window_profile))
    active_mask = window_profile > (p_peak_w * 0.1)
    active_vals = window_profile[active_mask]

    if len(active_vals) == 0:
        return {"p1": p_peak_w, "t1": 5, "p2": 100, "t2": 5, "r_c": 0.10}

    # p1: P90 of the active values
    p1 = float(np.percentile(active_vals, 90))

    # p2: P10 of the active values (standby)
    p2 = float(np.percentile(active_vals, 10))
    p2 = max(p2, 0)

    # t1: time spent above 50% of the peak
    high_mask = window_profile > (p_peak_w * 0.5)
    t1 = max(1, int(np.sum(high_mask) * DT_MIN))

    # t2: fraction of the low-state duration
    window_dur = w_end_min - w_start_min
    t2 = max(1, int((window_dur - t1) * 0.3))

    return {"p1": round(p1, 1), "t1": t1, "p2": round(p2, 1), "t2": t2, "r_c": 0.10}


def _cycle_from_segments(segments, fallback_profile):
    """Estimate one duty cycle from observed event segments or from a fallback proxy profile."""
    valid = [np.asarray(seg, dtype=float) for seg in segments if len(seg) > 0 and np.nanmax(seg) > 0]
    s_fallback = _smooth_signal(np.asarray(fallback_profile, dtype=float), kernel_size=3)
    if not valid:
        return extract_cycle_for_window(s_fallback, 0, len(s_fallback) * DT_MIN)

    low_vals = []
    high_vals = []
    high_runs = []
    low_runs = []
    seg_durations = []
    low_levels = []
    high_levels = []

    for seg in valid:
        states = _classify_segment_states(seg)
        seg_s = states["smoothed"]
        seg_durations.append(len(seg_s) * DT_MIN)
        if np.any(states["low_mask"]):
            low_vals.append(seg_s[states["low_mask"]])
            low_levels.append(states["low_level"])
        if np.any(states["high_mask"]):
            high_vals.append(seg_s[states["high_mask"]])
            high_levels.append(states["high_level"])
        high_runs.extend(_state_run_lengths(states["high_mask"], True))
        low_runs.extend(_state_run_lengths(states["low_mask"], True))

    if not low_vals and not high_vals:
        return extract_cycle_for_window(s_fallback, 0, len(s_fallback) * DT_MIN)

    fb_states = _classify_segment_states(s_fallback)
    fb_low = s_fallback[fb_states["low_mask"]]
    fb_high = s_fallback[fb_states["high_mask"]]

    # ON power p1: 60th percentile of the daily high-state samples, or the 85th percentile
    # of the low state as a proxy in the absence of any high state across the days.
    if high_vals:
        high_vals = np.concatenate(high_vals)
        p1_daily = float(np.percentile(high_vals, 60))
    else:
        p1_daily = float(np.percentile(np.concatenate(low_vals), 85))
             
    # Blend of the daily estimate (0.70) and the seasonal reference (0.30, 75th
    # percentile of its high state), clipped to a band around both anchors against
    # domination of p1 by either a noisy day or the reference alone.                                                                                                                                                  

         
    if len(fb_high) > 0:
        p1_ref = float(np.percentile(fb_high, 75))
        p1 = float(np.clip(0.70 * p1_daily + 0.30 * p1_ref,
                           max(p1_ref * 0.90, p1_daily * 0.85),
                           max(p1_ref * 1.20, p1_daily * 1.10, p1_ref + 50.0)))
    else:
        p1_ref = p1_daily
        p1 = p1_daily

    # OFF power p2: 45th percentile of the daily low-state samples (or 20th of the high state as a
    # proxy), blended 0.65 daily / 0.35 seasonal reference and capped below 55% of p1 to keep a real
    # ON/OFF contrast.
         
    if low_vals:
        low_vals = np.concatenate(low_vals)
        p2_daily = float(np.percentile(low_vals, 45))
    else:
        p2_daily = float(np.percentile(np.concatenate(high_vals), 20))
    if len(fb_low) > 0:
        p2_ref = float(np.percentile(fb_low, 50))
        p2 = float(np.clip(0.65 * p2_daily + 0.35 * p2_ref, 0.0, max(p1 * 0.55, p2_daily * 1.20)))
    else:
        p2 = p2_daily

    contrast = max(p1 - p2, 1.0)
    if contrast < max(80.0, p1 * 0.12):
        p2 = max(0.0, p1 - max(80.0, p1 * 0.14))

    if high_runs:
        t1 = max(15, int(np.median(high_runs) * DT_MIN))
    else:
        t1 = max(15, int(np.median(seg_durations) * 0.35))

    low_runs = [r for r in low_runs if r > 0]
    if low_runs:
        t2 = max(1, int(np.median(low_runs) * DT_MIN))
    else:
        t2 = max(1, int(np.median(seg_durations) * 0.15))

    if high_runs and np.mean(high_runs) > 0:
        r_c = float(np.clip(np.std(high_runs) / np.mean(high_runs), 0.05, 0.35))
    else:
        r_c = 0.10

    return {"p1": round(p1, 1), "t1": int(t1), "p2": round(p2, 1), "t2": int(t2), "r_c": round(r_c, 3)}


# ===================================================================
# 3. DAILY EVENTS AND STRUCTURED WINDOW RECONSTRUCTION
# ===================================================================


def extract_daily_events(day_profile, day_idx):
    """Detect the individual operating events present inside one daily profile."""
    raw = np.asarray(day_profile, dtype=float)
    smoothed = _smooth_signal(raw, kernel_size=3)
    day_max = float(np.nanmax(smoothed))
    if day_max <= 0:
        return []

    # Activity threshold at 16% of the daily peak, floored at 50 W to ignore standby
    # noise; the 0.62 factor as a looser mask threshold, preserving event edges below
    # the peak level.
         
    activity_thr = max(day_max * 0.16, 50.0)
    active_mask = smoothed >= activity_thr * 0.62
    active_mask = _fill_small_gaps(active_mask, max_gap=1)
    active_mask = _remove_short_regions(active_mask, min_len=2)
    spans = _contiguous_true_regions(active_mask)
    if not spans:
        return []

    events = []
    for span_start, span_end in spans:
        span = smoothed[span_start:span_end]
        if len(span) == 0 or np.nanmax(span) < activity_thr:
            continue

        span_range = float(np.nanmax(span) - np.nanmin(span))
        # Peak detection tuned to real grinding cycles: height above 68% of the span peak, prominence
        # above 16% of the span range (or 10% of its peak), minimum 2-slot spacing to avoid splitting
        # one cycle into spurious sub-peaks.
        peaks, _ = find_peaks(
            span,
            height=max(activity_thr, np.nanmax(span) * 0.68),
            prominence=max(span_range * 0.16, np.nanmax(span) * 0.10),
            distance=2,
        )
        if len(peaks) == 0:
            peaks = np.array([int(np.argmax(span))], dtype=int)

        cut_points = []
        for i in range(len(peaks) - 1):
            pk1 = peaks[i]
            pk2 = peaks[i + 1]
            # # Split of two peaks into separate events only with at least 4 slots apart and a
            # valley below 55% of the weaker peak and 70% of their mean: a shallow dip as a single
            # event.
            if (pk2 - pk1) < 4:
                continue
            valley = pk1 + int(np.argmin(span[pk1 : pk2 + 1]))
            valley_ratio = span[valley] / max(min(span[pk1], span[pk2]), 1.0)
            mean_ratio = span[valley] / max(np.mean([span[pk1], span[pk2]]), 1.0)
            if valley_ratio <= 0.55 and mean_ratio <= 0.70:
                cut_points.append(valley + 1)

        boundaries = [span_start] + [span_start + cp for cp in cut_points] + [span_end]
        for i in range(len(boundaries) - 1):
            seg_start = boundaries[i]
            seg_end = boundaries[i + 1]
            seg = raw[seg_start:seg_end]
            if len(seg) == 0 or np.nanmax(seg) < activity_thr:
                continue
            peak_slot = seg_start + int(np.argmax(seg))
            events.append({"day_idx": day_idx, "start_slot": int(seg_start),
                           "end_slot": int(seg_end), "peak_slot": int(peak_slot),
                           "start_min": int(seg_start * DT_MIN), "end_min": int(seg_end * DT_MIN),
                           "peak_min": int(peak_slot * DT_MIN + DT_MIN / 2),
                           "energy_Wh": float(np.nansum(seg) * DT_H)})

    return events


def _cluster_event_peaks(all_events):
    """Group nearby daily event peaks so recurring windows can be reconstructed seasonally."""
    if len(all_events) == 0:
        return np.array([], dtype=int)

    ordered = sorted(enumerate(all_events), key=lambda x: x[1]["peak_min"])
    labels = np.full(len(all_events), -1, dtype=int)
    clusters = []

    for idx, ev in ordered:
        peak = ev["peak_min"]
        best_cluster = None
        best_dist = None
        for cluster_id, cluster in enumerate(clusters):
            cluster_peaks = cluster["peaks"]
            cluster_center = float(np.median(cluster_peaks))
            new_min = min(cluster["min_peak"], peak)
            new_max = max(cluster["max_peak"], peak)
            dist = abs(peak - cluster_center)
            # Assign to a cluster only within 30 min of its center and with the whole spread under
            # 75 min: tolerances matching the day-to-day jitter of a recurring milling time without
            # merging two distinct daily sessions.
            if dist <= 30 and (new_max - new_min) <= 75:
                if best_dist is None or dist < best_dist:
                    best_cluster = cluster_id
                    best_dist = dist

        if best_cluster is None:
            best_cluster = len(clusters)
            clusters.append({"peaks": [peak], "min_peak": peak, "max_peak": peak})
        else:
            clusters[best_cluster]["peaks"].append(peak)
            clusters[best_cluster]["min_peak"] = min(clusters[best_cluster]["min_peak"], peak)
            clusters[best_cluster]["max_peak"] = max(clusters[best_cluster]["max_peak"], peak)

        labels[idx] = best_cluster

    return labels


def build_structured_windows(profile_15min, daily_profiles):
    """Rebuild recurring seasonal windows from the daily event catalogue and the mean profile."""
    # Plan of the main body (helper functions first, below this docstring):
    #   1. Extract the daily events and build the activity/power evidence arrays.
    #   2. Build the envelope and peak signals, then detect the active regions.
    #   3. Seed candidate clusters from event-peak groups, with a region-and-peak fallback.
    #   4. Sort the clusters, then merge the overlaps and absorb the weak clusters.
    #   5. Refine the clusters through shape-based splits, gap bridging and boundary cleanup.
    #   6. Cap the clusters to three windows, recover the edge activity and emit the window profiles.
    if daily_profiles is None or len(daily_profiles) == 0:
        return None

    # The helper functions below keep all split/merge decisions local to this
    # reconstruction routine because they depend on the evidence arrays built
    # from the same season only.
    def _best_valley_slot(signal, left_slot, right_slot):
        """Locate the lowest valley between two candidate peak neighborhoods."""
        left_slot = int(max(0, left_slot))
        right_slot = int(min(len(signal) - 1, right_slot))
        if right_slot <= left_slot:
            return left_slot
        return left_slot + int(np.nanargmin(signal[left_slot : right_slot + 1]))

    def _refresh_cluster(cluster):
        """Update support, dispersion and duty-cycle estimates after a cluster edit."""
        cluster["support"] = round(len(cluster["day_ids"]) / n_days, 4)
        cluster["start_std"] = (float(np.std(cluster["start_samples"]))
                                if len(cluster["start_samples"]) > 1 else 0.0)
        cluster["end_std"] = float(np.std(cluster["end_samples"])) if len(cluster["end_samples"]) > 1 else 0.0

        s_slot = max(0, int(cluster["start"] / DT_MIN))
        e_slot = min(N_SLOTS, int(cluster["end"] / DT_MIN))
        fallback = profile_15min[s_slot:e_slot]
        cluster["cycle"] = _cycle_from_segments(cluster["segments"], fallback)

        local_real = real_s[s_slot:e_slot]
        if len(local_real) > 0 and np.nanmax(local_real) > 0:
            cluster["peak"] = int((s_slot + int(np.nanargmax(local_real))) * DT_MIN + DT_MIN / 2)
        else:
            cluster["peak"] = (int(np.median(cluster["peak_samples"])) if cluster["peak_samples"]
                               else int((cluster["start"] + cluster["end"]) / 2))
        return cluster

    def _merge_clusters(clusters, idx):
        """Merge two adjacent window clusters when they actually describe one activity block."""
        left = clusters[idx]
        right = clusters[idx + 1]
        merged = {
            "start": min(left["start"], right["start"]), "end": max(left["end"], right["end"]),
            "energy_Wh": left["energy_Wh"] + right["energy_Wh"],
            "day_ids": set(left["day_ids"]) | set(right["day_ids"]),
            "start_samples": list(left["start_samples"]) + list(right["start_samples"]),
            "end_samples": list(left["end_samples"]) + list(right["end_samples"]),
            "peak_samples": list(left["peak_samples"]) + list(right["peak_samples"]),
            "segments": list(left["segments"]) + list(right["segments"]),
            "start_profile_acc": left["start_profile_acc"] + right["start_profile_acc"],
            "power_profile_acc": left["power_profile_acc"] + right["power_profile_acc"],
            "high_profile_acc": left["high_profile_acc"] + right["high_profile_acc"],
            "low_profile_acc": left["low_profile_acc"] + right["low_profile_acc"],
            "events": list(left["events"]) + list(right["events"]),
        }
        clusters[idx : idx + 2] = [_refresh_cluster(merged)]

    def _build_cluster_from_events(cluster_events, start_bound=None, end_bound=None):
        """Build one window candidate directly from the list of daily events assigned to it."""
        if not cluster_events:
            return None

        starts = [ev["start_min"] for ev in cluster_events]
        ends = [ev["end_min"] for ev in cluster_events]
        peaks = [ev["peak_min"] for ev in cluster_events]
        day_support = len({ev["day_idx"] for ev in cluster_events}) / n_days
        energy_share = sum(ev["energy_Wh"] for ev in cluster_events) / total_energy
        if day_support < max(2 / n_days, 0.08) or energy_share < 0.01:
            return None

        if start_bound is None:
            start_bound = int(np.floor(np.percentile(starts, 20) / DT_MIN) * DT_MIN)
        if end_bound is None:
            end_bound = int(np.ceil(np.percentile(ends, 80) / DT_MIN) * DT_MIN)

        pad_slots = int(np.clip(round(np.mean([np.std(starts), np.std(ends)]) / (4.0 * DT_MIN)), 0, 2))
        pad_min = pad_slots * DT_MIN
        start_min = max(
            int(start_bound), int(np.floor(np.percentile(starts, 20) / DT_MIN) * DT_MIN) - pad_min
        )
        end_min = min(int(end_bound), int(np.ceil(np.percentile(ends, 80) / DT_MIN) * DT_MIN) + pad_min)
        if end_min - start_min < 45:
            center = int(np.median(peaks))
            start_min = max(int(start_bound), center - 30)
            end_min = min(int(end_bound), center + 30)

        s_slot = max(0, int(start_min / DT_MIN))
        e_slot = min(N_SLOTS, int(end_min / DT_MIN))
        fallback = profile_15min[s_slot:e_slot]
        segments = []
        start_profile_acc = np.zeros(N_SLOTS, dtype=float)
        power_profile_acc = np.zeros(N_SLOTS, dtype=float)
        high_profile_acc = np.zeros(N_SLOTS, dtype=float)
        low_profile_acc = np.zeros(N_SLOTS, dtype=float)
        clipped_events = []
        for ev in cluster_events:
            seg_start = max(int(np.floor(start_min / DT_MIN)), int(ev["start_slot"]))
            seg_end = min(int(np.ceil(end_min / DT_MIN)), int(ev["end_slot"]))
            if seg_end <= seg_start:
                continue
            seg = np.asarray(daily_profiles[ev["day_idx"]][seg_start:seg_end], dtype=float)
            if len(seg) == 0 or np.nanmax(seg) <= 0:
                continue

            peak_slot = seg_start + int(np.nanargmax(seg))
            clipped_ev = dict(ev)
            clipped_ev["start_slot"] = int(seg_start)
            clipped_ev["end_slot"] = int(seg_end)
            clipped_ev["peak_slot"] = int(peak_slot)
            clipped_ev["start_min"] = int(seg_start * DT_MIN)
            clipped_ev["end_min"] = int(seg_end * DT_MIN)
            clipped_ev["peak_min"] = int(peak_slot * DT_MIN + DT_MIN / 2)
            clipped_ev["energy_Wh"] = float(np.nansum(seg) * DT_H)
            clipped_events.append(clipped_ev)

            segments.append(seg)
            start_profile_acc[seg_start] += 1.0
            power_profile_acc[seg_start:seg_end] += _smooth_signal(seg, kernel_size=3)
            states = _classify_segment_states(seg)
            low_profile_acc[seg_start:seg_end] += states["low_mask"].astype(float)
            high_profile_acc[seg_start:seg_end] += states["high_mask"].astype(float)

        if not clipped_events:
            return None

        starts = [ev["start_min"] for ev in clipped_events]
        ends = [ev["end_min"] for ev in clipped_events]
        peaks = [ev["peak_min"] for ev in clipped_events]
        day_ids = {ev["day_idx"] for ev in clipped_events}
        day_support = len(day_ids) / n_days
        energy_total = float(sum(ev["energy_Wh"] for ev in clipped_events))
        energy_share = energy_total / total_energy
        if day_support < max(2 / n_days, 0.08) or energy_share < 0.01:
            return None

        cycle = _cycle_from_segments(segments, fallback)

        local_real = real_s[s_slot:e_slot]
        if len(local_real) > 0 and np.nanmax(local_real) > 0:
            peak = int((s_slot + int(np.nanargmax(local_real))) * DT_MIN + DT_MIN / 2)
        else:
            peak = int(np.median(peaks))

        return {
            "start": start_min, "end": end_min, "peak": peak,
            "support": round(day_support, 4), "energy_Wh": energy_total,
            "start_std": float(np.std(starts)) if len(starts) > 1 else 0.0,
            "end_std": float(np.std(ends)) if len(ends) > 1 else 0.0,
            "cycle": cycle, "day_ids": day_ids,
            "start_samples": list(starts), "end_samples": list(ends),
            "peak_samples": list(peaks), "segments": segments,
            "start_profile_acc": start_profile_acc, "power_profile_acc": power_profile_acc,
            "high_profile_acc": high_profile_acc, "low_profile_acc": low_profile_acc,
            "events": clipped_events,
        }

    # --- 1. Extract daily events and build the activity evidence ---
    rows = np.asarray(daily_profiles, dtype=float)
    n_days = max(len(rows), 1)
    real_s = _smooth_signal(profile_15min, kernel_size=3)
    real_min = float(np.nanmin(real_s))
    real_range = float(np.nanmax(real_s) - real_min)
    real_norm = (real_s - real_min) / real_range if real_range > 1e-9 else np.zeros_like(real_s)

    all_events = []
    daily_active_minutes = []
    peak_marks = np.zeros((len(rows), N_SLOTS), dtype=float)
    active_marks = np.zeros((len(rows), N_SLOTS), dtype=float)
    power_norm = np.zeros((len(rows), N_SLOTS), dtype=float)

    for day_idx, row in enumerate(rows):
        row_s = _smooth_signal(row, kernel_size=3)
        row_max = float(np.nanmax(row_s))
        if row_max > 0:
            row_thr = max(row_max * 0.16, 50.0)
            active_mask = row_s >= row_thr * 0.62
            active_mask = _fill_small_gaps(active_mask, max_gap=1)
            active_mask = _remove_short_regions(active_mask, min_len=2)
            active_marks[day_idx] = active_mask.astype(float)
            power_norm[day_idx] = row_s / row_max

        events = extract_daily_events(row, day_idx)
        all_events.extend(events)
        daily_active_minutes.append(sum(ev["end_min"] - ev["start_min"] for ev in events))
        for ev in events:
            peak_marks[day_idx, ev["peak_slot"]] = 1.0

    if not all_events:
        return None

    # --- 2. Build the envelope and peak signals and detect the active regions ---
    activity_prob = _smooth_signal(active_marks.mean(axis=0), kernel_size=3)
    peak_prob = _smooth_signal(peak_marks.mean(axis=0), kernel_size=3)
    power_prob = _smooth_signal(power_norm.mean(axis=0), kernel_size=3)

    # Fused evidence envelope: # Fused evidence envelope: the activity probability leading (0.45) as the most direct
    # occupancy signal, backed by the normalized seasonal profile (0.35) and the mean
    # normalized power (0.20); the same mix on peak_signal to keep both consistent.
    envelope_signal = 0.45 * activity_prob + 0.35 * real_norm + 0.20 * power_prob
    peak_signal = 0.45 * peak_prob + 0.35 * real_norm + 0.20 * power_prob
    if float(np.nanmax(envelope_signal)) <= 0:
        return None

    # Active-region floor: 90% of the 40th percentile of the positive envelope, but never below a
    # hard 0.18, Active-region floor: 90% of the 40th percentile of the positive envelope, but never below a hard 0.18, for the survival of genuine low-activity windows and the exclusion of pure noise.
    envelope_floor = (
        max(0.18, float(np.nanpercentile(envelope_signal[envelope_signal > 0], 40)) * 0.90)
        if np.any(envelope_signal > 0)
        else 0.18
    )
    active_mask = envelope_signal >= envelope_floor
    active_mask = _fill_small_gaps(active_mask, max_gap=1)
    active_mask = _remove_short_regions(active_mask, min_len=2)
    active_regions = _contiguous_true_regions(active_mask)

    if not active_regions:
        fallback_mask = real_norm >= 0.15
        fallback_mask = _fill_small_gaps(fallback_mask, max_gap=1)
        fallback_mask = _remove_short_regions(fallback_mask, min_len=2)
        active_regions = _contiguous_true_regions(fallback_mask)
    if not active_regions:
        return None

    # --- 3. Seed clusters from event peaks, with a region-and-peak fallback ---
    total_energy = max(sum(ev["energy_Wh"] for ev in all_events), 1e-9)
    clusters = []
    global_peak = max(float(np.nanmax(real_s)), 1.0)
    global_range = max(float(np.nanmax(real_s) - np.nanmin(real_s)), 1.0)

    event_labels = _cluster_event_peaks(all_events)
    for label in sorted(set(event_labels.tolist())):
        if label < 0:
            continue
        cluster_events = [ev for ev, lab in zip(all_events, event_labels) if lab == label]
        built = _build_cluster_from_events(cluster_events)
        if built is not None:
            clusters.append(built)

    for region_start, region_end in ([] if clusters else active_regions):
        region_seg = real_s[region_start:region_end]
        if len(region_seg) == 0 or np.nanmax(region_seg) <= 0:
            continue

        region_max = float(np.nanmax(region_seg))
        region_range = max(float(np.nanmax(region_seg) - np.nanmin(region_seg)), 1.0)
        region_peaks, props = find_peaks(
            region_seg,
            height=max(60.0, region_max * 0.30, global_peak * 0.18),
            prominence=max(region_range * 0.10, global_range * 0.06, global_peak * 0.05),
            distance=2,
        )
        region_peaks = [region_start + int(pk) for pk in region_peaks]
        if len(region_peaks) == 0:
            region_peaks = [region_start + int(np.nanargmax(region_seg))]
            peak_scores = [1.0]
        else:
            prominences = props["prominences"]
            max_prom = max(float(np.nanmax(prominences)), 1.0)
            peak_scores = []
            for j, pk in enumerate(region_peaks):
                prom_norm = float(prominences[j]) / max_prom
                height_norm = float(real_s[pk]) / global_peak
                support_norm = float(activity_prob[pk]) / max(float(np.nanmax(activity_prob)), 1e-9)
                peak_scores.append(0.50 * prom_norm + 0.35 * height_norm + 0.15 * support_norm)

            kept_peaks = []
            kept_scores = []
            for pk, score, prom in zip(region_peaks, peak_scores, props["prominences"]):
                keep = score >= 0.22 or float(prom) >= max_prom * 0.30 or activity_prob[pk] >= 0.45
                if keep:
                    kept_peaks.append(pk)
                    kept_scores.append(score)
            region_peaks = kept_peaks or region_peaks
            peak_scores = kept_scores or peak_scores

        merged_peaks = [int(region_peaks[0])]
        merged_scores = [float(peak_scores[0])]
        for pk, score in zip(region_peaks[1:], peak_scores[1:]):
            prev_pk = merged_peaks[-1]
            prev_score = merged_scores[-1]
            valley_slot = _best_valley_slot(envelope_signal, prev_pk, pk)
            valley_level = float(envelope_signal[valley_slot])
            prev_level = float(max(envelope_signal[prev_pk], peak_signal[prev_pk], 1e-9))
            curr_level = float(max(envelope_signal[pk], peak_signal[pk], 1e-9))
            valley_ratio = valley_level / max(min(prev_level, curr_level), 1e-9)
            gap_min = (pk - prev_pk) * DT_MIN
            weak_peak = min(prev_score, score) < 0.28
            should_merge = (
                gap_min <= 30
                or (gap_min <= 60 and weak_peak)
                or (gap_min <= 120 and weak_peak and valley_ratio > 0.88)
            )
            if should_merge:
                if score > prev_score or real_s[pk] > real_s[prev_pk]:
                    merged_peaks[-1] = int(pk)
                    merged_scores[-1] = float(score)
            else:
                merged_peaks.append(int(pk))
                merged_scores.append(float(score))

        boundaries = [int(region_start)]
        for left_pk, right_pk in zip(merged_peaks[:-1], merged_peaks[1:]):
            valley_slot = _best_valley_slot(envelope_signal, left_pk, right_pk)
            boundary = max(boundaries[-1] + 1, min(right_pk, valley_slot + 1))
            boundaries.append(int(boundary))
        boundaries.append(int(region_end))

        for slot_start, slot_end in zip(boundaries[:-1], boundaries[1:]):
            start_bound = int(slot_start * DT_MIN)
            end_bound = int(slot_end * DT_MIN)
            cluster_events = [ev for ev in all_events if start_bound <= ev["peak_min"] < end_bound]
            if not cluster_events:
                cluster_events = [
                    ev
                    for ev in all_events
                    if max(start_bound, ev["start_min"]) < min(end_bound, ev["end_min"])
                ]
            if not cluster_events:
                continue

            built = _build_cluster_from_events(cluster_events, start_bound=start_bound, end_bound=end_bound)
            if built is not None:
                clusters.append(built)

    if not clusters:
        return None

    # --- 4. Sort the clusters, merge the overlaps and absorb the weak clusters ---
    clusters.sort(key=lambda x: (x["peak"], x["start"]))

    i = 0
    while i < len(clusters) - 1:
        curr = clusters[i]
        nxt = clusters[i + 1]
        overlap = min(curr["end"], nxt["end"]) - max(curr["start"], nxt["start"])
        min_dur = max(min(curr["end"] - curr["start"], nxt["end"] - nxt["start"]), 1)
        overlap_ratio = overlap / min_dur if overlap > 0 else 0.0
        peak_gap = abs(curr["peak"] - nxt["peak"])
        if overlap_ratio >= 0.60 and peak_gap <= 45:
            _merge_clusters(clusters, i)
            continue
        i += 1

    support_thr = max(3 / n_days, 0.14)
    energy_thr = 0.03

    while len(clusters) > 1:
        weak_idx = None
        for i, cl in enumerate(clusters):
            energy_share = cl["energy_Wh"] / total_energy
            duration = cl["end"] - cl["start"]
            if cl["support"] < support_thr or energy_share < energy_thr or duration < 45:
                weak_idx = i
                break
        if weak_idx is None:
            break

        if weak_idx == 0:
            merge_idx = 0
        elif weak_idx == len(clusters) - 1:
            merge_idx = weak_idx - 1
        else:
            left = clusters[weak_idx - 1]
            curr = clusters[weak_idx]
            right = clusters[weak_idx + 1]
            left_cost = max(0, curr["start"] - left["end"]) + 0.5 * abs(curr["peak"] - left["peak"])
            right_cost = max(0, right["start"] - curr["end"]) + 0.5 * abs(right["peak"] - curr["peak"])
            merge_idx = weak_idx - 1 if left_cost <= right_cost else weak_idx
        _merge_clusters(clusters, merge_idx)

    def _split_cluster_if_multimodal(cluster):
        """Split one wide cluster when its own average profile clearly contains two peaks."""
        events = list(cluster.get("events", []))
        if len(events) < max(4, int(np.ceil(0.12 * n_days))):
            return None

        s_slot = max(0, int(cluster["start"] / DT_MIN))
        e_slot = min(N_SLOTS, int(np.ceil(cluster["end"] / DT_MIN)))
        profile_mean = np.asarray(cluster["power_profile_acc"], dtype=float) / max(n_days, 1)
        local = profile_mean[s_slot:e_slot]
        if len(local) < 4 or np.nanmax(local) <= 0:
            return None

        local_s = _smooth_signal(local, kernel_size=3)
        local_range = float(np.nanmax(local_s) - np.nanmin(local_s))
        peaks, props = find_peaks(
            local_s,
            height=max(np.nanmax(local_s) * 0.30, 45.0),
            prominence=max(local_range * 0.10, np.nanmax(local_s) * 0.08),
            distance=2,
        )
        if len(peaks) < 2:
            return None

        split_choice = None
        split_score = None
        for pk1, pk2 in zip(peaks[:-1], peaks[1:]):
            if pk2 - pk1 < 3:
                continue
            valley = pk1 + int(np.nanargmin(local_s[pk1 : pk2 + 1]))
            left_level = float(max(local_s[pk1], 1e-9))
            right_level = float(max(local_s[pk2], 1e-9))
            valley_level = float(local_s[valley])
            valley_ratio = valley_level / max(min(left_level, right_level), 1e-9)
            if valley_ratio > 0.86:
                continue
            score = valley_ratio - 0.002 * (pk2 - pk1)
            if split_score is None or score < split_score:
                split_score = score
                split_choice = (pk1, pk2, valley)

        if split_choice is None:
            return None

        _, _, valley = split_choice
        boundary_min = int((s_slot + valley + 1) * DT_MIN)
        left_events = [ev for ev in events if ev["peak_min"] < boundary_min]
        right_events = [ev for ev in events if ev["peak_min"] >= boundary_min]
        if not left_events or not right_events:
            return None

        left_cluster = _build_cluster_from_events(
            left_events, start_bound=cluster["start"], end_bound=boundary_min
        )
        right_cluster = _build_cluster_from_events(
            right_events, start_bound=boundary_min, end_bound=cluster["end"]
        )
        if left_cluster is None or right_cluster is None:
            return None
        return [left_cluster, right_cluster]

    def _split_cluster_by_real_shape(cluster):
        """Split a cluster using the real seasonal profile when the event average is ambiguous."""
        events = list(cluster.get("events", []))
        if len(events) < 2:
            return None

        s_slot = max(0, int(cluster["start"] / DT_MIN))
        e_slot = min(N_SLOTS, int(np.ceil(cluster["end"] / DT_MIN)))
        local = real_s[s_slot:e_slot]
        if len(local) < 4 or np.nanmax(local) <= 0:
            return None

        local_s = _smooth_signal(local, kernel_size=3)
        local_max = float(np.nanmax(local_s))
        local_range = float(np.nanmax(local_s) - np.nanmin(local_s))
        if local_max <= 0 or local_range <= 0:
            return None

        peaks, props = find_peaks(
            local_s,
            height=max(local_max * 0.16, 25.0),
            prominence=max(local_range * 0.025, local_max * 0.015, 8.0),
            distance=2,
        )
        if len(peaks) < 2:
            return None

        strong_peaks = []
        for j, pk in enumerate(peaks):
            prom = float(props["prominences"][j]) if "prominences" in props else 0.0
            level = float(local_s[pk])
            if level >= local_max * 0.22 or prom >= max(local_range * 0.05, 20.0):
                strong_peaks.append(int(pk))
        peaks = np.asarray(strong_peaks, dtype=int)
        if len(peaks) < 2:
            return None

        boundaries = []
        for pk1, pk2 in zip(peaks[:-1], peaks[1:]):
            if pk2 - pk1 < 2:
                continue
            valley = pk1 + int(np.nanargmin(local_s[pk1 : pk2 + 1]))
            left_level = float(max(local_s[pk1], 1e-9))
            right_level = float(max(local_s[pk2], 1e-9))
            weak_level = min(left_level, right_level)
            valley_level = float(local_s[valley])
            valley_drop = weak_level - valley_level
            valley_ratio = valley_level / max(weak_level, 1e-9)
            # Late-peak flag past 17:30: # Late-peak flag past 17:30: an evening milling session typically close to the
            # afternoon one, hence relaxed split thresholds (0.12 level, 0.84 valley ratio below)
            # still enough to separate the two.
            late_peak = (s_slot + max(pk1, pk2)) >= int(17.5 * 60 / DT_MIN)
            should_split = valley_drop >= max(local_range * 0.05, 20.0) and (
                (weak_level >= local_max * 0.30 and valley_ratio <= 0.95)
                or (weak_level >= local_max * 0.20 and valley_ratio <= 0.88)
                or (late_peak and weak_level >= local_max * 0.12 and valley_ratio <= 0.84)
            )
            if should_split:
                boundaries.append(int((s_slot + valley + 1) * DT_MIN))

        boundaries = sorted(
            set(bd for bd in boundaries if cluster["start"] + 30 <= bd <= cluster["end"] - 30)
        )
        if not boundaries:
            return None

        interval_bounds = [cluster["start"]] + boundaries + [cluster["end"]]
        split_clusters = []
        for lo, hi in zip(interval_bounds[:-1], interval_bounds[1:]):
            side_events = []
            for ev in events:
                overlap_start = max(lo, int(ev["start_min"]))
                overlap_end = min(hi, int(ev["end_min"]))
                if overlap_end - overlap_start >= DT_MIN:
                    side_events.append(ev)
            built = _build_cluster_from_events(side_events, start_bound=lo, end_bound=hi)
            if built is None:
                return None
            split_clusters.append(built)

        if len(split_clusters) < 2:
            return None
        return split_clusters

    # --- 5. Refine the clusters: splits, gap bridging and boundary cleanup ---
    split_happened = True
    while split_happened:
        split_happened = False
        refined = []
        for cl in clusters:
            split_clusters = _split_cluster_if_multimodal(cl)
            if split_clusters is not None:
                refined.extend(split_clusters)
                split_happened = True
            else:
                refined.append(cl)
        clusters = refined

    split_happened = True
    while split_happened:
        split_happened = False
        refined = []
        for cl in clusters:
            split_clusters = _split_cluster_by_real_shape(cl)
            if split_clusters is not None:
                refined.extend(split_clusters)
                split_happened = True
            else:
                refined.append(cl)
        clusters = refined

    i = 0
    while i < len(clusters) - 1:
        curr = clusters[i]
        nxt = clusters[i + 1]
        if curr["end"] > nxt["start"]:
            peak_left = int(curr["peak"] / DT_MIN)
            peak_right = int(nxt["peak"] / DT_MIN)
            valley_slot = _best_valley_slot(real_s, peak_left, max(peak_left + 1, peak_right))
            boundary = int((valley_slot + 1) * DT_MIN)
            boundary = max(curr["start"] + 30, min(nxt["end"] - 30, boundary))
            if boundary <= curr["start"] or boundary >= nxt["end"]:
                _merge_clusters(clusters, i)
                continue
            curr["end"] = min(curr["end"], boundary)
            nxt["start"] = max(nxt["start"], boundary)
        i += 1

    i = 0
    while i < len(clusters) - 1:
        curr = clusters[i]
        nxt = clusters[i + 1]
        gap = nxt["start"] - curr["end"]
        if gap > 0:
            gap_start_slot = max(0, int(curr["end"] / DT_MIN) - 1)
            gap_end_slot = min(N_SLOTS - 1, int(nxt["start"] / DT_MIN))
            if gap_end_slot > gap_start_slot:
                gap_seg = real_s[gap_start_slot : gap_end_slot + 1]
                gap_level = float(np.nanmedian(gap_seg))
                gap_activity = float(np.nanmean(activity_prob[gap_start_slot : gap_end_slot + 1]))
                left_level = float(max(real_s[int(curr["peak"] / DT_MIN)], 1e-9))
                right_level = float(max(real_s[int(nxt["peak"] / DT_MIN)], 1e-9))
                bridge_ratio = gap_level / max(min(left_level, right_level), 1e-9)
                should_bridge = gap <= 240 and (bridge_ratio >= 0.22 or gap_activity >= 0.30)
                if should_bridge:
                    valley_slot = _best_valley_slot(real_s, gap_start_slot, gap_end_slot)
                    boundary = int((valley_slot + 1) * DT_MIN)
                    boundary = max(curr["start"] + 30, min(nxt["end"] - 30, boundary))
                    if boundary > curr["start"] and boundary < nxt["end"]:
                        curr["end"] = max(curr["end"], boundary)
                        nxt["start"] = min(nxt["start"], boundary)
        i += 1

    clusters = [_refresh_cluster(cl) for cl in clusters]

    i = 0
    while i < len(clusters) - 1:
        curr = clusters[i]
        nxt = clusters[i + 1]
        gap = max(0, nxt["start"] - curr["end"])
        peak_left = int(curr["peak"] / DT_MIN)
        peak_right = int(nxt["peak"] / DT_MIN)
        valley_slot = _best_valley_slot(real_s, peak_left, max(peak_left + 1, peak_right))
        left_level = float(max(real_s[peak_left], 1e-9))
        right_level = float(max(real_s[peak_right], 1e-9))
        valley_level = float(real_s[valley_slot])
        valley_ratio = valley_level / max(min(left_level, right_level), 1e-9)
        should_merge = gap <= 45 and valley_ratio >= 0.90
        if should_merge:
            _merge_clusters(clusters, i)
            continue
        i += 1

    clusters = [_refresh_cluster(cl) for cl in clusters]

    # --- 6. Cap to three windows, recover the edges and emit the profiles ---
    # The original RAMP model only supports three activity windows per appliance.
    # The analytical skeleton is therefore merged until it fits this native limit.
    max_windows = 3
    while len(clusters) > max_windows:
        merge_idx = None
        merge_cost = None
        for i in range(len(clusters) - 1):
            left = clusters[i]
            right = clusters[i + 1]
            gap = max(0, right["start"] - left["end"])
            peak_gap = abs(right["peak"] - left["peak"])
            valley_slot = _best_valley_slot(envelope_signal, int(left["peak"] / DT_MIN),
                                            int(right["peak"] / DT_MIN))
            valley_level = float(envelope_signal[valley_slot])
            level_ref = max(min(envelope_signal[int(left["peak"] / DT_MIN)],
                                envelope_signal[int(right["peak"] / DT_MIN)]), 1e-9)
            valley_ratio = valley_level / level_ref
            # Merge-cost trade-off across heterogeneous terms: raw gap in minutes, plus 0.35 x the
            # peak gap (down-weighted, distant peaks less decisive than the void between windows),
            # plus 120 x the valley ratio, a shallow dip between two windows (high ratio) marking them
            # the cheapest pair to merge first.
            cost = gap + 0.35 * peak_gap + 120.0 * valley_ratio
            if merge_cost is None or cost < merge_cost:
                merge_cost = cost
                merge_idx = i
        _merge_clusters(clusters, merge_idx)

    if clusters and len(clusters) < max_windows:

        def _event_is_covered(ev):
            for cl in clusters:
                if cl["start"] <= ev["peak_min"] < cl["end"]:
                    return True
                if max(cl["start"], ev["start_min"]) < min(cl["end"], ev["end_min"]):
                    return True
            return False

        uncovered = [ev for ev in all_events if not _event_is_covered(ev)]
        edge_specs = [
            ("early", [ev for ev in uncovered if ev["peak_min"] < clusters[0]["start"]]),
            ("late", [ev for ev in uncovered if ev["peak_min"] >= clusters[-1]["end"]]),
        ]

        for side, side_events in edge_specs:
            if not side_events or len(clusters) >= max_windows:
                continue

            labels = _cluster_event_peaks(side_events)
            candidate_clusters = []
            for label in sorted(set(labels.tolist())):
                if label < 0:
                    continue
                ev_group = [ev for ev, lab in zip(side_events, labels) if lab == label]
                built = _build_cluster_from_events(ev_group)
                if built is not None:
                    candidate_clusters.append(built)

            if not candidate_clusters:
                continue

            best = max(candidate_clusters, key=lambda cl: (cl["energy_Wh"], cl["support"]))
            if side == "early":
                if best["end"] <= clusters[0]["start"] + 30:
                    clusters.insert(0, _refresh_cluster(best))
            else:
                if best["start"] >= clusters[-1]["end"] - 30:
                    clusters.append(_refresh_cluster(best))

    weights = [cl["energy_Wh"] for cl in clusters]
    total_w = sum(weights)
    weights = [w / total_w for w in weights] if total_w > 0 else [1.0 / len(clusters)] * len(clusters)

    window_power_profiles = [np.clip(_smooth_signal(cl["power_profile_acc"] / max(n_days, 1),
                                                    kernel_size=3), 0.0, None) for cl in clusters]
    window_high_profiles = [np.clip(_smooth_signal(cl["high_profile_acc"] / max(n_days, 1),
                                                   kernel_size=3), 0.0, 1.0) for cl in clusters]
    window_low_profiles = [np.clip(_smooth_signal(cl["low_profile_acc"] / max(n_days, 1),
                                                  kernel_size=3), 0.0, 1.0) for cl in clusters]

    def _inject_boundary_support():
        """Preserve continuity at touching window boundaries when the real profile stays active."""
        if len(clusters) < 2:
            return

        peak_ref = float(np.nanmax(real_s)) if len(real_s) else 0.0
        boundary_thr = max(peak_ref * 0.08, 35.0)
        for i in range(len(clusters) - 1):
            left = clusters[i]
            right = clusters[i + 1]
            gap_slots = int(round((right["start"] - left["end"]) / DT_MIN))
            if gap_slots > 1:
                continue

            boundary_slot = int(round(left["end"] / DT_MIN))
            for slot in range(max(0, boundary_slot - 1), min(N_SLOTS, boundary_slot + 3)):
                real_val = float(real_s[slot])
                if real_val < boundary_thr:
                    continue

                current_total = float(window_power_profiles[i][slot] + window_power_profiles[i + 1][slot])
                target_total = max(real_val * 0.92, current_total)
                if target_total <= current_total + 1e-9:
                    continue

                deficit = target_total - current_total
                if slot < boundary_slot:
                    left_ratio = 0.78
                elif slot == boundary_slot:
                    left_ratio = 0.58
                else:
                    left_ratio = 0.30
                right_ratio = 1.0 - left_ratio

                window_power_profiles[i][slot] += deficit * left_ratio
                window_power_profiles[i + 1][slot] += deficit * right_ratio
                window_low_profiles[i][slot] = max(window_low_profiles[i][slot], left_ratio)
                window_low_profiles[i + 1][slot] = max(window_low_profiles[i + 1][slot], right_ratio)

    def _inject_late_tail_support():
        """Keep the small end-of-day tail when the real profile does not drop to zero."""
        if not clusters:
            return

        peak_ref = float(np.nanmax(real_s)) if len(real_s) else 0.0
        tail_thr = max(peak_ref * 0.015, 8.0)
        last_idx = len(clusters) - 1
        last_end_slot = int(np.ceil(clusters[-1]["end"] / DT_MIN))
        tail_stop = min(N_SLOTS, last_end_slot + 8)
        for slot in range(last_end_slot, tail_stop):
            real_val = float(real_s[slot])
            if real_val < tail_thr:
                continue
            target = real_val * (0.90 if slot <= last_end_slot + 2 else 0.82)
            if window_power_profiles[last_idx][slot] < target:
                window_power_profiles[last_idx][slot] = target
            window_low_profiles[last_idx][slot] = max(window_low_profiles[last_idx][slot], 0.65)

    _inject_boundary_support()
    _inject_late_tail_support()

    return {
        "windows": [(cl["start"], cl["end"]) for cl in clusters],
        "peak_centers": [cl["peak"] for cl in clusters],
        "cycles": [cl["cycle"] for cl in clusters],
        "window_weights": [round(w, 4) for w in weights],
        "window_supports": [cl["support"] for cl in clusters],
        "window_start_stds": [round(cl["start_std"], 1) for cl in clusters],
        "window_end_stds": [round(cl["end_std"], 1) for cl in clusters],
        "window_event_rates": [round(len(cl.get("events", [])) / max(n_days, 1), 4) for cl in clusters],
        "daily_active_minutes": daily_active_minutes,
        "window_start_profiles": [
            (_smooth_signal(cl["start_profile_acc"], kernel_size=3)
             / max(np.sum(_smooth_signal(cl["start_profile_acc"], kernel_size=3)), 1e-9)).tolist()
            for cl in clusters
        ],
        "window_power_profiles": [prof.tolist() for prof in window_power_profiles],
        "window_high_profiles": [prof.tolist() for prof in window_high_profiles],
        "window_low_profiles": [prof.tolist() for prof in window_low_profiles],
    }


# ===================================================================
# 4. GLOBAL PARAMETER EXTRACTION
# ===================================================================


def extract_analytical_params(profile_15min, windows, peak_centers, daily_profiles, structured):
    """Global RAMP parameters read off the mean profile, the daily profiles and
    the structured window reconstruction."""
    E_daily_Wh = float(np.nansum(profile_15min) * DT_H)
    print(f"   Daily energy: {E_daily_Wh:.1f} Wh ({E_daily_Wh / 1000:.2f} kWh)")

    total_window_min = sum(we - ws for ws, we in windows)
    p_peak = float(np.nanmax(profile_15min))

    # Nominal power: P70 of the daily peaks, kept close to the mean-profile peak.
    daily_peaks = np.nanmax(daily_profiles, axis=1)
    valid_peaks = daily_peaks[daily_peaks > 0]
    if len(valid_peaks) > 0:
        power_ref = float(np.percentile(valid_peaks, 70))
        power_est = float(np.clip(power_ref, p_peak * 0.95, p_peak * 1.35))
    else:
        power_est = p_peak * 1.10

    # func_time: mean daily active duration over the days showing activity.
    active_days = [v for v in structured["daily_active_minutes"] if v > 0]
    if active_days:
        func_time_est = float(np.mean(active_days))
    else:
        func_time_est = E_daily_Wh * 60.0 / power_est
    func_time_est = max(15, min(total_window_min * 0.98, func_time_est))

    # func_cycle: low quartile of the window ON durations, within 15-60 min.
    func_cycle_est = float(np.clip(np.percentile([c["t1"] for c in structured["cycles"]], 25), 15, 60))

    # random_var_w: median start/end dispersion relative to the window duration.
    var_terms = []
    for i, (ws, we) in enumerate(windows):
        dur = max(we - ws, 1)
        std_s = structured["window_start_stds"][i] if i < len(structured["window_start_stds"]) else 0
        std_e = structured["window_end_stds"][i] if i < len(structured["window_end_stds"]) else 0
        var_terms.append(np.mean([std_s, std_e]) / dur)
    random_var_w_est = float(np.clip(np.median(var_terms), 0.0, 0.08)) if var_terms else 0.04

    # occasional_use: fraction of the days showing any activity.
    n_active = sum(v > 0 for v in structured["daily_active_minutes"])
    occasional_use_est = float(np.clip(n_active / max(len(structured["daily_active_minutes"]), 1), 0.1, 1.0))

    window_weights = list(structured["window_weights"])
    cycles = list(structured["cycles"])

    # time_fraction_random_variability: relative day-to-day spread of active time.
    if len(active_days) > 1 and np.mean(active_days) > 0:
        tfv = float(np.clip(np.std(active_days) / np.mean(active_days), 0.0, 0.20))
    else:
        tfv = 0.05

    # event_concentration: entropy sharpness of the window start-time histograms.
    concentrations = []
    for prof in structured["window_start_profiles"]:
        arr = np.asarray(prof, dtype=float)
        if len(arr) == 0 or np.nansum(arr) <= 0:
            continue
        arr = arr / np.nansum(arr)
        entropy = -np.nansum(arr[arr > 0] * np.log(arr[arr > 0]))
        max_entropy = np.log(max(np.count_nonzero(arr), 2))
        sharp = 1.0 - entropy / max(max_entropy, 1e-9)
        concentrations.append(1.0 + 0.8 * sharp)
    event_concentration = float(np.clip(np.median(concentrations), 0.8, 1.8)) if concentrations else 1.0

    print(f"   Power (P70 of the daily peaks): {power_est:.1f} W")
    print(f"   func_time: {func_time_est:.1f} min")
    print(f"   {len(windows)} windows, {len(cycles)} cycles")
    for i, (ws, we) in enumerate(windows):
        pk_h = peak_centers[i] / 60.0 if i < len(peak_centers) else 0
        c = cycles[i]
        print(
            f"     W{i + 1}: {ws / 60:.1f}h-{we / 60:.1f}h ({we - ws}min, "
            f"w={window_weights[i]:.2f}, peak={pk_h:.1f}h)  "
            f"C{i + 1}: p1={c['p1']:.0f}W t1={c['t1']}min "
            f"p2={c['p2']:.0f}W t2={c['t2']}min"
        )

    return {
        "n_windows": len(windows),
        "power": round(power_est, 1),
        "func_time": round(func_time_est, 1),
        "func_cycle": round(func_cycle_est, 1),
        "random_var_w": round(random_var_w_est, 3),
        "time_fraction_random_variability": round(tfv, 3),
        "occasional_use": round(occasional_use_est, 2),
        "thermal_p_var": 0.0,
        "event_concentration": round(event_concentration, 3),
        "E_daily_Wh": round(E_daily_Wh, 1),
        "P_peak_mean_profile": round(p_peak, 1),
        "windows": windows,
        "window_weights": [round(w, 4) for w in window_weights],
        "window_supports": structured["window_supports"],
        "window_start_stds": structured["window_start_stds"],
        "window_end_stds": structured["window_end_stds"],
        "window_event_rates": structured["window_event_rates"],
        "window_start_profiles": structured["window_start_profiles"],
        "window_power_profiles": structured["window_power_profiles"],
        "window_high_profiles": structured["window_high_profiles"],
        "window_low_profiles": structured["window_low_profiles"],
        "peak_centers": peak_centers,
        "cycles": cycles,
    }


# ===================================================================
# 5. CONSOLE DIAGNOSTICS (proxy profile vs real profile)
# ===================================================================


def compute_theoretical_profile(params, n_slots=N_SLOTS):
    """Deterministic proxy: sum of the per-window mean power profiles, rescaled to
    the measured daily energy (console diagnostics only)."""
    profile = np.zeros(n_slots, dtype=float)
    target_energy = float(params.get("E_daily_Wh", 0.0))
    for prof in params["window_power_profiles"]:
        arr = np.asarray(prof, dtype=float)
        if len(arr) != n_slots:
            continue
        profile += arr
    current_energy = float(np.nansum(profile) * DT_H)
    if current_energy > 0 and target_energy > 0:
        profile *= target_energy / current_energy
    return profile


def compute_comparison_metrics(real_profile, estimated_profile):
    """Compute the compact analytical-versus-real metrics reported in step 3."""
    rmse = np.sqrt(np.nanmean((real_profile - estimated_profile) ** 2))
    p_range = np.nanmax(real_profile) - np.nanmin(real_profile)
    nrmse = rmse / p_range if p_range > 0 else float("inf")

    E_real = np.nansum(real_profile) * DT_H
    E_est = np.nansum(estimated_profile) * DT_H
    err_E_pct = (E_est - E_real) / E_real * 100 if E_real > 0 else 0

    P_peak_real = np.nanmax(real_profile)
    P_peak_est = np.nanmax(estimated_profile)
    err_P_pct = (P_peak_est - P_peak_real) / P_peak_real * 100 if P_peak_real > 0 else 0

    if np.std(real_profile) > 0 and np.std(estimated_profile) > 0:
        corr = np.corrcoef(real_profile, estimated_profile)[0, 1]
    else:
        corr = 0.0

    return {
        "NRMSE": round(nrmse, 4),
        "err_E_pct": round(err_E_pct, 1),
        "err_P_pct": round(err_P_pct, 1),
        "correlation": round(corr, 4),
    }


# ===================================================================
# 6. DATA-DRIVEN BOUNDS FOR THE STEP-4 OPTIMIZATION
# ===================================================================


def compute_param_bounds(daily_sub, params, k_sigma=K_SIGMA):
    """Data-driven optimization bounds, one (lo, hi) pair per step-4 parameter.

    For every parameter: center = step-3 analytical estimate, sigma = day-to-day
    standard deviation of the matching observable over the cluster days (daily peak
    for p1, minutes above the threshold for t1, and so on), and
    bound = center +/- k_sigma * sigma clipped to the physical RAMP limits.
    The _q10/_q90 suffixes are kept for compatibility with step 4 and
    simply mean lower/upper bound."""
    bounds = {}
    n_days, n_slots = daily_sub.shape
    slot_min = 1440.0 / n_slots
    windows = params["windows"]
    cycles = params["cycles"]
    n_win = len(windows)

    # ON threshold used to separate the high / low states of the step3 cycles
    if cycles:
        mids = [(c["p1"] + c["p2"]) / 2.0 for c in cycles]
        thr_global = float(np.mean(mids))
    else:
        thr_global = float(daily_sub.mean())

    # Helper: return (lo, hi) for center +/- k*sigma, clipped to the RAMP limits
    def sigma_bounds(center, sigma, ramp_lo, ramp_hi):
        sigma = max(float(sigma), 0.0)
        lo = max(ramp_lo, float(center) - k_sigma * sigma)
        hi = min(ramp_hi, float(center) + k_sigma * sigma)
        if hi <= lo:  # ensures hi > lo
            hi = lo + max(sigma, 1e-3)
        return lo, hi

    # ---- power: daily peak ----
    daily_peak = daily_sub.max(axis=1)
    p_lo, p_hi = sigma_bounds(daily_peak.mean(), daily_peak.std(), 0.0, float("inf"))
    bounds["power_q10"], bounds["power_q90"] = p_lo, p_hi

    # ---- func_time: ON minutes per day ----
    minutes_on = (daily_sub > thr_global).sum(axis=1) * slot_min
    ft_center = float(params.get("func_time", minutes_on.mean()))
    ft_lo, ft_hi = sigma_bounds(ft_center, minutes_on.std(), 0.0, 1440.0)
    bounds["func_time_q10"], bounds["func_time_q90"] = ft_lo, ft_hi

    # ---- func_cycle: mean duration of contiguous ON stretches per day ----
    cycle_durs = []
    for day_arr in daily_sub:
        on = (day_arr > thr_global).astype(int)
        flips = np.diff(np.concatenate(([0], on, [0])))
        starts = np.where(flips == 1)[0]
        ends = np.where(flips == -1)[0]
        if len(starts):
            cycle_durs.append(float(np.mean(ends - starts)) * slot_min)
    fc_center = float(params.get("func_cycle", np.mean(cycle_durs) if cycle_durs else slot_min))
    fc_sigma = float(np.std(cycle_durs)) if len(cycle_durs) > 1 else slot_min
    fc_lo, fc_hi = sigma_bounds(fc_center, fc_sigma, 1.0, 1440.0)
    bounds["func_cycle_q10"], bounds["func_cycle_q90"] = fc_lo, fc_hi

    # ---- random_var_w: relative deviation of the daily peak ----
    mean_peak = max(float(daily_peak.mean()), 1.0)
    rel_peak = np.abs(daily_peak - daily_peak.mean()) / mean_peak
    rv_center = float(params.get("random_var_w", rel_peak.mean()))
    rv_lo, rv_hi = sigma_bounds(rv_center, rel_peak.std(), 0.0, 1.0)
    bounds["random_var_w_q10"], bounds["random_var_w_q90"] = rv_lo, rv_hi

    # ---- occasional_use: fraction of active days (binomial std) ----
    active = (minutes_on > 0).astype(int)
    p_act = float(active.mean())
    sigma_p = float(np.sqrt(p_act * (1.0 - p_act) / max(n_days, 1)))
    occ_center = float(params.get("occasional_use", p_act))
    occ_lo, occ_hi = sigma_bounds(occ_center, sigma_p, 0.0, 1.0)
    bounds["occasional_use_q10"], bounds["occasional_use_q90"] = occ_lo, occ_hi

    # ---- time_frac_var: relative deviation of minutes_on ----
    mean_on = max(float(minutes_on.mean()), 1.0)
    rel_on = np.abs(minutes_on - minutes_on.mean()) / mean_on
    tfv_center = float(params.get("time_fraction_random_variability", rel_on.mean()))
    tfv_lo, tfv_hi = sigma_bounds(tfv_center, rel_on.std(), 0.0, 1.0)
    bounds["time_frac_var_q10"], bounds["time_frac_var_q90"] = tfv_lo, tfv_hi

    # ---- Bounds for each window i ----
    start_stds = params.get("window_start_stds", [0.0] * n_win)
    end_stds = params.get("window_end_stds", [0.0] * n_win)
    for i in range(n_win):
        ws_min, we_min = windows[i]
        s0 = max(0, int(ws_min / slot_min))
        s1 = min(n_slots, int(we_min / slot_min))
        win_data = daily_sub[:, s0:s1]
        cyc = cycles[i] if i < len(cycles) else {"p1": 0, "p2": 0, "t1": 0, "t2": 0, "r_c": 0}
        thr_win = (cyc["p1"] + cyc["p2"]) / 2.0

        # w_i_start / w_i_end: center = step3, sigma = day-to-day std
        ws_sigma = float(start_stds[i]) if i < len(start_stds) else slot_min
        we_sigma = float(end_stds[i]) if i < len(end_stds) else slot_min
        ws_lo, ws_hi = sigma_bounds(ws_min, ws_sigma, 0.0, 1440.0)
        we_lo, we_hi = sigma_bounds(we_min, we_sigma, 0.0, 1440.0)
        bounds[f"w{i}_start_q10"], bounds[f"w{i}_start_q90"] = ws_lo, ws_hi
        bounds[f"w{i}_end_q10"], bounds[f"w{i}_end_q90"] = we_lo, we_hi

        # p_i1: center = step3 cyc.p1; sigma = std of the daily peak
        daily_p1 = win_data.max(axis=1)
        p1_center = float(cyc.get("p1", daily_p1.mean()))
        p1_sigma = float(daily_p1.std()) if len(daily_p1) > 1 else max(p1_center * 0.1, 1.0)
        p1_lo, p1_hi = sigma_bounds(p1_center, p1_sigma, 0.0, float("inf"))
        bounds[f"c{i}_p1_q10"], bounds[f"c{i}_p1_q90"] = p1_lo, p1_hi

        # p_i2: center = step3 cyc.p2; sigma = std of the daily LOW-state means
        p2_per_day = []
        for day_arr in win_data:
            low = day_arr[day_arr < thr_win]
            if len(low):
                p2_per_day.append(float(low.mean()))
        p2_center = float(cyc.get("p2", np.mean(p2_per_day) if p2_per_day else 0.0))
        p2_sigma = float(np.std(p2_per_day)) if len(p2_per_day) > 1 else max(p2_center * 0.1, 1.0)
        p2_lo, p2_hi = sigma_bounds(p2_center, p2_sigma, 0.0, float("inf"))
        bounds[f"c{i}_p2_q10"], bounds[f"c{i}_p2_q90"] = p2_lo, p2_hi

        # t_i1: center = step3 cyc.t1; sigma = std of the daily ON durations
        t1_per_day = (win_data > thr_win).sum(axis=1) * slot_min
        t1_center = float(cyc.get("t1", t1_per_day.mean()))
        t1_sigma = float(t1_per_day.std()) if len(t1_per_day) > 1 else max(t1_center * 0.2, slot_min)
        t1_lo, t1_hi = sigma_bounds(t1_center, t1_sigma, 0.0, 1440.0)
        bounds[f"c{i}_t1_q10"], bounds[f"c{i}_t1_q90"] = t1_lo, t1_hi

        # t_i2: center = step3 cyc.t2; sigma = std of the daily LOW durations
        t2_per_day = (win_data <= thr_win).sum(axis=1) * slot_min
        t2_center = float(cyc.get("t2", t2_per_day.mean()))
        t2_sigma = float(t2_per_day.std()) if len(t2_per_day) > 1 else max(t2_center * 0.2, slot_min)
        t2_lo, t2_hi = sigma_bounds(t2_center, t2_sigma, 0.0, 1440.0)
        bounds[f"c{i}_t2_q10"], bounds[f"c{i}_t2_q90"] = t2_lo, t2_hi

        # r_ci: center = step3 cyc.r_c; sigma = std of the relative t1 dispersion
        rc_center = float(cyc.get("r_c", 0.0))
        if t1_per_day.mean() > 0 and len(t1_per_day) > 1:
            rel_t1 = np.abs(t1_per_day - t1_per_day.mean()) / t1_per_day.mean()
            rc_sigma = float(rel_t1.std())
        else:
            rc_sigma = 0.05
        rc_lo, rc_hi = sigma_bounds(rc_center, rc_sigma, 0.0, 1.0)
        bounds[f"c{i}_rc_q10"], bounds[f"c{i}_rc_q90"] = rc_lo, rc_hi

    return bounds


# ===================================================================
# 7. MAIN
# ===================================================================


def main():
    print("=" * 72)
    print("  STEP 3 - Analytical calibration")
    print("=" * 72)

    seasonal_profiles = load_seasonal_profiles()
    daily = load_daily_matrix()
    feat = load_clustered_features()
    print(f"{len(seasonal_profiles)} seasons, {len(daily)} days")

    all_params = {}
    for season in sorted(seasonal_profiles.keys()):
        sdata = seasonal_profiles[season]
        real_profile = sdata["mean_profile_W"]
        dominant_cluster = int(sdata["dominant_cluster"])
        print(f"\n-- {season} --")

        # Only the days of the dominant regular cluster: the windows must capture
        # the recurring behaviour, not the atypical days.
        selected_days = feat.index[(feat["season"] == season) & (feat["cluster"] == dominant_cluster)]
        if len(selected_days) == 0:
            raise ValueError(f"No day found for {season} " f"with dominant cluster {dominant_cluster}.")
        print(f"   {len(selected_days)} days used (dominant cluster {dominant_cluster})")
        daily_sub = daily.loc[daily.index.isin(selected_days)].values

        structured = build_structured_windows(real_profile, daily_sub)
        if structured is None or len(structured["windows"]) == 0:
            sys.exit(f"[ERROR] No recurring window could be reconstructed for {season}.")

        params = extract_analytical_params(
            real_profile, structured["windows"], structured["peak_centers"], daily_sub, structured
        )
        params["season"] = season
        params["bounds"] = compute_param_bounds(daily_sub, params)
        all_params[season] = params

        est_profile = compute_theoretical_profile(params)
        metrics = compute_comparison_metrics(real_profile, est_profile)
        print(
            f"   -> NRMSE={metrics['NRMSE']:.3f}, dE={metrics['err_E_pct']:+.1f}%, "
            f"dP={metrics['err_P_pct']:+.1f}%, r={metrics['correlation']:.3f}"
        )

    rows = []
    for season, params in sorted(all_params.items()):
        rows.append(
            {
                "season": season,
                "n_windows": params["n_windows"],
                "power": params["power"],
                "func_time": params["func_time"],
                "func_cycle": params["func_cycle"],
                "random_var_w": params["random_var_w"],
                "time_fraction_random_variability": params["time_fraction_random_variability"],
                "occasional_use": params["occasional_use"],
                "thermal_p_var": params["thermal_p_var"],
                "event_concentration": params["event_concentration"],
                "E_daily_Wh": params["E_daily_Wh"],
                "windows_json": json.dumps(params["windows"]),
                "window_weights_json": json.dumps(params["window_weights"]),
                "window_supports_json": json.dumps(params["window_supports"]),
                "window_start_stds_json": json.dumps(params["window_start_stds"]),
                "window_end_stds_json": json.dumps(params["window_end_stds"]),
                "window_event_rates_json": json.dumps(params["window_event_rates"]),
                "window_start_profiles_json": json.dumps(params["window_start_profiles"]),
                "window_power_profiles_json": json.dumps(params["window_power_profiles"]),
                "window_high_profiles_json": json.dumps(params["window_high_profiles"]),
                "window_low_profiles_json": json.dumps(params["window_low_profiles"]),
                "peak_centers_json": json.dumps(params["peak_centers"]),
                "cycles_json": json.dumps(params["cycles"]),
                "bounds_json": json.dumps(params["bounds"]),
            }
        )
    pd.DataFrame(rows).to_csv(ANALYTICAL_PARAMS_CSV, index=False)

    print("\nStep 3 complete.")
    print(f"  -> analytical parameters: {ANALYTICAL_PARAMS_CSV}")


if __name__ == "__main__":
    main()
