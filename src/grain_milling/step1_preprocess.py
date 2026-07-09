#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 1: from the raw meter CSV to a clean 15-minute series, the daily matrix
in watts, and the daily feature table screened by the entropy filter."""

import calendar
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    CLEAN_TS_CSV, DAILY_MATRIX_CSV, DAY_END_H, DAY_START_H, DT_MIN, ENTROPY_CSV,
    ENTROPY_THRESHOLD, FEATURES_CSV, FIG_DIR, INTERP_MAX_GAP_H, N_SLOTS, RAW_CSV,
    RAW_DATETIME_COL, RAW_POWER_COL, SEASON_ORDER, STUDY_PERIOD_END,
    STUDY_PERIOD_START, TIME_FREQ, month_to_season,
)

SEASON_COLORS = {"Dry season": "#E74C3C", "Transition": "#F39C12",
                 "Rainy season": "#3498DB"}
SEASON_LABELS_EN = {"Dry season": "Dry season", "Transition": "Transition season",
                    "Rainy season": "Rainy season"}


def season_label_en(season_name):
    """English label of a season, used on every exported figure."""
    return SEASON_LABELS_EN.get(season_name, season_name)


def load_clean_series():
    """Read the raw CSV and return the clean 15-minute power series (UTC index)."""
    df = pd.read_csv(RAW_CSV)
    # Keep a numeric core from raw values carrying decimal commas or stray characters.
    power = (df[RAW_POWER_COL].astype(str)
             .str.replace(",", ".", regex=False)
             .str.replace(r"[^\d\.\-eE+]", "", regex=True))
    df[RAW_POWER_COL] = pd.to_numeric(power, errors="coerce")
    # Chronological UTC index (meter native timezone); clip of negative readings to zero.
    df[RAW_DATETIME_COL] = pd.to_datetime(df[RAW_DATETIME_COL], errors="coerce", utc=True)
    df = df.dropna(subset=[RAW_DATETIME_COL, RAW_POWER_COL]).sort_values(RAW_DATETIME_COL)
    series = df.set_index(RAW_DATETIME_COL)[RAW_POWER_COL].clip(lower=0.0)
    # 15-minute mean grid; interpolation restricted to gaps of one hour at most.
    series = series.resample(TIME_FREQ).mean()
    series = series.interpolate(limit=int(INTERP_MAX_GAP_H * 60 / DT_MIN),
                                limit_direction="both")
    # Study window: every 15-minute slot from the first day to the end of the last day.
    start = pd.Timestamp(STUDY_PERIOD_START, tz="UTC")
    end = pd.Timestamp(STUDY_PERIOD_END, tz="UTC") + pd.Timedelta(days=1)
    series = series[(series.index >= start) & (series.index < end)]
    if series.empty:
        raise ValueError("No sample remains after applying the study-window filter.")
    return series


def day_profile(day_start, group):
    """96-slot profile of one day: missing slots interpolated, leftovers set to 0."""
    grid = pd.date_range(day_start, periods=N_SLOTS, freq=TIME_FREQ)
    return group.reindex(grid).interpolate(limit_direction="both").fillna(0.0).to_numpy()


def build_daily_tables(series):
    """Daily matrix in watts plus the entropy-screened daily feature table."""
    days, rows, feats = [], [], []
    for day_start, group in series.groupby(series.index.floor("D")):
        profile = day_profile(day_start, group)
        days.append(day_start)
        rows.append(profile)
        # Peak-normalized shape scored by the Shannon entropy of its 20-bin value
        # histogram; near-constant days score low and are considered uninformative.
        peak = float(np.max(profile))
        shape = profile / (peak if peak > 0 else 1.0)
        counts, _ = np.histogram(shape, bins=20)
        proba = counts[counts > 0] / counts.sum()
        entropy = float(-np.sum(proba * np.log(proba)))
        # Daytime shape features; slots 24-71 correspond to 06:00-18:00.
        p_peak = float(np.max(shape))
        p_mean = float(np.mean(shape))
        day_energy = float(np.sum(shape[24:72]))
        night_slots = np.concatenate([shape[:24], shape[72:N_SLOTS]])
        night_energy = float(np.sum(night_slots))
        feats.append({"date": day_start, "p_peak": p_peak, "p_mean": p_mean,
                      "lf": p_mean / p_peak if p_peak > 0 else 0.0,
                      "hour_of_peak_min": int(np.argmax(shape)) * int(DT_MIN),
                      "ratio_day_night": day_energy / (night_energy + 1.0),
                      "entropy": entropy})

    daily = pd.DataFrame(rows, index=pd.Index(days, name="date"),
                         columns=np.arange(N_SLOTS))
    pd.DataFrame({"date": days, "entropy": [f["entropy"] for f in feats]}).to_csv(
        ENTROPY_CSV, index=False)

    # Entropy screen; threshold halving down to a 0.005 floor on empty selection.
    threshold = ENTROPY_THRESHOLD
    for _ in range(4):
        features = pd.DataFrame([f for f in feats if f["entropy"] >= threshold])
        if not features.empty or threshold <= 0.005:
            break
        threshold = max(threshold / 2.0, 0.005)
        print(f"[ENTROPY] No day retained. Threshold relaxed to {threshold:.4f}.")
    if features.empty:
        raise ValueError("The entropy filter retained no day, even after relaxation.")
    print(f"[ENTROPY] Retained days = {len(features)} / {len(feats)} "
          f"(threshold = {threshold}).")
    return daily, features


def compute_daily_features(daily):
    """Descriptive per-day features in physical units, shared with step 2."""
    time_h = np.arange(N_SLOTS) * DT_MIN / 60.0
    day_mask = (time_h >= DAY_START_H) & (time_h < DAY_END_H)
    dt_h = DT_MIN / 60.0
    rows = []
    for day, row in daily.iterrows():
        vals = row.to_numpy(dtype=float)
        p_peak = float(np.max(vals))
        p_mean = float(np.mean(vals))
        e_day = float(np.sum(vals[day_mask]) * dt_h / 1000.0)
        e_night = float(np.sum(vals[~day_mask]) * dt_h / 1000.0)
        e_total = e_day + e_night
        rows.append({
            "date": day,
            "P_peak_W": p_peak,
            "P_mean_W": p_mean,
            "Load_factor": p_mean / p_peak if p_peak > 0 else 0.0,
            "peak_hour": float(time_h[int(np.argmax(vals))]),
            "E_kWh": float(np.sum(vals) * dt_h / 1000.0),
            "E_day_kWh": e_day,
            "E_night_kWh": e_night,
            "ratio_day": e_day / e_total if e_total > 0 else 0.0,
            "ratio_night": e_night / e_total if e_total > 0 else 0.0,
            "ramp_rate": float(np.mean(np.abs(np.diff(vals)))),
            "std_power": float(np.std(vals)),
            "completeness": 1.0,
            "n_valid": N_SLOTS,
        })
    feat = pd.DataFrame(rows).set_index("date").sort_index()
    # Cyclic (sin, cos) encoding of the peak hour, for a circular feature keeping 23:00 and 00:00 close in the DBSCAN feature space instead of far apart.
    feat["peak_sin"] = np.sin(2 * np.pi * feat["peak_hour"] / 24.0)
    feat["peak_cos"] = np.cos(2 * np.pi * feat["peak_hour"] / 24.0)
    feat["month"] = feat.index.month
    feat["season"] = feat["month"].apply(month_to_season)
    return feat


def figure_heatmap(daily, path):
    """Day-by-hour heatmap of the retained daily profiles."""
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(daily.values, aspect="auto", origin="lower", cmap="inferno",
                      interpolation="nearest", extent=[0, 24, 0, len(daily)])
    ax.set(xlabel="Hour of day [h]", ylabel="Day index",
           title="Heatmap of the retained daily profiles")
    fig.colorbar(image, ax=ax, label="Power [W]")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_season_profiles(daily, feat, path, with_days):
    """One panel per season: mean profile, months covered, day count and energy."""
    seasons = [s for s in SEASON_ORDER if s in feat["season"].values]
    if not seasons:
        return
    t = np.arange(N_SLOTS) * DT_MIN / 60.0
    fig, axes = plt.subplots(len(seasons), 1, figsize=(10, 4 * len(seasons)),
                             sharex=True, squeeze=False)
    for ax, season in zip(axes[:, 0], seasons):
        sub = daily.loc[daily.index.isin(feat.index[feat["season"] == season])]
        mean_profile = sub.values.mean(axis=0)
        energy = mean_profile.sum() * DT_MIN / 60.0 / 1000.0
        months = sorted(feat.loc[feat["season"] == season, "month"].unique())
        months_label = ", ".join(calendar.month_abbr[int(m)] for m in months)
        color = SEASON_COLORS[season]
        if with_days:
            for i, (_, row) in enumerate(sub.iterrows()):
                ax.plot(t, row.values, lw=0.3, color=color, alpha=0.15,
                        label="Individual days" if i == 0 else None)
        ax.plot(t, mean_profile, lw=2.5, color=color,
                label=(f"{season_label_en(season)} ({months_label}) - "
                       f"mean of {len(sub)} days, {energy:.2f} kWh/day"))
        ax.set_ylabel("Power [W]")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Hour [h]")
    fig.suptitle("Average daily profile by season", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    print("=" * 72)
    print("  STEP 1 - Preprocessing")
    print("=" * 72)

    series = load_clean_series()
    series.to_frame("P_W").to_csv(CLEAN_TS_CSV, index_label="Date")
    print(f"Samples in the study window: {len(series)} "
          f"({series.index.min()} -> {series.index.max()})")

    daily, features = build_daily_tables(series)
    daily.to_csv(DAILY_MATRIX_CSV)
    features.to_csv(FEATURES_CSV, index=False)

    retained = daily.loc[daily.index.isin(features["date"])]
    feat = compute_daily_features(retained)
    figure_heatmap(retained, os.path.join(FIG_DIR, "03_heatmap.png"))
    figure_season_profiles(retained, feat,
                           os.path.join(FIG_DIR, "07_season_profiles.png"), False)
    figure_season_profiles(retained, feat,
                           os.path.join(FIG_DIR, "07b_season_profiles_stacked.png"), True)

    print("Step 1 complete.")
    print(f"  -> clean time series: {CLEAN_TS_CSV}")
    print(f"  -> daily matrix:      {DAILY_MATRIX_CSV}")
    print(f"  -> retained features: {FEATURES_CSV}")
    print(f"  -> entropy log:       {ENTROPY_CSV}")


if __name__ == "__main__":
    main()
