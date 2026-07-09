# -*- coding: utf-8 -*-
"""step1_preprocess.py - Cold chain step 1: preprocessing of one client.

Curated CSV (data/<type>/<client>/<client>.csv, columns datetime_utc, power_W
and optional voltage) -> clean 15-min series -> daily matrix (96 slots) ->
9 features per day -> entropy filter (rejects flat profiles).
Outputs: timeseries_clean.csv, daily_matrix_W.csv, features_by_day.csv,
entropy_by_day.csv and figure 02_heatmap.png.
The client code comes from the PUE_CLIENT environment variable (config.py).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as C


def load_client_csv(client_code: str) -> pd.DataFrame:
    """Read the curated CSV of the client, sorted and without duplicates."""
    df = pd.read_csv(C.CLIENTS[client_code]["csv"])
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True,
                                        errors="coerce")
    df = df.dropna(subset=["datetime_utc", "power_W"])
    df = df.drop_duplicates(subset="datetime_utc", keep="first")
    return df.sort_values("datetime_utc").reset_index(drop=True)


def clean_and_resample(df: pd.DataFrame, period_start: str,
                       period_end: str) -> pd.DataFrame:
    """Period filter + V>200V filter, convert to UTC+1, resample to 15 min."""
    # Conversion UTC -> Benin local time (UTC+1), then naive local timestamps
    df["dt"] = (df["datetime_utc"]
                + pd.Timedelta(hours=C.LOCAL_TZ_OFFSET_H)).dt.tz_localize(None)

    # Restriction to the study period (period_end kept until end of day)
    start = pd.to_datetime(period_start).tz_localize(None)
    end = pd.to_datetime(period_end).tz_localize(None) + pd.Timedelta(days=1)
    df = df[(df["dt"] >= start) & (df["dt"] < end)].copy()
    if df.empty:
        raise RuntimeError(
            f"No data in the period {period_start} -> {period_end}.")

    # Mini-grid outage filter: power forced to NaN when the voltage collapses.
    # Only applied when the curated CSV carries a voltage column.
    if "voltage" in df.columns:
        voltage = pd.to_numeric(df["voltage"], errors="coerce")
        df.loc[voltage < C.VOLTAGE_MIN_VALID, "power_W"] = np.nan

    # Regular 15-min grid + linear interpolation (max gap INTERP_MAX_GAP_H)
    s = df.set_index("dt")["power_W"].astype(float).resample(C.TIME_FREQ).mean()
    max_gap_slots = int(C.INTERP_MAX_GAP_H * 60 / C.DT_MIN)
    s = s.interpolate(method="linear", limit=max_gap_slots,
                      limit_direction="both")
    return s.to_frame("power_W")


def daily_matrix(ts: pd.DataFrame) -> pd.DataFrame:
    """Reshape the 15-min series into a (days x 96 slots) matrix.

    Days with fewer than MIN_POINTS_PER_DAY valid slots are rejected; the
    residual gaps are interpolated and any remaining NaN forced to 0.
    """
    df = ts.copy()
    df["date"] = df.index.date
    df["slot"] = (df.index.hour * 60 + df.index.minute) // int(C.DT_MIN)
    pivot = df.pivot_table(index="date", columns="slot", values="power_W",
                           aggfunc="mean")
    pivot = pivot.reindex(columns=range(C.N_SLOTS))
    pivot = pivot[pivot.notna().sum(axis=1) >= C.MIN_POINTS_PER_DAY].copy()
    pivot = pivot.interpolate(axis=1, limit_direction="both").fillna(0.0)
    pivot.index = pd.to_datetime(pivot.index)
    pivot.columns = [int(c) for c in pivot.columns]
    return pivot


def compute_daily_features(daily_W: pd.DataFrame) -> pd.DataFrame:
    """Compute the 9 cold chain features for each day."""
    dt_h = C.DT_MIN / 60.0
    # Power matrix with one row per day and 96 quarter-hour slots, negatives clipped.
    P = np.clip(daily_W.values.astype(float), 0.0, None)

    E_kWh = (P.sum(axis=1) * dt_h) / 1000.0               # total daily energy
    P_peak = P.max(axis=1)                                # compressor peak
    P_mean = P.mean(axis=1)
    LF = np.where(P_peak > 0, P_mean / P_peak, 0.0)       # load factor

    # Peak time in minutes since midnight (middle of the slot) + circular encoding
    peak_min = P.argmax(axis=1) * C.DT_MIN + C.DT_MIN / 2.0
    angle = 2 * np.pi * (peak_min / 1440.0)

    # Day over night energy ratio (day = 6h-18h)
    day_slots = slice(int(C.DAY_START_H * 60 / C.DT_MIN),
                      int(C.DAY_END_H * 60 / C.DT_MIN))
    E_day = P[:, day_slots].sum(axis=1) * dt_h
    E_night = (P.sum(axis=1) * dt_h) - E_day
    # Fallback ratio at 10.0 for zero-night-energy days: a conventional high cap standing in for the otherwise-undefined day/night division, moderate to avoid inflating the feature scale.
    ratio_day_night = np.where(E_night > 0, E_day / E_night, 10.0)

    # Normalised Shannon entropy of the energy-normalised profile (in [0, 1])
    Esum = P.sum(axis=1)
    p_norm = np.where(Esum[:, None] > 0, P / Esum[:, None], 0.0)
    eps = 1e-12
    terms = np.where(p_norm > eps, p_norm * np.log(p_norm + eps), 0.0)
    entropy = -terms.sum(axis=1) / np.log(daily_W.shape[1])

    return pd.DataFrame({"E_kWh": E_kWh, "P_peak_W": P_peak,
                         "P_mean_W": P_mean, "Load_factor": LF,
                         "peak_hour_min": peak_min,
                         "peak_sin": np.sin(angle), "peak_cos": np.cos(angle),
                         "ratio_day_night": ratio_day_night,
                         "entropy": entropy}, index=daily_W.index)


def plot_heatmap(client_code: str, daily_W: pd.DataFrame) -> None:
    """02_heatmap.png - power heatmap, one row per day, hour on the x axis."""
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(daily_W.values.astype(float), aspect="auto", cmap="viridis",
                   extent=[0, 24, len(daily_W), 0])
    ax.set(xlabel="Hour of day [h]", ylabel="Day index",
           title=f"{client_code} - Daily power heatmap")
    fig.colorbar(im, ax=ax, label="Power [W]")
    fig.tight_layout()
    fig.savefig(C.client_fig_dir(client_code) / "02_heatmap.png",
                bbox_inches="tight")
    plt.close(fig)


def preprocess_client(client_code: str) -> dict:
    """Full step 1 for one client; returns a small summary dict."""
    print(f"\n=== STEP1 PREPROCESS : {client_code} ===")
    cinfo = C.CLIENTS[client_code]
    rdir = C.client_results_dir(client_code)

    df_raw = load_client_csv(client_code)
    print(f"  {len(df_raw)} raw rows read")

    ts = clean_and_resample(df_raw, cinfo["period_start"], cinfo["period_end"])
    ts.to_csv(rdir / C.CLEAN_TS_CSV)
    print(f"  15-min time series: {len(ts)} slots")

    daily_W = daily_matrix(ts)
    print(f"  Raw daily matrix: {len(daily_W)} days")

    feats = compute_daily_features(daily_W)

    # Entropy filter to drop the degenerate days with all energy concentrated in a single slot (dead meter or spurious spike), i.e. days with normalised Shannon entropy near zero; threshold at C.ENTROPY_THRESHOLD, low enough to spare genuine low-activity days.
    keep = feats["entropy"] >= C.ENTROPY_THRESHOLD
    entropy_log = feats[["entropy"]].copy()
    entropy_log["kept"] = keep.values
    daily_W_kept, feats_kept = daily_W.loc[keep], feats.loc[keep]
    print(f"  Entropy filter: {len(feats)} -> {int(keep.sum())} days "
          f"(threshold = {C.ENTROPY_THRESHOLD})")

    daily_W_kept.to_csv(rdir / C.DAILY_MATRIX_CSV)
    feats_kept.to_csv(rdir / C.FEATURES_CSV)
    entropy_log.to_csv(rdir / C.ENTROPY_CSV)
    plot_heatmap(client_code, daily_W_kept)

    return {"client": client_code, "n_days_raw": int(len(daily_W)),
            "n_days_kept": int(len(daily_W_kept))}


if __name__ == "__main__":
    preprocess_client(C.PUE_CLIENT)
