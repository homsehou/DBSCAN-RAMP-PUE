# -*- coding: utf-8 -*-
"""Step 1 - Preprocessing: raw meter CSV -> clean daily profiles + registers.
Acted rules: readings under 200 V are grid outages, dated and counted,
never zeros (mill meters inherit the outages of the voltage-equipped
meters of their mini-grid); gaps interpolated up to 30 min, a day with a
longer hole rejected and logged; an inactive day is excluded from the
shape matrix but counted for the use frequency (occasional_use), runs of
30+ inactive days being operation stops kept out of that frequency.
Run alone:  PUE_TYPE=grain_milling PUE_CLIENT=0016GBO python step1_preprocess.py
"""
import json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C   # shared settings of the repository
PUE_TYPE = os.environ.get("PUE_TYPE", "grain_milling")
CLIENT = os.environ.get("PUE_CLIENT", "0016GBO")
REPO = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO / "resultats" / PUE_TYPE / CLIENT

def neighbour_outages():
    """Outage slots agreed by every voltage-equipped meter of the grid."""
    votes = []
    for path in sorted((REPO / "data").glob(f"*/*/*{CLIENT[-3:]}.csv")):
        raw = pd.read_csv(path)
        if path.stem == CLIENT or "voltage" not in raw.columns:
            continue
        stamp = pd.to_datetime(raw["datetime_utc"], errors="coerce",
                               utc=True) + pd.Timedelta(hours=C.UTC_OFFSET_H)
        low = pd.Series((pd.to_numeric(raw["voltage"], errors="coerce")
                         < C.VOLTAGE_MIN).values, index=stamp)
        low = low[low.index.notna() & ~low.index.duplicated()]
        votes.append(low.resample("15min").max())
    if not votes:
        return None
    table = pd.concat(votes, axis=1)
    return table.index[table.notna().any(axis=1)
                       & table.fillna(False).astype(bool).all(axis=1)]

def load_series():
    """Raw CSV -> 15-minute power series, plus the dated outage episodes."""
    df = pd.read_csv(REPO / "data" / PUE_TYPE / CLIENT / f"{CLIENT}.csv")
    time_col = "datetime_utc" if "datetime_utc" in df.columns else "Date"
    power_col = ("power_W" if "power_W" in df.columns
                 else "Average_Active_power")
    df["dt"] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    # Logger artefact: a handful of rows are stamped at the 1970 epoch
    df = df[df["dt"] > pd.Timestamp("2015-01-01", tz="UTC")]
    df = df.dropna(subset=["dt"]).sort_values("dt")
    power = pd.to_numeric(df[power_col], errors="coerce").clip(lower=0)
    # Grid outage kept distinct from a deliberate switch-off
    if "voltage" in df.columns:
        volts = pd.to_numeric(df["voltage"], errors="coerce")
        power[volts.values < C.VOLTAGE_MIN] = np.nan
    series = pd.Series(power.values, index=df["dt"]
                       + pd.Timedelta(hours=C.UTC_OFFSET_H))
    series = series[~series.index.duplicated()].resample("15min").mean()
    if "voltage" not in df.columns:
        slots = neighbour_outages()   # mill meters have no voltage column
        if slots is not None:
            series[series.index.intersection(slots)] = np.nan
    # Optional study period restriction (bounds included)
    start = os.environ.get("PUE_PERIOD_START")
    end = os.environ.get("PUE_PERIOD_END")
    if start:
        series = series[pd.Timestamp(start, tz=series.index.tz):]
    if end:
        series = series[:pd.Timestamp(end, tz=series.index.tz)
                        + pd.Timedelta(days=1, minutes=-1)]
    # Outage register: every episode of consecutive invalid slots
    out = series.isna()
    grp = pd.Series(series.index[out],
                    index=(out != out.shift()).cumsum()[out].values
                    ).groupby(level=0)
    episodes = pd.DataFrame({"start": grp.min(),
                             "minutes": grp.size() * C.SLOT_MIN})
    episodes["end"] = episodes["start"] \
        + pd.to_timedelta(episodes["minutes"], unit="min")
    return series, episodes[["start", "end", "minutes"]]

def daily_split(series):
    """Interpolated day table, plus the register of the rejected days."""
    frame = series.to_frame("W")
    frame["day"] = frame.index.date
    frame["slot"] = frame.index.hour * 4 + frame.index.minute // C.SLOT_MIN
    table = frame.pivot_table(index="day", columns="slot", values="W",
                              dropna=False)
    table = table.reindex(columns=range(C.SLOTS_PER_DAY))
    valid = table.notna().sum(axis=1)
    # Gaps up to 30 min filled; a longer hole inside a day rejects the day
    table = table.interpolate(axis=1, limit=C.GAP_INTERP_SLOTS,
                              limit_direction="both")
    bad_few = valid < C.MIN_SLOTS_KEPT
    bad = bad_few | (table.isna().sum(axis=1) > 0)
    reasons = pd.DataFrame(
        {"reason": np.where(bad_few, "fewer than 90 valid slots",
                            f"gap of {C.GAP_REJECT_SLOTS}+ slots"),
         "missing_slots": C.SLOTS_PER_DAY - valid}, index=table.index)[bad]
    table = table[~bad]
    table.columns = [f"slot_{j}" for j in range(C.SLOTS_PER_DAY)]
    return table, reasons

def daily_features(table):
    """Descriptors of each day, inputs of the clustering in step 2."""
    P = table.to_numpy(float)
    peak = np.maximum(P.max(axis=1), 1e-9)
    # Peak hour on a circle: 23:45 stays next to 00:15
    angle = 2 * np.pi * (P.argmax(axis=1) + 0.5) / C.SLOTS_PER_DAY
    e_day = P[:, C.DAY_START_SLOT:C.DAY_END_SLOT].sum(axis=1)
    e_night = np.maximum(P.sum(axis=1) - e_day, 1e-9)
    return pd.DataFrame({
        "E_kWh": P.sum(axis=1) * 0.25 / 1000.0,
        "P_peak_W": peak,
        "Load_factor": P.mean(axis=1) / peak,
        "peak_sin": np.sin(angle),
        "peak_cos": np.cos(angle),
        "ratio_day_night": np.minimum(e_day / e_night,
                                      C.RATIO_DAY_NIGHT_CAP)},
        index=table.index)

def activity(feats):
    """Register of every kept day and per-season occasional_use."""
    energy = feats["E_kWh"]
    threshold = C.DORMANT_SHARE * float(energy.quantile(C.ACTIVE_REF_PCTL))
    active = energy >= threshold
    # Runs on the calendar, so a rejected day does not split a stop
    cal = active.reindex(pd.date_range(min(active.index),
                                       max(active.index)).date)
    off = cal.notna() & ~cal.infer_objects(copy=False).astype(bool)
    runs = (off != off.shift()).cumsum()
    stop = (off & (runs.map(runs.value_counts())
                   >= C.STOP_BLOCK_DAYS))[active.index]
    register = pd.DataFrame(
        {"season": [C.SEASONS[pd.Timestamp(d).month] for d in feats.index],
         "E_kWh": energy, "active": active, "operation_stop": stop})
    counted = register[~register["operation_stop"]]
    stops = register[register["operation_stop"]].groupby("season").size()
    summary = {s: {"occasional_use": round(float(r["active"].mean()), 3),
                   "n_active": int(r["active"].sum()),
                   "n_inactive": int((~r["active"]).sum()),
                   "n_stop_days": int(stops.get(s, 0))}
               for s, r in counted.groupby("season")}
    return register, summary, threshold

def figures(table, register, threshold):
    """Heatmap of the whole record and the active-day timeline."""
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(table.to_numpy(float), aspect="auto", cmap="inferno",
                   extent=[0, 24, len(table), 0])
    ax.set(xlabel="Hour of day", ylabel="Day index",
           title=f"{CLIENT} - power heatmap [W]")
    fig.colorbar(im, ax=ax, label="W")
    fig.savefig(OUT_DIR / "figures" / "02_heatmap.png", dpi=110,
                bbox_inches="tight")
    plt.close(fig)
    energy, active = register["E_kWh"], register["active"]
    fig, ax = plt.subplots(figsize=(12, 4))
    for m, colr, lab in ((active, "#0072B2", "Active day"),
                         (~active, "#B0B0B0", "Inactive day")):
        ax.bar(energy.index[m], energy[m], width=1.0, color=colr, label=lab)
    for day in register.index[register["operation_stop"]]:
        ax.axvspan(day, day + pd.Timedelta(days=1), color="#F5C4C4", zorder=0)
    ax.axhline(threshold, color="red", ls="--", lw=1,
               label=f"Activity threshold ({threshold:.2f} kWh)")
    ax.set(xlabel="Date", ylabel="Energy [kWh]", title=f"{CLIENT} - daily "
           f"energy ({100 * (~active).mean():.0f} % inactive, stops shaded)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    ax.grid(alpha=0.2)
    fig.savefig(OUT_DIR / "figures" / "03_activity_timeline.png",
                bbox_inches="tight", dpi=110)
    plt.close(fig)

def main():
    (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    series, outages = load_series()
    series.rename("power_W").to_csv(OUT_DIR / "timeseries_clean.csv")
    outages.to_csv(OUT_DIR / "outage_register.csv", index=False)
    table, rejected = daily_split(series)
    rejected.to_csv(OUT_DIR / "rejected_days.csv", index_label="day")
    feats = daily_features(table)
    register, summary, threshold = activity(feats)
    register.to_csv(OUT_DIR / "activity_register.csv", index_label="day")
    (OUT_DIR / "activity_summary.json").write_text(json.dumps(summary))
    # The shape chain (clustering, targets) sees the active days only
    keep = register["active"].values
    table[keep].to_csv(OUT_DIR / "daily_matrix_W.csv")
    feats[keep].to_csv(OUT_DIR / "daily_features.csv")
    figures(table, register, threshold)
    print(f"[step1] {CLIENT}: {int(keep.sum())} active days kept, "
          f"{int((~keep).sum())} inactive, {len(rejected)} rejected, "
          f"{len(outages)} outages -> {OUT_DIR.relative_to(REPO)}")

if __name__ == "__main__":
    main()
