# -*- coding: utf-8 -*-
"""
data.py — Data loading and preparation in a single pass.

Output, for Test 1 and Test 2:
  - DataFrame on the Temperature grid (~10 s) with columns:
    datetime, Tamb, Tair_in, Teau, P_abs_mean_W, dt_s, phase
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import config as C


# =============================================================================
# Sheet loading
# =============================================================================

def _load_power(xlsx: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx, sheet_name=C.POWER_SHEET)
    dt_str = df[C.POWER_DATE_COL].astype(str) + " " + df[C.POWER_TIME_COL].astype(str)
    df["datetime"] = pd.to_datetime(dt_str, format=C.POWER_DT_FORMAT, errors="coerce")
    df["P_abs_W"] = pd.to_numeric(df[C.POWER_VALUE_COL], errors="coerce").abs()
    return (df[["datetime", "P_abs_W"]]
            .dropna()
            .sort_values("datetime")
            .reset_index(drop=True))


def _load_temperature(xlsx: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx, sheet_name=C.TEMP_SHEET)
    df["datetime"] = pd.to_datetime(df[C.TEMP_TIME_COL], errors="coerce")
    out = (df[["datetime", C.TEMP_TAMB_COL, C.TEMP_TAIR_COL, C.TEMP_TEAU_COL]]
           .rename(columns={C.TEMP_TAMB_COL: "Tamb",
                            C.TEMP_TAIR_COL: "Tair_in",
                            C.TEMP_TEAU_COL: "Teau"})
           .dropna(subset=["datetime"])
           .sort_values("datetime")
           .reset_index(drop=True))
    for col in ("Tamb", "Tair_in", "Teau"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


# =============================================================================
# Power aggregation: 1 s samples -> 10 s Temperature grid
# =============================================================================

def _aggregate_power(power_1s: pd.DataFrame, temp_10s: pd.DataFrame) -> pd.DataFrame:
    tt = temp_10s["datetime"].to_numpy()
    pt = power_1s["datetime"].to_numpy()
    pp = power_1s["P_abs_W"].to_numpy(dtype=float)
    n = len(tt)
    p_mean = np.full(n, np.nan)
    for k in range(n - 1):
        i0 = np.searchsorted(pt, tt[k], side="left")
        i1 = np.searchsorted(pt, tt[k + 1], side="left")
        if i1 > i0:
            p_mean[k] = np.nanmean(pp[i0:i1])
    if n > 1:
        p_mean[-1] = p_mean[-2] if np.isfinite(p_mean[-2]) else 0.0
    out = temp_10s.copy()
    out["P_abs_mean_W"] = p_mean
    dt_s = out["datetime"].diff().dt.total_seconds().to_numpy(dtype=float).copy()
    dt_s[0] = np.nanmedian(dt_s[1:20]) if len(dt_s) > 1 else C.TEMP_SAMPLING_S
    out["dt_s"] = dt_s
    return out


# =============================================================================
# Phase labelling
# =============================================================================

def _label_phases(df: pd.DataFrame, phases: Dict[str, Tuple]) -> pd.Series:
    labels = np.full(len(df), "", dtype=object)
    t = df["datetime"].to_numpy()
    for name, (t0, t1) in phases.items():
        t0_np = np.datetime64(pd.Timestamp(t0))
        mask = t >= t0_np
        if t1 is not None:
            t1_np = np.datetime64(pd.Timestamp(t1))
            mask = mask & (t < t1_np)
        labels[mask] = name
    return pd.Series(labels, index=df.index)


# =============================================================================
# Door-opening instants (idealised protocol)
# =============================================================================

def build_door_times_s(ds: pd.DataFrame) -> np.ndarray:
    """Door-opening instants in seconds since ds.datetime[0], over the whole Phase B."""
    mask = (ds["phase"] == "B")
    if mask.sum() == 0:
        return np.array([], dtype=float)
    t0_ds = ds["datetime"].iloc[0]
    ds_B = ds[mask]
    t_start_B = (ds_B["datetime"].iloc[0] - t0_ds).total_seconds()
    t_end_B = (ds_B["datetime"].iloc[-1] - t0_ds).total_seconds()
    times = []
    k = 0
    while True:
        tk = t_start_B + k * C.DOOR_PERIOD_S
        if tk + C.DOOR_OPEN_S > t_end_B:
            break
        times.append(tk)
        k += 1
    return np.asarray(times, dtype=float)


# =============================================================================
# PUBLIC API
# =============================================================================

def load_test(test_name: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Loading of one full test.

    test_name : "Test1" or "Test2"
    Return : (dataset, door_times_s)
        dataset : 10 s DataFrame with columns datetime, Tamb, Tair_in, Teau,
                  P_abs_mean_W, dt_s, phase
        door_times_s : vector of door-opening instants (s since dataset start)
    """
    if test_name == "Test1":
        xlsx = C.DATA_TEST1
        phases = C.PHASES_TEST1
    elif test_name == "Test2":
        xlsx = C.DATA_TEST2
        phases = C.PHASES_TEST2
    else:
        raise ValueError(f"Unknown test_name: {test_name}")

    power = _load_power(xlsx)
    temp = _load_temperature(xlsx)
    ds = _aggregate_power(power, temp)
    ds["phase"] = _label_phases(ds, phases)

    # Only the points belonging to the four phases
    ds = ds[ds["phase"].isin(["A", "B", "C", "D"])].reset_index(drop=True)

    door_times_s = build_door_times_s(ds)
    return ds, door_times_s
