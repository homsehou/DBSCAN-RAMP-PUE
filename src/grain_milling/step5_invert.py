# -*- coding: utf-8 -*-
"""Step 5 - Anchors and surrogate inversion (methodology steps 7 and 8).

Reads the measured anchors of each seasonal target - session power p1
(P90 of the daily maxima, nameplate-capped), standby p2 (P5 of the
positive readings), burst anchor t1 (P75) - and the activity windows of
step 4, then inverts the target through the surrogate, the analytic
mean model of the native RAMP appliance: up to three duty regions laid
inside the activity windows (each window is at least one region; the
roughest window is split at its best hour cut while regions remain),
each region both a usage window and a duty-cycle window of the engine.
No simulation here; step 6 screens candidates around this inversion,
step 7 corrects them against the real engine. Also splits each cluster
into even/odd days for the later examination, and reads occasional_use
from step 1.
Run: PUE_TYPE=grain_milling PUE_CLIENT=0016GBO python step5_invert.py
"""
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C   # shared settings of the repository
PUE_TYPE = os.environ.get("PUE_TYPE", "grain_milling")
CLIENT = os.environ.get("PUE_CLIENT", "0016GBO")
RATED_W = float(os.environ.get("PUE_RATED_POWER_W") or 0) or None
REPO = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO / "resultats" / PUE_TYPE / CLIENT


def regions_from(target, p2, p1, windows):
    """Surrogate inversion: up to three duty regions in the windows.

    Every activity window starts as one region of its mean duty; while
    fewer than three regions exist, the region whose duty varies most
    is split at the hour cut flattening it best."""
    duty = np.clip((target - p2) / max(p1 - p2, 1e-6), 0.02, 0.98)
    spans = [[a // C.SLOT_MIN, b // C.SLOT_MIN] for a, b in windows]
    def sse(a, b):
        return float(np.var(duty[a:b]) * (b - a))
    while len(spans) < 3:
        gains = []
        for n, (a, b) in enumerate(spans):
            # A child region stays 2 h or longer, and the two children
            # must really differ - a sliver region would lock the cycle
            # phase and spike the mean profile
            cuts = [c for c in range(a + 8, b - 7, 4)]   # hour cuts
            if not cuts:
                continue
            c = min(cuts, key=lambda c: sse(a, c) + sse(c, b))
            if abs(float(duty[a:c].mean() - duty[c:b].mean())) < 0.02:
                continue
            gains.append((sse(a, b) - sse(a, c) - sse(c, b), n, c))
        if not gains or max(gains)[0] <= 0:
            break
        _, n, c = max(gains)
        spans[n:n + 1] = [[spans[n][0], c], [c, spans[n][1]]]
    return [{"start": a * C.SLOT_MIN, "end": b * C.SLOT_MIN,
             "duty": float(duty[a:b].mean())} for a, b in spans]


def power_candidates(maxima, target, p2):
    """Burst powers offered to the search, measured and bounded.

    Floor: the appliance must be able to draw the mean profile itself
    (a duty of at most 0.98 of the burst). Ceiling: the nameplate, or
    failing that the highest reading ever recorded."""
    low = (float(target.max()) - p2) / 0.98 + p2
    high = float(min(RATED_W or np.inf, max(maxima)))
    if high < low:
        return [float(low)]
    values = {float(np.clip(np.percentile(maxima, q), low, high))
              for q in C.SEARCH_P1_PCTLS}
    return sorted(values | {float(low), float(high)})


def standby_candidates(days, target):
    """Standby powers offered to the search, measured and bounded.

    Never above 90 % of the measured trough: a higher floor would lift
    the whole simulation over the target at night."""
    cap = C.P2_SAFETY * float(target.min())
    positive = days[days > 0]
    values = {float(min(np.percentile(positive, q), cap))
              for q in C.SEARCH_P2_PCTLS}
    return sorted(values | {0.0})


def prof_row(season, prof, **extra):
    """One 96-slot profile as a flat CSV row."""
    return {"season": season, **extra,
            **{f"slot_{j}": float(prof[j])
               for j in range(C.SLOTS_PER_DAY)}}


def main():
    daily = pd.read_csv(OUT_DIR / "daily_matrix_W.csv", index_col=0,
                        parse_dates=True)
    feats = pd.read_csv(OUT_DIR / "clustered_features.csv", index_col=0)
    targets = pd.read_csv(OUT_DIR / "seasonal_representative_profiles.csv")
    windows_all = pd.read_csv(OUT_DIR / "analytical_params.csv")
    occ_all = json.loads((OUT_DIR / "activity_summary.json").read_text())
    peak = float(daily.to_numpy(float).max())
    init = {"rated_power_W": RATED_W, "max_power_seen_W": peak,
            "rated_share_max": peak / RATED_W if RATED_W else None,
            "seasons": {}}
    rows = []
    for _, srow in targets.iterrows():
        season, dom = srow["season"], int(srow["cluster_dominant"])
        pick = (feats["season"] == season).values
        if dom != -1:
            pick &= (feats["cluster"] == dom).values
        days = daily.to_numpy(float)[pick]
        even = np.array([d.toordinal() % 2 == 0 for d in daily.index[pick]])
        split_ok = min(even.sum(), (~even).sum()) >= C.MIN_SPLIT_DAYS
        target_full = days.mean(axis=0)
        target = days[even].mean(axis=0) if split_ok else target_full
        target_val = days[~even].mean(axis=0) if split_ok else target_full
        # Measured anchors of methodology step 7
        maxima = days.max(axis=1)
        p95 = float(np.percentile(maxima, C.P1_PCTL))
        p1 = float(min(p95, RATED_W or np.inf))
        low = float(np.percentile(days[days > 0], C.P2_PCTL))
        p2 = float(min(low, C.P2_SAFETY * target_full.min()))
        meas_p75 = float(np.percentile(maxima, C.BURST_PCTL))
        t1_amp = int(np.clip(round(C.SLOT_MIN * meas_p75 / p1), 2,
                             C.SLOT_MIN))   # 2-min floor: engine-safe
        # Windows of step 4, cut into up to three duty regions
        wrow = windows_all[windows_all["season"] == season].iloc[0]
        windows = json.loads(wrow["windows_json"])
        energy = days.sum(axis=1) * 0.25 / 1000.0
        init["seasons"][season] = {
            "p1_W": p1, "p2_W": p2, "t1_amp_min": t1_amp,
            # Powers the step-6 search may try: the same estimators at
            # other percentiles, inside the physical bounds - never above
            # the nameplate or the highest reading, never below the mean
            # profile the appliance has to be able to draw
            "p1_candidates_W": power_candidates(maxima, target_full, p2),
            "p2_candidates_W": standby_candidates(days, target_full),
            "meas_p75_W": meas_p75,
            "regions": regions_from(target, p2, p1, windows),
            "cv_E_meas": float(np.std(energy) / np.mean(energy)),
            "occasional_use": occ_all.get(season, {}).get("occasional_use",
                                                          1.0),
            "holdout_valid": bool(split_ok),
            "rated_capped": bool(RATED_W is not None and p95 > RATED_W)}
        rows += [prof_row(season, p, kind=k) for k, p in
                 (("select", target), ("holdout", target_val),
                  ("full", target_full))]
        spans = ", ".join(
            f"{g['start'] // 60:02d}h-{g['end'] // 60:02d}h "
            f"d={g['duty']:.2f}"
            for g in init["seasons"][season]["regions"])
        print(f"[step5] {season}: p1 {p1:.0f} W  p2 {p2:.1f} W  "
              f"t_on {t1_amp} min  [{spans}]")
    pd.DataFrame(rows).to_csv(OUT_DIR / "calibration_targets.csv",
                              index=False)
    (OUT_DIR / "calibration_init.json").write_text(json.dumps(init))


if __name__ == "__main__":
    main()
