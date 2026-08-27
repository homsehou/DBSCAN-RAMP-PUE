# -*- coding: utf-8 -*-
"""Step 4 - Anchors and surrogate inversion (methodology steps 7 and 8).

Reads the measured anchors of each seasonal target - burst power p1
(P90 of the daily maxima, nameplate-capped), standby p2 (P5 of the
positive readings), burst duration t_on (P75 anchor) - then inverts the
target through the surrogate, the analytic mean model of the native RAMP
appliance: the day is cut into up to three duty regions (day/night, one
may wrap midnight) and each region receives the duty its mean asks for.
No simulation happens here; step 5 screens candidates around this
inversion, step 6 corrects them against the real engine. Also splits
each cluster into even/odd days for the later examination, and reads
occasional_use from step 1.
Run: PUE_TYPE=cold_chain PUE_CLIENT=0017SAM python step4_invert.py
"""
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C   # shared settings of the repository
PUE_TYPE = os.environ.get("PUE_TYPE", "cold_chain")
CLIENT = os.environ.get("PUE_CLIENT", "0017SAM")
RATED_W = float(os.environ.get("PUE_RATED_POWER_W") or 0) or None
REPO = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO / "resultats" / PUE_TYPE / CLIENT


def regions_from(target, p2, p1):
    """Surrogate inversion: up to three day regions of similar duty.

    Duty of slot j = (target_j - p2)/(p1 - p2). The day is cut at two
    hour boundaries (three contiguous regions, midnight always a cut, so
    each region is also a usage window of the engine and no switch-on
    event can leak its cycle into a neighbouring region); the cut
    keeping the duty flattest inside each region wins, and neighbouring
    regions of nearly equal duty are merged."""
    duty = np.clip((target - p2) / max(p1 - p2, 1e-6), 0.02, 0.98)
    if duty.max() - duty.min() < 0.05:      # flat day: a single region
        return [{"start": 0, "end": 1440, "duty": float(duty.mean())}]
    cuts = min(((j, k) for j in range(1, 23) for k in range(j + 1, 24)),
               key=lambda jk: sum(
                   float(np.var(duty[4 * a:4 * b]) * (b - a))
                   for a, b in ((0, jk[0]), (jk[0], jk[1]), (jk[1], 24))))
    regions = [{"start": a * 60, "end": b * 60,
                "duty": float(duty[4 * a:4 * b].mean())}
               for a, b in ((0, cuts[0]), (cuts[0], cuts[1]),
                            (cuts[1], 24))]
    merged = [regions[0]]               # neighbours within 0.02 merge
    for g in regions[1:]:
        if abs(g["duty"] - merged[-1]["duty"]) < 0.02:
            share = [merged[-1]["end"] - merged[-1]["start"],
                     g["end"] - g["start"]]
            merged[-1]["duty"] = float(np.average(
                [merged[-1]["duty"], g["duty"]], weights=share))
            merged[-1]["end"] = g["end"]
        else:
            merged.append(g)
    return merged


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
        energy = days.sum(axis=1) * 0.25 / 1000.0
        init["seasons"][season] = {
            "p1_W": p1, "p2_W": p2, "t1_amp_min": t1_amp,
            # Powers the step-5 search may try: the same estimators at
            # other percentiles, inside the physical bounds - never above
            # the nameplate or the highest reading, never below the mean
            # profile the appliance has to be able to draw
            "p1_candidates_W": power_candidates(maxima, target_full, p2),
            "p2_candidates_W": standby_candidates(days, target_full),
            "meas_p75_W": meas_p75,
            "cv_E_meas": float(np.std(energy) / np.mean(energy)),
            "occasional_use": occ_all.get(season, {}).get("occasional_use",
                                                          1.0),
            "holdout_valid": bool(split_ok),
            "rated_capped": bool(RATED_W is not None and p95 > RATED_W),
            "regions": regions_from(target, p2, p1)}
        rows += [prof_row(season, p, kind=k) for k, p in
                 (("select", target), ("holdout", target_val),
                  ("full", target_full))]
        spans = ", ".join(f"{g['start'] // 60:02d}h-{g['end'] // 60:02d}h "
                          f"d={g['duty']:.2f}"
                          for g in init["seasons"][season]["regions"])
        print(f"[step4] {season}: p1 {p1:.0f} W  p2 {p2:.1f} W  "
              f"t_on {t1_amp} min  [{spans}]")
    pd.DataFrame(rows).to_csv(OUT_DIR / "calibration_targets.csv",
                              index=False)
    (OUT_DIR / "calibration_init.json").write_text(json.dumps(init))


if __name__ == "__main__":
    main()
