# -*- coding: utf-8 -*-
"""Step 10 - Fleet summary: all calibrated clients of this PUE type.

Collects the calibration exports of every client found under
resultats/<type>/ and writes two tables at the type level:
    all_calibrations.csv    one row per client x season
    parameters_ranges.csv   min / median / max of the key quantities

Self-contained script. Run it alone with:
    PUE_TYPE=grain_milling python step10_summary.py
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PUE_TYPE = os.environ.get("PUE_TYPE", "grain_milling")
REPO = Path(__file__).resolve().parent.parent.parent
TYPE_DIR = REPO / "resultats" / PUE_TYPE

RANGE_COLS = ["p1_W", "p2_W", "t1_amp_min", "n_regions",
              "on_minutes_per_day", "occasional_use", "ratio_daily_max",
              "n_pass", "n_pass_holdout", "NRMSE", "err_E_pct",
              "err_P_pct", "cv_E_sim"]


def main():
    rows = []
    for client_dir in sorted(TYPE_DIR.iterdir()):
        export_path = client_dir / "calibration_export.json"
        if not client_dir.is_dir() or not export_path.exists():
            continue
        export = json.loads(export_path.read_text())
        if any("regions" not in sd.get("native", {})
               for sd in export["seasons"].values()):
            print(f"[step10] {client_dir.name}: old-format export, skipped")
            continue
        for season, sd in export["seasons"].items():
            rows.append({
                "client": client_dir.name, "season": season,
                "method_arm": sd["method_arm"],
                "p1_W": sd["p1_W"], "p2_W": sd["p2_W"],
                "t1_amp_min": sd["t1_amp_min"],
                "n_regions": len(sd["native"]["regions"]),
                # Expected running time of an active day, over the regions
                "on_minutes_per_day": sum(
                    g["duty"] * (g["end_min"] - g["start_min"])
                    for g in sd["native"]["regions"]),
                "occasional_use": sd["occasional_use"],
                "tfrv": sd["tfrv"],
                "ratio_daily_max": sd["ratio_daily_max"],
                "n_pass": sd["n_pass"],
                "n_pass_holdout": sd["n_pass_holdout"],
                "all_criteria_met": sd["all_criteria_met"],
                "activity_cv": sd["activity_cv"],
                "cv_E_meas": sd["cv_E_meas"], "cv_E_sim": sd["cv_E_sim"],
                **sd["fit"]})
    if not rows:
        print(f"[step10] no calibrated client under "
              f"{TYPE_DIR.relative_to(REPO)}")
        return
    df = pd.DataFrame(rows)
    df.to_csv(TYPE_DIR / "all_calibrations.csv", index=False)
    ranges = [{"parameter": c, "min": float(np.nanmin(df[c])),
               "median": float(np.nanmedian(df[c])),
               "max": float(np.nanmax(df[c])),
               "n": int(df[c].notna().sum())} for c in RANGE_COLS]
    pd.DataFrame(ranges).to_csv(TYPE_DIR / "parameters_ranges.csv",
                                index=False)
    print(f"[step10] {PUE_TYPE}: {len(df)} client-seasons, criteria passed "
          f"{int(df['n_pass'].sum())}/{6 * len(df)}")


if __name__ == "__main__":
    main()
