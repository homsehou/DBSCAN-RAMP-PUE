# -*- coding: utf-8 -*-
"""Step 8 - The final examination (methodology step 11).

Six criteria between target and simulation, all under their thresholds:
NRMSE and LDC and FFT errors under 10 %, energy and peak errors under
10 %, load factor within 0.05. Two halves guard against overfitting:
the model chosen on even days is judged on the odd days it never saw,
and that score is read against the split floor - the distance between
the two halves themselves, the best any model could do. The 95 %
margin over independent seed blocks says how much the score moves with
the dice; the amplitude note flags a burst power the meter cannot
identify. Everything is appended to calibration_export.json and written
to validation_metrics.csv for the figures and the fleet summary.
Run: PUE_TYPE=grain_milling PUE_CLIENT=0016GBO python step8_validation.py
"""
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C   # shared settings of the repository
PUE_TYPE = os.environ.get("PUE_TYPE", "grain_milling")
CLIENT = os.environ.get("PUE_CLIENT", "0016GBO")
REPO = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO / "resultats" / PUE_TYPE / CLIENT


def main():
    export = json.loads((OUT_DIR / "calibration_export.json").read_text())
    targets = pd.read_csv(OUT_DIR / "calibration_targets.csv")
    sims = pd.read_csv(OUT_DIR / "ramp_simulated_profiles.csv")
    seeds = pd.read_csv(OUT_DIR / "sim_seed_profiles.csv")
    slots = [f"slot_{j}" for j in range(C.SLOTS_PER_DAY)]
    def profile(frame, season, **match):
        rows = frame[frame["season"] == season]
        for key, value in match.items():
            rows = rows[rows[key] == value]
        return rows[slots].to_numpy(float)
    rows = []
    for season, sd in export["seasons"].items():
        full = profile(targets, season, kind="full")[0]
        select = profile(targets, season, kind="select")[0]
        holdout = profile(targets, season, kind="holdout")[0]
        sim = profile(sims, season)[0]
        sim_select = profile(targets, season, kind="sim_select")[0]
        # The six criteria on the delivered model, then out of sample
        m_fit = C.metrics(full, sim)
        m_hold = C.metrics(holdout, sim_select)
        floor = C.metrics(holdout, select)["NRMSE"]
        ci_nrmse, ci_fft = C.confidence_margin(
            full, profile(seeds, season))
        sd.update({"fit": m_fit, "n_pass": C.n_passed(m_fit),
                   "holdout": m_hold,
                   "n_pass_holdout": C.n_passed(m_hold),
                   "NRMSE_split_floor": floor,
                   "NRMSE_ci95": ci_nrmse, "FFT_ci95": ci_fft,
                   "all_criteria_met": C.n_passed(m_fit) == 6,
                   "amp_note": C.amp_note(sd["ratio_daily_max"],
                                          sd["t1_amp_min"],
                                          sd["rated_capped"])})
        rows.append({"season": season, **m_fit,
                     **{k + "_holdout": v for k, v in m_hold.items()},
                     **{k: sd[k] for k in C.SCALAR_KEYS}})
        print(f"[step8] {season}: fit {sd['n_pass']}/6  NRMSE "
              f"{m_fit['NRMSE']:.3f}+/-{ci_nrmse:.3f}  holdout "
              f"{m_hold['NRMSE']:.3f} (floor {floor:.3f})  "
              f"{sd['amp_note']}")
    pd.DataFrame(rows).to_csv(OUT_DIR / "validation_metrics.csv",
                              index=False)
    (OUT_DIR / "calibration_export.json").write_text(json.dumps(export))


if __name__ == "__main__":
    main()
