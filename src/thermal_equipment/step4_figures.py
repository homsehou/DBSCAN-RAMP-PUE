# -*- coding: utf-8 -*-
"""step4_figures.py - Cold chain step 4: calibration figures of one client.

Draws the calibration figures from the step 3 outputs: real vs simulated
profile (40), load duration curve (41) and FFT spectrum (42) for each
calibrated season, plus the parameter summary panel (50). The rendering
itself lives in figures.py; this script only reloads the data.
The client code comes from the PUE_CLIENT environment variable (config.py).
"""
import json

import pandas as pd

import config as C
import figures as figs


def profiles_by_season(csv_path) -> dict:
    """Map season -> 96-slot profile, from a CSV with slot_* columns."""
    df = pd.read_csv(csv_path)
    slot_cols = [c for c in df.columns if c.startswith("slot_")]
    return {str(r["season"]): r[slot_cols].values.astype(float)
            for _, r in df.iterrows()}


def season_tag(season: str) -> str:
    """Figure file name of a season: spaces replaced by underscores."""
    return season.replace(" ", "_")


def plot_client(client_code: str) -> None:
    print(f"\n=== STEP4 FIGURES : {client_code} ===")
    rdir = C.client_results_dir(client_code)
    fdir = C.client_fig_dir(client_code)
    export = json.loads((rdir / C.CALIB_EXPORT_JSON).read_text(
        encoding="utf-8"))
    targets = profiles_by_season(rdir / C.SEASONAL_PROFILES_CSV)
    sims = profiles_by_season(rdir / C.SIM_PROFILES_CSV)

    report = {}
    for season, sd in export["seasons"].items():
        if season not in targets or season not in sims:
            continue
        real, sim = targets[season], sims[season]
        metrics = sd["validation"]
        # Effective RAMP cycle windows, back in minutes for the plots
        windows = [[int(round(w["start_h"] * 60)), int(round(w["end_h"] * 60))]
                   for w in sd["windows"]]
        tag = season_tag(season)
        figs.fig_calibration(real, sim, windows, season, metrics,
                             bool(sd["converged"]),
                             path=str(fdir / f"40_calib_{tag}.png"))
        figs.fig_ldc(real, sim, season, str(fdir / f"41_ldc_{tag}.png"))
        figs.fig_fft(real, sim, season, str(fdir / f"42_fft_{tag}.png"),
                     fft_err=metrics["FFT_err"])
        print(f"  [{season}] figures 40/41/42 saved")

        # Nested window entry expected by the parameter summary panel
        report[season] = {
            "validation": metrics,
            "global_params": sd["global_params"],
            "windows": [{"window_id": w["window_id"],
                         "start_h": w["start_h"], "end_h": w["end_h"],
                         "duty_cycle": {"p1_W": w["p1_W"], "p2_W": w["p2_W"],
                                        "t1_min": w["t1_min"],
                                        "t2_min": w["t2_min"],
                                        "duty": w["duty"]}}
                        for w in sd["windows"]],
        }

    figs.fig_param_summary(report, str(fdir / "50_param_summary.png"),
                           client_label=client_code)
    print(f"  figure 50 saved -> {fdir}")


if __name__ == "__main__":
    plot_client(C.PUE_CLIENT)
