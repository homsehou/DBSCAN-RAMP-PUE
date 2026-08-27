# -*- coding: utf-8 -*-
"""Step 8 - Calibration figures: real vs simulated, in the reference style.

Four figures per season: 40 (filled real profile + dashed RAMP + duty
region boundaries + residual panel), 41 (load duration curve), 42 (FFT
spectrum), and the 50 panel of the calibrated native duty per region.

Self-contained. Run alone:
    PUE_TYPE=cold_chain PUE_CLIENT=0017SAM python step8_figures.py
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

PUE_TYPE = os.environ.get("PUE_TYPE", "cold_chain")
CLIENT = os.environ.get("PUE_CLIENT", "0017SAM")
REPO = Path(__file__).resolve().parent.parent.parent
OUT = REPO / "resultats" / PUE_TYPE / CLIENT
FIG = OUT / "figures"


def read_profiles(name):
    df = pd.read_csv(OUT / name)
    cols = [c for c in df.columns if c.startswith("slot_")]
    return {r["season"]: r[cols].to_numpy(float) for _, r in df.iterrows()}


def region_spans(regions):
    """Regions as in-day [start_min, end_min] pairs (wrap split in two)."""
    out = []
    for g in regions:
        if g["end_min"] > 1440:               # night region over midnight
            out += [[g["start_min"], 1440], [0, g["end_min"] - 1440]]
        else:
            out.append([g["start_min"], g["end_min"]])
    return out


def fig_calibration(real, sim, windows, season, m, tag):
    color = C.SEASON_COLORS.get(season, "gray")
    fig, axes = plt.subplots(2, 1, figsize=(15, 9),
                             gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    ax.fill_between(C.HOURS, real, alpha=0.3, color=color, label="Real profile")
    ax.plot(C.HOURS, real, color=color, lw=2)
    ax.plot(C.HOURS, sim, "k--", lw=2, label="Calibrated RAMP")
    for i, w in enumerate(windows):
        ax.axvline(w[0] / 60.0, color="green", ls=":", lw=1,
                   label="Duty region boundary" if i == 0 else "")
    ax.set_title(f"RAMP calibration -- {season} "
                 f"({len(windows)} duty regions)\n"
                 f"Average daily consumption="
                 f"{real.sum() * 0.25 / 1000:.2f} kWh/day  "
                 f"NRMSE={m['NRMSE']:.3f}  LDC={m['LDC_err']:.3f}  "
                 f"FFT={m['FFT_err']:.3f}  dE={m['err_E_pct']:+.1f}%  "
                 f"dP={m['err_P_pct']:+.1f}%  dLF={m['err_LF']:+.3f}")
    ax.set_ylabel("Power [W]")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9,
              framealpha=0.95)
    ax.grid(alpha=0.2)
    ax2 = axes[1]
    ax2.bar(C.HOURS, real - sim, width=0.15 * 0.9 * 100 / 60, color=color,
            alpha=0.6)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xlabel("Hour [h]")
    ax2.set_ylabel("Residual [W]")
    ax2.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / f"40_calib_{tag}.png", bbox_inches="tight", dpi=110)
    plt.close(fig)


def fig_ldc(real, sim, season, tag):
    x = np.arange(96) / 96 * 100
    color = C.SEASON_COLORS.get(season, "gray")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, np.sort(real)[::-1], color=color, lw=2, label="Real")
    ax.plot(x, np.sort(sim)[::-1], "k--", lw=2, label="RAMP")
    ax.set(xlabel="% of time", ylabel="Power [W]",
           title=f"Load duration curve -- {season}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / f"41_ldc_{tag}.png", bbox_inches="tight", dpi=110)
    plt.close(fig)


def fig_fft(real, sim, season, fft_err, tag):
    fft_r, fft_s = np.abs(np.fft.rfft(real)), np.abs(np.fft.rfft(sim))
    freq = np.fft.rfftfreq(96, d=0.25)
    color = C.SEASON_COLORS.get(season, "gray")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(freq[1:21], fft_r[1:21], color=color, lw=2, marker="o",
            markersize=4, label="Real")
    ax.plot(freq[1:21], fft_s[1:21], "k--", lw=2, marker="s",
            markersize=4, label="RAMP")
    ax.fill_between(freq[1:21], fft_r[1:21], fft_s[1:21], alpha=0.15,
                    color=color)
    ax.set(xlabel="Frequency [cycles/hour]", ylabel="Amplitude FFT",
           title=f"FFT spectrum -- {season}  (FFT_err = {fft_err:.4f})")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / f"42_fft_{tag}.png", bbox_inches="tight", dpi=110)
    plt.close(fig)


def fig_duty_panel(seasons):
    fig, axes = plt.subplots(len(seasons), 1, sharex=True,
                             figsize=(9, 2.4 * len(seasons)))
    for ax, (season, sd) in zip(np.atleast_1d(axes), seasons.items()):
        duty = np.zeros(96)
        for g in sd["native"]["regions"]:
            j = np.arange(g["start_min"] // 15, g["end_min"] // 15) % 96
            duty[j] = g["duty"]
        ax.step(C.HOURS, duty, where="post", color="#0072B2", lw=2)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("duty [-]")
        ax.set_title(f"{season}: p1 {sd['p1_W']:.0f} W, "
                     f"p2 {sd['p2_W']:.1f} W, t_on {sd['t1_amp_min']} min,"
                     f" occ {sd['occasional_use']:.2f},"
                     f" amplitude ratio {sd['ratio_daily_max']:.2f}",
                     fontsize=9)
    np.atleast_1d(axes)[-1].set_xlabel("Hour of day")
    fig.suptitle(f"{CLIENT} - calibrated native duty per region",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "50_param_summary.png", dpi=110)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    export = json.loads((OUT / "calibration_export.json").read_text())
    targets = read_profiles("seasonal_representative_profiles.csv")
    sims = read_profiles("ramp_simulated_profiles.csv")
    for season, sd in export["seasons"].items():
        real, sim = targets[season], sims[season]
        tag = season.replace(" ", "_")
        wins = region_spans(sd["native"]["regions"])
        fig_calibration(real, sim, wins, season, sd["fit"], tag)
        fig_ldc(real, sim, season, tag)
        fig_fft(real, sim, season, sd["fit"]["FFT_err"], tag)
    fig_duty_panel(export["seasons"])
    # Parameter card of the calibrated model, one row per season
    params = []
    for season, sd in export["seasons"].items():
        regions = sd["native"]["regions"]
        params.append({
            "season": season, "p1_W": sd["p1_W"], "p2_W": sd["p2_W"],
            "t_on_min": sd["t1_amp_min"], "tfrv": sd["tfrv"],
            "occasional_use": sd["occasional_use"],
            "ratio_daily_max": sd["ratio_daily_max"],
            "activity_cv": sd["activity_cv"], "n_regions": len(regions),
            "on_minutes": sum(g["duty"]
                              * (g["end_min"] - g["start_min"])
                              for g in regions)})
    pd.DataFrame(params).to_csv(OUT / "ramp_calibrated_params.csv",
                                index=False)
    print(f"[step8] {CLIENT}: calibration figures -> {FIG.relative_to(REPO)}")


if __name__ == "__main__":
    main()
