#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 5: report of the optimal RAMP appliance parameters kept after step 4.

Outputs: appliance_params_report.csv (one parameter per row, one season per
column), appliance_params_report_flat.csv (one row per season and window),
appliance_params_report.json (structured), figures/50_param_summary.png."""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    ANALYTICAL_PARAMS_CSV, FIG_DIR, PUE_CLIENT, RAMP_PARAMS_CSV, RESULTS_DIR,
    VALIDATION_CSV,
)

REPORT_CSV = os.path.join(RESULTS_DIR, "appliance_params_report.csv")
REPORT_FLAT_CSV = os.path.join(RESULTS_DIR, "appliance_params_report_flat.csv")
REPORT_JSON = os.path.join(RESULTS_DIR, "appliance_params_report.json")

SEASON_COLORS = {"Dry season": "#E74C3C", "Transition": "#F39C12",
                 "Rainy season": "#3498DB"}
SEASON_LABELS_EN = {"Dry season": "Dry season", "Transition": "Transition season",
                    "Rainy season": "Rainy season"}


def season_label_en(season_name):
    """English label of a season, used on the summary figure."""
    return SEASON_LABELS_EN.get(season_name, season_name)


def build_report(df_cal, df_val, df_ana):
    """Structured per-season report built from the three step-3/4 tables.
    Key insertion order load-bearing for the JSON output layout: keep it stable."""
    report = {}
    for _, row in df_cal.iterrows():
        season = row["season"]
        windows = json.loads(row["windows_json"])
        cycles = json.loads(row["cycles_json"])
        val_row = df_val[df_val["season"] == season].iloc[0]
        ana_row = df_ana[df_ana["season"] == season].iloc[0]
        ana_cycles = json.loads(ana_row["cycles_json"])
        ana_windows = json.loads(ana_row["windows_json"])

        season_report = {
            "season": season,
            "converged": bool(row["converged"]),
            "score": float(row["score"]),
            "n_evals": int(row["n_evals"]),
            "global_params": {
                "func_time_min": float(row["func_time"]),
                "func_cycle_min": float(row["func_cycle"]),
                "random_var_w": float(row["random_var_w"]),
                "occasional_use": float(row.get("occasional_use", 1.0)),
                "time_frac_var": float(row.get("time_frac_var", 0.1)),
                "thermal_p_var": float(row.get("thermal_p_var", 0.0)),
                "n_windows": int(row["n_windows"]),
            },
            "analytical_comparison": {
                "func_time_analytical": float(ana_row["func_time"]),
                "func_time_delta_pct": round(
                    (float(row["func_time"]) - float(ana_row["func_time"]))
                    / float(ana_row["func_time"]) * 100, 1
                ) if float(ana_row["func_time"]) > 0 else 0,
            },
            "windows": [],
            # Six step-4 validation metrics: NRMSE (Normalized Root-Mean-Square
            # Error, dimensionless), LDC_err (Load Duration Curve error, shape of
            # the sorted-power profile), FFT_err (Fast Fourier Transform spectral
            # error, periodicity match), err_E_pct (daily energy error in percent),
            # err_P_pct (peak power error in percent), err_LF (load-factor error,
            # load factor = mean power over peak power).
            "validation": {key: float(val_row[key])
                           for key in ("NRMSE", "LDC_err", "FFT_err",
                                       "err_E_pct", "err_P_pct", "err_LF")},
        }

        # Per-window detail: bounds, duration, duty cycle, analytical reference.
        for i, (ws, we) in enumerate(windows):
            cyc = cycles[i]
            w_detail = {
                "window_id": i + 1,
                "start_min": ws,
                "end_min": we,
                "start_h": round(ws / 60, 2),
                "end_h": round(we / 60, 2),
                "duration_min": we - ws,
                "duty_cycle": {
                    "p1_W": round(cyc["p1"], 1),
                    "t1_min": int(cyc["t1"]),
                    "p2_W": round(cyc["p2"], 1),
                    "t2_min": int(cyc.get("t2", 5)),
                    # r_c: RAMP duty-cycle random variability, fractional jitter on
                    # the ON/OFF cycle timing; default 0.1 for a mild ~10% spread
                    # around the nominal cycle when absent from the calibration.
                    "r_c": float(cyc.get("r_c", 0.1)),
                },
            }
            if i < len(ana_cycles):
                ac = ana_cycles[i]
                w_detail["analytical_comparison"] = {
                    "p1_analytical_W": round(ac["p1"], 1),
                    "p1_delta_pct": round(
                        (cyc["p1"] - ac["p1"]) / ac["p1"] * 100, 1
                    ) if ac["p1"] > 0 else 0,
                    "t1_analytical_min": int(ac["t1"]),
                }
            if i < len(ana_windows):
                w_detail["analytical_window"] = {
                    "start_analytical_min": ana_windows[i][0],
                    "end_analytical_min": ana_windows[i][1],
                }
            season_report["windows"].append(w_detail)

        report[season] = season_report
    return report


def print_report(report):
    """Compact console summary: one column per season."""
    seasons = sorted(report.keys())
    width = 28 + len(seasons) * 19
    print("\n" + "-" * width)
    print(f"  {'Parameter':<28s}" + "".join(f" | {s[:16]:>16s}" for s in seasons))
    print("  " + "-" * width)

    rows = [
        ("func_time [min]", lambda r: format(r["global_params"]["func_time_min"], ".1f")),
        ("func_cycle [min]", lambda r: format(r["global_params"]["func_cycle_min"], ".1f")),
        ("random_var_w", lambda r: format(r["global_params"]["random_var_w"], ".4f")),
        ("occasional_use", lambda r: format(r["global_params"]["occasional_use"], ".3f")),
        ("time_frac_var", lambda r: format(r["global_params"]["time_frac_var"], ".4f")),
        ("n_windows", lambda r: format(r["global_params"]["n_windows"], "d")),
        ("NRMSE", lambda r: format(r["validation"]["NRMSE"], ".4f")),
        ("LDC_err", lambda r: format(r["validation"]["LDC_err"], ".4f")),
        ("FFT_err", lambda r: format(r["validation"]["FFT_err"], ".4f")),
        ("dE [%]", lambda r: format(r["validation"]["err_E_pct"], "+.2f")),
        ("dP [%]", lambda r: format(r["validation"]["err_P_pct"], "+.2f")),
        ("dLF", lambda r: format(r["validation"]["err_LF"], "+.4f")),
        ("Score", lambda r: format(r["score"], ".4f")),
        ("Status", lambda r: "CONVERGED" if r["converged"] else "NOT CONVERGED"),
    ]
    for label, getter in rows:
        print(f"  {label:<28s}" + "".join(f" | {getter(report[s]):>16s}" for s in seasons))
    print("  " + "-" * width)


def save_comparison_csv(report, path):
    """Comparison table: one parameter per row, one season per column."""
    seasons = sorted(report.keys())
    rows = []

    def add_row(label, getter):
        row = {"parameter": label}
        for season in seasons:
            row[season] = getter(report[season])
        rows.append(row)

    add_row("func_time_min", lambda r: round(r["global_params"]["func_time_min"], 1))
    add_row("func_cycle_min", lambda r: round(r["global_params"]["func_cycle_min"], 1))
    add_row("random_var_w", lambda r: round(r["global_params"]["random_var_w"], 4))
    add_row("occasional_use", lambda r: round(r["global_params"]["occasional_use"], 3))
    add_row("time_frac_var", lambda r: round(r["global_params"].get("time_frac_var", 0.1), 4))
    add_row("thermal_p_var", lambda r: round(r["global_params"].get("thermal_p_var", 0.0), 4))
    add_row("n_windows", lambda r: int(r["global_params"]["n_windows"]))
    add_row("score", lambda r: round(r["score"], 4))
    add_row("converged", lambda r: bool(r["converged"]))
    for key in ("NRMSE", "LDC_err", "FFT_err", "err_E_pct", "err_P_pct", "err_LF"):
        row = {"parameter": key}
        for season in seasons:
            row[season] = round(report[season]["validation"][key], 4)
        rows.append(row)

    # Eight rows per window; empty cell when the season has fewer windows.
    window_specs = [
        ("start_h", lambda w: round(w["start_h"], 2)),
        ("end_h", lambda w: round(w["end_h"], 2)),
        ("duration_min", lambda w: w["duration_min"]),
        ("p1_W", lambda w: round(w["duty_cycle"]["p1_W"], 1)),
        ("t1_min", lambda w: w["duty_cycle"]["t1_min"]),
        ("p2_W", lambda w: round(w["duty_cycle"]["p2_W"], 1)),
        ("t2_min", lambda w: w["duty_cycle"]["t2_min"]),
        ("r_c", lambda w: round(w["duty_cycle"]["r_c"], 4)),
    ]
    max_windows = max(len(report[s]["windows"]) for s in seasons)
    for i in range(max_windows):
        for suffix, read_value in window_specs:
            row = {"parameter": f"window_{i + 1}_{suffix}"}
            for season in seasons:
                windows = report[season]["windows"]
                row[season] = read_value(windows[i]) if i < len(windows) else ""
            rows.append(row)

    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  -> {path}")


def save_flat_csv(report, path):
    """Flat detailed report: one row per season and window."""
    rows = []
    for season in sorted(report.keys()):
        r = report[season]
        gp = r["global_params"]
        val = r["validation"]
        for w in r["windows"]:
            dc = w["duty_cycle"]
            rows.append({
                "season": season, "converged": r["converged"], "score": r["score"],
                "n_evals": r["n_evals"], "func_time_min": gp["func_time_min"],
                "func_cycle_min": gp["func_cycle_min"], "random_var_w": gp["random_var_w"],
                "occasional_use": gp["occasional_use"],
                "time_frac_var": gp.get("time_frac_var", 0.1),
                "thermal_p_var": gp.get("thermal_p_var", 0.0),
                "n_windows": gp["n_windows"], "window_id": w["window_id"],
                "window_start_h": w["start_h"], "window_end_h": w["end_h"],
                "window_duration_min": w["duration_min"],
                "p1_W": dc["p1_W"], "t1_min": dc["t1_min"],
                "p2_W": dc["p2_W"], "t2_min": dc["t2_min"], "r_c": dc["r_c"],
                "NRMSE": val["NRMSE"], "LDC_err": val["LDC_err"],
                "FFT_err": val["FFT_err"], "err_E_pct": val["err_E_pct"],
                "err_P_pct": val["err_P_pct"], "err_LF": val["err_LF"],
            })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  -> {path}")


def fig_param_summary(report, path):
    """Figure 50: six-panel summary of the calibrated parameters by season."""
    seasons = sorted(report.keys())
    width = 0.25
    # One bar entry per (season, window), shared by panels 3 to 5.
    entries = [(s, w) for s in seasons for w in report[s]["windows"]]
    entry_labels = [f"{season_label_en(s)[:4]} W{w['window_id']}" for s, w in entries]
    entry_colors = [SEASON_COLORS.get(s, "gray") for s, _ in entries]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.35)

    def grouped_bars(ax, keys, xticklabels, title, ylabel):
        """One bar group per metric, one bar per season."""
        x = np.arange(len(keys))
        for j, season in enumerate(seasons):
            vals = [report[season]["validation"][k] for k in keys]
            ax.bar(x + j * width, vals, width, label=season_label_en(season)[:12],
                   color=SEASON_COLORS.get(season, "gray"), alpha=0.8)
        ax.set_xticks(x + width)
        ax.set_xticklabels(xticklabels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)

    grouped_bars(axes[0, 0], ["NRMSE", "LDC_err", "FFT_err"],
                 ["NRMSE", "LDC_err", "FFT_err"], "Shape metrics", "Value")
    grouped_bars(axes[0, 1], ["err_E_pct", "err_P_pct"],
                 ["dE (%)", "dP (%)"], "Energy and peak errors", "Error (%)")
    axes[0, 1].axhline(0, color="k", lw=0.5)

    # Activity windows as a Gantt-like chart.
    ax3 = axes[0, 2]
    for y_pos, (s, w) in enumerate(entries):
        ax3.barh(y_pos, w["end_h"] - w["start_h"], left=w["start_h"], height=0.6,
                 color=SEASON_COLORS.get(s, "gray"), alpha=0.7,
                 edgecolor="black", linewidth=0.5)
        ax3.text(w["start_h"] + (w["end_h"] - w["start_h"]) / 2, y_pos,
                 f"W{w['window_id']}", ha="center", va="center", fontsize=7)
    ax3.set_yticks(range(len(entries)))
    ax3.set_yticklabels([f"{season_label_en(s)[:8]} W{w['window_id']}" for s, w in entries],
                        fontsize=7)
    ax3.set_xlabel("Hour")
    ax3.set_xlim(0, 24)
    ax3.set_title("Activity windows")
    ax3.grid(alpha=0.2, axis="x")

    # ON power p1 by window.
    ax4 = axes[1, 0]
    ax4.bar(range(len(entries)), [w["duty_cycle"]["p1_W"] for _, w in entries],
            color=entry_colors, alpha=0.8)
    ax4.set_xticks(range(len(entries)))
    ax4.set_xticklabels(entry_labels, fontsize=7, rotation=30)
    ax4.set_ylabel("p1 [W]")
    ax4.set_title("ON power (p1) by window")
    ax4.grid(alpha=0.2)

    # ON/OFF durations t1/t2 by window.
    ax5 = axes[1, 1]
    x_pos = np.arange(len(entries))
    ax5.bar(x_pos - 0.15, [w["duty_cycle"]["t1_min"] for _, w in entries], 0.3,
            label="t1 (ON)", color="steelblue", alpha=0.8)
    ax5.bar(x_pos + 0.15, [w["duty_cycle"]["t2_min"] for _, w in entries], 0.3,
            label="t2 (OFF)", color="coral", alpha=0.8)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(entry_labels, fontsize=7, rotation=30)
    ax5.set_ylabel("Duration [min]")
    ax5.set_title("ON/OFF durations by window")
    ax5.legend(fontsize=7)
    ax5.grid(alpha=0.2)

    # Global parameters, variability terms rescaled to percent.
    ax6 = axes[1, 2]
    global_labels = ["func_time\n(min)", "func_cycle\n(min)", "var_w\n(%)",
                     "occ_use\n(%)", "tfv\n(%)", "therm_pv\n(%)"]
    x_g = np.arange(len(global_labels))
    for j, season in enumerate(seasons):
        gp = report[season]["global_params"]
        vals = [gp["func_time_min"], gp["func_cycle_min"], gp["random_var_w"] * 100,
                gp["occasional_use"] * 100, gp.get("time_frac_var", 0.1) * 100,
                gp.get("thermal_p_var", 0.0) * 100]
        ax6.bar(x_g + j * width, vals, width, label=season_label_en(season)[:12],
                color=SEASON_COLORS.get(season, "gray"), alpha=0.8)
    ax6.set_xticks(x_g + width)
    ax6.set_xticklabels(global_labels, fontsize=7)
    ax6.set_title("Global parameters")
    ax6.legend(fontsize=7)
    ax6.grid(alpha=0.2)

    fig.suptitle(f"PUE {PUE_CLIENT} -- Optimal appliance parameters (grain mill)",
                 fontsize=13, fontweight="bold")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def main():
    """Build every final parameter-report output from the calibrated results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 70)
    print(f"  STEP 5 -- Optimal appliance parameter report (PUE {PUE_CLIENT})")
    print("=" * 70)

    print("\n[1/3] Loading ...")
    df_cal = pd.read_csv(RAMP_PARAMS_CSV)
    df_val = pd.read_csv(VALIDATION_CSV)
    df_ana = pd.read_csv(ANALYTICAL_PARAMS_CSV)
    print(f"       {len(df_cal)} calibrated seasons")

    print("\n[2/3] Building the report ...")
    report = build_report(df_cal, df_val, df_ana)
    print_report(report)

    print("\n[3/3] Saving ...")
    save_comparison_csv(report, REPORT_CSV)
    save_flat_csv(report, REPORT_FLAT_CSV)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"  -> {REPORT_JSON}")
    fig_param_summary(report, os.path.join(FIG_DIR, "50_param_summary.png"))

    print("\n  Step 5 done.")


if __name__ == "__main__":
    main()
