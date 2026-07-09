# -*- coding: utf-8 -*-
"""step5_summary.py - Cold chain step 5: cross-client parameter summary.

Compiles the calibration_export.json of every calibrated client of the
PUE type into one table and one landscape 16:9 figure of the calibrated RAMP
parameter ranges (one panel per parameter: box plot per client plus the
individual season points). Clients without an export yet are skipped.
Outputs in resultats/<type>/: all_calibrations.csv,
parameters_ranges.csv and parameters_ranges_landscape.png.
"""
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import config as C

warnings.filterwarnings("ignore")

CLIENT_COLORS = {"0017SAM": "#E74C3C", "0018SAM": "#3498DB",
                 "0035SAM": "#9B59B6", "0151GBO": "#27AE60",
                 "0152GBO": "#E67E22"}
SEASON_MARKERS = {"Dry season": ("o", "Dry season"),
                  "Transition": ("s", "Transition season"),
                  "Rainy season": ("^", "Rainy season")}
# One panel per calibrated RAMP parameter: (group, label, column, description)
PARAMS = [
    ("Power", "power [W]", "power", "Appliance nominal power (= p_X1)"),
    ("Power", "p_X2 [W]", "p_X2", "OFF segment power (basal)"),
    ("Cycle durations", "func_cycle [min]", "func_cycle",
     "Min ON duration after switch-on"),
    ("Cycle durations", "t_11 [min]", "t_11",
     "Cycle 1 ON segment (high duty regime)"),
    ("Cycle durations", "t_21 [min]", "t_21",
     "Cycle 2 ON segment (mid duty regime)"),
    ("Cycle durations", "t_31 [min]", "t_31",
     "Cycle 3 ON segment (low duty regime)"),
    ("Stochasticity", "random_var_w", "random_var_w",
     "Window boundary variability"),
    ("Stochasticity", "time_fraction_random_variability",
     "time_fraction_random_variability", "Total ON time variability"),
    ("Stochasticity", "thermal_p_var", "thermal_p_var",
     "Thermal power variability"),
]


def collect_calibrations() -> pd.DataFrame:
    """One row per client x season, from the available calibration exports."""
    rows = []
    for client in C.CLIENTS:
        path = C.client_results_dir(client) / C.CALIB_EXPORT_JSON
        if not path.exists():
            print(f"  {client}: no calibration export yet, skipped")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for season, sd in data["seasons"].items():
            gp = sd["global_params"]
            cyc = {0: {}, 1: {}, 2: {}}
            for w in sd["windows"]:
                # duty (duty cycle): ON fraction of func_cycle within one
                # cycle, so t_X1 = duty * fc (ON) and t_X2 = (1 - duty) * fc (OFF)
                cyc[w["cycle_idx"]] = {"p1": w["p1_W"], "p2": w["p2_W"],
                                       "duty": w["duty"]}
            # p1/p2 shared across cycles
            p1 = max(cp.get("p1", 0) for cp in cyc.values())
            p2 = min((cp["p2"] for cp in cyc.values() if cp), default=0)
            fc = float(gp.get("func_cycle_min", 0))
            duty = [float(cyc[i].get("duty", 0)) for i in range(3)]
            rows.append({
                "client": data["client"], "season": season,
                "n_pass": int(sd["n_pass"]), "converged": bool(sd["converged"]),
                "power": float(p1), "p_X2": float(p2), "func_cycle": fc,
                "random_var_w": float(gp.get("random_var_w", 0)),
                "time_fraction_random_variability": float(
                    gp.get("time_frac_var", 0)),
                "thermal_p_var": float(gp.get("thermal_p_var", 0)),
                # ON/OFF durations: cycle 1 = high regime (duty_2), cycle 2 =
                # mid (duty_1), cycle 3 = low (duty_0)
                "t_11": duty[2] * fc, "t_12": (1 - duty[2]) * fc,
                "t_21": duty[1] * fc, "t_22": (1 - duty[1]) * fc,
                "t_31": duty[0] * fc, "t_32": (1 - duty[0]) * fc,
                "duty_0": duty[0], "duty_1": duty[1], "duty_2": duty[2],
                "func_time_min": float(gp.get("func_time_min", 0)),
                "occasional_use": float(gp.get("occasional_use", 1.0)),
                # Validation metrics, measured vs synthetic RAMP profile:
                # NRMSE (Normalized Root-Mean-Square Error), LDC_err (Load
                # Duration Curve error, sorted-power view), FFT_err (Fast
                # Fourier Transform spectrum error, periodicity view), err_LF
                # (Load Factor error, mean power over peak power)
                "NRMSE": float(sd["validation"]["NRMSE"]),
                "LDC_err": float(sd["validation"]["LDC_err"]),
                "FFT_err": float(sd["validation"]["FFT_err"]),
                "err_E_pct": float(sd["validation"]["err_E_pct"]),
                "err_P_pct": float(sd["validation"]["err_P_pct"]),
                "err_LF": float(sd["validation"]["err_LF"])})
    return pd.DataFrame(rows)


def fig_landscape(df: pd.DataFrame, png_path, csv_path) -> None:
    """Landscape 16:9 figure of the parameter ranges + summary CSV."""
    clients = sorted(df["client"].unique())
    n_cols = 3
    n_rows = (len(PARAMS) + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(24, 13.5))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.65, wspace=0.28,
                  bottom=0.07, top=0.90, left=0.05, right=0.97)

    summary_rows = []
    for idx, (group, label, col, descr) in enumerate(PARAMS):
        ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
        bp = ax.boxplot([df[df["client"] == c][col].values for c in clients],
                        positions=range(len(clients)), widths=0.6,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=1.5))
        for patch, client in zip(bp["boxes"], clients):
            patch.set_facecolor(CLIENT_COLORS.get(client, "#999"))
            patch.set_alpha(0.7)
        # Individual points, one marker per season
        for ci, client in enumerate(clients):
            for _, r in df[df["client"] == client].iterrows():
                marker = SEASON_MARKERS.get(r["season"], ("x", ""))[0]
                ax.scatter(ci + np.random.uniform(-0.15, 0.15), r[col], s=50,
                           marker=marker, alpha=0.8, edgecolor="black",
                           linewidth=0.5,
                           color=CLIENT_COLORS.get(client, "#999"))
        vals = df[col].dropna().values
        ax.set_title(f"{label}\nrange [{vals.min():.2g}, {vals.max():.2g}]   "
                     f"median = {np.median(vals):.2g}   (n={len(vals)})",
                     fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(clients)))
        ax.set_xticklabels(clients, fontsize=9, rotation=15)
        ax.set_xlabel(descr, fontsize=9, style="italic", color="#555",
                      labelpad=8)
        ax.grid(alpha=0.2, axis="y")
        summary_rows.append({"group": group, "parameter": label,
                             "description": descr, "min": float(vals.min()),
                             "max": float(vals.max()),
                             "median": float(np.median(vals)),
                             "mean": float(np.mean(vals)),
                             "std": float(np.std(vals)),
                             "n_samples": int(len(vals))})

    # Shared legend: one colour per client, one marker per season
    handles = [plt.Rectangle((0, 0), 1, 1, alpha=0.7, label=c,
                             color=CLIENT_COLORS.get(c, "#999"))
               for c in clients]
    handles.append(plt.Line2D([0], [0], marker="", color="white",
                              linestyle="None", label="   "))
    for marker, en_label in SEASON_MARKERS.values():
        handles.append(plt.Line2D([0], [0], marker=marker, color="gray",
                                  linestyle="None", markersize=10,
                                  markeredgecolor="black", label=en_label))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               fontsize=11, bbox_to_anchor=(0.5, 0.95), frameon=True)
    fig.suptitle(f"RAMP parameter ranges across {len(df)} calibrations "
                 f"({len(clients)} {C.PUE_TYPE.replace('_', ' ')} clients x "
                 f"{df['season'].nunique()} seasons)",
                 fontsize=15, fontweight="bold", y=0.985)
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    print(f"  -> {png_path}\n  -> {csv_path}")


def main() -> None:
    print(f"\n=== STEP5 SUMMARY : {C.PUE_TYPE} ===")
    df = collect_calibrations()
    if df.empty:
        print("  no calibrated client found, run step 3 first")
        return
    print(f"  {len(df)} calibrations ({df['client'].nunique()} clients, "
          f"{df['season'].nunique()} seasons)")
    df.to_csv(C.RESULTS_DIR / "all_calibrations.csv", index=False)
    fig_landscape(df, C.RESULTS_DIR / "parameters_ranges_landscape.png",
                  C.RESULTS_DIR / "parameters_ranges.csv")


if __name__ == "__main__":
    main()
