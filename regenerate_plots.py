# -*- coding: utf-8 -*-
"""
regenerate_plots.py — Regeneration of the measurement/model comparison
plots for Test 1 (calibration) and Test 2 (blind validation), starting
from the already identified parameters (results/params_Test1.json).

Changes with respect to the original plot:
  - Removal of the residual panel (2 panels: T_air, T_prod).
  - No reference to the ASHRAE Guideline 14 criterion.
  - Display of the NRMSE (normalised, in %).
  - Explicit display of the 7 calibration parameters on the figure.
  - Larger fonts, wider margins for readability.
  - Output: results/comparison_TestX_v2.png (originals preserved).
"""

from __future__ import annotations
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config as C
from src.data import load_test
from src.model import RegimeParams, simulate, metrics, metrics_by_phase


# =============================================================================
# NRMSE: normalised by the absolute mean of the measurements, in %
# (RMSE / |mean(measurements)| x 100). Identical to the classical CV-RMSE
# but without reference to an evaluation standard.
# =============================================================================

def nrmse_pct(sim: np.ndarray, meas: np.ndarray) -> float:
    valid = np.isfinite(sim) & np.isfinite(meas)
    s, m = sim[valid], meas[valid]
    if len(s) == 0:
        return float("nan")
    rmse = float(np.sqrt(np.mean((s - m) ** 2)))
    abs_mean = abs(float(np.mean(m)))
    if abs_mean < 1e-3:
        return float("nan")
    return 100.0 * rmse / abs_mean


# =============================================================================
# 2-PANEL PLOT (no residuals, no ASHRAE)
# =============================================================================

def plot_comparison_v2(ds: pd.DataFrame, T_air_sim: np.ndarray,
                       T_prod_sim: np.ndarray, rp: RegimeParams,
                       phase_metrics: dict, global_metrics: dict,
                       title: str, out_path: Path) -> None:
    t = ds["datetime"]
    Tair_meas = ds["Tair_in"].to_numpy(dtype=float)
    Tprod_meas = ds["Teau"].to_numpy(dtype=float)

    nrmse_air = nrmse_pct(T_air_sim, Tair_meas)
    nrmse_prod = nrmse_pct(T_prod_sim, Tprod_meas)

    fig, axes = plt.subplots(2, 1, figsize=(18, 11), sharex=True,
                              gridspec_kw={"height_ratios": [1, 1]})
    colors_ph = {"A": "#2ecc71", "B": "#e67e22",
                 "C": "#3498db", "D": "#9b59b6"}

    # --- Panel 1: T_air ---------------------------------------------------
    ax = axes[0]
    ax.plot(t, Tair_meas, color="#1f77b4", lw=1.0, alpha=0.9,
            label="measured T_air")
    ax.plot(t, T_air_sim, color="#d62728", lw=1.0, alpha=0.9,
            label="2R2C model T_air")
    for ph, col in colors_ph.items():
        mask = (ds["phase"] == ph).to_numpy()
        if mask.sum() > 0:
            idx0 = np.where(mask)[0][0]
            ax.axvline(t.iloc[idx0], color=col, ls="--", alpha=0.7, lw=1.4)
            ax.text(t.iloc[idx0], ax.get_ylim()[1], f"  {ph}", color=col,
                    fontsize=13, weight="bold", va="top")
    ax.set_ylabel("Internal air temperature  (°C)", fontsize=13)
    ax.legend(loc="upper right", fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.35)
    ax.tick_params(axis="both", labelsize=11)

    # --- Panel 2: T_prod --------------------------------------------------
    ax = axes[1]
    ax.plot(t, Tprod_meas, color="#2ca02c", lw=1.0, alpha=0.9,
            label="measured T_prod (water)")
    ax.plot(t, T_prod_sim, color="#9467bd", lw=1.0, alpha=0.9,
            label="2R2C model T_prod")
    for ph, col in colors_ph.items():
        mask = (ds["phase"] == ph).to_numpy()
        if mask.sum() > 0:
            idx0 = np.where(mask)[0][0]
            ax.axvline(t.iloc[idx0], color=col, ls="--", alpha=0.7, lw=1.4)
    ax.set_ylabel("Product temperature  (°C)", fontsize=13)
    ax.set_xlabel("Date and time", fontsize=13)
    ax.legend(loc="upper right", fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.35)
    ax.tick_params(axis="both", labelsize=11)

    # --- Parameter + metrics box -------------------------------------------
    rmse_air = global_metrics["Tair"]["rmse"]
    rmse_prod = global_metrics["Tprod"]["rmse"]
    bias_air = global_metrics["Tair"]["bias"]
    bias_prod = global_metrics["Tprod"]["bias"]
    r2_air = global_metrics["Tair"]["r2"]
    r2_prod = global_metrics["Tprod"]["r2"]

    rap_ratio = rp.R_ap_off / rp.R_ap_on
    renv_ratio = rp.R_env_off / rp.R_env_on
    info_lines = [
        f"2R2C model   "
        f"|   R_env : ON = {rp.R_env_on:.3f} K/W, OFF = {rp.R_env_off:.3f} K/W "
        f"(OFF/ON = {renv_ratio:.2f})   |   R_ap : ON = {rp.R_ap_on:.3f} K/W, "
        f"OFF = {rp.R_ap_off:.3f} K/W (OFF/ON = {rap_ratio:.2f})",
        f"C_air = {rp.C_air:.0f} J/K   |   COP = {rp.COP:.3f}   |   "
        f"E_door = {rp.E_door/1000:.1f} kJ   |   "
        f"RMSE T_air = {rmse_air:.2f} °C "
        f"(NRMSE = {nrmse_air:.1f} %)   |   "
        f"RMSE T_prod = {rmse_prod:.2f} °C "
        f"(NRMSE = {nrmse_prod:.1f} %)",
        "RMSE T_air per phase: " + "   ".join(
            f"{ph} = {phase_metrics.get(ph,{}).get('Tair',{}).get('rmse',np.nan):.2f} °C"
            for ph in ("A", "B", "C", "D")
        ),
    ]

    fig.tight_layout(rect=[0, 0.10, 1, 0.97])
    fig.text(0.5, 0.01, "\n".join(info_lines),
             ha="center", va="bottom", fontsize=11,
             bbox=dict(boxstyle="round,pad=0.6", facecolor="wheat",
                       alpha=0.9, edgecolor="#888"))

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN — parameter reload and resimulation of the 2 tests
# =============================================================================

def main():
    print("=" * 72)
    print("  Plot regeneration — version without residuals, without ASHRAE")
    print("=" * 72)

    # 1. Reload of the parameters identified on Test 1
    params_path = C.RESULTS_DIR / "params_Test1.json"
    with open(params_path, "r", encoding="utf-8") as f:
        params_json = json.load(f)
    rp_dict = params_json["rp"]
    rp = RegimeParams(**rp_dict)
    print(f"\n[OK] Parameters reloaded from: {params_path.name}")
    print(f"     R_env_on={rp.R_env_on:.3f}  R_env_off={rp.R_env_off:.3f}")
    print(f"     R_ap_on={rp.R_ap_on:.4f}  R_ap_off={rp.R_ap_off:.4f}")
    print(f"     C_air={rp.C_air:.0f} J/K  COP={rp.COP:.3f}  "
          f"E_door={rp.E_door/1000:.1f} kJ")

    # 2. Loop over the two tests
    for test_name, title in (
        ("Test1", "2R2C model — Test 1 (calibration)"),
        ("Test2", "2R2C model — Test 2 (blind validation, frozen Test 1 parameters)"),
    ):
        print(f"\n--- {test_name} ---")
        ds, door_times_s = load_test(test_name)
        T_air_sim, T_prod_sim = simulate(
            ds["datetime"].to_numpy(),
            ds["Tamb"].to_numpy(dtype=float),
            np.nan_to_num(ds["P_abs_mean_W"].to_numpy(dtype=float), nan=0.0),
            rp,
            float(ds["Tair_in"].iloc[0]),
            float(ds["Teau"].iloc[0]),
            door_times_s=door_times_s,
        )
        m_air = metrics(T_air_sim, ds["Tair_in"].to_numpy(dtype=float))
        m_prod = metrics(T_prod_sim, ds["Teau"].to_numpy(dtype=float))
        m_phases = metrics_by_phase(ds, T_air_sim, T_prod_sim)

        nrmse_air = nrmse_pct(T_air_sim,
                               ds["Tair_in"].to_numpy(dtype=float))
        nrmse_prod = nrmse_pct(T_prod_sim,
                                ds["Teau"].to_numpy(dtype=float))
        print(f"  RMSE  : T_air = {m_air['rmse']:.2f} °C   "
              f"T_prod = {m_prod['rmse']:.2f} °C")
        print(f"  NRMSE : T_air = {nrmse_air:.1f} %    "
              f"T_prod = {nrmse_prod:.1f} %  (norm. by |mean(measurements)|)")

        out_png = C.RESULTS_DIR / f"comparison_{test_name}_v2.png"
        plot_comparison_v2(ds, T_air_sim, T_prod_sim, rp, m_phases,
                            {"Tair": m_air, "Tprod": m_prod},
                            title=title, out_path=out_png)
        print(f"  [OK] Plot written: {out_png.name}")

    print(f"\n[OK] New plots in: {C.RESULTS_DIR}")


if __name__ == "__main__":
    main()
