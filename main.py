# -*- coding: utf-8 -*-
"""
main.py — Pipeline of the strict 2R2C model with parameters switched by
the compressor regime.

Strategy: calibration on Test 1 ONLY, blind validation on Test 2.
  1. Test 1 -> analytical identification of the 7 parameters (theta_1)
  2. Test 2 -> direct simulation with theta_1, WITHOUT re-identification
              (independent blind test)

The 7 parameters:
  R_env_on, R_env_off : envelope (R_int varies with the evaporator-driven
                       natural convection, R_paroi constant material)
  R_ap_on, R_ap_off  : air <-> product (same internal convection mechanism)
  C_air              : internal air capacity (constant, D+A mean)
  COP                : refrigeration efficiency (constant)
  E_door             : energy of one door opening (constant)

C_prod(T) : physical function (NIST), measured mass, NOT a parameter.

Usage:
    python3 main.py
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
from src.model import (RegimeParams, simulate, metrics, metrics_by_phase,
                        P_THRESHOLD_W)
from src.identify import identify_all


# =============================================================================
# COMPARISON PLOT (a single one per test)
# =============================================================================

def plot_comparison(ds: pd.DataFrame, T_air_sim: np.ndarray,
                     T_prod_sim: np.ndarray, rp: RegimeParams,
                     phase_metrics: dict, global_metrics: dict,
                     title: str, out_path: Path) -> None:
    t = ds["datetime"]
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True,
                              gridspec_kw={"height_ratios": [2, 2, 1]})
    colors_ph = {"A": "#2ecc71", "B": "#e67e22", "C": "#3498db", "D": "#9b59b6"}

    # Panel 1: T_air
    ax = axes[0]
    ax.plot(t, ds["Tair_in"], color="#1f77b4", lw=0.7, alpha=0.9,
            label="measured T_air")
    ax.plot(t, T_air_sim, color="#d62728", lw=0.7, alpha=0.9,
            label="model T_air")
    for ph, col in colors_ph.items():
        mask = (ds["phase"] == ph).to_numpy()
        if mask.sum() > 0:
            idx0 = np.where(mask)[0][0]
            ax.axvline(t.iloc[idx0], color=col, ls="--", alpha=0.6, lw=1.2)
            ax.text(t.iloc[idx0], ax.get_ylim()[1], f" {ph}", color=col,
                    fontsize=11, weight="bold", va="top")
    ax.set_ylabel("internal T_air (°C)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)

    # Panel 2: T_prod
    ax = axes[1]
    ax.plot(t, ds["Teau"], color="#2ca02c", lw=0.7, alpha=0.9,
            label="measured T_prod")
    ax.plot(t, T_prod_sim, color="#9467bd", lw=0.7, alpha=0.9,
            label="model T_prod")
    for ph, col in colors_ph.items():
        mask = (ds["phase"] == ph).to_numpy()
        if mask.sum() > 0:
            idx0 = np.where(mask)[0][0]
            ax.axvline(t.iloc[idx0], color=col, ls="--", alpha=0.6, lw=1.2)
    ax.set_ylabel("T_prod (water, °C)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Panel 3: residuals + ON-regime band
    ax = axes[2]
    ax.plot(t, T_air_sim - ds["Tair_in"].to_numpy(dtype=float),
            color="#d62728", lw=0.5, alpha=0.7, label="T_air residual")
    ax.plot(t, T_prod_sim - ds["Teau"].to_numpy(dtype=float),
            color="#9467bd", lw=0.5, alpha=0.7, label="T_prod residual")
    ax.axhline(0, color="k", lw=0.8, alpha=0.6)
    ax.axhline(1.0, color="g", lw=0.5, ls=":", alpha=0.6, label="±1 °C")
    ax.axhline(-1.0, color="g", lw=0.5, ls=":", alpha=0.6)
    ax.set_ylabel("residual (°C)")
    ax.set_xlabel("date and time")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9, ncol=3)
    ax.grid(True, alpha=0.3)

    # ON-regime band (grey) at the bottom of the residual panel
    P = np.nan_to_num(ds["P_abs_mean_W"].to_numpy(dtype=float), nan=0.0)
    on_mask = P > P_THRESHOLD_W
    y_lo = ax.get_ylim()[0]
    ax.fill_between(t, y_lo, y_lo + 0.3, where=on_mask, color="k", alpha=0.25,
                     step="mid")

    # Summary box BELOW the figure (outside the data area)
    rap_ratio = rp.R_ap_off / rp.R_ap_on
    renv_ratio = rp.R_env_off / rp.R_env_on
    is_valid = global_metrics['Tair']['cv_rmse_pct'] <= 30
    info_lines = [
        f"2R2C model — parameters switched on $P_{{elec}}$ > 30 W   "
        f"|   R_env : ON = {rp.R_env_on:.3f} K/W, OFF = {rp.R_env_off:.3f} K/W "
        f"(OFF/ON = {renv_ratio:.2f})   |   R_ap : ON = {rp.R_ap_on:.3f} K/W, "
        f"OFF = {rp.R_ap_off:.3f} K/W (OFF/ON = {rap_ratio:.2f})",
        f"C_air = {rp.C_air:.0f} J/K   |   COP = {rp.COP:.3f}   |   "
        f"E_door = {rp.E_door/1000:.1f} kJ   |   "
        f"RMSE T_air = {global_metrics['Tair']['rmse']:.2f} °C "
        f"(CV-RMSE = {global_metrics['Tair']['cv_rmse_pct']:.1f} %)   |   "
        f"RMSE T_prod = {global_metrics['Tprod']['rmse']:.2f} °C "
        f"(CV-RMSE = {global_metrics['Tprod']['cv_rmse_pct']:.1f} %)",
        "RMSE T_air per phase: " + "   ".join(
            f"{ph} = {phase_metrics.get(ph,{}).get('Tair',{}).get('rmse',np.nan):.2f} °C"
            for ph in ("A", "B", "C", "D")
        )
        + f"   |   ASHRAE Guideline 14 (CV-RMSE ≤ 30 %) : "
        + ("✓ PASSED" if is_valid else "✗ EXCEEDED"),
    ]

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.text(0.5, 0.005, "\n".join(info_lines),
             ha="center", va="bottom", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.6", facecolor="wheat",
                       alpha=0.9, edgecolor="#888"))

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# =============================================================================
# CALIBRATION ON TEST 1
# =============================================================================

def calibrate_on_test1() -> dict:
    """Analytical identification of the 7 parameters on Test 1.
    Simulation of Test 1 with these parameters and self-evaluation metrics.
    """
    test_name = "Test1"
    print(f"\n{'='*72}")
    print(f"  CALIBRATION: {test_name}")
    print(f"{'='*72}")

    ds, door_times_s = load_test(test_name)
    print(f"  {len(ds)} points, {len(door_times_s)} door openings")
    for ph in ("A", "B", "C", "D"):
        n_ph = (ds["phase"] == ph).sum()
        print(f"    Phase {ph}: {n_ph} points")
    P = np.nan_to_num(ds["P_abs_mean_W"].to_numpy(dtype=float), nan=0.0)
    frac_on = float((P > P_THRESHOLD_W).mean())
    print(f"  Compressor ON fraction: {frac_on*100:.1f} %")

    # Direct analytical identification
    res = identify_all(ds, door_times_s)
    rp = res["rp"]

    # Continuous simulation on Test 1 (self-evaluation)
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

    print(f"\n  Self-evaluation on Test 1 (calibration data):")
    print(f"    Global RMSE  : T_air = {m_air['rmse']:.2f} °C  "
          f"T_prod = {m_prod['rmse']:.2f} °C")
    print(f"    CV-RMSE      : T_air = {m_air['cv_rmse_pct']:.1f} %  "
          f"T_prod = {m_prod['cv_rmse_pct']:.1f} %  (ASHRAE 14 threshold 30 %)")
    print(f"    RMSE T_air per phase: "
          + "  ".join(f"{ph}={m_phases.get(ph,{}).get('Tair',{}).get('rmse',np.nan):.2f}"
                       for ph in ("A","B","C","D")))

    # Plot
    out_png = C.RESULTS_DIR / f"comparison_{test_name}.png"
    plot_comparison(ds, T_air_sim, T_prod_sim, rp, m_phases,
                     {"Tair": m_air, "Tprod": m_prod},
                     title=f"2R2C model — Test 1 (calibration)", out_path=out_png)

    # JSON export: identified parameters + self-evaluation
    out_json = C.RESULTS_DIR / f"params_{test_name}.json"
    out_json.write_text(json.dumps({
        "rp": rp.to_dict(),
        "phase_D_diagnostic": res["phaseD"],
        "phase_A_diagnostic": res["phaseA"],
        "phase_B_diagnostic": res["phaseB"],
        "C_air_final_J_per_K": res["C_air_final"],
        "metrics_global": {"Tair": m_air, "Tprod": m_prod},
        "metrics_by_phase": m_phases,
        "frac_ON": frac_on,
        "P_THRESHOLD_W": P_THRESHOLD_W,
    }, indent=2, default=str), encoding="utf-8")

    return {
        "test": test_name, "rp": rp, "ds": ds, "door_times_s": door_times_s,
        "T_air_sim": T_air_sim, "T_prod_sim": T_prod_sim,
        "metrics_global": {"Tair": m_air, "Tprod": m_prod},
        "metrics_by_phase": m_phases,
    }


# =============================================================================
# BLIND VALIDATION ON TEST 2 (frozen parameters from Test 1)
# =============================================================================

def validate_on_test2(rp: RegimeParams) -> dict:
    """Application of the rp parameters (identified on Test 1) to Test 2
    WITHOUT re-identification. Independent blind test."""
    test_name = "Test2"
    print(f"\n{'='*72}")
    print(f"  BLIND VALIDATION: {test_name} (frozen parameters from Test 1)")
    print(f"{'='*72}")

    ds, door_times_s = load_test(test_name)
    print(f"  {len(ds)} points, {len(door_times_s)} door openings")
    for ph in ("A", "B", "C", "D"):
        n_ph = (ds["phase"] == ph).sum()
        print(f"    Phase {ph}: {n_ph} points")

    # Direct simulation with parameters rp = theta_1
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

    print(f"\n  Blind simulation:")
    print(f"    Global RMSE  : T_air = {m_air['rmse']:.2f} °C  "
          f"T_prod = {m_prod['rmse']:.2f} °C")
    print(f"    CV-RMSE      : T_air = {m_air['cv_rmse_pct']:.1f} %  "
          f"T_prod = {m_prod['cv_rmse_pct']:.1f} %  (ASHRAE 14 threshold 30 %)")
    print(f"    RMSE T_air per phase: "
          + "  ".join(f"{ph}={m_phases.get(ph,{}).get('Tair',{}).get('rmse',np.nan):.2f}"
                       for ph in ("A","B","C","D")))

    # Plot
    out_png = C.RESULTS_DIR / f"comparison_{test_name}.png"
    plot_comparison(ds, T_air_sim, T_prod_sim, rp, m_phases,
                     {"Tair": m_air, "Tprod": m_prod},
                     title=f"2R2C model — Test 2 (blind validation, frozen theta_1)",
                     out_path=out_png)

    # JSON export: no parameters (they come from Test 1), metrics only
    out_json = C.RESULTS_DIR / f"validation_{test_name}.json"
    out_json.write_text(json.dumps({
        "params_source": "Test1 (frozen theta_1)",
        "rp_used": rp.to_dict(),
        "metrics_global": {"Tair": m_air, "Tprod": m_prod},
        "metrics_by_phase": m_phases,
    }, indent=2, default=str), encoding="utf-8")

    return {
        "test": test_name, "rp_used": rp, "ds": ds, "door_times_s": door_times_s,
        "T_air_sim": T_air_sim, "T_prod_sim": T_prod_sim,
        "metrics_global": {"Tair": m_air, "Tprod": m_prod},
        "metrics_by_phase": m_phases,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*72)
    print("  Strict 2R2C MODEL, parameters switched by compressor regime")
    print("  Strategy: calibration on Test 1 only, blind validation on Test 2")
    print("="*72)

    # 1. Calibration on Test 1
    res1 = calibrate_on_test1()
    rp1 = res1["rp"]

    # 2. Blind validation on Test 2 (frozen parameters)
    res2 = validate_on_test2(rp1)

    # 3. Summary
    rows = [
        {
            "test": "Test1 (calibration)",
            "R_env_on": rp1.R_env_on, "R_env_off": rp1.R_env_off,
            "R_ap_on": rp1.R_ap_on, "R_ap_off": rp1.R_ap_off,
            "C_air": rp1.C_air, "COP": rp1.COP, "E_door": rp1.E_door,
            "RMSE_Tair_C": res1["metrics_global"]["Tair"]["rmse"],
            "RMSE_Tprod_C": res1["metrics_global"]["Tprod"]["rmse"],
            "CV_RMSE_Tair_pct": res1["metrics_global"]["Tair"]["cv_rmse_pct"],
            "CV_RMSE_Tprod_pct": res1["metrics_global"]["Tprod"]["cv_rmse_pct"],
            "R2_Tair": res1["metrics_global"]["Tair"]["r2"],
            "R2_Tprod": res1["metrics_global"]["Tprod"]["r2"],
        },
        {
            "test": "Test2 (blind validation)",
            "R_env_on": rp1.R_env_on, "R_env_off": rp1.R_env_off,
            "R_ap_on": rp1.R_ap_on, "R_ap_off": rp1.R_ap_off,
            "C_air": rp1.C_air, "COP": rp1.COP, "E_door": rp1.E_door,
            "RMSE_Tair_C": res2["metrics_global"]["Tair"]["rmse"],
            "RMSE_Tprod_C": res2["metrics_global"]["Tprod"]["rmse"],
            "CV_RMSE_Tair_pct": res2["metrics_global"]["Tair"]["cv_rmse_pct"],
            "CV_RMSE_Tprod_pct": res2["metrics_global"]["Tprod"]["cv_rmse_pct"],
            "R2_Tair": res2["metrics_global"]["Tair"]["r2"],
            "R2_Tprod": res2["metrics_global"]["Tprod"]["r2"],
        },
    ]
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "summary.csv", index=False)

    print(f"\n{'='*72}")
    print(f"  SUMMARY")
    print(f"{'='*72}")
    print(f"\nParameters identified on Test 1 (theta_1):")
    print(f"  R_env  : ON {rp1.R_env_on:.3f}  OFF {rp1.R_env_off:.3f}  "
          f"(ratio {rp1.R_env_off/rp1.R_env_on:.2f})")
    print(f"  R_ap   : ON {rp1.R_ap_on:.3f}  OFF {rp1.R_ap_off:.3f}  "
          f"(ratio {rp1.R_ap_off/rp1.R_ap_on:.2f})")
    print(f"  C_air = {rp1.C_air:.0f} J/K   COP = {rp1.COP:.3f}   "
          f"E_door = {rp1.E_door/1000:.1f} kJ")

    for label, res in (("Test1 (calibration)", res1),
                        ("Test2 (blind validation)", res2)):
        m = res["metrics_global"]
        print(f"\n{label}:")
        print(f"  RMSE T_air = {m['Tair']['rmse']:.3f} °C   "
              f"CV-RMSE = {m['Tair']['cv_rmse_pct']:.1f} %  "
              f"({'PASSES' if m['Tair']['cv_rmse_pct'] <= 30 else 'EXCEEDS'} ASHRAE 14)")

    print(f"\n[OK] Results in: {C.RESULTS_DIR}")


if __name__ == "__main__":
    main()
