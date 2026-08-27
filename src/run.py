# -*- coding: utf-8 -*-
"""Single entry point: run the full step chain of one client.

Usage:  python src/run.py <pue_type> <client> [rated_power_W]
        python src/run.py cold_chain 0017SAM 276
        python src/run.py grain_milling 0016GBO

Each step is an independent script (one file per step); this runner only
launches them in order with the right environment variables.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

STEPS = {
    "cold_chain": [f"thermal_equipment/{s}" for s in (
        "step1_preprocess.py", "step2_clustering.py",
        "step3_cluster_figures.py", "step4_invert.py",
        "step5_search.py", "step6_correct.py", "step7_validation.py",
        "step8_figures.py", "step9_summary.py")],
    "grain_milling": [f"grain_milling/{s}" for s in (
        "step1_preprocess.py", "step2_clustering.py",
        "step3_cluster_figures.py", "step4_activity_windows.py",
        "step5_invert.py", "step6_search.py", "step7_correct.py",
        "step8_validation.py", "step9_figures.py",
        "step10_summary.py")],
}


STEPS["poultry_incubation"] = STEPS["cold_chain"]


def run(pue_type, client, rated_power_W=None, period_start=None,
        period_end=None):
    """Notebook-friendly call: runs the full chain of one client."""
    env = dict(os.environ, PUE_TYPE=pue_type, PUE_CLIENT=client)
    if rated_power_W:
        env["PUE_RATED_POWER_W"] = str(rated_power_W)
    if period_start:
        env["PUE_PERIOD_START"] = str(period_start)
    if period_end:
        env["PUE_PERIOD_END"] = str(period_end)
    for step in STEPS[pue_type]:
        result = subprocess.run(
            [sys.executable, str(REPO / "src" / step)], env=env,
            capture_output=True, text=True)
        print(result.stdout.strip().splitlines()[-1]
              if result.stdout.strip() else f"--- {step}")
        if result.returncode != 0:
            print(result.stdout, result.stderr)
            raise RuntimeError(f"failed at {step}")


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in STEPS:
        print(__doc__)
        return 1
    pue_type, client = sys.argv[1], sys.argv[2]
    env = dict(os.environ, PUE_TYPE=pue_type, PUE_CLIENT=client)
    if len(sys.argv) > 3:
        env["PUE_RATED_POWER_W"] = sys.argv[3]
    for step in STEPS[pue_type]:
        print(f"--- {step}")
        result = subprocess.run([sys.executable, str(REPO / "src" / step)],
                                env=env)
        if result.returncode != 0:
            print(f"FAILED at {step}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
