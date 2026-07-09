"""Single entry point of the RAMP calibration, valid for every PUE type.

Execution of the processing chain matching the requested type, with outputs written
to resultats/<type>/<client>/. Client-specific script versions, when they exist, live
in <pipeline>/overrides/<client>/; everything else is shared code.

Usage: python run.py <type> <client> [step ...]
"""
import os
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

SRC = Path(__file__).resolve().parent

# PUE type to processing chain mapping. Cold appliances and the poultry incubator
# are both thermostatic duty-cycle loads (compressor / heating element), so they
# share the `thermal_equipment` chain; each keeps its own data, notebooks and
# results folders, discovered by registry.py.
PIPELINES = {
    "grain_milling": "grain_milling",
    "cold_chain": "thermal_equipment",
    "poultry_incubation": "thermal_equipment",
}

# Pipelines whose scripts expect the client code as a command line argument.
ARGV_CLIENT = {"thermal_equipment"}

# Scripts of each chain, five steps each. For the thermal_equipment chain (cold
# appliances and the incubator): preprocessing, DBSCAN clustering, quantile-based
# calibration (Latin-Hypercube screening + Nelder-Mead), calibration figures,
# cross-client parameter summary.
STEPS_BY_PIPELINE = {
    "grain_milling": ["step1_preprocess.py", "step2_clustering.py",
                      "step3_analytical_calib.py", "step4_ramp_finetune.py",
                      "step5_param_report.py"],
    "thermal_equipment": ["step1_preprocess.py", "step2_clustering.py",
                          "step3_calibration.py", "step4_figures.py",
                          "step5_summary.py"],
}

# Human-readable label of each step, shown on the progress bar.
STEP_LABELS = {
    "step1_preprocess.py": "Preprocessing",
    "step2_clustering.py": "Clustering",
    "step3_calibration.py": "Calibration",
    "step3_analytical_calib.py": "Analytical calibration",
    "step4_ramp_finetune.py": "RAMP fine-tuning",
    "step4_figures.py": "Calibration figures",
    "step5_summary.py": "Summary",
    "step5_param_report.py": "Parameter report",
}


def run(pue_type, client, steps=None, period_start=None, period_end=None,
        rated_power_W=None):
    pipeline_dir = SRC / PIPELINES[pue_type]

    # PUE type and client code passed to the scripts through environment variables.
    # The repository root comes first on the Python path, so that the embedded ramp/
    # package (original core replaced by the updated one) is the one imported.
    env = os.environ.copy()
    env["PUE_TYPE"] = pue_type
    env["PUE_CLIENT"] = client
    # Optional client metadata entered in the notebook: the study period to
    # calibrate on and the appliance rated power. Left unset, the chain uses the
    # full span of the CSV and no power ceiling.
    if period_start:
        env["PUE_PERIOD_START"] = str(period_start)
    if period_end:
        env["PUE_PERIOD_END"] = str(period_end)
    if rated_power_W is not None:
        env["PUE_RATED_POWER_W"] = str(rated_power_W)
    env["MPLBACKEND"] = "Agg"          # figure saving without any graphical window
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SRC.parent), str(pipeline_dir), env.get("PYTHONPATH", "")])

    step_list = steps or STEPS_BY_PIPELINE[PIPELINES[pue_type]]
    print(f"Running the {pue_type} chain for {client} ...", flush=True)
    # One progress bar over the steps. The step output is captured rather than
    # streamed, so the notebook stays readable; on failure it is printed so the
    # error stays visible.
    bar = tqdm(step_list, unit="step", file=sys.stdout,
               bar_format="  {desc:<22} |{bar}| {n_fmt}/{total_fmt} steps [{elapsed}]")
    for step in bar:
        bar.set_description_str(STEP_LABELS.get(step, step))
        # Priority order: client-specific version, then type folder, then base chain.
        script = pipeline_dir / step
        if (SRC / pue_type / step).exists():
            script = SRC / pue_type / step
        if (pipeline_dir / "overrides" / client / step).exists():
            script = pipeline_dir / "overrides" / client / step

        command = [sys.executable, str(script)]
        if PIPELINES[pue_type] in ARGV_CLIENT:
            command.append(client)
        result = subprocess.run(command, cwd=str(pipeline_dir), env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            bar.close()
            print(result.stdout)
            raise RuntimeError(f"{step} failed for client {client}")
    bar.close()
    print(f"Done -- {len(step_list)} steps completed.", flush=True)


if __name__ == "__main__":
    arguments = sys.argv[1:]
    run(arguments[0], arguments[1], steps=arguments[2:] or None)
