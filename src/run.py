"""Whole chain, from the meter series to the published results.

Usage: python run.py [--workers 8] [--prefix rejeu/] [--from 4] [--to 7]
  workers: targets calibrated at the same time (one processor each)
  prefix : sub-folder of resultats/ for the calibration runs; with --prefix rejeu/, published
           results untouched and new ones ready for comparison
  from, to: first and last stages to run, numbered as below (stages 1 to 3 take a few minutes, the whole
           chain about fifteen hours on 15 processors)

Stages and output folders (inside resultats/<prefix>):
  1  daily profiles (step 1)                                   cibles_v9/ (always in resultats/)
  2  targets, season and month (step 2)                        cibles_v9/
  3  running power (step 3)                                    cibles_v9/puissances_mesurees.csv
  4  season calibration, setting of 2026-09-19 (step 4)        calib_directe9/
     season calibration, final setting (step 4)                calib_directe10/
  5  union of the two settings (step 5)                        calib_union_9_10/
  6  tie-break on the harmonics criterion (step 6)             arbitrage_fft/
  7  targets still missing that criterion, calibrated again
     with func_cycle down to 5 min (step 4)                    essai_fft_grille/
     final seasonal reference (step 7)                         reference_saison/
  8  months, func_cycle inherited from the season (step 4)     calib_mois/
  9  publication figures and tables, transfer analysis
     (steps 8 and 9)                                           <reference>/publication/
Resumption after a stop half-way (machine asleep, period without supply) at the point reached:
no new computation for a target already calibrated.
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
import settings as S

# Command-line arguments and fixed settings of the chain
parser = argparse.ArgumentParser()
parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
parser.add_argument("--prefix", default="")
parser.add_argument("--from", dest="start", type=int, default=1)
parser.add_argument("--to", dest="end", type=int, default=9)
args = parser.parse_args()
P = args.prefix
ENV = dict(os.environ, OMP_NUM_THREADS="1")      # one processor per target
GRID_DURATIONS = "5,10,15,30,60"
DEFAULT_DURATION = 15                            # func_cycle of a month without a seasonal equivalent
HERE = Path(__file__).resolve().parent           # folder of this file and of the step scripts


def run(script, *options, grain=None):
    """One step in the foreground, with a stop of the chain on failure."""
    env = dict(ENV, GRAIN=grain) if grain else ENV
    print(f"\n=== {script} {' '.join(options)}", flush=True)
    subprocess.run([sys.executable, str(HERE / script), *options], env=env, check=True)


def run_targets(script, targets, run_folder, options):
    """One process per target, several at a time, each with its own log file, and a stop on failure."""
    logs = S.RESULTS / run_folder / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def one(item):
        target, extra = item
        with open(logs / f"{target}.log", "w") as log:
            code = subprocess.run([sys.executable, str(HERE / script), target, *options, *extra],
                                  env=ENV, stdout=log, stderr=subprocess.STDOUT).returncode
        return target, code

    print(f"\n=== {script}: {len(targets)} targets into resultats/{run_folder}", flush=True)
    items = [(t, []) if isinstance(t, str) else t for t in targets]
    with ThreadPoolExecutor(args.workers) as pool:
        for n, (target, code) in enumerate(pool.map(one, items), start=1):
            print(f"   {n}/{len(items)} {target} {'done' if code == 0 else 'FAILED, see its log'}", flush=True)
            if code != 0:
                raise SystemExit(f"{target} failed: see {logs / (target + '.log')}")


def to_calibrate(grain):
    """Targets of a grain with enough days for a calibration."""
    t = pd.read_csv(S.TARGETS / f"cibles_{grain}.csv")
    return list(t[t.statut == "a calibrer"].nom)


def month_tasks():
    """Monthly targets, each with the func_cycle of the seasonal target containing the month.

    Only setting fixed beforehand, as a property of the appliance rather than of the client
    according to the transfer analysis (step 9): share of variance carried by the client 0.20
    for the cold chain and 0.23 for the mills. Search month by month of everything else."""
    # func_cycle of each seasonal target, by client, configuration and season
    duration = {}
    for path in sorted((S.RESULTS / f"{P}reference_saison").glob("*.json")):
        p = json.load(open(path))
        season, _, config = p["cible"].partition(" | ")
        duration[(p["client"], config, season)] = p["func_cycle"]
    # One task per monthly target, with the duration of its season (15 min by default)
    t = pd.read_csv(S.TARGETS / "cibles_mois.csv")
    tasks = []
    for _, r in t[t.statut == "a calibrer"].iterrows():
        month, _, config = str(r.cible).partition(" | ")
        season = S.SEASONS[S.MONTHS.index(month) + 1]
        L = duration.get((r.client, config, season), DEFAULT_DURATION)
        tasks.append((r.nom, ["--durations", str(L)]))
    return tasks


# Stages of the chain, from the one given by --from
if args.start <= 1 <= args.end:
    run("step1_daily_profiles.py")
if args.start <= 2 <= args.end:
    run("step2_targets.py", grain="saison")
    run("step2_targets.py", grain="mois")
if args.start <= 3 <= args.end:
    run("step3_running_power.py")
if args.start <= 4 <= args.end:
    season = to_calibrate("saison")
    run_targets("step4_calibrate.py", season, f"{P}calib_directe9",
                ["--run", f"{P}calib_directe9", "--setting", "2026-09-19"])
    run_targets("step4_calibrate.py", season, f"{P}calib_directe10",
                ["--run", f"{P}calib_directe10", "--setting", "final"])
if args.start <= 5 <= args.end:
    run("step5_union.py", f"{P}calib_union_9_10", f"{P}calib_directe9", f"{P}calib_directe10")
if args.start <= 6 <= args.end:
    run_targets("step6_fft_arbitration.py", to_calibrate("saison"), f"{P}arbitrage_fft",
                ["--run", f"{P}arbitrage_fft", "--reference", f"{P}calib_union_9_10",
                 "--candidates", f"{P}calib_directe10"])
if args.start <= 7 <= args.end:
    missing_fft = [p.stem for p in sorted((S.RESULTS / f"{P}arbitrage_fft").glob("*.json"))
                   if "FFT_err" in json.load(open(p))["manques"]]
    run_targets("step4_calibrate.py", missing_fft, f"{P}essai_fft_grille",
                ["--run", f"{P}essai_fft_grille", "--durations", GRID_DURATIONS])
    run("step7_final_reference.py", f"{P}reference_saison", f"{P}arbitrage_fft", f"{P}essai_fft_grille")
if args.start <= 8 <= args.end:
    run_targets("step4_calibrate.py", month_tasks(), f"{P}calib_mois",
                ["--run", f"{P}calib_mois", "--grain", "mois"])
if args.start <= 9 <= args.end:
    for folder, grain in ((f"{P}reference_saison", "saison"), (f"{P}calib_mois", "mois")):
        run("step8_publication.py", folder, "--grain", grain)
        run("step9_transfer.py", folder, "--grain", grain)
print("\nchain finished", flush=True)
