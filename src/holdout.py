"""Hold-out check of the seasonal chain: calibration on one part of the days, judgement on the other part.

Two splits (--split):
  semaines: calibration on the even calendar weeks, judgement on the odd weeks (35 targets);
  annees  : calibration on the first campaign year of the season, judgement on the last one
            (targets with at least 10 retained days in two different years only).

Same days as the published targets (same DBSCAN retention), split by the parity of the calendar
week counted from Monday 2024-01-01. Whole weeks rather than alternate days, so that two days of the
same week, alike, never sit on both sides. The seasonal chain (steps 4 to 7) runs unchanged on the
even weeks; the final model is then judged on the odd weeks with the validation contract of the
chain (thresholds from the noise of the odd weeks). Reference: the gap between the means of the
two halves of measured days, which is the best a model fitted on one half can reach on the other.

Usage: python holdout.py [--split semaines|annees] [--workers 15] [--from 1] [--to 3]
  1  targets of the two parts (step 2)          resultats/cibles_calage/ and cibles_test/ (annees: cibles_annee_...)
  2  seasonal chain on the calibration part (steps 4 to 7)   resultats/holdout/reference_saison/ (annees: holdout_annee/)
  3  judgement on the test part                              <same folder>/validation_hors_echantillon.csv
Published results (resultats/cibles_v9, reference_saison, calib_mois) read only, never written.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import settings as S
import validation as V

# Command-line arguments and folders of the check
parser = argparse.ArgumentParser()
parser.add_argument("--split", default="semaines", choices=["semaines", "annees"])
parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
parser.add_argument("--from", dest="start", type=int, default=1)
parser.add_argument("--to", dest="end", type=int, default=3)
args = parser.parse_args()
HERE = Path(__file__).resolve().parent
PUBLISHED = S.RESULTS / "cibles_v9"
TAG = "" if args.split == "semaines" else "annee_"
HALVES = {f"cibles_{TAG}calage": 0, f"cibles_{TAG}test": 1}
CALIBRATION, TEST = HALVES
OUT = S.RESULTS / ("holdout" if args.split == "semaines" else "holdout_annee")

# Targets of the two halves: step 2 on the days of one parity of weeks, daily profiles copied
if args.start <= 1 <= args.end:
    for folder, half in HALVES.items():
        (S.RESULTS / folder).mkdir(exist_ok=True)
        shutil.copytree(PUBLISHED / "journees", S.RESULTS / folder / "journees", dirs_exist_ok=True)
        shutil.copy(PUBLISHED / "puissances_mesurees.csv", S.RESULTS / folder)
        env = dict(os.environ, GRAIN="saison", CIBLES=folder, MOITIE=str(half), DECOUPAGE=args.split)
        print(f"\n=== step2_targets.py, weeks of parity {half} into resultats/{folder}", flush=True)
        subprocess.run([sys.executable, str(HERE / "step2_targets.py")], env=env, check=True)

# Seasonal chain on the even weeks, unchanged (steps 4 to 7 of run.py)
if args.start <= 2 <= args.end:
    env = dict(os.environ, CIBLES=CALIBRATION)
    subprocess.run([sys.executable, str(HERE / "run.py"), "--from", "4", "--to", "7", "--prefix", f"{OUT.name}/",
                    "--workers", str(args.workers)], env=env, check=True)


def halves(name):
    """Mean profile, noise and 95th percentiles of a target in each half."""
    out = {}
    for folder in HALVES:
        t = pd.read_csv(S.RESULTS / folder / "cibles_saison.csv").set_index("nom")
        if name not in t.index or t.loc[name, "statut"] != "a calibrer":
            return None
        r = t.loc[name]
        out[folder] = {"noise": dict(np.load(S.RESULTS / folder / "bruit_saison" / f"{name}.npz")),
                       "p95": {k[4:]: float(r[k]) for k in r.index if k.startswith("p95_")},
                       "jours": int(r.jours_retenus)}
    return out


def judge(y, sim, noise, p95):
    """Ten criteria, local shape and verdict of a profile against a target, contract of the chain."""
    m = V.criteria(y, sim, noise["pics"])
    local = V.local_shape(y, sim, noise["sd_1h"], noise["sd_2h"])
    verdict, missed = V.verdict(m, local, p95)
    return m, local, verdict, missed


# Judgement of each final model on the odd weeks, and gap between the two halves of measured days
if args.start <= 3 <= args.end:
    rows = []
    for path in sorted((OUT / "reference_saison").glob("*.json")):
        p, name = json.load(open(path)), path.stem
        h = halves(name)
        if h is None:
            rows.append({"cible": name, "famille": p["famille"], "statut": "moitie de test trop courte"})
            continue
        cal, test = h[CALIBRATION], h[TEST]
        sim = np.load(OUT / "reference_saison" / f"{name}_profil.npy")
        m, local, verdict, missed = judge(test["noise"]["cible"], sim, test["noise"], test["p95"])
        floor, floor_local, _, floor_missed = judge(test["noise"]["cible"], cal["noise"]["cible"], test["noise"], test["p95"])
        rows.append({"cible": name, "famille": p["famille"], "statut": "jugee",
                     "jours_calage": cal["jours"], "jours_test": test["jours"],
                     "verdict_calage": p["verdict"], "manques_calage": " ".join(p["manques"]),
                     "NRMSE_calage": p["mesures"]["NRMSE"],
                     "verdict_test": verdict, "manques_test": " ".join(missed),
                     **{f"{k}_test": v for k, v in (m | local).items()},
                     **{f"{k}_moities": v for k, v in (floor | floor_local).items()},
                     "manques_moities": " ".join(floor_missed)})
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "validation_hors_echantillon.csv", index=False)
    J = T[T.statut == "jugee"]
    print(f"\n{len(J)} targets judged on the odd weeks ({len(T) - len(J)} with a test half too short)")
    print("verdicts on the calibration weeks:", J.verdict_calage.value_counts().to_dict())
    print("verdicts on the test weeks       :", J.verdict_test.value_counts().to_dict())
    for k in ("NRMSE", "err_E_pct", "FFT_err"):
        a, b = J[f"{k}_test"].abs(), J[f"{k}_moities"].abs()
        print(f"{k}: test mean {a.mean():.4f} (median {a.median():.4f}, max {a.max():.4f}); "
              f"between halves mean {b.mean():.4f} (median {b.median():.4f}, max {b.max():.4f}); "
              f"test within the halves gap for {(a <= b).sum()} targets")
