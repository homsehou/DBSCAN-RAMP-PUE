"""Step 6: tie-break on the harmonics (FFT) criterion, all other criteria kept.

The final candidates of each target were saved during the calibration. They are replayed with
the check seeds, and the one with the smallest FFT gap is kept, provided it misses no new
criterion and does not worsen the shape (NRMSE and worst one-hour smoothed gap). The version of
the reference competes with the candidates.

Usage: python step6_fft_arbitration.py <target> --run <folder> --reference <folder> --candidates <folder>
       (folders inside resultats/)
"""
import argparse
import json
import shutil
import time
import numpy as np
import pandas as pd
import settings as S
import validation as V
import calibration as K
import plates as PL

SHAPE_MARGIN = 0.005          # margin on NRMSE and on the worst smoothed gap (absolute)

parser = argparse.ArgumentParser()
parser.add_argument("target")
parser.add_argument("--run", required=True)
parser.add_argument("--reference", required=True)
parser.add_argument("--candidates", required=True)
args = parser.parse_args()
NAME = args.target
OUT, REFERENCE, CANDIDATES = (S.RESULTS / f for f in (args.run, args.reference, args.candidates))
for d in ("entrees_ramp", "planches"):
    (OUT / d).mkdir(parents=True, exist_ok=True)
if (OUT / f"{NAME}.json").exists():
    print(NAME, "already arbitrated")
    raise SystemExit(0)
T0 = time.time()

base = json.load(open(REFERENCE / f"{NAME}.json"))
noise = dict(np.load(S.TARGETS / "bruit_saison" / f"{NAME}.npz"))
y = noise["cible"]
r = pd.read_csv(S.TARGETS / "cibles_saison.csv").set_index("nom").loc[NAME]
p95 = {k[4:]: float(r[k]) for k in r.index if k.startswith("p95_")}
plates = base["plaques_appareils"]


def judge(text):
    """Criteria, local shape and verdict of a RAMP file, with the check seeds."""
    days = K.simulate(text, K.SEEDS_CHECK[0], 20, days=True)
    profile = days.mean(axis=0)
    measures = V.criteria(y, profile, noise["pics"])
    local = V.local_shape(y, profile, noise["sd_1h"], noise["sd_2h"])
    verdict, missed = V.verdict(measures, local, p95)
    return {"profil": profile, "mesures": measures, "locale": local, "verdict": verdict, "manques": missed}


start = judge(open(REFERENCE / "entrees_ramp" / f"{NAME}.py").read())
print(f"{NAME}: start FFT {start['mesures']['FFT_err']:.3f}, NRMSE {start['mesures']['NRMSE']:.3f}, "
      f"missed {start['manques']}", flush=True)

# Candidates of the same calibration, replayed with the check seeds
kept = None
for f in json.load(open(CANDIDATES / "finalistes" / f"{NAME}.json")):
    e = judge(f["texte_ramp"])
    held = set(e["manques"]) <= set(start["manques"])
    shape_ok = (e["mesures"]["NRMSE"] <= start["mesures"]["NRMSE"] + SHAPE_MARGIN
                and e["locale"]["PEL_1h"] <= start["locale"]["PEL_1h"] + SHAPE_MARGIN)
    better = e["mesures"]["FFT_err"] < start["mesures"]["FFT_err"] - 1e-6
    print(f"   candidate {f['rang_candidat']} period {f['L_etoile']}: FFT {e['mesures']['FFT_err']:.3f}, "
          f"NRMSE {e['mesures']['NRMSE']:.3f}, missed {e['manques']}", flush=True)
    if held and shape_ok and better and (kept is None or e["mesures"]["FFT_err"] < kept[1]["mesures"]["FFT_err"]):
        kept = (f, e)

if kept is None:
    # Nothing better: the starting version is copied as it is
    for f in (f"{NAME}.json", f"{NAME}_profil.npy", f"entrees_ramp/{NAME}.py",
              f"planches/{NAME}.png", f"planches/{NAME}.pdf"):
        if (REFERENCE / f).exists():
            shutil.copy2(REFERENCE / f, OUT / f)
    print(f"{NAME}: start kept, {round(time.time() - T0)} s", flush=True)
    raise SystemExit(0)

f, e = kept
p = dict(base,
         fenetres=f["fenetres"],
         cycles=[{"p1": plates, "p2": [round(f["p2"] * q / sum(plates), 1) for q in plates],
                  "t1": cy["t1"], "t2": cy["t2"], "rapport_cyclique": round(cy["d"], 4), "plages": rg}
                 for cy, rg in zip(f["cycles"], f["plages"])],
         func_time=f["func_time"], func_cycle=f["func_cycle"], var_w=f["var_w"],
         continu=f["continu"], p2_W=round(f["p2"], 2), t1_haut=f["t1_haut"],
         L_etoile=f["L_etoile"], mesures=e["mesures"], locale=e["locale"],
         seuils={k: V.threshold(k, p95) for k in V.THRESHOLDS} | V.local_thresholds(p95),
         verdict=e["verdict"], manques=e["manques"],
         J_arbitrage=f["J_arbitrage"], etiquettes=f["etiquettes"],
         secondes=round(time.time() - T0),
         arbitrage_fft=f"candidat {f['rang_candidat']}, L* {f['L_etoile']} min")

np.save(OUT / f"{NAME}_profil.npy", e["profil"])
(OUT / "entrees_ramp" / f"{NAME}.py").write_text(f["texte_ramp"])
S.write_json(OUT / f"{NAME}.json", p)
PL.save(PL.plate(dict(p, profil=e["profil"]), noise), OUT / "planches" / NAME)
print(f"{NAME}: FFT {start['mesures']['FFT_err']:.3f} -> {e['mesures']['FFT_err']:.3f}, "
      f"NRMSE {start['mesures']['NRMSE']:.3f} -> {e['mesures']['NRMSE']:.3f}, "
      f"{p['verdict']}, {p['secondes']} s", flush=True)
