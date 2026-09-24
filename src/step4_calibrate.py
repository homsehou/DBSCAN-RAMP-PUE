"""Step 4: RAMP calibration of one target at the nameplate power.

Usage: python step4_calibrate.py <target> --run <folder> [--grain saison|mois]
                                 [--setting final|2026-09-19] [--durations 15,30,60]
  target   : column "nom" of resultats/cibles_v9/cibles_<grain>.csv (e.g. 0154GBO__Saison_des_pluies)
  run      : output folder, inside resultats/
  setting  : search setting of the calibration (see SETTINGS)
  durations: grid of func_cycle values (min)

Outputs in resultats/<run>/: <target>.json (parameters, criteria, verdict), <target>_profil.npy,
entrees_ramp/<target>.py (declaration runnable on its own), finalistes/<target>.json,
planches/<target>.png and .pdf. A target already calibrated is skipped.
Method: see calibration.py.
"""
import argparse
import json
import time
import numpy as np
import pandas as pd
import settings as S
import validation as V
import calibration as K
import plates as PL

# The two search settings used for the published results
SETTINGS = {"final": {"peak_weight": 0.0, "short_period": True},
            "2026-09-19": {"peak_weight": 0.03, "short_period": False}}
SLOTS = [f"slot_{i}" for i in range(S.SLOTS_PER_DAY)]

parser = argparse.ArgumentParser()
parser.add_argument("target")
parser.add_argument("--run", required=True)
parser.add_argument("--grain", default="saison", choices=["saison", "mois"])
parser.add_argument("--setting", default="final", choices=list(SETTINGS))
parser.add_argument("--durations", default="15,30,60")
args = parser.parse_args()
NAME = args.target
OUT = S.RESULTS / args.run
for d in ("entrees_ramp", "planches", "finalistes"):
    (OUT / d).mkdir(parents=True, exist_ok=True)
if (OUT / f"{NAME}.json").exists():
    print(NAME, "already calibrated")
    raise SystemExit(0)
T0 = time.time()

# Target, retained days and noise
r = pd.read_csv(S.TARGETS / f"cibles_{args.grain}.csv").set_index("nom").loc[NAME]
days = pd.read_csv(S.TARGETS / "journees" / f"{r.client}.csv")
clusters = pd.read_csv(S.TARGETS / f"clusters_{args.grain}" / f"{r.client}.csv")
d = (clusters[(clusters.cible == r.cible) & clusters.retenue]
     .merge(days, on=["jour", "mois"]).sort_values("jour"))
X, running = d[SLOTS].to_numpy(float), d.marche.to_numpy(bool)
noise = dict(np.load(S.TARGETS / f"bruit_{args.grain}" / f"{NAME}.npz"))
p95 = {k[4:]: float(r[k]) for k in r.index if k.startswith("p95_")}

# Appliance as a RAMP user knows it: nameplate(s) from the survey, measured standby power
config = "" if pd.isna(r.configuration) else str(r.configuration).split(" ")[0]
plates = list(S.TWO_APPLIANCE_NAMEPLATES[r.client]) if config == "depuis" else [S.NAMEPLATE_W[r.client]]
power = pd.read_csv(S.TARGETS / "puissances_mesurees.csv").fillna({"config": ""})
measured = power[(power.client == r.client) & (power.config == config)].iloc[0]
plate_text = " + ".join(f"{p:g} W" for p in plates)
header = f"{r.client}, {r.cible}: {r.famille}, nameplate {plate_text}. RAMP parameters calibrated at the nameplate."
app = K.Appliance(r.client, r.famille, plates, measured.veille_W, header)
print(f"{NAME}: {len(X)} days, nameplate {plate_text}, standby {measured.veille_W} W", flush=True)

res = K.calibrate(app, X, running, noise, p95, durations=tuple(int(v) for v in args.durations.split(",")),
                  log=lambda s: print(s, flush=True), **SETTINGS[args.setting])
c = res["choix"]
p = {"client": r.client, "cible": r.cible, "nom": NAME, "famille": r.famille,
     "fenetres": c["fenetres"],
     "cycles": [{"p1": plates, "p2": [round(cy["p2"] * q / app.nameplate, 1) for q in plates],
                 "t1": cy["t1"], "t2": cy["t2"], "rapport_cyclique": round(cy["d"], 4), "plages": rg}
                for cy, rg in zip(c["cycles"], c["plages"])],
     "plaques_appareils": plates, "statut_plaque": S.TWO_APPLIANCE_STATUS if len(plates) > 1
     else S.NAMEPLATE_STATUS[r.client],
     "func_time": c["func_time"], "func_cycle": c["func_cycle"], "occasional_use": round(c["occ"], 3),
     "var_w": c["var_w"], "continu": c["continu"], "veille_W": float(measured.veille_W),
     "p2_W": round(c["p2"], 2), "t1_haut": c["t1_haut"], "L_etoile": c["L_etoile"],
     "pointe_mediane_actifs_W": round(res["pointe_mediane_actifs"], 1),
     "marche_mesuree_W": float(measured.puissance_marche_W), "marche_sur_plaque": float(measured.coefficient),
     "jours_retenus": len(X), "part_marche": round(float(running.mean()), 3),
     "mesures": res["mesures"], "locale": res["locale"], "p95": p95,
     "seuils": {k: V.threshold(k, p95) for k in V.THRESHOLDS} | V.local_thresholds(p95),
     "verdict": res["verdict"], "manques": res["manques"], "realisme": res["realisme"],
     "mesures_98765": res["mesures_98765"], "J_arbitrage": c["J_arbitrage"],
     "n_geometries": res["n_geometries"], "n_candidats": res["n_candidats"],
     "etiquettes": [int(v) for v in c["etiquettes"]], "reglage": args.setting,
     "secondes": round(time.time() - T0)}

np.save(OUT / f"{NAME}_profil.npy", res["profil"])
(OUT / "entrees_ramp" / f"{NAME}.py").write_text(res["texte"])
S.write_json(OUT / f"{NAME}.json", p)
# numpy numbers turned into plain numbers for the JSON file
S.write_json(OUT / "finalistes" / f"{NAME}.json",
             json.loads(json.dumps(res["finalistes"], default=lambda o: o.item() if hasattr(o, "item") else str(o))))
PL.save(PL.plate(dict(p, profil=res["profil"]), noise), OUT / "planches" / NAME)
print(f"{NAME}: {res['verdict']}, missed {res['manques']}, NRMSE {res['mesures']['NRMSE']:.3f}, "
      f"{p['secondes']} s", flush=True)
