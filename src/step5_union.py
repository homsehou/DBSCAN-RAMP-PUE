"""Step 5: union of the two calibration settings, the best version of each seasonal target.

The runs compete target by target, judged by the same J as the calibration (without the
peak-time term), on the profile of the check seeds. The files of the winning version are copied
into the union folder, which is therefore a complete run.

Usage: python step5_union.py <union folder> <run> <run> [<run> ...]   (folders inside resultats/)
Output: the union folder and its table bilan_union.csv
"""
import json
import shutil
import sys
import numpy as np
import pandas as pd
import settings as S
import validation as V
import calibration as K

UNION = S.RESULTS / sys.argv[1]
RUNS = [S.RESULTS / n for n in sys.argv[2:]]
for d in ("entrees_ramp", "planches", "finalistes"):
    (UNION / d).mkdir(parents=True, exist_ok=True)
targets = pd.read_csv(S.TARGETS / "cibles_saison.csv").set_index("nom")
rows = []

for path in sorted(RUNS[0].glob("*.json")):
    name = path.stem
    noise = dict(np.load(S.TARGETS / "bruit_saison" / f"{name}.npz"))
    r = targets.loc[name]
    p95 = {k[4:]: float(r[k]) for k in r.index if k.startswith("p95_")}
    judge = {"sd_1h": noise["sd_1h"], "sd_2h": noise["sd_2h"], "pics": noise["pics"],
             "seuils": V.local_thresholds(p95)}
    scores = {run: K.objective(noise["cible"], np.load(run / f"{name}_profil.npy"), judge) for run in RUNS}
    winner = min(scores, key=scores.get)
    j = json.load(open(winner / f"{name}.json"))
    rows.append({"cible": name, "essai": winner.name, "J": round(scores[winner], 3),
                 **{f"J_{run.name}": round(s, 3) for run, s in scores.items()},
                 "verdict": j["verdict"], "n_manques": len(j["manques"]),
                 "manques": " ".join(j["manques"]), "NRMSE": round(j["mesures"]["NRMSE"], 3),
                 "err_E_pct": round(j["mesures"]["err_E_pct"], 1),
                 "ECART_PICS_h": j["mesures"]["ECART_PICS_h"], "L_etoile": j["L_etoile"]})
    for f in (f"{name}.json", f"{name}_profil.npy", f"entrees_ramp/{name}.py", f"planches/{name}.png",
              f"planches/{name}.pdf", f"finalistes/{name}.json"):
        if (winner / f).exists():
            shutil.copy2(winner / f, UNION / f)

summary = pd.DataFrame(rows)
summary.to_csv(UNION / "bilan_union.csv", index=False)
print(summary.essai.value_counts().to_string())
print("criteria met %d / %d | NRMSE mean %.3f median %.3f max %.3f | %d targets > 10 %%"
      % (10 * len(summary) - summary.n_manques.sum(), 10 * len(summary), summary.NRMSE.mean(),
         summary.NRMSE.median(), summary.NRMSE.max(), (summary.NRMSE > 0.10).sum()))
print(summary.verdict.value_counts().to_string())
