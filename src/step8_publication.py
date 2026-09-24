"""Step 8: publication figures and tables of a reference folder.

Usage: python step8_publication.py <reference> [--grain saison|mois]   (folder inside resultats/)

Outputs in resultats/<reference>/publication/:
  figures/cibles/<target>.pdf and .png  one figure per target (profile and residuals)
  figures/multiples_<family>.pdf        the targets of a family on one plate
  figures/multiples_client_<client>.pdf monthly grain only: the months of a client, four columns
  figures/criteres.pdf                  gaps relative to the threshold, every criterion
  figures/plancher_journees.pdf         floor of the perfect model against the number of days
  figures/observe_plancher.pdf          observed gap against floor, four criteria
  figures/parametres_<client>.pdf       retained RAMP parameters, per client and fleet
  tableaux/*.md and *.tex               sample, results, ranges of variation
"""
import argparse
import json
import numpy as np
import pandas as pd
import settings as S
import validation as V
import publication_figures as F

parser = argparse.ArgumentParser()
parser.add_argument("reference")
parser.add_argument("--grain", default="saison", choices=["saison", "mois"])
args = parser.parse_args()
REFERENCE = S.RESULTS / args.reference
OUT = REFERENCE / "publication"
(OUT / "figures" / "cibles").mkdir(parents=True, exist_ok=True)
(OUT / "tableaux").mkdir(parents=True, exist_ok=True)
KEYS = list(F.NAMES)


def load():
    """Results of the reference, the noise of each target, and the summary table."""
    ps, noises, rows = [], [], []
    for path in sorted(REFERENCE.glob("*.json")):
        p = json.load(open(path))
        p["profil"] = np.load(REFERENCE / f"{p['nom']}_profil.npy")
        noise = dict(np.load(S.TARGETS / f"bruit_{args.grain}" / f"{p['nom']}.npz"))
        ps.append(p)
        noises.append(noise)
        values = {**p["mesures"], **p["locale"]}
        rows.append({"nom": p["nom"], "client": p["client"], "cible": p["cible"],
                     "famille": p["famille"], "jours_retenus": p["jours_retenus"],
                     "verdict": p["verdict"], "n_manques": len(p["manques"]),
                     **{k: values[k] for k in KEYS},
                     **{f"seuil_{k}": p["seuils"][k] for k in KEYS},
                     **{f"p95_{k}": p["p95"][k] for k in KEYS}})
    return ps, noises, pd.DataFrame(rows)


def factor(k):
    """Conversion to percent: gaps already in %, hours over 24 h, standard deviations unchanged."""
    if k in ("err_E_pct", "err_P_pct"):
        return 1.0
    if k == "ECART_PICS_h":
        return 100 / 24
    if k in ("ELM_1h", "ELM_2h"):
        return 1.0
    return 100.0


def table_ranges(b):
    """Range of variation of each criterion, in percent, with its threshold."""
    rows = []
    for k in KEYS:
        f = factor(k)
        v, s = b[k].abs() * f, b[f"seuil_{k}"] * f
        out = int((b[k].abs() > b[f"seuil_{k}"] + 1e-12).sum())
        nominal = V.THRESHOLDS[k] * f if k in V.THRESHOLDS else np.nan
        unit = "σ" if k in ("ELM_1h", "ELM_2h") else "%"
        rows.append({"critere": F.NAMES[k], "unite": unit,
                     "seuil nominal": "95e centile du bruit" if np.isnan(nominal) else F.comma(nominal),
                     "seuil effectif": f"{F.comma(s.min())} à {F.comma(s.max())}",
                     "minimum": F.comma(v.min()), "moyenne": F.comma(v.mean()),
                     "mediane": F.comma(v.median()), "maximum": F.comma(v.max()),
                     "hors seuil": f"{out} / {len(b)}",
                     "part du seuil (mediane)": F.comma(100 * (b[k].abs() / b[f"seuil_{k}"]).median(), 0) + " %"})
    return pd.DataFrame(rows)


def table_families(b):
    """Results per family: criteria met, NRMSE and energy, verdicts."""
    rows = []
    for fam, g in list(b.groupby("famille")) + [("ensemble", b)]:
        rows.append({"categorie": F.FAMILY.get(fam, "ensemble"), "cibles": len(g),
                     "journees": int(g.jours_retenus.sum()),
                     "criteres tenus": f"{10 * len(g) - g.n_manques.sum()} / {10 * len(g)}",
                     "NRMSE moyen": F.comma(100 * g.NRMSE.mean()) + " %",
                     "NRMSE median": F.comma(100 * g.NRMSE.median()) + " %",
                     "NRMSE max": F.comma(100 * g.NRMSE.max()) + " %",
                     "energie moyenne": F.comma(g.err_E_pct.abs().mean()) + " %",
                     "validees": int((g.verdict == "validee").sum()),
                     "non verifiables": int((g.verdict == "non verifiable").sum()),
                     "rejetees": int((g.verdict == "rejetee").sum())})
    return pd.DataFrame(rows)


def table_targets(b):
    """One row per target: days, four key criteria, verdict."""
    t = b[["client", "cible", "jours_retenus", "NRMSE", "FFT_err", "err_E_pct",
           "ECART_PICS_h", "n_manques", "verdict"]].copy()
    t["cible"] = t.cible.str.replace("Saison seche", "sèche").str.replace("Saison des pluies", "pluies")
    for k, d in (("NRMSE", 1), ("FFT_err", 1), ("err_E_pct", 1), ("ECART_PICS_h", 2)):
        t[k] = (t[k] * (100 if k in ("NRMSE", "FFT_err") else 1)).map(lambda x: F.comma(x, d))
    t["criteres tenus"] = 10 - t.pop("n_manques")
    return t.rename(columns={"jours_retenus": "journées", "NRMSE": "NRMSE (%)",
                             "FFT_err": "harmoniques (%)", "err_E_pct": "énergie (%)",
                             "ECART_PICS_h": "heure des pics (h)"})


ps, noises, b = load()
targets = pd.read_csv(S.TARGETS / f"cibles_{args.grain}.csv")
targets = targets[targets.statut == "a calibrer"]
print(f"{len(ps)} targets read in {REFERENCE.name}", flush=True)

# One figure per target
for p, noise in zip(ps, noises):
    F.save(F.figure_target(p, noise), OUT / "figures" / "cibles" / p["nom"])
print("figures per target: done", flush=True)

# One plate of small multiples per family, in chronological order
for fam in ("cold_chain", "grain_milling", "poultry_incubation"):
    group = [(p, z) for p, z in zip(ps, noises) if p["famille"] == fam]
    group.sort(key=lambda pz: (pz[0]["client"], F.period_rank(pz[0]["cible"].partition(" | ")[0])))
    F.save(F.figure_multiples([p for p, _ in group], [z for _, z in group]),
           OUT / "figures" / f"multiples_{fam}")

# Monthly grain: one plate per client, four columns, months in chronological order
if args.grain == "mois":
    for client in sorted({p["client"] for p in ps}):
        group = [(p, z) for p, z in zip(ps, noises) if p["client"] == client]
        group.sort(key=lambda pz: (pz[0]["cible"].partition(" | ")[2],
                                   F.period_rank(pz[0]["cible"].partition(" | ")[0])))
        title = f"{client}, {F.FAMILY[group[0][0]['famille']]}, {len(group)} mois calibrés"
        F.save(F.figure_multiples([p for p, _ in group], [z for _, z in group], columns=4,
                                  row_height=1.45, title=title),
               OUT / "figures" / f"multiples_client_{client}")
print("small multiples: done", flush=True)

# Summary figures
F.save(F.figure_criteria(b), OUT / "figures" / "criteres")
F.save(F.figure_floor(targets), OUT / "figures" / "plancher_journees")
F.save(F.figure_observed_floor(b), OUT / "figures" / "observe_plancher")
for (client, plates), g in pd.DataFrame(
        [{"client": p["client"], "plaques": " + ".join(f"{q:g}" for q in p["plaques_appareils"]),
          "i": i} for i, p in enumerate(ps)]).groupby(["client", "plaques"]):
    group = [ps[i] for i in g.i]
    title = f"{client}, {F.FAMILY[group[0]['famille']]}, plaque {plates.replace('+', '+ ')} W"
    F.save(F.figure_parameters(group, title),
           OUT / "figures" / f"parametres_{client}_{plates.replace(' + ', '_')}")
print("summary figures: done", flush=True)

# Tables
tables = OUT / "tableaux"
F.write_table(table_families(b), tables, "resultats_par_categorie", "Résultats par catégorie d'usage")
F.write_table(table_ranges(b), tables, "plages_criteres", "Plage de variation de chaque critère de validation")
F.write_table(table_targets(b), tables, "resultats_par_cible", "Résultats cible par cible")
b.to_csv(tables / "bilan_complet.csv", index=False)
print(f"tables: done\noutput: {OUT}", flush=True)
