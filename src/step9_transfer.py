"""Step 9: what transfers from one appliance to another.

For each retained parameter, this step separates what comes from the type of appliance from what
comes from the client. The measure is the share of variance carried by the client within a family:

    share = var(means per client) / (var(means per client) + mean of the within-client variances)

Close to 0, the parameter depends only on the family and the season, so it transfers.
Close to 1, it belongs to the client, so it has to be measured or surveyed.

Usage: python step9_transfer.py <reference> [--grain saison|mois]   (folder inside resultats/)
Outputs in resultats/<reference>/publication/tableaux/:
  transfert_variance.md / .tex     share of variance carried by the client
  transfert_reglages.md / .tex     most frequent value of each setting, per family
  transfert_niveaux.md / .tex      usage level per client
  type_qualite.md / .tex           calibration quality per type of appliance
  type_fiche.md / .tex             RAMP input sheet per type
  transfert_parametres.csv         the detail, one row per target
and the figure publication/figures/transfert.pdf
"""
import argparse
import json
import numpy as np
import pandas as pd
import settings as S
import publication_figures as F

parser = argparse.ArgumentParser()
parser.add_argument("reference")
parser.add_argument("--grain", default="saison", choices=["saison", "mois"])
args = parser.parse_args()
REFERENCE = S.RESULTS / args.reference
OUT = REFERENCE / "publication" / "tableaux"
OUT.mkdir(parents=True, exist_ok=True)
(REFERENCE / "publication" / "figures").mkdir(parents=True, exist_ok=True)

PARAMETER_NAMES = {"func_cycle": "Durée de marche (func_cycle)", "L_etoile": "Période du cycle",
                   "debut_h": "Début de fenêtre", "fin_h": "Fin de fenêtre",
                   "etendue_h": "Étendue des fenêtres", "part_ft": "Part de func_time",
                   "var_w": "Variabilité horaire (random_var_w)", "continu": "Mode de cycle",
                   "d_moy": "Rapport cyclique moyen", "heures_eq": "Heures équivalentes par jour",
                   "kWh_j": "Énergie journalière"}
TRANSFERABLE = ["func_cycle", "L_etoile", "debut_h", "fin_h", "etendue_h", "part_ft"]
OWN = ["d_moy", "heures_eq", "kWh_j"]


def read():
    """Table of the retained parameters, one row per target."""
    rows = []
    for path in sorted(REFERENCE.glob("*.json")):
        p = json.load(open(path))
        profile = np.load(REFERENCE / f"{p['nom']}_profil.npy")
        windows = p["fenetres"]
        span = sum(b - a for a, b in windows)
        plate = sum(p["plaques_appareils"])
        rows.append({"client": p["client"], "famille": p["famille"],
                     "saison": p["cible"].partition(" | ")[0], "nom": p["nom"],
                     "n_fenetres": len(windows), "etendue_h": span / 60,
                     "debut_h": min(a for a, b in windows) / 60,
                     "fin_h": max(b for a, b in windows) / 60,
                     "func_cycle": p["func_cycle"], "L_etoile": p["L_etoile"],
                     "part_ft": p["func_time"] / span, "var_w": p["var_w"],
                     "continu": p["continu"], "occ": p["occasional_use"],
                     "t1_haut": p["t1_haut"], "p2_sur_plaque": p["p2_W"] / plate,
                     "d_moy": float(np.mean([c["rapport_cyclique"] for c in p["cycles"]])),
                     "heures_eq": profile.sum() / 4 / plate, "kWh_j": profile.sum() / 4 / 1000,
                     "marche_sur_plaque": p["marche_sur_plaque"], "jours": p["jours_retenus"],
                     "verdict": p["verdict"], "n_manques": len(p["manques"]),
                     "NRMSE": p["mesures"]["NRMSE"], "FFT": p["mesures"]["FFT_err"],
                     "E": p["mesures"]["err_E_pct"]})
    return pd.DataFrame(rows)


def client_share(g, col):
    """Share of variance carried by the client, within a family."""
    between = g.groupby("client")[col].mean().var(ddof=0)
    within = g.groupby("client")[col].apply(lambda s: s.var(ddof=0)).mean()
    return np.nan if between + within == 0 else between / (between + within)


def table_variance(b):
    """Share of variance carried by the client, parameter by parameter."""
    families = [f for f in ("cold_chain", "grain_milling") if b[b.famille == f].client.nunique() > 1]
    rows = []
    for col in TRANSFERABLE + OWN:
        row = {"parametre": PARAMETER_NAMES[col], "porte par": "type" if col in TRANSFERABLE else "client"}
        for f in families:
            row[F.FAMILY[f]] = F.comma(client_share(b[b.famille == f], col), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def table_settings(b):
    """Most frequent value of each setting and its frequency, per family."""
    rows = []
    for col, name in (("func_cycle", "Durée de marche (min)"), ("L_etoile", "Période du cycle (min)"),
                      ("var_w", "random_var_w"), ("continu", "continuous_duty_cycle"),
                      ("n_fenetres", "Nombre de fenêtres"), ("t1_haut", "t_i1 du cycle haut (min)")):
        row = {"reglage": name}
        for f, g in b.groupby("famille"):
            v = g[col].value_counts()
            row[F.FAMILY[f]] = f"{v.index[0]:g} ({v.iloc[0]}/{len(g)})"
        rows.append(row)
    row = {"reglage": "p_i2 / plaque (médiane)"}
    for f, g in b.groupby("famille"):
        row[F.FAMILY[f]] = F.comma(g.p2_sur_plaque.median(), 2)
    rows.append(row)
    return pd.DataFrame(rows)


def table_levels(b):
    """Usage level per client: equivalent hours, energy, running coefficient."""
    t = b.groupby(["famille", "client"]).agg(
        cibles=("nom", "size"), h_eq_min=("heures_eq", "min"), h_eq_max=("heures_eq", "max"),
        kWh_j=("kWh_j", "mean"), coefficient=("marche_sur_plaque", "first")).reset_index()
    t["famille"] = t.famille.map(F.FAMILY)
    for col, d in (("h_eq_min", 2), ("h_eq_max", 2), ("kWh_j", 2), ("coefficient", 2)):
        t[col] = t[col].map(lambda v: F.comma(v, d))
    return t.rename(columns={"h_eq_min": "heures éq. min", "h_eq_max": "heures éq. max",
                             "kWh_j": "énergie (kWh/j)"})


def table_quality(b, targets):
    """Quality reached per type: criteria met, gaps, noise floor, verdicts."""
    rows = []
    for fam, g in b.groupby("famille"):
        p = targets[targets.famille == fam]
        rows.append({
            "type": F.FAMILY[fam], "cibles": len(g), "clients": g.client.nunique(),
            "journées": int(g.jours.sum()),
            "critères tenus": f"{10 * len(g) - g.n_manques.sum()} / {10 * len(g)}",
            "NRMSE moyen": F.comma(100 * g.NRMSE.mean()) + " %",
            "harmoniques moyen": F.comma(100 * g.FFT.mean()) + " %",
            "énergie moyenne": F.comma(g.E.abs().mean()) + " %",
            "plancher NRMSE": F.comma(100 * p.p95_NRMSE.mean()) + " %",
            "validées": int((g.verdict == "validee").sum()),
            "non vérifiables": int((g.verdict == "non verifiable").sum()),
            "rejetées": int((g.verdict == "rejetee").sum())})
    return pd.DataFrame(rows)


def table_sheet(b):
    """RAMP input sheet per type: most frequent value, or median and observed range."""
    rows = []

    def row(name, col, f=1.0, d=2, mode=False):
        r = {"paramètre RAMP": name}
        for fam, g in b.groupby("famille"):
            v = g[col] * f
            if mode:
                count = v.value_counts()
                r[F.FAMILY[fam]] = f"{count.index[0]:g} ({count.iloc[0]}/{len(g)})"
            else:
                r[F.FAMILY[fam]] = (f"{F.comma(v.median(), d)}  "
                                    f"[{F.comma(v.min(), d)} à {F.comma(v.max(), d)}]")
        rows.append(r)

    row("func_cycle (min)", "func_cycle", mode=True)
    row("période du cycle (min)", "L_etoile", mode=True)
    row("random_var_w", "var_w", mode=True)
    row("continuous_duty_cycle", "continu", mode=True)
    row("nombre de fenêtres", "n_fenetres", mode=True)
    row("étendue des fenêtres (h)", "etendue_h", d=1)
    row("func_time / étendue", "part_ft")
    row("p_i2 / plaque", "p2_sur_plaque")
    row("t_i1 du cycle haut (min)", "t1_haut", d=0)
    row("rapport cyclique moyen", "d_moy")
    row("occasional_use", "occ")
    row("heures équivalentes par jour", "heures_eq", d=1)
    return pd.DataFrame(rows)


b = read()
b.to_csv(OUT / "transfert_parametres.csv", index=False)
variance = table_variance(b)
F.save(F.figure_transfer(variance, b[["famille", "client", "heures_eq"]]),
       REFERENCE / "publication" / "figures" / "transfert")
F.write_table(variance, OUT, "transfert_variance", "Part de variance portée par le client, à l'intérieur d'une famille")
F.write_table(table_settings(b), OUT, "transfert_reglages", "Réglage le plus courant par type d'équipement, et sa fréquence")
F.write_table(table_levels(b), OUT, "transfert_niveaux", "Niveau d'usage, client par client")

targets = pd.read_csv(S.TARGETS / f"cibles_{args.grain}.csv")
targets = targets[targets.statut == "a calibrer"]
F.write_table(table_quality(b, targets), OUT, "type_qualite", "Qualité de la calibration par type de PUE")
F.write_table(table_sheet(b), OUT, "type_fiche", "Fiche de saisie RAMP par type : valeur retenue, et plage observée")

print(F.markdown(variance))
print()
for f, g in b.groupby("famille"):
    means = g.groupby("client").kWh_j.mean()
    if len(means) > 1:
        print(f"{F.FAMILY[f]:<10} ratio of the highest to the lowest user: {means.max() / means.min():.1f} "
              f"({means.min():.2f} to {means.max():.2f} kWh/day)")
print(f"\ntables written in {OUT}")
