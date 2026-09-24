"""Step 7: final reference, the reference plus the gains of a run with a wider duration grid.

Same rule as step 6: replacement of a target by the version of the run only with no new missed
criterion, neither NRMSE nor the worst one-hour smoothed gap worse (margin 0.005), and a better
harmonics (FFT) criterion. Unchanged copy of the targets absent from the run.

Usage: python step7_final_reference.py <output> <reference> <run> [<run> ...]   (folders inside resultats/)
Output: the output folder and its table bilan_reference.csv
"""
import json
import shutil
import sys
import pandas as pd
import settings as S

# Output folder, reference and runs in competition
OUT = S.RESULTS / sys.argv[1]
REFERENCE = S.RESULTS / sys.argv[2]
RUNS = [S.RESULTS / n for n in sys.argv[3:]]
SHAPE_MARGIN = 0.005          # margin on NRMSE and on the worst smoothed gap (absolute)

for d in ("entrees_ramp", "planches", "finalistes"):
    (OUT / d).mkdir(parents=True, exist_ok=True)


def gain(a, b):
    """Replacement of a by b: no new missed criterion, no worse shape, better FFT."""
    held = set(b["manques"]) <= set(a["manques"])
    shape_ok = (b["mesures"]["NRMSE"] <= a["mesures"]["NRMSE"] + SHAPE_MARGIN
                and b["locale"]["PEL_1h"] <= a["locale"]["PEL_1h"] + SHAPE_MARGIN)
    return held and shape_ok and b["mesures"]["FFT_err"] < a["mesures"]["FFT_err"] - 1e-6


def copy(source, name):
    """Every file of a target, from the folder of the kept version."""
    for f in (f"{name}.json", f"{name}_profil.npy", f"entrees_ramp/{name}.py",
              f"planches/{name}.png", f"planches/{name}.pdf", f"finalistes/{name}.json"):
        if (source / f).exists():
            shutil.copy2(source / f, OUT / f)


# Kept version of every target, and its row in the summary table
rows = []
for path in sorted(REFERENCE.glob("*.json")):
    name = path.stem
    kept, p = REFERENCE, json.load(open(path))
    for run in RUNS:
        candidate = run / f"{name}.json"
        if candidate.exists():
            q = json.load(open(candidate))
            if gain(p, q):
                kept, p = run, q
    copy(kept, name)
    rows.append({"cible": name, "source": kept.name, "verdict": p["verdict"],
                 "n_manques": len(p["manques"]), "manques": " ".join(p["manques"]),
                 "NRMSE": round(p["mesures"]["NRMSE"], 3),
                 "FFT_err": round(p["mesures"]["FFT_err"], 3),
                 "err_E_pct": round(p["mesures"]["err_E_pct"], 1),
                 "ECART_PICS_h": p["mesures"]["ECART_PICS_h"],
                 "ELM_1h": round(p["locale"]["ELM_1h"], 2),
                 "L_etoile": p["L_etoile"], "func_cycle": p["func_cycle"]})

# Summary table and key figures of the final reference
summary = pd.DataFrame(rows)
summary.to_csv(OUT / "bilan_reference.csv", index=False)
n = len(summary)
print(summary.source.value_counts().to_string())
print("criteria met %d / %d" % (10 * n - summary.n_manques.sum(), 10 * n))
print("NRMSE mean %.3f median %.3f max %.3f | %d targets > 10 %%"
      % (summary.NRMSE.mean(), summary.NRMSE.median(), summary.NRMSE.max(), (summary.NRMSE > 0.10).sum()))
print("FFT mean %.3f median %.3f max %.3f | %d targets <= 10 %%"
      % (summary.FFT_err.mean(), summary.FFT_err.median(), summary.FFT_err.max(), (summary.FFT_err <= 0.10).sum()))
print("energy |mean| %.1f %% max %.1f %%" % (summary.err_E_pct.abs().mean(), summary.err_E_pct.abs().max()))
print(summary.verdict.value_counts().to_string())
