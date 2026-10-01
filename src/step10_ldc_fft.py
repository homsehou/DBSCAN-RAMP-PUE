"""Step 10: load duration curves and harmonics of the retained models, against the measured profiles.

Same definitions as the working plates (plates.py):
  load duration curve = mean profile sorted by decreasing power;
  harmonics = amplitudes of the first ten daily harmonics, 2 |FFT| / 96.
Read-only use of the committed results.

Usage: python step10_ldc_fft.py
Outputs in resultats/courbes/ (PNG only):
  ldc_saison.png, fft_saison.png            the 36 seasonal targets, one panel per target
  ldc_mois_<client>.png, fft_mois_<client>.png   the monthly targets of each client
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import settings as S
import publication_figures as F

OUT = S.RESULTS / "courbes"
HOURS = np.arange(S.SLOTS_PER_DAY) / 4
MONTH_ABBR = {"Janvier": "janv.", "Fevrier": "févr.", "Mars": "mars", "Avril": "avril", "Mai": "mai",
              "Juin": "juin", "Juillet": "juil.", "Aout": "août", "Septembre": "sept.", "Octobre": "oct.",
              "Novembre": "nov.", "Decembre": "déc."}


def amplitudes(y):
    """Amplitudes of the first ten daily harmonics of a profile."""
    return 2 * np.abs(np.fft.rfft(y))[1:11] / len(y)


def short_title(target):
    """Short period name for a panel title, for instance 'sèche, av. 09/2025'."""
    t = target.replace(" | ", " ").replace("Saison des pluies", "pluies").replace("Saison seche", "sèche")
    t = t.replace("_active", "").replace(" active", "").replace("_", " ")
    for month, abbr in MONTH_ABBR.items():
        t = t.replace(month, abbr)
    for word, abbr in (("avant", "av."), ("depuis", "dep.")):
        if f" {word} " in t:
            start, date = t.split(f" {word} ")
            t = f"{start}, {abbr} {date[5:7]}/{date[:4]}"
    return t


def pairs(folder, grain):
    """Measured target profile and retained model profile, one row of the summary table per target."""
    table = pd.read_csv(S.RESULTS / folder / "publication" / "tableaux" / "bilan_complet.csv")
    for _, row in table.iterrows():
        measured = np.load(S.TARGETS / f"bruit_{grain}" / f"{row.nom}.npz")["cible"]
        model = np.load(S.RESULTS / folder / f"{row.nom}_profil.npy")
        yield row, np.asarray(measured, float), model


def draw_ldc(ax, measured, model):
    """Measured load duration curve (area) and simulated one (line)."""
    ax.fill_between(HOURS, np.sort(measured)[::-1], color=F.AREA, lw=0, step="post")
    ax.plot(HOURS, np.sort(measured)[::-1], color=F.MEASURED, lw=0.8, drawstyle="steps-post")
    ax.plot(HOURS, np.sort(model)[::-1], color=F.BLUE, lw=1.0, drawstyle="steps-post")
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 12, 24])
    ax.set_ylim(0, None)


def draw_fft(ax, measured, model):
    """Amplitudes of the first ten harmonics: measure as bars, model as dots."""
    rank = np.arange(1, 11)
    ax.bar(rank, amplitudes(measured), color=F.AREA, edgecolor=F.MEASURED, lw=0.4, width=0.75)
    ax.plot(rank, amplitudes(model), "o", color=F.BLUE, ms=2.4)
    ax.set_xticks([1, 5, 10])
    ax.set_ylim(0, None)


def grid(panels, draw, xlabel, name, ncol=6, height=1.05):
    """Small multiples, one panel per target, common legend at the bottom."""
    nrow = int(np.ceil(len(panels) / ncol))
    with plt.rc_context(F.STYLE):
        fig, axes = plt.subplots(nrow, ncol, figsize=(F.DOUBLE, height * nrow + 0.5), layout="constrained",
                                 squeeze=False)
        for ax, (title, measured, model, label) in zip(axes.flat, panels):
            draw(ax, measured, model)
            ax.set_title(title, loc="left", fontsize=6.5)
            ax.text(0.97, 0.93, label, transform=ax.transAxes, ha="right", va="top", fontsize=6)
            ax.tick_params(labelsize=6)
        for ax in list(axes.flat)[len(panels):]:
            ax.set_axis_off()
        for ax in axes[-1]:
            ax.set_xlabel(xlabel, fontsize=6.5)
        for ax in axes[:, 0]:
            ax.set_ylabel("W", fontsize=6.5)
        measure_key = plt.Rectangle((0, 0), 1, 1, color=F.AREA, ec=F.MEASURED, lw=0.5)
        if draw is draw_ldc:
            model_key = plt.Line2D([], [], color=F.BLUE, lw=1.2)
        else:
            model_key = plt.Line2D([], [], color=F.BLUE, marker="o", ms=3, ls="none")
        fig.legend([measure_key, model_key], ["mesure", "modèle RAMP"], loc="outside lower center", ncol=2,
                   fontsize=7, frameon=False)
        fig.savefig(OUT / f"{name}.png", dpi=200)
        plt.close(fig)
    print("figure", name, flush=True)


# Two kinds of figure: load duration curve and harmonics
KINDS = ((draw_ldc, "LDC_err", "durée cumulée (h)", "ldc"),
         (draw_fft, "FFT_err", "rang de l'harmonique", "fft"))
OUT.mkdir(exist_ok=True)

# Seasons: the 36 targets, sorted by family and then by client
families = list(F.FAMILY)
seasons = sorted(pairs("reference_saison", "saison"), key=lambda p: (families.index(p[0].famille), p[0].nom))
for draw, key, xlabel, kind in KINDS:
    panels = [(f"{r.client} {F.FAMILY[r.famille]}\n{short_title(r.cible)}", y, sim,
               f"{key[:3]} {F.comma(100 * r[key])} %") for r, y, sim in seasons]
    grid(panels, draw, xlabel, f"{kind}_saison", height=1.3)

# Months: one figure per client, in calendar order
months = list(pairs("calib_mois", "mois"))
for client in sorted({r.client for r, _, _ in months}):
    mine = sorted([p for p in months if p[0].client == client], key=lambda p: S.MONTHS.index(p[0].cible.split(" ")[0]))
    for draw, key, xlabel, kind in KINDS:
        panels = [(short_title(r.cible), y, sim, f"{key[:3]} {F.comma(100 * r[key])} %") for r, y, sim in mine]
        grid(panels, draw, xlabel, f"{kind}_mois_{client}", height=1.0)
