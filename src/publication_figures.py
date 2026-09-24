"""Publication figures, distinct from the working plates of plates.py (target-by-target check).

Differences with the working plates:
  - width set to the double column of journals (7.16 inches), 8 pt font at final size, vector
    PDF with embedded fonts and PNG at 600 dpi;
  - no green and red banner: the criteria go into a table, the figure shows the data;
  - colour-blind safe colours (Okabe-Ito), measurement in neutral grey and model in blue,
    never colour alone to carry information;
  - residuals relative to the standard deviation of the noise, so comparable between targets.
The text of the figures stays in French, as in the published results and the method guides.

Figures:
  figure_target(p, noise)            profile and residuals of one target
  figure_multiples(ps, noises)       several targets on one plate, same visual grammar
  figure_criteria(summary)           every target, each gap relative to its threshold
  figure_floor(targets)              floor of the perfect model against the number of days
  figure_observed_floor(summary)     observed gap against floor, per criterion
  figure_parameters(ps, title)       retained RAMP parameters, per season
  figure_transfer(variance, levels)  what transfers from one appliance to another
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator
import settings as S

# Standard journal widths, in inches
COLUMN, DOUBLE = 3.46, 7.16

# Okabe-Ito colours: enough separation for colour-vision deficiencies
BLUE, ORANGE, GREEN, VERMILION, PINK = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"
MEASURED, AREA, NOISE, SHADE = "#3D3D3D", "#E8E8E6", "#C9C9C6", "#9A9A96"
CYCLES = [BLUE, ORANGE, GREEN]

STYLE = {
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "axes.labelpad": 3.0,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.4, "grid.alpha": 1.0,
    "axes.axisbelow": True, "legend.frameon": False, "legend.handlelength": 1.8,
    "legend.borderpad": 0.0, "legend.columnspacing": 1.2, "legend.handletextpad": 0.6,
    "lines.solid_capstyle": "round", "figure.dpi": 150, "savefig.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}
NAMES = {"NRMSE": "NRMSE", "LDC_err": "Courbe classée", "FFT_err": "Harmoniques",
         "err_E_pct": "Énergie", "err_P_pct": "Pointe", "err_LF": "Facteur de charge",
         "CORR_h": "Corrélation horaire", "TVD_h": "Distance horaire",
         "EXT_prof": "Étendue du profil", "ECART_PICS_h": "Heure des pics",
         "ELM_1h": "Forme locale 1 h", "ELM_2h": "Forme locale 2 h",
         "PEL_1h": "Pire écart lissé 1 h"}
FAMILY = {"cold_chain": "froid", "grain_milling": "moulin", "poultry_incubation": "couveuse"}
SHORT = {"Saison seche": "sèche", "Mai": "mai", "Saison des pluies": "pluies", "Octobre": "octobre"}


def period_rank(name):
    """Chronological rank of a season or a month, to sort the targets.

    May and October are both months and seasons: they keep their month rank, and the two long
    seasons fit in between (dry season first, rainy season after May)."""
    if name == "Saison seche":
        return -1
    if name == "Saison des pluies":
        return 4.5
    return S.MONTHS.index(name) if name in S.MONTHS else 99


def comma(x, d=1):
    """Number written the French way, with the true minus sign."""
    return f"{x:.{d}f}".replace(".", ",").replace("-", "−")


def target_title(p):
    """Short title of a target: client, use, season, configuration."""
    period, _, config = p["cible"].partition(" | ")
    suffix = f" ({config.replace('avant', 'avant le').replace('depuis', 'depuis le')})" if config else ""
    return f"{p['client']}, {FAMILY[p['famille']]}, {SHORT.get(period, period.lower())}{suffix}"


def step(y):
    """One step per quarter of an hour, extended up to 24 h."""
    return np.r_[y, y[-1]]


def hour_axis(ax, every=3):
    """Axis of the hours, from 0 to 24, without crowded ticks."""
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, every))
    ax.set_xticklabels([f"{h} h" for h in range(0, 25, every)])


def _profile(ax, target, sim, noise, peaks, legend=True):
    """Measured profile, its noise band and the model, on an existing axis."""
    h = np.arange(97) / 4
    ax.fill_between(h, step(noise["bas"]), step(noise["haut"]), step="post",
                    color=NOISE, lw=0, zorder=1)
    ax.fill_between(h, 0, step(target), step="post", color=AREA, lw=0, zorder=0)
    ax.step(h, step(target), where="post", color=MEASURED, lw=1.2, zorder=3, label="mesure")
    ax.step(h, step(sim), where="post", color=BLUE, lw=1.4, zorder=4, label="modèle RAMP")
    # Top margin kept for the peak marks, so that they are never cut
    top = max(target.max(), noise["haut"].max(), sim.max(), 1e-9)
    upper = top * (1.16 if len(peaks) else 1.06)
    if len(peaks):
        ax.plot(peaks / 4 + 0.125, np.full(len(peaks), 1.07 * top), marker="v", ls="none",
                color=VERMILION, ms=3.5, zorder=5, clip_on=False, label="pic marqué")
    ax.set_ylim(0, upper)
    hour_axis(ax)
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: comma(v, 0) if (v >= 10 or float(v).is_integer()) else comma(v, 1)))
    if legend:
        ax.legend(loc="upper left", ncol=3, bbox_to_anchor=(0, 1.02), borderaxespad=0)


def _usage_band(ax, p):
    """Dedicated band: usage windows above, ranges of each cycle below."""
    for w in p["fenetres"]:
        ax.add_patch(plt.Rectangle((w[0] / 60, 0.58), (w[1] - w[0]) / 60, 0.34, color=SHADE, lw=0))
    for k, cy in enumerate(p["cycles"]):
        for w in cy["plages"]:
            ax.add_patch(plt.Rectangle((w[0] / 60, 0.08), (w[1] - w[0]) / 60, 0.34,
                                       color=CYCLES[k], lw=0))
    ax.set_ylim(0, 1)
    ax.set_yticks([0.75, 0.25], ["fenêtres", "cycles"])
    ax.tick_params(length=0, labelsize=7)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def figure_target(p, noise, height=4.6):
    """One target: profile and model, usage band, then residual relative to the noise."""
    target, sim = np.asarray(noise["cible"], float), np.asarray(p["profil"], float)
    with plt.rc_context(STYLE):
        fig, (a, u, r) = plt.subplots(3, 1, figsize=(DOUBLE, height),
                                      height_ratios=[2.4, 0.34, 1], sharex=True,
                                      layout="constrained")
        _profile(a, target, sim, noise, np.asarray(noise["pics"], int))
        a.set_ylabel("puissance (W)")
        a.set_title(target_title(p) + f", {p['jours_retenus']} journées", loc="left", pad=14)
        _usage_band(u, p)

        # Residual relative to the standard deviation of the noise: comparable between targets
        h = np.arange(96) / 4
        z = (sim - target) / np.maximum(noise["sd_1h"], 1e-9)
        r.axhspan(-2, 2, color=NOISE, lw=0, zorder=0)
        r.axhspan(-1, 1, color="#E4E4E1", lw=0, zorder=1)
        r.bar(h, z, width=0.25, align="edge", lw=0, zorder=2,
              color=np.where(z >= 0, BLUE, VERMILION))
        r.axhline(0, color=MEASURED, lw=0.6, zorder=3)
        lim = max(3.2, 1.1 * np.abs(z).max())
        r.set_ylim(-lim, lim)
        r.set_ylabel("écart / bruit (σ)")
        r.set_xlabel("heure de la journée")
        r.text(0.004, 0.96, "bandes ± 1 σ et ± 2 σ du bruit d'échantillonnage", fontsize=7,
               color="#6A6A66", va="top", ha="left", transform=r.transAxes,
               bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0))
    return fig


def short_title(p):
    """Title of a panel when the client is already named: period and fleet only."""
    period, _, config = p["cible"].partition(" | ")
    name = SHORT.get(period, period.lower()).replace("fevrier", "février").replace("aout", "août")
    name = name.replace("decembre", "décembre")
    return name + (" (" + config.split(" ")[0] + ")" if config else "")


def figure_multiples(ps, noises, columns=2, row_height=1.55, title=None):
    """Several targets on one plate, profiles only, same visual grammar.

    title: when given, overall title of the plate and short titles of the panels."""
    n = len(ps)
    rows = int(np.ceil(n / columns))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(rows, columns, figsize=(DOUBLE, row_height * rows),
                                 layout="constrained", squeeze=False)
        for i, (p, noise) in enumerate(zip(ps, noises)):
            ax = axes[i // columns][i % columns]
            target, sim = np.asarray(noise["cible"], float), np.asarray(p["profil"], float)
            _profile(ax, target, sim, noise, np.asarray(noise["pics"], int), legend=False)
            ax.set_title(short_title(p) if title else target_title(p), loc="left", fontsize=7.5, pad=2)
            ax.tick_params(labelsize=7)
            if i % columns == 0:
                ax.set_ylabel("puissance (W)")
            if i // columns == rows - 1:
                ax.set_xlabel("heure de la journée")
            hour_axis(ax, every=6)
        for j in range(n, rows * columns):
            axes[j // columns][j % columns].set_axis_off()
        handles = [plt.Line2D([], [], color=MEASURED, lw=1.2), plt.Line2D([], [], color=BLUE, lw=1.4),
                   plt.Rectangle((0, 0), 1, 1, color=NOISE),
                   plt.Line2D([], [], color=VERMILION, marker="v", ls="none", ms=3.5)]
        fig.legend(handles, ["mesure", "modèle RAMP", "bruit de la moyenne (90 %)", "pic marqué"],
                   loc="outside lower center", ncol=4, fontsize=7.5)
        if title:
            fig.suptitle(title, fontsize=8.5, weight="bold", x=0.01, ha="left")
    return fig


MARKERS = {"cold_chain": ("o", BLUE), "grain_milling": ("s", ORANGE),
           "poultry_incubation": ("^", GREEN)}


def figure_criteria(summary, keys=None):
    """Each gap relative to its threshold: 1 = limit of the contract, logarithmic scale.

    summary: table with one row per target, the columns of the criteria and of the thresholds
    (`seuil_<k>`), plus `famille`.
    """
    keys = keys or list(NAMES)
    ratios = {k: (summary[k].abs() / summary[f"seuil_{k}"]).to_numpy() for k in keys}
    order = sorted(keys, key=lambda k: np.median(ratios[k]))
    fam = summary["famille"].to_numpy()
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(DOUBLE, 0.34 * len(keys) + 1.3), layout="constrained")
        ax.axvspan(1, 40, color="#F4F0EC", lw=0, zorder=0)
        ax.axvline(1, color=VERMILION, lw=1.0, zorder=1)
        for i, k in enumerate(order):
            v = np.clip(ratios[k], 0.025, 40)
            shift = np.linspace(-0.17, 0.17, len(v))
            for f, (marker, colour) in MARKERS.items():
                s = fam == f
                ax.plot(v[s], np.full(s.sum(), i) + shift[s], marker, ms=3.0, mfc=colour,
                        mec="white", mew=0.3, ls="none", alpha=0.8, zorder=2,
                        label=FAMILY[f] if i == 0 else None)
            ax.plot([np.median(ratios[k])], [i], "|", color=MEASURED, ms=13, mew=1.6, zorder=3)
            out = int((ratios[k] > 1).sum())
            ax.text(38, i, f"{out}", ha="right", va="center", fontsize=7.5,
                    color=VERMILION if out else "#8A8A86")
        ax.set_yticks(range(len(order)), [NAMES[k] for k in order])
        ax.set_xscale("log")
        ax.set_xlim(0.023, 45)
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda v, _: {0.1: "0,1", 1.0: "1", 10.0: "10"}.get(round(v, 3), "")))
        ax.set_xlabel("écart rapporté au seuil du contrat (1 = limite ; les écarts nuls sont "
                      "ramenés au bord gauche)")
        ax.set_ylim(-0.8, len(order) - 0.2)
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower right", ncol=3, bbox_to_anchor=(1, 1.0), borderaxespad=0)
        ax.set_title(f"{len(summary)} cibles, trait noir : médiane du critère ; "
                     "chiffre à droite : cibles hors seuil", loc="left", fontsize=7.5)
    return fig


def figure_floor(targets, nrmse_threshold=0.10, cap=2.0):
    """Floor of the perfect model against the number of days: the limit of verifiability."""
    j = targets["jours_retenus"].to_numpy(float)
    y = targets["p95_NRMSE"].to_numpy(float)
    fam = targets["famille"].to_numpy()
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(COLUMN * 1.55, 2.6), layout="constrained")
        ax.axhspan(cap * nrmse_threshold, 1.0, color="#F4F0EC", lw=0, zorder=0)
        ax.axhline(cap * nrmse_threshold, color=VERMILION, lw=1.0, zorder=1)
        ax.axhline(nrmse_threshold, color=MEASURED, lw=0.8, ls=(0, (4, 2)), zorder=1)
        for f, (marker, colour) in MARKERS.items():
            s = fam == f
            ax.plot(j[s], y[s], marker, ms=4, mfc=colour, mec="white", mew=0.4,
                    ls="none", label=FAMILY[f], zorder=2)
        ax.set_xscale("log")
        ax.set_xticks([10, 20, 50, 100, 200], ["10", "20", "50", "100", "200"])
        ax.set_xlabel("journées retenues dans la cible")
        ax.set_ylabel("plancher du NRMSE\n(modèle parfait, 95e centile)")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: comma(100 * v, 0) + " %"))
        ax.set_ylim(0, min(0.58, 1.14 * y.max()))
        ax.legend(loc="lower left", ncol=3, bbox_to_anchor=(0, 1.0), borderaxespad=0)
        ax.text(225, cap * nrmse_threshold + 0.008, "au-dessus : cible non vérifiable",
                fontsize=7, color=VERMILION, ha="right")
        ax.text(225, nrmse_threshold + 0.008, "seuil nominal du contrat", fontsize=7,
                color="#6A6A66", ha="right")
    return fig


UNIT = {"NRMSE": " (%)", "LDC_err": " (%)", "FFT_err": " (%)", "err_E_pct": " (%)",
        "err_P_pct": " (%)", "err_LF": " (%)", "CORR_h": " (%)", "TVD_h": " (%)",
        "EXT_prof": " (%)", "ECART_PICS_h": " (h)", "ELM_1h": " (σ)", "ELM_2h": " (σ)",
        "PEL_1h": " (%)"}


def _as_percent(k, v):
    """Values on a readable scale: percentage, or hours as they are."""
    if k == "ECART_PICS_h" or k in ("ELM_1h", "ELM_2h"):
        return v
    return v * (1.0 if k.endswith("_pct") else 100.0)


def figure_observed_floor(summary, keys=("NRMSE", "FFT_err", "err_P_pct", "ECART_PICS_h")):
    """Observed gap against the floor of the perfect model, one panel per criterion."""
    fam = summary["famille"].to_numpy()
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(keys), figsize=(DOUBLE, 2.3), layout="constrained")
        for i, (ax, k) in enumerate(zip(np.atleast_1d(axes), keys)):
            x = _as_percent(k, summary[f"p95_{k}"].abs().to_numpy(float))
            y = _as_percent(k, summary[k].abs().to_numpy(float))
            top = 1.12 * max(x.max(), y.max())
            ax.plot([0, top], [0, top], color=MEASURED, lw=0.8, ls=(0, (4, 2)), zorder=1)
            for f, (marker, colour) in MARKERS.items():
                s = fam == f
                ax.plot(x[s], y[s], marker, ms=3.2, mfc=colour, mec="white", mew=0.3,
                        ls="none", zorder=2, label=FAMILY[f] if i == 0 else None)
            ax.set_xlim(0, top)
            ax.set_ylim(0, top)
            ax.set_aspect("equal")
            ax.set_title(NAMES[k] + UNIT[k], loc="left", fontsize=7.5)
            ax.set_xlabel("plancher")
            ax.tick_params(labelsize=7)
            below = int((y < x).sum())
            ax.text(0.03 * top, 0.97 * top, f"{below} / {len(y)} sous\nle plancher",
                    ha="left", va="top", fontsize=7, color="#6A6A66")
        a0 = np.atleast_1d(axes)[0]
        a0.set_ylabel("écart du modèle")
        fig.legend(loc="outside lower center", ncol=3, fontsize=7.5)
        fig.suptitle("sous la diagonale : le modèle s'écarte moins de la cible que le bruit "
                     "d'échantillonnage", fontsize=7.5, x=0.01, ha="left")
    return fig


def figure_powers(ps, ax):
    """Nameplate entered in RAMP, measured and simulated daily peak, running power."""
    periods = [SHORT.get(s, s.lower()) for s in (p["cible"].partition(" | ")[0] for p in ps)]
    x = np.arange(len(ps))
    plate = sum(ps[0]["plaques_appareils"])
    ax.axhline(plate, color=MEASURED, lw=1.0, ls=(0, (4, 2)), zorder=1)
    ax.text(len(ps) - 0.5, plate, f" plaque\n {comma(plate, 0)} W", fontsize=6.5,
            va="center", ha="left", color=MEASURED)
    ax.bar(x - 0.19, [p["realisme"]["pointe_P95_mes"] for p in ps], width=0.36, color=MEASURED,
           lw=0, label="pointe mesurée")
    ax.bar(x + 0.19, [p["realisme"]["pointe_P95_sim"] for p in ps], width=0.36, color=BLUE,
           lw=0, label="pointe du modèle")
    ax.plot(x, [p["marche_mesuree_W"] for p in ps], "_", color=VERMILION, ms=11, mew=1.6,
            ls="none", zorder=3, label="marche mesurée")
    ax.set_xticks(x, periods)
    ax.set_ylim(0, 1.55 * plate)
    ax.set_ylabel("puissance (W)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=1, loc="upper right", fontsize=6.5, borderaxespad=0.3)
    ax.set_title("Puissances, 95e centile journalier", loc="left", fontsize=7.5, pad=3)


def figure_parameters(ps, title):
    """Retained RAMP parameters of a client: windows, duty cycle, durations."""
    ps = sorted(ps, key=lambda p: period_rank(p["cible"].partition(" | ")[0]))
    periods = [SHORT.get(s, s.lower()) for s in (p["cible"].partition(" | ")[0] for p in ps)]
    with plt.rc_context(STYLE):
        fig, ((a, d), (b, c)) = plt.subplots(2, 2, figsize=(DOUBLE, 4.0), layout="constrained",
                                             width_ratios=[1.5, 1])
        figure_powers(ps, d)
        for j, p in enumerate(ps):
            y = len(ps) - 1 - j
            for w in p["fenetres"]:
                a.add_patch(plt.Rectangle((w[0] / 60, y + 0.10), (w[1] - w[0]) / 60, 0.16,
                                          color=SHADE, lw=0))
            for k, cy in enumerate(p["cycles"]):
                for w in cy["plages"]:
                    a.add_patch(plt.Rectangle((w[0] / 60, y - 0.26), (w[1] - w[0]) / 60, 0.30,
                                              color=CYCLES[k], lw=0))
        a.set_yticks(range(len(ps)), periods[::-1])
        a.set_ylim(-0.6, len(ps) - 0.4)
        hour_axis(a, every=6)
        a.grid(axis="y", visible=False)
        a.set_title("Fenêtres (gris) et plages des cycles", loc="left", fontsize=7.5)

        width = 0.26
        for k in range(3):
            d = [p["cycles"][k]["rapport_cyclique"] if k < len(p["cycles"]) else np.nan for p in ps]
            b.bar(np.arange(len(ps)) + (k - 1) * width, d, width=width, color=CYCLES[k], lw=0,
                  label=f"cycle {k + 1}")
        b.set_xticks(range(len(ps)), periods)
        b.set_ylabel("rapport cyclique")
        b.set_ylim(0, 1.32)
        b.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0,25", "0,50", "0,75", "1"])
        b.grid(axis="x", visible=False)
        b.legend(ncol=3, loc="upper center", fontsize=6.5, borderaxespad=0.3,
                 columnspacing=0.8, handlelength=1.2)
        b.set_title("Part du temps en marche", loc="left", fontsize=7.5, pad=3)

        c.bar(np.arange(len(ps)) - 0.18, [p["func_cycle"] for p in ps], width=0.36,
              color=BLUE, lw=0, label="durée de marche")
        c.bar(np.arange(len(ps)) + 0.18, [p["L_etoile"] for p in ps], width=0.36,
              color=ORANGE, lw=0, label="période du cycle")
        for j, p in enumerate(ps):
            c.text(j, max(p["func_cycle"], p["L_etoile"]) * 1.05,
                   f"{comma(100 * p['func_time'] / 1440, 0)} %",
                   ha="center", va="bottom", fontsize=6.5, color="#6A6A66")
        c.set_xticks(range(len(ps)), periods)
        c.set_ylabel("minutes")
        c.set_ylim(0, 1.75 * max(max(p["func_cycle"], p["L_etoile"]) for p in ps))
        c.grid(axis="x", visible=False)
        c.legend(ncol=1, loc="upper right", fontsize=6.5, borderaxespad=0.3)
        c.set_title("Durées, et func_time en gris (part de la journée)", loc="left",
                    fontsize=7.5, pad=3)
        fig.suptitle(title, fontsize=8.5, weight="bold", x=0.01, ha="left")
    return fig


def figure_transfer(variance, levels):
    """What transfers from one appliance to another.

    variance: share of variance carried by the client, per parameter and per family
    (columns `parametre`, `porte par`, then one column per family).
    levels: one row per target, columns `famille`, `client`, `heures_eq`.
    """
    families = [c for c in variance.columns if c not in ("parametre", "porte par")]
    order = variance.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(order))
    with plt.rc_context(STYLE):
        fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 3.2), layout="constrained",
                                   width_ratios=[1.7, 1])
        a.axvspan(0.5, 1.0, color="#F4F0EC", lw=0, zorder=0)
        a.axvline(0.5, color=VERMILION, lw=1.0, zorder=1)
        width = 0.36
        for i, fam in enumerate(families):
            v = order[fam].astype(str).str.replace(",", ".").astype(float)
            a.barh(y + (i - 0.5) * width, v, height=width, lw=0, zorder=2,
                   color=(BLUE if i == 0 else ORANGE), label=fam)
        a.set_yticks(y, order.parametre, fontsize=7)
        a.set_xlim(0, 1)
        a.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "0,25", "0,50", "0,75", "1"])
        a.set_xlabel("part de variance portée par le client")
        a.set_ylim(-0.6, len(order) - 0.4)
        a.grid(axis="y", visible=False)
        a.legend(ncol=1, loc="upper right", fontsize=7, borderaxespad=0.4)
        a.set_title("à gauche du trait : propriété de la machine ; à droite : de l'exploitant",
                    loc="left", fontsize=7.5, pad=3)

        for i, (fam, g) in enumerate(levels.groupby("famille")):
            marker, colour = MARKERS.get(g.famille.iloc[0], ("o", BLUE))
            for j, (client, h) in enumerate(g.groupby("client")):
                v = h.heures_eq.to_numpy()
                b.plot(v, np.full(len(v), i) + np.linspace(-0.18, 0.18, len(v)), marker, ms=3.2,
                       mfc=colour, mec="white", mew=0.3, ls="none", zorder=2)
            b.plot([g.heures_eq.min(), g.heures_eq.max()], [i, i], color="#BFBFBB", lw=0.8, zorder=1)
        b.set_yticks(range(levels.famille.nunique()),
                     [FAMILY.get(f, f) for f in sorted(levels.famille.unique())])
        b.set_xscale("log")
        b.set_xlabel("heures équivalentes par jour")
        b.set_xticks([0.05, 0.5, 5], ["0,05", "0,5", "5"])
        b.grid(axis="y", visible=False)
        b.set_ylim(-0.7, levels.famille.nunique() - 0.3)
        b.set_title("Niveau d'usage par cible", loc="left", fontsize=7.5, pad=3)
    return fig


def save(fig, path):
    """Vector PDF with embedded fonts and PNG at 600 dpi, then the figure is closed."""
    fig.savefig(str(path) + ".pdf")
    fig.savefig(str(path) + ".png")
    plt.close(fig)


def markdown(t):
    """Markdown table written without extra dependency: a header row, a separator row."""
    columns = [str(c) for c in t.columns]
    rows = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, r in t.iterrows():
        rows.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(rows)


def write_table(t, folder, name, title):
    """One table in Markdown and in LaTeX booktabs, ready to insert."""
    (folder / f"{name}.md").write_text(f"**{title}**\n\n" + markdown(t) + "\n")
    (folder / f"{name}.tex").write_text(
        t.to_latex(index=False, escape=True, caption=title, label=f"tab:{name}", position="htbp"))
