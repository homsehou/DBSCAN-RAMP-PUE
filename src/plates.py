"""Calibration plate of one target: criteria banner, profile, windows and ranges, gap against
the noise of the mean, load duration curve and harmonics.

p is the result of one target (JSON written by step 4): client, target, windows, cycles,
nameplates, global settings, criteria, local shape, p95, verdict, realism, and the simulated
profile (key "profil"). noise is the file bruit_<grain>/<target>.npz of step 2 (target, 90 %
band, peaks).

The text of the figures stays in French, as in the published results and the method guides.
Format: 10 x 10.4 inches, PNG at 150 dpi and PDF; fonts from 11 to 15 pt; decimal comma.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import settings as S
import validation as V

STYLE = {"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 13,
         "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
         "figure.titlesize": 15, "axes.spines.top": False, "axes.spines.right": False,
         "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.6,
         "figure.dpi": 100, "savefig.dpi": 150}
REAL, MODEL, BAND = "#e8735a", "#111111", "#bdbdbd"
OK, KO, INFO = "#d9efd5", "#f6d3cf", "#ececec"
WINDOW_GREEN, CYCLE_COLOURS = "#7fb77e", ["#2a78d6", "#eda100", "#8e5bb5"]
NAMES = {"NRMSE": "NRMSE", "LDC_err": "LDC", "FFT_err": "FFT", "err_E_pct": "Énergie",
         "err_P_pct": "Pointe", "err_LF": "Fact. de charge", "CORR_h": "CORR_h",
         "TVD_h": "TVD_h", "EXT_prof": "EXT_prof", "ECART_PICS_h": "Heure des pics",
         "ELM_1h": "Forme 1 h", "ELM_2h": "Forme 2 h", "PEL_1h": "Pire écart 1 h"}
LABEL = {"Saison seche": "Sèche", "Mai": "Mai", "Saison des pluies": "Pluies", "Octobre": "Octobre"}
FAMILY = {"cold_chain": "froid", "grain_milling": "moulin à grains",
          "poultry_incubation": "couveuse"}
VERDICT = {"validee": ("validée", OK), "rejetee": ("rejetée", KO),
           "non verifiable": ("non vérifiable", "#fde9b8")}


def comma(x, decimals=1):
    """Number written the French way: 6,7 rather than 6.7."""
    return f"{x:.{decimals}f}".replace(".", ",").replace("-", "−")


def watts(q):
    """Power written the French way: 7 350 W."""
    return f"{q:,.0f}".replace(",", " ") + " W"


def unit(k):
    """Display factor and unit: hours, standard deviations (local shape) or percentage."""
    if k == "ECART_PICS_h":
        return 1.0, " h"
    if k in ("ELM_1h", "ELM_2h"):
        return 1.0, " σ"
    return (1.0, " %") if k.endswith("pct") else (100.0, " %")


def box(ax, x, y, title, text, colour):
    """One box of the banner: name in bold, value below."""
    ax.add_patch(plt.Rectangle((x + 0.02, y + 0.05), 0.96, 0.9, color=colour, lw=0))
    ax.text(x + 0.5, y + 0.68, title, ha="center", va="center", fontsize=12, weight="bold")
    ax.text(x + 0.5, y + 0.30, text, ha="center", va="center", fontsize=12)


def banner(ax, p):
    """Three rows of five boxes: ten criteria, local shape, verdict and daily peak."""
    ax.set_axis_off()
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 3)
    limits = {k: V.threshold(k, p["p95"]) for k in V.THRESHOLDS}
    limits.update(V.local_thresholds(p["p95"]))
    values = {**p["mesures"], **p["locale"]}
    for i, k in enumerate(list(V.THRESHOLDS) + ["ELM_1h", "ELM_2h", "PEL_1h"]):
        factor, u = unit(k)
        met = abs(values[k]) <= limits[k]
        sign = "+" if k in ("err_E_pct", "err_P_pct") and values[k] > 0 else ""
        box(ax, i % 5, 2 - i // 5, NAMES[k], f"{sign}{comma(values[k] * factor)}{u} "
            f"{'≤' if met else '>'} {comma(limits[k] * factor)}", OK if met else KO)
    text, colour = VERDICT[p["verdict"]]
    box(ax, 3, 0, "Verdict", text, colour)
    r = p.get("realisme", {})
    if r:
        box(ax, 4, 0, "Pointe jour. P95", f"{watts(r['pointe_P95_sim'])[:-2]} / {watts(r['pointe_P95_mes'])}", INFO)


def step(y):
    """One step per quarter of an hour, up to 24 h."""
    return np.r_[y, y[-1]]


def plate(p, noise):
    """Plate of one target, readable on screen (1000 px) as on a full thesis page."""
    target, sim = np.asarray(noise["cible"], float), np.asarray(p["profil"], float)
    peaks = np.asarray(noise["pics"], int)
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(10, 10.4), layout="constrained")
        gs = fig.add_gridspec(5, 2, height_ratios=[1.6, 3.0, 0.8, 1.2, 1.9])
        period, _, config = p["cible"].partition(" | ")
        met = sum(abs(p["mesures"][k]) <= V.threshold(k, p["p95"]) for k in V.THRESHOLDS)
        plates = " + ".join(watts(q) for q in p["plaques_appareils"])
        fig.suptitle(f"{p['client']}, {FAMILY[S.CLIENTS[p['client']]]}, {LABEL.get(period, period).lower()}"
                     + (f" ({config})" if config else "")
                     + f"\n{p['jours_retenus']} journées, plaque {plates}, "
                     f"{met} critères tenus sur 10", weight="bold", fontsize=15)
        banner(fig.add_subplot(gs[0, :]), p)

        # Profile: area of the mean measured profile, RAMP model dashed, marked peaks
        h = np.arange(97) / 4
        a = fig.add_subplot(gs[1, :])
        a.fill_between(h, 0, step(target), step="post", color=REAL, alpha=0.35, lw=0)
        a.step(h, step(target), where="post", color=REAL, lw=2.0, label="profil réel moyen")
        a.step(h, step(sim), where="post", color=MODEL, lw=2.4, ls=(0, (5, 3)), label="modèle RAMP")
        if len(peaks):
            a.plot(peaks / 4 + 0.125, target[peaks] * 1.05, "v", color="#6a3d9a", ms=10,
                   label="pics marqués de la cible")
        a.set_ylim(0, 1.18 * max(target.max(), sim.max(), 1e-6))
        a.set_xlim(0, 24)
        a.set_xticks(range(0, 25, 2))
        a.tick_params(labelbottom=False)
        a.set_ylabel("puissance (W)")
        a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: comma(v, 0 if float(v).is_integer() else 1)))
        a.legend(loc="lower left", bbox_to_anchor=(0, 1.0, 1, 0.1), mode="expand", ncol=3,
                 frameon=False, handlelength=2.4, borderaxespad=0.2)

        # Usage windows and ranges of each cycle, each on its own band
        g = fig.add_subplot(gs[2, :], sharex=a)
        n = len(p["cycles"])
        for w in p["fenetres"]:
            g.barh(n, (w[1] - w[0]) / 60, left=w[0] / 60, height=0.7, color=WINDOW_GREEN)
        for k, cy in enumerate(p["cycles"]):
            for w in cy["plages"]:
                g.barh(n - 1 - k, (w[1] - w[0]) / 60, left=w[0] / 60, height=0.7, color=CYCLE_COLOURS[k])
        g.set_yticks(range(n, -1, -1), ["fenêtres"] + [f"cycle {k + 1}" for k in range(n)])
        g.set_ylim(-0.6, n + 0.6)
        g.grid(axis="y", visible=False)
        g.tick_params(labelbottom=False, labelsize=11)

        # Model minus measured, against the 90 % band of the noise of the mean
        r = fig.add_subplot(gs[3, :], sharex=a)
        res = sim - target
        r.fill_between(h, step(noise["bas"] - target), step(noise["haut"] - target), step="post",
                       color=BAND, alpha=0.6, lw=0)
        r.bar(np.arange(96) / 4, res, width=0.25, align="edge",
              color=np.where(res >= 0, "#3b6ea8", "#c0504d"), alpha=0.9)
        r.axhline(0, color="black", lw=0.8)
        r.set_title("Écart modèle − réel : bleu au-dessus, rouge en dessous, gris = bruit de la moyenne (90 %)",
                    loc="left", fontsize=12)
        r.set_ylabel("écart (W)")
        r.set_xlabel("heure de la journée (h)")

        # Load duration curve, in cumulated hours
        c = fig.add_subplot(gs[4, 0])
        hours = np.arange(1, 97) / 4
        c.plot(hours, np.sort(target)[::-1], color=REAL, lw=2.0, label="réel")
        c.plot(hours, np.sort(sim)[::-1], color=MODEL, lw=2.2, ls=(0, (5, 3)), label="modèle")
        c.set_title(f"Courbe classée, LDC {comma(100 * p['mesures']['LDC_err'])} %", loc="left")
        c.set_xlabel("durée cumulée (h)")
        c.set_ylabel("puissance (W)")
        c.set_xlim(0, 24)
        c.set_xticks(range(0, 25, 4))
        c.legend(frameon=False)

        # Daily harmonics, as amplitudes in watts
        d = fig.add_subplot(gs[4, 1])
        k = np.arange(1, 11)
        amp = lambda y: 2 * np.abs(np.fft.rfft(y))[1:11] / len(y)
        d.bar(k - 0.19, amp(target), width=0.38, color=REAL, label="réel")
        d.bar(k + 0.19, amp(sim), width=0.38, color="0.3", label="modèle")
        d.set_title(f"Harmoniques, FFT {comma(100 * p['mesures']['FFT_err'])} %", loc="left")
        d.set_xlabel("rang (période 24 h / rang)")
        d.set_ylabel("amplitude (W)")
        d.set_xticks(k)
        d.grid(axis="x", visible=False)
        d.legend(frameon=False)
    return fig


def save(fig, path):
    """PNG at 150 dpi and vector PDF, then the figure is closed."""
    fig.savefig(str(path) + ".png", dpi=150)
    fig.set_layout_engine("none")          # layout already computed at the first export
    fig.savefig(str(path) + ".pdf")
    plt.close(fig)
