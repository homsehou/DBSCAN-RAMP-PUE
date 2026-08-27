# -*- coding: utf-8 -*-
"""Step 3 - Clustering figures: the twelve visual diagnostics of steps
1-2 (season profiles 07/07b, k-distance 08, cluster histogram/profiles,
representative profiles 09 + baseline, feature scatters 12-14, season x
cluster bar 15). Self-contained; run alone with:
    PUE_TYPE=cold_chain PUE_CLIENT=0017SAM python step3_cluster_figures.py
"""
import calendar, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C   # shared settings of the repository
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

PUE_TYPE = os.environ.get("PUE_TYPE", "cold_chain")
CLIENT = os.environ.get("PUE_CLIENT", "0017SAM")
REPO = Path(__file__).resolve().parent.parent.parent
OUT = REPO / "resultats" / PUE_TYPE / CLIENT
FIG = OUT / "figures"
# Orange reserved for DBSCAN noise, never a cluster colour


def ccol(cid):
    return "#FF7F0E" if cid == -1 else C.PALETTE[int(cid) % 10]

def legend_right(ax, fs=8):
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=fs)
    ax.grid(alpha=0.2)

def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight", dpi=110)
    plt.close(fig)

def season_panels(daily, feats, mode, title, name):
    """One panel per season: mean profile (07), all days behind the mean
    (07b) or dominant-cluster mean with filled area (baseline)."""
    present = [s for s in C.SEASON_ORDER if s in feats["season"].values]
    fig, axes = plt.subplots(len(present), 1, sharex=True,
                             figsize=(10, 3.8 * len(present)))
    for ax, season in zip(np.atleast_1d(axes), present):
        pick = (feats["season"] == season).values
        label_extra = ""
        if mode == "baseline":            # dominant cluster of the season
            valid = feats["cluster"][pick & (feats["cluster"] != -1).values]
            dom = int(valid.value_counts().index[0]) if len(valid) else -1
            if dom != -1:
                pick &= (feats["cluster"] == dom).values
            label_extra = f" - cluster C{dom}"
        days = daily.values[pick]
        mean_p, color = np.nanmean(days, axis=0), C.SEASON_COLORS[season]
        months = ", ".join(calendar.month_abbr[m] for m in sorted(
            {d.month for d in feats.index[pick]}))
        if mode == "stacked":
            ax.set_facecolor("#F8F8F8")
            for row in days:
                ax.plot(C.HOURS, row, lw=0.3, color=color, alpha=0.15)
        if mode == "baseline":
            ax.fill_between(C.HOURS, 0, mean_p, alpha=0.15, color=color)
        ax.plot(C.HOURS, mean_p, lw=2.6, color=color,
                label=f"{season}{label_extra} (n={len(days)})")
        ax.set_ylim(0, max(float(np.nanmax(days if mode == "stacked"
                                           else mean_p)), 10.0) * 1.12)
        ax.text(0.02, 0.94, f"{season}\n({months})\nAverage daily "
                f"consumption = {mean_p.sum() * 0.25 / 1000:.2f} kWh/day",
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                va="top", bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                    ec=color, alpha=0.9))
        ax.set_ylabel("Power [W]")
        legend_right(ax)
    np.atleast_1d(axes)[-1].set_xlabel("Hour [h]")
    fig.suptitle(title, fontweight="bold", y=1.01)
    save(fig, name)

def k_distance(daily, feats):
    """Reproduction of the hybrid representation and its epsilon rule."""
    P = daily.to_numpy(float)
    shape = P / np.maximum(P.max(axis=1)[:, None], 1e-9)
    pca = PCA(n_components=min(10, len(P) - 1)).fit(shape)
    keep = int(np.searchsorted(pca.explained_variance_ratio_.cumsum(),
                               0.90) + 1)
    Z = np.hstack([pca.transform(shape)[:, :keep],
                   StandardScaler().fit_transform(feats[C.FEATURES].values)])
    dist, _ = NearestNeighbors(n_neighbors=5).fit(Z).kneighbors(Z)
    k_dist, eps = np.sort(dist[:, -1]), float(np.percentile(dist[:, -1], 90))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_dist, color="steelblue", lw=1.5)
    ax.axhline(eps, color="red", ls="--", lw=1.5,
               label=f"eps (P90)={eps:.4f}")
    ax.set(xlabel="Days (sorted by 5-NN distance)", ylabel="5-NN distance",
           title=f"k-distance plot (k=5, n={len(P)})")
    legend_right(ax, 9)
    save(fig, "08_k_distance_plot.png")

def main():
    FIG.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(OUT / "daily_matrix_W.csv", index_col=0,
                        parse_dates=True)
    feats = pd.read_csv(OUT / "clustered_features.csv", index_col=0,
                        parse_dates=True)
    labels = feats["cluster"].to_numpy(int)
    season_panels(daily, feats, "mean", "Average profile by season",
                  "07_season_profiles.png")
    season_panels(daily, feats, "stacked", "Average daily profiles by "
                  "season", "07b_season_profiles_stacked.png")
    season_panels(daily, feats, "baseline", "Baseline seasonal "
                  "representative profiles",
                  "baseline_season_representative_profiles.png")
    k_distance(daily, feats)

    unique, counts = np.unique(labels, return_counts=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Noise" if c == -1 else f"C{c}" for c in unique], counts,
                  color=[ccol(c) for c in unique], edgecolor="k")
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"n={c}\n({100 * c / counts.sum():.1f}%)", ha="center",
                va="bottom", fontsize=10, fontweight="bold")
    ax.set(xlabel="Cluster", ylabel="Number of days",
           title=f"DBSCAN cluster distribution (total = {counts.sum()} days)")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.grid(alpha=0.2, axis="y")
    save(fig, "08_cluster_histogram.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    for lab in sorted(set(labels)):
        prof = np.nanmean(daily.values[labels == lab], axis=0)
        ax.plot(C.HOURS, prof, color=ccol(lab), lw=2.5,
                ls="--" if lab == -1 else "-",
                label=(f"Noise (n={(labels == lab).sum()})" if lab == -1
                       else f"Cluster {lab}  (n={(labels == lab).sum()})"))
    ax.set(xlabel="Hour [h]", ylabel="Power [W]",
           title="Cluster mean profiles (clean view)")
    legend_right(ax, 9)
    save(fig, "07_cluster_profiles.png")

    targets = pd.read_csv(OUT / "seasonal_representative_profiles.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    for _, r in targets.iterrows():
        ax.plot(C.HOURS, [r[f"slot_{j}"] for j in range(96)], lw=2.5,
                color=C.SEASON_COLORS[r["season"]],
                label=f"{r['season']} (n={r['n_days']})")
    ax.set(xlabel="Hour [h]", ylabel="Power [W]",
           title="Seasonal representative profiles (calibration targets)")
    legend_right(ax, 9)
    save(fig, "09_seasonal_repr_profiles.png")

    for num, x, y in (("12", "E_kWh", "P_peak_W"), ("13", "E_kWh",
                      "Load_factor"), ("14", "P_peak_W", "peak_sin")):
        fig, ax = plt.subplots(figsize=(7, 5))
        for cid in sorted(feats["cluster"].unique()):
            sub = feats[feats["cluster"] == cid]
            ax.scatter(sub[x], sub[y], c=ccol(cid), s=18, alpha=0.6,
                       edgecolor="k", linewidth=0.3,
                       label=(f"Noise (n={len(sub)})" if cid == -1
                              else f"C{cid} (n={len(sub)})"))
        ax.set(xlabel=x, ylabel=y, title=f"{x} vs {y} by cluster")
        legend_right(ax)
        save(fig, f"{num}_scatter_{x.split('_')[0]}_{y.split('_')[0]}.png")

    seasons = [s for s in C.SEASON_ORDER if s in feats["season"].values]
    clusters = sorted(feats["cluster"].unique())
    width = 0.8 / max(1, len(clusters))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, cid in enumerate(clusters):
        counts = [((feats["season"] == s) & (feats["cluster"] == cid)).sum()
                  for s in seasons]
        bars = ax.bar([j + i * width for j in range(len(seasons))], counts,
                      width, color=ccol(cid), edgecolor="k", linewidth=0.5,
                      alpha=0.85, label="Noise" if cid == -1 else f"C{cid}")
        for bar, c in zip(bars, counts):
            if c:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"n={c}", ha="center", va="bottom", fontsize=8,
                        fontweight="bold")
    ax.set_xticks([j + width * (len(clusters) - 1) / 2
                   for j in range(len(seasons))])
    ax.set_xticklabels(seasons, fontsize=9)
    ax.set(xlabel="Season", ylabel="Number of days",
           title="Cluster distribution by season (with cluster sizes)")
    legend_right(ax)
    save(fig, "15_season_cluster_bar.png")
    print(f"[step3] {CLIENT}: 12 clustering figures -> "
          f"{FIG.relative_to(REPO)}")


if __name__ == "__main__":
    main()
