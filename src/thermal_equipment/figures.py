# -*- coding: utf-8 -*-
"""
figures.py - Shared plotting functions for the cold chain (and incubator).

Each function draws one figure of the chain from the data prepared by the
steps; step2_clustering.py and step4_figures.py call them.

Functions:
  fig_season_profiles     : 07_season_profiles.png
  fig_season_stacked      : 07b_season_profiles_stacked.png
  fig_calibration         : 40_calib_<season>.png
  fig_ldc                 : 41_ldc_<season>.png
  fig_fft                 : 42_fft_<season>.png
  fig_param_summary       : 50_param_summary.png
"""

from __future__ import annotations

import calendar
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import config as C

warnings.filterwarnings("ignore")


# =============================================================================
# PLOT STYLE CONSTANTS
# =============================================================================
SEASON_COLORS = {
    "Dry season": "#E74C3C",
    "Transition": "#F39C12",
    "Rainy season": "#3498DB",
}

SEASON_LABELS_EN = {
    "Dry season": "Dry season",
    "Transition": "Transition season",
    "Rainy season": "Rainy season",
}


def season_label_en(season: str) -> str:
    return SEASON_LABELS_EN.get(season, season)


def time_axis_h() -> np.ndarray:
    return np.arange(C.N_SLOTS) * C.DT_MIN / 60.0


def profile_daily_energy_kwh(profile: np.ndarray) -> float:
    return float(np.nansum(profile) * C.DT_MIN / 60.0 / 1000.0)


def apply_adaptive_axis(ax, signals):
    flat = np.concatenate([np.ravel(s) for s in signals])
    flat = flat[np.isfinite(flat)]
    if len(flat) == 0:
        return
    y_max = max(float(flat.max()), 10.0)
    upper = y_max * 1.12
    ax.set_ylim(0, upper)


# =============================================================================
# STEP 1 - Daily profiles
# =============================================================================


def fig_season_profiles(daily: pd.DataFrame, feats: pd.DataFrame,
                        path: str) -> None:
    """07_season_profiles.png."""
    feats = feats.copy()
    feats["month"] = pd.to_datetime(feats.index).month
    feats["season"] = feats["month"].apply(C.month_to_season)
    season_order = C.SEASON_ORDER
    present = [s for s in season_order if s in feats["season"].values]
    if not present:
        return

    fig, axes = plt.subplots(len(present), 1, figsize=(10, 3.8 * len(present)),
                              sharex=True)
    if len(present) == 1:
        axes = [axes]
    t = time_axis_h()
    for ax, season in zip(axes, present):
        days_idx = feats.index[feats["season"] == season]
        sub = daily.loc[daily.index.isin(days_idx)]
        if sub.empty:
            continue
        mean_profile = np.nanmean(sub.values, axis=0)
        avg_energy = profile_daily_energy_kwh(mean_profile)
        period_months = sorted(feats.loc[feats["season"] == season,
                                          "month"].unique())
        period_label = ", ".join(calendar.month_abbr[int(m)]
                                  for m in period_months)
        ax.plot(t, mean_profile, lw=2.6,
                color=SEASON_COLORS.get(season, "gray"),
                label=f"{season_label_en(season)} (n={len(sub)})")
        apply_adaptive_axis(ax, [mean_profile])
        ax.text(0.02, 0.94,
                "\n".join([
                    season_label_en(season),
                    f"({period_label})",
                    f"Average daily consumption = {avg_energy:.2f} kWh/day",
                ]),
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=SEASON_COLORS.get(season, "gray"), alpha=0.9))
        ax.set_ylabel("Power [W]")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Hour [h]")
    fig.suptitle("Average profile by season", fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_season_stacked(daily: pd.DataFrame, feats: pd.DataFrame,
                       path: str) -> None:
    """07b_season_profiles_stacked.png."""
    feats = feats.copy()
    feats["month"] = pd.to_datetime(feats.index).month
    feats["season"] = feats["month"].apply(C.month_to_season)
    season_order = C.SEASON_ORDER
    present = [s for s in season_order if s in feats["season"].values]
    if not present:
        return
    fig, axes = plt.subplots(len(present), 1, figsize=(10, 4 * len(present)),
                              sharex=True)
    if len(present) == 1:
        axes = [axes]
    t = time_axis_h()
    for ax, season in zip(axes, present):
        sub = daily.loc[daily.index.isin(
            feats.index[feats["season"] == season])]
        if sub.empty:
            continue
        mean_profile = np.nanmean(sub.values, axis=0)
        avg_energy = profile_daily_energy_kwh(mean_profile)
        period_months = sorted(feats.loc[feats["season"] == season,
                                          "month"].unique())
        period_label = ", ".join(calendar.month_abbr[int(m)]
                                  for m in period_months)
        ax.set_facecolor("#F8F8F8")
        for _, row in sub.iterrows():
            ax.plot(t, row.values, lw=0.3,
                    color=SEASON_COLORS.get(season, "gray"), alpha=0.15)
        ax.plot(t, mean_profile, lw=2.5,
                color=SEASON_COLORS.get(season, "gray"),
                label=f"{season_label_en(season)} (n={len(sub)})")
        apply_adaptive_axis(ax, [sub.values, mean_profile])
        ax.text(0.02, 0.95,
                "\n".join([
                    season_label_en(season),
                    f"({period_label})",
                    f"Average daily consumption = {avg_energy:.2f} kWh/day",
                ]),
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=SEASON_COLORS.get(season, "gray"), alpha=0.9))
        ax.set_ylabel("Power [W]")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Hour [h]")
    fig.suptitle("Average daily profiles by season", fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# Cluster palette without orange (orange reserved for "Noise") to avoid
# any visual confusion between clusters and noise
CLUSTER_PALETTE = [
    "#1f77b4",  # blue     (C0)
    "#2ca02c",  # green    (C1)
    "#d62728",  # red      (C2)
    "#9467bd",  # violet   (C3)
    "#8c564b",  # brown    (C4)
    "#e377c2",  # pink     (C5)
    "#17becf",  # cyan     (C6)
    "#bcbd22",  # olive    (C7)
    "#1a9850",  # dark green (C8)
    "#7f7f7f",  # gray     (C9)
]


def _cluster_color(cluster_id: int) -> str:
    if cluster_id == -1:
        return "#FF7F0E"  # orange = Noise (reserved, not part of the palette)
    return CLUSTER_PALETTE[int(cluster_id) % len(CLUSTER_PALETTE)]


def fig_k_distance_plot(daily: pd.DataFrame, feats: pd.DataFrame,
                         path: str, k: int = 5) -> None:
    """08_k_distance_plot.png - k-distance plot for DBSCAN epsilon selection."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors

    n = len(daily)
    if n < k + 1:
        return

    # Reproduce the PCA + standardized features of the hybrid step2
    norm = []
    for _, row in daily.iterrows():
        v = np.clip(row.values, 0.0, None)
        m = v.max()
        norm.append(v / m if m > 0 else v)
    norm = np.array(norm)
    n_comp = min(C.DBSCAN_N_COMPONENTS, norm.shape[1], n - 1)
    n_comp = max(1, n_comp)
    pca = PCA(n_components=n_comp)
    Z_shape = pca.fit_transform(norm)
    var_kept = pca.explained_variance_ratio_.cumsum()
    # Cumulative-variance cutoff at 0.90 (90%): enough shape components to preserve
    # the daily-profile geometry, # high enough to drop the low-variance PCA tail, mostly noise in the distance metric.
    n_keep = int(np.searchsorted(var_kept, 0.90) + 1)
    Z_shape = Z_shape[:, :n_keep]
    F = feats[C.CLUSTERING_FEATURES].values
    Z_feat = StandardScaler().fit_transform(F)
    # Hybrid representation: weighted side-by-side stack of the shape components
    # (Z_shape) and the standardized engineered features (Z_feat), # for a single DBSCAN distance blending daily-curve geometry with the scalar
    # descriptors.
    # W_TS and W_FEAT as the relative pull of each block on that distance
    # (higher W_TS for shape-driven grouping, higher W_FEAT for feature-driven).
    Z = np.concatenate([C.DBSCAN_W_TS * Z_shape,
                         C.DBSCAN_W_FEAT * Z_feat], axis=1)

    k_eff = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k_eff).fit(Z)
    distances, _ = nn.kneighbors(Z)
    k_dist = np.sort(distances[:, -1])
    eps = float(np.percentile(distances[:, -1],
                                C.DBSCAN_EPS_PERCENTILE))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(len(k_dist)), k_dist, color="steelblue", lw=1.5)
    ax.axhline(eps, color="red", linestyle="--", lw=1.5,
                label=f"eps (P{C.DBSCAN_EPS_PERCENTILE})={eps:.4f}")
    ax.set_xlabel(f"Days (sorted by {k_eff}-NN distance)")
    ax.set_ylabel(f"{k_eff}-NN distance")
    ax.set_title(f"k-distance plot (k={k_eff}, n={n})")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_baseline_season_representative_profiles(daily: pd.DataFrame,
                                                  feats: pd.DataFrame,
                                                  labels: np.ndarray,
                                                  path: str) -> None:
    """baseline_season_representative_profiles.png - 1 panel per season
    showing the mean profile of the dominant cluster."""
    feats = feats.copy()
    feats["month"] = pd.to_datetime(feats.index).month
    feats["season"] = feats["month"].apply(C.month_to_season)
    feats["cluster"] = labels
    seasons = [s for s in C.SEASON_ORDER if s in feats["season"].values]
    if not seasons:
        return

    fig, axes = plt.subplots(len(seasons), 1, figsize=(10, 3.6 * len(seasons)),
                              sharex=True)
    if len(seasons) == 1:
        axes = [axes]
    t = time_axis_h()
    for ax, season in zip(axes, seasons):
        sub_idx = feats[feats["season"] == season].index
        if len(sub_idx) == 0:
            continue
        sub_clusters = feats.loc[sub_idx, "cluster"]
        valid = sub_clusters[sub_clusters != -1]
        if len(valid) == 0:
            cluster_dom = -1
            mask = feats.index.isin(sub_idx)
        else:
            cluster_dom = int(valid.value_counts().index[0])
            mask = (feats.index.isin(sub_idx)
                    & (feats["cluster"] == cluster_dom))
        days = daily.loc[daily.index.isin(feats.index[mask])]
        if days.empty:
            continue
        mean_p = np.nanmean(days.values, axis=0)
        avg_e = profile_daily_energy_kwh(mean_p)
        color = SEASON_COLORS.get(season, "gray")
        ax.plot(t, mean_p, lw=2.6, color=color,
                label=f"{season_label_en(season)} - cluster C{cluster_dom} "
                      f"(n={len(days)})")
        ax.fill_between(t, 0, mean_p, alpha=0.15, color=color)
        apply_adaptive_axis(ax, [mean_p])
        ax.text(0.02, 0.95,
                f"Average daily consumption = {avg_e:.2f} kWh/day",
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=color, alpha=0.9))
        ax.set_ylabel("Power [W]")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Hour [h]")
    fig.suptitle("Baseline seasonal representative profiles",
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_cluster_histogram(labels: np.ndarray, path: str) -> None:
    """09_cluster_histogram.png - DBSCAN cluster distribution bar chart
    with counts annotated above each bar."""
    unique, counts = np.unique(labels, return_counts=True)
    colors = [_cluster_color(cid) for cid in unique]
    bar_labels = ["Noise" if cid == -1 else f"C{cid}" for cid in unique]
    total = int(counts.sum())
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(bar_labels, counts, color=colors, edgecolor="k")
    # Add the counts (n and %) above the bars
    for bar, c in zip(bars, counts):
        pct = 100.0 * c / total if total > 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"n={int(c)}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=10,
                fontweight="bold")
    ax.set(xlabel="Cluster", ylabel="Number of days",
           title=f"DBSCAN cluster distribution (total = {total} days)")
    ax.grid(alpha=0.2, axis="y")
    # Extend the y-axis so the annotations are not cut off
    ax.set_ylim(0, max(counts) * 1.15)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_cluster_profiles_clean(daily: pd.DataFrame, labels: np.ndarray,
                                path: str) -> None:
    """10b_cluster_profiles_W_clean.png - Per-cluster MEAN only."""
    t = time_axis_h()
    fig, ax = plt.subplots(figsize=(10, 5))
    unique = sorted(set(labels))
    for lab in unique:
        mask = labels == lab
        if mask.sum() == 0:
            continue
        color = _cluster_color(lab)
        prof = np.nanmean(daily.values[mask], axis=0)
        title_label = f"Noise (n={mask.sum()})" if lab == -1 else \
                      f"Cluster {lab}  (n={mask.sum()})"
        ls = "--" if lab == -1 else "-"
        ax.plot(t, prof, color=color, lw=2.5, ls=ls, label=title_label)
    ax.set(xlabel="Hour [h]", ylabel="Power [W]",
           title="Cluster mean profiles (clean view)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_scatter_features(feats: pd.DataFrame, x_col: str, y_col: str,
                          x_label: str, y_label: str, title: str,
                          path: str) -> None:
    """12/13/14_scatter_<feat1>_<feat2>.png."""
    fig, ax = plt.subplots(figsize=(7, 5))
    if "cluster" in feats.columns:
        for cid in sorted(feats["cluster"].unique()):
            sub = feats[feats["cluster"] == cid]
            color = _cluster_color(cid)
            label = f"Noise (n={len(sub)})" if cid == -1 else \
                    f"C{cid} (n={len(sub)})"
            ax.scatter(sub[x_col], sub[y_col], c=color, s=18, alpha=0.6,
                       edgecolor="k", linewidth=0.3, label=label)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    else:
        ax.scatter(feats[x_col], feats[y_col], s=18, alpha=0.6, c="#888888")
    ax.set(xlabel=x_label, ylabel=y_label, title=title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_season_cluster_bar(feats: pd.DataFrame, path: str) -> None:
    """15_season_cluster_bar.png - Cluster distribution by season
    with counts (n=X) annotated above each bar."""
    if "cluster" not in feats.columns:
        return
    feats = feats.copy()
    if "season" not in feats.columns:
        feats["month"] = pd.to_datetime(feats.index).month
        feats["season"] = feats["month"].apply(C.month_to_season)
    seasons = [s for s in C.SEASON_ORDER if s in feats["season"].values]
    clusters = sorted(feats["cluster"].unique())
    fig, ax = plt.subplots(figsize=(11, 5.5))
    width = 0.8 / max(1, len(clusters))
    max_count = 0
    for i, cid in enumerate(clusters):
        counts = []
        for s in seasons:
            sub = feats[(feats["season"] == s) & (feats["cluster"] == cid)]
            counts.append(len(sub))
        max_count = max(max_count, max(counts) if counts else 0)
        color = _cluster_color(cid)
        label = "Noise" if cid == -1 else f"C{cid}"
        positions = [j + i * width for j in range(len(seasons))]
        bars = ax.bar(positions, counts, width, color=color, edgecolor="k",
                       linewidth=0.5, label=label, alpha=0.85)
        # Counts n=X above each bar
        for bar, c in zip(bars, counts):
            if c > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"n={int(c)}",
                        ha="center", va="bottom", fontsize=8,
                        fontweight="bold")
    ax.set_xticks([j + width * (len(clusters) - 1) / 2
                   for j in range(len(seasons))])
    ax.set_xticklabels([season_label_en(s) for s in seasons], fontsize=9)
    ax.set(xlabel="Season", ylabel="Number of days",
           title="Cluster distribution by season (with cluster sizes)")
    # Legend on the right so it does not hide the bars
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    ax.set_ylim(0, max_count * 1.15)
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# STEP 4 - Calibration figures
# =============================================================================
def fig_calibration(real: np.ndarray, sim: np.ndarray, windows: list,
                    season: str, metrics: dict, converged: bool,
                    path: str) -> None:
    """40_calib_<season>.png - 2-panel real vs RAMP simulated."""
    t = time_axis_h()
    color = SEASON_COLORS.get(season, "gray")
    s_label = season_label_en(season)
    avg_e = profile_daily_energy_kwh(real)
    n_win = len(windows)

    fig, axes = plt.subplots(2, 1, figsize=(15, 9),
                              gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    ax.fill_between(t, real, alpha=0.3, color=color, label="Real profile")
    ax.plot(t, real, color=color, lw=2)
    ax.plot(t, sim, "k--", lw=2, label="Calibrated RAMP")
    for i, w in enumerate(windows):
        ws, we = w[0], w[1]
        ax.axvspan(ws / 60.0, we / 60.0, alpha=0.08, color="green",
                   label="Windows" if i == 0 else "")
    title_lines = [
        f"RAMP calibration -- {s_label} "
        f"({n_win} windows)",
        (f"Average daily consumption={avg_e:.2f} kWh/day  "
         f"NRMSE={metrics.get('NRMSE', 0):.3f}  "
         f"LDC={metrics.get('LDC_err', 0):.3f}  "
         f"FFT={metrics.get('FFT_err', 0):.3f}  "
         f"dE={metrics.get('err_E_pct', 0):+.1f}%  "
         f"dP={metrics.get('err_P_pct', 0):+.1f}%  "
         f"dLF={metrics.get('err_LF', 0):+.3f}")
    ]
    ax.set_title("\n".join(title_lines))
    ax.set_ylabel("Power [W]")
    # Legend placed to the right so it does not overlap the plotted data.
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.2)

    ax2 = axes[1]
    res = real - sim
    ax2.bar(t, res, width=C.DT_MIN / 60 * 0.9, color=color, alpha=0.6)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xlabel("Hour [h]")
    ax2.set_ylabel("Residual [W]")
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_ldc(real: np.ndarray, sim: np.ndarray, season: str, path: str) -> None:
    """41_ldc_<season>.png - Load duration curve."""
    r_sorted = np.sort(real)[::-1]
    s_sorted = np.sort(sim)[::-1]
    x = np.arange(len(r_sorted)) / max(1, len(r_sorted)) * 100
    color = SEASON_COLORS.get(season, "gray")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, r_sorted, color=color, lw=2, label="Real")
    ax.plot(x, s_sorted, "k--", lw=2, label="RAMP")
    ax.set_xlabel("% of time")
    ax.set_ylabel("Power [W]")
    ax.set_title(f"Load duration curve -- {season_label_en(season)}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_fft(real: np.ndarray, sim: np.ndarray, season: str, path: str,
            fft_err: float = None) -> None:
    """42_fft_<season>.png - FFT spectrum comparison."""
    fft_r = np.abs(np.fft.rfft(real))
    fft_s = np.abs(np.fft.rfft(sim))
    freq = np.fft.rfftfreq(len(real), d=C.DT_MIN / 60.0)
    n_show = min(20, len(freq) - 1)
    freq_show = freq[1:n_show + 1]
    color = SEASON_COLORS.get(season, "gray")
    if fft_err is None:
        # compute simple FFT error (L2-norm log)
        Rh = np.log(fft_r[1:C.FFT_N_KEEP + 1] + 1.0)
        Sh = np.log(fft_s[1:C.FFT_N_KEEP + 1] + 1.0)
        den = max(Rh.max() - Rh.min(), 1e-9)
        fft_err = float(np.sqrt(np.mean((Rh - Sh) ** 2)) / den)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(freq_show, fft_r[1:n_show + 1], color=color, lw=2,
            marker="o", markersize=4, label="Real")
    ax.plot(freq_show, fft_s[1:n_show + 1], "k--", lw=2,
            marker="s", markersize=4, label="RAMP")
    ax.fill_between(freq_show, fft_r[1:n_show + 1], fft_s[1:n_show + 1],
                    alpha=0.15, color=color)
    ax.set_xlabel("Frequency [cycles/hour]")
    ax.set_ylabel("Amplitude FFT")
    ax.set_title(f"FFT spectrum -- {season_label_en(season)}  "
                 f"(FFT_err = {fft_err:.4f})")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# STEP 5 - Param summary
# =============================================================================


def fig_param_summary(report: dict, path: str, client_label: str = "") -> None:
    """50_param_summary.png - 6-panel summary of params/metrics across seasons.

    Legends at the top of the figure (global) so they do not overlap the axes
    of neighboring panels. Extended width-space to avoid overlaps.
    """
    seasons = sorted(report.keys())
    if not seasons:
        return
    n_seasons = len(seasons)

    # Very wide figure + larger wspace so that the ylabels of the right-hand
    # panels are not hidden by the legends to their left
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.55,
                   bottom=0.10, top=0.90)
    width = 0.25

    # 1. Shape metrics (NRMSE, LDC, FFT)
    ax1 = fig.add_subplot(gs[0, 0])
    metric_names = ["NRMSE", "LDC_err", "FFT_err"]
    x = np.arange(len(metric_names))
    for j, season in enumerate(seasons):
        vals = [report[season]["validation"].get(m, 0) for m in metric_names]
        color = SEASON_COLORS.get(season, "gray")
        ax1.bar(x + j * width, vals, width,
                label=season_label_en(season)[:12], color=color, alpha=0.8)
    ax1.axhline(0.15, color="red", linestyle="--", lw=1, alpha=0.5,
                label="Threshold")
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(metric_names, fontsize=8)
    ax1.set_ylabel("Value")
    ax1.set_title("Shape metrics")
    # Legend removed from the individual panels (added at the top of the figure)
    ax1.grid(alpha=0.2)

    # 2. Energy & peak errors
    ax2 = fig.add_subplot(gs[0, 1])
    err_names = ["err_E_pct", "err_P_pct"]
    for j, season in enumerate(seasons):
        vals = [report[season]["validation"].get(m, 0) for m in err_names]
        color = SEASON_COLORS.get(season, "gray")
        ax2.bar(np.arange(len(err_names)) + j * width, vals, width,
                label=season_label_en(season)[:12], color=color, alpha=0.8)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.axhline(10, color="red", linestyle="--", lw=1, alpha=0.5)
    ax2.axhline(-10, color="red", linestyle="--", lw=1, alpha=0.5)
    ax2.set_xticks(np.arange(len(err_names)) + width)
    ax2.set_xticklabels(["dE (%)", "dP (%)"], fontsize=8)
    ax2.set_ylabel("Error (%)")
    ax2.set_title("Energy and peak errors")
    # Legend removed (added at the top of the global figure)
    ax2.grid(alpha=0.2)

    # 3. Activity windows (Gantt)
    ax3 = fig.add_subplot(gs[0, 2])
    y_pos = 0
    y_labels, y_ticks = [], []
    for season in seasons:
        for w in report[season].get("windows", []):
            color = SEASON_COLORS.get(season, "gray")
            sh = w["start_h"]
            eh = w["end_h"]
            ax3.barh(y_pos, eh - sh, left=sh, height=0.6,
                     color=color, alpha=0.7, edgecolor="black", lw=0.5)
            ax3.text(sh + (eh - sh) / 2, y_pos,
                     f"F{w.get('window_id', 0)}",
                     ha="center", va="center", fontsize=7)
            y_labels.append(f"{season_label_en(season)[:8]} "
                            f"F{w.get('window_id', 0)}")
            y_ticks.append(y_pos)
            y_pos += 1
    ax3.set_yticks(y_ticks)
    ax3.set_yticklabels(y_labels, fontsize=7)
    ax3.set_xlabel("Hour")
    ax3.set_xlim(0, 24)
    ax3.set_title("Activity windows")
    ax3.grid(alpha=0.2, axis="x")

    # 4. ON power p1 by window
    ax4 = fig.add_subplot(gs[1, 0])
    data_p1, labels_p1, colors_p1 = [], [], []
    for season in seasons:
        for w in report[season].get("windows", []):
            data_p1.append(w["duty_cycle"]["p1_W"])
            labels_p1.append(f"{season_label_en(season)[:4]} "
                             f"F{w.get('window_id', 0)}")
            colors_p1.append(SEASON_COLORS.get(season, "gray"))
    if data_p1:
        ax4.bar(range(len(data_p1)), data_p1, color=colors_p1, alpha=0.8)
        ax4.set_xticks(range(len(data_p1)))
        ax4.set_xticklabels(labels_p1, fontsize=8, rotation=45,
                              ha="right", rotation_mode="anchor")
    ax4.set_ylabel("p1 [W]")
    ax4.set_xlabel("Window")
    ax4.set_title("ON power (p1) by window")
    ax4.grid(alpha=0.2)
    ax4.tick_params(axis="x", which="major", pad=2)

    # 5. Durations t1/t2 by window
    ax5 = fig.add_subplot(gs[1, 1])
    t1_vals, t2_vals, labels_t = [], [], []
    for season in seasons:
        for w in report[season].get("windows", []):
            t1_vals.append(w["duty_cycle"]["t1_min"])
            t2_vals.append(w["duty_cycle"]["t2_min"])
            labels_t.append(f"{season_label_en(season)[:4]} "
                            f"F{w.get('window_id', 0)}")
    x_pos = np.arange(len(t1_vals))
    if len(t1_vals) > 0:
        ax5.bar(x_pos - 0.15, t1_vals, 0.3, label="t1 (ON)",
                color="steelblue", alpha=0.8)
        ax5.bar(x_pos + 0.15, t2_vals, 0.3, label="t2 (OFF)",
                color="coral", alpha=0.8)
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(labels_t, fontsize=8, rotation=45,
                             ha="right", rotation_mode="anchor")
    ax5.set_ylabel("Duration [min]")
    ax5.set_xlabel("Window")
    ax5.set_title("ON/OFF durations by window")
    # Legend on the right so it does not hide the bars
    # Local legend on ax5 (the ON/OFF durations panel with its own t1/t2 key)
    ax5.legend(loc="upper right", fontsize=8)
    ax5.grid(alpha=0.2)
    ax5.tick_params(axis="x", which="major", pad=2)

    # 6. Global parameters
    ax6 = fig.add_subplot(gs[1, 2])
    global_data = []
    for season in seasons:
        gp = report[season].get("global_params", {})
        global_data.append([
            gp.get("func_time_min", 0),
            gp.get("func_cycle_min", 0),
            gp.get("random_var_w", 0) * 100,
            gp.get("occasional_use", 0) * 100,
            gp.get("time_frac_var", 0) * 100,
            gp.get("thermal_p_var", 0) * 100,
        ])
    global_data = np.array(global_data) if global_data else np.zeros((1, 6))
    global_labels = ["func_time\n(min)", "func_cycle\n(min)", "var_w\n(%)",
                     "occ_use\n(%)", "tfv\n(%)", "therm_pv\n(%)"]
    x_g = np.arange(len(global_labels))
    for j, season in enumerate(seasons):
        color = SEASON_COLORS.get(season, "gray")
        ax6.bar(x_g + j * width, global_data[j], width,
                label=season_label_en(season)[:12], color=color, alpha=0.8)
    ax6.set_xticks(x_g + width)
    ax6.set_xticklabels(global_labels, fontsize=8)
    ax6.set_title("Global parameters")
    # Legend removed (added at the top of the global figure)
    ax6.grid(alpha=0.2)

    # Global legend at the top (season colors)
    handles = []
    for season in seasons:
        color = SEASON_COLORS.get(season, "gray")
        handles.append(plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.85,
                                       label=season_label_en(season)))
    handles.append(plt.Line2D([0], [0], color="red", linestyle="--", lw=1.5,
                                label="Threshold"))
    fig.legend(handles=handles, loc="upper center",
                bbox_to_anchor=(0.5, 0.97), ncol=len(handles),
                fontsize=10, frameon=True)

    fig.suptitle(f"{client_label} -- Optimal appliance parameters (cold chain)",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
