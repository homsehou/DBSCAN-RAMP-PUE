#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 2: hybrid DBSCAN clustering of the daily profiles and construction of the
representative seasonal profiles used by the calibration steps."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from config import (
    CLEAN_TS_CSV, CLUSTERED_CSV, CLUSTER_FREQ_BY_SEASON_CSV, CLUSTER_FREQ_GLOBAL_CSV,
    CLUSTER_PROFILES_CSV, DAILY_MATRIX_CSV, DBSCAN_EPS_PERCENTILE, DBSCAN_K_NEIGHBORS,
    DBSCAN_N_COMPONENTS, DBSCAN_W_FEAT, DBSCAN_W_TS, DT_MIN, FEATURES_CSV,
    FEATURES_WITH_CLUSTERS_CSV, FIG_DIR, CLUSTERING_FEATURES, N_SLOTS,
    PROFILE_STATISTIC_FOR_CALIBRATION, SEASONAL_PROFILES_CSV,
    SEASONAL_PROFILES_MEAN_CSV, SEASON_MAP_LOCAL_TO_PROJECT, SEASON_ORDER,
    local_season_of_date,
)
from step1_preprocess import compute_daily_features, day_profile, season_label_en

CLUSTER_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                  "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#aec7e8"]

# Seed of the PCA solver. For series longer than 500 days, randomized SVD in scikit-learn, hence a fixed seed for
# reproducible runs matching the archived baseline clustering (shorter series on the exact solver, seed unused).
PCA_SEED = 3


def cluster_color(cluster_id):
    """Stable display color of one cluster; noise (-1) sharing the orange tone."""
    return "#ff7f0e" if cluster_id == -1 else CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]


def cluster_name(cluster_id):
    """Short display name of one cluster label."""
    return "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"


def run_hybrid_dbscan(series, features):
    """Hybrid DBSCAN on peak-normalized shapes plus daily features."""
    days, rows = [], []
    for day_start, group in series.groupby(series.index.floor("D")):
        profile = day_profile(day_start, group)
        peak = float(np.max(profile))
        rows.append(profile / (peak if peak > 0 else 1.0))
        days.append(day_start)
    dates_df = pd.DataFrame({"date": days})

    # Block 1: normalized day shapes, standardized then compressed by PCA.
    x_scaled = StandardScaler().fit_transform(np.vstack(rows))
    z_ts = PCA(n_components=min(DBSCAN_N_COMPONENTS, x_scaled.shape[1]),
               random_state=PCA_SEED).fit_transform(x_scaled)

    # Block 2: standardized aggregated features of the entropy-retained days only
    # (entropy-retained = days surviving the step1 Shannon-entropy screen on the
    # peak-normalized shape, degenerate flat days below the threshold).
    aligned = dates_df.merge(features[["date"] + CLUSTERING_FEATURES],
                             on="date", how="inner")
    z_ts = z_ts[dates_df["date"].isin(aligned["date"]).values]
    agg = aligned[CLUSTERING_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    agg = StandardScaler().fit_transform(agg)
    hybrid = np.hstack([DBSCAN_W_TS * z_ts, DBSCAN_W_FEAT * agg])

    # eps from the k-distance curve: high percentile of the k-th neighbor distances.
    k_eff = min(max(3, DBSCAN_K_NEIGHBORS), len(hybrid))
    neigh = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    neigh.fit(hybrid)
    distances, _ = neigh.kneighbors(hybrid)
    k_distances = np.sort(distances[:, -1])
    # Fallback eps of 0.5 for the degenerate case of a zero percentile (identical
    # collapsed points), a safe non-zero radius to keep DBSCAN from an empty run.
    eps = float(np.percentile(k_distances, DBSCAN_EPS_PERCENTILE)) or 0.5

    labels_df = aligned[["date"]].copy()
    labels_df["cluster"] = DBSCAN(eps=eps, min_samples=k_eff,
                                  metric="euclidean").fit_predict(hybrid)
    return labels_df, eps, k_eff, k_distances


def build_clustered_table(daily, features, labels_df):
    """Descriptive per-day table with cluster labels and both season namings."""
    retained = daily.loc[daily.index.isin(pd.to_datetime(features["date"]))]
    table = compute_daily_features(retained)
    table = (table.merge(labels_df, left_index=True, right_on="date", how="inner")
             .set_index("date").sort_index())
    table["season_local"] = table.index.to_series().apply(local_season_of_date)
    table["season"] = table["season_local"].map(SEASON_MAP_LOCAL_TO_PROJECT)
    table.to_csv(CLUSTERED_CSV)
    return table


def save_cluster_counts(clustered):
    """Global and per-season day counts of each DBSCAN cluster."""
    global_counts = clustered.groupby("cluster").size().rename("count").reset_index()
    global_counts.to_csv(CLUSTER_FREQ_GLOBAL_CSV, index=False)
    local_counts = (clustered.groupby(["season_local", "cluster"]).size()
                    .rename("count").reset_index())
    local_counts.to_csv(CLUSTER_FREQ_BY_SEASON_CSV, index=False)
    return global_counts, local_counts


def compute_cluster_profiles(daily, clustered):
    """Per-cluster daily curves plus mean and median profiles, in W and normalized."""
    profiles = {}
    for cluster_id in sorted(c for c in clustered["cluster"].unique() if c != -1):
        matrix = daily.loc[clustered.index[clustered["cluster"] == cluster_id]].values.astype(float)
        maxs = np.nanmax(matrix, axis=1)
        maxs[maxs == 0] = 1.0
        normalized = matrix / maxs[:, None]
        profiles[cluster_id] = {
            "n_days": matrix.shape[0], "days": matrix,
            "mean_W": np.nanmean(matrix, axis=0),
            "median_W": np.nanmedian(matrix, axis=0),
            "mean_norm": np.nanmean(normalized, axis=0),
            "median_norm": np.nanmedian(normalized, axis=0),
        }
    return profiles


def save_cluster_profiles(profiles):
    """Cluster-profile summary CSV, one row per cluster and statistic."""
    labels = [f"{slot * 15 // 60:02d}:{slot * 15 % 60:02d}" for slot in range(N_SLOTS)]
    rows = []
    for cluster_id, p in profiles.items():
        for stat in ("mean_W", "median_W", "mean_norm", "median_norm"):
            row = {"cluster": cluster_id, "stat": stat}
            for label, value in zip(labels, p[stat]):
                row[label] = value
            rows.append(row)
    pd.DataFrame(rows).to_csv(CLUSTER_PROFILES_CSV, index=False, encoding="utf-8")


def build_seasonal_profiles(clustered, daily):
    """Mean representative profile per season, from the dominant regular cluster."""
    rows = []
    for season in SEASON_ORDER:
        subset = clustered.loc[(clustered["season"] == season) & (clustered["cluster"] != -1)]
        if subset.empty:
            continue
        dominant = int(subset["cluster"].mode().iloc[0])
        season_days = subset.index[subset["cluster"] == dominant]
        mean_profile = np.nanmean(daily.loc[daily.index.isin(season_days)].values, axis=0)
        row = {"season": season, "dominant_cluster": dominant,
               "n_days": int(len(season_days)),
               "profile_statistic": PROFILE_STATISTIC_FOR_CALIBRATION}
        row.update({f"slot_{i:02d}": float(mean_profile[i]) for i in range(N_SLOTS)})
        rows.append(row)
    df_mean = pd.DataFrame(rows)
    df_mean.to_csv(SEASONAL_PROFILES_MEAN_CSV, index=False)
    df_mean.to_csv(SEASONAL_PROFILES_CSV, index=False)
    return df_mean


def figure_kdistance(k_distances, eps, k_eff, path):
    """k-distance curve with the selected eps radius."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(k_distances, "k-", lw=1.6, label=f"{k_eff}-distance")
    ax.axhline(eps, color="#D62728", lw=1.4, ls="--",
               label=f"Hybrid eps ({DBSCAN_EPS_PERCENTILE}th percentile) = {eps:.3f}")
    ax.set(xlabel="Sorted days", ylabel=f"Distance to the {k_eff}th nearest neighbor",
           title="Hybrid DBSCAN k-distance plot")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_cluster_histogram(labels, path):
    """Number of retained days assigned to each cluster, noise included."""
    unique, counts = np.unique(labels, return_counts=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for cluster_id, count in zip(unique, counts):
        ax.bar(cluster_name(cluster_id), count, color=cluster_color(cluster_id),
               edgecolor="k", label=f"{cluster_name(cluster_id)} ({count} days)")
    ax.set(xlabel="Cluster", ylabel="Number of days",
           title="DBSCAN cluster distribution")
    ax.legend()
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_cluster_profiles(profiles, path):
    """One panel per cluster: individual daily curves and their mean profile."""
    if not profiles:
        return
    t = np.arange(N_SLOTS) * DT_MIN / 60.0
    fig, axes = plt.subplots(1, len(profiles), figsize=(6 * len(profiles), 5),
                             squeeze=False)
    for ax, (cluster_id, p) in zip(axes[0], sorted(profiles.items())):
        color = cluster_color(cluster_id)
        for i, row in enumerate(p["days"]):
            ax.plot(t, row, color=color, alpha=0.08, lw=0.6,
                    label="Individual days" if i == 0 else None)
        ax.plot(t, p["mean_W"], color=color, lw=2.5, label="Mean profile")
        ax.set(title=f"Cluster {cluster_id} (n={p['n_days']})", xlabel="Hour [h]")
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0, 0].set_ylabel("Power [W]")
    fig.suptitle("Cluster profiles (W)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_scatter(clustered, x_col, y_col, path):
    """Retained days in feature space, one color per cluster, noise on top."""
    cluster_ids = sorted(clustered["cluster"].unique(), key=lambda c: (c == -1, c))
    fig, ax = plt.subplots(figsize=(7, 5))
    for cluster_id in cluster_ids:
        mask = clustered["cluster"] == cluster_id
        ax.scatter(clustered.loc[mask, x_col], clustered.loc[mask, y_col],
                   s=70 if cluster_id == -1 else 50,
                   alpha=0.85 if cluster_id == -1 else 0.75,
                   c=cluster_color(cluster_id),
                   edgecolors="red" if cluster_id == -1 else "k", lw=0.6,
                   zorder=5 if cluster_id == -1 else 3,
                   label=f"{cluster_name(cluster_id)} (n={mask.sum()})")
    ax.set(xlabel=x_col, ylabel=y_col, title=f"DBSCAN - {x_col} vs {y_col}")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_season_cluster_bar(clustered, path):
    """Seasonal distribution of the retained days across the discovered clusters."""
    counts = pd.crosstab(clustered["season"], clustered["cluster"])
    cluster_ids = sorted(counts.columns)
    fig, ax = plt.subplots(figsize=(8, 5))
    counts[cluster_ids].plot(kind="bar", stacked=True, ax=ax, edgecolor="k",
                             color=[cluster_color(c) for c in cluster_ids])
    ax.set(xlabel="Season", ylabel="Number of days",
           title="Season vs cluster distribution")
    ax.set_xticklabels([season_label_en(lab.get_text()) for lab in ax.get_xticklabels()],
                       rotation=0)
    ax.legend(title="Cluster")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_seasonal_profiles(clustered, daily, df_mean, path):
    """One panel per season: regular days of the dominant cluster and their mean."""
    if df_mean.empty:
        return
    t = np.arange(N_SLOTS) * DT_MIN / 60.0
    seasons = list(df_mean["season"])
    fig, axes = plt.subplots(len(seasons), 1, figsize=(12, 4.5 * len(seasons)),
                             sharex=True, squeeze=False)
    for ax, (_, row) in zip(axes[:, 0], df_mean.iterrows()):
        season_days = clustered.index[(clustered["season"] == row["season"])
                                      & (clustered["cluster"] == row["dominant_cluster"])]
        for i, day_row in enumerate(daily.loc[daily.index.isin(season_days)].values):
            ax.plot(t, day_row, color="#C8C8C8", alpha=0.18, lw=0.9,
                    label="Regular days" if i == 0 else None)
        mean_profile = row.filter(regex=r"^slot_").values.astype(float)
        ax.plot(t, mean_profile, color="#1F77B4", lw=2.4,
                label=f"Mean regular profile (n={row['n_days']})")
        ax.set_title(f"{season_label_en(row['season'])} - retained regular days")
        ax.set_ylabel("Power [W]")
        ax.grid(alpha=0.2)
        ax.legend()
    axes[-1, 0].set_xlabel("Hour [h]")
    axes[-1, 0].set_xticks(np.arange(0, 25, 1))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    print("=" * 72)
    print("  STEP 2 - Clustering")
    print("=" * 72)

    series = pd.read_csv(CLEAN_TS_CSV, index_col="Date", parse_dates=True)["P_W"]
    daily = pd.read_csv(DAILY_MATRIX_CSV, index_col=0, parse_dates=True)
    features = pd.read_csv(FEATURES_CSV, parse_dates=["date"])
    print(f"Clean samples: {len(series)}; retained days: {len(features)}")

    labels_df, eps, k_eff, k_distances = run_hybrid_dbscan(series, features)
    labels_df.to_csv(FEATURES_WITH_CLUSTERS_CSV, index=False)
    print(f"DBSCAN eps = {eps:.4f}, min_samples = {k_eff}")

    clustered = build_clustered_table(daily, features, labels_df)
    global_counts, local_counts = save_cluster_counts(clustered)
    print(global_counts.to_string(index=False))
    print(local_counts.to_string(index=False))

    retained = daily.loc[daily.index.isin(clustered.index)]
    profiles = compute_cluster_profiles(retained, clustered)
    save_cluster_profiles(profiles)
    df_mean = build_seasonal_profiles(clustered, daily)

    figure_kdistance(k_distances, eps, k_eff, os.path.join(FIG_DIR, "08_k_distance_plot.png"))
    figure_cluster_histogram(labels_df["cluster"].values,
                             os.path.join(FIG_DIR, "09_cluster_histogram.png"))
    figure_cluster_profiles(profiles, os.path.join(FIG_DIR, "10_cluster_profiles_W.png"))
    figure_scatter(clustered, "E_kWh", "P_peak_W", os.path.join(FIG_DIR, "12_scatter_E_Ppeak.png"))
    figure_scatter(clustered, "E_kWh", "Load_factor", os.path.join(FIG_DIR, "13_scatter_E_LF.png"))
    figure_scatter(clustered, "P_peak_W", "peak_hour", os.path.join(FIG_DIR, "14_scatter_Ppeak_peakH.png"))
    figure_season_cluster_bar(clustered, os.path.join(FIG_DIR, "15_season_cluster_bar.png"))
    figure_seasonal_profiles(clustered, daily, df_mean,
                             os.path.join(FIG_DIR, "baseline_season_representative_profiles.png"))

    print("Step 2 complete.")
    print(f"  -> cluster labels:      {FEATURES_WITH_CLUSTERS_CSV}")
    print(f"  -> clustered table:     {CLUSTERED_CSV}")
    print(f"  -> seasonal profiles:   {SEASONAL_PROFILES_CSV}")


if __name__ == "__main__":
    main()
