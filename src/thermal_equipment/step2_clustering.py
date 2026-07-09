# -*- coding: utf-8 -*-
"""step2_clustering.py - Cold chain step 2: hybrid DBSCAN clustering.

Methodology: PCA on the peak-normalised daily profiles (shape block,
cut at 90% explained variance) + 6 standardised features, weighted
concatenation, DBSCAN with eps = 90th percentile of the k-distances. The
dominant cluster of each season gives the representative seasonal profile.
Outputs: clustered_features.csv, features_with_clusters.csv,
cluster_profiles_W.csv, cluster frequency tables,
seasonal_representative_profiles.csv and the step 2 diagnostic figures.
The client code comes from the PUE_CLIENT environment variable (config.py).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import config as C
import figures as figs


def normalize_profile(profile: np.ndarray) -> np.ndarray:
    """Normalise one day ("peak" = /max, "energy" = /sum): shape only."""
    p = np.clip(profile, 0.0, None)
    m = p.max() if C.NORMALIZE_METHOD == "peak" else p.sum()
    return p / m if m > 0 else p


def hybrid_dbscan(daily_W: pd.DataFrame,
                  feats: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """DBSCAN labels + diagnostic info (eps, cluster and noise counts)."""
    n = len(daily_W)

    # Block 1: PCA on the normalised profiles, cut at 90% explained variance
    norm = np.array([normalize_profile(row.values)
                     for _, row in daily_W.iterrows()])
    n_comp = max(1, min(C.DBSCAN_N_COMPONENTS, norm.shape[1], n - 1))
    pca = PCA(n_components=n_comp)
    Z_shape = pca.fit_transform(norm)
    var_kept = pca.explained_variance_ratio_.cumsum()
    n_keep = max(1, min(int(np.searchsorted(var_kept, 0.90) + 1),
                        Z_shape.shape[1]))
    Z_shape = Z_shape[:, :n_keep]

    # Block 2: the 6 standardised clustering features
    Z_feat = StandardScaler().fit_transform(feats[C.CLUSTERING_FEATURES].values)

    # Weighted concatenation of the two blocks
    Z = np.concatenate([C.DBSCAN_W_TS * Z_shape, C.DBSCAN_W_FEAT * Z_feat],
                       axis=1)

    # eps at the 90th percentile of the k-distances: automated eps selection in
    # place of the manual elbow reading of the k-distance curve, high enough to
    # fold sparse days into noise instead of forcing them into thin clusters.
    k = max(2, min(C.DBSCAN_K_NEIGHBORS, n - 1))
    distances, _ = NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(Z)
    eps = float(np.percentile(distances[:, -1], C.DBSCAN_EPS_PERCENTILE))

    labels = DBSCAN(eps=eps, min_samples=k).fit(Z).labels_
    info = {"eps": eps,
            "n_clusters": int(len(set(labels)) - (1 if -1 in labels else 0)),
            "n_noise": int((labels == -1).sum())}
    return labels, info


def cluster_profiles(daily_W: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Mean and median profile (96 slots) of each cluster, noise included."""
    rows = []
    for lab in sorted(set(labels)):
        sub = daily_W.values[labels == lab]
        row = {"cluster": int(lab), "n_days": int(sub.shape[0])}
        mean_profile = sub.mean(axis=0)
        median_profile = np.median(sub, axis=0)
        for j, v in enumerate(mean_profile):
            row[f"mean_slot_{j}"] = float(v)
        for j, v in enumerate(median_profile):
            row[f"med_slot_{j}"] = float(v)
        rows.append(row)
    return pd.DataFrame(rows)


def freq_by_season(daily_W: pd.DataFrame,
                   labels: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Number of days per cluster, globally and per season."""
    seasons = pd.Series(daily_W.index).dt.month.apply(C.month_to_season)
    df = pd.DataFrame({"cluster": labels, "season": seasons.values})
    by_global = df.groupby("cluster").size().reset_index(name="n_days")
    by_season = (df.groupby(["season", "cluster"]).size()
                 .reset_index(name="n_days"))
    return by_global, by_season


def seasonal_representative_profiles(daily_W: pd.DataFrame,
                                     labels: np.ndarray) -> pd.DataFrame:
    """Mean profile of the days of the dominant cluster, per season.

    Noise days (-1) never dominant: for a season with noise only, fallback to
    all its days with the dominant cluster labelled -1.
    """
    seasons = pd.Series(daily_W.index).dt.month.apply(C.month_to_season).values
    rows = []
    for season in C.SEASON_ORDER:
        mask = seasons == season
        if mask.sum() == 0:
            continue
        valid = labels[mask][labels[mask] != -1]
        if len(valid) == 0:
            cluster_dom, mask_days = -1, mask
        else:
            cluster_dom = int(pd.Series(valid).value_counts().index[0])
            mask_days = mask & (labels == cluster_dom)
        profile = daily_W.values[mask_days].mean(axis=0)
        row = {"season": season, "cluster_dominant": cluster_dom,
               "n_days_dominant": int(mask_days.sum())}
        for j in range(daily_W.shape[1]):
            row[f"slot_{j}"] = float(profile[j])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_step2_figures(client_code: str, daily_W: pd.DataFrame,
                       feats_out: pd.DataFrame, seas_repr: pd.DataFrame,
                       labels: np.ndarray) -> None:
    """Step 2 diagnostic figures (shared plot style, English labels)."""
    fdir = C.client_fig_dir(client_code)

    # Season profiles: mean per season, plain and stacked with all the days
    figs.fig_season_profiles(daily_W, feats_out,
                             str(fdir / "07_season_profiles.png"))
    figs.fig_season_stacked(daily_W, feats_out,
                            str(fdir / "07b_season_profiles_stacked.png"))

    # Cluster diagnostics: mean profiles, size histogram, k-distance plot
    figs.fig_cluster_profiles_clean(daily_W, labels,
                                    str(fdir / "07_cluster_profiles.png"))
    figs.fig_cluster_histogram(labels, str(fdir / "08_cluster_histogram.png"))
    figs.fig_k_distance_plot(daily_W, feats_out,
                             str(fdir / "08_k_distance_plot.png"))

    # 09 - representative profile of the dominant cluster, per season
    t = np.arange(C.N_SLOTS) * C.DT_MIN / 60.0
    slot_cols = [c for c in seas_repr.columns if c.startswith("slot_")]
    fig, ax = plt.subplots(figsize=(8, 4))
    for _, row in seas_repr.iterrows():
        ax.plot(t, row[slot_cols].values.astype(float), lw=1.8,
                color=figs.SEASON_COLORS.get(row["season"], "gray"),
                label=f"{figs.season_label_en(row['season'])} "
                      f"(n={row['n_days_dominant']})")
    ax.set(xlabel="Hour of day [h]", ylabel="Power [W]",
           title=f"{client_code} - Seasonal representative profiles")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(fdir / "09_seasonal_repr_profiles.png",
                bbox_inches="tight")
    plt.close(fig)

    # Feature scatters coloured by cluster + season x cluster distribution
    figs.fig_scatter_features(feats_out, "E_kWh", "P_peak_W", "Energy [kWh]",
                              "Peak [W]", "Energy vs peak power",
                              str(fdir / "12_scatter_E_Ppeak.png"))
    figs.fig_scatter_features(feats_out, "E_kWh", "Load_factor",
                              "Energy [kWh]", "Load factor",
                              "Energy vs load factor",
                              str(fdir / "13_scatter_E_LF.png"))
    figs.fig_scatter_features(feats_out, "P_peak_W", "peak_hour_min",
                              "Peak [W]", "Peak hour [min]",
                              "Peak power vs peak hour",
                              str(fdir / "14_scatter_Ppeak_peakH.png"))
    figs.fig_season_cluster_bar(feats_out,
                                str(fdir / "15_season_cluster_bar.png"))
    figs.fig_baseline_season_representative_profiles(
        daily_W, feats_out, labels,
        str(fdir / "baseline_season_representative_profiles.png"))


def cluster_client(client_code: str) -> dict:
    """Full step 2 for one client; returns a small summary dict."""
    print(f"\n=== STEP2 CLUSTERING : {client_code} ===")
    rdir = C.client_results_dir(client_code)
    daily_W = pd.read_csv(rdir / C.DAILY_MATRIX_CSV, index_col=0,
                          parse_dates=True)
    feats = pd.read_csv(rdir / C.FEATURES_CSV, index_col=0, parse_dates=True)
    print(f"  {len(daily_W)} days in input")

    labels, info = hybrid_dbscan(daily_W, feats)
    print(f"  DBSCAN: eps={info['eps']:.4f}, n_clusters={info['n_clusters']}, "
          f"n_noise={info['n_noise']}")

    feats_out = feats.copy()
    feats_out["cluster"] = labels
    feats_out.to_csv(rdir / C.CLUSTERED_CSV)
    feats_out.to_csv(rdir / C.FEATURES_WITH_CLUSTERS_CSV)
    cluster_profiles(daily_W, labels).to_csv(rdir / C.CLUSTER_PROFILES_CSV,
                                             index=False)
    by_global, by_season = freq_by_season(daily_W, labels)
    by_global.to_csv(rdir / C.CLUSTER_FREQ_GLOBAL_CSV, index=False)
    by_season.to_csv(rdir / C.CLUSTER_FREQ_BY_SEASON_CSV, index=False)
    seas_repr = seasonal_representative_profiles(daily_W, labels)
    seas_repr.to_csv(rdir / C.SEASONAL_PROFILES_CSV, index=False)

    plot_step2_figures(client_code, daily_W, feats_out, seas_repr, labels)

    return {"client": client_code, "n_days": int(len(daily_W)),
            "n_clusters": info["n_clusters"], "n_noise": info["n_noise"],
            "n_seasons_kept": int(seas_repr.shape[0])}


if __name__ == "__main__":
    cluster_client(C.PUE_CLIENT)
