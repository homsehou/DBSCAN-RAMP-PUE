# -*- coding: utf-8 -*-
"""Step 2 - Clustering: group similar days, build the seasonal targets.

Grouping of similar days by DBSCAN on shape and descriptors, atypical
days left in a noise group. Per season, mean profile of the dominant
cluster as the calibration target of step 4.

Input : resultats/<type>/<client>/daily_matrix_W.csv, daily_features.csv
Output: clustered_features.csv                 features + cluster label
        seasonal_representative_profiles.csv  one target per season
        figures/08_k_distance.png, 09_seasonal_targets.png,
        10_cluster_histogram.png

Self-contained script. Run it alone with:
    PUE_TYPE=cold_chain PUE_CLIENT=0017SAM python step2_clustering.py
"""
import os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C   # shared settings of the repository
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# --- Settings ------------------------------------------------------------------
PUE_TYPE = os.environ.get("PUE_TYPE", "cold_chain")
CLIENT = os.environ.get("PUE_CLIENT", "0017SAM")
REPO = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO / "resultats" / PUE_TYPE / CLIENT


# Seasons of central Benin; May and October get their own model each


def hybrid_dbscan(table, feats):
    """One euclidean distance mixing day SHAPE and day DESCRIPTORS.

    Shape block: profiles normalised by their peak, compressed by PCA.
    Feature block: the six descriptors, standardised. DBSCAN radius at the
    90th percentile of the k-nearest-neighbour distances, in place of a
    manual elbow reading.
    """
    P = table.to_numpy(float)
    shape = P / np.maximum(P.max(axis=1)[:, None], 1e-9)
    pca = PCA(n_components=min(C.PCA_MAX_COMPONENTS, len(table) - 1))
    z_shape = pca.fit_transform(shape)
    enough = np.searchsorted(pca.explained_variance_ratio_.cumsum(),
                             C.PCA_VARIANCE_KEPT) + 1
    z_shape = z_shape[:, :max(1, int(enough))]
    z_feat = StandardScaler().fit_transform(feats[C.FEATURES].to_numpy(float))
    z = np.hstack([z_shape, z_feat])
    dist, _ = NearestNeighbors(n_neighbors=C.K_NEIGHBORS).fit(z).kneighbors(z)
    k_dist = np.sort(dist[:, -1])
    eps = float(np.percentile(k_dist, C.EPS_PERCENTILE)) or 0.5
    labels = DBSCAN(eps=eps, min_samples=C.K_NEIGHBORS).fit_predict(z)
    return labels, k_dist, eps


def seasonal_targets(table, labels):
    """Per season: mean profile of the days of the dominant cluster."""
    season_of_day = pd.Series(
        [C.SEASONS[d.month] for d in table.index], index=table.index)
    rows = []
    for season in ["Dry season", "May", "Rainy season", "October"]:
        in_season = (season_of_day == season).to_numpy()
        if not in_season.any():
            continue
        season_labels = labels[in_season]
        valid = season_labels[season_labels != -1]
        # Fallback on every day of a season left without a cluster
        dominant = int(pd.Series(valid).mode()[0]) if len(valid) else -1
        chosen = in_season & ((labels == dominant) | (dominant == -1))
        profile = table.to_numpy(float)[chosen].mean(axis=0)
        rows.append({"season": season, "cluster_dominant": dominant,
                     "n_days": int(chosen.sum()),
                     **{f"slot_{j}": profile[j] for j in range(C.SLOTS_PER_DAY)}})
    return pd.DataFrame(rows)


def figures(k_dist, eps, labels, targets, fig_dir):
    """Three visual checks: the radius rule, the targets, the clusters."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(k_dist, lw=1.5)
    ax.axhline(eps, color="red", ls="--", label=f"eps = P{C.EPS_PERCENTILE}")
    ax.set_xlabel("Days (sorted)")
    ax.set_ylabel(f"Distance to {C.K_NEIGHBORS}th neighbour")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "08_k_distance.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    hours = np.arange(C.SLOTS_PER_DAY) * 0.25
    for _, row in targets.iterrows():
        ax.plot(hours, [row[f"slot_{j}"] for j in range(C.SLOTS_PER_DAY)],
                label=f"{row['season']} ({row['n_days']} days)")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Power [W]")
    ax.set_title(f"{CLIENT} - seasonal calibration targets")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "09_seasonal_targets.png", dpi=110)
    plt.close(fig)

    counts = pd.Series(labels).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    names = ["noise" if c == -1 else f"cluster {c}" for c in counts.index]
    ax.bar(names, counts.values, color="#0072B2")
    ax.set_ylabel("Days")
    fig.tight_layout()
    fig.savefig(fig_dir / "10_cluster_histogram.png", dpi=110)
    plt.close(fig)


def main():
    table = pd.read_csv(OUT_DIR / "daily_matrix_W.csv", index_col=0,
                        parse_dates=True)
    feats = pd.read_csv(OUT_DIR / "daily_features.csv", index_col=0,
                        parse_dates=True)
    labels, k_dist, eps = hybrid_dbscan(table, feats)
    feats["cluster"] = labels
    feats["season"] = [C.SEASONS[d.month] for d in feats.index]
    feats.to_csv(OUT_DIR / "clustered_features.csv")
    targets = seasonal_targets(table, labels)
    targets.to_csv(OUT_DIR / "seasonal_representative_profiles.csv",
                   index=False)
    figures(k_dist, eps, labels, targets, OUT_DIR / "figures")
    n_noise = int((labels == -1).sum())
    print(f"[step2] {CLIENT}: {labels.max() + 1} clusters, "
          f"{n_noise} noise days, {len(targets)} seasonal targets")


if __name__ == "__main__":
    main()
