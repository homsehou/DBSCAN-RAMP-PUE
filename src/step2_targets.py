"""Step 2: targets to calibrate, days retained by DBSCAN and sampling noise of each target.

Target: one client, one season (or one calendar month with GRAIN=mois) and one fleet configuration.
  - Fleet: before or since the change of appliances, at the date seen at the meter; calibration
    of the incubator 0152GBO over its operating period only.
  - Removal of the out-of-service periods (at least 21 consecutive days without running), never
    calibrated; isolated days without running kept in the target (occasional use).
  - DBSCAN search of atypical days among the running days, in a space of three principal
    components of the daily shape and six bounded descriptors, with the radius at the knee of
    the sorted distances to the 4th neighbour; flagged days kept above 10 % of the energy.
  - Sampling noise by a bootstrap of blocks of 7 consecutive days: mean and 95th percentile of
    the gap of a perfect model for the ten criteria and the local shape, local standard
    deviations and 90 % band.

Outputs in resultats/cibles_v9/ (suffix saison or mois):
  clusters_<grain>/<client>.csv   fate of every day
  bruit_<grain>/<target>.npz      target profile, local standard deviations, band, peaks
  cibles_<grain>.csv              one row per target
  dbscan_sensibilite_<grain>.csv  share of flagged days for other radii
  hors_service.csv                out-of-service periods
"""
import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import settings as S
import validation as V

# Grain of the targets and output folders
OUT = S.TARGETS
GRAIN = os.environ.get("GRAIN", "saison")      # "mois" for one target per calendar month
for d in (f"clusters_{GRAIN}", f"bruit_{GRAIN}"):
    (OUT / d).mkdir(exist_ok=True)
SLOTS = [f"slot_{i}" for i in range(S.SLOTS_PER_DAY)]
NEAR_ZERO_KWH = 0.05    # daily energy of a target too small for a calibration
MAX_FLAGGED_SHARE = 0.10  # energy share of the flagged days, threshold for keeping all of them
DRAWS = 200             # bootstrap means per use (band, then perfect model)
BLOCK = 7               # consecutive days per bootstrap block


def configuration(client, day):
    """Fleet configuration of a day, or state of the incubator."""
    if client in S.FLEET_CHANGE:
        d = S.FLEET_CHANGE[client]
        return f"avant {d}" if day < d else f"depuis {d}"
    if client == "0152GBO":
        start, end = S.INCUBATOR_ACTIVE
        return "active" if start <= day <= end else "hors service"
    return ""


def out_of_service(days, peak, threshold):
    """Flag of the days belonging to a period of at least 21 days without running."""
    # Runs of consecutive days with the same running state, and their span in days
    j = pd.to_datetime(pd.Series(days))
    running = pd.Series(peak >= threshold)
    run = (running != running.shift()).cumsum()
    span = j.groupby(run).transform(lambda x: (x.max() - x.min()).days + 1)
    return (~running & (span >= S.OUT_OF_SERVICE_DAYS)).to_numpy()


def descriptors(X):
    """Six bounded day descriptors: energy, peak, load factor, peak hour (sine and cosine),
    share of the energy between 6:00 and 18:00."""
    E = X.sum(axis=1) * 0.25 / 1000.0
    P = X.max(axis=1)
    LF = np.divide(X.mean(axis=1), P, out=np.zeros(len(X)), where=P > 0)
    h = X.argmax(axis=1) * 2 * np.pi / S.SLOTS_PER_DAY
    tot = X.sum(axis=1)
    day_share = np.divide(X[:, 24:72].sum(axis=1), tot, out=np.zeros(len(X)), where=tot > 0)
    return np.column_stack([E, P, LF, np.sin(h), np.cos(h), day_share])


def feature_space(X):
    """Day coordinates: three principal components of the shape, not rescaled, and the six
    standardised descriptors."""
    amp = X.max(axis=1, keepdims=True)
    comp = PCA(n_components=min(3, len(X) - 1), random_state=0).fit_transform(
        X / np.where(amp > 0, amp, 1.0))
    return np.hstack([comp, StandardScaler().fit_transform(descriptors(X))])


def neighbour_distances(F):
    """Sorted distance of each day to its 4th neighbour (min_samples - 1)."""
    k = S.DBSCAN_MIN_SAMPLES - 1
    return np.sort(np.sort(np.linalg.norm(F[:, None] - F[None, :], axis=2), axis=1)[:, k])


def knee_radius(d):
    """Radius at the knee of the sorted curve, the point farthest below the chord."""
    x = np.linspace(0, 1, len(d))
    y = (d - d[0]) / max(d[-1] - d[0], 1e-12)
    return float(d[np.argmax(x - y)])


def dbscan(X, radius="knee"):
    """DBSCAN labels, -1 for an atypical day, and the radius in use.

    radius: "knee" for the automated radius, or a percentile of the distances for the
    sensitivity table."""
    if len(X) <= S.DBSCAN_MIN_SAMPLES:
        return np.zeros(len(X), int), np.nan
    F = feature_space(X)
    d = neighbour_distances(F)
    eps = knee_radius(d) if radius == "knee" else float(np.percentile(d, radius))
    eps = max(eps, 1e-6)
    return DBSCAN(eps=eps, min_samples=S.DBSCAN_MIN_SAMPLES).fit_predict(F), eps


def block_draw(n, rng):
    """Indices of a bootstrap draw by blocks of 7 consecutive days."""
    starts = rng.integers(0, max(n - BLOCK + 1, 1), int(np.ceil(n / BLOCK)))
    return np.concatenate([np.arange(s, min(s + BLOCK, n)) for s in starts])[:n]


def noise(Vd, peaks, rng):
    """Sampling noise of a target, with its days sorted by date.

    First 200 bootstrap means: local standard deviations of the smoothed residual (one and two
    hours) and 90 % band. Next 200: role of the perfect model, with the mean and the 95th
    percentile of its gap to the target for every criterion.
    """
    y = Vd.mean(axis=0)
    amp = max(float(np.ptp(y)), 1e-9)
    B = np.array([Vd[block_draw(len(Vd), rng)].mean(axis=0) for _ in range(2 * DRAWS)])
    ref, perfect = B[:DRAWS], B[DRAWS:]
    sd_1h = np.maximum(np.std([S.smooth(b - y, 4) for b in ref], axis=0), 0.02 * amp)
    sd_2h = np.maximum(np.std([S.smooth(b - y, 8) for b in ref], axis=0), 0.02 * amp)
    gaps = pd.DataFrame([{**V.criteria(y, b, peaks), **V.local_shape(y, b, sd_1h, sd_2h)}
                         for b in perfect]).abs()
    arrays = {"cible": y, "sd_1h": sd_1h, "sd_2h": sd_2h, "pics": np.asarray(peaks, int),
              "bas": np.percentile(ref, 5, axis=0), "haut": np.percentile(ref, 95, axis=0)}
    return gaps.mean(), gaps.quantile(0.95), arrays


def periods(days, flag):
    """Runs of flagged days, as (first day, last day, number of days)."""
    s = pd.Series(flag)
    run = (s != s.shift()).cumsum()
    return [(days[g.index[0]], days[g.index[-1]], len(g)) for _, g in s[s].groupby(run[s])]


# Targets of every client, one seed for the whole bootstrap
rng = np.random.default_rng(20260919)
summary, sensitivity, stops = [], [], []
for client, family in S.CLIENTS.items():
    # Running state, out-of-service periods and fleet configuration of each day
    jr = pd.read_csv(OUT / "journees" / f"{client}.csv").sort_values("jour").reset_index(drop=True)
    X_all = jr[SLOTS].to_numpy(float)
    peak = X_all.max(axis=1)
    threshold = S.CLIENT_RUNNING_THRESHOLD_W.get(client, S.RUNNING_THRESHOLD_W[family])
    jr["marche"] = peak >= threshold
    jr["hors_service"] = out_of_service(jr.jour, peak, threshold)
    jr["config"] = [configuration(client, j) for j in jr.jour]
    jr.loc[jr.config == "hors service", "hors_service"] = True
    for start, end, n in periods(list(jr.jour), jr.hors_service.to_numpy()):
        e = X_all[(jr.jour >= start) & (jr.jour <= end)].sum(axis=1).mean() * 0.25 / 1000
        stops.append({"client": client, "debut": start, "fin": end, "journees": n,
                      "kWh_j": round(float(e), 3)})
    # Period of each day (season or month), then one target per period and configuration
    period_of = S.season_of_month if GRAIN == "saison" else (lambda m: S.MONTHS[int(str(m)[5:7]) - 1])
    jr["saison"] = jr["mois"].map(period_of)
    jr["cible"] = [s if c == "" else f"{s} | {c}" for s, c in zip(jr.saison, jr.config)]
    jr["cluster"], jr["retenue"], jr["moitie"] = -2, False, -1
    for target, block in jr.groupby("cible"):
        # Days in service of the target, and its identity card
        service = block[~block.hors_service]
        X = service[SLOTS].to_numpy(float)
        row = {"client": client, "famille": family, "cible": target,
               "nom": S.file_name(client, target), "saison": block.saison.iloc[0],
               "configuration": block.config.iloc[0], "jours": len(block),
               "jours_hors_service": int(block.hors_service.sum()), "jours_en_service": len(service),
               "premier_jour": block.jour.min(), "dernier_jour": block.jour.max()}
        if len(service) == 0:
            summary.append({**row, "jours_retenus": 0, "statut": "hors service"})
            continue
        # DBSCAN on the running days only. Reason: days without running, all close to zero, as
        # the main group otherwise, with every day of use left as an outlier. Days without
        # running kept in the target (occasional use), with the label -3.
        active = service.marche.to_numpy()
        lab = np.full(len(X), -3)
        lab[active], eps = dbscan(X[active])
        E = X.sum(axis=1)
        flagged_share = float(E[lab == -1].sum() / max(E.sum(), 1e-9))
        kept_all = flagged_share > MAX_FLAGGED_SHARE
        retained = np.ones(len(X), bool) if kept_all else lab != -1
        # Sensitivity of the flagged share and of the energy bias to the radius
        for r in ("knee", 80, 90, 95):
            l2 = np.full(len(X), -3)
            l2[active], e2 = dbscan(X[active], r)
            sensitivity.append({"client": client, "cible": target,
                                "rayon": "coude" if r == "knee" else r, "eps": e2,
                                "part_ecartee": float(np.mean(l2[active] == -1)) if active.any() else 0.0,
                                "biais_E_pct": 100 * (E[l2 != -1].mean() / max(E.mean(), 1e-9) - 1)})
        # Retained days, their daily energy and the status of the target
        Vd = X[retained]
        kwh = float(Vd.mean(axis=0).sum() * 0.25 / 1000) if len(Vd) else 0.0
        jr.loc[service.index, "cluster"] = lab
        jr.loc[service.index, "retenue"] = retained
        jr.loc[service.index[retained], "moitie"] = np.arange(retained.sum()) % 2
        if len(Vd) < S.MIN_TARGET_DAYS:
            status = "non calibrable (moins de 10 journees)"
        elif kwh < NEAR_ZERO_KWH:
            status = "quasi nulle"
        else:
            status = "a calibrer"
        row.update({"jours_retenus": len(Vd), "jours_ecartes_dbscan": int((lab == -1).sum()),
                    "eps": eps, "part_E_ecartee": round(flagged_share, 4),
                    "atypiques_gardees": bool(kept_all),
                    "kWh_j_service": round(float(E.mean() * 0.25 / 1000), 3), "kWh_j": round(kwh, 3),
                    "biais_E_pct": round(100 * (kwh / max(E.mean() * 0.25 / 1000, 1e-9) - 1), 2),
                    "part_marche": round(float(service.marche[retained].mean()), 3),
                    "statut": status})
        # Marked peaks and sampling noise, for the targets to calibrate only
        if status == "a calibrer":
            peaks = V.robust_peaks(Vd)
            mean_gap, p95, arrays = noise(Vd, peaks, rng)
            row.update({"pics_slots": " ".join(map(str, peaks))})
            row.update({f"plancher_{k}": v for k, v in mean_gap.items()})
            row.update({f"p95_{k}": v for k, v in p95.items()})
            np.savez(OUT / f"bruit_{GRAIN}" / f"{row['nom']}.npz", **arrays)
        summary.append(row)
    jr.drop(columns=SLOTS).to_csv(OUT / f"clusters_{GRAIN}" / f"{client}.csv", index=False)

# Tables of the targets, of the DBSCAN sensitivity and of the out-of-service periods
t = pd.DataFrame(summary)
t.to_csv(OUT / f"cibles_{GRAIN}.csv", index=False)
pd.DataFrame(sensitivity).to_csv(OUT / f"dbscan_sensibilite_{GRAIN}.csv", index=False)
pd.DataFrame(stops).to_csv(OUT / "hors_service.csv", index=False)
print("targets:", len(t), "| to calibrate:", int((t.statut == "a calibrer").sum()))
