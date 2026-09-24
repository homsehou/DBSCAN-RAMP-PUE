"""Validation contract: ten criteria, local shape, noise-relative thresholds and verdict.

Criterion met for a gap below max(nominal threshold, min(95th percentile of a perfect model,
2 x nominal threshold)). The 95th percentile from a block bootstrap of the retained days
(step 2), hence a criterion met by a perfect model in 95 % of the draws. Target "non
verifiable" for its own noise above twice the nominal threshold on NRMSE or EXT_prof: days
too few or too scattered for any judgement of a model at the 10 % level.
"""
import numpy as np
from scipy.signal import find_peaks
import settings as S

THRESHOLDS = {**S.THRESH_10, **S.SHAPE_THRESHOLDS, "ECART_PICS_h": 1.0}
CAP_FACTOR = 2.0        # cap of the noise tolerance, as a multiple of the nominal threshold
MISSED_PEAK_H = 3.0     # gap assigned to a target peak without any model peak within 3 h
PEL_MIN = 0.15          # smallest tolerated worst one-hour smoothed gap, as a share of amplitude


def dominant_peaks(profile):
    """Peaks structuring the shape of a profile, with thresholds relative to its amplitude."""
    # Light smoothing over three slots, then peaks of the whole day
    sig = np.convolve(np.asarray(profile, float), np.ones(3) / 3, mode="same")
    s_max = float(np.nanmax(sig))
    s_range = float(np.nanmax(sig) - np.nanmin(sig))
    if s_max <= 0 or s_range <= 0:
        return np.array([], int)
    p, _ = find_peaks(sig, height=s_max * 0.16,
                      prominence=max(s_range * 0.08, s_max * 0.06), distance=1)
    # Second search after 18:00 with lower thresholds, for the small evening peaks
    late = int(18 * 60 / S.SLOT_MIN)
    pt = np.array([], int)
    if late < len(sig):
        lp, _ = find_peaks(sig[late:], height=s_max * 0.08,
                           prominence=max(s_range * 0.03, s_max * 0.03), distance=1)
        pt = late + lp
    return np.unique(np.concatenate([p, pt])).astype(int)


def robust_peaks(X, share=0.6, draws=40, tol=2, seed=20260910):
    """Peaks of the mean profile found again in at least `share` of random half-samples of days.

    Reason: sampling noise left in the mean of some twenty days, with small bumps taken for
    peaks by the detector. Only the peaks present in most half-samples count as habits.
    """
    X = np.asarray(X, float)
    pr = dominant_peaks(X.mean(axis=0))
    if len(X) < 8 or len(pr) == 0:
        return pr
    # Vote of each half-sample for the peaks of the full mean, within `tol` slots
    rng = np.random.default_rng(seed)
    votes = np.zeros(len(pr))
    for _ in range(draws):
        idx = rng.permutation(len(X))[: len(X) // 2]
        found = dominant_peaks(X[idx].mean(axis=0))
        if len(found) == 0:
            continue
        for i, p in enumerate(pr):
            if np.min(np.abs(found - p)) <= tol:
                votes[i] += 1
    return pr[votes / draws >= share]


def peak_time_gap_h(target_peaks, sim):
    """Worst time gap, in hours, between the marked peaks of the target and those of the model."""
    sim_peaks = dominant_peaks(sim)
    if len(target_peaks) == 0:
        return 0.0
    gaps = []
    for p in target_peaks:
        d = np.abs(sim_peaks - p) if len(sim_peaks) else np.array([96])
        d = np.minimum(d, 96 - d) / 4          # distance on a circular day, in hours
        gaps.append(min(float(d.min()), MISSED_PEAK_H))
    return max(gaps)


def criteria(target, sim, target_peaks):
    """The ten gaps between the target and the model."""
    return {**S.metrics(target, sim), **S.shape(target, sim),
            "ECART_PICS_h": peak_time_gap_h(target_peaks, sim)}


def local_shape(target, sim, sd_1h, sd_2h):
    """Local gaps of the model, smoothed over one and two hours.

    ELM_1h, ELM_2h : worst smoothed gap, relative to the bootstrap standard deviation of the slot
    PEL_1h         : worst one-hour smoothed gap, relative to the amplitude of the target
    """
    r = np.asarray(sim, float) - np.asarray(target, float)
    amp = max(float(np.ptp(target)), 1e-9)
    return {"ELM_1h": float(np.max(np.abs(S.smooth(r, 4)) / sd_1h)),
            "ELM_2h": float(np.max(np.abs(S.smooth(r, 8)) / sd_2h)),
            "PEL_1h": float(np.max(np.abs(S.smooth(r, 4))) / amp)}


def threshold(k, p95):
    """Effective threshold of a criterion: nominal, raised to the noise, capped at twice nominal."""
    return max(THRESHOLDS[k], min(abs(p95[k]), CAP_FACTOR * THRESHOLDS[k]))


def local_thresholds(p95):
    """Thresholds of the local shape: 95th percentile of a perfect model, with 0.15 as floor of PEL."""
    return {"ELM_1h": p95["ELM_1h"], "ELM_2h": p95["ELM_2h"], "PEL_1h": max(PEL_MIN, p95["PEL_1h"])}


def verdict(m, local, p95):
    """Verdict of a target and list of the missed criteria.

    validee        : ten criteria and local shape met, to be confirmed by eye on the plate
    rejetee        : at least one criterion missed
    non verifiable : noise of the days above the cap on NRMSE or EXT_prof
    """
    missed = [k for k in THRESHOLDS if abs(m[k]) > threshold(k, p95)]
    missed += [k for k, s in local_thresholds(p95).items() if local[k] > s]
    if any(p95[k] > CAP_FACTOR * THRESHOLDS[k] for k in ("NRMSE", "EXT_prof")):
        return "non verifiable", missed
    return ("validee" if not missed else "rejetee"), missed
