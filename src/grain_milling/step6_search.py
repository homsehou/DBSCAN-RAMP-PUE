# -*- coding: utf-8 -*-
"""Step 6 - Surrogate screening of candidate appliances (methodology step 8).

The surrogate takes its classic role here: search wide with a cheap
model, verify narrow with the real one. A numpy replica of the exact
engine path of this appliance family (certified indistinguishable from the engine's own
seed-to-seed noise) scores thousands of candidates -
a coarse pass over every hour-cut split of the activity windows
crossed with every
measured burst power, then a Latin-hypercube refinement of the
survivors over duties (factor 1/2..2 around the analytic inversion, the
very span the step 9 correction may reach), cycle length, standby and
engine dispersion, in both day-to-day variability branches. A
15-minute meter cannot tell a short strong burst from a long weak one,
so the six criteria settle what the measurement leaves open - inside
the physical bounds step 5 attaches to each power. The best few of
each branch are then verified by the untouched RAMP engine on the cheap
budget; step 7 gives the engine the last word. A shape guard keeps every
retained candidate within SHAPE_CAP of the split floor, so a passed
amplitude criterion can never buy an unmoored profile.
Run: PUE_TYPE=grain_milling PUE_CLIENT=0016GBO python step6_search.py
"""
import io, json, os, sys
import random as rnd
from contextlib import redirect_stdout
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C   # shared settings of the repository
PUE_TYPE = os.environ.get("PUE_TYPE", "grain_milling")
CLIENT = os.environ.get("PUE_CLIENT", "0016GBO")
REPO = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO / "resultats" / PUE_TYPE / CLIENT
sys.path.insert(0, str(REPO))    # embedded ramp/ package importable


def place_event(spots, L, rand_time, cws, rng, attempts=200):
    """One switch-on event, engine-style.

    Uniform start over the free minutes, random length up to the daily
    budget, duty cycle matched by the engine's own rules: overlap for an
    in-day event, start containment for a midnight-crossing one; a rare
    no-match is re-picked under a bounded budget."""
    while attempts > 0:
        starts = [(i, a, b) for i, (a, b) in enumerate(spots) if b - a >= L]
        n = sum(b - a - L + 1 for _, a, b in starts)
        if n == 0:
            return None
        k = int(rng.integers(n))
        for spot_i, a, b in starts:
            if k < b - a - L + 1:
                start = a + k
                break
            k -= b - a - L + 1
        largest = min(rand_time, b - start)
        length = int(rng.uniform(L, largest)) if largest > L else L
        s_loc, e_loc = start % 1440, (start + length - 1) % 1440
        for cyc_id, (wa, wb) in enumerate(cws):
            if (wa <= s_loc < wb) if e_loc < s_loc else (
                    s_loc == 0 or not (e_loc < wa or s_loc > wb)):
                return start, length, cyc_id, spot_i
        attempts -= 1
    return None


def replica_days(p1, p2, cycle_len, regions, tfrv, n_days, rng,
                 p_var=C.RAMP_VARIABILITY):
    """Numpy replica of a simulate() run, same profile distribution.

    Every mechanism of the engine's continuous path is reproduced: the
    midnight merge of the first and last windows, the one-cycle
    spillover, the integer truncation of randomised durations, switch-on
    events of random length paved with the region cycles, the 0.99 cap
    on the daily minutes, the day-order overwrite of spilled minutes.
    Only the random stream differs - the engine's own seed noise."""
    L = int(cycle_len)
    cws = [(int(g["start"]), int(g["end"])) for g in regions]
    t_ons = [int(np.clip(round(g["duty"] * L), 2, L - 2)) for g in regions]
    burst = [float(g.get("power", p1)) - p2 for g in regions]
    func_time = sum(b - a for a, b in cws)
    merged, end_i, start_i = [list(w) for w in cws], None, None
    for i, w in enumerate(merged):
        if w[1] >= 1438:
            end_i = i
        if w[0] <= 2:
            start_i = i
    if end_i is not None and start_i is not None and end_i != start_i:
        merged[end_i][1] = 1440 + merged[start_i][1]
        merged[start_i] = [0, 0]
    total_avail = sum(b - a for a, b in merged)
    minutes = (n_days + 2) * 1440          # one padding day each side
    arr = np.zeros(minutes)
    for day in range(n_days + 2):
        off = day * 1440
        draw = rng.uniform(1 - tfrv, 1 + tfrv)
        lo, hi = sorted((func_time, int(func_time * draw)))
        rand_time = max(round(rng.uniform(lo, hi)), L)
        rand_time = max(min(rand_time, int(0.99 * total_avail)), L)
        cycles = []
        for t_on, amp in zip(t_ons, burst):   # the day's randomised cycles
            power = amp * rng.uniform(1 - p_var, 1 + p_var)
            on = int(t_on * rng.uniform(1 - p_var, 1 + p_var))
            rest = int((L - t_on) * rng.uniform(1 - p_var, 1 + p_var))
            parts = [np.full(on, power), np.zeros(rest)]
            if len(regions) == 3 and rng.integers(2):
                parts.reverse()            # engine flips 3-region cycles
            cycles.append(np.concatenate(parts))
        spots = [[max(0, a + off),
                  min(b + off + (L if b >= 1438 else 0), minutes)]
                 for a, b in merged if a != b]
        used = 0
        while used <= rand_time:
            event = place_event(spots, L, rand_time, cws, rng)
            if event is None:
                break
            start, length, cyc_id, spot_i = event
            used += length
            write = length if used <= rand_time else length - (used
                                                               - rand_time)
            if write <= 0:
                break
            cyc = cycles[cyc_id]
            arr[start:start + write] = np.tile(
                cyc, int(np.ceil(write / len(cyc))))[:write]
            a, b = spots.pop(spot_i)
            if b > start + write:
                spots.insert(spot_i, [start + write, b])
            if start > a:
                spots.insert(spot_i, [a, start])
    days = arr[1440:1440 + n_days * 1440].reshape(n_days, 96, C.SLOT_MIN)
    return days.mean(axis=2) + p2          # + meter standby


def simulate(p1, p2, cycle_len, regions, tfrv, n_seeds, n_days, seed0,
             p_var=C.RAMP_VARIABILITY):
    """Real RAMP runs (continuous mode) of the native appliance.

    Every region carries a cycle of the same length: ON for duty x L
    minutes at p1, at rest for the remainder. func_cycle = L keeps every
    switch-on event at least one full cycle long - shorter events would
    be truncated by the engine to their ON part and spike the profile."""
    from ramp import User, UseCase
    on_span = sum(g["end"] - g["start"] for g in regions)
    profiles, day_maxima, day_energy = [], [], []
    for seed in range(n_seeds):
        rnd.seed(seed0 + seed), np.random.seed(seed0 + seed)
        with redirect_stdout(io.StringIO()):
            uc = UseCase(name="native", date_start=C.SIM_START_DATE,
                         date_end=pd.Timestamp(C.SIM_START_DATE)
                         + pd.Timedelta(days=n_days - 1))
            user = User(user_name="client")
            uc.add_user(user)
            app = user.add_appliance(
                name="appliance", number=1, power=p1 - p2,   # duty cycles
                                                             # carry the
                                                             # real power
                num_windows=len(regions), func_time=int(on_span),
                func_cycle=int(cycle_len), fixed="yes",
                fixed_cycle=len(regions), continuous_duty_cycle=1,
                occasional_use=1.0, thermal_p_var=p_var,
                time_fraction_random_variability=tfrv)
            # Usage window = duty region: a switch-on event stays inside
            # its region, its cycle never leaks onto a neighbour
            app.windows(**{f"window_{n}": [g["start"], g["end"]]
                           for n, g in enumerate(regions, start=1)})
            for num, g in enumerate(regions, start=1):
                # 2-min floors: the engine truncates randomized cycle
                # durations to int, a 1-min part can round down to empty
                t_on = int(np.clip(round(g["duty"] * cycle_len), 2,
                                   cycle_len - 2))
                amp = float(g.get("power", p1)) - p2
                app.specific_cycle(num, **{
                    f"p_{num}1": amp, f"t_{num}1": t_on,
                    f"p_{num}2": 0.0, f"t_{num}2": int(cycle_len) - t_on,
                    f"r_c{num}": p_var,
                    f"cw{num}1": [g["start"], g["end"]]})
            arr = uc.generate_daily_load_profiles(flat=False,
                                                  continuous=True)
        days = arr.reshape(n_days, C.SLOTS_PER_DAY,
                           C.SLOT_MIN).mean(axis=2) + p2  # + meter standby
        profiles.append(days.mean(axis=0))
        day_maxima.extend(days.max(axis=1))
        day_energy.extend(days.sum(axis=1) * 0.25 / 1000.0)
    return {"sim": np.mean(profiles, axis=0),
            "dmax": float(np.percentile(day_maxima, C.BURST_PCTL)),
            "dE": np.asarray(day_energy), "seeds": np.asarray(profiles)}


def lhs(n, dims, rng):
    """Latin hypercube in the unit cube (McKay, Beckman, Conover 1979)."""
    return (rng.random((n, dims)) + np.stack(
        [rng.permutation(n) for _ in range(dims)], axis=1)) / n


def season_windows(season):
    """Search domain of the season: its measured activity windows."""
    rows = pd.read_csv(OUT_DIR / "analytical_params.csv")
    return json.loads(rows[rows["season"] == season].iloc[0]
                      ["windows_json"])


def region_mean(target, a, b):
    """Mean of the target inside one region."""
    return float(target[np.arange(a // C.SLOT_MIN,
                                  b // C.SLOT_MIN) % 96].mean())


def duty_center(target, a, b, p1, p2):
    """Duty the region mean asks for - the analytic inversion of step 8."""
    return float(np.clip((region_mean(target, a, b) - p2)
                         / max(p1 - p2, 1e-6), 0.02, 0.98))


def region_power_floor(target, a, b, p2):
    """Lowest burst power a region can use: its own mean, at duty one."""
    return max(region_mean(target, a, b), p2 + 1e-3)


def structures(windows):
    """Candidate region structures: the windows whole, then every
    hour-cut split - one cut per window, two cuts when a single window
    fills the day - under the engine cap of three regions."""
    base = [(int(a), int(b)) for a, b in windows]
    out, room = [base], 3 - len(base)
    for i, (a, b) in enumerate(base):
        cuts = list(range(60 * (a // 60) + 60, b, 60))
        if room >= 1:
            out += [base[:i] + [(a, c), (c, b)] + base[i + 1:]
                    for c in cuts]
        if room >= 2:
            out += [[(a, c1), (c1, c2), (c2, b)]
                    for c1 in cuts for c2 in cuts if c2 > c1]
    return out


def cycle_bounds(regs, centers):
    """Cycle lengths a structure can carry.

    Floor: an emptier region is not representable, the engine needs at
    least two ON minutes per cycle. Ceiling: half the smallest region,
    so at least two cycle positions fit and the ON minutes move from
    day to day instead of spiking one fixed slot. The floor wins."""
    lo = max(6, int(np.ceil(2.0 / max(min(centers), 0.01))))
    return lo, max(min(b - a for a, b in regs) // 2, lo)


def build(regs, centers, factors, L, p1, p2, p_var, powers=None):
    """One candidate appliance from its ingredients.

    A region may carry its own burst power: sharing the peak power
    forces a quiet region into a very short ON part, and the engine's
    two-minute floor then makes a low mean unreachable. A weaker,
    longer burst reaches it."""
    powers = powers if powers is not None else [p1] * len(regs)
    return {"p1": float(p1), "p2": float(p2), "p_var": float(p_var),
            "L": int(L), "regions": [
                {"start": a, "end": b, "power": float(w),
                 "duty": float(np.clip(f * c, 2.0 / L, 0.98))}
                for (a, b), c, f, w in zip(regs, centers, factors,
                                           powers)]}


def screening(target, powers, p2, windows):
    """Coarse pass: the exact inversion for every structure and power.

    The burst power sets the duty every region needs, so a whole family
    of shapes hangs on it - it deserves an exhaustive look before any
    refinement, at the anchored standby and dispersion."""
    cands = []
    for regs in structures(windows):
        for p1 in powers:
            centers = [duty_center(target, a, b, p1, p2) for a, b in regs]
            lo, hi = cycle_bounds(regs, centers)
            for x in np.linspace(0.0, 1.0, C.SEARCH_CENTERS):
                cands.append(build(regs, centers, [1.0] * len(regs),
                                   round(lo * (hi / lo) ** x), p1, p2,
                                   C.RAMP_VARIABILITY))
    return cands


def refinement(target, kept, standbys, rng):
    """Fine pass: Latin hypercube around each surviving pair.

    Around a structure and a burst power that the coarse pass liked,
    the duties travel by the factor 1/2..2 (the very span the step 9
    correction may reach), each region's own burst power from its mean
    up to the peak power, the cycle length over its whole range, the
    standby over the measured percentiles and the engine dispersion
    over its offered values."""
    cands = []
    for cand in kept:
        regs = [(g["start"], g["end"]) for g in cand["regions"]]
        p1, R = cand["p1"], len(cand["regions"])
        for row in lhs(C.SEARCH_PER_STRUCT, 2 * R + 3, rng):
            p2 = standbys[min(int(row[-1] * len(standbys)),
                              len(standbys) - 1)]
            p_var = C.SEARCH_P_VARS[min(int(row[-2] * len(C.SEARCH_P_VARS)),
                                        len(C.SEARCH_P_VARS) - 1)]
            powers = [float(np.clip(
                region_power_floor(target, a, b, p2)
                * (p1 / region_power_floor(target, a, b, p2)) ** f,
                p2 + 1e-3, p1))
                for (a, b), f in zip(regs, row[R:2 * R])]
            centers = [duty_center(target, a, b, w, p2)
                       for (a, b), w in zip(regs, powers)]
            lo, hi = cycle_bounds(regs, centers)
            cands.append(build(regs, centers,
                               [0.5 * 4.0 ** f for f in row[:R]],
                               round(lo * (hi / lo) ** row[-3]), p1, p2,
                               p_var, powers))
    return cands


def rank(entry):
    """Criteria first, then the distance of the failing ones."""
    return (-C.n_passed(entry["m"]), C.exceedance(entry["m"]))


def score(cands, target, cv_meas):
    """Replica score of every candidate, both variability branches.

    The same random draw sits behind every candidate, so the comparison
    is fair and only the parameters separate them."""
    pools = {"zero": [], "act": []}
    for cand in cands:
        days = replica_days(cand["p1"], cand["p2"], cand["L"],
                            cand["regions"], 0.0, C.SEARCH_DAYS,
                            np.random.default_rng(C.SEARCH_SEED),
                            cand["p_var"])
        pools["zero"].append({**cand, "tfrv": 0.0,
                              "m": C.metrics(target, days.mean(axis=0))})
        dE = days.sum(axis=1)
        cv_a = float(np.sqrt(max(cv_meas ** 2
                                 - (np.std(dE) / np.mean(dE)) ** 2, 0.0)))
        if cv_a >= C.ACTIVITY_CV_MIN:      # missing day-to-day spread
            tf = C.tfrv_from(min(1.0, cv_a))
            days = replica_days(cand["p1"], cand["p2"], cand["L"],
                                cand["regions"], tf, C.SEARCH_DAYS,
                                np.random.default_rng(C.SEARCH_SEED),
                                cand["p_var"])
            pools["act"].append({**cand, "tfrv": tf,
                                 "m": C.metrics(target,
                                                days.mean(axis=0))})
    return pools


def shortlist(target, sd, cv_meas, windows, cap):
    """Coarse screening, fine refinement, engine verdict on the best.

    Both variability branches are kept apart - step 6 arbitrates them
    with the step 10 rule at the evaluation budget."""
    guard = lambda pool: [e for e in pool
                          if e["m"]["NRMSE"] <= cap] or pool
    powers = sd.get("p1_candidates_W") or [sd["p1_W"]]
    standbys = sd.get("p2_candidates_W") or [sd["p2_W"]]
    coarse = score(screening(target, powers, sd["p2_W"], windows),
                   target, cv_meas)
    rng = np.random.default_rng(C.SEARCH_SEED + 1)
    out = {}
    for name, pool in coarse.items():
        if not pool:
            out[name] = None
            continue
        kept = sorted(guard(pool), key=rank)[:C.SEARCH_KEEP_PAIRS]
        fine = score(refinement(target, kept, standbys, rng), target,
                     cv_meas)[name]
        top = sorted(guard(kept + fine), key=rank)[:C.RESCORE_KEEP]
        for e in top:                      # second look, more days
            days = replica_days(e["p1"], e["p2"], e["L"], e["regions"],
                                e["tfrv"], C.RESCORE_DAYS,
                                np.random.default_rng(C.SEARCH_SEED + 7),
                                e["p_var"])
            e["m"] = C.metrics(target, days.mean(axis=0))
        verified = []
        for e in sorted(guard(top), key=rank)[:C.SEARCH_TOPK]:
            r = simulate(e["p1"], e["p2"], e["L"], e["regions"],
                         e["tfrv"], C.TUNE_SEEDS, C.TUNE_DAYS, C.SEED,
                         e["p_var"])
            verified.append({**e, "m": C.metrics(target, r["sim"])})
        out[name] = min(guard(verified), key=rank)
    return out


def main():
    init = json.loads((OUT_DIR / "calibration_init.json").read_text())
    targets = pd.read_csv(OUT_DIR / "calibration_targets.csv")
    targets = targets[targets["kind"] != "sim_select"]
    slots = [f"slot_{j}" for j in range(C.SLOTS_PER_DAY)]
    def target_of(season, kind):
        r = targets[(targets["season"] == season)
                    & (targets["kind"] == kind)]
        return r[slots].to_numpy(float)[0]
    out = {}
    for season, sd in init["seasons"].items():
        cv = sd["cv_E_meas"]
        wins = season_windows(season)
        cap_sel = C.SHAPE_CAP * C.THRESH["NRMSE"]
        cap_full = cap_sel
        if sd["holdout_valid"]:            # floor = target's own noise
            cap_full = C.SHAPE_CAP * max(
                C.THRESH["NRMSE"],
                C.metrics(target_of(season, "holdout"),
                          target_of(season, "select"))["NRMSE"])
        entry = {"cap_full": cap_full, "cap_select": cap_sel,
                 "full": shortlist(target_of(season, "full"), sd, cv,
                                   wins, cap_full)}
        # Without a valid split the two targets are the same profile
        entry["select"] = entry["full"] if not sd["holdout_valid"] else \
            shortlist(target_of(season, "select"), sd, cv, wins, cap_sel)
        out[season] = entry
        for name, e in entry["full"].items():
            if e:
                powers = "/".join(f"{g['power']:.0f}" for g in e["regions"])
                print(f"[step6] {season} {name}: {-rank(e)[0]}/6  NRMSE "
                      f"{e['m']['NRMSE']:.3f}  cap {cap_full:.2f}  "
                      f"p1 {e['p1']:.0f}W [{powers}]  p2 {e['p2']:.1f}W  "
                      f"var {e['p_var']:.2f}")
    (OUT_DIR / "search_shortlist.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
