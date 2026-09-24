"""RAMP calibration of one target, with the nameplate power entered as such.

Rule of the method: nameplate from the field survey as the power entered in RAMP
(power = p_i1 = nameplate), and every other parameter fitted to the shape.

Steps for one target:
  1. exact geometry: optimal three-level staircase under the RAMP rules (at most two ranges per
     cycle, one to three windows), on the profile of the running days and on the profile of all
     days, by plain least squares and by least squares weighted by the noise;
  2. rendering by the engine: occupancy profile of each cycle (RAMP engine, 1 W appliance),
     levels by least squares bounded by the nameplate, six orders of the cycles, short grid of
     settings; choice on J = NRMSE + 0.5 x worst one-hour smoothed gap, plus the local shape and
     the energy (see objective_terms);
  3. realisation at the nameplate: p_i1 = nameplate, p_i2 = measured standby bounded by 0.8
     times the lowest level, t_i1 of the highest cycle derived from the median daily peak, other
     t_i1 = duty cycle x period, t_i2 derived from the duty cycle, then a fixed-point correction
     on the mean of each zone, run on the engine;
  4. shift of the borders of ranges and windows by one slot, kept for a lower J;
  5. arbitration of the finalists on independent seeds, check with two other seeds, verdict.

Two search settings behind the published results (see run.py):
  - "final": peak_weight = 0, short period of 15 min in competition with the long period;
  - "2026-09-19": peak_weight = 0.03, long period only.
"""
import io
import itertools
import random
import sys
import warnings
from contextlib import redirect_stdout, redirect_stderr
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
import settings as S
import validation as V

sys.path.insert(0, str(S.ROOT))            # access to the unmodified RAMP engine in ramp/

# Grid of func_cycle values (min), widened to 5 and 10 min for the targets still missing the
# harmonics criterion: better rendering of the one-to-two-hour bumps of the compressors.
DURATIONS = (15, 30, 60)
WINDOW_SHARES = (1.0, 0.75)            # values of func_time as a share of the span of the windows
VARIABILITY = (0.0, 0.1)               # values of random_var_w
N_WINDOWS = (1, 2, 3)                  # numbers of windows tried for the staircase
ORDERS_KEPT = 2                        # best orders of the cycles kept for the grid
N_REALISED = 5                         # candidates realised at the nameplate, then refined
N_FIXED_POINT = 4                      # iterations of the fixed point at the first realisation
PEL_WEIGHT = 0.5                       # weight of the worst one-hour smoothed gap in J
ENERGY_TOLERANCE = 5.0                 # energy gap tolerated by the objective (%)
# Short period of the cycles (min). Random draw by the engine of the starting phase of every
# switch-on (high or low, 50/50). For a forced switch-on (opening of a dense window, border
# between two cycles, noon for a 24 h window declared in two halves), a peak then a trough in
# the mean, over one period t1 + t2. Short period: transient held within one slot.
SHORT_PERIOD = 15
# Seeds of the engine runs: occupancy profiles, fixed point, borders, arbitration and check
SEED_UNIT, SEED_FIXED, SEED_BORDERS = 424242, 777, 31000
SEED_ARBITRATION, SEEDS_CHECK = 555000, (1042, 98765)


class Appliance:
    """A productive use seen from the side of a RAMP user: family, nameplate(s), measured standby power."""

    def __init__(self, client, family, nameplates, standby, header):
        self.client, self.family = client, family
        self.nameplates = [float(p) for p in nameplates]
        self.nameplate = sum(self.nameplates)       # total power of the fleet
        self.standby = float(standby)
        self.header = header

    def modes(self):
        """Cycle modes in competition: both for the mills, the continuous one for the other families."""
        return (1, 0) if self.family == "grain_milling" else (1,)


# --- RAMP engine --------------------------------------------------------------
class Sink(io.TextIOBase):
    """Sink for the output of the engine, against a range warning filling the logs."""

    def write(self, text):
        return len(text)


def simulate(text, seed, n_seeds=20, n_days=480, days=False):
    """Mean profile, or 15-minute days, produced by the RAMP engine for a declaration."""
    from ramp import User, UseCase
    # Declaration text turned into its function declarer(user)
    space = {}
    exec(text, space)
    # One engine run of n_days per seed, then 15-minute means of the minute profile
    out = []
    for g in range(n_seeds):
        random.seed(seed + g)
        np.random.seed(seed + g)
        with warnings.catch_warnings(), redirect_stdout(Sink()), redirect_stderr(Sink()):
            warnings.simplefilter("ignore")
            uc = UseCase(name="pue", date_start=S.SIM_START_DATE,
                         date_end=pd.Timestamp(S.SIM_START_DATE) + pd.Timedelta(days=n_days))
            user = User(user_name="client", num_users=1)
            uc.add_user(user)
            space["declarer"](user)
            arr = uc.generate_daily_load_profiles(flat=False, continuous=True)
        out.append(arr.reshape(n_days + 1, 96, 15).mean(axis=2)[1:])
    out = np.concatenate(out)
    return out if days else out.mean(axis=0)


# --- RAMP declaration -----------------------------------------------------------
def ramp_windows(windows):
    """Split of a 24 h window into two halves, against a spill of the engine over the next day."""
    return [[0, 720], [720, 1440]] if windows == [[0, 1440]] else windows


def declaration(app, r, cycles, unit=False):
    """RAMP input text in the form written by a RAMP user, runnable on its own.

    r: settings (windows, ranges, func_time, func_cycle, var_w, continuous mode, occasional use).
    cycles: t1, t2 and p2 (standby of the fleet) of each cycle. One appliance per nameplate;
    with unit=True, a single 1 W appliance with p1 and p2 taken as given (occupancy profile).
    """
    win = ", ".join(f"window_{n}={w}" for n, w in enumerate(ramp_windows(r["fenetres"]), start=1))
    plates = [1.0] if unit else app.nameplates
    # One add_appliance block per nameplate, then one specific_cycle line per cycle
    lines = [f"# {app.header}", "", "", "def declarer(user):"]
    for n, plate in enumerate(plates, start=1):
        name, a = (app.family, "app") if len(plates) == 1 else (f"{app.family}_{n}", f"app{n}")
        share = 1.0 if unit else plate / app.nameplate
        lines += [f"    {a} = user.add_appliance(name=\"{name}\", number=1, power={plate:g},",
                  f"        num_windows={len(ramp_windows(r['fenetres']))}, func_time={r['func_time']}, "
                  f"func_cycle={r['func_cycle']},",
                  f"        fixed=\"yes\", fixed_cycle={len(cycles)}, continuous_duty_cycle={r['continu']},",
                  f"        occasional_use={r['occ']:.3f}, flat=\"no\", thermal_p_var=0,",
                  "        time_fraction_random_variability=0, pref_index=0, wd_we_type=2)",
                  f"    {a}.windows(random_var_w={r['var_w']}, {win})"]
        for i, (c, rg) in enumerate(zip(cycles, r["plages"]), start=1):
            p1 = c["p1"] if unit else plate
            p2 = c["p2"] if unit else c["p2"] * share
            cw = ", ".join(f"cw{i}{j}={w}" for j, w in enumerate(rg, start=1))
            lines.append(f"    {a}.specific_cycle({i}, p_{i}1={p1:g}, t_{i}1={c['t1']}, "
                         f"p_{i}2={p2:.1f}, t_{i}2={c['t2']}, r_c{i}=0, {cw})")
    # Stand-alone runner at the bottom of the file
    lines += ["", "",
              "if __name__ == \"__main__\":",
              "    # mean profile of 480 simulated days; RAMP engine of this repository",
              "    # (this file lives in resultats/<run>/entrees_ramp/)",
              "    import sys, pathlib",
              "    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))",
              "    from ramp import User, UseCase",
              f"    uc = UseCase(name=\"pue\", date_start=\"{S.SIM_START_DATE}\", date_end=\"2025-04-25\")",
              "    user = User(user_name=\"client\", num_users=1)",
              "    uc.add_user(user)",
              "    declarer(user)",
              "    profile = uc.generate_daily_load_profiles(flat=False, continuous=True)",
              "    print(profile.reshape(-1, 96, 15)[1:].mean(axis=(0, 2)).round(1))"]
    return "\n".join(lines) + "\n"


# --- exact geometry: optimal staircase under the RAMP rules ----------------------
def dp_staircase(y, levels, w, max_ranges=2, max_windows=3):
    """Best labelling of y (0 = outside windows, c = cycle c) for given levels.

    Gaps weighted by w; at most max_ranges ranges per cycle and max_windows windows; midnight as
    a border. Output: (cost, labels).

    State of the dynamic programme: current label, number of ranges already opened per cycle,
    and number of windows already opened."""
    n, K = len(y), len(levels)
    v = np.r_[0.0, levels]
    P, F = max_ranges + 1, max_windows + 1
    dims = (K + 1,) + (P,) * K + (F,)
    INF = 1e30
    # Costs of the first slot, for every possible label
    cost = np.full(dims, INF)
    back = []
    for l in range(K + 1):
        idx = [l] + [0] * K + [0]
        if l > 0:
            idx[l], idx[-1] = 1, 1
        cost[tuple(idx)] = w[0] * (y[0] - v[l]) ** 2
    # Forward pass: best cost of each state at each slot, with the previous label kept for the way back
    for t in range(1, n):
        new = np.full(dims, INF)
        choice = {}
        for l2 in range(K + 1):
            local = w[t] * (y[t] - v[l2]) ** 2
            for l1 in range(K + 1):
                src = cost[l1]
                if l2 == l1 or l2 == 0:
                    cand = src
                else:
                    # New range of cycle l2, and a new window after the off state
                    cand = np.full(src.shape, INF)
                    sl_src, sl_dst = [slice(None)] * (K + 1), [slice(None)] * (K + 1)
                    sl_src[l2 - 1], sl_dst[l2 - 1] = slice(0, P - 1), slice(1, P)
                    if l1 == 0:
                        sl_src[K], sl_dst[K] = slice(0, F - 1), slice(1, F)
                    cand[tuple(sl_dst)] = src[tuple(sl_src)]
                best = new[l2]
                better = cand + local < best
                if better.any():
                    best[better] = cand[better] + local
                    new[l2] = best
                    choice.setdefault(l2, np.full(src.shape, -1))
                    choice[l2][better] = l1
        back.append(choice)
        cost = new
    # Way back from the best final state to the labels of every slot
    i = np.unravel_index(np.argmin(cost), cost.shape)
    best_cost, state, labels = float(cost[i]), list(i), [int(i[0])]
    for t in range(n - 1, 0, -1):
        l2 = state[0]
        l1 = int(back[t - 1][l2][tuple(state[1:])])
        prev = list(state[1:])
        if l2 != l1 and l2 != 0:
            prev[l2 - 1] -= 1
            if l1 == 0:
                prev[K] -= 1
        state = [l1] + prev
        labels.append(l1)
    return best_cost, np.array(labels[::-1])


def optimal_staircase(y, w, K=3, max_windows=3, tries=40, seed=0):
    """Staircase with K free levels: exact labelling alternated with weighted mean levels.

    Output: the labels (0 = outside windows, 1..K = cycle). Starting levels drawn among
    quantiles of the useful part of the profile, 40 starts, 15 alternations at most."""
    rng = np.random.default_rng(seed)
    useful = y[y > 0.02 * y.max()] if (y > 0.02 * y.max()).any() else y
    q = np.quantile(useful, np.linspace(0.05, 0.95, 12))
    best = (np.inf, None)
    for _ in range(tries):
        lev = np.sort(rng.choice(q, K, replace=False)).astype(float)
        prev = np.inf
        for _ in range(15):
            c, e = dp_staircase(y, lev, w, 2, max_windows)
            for k in range(K):
                if (e == k + 1).any():
                    lev[k] = np.average(y[e == k + 1], weights=w[e == k + 1])
            if prev - c < 1e-9 * max(c, 1):
                break
            prev = c
        c, e = dp_staircase(y, lev, w, 2, max_windows)
        if c < best[0]:
            best = (c, e.copy())
    return best[1]


def renumber(e):
    """Labels 1..n in the order of the cycles present, without the empty cycles."""
    present = [k for k in sorted(set(e.tolist())) if k > 0]
    out = np.zeros_like(e)
    for n, k in enumerate(present, start=1):
        out[e == k] = n
    return out


def three_cycles(e):
    """Geometry brought to three cycles.

    Random draw of the pattern of each cycle (high or low phase first) by the engine only from
    three cycles on; with one or two cycles, a high phase at every switch-on and a level
    overestimated by 7 to 81 %. New cycle made of the second range of a two-range cycle, or
    otherwise of the second half of the longest range."""
    e = e.copy()
    while e.max() < 3:
        _, ranges = geometry(e)
        new = e.max() + 1
        double = [k for k, p in enumerate(ranges, start=1) if len(p) == 2]
        if double:
            a, b = ranges[double[0] - 1][1]
        else:
            a, b = max((p for rg in ranges for p in rg), key=lambda p: p[1] - p[0])
            if b - a < 30:
                break
            a = (a + b) // 30 * 15
        e[a // 15:b // 15] = new
    return e


def geometry(e):
    """Windows and ranges (minutes) of each cycle, read from the labels."""
    # Ranges: runs of the same non-zero label
    ranges, a = {}, 0
    for i in range(1, 97):
        if i == 96 or e[i] != e[a]:
            if e[a] > 0:
                ranges.setdefault(int(e[a]), []).append([15 * a, 15 * i])
            a = i
    # Windows: runs of non-zero labels, whatever the cycle
    on = np.r_[False, e > 0, False]
    starts, ends = np.flatnonzero(~on[:-1] & on[1:]), np.flatnonzero(on[:-1] & ~on[1:])
    windows = [[15 * int(x), 15 * int(y)] for x, y in zip(starts, ends)]
    return windows, [ranges[k] for k in sorted(ranges)]


def valid(e):
    """Labels meeting the RAMP rules: two ranges per cycle, three windows."""
    windows, ranges = geometry(e)
    return 0 < len(windows) <= 3 and all(len(p) <= 2 for p in ranges)


# --- selection on the output of the engine ---------------------------------------
def objective_terms(y, profile, judge):
    """Terms of J: NRMSE, 0.5 x worst one-hour smoothed gap, local shape beyond its noise
    threshold, energy gap beyond 5 %, and the peak-time term for a setting with a weight on it.

    judge: local standard deviations, local shape thresholds and marked peaks of the target, and
    weight of the peak-time term (zero in the final setting, with the peak time kept for the
    validation only)."""
    m = S.metrics(y, profile)
    loc = V.local_shape(y, profile, judge["sd_1h"], judge["sd_2h"])
    excess = sum(max(0.0, loc[k] / judge["seuils"][k] - 1) for k in ("ELM_1h", "ELM_2h"))
    terms = {"NRMSE": m["NRMSE"], "pire_ecart_1h": PEL_WEIGHT * loc["PEL_1h"], "forme_locale": 0.1 * excess}
    if judge.get("peak_weight", 0.0) > 0:
        terms["heure_pics"] = judge["peak_weight"] * V.peak_time_gap_h(judge["pics"], profile)
    terms["energie"] = 0.1 * max(0.0, abs(m["err_E_pct"]) - ENERGY_TOLERANCE)
    return terms


def total(terms):
    """Sum of the terms of J.

    Addition term after term in the 2026-09-19 setting; sum() in the final setting, with its
    compensation of rounding errors since Python 3.12. Both ways kept, for the published
    results of each setting bit for bit."""
    if "heure_pics" not in terms:
        return sum(terms.values())
    j = 0.0
    for v in terms.values():
        j += v
    return j


def objective(y, profile, judge):
    """J = total of the terms of objective_terms."""
    return total(objective_terms(y, profile, judge))


def settings_of(e, L, share, var_w, continuous, occ):
    """RAMP settings of a geometry, with the labels in the declaration order of the cycles."""
    windows, ranges = geometry(e)
    span = sum(b - a for a, b in windows)
    return {"etiquettes": e, "fenetres": windows, "plages": ranges, "func_cycle": L,
            "func_time": int(min(span, max(L, round(share * span)))), "var_w": var_w,
            "continu": continuous, "occ": occ}


def evaluate(app, y, r, cache, judge):
    """Occupancy of each cycle from the engine, levels by least squares bounded by the nameplate."""
    key = (tuple(r["etiquettes"]), r["func_cycle"], r["func_time"], r["var_w"], r["continu"], r["occ"])
    if key not in cache:
        # One engine run per cycle, with a 1 W appliance switched on in that cycle only
        L = r["func_cycle"]
        cols = []
        for k in range(len(r["plages"])):
            cyc = [{"p1": float(i == k), "p2": float(i == k), "t1": L // 2, "t2": L - L // 2}
                   for i in range(len(r["plages"]))]
            cols.append(simulate(declaration(app, r, cyc, unit=True), SEED_UNIT, 10))
        # Levels of the cycles: least squares on the occupancy profiles, within (0, nameplate)
        B = np.column_stack(cols)
        levels = lsq_linear(B, y, bounds=(0, app.nameplate)).x
        cache[key] = dict(r, B=B, niveaux=levels, J=objective(y, B @ levels, judge))
    return cache[key]


def candidates(app, y, geometries, cache, judge, durations):
    """For each geometry: six orders of the cycles, then a short grid on the two best orders."""
    out = []
    for e, occ in geometries:
        # Six orders of the cycles, ranked on the base settings
        n = int(e.max())
        permuted = []
        for order in itertools.permutations(range(1, n + 1)):
            e2 = np.zeros_like(e)
            for rank, k in enumerate(order, start=1):
                e2[e == k] = rank
            permuted.append(e2)
        base = [evaluate(app, y, settings_of(e2, 15, 1.0, 0.0, app.modes()[0], occ), cache, judge)
                for e2 in permuted]
        # Short grid of settings on the two best orders
        for c in sorted(base, key=lambda c: c["J"])[:ORDERS_KEPT]:
            for L, share, var_w, continuous in itertools.product(durations, WINDOW_SHARES,
                                                                 VARIABILITY, app.modes()):
                r = settings_of(c["etiquettes"], L, share, var_w, continuous, occ)
                if L <= sum(b - a for a, b in r["fenetres"]):
                    out.append(evaluate(app, y, r, cache, judge))
        out += base
    # Removal of the duplicates coming from the cache, then ranking on J
    unique = {id(c): c for c in out}
    return sorted(unique.values(), key=lambda c: c["J"])


# --- realisation at the nameplate ------------------------------------------------
def nameplate_cycles(app, d, t1_high, period, p2):
    """t1, t2 and p2 of each cycle for duty cycles d, with p1 = nameplate.

    t1 of the highest cycle from the median daily peak; t1 of the other cycles as duty cycle x
    period; t2 from the duty cycle."""
    high = int(np.argmax(d))
    t1_max = max(t1_high, int(np.ceil(d[high] / (1 - d[high]))))
    cycles = []
    for k, dk in enumerate(d):
        t1 = t1_max if k == high else int(np.clip(round(dk * period), 1, t1_max))
        t2 = int(np.clip(round(t1 * (1 - dk) / dk), 1, 1440))
        cycles.append({"t1": t1, "t2": t2, "p2": p2, "d": float(dk)})
    return cycles


def zones(r):
    """Slots of each cycle, in the declaration order."""
    return [np.flatnonzero(r["etiquettes"] == k) for k in range(1, len(r["plages"]) + 1)]


def fixed_point(app, y, r, cycles, t1_high, period, p2, iterations, seed):
    """Correction of the duty cycles on the engine, up to a simulated mean of each zone equal to
    the target mean. Ratio bounded to (0.3, 3) at each iteration, duty cycle to (0.001, 0.95)."""
    Z = zones(r)
    target = np.array([y[z].mean() for z in Z])
    d = np.array([c["d"] for c in cycles])
    for it in range(iterations):
        sim = simulate(declaration(app, r, cycles), seed + it, 4)
        obs = np.array([sim[z].mean() for z in Z])
        occupied = np.array([r["B"].sum(axis=1)[z].mean() for z in Z]) if "B" in r else np.full(len(Z), r["occ"])
        d = np.clip(d * np.clip((target - p2 * occupied) / np.maximum(obs - p2 * occupied, 1e-6), 0.3, 3.0),
                    1e-3, 0.95)
        cycles = nameplate_cycles(app, d, t1_high, period, p2)
    return cycles


def periods(c, short_period):
    """Periods in competition: the long one of the engine audit, plus the short one if enabled."""
    long_period = 30 if c["continu"] == 1 else 60
    return (long_period, SHORT_PERIOD) if short_period else (long_period,)


def realise(app, y, c, median_peak, period):
    """First realisation of a candidate at the nameplate for one period, then the fixed point."""
    # Standby bounded by 0.8 x the lowest level, duty cycles from the levels, t1 of the high cycle
    p2 = float(min(app.standby, 0.8 * max(c["niveaux"].min(), 0.0)))
    d = np.clip((c["niveaux"] - p2) / (app.nameplate - p2), 1e-3, 0.95)
    t1_high = int(np.clip(round(15 * (median_peak - p2) / (app.nameplate - p2)), 1, 15))
    cycles = nameplate_cycles(app, d, t1_high, period, p2)
    cycles = fixed_point(app, y, c, cycles, t1_high, period, p2, N_FIXED_POINT, SEED_FIXED)
    return dict(c, cycles=cycles, t1_haut=t1_high, L_etoile=period, p2=p2)


def borders(e):
    """Positions of the changes of label."""
    return [i for i in range(1, 96) if e[i] != e[i - 1]]


def refine_borders(app, y, c, judge, passes=3):
    """Shift of each border of a range or window by one slot, kept for a lower J.

    At most three passes; after them, two more iterations of the fixed point."""
    J = objective(y, simulate(declaration(app, c, c["cycles"]), SEED_BORDERS, 4), judge)
    for _ in range(passes):
        gain = False
        for i in borders(c["etiquettes"]):
            # Shift to the left, then to the right, of the border at slot i
            for left in (True, False):
                e2 = c["etiquettes"].copy()
                if left:
                    e2[i] = e2[i - 1]
                else:
                    e2[i - 1] = e2[i]
                if set(e2.tolist()) != set(c["etiquettes"].tolist()) or not valid(e2):
                    continue
                r2 = dict(c, **settings_of(e2, c["func_cycle"], 1.0, c["var_w"], c["continu"], c["occ"]))
                r2["func_time"] = int(min(sum(b - a for a, b in r2["fenetres"]), c["func_time"]))
                J2 = objective(y, simulate(declaration(app, r2, c["cycles"]), SEED_BORDERS, 4), judge)
                if J2 < J - 1e-4:
                    c, J, gain = r2, J2, True
                    break
        if not gain:
            break
    c.pop("B", None)
    c["cycles"] = fixed_point(app, y, c, c["cycles"], c["t1_haut"], c["L_etoile"], c["p2"], 2, SEED_FIXED + 10)
    return c


# --- realism check (outside the contract) ----------------------------------------
def realism(X, Sd):
    """Daily peaks and 15-minute values, measured (X) against simulated (Sd)."""
    return {"pointe_P50_mes": float(np.median(X.max(axis=1))), "pointe_P50_sim": float(np.median(Sd.max(axis=1))),
            "pointe_P95_mes": float(np.percentile(X.max(axis=1), 95)),
            "pointe_P95_sim": float(np.percentile(Sd.max(axis=1), 95)),
            "quart_P99_mes": float(np.percentile(X, 99)), "quart_P99_sim": float(np.percentile(Sd, 99)),
            "part_quarts_sim_au_dessus_max_mes": float(np.mean(Sd > X.max()))}


# --- complete chain -----------------------------------------------------------------
def calibrate(app, X, running, noise, p95, peak_weight=0.0, short_period=True, durations=DURATIONS,
              log=print):
    """Calibration of one target. X: retained days; running: flag of the days with the appliance running."""
    # Occasional use, and the profiles read: running days only (if some days without running), all days
    y = X.mean(axis=0)
    occ = float(running.mean()) if running.mean() < 0.95 else 1.0
    readings = [(X[running].mean(axis=0), occ)] if occ < 1 else []
    readings.append((y, 1.0))
    # Geometries by plain least squares, and weighted by the inverse of the noise variance,
    # in line with the local shape test and its gaps relative to the noise of the slot
    weights = [np.ones(96), 1.0 / np.maximum(noise["sd_1h"], 1e-9) ** 2]
    geometries, seen = [], set()
    for profile, o in readings:
        for w in weights:
            for f in N_WINDOWS:
                e = three_cycles(renumber(optimal_staircase(profile, w / w.mean(), 3, f)))
                if valid(e) and (tuple(e), o) not in seen:
                    seen.add((tuple(e), o))
                    geometries.append((e, o))
    log(f"   {len(geometries)} geometries")
    # Ranking of every candidate on J
    judge = {"sd_1h": noise["sd_1h"], "sd_2h": noise["sd_2h"], "pics": noise["pics"],
             "seuils": V.local_thresholds(p95), "peak_weight": peak_weight}
    cache = {}
    ranked = candidates(app, y, geometries, cache, judge, durations)
    log(f"   {len(ranked)} candidates, J {ranked[0]['J']:.3f} to {ranked[-1]['J']:.3f}")

    # Realisation at the nameplate of the five best candidates, for each period in competition
    active = X[running] if running.any() else X
    median_peak = float(np.median(active.max(axis=1)))
    finalists, trace = [], []
    for rank, candidate in enumerate(ranked[:N_REALISED], start=1):
        for period in periods(candidate, short_period):
            c = refine_borders(app, y, realise(app, y, dict(candidate), median_peak, period), judge)
            s = simulate(declaration(app, c, c["cycles"]), SEED_ARBITRATION, 20)
            terms = objective_terms(y, s, judge)
            c["J_arbitrage"] = total(terms)
            finalists.append(c)
            # Trace of every finalist, for the reading of the selection and for step 6
            trace.append({"rang_candidat": rank, "L_etoile": period, "J_arbitrage": c["J_arbitrage"],
                          "termes": terms, "ecart_pics_h": V.peak_time_gap_h(judge["pics"], s),
                          "fenetres": c["fenetres"], "plages": c["plages"],
                          "cycles": [{"t1": cy["t1"], "t2": cy["t2"], "d": cy["d"]} for cy in c["cycles"]],
                          "p2": c["p2"], "t1_haut": c["t1_haut"], "func_time": c["func_time"],
                          "func_cycle": c["func_cycle"], "var_w": c["var_w"], "continu": c["continu"],
                          "etiquettes": [int(v) for v in c["etiquettes"]],
                          "texte_ramp": declaration(app, c, c["cycles"]), "profil": s.tolist()})
            log(f"   finalist period {period} min, arbitration J {c['J_arbitrage']:.3f}")
    choice = min(finalists, key=lambda c: c["J_arbitrage"])

    # Check of the chosen finalist with two other seeds, then verdict of the contract
    text = declaration(app, choice, choice["cycles"])
    Sd = simulate(text, SEEDS_CHECK[0], 20, days=True)
    profile = Sd.mean(axis=0)
    measures = V.criteria(y, profile, noise["pics"])
    local = V.local_shape(y, profile, noise["sd_1h"], noise["sd_2h"])
    verdict, missed = V.verdict(measures, local, p95)
    other = simulate(text, SEEDS_CHECK[1], 20)
    return {"texte": text, "profil": profile, "choix": choice, "mesures": measures, "locale": local,
            "verdict": verdict, "manques": missed, "realisme": realism(X, Sd),
            "mesures_98765": V.criteria(y, other, noise["pics"]), "pointe_mediane_actifs": median_peak,
            "occ": occ, "n_geometries": len(geometries), "n_candidats": len(ranked), "finalistes": trace}
