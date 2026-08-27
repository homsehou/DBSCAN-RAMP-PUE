# -*- coding: utf-8 -*-
"""Step 6 - Engine verification and damped correction (methodology step 9).

Two arms compete for every season, both judged by the untouched RAMP
engine (continuous mode, no midnight dip by construction) at the
evaluation budget. The classic arm is the historic step 9 flow: the
step 4 inversion approached by free mean corrections, then polished by
the bounded damped ratio; day-to-day spread through the native time
variability, days without activity as occasional_use. The screened arm
starts from the step 5 shortlist - which may prefer another measured
burst power to the anchor: the step 10 rule arbitrates its two
variability branches at the evaluation budget, then the same damped
ratio corrects it under a guard - a correction is kept only when the
criteria improve, so the search gains cannot be written off. The arm
passing more criteria delivers the season (ties on the remaining
distance); the model chosen on even days is kept for the examination
(step 7) and every calibrated native parameter is written to
appliance_parameters.csv.
Run: PUE_TYPE=cold_chain PUE_CLIENT=0017SAM python step6_correct.py
"""
import io, json, os, sys
import random as rnd
from contextlib import redirect_stdout
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C   # shared settings of the repository
PUE_TYPE = os.environ.get("PUE_TYPE", "cold_chain")
CLIENT = os.environ.get("PUE_CLIENT", "0017SAM")
REPO = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO / "resultats" / PUE_TYPE / CLIENT
sys.path.insert(0, str(REPO))    # embedded ramp/ package importable


def region_means(profile, regions):
    """Mean of a 96-slot profile inside each region (wrap handled)."""
    return [float(profile[np.arange(g["start"] // C.SLOT_MIN,
                                    g["end"] // C.SLOT_MIN) % 96].mean())
            for g in regions]


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
                name="appliance", number=1, power=p1 - p2,   # the duty
                                                             # cycles carry
                                                             # the real
                                                             # powers
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


def correct_season(sd, target_sel, target_full, target_hold, short):
    """Both arms of one season; the better delivered model wins."""
    p1, p2, meas_p75 = sd["p1_W"], sd["p2_W"], sd["meas_p75_W"]
    for key in ("p1_candidates_W", "p2_candidates_W"):
        sd.pop(key, None)               # travelled to step 5, not exported

    def fit(target, tfrv, n_polish, regions0, cycle_len, classic, cap,
            p1=p1, p2=p2, p_var=C.RAMP_VARIABILITY):
        """One model fit: cheap approach rounds, then damped polish at
        the evaluation budget. classic=True is the historic loop (mean
        corrections free); classic=False starts from a screened
        candidate and keeps a correction only when the criteria improve
        and the shape stays under the cap."""
        regions = [dict(g) for g in regions0]
        len_floor = int(np.ceil(2.0 / max(min(g["duty"] for g in regions),
                                          0.01)))
        min_len = min(g["end"] - g["start"] for g in regions) // 2
        want = region_means(target, regions)
        def corrected(r, damp, regs):
            for g, w, got in zip(regs, want, region_means(r["sim"], regs)):
                ratio = np.clip((w - p2) / max(got - p2, 1e-3),
                                C.CORRECT_LOW, C.CORRECT_HIGH)
                g["duty"] = float(np.clip(g["duty"] * ratio ** damp,
                                          2.0 / cycle_len, 0.98))
        # Cheap approach rounds, classic arm only: a screened candidate
        # already carries a cycle and a burst power the six criteria
        # picked and the engine verified, and chasing the P75 amplitude
        # band would undo that choice
        for _ in range(C.FIXED_POINT_ROUNDS if classic else 0):
            r = simulate(p1, p2, cycle_len, regions, tfrv, C.TUNE_SEEDS,
                         C.TUNE_DAYS, C.SEED, p_var)
            amp = r["dmax"] / meas_p75 if meas_p75 > 0 else 1.0
            amp_ok = C.AMP_LOW <= amp <= C.AMP_HIGH
            if amp_ok and C.n_passed(C.metrics(target, r["sim"])) == 6:
                break
            if not amp_ok:
                # Too strong a daily maximum = bursts too long: shrink
                # the cycle (ON parts scale with it), and conversely -
                # never below the representability floor
                cycle_len = int(np.clip(round(cycle_len / amp),
                                        max(6, len_floor),
                                        max(min_len, 6)))
            corrected(r, 1.0, regions)
        r = simulate(p1, p2, cycle_len, regions, tfrv, C.EVAL_SEEDS,
                     C.EVAL_DAYS, C.EVAL_SEED, p_var)
        m, peak = C.metrics(target, r["sim"]), int(np.argmax(target))
        n_rescue = 0
        for i in range(n_polish):       # evaluation-grade rounds, damped
            if (C.n_passed(m) == 6 and m["NRMSE"] <= C.POLISH_NRMSE
                    and abs(r["sim"][peak] - target[peak])
                    <= C.POLISH_PEAK_TOL * target[peak]):
                break
            n_rescue += 1
            damp = C.DAMP_FIRST if i < C.DAMP_SWITCH else C.DAMP_LATE
            if classic:
                corrected(r, damp, regions)
                r = simulate(p1, p2, cycle_len, regions, tfrv,
                             C.EVAL_SEEDS, C.EVAL_DAYS, C.EVAL_SEED,
                             p_var)
                m = C.metrics(target, r["sim"])
            else:
                cand = [dict(g) for g in regions]
                corrected(r, damp, cand)
                r2 = simulate(p1, p2, cycle_len, cand, tfrv,
                              C.EVAL_SEEDS, C.EVAL_DAYS, C.EVAL_SEED,
                              p_var)
                m2 = C.metrics(target, r2["sim"])
                if ((-C.n_passed(m2), C.exceedance(m2))
                        < (-C.n_passed(m), C.exceedance(m))
                        and m2["NRMSE"] <= max(cap, m["NRMSE"])):
                    regions, r, m = cand, r2, m2
                else:
                    break               # a screened start never worsens
        r.update({"regions": regions, "m": m, "n_rescue": n_rescue,
                  "cycle_len": cycle_len, "tfrv": tfrv, "p1": p1,
                  "p2": p2, "p_var": p_var})
        return r

    # Classic arm - the historic flow, numerically unchanged.
    # Cycle length: the burst anchor asks duty_max x L = t1_amp; the
    # engine's 2-min ON floor asks L >= 2/duty_min - the floor wins
    for g in sd["regions"]:
        g.setdefault("power", p1)    # classic arm: one power for all
    duties = [g["duty"] for g in sd["regions"]]
    min_len = min(g["end"] - g["start"] for g in sd["regions"]) // 2
    len_floor = int(np.ceil(2.0 / max(min(duties), 0.01)))
    L = int(np.clip(max(round(sd["t1_amp_min"] / max(duties)), len_floor),
                    sd["t1_amp_min"] + 2, max(min_len,
                                              sd["t1_amp_min"] + 2)))
    no_cap = float("inf")
    sel = fit(target_sel, 0.0, C.POLISH_SELECT, sd["regions"], L, True,
              no_cap)
    L, tfrv = sel["cycle_len"], 0.0
    cv0 = float(np.std(sel["dE"]) / np.mean(sel["dE"]))
    cv_a = float(min(1.0, np.sqrt(max(sd["cv_E_meas"] ** 2 - cv0 ** 2,
                                      0.0))))
    if np.isfinite(cv_a) and cv_a >= C.ACTIVITY_CV_MIN:
        cand = fit(target_sel, C.tfrv_from(cv_a), C.POLISH_SELECT,
                   sd["regions"], L, True, no_cap)
        L = cand["cycle_len"]
        # Step 10 rule: the measured spread is kept unless the shape
        # degrades - as many criteria passed, NRMSE within the margin
        if C.n_passed(cand["m"]) >= C.n_passed(sel["m"]) \
                and cand["m"]["NRMSE"] <= max(
                    C.ACTIVITY_NRMSE_CAP,
                    sel["m"]["NRMSE"] + C.ACTIVITY_NRMSE_MARGIN):
            tfrv, sel = C.tfrv_from(cv_a), cand
    d = fit(target_full, tfrv, C.POLISH_DELIVERED, sd["regions"], L, True,
            no_cap)
    L = d["cycle_len"]
    if C.n_passed(d["m"]) < 6 and tfrv > 0.0:
        cand = fit(target_full, 0.0, C.POLISH_DELIVERED, sd["regions"], L,
                   True, no_cap)
        L = cand["cycle_len"]
        if C.n_passed(cand["m"]) > C.n_passed(d["m"]):
            tfrv, d = 0.0, cand   # dropped only when strictly better

    # Screened arm - step 10 rule between the two branches at the
    # evaluation budget, then the guarded polish of the winner
    def screened(kind, target, n_polish, cap):
        got = {}
        for name, e in short[kind].items():
            if e:
                r = simulate(e["p1"], e["p2"], e["L"], e["regions"],
                             e["tfrv"], C.EVAL_SEEDS, C.EVAL_DAYS,
                             C.EVAL_SEED, e["p_var"])
                got[name] = {**e, "m": C.metrics(target, r["sim"])}
        pick = got.get("zero") or got.get("act")
        if "zero" in got and "act" in got \
                and C.n_passed(got["act"]["m"]) >= C.n_passed(
                    got["zero"]["m"]) \
                and got["act"]["m"]["NRMSE"] <= max(
                    C.ACTIVITY_NRMSE_CAP,
                    got["zero"]["m"]["NRMSE"] + C.ACTIVITY_NRMSE_MARGIN):
            pick = got["act"]
        return fit(target, pick["tfrv"], n_polish, pick["regions"],
                   pick["L"], False, cap, pick["p1"], pick["p2"],
                   pick["p_var"])

    sel_s = screened("select", target_sel, C.POLISH_SELECT,
                     short["cap_select"])
    d_s = screened("full", target_full, C.POLISH_DELIVERED,
                   short["cap_full"])
    # The screened arm delivers only if it passes more criteria AND
    # holds out of sample: freeing the powers is only defensible while
    # the odd days the model never saw agree with the even ones
    better = ((-C.n_passed(d_s["m"]), C.exceedance(d_s["m"]))
              < (-C.n_passed(d["m"]), C.exceedance(d["m"])))
    veto = False
    if better and C.HOLDOUT_VETO and sd["holdout_valid"]:
        h_c = C.metrics(target_hold, sel["sim"])
        h_s = C.metrics(target_hold, sel_s["sim"])
        veto = ((-C.n_passed(h_s), C.exceedance(h_s))
                > (-C.n_passed(h_c), C.exceedance(h_c)))
    arm = "classic"
    if better and not veto:
        arm, sel, d = "screened", sel_s, d_s
        tfrv, L = d_s["tfrv"], d_s["cycle_len"]

    duty_max = max(g["duty"] for g in d["regions"])
    sd.update({
        "p1_W": d["p1"], "p1_anchor_W": p1,
        "p2_W": d["p2"], "p2_anchor_W": p2,
        "thermal_p_var": d["p_var"], "holdout_veto": bool(veto),
        "t1_amp_min": int(np.clip(round(duty_max * L), 2, L - 2)),
        "tfrv": tfrv, "activity_cv": cv_a, "n_rescue": d["n_rescue"],
        "method_arm": arm,
        "ratio_daily_max": float(d["dmax"] / meas_p75) if meas_p75
        else 1.0,
        "cv_E_sim": float(np.std(d["dE"]) / np.mean(d["dE"])),
        "native": {"func_time_min": sum(g["end"] - g["start"]
                                        for g in d["regions"]),
                   "cycle_len_min": int(L),
                   "tfrv": tfrv, "occasional_use": sd["occasional_use"],
                   "regions": [{"start_min": g["start"],
                                "end_min": g["end"],
                                "power_W": round(float(g.get("power",
                                                             d["p1"])), 1),
                                "duty": round(g["duty"], 4),
                                "t_on_min": int(np.clip(
                                    round(g["duty"] * L), 2, L - 2))}
                               for g in d["regions"]]}})
    return sd, d, sel


def main():
    init = json.loads((OUT_DIR / "calibration_init.json").read_text())
    short_all = json.loads((OUT_DIR / "search_shortlist.json").read_text())
    targets = pd.read_csv(OUT_DIR / "calibration_targets.csv")
    targets = targets[targets["kind"] != "sim_select"]   # rerun-safe
    slots = [f"slot_{j}" for j in range(C.SLOTS_PER_DAY)]
    def target_of(season, kind):
        r = targets[(targets["season"] == season)
                    & (targets["kind"] == kind)]
        return r[slots].to_numpy(float)[0]
    export = {k: v for k, v in init.items() if k != "seasons"}
    export.update({"method": "native_surrogate_search",
                   "engine": "RAMP continuous, untouched", "seasons": {}})
    sims, extra_rows, seed_rows, params = [], [], [], []
    for season, sd in init["seasons"].items():
        sd, d, sel = correct_season(sd, target_of(season, "select"),
                                    target_of(season, "full"),
                                    target_of(season, "holdout"),
                                    short_all[season])
        sd.pop("regions")               # superseded by native["regions"]
        export["seasons"][season] = sd
        sims.append({"season": season,
                     **{s: float(v) for s, v in zip(slots, d["sim"])}})
        extra_rows.append({"season": season, "kind": "sim_select",
                           **{s: float(v)
                              for s, v in zip(slots, sel["sim"])}})
        seed_rows += [{"season": season, "seed": i,
                       **{s: float(v) for s, v in zip(slots, prof)}}
                      for i, prof in enumerate(d["seeds"])]
        nat = sd["native"]
        row = {"season": season, "method_arm": sd["method_arm"],
               "p1_W": round(sd["p1_W"], 1),
               "p1_anchor_W": round(sd["p1_anchor_W"], 1),
               "p2_W": round(sd["p2_W"], 2),
               "p2_anchor_W": round(sd["p2_anchor_W"], 2),
               "n_regions": len(nat["regions"]),
               "cycle_len_min": nat["cycle_len_min"],
               "func_time_min": nat["func_time_min"],
               "func_cycle_min": nat["cycle_len_min"],
               "thermal_p_var": sd["thermal_p_var"],
               "tfrv": round(nat["tfrv"], 3),
               "occasional_use": nat["occasional_use"]}
        for i, g in enumerate(nat["regions"], start=1):
            row.update({
                f"region{i}_window": f"{g['start_min']}-{g['end_min']}",
                f"region{i}_power_W": g["power_W"],
                f"region{i}_duty": g["duty"],
                f"region{i}_t_on_min": g["t_on_min"],
                f"region{i}_t_off_min": nat["cycle_len_min"]
                - g["t_on_min"]})
        params.append(row)
        print(f"[step6] {season}: {sd['method_arm']}  "
              f"p1 {sd['p1_W']:.0f}W (ancre {sd['p1_anchor_W']:.0f}W)  "
              f"p2 {sd['p2_W']:.1f}W  var {sd['thermal_p_var']:.2f}  "
              f"{'VETO ' if sd['holdout_veto'] else ''}"
              f"{len(nat['regions'])} regions  L {nat['cycle_len_min']} "
              f"min  tfrv {sd['tfrv']:.2f}  "
              f"amp {sd['ratio_daily_max']:.2f}  "
              f"rescue {sd['n_rescue']}")
    pd.concat([targets, pd.DataFrame(extra_rows)]).to_csv(
        OUT_DIR / "calibration_targets.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(OUT_DIR / "sim_seed_profiles.csv",
                                   index=False)
    pd.DataFrame(sims).to_csv(OUT_DIR / "ramp_simulated_profiles.csv",
                              index=False)
    pd.DataFrame(params).to_csv(OUT_DIR / "appliance_parameters.csv",
                                index=False)
    (OUT_DIR / "calibration_export.json").write_text(json.dumps(export))


if __name__ == "__main__":
    main()
