"""Step 1: conversion of the meter series of each client into complete days of 96 slots.

Cleaning rules, in this order:
  - removal of the integration periods longer than 15 minutes;
  - empty slots under a voltage below 200 V, the power being unknown there, not zero;
  - linear interpolation of the gaps of at most 30 minutes (two slots);
  - no filling and no zero for a longer gap: rejection of the whole day;
  - start of the study at the commissioning of the site, or at the arrival of the appliance
    when later;
  - flag of site-wide rejection for a day rejected at every client of the same site.

Outputs in resultats/cibles_v9/:
  journees/<client>.csv          one row per retained day, 96 slot columns
  registres/<client>_jours.csv   fate of every day and reason for a rejection
  registres/<client>_mois.csv    monthly summary
  bilan_mensuel.csv
"""
import numpy as np
import pandas as pd
import settings as S

# Output folders of the daily profiles and of the registers
OUT = S.TARGETS
for d in ("journees", "registres"):
    (OUT / d).mkdir(parents=True, exist_ok=True)
LOCAL = "Etc/GMT-1"                    # Benin time zone, UTC+1
BEFORE_START = "avant le debut d'etude"


def study_start(client):
    """First studied day: commissioning of the site, or arrival of the appliance if later."""
    site = S.COMMISSIONING[client[-3:]]
    return max(site, S.CLIENT_START.get(client, site))


def fill_short_gaps(grid):
    """Interpolation of the gaps of at most 30 minutes, longer gaps left empty."""
    limit = S.GAP_MAX_MIN // S.SLOT_MIN
    gap = grid.isna()
    length = gap.groupby((gap != gap.shift()).cumsum()).transform("sum")
    return grid.interpolate(limit_area="inside").where(~gap | (length <= limit))


def client_days(client):
    """Retained days of a client, and the register of every day with the reason of a rejection."""
    # Meter series in local time, without long integration periods nor duplicates
    d = pd.read_csv(S.DATA / S.CLIENTS[client] / client / f"{client}.csv", low_memory=False)
    d["t"] = pd.to_datetime(d["t"], utc=True).dt.tz_convert(LOCAL)
    d = d[(d["duree_s"].isna()) | (d["duree_s"] <= S.PERIOD_MAX_S)]
    d = d.sort_values("t").drop_duplicates(subset="t").set_index("t")
    p = pd.to_numeric(d["true_power_avg"], errors="coerce")
    v = pd.to_numeric(d["voltage_avg"], errors="coerce")
    # Removal of the periods without supply: unknown power there, not a zero power
    off = v.notna() & (v < S.VOLTAGE_OUT)
    p = p.mask(off)
    # 15-minute grid, then interpolation of the short gaps only
    grid = p.resample(f"{S.SLOT_MIN}min").mean()
    off_slots = off.resample(f"{S.SLOT_MIN}min").max().fillna(False)
    filled = fill_short_gaps(grid)
    start = study_start(client)
    days, register = [], []
    # Day-by-day decision, with the reason of each rejection kept in the register
    for day, block in filled.groupby(filled.index.normalize()):
        b = block.reindex(pd.date_range(day, periods=S.SLOTS_PER_DAY,
                                        freq=f"{S.SLOT_MIN}min", tz=LOCAL))
        missing = int(b.isna().sum())
        holes = b.isna().astype(int)
        worst = int((holes.groupby((holes != holes.shift()).cumsum())
                     .transform("sum") * holes).max() or 0)
        n_off = int(off_slots.reindex(b.index, fill_value=False).sum())
        if str(day.date()) < start:
            reason = BEFORE_START
        elif S.SLOTS_PER_DAY - missing < S.MIN_VALID_SLOTS:
            reason = "moins de 90 creneaux valides"
        elif missing > 0:
            reason = "trou de plus de 30 min"
        else:
            reason = ""
        kept = reason == ""
        vals = b.to_numpy(float)
        register.append({"jour": str(day.date()), "mois": day.strftime("%Y-%m"),
                         "creneaux_manquants": missing, "pire_trou": worst,
                         "creneaux_en_coupure": n_off, "garde": kept,
                         "actif": bool(kept and np.nansum(vals) > 0), "motif": reason,
                         "energie_kWh": float(np.nansum(vals) * 0.25 / 1000),
                         "pointe_W": float(np.nanmax(vals)) if kept else np.nan})
        if kept:
            days.append([str(day.date()), day.strftime("%Y-%m")] + list(vals))
    cols = ["jour", "mois"] + [f"slot_{i}" for i in range(S.SLOTS_PER_DAY)]
    return pd.DataFrame(days, columns=cols), pd.DataFrame(register)


# Daily profiles and register of every client
days, registers = {}, {}
for client in S.CLIENTS:
    days[client], registers[client] = client_days(client)

# Flag of the days rejected at every studied client of the same site on the same date
for site in ("SAM", "GBO"):
    clients = [c for c in S.CLIENTS if c.endswith(site)]
    tab = pd.concat([registers[c] for c in clients])
    tab = tab[tab.motif != BEFORE_START]
    common = (~tab.garde).groupby(tab.jour).all()
    for c in clients:
        r = registers[c]
        r["rejet_commun_site"] = (~r.garde & (r.motif != BEFORE_START)
                                  & r.jour.map(common).eq(True))

# Files per client, monthly summary and progress line
summary = []
for client, family in S.CLIENTS.items():
    dy, reg = days[client], registers[client]
    dy.to_csv(OUT / "journees" / f"{client}.csv", index=False)
    reg.to_csv(OUT / "registres" / f"{client}_jours.csv", index=False)
    studied = reg[reg.motif != BEFORE_START]
    month = (studied.groupby("mois")
                    .agg(jours_calendaires=("jour", "size"),
                         jours_propres=("garde", "sum"),
                         jours_actifs=("actif", "sum"),
                         energie_kWh=("energie_kWh", "sum"),
                         creneaux_en_coupure=("creneaux_en_coupure", "sum"))
                    .reset_index())
    month["occasional_use"] = month.jours_actifs / month.jours_propres.replace(0, np.nan)
    month["signale_mois_pauvre"] = month.jours_propres < S.MIN_DAYS_FLAG
    month["client"], month["famille"] = client, family
    month.to_csv(OUT / "registres" / f"{client}_mois.csv", index=False)
    summary.append(month)
    rejected = studied[~studied.garde]
    print(f"{client:8s} start {study_start(client)}  {len(studied):>4d} studied days  "
          f"{int(studied.garde.sum()):>4d} kept  rejected {100 * len(rejected) / len(studied):4.1f} %  "
          f"(site-wide {100 * rejected.rejet_commun_site.mean():3.0f} %)", flush=True)

total = pd.concat(summary, ignore_index=True)
total.to_csv(OUT / "bilan_mensuel.csv", index=False)
print("\ntotal:", len(total), "client-month pairs;", int(total.signale_mois_pauvre.sum()), "flagged as poor")
