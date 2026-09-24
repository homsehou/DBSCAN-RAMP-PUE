"""Step 3: running power and standby power of each appliance, measured at the meter.

Meter data per 15-minute slot: mean power and maximum current reached. Slot of continuous
running: mean current at 90 % of the maximum current at least, hence a mean power equal to the
actual power of the appliance at that time.

Two groups among those slots:
  - standby: median of the low group, split from the high group by Otsu's method on the
    logarithm of the power;
  - running: main peak of the histogram (bins of a tenth of a decade) above five times the
    standby power, then median of the slots within half a bin of that peak. Reason: a median
    of the whole high group mixing high standby and running (0017SAM: 28 W instead of 140 W).
Coefficient = running power / nameplate power. Below 50 slots of continuous running (0016GBO,
a mill never running a full quarter of an hour), no measured running power: median
coefficient of the family applied to the nameplate, with a flag.

Running power never entered in RAMP, used as a plausibility check only.
Output: resultats/cibles_v9/puissances_mesurees.csv (one row per client and fleet)
"""
import numpy as np
import pandas as pd
import settings as S

# Nameplate per fleet, with the sum of both nameplates for a two-appliance fleet
NAMEPLATE = dict(S.NAMEPLATE_W)
NAMEPLATE.update({(c, "depuis"): sum(p) for c, p in S.TWO_APPLIANCE_NAMEPLATES.items()})


def otsu_threshold(x, n_bins=64):
    """Threshold with the best separation of two groups (Otsu's method)."""
    h, edges = np.histogram(x, n_bins)
    centres = 0.5 * (edges[1:] + edges[:-1])
    w0, w1 = np.cumsum(h), np.cumsum(h[::-1])[::-1]
    m0 = np.cumsum(h * centres) / np.maximum(w0, 1)
    m1 = (np.cumsum((h * centres)[::-1]) / np.maximum(np.cumsum(h[::-1]), 1))[::-1]
    variance = w0[:-1] * w1[1:] * (m0[:-1] - m1[1:]) ** 2
    return edges[1:][np.argmax(variance)]


# Slots of continuous running of each client and fleet, then standby and running power
rows = []
for client in S.CLIENTS:
    d = pd.read_csv(S.DATA / S.CLIENTS[client] / client / f"{client}.csv",
                    usecols=["t", "true_power_avg", "current_max", "current_avg", "voltage_avg"])
    d["jour"] = (pd.to_datetime(d.t, utc=True, format="mixed")
                 .dt.tz_convert("Africa/Porto-Novo").dt.strftime("%Y-%m-%d"))
    d = d[(d.voltage_avg > S.VOLTAGE_OUT) & (d.true_power_avg > 0)]
    # Fleet configuration of each slot, and operating period only for the incubator
    if client in S.FLEET_CHANGE:
        d["config"] = np.where(d.jour < S.FLEET_CHANGE[client], "avant", "depuis")
    elif client == "0152GBO":
        start, end = S.INCUBATOR_ACTIVE
        d = d[(d.jour >= start) & (d.jour <= end)].assign(config="active")
    else:
        d["config"] = ""
    for config, g in d.groupby("config"):
        full = g[g.current_avg >= 0.9 * g.current_max].true_power_avg.to_numpy()
        standby = float(np.median(full[full < 10 ** otsu_threshold(np.log10(full))]))
        high = full[full >= 5 * standby]
        row = {"client": client, "config": config, "famille": S.CLIENTS[client],
               "quarts_heure_pleins": len(full), "veille_W": round(standby, 1),
               "plaque_W": NAMEPLATE.get((client, config), NAMEPLATE.get(client))}
        if len(high) >= 50:
            h, edges = np.histogram(np.log10(high), np.arange(np.log10(5 * standby), 4.3, 0.1))
            peak = edges[np.argmax(h)] + 0.05
            around = high[np.abs(np.log10(high) - peak) <= 0.05]
            row.update({"en_marche": len(around), "puissance_marche_W": round(float(np.median(around)), 1),
                        "marche_p95_W": round(float(np.percentile(high, 95)), 1), "source": "mesure"})
        else:
            row.update({"en_marche": len(high), "puissance_marche_W": np.nan,
                        "source": "coefficient de famille"})
        rows.append(row)

# Coefficient running power / nameplate, and median of the family where no measure exists
t = pd.DataFrame(rows)
t["coefficient"] = (t.puissance_marche_W / t.plaque_W).round(2)
family_coef = t[t.source == "mesure"].groupby("famille").coefficient.median()
missing = t.source != "mesure"
t.loc[missing, "coefficient"] = t.loc[missing, "famille"].map(family_coef)
t.loc[missing, "puissance_marche_W"] = (t.loc[missing, "coefficient"] * t.loc[missing, "plaque_W"]).round(1)
t.to_csv(S.TARGETS / "puissances_mesurees.csv", index=False)
print(t[["client", "config", "veille_W", "plaque_W", "puissance_marche_W", "coefficient", "source"]]
      .to_string(index=False))
