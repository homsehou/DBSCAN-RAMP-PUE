# DBSCAN-RAMP-PUE V3

Load profile modelling of productive uses of electricity (PUE) on two rural mini-grids in
Benin (Samionta and Gbowele). Smart-meter series of nine appliances (four cold-chain
appliances, four grain mills, one poultry incubator) are cut into daily profiles, the typical
days are selected with DBSCAN, and one RAMP appliance per client is calibrated so that its
simulated mean profile matches the measured one, season by season and then month by month.

Two rules hold throughout:

- the RAMP engine (`ramp/`) is used through its public interface and never modified;
- the power entered in RAMP (`power` and `p_i1`) is always the nameplate power from the field
  survey, never a power fitted to the meter. Every other parameter is fitted to the shape.

## Results

| | Seasonal targets | Monthly targets |
|---|---|---|
| Targets calibrated | 36 | 98 |
| Days behind them | 2 651 | 2 615 |
| Criteria met | 309 / 360 | 801 / 980 |
| NRMSE, mean (median, max) | 8.3 % (8.2 %, 15.7 %) | 9.0 % (8.6 %, 16.9 %) |
| Energy gap, mean of absolute values (max) | 2.4 % (17.1 %) | 1.7 % (5.8 %) |
| Verdicts: validated / rejected / not verifiable | 4 / 14 / 18 | 5 / 33 / 60 |

A target is *not verifiable* when its own days are too few or too scattered for any model,
even a perfect one, to be judged at the 10 % level (see the validation contract below). This is
the main limit of the monthly grain: half as many days per target as the seasonal one.

What transfers from one appliance to another (share of the variance carried by the client,
within a family; close to 0 the parameter belongs to the type of appliance, close to 1 to the
client): `func_cycle` 0.20 for the cold chain and 0.23 for the mills, cycle period 0.22 and
0.20; mean duty cycle 0.74 and 0.82, daily energy 0.62 and 0.79. The running duration is a
property of the machine, the level of use a property of the operator.

## Repository layout

```
├── data/<type>/<client>/<client>.csv      smart-meter series, one file per client
├── notebooks/<type>/<client>.ipynb        one notebook per client, shipped executed
├── ramp/                                  RAMP engine, untouched
├── resultats/
│   ├── cibles_v9/                         daily profiles, targets, noise, measured powers
│   ├── reference_saison/                  final seasonal results (36 targets)
│   └── calib_mois/                        monthly results (98 targets)
├── src/                                   the chain, one script per step
└── requirements.txt                       pinned library versions (Python 3.12)
```

Inside `reference_saison/` and `calib_mois/`:

- `<target>.json`: retained RAMP parameters, the ten criteria, the local shape, the thresholds
  and the verdict;
- `<target>_profil.npy`: the simulated mean profile (96 slots of 15 minutes);
- `entrees_ramp/<target>.py`: the RAMP declaration, as a user would write it; it runs on its own;
- `planches/<target>.png`: working plate of the target (criteria banner, profile, windows,
  gap against the noise, load duration curve, harmonics);
- `publication/figures/` and `publication/tableaux/`: publication figures and tables
  (Markdown and LaTeX).

Season names, month names, file names and result keys keep their French spelling
("Saison seche", "Mai", `cibles_saison.csv`, `manques`), as does the text of the figures:
they are the labels of the published results and of the method guides.

## Running

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name dbscan-ramp-pue-v3-github \
    --display-name "Python (DBSCAN-RAMP-PUE V3, GitHub)"   # kernel named in the notebooks
.venv/bin/jupyter lab                       # notebooks in the browser
.venv/bin/python src/run.py --prefix rejeu/ # whole chain, results in resultats/rejeu/
```

The notebooks read the committed results and replay the RAMP engine on the published input
files: each replay gives back the published profile exactly (36 seasonal targets checked).
The whole chain takes about fifteen hours on 15 processors; with `--prefix rejeu/` the
committed results stay untouched and the new ones land next to them for comparison. A run
that stops half-way resumes where it stopped (`--from` restarts at a given stage). Each step
also runs alone, for instance:

```
cd src
python step4_calibrate.py 0154GBO__Saison_des_pluies --run essai --setting final
```

## The chain

| Step | Script | What it does |
|---|---|---|
| 1 | `step1_daily_profiles.py` | Complete days of 96 slots: long integration periods discarded, voltage collapses left empty (power unknown, not zero), gaps up to 30 minutes interpolated, longer gaps reject the day |
| 2 | `step2_targets.py` | One target per client, season (or month) and fleet configuration; out-of-service periods removed; typical days kept by DBSCAN; noise of the mean by a block bootstrap of 7-day blocks |
| 3 | `step3_running_power.py` | Standby and running power measured at the meter, used as a plausibility check only |
| 4 | `step4_calibrate.py` | RAMP calibration of one target at the nameplate (method in `calibration.py`) |
| 5 | `step5_union.py` | Best of the two search settings, target by target |
| 6 | `step6_fft_arbitration.py` | Among the finalists, the smallest harmonics gap, provided no criterion is lost and the shape does not worsen |
| 7 | `step7_final_reference.py` | Same rule, against a run with durations down to 5 minutes on the targets still missing the harmonics criterion |
| 8 | `step8_publication.py` | Publication figures and tables |
| 9 | `step9_transfer.py` | What transfers from one appliance to another |

Shared modules: `settings.py` (clients, nameplates, dates, thresholds, metrics),
`validation.py` (validation contract), `calibration.py`, `plates.py`,
`publication_figures.py`. `run.py` chains everything.

## Method in short

**Calibration of one target** (`calibration.py`). The exact geometry comes first: an optimal
three-level staircase under the RAMP rules (at most two ranges per cycle, one to three windows),
found by dynamic programming with 40 starts, on the profile of the running days and on the
profile of all days. The engine then renders each geometry: the occupancy profile of each
cycle is simulated with a 1 W appliance, and the levels come from least squares bounded by the
nameplate. Six orders of the cycles and a short grid (`func_cycle` 15, 30 or 60 min, window
share 1.0 or 0.75, `random_var_w` 0 or 0.1) are ranked on

J = NRMSE + 0.5 x worst one-hour smoothed gap + 0.1 x local shape beyond its noise threshold
+ 0.1 x energy gap beyond 5 %.

The five best candidates are realised at the nameplate (p_i1 = nameplate, t_i1 and t_i2 from
the duty cycle, a fixed-point loop on the engine), their borders are refined by one slot, and
the finalists are arbitrated on independent seeds. Three cycles are always declared: below
three, the engine starts every switch-on with the high phase and overestimates the level.

**Two search settings** produced the published seasonal results, and both are kept so that the
chain gives them back exactly:

- `2026-09-19`: J also carries 0.03 x peak-time gap, one cycle period (30 min, 60 min for the
  mills in non-continuous mode);
- `final`: no peak-time term (the peak time serves the validation only), and a short period of
  15 minutes competes with the long one.

Removing the peak-time term changes the ranking of the candidates, and a good geometry can
leave the five best (0016GBO, dry season). Step 5 therefore keeps, for each target, the version
with the lowest J. The monthly targets use the final setting, with `func_cycle` inherited from
the seasonal target that contains the month.

**Validation contract** (`validation.py`). Ten criteria: NRMSE, load duration curve, harmonics
(FFT), energy, peak, load factor, hourly correlation, hourly distance, profile extremes and
peak time; plus three local shape measures at one and two hours. Each criterion is met when its
gap stays below max(nominal threshold, min(95th percentile of a perfect model, 2 x nominal
threshold)), the 95th percentile coming from the block bootstrap of the target's own days.

## Provenance and integrity

- `resultats/reference_saison/` and `resultats/calib_mois/` are exact copies of the results
  presented in the two method guides. The only lines changed are the stand-alone runner at the
  bottom of each `entrees_ramp/*.py`, so that it finds the engine of this repository; the RAMP
  declarations themselves are untouched.
- The code of this repository is an English translation of the working code. Before release it
  was checked bit for bit against the published results: steps 1 to 3 (every file identical),
  step 4 with both settings, with the widened grid and on a monthly target (profile, RAMP
  declaration, criteria and finalists identical), steps 5 to 7 on the published runs (every
  result identical), steps 8 and 9 (every table and every figure identical, byte for byte).
- Fixed seeds, pinned library versions, Python 3.12. From Python 3.12 on, `sum()` compensates
  rounding errors; the final setting relies on it, while the 2026-09-19 setting added the
  terms of J one by one. Both ways are kept in `calibration.total`.
