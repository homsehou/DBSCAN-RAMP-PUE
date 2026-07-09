# DBSCAN-RAMP-PUE

Load profile modelling of productive uses of electricity (PUE) on rural mini-grids in
Benin. 
Smart-meter measurements are clustered with DBSCAN, then the stochastic 
[RAMP](https://github.com/RAMP-project/RAMP) model is calibrated so that its simulated 
profiles match the measured ones, season by season.

## Repository layout

```
├── data/<type>/<client>/        one smart-meter CSV per client
├── notebooks/<type>/            one notebook per client — Run All executes the chain
├── src/
│   ├── run.py                   single entry point: run(type, client)
│   ├── registry.py              dynamic client registry, read from the data/ tree
│   ├── grain_milling/           mill chain: 5 steps + config
│   └── thermal_equipment/       cold appliances + incubator: 5 steps + config
├── ramp/                        embedded RAMP source 
├── requirements.txt             exact package versions used for the committed results
└── resultats/<type>/<client>/   outputs: CSV tables, calibrated parameters, figures
```
There is no client list to maintain: `src/registry.py` discovers the clients from the
`data/<type>/<client>/<client>.csv` tree. Each client's study period and rated power
are entered in its notebook, so clients stay anonymous — only their code appears here.

## Studying an existing client

1. Open the client's notebook under `notebooks/<type>/` with the
   `Python (DBSCAN-RAMP-PUE)` kernel.
2. Run all cells. The full chain executes on the spot (preprocessing, clustering,
   calibration, figures, summary) with live progress, then the notebook displays the
   results: identity card, measured seasonal profiles and heatmap, clustering figures,
   one measured-versus-simulated block per season with the six validation criteria,
   and the calibrated RAMP parameters. Expect 10 to 15 minutes per client.

## Adding a new PUE, from CSV to results

1. **Drop the data**: create `data/<type>/<code>/` and place the meter series as
   `<code>.csv`. The cold chain and the incubator read a curated series with columns
   `datetime_utc, power_W` (and optional `voltage`); the mill chain reads the raw meter
   export with columns `Reference_No., Date, Average_Active_power, ...`.
2. **Duplicate a notebook** of the same PUE type under `notebooks/<type>/`, rename it
   `<code>.ipynb`, and fill its first code cell — the client code, the type, the study
   period to calibrate on and, for a cold appliance, its rated power:
   ```python
   CLIENT = "<code>"
   PUE_TYPE = "<type>"
   PERIOD_START = "2024-10-01"
   PERIOD_END = "2025-11-30"
   RATED_POWER_W = 150          # None if not applicable
   ```
3. **Run all cells.** The chain runs and the notebook fills itself with the results.
   Outputs also land under `resultats/<type>/<code>/` (tables, figures, calibrated
   parameters).

## Results

`resultats/<type>/<client>/` holds the cleaned time series, the daily matrix, the
clustering tables, the calibrated parameters (`ramp_calibrated_params.csv`), the
validation table (`validation_metrics.csv`), the simulated profiles and all figures.
`resultats/<type>/` also carries a cross-client summary produced by the last step of
each chain.

