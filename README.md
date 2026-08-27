# 2R2C thermal model of a vertical freezer (Roch RUF-295-J)

Grey-box thermal model of an off-grid freezer used for productive uses of
electricity (PUE) in rural solar mini-grids (Benin). Two thermal nodes
(internal air, product) and two thermal resistances whose values switch
with the compressor regime (ON/OFF). Calibration on a first laboratory
test, blind validation on a second independent test.

## Repository content

```
Modele_RC_2R2C_congelateur/
├── main.py                 # full pipeline: calibration (Test 1) + blind validation (Test 2)
├── regenerate_plots.py     # comparison figures from the saved parameters
├── config.py               # physical constants, paths, protocol metadata
├── src/
│   ├── data.py             # Excel loading, power aggregation, phase labelling
│   ├── model.py            # 2R2C equations and continuous simulation
│   └── identify.py         # analytical identification of the 7 parameters
├── data_ref/
│   ├── Test1/Collected_data.xlsx   # calibration data (power 1 s, temperatures 10 s)
│   └── Test2/Collected_data.xlsx   # blind validation data
└── results/                # figures, parameters (JSON), summary (CSV)
```

## How to run

```bash
pip install -r requirements.txt
python3 main.py               # calibration + validation (about 1 min)
python3 regenerate_plots.py   # presentation figures (comparison_TestX_v2.png)
```

All paths are relative to the repository root: no configuration needed.

## Test protocol

Each test covers four phases on an instrumented freezer loaded with
14 water sachets (6.734 kg):

| Phase | Content | Identified parameters |
|-------|---------|----------------------|
| A | Continuous pulldown, +30 to -25 °C | R_env_on, R_ap_on, COP |
| B | Thermostatic cycling with one door opening per hour (24 openings) | E_door |
| C | Thermostatic cycling without openings | none (consistency check) |
| D | Compressor off, free temperature rise | R_env_off, R_ap_off, C_air |

## Expected results

Parameters identified on Test 1:

| Parameter | Value |
|-----------|-------|
| R_env ON / OFF | 0.488 / 2.431 K/W (ratio 4.99) |
| R_ap ON / OFF | 0.055 / 0.164 K/W (ratio 2.96) |
| C_air | 3 887 J/K |
| COP | 1.173 |
| E_door | 83.8 kJ per opening |

Model accuracy:

| Test | RMSE T_air | CV-RMSE (ASHRAE 14 threshold: 30 %) |
|------|-----------|--------------------------------------|
| Test 1 (calibration) | 2.76 °C | 11.4 % |
| Test 2 (blind validation) | 5.85 °C | 24.0 % |

The identification is fully deterministic: a new run reproduces these
values exactly.
