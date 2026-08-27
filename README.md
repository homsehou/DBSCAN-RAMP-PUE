# DBSCAN-RAMP-PUE-v3

Load profile modelling of productive uses of electricity (PUE) on rural
mini-grids in Benin. Smart-meter measurements are clustered with DBSCAN,
then **one native RAMP appliance per client** is calibrated so that its
simulated profiles match the measured ones, season by season, against six
validation criteria. The RAMP engine is **never modified**: a surrogate
(the analytic mean model of the native appliance) inverts each target
into native parameters, and a damped loop against the real engine
corrects them.

**v3 design rules.** Exactly **one self-contained script per step**, every
file **under 200 lines** (comments included), written for a Python
beginner to read, understand and reproduce alone. Exactly **one shared
file**, `src/config.py`, holding the settings that must be identical
everywhere: cleaning thresholds, clustering parameters, the season
calendar, figure colours, random seeds, the six validation criteria and
the diagnostic helpers. Nothing else is shared — no registry, no utils, and
no step script imports another step script. Each step still runs on its own
with two environment variables. The only library-style code is the embedded
RAMP engine (`ramp/`, third-party, untouched — the colleague's continuous
mode, no local extension).

| Committee recommendation (2026-05-27) | Answer | Where |
|---|---|---|
| RAMP outputs dip at midnight | The engine runs in the colleague's continuous mode: switch-on events cross midnight naturally, no dip by construction | `ramp/core/core.py` (continuous mode) |
| Shape and powers of the simulated profiles | Native single-appliance calibration through a surrogate: cold chain as up to three duty-cycle regions over the day, mills as ON-budget inside the step-4 activity windows; powers offered as measured candidates inside physical bounds rather than fixed in advance, since a 15-minute meter cannot identify the burst; a fast replica of the engine screens candidate region/duty/power/cycle settings, the untouched engine verifies the best and the holdout has a veto against overfitting | `step4_invert.py` + `step5_search.py` + `step6_correct.py` (thermal), `step5_invert.py` + `step6_search.py` + `step7_correct.py` (mills) |
| Day-to-day realism | Native time variability (tfrv) matched to the measured energy CV, kept per season only when the shape stays intact; days without activity restored by the native `occasional_use`, measured in step 1 | same calibration scripts |
| Time-flexibility per PUE, for MicroGridsPy | **On hold** (own future track): the study and its scripts left the working repository, the write-up waits in `Bibliographie/99_reserve_flexibilite/` | — |
| Model choice vs literature | Literature review (preprocessing, clustering, calibration; flexibility on hold) + step-by-step retained methodology | `Bibliographie/00_revues_redigees/`, `Bibliographie/04_methodologie_calibration/` |

## Repository layout

```
├── Bibliographie/
│   ├── 00_revues_redigees/      the literature review (PDF + source)
│   ├── 01_notes_de_lecture/     verbatim reading notes, one entry per reference
│   ├── 02_guide_methodologique/ script-by-script guide (PDF + source)
│   ├── 03_checklist_audit/      quality checklist (PDF + source)
│   ├── 04_methodologie_calibration/ the retained methodology, step by step
│   ├── 99_reserve_flexibilite/  flexibility part, on hold (own future track)
│   └── references_pdf/          the cited PDFs, linked from the documents
├── data/        one smart-meter CSV per client
├── notebooks/   one notebook per client (Run All = full chain)
├── ramp/        embedded RAMP 0.5.2, continuous mode, untouched
├── resultats/   outputs per client (+ _baseline_bandes_2026-08-26/,
│                the frozen before/after reference of the refactoring)
├── src/
│   ├── run.py                       single entry point
│   ├── thermal_equipment/           cold chain + incubator, steps 1-8
│   └── grain_milling/               mills, steps 1-9
└── requirements.txt                pinned library versions
```

## Running

```
python3 -m venv .venv                               # once
.venv/bin/pip install -r requirements.txt           # pinned versions
.venv/bin/python src/run.py cold_chain 0017SAM 300  # one client, full chain
.venv/bin/jupyter lab                               # notebooks in the browser
```

Notebooks ship executed — results visible without running anything. Every
step script also runs alone, e.g.:

```
PUE_TYPE=cold_chain PUE_CLIENT=0017SAM .venv/bin/python \
    src/thermal_equipment/step6_correct.py
```

## Method in one paragraph

Each client is ONE native RAMP appliance; the engine itself is never
edited. A cold-chain day is cut into up to three duty regions (night /
day / evening, midnight always a boundary): each region is both a usage
window and a duty-cycle window, ON for `duty x L` minutes at its own
burst power, then at rest. A mill runs sessions of at least t1 minutes
inside the step-4 activity windows until the daily ON budget is spent.

The measured anchors no longer fix the powers, they bound them. A
15-minute meter cannot tell a short strong burst from a long weak one,
so pinning the burst power on a percentile decided in advance answers a
question the measurement leaves open. Each anchor now offers a set of
measured candidates - percentiles of the daily maxima for the burst
power, of the positive readings for the standby - inside hard physical
bounds: never above the nameplate or the highest reading, never below
the mean profile the appliance has to be able to draw, and never above
90 % of the measured trough for the standby. The six criteria choose
among them, the engine verifies, and the holdout has a veto. Every
delivered value travels next to its anchor (`p1_anchor_W`,
`p2_anchor_W`) so the gap stays auditable. Calibrated this way: the
burst power of each region, the standby, the engine dispersion, the
cycle length, the region cuts and the day-to-day time variability.
`occasional_use` stays measured, and `func_time` stays tied to the
regions - that identity is what keeps a switch-on event from carrying
its cycle onto a neighbour.

The surrogate works in two stages: the analytic mean model inverts each
seasonal target into a first set of parameters, then a numpy replica of
the exact engine path - certified indistinguishable from the engine's own
seed-to-seed noise - screens candidates around it,
coarsely over every structure crossed with every offered power, then by
Latin hypercube over the survivors. Only the best go to the real engine
for verdict. A damped loop against the real engine (bounded ratio, 80 %
then 60 % of each step) closes the fit - freely on the classic arm,
under an improve-or-keep guard on the screened arm - and the better arm
delivers the season, unless the holdout refuses it. The native
`occasional_use`, measured in step 1, restores the days without
activity; the native time variability restores the day-to-day energy
spread where it does not cost shape. Validation: six criteria (NRMSE,
LDC, FFT, energy, peak, load factor), against the fitting target AND
against a holdout of odd calendar days, reported next to the split floor
(the distance between the two half-populations, the best any model could
reach) and to a 95 % seed margin.

## Provenance and integrity

- Four seasonal models per client: dry season (Nov-Apr), May, rainy
  season (Jun-Sep) and October - the two shoulder months each get their
  own model. The per-criterion scores of the native models are read
  against the frozen banded baseline
  (`resultats/_baseline_bandes_2026-08-26/`): the native parameters are
  coarser than the retired 96-band extension, and the six criteria state
  the fineness they reach - that comparison is part of the acted
  refactoring protocol.
- Cleaning registers per client: dated grid outages, rejected days with
  their reason, activity register with operation stops and the measured
  `occasional_use` per season.
- Reproducible: fixed seeds, pinned versions (inline in
  `requirements.txt`), every figure visually inspected before delivery.
