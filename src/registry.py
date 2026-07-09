"""Dynamic client registry, driven by the data/ folder tree.

A client exists as soon as its CSV is present at data/<type>/<client>/<client>.csv.
Nothing else has to be declared: adding a client never touches shared files. The
study period and any appliance rating are entered in the client's notebook (see
run.run), not stored in the repository, so clients stay anonymous — only their
code appears here.
"""
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]

EQUIPMENT_DEFAULT = {"grain_milling": "Grain mill", "cold_chain": "Cold appliance",
                     "poultry_incubation": "Egg incubator"}


def clients(pue_type):
    """Client codes of one PUE type, discovered from the data folders."""
    base = REPO / "data" / pue_type
    return sorted(d.name for d in base.iterdir()
                  if d.is_dir() and (d / f"{d.name}.csv").exists())


def client_info(pue_type, client):
    """Identity of one client: code, CSV path, site and a generic equipment label.

    The default study period is the full span of the CSV; the notebook overrides
    it when a shorter calibration window is wanted. The first column holding dates
    is used, so both input schemas (mill exports and extracted series) are accepted.
    """
    folder = REPO / "data" / pue_type / client
    info = {"client": client, "pue_type": pue_type, "csv": folder / f"{client}.csv",
            "site": client[-3:], "equipment": EQUIPMENT_DEFAULT.get(pue_type, "PUE")}

    header = pd.read_csv(info["csv"], nrows=0).columns
    date_col = "datetime_utc" if "datetime_utc" in header else "Date"
    dates = pd.to_datetime(pd.read_csv(info["csv"], usecols=[date_col])[date_col],
                           utc=True, errors="coerce")
    info["period_start"] = str(dates.min().date())
    info["period_end"] = str(dates.max().date())
    return info
