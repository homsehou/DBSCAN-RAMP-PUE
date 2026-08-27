# -*- coding: utf-8 -*-
"""
config.py — Central configuration of the 2R2C model (V2).

Content limited to:
  - PHYSICAL CONSTANTS (water properties, thermodynamic constants)
  - PATHS (data, results)
  - PROTOCOL METADATA (water mass, door-opening duration, test phases)

No arbitrary bounds on the identified parameters. Identification relies
on physical priors (Bayesian approach) or unconstrained maximum
likelihood (EKF+MLE).
"""

from pathlib import Path

# =============================================================================
# 1. PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_TEST1 = PROJECT_ROOT / "data_ref" / "Test1" / "Collected_data.xlsx"
DATA_TEST2 = PROJECT_ROOT / "data_ref" / "Test2" / "Collected_data.xlsx"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# =============================================================================
# 2. SHEET AND COLUMN NAMES (Collected_data.xlsx)
# =============================================================================
POWER_SHEET = "Power"
POWER_DATE_COL = "Date"
POWER_TIME_COL = "Heure"
POWER_VALUE_COL = "P (W)"
POWER_DT_FORMAT = "%m/%d/%Y %I:%M:%S %p"

TEMP_SHEET = "Temperature"
TEMP_TIME_COL = "time"
TEMP_TAMB_COL = "Tamb"
TEMP_TAIR_COL = "Tair_in"
TEMP_TEAU_COL = "Teau"

# =============================================================================
# 3. TIME BOUNDS OF THE TEST PHASES
# =============================================================================
PHASES_TEST1 = {
    "A": ("2026-02-02 19:35:00", "2026-02-03 05:30:00"),
    "B": ("2026-02-03 05:30:00", "2026-02-04 05:30:00"),
    "C": ("2026-02-04 05:30:00", "2026-02-04 10:30:00"),
    "D": ("2026-02-04 10:30:00", None),
}
PHASES_TEST2 = {
    "A": ("2026-02-04 19:35:00", "2026-02-05 05:30:00"),
    "B": ("2026-02-05 05:30:00", "2026-02-06 05:30:00"),
    "C": ("2026-02-06 05:30:00", "2026-02-06 10:30:00"),
    "D": ("2026-02-06 10:30:00", None),
}

# =============================================================================
# 4. PHYSICAL PROPERTIES OF WATER (source: NIST Webbook)
# =============================================================================
N_SACHETS = 14
M_SACHET_KG = 0.481
M_EAU_KG = N_SACHETS * M_SACHET_KG             # 6.734 kg

C_LIQ_J_KG_K = 4186.0                          # liquid water at 20 C
C_ICE_J_KG_K = 2100.0                          # ice at 0 C
L_F_J_KG = 334000.0                            # latent heat of fusion
T_FREEZE_C = 0.0                               # melting point
EPSILON_FREEZE_C = 0.5                         # half-plateau width (regularisation)

# =============================================================================
# 5. AMBIENT AIR PROPERTIES
# =============================================================================
RHO_AIR_KG_M3 = 1.2
CP_AIR_J_KG_K = 1005.0

# =============================================================================
# 6. APPLIANCE GEOMETRY (Roch RUF-295-J)
# =============================================================================
V_INTERNE_L = 159.0
V_INTERNE_M3 = V_INTERNE_L / 1000.0

# =============================================================================
# 7. INSTRUMENTATION — UNCERTAINTIES
# =============================================================================
# Type-K thermocouples: +/- (0.2 % + 0.5 C)
SIGMA_TEMP_C = 0.5                             # temperature noise std (sensor)

# =============================================================================
# 8. DOOR-OPENING PROTOCOL
# =============================================================================
DOOR_PERIOD_S = 3600.0                         # one opening per hour
DOOR_OPEN_S = 30.0                             # duration of one opening
# During openings, the Option B model replaces the constant power
# E_door/30s by an exponential coupling of T_air towards T_amb with a
# time constant tau_door (parameter to identify).

# =============================================================================
# 9. COMPRESSOR ON/OFF DETECTION (prerequisite for Q_c = COP * P_elec)
# =============================================================================
POWER_HYST_ON_FRAC = 0.60
POWER_HYST_OFF_FRAC = 0.40
POWER_MIN_ON_S = 20
POWER_MIN_OFF_S = 20

# =============================================================================
# 10. NUMERICAL INTEGRATION
# =============================================================================
TEMP_SAMPLING_S = 10.0                         # nominal sampling step of T measurements
INTEGRATION_SUBSTEP_S = 2.0                    # Euler integration substep
# (2 s gives an accuracy below 0.01 C over 5000 s, checked by unit test)

# =============================================================================
# 11. REFERENCE COMPUTATIONS (diagnostics, not used in optimisation)
# =============================================================================
def c_prod_liquid():
    """Thermal capacity of the product, water 100 % liquid (J/K)."""
    return M_EAU_KG * C_LIQ_J_KG_K              # ~28 172 J/K

def c_prod_ice():
    """Thermal capacity of the product, water 100 % solid (J/K)."""
    return M_EAU_KG * C_ICE_J_KG_K              # ~14 141 J/K

def c_air_pure():
    """Thermal capacity of the compartment air alone (J/K). Physical floor."""
    return V_INTERNE_M3 * RHO_AIR_KG_M3 * CP_AIR_J_KG_K  # ~192 J/K
