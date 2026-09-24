# 0097SAM, Fevrier : moulin à grains, plaque 7350 W. Parametres RAMP calibres le 19/09/2026 (plan d'audit, etape 3).


def declarer(user):
    app = user.add_appliance(name="grain_milling", number=1, power=7350,
        num_windows=1, func_time=641, func_cycle=15,
        fixed="yes", fixed_cycle=3, continuous_duty_cycle=0,
        occasional_use=1.000, flat="no", thermal_p_var=0,
        time_fraction_random_variability=0, pref_index=0, wd_we_type=2)
    app.windows(random_var_w=0.0, window_1=[420, 1290])
    app.specific_cycle(1, p_11=7350, t_11=1, p_12=16.0, t_12=9, r_c1=0, cw11=[420, 1020], cw12=[1230, 1290])
    app.specific_cycle(2, p_21=7350, t_21=3, p_22=16.0, t_22=10, r_c2=0, cw21=[1020, 1110], cw22=[1140, 1230])
    app.specific_cycle(3, p_31=7350, t_31=10, p_32=16.0, t_32=12, r_c3=0, cw31=[1110, 1140])


if __name__ == "__main__":
    # mean profile of 480 simulated days; RAMP engine of this repository
    # (this file lives in resultats/<run>/entrees_ramp/)
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
    from ramp import User, UseCase
    uc = UseCase(name="pue", date_start="2024-01-01", date_end="2025-04-25")
    user = User(user_name="client", num_users=1)
    uc.add_user(user)
    declarer(user)
    profil = uc.generate_daily_load_profiles(flat=False, continuous=True)
    print(profil.reshape(-1, 96, 15)[1:].mean(axis=(0, 2)).round(1))
