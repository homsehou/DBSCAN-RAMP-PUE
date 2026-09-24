# 0154GBO, Saison seche : moulin à grains, plaque 7500 W. Parametres RAMP calibres le 19/09/2026 (plan d'audit, etape 3).


def declarer(user):
    app = user.add_appliance(name="grain_milling", number=1, power=7500,
        num_windows=1, func_time=870, func_cycle=30,
        fixed="yes", fixed_cycle=3, continuous_duty_cycle=1,
        occasional_use=1.000, flat="no", thermal_p_var=0,
        time_fraction_random_variability=0, pref_index=0, wd_we_type=2)
    app.windows(random_var_w=0.0, window_1=[405, 1320])
    app.specific_cycle(1, p_11=7500, t_11=1, p_12=0.0, t_12=21, r_c1=0, cw11=[450, 1005])
    app.specific_cycle(2, p_21=7500, t_21=1, p_22=0.0, t_22=900, r_c2=0, cw21=[405, 450], cw22=[1200, 1320])
    app.specific_cycle(3, p_31=7500, t_31=6, p_32=0.0, t_32=16, r_c3=0, cw31=[1005, 1200])


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
