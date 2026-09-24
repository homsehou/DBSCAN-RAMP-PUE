# 0016GBO, Janvier : moulin à grains, plaque 7500 W. Parametres RAMP calibres le 19/09/2026 (plan d'audit, etape 3).


def declarer(user):
    app = user.add_appliance(name="grain_milling", number=1, power=7500,
        num_windows=2, func_time=394, func_cycle=15,
        fixed="yes", fixed_cycle=3, continuous_duty_cycle=0,
        occasional_use=0.714, flat="no", thermal_p_var=0,
        time_fraction_random_variability=0, pref_index=0, wd_we_type=2)
    app.windows(random_var_w=0.1, window_1=[525, 690], window_2=[720, 1080])
    app.specific_cycle(1, p_11=7500, t_11=1, p_12=10.0, t_12=36, r_c1=0, cw11=[525, 630], cw12=[885, 1005])
    app.specific_cycle(2, p_21=7500, t_21=1, p_22=10.0, t_22=16, r_c2=0, cw21=[630, 690], cw22=[720, 855])
    app.specific_cycle(3, p_31=7500, t_31=3, p_32=10.0, t_32=29, r_c3=0, cw31=[855, 885], cw32=[1005, 1080])


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
