# 0151GBO, Juillet | avant 2025-09-17 : froid, plaque 220 W. Parametres RAMP calibres le 19/09/2026 (plan d'audit, etape 3).


def declarer(user):
    app = user.add_appliance(name="cold_chain", number=1, power=220,
        num_windows=2, func_time=968, func_cycle=15,
        fixed="yes", fixed_cycle=3, continuous_duty_cycle=1,
        occasional_use=1.000, flat="no", thermal_p_var=0,
        time_fraction_random_variability=0, pref_index=0, wd_we_type=2)
    app.windows(random_var_w=0.0, window_1=[0, 900], window_2=[1065, 1440])
    app.specific_cycle(1, p_11=220, t_11=1, p_12=9.7, t_12=29, r_c1=0, cw11=[0, 510], cw12=[1230, 1440])
    app.specific_cycle(2, p_21=220, t_21=1, p_22=9.7, t_22=999, r_c2=0, cw21=[795, 900], cw22=[1065, 1230])
    app.specific_cycle(3, p_31=220, t_31=9, p_32=9.7, t_32=21, r_c3=0, cw31=[510, 795])


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
