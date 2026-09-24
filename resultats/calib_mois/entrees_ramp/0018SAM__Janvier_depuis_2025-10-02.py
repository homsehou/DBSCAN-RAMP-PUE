# 0018SAM, Janvier | depuis 2025-10-02 : froid, plaque 276 W + 220 W. Parametres RAMP calibres le 19/09/2026 (plan d'audit, etape 3).


def declarer(user):
    app1 = user.add_appliance(name="cold_chain_1", number=1, power=276,
        num_windows=2, func_time=1365, func_cycle=30,
        fixed="yes", fixed_cycle=3, continuous_duty_cycle=1,
        occasional_use=0.667, flat="no", thermal_p_var=0,
        time_fraction_random_variability=0, pref_index=0, wd_we_type=2)
    app1.windows(random_var_w=0.0, window_1=[0, 1020], window_2=[1080, 1440])
    app1.specific_cycle(1, p_11=276, t_11=4, p_12=4.5, t_12=25, r_c1=0, cw11=[0, 555], cw12=[705, 795])
    app1.specific_cycle(2, p_21=276, t_21=2, p_22=4.5, t_22=25, r_c2=0, cw21=[915, 1020], cw22=[1080, 1440])
    app1.specific_cycle(3, p_31=276, t_31=12, p_32=4.5, t_32=4, r_c3=0, cw31=[555, 705], cw32=[795, 915])
    app2 = user.add_appliance(name="cold_chain_2", number=1, power=220,
        num_windows=2, func_time=1365, func_cycle=30,
        fixed="yes", fixed_cycle=3, continuous_duty_cycle=1,
        occasional_use=0.667, flat="no", thermal_p_var=0,
        time_fraction_random_variability=0, pref_index=0, wd_we_type=2)
    app2.windows(random_var_w=0.0, window_1=[0, 1020], window_2=[1080, 1440])
    app2.specific_cycle(1, p_11=220, t_11=4, p_12=3.5, t_12=25, r_c1=0, cw11=[0, 555], cw12=[705, 795])
    app2.specific_cycle(2, p_21=220, t_21=2, p_22=3.5, t_22=25, r_c2=0, cw21=[915, 1020], cw22=[1080, 1440])
    app2.specific_cycle(3, p_31=220, t_31=12, p_32=3.5, t_32=4, r_c3=0, cw31=[555, 705], cw32=[795, 915])


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
