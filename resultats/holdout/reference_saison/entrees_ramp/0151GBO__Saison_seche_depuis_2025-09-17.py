# 0151GBO, Saison seche | depuis 2025-09-17: cold_chain, nameplate 220 W + 220 W. RAMP parameters calibrated at the nameplate.


def declarer(user):
    app1 = user.add_appliance(name="cold_chain_1", number=1, power=220,
        num_windows=2, func_time=1440, func_cycle=15,
        fixed="yes", fixed_cycle=3, continuous_duty_cycle=1,
        occasional_use=1.000, flat="no", thermal_p_var=0,
        time_fraction_random_variability=0, pref_index=0, wd_we_type=2)
    app1.windows(random_var_w=0.1, window_1=[0, 720], window_2=[720, 1440])
    app1.specific_cycle(1, p_11=220, t_11=4, p_12=5.0, t_12=24, r_c1=0, cw11=[885, 1185])
    app1.specific_cycle(2, p_21=220, t_21=12, p_22=5.0, t_22=16, r_c2=0, cw21=[465, 885], cw22=[1185, 1290])
    app1.specific_cycle(3, p_31=220, t_31=12, p_32=5.0, t_32=6, r_c3=0, cw31=[0, 465], cw32=[1290, 1440])
    app2 = user.add_appliance(name="cold_chain_2", number=1, power=220,
        num_windows=2, func_time=1440, func_cycle=15,
        fixed="yes", fixed_cycle=3, continuous_duty_cycle=1,
        occasional_use=1.000, flat="no", thermal_p_var=0,
        time_fraction_random_variability=0, pref_index=0, wd_we_type=2)
    app2.windows(random_var_w=0.1, window_1=[0, 720], window_2=[720, 1440])
    app2.specific_cycle(1, p_11=220, t_11=4, p_12=5.0, t_12=24, r_c1=0, cw11=[885, 1185])
    app2.specific_cycle(2, p_21=220, t_21=12, p_22=5.0, t_22=16, r_c2=0, cw21=[465, 885], cw22=[1185, 1290])
    app2.specific_cycle(3, p_31=220, t_31=12, p_32=5.0, t_32=6, r_c3=0, cw31=[0, 465], cw32=[1290, 1440])


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
    profile = uc.generate_daily_load_profiles(flat=False, continuous=True)
    print(profile.reshape(-1, 96, 15)[1:].mean(axis=(0, 2)).round(1))
