# 0017SAM, Saison seche | depuis 2025-09-25: cold_chain, nameplate 276 W + 276 W. RAMP parameters calibrated at the nameplate.


def declarer(user):
    app1 = user.add_appliance(name="cold_chain_1", number=1, power=276,
        num_windows=2, func_time=1395, func_cycle=15,
        fixed="yes", fixed_cycle=3, continuous_duty_cycle=1,
        occasional_use=1.000, flat="no", thermal_p_var=0,
        time_fraction_random_variability=0, pref_index=0, wd_we_type=2)
    app1.windows(random_var_w=0.1, window_1=[0, 1125], window_2=[1170, 1440])
    app1.specific_cycle(1, p_11=276, t_11=1, p_12=4.0, t_12=21, r_c1=0, cw11=[780, 1125], cw12=[1170, 1350])
    app1.specific_cycle(2, p_21=276, t_21=2, p_22=4.0, t_22=14, r_c2=0, cw21=[615, 780], cw22=[1350, 1410])
    app1.specific_cycle(3, p_31=276, t_31=10, p_32=4.0, t_32=60, r_c3=0, cw31=[0, 615], cw32=[1410, 1440])
    app2 = user.add_appliance(name="cold_chain_2", number=1, power=276,
        num_windows=2, func_time=1395, func_cycle=15,
        fixed="yes", fixed_cycle=3, continuous_duty_cycle=1,
        occasional_use=1.000, flat="no", thermal_p_var=0,
        time_fraction_random_variability=0, pref_index=0, wd_we_type=2)
    app2.windows(random_var_w=0.1, window_1=[0, 1125], window_2=[1170, 1440])
    app2.specific_cycle(1, p_11=276, t_11=1, p_12=4.0, t_12=21, r_c1=0, cw11=[780, 1125], cw12=[1170, 1350])
    app2.specific_cycle(2, p_21=276, t_21=2, p_22=4.0, t_22=14, r_c2=0, cw21=[615, 780], cw22=[1350, 1410])
    app2.specific_cycle(3, p_31=276, t_31=10, p_32=4.0, t_32=60, r_c3=0, cw31=[0, 615], cw32=[1410, 1440])


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
