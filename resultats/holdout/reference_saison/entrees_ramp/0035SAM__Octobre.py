# 0035SAM, Octobre: cold_chain, nameplate 276 W. RAMP parameters calibrated at the nameplate.


def declarer(user):
    app = user.add_appliance(name="cold_chain", number=1, power=276,
        num_windows=3, func_time=990, func_cycle=30,
        fixed="yes", fixed_cycle=3, continuous_duty_cycle=1,
        occasional_use=1.000, flat="no", thermal_p_var=0,
        time_fraction_random_variability=0, pref_index=0, wd_we_type=2)
    app.windows(random_var_w=0.1, window_1=[0, 420], window_2=[750, 1050], window_3=[1155, 1440])
    app.specific_cycle(1, p_11=276, t_11=3, p_12=0.5, t_12=75, r_c1=0, cw11=[825, 870], cw12=[1215, 1440])
    app.specific_cycle(2, p_21=276, t_21=1, p_22=0.5, t_22=106, r_c2=0, cw21=[0, 420], cw22=[750, 825])
    app.specific_cycle(3, p_31=276, t_31=1, p_32=0.5, t_32=999, r_c3=0, cw31=[870, 1050], cw32=[1155, 1215])


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
