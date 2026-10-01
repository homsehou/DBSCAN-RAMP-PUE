# 0043SAM, Saison seche: grain_milling, nameplate 7500 W. RAMP parameters calibrated at the nameplate.


def declarer(user):
    app = user.add_appliance(name="grain_milling", number=1, power=7500,
        num_windows=1, func_time=630, func_cycle=15,
        fixed="yes", fixed_cycle=3, continuous_duty_cycle=1,
        occasional_use=0.634, flat="no", thermal_p_var=0,
        time_fraction_random_variability=0, pref_index=0, wd_we_type=2)
    app.windows(random_var_w=0.0, window_1=[420, 1275])
    app.specific_cycle(1, p_11=7500, t_11=3, p_12=6.0, t_12=11, r_c1=0, cw11=[420, 465], cw12=[1245, 1275])
    app.specific_cycle(2, p_21=7500, t_21=3, p_22=6.0, t_22=11, r_c2=0, cw21=[720, 915])
    app.specific_cycle(3, p_31=7500, t_31=11, p_32=6.0, t_32=12, r_c3=0, cw31=[465, 720], cw32=[915, 1245])


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
