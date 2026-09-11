from astropy.table import Table

density_diagnostic_points = [
    # (diagnostic, log_nM, log_nobs, independent_subset)
    ("[S II] 6717/6731", 3.06, 2.309, True),
    ("[O II] 3727/3729", 3.16, 2.540, True),
    ("[Fe III] 4986/4881", 3.74, 2.685, True),
    ("[Fe III] 4658/4986", 3.84, 2.395, True),
    ("[Cl III] 5538/5518", 3.99, 3.070, True),
    ("[Fe III] 4986/4702", 4.04, 2.479, False),
    ("[S II] 4070/6717", 4.64, 2.593, False),
    ("[O II] 7320/3729", 4.69, 2.868, False),
    ("[S II] 4070/(6717+6731)", 4.85, 2.757, True),
    ("[S II] (4070+4075)/(6717+6731)", 4.91, 2.979, False),
    ("[O II] 7320/(3727+3729)", 5.03, 2.942, True),
    ("[O II] 7320/3727", 5.10, 3.127, False),
    ("[Fe III] 4702/4881", 6.19, 3.696, True),
    ("[Fe II] 8617/7155", 6.31, 3.333, True),
    ("[Fe III] 4658/4881", 6.36, 3.608, False),
    ("[Fe III] 4658/4702", 6.90, 3.457, False),
]

ori_tab = Table(
    rows=density_diagnostic_points,
    names=[
        "ratio",
        "log10(n_M)",
        "log10(n_obs)",
        "independent?",
    ],
)
