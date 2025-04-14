import numpy as np


# Median
def median(R, M, pars, par_id=None):
    """returns the log-median of PGV the ground motion model
    equation 3.1-3.4 in [Bommer et al. 2021], for reference vs30=200 m/s
    all input arrays should be broadcast-ready, i.e., with
    properly aligned dimensions, but can have any number of
    dimensions

    inputs
    R, M - broadcast-ready (aligned) numpy arrays representing
                 distances, magnitudes
    pars       - ground motion model coefficients aligned with R, M
    par_id     - index array for parameter names/id's
    """

    # fixed parameters - hinge distances, reference vs30
    Rh0, Rh1, Rh2 = 1.0, 7.0, 12.0

    # organize parameter views into dictionary
    c = gen_dict_like(pars, par_id)

    # local symbols just for code readability
    c1, c2 = c["c1"], c["c2"]
    c3, c4, c5 = c["c3"], c["c4"], c["c5"]
    c6, c7 = c["c6"], c["c7"]

    # distance saturation, eq. (3.3)
    h = np.exp(c6 + c7 * M)

    # from hypocentral R to Rf, "effective" R as in eq. (3.2)
    Rf = (R**2 + h**2) ** 0.5

    # division of distance range into 3 sections by clipping as in eq. (3.4 a, b, c)
    lnR0 = np.log(np.clip(Rf / Rh0, 0.0, Rh1 / Rh0))
    lnR1 = np.log(np.clip(Rf / Rh1, 1.0, Rh2 / Rh1))
    lnR2 = np.log(np.clip(Rf / Rh2, 1.0, None))

    # combine (3.4 a, b, c)
    g = c3 * lnR0 + c4 * lnR1 + c5 * lnR2

    # total (3.1)
    lnPGV = c1 + c2 * M + g

    return lnPGV


# Relative amplification
def amplification(vs30, pars, par_id=None):
    """returns the relative site response amplification conditional
    on vs30, equation (3.1) [Bommer et al. 2021]
    all input arrays should be broadcast-ready, i.e., with
    properly aligned dimensions, but can have any number of
    dimensions

    inputs
    vs30 - broadcast-ready (aligned) numpy arrays representing
                vs30 for the sites
    pars       - ground motion model coefficients aligned with vs30
    par_id     - index array for parameter names/id's
    """

    # fixed parameters - reference vs30
    vs30_ref = 200.0

    # organize parameter views into dictionary
    c = gen_dict_like(pars, par_id)

    lnAF = c["c8"] * np.log(vs30 / vs30_ref)

    return lnAF


# UTILITY
def gen_dict_like(c, p):
    if p is None:
        return c
    else:
        return {par: c[..., i] for i, par in enumerate(p)}
