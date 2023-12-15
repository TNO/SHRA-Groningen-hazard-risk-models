import numpy as np
import scipy.special as sp


def spectral_periods():
    """
    returns the spectral periods used in the fragility and consequence models
    the periods are mention in [Crowley and Pinho, 2019/2020, Section 4.1.2]
    """
    return np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0])


def fragility_poe(lnIM, b0, b1, lnSD_lim, beta):
    """
    returns probability of exceedance of a limit state
    [Crowley and Pinho, 2019, Eq. 5.2-5.4, ..., 2020, Eq. 5.1-5.3]

    inputs
    lnIM   - broadcast-ready (aligned) numpy array representing
             the log of the ground motion intensity measure
    b0, b1 - fragility coefficients aligned with lnIM
    lnSD_lim - limit states
    """
    lnSD = b0 + b1 * lnIM
    arg = (lnSD - lnSD_lim) / beta

    poe = 0.5 * (1 + sp.erf(arg / np.sqrt(2.0)))

    return poe
