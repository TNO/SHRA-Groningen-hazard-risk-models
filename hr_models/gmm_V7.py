import numpy as np


# REFERENCE
def reference_median(R, M, T, pars, par_id=None):
    """returns the sum of the path and source term of the ground motion model
    equation 9.1 in [Bommer et al. 2021]

    inputs
    R, M, T - broadcast-ready (aligned) numpy arrays representing
              distances, magnitudes, spectral periods
    pars    - ground motion model coefficients aligned with R, M, T,
              can be a regular array or a structured array
    par_id  - array with parameter names (ids), if None then assume
              pars is structured array
    """

    gs = source_median(M, pars, par_id=par_id)
    gp = path_median(R, M, T, pars, par_id=par_id)
    lnY = gs + gp

    return lnY


def reference_ac_realization(R, M, T, tau, phi_ss, eps_ref, pars, par_id=None):
    """
    inputs
    R, M, T, tau, phi_ss, eps_ref, eps_af - broadcast-ready (aligned) numpy arrays representing
              distances, magnitudes, spectral periods
    pars    - ground motion model coefficients aligned with R, M, T,
              can be a regular array or a structured array
    par_id  - array with parameter names (ids), if None then assume
              pars is structured array
    """
    lnY_median = reference_median(R, M, T, pars, par_id)
    varY = reference_ac_variance(R, M, T, tau, phi_ss)
    lnY = lnY_median + eps_ref * np.sqrt(varY)

    return lnY


def reference_gm_realization(R, M, T, tau, phi_ss, eps_ref, pars, par_id=None):
    """
    inputs
    R, M, T, tau, phi_ss, eps_ref, eps_af - broadcast-ready (aligned) numpy arrays representing
              distances, magnitudes, spectral periods
    pars    - ground motion model coefficients aligned with R, M, T,
              can be a regular array or a structured array
    par_id  - array with parameter names (ids), if None then assume
              pars is structured array
    """
    lnY_median = reference_median(R, M, T, pars, par_id)
    varY = reference_gm_variance(tau, phi_ss)
    lnY = lnY_median + eps_ref * np.sqrt(varY)

    return lnY


# REFERENCE - PATH
def path_median(R, M, T, pars, par_id=None):
    """returns the path term of the ground motion model
    equation 9.3 - 9.7 in [Bommer et al. 2021]
    all input arrays should be broadcast-ready, i.e., with
    properly aligned dimensions, but can have any number of
    dimensions

    inputs
    R, M, T - broadcast-ready (aligned) numpy arrays representing
              distances, magnitudes, spectral periods
    pars    - ground motion model coefficients aligned with R, M, T
    par_id - index array for parameter names/id's
    """
    # additional GMM-V7 model parameters
    Rh0, Rh1, Rh2, Rh3 = 3.0, 7.0, 12.0, 25.0
    Mr = 3.875

    # organize parameter views into dictionary
    c = gen_dict_like(pars, par_id)

    # local symbols just for code readability
    r0a, r0b, r0c, r0d = c["r0a"], c["r0b"], c["r0c"], c["r0d"]
    r1a, r1b, r1c, r1d = c["r1a"], c["r1b"], c["r1c"], c["r1d"]
    r2a, r2b, r2c, r2d = c["r2a"], c["r2b"], c["r2c"], c["r2d"]
    r3a, r3b, r3c, r3d = c["r3a"], c["r3b"], c["r3c"], c["r3d"]

    # division of distance range into 4 sections
    lnR0 = np.log(np.clip(R / Rh0, 1.0, Rh1 / Rh0))
    lnR1 = np.log(np.clip(R / Rh1, 1.0, Rh2 / Rh1))
    lnR2 = np.log(np.clip(R / Rh2, 1.0, Rh3 / Rh2))
    lnR3 = np.log(np.clip(R / Rh3, 1.0, None))

    # (9.4) - section 0
    r0 = np.where(M > Mr, r0a + r0c * np.tanh(r0d * (M - Mr)), r0a + r0b * (M - Mr))

    # (9.5) - section 1
    r1 = np.where(
        np.logical_and(T <= 0.2, M > Mr),
        r1a + r1c * np.tanh(r1d * (M - Mr)),
        r1a + r1b * (M - Mr),
    )

    # (9.6) - section 2
    r2 = np.where(
        np.logical_and(T <= 0.5, M > Mr),
        r2a + r2c * np.tanh(r2d * (M - Mr)),
        r2a + r2b * (M - Mr),
    )

    # (9.7) - section 3
    r3 = np.where(M > Mr, r3a + r3c * np.tanh(r3d * (M - Mr)), r3a + r3b * (M - Mr))

    # (9.3) - compose
    gp = (r0 * lnR0) + (r1 * lnR1) + (r2 * lnR2) + (r3 * lnR3)

    return gp


# REFERENCE - SOURCE
def source_median(M, pars, par_id=None):
    """returns the source term of the ground motion model
    equation 9.2 in [Bommer et al. 2021]
    both input arrays should be broadcast-ready, i.e., with
    properly aligned dimensions, but can have any number of
    dimensions

    inputs
    M      - broadcast-ready (aligned) numpy array representing
             magnitudes
    pars   - ground motion model coefficients aligned with M
    par_id - index array for parameter names/id's
    """

    # organize parameter views into dictionary
    c = gen_dict_like(pars, par_id)

    # additional GMM-V7 model parameters
    Mm = 4.75

    # local symbols just for code readability
    m0, m1, m2, m3, m4 = c["m0"], c["m1"], c["m2"], c["m3"], c["m4"]

    # (9.2)
    gs = np.where(
        M < Mm,
        m0 + m1 * (M - Mm) + m2 * (M - Mm) ** 2,
        m0 + m3 * (M - Mm) + m4 * (M - Mm) ** 2,
    )

    return gs


# REFERENCE - VARIANCE
def c2c_variance(R, M, T):
    """returns the component-to-component variance
    equations 9.10 and 9.11 in [Bommer et al 2021]

    inputs
    R, M, T - broadcast-ready (aligned) numpy arrays representing
              distances [km], magnitudes, spectral periods [s]
    """
    Mclip = np.clip(M, 3.6, 5.6)
    Tclip = np.clip(T, 0.1, 0.85)

    # (9.10a)
    var_0_1 = 0.026 + 1.03 * (5.6 - Mclip) * R ** (-2.22)

    # (9.10b)
    var_0_85 = 0.045 + 5.315 * (5.6 - Mclip) * R ** (-2.92)

    # (9.11)
    d_var = var_0_85 - var_0_1
    var_c2c = var_0_1 + d_var * np.log(Tclip / 0.1) / np.log(0.85 / 0.1)

    return var_c2c


def reference_gm_variance(tau, phi_ss):
    """returns the component-to-component variance
    equations 9.10 and 9.11 in [Bommer et al 2021]

    inputs
    tau, phiss - broadcast-ready (aligned) numpy arrays representing
              distances [km], magnitudes, spectral periods [s], source (tau)
              and path (phi_ss) standard deviations
    """

    total_variance = tau**2 + phi_ss**2

    return total_variance


def reference_ac_variance(R, M, T, tau, phi_ss):
    """returns the component-to-component variance
    equations 9.10 and 9.11 in [Bommer et al 2021]

    inputs
    R, M, T, tau, phiss - broadcast-ready (aligned) numpy arrays representing
              distances [km], magnitudes, spectral periods [s], source (tau)
              and path (phi_ss) standard deviations
    """

    total_variance = reference_gm_variance(tau, phi_ss) + c2c_variance(R, M, T)

    return total_variance


# SURFACE
def surface_median(R, M, T, ref_pars, af_pars, ref_par_id=None, af_par_id=None):
    """returns the sum of the path and source term of the ground motion model
    equation 9.1 in [Bommer et al. 2021]

    inputs
    inputs
    R, M, T - broadcast-ready (aligned) numpy arrays representing
              distances, magnitudes, spectral periods
    pars    - ground motion model coefficients aligned with R, M, T,
              can be a regular array or a structured array
    par_id  - array with parameter names (ids), if None then assume
              pars is structured array
    """
    lnY = reference_median(R, M, T, ref_pars, ref_par_id)
    lnAF = af_median(R, M, lnY, af_pars, af_par_id)

    return lnY + lnAF


def surface_ac_realization(
    R,
    M,
    T,
    tau,
    phi_ss,
    eps_ref,
    eps_af,
    ref_pars,
    af_pars,
    ref_par_id=None,
    af_par_id=None,
):
    """
    inputs
    R, M, T, tau, phi_ss, eps_ref, eps_af - broadcast-ready (aligned) numpy arrays representing
              distances, magnitudes, spectral periods
    pars    - ground motion model coefficients aligned with R, M, T,
              can be a regular array or a structured array
    par_id  - array with parameter names (ids), if None then assume
              pars is structured array
    """
    lnY = reference_ac_realization(R, M, T, tau, phi_ss, eps_ref, ref_pars, ref_par_id)
    lnAF = af_realization(R, M, lnY, eps_af, af_pars, af_par_id)

    return lnY + lnAF


def surface_gm_realization(
    R,
    M,
    T,
    tau,
    phi_ss,
    eps_ref,
    eps_af,
    ref_pars,
    af_pars,
    ref_par_id=None,
    af_par_id=None,
):
    """
    inputs
    R, M, T, tau, phi_ss, eps_ref, eps_af - broadcast-ready (aligned) numpy arrays representing
              distances, magnitudes, spectral periods
    pars    - ground motion model coefficients aligned with R, M, T,
              can be a regular array or a structured array
    par_id  - array with parameter names (ids), if None then assume
              pars is structured array
    """
    lnY = reference_gm_realization(
        R, M, T, tau, phi_ss, eps_ref, ref_pars, par_id=ref_par_id
    )
    lnAF = af_realization(R, M, lnY, eps_af, af_pars, af_par_id)

    return lnY + lnAF


# SITE AMPLIFICATION
def af_median(R, M, lnY, pars, par_id=None):
    """returns the site amplification factor for the spectral
    accelerations conditional on R, M, SA,
    equations 9.12, 9.13 and 9.14 in [Bommer et al 2021]

    inputs
    R, M, SA - aligned arrays of distances, magnitudes, spectral
               accelerations
    pars     - site amplification coeffs
    """
    # organize parameter views into dictionary
    c = gen_dict_like(pars, par_id)

    # local symbols just for code readability
    Afmin, Afmax = c["AFmin"], c["AFmax"]

    # linear effects conditional on R, M, and zones/branches (through par)
    lnAFlin = af_median_linear(R, M, pars, par_id)

    # nonlinear damping conditional on SA, and zones/branches
    lnAFnonlin = af_median_nonlinear(lnY, pars, par_id)

    # total
    lnAF = lnAFlin + lnAFnonlin

    # clip
    lnAFclipped = np.clip(lnAF, np.log(Afmin), np.log(Afmax))

    return lnAFclipped


def af_max(pars, par_id=None):
    # organize parameter views into dictionary
    c = gen_dict_like(pars, par_id)

    # local symbols just for code readability
    Afmax = c["AFmax"]

    return np.log(Afmax)


def af_median_linear(R, M, pars, par_id=None):
    """returns the linear site amplification factor for the spectral
    accelerations conditional on R, M
    equations 9.12, 9.13 and 9.14 in [Bommer et al 2021]

    inputs
    R, M   - aligned arrays of distances, magnitudes
    pars   - site amplification coeffs
    """
    # organize parameter views into dictionary
    c = gen_dict_like(pars, par_id)

    # local symbols just for code readability
    a0, a1, a2, a3 = c["a0"], c["a1"], c["a2"], c["a3"]
    b0, b1, b2 = c["b0"], c["b1"], c["b2"]
    Ma, Mb, Mref2 = c["Ma"], c["Mb"], c["Mref2"]
    Rref = c["Rref"]

    # prep
    lnR = np.log(R)
    lnRref = np.log(Rref)

    # (9.14)
    Mmin, Mmax = np.minimum(Ma, Mb), np.maximum(Ma, Mb)
    Mref1 = np.clip(
        Ma + (Mb - Ma) * (lnR - np.log(3.0)) / np.log(60.0 / 3.0), Mmin, Mmax
    )

    # (9.13) - contains error - correct version in (8.2)? or still mistake?
    f1 = (
        (a0 + a1 * lnR)
        + (b0 + b1 * lnR) * (np.minimum(M, Mref1) - Mref1)
        + a2 * (lnR - lnRref) ** 2
        + b2 * (np.minimum(M, Mref1) - Mref2) ** 2
        + a3 * (np.maximum(M, Mref1) - Mref1)
    )

    return f1


def af_median_nonlinear(lnY, pars, par_id=None):
    """returns the non-linear amplification factor for the spectral
    accelerations
    equation 9.12 in [Bommer et al 2021]

    inputs
    lnY      - N-dim bc-aligned array of spectral acceleration values
    pars     - (N+1)-dim bc-aligned or N-dim structured array of site
               amplification model coefficients
    par_id   - 1-D array of model coefficient identifiers, or None
               if pars is a structured array
    """
    # organize parameter views into dictionary
    c = gen_dict_like(pars, par_id)

    # local symbols just for code readability
    f2, f3 = c["f2"], c["f3"]
    Afscale = c["AFscale"]

    # for spectral accelerations from cm/s2 to g just
    # to determine the AF - does not affect the unit of SA outside of this scope
    Y = np.exp(lnY) / Afscale

    # (9.12)
    lnAFnonlin = f2 * np.log((Y + f3) / f3)

    return lnAFnonlin


def af_realization(R, M, lnY, eps, pars, par_id=None):
    lnAF_median = af_median(R, M, lnY, pars, par_id)
    varAF = af_variance(lnY, pars, par_id)
    lnAF = lnAF_median + eps * np.sqrt(varAF)

    return lnAF


# SITE AMPLIFICATION VARIANCE
def af_variance(lnY, pars, par_id=None):
    """returns sigma fo the site amplification model for spectral accelerations
    equations 9.15 in [Bommer et al 2021]

    inputs
    lnSA   - ln of spectral acceleration (in units of g)
    pars   - site amplification coeffs for specific zone and spectral
             acceleration, coming from afcoeffs[zone][sakey]
             see also doc of this module and of PoE_calc.py
    """
    # organize parameter views into dictionary
    c = gen_dict_like(pars, par_id)

    # local symbols just for code readability
    s1, s2, xl, xh = c["s1"], c["s2"], c["xl"], c["xh"]

    # (9.15)
    smin, smax = np.minimum(s1, s2), np.maximum(s1, s2)
    phi_s2s = np.clip(s1 + (s2 - s1) * (lnY - np.log(xl)) / np.log(xh / xl), smin, smax)

    # from standard deviation to variance
    var_s2s = phi_s2s**2

    return var_s2s


# LOGIC TREE
def branch_median_weights(M):
    """return the magnitude-dependent logic tree weights according to
    equation 9.8 in [Bommer et al. 2021]
    the equation has been modified/corrected a bit by clipping at the 3.6 and 5.0
    boundaries, as intended according to Figure 9.1
    """

    Mclip = np.clip(M, 3.6, 5.0)[..., np.newaxis]
    # Mclip = Mclip.reshape(M.shape + (1,))

    # Table 9.1 [Lower, CentralLower, CentralUpper, Upper]
    wL = np.array([0.2, 0.3, 0.3, 0.2])
    wH = np.array([0.1, 0.2, 0.3, 0.4])

    # (9.8)
    weights = wL + (wH - wL) * (Mclip - 3.6) / (5.0 - 3.6)

    return weights


def branch_s2s_epsilons():
    eps = np.array([-1.645, 0.0, 1.645])
    return eps


def branch_s2s_weights():
    # weights = np.array([0.185, 0.63, 0.185])
    # note that the above reproduces the second order moment of
    # the lognormal distribution
    # the below is a (conservative - for rare events) approximation
    weights = np.array([0.2, 0.6, 0.2])
    return weights


# WIERDEN
def wierde_periods():
    periods = np.array([0.01, 0.1, 0.2, 0.5, 1.0, 2.0])
    return periods


def wierde_factors():
    factors = np.array([0.2, 0.25, 0.35, 0.35, 0.1, 0.05])
    return factors


# UTILITY
def gen_dict_like(c, p):
    if p is None:
        return c
    else:
        return {par: c[..., i] for i, par in enumerate(p)}
