import numpy as np


# REFERENCE
def reference_median(R, M, T, pars, par_id=None):
    """returns the sum of the path and source term of the ground motion model
    equation 5.1 in [Bommer et al. 2019]

    inputs
    R, M, T - broadcast-ready (aligned) numpy arrays representing
              distances, magnitudes, spectral periods
    pars    - ground motion model coefficients aligned with R, M, T,
              can be a regular array or a structured array
    par_id  - array with parameter names (ids), if None then assume
              pars is structured array

    output
    """

    gs = source_median(M, pars, par_id=par_id)
    gp = path_median(R, M, T, pars, par_id=par_id)
    lnY = gs + gp

    return lnY


def reference_median_ds(R, M, pars, par_id=None):
    """returns the sum of the path and source term of the ground motion model
    equation 5.1 in [Bommer et al. 2019]

    inputs
    R, M - broadcast-ready (aligned) numpy arrays representing
              distances, magnitudes, spectral periods
    pars    - ground motion model coefficients aligned with R, M,
              can be a regular array or a structured array
    par_id  - array with parameter names (ids), if None then assume
              pars is structured array

    output
    """

    gs = source_median_ds(M, pars, par_id=par_id)
    gp = path_median_ds(R, M, pars, par_id=par_id)
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


# PATH
def path_median(R, M, T, pars, par_id=None):
    """returns the path term of the ground motion model
    equation 5.3a-c in [Bommer et al. 2019]
    all input arrays should be broadcast-ready, i.e., with
    properly aligned dimensions, but can have any number of
    dimensions

    inputs
    R, M, T - broadcast-ready (aligned) numpy arrays representing
              distances, magnitudes, spectral periods
    pars    - ground motion model coefficients aligned with R, M, T
    par_id - index array for parameter names/id's
    """

    # organize parameter views into dictionary
    c = gen_dict_like(pars, par_id)

    # local symbols just for code readability
    r0, r1, r2, r3, r4, r5 = c["r0"], c["r1"], c["r2"], c["r3"], c["r4"], c["r5"]

    # division of distance range into 4 sections
    lnR0 = np.log(np.clip(R / 3.0, 1.0, 7.0 / 3.0))
    lnR1 = np.log(np.clip(R / 7.0, 1.0, 12.0 / 7.0))
    lnR2 = np.log(np.clip(R / 12.0, 1.0, None))

    # (5.3a, b, c)
    gp = (r0 + r1 * M) * lnR0 + (r2 + r3 * M) * lnR1 + (r4 + r5 * M) * lnR2
    return gp


def path_median_ds(R, M, pars, par_id=None):
    """
    inputs
    R, M    - broadcast-ready (aligned) numpy arrays representing
              distances, magnitudes, spectral periods
    pars    - ground motion model coefficients aligned with R, M, T
    par_id - index array for parameter names/id's
    """

    # organize parameter views into dictionary
    c = gen_dict_like(pars, par_id)

    # local symbols just for code readability
    r6, r7, r8, r9, r10 = c["r6"], c["r7"], c["r8"], c["r9"], c["r10"]

    # division of distance range into 2 sections
    lnR0 = np.log(np.clip(R / 3.0, 1.0, 12.0 / 3.0))
    lnR1 = np.log(np.clip(R / 12.0, 1.0, None))

    # clip range of M
    Mclip = np.clip(M, 3.25, 6.0)

    # (6.14a, b, c)
    gp = (r6 + r7 * Mclip) * lnR0**r8 + (r9 + r10 * Mclip) * lnR1

    return gp


# SOURCE
def source_median(M, pars, par_id=None):
    """returns the source term of the ground motion model
    equation 5.2a-c in [Bommer et al. 2019]
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

    # local symbols just for code readability
    m0, m1, m2, m3, m4, m5 = c["m0"], c["m1"], c["m2"], c["m3"], c["m4"], c["m5"]

    # (5.2a, b, c)
    gs0 = m0 + m1 * (M - 4.7) + m2 * (M - 4.7) ** 2
    gs1 = m0 + m3 * (M - 4.7)
    gs2 = m0 + m3 * (5.45 - 4.7) + m4 * (M - 5.45) + m5 * (M - 5.45) ** 2

    # looks a bit awkard but hey - as long as broadcasting works
    gs = np.where(M <= 4.7, gs0, gs1)
    gs = np.where(M > 5.45, gs2, gs)

    return gs


def source_median_ds(M, pars, par_id=None):
    """returns the source term of the ground motion model
    equation 5.2a-c in [Bommer et al. 2019]
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

    # local symbols just for code readability
    m6, m7, m8, m9 = c["m6"], c["m7"], c["m8"], c["m9"]

    # cliprange in M
    Mclip = np.clip(M, 3.25, None)

    # (6.13a, b)
    fs0 = m6 + m7 * (Mclip - 5.25)
    fs1 = m6 + m8 * (M - 5.25) + m9 * (M - 5.25) ** 2

    fs = np.where(M <= 5.25, fs0, fs1)

    return fs


# REFERENCE VARIANCE
def c2c_variance(R, M, T):
    """returns the component-to-component variance
    equations 5.5 and 5.6 in [Bommer et al 2019]

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


def c2c_variance_ds(R, M):
    """returns the component-to-component variance
    equation 6.16 in [Bommer et al 2018]

    inputs
    R, M - broadcast-ready (aligned) numpy arrays representing
              distances [km], magnitudes, spectral periods [s]
    """

    # (6.16)
    Mclip = np.clip(M, 3.6, 5.6)
    var_c2c = 0.0299 + 2.434 * (5.6 - Mclip) * R**-1.95

    return var_c2c


def reference_gm_variance(tau, phi_ss):
    """returns the component-to-component variance
    equations 5.4a-b in [Bommer et al 2019]

    inputs
    tau, phiss - broadcast-ready (aligned) numpy arrays representing
              distances [km], magnitudes, spectral periods [s], source (tau)
              and path (phi_ss) standard deviations
    """

    total_variance = tau**2 + phi_ss**2

    return total_variance


def reference_gm_variance_ds(pars, par_id=None):
    """returns the component-to-component variance
    equations 5.4a-b in [Bommer et al 2019]

    inputs

    """
    # organize parameter views into dictionary
    c = gen_dict_like(pars, par_id)

    # local symbols just for code readability

    total_variance = c["tau"] ** 2 + c["phi"] ** 2

    return total_variance


def reference_ac_variance(R, M, T, tau, phi_ss):
    """returns the component-to-component variance
    equations 5.4a-b in [Bommer et al 2019]

    inputs
    R, M, T, tau, phiss - broadcast-ready (aligned) numpy arrays representing
              distances [km], magnitudes, spectral periods [s], source (tau)
              and path (phi_ss) standard deviations
    """

    total_variance = reference_gm_variance(tau, phi_ss) + c2c_variance(R, M, T)

    return total_variance


def reference_ac_variance_ds(R, M, pars, par_id=None):
    """returns the component-to-component variance
    equations 5.4a-b in [Bommer et al 2019]

    inputs
    R, M,  - broadcast-ready (aligned) numpy arrays representing
              distances [km], magnitudes
    """

    ref_variance = reference_gm_variance_ds(pars, par_id=par_id)

    total_variance = ref_variance + c2c_variance_ds(R, M)

    return total_variance


# SURFACE
def surface_median(R, M, T, ref_pars, af_pars, ref_par_id=None, af_par_id=None):
    """returns a quasi-median consisting of the combination of median reference
    and median amplification conditional on the median reference

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
    R, M, T - broadcast-ready (aligned) numpy arrays representing
              distances, magnitudes, spectral periods
    pars    - ground motion model coefficients aligned with R, M, T,
              can be a regular array or a structured array
    par_id  - array with parameter names (ids), if None then assume
              pars is structured array
    """
    # lnY_median = reference_median(R, M, T, ref_pars, ref_par_id)
    # varY = reference_ac_variance(R, M, T, tau, phi_ss)
    # lnY = lnY_median + eps_ref * np.sqrt(varY)
    # lnAF_median = af_median(R, M, lnY, af_pars, af_par_id)
    # varAF = af_variance(lnY, af_pars, af_par_id)
    # lnAF = lnAF_median + eps_af * np.sqrt(varAF)
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
    R, M, T - broadcast-ready (aligned) numpy arrays representing
              distances, magnitudes, spectral periods
    pars    - ground motion model coefficients aligned with R, M, T,
              can be a regular array or a structured array
    par_id  - array with parameter names (ids), if None then assume
              pars is structured array
    """
    # lnY_median = reference_median(R, M, T, ref_pars, ref_par_id)
    # varY = reference_ac_variance(R, M, T, tau, phi_ss)
    # lnY = lnY_median + eps_ref * np.sqrt(varY)
    # lnAF_median = af_median(R, M, lnY, af_pars, af_par_id)
    # varAF = af_variance(lnY, af_pars, af_par_id)
    # lnAF = lnAF_median + eps_af * np.sqrt(varAF)
    lnY = reference_gm_realization(R, M, T, tau, phi_ss, eps_ref, ref_pars, ref_par_id)
    lnAF = af_realization(R, M, lnY, eps_af, af_pars, af_par_id)

    return lnY + lnAF


# SITE AMPLIFICATION
def af_median(R, M, lnSA, pars, par_id=None):
    """returns the site amplification factor for the spectral
    accelerations conditional on R, M, SA,
    equations 5.7a-b in [Bommer et al 2019]

    inputs
    R, M, SA - aligned arrays of distances, magnitudes, spectral
               accelerations
    pars     - site amplification coeffs
    """

    # organize parameter views into dictionary
    c = gen_dict_like(pars, par_id)

    # local symbols just for code readability
    Afmin, Afmax = c["Afmin"], c["Afmax"]

    # linear effects conditional on R, M, and zones/branches (through par)
    lnAFlin = af_median_linear(R, M, pars, par_id)

    # nonlinear damping conditional on linear AF, SA, and zones/branches
    lnAFnonlin = af_median_nonlinear(lnSA, pars, par_id)

    # total
    lnAF = lnAFlin + lnAFnonlin

    # clip
    lnAFclipped = np.clip(lnAF, np.log(Afmin), np.log(Afmax))

    return lnAFclipped


def af_max(pars, par_id=None):
    # organize parameter views into dictionary
    c = gen_dict_like(pars, par_id)

    # local symbols just for code readability
    Afmax = c["Afmax"]

    return np.log(Afmax)


def af_median_linear(R, M, pars, par_id=None):
    """returns the linear site amplification factor for the spectral
    accelerations conditional on R, M
    equations 5.8 in [Bommer et al 2019]

    inputs
    R, M   - aligned arrays of distances, magnitudes
    pars   - site amplification coeffs
    """
    # organize parameter views into dictionary
    c = gen_dict_like(pars, par_id)

    # local symbols just for code readability
    a0, a1 = c["a0"], c["a1"]
    b0, b1 = c["b0"], c["b1"]
    M1, M2 = c["M1"], c["M2"]

    # prep
    lnR = np.log(R)

    # (5.9)
    Mref = M1 - ((lnR - np.log(3.0)) / np.log(60.0 / 3.0)) * (M1 - M2)

    # (5.8)
    f1 = (a0 + a1 * lnR) + (b0 + b1 * lnR) * np.clip(M - Mref, None, 0)

    return f1


def af_median_nonlinear(lnY, pars, par_id=None):
    """returns the non-linear amplification factor for the spectral
    accelerations
    equation 5.7 in [Bommer et al 2019]

    inputs
    lnY      - N-dim bc-aligned array of intensity values
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


def af_median_linear_pgv(R, M, pars, par_id=None):
    """returns the linear site amplification factor for the spectral
    accelerations conditional on R, M
    equations 5.10 in [Bommer et al 2019]

    inputs
    R, M   - aligned arrays of distances, magnitudes
    pars   - site amplification coeffs
    """
    # organize parameter views into dictionary
    c = gen_dict_like(pars, par_id)

    # local symbols just for code readability
    a0, a1 = c["a0"], c["a1"]
    b0, b1 = c["b0"], c["b1"]
    M1, d = c["M1"], c["d"]

    # prep
    lnR = np.log(R)

    # (5.10)
    Mlo = np.clip(M, None, M1)
    Mhi = np.clip(M, M1, None)
    f1 = (a0 + a1 * lnR) + (b0 + b1 * lnR) * (Mlo - M1) + d * (Mhi - M1)

    return f1


def af_median_linear_ds(vs30):
    """returns duration site amplification factor
    equation 6.15 in [Bommer et al 2018]
    input:
    vs30  -  float, average shear wave velocity of top 30m
    """
    fsite = -0.2246 * np.log(np.minimum([vs30, 600.0]) / 600.0)
    return fsite


def af_realization(R, M, lnY, eps, pars, par_id=None):
    lnAF_median = af_median(R, M, lnY, pars, par_id)
    varAF = af_variance(lnY, pars, par_id)
    lnAF = lnAF_median + eps * np.sqrt(varAF)

    return lnAF


# SITE AMPLIFICATION VARIANCE
def af_variance(lnY, pars, par_id=None):
    """returns sigma fo the site amplification model for spectral accelerations
    equations 5.11 in [Bommer et al 2019]

    inputs
    lnSY   - ln of spectral acceleration (in units of g) or PGV (cm/s)
    pars   - site amplification coeffs for specific zone and spectral
             acceleration, coming from afcoeffs[zone][sakey]
             see also doc of this module and of PoE_calc.py
    """
    # organize parameter views into dictionary
    c = gen_dict_like(pars, par_id)

    # local symbols just for code readability
    s1, s2, xl, xh = c["phiS2S_1"], c["phi_S2S_2"], c["Sa_low"], c["Sa_high"]

    # (9.15)
    smin, smax = np.minimum(s1, s2), np.maximum(s1, s2)
    phi_s2s = np.clip(s1 + (s2 - s1) * (lnY - np.log(xl)) / np.log(xh / xl), smin, smax)

    # from standard deviation to variance
    var_s2s = phi_s2s**2

    return var_s2s


# UTILITY
def gen_dict_like(c, p):
    if p is None:
        return c
    else:
        return {par: c[..., i] for i, par in enumerate(p)}
