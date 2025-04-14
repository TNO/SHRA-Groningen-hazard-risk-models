"""
Generate hazard prep: conditional probabilities of exceedance
"""

import sys
import logging
import timeit
import scipy.stats as st
import numpy as np
import xarray as xr

from chaintools.chaintools.tools_configuration import preamble
from chaintools.chaintools import tools_xarray as tx
from chaintools.chaintools import tools_grid as tg


def main(args):
    config = preamble(args)
    logging.info("starting module %s in file %s", __name__, __file__)
    start = timeit.default_timer()

    run_core(config)

    stop = timeit.default_timer()
    total_time = stop - start
    logging.info(f"total time: {total_time / 60:.2f} mins")

    return


def run_core(config):
    # open gmm configuration and tabular data
    gmm_tables = tx.open("gmm_tables", config)
    weights = tx.open("weights", config)
    rate_multiplier = tx.open("rate_multiplier", config)

    # STAGE 0: set up dimensions and coordinates for ground motion, then store
    output_ds = tx.prepare_ds(config)
    lnSA_ref = np.log(output_ds["SA_reference"])
    lnSA_srf = np.log(output_ds["SA_surface"])
    tx.store(output_ds, "output", config, mode=config.get("file_mode", "w-"))

    # STAGE 1: calculate reference ground motion distributions
    logging.info("calculating reference exceedence probabilities")
    reference_poe = calculate_reference_poe(gmm_tables, lnSA_ref)
    reference_poe = tx.weighted_sum(reference_poe, weights, rate_multiplier)
    output_ds["reference_poe"] = reference_poe
    tx.store(output_ds, "output", config)

    # STAGE 2: compute probability of exceedance (poe) conditional on reference
    # ground motions (tabulated along gm_reference dimension)
    # delay actual computation until the next stage
    conditional_surface_poe = calculate_conditional_surface_poe(
        gmm_tables, lnSA_ref, lnSA_srf, config
    )

    # STAGE 3: marginalize over reference ground motions
    # (conditional on magnitude/distance), and over the core dimension
    # of the provided provided weights -- most probably the logic tree --,
    # to obtain surface poe, still conditional on magnitude/distance
    logging.info("calculating surface exceedence probabilities")
    reference_pmf = tg.bin_diff(reference_poe, "gm_reference", fill_value=0.0).persist()
    output_ds["surface_poe"] = calculate_surface_poe(
        reference_pmf, conditional_surface_poe, weights
    )
    tx.store(output_ds, "output", config)


def calculate_reference_poe(gmm_tables, im_ref):
    """
    Calculate exceedence probabilities of all ground motion components at reference level
    """
    # shorthands
    median = gmm_tables["reference_median"]
    sd = np.sqrt(gmm_tables["reference_variance"])

    # calculate exceedence probabilities
    reference_poe = xr.apply_ufunc(
        st.norm.sf,
        im_ref,
        median,
        sd,
        dask="parallelized",
        output_dtypes=[float],
    )

    return reference_poe


def calculate_surface_poe(reference_pmf, conditional_poe_srf, weights):
    relevant_weights, marginalize_dims = tx.prepare_weights(
        weights, reference_pmf, conditional_poe_srf
    )
    marginalize_dims = marginalize_dims | {"gm_reference"}

    surface_poe = xr.dot(
        reference_pmf,
        conditional_poe_srf,
        *relevant_weights.values(),
        dim=marginalize_dims,
        optimize=True,
    )

    return surface_poe


def calculate_conditional_surface_poe(gmm_tables, lnSA_ref, lnSA_srf, config):
    """
    Calculate conditional probability of exceedance at surface
    given reference ground motions (tabulated along gm_reference dimension)
    """

    # treatment of wierden -- imported from V7, allowed in other models
    # note that this is actually a log of a factor
    wierde_factor = gmm_tables.get("wierde_factor", 0.0)

    # shorthands
    lnAF = gmm_tables["af_median"]
    lnAF_std = np.sqrt(gmm_tables["af_variance"])
    s2s_epsilons = gmm_tables["s2s_epsilons"]

    # median motions at surface level
    median = lnSA_ref + lnAF + wierde_factor

    # next steps depend on the treatment of s2s variability, either
    # as aleatory or epistemic
    s2s_mode = get_s2s_mode(gmm_tables, config)
    if s2s_mode == "aleatory":
        # AF is modeled as a lognormal distribution
        # first interpolated linearly to the center of the gm_reference bin (between two nodes)
        mu = tg.bin_average(median, "gm_reference")
        sigma = tg.bin_average(lnAF_std, "gm_reference")
        # then, conditional poe is computed using the survival function
        conditional_poe_srf = xr.apply_ufunc(
            st.norm.sf,
            lnSA_srf,
            mu,
            sigma,
            dask="parallelized",
        ).fillna(0.0)
    elif s2s_mode == "epistemic":
        # AF is modeled as a 3pt discrete distribution (3pt on 3 s2s branches)
        delta_lnAF = s2s_epsilons * lnAF_std
        realization = median + delta_lnAF
        # construct a linear off-ramp function corresponding to the gm_reference bin
        # first, determine the range in gm_surface occupied by the gm_reference bin
        delta = (
            tg.bin_diff(-1 * realization, "gm_reference").fillna(0.0).clip(1e-10, None)
        )
        # then, construct a linear function on that range and clip it,
        # thus forming the linear off-ramp
        mu = tg.bin_average(realization, "gm_reference")
        conditional_poe_srf = (0.5 - (lnSA_srf - mu) / delta).fillna(0.0).clip(0.0, 1.0)
        # this is more or less analogous to aleatory case,
        # where we have a sigmoid function in place (sf of normal distribution)
    else:
        raise ValueError(f"unknown s2s_mode: {s2s_mode}")

    return conditional_poe_srf


def get_s2s_mode(gmm_tables, config):
    gmm_version = gmm_tables["gmm_version"]
    if gmm_version in ["GMM-V5", "GMM-V6"]:
        s2s_mode_default = "aleatory"
    else:
        s2s_mode_default = "epistemic"

    return config.get("s2s_mode", s2s_mode_default)


if __name__ == "__main__":
    main(sys.argv)
