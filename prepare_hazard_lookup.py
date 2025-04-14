"""
Generate hazard prep: conditional probabilities of exceedance
"""
import sys
import logging
import timeit
import scipy.stats as st
import numpy as np
import xarray as xr
from dask.distributed import progress

from chaintools.chaintools.tools_configuration import preamble
from chaintools.chaintools import tools_xarray as tx


def main(args):
    module_name = "hazard_prep"
    config, client = preamble(args, module_name)
    logging.info(f"starting {module_name}")
    start = timeit.default_timer()

    # open gmm configuration and tabular data
    gmm_tables = tx.open("gmm_tables", config)

    # STAGE 1: compute probability of exceedance (poe) conditional on reference
    # ground motions (tabulated along gm_reference dimension)
    conditional_poe_srf = get_conditional_exceedance_curves(gmm_tables, config)

    # STAGE 2: marginalize over reference ground motions
    # (conditional on magnitude/distance), and over logic tree,
    # to obtain surface poe conditional on magnitude/distance
    logging.info("calculating mean exceedance curves at surface")
    poe_mean = get_mean_exceedance_curves(gmm_tables, conditional_poe_srf)
    storage_task = tx.store(poe_mean, "hazard_prep", config, mode="w-", compute=False)
    job = client.compute(storage_task)
    progress(job)

    # full logic tree only if requested
    if config.get("full_logictree", False):
        # report intermediate timing
        stop = timeit.default_timer()
        total_time = stop - start
        logging.info(f"intermediate time: {total_time / 60:.2f} mins")

        # marginalize surface ground motions over gm_reference only
        logging.info("calculating full logic tree exceedance curves at surface")
        poe = get_full_lt_exceedence_curves(gmm_tables, conditional_poe_srf)
        storage_task = tx.store(poe, "hazard_prep", config, mode="a", compute=False)
        job = client.compute(storage_task)
        progress(job)

    # report timing
    stop = timeit.default_timer()
    total_time = stop - start
    logging.info(f"total time: {total_time / 60:.2f} mins")

    return


def get_full_lt_exceedence_curves(gmm_tables, conditional_poe_srf):
    # treatment of rate multiplier -- implementation of magnitude-dependent weights
    # should be used when marginalizing over logic tree
    rate_multiplier = gmm_tables.get("rate_multiplier", 1.0)

    poe_srf = xr.dot(
        gmm_tables["reference_pmf"],
        conditional_poe_srf,
        dims=["gm_reference"],
        optimize=True,
    )

    poe = xr.Dataset(
        {
            "surface_poe": poe_srf,
            "reference_poe": gmm_tables["reference_poe"],
            "rate_multiplier": rate_multiplier,
        }
    )

    return poe


def get_mean_exceedance_curves(gmm_tables, conditional_poe_srf):
    # get logic tree
    logictree, logictree_af, logictree_af_dims = get_logic_tree(gmm_tables)

    # marginalize conditional motions over logic tree to get the mean
    conditional_poe_srf_mean = xr.dot(
        conditional_poe_srf,
        *logictree_af.values(),
        dims=logictree_af_dims,
        optimize=True,
    )

    # marginalize conditional mean surface motions over mean reference motions
    poe_srf_mean = xr.dot(
        gmm_tables["reference_pmf_mean"],
        conditional_poe_srf_mean,
        dims=["gm_reference"],
    )

    # collect and store
    poe_mean = xr.Dataset(
        {
            "surface_poe_mean": poe_srf_mean,
            "reference_poe_mean": gmm_tables["reference_poe"],
        }
    ).merge(logictree)

    return poe_mean


def get_conditional_exceedance_curves(gmm_tables, config):
    """
    Calculate conditional probability of exceedance at surface
    given reference ground motions (tabulated along gm_reference dimension)
    """
    # treatment of site-to-site (s2s) variability
    # defaults can be overridden in config
    gmm_version = gmm_tables["gmm_version"]
    if gmm_version in ["GMM-V5", "GMM-V6"]:
        s2s_mode_default = "aleatory"
    else:
        s2s_mode_default = "epistemic"
    s2s_mode = config.get("s2s_mode", s2s_mode_default)

    # treatment of wierden -- imported from V7, allowed in other models
    # note that this is actually a log of a factor
    wierde_factor = gmm_tables.get("wierde_factor", 0.0)

    # shorthands
    lnAF = gmm_tables["af_median"]
    lnAF_std = np.sqrt(gmm_tables["af_variance"])
    s2s_epsilons = gmm_tables["s2s_epsilons"]
    lnSA_ref = np.log(gmm_tables["SA_reference"])
    lnSA_srf = np.log(gmm_tables["SA_surface"])
    surface_nodes = lnSA_srf

    # median motions at surface level
    median = lnSA_ref + lnAF + wierde_factor

    if s2s_mode == "aleatory":
        # AF is modeled as a lognormal distribution
        # first interpolated linearly to the center of the gm_reference bin (between two nodes)
        mu = bin_interpolate(median, "gm_reference")
        sigma = bin_interpolate(lnAF_std, "gm_reference")
        # then, conditional poe is computed using the survival function
        conditional_poe_srf = xr.apply_ufunc(
            st.norm.sf, surface_nodes, mu, sigma, dask="parallelized"
        ).fillna(0.0)
    elif s2s_mode == "epistemic":
        # AF is modeled as a 3pt discrete distribution (3pt on 3 s2s branches)
        delta_lnAF = s2s_epsilons * lnAF_std
        realization = median + delta_lnAF
        # construct a linear off-ramp function corresponding to the gm_reference bin
        # first, determine the range in gm_surface occupied by the gm_reference bin
        delta = bin_diff(realization, "gm_reference").fillna(0.0).clip(1e-10, None)
        # then, construct a linear function on that range and clip it,
        # thus forming the linear off-ramp
        mu = bin_interpolate(realization, "gm_reference")
        conditional_poe_srf = (
            (0.5 - (surface_nodes - mu) / delta).fillna(0.0).clip(0.0, 1.0)
        )
        # this is more or less analogous to aleatory case,
        # where we have a sigmoid function in place (sf of normal distribution)
    else:
        raise ValueError(f"unknown s2s_mode: {s2s_mode}")

    return conditional_poe_srf


def get_logic_tree(gmm_tables):
    logictree = gmm_tables[[v for v in gmm_tables if "logic_tree:" in v]]
    logictree_af_dims = ["branch_s2s"]
    logictree_ref = logictree.drop_dims(logictree_af_dims)
    logictree_ref_dims = [d for d in logictree_ref.dims if d.startswith("branch")]
    logictree_af = logictree.drop_dims(logictree_ref_dims)
    return logictree, logictree_af, logictree_af_dims


def bin_diff(value, dim, fill_value=None):
    if fill_value is None:
        shift_value = value.shift({dim: -1})
    else:
        shift_value = value.shift({dim: -1}, fill_value=fill_value)

    return shift_value - value


def bin_interpolate(value, dim, fill_value=None):
    if fill_value is None:
        shift_value = value.shift({dim: -1})
    else:
        shift_value = value.shift({dim: -1}, fill_value=fill_value)

    value = 0.5 * (value + shift_value)
    return value


if __name__ == "__main__":
    main(sys.argv)
