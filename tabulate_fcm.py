"""
Generate tables of fragity and consequence model parameters conditional
on surface ground motions
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
    module_name = "fcm_tables"
    config, client = preamble(args, module_name)
    logging.info(f"starting {module_name}")
    start = timeit.default_timer()

    # set up coordinates dataset and prepare for DASK
    coords_ds = tx.prepare_ds(config)

    # open gmm configuration data
    fcm_config = tx.open("fcm_config", config)

    # generate samples - prepare DASK task
    logging.info("generating tables")
    samples = fcm_tables(fcm_config, coords_ds)

    # set up output logistics
    storage_task = tx.store(samples, module_name, config, compute=False)

    # execute
    job = client.compute(storage_task)
    progress(job)

    stop = timeit.default_timer()
    total_time = stop - start
    logging.info(f"total time: {total_time / 60:.2f} mins")

    return


def fcm_tables(fcm_config, sample_ds):
    """
    Generate tables of fragity and consequence model parameters conditional
    on surface ground motions

    Parameters
    ----------
    fcm_config : xarray.Dataset
        Configuration data for fragility and consequence models
    sample_ds : xarray.Dataset
        Rudimentary Dataset with dimension information


    Returns
    -------
    output_ds : xarray.Dataset or dict
        Dataset with fragility and consequence model parameters

    """
    table_ds = sample_ds

    # define shorthands from sample structure
    lnsa = np.log(table_ds["SA_g_surface"])

    # generate fragility model samples
    str_poe = fcm_structural_poe(lnsa, fcm_config)
    str_pod = fcm_structural_pod(str_poe, fcm_config)
    str_pod_cond_poe = fcm_structural_pod_cond_poe(fcm_config)
    chm_poe = fcm_chimney_poe(lnsa, fcm_config)
    chm_pod = fcm_chimney_pod(lnsa, fcm_config)
    table_ds = table_ds.merge(
        {
            "structural_poe": str_poe,
            "structural_pod": str_pod,
            "structural_pod_cond_poe": str_pod_cond_poe,
            "chimney_poe": chm_poe,
            "chimney_pod": chm_pod,
        }
    ).fillna(0.0)

    # copy logic tree weights
    logictree = fcm_config[[v for v in fcm_config if "logic_tree:" in v]]
    table_ds = table_ds.merge(logictree)

    return table_ds


def fcm_structural_pod_cond_poe(fcm_config):
    # lookup table for probability of dying conditional on exceeding
    # a certain limit state
    pod = fcm_config["probability_of_dying"].sel({"limit_state": ["CS1", "CS2", "CS3"]})

    # generate table for additional probability of dying conditional on reaching
    # a certain limit state (having accounted for the states below)
    pod_cond_poe = pod - pod.shift(limit_state=1, fill_value=0.0)

    return pod_cond_poe


def fcm_structural_pod(poe, fcm_config):
    # lookup table for probability of dying conditional on exceeding
    # a certain limit state
    pod_cond_poe = fcm_structural_pod_cond_poe(fcm_config)

    # marginalize over limit states
    pod = xr.dot(pod_cond_poe, poe, dims=["limit_state"])

    return pod


def fcm_structural_poe(lnIM_g, fcm_config):
    b0 = fcm_config["fragility_parameters"].sel(
        {"parameter_fragility": "b0"}, drop=True
    )
    b1 = fcm_config["fragility_parameters"].sel(
        {"parameter_fragility": "b1"}, drop=True
    )
    sigma = fcm_config["fragility_parameters"].sel(
        {"parameter_fragility": "sigma"}, drop=True
    )
    lnSD_lim = np.log(fcm_config["displacement_limit"])

    # in the original formulation, the number
    # of standard deviations is defined as:
    # epsilon = (lnSD - lnSD_lim) / sigma,
    # and since the displacement is defined as:
    # lnSD = b0 + b1 * lnIM_g
    # we can express
    # hence, also:

    lnIM_g_sigma = sigma / b1
    lnIM_g_limit = (lnSD_lim - b0) / b1
    epsilon = (lnIM_g - lnIM_g_limit) / lnIM_g_sigma

    poe = xr.apply_ufunc(st.norm.cdf, epsilon, dask="allowed")

    return poe


def fcm_chimney_poe(lnPGA, fcm_config):
    mu = fcm_config["fragility_parameters"].sel(
        {"parameter_fragility": "Median_PGA_chf"}, drop=True
    )
    beta = fcm_config["fragility_parameters"].sel(
        {"parameter_fragility": "beta_chf"}, drop=True
    )

    beta = beta.where(beta > 0.0)  # nan where no info
    lnPGA_lim = np.log(mu.where(mu > 0.0))  # nan where no info

    epsilon = (lnPGA - lnPGA_lim) / beta
    poe = xr.apply_ufunc(st.norm.cdf, epsilon, dask="allowed")
    poe = poe.fillna(0.0)

    return poe


def fcm_chimney_pod(lnPGA, fcm_config):
    mu = fcm_config["consequence_parameters"].sel(
        {"parameter_consequence": "Median_PGA_chd"}, drop=True
    )
    beta = fcm_config["consequence_parameters"].sel(
        {"parameter_consequence": "beta_chd"}, drop=True
    )

    beta = beta.where(beta > 0.0)  # nan where no info
    lnPGA_lim = np.log(mu.where(mu > 0.0))  # nan where no info

    epsilon = (lnPGA - lnPGA_lim) / beta
    pod = xr.apply_ufunc(st.norm.cdf, epsilon, dask="allowed")
    pod = pod.fillna(0.0)

    return pod


if __name__ == "__main__":
    main(sys.argv)
