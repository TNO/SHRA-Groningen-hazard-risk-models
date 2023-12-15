"""
Generate risk prep: conditional probabilities of damage/collapse/death
"""

import sys
import logging
import timeit
import xarray as xr
from dask.distributed import progress

from chaintools.chaintools.tools_configuration import preamble
from chaintools.chaintools import tools_xarray as tx


def main(args):
    module_name = "risk_prep"
    config, client = preamble(args, module_name)
    logging.info(f"starting {module_name}")
    start = timeit.default_timer()

    # open gmm configuration and tabular data
    fcm_tables = tx.open("fcm_tables", config)
    im_prep = tx.open("im_prep", config)

    # set up logic tree, separate in reference and amplification components
    gmm_logictree = im_prep[[v for v in im_prep if "logic_tree:" in v]]
    tx.store(gmm_logictree, "risk_prep", config, mode="w-", compute=True)
    fcm_logictree = fcm_tables[[v for v in fcm_tables if "logic_tree:" in v]]
    tx.store(fcm_logictree, "risk_prep", config, mode="a", compute=True)

    # MAIN calculations stage 1 - mean risk
    saavg_pmf = im_prep["surface_pmf_mean"].sel(IM_FCM="SaAvg", drop=True)
    pga_pmf = im_prep["surface_pmf_mean"].sel(IM_FCM="PGA", drop=True)

    risk_prep = get_mean_risk(
        fcm_tables,
        saavg_pmf,
        pga_pmf,
        fcm_logictree,
    )
    storage_task = tx.store(risk_prep, "risk_prep", config, mode="a", compute=False)
    job = client.compute(storage_task)
    progress(job)

    if config.get("full_logictree", False):
        # report timing
        stop = timeit.default_timer()
        total_time = stop - start
        logging.info(f"intermediate time: {total_time / 60:.2f} mins")

        # treatment of rate multiplier -- implementation of magnitude-dependent weights
        # should be used when marginalizing over logic tree
        rate_multiplier = im_prep.get("rate_multiplier", 1.0)
        if "rate_multiplier" in im_prep:
            tx.store(rate_multiplier, "risk_prep", config, mode="a", compute=True)

        # MAIN calculations stage 1 - full logic tree
        # shorthands
        saavg_pmf = im_prep["surface_pmf"].sel(IM_FCM="SaAvg", drop=True)
        pga_pmf = im_prep["surface_pmf"].sel(IM_FCM="PGA", drop=True)
        saavg_pmf.attrs["IM_FCM"] = "SaAvg"
        pga_pmf.attrs["IM_FCM"] = "PGA"
        risk_prep = get_logictree_risk(fcm_tables, saavg_pmf, pga_pmf)
        storage_task = tx.store(risk_prep, "risk_prep", config, mode="a", compute=False)
        job = client.compute(storage_task)
        progress(job)

    # report timing
    stop = timeit.default_timer()
    total_time = stop - start
    logging.info(f"total time: {total_time / 60:.2f} mins")

    return


def get_mean_risk(
    fcm_tables,
    saavg_pmf_mean,
    pga_pmf_mean,
    fcm_logictree,
):
    # first contract over logic tree
    lt_dims = [v for v in fcm_logictree.dims if v.startswith("branch_")]

    structural_poe_mean = xr.dot(
        fcm_tables["structural_poe"],
        *fcm_logictree.values(),
        dims=lt_dims,
        optimize=True,
    )
    structural_pod_mean = xr.dot(
        fcm_tables["structural_pod"],
        *fcm_logictree.values(),
        dims=lt_dims,
        optimize=True,
    )
    chimney_poe_mean = xr.dot(
        fcm_tables["chimney_poe"],
        *fcm_logictree.values(),
        dims=lt_dims,
        optimize=True,
    )
    chimney_pod_mean = xr.dot(
        fcm_tables["chimney_pod"],
        *fcm_logictree.values(),
        dims=lt_dims,
        optimize=True,
    )

    # then contract over gm_surface
    risk_prep = xr.Dataset(
        {
            "structural_poe_mean": xr.dot(
                saavg_pmf_mean,
                structural_poe_mean,
                dims=["gm_surface"],
            ),
            "structural_pod_mean": xr.dot(
                saavg_pmf_mean,
                structural_pod_mean,
                dims=["gm_surface"],
            ),
            "chimney_poe_mean": xr.dot(
                pga_pmf_mean,
                chimney_poe_mean,
                dims=["gm_surface"],
            ),
            "chimney_pod_mean": xr.dot(
                pga_pmf_mean,
                chimney_pod_mean,
                dims=["gm_surface"],
            ),
        }
    )

    risk_prep["LPR_mean"] = calculate_LPR(
        risk_prep["structural_pod_mean"],
        risk_prep["chimney_pod_mean"],
    )

    return risk_prep


def get_logictree_risk(fcm_tables, saavg_pmf, pga_pmf):
    risk_prep = xr.Dataset(
        {
            "structural_poe": xr.dot(
                saavg_pmf,
                fcm_tables["structural_poe"],
                dims=["gm_surface"],
            ),
            "structural_pod": xr.dot(
                saavg_pmf,
                fcm_tables["structural_pod"],
                dims=["gm_surface"],
            ),
            "chimney_poe": xr.dot(
                pga_pmf,
                fcm_tables["chimney_poe"],
                dims=["gm_surface"],
            ),
            "chimney_pod": xr.dot(
                pga_pmf,
                fcm_tables["chimney_pod"],
                dims=["gm_surface"],
            ),
        }
    )

    risk_prep["LPR"] = calculate_LPR(
        risk_prep["structural_pod"],
        risk_prep["chimney_pod"],
    )

    return risk_prep


def calculate_LPR(structural_pod, chimney_pod):
    LPR_outside = structural_pod.sel({"location": "outside"}) + chimney_pod
    LPR_inside = structural_pod.sel({"location": "inside"})
    LPR = 0.99 * LPR_inside + 0.01 * LPR_outside

    return LPR


if __name__ == "__main__":
    main(sys.argv)
