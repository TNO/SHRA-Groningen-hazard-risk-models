"""
Generate risk prep: conditional probabilities of damage/collapse/death
"""

import sys
import logging
import timeit
import xarray as xr
from tqdm import tqdm

from chaintools.chaintools.tools_configuration import preamble
from chaintools.chaintools import tools_xarray as tx


def assign_defaults(config):
    config.setdefault("file_mode", "w-")
    return config


def main(args):
    config = preamble(args)
    assign_defaults(config)
    logging.info("starting module %s in file %s", __name__, __file__)
    start = timeit.default_timer()

    run_core(config)

    # report timing
    stop = timeit.default_timer()
    total_time = stop - start
    logging.info(f"total time: {total_time / 60:.2f} mins")

    return


def run_core(config):
    # open gmm configuration and tabular data
    fcm_tables = tx.open("fcm_tables", config)
    im_prep = tx.open("im_prep", config)
    weights = tx.open("weights", config)

    # preprocessing
    if weights is None:
        weights = xr.Dataset()
    fcm_tables.persist()
    weights.persist()

    # calculate risk prep
    logging.info("integrate gmm and fcm models")
    storage_kwargs = {"mode": config["file_mode"]}
    for zone in tqdm(im_prep["zone"].data, desc="zone"):
        prep_z = im_prep.sel(zone=[zone])
        risk_prep = calculate_risk_prep(fcm_tables, prep_z, weights)
        tx.store(risk_prep, "output", config, **storage_kwargs)
        storage_kwargs["append_dim"] = "zone"
        storage_kwargs["mode"] = "a"


def calculate_poe(im_pmf, fragility, rate_multiplier, weights):
    weights, marginalize_dims = tx.prepare_weights(weights, im_pmf, fragility)
    marginalize_dims = marginalize_dims | {"gm_surface"}

    # V7: include rate multiplier; harmless in other cases
    if "b_median" not in weights.dims:
        rate_multiplier = xr.DataArray(1.0)

    return xr.dot(
        im_pmf,
        rate_multiplier,
        fragility,
        *weights.values(),
        dim=marginalize_dims,
        optimize=True,
    )


def calculate_risk_prep(fcm_tables, im_prep, weights):
    # prepare IMs
    saavg_pmf = im_prep["surface_pmf"].sel(IM_FCM="SaAvg", drop=True)
    pga_pmf = im_prep["surface_pmf"].sel(IM_FCM="PGA", drop=True)
    saavg_pmf.attrs["IM_FCM"] = "SaAvg"
    pga_pmf.attrs["IM_FCM"] = "PGA"

    # prepare output
    risk_prep = xr.Dataset()

    # treatment of rate multiplier
    if "rate_multiplier" in im_prep:
        rate_multiplier = im_prep["rate_multiplier"]
        if "b_median" not in weights.dims:
            risk_prep["rate_multiplier"] = rate_multiplier
    else:
        rate_multiplier = xr.DataArray(1.0)

    # calculate exceedence probabilities
    risk_prep["structural_poe"] = calculate_poe(
        saavg_pmf,
        fcm_tables["structural_poe"],
        rate_multiplier,
        weights,
    )
    risk_prep["structural_pod"] = calculate_poe(
        saavg_pmf,
        fcm_tables["structural_pod"],
        rate_multiplier,
        weights,
    )
    risk_prep["chimney_poe"] = calculate_poe(
        pga_pmf,
        fcm_tables["chimney_poe"],
        rate_multiplier,
        weights,
    )
    risk_prep["chimney_pod"] = calculate_poe(
        pga_pmf,
        fcm_tables["chimney_pod"],
        rate_multiplier,
        weights,
    )

    # calculate LPR
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
