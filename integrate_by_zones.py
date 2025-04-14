"""
Module to integrate pre-calculated conditional (exceedance) event probabilities over conditions.
The conditions are supplied in term of occurrence rates (e.g. seismicity rates).
The result is a set of output (exceedance) event rates.
Also, the conditional probabilities can be seen to provide a set of filters on the input rates.
The output rates are always a fraction of the input rates. The fraction is determined by the
pre-calculated conditional probabilities.
Apart from the input rates and the conditional probabilities, the module also allows the specification
of supplementary weights, or weight/probability distributions that are used to marginalize the
input rates and the conditional probabilities.
"""

import sys
import logging
import timeit
import numpy as np
import xarray as xr
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from chaintools.chaintools.tools_configuration import preamble
from chaintools.chaintools import tools_xarray as tx


def assign_defaults(config):
    config.setdefault("file_mode", "w-")
    config.setdefault("marginalize_dims", ["distance_rupture", "magnitude"])
    config.setdefault("suffix", {"mean": "-[lt-mean]"})


def main(args):
    config = preamble(args)
    assign_defaults(config)
    logging.info("starting module %s in file %s", __name__, __file__)
    start = timeit.default_timer()

    # open data sources
    rates = tx.open("rates", config)
    conditional_data = tx.open("conditional_probabilities", config)
    weights = tx.open("weights", config)
    exposure_grid = tx.open("exposure_grid", config)
    rate_multiplier = tx.open("rate_multiplier", config)

    # prepare surface grid nodes
    # find relevant combinations of zone and x,y
    zones = conditional_data["zone"].data
    zone_x_y = get_exposure_nodes(exposure_grid, zones)

    # config data
    marginalize_dims = config["marginalize_dims"]
    logging.info("iterating over zones")
    storage_kwargs = {"mode": config["file_mode"]}
    tqdm_node_kwargs = {
        "total": zone_x_y.sizes["zone_x_y"],
        "desc": "total nodes",
        "position": 0,
    }
    with logging_redirect_tqdm(), tqdm(**tqdm_node_kwargs) as pbar_node:
        for zone in tqdm(zones, position=1, desc="zones"):
            # for each zone collect the relevant nodes using the exposure grid
            rates_z, probabilities_z = extract_zone_data(
                rates, conditional_data, zone_x_y, zone
            )
            output_rates_z = compute_output_rates_mean(
                rates_z,
                rate_multiplier,
                weights,
                probabilities_z,
                marginalize_dims,
            )

            # add suffix
            output_rates_z = tx.add_suffix(output_rates_z, config["suffix"]["mean"])

            # store flattenend, using zone, x, y for coordinates
            tx.store(output_rates_z, "output", config, **storage_kwargs)
            storage_kwargs["append_dim"] = "zone_x_y"
            storage_kwargs["mode"] = "a"
            pbar_node.update(len(output_rates_z["zone_x_y"]))

    stop = timeit.default_timer()
    total_time = stop - start
    logging.info(f"total time: {total_time / 60:.2f} mins")

    return


def compute_output_rates_mean(
    rates,
    rate_multiplier,
    weights,
    conditional_data,
    marginalize_dims,
):
    if rate_multiplier is None:
        rate_multiplier = xr.DataArray(1.0)
    if weights is None:
        weights = xr.Dataset()

    # turn input dataarrays into datasets
    if isinstance(conditional_data, xr.DataArray):
        conditional_data = conditional_data._to_dataset_whole()
    if isinstance(rates, xr.DataArray):
        rates = rates._to_dataset_whole()
    if isinstance(weights, xr.DataArray):
        weights = weights._to_dataset_whole()

    # marginalize rates first
    w_rates, w_rates_dims = tx.prepare_weights(weights, rates)
    dot_args = (*rates.values(), *w_rates.values())
    dot_kwargs = {"dim": w_rates_dims, "optimize": True}
    rates_mean = xr.dot(*dot_args, **dot_kwargs)

    # then marginalize conditional probabilities
    w_prob, w_prob_dims = tx.prepare_weights(weights, rate_multiplier, conditional_data)
    dot_args = (rate_multiplier, *w_prob.values())
    dot_kwargs = {"dim": w_prob_dims, "keep_attrs": True, "optimize": True}
    probs_mean = conditional_data.map(xr.dot, args=dot_args, **dot_kwargs)

    # finally marginalize over the specified dimensions (distance, magnitude)
    dot_args = (rates_mean,)
    dot_kwargs = {"dim": marginalize_dims, "keep_attrs": True, "optimize": True}
    output_rates_mean = probs_mean.map(xr.dot, args=dot_args, **dot_kwargs)

    return output_rates_mean


def get_exposure_nodes(exposure_grid, zones=None, availability_id=None):
    # determine relevant zones
    if zones is None:
        zones = exposure_grid["zone"].data
    zones = np.atleast_1d(zones)

    # to know where the exposure grid is available we check one variable
    if availability_id is None:
        availability_id = "contributing_to_zone"  # feels a bit ad-hoc

    egz = (
        exposure_grid[availability_id]
        .sel({"zone": zones})
        .stack(zone_x_y=("zone", "x", "y"))
        .compute()
    )
    egz = egz.where(egz, drop=True)
    zone_x_y = egz["zone_x_y"]

    return zone_x_y


def extract_zone_data(xy_data, zone_data, exposure_grid, zone_id):
    # determine the x,y coordinates contributing to this zone
    zone = exposure_grid["zone"].sel({"zone": zone_id}).data
    x_y = exposure_grid.sel({"zone": zone_id})["zone_x_y"].data

    # select the source distribution data for the nodes contributing to this zone
    # then add the zone identifier to the multiindex
    xy_data_z = xy_data.sel({"x_y": x_y}).reset_index("x_y").rename({"x_y": "zone_x_y"})
    xy_data_z["zone"] = xr.DataArray(zone, dims="zone_x_y")
    xy_data_z = xy_data_z.set_xindex(["zone", "x", "y"])

    # select the zone_data for this zone
    zonedata_z = zone_data.sel({"zone": zone_id}, drop=True)

    return xy_data_z, zonedata_z


if __name__ == "__main__":
    main(sys.argv)
