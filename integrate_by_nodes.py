"""
Module to calculate risk due to a source distribution, in terms of exceedance
probabilities of damage and collapse states, as well as probabilities of loss
of life.
"""

import sys
import logging
import timeit
import xarray as xr
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from chaintools.chaintools.tools_configuration import preamble, batched
from chaintools.chaintools.tools_statistics import xr_weighted_fractiles
from chaintools.chaintools import tools_xarray as tx
from chaintools.chaintools import tools_grid as tg

from integrate_by_zones import (
    get_exposure_nodes,
    extract_zone_data,
    compute_output_rates_mean,
)


def assign_defaults(config):
    config.setdefault("file_mode", "w-")
    config.setdefault("marginalize_dims", ["distance_rupture", "magnitude"])
    config.setdefault("fractile", None)
    config.setdefault("suffix", {"mean": "-[lt-mean]", "fractiles": "-[lt-fractiles]"})
    config.setdefault("batch_size", 10)


def main(args):
    config = preamble(args)
    assign_defaults(config)
    logging.info("starting module %s in file %s", __name__, __file__)
    start = timeit.default_timer()

    run_core(config)

    stop = timeit.default_timer()
    total_time = stop - start
    logging.info("total time: %.2f mins", total_time / 60)

    return


def run_core(config):
    # open data sources
    rates = tx.open("rates", config)
    conditional_data = tx.open("conditional_probabilities", config)
    weights = tx.open("weights", config)
    exposure_grid = tx.open("exposure_grid", config)

    # TODO : incorporate rate_multiplier either through rates or weights
    rate_multiplier = tx.open("rate_multiplier", config)

    # prepare surface grid nodes
    # find relevant combinations of zone and x,y
    zones = conditional_data["zone"].data
    zone_x_y = get_exposure_nodes(exposure_grid, zones)

    # config data
    fractiles = tg.make_xarray_based("fractile", config["fractile"])
    marginalize_dims = config["marginalize_dims"]
    batch_size = config["batch_size"]

    logging.info("iterating over zones and surface node batches")
    storage_kwargs = {"mode": config["file_mode"], "compute": True}
    tqdm_node_kwargs = {
        "total": zone_x_y.sizes["zone_x_y"],
        "desc": "total nodes",
        "position": 0,
    }
    with logging_redirect_tqdm(), tqdm(**tqdm_node_kwargs) as pbar_node:
        for zone in tqdm(zones, desc="total zones", position=1):
            # extract data for the current zone
            rates_z, probabilities_z = extract_zone_data(
                rates, conditional_data, zone_x_y, zone
            )

            # iterate over batches of surface nodes for this zone
            for i_range in create_tqdm_batch(batch_size, rates_z):
                rates_x_y = rates_z.isel({"zone_x_y": [*i_range]})
                output_rates = compute_output_rates_statistics(
                    rates_x_y,
                    rate_multiplier,
                    weights,
                    probabilities_z,
                    marginalize_dims,
                    fractiles,
                    config,
                )
                tx.store(output_rates, "output", config, **storage_kwargs)

                # prepare for next iteration
                pbar_node.update(len(i_range))
                storage_kwargs["append_dim"] = "zone_x_y"
                storage_kwargs["mode"] = "a"


def create_tqdm_batch(batch_size, rates_z):
    n = rates_z.sizes["zone_x_y"]
    n_batch = -(n // -batch_size)  # upside-down floor division
    tqdm_batch = tqdm(
        batched(range(n), batch_size),
        total=n_batch,
        desc=f"- current zone node batches (n={batch_size})",
        position=2,
        leave=False,
    )

    return tqdm_batch


def compute_output_rates_statistics(
    rates,
    rate_multiplier,
    weights,
    conditional_data,
    marginalize_dims,
    fractiles,
    config,
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

    if fractiles is None:
        # if there is no need for fractiles we can be a lot more efficient
        output_rates_mean = compute_output_rates_mean(
            rates,
            rate_multiplier,
            weights,
            conditional_data,
            marginalize_dims,
        )
        output_rates_statistics = tx.add_suffix(
            output_rates_mean, config["suffix"]["mean"]
        )
    else:
        # marginalization over the dimensions in marginalize_dims
        dot_args = (*rates.values(), rate_multiplier)
        dot_kwargs = {"dim": marginalize_dims, "keep_attrs": True, "optimize": True}
        output_rates = conditional_data.map(xr.dot, args=dot_args, **dot_kwargs)

        # calculate summary statistics: 2 steps
        w_full, w_dims = tx.prepare_weights(weights, output_rates)

        # step 1: calculate weighted mean
        dot_args = (*w_full.values(),)
        dot_kwargs = {"dim": w_dims, "keep_attrs": True, "optimize": True}
        output_rates_mean = output_rates.map(xr.dot, args=dot_args, **dot_kwargs)
        output_rates_statistics = tx.add_suffix(
            output_rates_mean, config["suffix"]["mean"]
        )

        # step 2: calculate weighted fractiles
        output_rates_fractiles = compute_weighted_fractiles(
            output_rates, w_full, w_dims, fractiles
        )
        output_rates_statistics = output_rates_statistics.merge(
            tx.add_suffix(output_rates_fractiles, config["suffix"]["fractiles"])
        )

    return output_rates_statistics


def compute_weighted_fractiles(output_rates, weights, w_dims, fractiles):
    # construct full weight tensor
    w_full = xr.dot(*weights.values())

    # flatten by stacking
    output_rates_flat = output_rates.stack({"__lt__": w_dims})
    w_full_flat = w_full.stack({"__lt__": w_dims})

    # make sure they are properly aligned
    output_rates_flat, w_full_flat = xr.align(
        output_rates_flat, w_full_flat, join="exact"
    )

    # compute fractiles
    output_rates_fractiles = output_rates_flat.map(
        xr_weighted_fractiles,
        args=(w_full_flat, fractiles, "__lt__"),
        keep_attrs=True,
    )

    return output_rates_fractiles


if __name__ == "__main__":
    main(sys.argv)
