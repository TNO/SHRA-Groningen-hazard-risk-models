"""
Module to calculate hazard curves due to a source distribution.
"""

import sys
import logging
import timeit
import numpy as np
import xarray as xr
from tqdm import tqdm
from xarray_einstats.numba import searchsorted_ufunc

from chaintools.chaintools.tools_configuration import preamble, batched
from chaintools.chaintools import tools_xarray as tx
from chaintools.chaintools import tools_grid as tg


def assign_defaults(config):
    config.setdefault("return_periods", [475.0, 2475.0])
    config.setdefault("file_mode", "a")
    config.setdefault("dim", "gm_surface")
    config.setdefault("coords", "SA_g_surface")
    config.setdefault("suffix", "-[return_periods]")
    config.setdefault("batch_dim", "zone_x_y")
    config.setdefault("batch_size", 10)

    return


def main(args):
    config = preamble(args)
    logging.info("starting module %s in file %s", __name__, __file__)
    assign_defaults(config)
    start = timeit.default_timer()

    run_core(config)

    stop = timeit.default_timer()
    total_time = stop - start
    logging.info(f"total time: {total_time / 60:.2f} mins")

    return


def run_core(config):
    # open data sources
    exceedance_data = tx.open("exceedance_data", config)

    # make sure exceedance_data is a dataset to be able to map
    if isinstance(exceedance_data, xr.DataArray):
        exceedance_data = exceedance_data._to_dataset_whole()

    # config data
    return_periods = tg.make_xarray_based("return_periods", config["return_periods"])
    h_dim = config["dim"]
    h_coords = config["coords"]
    it_dim = config["batch_dim"]
    batch_size = config["batch_size"]

    n = exceedance_data.sizes[it_dim]
    storage_kwargs = {"mode": config["file_mode"]}
    with tqdm(total=n, desc=f"iterator {it_dim}", position=0) as pbar:
        for i_range in batched(range(n), batch_size):
            exc_dat = exceedance_data.isel({it_dim: [*i_range]})
            hazard = exc_dat.map(
                inverse_interpolate_hazard, args=(return_periods, h_dim, h_coords)
            )

            # store results
            hazard = tx.add_suffix(hazard, config["suffix"])
            tx.store(hazard, "output", config, **storage_kwargs)

            # prepare for next iteration
            pbar.update(len(i_range))
            storage_kwargs["append_dim"] = it_dim
            storage_kwargs["mode"] = "a"


def inverse_interpolate_hazard(exceedance_rates, return_periods, h_dim, h_coords):
    x = exceedance_rates
    y = exceedance_rates[h_coords]
    x_lookup = 1 / return_periods

    # find the index of the interval where the return period is located
    index = find_index(x, x_lookup, h_dim).compute()

    # inverse interpolate the hazard by linear interpolation in log space
    hazard = interpolate_in_log(x, y, x_lookup, index, h_dim).compute()

    return hazard


def find_index(x, x_lookup, h_dim):
    # put exceedance periods on the "x" - axis
    # introduce minus sign to force ascending order
    x_reversed = -x.reset_coords(drop=True).fillna(0.0)

    # the location to interpolate return frequencies
    # introduce minus sign to force ascending order
    x_lookup_reversed = -x_lookup

    # search and find location x_index of x in x_range, then lookup corresponding
    # exceedence frequency (x) and spectral acceleration (y)
    # take log for smooth interpolation
    x_index = (
        searchsorted(x_reversed, x_lookup_reversed, h_dim)
        .astype(int)
        .clip(2, x_reversed.sizes[h_dim] - 2)
    )

    return x_index


def interpolate_in_log(x, y, x_lookup, index, h_dim):
    log_x_lookup = np.log(x_lookup)

    with np.errstate(divide="ignore"):
        log_x_up = np.log(x.isel({h_dim: index}))
        log_x_down = np.log(x.isel({h_dim: index - 1}))
        log_y_up = np.log(y.isel({h_dim: index}))
        log_y_down = np.log(y.isel({h_dim: index - 1}))

    frac_up = (log_x_lookup - log_x_down) / (log_x_up - log_x_down)
    frac_down = 1 - frac_up
    log_y_interpolated = frac_up * log_y_up + frac_down * log_y_down
    y_interpolated = np.exp(log_y_interpolated)

    return y_interpolated


def searchsorted(x_range, x, h_dim):
    return xr.apply_ufunc(
        searchsorted_ufunc,
        x_range,
        x,
        input_core_dims=[[h_dim], x.dims],
        output_core_dims=[x.dims],
        dask="allowed",
    )


if __name__ == "__main__":
    main(sys.argv)
