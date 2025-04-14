"""
Module to take input, write output.
It allows to, e.g. change file type, make selections etc.
Also, a weighted average can be calculated.
"""

import logging
import timeit
import sys
import xarray as xr

from chaintools.chaintools.tools_configuration import preamble
from chaintools.chaintools import tools_xarray as tx


def assign_defaults(config):
    config.setdefault("file_mode", "w-")


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
    # open data source
    data_in = tx.open("input", config)
    rate_multiplier = tx.open(
        "rate_multiplier", config, default=xr.DataArray(1.0)
    )  # optional
    weights = tx.open("weights", config)  # optional

    # preprocess weights -select relevant ones and determine marginalizable dimensions
    weights, summation_dims = tx.prepare_weights(weights, data_in)

    # weights can be used to do a weighted sum over some dimensions
    if len(summation_dims) > 0:
        if isinstance(data_in, xr.Dataset):
            dot_args = (*weights.values(), rate_multiplier)
            dot_kwargs = {"dim": summation_dims, "keep_attrs": True, "optimize": True}
            data_out = data_in.map(xr.dot, args=dot_args, **dot_kwargs)
        else:
            data_out = xr.dot(
                data_in, *weights.values(), rate_multiplier, dim=summation_dims
            )
            data_out.name = data_in.name
    else:
        data_out = data_in

    # export data
    tx.store(data_out, "output", config, mode=config["file_mode"])


if __name__ == "__main__":
    main(sys.argv)
