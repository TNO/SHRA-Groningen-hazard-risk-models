import sys
import logging
import timeit
import numpy as np
import xarray as xr

from chaintools.chaintools.tools_configuration import preamble
from chaintools.chaintools import tools_xarray as tx
from chaintools.chaintools import tools_grid as tg

from integrate_ds1 import calculate_ds1_poe
from aggregate_exposure import (
    preprocess_exposure_data,
    preprocess_source_grid,
    prepare_azimuth_distance,
)


def assign_defaults(config):
    config.setdefault("rupture_azimuth", -30.0)
    config.setdefault("output_name", "fast_risk")
    config.setdefault("gmm_component", "maxrot")
    config.setdefault("source_spatial_coordinates", ["x", "y"])
    config.setdefault("exposure_spatial_coordinates", ["x", "y"])
    config.setdefault("file_mode", "w-")

    return


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
    rupture_prep = tx.open("rupture_prep", config)
    source_grid = tx.open("source_grid", config)
    exposure_grid_ds1 = tx.open("exposure_grid", config)
    gmm_xr = tx.open("gmm_coefficients", config)
    fcm_xr = tx.open("fcm_coefficients", config)

    # preprocess source - extract and flatten the grid
    source_grid = preprocess_source_grid(source_grid, config)

    # load data
    rupture_prep.load()

    exposure_grid_ds1 = preprocess_exposure_data(exposure_grid_ds1, config)

    # prepare azimuth and distance for all combination of source and exposure grid nodes
    distances = prepare_azimuth_distance(
        exposure_grid_ds1, source_grid, rupture_prep, config
    )["distance_hypocenter"]

    distance_grid = calculate_distance_grid(distances, exposure_grid_ds1, rupture_prep)

    poe = calculate_ds1_poe(
        gmm_xr,
        fcm_xr,
        distance_grid["distance_hypocenter"],
        rupture_prep["magnitude"],
        exposure_grid_ds1["vs30"],
    ).sel({"component": config["gmm_component"]})

    distance_grid = distance_grid.sel({"tno_typology": poe["type"]})

    result = xr.dot(
        distance_grid,
        poe,
        dim=["vs30", "type", "distance_hypocenter"],
        optimize=True,
    ).expand_dims(dim={"limit_state": ["DS1"]})

    result.name = config["output_name"]
    if "_loc_s_" in result.dims:
        result = result.unstack("_loc_s_")
    tx.store(result, "output", config, mode=config["file_mode"])

    return


def calculate_distance_grid(dst, eg, rp):
    """
    Map the distances of the exposure grid relative
    to the forecast grid onto the rupture_prep grid
    """
    distances = rp["distance_hypocenter"].load()

    lnd = np.log(distances)
    lnd_start = lnd[0]
    lnd_stop = lnd[-1]
    lnd_step = lnd[1] - lnd[0]

    # aggregate on grid
    dst_grid = tg.aggregate_to_grid(
        dst,
        target_step=lnd_step,
        weights=eg,
        target_start=lnd_start,
        target_stop=lnd_stop,
        marginalize_dims=["_loc_e_"],
    )

    # replace log distances by linear distances
    dst_grid["distance_hypocenter"] = distances

    return dst_grid


if __name__ == "__main__":
    main(sys.argv)
