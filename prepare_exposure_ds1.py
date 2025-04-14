"""
Prepares the exposure grid and exposure database for the DS1 risk calculation.
"""

import sys
import logging
import timeit

from chaintools.chaintools.tools_configuration import preamble
from chaintools.chaintools import tools_xarray as tx
from chaintools.chaintools import tools_grid as tg


def assign_defaults(config):
    config.setdefault("exposure_spatial_coordinates", ["x", "y"])
    config.setdefault("file_mode", "w-")
    config.setdefault("grid_spacing", 1000.0)
    config.setdefault("vs30_resolution", 10.0)
    config.setdefault("building_dim", "bag_building_id")
    return config


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
    vs30 = tx.open("vs30", config)
    edb_xr = tx.open("exposure_input", config)

    # step 1: get the forecast and the exposure database ready
    logging.info("preprocessing database")
    edb_xr = preprocess_database(edb_xr, vs30, config)
    tx.store(edb_xr, "exposure_database", config, mode=config["file_mode"])

    # step 2: aggregate database on regular grid
    logging.info("aggregating database to grid")
    exposure_grid = get_exposure_grid(edb_xr, config)
    exposure_grid.name = "count"

    tx.store(exposure_grid, "exposure_grid", config, mode=config["file_mode"])

    return


def get_exposure_grid(edb, config):
    """Aggregate the exposure database on a regular grid"""
    target = config["exposure_spatial_coordinates"] + ["vs30"]
    target_step = [config["grid_spacing"]] * 2 + [config["vs30_resolution"]]

    bld_grid = (
        edb.groupby("tno_typology")
        .map(
            lambda edb_block: tg.aggregate_to_grid(
                samples=edb_block,
                target=target,
                target_step=target_step,
                marginalize_dims=config["building_dim"],
                order=1,
            )
        )
        .fillna(0)
    )

    return bld_grid


def preprocess_database(edb, vs30, config):
    """Preprocess the exposure database by determining the vs30 per building"""
    # select types of fragility model
    edb = edb.dropna(config["building_dim"])
    edb["vs30"] = vs30.sel({"postcode": edb["postcode"]}).drop_vars("postcode")

    return edb


if __name__ == "__main__":
    main(sys.argv)
