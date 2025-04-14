"""
Module to calculate risk due to a source distribution, in terms of exceedance
probabilities of damage and collapse states, as well as probabilities of loss
of life.
"""

import sys
import logging
import timeit
import numpy as np
import xarray as xr
from tqdm import tqdm

from chaintools.chaintools.tools_configuration import preamble
from chaintools.chaintools import tools_xarray as tx


def assign_defaults(config):
    config.setdefault("file_mode", "w-")
    config.setdefault("building_dim", "bag_building_id")
    config.setdefault("suffix", "-[vc-mean]")
    config.setdefault("interpolation_method", "linear")


def main(args):
    config = preamble(args)
    logging.info("starting module %s in file %s", __name__, __file__)
    assign_defaults(config)
    start = timeit.default_timer()

    run_core(config, start)

    return


def run_core(config, start):
    # open data sources
    zone_data = tx.open("zone_data", config)
    exposure_database = tx.open("exposure_database", config)

    # determine common zones and disparities
    edb_zones = np.unique(exposure_database["zone"])
    missing_zones = np.setdiff1d(edb_zones, zone_data["zone"])
    if missing_zones.size > 0:
        logging.info(
            f"missing zones in zone_data: {missing_zones}; skipping these zones"
        )
    common_zones = np.intersect1d(zone_data.zone, edb_zones)

    # select exposure data for common zones
    bdim = config["building_dim"]
    edb = exposure_database.sel(
        {bdim: np.isin(exposure_database["zone"], common_zones)}
    )

    # group, iterate, interpolate and store
    logging.info("iterate over zones")
    storage_kwargs = {"mode": config["file_mode"]}
    with tqdm(total=len(edb[bdim]), desc="buildings", position=0) as pbar:
        for z, dbz in tqdm(edb.groupby("zone"), desc="zones", position=1):
            # select zone, unstack, interpolate location, update bookkeeping
            zd = zone_data.sel(zone=z).unstack("zone_x_y")
            values = zd.interp(
                x=dbz["x"], y=dbz["y"], method=config["interpolation_method"]
            )
            values = values.assign_coords(
                {"zone": (bdim, np.full_like(values[bdim].data, z))}
            )

            # select surface condition based on database
            if "surface_condition" in values and "surface_condition" in dbz:
                sc_values = (
                    values.sel({"surface_condition": dbz["surface_condition"]})
                    .drop_vars("surface_condition")
                    .expand_dims({"surface_condition": ["sc_flag"]})
                )
                values = xr.concat([sc_values, values], "surface_condition")
                values["sc_flag"] = dbz["surface_condition"]
                values = values.set_coords("sc_flag")

            # calculate mean over vulnerability class
            if "vulnerability_class" in values and "vc_matrix" in dbz:
                values_w = values.weighted(dbz["vc_matrix"])
                values_mean = values_w.mean(dim="vulnerability_class")
                values = values.merge(tx.add_suffix(values_mean, config["suffix"]))

            # store, prepare next iteration, update progress bar
            tx.store(values, "output", config, **storage_kwargs)
            storage_kwargs["append_dim"] = bdim
            storage_kwargs["mode"] = "a"
            pbar.update(len(dbz[bdim]))

    stop = timeit.default_timer()
    total_time = stop - start
    logging.info(f"total time: {total_time / 60:.2f} mins")


if __name__ == "__main__":
    main(sys.argv)
