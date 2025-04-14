"""
Prepares the exposure grid and exposure database for the hazard and risk
calculations. The exposure grid is a regular grid covering the entire extent
of the zonation.
"""

import sys
import logging
import timeit
import numpy as np
import xarray as xr
import xarray_einstats as xe

from chaintools.chaintools.tools_configuration import preamble
from chaintools.chaintools import tools_xarray as tx
from chaintools.chaintools import tools_geometry as tg
from chaintools.chaintools import tools_grid as gr


def assign_defaults(config):
    config.setdefault("grid_crs", "EPSG:28992")
    config.setdefault("grid_spacing", 1000.0)
    config.setdefault("grid_anchor", [0.0, 0.0])
    config.setdefault("zone_id", "zone")
    config.setdefault("ignore_zones", None)
    config.setdefault("file_mode", "w-")
    config.setdefault("building_id", "bag_building_id")
    return config


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
    # open input data
    exposure_data = tx.open("exposure_input", config)
    zonation_xr = tx.open("zonation", config)

    # prepare exposure grid
    exposure_grid = prepare_exposure_grid(zonation_xr, config)

    # if an exposure database is provided, read and process it
    mode = config["file_mode"]
    if exposure_data is not None:
        # read edb file
        exposure_database = prepare_exposure_data(exposure_data, zonation_xr, config)

        # transfer from edb to grid
        exposure_grid["count"] = grid_count(exposure_database, config)
        if "occupancy" in exposure_database:
            exposure_grid["occupancy"] = grid_occupancy(exposure_database, config)

        # transfer form grid to edb
        exposure_database["x_grd"] = (
            exposure_grid["x"]
            .interp({"x": exposure_database["x"]}, method="nearest")
            .rename({"x": "x_grd"})
        )
        exposure_database["y_grd"] = (
            exposure_grid["y"]
            .interp({"y": exposure_database["y"]}, method="nearest")
            .rename({"y": "y_grd"})
        )
        exposure_database = exposure_database.set_coords(["x_grd", "y_grd"])

        # store
        tx.store(exposure_database, "exposure_database", config, mode=mode)

    # store exposure grid
    tx.store(exposure_grid, "exposure_grid", config, mode=mode)


def prepare_exposure_data(edb_xr, zonation_xr, config):
    # transform to geodataframes
    edb_gdf = tx.to_geopandas(edb_xr, to_crs=config["grid_crs"])
    zonation_gdf = tx.to_geopandas(zonation_xr, to_crs=config["grid_crs"])

    # assign zones to buildings
    edb_gdf = tg.apply_zone_assignment(
        edb_gdf,
        zonation_gdf,
        zone_id=config["zone_id"],
        zone_distance_id="zone_distance",
    )

    # transform back
    edb_xr = xr.Dataset.from_dataframe(edb_gdf)
    edb_xr = edb_xr.set_coords(["x", "y", "community", "zone"])

    # reorganize all database information in xarray
    exposure_database = fill_exposure_database(edb_xr, config["building_id"])
    exposure_database.rio.write_crs(config["grid_crs"], inplace=True)

    return exposure_database


def grid_occupancy(exposure_database, config):
    occupancy_grid = exposure_database.groupby(["zone", "surface_condition"]).map(
        lambda edb_block: gr.aggregate_to_grid(
            samples=edb_block[["x", "y"]].reset_coords(["x", "y"]),
            weights=edb_block["occupancy"]
            * edb_block["population_whereabouts"]
            * edb_block["vc_matrix"],
            marginalize_dims=[config["building_id"]],
            target_step=config["grid_spacing"],
        )
    )

    return occupancy_grid


def grid_count(exposure_database, config):
    if "vc_matrix" in exposure_database:
        count_grid = exposure_database.groupby(["zone", "surface_condition"]).map(
            lambda edb_block: gr.aggregate_to_grid(
                samples=edb_block[["x", "y"]].reset_coords(["x", "y"]),
                weights=edb_block["vc_matrix"],
                marginalize_dims=[config["building_id"]],
                target_step=config["grid_spacing"],
            )
        )
    else:
        count_grid = exposure_database.groupby("zone").map(
            lambda edb_block: gr.aggregate_to_grid(
                samples=edb_block[["x", "y"]].reset_coords(["x", "y"]),
                marginalize_dims=[config["building_id"]],
                target_step=config["grid_spacing"],
            )
        )

    return count_grid


def prepare_exposure_grid(zonation_xr, config):
    grid_spacing = config["grid_spacing"]
    grid_anchor = config["grid_anchor"]

    # transform zonation to geodataframe
    zone_id = config["zone_id"]
    zonation_gdf = tx.to_geopandas(zonation_xr, to_crs=config["grid_crs"])
    zonation_gdf = zonation_gdf.rename_axis(zone_id)

    # first drop zones to be ignored (e.g., lakes)
    if config["ignore_zones"] is not None:
        zonation_gdf = zonation_gdf.drop(labels=config["ignore_zones"], axis=0)

    # prepare grid covering entire zonation extent
    surface_grid = tg.define_grid_spanning_zonation(
        grid_spacing, grid_anchor, zonation_gdf
    )

    # put zonation on xarray dimension
    zone_geometry = xr.DataArray.from_series(zonation_gdf["geometry"])

    # for each node in the grid, construct a (square) buffer with twice the
    # spacing as radius, so that we know that if this node has any overlap with
    # a zone polygon, this node may be required for bilinear interpolation within
    # that zone
    overlap = tg.xr_cell_polygon_overlap_fraction(
        surface_grid,
        zone_geometry,
        grid_spacing,  # half size of the square
        cap_style="square",  # square buffer
    )
    exposure_grid = xr.Dataset(
        {
            "contributing_to_zone": (overlap > 0.0),
            "contributing": (overlap > 0.0).any(zone_id),
        }
    )
    exposure_grid.rio.write_crs(zonation_gdf.crs, inplace=True)

    return exposure_grid


def fill_exposure_database(edb_xr, building_id):
    list_of_vars = [
        v for v in ["x", "y", "community", "zone", "zone_distance"] if v in edb_xr
    ]
    output_edb = edb_xr[list_of_vars]

    # usage data
    use_selection = [v for v in edb_xr if "use_" in v]
    if len(use_selection) > 0:
        uses = np.unique(edb_xr[use_selection].fillna("").to_array().data)
        use = xr.Dataset().expand_dims(use_function=uses[1:])
        use_matrix = xe.zeros_ref(
            edb_xr, use, dims=[building_id, "use_function"], dtype=int
        )
        for i, col in enumerate(use_selection):
            subset = edb_xr[col].dropna(building_id)
            use_matrix.loc[
                {
                    "use_function": subset,
                    building_id: subset[building_id],
                }
            ] = (
                i + 1
            )
        output_edb["use"] = use_matrix

    # vulnerability data
    system_selection = [v for v in edb_xr if "system_" in v]
    if len(system_selection) > 0:
        codes = np.unique(edb_xr[system_selection].fillna("").to_array().data)
        vc = xr.Dataset().expand_dims(vulnerability_class=codes[1:])
        vc_matrix = xe.zeros_ref(edb_xr, vc, dims=[building_id, "vulnerability_class"])
        for i in range(1, 11):
            syst = f"system_{i}"
            prob = f"s_probability_{i}"
            subset = edb_xr[[syst, prob]].dropna(building_id)
            vc_matrix.loc[
                {
                    "vulnerability_class": subset[syst],
                    building_id: subset[building_id],
                }
            ] = subset[prob]
        output_edb["vc_matrix"] = vc_matrix

    # occupancy data
    occupancy_selection = [v for v in edb_xr if "sum_pop_" in v]
    if len(occupancy_selection) > 0:
        output_edb["occupancy"] = (
            xr.concat(
                [
                    xr.concat(
                        [
                            edb_xr["sum_pop_in_day"],
                            edb_xr["sum_pop_pas_day"],
                            edb_xr["sum_pop_runners_out_day"],
                        ],
                        dim="population",
                    ),
                    xr.concat(
                        [
                            edb_xr["sum_pop_in_night"],
                            edb_xr["sum_pop_pas_night"],
                            edb_xr["sum_pop_runners_out_night"],
                        ],
                        dim="population",
                    ),
                ],
                dim="time_of_day",
            )
            .rename("occupancy")
            .fillna(0.0)
            .assign_coords(
                population=["inside", "passing", "runners_out"],
                time_of_day=["day", "night"],
            )
        )

        output_edb["population_whereabouts"] = xr.DataArray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
            ],
            dims=["location", "population"],
        ).assign_coords(location=["inside", "outside"])

        output_edb["uniform_whereabouts"] = xr.DataArray(
            [0.99, 0.01],
            dims="location",
        ).assign_coords(location=["inside", "outside"])

    wierde_flag = None
    if "wiede_flag" in edb_xr:
        wierde_flag = "wiede_flag"
    elif "wierde_flag" in edb_xr:
        wierde_flag = "wierde_flag"

    if wierde_flag is not None:
        output_edb["surface_condition"] = xr.where(
            edb_xr[wierde_flag], "wierde", "regular"
        )

    return output_edb


if __name__ == "__main__":
    main(sys.argv)
