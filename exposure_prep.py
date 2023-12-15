"""	
Prepares the exposure grid and exposure database for the hazard and risk
calculations. The exposure grid is a regular grid covering the entire extent
of the zonation. 
"""
import sys
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray
import xarray_einstats as xe
import timeit

from chaintools.chaintools.tools_configuration import configure
from chaintools.chaintools import tools_xarray as tx
from chaintools.chaintools import tools_geometry as tg
from chaintools.chaintools import tools_grid as gr

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


def main(args):
    module_name = "exposure_prep"
    logging.info(f"starting {module_name}")
    config = configure(args[1:], module_name)
    start = timeit.default_timer()

    # retrieve relevant info from config, or set defaults
    zonation_file = config.get("zonation_file", None)
    edb_file = config.get("edb_file", None)

    # parse zonation file, put in xarray
    exposure_grid, zonation_gdf = prepare_exposure_grid(zonation_file, config)

    # if and edb file is provided, read and process it
    mode = "w-"
    if edb_file is not None:
        # read edb file
        exposure_db = import_exposure_database(edb_file, zonation_gdf, config)

        # transfer from edb to grid
        exposure_grid["occupancy"] = grid_occupancy(exposure_db, config)

        # store
        tx.store(exposure_db, "exposure_database", config, mode=mode)
        mode = "a"

    # store exposure grid
    tx.store(exposure_grid, "exposure_grid", config, mode=mode)

    stop = timeit.default_timer()
    total_time = stop - start
    logging.info(f"total time: {total_time / 60:.2f} mins")

    return


def import_exposure_database(edb_file, zonation_gdf, config):
    grid_crs = config.get("grid_crs", 28992)

    edb_gdf = get_exposure_database_gdf(edb_file, crs=grid_crs)

    # transfer from zonation to edb
    edb_gdf = assign_zones(edb_gdf, zonation_gdf)

    # reorganize in xarray
    exposure_database = edb_to_xr(edb_gdf)

    return exposure_database


def grid_occupancy(exposure_database, config):
    grid_spacing = config.get("grid_spacing", 1000.0)
    occupancy_grid = exposure_database.groupby("zone").map(
        lambda edb_block: gr.samples_to_density_grid(
            samples=edb_block[["x", "y"]],
            weights=edb_block["occupancy"],
            marginalize_dims=["bag_building_id"],
            target_step=grid_spacing,
        )
    )

    return occupancy_grid


def prepare_exposure_grid(zonation_file, config):
    grid_crs = config.get("grid_crs", 28992)
    grid_spacing = config.get("grid_spacing", 1000.0)
    grid_anchor = config.get("grid_anchor", [0.0, 0.0])
    zonation_id = config.get("zone_id", "ID_V6")

    zonation_gdf = get_zonation_gdf(zonation_file, grid_crs, zonation_id)
    zonation_xr = xr.Dataset(zonation_gdf)

    # prepare grid covering entire zonation extent
    surface_grid = define_surface_grid(grid_spacing, grid_anchor, zonation_gdf)

    # for each node in the grid, construct a (square) buffer with twice the
    # spacing as radius, so that we know that if this node has any overlap with
    # a zone polygon, this node may be required for bilinear interpolation within
    # that zone
    overlap = tg.xr_cell_polygon_overlap_fraction(
        surface_grid, zonation_xr["geometry"], grid_spacing
    )
    exposure_grid = xr.Dataset(
        {
            "overlap_fraction": overlap,
            "contributing": (overlap > 0.0).any("zone"),
        }
    )

    return exposure_grid, zonation_gdf


def assign_zones(edb_gdf, zone_gdf):
    # first drop the two zones that are lakes and have no ground motion model
    # TODO: give this a more general treatment
    zone_gdf = zone_gdf.drop(labels=["2813", "3411"], axis=0)
    zone_assignment = gpd.sjoin_nearest(
        edb_gdf[["geometry"]], zone_gdf, how="left", distance_col="zone_distance"
    ).rename(columns={"index_right": "zone"})
    return edb_gdf.join(zone_assignment[["zone", "zone_distance"]])


def edb_to_xr(edb_gdf):
    edb_xr_tmp = xr.Dataset(edb_gdf)
    uses = np.unique(
        edb_xr_tmp[["main_use", "secondary_use"]].fillna("").to_array().data
    )[1:]
    use = xr.Dataset().expand_dims(use_function=uses)
    use_matrix = xe.zeros_ref(
        edb_xr_tmp, use, dims=["bag_building_id", "use_function"], dtype=int
    )
    for i, col in enumerate(["main_use", "secondary_use"]):
        id = "bag_building_id"
        subset = edb_xr_tmp[col].dropna(id)
        use_matrix.loc[
            {
                "use_function": subset,
                "bag_building_id": subset[id],
            }
        ] = (
            i + 1
        )

    codes = np.unique(
        edb_xr_tmp[[f"system_{i+1}" for i in range(10)]].fillna("").to_array().data
    )[1:]
    vc = xr.Dataset().expand_dims(vulnerability_class=codes)
    vc_matrix = xe.zeros_ref(
        edb_xr_tmp, vc, dims=["bag_building_id", "vulnerability_class"]
    )
    for i in range(1, 11):
        syst = f"system_{i}"
        prob = f"s_probability_{i}"
        id = "bag_building_id"
        subset = edb_xr_tmp[[syst, prob]].dropna(id)
        vc_matrix.loc[
            {
                "vulnerability_class": subset[syst],
                "bag_building_id": subset[id],
            }
        ] = subset[prob]

    occupancy = (
        xr.concat(
            [
                xr.concat(
                    [
                        edb_xr_tmp["sum_pop_in_day"],
                        edb_xr_tmp["sum_pop_pas_day"],
                        edb_xr_tmp["sum_pop_runners_out_day"],
                        xr.ones_like(edb_xr_tmp["sum_pop_in_day"]),
                    ],
                    dim="population",
                ),
                xr.concat(
                    [
                        edb_xr_tmp["sum_pop_in_night"],
                        edb_xr_tmp["sum_pop_pas_night"],
                        edb_xr_tmp["sum_pop_runners_out_night"],
                        xr.ones_like(edb_xr_tmp["sum_pop_in_day"]),
                    ],
                    dim="population",
                ),
            ],
            dim="time_of_day",
        )
        .rename("occupancy")
        .fillna(0.0)
        .assign_coords(
            population=["inside", "passing", "runners_out", "uniform"],
            time_of_day=["day", "night"],
        )
    )

    population_whereabouts = xr.DataArray(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]],
        dims=["location", "population"],
    ).assign_coords(location=["inside", "outside"])

    uniform_whereabouts = xr.DataArray([0.99, 0.01], dims="location").assign_coords(
        location=["inside", "outside"]
    )

    surface_condition = xr.where(edb_xr_tmp.wiede_flag, "wierde", "regular")

    output_edb = edb_xr_tmp[["x", "y", "community", "zone", "zone_distance"]]
    output_edb = output_edb.merge(
        xr.Dataset(
            {
                "use": use_matrix,
                "vc_matrix": vc_matrix,
                "occupancy": occupancy,
                "surface_condition": surface_condition,
                "population_whereabouts": population_whereabouts,
                "uniform_whereabouts": uniform_whereabouts,
            }
        )
    ).rio.write_crs(edb_gdf.crs.to_epsg())

    return output_edb


def get_exposure_database_gdf(edb_file, crs=28992):
    edb_file_path = tx.construct_path(edb_file)
    edb = pd.read_csv(
        edb_file_path, dtype={"bag_building_id": str, "wiede_flag": bool}
    ).set_index("bag_building_id")

    db_xy = gpd.points_from_xy(edb["point_x"], edb["point_y"], crs=28992)
    edb_gdf = gpd.GeoDataFrame(edb, geometry=db_xy.to_crs(crs))
    edb_gdf = edb_gdf.join(edb_gdf.get_coordinates())

    return edb_gdf


def get_zonation_gdf(zonation_file, grid_crs, zone_id):
    zonation_file_path = tx.construct_path(zonation_file)
    zonegdf = gpd.read_file(zonation_file_path).to_crs(grid_crs)
    zonegdf["zone"] = zonegdf[zone_id].astype(str)
    zonegdf = zonegdf.set_index("zone")[["geometry"]]
    zonegdf = zonegdf.join(zonegdf.area.rename("area"))
    zonegdf = zonegdf.join(zonegdf.bounds)
    return zonegdf


def define_surface_grid(grid_spacing, grid_anchor, zonegdf):
    anchor = np.asarray(grid_anchor)
    spacing = np.asarray(grid_spacing)

    minx, miny, maxx, maxy = zonegdf.total_bounds
    min = np.array([minx, miny])
    max = np.array([maxx, maxy])
    min = np.floor((min - anchor) / spacing) * spacing + anchor
    max = np.ceil((max - anchor) / spacing) * spacing + anchor

    x_range = np.arange(min[0], max[0] + spacing, spacing)
    y_range = np.arange(min[1], max[1] + spacing, spacing)
    surface_grid = xr.Dataset({"x": x_range, "y": y_range})
    surface_grid = surface_grid.rio.write_crs(zonegdf.crs.to_epsg())

    return surface_grid


if __name__ == "__main__":
    main(sys.argv)
