import sys
import logging
import timeit
import numpy as np
import xarray as xr
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from chaintools.chaintools.tools_configuration import preamble
from chaintools.chaintools import tools_xarray as tx
from chaintools.chaintools import tools_grid as tg


def assign_defaults(config):
    config.setdefault("rupture_azimuth", -30.0)
    config.setdefault("output_name", "fast_risk")
    config.setdefault("source_spatial_coordinates", ["x", "y"])
    config.setdefault("exposure_spatial_coordinates", ["x", "y"])
    config.setdefault("file_mode", "w-")
    config.setdefault("store_per_zone", False)
    config.setdefault("per_zone_suffix", "_per_zone")

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
    exposure_grid = tx.open("exposure_grid", config)
    weights = tx.open("weights", config)
    conditional_data = tx.open("conditional_data", config)

    logging.info("preprocessing source and exposure grids")

    # preprocess source - extract and flatten the grid
    source_grid = preprocess_source_grid(source_grid, config)

    # preprocess exposure -
    exposure_grid = preprocess_exposure_data(exposure_grid, config)

    # preprocess weights -select relevant ones and determine marginalizable dimensions
    weights, w_dims = tx.prepare_weights(weights, conditional_data)

    # prepare azimuth and distance for all combination of source and exposure grid nodes
    logging.info("prepare relative azimuth and distance")
    azi_dst = prepare_azimuth_distance(exposure_grid, source_grid, rupture_prep, config)

    # bookkeeping
    storage_kwargs = {"mode": config["file_mode"]}

    # determine dimensions to sum/marginalize over
    sum_dims = set(w_dims) | set(exposure_grid.dims) | set(rupture_prep.dims)
    sum_dims = sum_dims - {"magnitude", "zone", "_loc_e_"}

    # loop over zones with log redirection
    total = 0.0
    logging.info("loop over zones")
    with logging_redirect_tqdm():
        for zone in tqdm(conditional_data["zone"].data, desc="zone"):
            # select zone-specific exposure grid nodes
            if not zone in exposure_grid["zone"]:
                continue
            exposure_grid_z = exposure_grid.sel({"zone": zone}, drop=True)  # .load()

            # select corresponding azimuth and distance
            azi_dst_z_smp = azi_dst.sel({"zone": zone})

            # map to azimuth / hypocentral distance grid relavant for the rupture
            exposure_z = calculate_azimuth_distance_grid(
                azi_dst_z_smp, exposure_grid_z, rupture_prep
            )

            # select conditional data for the zone
            cd_z = conditional_data.sel({"zone": zone}, drop=True)

            # calculate inner product
            # - exposure mapped on azimuth / hypocentral distance coordinates
            # - rupture_prep: maps azimuths and distances to rupture distance distribution
            # - conditional data: conditional on rupture distance
            # - logic tree weights (optional)
            dot_args = [exposure_z, rupture_prep, cd_z, *weights.values()]
            result_per_zone = xr.dot(*dot_args, dim=sum_dims, optimize=True)
            result_per_zone = result_per_zone.compute()

            # store result
            if config["store_per_zone"]:
                store_result = result_per_zone.expand_dims(dim={"zone": [zone]})
                store_result.name = config["output_name"] + config["per_zone_suffix"]
                if "_loc_s_" in store_result.dims:
                    store_result = store_result.unstack("_loc_s_")
                tx.store(store_result, "output", config, **storage_kwargs)

                # prepare for next iteration
                storage_kwargs["append_dim"] = "zone"
                storage_kwargs["mode"] = "a"

            # accumulate total
            total = total + result_per_zone

    # store total result
    total.name = config["output_name"]
    if "_loc_s_" in total.dims:
        total = total.unstack("_loc_s_")
    tx.store(total, "output", config, mode=storage_kwargs["mode"])

    return


def preprocess_exposure_data(exposure_data, config):
    """
    Preprocess the exposure grid to extract the flattened grid
    """
    if exposure_data is None:
        return None
    exposure_data.load()

    x_dim, y_dim = config["exposure_spatial_coordinates"]
    grid_coords = [x_dim, y_dim]
    if "zone" in exposure_data.coords:
        grid_coords = ["zone"] + grid_coords

    grid_dims = set()
    for c in grid_coords:
        grid_dims |= set(exposure_data[c].dims)
    grid_dims = list(grid_dims)
    if len(grid_dims) > 1:
        eg_flat = exposure_data.stack({"_loc_e_": grid_dims})
    else:
        eg_flat = exposure_data
        spatial_index = grid_dims[0]
        eg_flat = eg_flat.rename({spatial_index: "_loc_e_"}).set_xindex(grid_coords)

    sum_dims = set(eg_flat.dims) - {"_loc_e_"}
    active = eg_flat.sum(sum_dims) > 0.0

    eg_flat = (
        eg_flat.where(active, drop=True)
        .fillna(0.0)
        .rename({x_dim: "_x_e_", y_dim: "_y_e_"})
    )

    return eg_flat


def preprocess_source_grid(forecast, config):
    """
    Preprocess the forecast to extract its grid
    """
    x_dim, y_dim = config["source_spatial_coordinates"]

    forecast_grid = xr.zeros_like(forecast[x_dim] + forecast[y_dim])
    if x_dim in forecast_grid.dims and y_dim in forecast_grid.dims:
        forecast_grid = forecast_grid.stack({"_loc_s_": [x_dim, y_dim]})

    return forecast_grid


def calculate_azimuth_distance_grid(azi_dst, eg, rp):
    """
    Map the azimuths and distances of the exposure grid relative
    to the forecast grid onto the rupture_prep grid
    """
    distances = rp["distance_hypocenter"].load()
    azimuths = rp["azimuth"].load()

    lnd = np.log(distances)
    lnd_start = lnd[0]
    lnd_stop = lnd[-1]
    lnd_step = lnd[1] - lnd[0]
    azi_start = azimuths[0]
    azi_stop = azimuths[-1]
    azi_step = azimuths[1] - azimuths[0]

    # define grid
    step = np.array([azi_step, lnd_step])
    start = np.array([azi_start, lnd_start])
    stop = np.array([azi_stop, lnd_stop])

    # aggregate on grid
    azi_dst_grid = tg.aggregate_to_grid(
        azi_dst,
        target_step=step,
        weights=eg,
        target_start=start,
        target_stop=stop,
        marginalize_dims=["_loc_e_"],
    )

    # replace log distances by linear distances
    azi_dst_grid["distance_hypocenter"] = distances

    return azi_dst_grid


def prepare_azimuth_distance(eg, fc, rp, config):
    """
    Prepare the azimuth and distance between the exposure grid and the forecast grid
    """
    x_e, y_e = "_x_e_", "_y_e_"
    x_s, y_s = config["source_spatial_coordinates"]
    rupture_azimuth = tg.make_xarray_based(
        "rupture_azimuth", np.atleast_1d(config["rupture_azimuth"])
    ).squeeze()
    azi, dst = get_azimuth_distance(
        eg[x_e],
        eg[y_e],
        fc[x_s],
        fc[y_s],
        rupture_azimuth,
        rp["rupture_depth"],
    )

    out = xr.Dataset({"azimuth": azi, "distance_hypocenter": np.log(dst)})

    return out


def get_azimuth_distance(x0, y0, x1, y1, azimuth, depth):
    """
    Calculates azimuth and distance between two points in 3D space.
    """
    dx = (x1 - x0) / 1000.0
    dy = (y1 - y0) / 1000.0
    dz = depth
    azi = relative_azimuth(dx, dy, azimuth)
    distance = np.sqrt(dx**2 + dy**2 + dz**2)

    return azi, distance


def relative_azimuth(dx, dy, azimuth):
    """
    Calculates reduced relative angles for a grid of distances. Uses symmetry to map angles to the first quadrant.
    """
    angles_reduced = (azimuth - np.arctan2(dx, dy) * (180.0 / np.pi)) % 180.0
    angles_reduced = xr.where(
        angles_reduced <= 90.0, angles_reduced, 180.0 - angles_reduced
    )
    return angles_reduced


if __name__ == "__main__":
    main(sys.argv)
