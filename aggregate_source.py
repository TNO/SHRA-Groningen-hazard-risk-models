import sys
import logging
import timeit
import numpy as np
import xarray as xr
from tqdm import tqdm

from chaintools.chaintools.tools_configuration import preamble, batched
from chaintools.chaintools import tools_xarray as tx


def assign_defaults(config):
    config.setdefault("rupture_azimuth", -30.0)
    config.setdefault("source_spatial_dimensions", ["x", "y"])
    config.setdefault("source_spatial_coordinates", ["x", "y"])
    config.setdefault("file_mode", "w-")
    config.setdefault("output_id", "seismicity_rate")
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
    rupture_prep = tx.open("rupture_prep", config)
    forecast = tx.open("forecast", config)
    exposure_grid = tx.open("exposure_grid", config)
    weights = tx.open("weights", config)

    # preprocessing
    forecast = preprocess_forecast(forecast, rupture_prep["magnitude"], config)

    # preprocess grid - find surface nodes where we need data
    exposure_grid.load()
    eg_x_y = get_exposure_x_y(exposure_grid)

    # preprocess weights -select relevant ones and determine marginalizable dimensions
    weights, marginalize_dims = tx.prepare_weights(weights, forecast, rupture_prep)
    marginalize_dims = marginalize_dims | {"loc_s"}

    # determine relative geometry of surface (eg_x_y) and subsurface (forecast)
    # points
    azi, dst = get_azimuth_distance(
        forecast,
        eg_x_y,
        config["rupture_azimuth"],
        rupture_prep["rupture_depth"],
    )

    # loop over exposure grid points, calculate seismicity rate and store
    storage_kwargs = {"mode": config["file_mode"]}
    n = len(eg_x_y)
    batch_size = config["batch_size"]
    n_batch = n // batch_size + 1
    b_iterator = tqdm(batched(range(n), batch_size), desc="node batches", total=n_batch)
    for i_range in b_iterator:
        seismicity = calculate_radial_seismicity(
            rupture_prep,
            forecast,
            dst.isel({"x_y": [*i_range]}),
            azi.isel({"x_y": [*i_range]}),
            weights,
            marginalize_dims,
        )
        seismicity.name = config["output_id"]

        # store the result
        tx.store(seismicity, "output", config, **storage_kwargs)

        # prepare for next iteration
        storage_kwargs["append_dim"] = "x_y"
        storage_kwargs["mode"] = "a"


def calculate_radial_seismicity(
    rupture_prep, forecast, dst, azi, weights, marginalize_dims
):
    rupture_prep_interpolated = rupture_prep.interp(
        azimuth=azi,
        distance_hypocenter=dst,
        method="linear",
    ).fillna(0.0)

    # inner product with source distribution at subsurface nodes
    radial_seismicity = xr.dot(
        forecast,
        rupture_prep_interpolated,
        *weights.values(),
        dim=marginalize_dims,
        optimize=True,
    )

    return radial_seismicity


def get_exposure_x_y(exposure_grid):
    eg = exposure_grid.stack({"x_y": ("x", "y")})
    eg_x_y = eg["x_y"].where(eg["contributing"], drop=True)

    return eg_x_y


def get_azimuth_distance(fc, eg, azimuth, depth):
    dx = (fc["x"] - eg["x"]) / 1000.0
    dy = (fc["y"] - eg["y"]) / 1000.0
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


def preprocess_forecast(seismicity, target_mags, config):
    # input seismicity are magnitude survival count per spatial unit
    # (total count * survival probability)
    # we need magnitude bin count per spatial unit
    # (total count * magnitude probability mass distribution)
    # therefore: we will interpolate the survival rates at bin edges
    # then subtract to obtain the counts per bin
    dm = target_mags[1] - target_mags[0]
    mmin = seismicity["magnitude"][0]
    m_lower = (target_mags - 0.5 * dm).clip(min=mmin)
    m_upper = (target_mags + 0.5 * dm).clip(min=mmin)
    count_lower = seismicity.interp(magnitude=m_lower, method="linear").fillna(0.0)
    count_upper = seismicity.interp(magnitude=m_upper, method="linear").fillna(0.0)
    seismicity_pmf = count_lower - count_upper

    sdim = config["source_spatial_dimensions"]
    if isinstance(sdim, str):
        seismicity_pmf = seismicity_pmf.rename({sdim: "loc_s"})
    else:
        seismicity_pmf = seismicity_pmf.stack(loc_s=sdim).reset_index("loc_s")

    return seismicity_pmf


if __name__ == "__main__":
    main(sys.argv)
