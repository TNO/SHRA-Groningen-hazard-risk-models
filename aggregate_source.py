import sys
import logging
import timeit
import numpy as np
import xarray as xr
from dask.distributed import progress

from chaintools.chaintools.tools_configuration import preamble
from chaintools.chaintools import tools_xarray as tx


def main(args):
    module_name = "source_integrator"
    config, client = preamble(args, module_name)
    logging.info(f"starting {module_name}")
    assign_defaults(config)
    start = timeit.default_timer()

    # open data sources
    rupture_prep = tx.open("rupture_prep", config, chunking_allowed=False)
    forecast = tx.open("forecast", config, chunking_allowed=False)
    exposure_grid = tx.open("exposure_grid", config, chunking_allowed=False)

    # preprocessing
    rp_mags = rupture_prep["magnitude"]
    rp = rupture_prep["probability_density_azimuth_smoothed"]
    forecast = preprocess_ssm(forecast, rp_mags, config)
    sr = forecast["seismicity_rate"]
    eg = exposure_grid.stack(x_y=("x", "y"))
    eg = eg["contributing"].where(eg["contributing"], drop=True)
    logictree = forecast[[v for v in forecast if "logic_tree" in v]]

    # compute azimuth and distance
    azi, dst = get_azimuth_distance(rupture_prep, sr, eg, config["rupture_azimuth"])

    # chunk
    ds = xr.Dataset({"d": dst, "a": azi, "sr": sr, "rp": rp})
    chunks = {d: config["chunks"].get(d, "auto") for d in ds.dims}
    ds = ds.chunk(chunks).unify_chunks()

    # main calculations
    # interpolate rupture prep to get rupture distance distribution conditional on
    # the specific azimuth and hypocentral distance of subsurface nodes
    rp_int = (
        ds["rp"]
        .interp(azimuth=ds["a"], distance_hypocenter=ds["d"], method="linear")
        .fillna(0.0)
    )

    # inner product with source distribution at subsurface nodes
    sr_mean = xr.dot(ds["sr"], *logictree.values(), optimize=True)
    seismicity_mean = xr.dot(sr_mean, rp_int, dims="loc_s").reset_index("x_y")

    out_ds = xr.Dataset({"seismicity_rate_mean": seismicity_mean})
    out_ds = out_ds.merge(logictree, combine_attrs="no_conflicts")
    out_ds.assign_attrs(**config)

    if config.get("full_logictree", False):
        sr = xr.dot(ds["sr"], rp_int, dims="loc_s").reset_index("x_y")
        out_ds["seismicity_rate"] = sr

    # finally, store the output
    storage_task = tx.store(
        out_ds, "source_distribution", config, mode="w-", compute=False
    )

    # launch and monitor
    job = client.compute(storage_task)
    progress(job)

    stop = timeit.default_timer()
    total_time = stop - start
    logging.info(f"total time: {total_time / 60:.2f} mins")

    return


def get_azimuth_distance(rupture_prep, fc, eg, azimuth):
    dx = (fc["x"] - eg["x"]) / 1000.0
    dy = (fc["y"] - eg["y"]) / 1000.0
    dz = rupture_prep.rupture_depth
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


def assign_defaults(config):
    config["rupture_azimuth"] = config.get("rupture_azimuth", -30.0)
    config["chunks"] = config.get("chunks", {}) | {
        "loc_s": -1,
        "x_y": 1,
        "magnitude": 10,
        "distance_hypocenter": -1,
        "distance_rupture": -1,
        "azimuth": -1,
    }
    config["source_spatial_dimensions"] = config.get(
        "source_spatial_dimensions", ["x", "y"]
    )
    config["source_spatial_coordinates"] = config.get(
        "source_spatial_coordinates", ["x", "y"]
    )
    config["full_logictree"] = config.get("full_logictree", False)
    return


def preprocess_ssm(seismicity, target_mags, config):
    # NOTE that this is quite ad-hoc, and should be more smoothly integrated
    sdim = config["source_spatial_dimensions"]

    seismicity = seismicity.rename({"mmax": "branch_mmax", "magnitude": "m_tmp"})

    dm = target_mags[1] - target_mags[0]
    count_lower = seismicity.interp(
        m_tmp=target_mags - 0.5 * dm, method="linear"
    ).fillna(0.0)
    count_upper = seismicity.interp(
        m_tmp=target_mags + 0.5 * dm, method="linear"
    ).fillna(0.0)
    seismicity_pmf = count_lower - count_upper

    seismicity_ds = xr.Dataset({"seismicity_rate": seismicity_pmf})
    seismicity_ds["logic_tree:branch_mmax"] = xr.DataArray(
        [0.27, 0.405, 0.1875, 0.1075, 0.025, 0.005], dims="branch_mmax"
    )

    if isinstance(sdim, str):
        seismicity_ds = seismicity_ds.rename({sdim: "loc_s"})
    else:
        seismicity_ds = seismicity_ds.stack(loc_s=sdim).reset_index("loc_s")

    return seismicity_ds


if __name__ == "__main__":
    main(sys.argv)
