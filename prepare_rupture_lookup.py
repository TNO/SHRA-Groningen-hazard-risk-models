"""
Generate rupture_prep: distribution of rupture distances as a function of epicentral distance, magnitude, and
azimuth.
"""

import sys
import logging
import timeit
import numpy as np
import xarray as xr
import dask.array as da
from flox.xarray import xarray_reduce
from scipy.ndimage import gaussian_filter1d

from hr_models import rupture as rup
from chaintools.chaintools.tools_configuration import preamble
from chaintools.chaintools.tools_xarray import prepare_ds, store


def assign_defaults(config):
    config.setdefault("file_mode", "w-")
    config.setdefault("rng_seed", 42)
    config.setdefault("azimuth_sd", 30.0)
    config.setdefault("rupture_length_logsd", 0.190)
    config.setdefault("rupture_depth", 3.0)
    config.setdefault("n_sample", 1_000_000)
    config.setdefault("n_workers", 8)
    config.setdefault(
        "independent_dimensions", ["magnitude", "distance_hypocenter", "azimuth"]
    )
    config.setdefault("chunk", {})
    config["rupture_model"] = rup.default_parameters | config.get("rupture_model", {})
    config["dimensions"]["azimuth"]["interval"] = [0.0, 90.0]

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
    # prepare data dimensions and prepare random samples
    ds = prepare_ds(config)
    ds = extend_ds(ds, config)
    epsilon = get_random_samples(ds, config)

    # compute rupture distance
    logging.info("computing rupture distance distribution")
    rupture_distance = get_rupture_distance(ds, epsilon, config)
    rupture_distance_distribution = get_rupture_distance_distribution(
        ds, rupture_distance
    ).compute()

    logging.info("marginalize azimuth distribution")
    smoothed_rupture_distance_distribution = smooth_azimuth(
        rupture_distance_distribution, config
    )

    # store
    ds = ds.merge(
        {
            "probability_density": rupture_distance_distribution,
            "probability_density_azimuth_smoothed": smoothed_rupture_distance_distribution,
        }
    )
    store(ds, "output", config, mode=config["file_mode"])


def extend_ds(ds, config):
    ds = ds.assign_coords({"rupture_depth": config["rupture_depth"]})

    distance_epicenter = np.clip(
        np.sqrt(ds["distance_hypocenter"] ** 2 - ds["rupture_depth"] ** 2), 0.0, None
    ).data

    ds = ds.assign_coords(
        {"distance_epicenter": ("distance_hypocenter", distance_epicenter)}
    )

    return ds


def get_rupture_distance_distribution(ds, distance_rupture):
    spacing = ds["distance_rupture"].attrs.get("sequence_spacing", "linear")
    if spacing in ["log", "exp", "geom"]:
        nodes = np.log(ds["distance_rupture"])
        values = np.log(distance_rupture)
    else:
        nodes = ds["distance_rupture"]
        values = distance_rupture

    start = nodes.isel(distance_rupture=0, drop=True)
    step = nodes.isel(distance_rupture=1, drop=True) - start
    label_range = range(len(nodes))

    index, remainder = np.divmod(values - start, step)
    i0 = index.astype(int).rename("distance_rupture")
    i1 = i0 + 1
    w1 = (remainder / step).rename("probability_density")
    w0 = 1.0 - w1
    w = xr.concat([w0, w1], dim="__span__") / values.sizes["__sample__"]
    i = xr.concat([i0, i1], dim="__span__")

    rupture_distribution = (
        xarray_reduce(
            w,
            i,
            func="sum",
            dim=["__span__", "__sample__"],
            expected_groups=label_range,
            engine="flox",
            method="map-reduce",
        )
        .fillna(0.0)
        .assign_coords(ds.coords)
    )

    return rupture_distribution


def smooth_azimuth(distribution, config):
    sigma = config["azimuth_sd"] * distribution.sizes["azimuth"] / 90.0
    azimuth_smoothed_distribution = xr.apply_ufunc(
        gaussian_filter1d,
        distribution,
        sigma,
        input_core_dims=[["azimuth"], []],
        output_core_dims=[["azimuth"]],
        kwargs={"mode": "mirror", "axis": -1},
        keep_attrs=True,
    )
    return azimuth_smoothed_distribution


def get_rupture_distance(ds, epsilon, config):
    rupture_length = xr.apply_ufunc(
        lambda m: rup.rupture_length(m, config["rupture_model"]), ds["magnitude"]
    )
    mean_log_length = np.log(rupture_length)
    sigma_log_length = np.log(10) * config["rupture_length_logsd"]
    rupture_length = np.exp(
        (mean_log_length + sigma_log_length * epsilon["standard_normal"])
    )
    offset = rupture_length * epsilon["uniform"]

    rupture_distance = xr.apply_ufunc(
        rup.rupture_distance,
        ds["distance_hypocenter"],
        ds["rupture_depth"],
        ds["azimuth"],
        offset,
        dask="allowed",
    )

    return rupture_distance


def get_random_samples(ds, config, sample_dim=None):
    # batch dimensions - all elements of these dimensions
    # recieve their own random sample
    independent_dimensions = config["independent_dimensions"]
    coords = ds[independent_dimensions].coords
    sizes = tuple(coords.sizes.values())

    # add sample dimension
    if sample_dim is None:
        sample_dim = "__sample__"
    dimensions = independent_dimensions + [sample_dim]
    n_sample = config["n_sample"]
    sizes = sizes + (n_sample,)

    # determine chunking - straight from config; should be the same as ds
    chunk_spec = config["chunk"]
    chunk_spec[sample_dim] = -1
    chunk_sizes = tuple(chunk_spec.get(dim, "auto") for dim in dimensions)

    rng = da.random.default_rng(config["rng_seed"])
    ds_epsilon = xr.Dataset(
        {
            "uniform": xr.DataArray(
                rng.uniform(size=sizes, chunks=chunk_sizes),
                dims=dimensions,
                coords=coords,
            ),
            "standard_normal": xr.DataArray(
                rng.standard_normal(size=sizes, chunks=chunk_sizes),
                dims=dimensions,
                coords=coords,
            ),
        }
    )

    return ds_epsilon


if __name__ == "__main__":
    main(sys.argv)
