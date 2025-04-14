"""
DS1 exposure
"""

import sys
import logging
import timeit
from math import fabs, erf, erfc

import numpy as np
import xarray as xr
from numba import vectorize, float64

from chaintools.chaintools.tools_configuration import preamble
from chaintools.chaintools import tools_xarray as tx
from chaintools.chaintools import tools_grid as tg
import hr_models.gmm_empirical as gmm


def assign_defaults(config):
    config.setdefault("source_depth", 3.0)
    config.setdefault("log_distance_resolution", 0.05)
    config.setdefault("source_spatial_index", "loc_s")
    config.setdefault("source_spatial_coordinates", ["x", "y"])
    config.setdefault("exposure_spatial_index", "loc_e")
    config.setdefault("exposure_spatial_coordinates", ["x", "y"])
    config.setdefault("file_mode", "w-")
    config.setdefault("component", "maxrot")
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
    forecast = tx.open("forecast", config)
    weights = tx.open("weights", config)
    gmm_xr = tx.open("gmm_coefficients", config)
    fcm_xr = tx.open("fcm_coefficients", config)
    exposure_grid = tx.open("exposure_grid", config).load()
    exposure_database = tx.open("exposure_database", config).load()

    # preprocessing - step 1: get the forecast and the exposure database ready
    logging.info("preprocessing forecast")
    forecast = preprocess_forecast(forecast, weights, config).compute()

    # preprocess - step 2: get regular grid points to interpolate the database on
    logging.info("preprocessing exposure grid")
    exposure_grid = preprocess_exposure_grid(exposure_grid, config)

    # calculate - step 1: collect the seismic demand as a function of distance
    logging.info("collecting seismicity on a distance grid")
    xy_to_r_mapper = grid_to_distance_mapper(forecast, exposure_grid, config)
    forecast_sf_r = xr.dot(xy_to_r_mapper, forecast, optimize=True)

    # calculate - step 2: calculate the probability of exceedance on the grid
    logging.info("calculating the rate of exceedance on the distance grid")
    poe = calculate_ds1_poe(
        gmm_xr,
        fcm_xr,
        forecast_sf_r["distance_hypocenter"],
        forecast_sf_r["magnitude"],
        exposure_grid["vs30"],
    ).compute()

    # calculate - step 3: use "summation by parts" to integrate the source
    logging.info("integrating capacity and demand")
    rate_of_exceedance = tg.expectation_by_parts(
        forecast_sf_r, poe, dim="magnitude", average=True
    )

    # calculate - step 4: calculate the exposure according to the database
    logging.info("calculating the exposure according to the database")
    exposure = get_exposure(exposure_database, rate_of_exceedance, config).rename(
        "exposure"
    )
    tx.store(exposure, "output", config, mode=config["file_mode"])

    return


def get_exposure(edb_xr, rate_of_exceedance, config):
    edb_xr = edb_xr.where(
        edb_xr["tno_typology"].isin(rate_of_exceedance["type"]), drop=True
    )
    x_edb, y_edb = config["exposure_spatial_coordinates"]
    int_dict = {
        x_edb: edb_xr[x_edb],
        y_edb: edb_xr[y_edb],
        "vs30": edb_xr["vs30"],
    }
    roe_exposure = (
        rate_of_exceedance.unstack("loc_e")
        .interp(int_dict)
        .sel({"type": edb_xr["tno_typology"]})
    )
    roe_exposure = roe_exposure.assign_coords({"limit_state": "DS1"})
    return roe_exposure


def grid_to_distance_mapper(forecast_grid, exposure_grid, config):
    grid_distances = get_distances(forecast_grid, exposure_grid, config).load()
    mapper = tg.aggregate_to_grid(
        samples=np.log(grid_distances),
        target_step=config["log_distance_resolution"],
    )
    mapper = mapper.assign_coords(
        {"distance_hypocenter": np.exp(mapper["distance_hypocenter"])}
    )

    return mapper


def calculate_ds1_poe(gmm_xr, fcm_xr, distance_hypocenter, magnitude, vs30):
    # ground motion -> demand
    lnpgv = xr.apply_ufunc(
        gmm.median,
        distance_hypocenter,
        magnitude,
        gmm_xr,
        kwargs={"par_id": gmm_xr["coefficient"].data},
        input_core_dims=[[], [], ["coefficient"]],
        exclude_dims=set(("coefficient",)),
        dask="allowed",
    )
    lnAF = xr.apply_ufunc(
        gmm.amplification,
        vs30,
        gmm_xr,
        kwargs={"par_id": gmm_xr["coefficient"].data},
        input_core_dims=[[], ["coefficient"]],
        exclude_dims=set(("coefficient",)),
        dask="allowed",
    )
    sigma = gmm_xr.sel({"coefficient": "sigma"}, drop=True)

    # capacity is defined in mm/s: add log(10) to demand to convert cm/s --> mm/s
    demand = lnpgv + lnAF + np.log(10)

    # fragility -> capacity
    theta = fcm_xr.sel({"coefficient": "theta"}, drop=True)
    beta = fcm_xr.sel({"coefficient": "beta"}, drop=True)
    capacity = np.log(theta)

    # combined variability
    st_dev = (beta**2 + sigma**2) ** 0.5
    epsilon = (demand - capacity) / st_dev

    # calculate exceedance probability
    poe = xr.apply_ufunc(cdf_numba, epsilon, dask="allowed")

    return poe


def get_distances(source_grid, exposure_grid, config):
    x_e, y_e = config["exposure_spatial_coordinates"]
    x_s, y_s = "_x_s_", "_y_s_"
    grid_distances = hypocentral_distance(
        source_grid[x_s],
        source_grid[y_s],
        exposure_grid[x_e],
        exposure_grid[y_e],
        config["source_depth"],
    ).rename("distance_hypocenter")

    return grid_distances


def preprocess_exposure_grid(exposure_grid, config):
    stack_var = config["exposure_spatial_index"]
    active = exposure_grid.sum(["tno_typology", "vs30"]) > 0
    x_e, y_e = config["exposure_spatial_coordinates"]
    exposure_grid = (
        exposure_grid.where(active).stack({stack_var: [x_e, y_e]}).dropna(stack_var)
    )

    return exposure_grid


def preprocess_database(edb, fcm_xr, vs30, config):
    # select types of fragility model
    edb = edb.sel(
        {config["building_dim"]: np.isin(edb["tno_typology"], fcm_xr["type"])}
    )
    edb["vs30"] = vs30.sel({"postcode": edb["postcode"]}).drop_vars("postcode")

    return edb


def hypocentral_distance(fc_x, fc_y, db_x, db_y, depth):
    dx = (fc_x - db_x) / 1000.0
    dy = (fc_y - db_y) / 1000.0
    dz = depth
    d = np.sqrt(dx**2 + dy**2 + dz**2)

    return d


def preprocess_forecast(forecast, weights, config):
    # marginalize the conditions for which weights have been supplied
    weights, marginalize_dims = tx.prepare_weights(weights, forecast)
    forecast_mean = xr.dot(
        forecast.fillna(0),
        *weights.values(),
        dim=marginalize_dims,
        optimize=True,
    )

    # for efficiency stack the forecast on the spatial dimensions
    stack_var = config["source_spatial_index"]
    x_s, y_s = config["source_spatial_coordinates"]
    forecast_mean = (
        forecast_mean.stack({stack_var: [x_s, y_s]})
        .dropna(stack_var)
        .rename({x_s: "_x_s_", y_s: "_y_s_"})
    )

    return forecast_mean


@vectorize([float64(float64)])
def cdf_numba(a):
    NPY_SQRT1_2 = 1.0 / np.sqrt(2)
    x = a * NPY_SQRT1_2
    z = fabs(x)

    if z < NPY_SQRT1_2:
        y = 0.5 + 0.5 * erf(x)
    else:
        y = 0.5 * erfc(z)
        if x > 0:
            y = 1.0 - y

    return y


if __name__ == "__main__":
    main(sys.argv)
