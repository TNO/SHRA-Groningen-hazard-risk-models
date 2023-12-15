"""	
Module to calculate hazard curves due to a source distribution.
"""
import sys
import logging
import timeit
import numpy as np
import xarray as xr
from xarray_einstats.numba import searchsorted_ufunc
from tqdm import tqdm

from chaintools.chaintools.tools_configuration import preamble
from chaintools.chaintools import tools_xarray as tx


def main(args):
    module_name = "hazard_integrator"
    config, _ = preamble(args, module_name)
    logging.info(f"starting {module_name}")
    start = timeit.default_timer()

    # open data sources
    src_dist = tx.open("source_distribution", config)
    hazard_prep = tx.open("hazard_prep", config)
    exposure_grid = tx.open("exposure_grid", config, chunking_allowed=False)

    # preprocessing -  repair multi-index that cannot be serialized on disk
    src_dist = src_dist.set_xindex(["x", "y"])

    # shorthands
    # sd organized by x,y; hp organized by zone
    # the mean source distribution is forced into memory
    sd_mean = src_dist["seismicity_rate_mean"].persist()
    hp_mean = hazard_prep["surface_poe_mean"]

    # prepare surface grid nodes
    # find relevant combinations of zone and x,y
    relevant_zones = hazard_prep["zone"].data
    egz = get_exposure_nodes(exposure_grid, relevant_zones)

    # get return periods from config
    return_periods = config.get("return_periods", [475.0, 2475.0])

    storage_kwargs = {"mode": "w-", "compute": True}
    for zone in tqdm(relevant_zones):
        # for each zone collect the relevant nodes using the exposure grid
        sd_mean_z, hp_mean_z = extract_zone_data(sd_mean, hp_mean, egz, zone)

        # calculate hazard curves per zone
        hazard_curves_mean_z = xr.dot(
            sd_mean_z, hp_mean_z, dims=["distance_rupture", "magnitude"]
        ).compute()

        # and the inverse-interpolated hazard per return period
        hazard = calculate_hazard(hazard_curves_mean_z, return_periods)
        ds = xr.Dataset(
            {
                "hazard_curves_mean": hazard_curves_mean_z,
                "hazard_mean": hazard,
            }
        )

        # store flattenend, using zone, x, y for coordinates
        tx.store(ds, "hazard", config, **storage_kwargs)
        storage_kwargs["append_dim"] = "zone_x_y"
        storage_kwargs["mode"] = "a"

    stop = timeit.default_timer()
    total_time = stop - start
    logging.info(f"total time: {total_time / 60:.2f} mins")

    return


def calculate_hazard(exceedence_frequencies, return_periods):
    # convert return periods to dimension in xarray
    rp = xr.DataArray(
        np.array(return_periods), coords={"return_period": return_periods}
    )

    # put exceedance periods on the "x" - axis
    # introduce minus sign to force ascending order
    x_range = -exceedence_frequencies.reset_coords(drop=True).fillna(0.0)

    # the location to interpolate return frequencies
    # introduce minus sign to force ascending order
    x = -(1 / rp)

    # put spectral accelerations on the "y" - axis
    # transform to log for smooth interpolation
    y_range = exceedence_frequencies["SA_g_surface"].reset_coords(drop=True)

    # search and find location x_index of x in x_range, then lookup corresponding
    # exceedence frequency (x) and spectral acceleration (y)
    # take log for smooth interpolation
    x_index = (
        searchsorted(x_range, x).astype(int).clip(2, x_range.sizes["gm_surface"] - 2)
    )
    with np.errstate(divide="ignore"):
        log_x_up = np.log(-x_range.isel(gm_surface=x_index))
        log_x_down = np.log(-x_range.isel(gm_surface=x_index - 1))
        y_up = np.log(y_range.isel(gm_surface=x_index))
        y_down = np.log(y_range.isel(gm_surface=x_index - 1))

        # linear interpolation in log domain
        log_x = np.log(-x)
    frac_up = (log_x - log_x_down) / (log_x_up - log_x_down)
    frac_down = 1 - frac_up
    hazard = np.exp(frac_up * y_up + frac_down * y_down)

    return hazard


def searchsorted(x_range, x):
    return xr.apply_ufunc(
        searchsorted_ufunc,
        x_range,
        x,
        input_core_dims=[["gm_surface"], x.dims],
        output_core_dims=[x.dims],
        dask="allowed",
    )


def get_exposure_nodes(exposure_grid, relevant_zones):
    egz = (
        exposure_grid["overlap_fraction"]
        .astype(bool)
        .rename("contributing")
        .sel({"zone": relevant_zones})
        .stack(zone_x_y=("zone", "x", "y"))
    )
    egz = egz.where(egz, drop=True)
    return egz


def extract_zone_data(sd, hp, egz, zone):
    """
    sd: seismicity_rate
    hp: hazard_prep
    egz: exposure nodes
    """
    # determin the x,y coordinates contributing to this zone
    eg_zone = egz.sel({"zone": zone})
    x_y = eg_zone["zone_x_y"].data

    # select the hazard_prep data for this zone
    hp_z = hp.sel({"zone": zone}, drop=True)

    # select the source distribution data for the nodes contributing to this zone
    sd_z = sd.sel({"x_y": x_y}).reset_index("x_y")

    # some bookkeeping, adding back zone coordinate, rename x_y to zone_x_y
    sd_z["zone"] = xr.full_like(sd_z["x_y"], zone, dtype=object)
    sd_z = sd_z.rename({"x_y": "zone_x_y"})

    return sd_z, hp_z


if __name__ == "__main__":
    main(sys.argv)
