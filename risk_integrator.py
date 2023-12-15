"""	
Module to calculate risk due to a source distribution, in terms of exceedance
probabilities of damage and collapse states, as well as probabilities of loss
of life.
"""
import sys
import logging
import timeit
import xarray as xr
from tqdm import tqdm

from chaintools.chaintools.tools_configuration import preamble
from chaintools.chaintools import tools_xarray as tx


def main(args):
    module_name = "risk_integrator"
    config, _ = preamble(args, module_name)
    logging.info(f"starting {module_name}")
    start = timeit.default_timer()

    # open data sources
    src_dist = tx.open("source_distribution", config)
    risk_prep = tx.open("risk_prep", config)
    exposure_grid = tx.open("exposure_grid", config, chunking_allowed=False)

    # preprocessing
    src_dist = src_dist.set_xindex(["x", "y"])

    # shorthands
    # sd organized by x,y; hp organized by zone
    # the mean source distribution is forced into memory
    sd_mean = src_dist["seismicity_rate_mean"].persist()
    risk_names = [
        "structural_poe",
        "chimney_poe",
        "structural_pod",
        "chimney_pod",
        "LPR",
    ]
    risk_mean_names = [n + "_mean" for n in risk_names]
    rp_mean = risk_prep[risk_mean_names]

    # prepare surface grid nodes
    # find relevant combinations of zone and x,y
    relevant_zones = risk_prep["zone"].data
    egz = get_exposure_nodes(exposure_grid, relevant_zones)

    storage_kwargs = {"mode": "w-", "compute": True}
    for zone in tqdm(relevant_zones):
        # for each zone collect the relevant nodes using the exposure grid
        sd_mean_z, rp_mean_z = extract_zone_data(sd_mean, rp_mean, egz, zone)

        # calculate risk per zone
        contract = lambda da: xr.dot(
            sd_mean_z, da, dims=["distance_rupture", "magnitude"]
        )
        risk_mean_z = rp_mean_z.map(contract).compute()

        # store flattenend, using zone, x, y for coordinates
        tx.store(risk_mean_z, "risk", config, **storage_kwargs)
        storage_kwargs["append_dim"] = "zone_x_y"
        storage_kwargs["mode"] = "a"

    stop = timeit.default_timer()
    total_time = stop - start
    logging.info(f"total time: {total_time / 60:.2f} mins")

    return


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


def extract_zone_data(sd, rp, egz, zone):
    # determin the x,y coordinates contributing to this zone
    eg_zone = egz.sel({"zone": zone})
    x_y = eg_zone["zone_x_y"].data

    # select the risk_prep data for this zone
    hp_z = rp.sel({"zone": zone}, drop=True)

    # select the source distribution data for the nodes contributing to this zone
    sd_z = sd.sel({"x_y": x_y}).reset_index("x_y")

    # some bookkeeping, adding back zone coordinate, rename x_y to zone_x_y
    sd_z["zone"] = xr.full_like(sd_z["x_y"], zone, dtype=object)
    sd_z = sd_z.rename({"x_y": "zone_x_y"})

    return sd_z, hp_z


if __name__ == "__main__":
    main(sys.argv)
