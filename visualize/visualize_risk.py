"""
Module to visualize risk
This is quite ad hoc for the moment
"""

import sys
import logging
import time
import datetime

from chaintools.chaintools.tools_configuration import preamble
from chaintools.chaintools import tools_xarray as tx
from visualize.visualization_tools import plot_risk_maps


def main(args):
    config = preamble(args)
    logging.info("starting module %s in file %s", __name__, __file__)

    risk = tx.open("risk", config)
    zones = tx.open("zonation", config)
    zones = tx.to_geopandas(zones, to_crs=config["grid_crs"])
    fig_path = tx.construct_path(config["data_sinks"]["visualization_results"]["path"])
    fig_path.mkdir(parents=True, exist_ok=True)

    if "lt_median_choice" in risk.dims:
        lt_selection = config.get(
            "lt_median_choice", risk["lt_median_choice"].values[-1]
        )
        risk = risk.sel(lt_median_choice=lt_selection)
    plot_risk_maps(risk, zones, fig_path=fig_path)


if __name__ == "__main__":
    time0 = time.time()
    # First command-line argument is passed as the path to the configuration file or else default is used
    args = (
        sys.argv[:]
        if sys.argv[1:]
        else ["dummy", "test_vis.yml", "--task", "test_vis_risk"]
    )
    main(args)
    time1 = time.time()
    print(f"Done in {str(datetime.timedelta(seconds=int(time1 - time0)))} (hh:mm:ss)")
