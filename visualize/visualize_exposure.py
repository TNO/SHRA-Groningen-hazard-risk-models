"""
Module to visualize exposure
This is quite ad hoc for the moment
"""

import sys
import logging
import time
import datetime

from chaintools.chaintools.tools_configuration import preamble
from chaintools.chaintools import tools_xarray as tx
from visualize.visualization_tools import (
    plot_exposure_curves,
    plot_typology_curves,
    plot_ds_curves,
    create_ncg_table,
)


def main(args):
    config = preamble(args)
    logging.info("starting module %s in file %s", __name__, __file__)

    fig_path = tx.construct_path(config["data_sinks"]["visualization_results"]["path"])
    fig_path.mkdir(parents=True, exist_ok=True)
    try:
        exposure_risk = tx.open("exposure_risk", config)
    except KeyError:
        exposure_risk = None
    try:
        exposure_db = tx.open("exposure_database", config)
    except KeyError:
        exposure_db = None
    try:
        exposure_ds1 = tx.open("exposure_ds1", config)
    except KeyError:
        exposure_ds1 = None

    if exposure_risk is not None:
        create_ncg_table(exposure_risk, fig_path=fig_path)
        plot_exposure_curves(exposure_risk, fig_path=fig_path)

        if exposure_ds1 is not None and "limit_state" in exposure_risk:
            plot_ds_curves(exposure_risk, exposure_ds1, fig_path=fig_path)
        if exposure_db is not None:
            plot_typology_curves(exposure_risk, exposure_db, fig_path=fig_path)


if __name__ == "__main__":
    time0 = time.time()
    # First command-line argument is passed as the path to the configuration file or else default is used
    args = (
        sys.argv[:]
        if sys.argv[1:]
        else ["dummy", "test_vis.yml", "--task", "test_vis_exposure2"]
    )
    main(args)
    time1 = time.time()
    print(f"Done in {str(datetime.timedelta(seconds=int(time1 - time0)))} (hh:mm:ss)")
