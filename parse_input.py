"""
Convert all FCM and GMM coefficient files into xarray data structures
"""

import sys
import logging
import timeit
from convert import convert_GMM
from convert import convert_FCM
from chaintools.chaintools.tools_configuration import preamble
from chaintools.chaintools.tools_xarray import store, construct_path


def main(args):
    config = preamble(args)
    logging.info("starting module %s in file %s", __name__, __file__)
    start = timeit.default_timer()

    gmm_data = convert_GMM.convert(construct_path(config["gmm_path"]))
    for key, value in gmm_data.items():
        store(value.ds, "gmm_config", config, group=key)

    fcm_data = convert_FCM.convert(construct_path(config["fcm_path"]))
    for key, value in fcm_data.items():
        store(value.ds, "fcm_config", config, group=key)

    stop = timeit.default_timer()
    total_time = stop - start
    logging.info(f"total time: {total_time / 60:.2f} mins")


if __name__ == "__main__":
    main(sys.argv)
