"""
Convert all FCM and GMM coefficient files into xarray data structures
"""
import sys
import logging
from convert import convert_GMM
from convert import convert_FCM
from chaintools.chaintools.tools_configuration import configure
from chaintools.chaintools.tools_xarray import store, construct_path

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

def main(args):
    module_name = "parse_input"
    logging.info(f"starting {module_name}")
    config = configure(args[1:], module_name)

    gmm_data = convert_GMM.convert(construct_path(config["gmm_path"]))
    for key, value in gmm_data.items():
        store(value.ds, "gmm_config", config, group=key)

    fcm_data = convert_FCM.convert(construct_path(config["fcm_path"]))
    for key, value in fcm_data.items():
        store(value.ds, "fcm_config", config, group=key)


if __name__ == "__main__":
    main(sys.argv)
