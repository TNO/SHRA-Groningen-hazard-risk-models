"""
Convert GMM coefficient files into xarray data structures

This script converts the original coefficient files for the ground motion models
of Bommer et al. into a xarray data structures, exported in netCDF/HDF5 format.

"""

from pathlib import Path
import datatree as dt

try:
    import convert_GMMV5
    import convert_GMMV6
    import convert_GMMV7
except ImportError:
    from convert import convert_GMMV5
    from convert import convert_GMMV6
    from convert import convert_GMMV7

base_path = "./convert/res/"
output_file = "./GMM_config.h5"


def convert(base_path):
    output = {}

    output["GMM-V5"] = convert_GMMV5.convert(Path(base_path) / "V5")
    output["GMM-V6"] = convert_GMMV6.convert(Path(base_path) / "V6")
    output["GMM-V7"] = convert_GMMV7.convert(Path(base_path) / "V7")

    datatree = dt.DataTree.from_dict(output)
    return datatree


if __name__ == "__main__":
    ds = convert(base_path)
    ds.to_netcdf(output_file, engine="h5netcdf")
