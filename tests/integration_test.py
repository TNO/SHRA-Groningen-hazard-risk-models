import os
import importlib
import tempfile
import collections
import yaml
import xarray as xr
from pathlib import Path
from copy import deepcopy

from chaintools.chaintools import tools_configuration as tc
from chaintools.chaintools import tools_xarray as tx

dirname = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(dirname, "config_test.yml")


def tst_module(module):
    if isinstance(module, str):
        modules = [module]
    elif isinstance(module, list):
        modules = module
    else:
        raise TypeError("module must be a string or a list of strings")
    config = tc.configure([yaml_path])
    for task in config["tasks"].values():
        if task["module"]["python_module"] in modules:
            task["configuration"].update(config.get("generic", {}))
            # each task may change the current working directory
            # this is bound to go wrong for relative paths
            # so we restore the original working directory after each task
            restore_dir = os.getcwd()
            try:
                tst_task(task)
            except Exception as e:
                print(f"Error in task {task['module']['python_module']}: {e}")
                raise e
            finally:
                os.chdir(restore_dir)


def tst_task(task):
    config_ref = deepcopy(task["configuration"])
    config_tst = deepcopy(task["configuration"])
    config_ref["data_sinks"] = rename_zarr_to_zip(config_ref["data_sinks"])
    if "data_sources" in config_tst:
        config_tst["data_sources"] = rename_zarr_to_zip(config_tst["data_sources"])
    with tempfile.TemporaryDirectory() as dir:
        config_tst["data_sinks"] = move_to_dir(config_tst["data_sinks"], dir)
        temp_config_path = build_config_file_for_task(config_tst, dir).as_posix()
        module = get_module(
            task["module"]["python_module"],
            task["module"].get("python_submodule", None),
        )
        module.main(["dummy", temp_config_path])
        for sink_name in config_tst["data_sinks"]:
            ds_tst = tx.data_source(**config_tst["data_sinks"][sink_name])
            ds_ref = tx.data_source(**config_ref["data_sinks"][sink_name])
            if not isinstance(ds_ref, Path):
                if isinstance(ds_tst, (xr.DataArray, xr.Dataset)):
                    dims = list(ds_ref.dims)
                    ds_tst = ds_tst.transpose(*dims)
                    ds_ref = ds_ref.transpose(*dims)
                    xr.testing.assert_allclose(ds_tst, ds_ref)
                elif isinstance(ds_tst, xr.DataTree):
                    assert ds_tst.equals(ds_ref)
            else:
                print(
                    "Only tested that the main did not crash. Did not compare against reference"
                )
                print("Should only happen for visualization results")


def test_parse_input():
    tst_module("parse_input")


def test_gmm_tables():
    tst_module("tabulate_gmm")


def test_fcm_tables():
    tst_module("tabulate_fcm")


def test_prepare_hazard_lookup():
    tst_module("prepare_hazard_lookup")


def test_prepare_im_lookup():
    tst_module("prepare_im_lookup")


def test_prepare_im_lookup_mom():
    tst_module("prepare_im_lookup_mom")


def test_prepare_risk_lookup():
    tst_module("prepare_risk_lookup")


def test_process():
    tst_module("process")


def test_prepare_rupture_lookup():
    tst_module("prepare_rupture_lookup")


def test_exposure_prep():
    tst_module("prepare_exposure")


def test_aggregate_source():
    tst_module("aggregate_source")


def test_aggregate_exposure():
    tst_module("aggregate_exposure")


def test_aggregate_exposure_ds1():
    tst_module("aggregate_exposure_ds1")


def test_prepare_exposure_ds1():
    tst_module("prepare_exposure_ds1")


def test_integrate_by_zones():
    tst_module("integrate_by_zones")


def test_extract_hazard():
    tst_module("extract_hazard")


def test_integrate_by_nodes():
    tst_module("integrate_by_nodes")


def test_confront_exposure():
    tst_module("confront_exposure")


def test_integrate_ds1():
    tst_module("integrate_ds1")


def test_visualize_hazard():
    tst_module("visualize_hazard")


def test_visualize_risk():
    tst_module("visualize_risk")


def test_exposure_visualization():
    pass
    tst_module("visualize_exposure")


def build_config_file_for_task(config: dict, dir: str, config_name: str = None) -> Path:
    """
    Write and save a new configuration .yaml file for a specific task.
    :param config: Dictionary with configuration for a specific task
    :param dir: Directory where the configuration file will be stored
    :param config_name: Optional, name of the configuration file.
    :return: Returns the path to the new configuration file.
    """
    if config_name is None:
        config_name = "temp_config.yml"
    yml_path = Path(dir) / config_name
    with open(yml_path, "w") as f:
        yaml.dump(config, f)

    return yml_path


def rename_zarr_to_zip(data_stores_in):
    data_stores = deepcopy(data_stores_in)
    for name, data in data_stores.items():
        data_stores[name] = _rename_zarr_to_zip(data)
    return data_stores


def _rename_zarr_to_zip(data):
    if isinstance(data, collections.abc.Sequence):
        return [_rename_zarr_to_zip(d) for d in data]
    elif isinstance(data, collections.abc.Mapping):
        if "path" in data:
            path = tx.construct_path(data["path"])
            if path.suffix == ".zarr":
                data["path"] = path.with_suffix(".zip").as_posix()
        return data


def move_to_dir(data_stores_in, dir):
    data_stores = deepcopy(data_stores_in)
    for name, data in data_stores.items():
        data_stores[name] = _move_to_dir(data, dir)
    return data_stores


def _move_to_dir(data, dir):
    if isinstance(data, collections.abc.Sequence):
        return [_move_to_dir(d, dir) for d in data]
    elif isinstance(data, collections.abc.Mapping):
        if "path" in data:
            path = tx.construct_path(data["path"])
            data["path"] = [dir, path.name]
        return data


def get_module(module_name, sub_module):
    if not sub_module:
        module = importlib.import_module(module_name)
    else:
        module_name = "." + module_name
        module = importlib.import_module(module_name, sub_module)
    return module


if __name__ == "__main__":
    test_exposure_prep()
    test_parse_input()
    test_gmm_tables()
    test_fcm_tables()
    test_prepare_hazard_lookup()
    test_prepare_im_lookup()
    test_prepare_im_lookup_mom()
    test_prepare_risk_lookup()
    test_process()
    test_prepare_rupture_lookup()
    test_aggregate_source()
    test_integrate_by_zones()
    test_extract_hazard()
    test_integrate_by_nodes()
    test_confront_exposure()
    test_integrate_ds1()
    test_visualize_hazard()
    test_visualize_risk()
