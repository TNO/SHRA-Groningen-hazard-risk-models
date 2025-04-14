"""
Generate tables of ground motion distribution parameters

Generate tables of ground motion distibution parameters conditional
on ranges for distances, magnitudes, as specified in the configuration file
provided as a first argument on the command line
"""

import sys
import logging
import timeit
import numpy as np
import xarray as xr

from hr_models import gmm_V5V6, gmm_V7
from chaintools.chaintools.tools_configuration import preamble
from chaintools.chaintools import tools_xarray as tx


def main(args):
    config = preamble(args)
    logging.info("starting module %s in file %s", __name__, __file__)
    start = timeit.default_timer()

    run_core(config)

    stop = timeit.default_timer()
    total_time = stop - start
    logging.info(f"total time: {total_time / 60:.2f} mins")

    return


def run_core(config):
    # set up coordinates dataset and prepare for DASK
    table_ds = tx.prepare_ds(config)

    # open gmm configuration data
    gmm_config = tx.open("gmm_config", config)

    # select only spectral periods - no PGV or duration beyond this point
    im_selection = gmm_config["T"].compute().notnull()
    gmm_config = gmm_config.sel(IM=im_selection)

    logging.info("calculating basic tables")
    table_ds = gmm_tables(gmm_config, table_ds)
    tx.store(table_ds, "tables", config, mode=config.get("file_mode", "w-"))

    logging.info("exporting logic tree weights")
    logic_tree = gmm_config[[v for v in gmm_config if v.startswith("w_")]]
    tx.store(logic_tree, "logic_tree", config, mode=config.get("file_mode", "w-"))


def gmm_tables(gmm_config, table_ds):
    # determine functions to use
    # this may be a use case for a class; but this seems to work OK
    gmm_version = gmm_config["gmm_version"]
    if gmm_version in ["GMM-V7"]:
        package = gmm_V7
    elif gmm_version in ["GMM-V5", "GMM-V6"]:
        package = gmm_V5V6

    # define shorthands from sample structure
    r = table_ds["distance_rupture"]
    m = table_ds["magnitude"]
    lnsa = np.log(table_ds["SA_reference"])

    # save attrs, for some reason they get lost in the process
    r_attrs = r.attrs
    m_attrs = m.attrs

    # define shorthands from gmm_config structure
    t = gmm_config["T"]
    ref_pars = gmm_config["median_parameters"]
    ref_par_ids = gmm_config["parameter_median"]
    af_pars = gmm_config["af_parameters"]
    af_par_ids = gmm_config["parameter_af"]
    tau = gmm_config["tau"]
    phiss = gmm_config["phiss"]

    # perform calculations
    # apply_ufunc takes care of maintaining xarray metadata
    table_ds["reference_median"] = xr.apply_ufunc(
        package.reference_median,
        r,
        m,
        t,
        ref_pars,
        kwargs={"par_id": ref_par_ids.values},
        input_core_dims=[[], [], [], ["parameter_median"]],
        exclude_dims=set(("parameter_median",)),
        dask="parallelized",
        output_dtypes=[float],
    )

    table_ds["reference_ac_variance"] = xr.apply_ufunc(
        package.reference_ac_variance,
        r,
        m,
        t,
        tau,
        phiss,
        dask="parallelized",
        output_dtypes=[float],
    )

    table_ds["reference_gm_variance"] = xr.apply_ufunc(
        package.reference_gm_variance,
        tau,
        phiss,
        dask="parallelized",
        output_dtypes=[float],
    )

    table_ds["reference_variance"] = xr.Dataset(
        {
            "arbitrary_component": table_ds["reference_ac_variance"],
            "geometric_mean": table_ds["reference_gm_variance"],
        }
    ).to_array(dim="component")

    table_ds["surface_median"] = xr.apply_ufunc(
        package.surface_median,
        r,
        m,
        t,
        ref_pars,
        af_pars,
        kwargs={"ref_par_id": ref_par_ids.values, "af_par_id": af_par_ids.values},
        input_core_dims=[[], [], [], ["parameter_median"], ["parameter_af"]],
        exclude_dims=set(("parameter_median", "parameter_af")),
        dask="parallelized",
        output_dtypes=[float],
    )

    table_ds["af_median"] = xr.apply_ufunc(
        package.af_median,
        r,
        m,
        lnsa,
        af_pars,
        kwargs={"par_id": af_par_ids.values},
        input_core_dims=[[], [], [], ["parameter_af"]],
        exclude_dims=set(("parameter_af",)),
        dask="parallelized",
        output_dtypes=[float],
    )

    table_ds["af_median_nonlinear"] = xr.apply_ufunc(
        package.af_median_nonlinear,
        lnsa,
        af_pars,
        kwargs={"par_id": af_par_ids.values},
        input_core_dims=[[], ["parameter_af"]],
        exclude_dims=set(("parameter_af",)),
        dask="parallelized",
        output_dtypes=[float],
    )

    table_ds["af_variance"] = xr.apply_ufunc(
        package.af_variance,
        lnsa,
        af_pars,
        kwargs={"par_id": af_par_ids.values},
        input_core_dims=[[], ["parameter_af"]],
        exclude_dims=set(("parameter_af",)),
        dask="parallelized",
        output_dtypes=[float],
    )

    table_ds["af_median_linear"] = xr.apply_ufunc(
        package.af_median_linear,
        r,
        m,
        af_pars,
        kwargs={"par_id": af_par_ids.values},
        input_core_dims=[[], [], ["parameter_af"]],
        exclude_dims=set(("parameter_af",)),
        dask="parallelized",
        output_dtypes=[float],
    )

    if gmm_version in ["GMM-V7"]:
        median_weights = xr.apply_ufunc(
            package.median_weights,
            m,
            output_core_dims=[["b_median"]],
            dask="parallelized",
            output_dtypes=[float],
        ).assign_coords(
            {"b_median": ["Lower", "CentralLower", "CentralUpper", "Upper"]}
        )  # assign coords to ensure proper alignment

        # following serves as a multiplier on the event rates, to account for
        # the fact that the branch median weights are not constant across
        # the magnitude range
        rate_multiplier = median_weights / gmm_config["w_median"]
    else:
        rate_multiplier = xr.DataArray(1)  # trivial multiplier for GMM-V5 and GMM-V6
    table_ds["rate_multiplier"] = rate_multiplier

    # V7 elements that have been adopted in all previous versions
    table_ds["s2s_epsilons"] = gmm_config["s2s_epsilons"]
    table_ds["af_delta_s2s"] = table_ds["s2s_epsilons"] * np.sqrt(
        table_ds["af_variance"]
    )
    table_ds["wierde_factor"] = gmm_config["wierde_factor"]

    # restore attributes
    table_ds["distance_rupture"].attrs = r_attrs
    table_ds["magnitude"].attrs = m_attrs

    return table_ds


if __name__ == "__main__":
    main(sys.argv)
