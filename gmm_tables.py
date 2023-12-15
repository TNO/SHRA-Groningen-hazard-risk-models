"""
Generate tables of ground motion distribution parameters

Generate tables of ground motion distibution parameters conditional
on ranges for distances, magnitudes, as specified in the configuration file
provided as a first argument on the command line
"""
import sys
import logging
import timeit
import scipy.stats as st
import numpy as np
import xarray as xr
from dask.distributed import progress

from models import gmm_V5V6, gmm_V7
from chaintools.chaintools.tools_configuration import preamble
from chaintools.chaintools import tools_xarray as tx


def main(args):
    module_name = "gmm_tables"
    config, client = preamble(args, module_name)
    logging.info(f"starting {module_name}")
    start = timeit.default_timer()

    # set up coordinates dataset and prepare for DASK
    table_ds = tx.prepare_ds(config)

    # open gmm configuration data
    gmm_config = tx.open("gmm_config", config)

    # select only spectral periods - no PGV or duration beyond this point
    gmm_config = gmm_config.sel(IM=gmm_config["T"].compute().notnull())

    logging.info("calculating basic tables")
    table_ds = gmm_tables(gmm_config, table_ds)
    storage_task = tx.store(table_ds, module_name, config, mode="w-", compute=False)
    job = client.compute(storage_task)
    progress(job)

    logging.info("calculating reference exceedence probabilities")
    table_ds = tx.chunk(table_ds, config["chunks"])
    table_ds = calculate_reference_poe(table_ds)
    storage_task = tx.store(table_ds, module_name, config, mode="a", compute=False)
    job = client.compute(storage_task)
    progress(job)

    logging.info("calculating reference probabily densities")
    table_ds = tx.chunk(table_ds, config["chunks"])
    table_ds = calculate_reference_pmf(table_ds)
    storage_task = tx.store(table_ds, module_name, config, mode="a", compute=False)
    job = client.compute(storage_task)
    progress(job)

    stop = timeit.default_timer()
    total_time = stop - start
    logging.info(f"total time: {total_time / 60:.2f} mins")

    return


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
    # apply_ufunc takes case of mainaining xarray metadata
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

    table_ds["af_max"] = xr.apply_ufunc(
        package.af_max,
        af_pars,
        kwargs={"par_id": af_par_ids.values},
        input_core_dims=[["parameter_af"]],
        exclude_dims=set(("parameter_af",)),
        dask="parallelized",
        output_dtypes=[float],
    )

    if gmm_version in ["GMM-V7"]:
        table_ds["branch_median_weights"] = xr.apply_ufunc(
            package.branch_median_weights,
            m,
            output_core_dims=[["branch_median"]],
            dask="parallelized",
            output_dtypes=[float],
        ).assign_coords(
            {"branch_median": ["Lower", "CentralLower", "CentralUpper", "Upper"]}
        )

        # following serves as a multiplier on the event rates, to account for
        # the fact that the branch median weights are not constant across
        # the magnitude range
        table_ds["rate_multiplier"] = (
            table_ds["branch_median_weights"] / gmm_config["logic_tree:branch_median"]
        )

    # V7 elements that have been adopted in all previous versions
    table_ds["s2s_epsilons"] = gmm_config["s2s_epsilons"]
    table_ds["af_delta_s2s"] = table_ds["s2s_epsilons"] * np.sqrt(
        table_ds["af_variance"]
    )
    table_ds["wierde_factor"] = gmm_config["wierde_factor"]

    # copy logic tree weights
    logictree = gmm_config[[v for v in gmm_config if "logic_tree:" in v]]
    table_ds = table_ds.merge(logictree)

    # restore attributes
    table_ds["distance_rupture"].attrs = r_attrs
    table_ds["magnitude"].attrs = m_attrs

    return table_ds


def calculate_reference_poe(table_ds):
    """
    Calculate exceedence probabilities of all ground motion components at reference level
    """

    # two shorthands
    lnsa = np.log(table_ds["SA_reference"])
    var = xr.concat(
        [
            table_ds["reference_ac_variance"].expand_dims({"component": ["arbitrary"]}),
            table_ds["reference_gm_variance"].expand_dims(
                {"component": ["geometric_mean"]}
            ),
        ],
        dim="component",
    )

    # calculate exceedence probabilities
    table_ds["reference_poe"] = xr.apply_ufunc(
        st.norm.sf,
        lnsa,
        table_ds["reference_median"],
        np.sqrt(var),
        dask="parallelized",
        output_dtypes=[float],
    )

    # copy logic tree weights
    logictree = table_ds[[v for v in table_ds if "logic_tree:" in v]]
    table_ds = table_ds.merge(logictree)
    logictree_ref = logictree.drop_dims("branch_s2s")
    logictree_ref_dims = [d for d in logictree_ref.dims if d.startswith("branch")]

    rate_multiplier = table_ds.get("rate_multiplier", 1.0)
    table_ds["reference_poe_mean"] = xr.dot(
        rate_multiplier * table_ds["reference_poe"],
        *logictree_ref.values(),
        dims=logictree_ref_dims,
        optimize=True,
    )

    return table_ds


def calculate_reference_pmf(table_ds):
    """
    Calculate probability mass functions of all ground motion components at reference level
    """

    table_ds["reference_pmf"] = table_ds["reference_poe"] - table_ds[
        "reference_poe"
    ].shift({"gm_reference": -1}, fill_value=0.0)

    table_ds["reference_pmf_mean"] = table_ds["reference_poe_mean"] - table_ds[
        "reference_poe_mean"
    ].shift({"gm_reference": -1}, fill_value=0.0)

    return table_ds


if __name__ == "__main__":
    main(sys.argv)
