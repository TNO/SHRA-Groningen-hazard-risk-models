"""
Generate intensity measure prep, i.e. the ground motion intensity in terms
of the intensity measures use in the fragility and consequence models.
This version is based on the method of moments.
"""

import sys
import logging
import timeit
import numpy as np
import xarray as xr
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from math import fabs, erf, erfc
from numba import vectorize, float64

from chaintools.chaintools import tools_configuration as tc
from chaintools.chaintools import tools_xarray as tx
from chaintools.chaintools import tools_grid as tg


def assign_defaults(config, gmm_tables):
    gmm_version = gmm_tables["gmm_version"].item()
    if gmm_version in ["GMM-V5", "GMM-V6"]:
        s2s_mode_default = "aleatory"
    else:
        s2s_mode_default = "epistemic"

    # treatment of site-to-site (s2s) variability
    # defaults can be overridden in config
    s2s_mode = config.get("s2s_mode", "default")
    if s2s_mode == "default":
        config["s2s_mode"] = s2s_mode_default

    config.setdefault("s2s_p2p_mode", "consistent")
    config.setdefault("file_mode", "w-")


def main(args):
    config = tc.preamble(args)
    logging.info("starting module %s in file %s", __name__, __file__)
    start = timeit.default_timer()

    run_core(config, start)

    stop = timeit.default_timer()
    total_time = stop - start
    logging.info(f"total time: {total_time / 60:.2f} mins")

    return


def run_core(config, start):
    # default config assignment depends on the gmm version
    # hence it is postponed until the gmm tables are opened

    # open gmm configuration and tabular data
    fcm_config = tx.open("fcm_config", config)
    gmm_config = tx.open("gmm_config", config)
    gmm_tables = tx.open("gmm_tables", config)
    p2p_empirical = tx.open("p2p_empirical", config)

    # assign default configuration values
    assign_defaults(config, gmm_tables)

    # preprocessing
    gmm_config, gmm_tables = preprocess(
        config,
        fcm_config,
        gmm_config,
        gmm_tables,
        p2p_empirical,
    )

    # prepare output dataset
    output_ds = tx.prepare_ds(config)
    lnSA_ref = np.log(output_ds["SA_reference"])

    logging.info(f"generate reference marginal pmf")
    output_ds["reference_full_pmf"] = discretize_normal_distribution(
        gmm_tables["reference_median"],
        gmm_tables["reference_variance"],
        lnSA_ref,
        "gm_reference",
        mode="pmf",
        shift=True,
        clip=True,
    )
    tx.store(output_ds, "output", config, mode=config["file_mode"])
    stop = timeit.default_timer()
    total_time = stop - start
    logging.info(f"intermediate time: {total_time / 60:.2f} mins")

    logging.info(f"generate first and second reference moments")
    input_ds = tx.open("output", config)
    output_ds = tx.prepare_ds(config)
    reference_mean = gmm_tables["reference_median"]
    reference_variance = generate_reference_variance(gmm_tables, gmm_config)
    reference_moments_fcm = extract_fcm_intensity_measures(
        reference_mean, reference_variance
    )
    output_ds["reference_median"] = reference_moments_fcm["mean"]
    output_ds["reference_variance"] = reference_moments_fcm["var"]
    tx.store(output_ds, "output", config)
    del reference_mean, reference_variance, reference_moments_fcm
    stop = timeit.default_timer()
    total_time = stop - start
    logging.info(f"intermediate time: {total_time / 60:.2f} mins")

    logging.info(f"generate expected amplification factors")
    input_ds = tx.open("output", config)
    output_ds = tx.prepare_ds(config)
    output_ds["af_median"] = gmm_tables["af_median"]
    # store on disk because of  : https://github.com/dask/dask/issues/874
    output_ds["mean_mean_af"] = xr.dot(
        input_ds["reference_full_pmf"],
        output_ds["af_median"],
        dim="gm_reference",
    )
    tx.store(output_ds, "output", config)
    stop = timeit.default_timer()
    total_time = stop - start
    logging.info(f"intermediate time: {total_time / 60:.2f} mins")

    logging.info(f"generate surface mean")
    input_ds = tx.open("output", config)
    output_ds = tx.prepare_ds(config)
    lnSA_srf = np.log(output_ds["SA_surface"])
    output_ds["surface_full_median"] = generate_surface_mean(gmm_tables, input_ds)
    tx.store(output_ds, "output", config)
    stop = timeit.default_timer()
    total_time = stop - start
    logging.info(f"intermediate time: {total_time / 60:.2f} mins")

    logging.info(f"generate surface covariance")
    input_ds = tx.open("output", config)
    output_ds = tx.prepare_ds(config)
    surface_covariance = generate_surface_covariance(
        gmm_tables, gmm_config, input_ds, config
    )
    output_ds["surface_full_marginal_variance"] = surface_covariance.sel(
        IM_T=surface_covariance["IM"]
    ).drop_vars("IM_T")
    surface_moments_fcm = extract_fcm_intensity_measures(
        input_ds["surface_full_median"],
        surface_covariance,
    )
    output_ds["surface_median"] = surface_moments_fcm["mean"]
    output_ds["surface_variance"] = surface_moments_fcm["var"]
    tx.store(output_ds, "output", config)
    del surface_moments_fcm
    stop = timeit.default_timer()
    total_time = stop - start
    logging.info(f"intermediate time: {total_time / 60:.2f} mins")

    logging.info(f"generate surface pmf for fragility")
    input_ds = tx.open("output", config)
    output_ds = tx.prepare_ds(config)
    surface_fcm_marginal_variance = (
        input_ds["surface_variance"]
        .sel(IM_FCM_T=input_ds["IM_FCM"])
        .drop_vars("IM_FCM_T")
    )
    output_ds["surface_pmf"] = discretize_normal_distribution(
        input_ds["surface_median"],
        surface_fcm_marginal_variance,
        lnSA_srf,
        "gm_surface",
        mode="pmf",
        shift=True,
        clip=True,
    )
    tx.store(output_ds, "output", config)
    del surface_fcm_marginal_variance
    stop = timeit.default_timer()
    total_time = stop - start
    logging.info(f"intermediate time: {total_time / 60:.2f} mins")

    if "hazard" in config["data_sinks"]:
        logging.info(f"generate marginal surface sf for hazard")
        it_dim = "IM"
        batch_size = 1
        tqdm_node_kwargs = {
            "total": gmm_tables.sizes[it_dim],
            "desc": f"Iterate over {it_dim}",
            "position": 0,
        }
        input_ds = tx.open("output", config).chunk({it_dim: batch_size})
        kwargs = {"mode": config["file_mode"]}
        with logging_redirect_tqdm(), tqdm(**tqdm_node_kwargs) as pbar:
            for batch in tc.batched(gmm_tables[it_dim].values, batch_size):
                output_ds = xr.Dataset()
                output_ds["reference_poe"] = discretize_normal_distribution(
                    gmm_tables["reference_median"].sel({it_dim: list(batch)}),
                    gmm_tables["reference_variance"].sel({it_dim: list(batch)}),
                    lnSA_ref,
                    "gm_reference",
                    mode="sf",
                    shift=False,
                    clip=False,
                )

                output_ds["surface_poe"] = discretize_normal_distribution(
                    input_ds["surface_full_median"].sel({it_dim: list(batch)}),
                    input_ds["surface_full_marginal_variance"].sel(
                        {it_dim: list(batch)}
                    ),
                    lnSA_srf,
                    "gm_surface",
                    mode="sf",
                    shift=False,
                    clip=False,
                )

                tx.store(output_ds, "hazard", config, **kwargs)
                pbar.update(len(batch))
                kwargs["mode"] = "a"
                kwargs["append_dim"] = it_dim


def preprocess(config, fcm_config, gmm_config, gmm_tables, p2p_empirical):
    if config["s2s_mode"] == "aleatory":
        gmm_tables = gmm_tables.drop_dims("b_s2s")
    elif config["s2s_mode"] == "epistemic":
        sigma = np.sqrt(gmm_tables["af_variance"])
        median = gmm_tables["af_median"]
        gmm_tables["af_median"] = median + sigma * gmm_tables["s2s_epsilons"]
        gmm_tables["af_variance"] = 0.0
    else:
        raise ValueError(f"unknown s2s_mode: {config['s2s_mode']}")

    # select the spectral periods that are of interest to the FCM
    # note that we assume here that all are present and
    # we don't have to interpolate
    im_selection = {}
    im_selection["IM"] = fcm_config["IM"].data
    gmm_tables = gmm_tables.sel(im_selection)
    im_selection["IM_T"] = fcm_config["IM"].data
    gmm_config = gmm_config.sel(im_selection)
    if p2p_empirical is not None:
        p2p_empirical = p2p_empirical.sel(im_selection)
        gmm_config = gmm_config.merge(p2p_empirical)
    gmm_config["AF_correlation_matrix"] = create_s2s_correlation_matrix(
        gmm_config, config
    )

    # remove keys that interfere with repeated reading of the output file
    for k in ["isel", "islice", "thin"]:
        if k in config:
            config.pop(k)

    return gmm_config, gmm_tables


def create_s2s_correlation_matrix(conf, config):
    used_modes = []
    size = conf.sizes["IM"]
    s2s_p2p_mode = np.atleast_1d(config["s2s_p2p_mode"])
    if "zero" in s2s_p2p_mode:
        zerocorr = xr.DataArray(np.eye(size), dims=["IM", "IM_T"]).expand_dims(
            {"s2s_p2p_mode": ["zero"]}
        )
        used_modes.append(zerocorr)
    if "consistent" in s2s_p2p_mode:
        consistentcorr = conf["correlation_matrix"].expand_dims(
            {"s2s_p2p_mode": ["consistent"]}
        )
        used_modes.append(consistentcorr)
    if "full" in s2s_p2p_mode:
        fullcorr = xr.DataArray(np.ones((size, size)), dims=["IM", "IM_T"]).expand_dims(
            {"s2s_p2p_mode": ["full"]}
        )
        used_modes.append(fullcorr)
    if "empirical" in s2s_p2p_mode:
        empiricalcorr = conf["correlation_matrix_empirical"].expand_dims(
            {"s2s_p2p_mode": ["empirical"]}
        )
        used_modes.append(empiricalcorr)

    # select and sort according to configuration
    corr = xr.concat(used_modes, dim="s2s_p2p_mode").sel(
        s2s_p2p_mode=config["s2s_p2p_mode"]
    )

    return corr


def generate_reference_variance(tab, conf):
    # get marginal variance
    var_ref = tab["reference_variance"]

    # construct the covariance matrix
    sigma_ref = np.sqrt(var_ref)
    sigma_ref_T = sigma_ref.rename({"IM": "IM_T"})
    cor_ref = conf["correlation_matrix"]
    covar_ref = cor_ref * sigma_ref * sigma_ref_T

    return covar_ref


def generate_surface_mean(tab, input_ds):
    # shorthands
    mean_ref = tab["reference_median"]
    wierde_factor = tab.get("wierde_factor", 0.0)
    mean_mean_af = input_ds["mean_mean_af"]

    # STEP 1: calculate the mean of the surface ground motions
    # mean of the surface ground motions, include wierde factor
    mean_srf = mean_ref + mean_mean_af + wierde_factor

    return mean_srf


def generate_surface_covariance(gmm_tables, gmm_config, previous_results, config):
    # shorthands
    mean_ref = gmm_tables["reference_median"]
    sigma_ref = np.sqrt(gmm_tables["reference_variance"])
    lnsa_ref = np.log(gmm_tables["SA_reference"])
    conditional_mean_af = previous_results["af_median"]
    conditional_var_af = gmm_tables["af_variance"]
    cor_af = gmm_config["AF_correlation_matrix"]
    cor_ref = gmm_config["correlation_matrix"]
    reference_pmf = previous_results["reference_full_pmf"]
    mean_mean_af = previous_results["mean_mean_af"]

    # Calculate the variance of the surface ground motions
    # uses law of total variance and a number of approximations

    # variance of the expectation/mean
    var_mean_af = xr.dot(
        reference_pmf,
        (conditional_mean_af - mean_mean_af) ** 2,
        dim="gm_reference",
    )

    # mean/expectation of the variance
    if config["s2s_mode"] == "aleatory":
        mean_var_af = xr.dot(
            reference_pmf,
            conditional_var_af,
            dim="gm_reference",
        )
    else:
        mean_var_af = 0.0

    # cross-covariance of the mean of the amplification and the mean of the reference
    # note this is calculated on the IM dimension only, i.e., the diagonal of the

    # cross-covariance
    covar_mean_ref_af = xr.dot(
        reference_pmf,
        (conditional_mean_af - mean_mean_af) * (lnsa_ref - mean_ref),
        dim="gm_reference",
    )

    # auto-covariance of the reference
    sigma_ref_T = sigma_ref.rename({"IM": "IM_T"})
    covar_ref = cor_ref * sigma_ref * sigma_ref_T

    # auto-covariance of the amplification induced by the reference - approximations
    sqrt_var_mean_af = np.sqrt(var_mean_af)
    sqrt_var_mean_af_T = sqrt_var_mean_af.rename({"IM": "IM_T"})
    covar_mean_af = cor_ref * sqrt_var_mean_af * sqrt_var_mean_af_T
    if config["s2s_mode"] == "aleatory":
        sqrt_mean_var_af = np.sqrt(mean_var_af)
        sqrt_mean_var_af_T = sqrt_mean_var_af.rename({"IM": "IM_T"})
        covar_var_af = cor_af * sqrt_mean_var_af * sqrt_mean_var_af_T
    else:
        covar_var_af = 0.0
    covar_af = covar_mean_af + covar_var_af
    covar_mean_ref_af_T = covar_mean_ref_af.rename({"IM": "IM_T"})
    covar_ref_af = cor_ref * (
        (sigma_ref_T / sigma_ref) * covar_mean_ref_af
        + (sigma_ref / sigma_ref_T) * covar_mean_ref_af_T
    )

    # total covariance
    covar_srf = covar_ref + covar_ref_af + covar_af

    return covar_srf


def extract_fcm_intensity_measures(mean, covar):
    PGA = "Sa[0.01]"
    lnPGA_mean = mean.sel(IM=PGA, drop=True)
    lnPGA_var = covar.sel(IM=PGA, IM_T=PGA, drop=True)
    lnSAavg_mean = mean.mean(dim="IM")
    lnSAavg_var = covar.mean(dim=["IM", "IM_T"])
    lnPGA_lnSAavg_covar = covar.sel(IM=PGA, drop=True).mean(dim="IM_T")

    full_mean = xr.concat([lnPGA_mean, lnSAavg_mean], dim="IM_FCM")
    full_covar = xr.concat(
        [
            xr.concat([lnPGA_var, lnPGA_lnSAavg_covar], dim="IM_FCM"),
            xr.concat([lnPGA_lnSAavg_covar, lnSAavg_var], dim="IM_FCM"),
        ],
        dim="IM_FCM_T",
    )

    moments = xr.Dataset({"mean": full_mean, "var": full_covar})
    moments = moments.assign_coords(
        {"IM_FCM": ["PGA", "SaAvg"], "IM_FCM_T": ["PGA", "SaAvg"]}
    )

    return moments


def discretize_normal_distribution(
    mean, var, x_range, dim, mode="pmf", shift=False, clip=True
):
    """
    Calculate probability mass centered on each node in x_range
    """
    # shift x_range to center the intervals
    if shift:
        spacing = x_range.isel({dim: 1}, drop=True) - x_range.isel({dim: 0}, drop=True)
    else:
        spacing = 0.0

    # calculate exceedence probabilities
    dist = xr.apply_ufunc(
        sf_numba_3,
        x_range - 0.5 * spacing,
        mean,
        var,
        dask="allowed",
    )

    if mode == "pmf":
        # squeeze all probabiliy below the lower bound on the first index
        # the same will also be done for the upper bound in the next step
        if clip:
            dist[{dim: 0}] = 1.0

        # determine difference between consecutive exceedence probabilities
        dist = tg.bin_diff(dist, dim, fill_value=0.0)

        if not clip:
            # remove the probability mass above the upper bound
            dist[{dim: -1}] = 0.0

    return dist


@vectorize([float64(float64, float64, float64)])
def sf_numba_3(a, mean, var):
    NPY_SQRT1_2 = 1.0 / np.sqrt(2 * var)
    x = (mean - a) * NPY_SQRT1_2
    z = fabs(x)

    if z < NPY_SQRT1_2:
        y = 0.5 + 0.5 * erf(x)
    else:
        y = 0.5 * erfc(z)
        if x > 0:
            y = 1.0 - y

    return y


if __name__ == "__main__":
    main(sys.argv)
