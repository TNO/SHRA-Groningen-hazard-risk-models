"""
Generate intensity measure prep, i.e. the ground motion intensity in terms
of the intensity measures use in the fragility and consequence models.
"""

import sys
import logging
import timeit
import numpy as np
import xarray as xr
import xarray_einstats.stats as xe_st
import xarray_einstats.numba as xenum
import dask.array as da
from tqdm import tqdm

from hr_models import gmm_V5V6, gmm_V7
from chaintools.chaintools.tools_configuration import preamble
from chaintools.chaintools import tools_xarray as tx


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
    config.setdefault("store_epsilon", False)
    config.setdefault("rng_seed", 42)
    config.setdefault("n_sample", 1_000)
    config.setdefault("n_batch", 1)
    config.setdefault("independent_dimensions", ["magnitude", "distance_rupture"])
    config.setdefault("file_mode", "w-")


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
    # open gmm configuration and tabular data
    fcm_config = tx.open("fcm_config", config)
    gmm_config = tx.open("gmm_config", config)
    gmm_tables = tx.open("gmm_tables", config)
    weights = tx.open("weights", config)

    # assign default configuration values
    assign_defaults(config, gmm_tables)

    # preprocessing
    if weights is None:
        weights = xr.Dataset()
    gmm_config, gmm_tables = preprocess(config, fcm_config, gmm_config, gmm_tables)

    # prepare output
    output_ds = tx.prepare_ds(config)
    SA_ref = output_ds["SA_reference"]
    SA_srf = output_ds["SA_surface"]

    # treatment of the rate multiplier
    if "b_median" in weights.dims:
        rate_multiplier = gmm_tables["rate_multiplier"]
    else:
        rate_multiplier = xr.DataArray(1.0)
        # if not used here, carry through
        output_ds["rate_multiplier"] = gmm_tables["rate_multiplier"]
    tx.store(output_ds, "output", config, mode=config["file_mode"])

    # set up the dask random number generator and prepare samples
    rng = da.random.default_rng(config["rng_seed"])
    epsilon = generate_epsilon(gmm_tables, gmm_config, config, rng)

    # because of the huge dimensions of the data we allow ourselves a
    # for loop over the zones, exploiting the append facility of xarray and zarr
    storage_kwargs = {"mode": "a"}
    logging.info("generating ground motion intensity distributions")
    for zone in tqdm(gmm_tables["zone"].data, desc="zone"):
        tab = gmm_tables.sel({"zone": [zone]})
        conf = gmm_config.sel({"zone": [zone]})

        # generate ground motion realizations based on random sampling
        gm_ref, gm_srf = gm_realization(tab, conf, epsilon)

        # extract and process histograms
        output_ds = xr.Dataset()
        output_ds["reference_pmf"] = calculate_histogram(
            gm_ref, weights, rate_multiplier, SA_ref
        )
        output_ds["surface_pmf"] = calculate_histogram(
            gm_srf, weights, rate_multiplier, SA_srf
        )

        # store
        tx.store(output_ds, "output", config, **storage_kwargs)

        # prepare for next iteration
        storage_kwargs["append_dim"] = "zone"


def preprocess(config, fcm_config, gmm_config, gmm_tables):
    if config["s2s_mode"] == "aleatory":
        gmm_tables = gmm_tables.drop_dims("b_s2s")

    # select the spectral periods that are of interest to the FCM
    # note that we assume here that all are present and
    # we don't have to interpolate
    im_selection = {}
    im_selection["IM"] = fcm_config["IM"].data
    gmm_tables = gmm_tables.sel(im_selection)
    im_selection["IM_T"] = fcm_config["IM"].data
    gmm_config = gmm_config.sel(im_selection)

    return gmm_config, gmm_tables


def calculate_histogram(gm, weights, rate_multiplier, gm_nodes):
    # extract intensity measures
    gm = extract_IMs(gm)

    # calculate logictree histogram/pmf at surface level
    hist = gm_histogram(gm, gm_nodes).drop_vars(["left_edges", "right_edges"])

    weights, marginalize_dims = tx.prepare_weights(weights, hist)
    if "b_median" not in weights.dims:
        rate_multiplier = xr.DataArray(1.0)

    # marginalize over the core dimensions of the provided weights
    hist = xr.dot(
        hist,
        rate_multiplier,
        *weights.values(),
        dim=marginalize_dims,
        optimize=True,
    )

    return hist


def gm_histogram(lnIM, Sa_range, sample_dim=None, batch_dim=None):
    if sample_dim is None:
        sample_dim = "__sample__"
    if batch_dim is None:
        batch_dim = "__batch__"

    # determine histogram bins
    lnSa_range = np.log(Sa_range)
    range_dim = lnSa_range.dims[0]

    # these values are the histogram bin centers, so we should
    # determine the bin edges by shifting and averaging
    # finally we add -inf and inf to the bin edges, extending the
    # length by one
    lnSa_range = 0.5 * (lnSa_range + lnSa_range.shift({range_dim: 1}))
    lnSa_range_np = lnSa_range.values
    lnSa_range_np[0] = -np.inf
    lnSa_range_np = np.append(lnSa_range_np, [np.inf])
    n_samples = lnIM.sizes[sample_dim]

    lnIM_hist = (
        xenum.histogram(
            lnIM,
            bins=lnSa_range_np,
            dims=[sample_dim],
            dask="parallelized",
            output_dtypes=[float],
        )
        .rename({"bin": range_dim})
        .assign_coords(lnSa_range.coords)
        .mean(batch_dim)
        / n_samples
    )

    return lnIM_hist


def gm_realization(gmm_tables, gmm_config, epsilon):
    # identify model
    gmm_version = gmm_tables["gmm_version"].data[()]
    if gmm_version in ["GMM-V5", "GMM-V6"]:
        gmm_package = gmm_V5V6
    else:
        gmm_package = gmm_V7

    # construct reference realization
    sigma = np.sqrt(gmm_tables["reference_ac_variance"])
    realization_ref = gmm_tables["reference_median"] + epsilon["epsilon_ref"] * sigma

    # determine af parameters conditional on reference realization
    realization_af = xr.apply_ufunc(
        gmm_package.af_realization,
        gmm_tables["distance_rupture"],
        gmm_tables["magnitude"],
        realization_ref,
        epsilon["epsilon_af"],
        gmm_config["af_parameters"],
        kwargs={"par_id": gmm_config["parameter_af"].values},
        input_core_dims=[[], [], [], [], ["parameter_af"]],
        exclude_dims=set(("parameter_af",)),
        dask="parallelized",
        output_dtypes=[float],
    )

    # the complete realization at the surface
    wierde_factor = gmm_tables.get("wierde_factor", 0.0)
    realization_srf = realization_ref + realization_af + wierde_factor

    return realization_ref, realization_srf


def extract_IMs(gm_realization):
    lnSaAvg = gm_realization.mean("IM")
    lnPGA = gm_realization.sel({"IM": "Sa[0.01]"}, drop=True)
    lnIM = xr.concat(
        [lnSaAvg, lnPGA], dim=xr.DataArray(["SaAvg", "PGA"], dims="IM_FCM")
    )
    return lnIM


def generate_epsilon(
    gmm_tables, gmm_config, config, rng, sample_dim=None, batch_dim=None
):
    if sample_dim is None:
        sample_dim = "__sample__"
    if batch_dim is None:
        batch_dim = "__batch__"

    # treatment of period-to-period correlation in aleatory s2s variability
    # alternatives: ["zero", "consistent", "full"]
    # GMM-V5/GMM-V6 prescription: "zero"
    # TNO recommendation: "consistent"
    # GMM-V7 implicit prescriptionin epistemic treatment: "full"
    s2s_mode = config["s2s_mode"]
    s2s_p2p_mode = config["s2s_p2p_mode"]

    im_dims = ["IM", "IM_T"]
    dims, coords, sizes, chunks = epsilon_metadata(
        gmm_tables, config, im_dims[0], sample_dim, batch_dim
    )

    correlation_matrix = gmm_config["correlation_matrix"].load()

    # generated correlated samples
    epsilon_ref = get_random_samples(
        correlation_matrix,
        dims,
        sizes,
        chunks,
        im_dims,
        p2p_alternatives=False,
        rng=rng,
    )

    if s2s_mode == "epistemic":
        # on the "b_s2s" dimension, 3 samples
        epsilon_af = gmm_tables["s2s_epsilons"]
    elif s2s_mode == "aleatory":
        # on the "__sample__" dimension
        # if desired we also get a p2p dimensions
        epsilon_af = get_random_samples(
            correlation_matrix,
            dims,
            sizes,
            chunks,
            im_dims,
            p2p_alternatives=True,
            rng=rng,
        ).sel({"s2s_p2p_mode": s2s_p2p_mode})
    else:
        raise ValueError(f"Unknown s2s_mode: {s2s_mode}")

    ds = xr.Dataset(
        {"epsilon_ref": epsilon_ref, "epsilon_af": epsilon_af}
    ).assign_coords(coords)

    return ds


def epsilon_metadata(ds, config, im_dim, sample_dim, batch_dim):
    independent_dimensions = config["independent_dimensions"]
    dimensions = independent_dimensions + [im_dim]
    coords = ds[dimensions].coords
    sizes = tuple(coords.sizes.values())

    n_samples = config["n_sample"]
    n_batch = config["n_batch"]
    sizes = sizes + (n_batch, n_samples)
    dimensions = dimensions + [batch_dim, sample_dim]

    chunk_spec = config.get("chunk", {})
    chunk_spec[batch_dim] = 1
    chunk_spec[sample_dim] = -1
    chunks = tuple(chunk_spec.get(dim, -1) for dim in dimensions)

    return dimensions, coords, sizes, chunks


def get_cholesky_L(correlation_matrix, im_dims=None):
    if im_dims is None:
        im_dims = ["IM", "IM_T"]

    # cholesky for the reference samples
    cholesky_L = xe_st.cholesky(correlation_matrix, dims=im_dims)

    return cholesky_L


def get_random_samples(
    correlation_matrix, dims, sizes, chunks, im_dims, p2p_alternatives, rng
):
    uncorrelated_samples = xr.DataArray(
        rng.standard_normal(size=sizes, chunks=chunks), dims=dims
    )

    # cholesky procedure to get correlated samples
    cholesky_L = get_cholesky_L(correlation_matrix, im_dims)
    IM, IM_T = im_dims
    correlated_samples = xr.dot(
        cholesky_L, uncorrelated_samples.rename({IM: IM_T}), dim=[IM_T]
    )

    if p2p_alternatives:
        # return all alternatives, later select what is needed
        us = uncorrelated_samples
        cs = correlated_samples
        fs = uncorrelated_samples.isel({IM: 0}, drop=True)
        bc = xr.broadcast(us, cs, fs)
        samples = xr.concat(bc, dim="s2s_p2p_mode")
        samples = samples.assign_coords(
            {"s2s_p2p_mode": ["zero", "consistent", "full"]}
        )
    else:
        samples = correlated_samples

    return samples


if __name__ == "__main__":
    main(sys.argv)
