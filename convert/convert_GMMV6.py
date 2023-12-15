import os
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

base_path = "./convert/res/V6/"
output_file = "./GMM_config.h5"


def convert(base_path):
    # set input and output file names
    median_file = Path(base_path) / "gmpe_medians_NS_B_20190314_v6.csv"
    sigma_file = Path(base_path) / "gmpe_sigmas_NS_B_20190325_v6.csv"
    af_file = Path(base_path) / "gmpeSurfaceAmplificationModel_20190326_v6.csv"
    p2p_file = Path(base_path) / "gmpe_period2period_correlations_20190325_v6.csv"

    # read original data files
    median_df = pd.read_csv(median_file, index_col=0)
    sigma_df = pd.read_csv(sigma_file, index_col=0)
    p2p_df = pd.read_csv(p2p_file, index_col=0)
    af_df = pd.read_csv(af_file, index_col=(0, 1))

    # disentangle colummn names for reshaping
    median_parameters, median_levels = np.array(
        [a.split("_") for a in median_df.columns]
    ).transpose()
    tau_levels = sigma_df.columns[0:4]
    phiss_levels = sigma_df.columns[4:]

    # apply customary branch names
    median_branch_name_mapping = {
        "level1": "Lower",
        "level2": "CentralLower",
        "level3": "CentralUpper",
        "level4": "Upper",
    }
    median_branches = [median_branch_name_mapping[k] for k in median_levels]

    tau_branch_name_mapping = {
        "tau_level1": "Lower",
        "tau_level2": "CentralLower",
        "tau_level3": "CentralUpper",
        "tau_level4": "Upper",
    }
    tau_branches = [tau_branch_name_mapping[k] for k in tau_levels]

    phiss_branch_name_mapping = {"phi_ss_level1": "Lower", "phi_ss_level2": "Upper"}
    phiss_branches = [phiss_branch_name_mapping[k] for k in phiss_levels]

    # create parameter dataset and reshape
    # xarray reshaping procedure obtained from
    # https://stackoverflow.com/questions/62592803/xarray-equivalent-of-np-reshape
    # generate id's for intensity measures
    im_ids_median = list(map(lambda p: f"Sa[{p}]", median_df.index))
    im_ids_median[0] = "PGV"
    median_p = (
        xr.DataArray(
            median_df,
            # name = "gmm_median_parameters",
            dims=("IM", "p_b"),
            coords={
                "IM": im_ids_median,
                "parameter_median": ("p_b", median_parameters),
                "branch_median": ("p_b", median_branches),
            },
            attrs={
                # assemble any data that may be of interest
                "source": [os.path.basename(median_file)]
            },
        )
        .set_index(
            # reshaping step 1
            # create multi-index on the flattened dimensions
            {"p_b": ("parameter_median", "branch_median")}
        )
        .unstack(
            # reshaping step 2
            # and unflatten
            "p_b"
        )
    )

    af_p_tmp = (
        xr.DataArray(
            af_df,
            dims=("z_t", "parameter_af"),
            attrs={
                # assemble any data that may be of interest
                "source": [os.path.basename(af_file)]
            },
        )
        .unstack("z_t")
        .rename({"Zone": "zone", "T": "IM"})
    )

    im_ids_af = list(map(lambda p: f"Sa[{float(p)}]", af_p_tmp["IM"].values))
    im_ids_af[0] = "PGV"
    zone_ids = list(map(str, af_p_tmp["zone"].values))
    af_p_tmp = af_p_tmp.assign_coords({"zone": zone_ids, "IM": im_ids_af})

    # add extra column/parameter_af to take care of scaling from cm/s2 to g
    af_scale = xr.full_like(af_p_tmp["IM"], 981.0, dtype=float).assign_coords(
        {"parameter_af": "AFscale"}
    )
    af_p = xr.concat((af_p_tmp, af_scale), "parameter_af")
    af_p.loc[{"IM": "PGV", "parameter_af": "AFscale"}] = 1.0

    # xarray does not seem to support the same coordinates on more than 1 dimension
    im_ids_p2p = list(map(lambda p: f"Sa[{float(p)}]", p2p_df.index))
    im_ids_p2p2 = list(map(lambda p: f"Sa[{float(p)}]", p2p_df.columns))
    corr_mat = xr.DataArray(
        p2p_df.values,
        dims=("IM", "IM_T"),
        coords={"IM": im_ids_p2p, "IM_T": im_ids_p2p2},
        attrs={"source": [os.path.basename(p2p_file)]},
    )

    im_ids_sigma = list(map(lambda p: f"Sa[{float(p)}]", sigma_df.index))
    im_ids_sigma[0] = "PGV"
    tau = xr.DataArray(
        sigma_df.iloc[:, 0:4],
        dims=("IM", "branch_tau"),
        coords={"IM": im_ids_sigma, "branch_tau": tau_branches},
        attrs={"source": [os.path.basename(sigma_file)]},
    )

    phiss = xr.DataArray(
        sigma_df.iloc[:, 4:],
        dims=("IM", "branch_phiss"),
        coords={"IM": im_ids_sigma, "branch_phiss": phiss_branches},
        attrs={"source": [os.path.basename(sigma_file)]},
    )

    t = xr.DataArray(
        # extract spectral periods
        sigma_df.index.values[1:],
        dims=("IM",),
        coords={"IM": im_ids_sigma[1:]},
    )

    # prepare wierde -- adopted from GMM-V7
    wierde_factor = (
        xr.DataArray(
            data=np.array(
                [[0.2, 0.25, 0.35, 0.35, 0.1, 0.05], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
            ),
            dims=["surface_condition", "T"],
            coords={
                "surface_condition": ["wierde", "regular"],
                "T": [0.01, 0.1, 0.2, 0.5, 1.0, 2.0],
            },
        )
        .interp({"T": t}, method="linear")
        .fillna(0.0)  # assume 0.0 for periods outside of interpolation range
    )

    # s2s branches -- adopted from GMM-V7
    s2s_epsilons = xr.DataArray(
        np.array([-1.645, 0.0, 1.645]),
        coords={"branch_s2s": ["Lower", "Central", "Upper"]},
    )

    lw_branch_median_tau_weights = xr.DataArray(
        [
            [0.1, 0.0, 0.0, 0.0],
            [0.0, 0.3, 0.0, 0.0],
            [0.0, 0.0, 0.3, 0.0],
            [0.0, 0.0, 0.0, 0.3],
        ],
        coords={
            "branch_median": ["Lower", "CentralLower", "CentralUpper", "Upper"],
            "branch_tau": ["Lower", "CentralLower", "CentralUpper", "Upper"],
        },
    )

    lw_branch_phiss_weights = xr.DataArray(
        [0.5, 0.5], coords={"branch_phiss": ["Lower", "Upper"]}
    )

    lw_branch_s2s_weights = xr.DataArray(
        [0.2, 0.6, 0.2], coords={"branch_s2s": ["Lower", "Central", "Upper"]}
    )

    # combine into single dataset
    ds = (
        xr.Dataset(
            data_vars={
                "af_parameters": af_p,
                "median_parameters": median_p,
                "correlation_matrix": corr_mat,
                "tau": tau,
                "phiss": phiss,
                "T": t,
                "wierde_factor": wierde_factor,
                "s2s_epsilons": s2s_epsilons,
                "logic_tree:branch_median_tau": lw_branch_median_tau_weights,
                "logic_tree:branch_phiss": lw_branch_phiss_weights,
                "logic_tree:branch_s2s": lw_branch_s2s_weights,
            },
            coords={"gmm_version": "GMM-V6"},
            attrs={"reference": "Bommer et al. (2019)"},
        )
        .transpose("zone", "branch_median", "branch_tau", "branch_phiss", "IM", ...)
        .sortby("T")
    )

    return ds


if __name__ == "__main__":
    ds = convert(base_path)
    ds.to_netcdf(output_file, group="GMM-V6", mode="a", engine="h5netcdf")
