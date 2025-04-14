"""
Convert FCM coefficient files into xarray data structures

This script converts the original coefficient files for the fragility 
and consequence models of Crowley et al. into a xarray data structures,
exported in netCDF/HDF5 format. 

"""

import numpy as np
import pandas as pd
import yaml
import xarray as xr
from pathlib import Path

base_path = "./convert/res/"
output_file = "./FCM_config.h5"


def convert(base_path):
    # collect datasets in dictionary - later converted to DataTree
    output = {}

    # V5
    frag_files = {
        "Lower": Path(base_path) / "V5/NAM_V5_Fragility_LowerBranch_29-09-17.csv",
        "Middle": Path(base_path) / "V5/NAM_V5_Fragility_MiddleBranch_29-09-17.csv",
        "Upper": Path(base_path) / "V5/NAM_V5_Fragility_UpperBranch_29-09-17.csv",
    }
    cons_files = {
        "Lower": Path(base_path) / "V5/NAM_V5_Fatality_LowerBranch_29-09-17.csv",
        "Middle": Path(base_path) / "V5/NAM_V5_Fatality_MiddleBranch_29-09-17.csv",
        "Upper": Path(base_path) / "V5/NAM_V5_Fatality_UpperBranch_29-09-17.csv",
    }
    code_conversion_file = Path(base_path) / "V5/short_structural_code.yaml"

    # V5 specific conversion table for structural_types / vulnerability_classes
    with open(code_conversion_file, "r") as stream:
        try:
            convtable = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    label = "FCM-V5"
    ds5 = convert_csv_to_ds(frag_files, cons_files).assign_coords(
        {"fcm_version": label}
    )
    vul_cl = [convtable[a[1:-1]] for a in ds5["vulnerability_class"].values]
    ds5 = ds5.assign_coords({"vulnerability_class": vul_cl})
    output[label] = ds5

    # V6
    frag_files = {
        "Lower": Path(base_path) / "V6/Fragility_v6_LowerBranch_20190215.csv",
        "Middle": Path(base_path) / "V6/Fragility_v6_MiddleBranch_20190215.csv",
        "Upper": Path(base_path) / "V6/Fragility_v6_UpperBranch_20190215.csv",
    }
    cons_files = {
        "Lower": Path(base_path) / "V6/Fatality_v6_LowerBranch_20190215.csv",
        "Middle": Path(base_path) / "V6/Fatality_v6_MiddleBranch_20190215.csv",
        "Upper": Path(base_path) / "V6/Fatality_v6_UpperBranch_20190215.csv",
    }

    avgsa_periods = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]
    avgsa_labels = [f"Sa[{a}]" for a in avgsa_periods]

    label = "FCM-V6"
    ds6 = convert_csv_to_ds(frag_files, cons_files).assign_coords(
        {
            "fcm_version": label,
            "IM": avgsa_labels,
            "T": ("IM", avgsa_periods),
        }
    )
    output[label] = ds6
    # V7
    frag_files = {
        "Lower": Path(base_path) / "V7/Fragility_v7_LowerBranch_20200127.csv",
        "Middle": Path(base_path) / "V7/Fragility_v7_MiddleBranch_20200127.csv",
        "Upper": Path(base_path) / "V7/Fragility_v7_UpperBranch_20200127.csv",
    }
    cons_files = {
        "Lower": Path(base_path) / "V7/Fatality_v7_LowerBranch_20200127.csv",
        "Middle": Path(base_path) / "V7/Fatality_v7_MiddleBranch_20200127.csv",
        "Upper": Path(base_path) / "V7/Fatality_v7_UpperBranch_20200127.csv",
    }

    label = "FCM-V7"
    ds7 = convert_csv_to_ds(frag_files, cons_files).assign_coords(
        {
            "fcm_version": label,
            "IM": avgsa_labels,
            "T": ("IM", avgsa_periods),
        }
    )
    output[label] = ds7

    # TNO2020
    conversion_file = Path(base_path) / "TNO2020/conversie_v3.yaml"

    label = "FCM-TNO2020"
    dsTNO2020 = convert_to_TNO2020(ds6, ds7, conversion_file).assign_coords(
        {
            "fcm_version": label,
            "IM": avgsa_labels,
            "T": ("IM", avgsa_periods),
        }
    )
    output[label] = dsTNO2020

    # combine all
    datatree = xr.DataTree.from_dict(output)
    return datatree


def convert_to_TNO2020(ds6, ds7, conversion_file):
    with open(conversion_file, "r") as stream:
        try:
            conv_table = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    conv_info = (
        xr.DataArray(pd.DataFrame(conv_table))
        .to_dataset(dim="dim_0")
        .set_index(dim_1="namklasse")
        .rename(dim_1="vulnerability_class")
    )

    ds6, ds7, conv_info = xr.align(ds6, ds7, conv_info, join="right")

    fcm_tno = xr.where(conv_info["base"] == "v6", ds6, ds7)

    frag_tno = fcm_tno["fragility_parameters"]
    cons_tno = fcm_tno["probability_of_dying"]

    # adjust central branch
    frag_tno_midbranch = xr.where(
        conv_info["frag_up2mid"] == "no",
        frag_tno.sel(b_fragility="Middle"),
        frag_tno.sel(b_fragility="Upper"),
    )
    b0_mid = frag_tno_midbranch.sel(parameter_fragility="b0")
    b1_mid = frag_tno_midbranch.sel(parameter_fragility="b1")
    b0_shifted = xr.where(
        conv_info["frag_med_shift"] == "no", b0_mid, b0_mid - b1_mid * np.log(0.85)
    )

    # reconstuct fragility branches
    mult = xr.DataArray(
        [-1.73, 0.0, 1.73], 
        coords={"b_fragility": ["Lower", "Middle", "Upper"]},
    )
    b0_new = b0_shifted + mult * conv_info["modelonzekerheid"].astype(float)

    frag_tno.loc[{"parameter_fragility": "b0"}] = b0_new
    fcm_tno["fragility_parameters"] = frag_tno

    ratio = cons_tno.sel(b_consequence="Upper") / cons_tno.sel(b_consequence="Middle")
    cons_tno_shifted = cons_tno.shift({"b_consequence": -1})
    cons_tno_shifted.loc[{"b_consequence": "Upper"}] = (
        ratio * cons_tno_shifted.loc[{"b_consequence": "Middle"}]
    )
    cons_tno_new = xr.where(
        conv_info["conseq_up2mid"] == "no", cons_tno, cons_tno_shifted
    )
    fcm_tno["probability_of_dying"] = cons_tno_new

    # since the xr.where operations above broadcast everything to "vulnerability_class"
    # we want to remove that that superfluos dimension from the logic tree branch weights
    for v in fcm_tno:
        if v.startswith("w_"):
            fcm_tno[v] = ds7[v]

    return fcm_tno


def convert_csv_to_ds(frag_files, cons_files):
    # read original data files using pandas
    frag_df = {k: pd.read_csv(v, index_col=0) for k, v in frag_files.items()}
    cons_df = {k: pd.read_csv(v, index_col=0) for k, v in cons_files.items()}

    # create xarray data
    frag_xr = {br: create_frag_xr(br, df) for br, df in frag_df.items()}
    frag_limit_xr = {br: create_frag_limit_xr(br, df) for br, df in frag_df.items()}
    cons_pod_xr = {br: create_cons_pod_xr(br, df) for br, df in cons_df.items()}
    cons_chimney_xr = {br: create_cons_chimney_xr(br, df) for br, df in cons_df.items()}

    pod = xr.concat(cons_pod_xr.values(), "b_consequence").dropna("limit_state")

    # compile into dataset
    ds = xr.Dataset(
        {
            "fragility_parameters": xr.concat(frag_xr.values(), "b_fragility"),
            "displacement_limit": xr.concat(frag_limit_xr.values(), "b_fragility"),
            "consequence_parameters": xr.concat(
                cons_chimney_xr.values(), "b_consequence"
            ),
            "probability_of_dying": pod,
            "w_fragility": xr.DataArray(
                [0.17, 0.66, 0.17],
                coords={"b_fragility": ["Lower", "Middle", "Upper"]},
                attrs={
                    "support_dims": "b_fragility",
                    "distribution_type": "probability_mass",
                },
            ),
            "w_consequence": xr.DataArray(
                [0.25, 0.5, 0.25],
                coords={"b_consequence": ["Lower", "Middle", "Upper"]},
                attrs={
                    "support_dims": "b_consequence",
                    "distribution_type": "probability_mass",
                },
            ),
        }
    )

    return ds


# create consequence probability-of-dying dataset and reshape
# the "inside" and "outside" distinction is separated into a new dimension
# called "location"
def create_cons_pod_xr(branch_id, df):
    locations, states = np.array(
        [a[3:].split("|") for a in df.columns[0:6]]
    ).transpose()
    return (
        xr.DataArray(
            df.iloc[:, 0:6],
            dims=("vulnerability_class", "l_s"),
            coords={
                "vulnerability_class": df.index.values,
                "location": ("l_s", locations),
                "limit_state": ("l_s", states),
                "b_consequence": branch_id,
            },
        )
        .set_index(
            # reshaping step 1
            # create multi-index on the flattened dimensions
            {"l_s": ("location", "limit_state")}
        )
        .unstack(
            # reshaping step 2
            # and unflatten
            "l_s"
        )
    )


def create_cons_chimney_xr(branch_id, df):
    return xr.DataArray(
        df.iloc[:, 6:],
        dims=("vulnerability_class", "parameter_consequence"),
        coords={
            "vulnerability_class": df.index.values,
            "parameter_consequence": df.columns[6:],
            "b_consequence": branch_id,
        },
    )


def create_frag_limit_xr(branch_id, df):
    states = np.array([a[3:] for a in df.columns[7:13]])
    return xr.DataArray(
        df.iloc[:, 7:13],
        dims=("vulnerability_class", "limit_state"),
        coords={
            "vulnerability_class": df.index.values,
            "limit_state": states,
            "b_fragility": branch_id,
        },
    )


def create_frag_par_xr(branch_id, df):
    return xr.DataArray(
        df.iloc[:, 0:7],
        dims=("vulnerability_class", "parameter_fragility"),
        coords={
            "vulnerability_class": df.index.values,
            "parameter_fragility": df.columns[0:7],
            "b_fragility": branch_id,
        },
    )


def create_frag_chimney_xr(branch_id, df):
    return xr.DataArray(
        df.iloc[:, 13:],
        dims=("vulnerability_class", "parameter_fragility"),
        coords={
            "vulnerability_class": df.index.values,
            "parameter_fragility": df.columns[13:],
            "b_fragility": branch_id,
        },
    )


def create_frag_xr(branch_id, df):
    return xr.concat(
        (create_frag_par_xr(branch_id, df), create_frag_chimney_xr(branch_id, df)),
        "parameter_fragility",
    )


if __name__ == "__main__":
    ds = convert(base_path)
    ds.to_netcdf(output_file, engine="h5netcdf")
