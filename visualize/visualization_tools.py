import matplotlib.pyplot as plt
import numpy as np
from matplotlib import path as mpath
import matplotlib.patches as patches
import os
import pandas as pd
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray as xr

map_xlim = [225000, 275000]
map_ylim = [560000, 614000]


def plot_coast(ref="plt", color="grey", lw=1):

    files = ["res/kust.csv"]
    res_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))

    for file in files:
        path = os.path.join(res_dir, file)
        outline = np.genfromtxt(path, names=True, delimiter=",")
        if ref == "plt":
            plt.plot(outline["x"], outline["y"], c=color, lw=lw, zorder=4)
        else:
            ref.plot(outline["x"], outline["y"], c=color, lw=lw, zorder=4)


def plot_cities(ref="plt", symbol="s", s=15, color="grey", fontsize=8, add_name=True):
    source_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "res/plaatsen.csv"
    )
    cities = np.genfromtxt(source_path, names=True, delimiter=",", dtype="f8,f8,U50")
    for city in cities:
        x_offset = 3000
        y_offset = 3000
        if city["plaats"] == "Delfzijl":
            x_offset = 6500
            y_offset = -2000
        if city["plaats"] == "Hoogezand":
            y_offset = -2000
        if city["plaats"] == "Ten Boer":
            x_offset = -500
        if city["plaats"] == "Groningen":
            x_offset = 8000
        if city["plaats"] == "Winschoten":
            x_offset = 8000
        if city["plaats"] == "Loppersum":
            x_offset = 7000
        if ref == "plt":
            plt.scatter(city["x"], city["y"], marker=symbol, s=s, color=color, zorder=5)
            if add_name:
                plt.text(
                    city["x"] - x_offset,
                    city["y"] - y_offset,
                    city["plaats"],
                    color=color,
                    fontsize=fontsize,
                    zorder=5,
                )
        else:
            ref.scatter(city["x"], city["y"], marker=symbol, s=s, color=color)
            if add_name:
                ref.text(
                    city["x"] - x_offset,
                    city["y"] - y_offset,
                    city["plaats"],
                    color=color,
                    fontsize=fontsize,
                    zorder=5,
                )


def plot_outline(ref="plt", color="k", lw=2):
    source_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "res/Groningen_field_outline.csv"
    )
    outline = np.genfromtxt(source_path, names=True, delimiter=",")
    if ref == "plt":
        plt.plot(outline["x"], outline["y"], c=color, lw=lw, zorder=6)
    else:
        ref.plot(outline["x"], outline["y"], c=color, lw=lw, zorder=6)


def create_zonation_geometry_from_data_structure(hazard_data, zones):

    relevant_zones = np.unique(hazard_data.zone)
    zone_geoms = {}
    for zone in relevant_zones:
        zone_polygon = zones.loc[zone].geometry
        if zone_polygon.geom_type == "Polygon":
            polys = [get_coords_from_poly(zone_polygon)]
        else:
            polys = [get_coords_from_poly(a) for a in zone_polygon.geoms]

        zone_geoms[zone] = {"polygons": polys}

    return zone_geoms


def get_coords_from_poly(polygon):
    xx, yy = polygon.exterior.coords.xy
    xx, yy = np.array(xx.tolist()), np.array(yy.tolist())
    coords = np.hstack([xx[:, None], yy[:, None]])
    return coords


def ring_coding(coords):
    # code used from
    # https://sgillies.net/2010/04/06/painting-punctured-polygons-with-matplotlib.html
    # The codes will be all "LINETO" commands, except for "MOVETO"s at the
    # beginning of each subpath
    n = len(coords)
    codes = np.ones(n, dtype=mpath.Path.code_type) * mpath.Path.LINETO
    codes[0] = mpath.Path.MOVETO
    return codes


def pathify(polygon_collection):
    # code used from
    # https://sgillies.net/2010/04/06/painting-punctured-polygons-with-matplotlib.html
    # Convert coordinates to path vertices. Objects produced by Shapely's
    # analytic methods have the proper coordinate order, no need to sort.
    vertices = np.concatenate([np.asarray(pol) for pol in polygon_collection])
    codes = np.concatenate([ring_coding(pol) for pol in polygon_collection])
    return mpath.Path(vertices, codes)


def add_zone_plot(
    ax,
    zone_property,
    zone_geom,
    vmin,
    vmax,
    cmap,
    show_outline=False,
    clip_outline=True,
    **kwargs,
):

    polygon_path = pathify(zone_geom["polygons"])
    polygon_patch = patches.PathPatch(polygon_path, fc="none", ec="none")
    # ax.scatter(zone_hazard.x, zone_hazard.y,c=zone_hazard, s=1, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
    zone_property = zone_property.unstack("zone_x_y")
    zone_property = zone_property.sortby(zone_property.x)
    zone_property = zone_property.sortby(zone_property.y)

    im = (
        zone_property.transpose("y", "x")
        .fillna(0.0)
        .plot.imshow(
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            interpolation="bilinear",
            add_colorbar=False,
            **kwargs,
        )
    )

    if show_outline:
        for i in range(len(zone_geom["polygons"])):
            x, y = zone_geom["polygons"][i][:, 0], zone_geom["polygons"][i][:, 1]
            ax.plot(x, y, c="k")
    if clip_outline:
        ax.add_patch(polygon_patch)
        im.set_clip_path(polygon_patch)


def find_extremes_per_zone(hazard, geometry):

    polygon_path = pathify(geometry["polygons"])
    x_y_flat = np.concatenate([a for a in geometry["polygons"]], axis=0)
    hazard_unstacked = hazard.unstack("zone_x_y")

    # usually the extremes are found at the boundary
    # we choose not to subsample the zone boundaries and /or do line searches
    # instead we accept the finite granularity of the chosen discretization
    edge_vals = np.concatenate(
        [hazard_unstacked.interp(x=x, y=y).values[:, None] for x, y in x_y_flat],
        axis=-1,
    ).T
    min_ind, max_ind = np.argmin(edge_vals, axis=0), np.argmax(edge_vals, axis=0)
    edge_min_val, edge_max_val = np.diag(edge_vals[min_ind]), np.diag(
        edge_vals[max_ind]
    )
    edge_min_loc, edge_max_loc = x_y_flat[min_ind], x_y_flat[max_ind]

    # sometimes the extreme is a local extreme at the inside of a polygon
    # assuming (bi) linear interpolation this can only happen at the inner grid points
    # it may happen that for a small zone all grid points are outside of the polygon
    hazard_x_grid, hazard_y_grid = np.meshgrid(
        hazard_unstacked.x, hazard_unstacked.y, indexing="ij"
    )
    x_flat, y_flat = hazard_x_grid.flatten(), hazard_y_grid.flatten()
    hazard_xy = np.concatenate([x_flat[:, None], y_flat[:, None]], axis=-1)
    inside = polygon_path.contains_points(hazard_xy)
    hazard_inside = hazard_xy[inside]

    try:
        grid_vals = np.concatenate(
            [
                hazard_unstacked.interp(x=x, y=y).values[:, None]
                for x, y in hazard_inside
            ],
            axis=-1,
        ).T
        min_ind, max_ind = np.argmin(grid_vals, axis=0), np.argmax(grid_vals, axis=0)
        grid_min_val, grid_max_val = np.diag(grid_vals[min_ind]), np.diag(
            grid_vals[max_ind]
        )
        grid_min_loc, grid_max_loc = x_y_flat[min_ind], x_y_flat[max_ind]
    except ValueError:
        grid_min_val, grid_max_val = np.full_like(
            edge_min_val, fill_value=np.inf
        ), np.full_like(edge_max_val, -np.inf)
        grid_min_loc = grid_max_loc = [0, 0]

    min_val, max_val, min_loc, max_loc = (
        np.zeros_like(edge_min_val),
        np.zeros_like(edge_max_val),
        np.zeros_like(edge_min_loc),
        np.zeros_like(edge_max_loc),
    )
    for i in range(len(min_val)):
        if edge_min_val[i] < grid_min_val[i]:
            min_val[i] = edge_min_val[i]
            min_loc[i] = edge_min_loc[i]
        else:
            min_val[i] = grid_min_val[i]
            min_loc[i] = grid_min_loc[i]

        if edge_max_val[i] > grid_max_val[i]:
            max_val[i] = edge_max_val[i]
            max_loc[i] = edge_max_loc[i]
        else:
            max_val[i] = grid_max_val[i]
            max_loc[i] = grid_max_loc[i]

    return min_val, max_val, min_loc, max_loc


def get_global_min_max(hazard, zone_geometries, select_zones):
    glob_min, glob_max = [], []
    min_loc, max_loc = [], []
    for z in select_zones:
        zone_min, zone_max, zone_min_loc, zone_max_loc = find_extremes_per_zone(
            hazard.sel(zone=z), zone_geometries[z]
        )
        glob_min.append(zone_min)
        glob_max.append(zone_max)
        min_loc.append(zone_min_loc)
        max_loc.append(zone_max_loc)

    min_vals = np.concatenate([m[..., None] for m in glob_min], axis=-1)
    min_ind = min_vals.argmin(axis=1)
    max_vals = np.concatenate([m[..., None] for m in glob_max], axis=-1)
    max_ind = max_vals.argmax(axis=1)

    min_return = np.diag(min_vals[:, min_ind])
    max_return = np.diag(max_vals[:, max_ind])

    min_loc = np.concatenate([m[..., None] for m in min_loc], axis=-1)
    max_loc = np.concatenate([m[..., None] for m in max_loc], axis=-1)

    min_loc_return = min_loc[..., min_ind][
        np.arange(len(min_return)), :, np.arange(len(min_return))
    ]
    max_loc_return = max_loc[..., max_ind][
        np.arange(len(max_return)), :, np.arange(len(max_return))
    ]

    return min_return, max_return, min_loc_return, max_loc_return


def subplot_geometry(n):
    if n == 1:
        rows, columns = 1, 1
    elif n == 2:
        rows, columns = 1, 2
    elif n <= 4:
        rows, columns = 2, 2
    elif n <= 6:
        rows, columns = 3, 2
    elif n <= 9:
        rows, columns = 3, 3
    elif n <= 12:
        rows, columns = 4, 3
    elif n <= 15:
        rows, columns = 5, 3
    elif n <= 20:
        rows, columns = 5, 4
    elif n <= 25:
        rows, columns = 5, 5
    elif n <= 30:
        rows, columns = 6, 5
    elif n <= 35:
        rows, columns = 7, 5
    elif n <= 40:
        rows, columns = 8, 5
    else:
        raise ValueError("too many items for subplot configuration")
    return rows, columns


def plot_hazard_maps(
    hazard,
    zones,
    fig_path=None,
    return_period=475,
    IM="Sa[0.01]",
    surface_condition="regular",
    select_time=None,
    select_zones=None,
    show_max=False,
    cmap="hot_r",
    vmin=None,
    vmax=None,
):
    zone_geometries = create_zonation_geometry_from_data_structure(hazard, zones)
    selection = hazard["surface_poe-[lt-mean]-[return_periods]"].sel(
        return_periods=return_period, IM=IM, surface_condition=surface_condition
    )
    # If the selection has more dimensions than 'time' and 'zone_x_y' we currently don't support further selection
    # We will just take the first entry along each other dim
    for dim in selection.dims:
        if dim not in ["time", "zone_x_y"]:
            print(f'Selecting "{selection[dim][0].item()}" on dimension "{dim}"')
            selection = selection.isel({dim: 0})

    if not select_zones:
        select_zones = np.unique([z.item() for z in selection.zone])
    if not select_time:
        select_time = selection.time

    times = pd.to_datetime(selection.time)
    if times[0].month == 10:
        # Input is in gasyears
        times_for_name = [f"GY{t.year}" for t in times]
        times = [f"GY{t.year}/{t.year+1}" for t in times]
    else:
        times = times_for_name = [f"{t.year}" for t in times]

    if show_max:
        print("Determining max value, this takes a couple of minutes")
        min_val, max_val, _, loc_max = get_global_min_max(
            selection.sel(time=select_time), zone_geometries, select_zones
        )
        print("Done, continue to plotting")
        vmin, vmax = np.min(min_val), np.max(max_val)
        vmin = 0  # Overwrite
    else:
        loc_max = None
        if vmax is None:
            vmax = 0.15
        if vmin is None:
            vmin = 0

    # We always create both an overview plot AND individual plots
    name_postfix = ""
    if selection.component.item() != "geometric_mean":
        name_postfix += f"_{selection.component.item()}"
    if selection.surface_condition.item() != "regular":
        name_postfix += f"_{selection.surface_condition.item()}"
    name_postfix += ".png"

    rows, columns = subplot_geometry(len(select_time))
    overview_fig = plt.figure(num=0, figsize=(columns + 2, rows + 2))

    for i, t in enumerate(select_time):
        year = times[i]
        temporal_selection = selection.sel(time=t)

        # Plot in separate figure and save
        plt.figure(num=i + 1)
        ax = plt.gca()
        for z in select_zones:
            add_zone_plot(
                ax,
                temporal_selection.sel(zone=z),
                zone_geometries[z],
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
        plot_cities(add_name=True)
        plot_coast()
        plot_outline()
        plt.xlim(map_xlim)
        plt.ylim(map_ylim)
        plt.title(
            f"{year} {selection.IM.item()} return period: {int(selection.return_periods.item())} years",
            fontdict={"fontsize": 8},
        )
        if show_max:
            xmax, ymax = loc_max[i]
            plt.scatter(
                xmax, ymax, c="blue", s=10, label=(f"Maximum: {max_val[i]:.2g}")
            )
            plt.legend(loc=3, frameon=False, prop={"size": 8})
        ax.set_aspect("equal")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "10 %", pad="10 %")
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
        cb1.set_label("acceleration (g)")

        name = (
            f"hazard_{times_for_name[i]}_{selection.IM.item()}_{int(selection.return_periods.item())}"
            + name_postfix
        )
        if fig_path is not None:
            plt.savefig(os.path.join(fig_path, name), dpi=300)
            plt.close()

        # Plot in the overview figure
        plt.figure(num=0)
        ax = plt.subplot(rows, columns, i + 1, frame_on=False, xticks=[], yticks=[])
        for z in select_zones:
            add_zone_plot(
                ax,
                temporal_selection.sel(zone=z),
                zone_geometries[z],
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
        plot_cities(s=12 / float(columns), add_name=False)
        plot_coast(lw=2 / float(columns))
        plot_outline(lw=2 / float(columns))
        plt.xlabel("")
        plt.ylabel("")
        plt.xlim(map_xlim)
        plt.ylim(map_ylim)
        plt.title(year, fontdict={"fontsize": 8})
        if show_max:
            xmax, ymax = loc_max[i]
            plt.scatter(
                xmax,
                ymax,
                c="blue",
                s=10 / float(columns),
                label=(f"Maximum: {max_val[i]:.2g}"),
            )
            plt.legend(loc=3, frameon=False, prop={"size": 18 / float(columns)})
        ax.set_aspect("equal")

    # Finish overview figure
    # add overview map if there's space
    if i + 2 < rows * columns:
        axf = plt.subplot(rows, columns, i + 2, frame_on=False, xticks=[], yticks=[])
        axf.set_aspect("equal")
        plot_coast(lw=2 / float(columns))
        plot_outline(lw=2 / float(columns))
        plot_cities(s=12 / float(columns), fontsize=12 / float(columns), color="k")
        plt.xlim(map_xlim)
        plt.ylim(map_ylim)
    plt.suptitle(
        f"{selection.IM.item()} return period: {int(selection.return_periods.item())} years"
    )
    overview_fig.subplots_adjust(right=0.75)
    cbar_ax = overview_fig.add_axes([0.78, 0.2, 0.02, 0.6])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb1 = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm)
    cb1.set_label("acceleration (g)", fontsize=7)
    cb1.ax.tick_params(labelsize=6)
    name = (
        f"hazard_overview_{selection.IM.item()}_{int(selection.return_periods.item())}"
        + name_postfix
    )
    if fig_path is not None:
        plt.savefig(os.path.join(fig_path, name), dpi=300)
        plt.close()


def plot_risk_maps(
    risk,
    zones,
    fig_path=None,
    vulnerability_class="URM1F_B",
    surface_condition="regular",
    select_time=None,
    select_zones=None,
    show_max=False,
    vmin=None,
    vmax=None,
):
    zone_geometries = create_zonation_geometry_from_data_structure(risk, zones)
    selection = risk["LPR-[lt-mean]"].sel(
        vulnerability_class=vulnerability_class, surface_condition=surface_condition
    )

    if not select_zones:
        select_zones = np.unique([z.item() for z in selection.zone])
    if not select_time:
        select_time = selection.time

    times = pd.to_datetime(selection.time)
    if times[0].month == 10:
        # Input is in gasyears
        times_for_name = [f"GY{t.year}" for t in times]
        times = [f"GY{t.year}/{t.year+1}" for t in times]
    else:
        times = times_for_name = [f"{t.year}" for t in times]

    if show_max:
        print("Determining max value, this takes a couple of minutes")
        min_val, max_val, _, loc_max = get_global_min_max(
            selection.sel(time=select_time), zone_geometries, select_zones
        )
        print("Done, continue to plotting")
        vmin, vmax = np.min(min_val), np.max(max_val)
    else:
        loc_max = None
        vmin, vmax = 1e-7, 1e-5

    # We always create both an overview plot AND individual plots
    name_postfix = ""
    if selection.surface_condition.item() != "regular":
        name_postfix += f"_{selection.surface_condition.item()}"
    name_postfix += ".png"

    rows, columns = subplot_geometry(len(select_time))
    overview_fig = plt.figure(num=0, figsize=(columns + 2, rows + 2))

    cmap = mpl.colormaps["hot_r"]
    cmap.set_over("magenta")

    for i, t in enumerate(select_time):
        year = times[i]
        temporal_selection = selection.sel(time=t)

        # Plot in separate figure and save
        plt.figure(num=i + 1)
        ax = plt.gca()
        for z in select_zones:
            add_zone_plot(
                ax,
                temporal_selection.sel(zone=z),
                zone_geometries[z],
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                norm=mpl.colors.LogNorm(),
            )
        plot_cities(add_name=True)
        plot_coast()
        plot_outline()
        plt.xlim(map_xlim)
        plt.ylim(map_ylim)
        plt.title(
            f"{year} {selection.vulnerability_class.item()}", fontdict={"fontsize": 8}
        )
        if show_max:
            xmax, ymax = loc_max[i]
            plt.scatter(
                xmax, ymax, c="blue", s=10, label=(f"Maximum: {max_val[i]:.2g}")
            )
            plt.legend(loc=3, frameon=False, prop={"size": 8})
        ax.set_aspect("equal")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "10 %", pad="10 %")
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
        cb1.set_label("LPR")

        name = (
            f"risk_{times_for_name[i]}_{selection.vulnerability_class.item()}"
            + name_postfix
        )
        if fig_path is not None:
            plt.savefig(os.path.join(fig_path, name), dpi=300)
            plt.close()

        # Plot in the overview figure
        plt.figure(num=0)
        ax = plt.subplot(rows, columns, i + 1, frame_on=False, xticks=[], yticks=[])
        for z in select_zones:
            add_zone_plot(
                ax,
                temporal_selection.sel(zone=z),
                zone_geometries[z],
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
        plot_cities(s=12 / float(columns), add_name=False)
        plot_coast(lw=2 / float(columns))
        plot_outline(lw=2 / float(columns))
        plt.xlabel("")
        plt.ylabel("")
        plt.xlim(map_xlim)
        plt.ylim(map_ylim)
        plt.title(year, fontdict={"fontsize": 8})
        if show_max:
            xmax, ymax = loc_max[i]
            plt.scatter(
                xmax,
                ymax,
                c="blue",
                s=10 / float(columns),
                label=(f"Maximum: {max_val[i]:.2g}"),
            )
            plt.legend(loc=3, frameon=False, prop={"size": 18 / float(columns)})
        ax.set_aspect("equal")

    # Finish overview figure
    # add overview map if there's space
    if i + 2 < rows * columns:
        axf = plt.subplot(rows, columns, i + 2, frame_on=False, xticks=[], yticks=[])
        axf.set_aspect("equal")
        plot_coast(lw=2 / float(columns))
        plot_outline(lw=2 / float(columns))
        plot_cities(s=12 / float(columns), fontsize=12 / float(columns), color="k")
        plt.xlim(map_xlim)
        plt.ylim(map_ylim)
    plt.suptitle(f"{selection.vulnerability_class.item()}")
    overview_fig.subplots_adjust(right=0.75)
    cbar_ax = overview_fig.add_axes([0.78, 0.2, 0.02, 0.6])
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    cb1 = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm)
    cb1.set_label("LPR", fontsize=7)
    cb1.ax.tick_params(labelsize=6)
    name = f"risk_overview_{selection.vulnerability_class.item()}" + name_postfix
    if fig_path is not None:
        plt.savefig(os.path.join(fig_path, name), dpi=300)
        plt.close()


def get_vc_curves(exposure_risk, exposure_db):
    lpr = exposure_risk["LPR-[lt-mean]"].sortby("bag_building_id")
    edb = exposure_db.sortby("bag_building_id")

    # check consistency building databases
    initial_length = len(lpr["bag_building_id"])
    exclude_dims = [d for d in lpr.dims if d != "bag_building_id"] + [
        d for d in edb.dims if d != "bag_building_id"
    ]
    lpr, edb = xr.align(lpr, edb, exclude=exclude_dims, join="inner")
    assert (
        len(lpr["bag_building_id"]) != 0
    ), "Building databases do not have a single match"
    if len(lpr["bag_building_id"]) != initial_length:
        print(
            "Notice: {} buildings are not in exposure_db".format(
                initial_length - len(lpr["bag_building_id"])
            )
        )

    vc_prob = edb.vc_matrix
    vc_mask = vc_prob > 0
    contributions = vc_mask * lpr
    sorting = (-1 * contributions).argsort(
        axis=contributions.dims.index("bag_building_id")
    )

    final_lpr = xr.zeros_like(lpr).assign_coords(
        bag_building_id=np.arange(len(lpr["bag_building_id"]))
    )
    weights = xr.zeros_like(final_lpr)

    for t_ind, t in enumerate(sorting["time"]):
        for vc_ind, vc in enumerate(sorting["vulnerability_class"]):
            local_sorting = sorting.isel(time=t_ind, vulnerability_class=vc_ind).values
            final_lpr.loc[dict(time=t, vulnerability_class=vc)] = contributions.isel(
                time=t_ind, vulnerability_class=vc_ind
            )[local_sorting].values
            weights.loc[dict(time=t, vulnerability_class=vc)] = vc_prob.isel(
                vulnerability_class=vc_ind
            )[local_sorting].values

    weights = weights.cumsum(dim="bag_building_id")
    return final_lpr, weights


def get_typology_cmap():
    all_typology_keys = [
        "RC3L",
        "URM1F_HC",
        "URM1F_B",
        "URM3L",
        "URM2L",
        "URM3M_D",
        "URM3M_U",
        "URM4L",
        "URM8L",
        "URM9L",
        "RC3M",
        "RC2",
        "PC2",
        "URM3M_B",
        "PC1L",
        "PC1M",
        "PC2L",
        "PC3H",
        "PC3L",
        "PC3M",
        "PC4L",
        "PC4M",
        "PC5L",
        "RC1H",
        "RC1L",
        "RC1M",
        "RC2L",
        "RC2M",
        "RC3H",
        "RC4L",
        "RC4M",
        "S1H",
        "S1L",
        "S1M",
        "S2H",
        "S2L",
        "S2M",
        "S3",
        "S3L",
        "S3M",
        "S4L",
        "S4M",
        "S5L",
        "S5M",
        "URM10",
        "URM1F_HA",
        "URM1L",
        "URM1M",
        "URM1_F",
        "URM1_O",
        "URM2M",
        "URM3M",
        "URM4L ",
        "URM5L",
        "URM5M",
        "URM6L",
        "URM6M",
        "URM7L",
        "URM7M",
        "URM8M",
        "W1 ",
        "W1L",
        "W1M",
        "W2",
        "W2L",
        "W2M",
        "W3",
        "W3L",
        "W4L",
        "W4M",
        "W5L",
        "W5M",
        "W6L",
        "W6M",
        "W7L",
        "W7M",
    ]
    # To get often visualized typologies clearly separate colors
    j = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    typ_colors = [
        plt.get_cmap("tab20")(j[i % 20]) for i, _ in enumerate(all_typology_keys)
    ]

    return {key: value for key, value in zip(all_typology_keys, typ_colors)}


def plot_typology_curves(exposure_risk, exposure_db, select_time=None, fig_path=None):
    cmap_dict = get_typology_cmap()
    if not select_time:
        select_time = exposure_risk.time

    # GMM-V7 logic tree madness
    if exposure_risk is not None:
        if "lt_median_choice" in exposure_risk:
            exposure_risk = exposure_risk.isel(lt_median_choice=0)

    times = pd.to_datetime(select_time)
    if times[0].month == 10:
        # Input is in gasyears
        times_for_name = [f"GY{t.year}" for t in times]
        times = [f"GY{t.year}/{t.year+1}" for t in times]
    else:
        times = times_for_name = [f"{t.year}" for t in times]

    typology_curves_y, typology_curves_x = get_vc_curves(exposure_risk, exposure_db)

    for i, t in enumerate(select_time):
        year = times[i]
        ty, tx = typology_curves_y.sel(time=t), typology_curves_x.sel(time=t)
        color_cutoff = np.sort(ty.isel(bag_building_id=0))[-8]

        # Plot in separate figure and save
        plt.figure(num=i + 1)
        for vc_ind, _ in enumerate(ty["vulnerability_class"]):
            vc = ty["vulnerability_class"][vc_ind].item()
            ty_vc, tx_vc = ty.isel(vulnerability_class=vc_ind), tx.isel(
                vulnerability_class=vc_ind
            )
            if ty_vc[0] >= color_cutoff:
                plt.loglog(tx_vc, ty_vc, lw=2, c=cmap_dict[vc], label=vc)
            else:
                plt.loglog(tx_vc, ty_vc, c="k", alpha=0.3, lw=1)

        xlim = [1e0, 1e5]
        ylim = [1e-8, 1e-3]
        plt.plot(xlim, [1e-5, 1e-5], "--", color="k")
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel("Number of buildings exceeding this risk")
        plt.ylabel("Local Personal Risk")
        plt.legend(loc=3)
        plt.title(year)
        if fig_path is not None:
            plt.savefig(
                os.path.join(fig_path, f"typology_curve_{times_for_name[i]}.png"),
                dpi=300,
            )
            plt.close()


def plot_ds_curves(
    exposure_risk, exposure_ds1, select_time=None, fig_path=None, component="maxrot"
):
    if not select_time:
        select_time = exposure_risk.time

    spoe_key = "structural_poe-[lt-mean]-[vc-mean]"

    # GMM-V7 logic tree madness
    if exposure_risk is not None:
        if "lt_median_choice" in exposure_risk:
            exposure_risk = exposure_risk.isel(lt_median_choice=0)

    times = pd.to_datetime(select_time)
    if times[0].month == 10:
        # Input is in gasyears
        times_for_name = [f"GY{t.year}" for t in times]
        times = [f"GY{t.year}/{t.year+1}" for t in times]
    else:
        times = times_for_name = [f"{t.year}" for t in times]

    ds1 = exposure_ds1.exposure.sel(component=component)
    ds2 = exposure_risk[spoe_key].sel(limit_state="DS2")
    ds3 = exposure_risk[spoe_key].sel(limit_state="DS3")

    expectation_values = np.concatenate(
        [
            ds1.sum(dim="bag_building_id").values[:, None],
            ds2.sum(dim="bag_building_id").values[:, None],
            ds3.sum(dim="bag_building_id").values[:, None],
        ],
        axis=-1,
    )

    ev = xr.DataArray(
        expectation_values, coords={"time": times, "limit_state": ["DS1", "DS2", "DS3"]}
    )
    ev.to_pandas().to_csv(os.path.join(fig_path, "damage_state_table.csv"))

    for i, t in enumerate(select_time):
        year = times[i]
        ds1_local = ds1.sel(time=t)
        ds2_local = ds2.sel(time=t)
        ds3_local = ds3.sel(time=t)

        # Plot in separate figure and save
        plt.figure(num=i + 1)

        plt.plot(
            np.arange(len(ds1_local)) + 1,
            sorted(ds1_local.values)[::-1],
            lw=2,
            c="k",
            label="DS1",
        )
        plt.plot(
            np.arange(len(ds2_local)) + 1,
            sorted(ds2_local.values)[::-1],
            ls="--",
            lw=2,
            c="k",
            label="DS2",
        )
        plt.plot(
            np.arange(len(ds3_local)) + 1,
            sorted(ds3_local.values)[::-1],
            ls=":",
            lw=2,
            c="k",
            label="DS3",
        )

        xlim = [1e0, 4e5]
        ylim = [1e-7, 1e-1]
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel("Number of buildings")
        plt.ylabel("Annual probability of exceedance")
        plt.legend(loc=3)
        plt.title(year)
        if fig_path is not None:
            plt.savefig(
                os.path.join(fig_path, f"damagestate_{times_for_name[i]}.png"), dpi=300
            )
            plt.close()


def plot_exposure_curves(exposure_risk, select_time=None, fig_path=None):
    # Allow for potential double-naming. To be fixed upstream
    mean_key = "LPR-[lt-mean]-[vc-mean]"
    frac_key = "LPR-[lt-fractiles]-[vc-mean]"

    if not select_time:
        select_time = exposure_risk.time

    # GMM-V7 logic tree madness - just ignore here and choose High
    if "lt_median_choice" in exposure_risk:
        exposure_risk = exposure_risk.isel(lt_median_choice=-1)

    times = pd.to_datetime(select_time)
    if times[0].month == 10:
        # Input is in gasyears
        times_for_name = [f"GY{t.year}" for t in times]
        times = [f"GY{t.year}/{t.year+1}" for t in times]
    else:
        times = times_for_name = [f"{t.year}" for t in times]

    for i, t in enumerate(select_time):
        year = times[i]

        # Plot in separate figure and save
        plt.figure(num=i + 1)
        lpr_mean = exposure_risk[mean_key].sel(time=t).values
        plt.plot(
            np.arange(len(lpr_mean)) + 1,
            sorted(lpr_mean)[::-1],
            lw=2,
            c="k",
            label="Mean LPR",
        )
        if frac_key in exposure_risk:
            add_risk_fractiles(exposure_risk[frac_key], t)
        plt.xscale("log")
        plt.yscale("log")
        xlim = [1e0, 2e5]
        ylim = [1e-8, 1e-3]
        plt.plot(xlim, [1e-5, 1e-5], "--", color="k")
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel("Number of buildings exceeding this risk")
        plt.ylabel("Local Personal Risk")
        plt.legend(loc=1)
        plt.title(year)
        if fig_path is not None:
            plt.savefig(
                os.path.join(fig_path, f"LPR_curve_{times_for_name[i]}.png"), dpi=300
            )
            plt.close()

    norm = 10**-5
    mean_norm_exceedance = (
        (exposure_risk[mean_key] > norm).sum(dim="bag_building_id").values[:, None]
    )
    mne = xr.DataArray(mean_norm_exceedance, coords={"time": times, "norm": [norm]})
    mne.to_pandas().to_csv(os.path.join(fig_path, "mean_norm_exceedance_table.csv"))
    if frac_key in exposure_risk:
        fractile_norm_exceedance = (
            (exposure_risk[frac_key] > norm).sum(dim="bag_building_id").values
        )
        qne = xr.DataArray(
            fractile_norm_exceedance,
            coords={
                "time": times,
                "fractiles": exposure_risk[frac_key]["fractile"].values,
            },
        )
        qne.to_pandas().to_csv(
            os.path.join(fig_path, "fractile_norm_exceedance_table.csv")
        )


def add_risk_fractiles(risk_frac, t):
    if len(risk_frac["fractile"]) >= 2:
        min_q = risk_frac["fractile"].min().item()
        max_q = risk_frac["fractile"].max().item()
        min_lpr_fractile = risk_frac.sel(time=t, fractile=min_q).values
        max_lpr_fractile = risk_frac.sel(time=t, fractile=max_q).values
        plt.fill_between(
            np.arange(len(min_lpr_fractile)) + 1,
            sorted(min_lpr_fractile)[::-1],
            sorted(max_lpr_fractile)[::-1],
            color="grey",
            alpha=0.8,
            label=f"P{int(min_q*100)}-P{int(max_q*100)} interval",
        )

        for q in risk_frac["fractile"]:
            if q.item() != min_q and q.item() != max_q:
                lpr_fractile = risk_frac.sel(time=t, fractile=q).values
                plt.plot(
                    np.arange(len(lpr_fractile)) + 1,
                    sorted(lpr_fractile)[::-1],
                    lw=1,
                    c="k",
                    label=f"P{int(q.item()*100)}",
                )

    else:
        for q in risk_frac["fractile"]:
            lpr_fractile = risk_frac.sel(time=t, fractile=q).values
            plt.plot(
                np.arange(len(lpr_fractile)) + 1,
                sorted(lpr_fractile)[::-1],
                lw=1,
                c="k",
                label=f"P{int(q.item()*100)}",
            )


def create_ncg_table(exposure_risk, select_time=None, fig_path=None, norm=10**-5):
    # Allow for potential double-naming. To be fixed upstream
    mean_key = "LPR-[lt-mean]-[vc-mean]"
    frac_key = "LPR-[lt-fractiles]-[vc-mean]"

    if not select_time:
        select_time = exposure_risk.time

    # GMM-V7 logic tree madness - just ignore here and choose High
    if "lt_median_choice" in exposure_risk:
        exposure_risk = exposure_risk.isel(lt_median_choice=-1)

    times = pd.to_datetime(select_time)
    if times[0].month == 10:
        # Input is in gasyears
        times = [f"GY{t.year}/{t.year+1}" for t in times]
    else:
        times = [f"{t.year}" for t in times]

    sorting = exposure_risk[mean_key].sel(time=select_time[0]).argsort()[::-1].values
    bag_id = exposure_risk["bag_building_id"].values[sorting]
    x, y = (
        exposure_risk["x"].values[sorting, None],
        exposure_risk["y"].values[sorting, None],
    )
    ranking = np.arange(1, len(x) + 1)[:, None]

    table = np.zeros_like(exposure_risk[mean_key]).astype(int)
    if frac_key in exposure_risk and 0.9 in exposure_risk[frac_key]["fractile"]:
        table[exposure_risk[frac_key].sel(fractile=0.9) >= norm] = 2
    table[exposure_risk[mean_key] >= norm] = 1

    table_sorted = table[sorting]
    columns = ["x", "y", f"Mean LPR {times[0]} rank"] + times
    ncg = xr.DataArray(
        np.concatenate((x, y, ranking, table_sorted), axis=-1),
        coords={"bag_building_id": bag_id, "columns": columns},
    )
    dtype = dict(zip(columns, [float, float, int] + len(times) * [int]))
    ncg = ncg.to_pandas().astype(dtype)
    ncg.to_csv(os.path.join(fig_path, "ncg_table.csv"))

    alpha = [0.5, 0.9, 0.95]
    alpha_ind = [int(np.ceil(len(bag_id) / (1 / a))) for a in alpha]
    expected_lpr = exposure_risk[mean_key].mean(dim="bag_building_id").values[:, None]
    sorted_lpr_values = np.sort(
        exposure_risk[mean_key].values,
        axis=list(exposure_risk[mean_key].dims).index("bag_building_id"),
    )
    var_lpr = sorted_lpr_values[np.array(alpha_ind), :].T
    pot_lpr = np.array(
        [sorted_lpr_values[a:, :].sum(axis=0) / (len(bag_id) - a) for a in alpha_ind]
    ).T

    table = np.concatenate((expected_lpr, var_lpr, pot_lpr), axis=-1)
    columns = (
        ["LPR expectation"] + [f"VaR_{a}" for a in alpha] + [f"PoT_{a}" for a in alpha]
    )
    risk_metrics = xr.DataArray(table, coords={"time": times, "columns": columns})
    risk_metrics.to_pandas().to_csv(
        os.path.join(fig_path, "risk_metrics.csv"), float_format="%.5e"
    )
