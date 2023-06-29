#%%
import xarray as xr
import geopandas as gpd

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

import seaborn as sns
import numpy as np
import metpy.calc as mpcalc
import datashader as dsh

from metpy.units import units
from datashader.mpl_ext import dsshow
from windrose import WindroseAxes

from utils import *
from const import *


def plot_pred_true(ds):

    figure, axis = plt.subplots(1, 2, figsize=(16, 8))
    figure.tight_layout(pad=7.0)
    ds.test_2019.groupby("time").mean().mul(1e6)[[S5P_OBS_COL, S5P_PRED_COL]].plot.line(
        ax=axis[0]
    )

    dsartist = dsshow(
        ds.test_2019[[S5P_OBS_COL, S5P_PRED_COL]].mul(1e6),
        dsh.Point(S5P_OBS_COL, S5P_PRED_COL),
        dsh.count(),
        norm="linear",
        aspect="auto",
        ax=axis[1],
    )

    plt.colorbar(dsartist)

    axis[0].set_title(r"a) Time series trend of OBS_NO$_{2}$ and BAU_NO$_{2}$")
    axis[0].set_xlabel("Date")
    axis[0].set_ylabel(f"$10^{{{-6}}}$ $mol/m^2$")

    axis[1].set_title(f"b) Scatter Plot of OBS_NO$_{2}$ and BAU_NO$_{2}$")
    axis[1].set_xlabel(r"OBS_NO$_{2}$ ($10^{{{-6}}}$ $mol/m^2$)")
    axis[1].set_ylabel(r"BAU_NO$_{2}$ ($10^{{{-6}}}$ $mol/m^2$)")
    axis[1].annotate(
        "$R^2$ = {:.3f}".format(
            r2_score(ds.test_2019[S5P_OBS_COL], ds.test_2019[S5P_PRED_COL])
        ),
        (10, 750),
    )
    line = mlines.Line2D([0, 1], [0, 1], color="red", label="1:1 line")
    transform = axis[1].transAxes
    line.set_transform(transform)
    axis[1].add_line(line)
    axis[1].legend()


def plot_fire_conflict():

    # data_df = prep_fire_df() if data_type == "Fire Spot" else prep_conflict_df()
    # color = "orange" if data_type == "Fire Spot" else "red"
    color_fire = "red"
    color_conflict = "orange"
    label_fire = "Fire Spot"
    label_conflict = "Conflict Spot"
    label_coal = "Coal power plant"

    years = [2020, 2021, 2022]

    coal_gdf = gpd.read_file(UK_COAL_SHP)

    bound_lv1 = gpd.read_file(UK_SHP_ADM1)
    bound_lv0 = gpd.read_file(UK_SHP_ADM0)

    fire_df = gpd.clip(prep_fire_df(), bound_lv0.geometry)
    conflict_df = gpd.clip(prep_conflict_df(), bound_lv0.geometry)

    # Plot fire locations
    figure_fire, ax_fire = plt.subplots(6, 3, figsize=(24, 5 * 6), layout="constrained")
    for j, y in enumerate(years):

        sd_ed = PERIOD_DICT[y]
        tks = list(sd_ed.keys())
        for i, tk in enumerate(tks):

            t = sd_ed[tk]

            sd = np.datetime64(f"{y}-{t['sm']}-{t['sd']}T00:00:00.000000000")
            ed = np.datetime64(f"{y}-{t['em']}-{t['ed']}T00:00:00.000000000")

            mask = (fire_df["DATETIME"] > sd) & (fire_df["DATETIME"] <= ed)
            df = fire_df.loc[mask]

            coal_gdf.plot(
                ax=ax_fire[i][j], color=COAL_COLOR, markersize=20, label=label_coal
            )
            df.plot(ax=ax_fire[i][j], color=color_fire, markersize=2, label=label_fire)
            bound_lv1.plot(
                ax=ax_fire[i][j], facecolor="None", edgecolor="black", lw=0.2
            )
            ax_fire[i][j].legend(
                bbox_to_anchor=(0, 0),
                loc="lower left",
            )
            ax_fire[i][j].set_title(f"{y}[{tk}]", fontsize=25)
    figure_fire.suptitle(
        f"Satellite-captured fire spots from the end of Febuary to July in 2020, 2021, 2022",
        fontsize=28,
    )
    # Plot conflict locations
    figure_conflict, ax_conflict = plt.subplots(
        4, 3, figsize=(24, 5 * 4), layout="constrained"
    )
    sd_ed = PERIOD_DICT[2022]

    tks = list(sd_ed.keys())

    j = 0
    for i, tk in enumerate(tks):

        i = int(i / 3)

        i = i if i == 0 else i + 1
        j = j if j < 3 else j - 3
        t = sd_ed[tk]
        sd = np.datetime64(f"2022-{t['sm']}-{t['sd']}T00:00:00.000000000")
        ed = np.datetime64(f"2022-{t['em']}-{t['ed']}T00:00:00.000000000")

        mask = (fire_df["DATETIME"] > sd) & (fire_df["DATETIME"] <= ed)
        masked_fire_df = fire_df.loc[mask]

        mask = (conflict_df["DATETIME"] > sd) & (conflict_df["DATETIME"] <= ed)
        masked_conflict_df = conflict_df.loc[mask]

        coal_gdf.plot(
            ax=ax_conflict[i][j], color=COAL_COLOR, markersize=20, label=label_coal
        )
        masked_fire_df.plot(
            ax=ax_conflict[i][j],
            color=color_fire,
            markersize=2,
            label=label_fire,
        )
        bound_lv1.plot(
            ax=ax_conflict[i][j], facecolor="None", edgecolor="black", lw=0.2
        )
        ax_conflict[i][j].legend(
            bbox_to_anchor=(0, 0),
            loc="lower left",
        )
        ax_conflict[i][j].set_title(f"2022[{tk}]", fontsize=25)

        coal_gdf.plot(
            ax=ax_conflict[i + 1][j], color=COAL_COLOR, markersize=20, label=label_coal
        )
        masked_conflict_df.plot(
            ax=ax_conflict[i + 1][j],
            color=color_conflict,
            markersize=2,
            label=label_conflict,
        )
        bound_lv1.plot(
            ax=ax_conflict[i + 1][j], facecolor="None", edgecolor="black", lw=0.2
        )

        ax_conflict[i + 1][j].legend(bbox_to_anchor=(0, 0), loc="lower left")
        ax_conflict[i + 1][j].set_title(f"2022[{tk}]", fontsize=25)

        j = j + 1

    figure_conflict.suptitle(
        f"Satellite-captured fire spots and Statistic conflict locations from Febuary to July in 2022",
        fontsize=28,
    )


def plot_ax_line(
    ds,
    geometry,
    location,
    gdf,
    ax,
    year,
    set_ylabel=False,
    get_table=False,
    event="covid",
):

    vl_covid_clr = "#33a02c"
    vl_war_clr = "#1f78b4"
    label_war = "War start date"
    label_covid = "Lockdown period"

    ls_covid = "dashed"
    ls_war = "dashed"
    lw = 2.5

    pred_truth_diff = ["cyan", "red", "#feb24c"]

    sd = np.datetime64(f"{year}-01-01T00:00:00.000000000")
    ed = np.datetime64(f"{year}-12-31T00:00:00.000000000")
    ds_clip = ds.rio.clip(geometry, gdf.crs).sel(time=slice(sd, ed))[
        [S5P_PRED_COL, S5P_OBS_COL]
    ]

    ds_clip_plot = ds_clip.mean(dim=["lat", "lon"], skipna=True)

    # calculate covid stats
    sd_cv19 = np.datetime64(f"{year}-04-18T00:00:00.000000000")
    ed_cv19 = np.datetime64(f"{year}-05-08T00:00:00.000000000")
    if event == "war":
        sd_cv19 = np.datetime64(f"{year}-02-25T00:00:00.000000000")
        ed_cv19 = np.datetime64(f"{year}-03-26T00:00:00.000000000")
        if year == 2020:
            ed_cv19 = np.datetime64(f"{year}-03-25T00:00:00.000000000")
    ds_clip_covid = ds_clip.sel(time=slice(sd_cv19, ed_cv19)).mean(dim=["lat", "lon"])
    obs_bau = (
        (ds_clip_covid[S5P_OBS_COL] - ds_clip_covid[S5P_PRED_COL])
        * 100
        / ds_clip_covid[S5P_PRED_COL]
    )
    t = obs_bau.values.shape
    obs_bau_std = np.nanstd(np.average(obs_bau.values.reshape(3, -1), axis=0))
    obs_bau_mean = obs_bau.mean(dim=["time"], skipna=True).item()

    # ploting
    org_df = ds_clip_plot.to_dataframe()
    df = get_nday_mean(org_df, nday=7)

    df[OBS_PRED_CHNAGE] = df[S5P_OBS_COL] - df[S5P_PRED_COL]
    df[[S5P_PRED_COL, S5P_OBS_COL]].plot.line(
        ax=ax, color=pred_truth_diff, legend=False
    )
    ax.set_ylim([df[S5P_OBS_COL].min() - 20, df[S5P_OBS_COL].max() + 50])

    if set_ylabel:
        ax.set_ylabel(NO2_UNIT)
    ax.grid(color="#d9d9d9")
    ax.set_title(f"{location}-{year}", fontsize=18)
    handles, labels = ax.get_legend_handles_labels()

    if year != 2019:
        covid_line = ax.axvline(
            x=np.datetime64(f"{year}-03-25T00:00:00.000000000"),
            color=vl_covid_clr,
            linewidth=lw,
            linestyle=ls_covid,
        )
        ax.axvline(
            x=np.datetime64(f"{year}-05-11T00:00:00.000000000"),
            color=vl_covid_clr,
            linewidth=lw,
            linestyle=ls_covid,
        )
        war_line = ax.axvline(
            x=np.datetime64(f"{year}-02-24T00:00:00.000000000"),
            color=vl_war_clr,
            linewidth=lw,
            linestyle=ls_war,
        )

        handles = handles + [war_line]
        labels = labels + [label_war]

        handles = handles + [covid_line]
        labels = labels + [label_covid]
        if get_table:
            return (obs_bau_mean, obs_bau_std, handles, labels)
        return handles, labels

    return obs_bau_mean, obs_bau_std if get_table else 1


def plot_ppl_obs_bau_line_mlt(org_ds):

    ds_2019 = prep_ds(org_ds, 2019)
    ds_2020 = prep_ds(org_ds, 2020)
    ds_2021 = prep_ds(org_ds, 2021)
    ds_2022 = prep_ds(org_ds, 2022)

    coal_gdf = gpd.read_file(UK_COAL_SHP)
    coal_gdf.crs = "EPSG:4326"
    # coal_gdf["buffer"] = coal_gdf.geometry.buffer(0.1, cap_style=3)

    for i, ppl_name in enumerate(coal_gdf.name.values):

        # geometry = coal_gdf.loc[coal_gdf["name"] == ppl_name]["buffer"].geometry
        geometry = coal_gdf.loc[coal_gdf["name"] == ppl_name].geometry
        fig, ax = plt.subplots(1, 4, figsize=(28, 4))

        plot_ax_line(
            ds_2019, geometry, ppl_name, coal_gdf, ax[0], 2019, set_ylabel=True
        )
        plot_ax_line(
            ds_2020,
            geometry,
            ppl_name,
            coal_gdf,
            ax[1],
            2020,
        )
        handles, labels = plot_ax_line(
            ds_2021, geometry, ppl_name, coal_gdf, ax[2], 2021
        )
        plot_ax_line(ds_2022, geometry, ppl_name, coal_gdf, ax[3], 2022)

        fig.legend(
            handles,
            labels,
            ncol=5,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.01),
            fontsize=18,
        )


def plot_obs_bau_pop_line_mlt(org_ds, event="covid"):

    ds_2019 = prep_ds(org_ds, 2019)
    ds_2020 = prep_ds(org_ds, 2020)
    ds_2021 = prep_ds(org_ds, 2021)
    ds_2022 = prep_ds(org_ds, 2022)

    col = "ADM2_EN"
    bound_pop = gpd.read_file(UK_SHP_ADM2)

    list_city = POP_ADMS

    nrows, ncols = len(list_city), 4
    fig, ax = plt.subplots(
        nrows, ncols, figsize=(12 * ncols, 6 * nrows), layout="constrained"
    )
    year_col = [i for i in range(2020, 2023)]
    name_col = ["mean", "var"]
    table_dict = {}

    for y in year_col:
        for n in name_col:
            table_dict[f"{y}_{n}"] = []

    table_dict["city"] = []
    for i, city in enumerate(list_city):

        geometry = bound_pop.loc[bound_pop[col] == city].geometry
        # fig, ax = plt.subplots(1, 3, figsize=(16, 4))
        plot_ax_line(
            ds_2019,
            geometry,
            city,
            bound_pop,
            ax[i][0],
            2019,
            set_ylabel=True,
            event=event,
        )
        obs_bau_2020 = plot_ax_line(
            ds_2020,
            geometry,
            city,
            bound_pop,
            ax[i][1],
            2020,
            get_table=True,
            event=event,
        )
        # with handles and labels
        obs_bau_2021 = plot_ax_line(
            ds_2021,
            geometry,
            city,
            bound_pop,
            ax[i][2],
            2021,
            get_table=True,
            event=event,
        )
        obs_bau_2022 = plot_ax_line(
            ds_2022,
            geometry,
            city,
            bound_pop,
            ax[i][3],
            2022,
            get_table=False,
            event=event,
        )

        table_dict["2022_mean"].append(obs_bau_2022[0])
        table_dict["2022_var"].append(obs_bau_2022[1])

        table_dict["2021_mean"].append(obs_bau_2021[0])
        table_dict["2021_var"].append(obs_bau_2021[1])

        table_dict["2020_mean"].append(obs_bau_2020[0])
        table_dict["2020_var"].append(obs_bau_2020[1])

        table_dict["city"].append(city)

    fig.legend(
        obs_bau_2021[2],
        obs_bau_2021[3],
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        fontsize=40,
    )
    return pd.DataFrame.from_dict(table_dict).round(1)


def plot_weather_params(ds, event="covid"):

    # border_df = get_boundary_cities()
    # conflict_df = get_monthly_conflict()

    u10 = ds.era5["u10"]
    v10 = ds.era5["v10"]
    ds.era5["wind"] = np.sqrt(u10**2 + v10**2)

    year_target = 2022
    list_color = ["#1b9e77", "#e7298a"]

    if event == "covid":
        year_target = 2020
        list_color = ["#1b9e77", "#d95f02"]

    years = [2019, year_target]
    sd_ed = PERIOD_DICT[year_target]

    tks = list(sd_ed.keys())

    var_label_dict = {
        # "wind": "Wind speed (m/s)",
        "blh": "Boundary layer height (m)",
        # "z": "Geopotential (m\u00b2/s\u00b22)",
    }
    # x_range_dict = {"wind": [2, 4.5], "blh": [400, 900], "t2m": [275, 285]}
    list_var = list(var_label_dict.keys())

    nrows = 1 if len(tks) == 2 else 2
    ncols = int(len(tks) / nrows)
    figure, ax = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), layout="constrained"
    )
    ylabel = "Relative Frequency (%)"

    k = 0
    j = 0
    for i, tk in enumerate(tks):
        i = int(i / ncols)
        j = j if j < ncols else 0
        t = sd_ed[tk]
        ax_sub = ax[i][j] if nrows > 1 else ax[j]
        for var in list_var:
            for year, color in zip(years, list_color):
                sd = np.datetime64(f"{year}-{t['sm']}-{t['sd']}{HOUR_STR}")
                ed = np.datetime64(f"{year}-{t['em']}-{t['ed']}{HOUR_STR}")
                # sel_ds = ds.era5.sel(time=slice(sd, ed)).mean("time")
                sel_ds = ds.era5.sel(time=slice(sd, ed))
                ts_data = sel_ds[var].values.reshape(-1)
                ts_data = ts_data[~np.isnan(ts_data)]

                ws = np.ones_like(ts_data) * 100 / ts_data.size
                ax_sub.hist(
                    ts_data,
                    bins=15,
                    weights=ws,
                    ec=color,
                    fc="None",
                    histtype="step",
                    label=f"{year}",
                    linewidth=3 if year == 2022 or year == 2019 else 1,
                )
                ax_sub.legend()
                ax_sub.set_title(f"{INDEX_FIG[k]}) {tk}", fontsize=20)
                ax_sub.set_xlabel(var_label_dict[var])
                # ax_sub.set_xlim(x_range_dict[var])
                ax_sub.set_ylabel(ylabel)
                ax_sub.legend(loc="upper right")
            k += 1
            j += 1


def plot_obs_bau_bubble(org_ds, year):

    ds = prep_ds(org_ds, year)

    bound_lv1 = gpd.read_file(UK_SHP_ADM1)
    adm_col = "ADM2_EN"
    bound_pop, crs = get_bound_pop()

    list_city = POP_ADMS
    sd_ed = PERIOD_DICT[year]

    dict_no2_change = {}

    tks = list(sd_ed.keys())

    for tk in tks:

        dict_no2_change[tk] = []

        t = sd_ed[tk]
        sd = np.datetime64(f"{year}-{t['sm']}-{t['sd']}T00:00:00.000000000")
        ed = np.datetime64(f"{year}-{t['em']}-{t['ed']}T00:00:00.000000000")

        for city in list_city:

            geometry = bound_pop.loc[bound_pop[adm_col] == city].geometry
            city_ds = (
                ds.rio.clip(geometry, crs)
                .mean(dim=["lat", "lon"])
                .sel(time=slice(sd, ed))
                .mean("time")[[S5P_PRED_COL, S5P_OBS_COL]]
            )
            dict_no2_change[tk].append(
                (city_ds[S5P_OBS_COL].item() - city_ds[S5P_PRED_COL].item())
                * 100
                / city_ds[S5P_PRED_COL].item()
            )

    df_no2_change = pd.DataFrame.from_dict(dict_no2_change)
    df_no2_change["Population"] = bound_pop["Population"].values
    geo_df = gpd.GeoDataFrame(
        df_no2_change, crs=crs, geometry=bound_pop.geometry.centroid
    )

    nrows = int(len(tks) / 2)
    ncols = 2
    figure, ax = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 5 * nrows), layout="constrained"
    )

    j = 0
    for i, col in enumerate(tks):

        i = int(i / 2)
        j = 0 if j > 1 else j
        sub_ax = ax[i][j] if nrows > 1 else ax[j]
        bound_lv1 = gpd.read_file(UK_SHP_ADM1)
        bound_lv1.plot(ax=sub_ax, facecolor="white", edgecolor="black", lw=0.7)
        norm_val = 15
        g = sns.scatterplot(
            data=geo_df,
            x=geo_df.centroid.x,
            y=geo_df.centroid.y,
            hue=col,
            hue_norm=(-15, 15),
            size="Population",
            sizes=(50, 500),
            palette=CMAP_NO2,
            ax=sub_ax,
            edgecolor="black",
            linewidth=1,
        )

        # g.legend(
        #     bbox_to_anchor=(1.0, 1.0),
        #     ncol=1,
        #     bbox_transform=sub_ax.transAxes,
        # )

        norm = plt.Normalize(-15, 15)
        sm = plt.cm.ScalarMappable(cmap=CMAP_NO2, norm=norm)
        sm.set_array([])

        # clb = g.figure.colorbar(
        #     sm,
        #     ax=ax[i][j - 1],
        #     fraction=0.047,
        #     orientation="horizontal",
        #     extend="both",
        #     label="NO$_{2}$ col. change (%)",
        # )

        g.set(title=rf"{INDEX_FIG[j]}) {year}_OBS[{col}] - {year}_BAU[{col}]")

        h, l = g.get_legend_handles_labels()
        l = [f"{li}M" if li != "Population" else li for li in l]
        legend = sub_ax.legend(
            h[-6:],
            l[-6:],
            bbox_to_anchor=(0, 0),
            loc="lower left",
            borderaxespad=0.0,
            # fontsize=13,
            edgecolor="black",
        )
        legend.get_frame().set_alpha(None)
        # legend.get_frame().set_facecolor((0, 0, 1, 0.1))
        j += 1
    figure.colorbar(
        sm,
        ax=ax[:, :] if nrows > 1 else ax[:],
        # ax=ax[:,1],
        # fraction=0.47,
        orientation="horizontal",
        extend="both",
        label="NO$_{2}$ col. change (%)",
        location="bottom",
        shrink=0.4,
    )
    plt.suptitle(rf"OBS_NO$_{2}$ - BAU_NO$_{2}$ difference (Major cities)", fontsize=18)
    return geo_df


def plot_obs_change_adm2():

    adm_col = "ADM2_EN"
    bound_lv2 = gpd.read_file(UK_SHP_ADM2)
    coal_gdf = gpd.read_file(UK_COAL_SHP)
    boundary = get_boundary_cities()
    org_ds = prep_s5p_ds()

    conflict_ds = get_monthly_conflict()

    year_target = 2022
    sd_ed = PERIOD_DICT[year_target]
    year_srcs = [i for i in range(2019, year_target)]

    tks = list(sd_ed.keys())

    pixel_change_dict = {}
    for year in year_srcs + [year_target]:

        for tk in tks:
            t = sd_ed[tk]
            sd = np.datetime64(f"{year}-{t['sm']}-{t['sd']}{HOUR_STR}")
            ed = np.datetime64(f"{year}-{t['em']}-{t['ed']}{HOUR_STR}")
            pixel_change_dict[f"{year}_{tk}"] = org_ds.sel(time=slice(sd, ed)).mean(
                "time"
            )[[S5P_OBS_COL]]

        pixel_change_dict[f"self_{year}"] = (
            (
                pixel_change_dict[f"{year}_{tks[1]}"]
                - pixel_change_dict[f"{year}_{tks[0]}"]
            )
            * 100
            / pixel_change_dict[f"{year}_{tks[0]}"]
        )
        self_year_change = []

        for adm2 in bound_lv2[adm_col].values:
            geometry = bound_lv2.loc[bound_lv2[adm_col] == adm2].geometry
            self_year_change.append(
                pixel_change_dict[f"self_{year}"]
                .rio.clip(geometry, bound_lv2.crs)
                .mean(dim=["lat", "lon"])[S5P_OBS_COL]
                .item()
            )
        bound_lv2[f"self_{year}"] = self_year_change

    for tk in tks:
        for year in year_srcs:

            pixel_change_year = (
                (
                    pixel_change_dict[f"{year_target}_{tk}"]
                    - pixel_change_dict[f"{year}_{tk}"]
                )
                * 100
                / pixel_change_dict[f"{year}_{tk}"]
            )
            year_tk_items = []
            for adm2 in bound_lv2[adm_col].values:
                geometry = bound_lv2.loc[bound_lv2[adm_col] == adm2].geometry

                # cal obs deweather no2 ds
                change_adm_no2_ds = (
                    pixel_change_year.rio.clip(geometry, bound_lv2.crs)
                    .mean(dim=["lat", "lon"])[S5P_OBS_COL]
                    .item()
                )
                year_tk_items.append(change_adm_no2_ds)
            bound_lv2[f"inter_{year}_{tk}"] = year_tk_items

    # self_ref_fig, self_ref_ax = plt.subplots(
    #     2, 2, figsize=(6 * 2, 5.2 * 2), layout="constrained"
    # )
    # inter_ref_fig, inter_ref_ax = plt.subplots(
    #     2, 3, figsize=(6 * 3, 5.2 * 2), layout="constrained"
    # )
    ncols = 3
    nrows = len(tks)
    inter_war_fig, inter_war_ax = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 5 * nrows), layout="constrained"
    )

    # self refugee change
    # j = 0
    # k = 0
    # for i, year in enumerate(year_srcs + [year_target]):
    #     i = int(i / 2)
    #     j = 0 if j > 1 else j
    #     col = f"self_{year}"
    #     bound_lv2.plot(
    #         column=col,
    #         ax=self_ref_ax[i][j],
    #         cmap=CMAP_NO2,
    #         vmin=-70,
    #         vmax=70,
    #         legend=False,
    #     )
    #     self_ref_ax[i][j].set_title(
    #         rf"{INDEX_FIG[k]}) {year}[{tks[1]}] - {year}[{tks[0]}]", fontsize=14
    #     )
    #     bound_lv2.plot(
    #         ax=self_ref_ax[i][j], facecolor="None", edgecolor="black", lw=0.2
    #     )
    #     self_ref_conflict = conflict_ds.loc[conflict_ds[f"conflict_{tks[1]}"] > 2]
    #     self_ref_conflict.plot(
    #         ax=self_ref_ax[i][j],
    #         facecolor="None",
    #         edgecolor=EDGE_COLOR_CONFLICT,
    #         lw=1,
    #     )
    #     coal_gdf.plot(
    #         ax=self_ref_ax[i][j],
    #         color=COAL_COLOR,
    #         markersize=20,
    #         label="CPP",
    #     )
    #     boundary.plot(
    #         ax=self_ref_ax[i][j],
    #         facecolor="None",
    #         edgecolor=EDGE_COLOR_BORDER,
    #         lw=1,
    #     )
    #     handles, _ = self_ref_ax[i][j].get_legend_handles_labels()
    #     self_ref_ax[i][j].legend(
    #         handles=[*LG_CONFLICT, *LG_BORDER, *handles], loc="lower left"
    #     )
    #     j += 1
    #     k += 1
    # norm = plt.Normalize(-70, 70)
    # sm = plt.cm.ScalarMappable(cmap=CMAP_NO2, norm=norm)
    # sm.set_array([])
    # self_ref_fig.colorbar(
    #     sm,
    #     ax=self_ref_ax[:, :],
    #     # ax=ax[:,1],
    #     # fraction=0.47,
    #     orientation="horizontal",
    #     extend="both",
    #     label="NO$_{2}$ col. change (%)",
    #     location="bottom",
    #     shrink=0.4,
    # )
    # self_ref_fig.suptitle(
    #     rf'OBS "before-during" change estimates (City level)', fontsize=18
    # )
    # # inter refugee change
    # for i, tk in enumerate(tks[:1]):
    #     for j, year in enumerate(year_srcs):
    #         col = f"inter_{year}_{tk}"
    #         bound_lv2.plot(
    #             column=col,
    #             ax=inter_ref_ax[i][j],
    #             cmap=CMAP_NO2,
    #             vmin=-70,
    #             vmax=70,
    #             legend=False,
    #         )
    #         inter_ref_ax[i][j].set_title(
    #             f"{year_target}[{tk}] - {year}[{tk}]", fontsize=16
    #         )
    #         bound_lv2.plot(
    #             ax=inter_ref_ax[i][j], facecolor="None", edgecolor="black", lw=0.2
    #         )
    #         if i > 0:
    #             inter_ref_conflict = conflict_ds.loc[
    #                 conflict_ds[f"conflict_{tks[1]}"] > 2
    #             ]
    #             inter_ref_conflict.plot(
    #                 ax=inter_ref_ax[i][j],
    #                 facecolor="None",
    #                 edgecolor=EDGE_COLOR_CONFLICT,
    #                 lw=1,
    #             )
    #             coal_gdf.plot(
    #                 ax=inter_ref_ax[i][j],
    #                 color=COAL_COLOR,
    #                 markersize=20,
    #                 label="CPP",
    #             )
    #             boundary.plot(
    #                 ax=inter_ref_ax[i][j],
    #                 facecolor="None",
    #                 edgecolor=EDGE_COLOR_BORDER,
    #                 lw=1,
    #             )
    #             handles, _ = inter_ref_ax[i][j].get_legend_handles_labels()
    #             inter_ref_ax[i][j].legend(
    #                 handles=[*LG_CONFLICT, *LG_BORDER, *handles], loc="lower left"
    #             )

    # norm = plt.Normalize(-70, 70)
    # sm = plt.cm.ScalarMappable(cmap=CMAP_NO2, norm=norm)
    # sm.set_array([])
    # inter_ref_fig.colorbar(
    #     sm,
    #     ax=inter_ref_ax[:, :],
    #     # ax=ax[:,1],
    #     # fraction=0.47,
    #     orientation="horizontal",
    #     extend="both",
    #     label="NO$_{2}$ col. change (%)",
    #     location="bottom",
    #     shrink=0.4,
    # )
    # inter_ref_fig.suptitle(
    #     rf'OBS "year-to-year" change estimates (City level)', fontsize=20
    # )

    # inter war change
    for i, tk in enumerate(tks):
        for j, year in enumerate(year_srcs):
            col = f"inter_{year}_{tk}"
            bound_lv2.plot(
                column=col,
                ax=inter_war_ax[i][j],
                cmap=CMAP_NO2,
                vmin=-30,
                vmax=30,
                legend=False,
            )
            inter_war_ax[i][j].set_title(
                f"{year_target}[{tk}] - {year}[{tk}]", fontsize=18
            )
            bound_lv2.plot(
                ax=inter_war_ax[i][j], facecolor="None", edgecolor="black", lw=0.2
            )
            inter_war_conflict = conflict_ds.loc[conflict_ds[f"conflict_{tk}"] > 10]
            inter_war_conflict.plot(
                ax=inter_war_ax[i][j],
                facecolor="None",
                edgecolor=EDGE_COLOR_CONFLICT,
                lw=1,
            )
            coal_gdf.plot(
                ax=inter_war_ax[i][j],
                color=COAL_COLOR,
                markersize=20,
                label="CPP",
            )
            boundary.plot(
                ax=inter_war_ax[i][j],
                facecolor="None",
                edgecolor=EDGE_COLOR_BORDER,
                lw=1,
            )
            handles, _ = inter_war_ax[i][j].get_legend_handles_labels()
            inter_war_ax[i][j].legend(
                handles=[*LG_CONFLICT, *LG_BORDER, *handles], loc="lower left"
            )

    norm = plt.Normalize(-30, 30)
    sm = plt.cm.ScalarMappable(cmap=CMAP_NO2, norm=norm)
    sm.set_array([])
    inter_war_fig.colorbar(
        sm,
        ax=inter_war_ax[:, :],
        # ax=ax[:,1],
        # fraction=0.47,
        orientation="horizontal",
        extend="both",
        label="NO$_{2}$ col. change (%)",
        location="bottom",
        shrink=0.4,
    )
    inter_war_fig.suptitle(
        rf'OBS "year-to-year" change estimates (City level)', fontsize=22
    )

    return bound_lv2


def plot_obs_bau_adm2(org_ds):
    year_event = 2022
    year_bau = 2019
    border_df = get_boundary_cities()
    conflict_df = get_monthly_conflict()
    coal_gdf = gpd.read_file(UK_COAL_SHP)

    ds_war = prep_ds(org_ds, year_event)
    ds_bau = prep_ds(org_ds, year_bau)

    conflict_ds = prep_conflict_df()
    fire_ds = prep_fire_df()

    bound_lv2 = gpd.read_file(UK_SHP_ADM2)
    sd_ed = PERIOD_DICT[year_event]

    adm_col = "ADM2_EN"

    dict_obs_bau = {}
    dict_y2y = {}
    dict_conflict_change = {}
    dict_fire_change = {}

    list_adm = bound_lv2[adm_col].values

    tks = list(sd_ed.keys())
    for tk in tks:

        dict_obs_bau[tk] = []
        dict_y2y[tk] = []
        dict_conflict_change[tk] = []
        dict_fire_change[tk] = []

        t = sd_ed[tk]
        sd_event = np.datetime64(f"{year_event}-{t['sm']}-{t['sd']}T00:00:00.000000000")
        ed_event = np.datetime64(f"{year_event}-{t['em']}-{t['ed']}T00:00:00.000000000")

        sd_bau = np.datetime64(f"{year_bau}-{t['sm']}-{t['sd']}T00:00:00.000000000")
        ed_bau = np.datetime64(f"{year_bau}-{t['em']}-{t['ed']}T00:00:00.000000000")

        for adm in list_adm:
            geometry = bound_lv2.loc[bound_lv2[adm_col] == adm].geometry

            # cal obs deweather no2 ds
            adm_obs_bau_ds = (
                ds_war.rio.clip(geometry, bound_lv2.crs)
                .mean(dim=["lat", "lon"])
                .sel(time=slice(sd_event, ed_event))
                .mean("time")[[S5P_PRED_COL, S5P_OBS_COL]]
            )

            adm_y2y_ds = (
                ds_bau.rio.clip(geometry, bound_lv2.crs)
                .mean(dim=["lat", "lon"])
                .sel(time=slice(sd_bau, ed_bau))
                .mean("time")[[S5P_OBS_COL]]
            )

            dict_obs_bau[tk].append(
                (
                    adm_obs_bau_ds[S5P_OBS_COL].item()
                    - adm_obs_bau_ds[S5P_PRED_COL].item()
                )
                * 100
                / adm_obs_bau_ds[S5P_PRED_COL].item()
            )
            dict_y2y[tk].append(
                (adm_obs_bau_ds[S5P_OBS_COL].item() - adm_y2y_ds[S5P_OBS_COL].item())
                * 100
                / adm_y2y_ds[S5P_OBS_COL].item()
            )

            # cal conflict_ds
            mask_date = (conflict_ds["DATETIME"] > sd_event) & (
                conflict_ds["DATETIME"] <= ed_event
            )
            amd_cflt_ds = conflict_ds.loc[mask_date]
            amd_cflt_ds = gpd.clip(amd_cflt_ds, geometry)
            dict_conflict_change[tk].append(len(amd_cflt_ds))

            # cal fire ds
            mask_date = (fire_ds["DATETIME"] > sd_event) & (
                fire_ds["DATETIME"] <= ed_event
            )
            amd_fire_ds = fire_ds.loc[mask_date]
            amd_fire_ds = gpd.clip(amd_fire_ds, geometry)
            dict_fire_change[tk].append(len(amd_fire_ds))

        bound_lv2[f"obs_bau_{tk}"] = dict_obs_bau[tk]
        bound_lv2[f"y2y_{tk}"] = dict_y2y[tk]
        bound_lv2[f"conflict_{tk}"] = dict_conflict_change[tk]
        bound_lv2[f"fire_{tk}"] = dict_fire_change[tk]

    nrows = len(tks)
    ncols = 4
    figure, ax = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 5 * nrows), layout="constrained"
    )

    obs_bau_mean_war_adm2 = []
    obs_bau_mean_border_amd2 = []
    obs_bau_mean_normal_adm2 = []

    obs_bau_std_war_adm2 = []
    obs_bau_std_border_amd2 = []
    obs_bau_std_normal_adm2 = []

    y2y_mean_war_adm2 = []
    y2y_mean_border_amd2 = []
    y2y_mean_normal_adm2 = []

    y2y_std_war_adm2 = []
    y2y_std_border_amd2 = []
    y2y_std_normal_adm2 = []
    vmin, vmax = -35, 35
    for i, tk in enumerate(tks):

        legend = False if i < (len(tks) - 1) else True
        bound_lv2.plot(
            column=f"y2y_{tk}",
            ax=ax[i][0],
            legend=legend,
            cmap=CMAP_NO2,
            vmin=vmin,
            vmax=vmax,
            legend_kwds={
                "label": r"NO$_{2}$ col. change (%)",
                "orientation": "horizontal",
                "extend": "both",
                "shrink": 0.8,
            },
        )
        bound_lv2.plot(
            column=f"obs_bau_{tk}",
            ax=ax[i][1],
            legend=legend,
            cmap=CMAP_NO2,
            vmin=vmin,
            vmax=vmax,
            legend_kwds={
                "label": r"NO$_{2}$ col. change (%)",
                "orientation": "horizontal",
                "extend": "both",
                "shrink": 0.8,
            },
        )
        bound_lv2.plot(
            column=f"conflict_{tk}",
            ax=ax[i][2],
            legend=legend,
            cmap=CMAP_CONFLICT,
            vmin=0,
            vmax=300,
            legend_kwds={
                "label": "Number of conflict spots",
                "orientation": "horizontal",
                "extend": "both",
                "shrink": 0.8,
            },
        )

        bound_lv2.plot(
            column=f"fire_{tk}",
            ax=ax[i][3],
            legend=legend,
            cmap=CMAP_FIRE,
            vmin=0,
            vmax=600,
            legend_kwds={
                "label": "Number of fire spots",
                "orientation": "horizontal",
                "extend": "both",
                "shrink": 0.8,
            },
        )
        threshold_conflict_point = 5 if i < 2 else 18
        # threshold_conflict_point = 2
        event_bound = conflict_df.loc[
            conflict_df[f"conflict_{tk}"] > threshold_conflict_point
        ]
        event_bound.plot(
            ax=ax[i][0], facecolor="None", edgecolor=EDGE_COLOR_CONFLICT, lw=1
        )
        event_bound.plot(
            ax=ax[i][1], facecolor="None", edgecolor=EDGE_COLOR_CONFLICT, lw=1
        )
        # border_df.plot(ax=ax[i][0], facecolor="None", edgecolor=EDGE_COLOR_BORDER, lw=1)
        # border_df.plot(ax=ax[i][1], facecolor="None", edgecolor=EDGE_COLOR_BORDER, lw=1)

        # extract scores
        war_ds = bound_lv2.loc[bound_lv2[adm_col].isin(event_bound[adm_col].tolist())]
        border_ds = bound_lv2.loc[bound_lv2[adm_col].isin(border_df[adm_col].tolist())]
        # normal_ds = bound_lv2.loc[
        #     ~bound_lv2[adm_col].isin(
        #         border_df[adm_col].tolist() + event_bound[adm_col].tolist()
        #     )
        # ]
        normal_ds = bound_lv2.loc[
            ~bound_lv2[adm_col].isin(event_bound[adm_col].tolist())
        ]

        obs_bau_mean_war_adm2.append(war_ds[f"obs_bau_{tk}"].mean())
        obs_bau_mean_border_amd2.append(border_ds[f"obs_bau_{tk}"].mean())
        obs_bau_mean_normal_adm2.append(normal_ds[f"obs_bau_{tk}"].mean())

        obs_bau_std_war_adm2.append(war_ds[f"obs_bau_{tk}"].std())
        obs_bau_std_border_amd2.append(border_ds[f"obs_bau_{tk}"].std())
        obs_bau_std_normal_adm2.append(normal_ds[f"obs_bau_{tk}"].std())

        y2y_mean_war_adm2.append(war_ds[f"y2y_{tk}"].mean())
        y2y_mean_border_amd2.append(border_ds[f"y2y_{tk}"].mean())
        y2y_mean_normal_adm2.append(normal_ds[f"y2y_{tk}"].mean())

        y2y_std_war_adm2.append(war_ds[f"y2y_{tk}"].std())
        y2y_std_border_amd2.append(border_ds[f"y2y_{tk}"].std())
        y2y_std_normal_adm2.append(normal_ds[f"y2y_{tk}"].std())

        for j in range(len(ax[i])):
            coal_gdf.plot(
                ax=ax[i][j],
                color=COAL_COLOR,
                markersize=20,
                label="CPP",
            )
            bound_lv2.plot(ax=ax[i][j], facecolor="None", edgecolor="black", lw=0.05)
            ax[i][j].legend(loc="lower left")

        handles, _ = ax[i][0].get_legend_handles_labels()
        ax[i][0].legend(handles=[*LG_CONFLICT, *handles], loc="lower left")

        ax[i][0].set_title(
            rf"{year_event}_OBS[{tk}] - {year_bau}_OBS[{tk}]", fontsize=14
        )
        ax[i][1].legend(handles=[*LG_CONFLICT, *handles], loc="lower left")

        ax[i][1].set_title(
            rf"{year_event}_OBS[{tk}] - {year_event}_BAU[{tk}]", fontsize=14
        )
        ax[i][2].set_title(f"Conflict Locations {year_event}[{tk}]", fontsize=14)
        ax[i][3].set_title(f"Fire Locations {year_event}[{tk}]", fontsize=14)
    plt.suptitle(
        rf"OBS_NO$_{2}$ -  BAU_NO$_{2}$ difference , Conflict locations, and Fire spots 2022[Feb - Jul] (City level)",
        fontsize=18,
    )

    mean_std_df = pd.DataFrame()
    mean_std_df["time"] = tks

    mean_std_df["mean_normal_obs_bau"] = obs_bau_mean_normal_adm2
    mean_std_df["std_normal_obs_bau"] = obs_bau_std_normal_adm2
    mean_std_df["mean_normal_y2y"] = y2y_mean_normal_adm2
    mean_std_df["std_normal_y2y"] = y2y_std_normal_adm2

    mean_std_df["mean_war_obs_bau"] = obs_bau_mean_war_adm2
    mean_std_df["std_war_obs_bau"] = obs_bau_std_war_adm2
    mean_std_df["mean_war_y2y"] = y2y_mean_war_adm2
    mean_std_df["std_war_y2y"] = y2y_std_war_adm2

    # mean_std_df["mean_border_obs_bau"] = obs_bau_mean_border_amd2
    # mean_std_df["std_border_obs_bau"] = obs_bau_std_border_amd2
    # mean_std_df["mean_border_y2y"] = y2y_mean_border_amd2
    # mean_std_df["std_border_y2y"] = y2y_std_border_amd2

    return bound_lv2, mean_std_df



# %%
