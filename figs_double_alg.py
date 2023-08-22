# %%

import matplotlib.pyplot as plt
import xarray as xr
import geopandas as gpd
import seaborn as sns
import numpy as np
import calendar
import matplotlib.cm as cm
import metpy.calc as mpcalc
import windrose
import matplotlib.dates as mdates

from utils import *
from const import *
from tabs import *

S5P_VERSIONS = ["S5P_v1.x", "S5P_v2.4"]


def compare_obs_bau(ds1, ds2, fig_name=None):
    bau_var = "BAU_S5P"
    obs_var = "OBS_S5P"
    list_ds = [ds1, ds2]
    list_var = [obs_var, bau_var, obs_var]

    cmaps = ["YlOrBr", "YlOrBr", "YlOrBr"]
    dates_2020 = [LCKDWN_SD, LCKDWN_ED]
    dates_2022 = ["2022-02-24", "2022-07-31"]

    # tits = ["Prelockdown", "Lockdown"]

    nrs, ncs = 2, 3
    w, h = 5 * ncs, 5 * nrs
    for date, year, index in zip(
        [dates_2020, dates_2022], ["2020", "2022"], ["a", "b"]
    ):
        fig, axes = plt.subplots(nrs, ncs, figsize=(w, h), layout="constrained")

        for i, (ds, ver) in enumerate(zip(list_ds, S5P_VERSIONS)):
            for j, (var, cmap) in enumerate(zip(list_var, cmaps)):
                ax = axes[i, j]

                ds_var = ds.dw_ds[var]
                sd, ed = date[0], date[1]

                y = year

                if j == 2:
                    y19 = "2019"
                    sd, ed = date[0].replace(year, y19), date[1].replace(year, y19)
                    y = y19

                print(sd, ed)

                sel_ds = ds_var.sel(time=slice(sd, ed)).mean("time")
                cb_mean = sel_ds.plot(cmap=cmap, ax=ax, vmin=10, vmax=50)

                ax.set_title(f"{ver} {var.split('_')[0]} {y}", fontsize=20)
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.set_yticks([])
                ax.set_yticklabels([])
                ax.set_ylabel("")
                ax.set_xlabel("")
                cb_mean.colorbar.remove()

        cb_mean = fig.colorbar(
            cb_mean,
            ax=axes,
            shrink=0.4,
            extend="both",
            location="bottom",
        )
        cb_mean.set_label(NO2_UNIT, size=20)
        cb_mean.ax.tick_params(labelsize=20)
        if fig_name:
            path_ = f"figures/{fig_name}_{index}.tiff"
            fig.savefig(path_, format="tiff", dpi=300)


def plt_scatter_map_covid_2alg(df_1, df_2, fig_name=None):
    nrs, ncs = 2, 2
    w, h = 5 * ncs, 4 * nrs
    bound_lv1 = gpd.read_file(UK_SHP_ADM1)
    index_fig = [["1", "2"], ["3", "4"]]
    fis = ["a", "b"]
    huem = 35

    fig1, axes1 = plt.subplots(nrs, ncs, figsize=(w, h), layout="constrained")
    fig2, axes2 = plt.subplots(nrs, ncs, figsize=(w, h), layout="constrained")

    for fi, p, fig, axes in zip(fis, DATE_2020.keys(), [fig1, fig2], [axes1, axes2]):
        for i, (df, v) in enumerate(zip([df_1, df_2], S5P_VERSIONS)):
            for j, col in enumerate(["2020-2019", OBS_BAU_COL]):
                ax = axes[i, j]
                bound_lv1.plot(ax=ax, facecolor="white", edgecolor="black", lw=0.7)
                g = sns.scatterplot(
                    data=df,
                    x=df.centroid.x,
                    y=df.centroid.y,
                    hue=f"{col} {p}",
                    hue_norm=(-1 * huem, huem),
                    size="Population",
                    sizes=(50, 500),
                    palette=CMAP_NO2,
                    ax=ax,
                    edgecolor="black",
                    linewidth=1,
                )
                g.set(title=f"({fi}{index_fig[i][j]}) {v} - {col} \n {p}")
                ax.get_legend().remove()
                if i < 1 and j < 1:
                    x, y, arrow_length = 0.05, 0.95, 0.12
                    ax.annotate(
                        "N",
                        xy=(x, y),
                        xytext=(x, y - arrow_length),
                        arrowprops=dict(facecolor="black", width=5, headwidth=12),
                        ha="center",
                        va="center",
                        fontsize=15,
                        xycoords=ax.transAxes,
                    )

                    h, l = g.get_legend_handles_labels()
                    l = [f"{li}M" if li != "Population" else li for li in l]
                    legend = ax.legend(
                        h[-6:],
                        l[-6:],
                        bbox_to_anchor=(0, 0),
                        loc="lower left",
                        borderaxespad=0.0,
                        # fontsize=13,
                        edgecolor="black",
                    )
                    legend.get_frame().set_alpha(None)

                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.set_yticks([])
                ax.set_yticklabels([])

        norm = plt.Normalize(-1 * huem, huem)
        sm = plt.cm.ScalarMappable(cmap=CMAP_NO2, norm=norm)
        sm.set_array([])
        fig.colorbar(
            sm,
            ax=axes,
            # fraction=0.47,
            orientation="horizontal",
            extend="both",
            label="NO$_{2}$ col. change (%)",
            location="bottom",
            shrink=0.4,
        )
        if fig_name:
            path_ = f"figures/{fig_name}_{fi}.tiff"
            fig.savefig(path_, format="tiff", dpi=300)


def plt_line_ts_adm_2alg(ds_1, ds_2, dict_city, eng_name, fig_name=None):
    list_city = dict_city.keys()
    nrs, ncs = len(list_city), 4
    w, h = 2.3 * ncs, 2 * nrs
    # fig, axes = plt.subplots(nrs, ncs, figsize=(w, h), layout="constrained")

    fig = plt.figure(layout="constrained", figsize=(w, h))
    subfigs = fig.subfigures(nrs, 1, wspace=0.07)

    for i, city in enumerate(list_city):
        txt = subfigs[i].suptitle(eng_name[city], fontsize="x-large")
        axes = subfigs[i].subplots(1, ncs, sharey=True)
        for j, (ds, v) in enumerate(zip([ds_1, ds_2], S5P_VERSIONS)):
            df_city = ds[city].mean(["lat", "lon"], skipna=True).to_dataframe()
            for k, y in enumerate([2020, 2022]):
                df = get_nday_mean(df_city[f"{y}-02-01":f"{y}-07-31"], 7)
                ax = axes[2 * j + k]
                sns.lineplot(
                    df,
                    x="time",
                    y=S5P_PRED_COL,
                    ax=ax,
                    legend="brief",
                    label="BAU",
                    color="#d95f02",
                    linewidth=1.5,
                )
                sns.lineplot(
                    df,
                    x="time",
                    y=S5P_OBS_COL,
                    ax=ax,
                    legend="brief",
                    label="OBS",
                    color="#41b6c4",
                    linewidth=1.5,
                )
                ax.get_legend().remove()
                ax.set_title(f"{v}", size=10, loc="left")
                ax.set_title(f"{y}", size=10, loc="right")
                if y == 2020:
                    for d in [LCKDWN_SD, LCKDWN_ED]:
                        d = pd.to_datetime(d)
                        cl = ax.axvline(x=d, color="red", linestyle="--")
                if y == 2022:
                    wd = pd.to_datetime(WAR_SD)
                    wl = ax.axvline(x=wd, color="green", linestyle="--")

                ax.grid(visible=True, which="major", color="black", linewidth=0.1)
                ax.grid(visible=True, which="minor", color="black", linewidth=0.1)

                fmt = mdates.DateFormatter("%b")
                locator = mdates.MonthLocator()
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(fmt)
                ax.set_ylabel(NO2_UNIT)
                ax.set_ylim(dict_city[city]["min"], dict_city[city]["max"])
                # ax.tick_params(axis="both", which="major", labelsize=11)

                ax.set_xlabel("")
                ax.set_xticklabels(["", "Mar", "", "May", "", "Jul", ""])
                if i != len(list_city) - 1:
                    for tick in ax.xaxis.get_major_ticks():
                        tick.tick1line.set_visible(False)
                        tick.tick2line.set_visible(False)
                        tick.label1.set_visible(False)
                        tick.label2.set_visible(False)

                if (2 * j + k) > 0:
                    for tick in ax.yaxis.get_major_ticks():
                        tick.tick1line.set_visible(False)
                        tick.tick2line.set_visible(False)
                        tick.label1.set_visible(False)
                        tick.label2.set_visible(False)
                try:
                    handles, labels = ax.get_legend_handles_labels()
                    handles = handles + [cl] + [wl]
                    labels = labels + ["COVID-19 Lockdown"] + ["War start date"]
                    lgd = fig.legend(
                        handles,
                        labels,
                        ncol=4,
                        loc="upper center",
                        bbox_to_anchor=(0.5, -0.01),
                        fontsize=12,
                    )
                except:
                    pass
    if fig_name:
        path_ = f"figures/{fig_name}.tiff"
        fig.savefig(
            path_,
            format="tiff",
            dpi=300,
            bbox_inches="tight",
        )


def plt_feature_importance(ds1, ds2, fig_name=None):
    nrs, ncs = 1, 2
    w, h = 6 * ncs, 6 * nrs
    fig, axes = plt.subplots(nrs, ncs, figsize=(w, h), layout="constrained")

    for i, (ds, v) in enumerate(zip([ds1, ds2], S5P_VERSIONS)):
        df = pd.DataFrame()

        df["Feature importance"] = ds.model.feature_importances_
        df["Feature name"] = FEATURE_NAMES
        df = df.sort_values(by=["Feature importance"], ascending=False)

        sns.set_theme(style="whitegrid")
        ax = axes[i]
        sns.barplot(
            data=df,
            y="Feature name",
            x="Feature importance",
            orient="h",
            ax=ax,
            color="#1b9e77",
        )
        ax.set_title(v)
        ax.set_xlim(0, 16000)
    if fig_name:
        path_ = f"figures/{fig_name}.tiff"
        fig.savefig(
            path_,
            format="tiff",
            dpi=300,
            bbox_inches="tight",
        )
