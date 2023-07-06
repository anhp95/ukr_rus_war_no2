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


def plt_scatter_map_covid_2alg(df_1, df_2):
    nrs, ncs = 2, 2
    w, h = 5 * ncs, 4 * nrs
    bound_lv1 = gpd.read_file(UK_SHP_ADM1)
    index_fig = [["a1", "b1"], ["a2", "b2"]]
    huem = 35

    fig1, axes1 = plt.subplots(nrs, ncs, figsize=(w, h), layout="constrained")
    fig2, axes2 = plt.subplots(nrs, ncs, figsize=(w, h), layout="constrained")

    for p, fig, axes in zip(DATE_2020.keys(), [fig1, fig2], [axes1, axes2]):
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
                g.set(title=f"({index_fig[i][j]}) {v} - {col} {p}")
                ax.get_legend().remove()
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


def plt_line_ts_adm_2alg(ds_1, ds_2, dict_city):
    list_city = dict_city.keys()
    nrs, ncs = len(list_city), 4
    w, h = 2.3 * ncs, 2 * nrs
    # fig, axes = plt.subplots(nrs, ncs, figsize=(w, h), layout="constrained")

    fig = plt.figure(layout="constrained", figsize=(w, h))
    subfigs = fig.subfigures(nrs, 1, wspace=0.07)

    for i, city in enumerate(list_city):
        subfigs[i].suptitle(city, fontsize="x-large")
        axes = subfigs[i].subplots(1, ncs, sharey=True)
        for j, (ds, v) in enumerate(zip([ds_1, ds_2], S5P_VERSIONS)):
            df_city = ds[city].mean(["lat", "lon"], skipna=True).to_dataframe()
            for k, y in enumerate([2021, 2022]):
                df = get_nday_mean(df_city[f"{y}-02-15":f"{y}-05-31"], 3)
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
                ax.tick_params(axis="both", which="major", labelsize=11)
                ax.set_xticklabels(["", "Mar", "", "May", "", "Jul", ""])
                ax.set_xlabel("")
                if i != len(list_city) - 1:
                    ax.set_xticklabels(["", "", "", "", "", "", ""])
                try:
                    handles, labels = ax.get_legend_handles_labels()
                    handles = handles + [cl] + [wl]
                    labels = labels + ["COVID-19 Lockdown"] + ["War start date"]
                    fig.legend(
                        handles,
                        labels,
                        ncol=4,
                        loc="upper center",
                        bbox_to_anchor=(0.5, -0.01),
                        fontsize=12,
                    )
                except:
                    pass
