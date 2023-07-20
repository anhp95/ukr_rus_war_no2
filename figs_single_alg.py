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
import matplotlib.lines as mlines
import datashader as dsh

from metpy.units import units
from calendar import monthrange

from datashader.mpl_ext import dsshow
from sklearn.metrics import r2_score
from scipy import stats

from utils import *
from const import *
from tabs import *


def plt_scatter_map_covid(df_change):
    nrs, ncs = 2, 2
    w, h = 5 * ncs, 4 * nrs
    fig, axes = plt.subplots(nrs, ncs, figsize=(w, h), layout="constrained")
    bound_lv1 = gpd.read_file(UK_SHP_ADM1)
    index_fig = [["a", "b"], ["c", "d"]]
    huem = 35
    for i, p in enumerate(DATE_2020.keys()):
        for j, col in enumerate(["2020-2019", OBS_BAU_COL]):
            bound_lv1.plot(ax=axes[i, j], facecolor="white", edgecolor="black", lw=0.7)
            g = sns.scatterplot(
                data=df_change,
                x=df_change.centroid.x,
                y=df_change.centroid.y,
                hue=f"{col} {p}",
                hue_norm=(-1 * huem, huem),
                size="Population",
                sizes=(50, 500),
                palette=CMAP_NO2,
                ax=axes[i, j],
                edgecolor="black",
                linewidth=1,
            )
            g.set(title=f"({index_fig[i][j]}) {col} {p}")
            axes[i, j].get_legend().remove()
            h, l = g.get_legend_handles_labels()
            l = [f"{li}M" if li != "Population" else li for li in l]
            legend = axes[i, j].legend(
                h[-6:],
                l[-6:],
                bbox_to_anchor=(0, 0),
                loc="lower left",
                borderaxespad=0.0,
                # fontsize=13,
                edgecolor="black",
            )
            legend.get_frame().set_alpha(None)
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


def plt_line_ts_adm(ds, list_city):
    nrs, ncs = len(list_city), 4
    w, h = 5 * ncs, 3 * nrs
    fig, axes = plt.subplots(nrs, ncs, figsize=(w, h), layout="constrained")

    for i, city in enumerate(list_city):
        df_city = ds[city].mean(["lat", "lon"], skipna=True).to_dataframe()

        for j, y in enumerate([2019, 2020, 2021, 2022]):
            df = get_nday_mean(df_city[f"{y}-02-01":f"{y}-07-31"], 5)
            ax = axes[i, j]
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
            ax.set_title(f"{city} - {y}", size=15)
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
            # ax.tick_params(axis='both', which='major', labelsize=15)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles = handles + [cl] + [wl]
    labels = labels + ["COVID-19 Lockdown"] + ["War start date"]
    fig.legend(
        handles,
        labels,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        fontsize=25,
    )


def plot_obs_year():
    org_ds = prep_s5p_ds()
    bound_lv1 = gpd.read_file(UK_SHP_ADM1)
    years = [2019, 2020, 2021, 2022]
    nrs, ncs = 4, 12
    w, h = 4 * ncs, 4 * nrs
    fig_obs, ax_obs = plt.subplots(nrs, ncs, figsize=(w, h), layout="constrained")
    fig_count, ax_count = plt.subplots(nrs, ncs, figsize=(w, h), layout="constrained")

    for i, y in enumerate(years):
        for m in range(1, 13):
            _, ed = monthrange(y, m)

            s5p_no2_year = org_ds.sel(time=org_ds.time.dt.year.isin([y]))
            s5p_month = s5p_no2_year.sel(time=slice(f"{y}-{m}-01", f"{y}-{m}-{ed}"))
            s5p_month = s5p_month[S5P_OBS_COL][:, 0, :, :]
            s5p_mean = s5p_month.mean("time")

            s5p_count = s5p_month.notnull().sum("time")
            s5p_count = s5p_count.where(s5p_count > 0)

            cb_mean = s5p_mean.plot(
                cmap="YlOrRd",
                vmin=10,
                vmax=100,
                ax=ax_obs[i, m - 1],
            )
            cb_mean.colorbar.remove()

            cb_count = s5p_count.plot(
                vmin=1,
                vmax=30,
                ax=ax_count[i, m - 1],
            )
            cb_count.colorbar.remove()

            for axi in [ax_obs, ax_count]:
                axi[i, m - 1].set_title(f"{y}- {calendar.month_name[m]}", fontsize=30)
                axi[i, m - 1].set_ylabel("")
                axi[i, m - 1].set_xlabel("")

    cb_count = fig_count.colorbar(
        cb_count,
        ax=ax_count,
        shrink=0.4,
        extend="both",
        location="bottom",
    )
    cb_count.set_label("Number of observations", size=30)
    cb_count.ax.tick_params(labelsize=20)

    cb_mean = fig_obs.colorbar(
        cb_mean,
        ax=ax_obs,
        shrink=0.4,
        extend="both",
        location="bottom",
    )
    cb_mean.set_label(NO2_UNIT, size=30)
    cb_mean.ax.tick_params(labelsize=30)


def plt_met_dist(ds, var="blh"):
    u10 = ds.era5["u10"]
    v10 = ds.era5["v10"]
    ds.era5["wind"] = np.sqrt(u10**2 + v10**2)

    var_label_dict = {
        "wind": "Wind speed (m/s)",
        "blh": "Boundary layer height (m)",
    }
    ylabel = "Relative Frequency (%)"
    # var = "blh"

    for i, p in enumerate(DATE_2020.keys()):
        nrs, ncs = 1, 1
        w, h = 5 * ncs, 4 * nrs
        figure, axes = plt.subplots(nrs, ncs, figsize=(w, h), layout="constrained")

        sd20, ed20 = DATE_2020[p]
        sd19, ed19 = to_d19(sd20), to_d19(ed20)

        ds20 = ds.era5.sel(time=slice(sd20, ed20))
        ds19 = ds.era5.sel(time=slice(sd19, ed19))

        ts20 = ds20[var].values.reshape(-1)
        ts20 = ts20[~np.isnan(ts20)]
        ws20 = np.ones_like(ts20) * 100 / ts20.size

        ts19 = ds19[var].values.reshape(-1)
        ts19 = ts19[~np.isnan(ts19)]
        ws19 = np.ones_like(ts19) * 100 / ts19.size

        axes.hist(
            ts20,
            bins=15,
            weights=ws20,
            ec="#1b9e77",
            fc="None",
            histtype="step",
            label=f"2020",
        )
        axes.hist(
            ts19,
            bins=15,
            weights=ws19,
            ec="#e7298a",
            fc="None",
            histtype="step",
            label=f"2019",
        )
        axes.legend()
        axes.set_title(f"{p}", fontsize=15)
        axes.set_xlabel(var_label_dict[var])
        axes.set_ylabel(ylabel)
        axes.legend(loc="upper right")


def plt_wind_rose(ds, year):
    wind_var, u10_var, v10_var = "wind", "u10", "v10"

    bins = [i for i in range(0, 7)]

    u10, v10 = ds.era5[u10_var], ds.era5[v10_var]
    ds.era5[wind_var] = np.sqrt(u10**2 + v10**2)
    nrs, ncs = 1, 2
    fig = plt.figure(figsize=(nrs * 10, ncs * 10), constrained_layout=True)

    for i, p in enumerate(DATE_2020.keys()):
        sd, ed = DATE_2020[p]
        if year == 2019:
            sd, ed = to_d19(sd), to_d19(ed)
        dsw = ds.era5.sel(time=slice(sd, ed))[[wind_var, u10_var, v10_var]]

        wf, uf, vf = [], [], []

        wf = np.concatenate((wf, dsw[wind_var].values.reshape(-1)), axis=None)
        uf = np.concatenate((uf, dsw[u10_var].values.reshape(-1)), axis=None)
        vf = np.concatenate((vf, dsw[v10_var].values.reshape(-1)), axis=None)

        wf = wf[~np.isnan(wf)]
        uf = uf[~np.isnan(uf)]
        vf = vf[~np.isnan(vf)]

        uf = units.Quantity(uf, "m/s")
        vf = units.Quantity(vf, "m/s")

        wind_d = mpcalc.wind_direction(uf, vf)
        ax = fig.add_subplot(nrs, ncs, i + 1, projection="windrose")

        ax.contourf(
            wind_d.magnitude,
            wf,
            normed=True,
            bins=bins,
            lw=3,
            cmap=cm.Spectral_r,
        )
        ax.contour(wind_d.magnitude, wf, normed=True, bins=bins, colors="black")
        ax.set_legend(title="Windspeed: m/s")
        ax.set_title(f"{year} - {p}", fontsize=15)

    return wind_d, wf


def plt_scatter_war(ds, v):
    """
    Plot scatter plot of obs_bau and obs_change values for each reported conflict locations
    """
    # plotting
    index = ["a", "b", "c", "d", "e"]
    clrs = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#666666"]
    plt = {e: clr for e, clr in zip(EVENTS, clrs)}
    city_dfs = cal_change_war_point(ds)
    list_dfs = []

    xc, yc, hc = "OBS_BAU - 2022 (%)", "OBS_CHANGE - 2022-2019 (%)", "Event type"

    for i, city in enumerate(city_dfs.keys()):
        df = city_dfs[city]
        df.columns = [xc, yc, hc]
        list_dfs.append(df)
    war_cities = pd.concat(list_dfs, ignore_index=True)

    p = sns.jointplot(
        war_cities,
        x=xc,
        y=yc,
        kind="hex",
        joint_kws=dict(bins="log"),
    )

    p.fig.suptitle(f"{v} (n={len(war_cities)})", size=15)
    p.ax_joint.axhline(y=0, color="red", linestyle="--")
    p.ax_joint.axvline(x=0, color="red", linestyle="--")
    p.ax_joint.set_xlim(-200, 700)
    p.ax_joint.set_ylim(-400, 600)

    for patch in p.ax_marg_x.patches:
        patch.set_facecolor("#d95f02")

    for patch in p.ax_marg_y.patches:
        patch.set_facecolor("#41b6c4")
    return city_dfs


def plt_war_fire():
    cf_df = prep_conflict_df()
    fire_df = prep_fire_df()

    b1 = gpd.read_file(UK_SHP_ADM1)
    b0 = gpd.read_file(UK_SHP_ADM0)
    cf_df = gpd.clip(cf_df, b0.geometry)
    fire_df = gpd.clip(fire_df, b0.geometry)

    moi = [4, 5, 6, 7]
    moi = [i for i in range(1, 13)]
    nrs, ncs = len(moi), 3
    w, h = 5.5 * ncs, 4 * nrs
    fig, axes = plt.subplots(nrs, ncs, figsize=(w, h), layout="constrained")
    ys, ms = [y for y in range(2021, 2023)], [m for m in moi]
    for j, y in enumerate(ys):
        fire_y = fire_df[fire_df["DATETIME"].dt.year == y]
        for i, m in enumerate(ms):
            b1.plot(ax=axes[i, j], facecolor="None", edgecolor="black", lw=0.2)
            fire_m = fire_y[fire_y["DATETIME"].dt.month == m]
            fire_m.plot(ax=axes[i, j], color="red", markersize=2)
            axes[i, j].set_title(
                f"Fire Spots ({y} - {calendar.month_name[m]})", size=20
            )

    for j, m in enumerate(ms):
        b1.plot(ax=axes[j, 2], facecolor="None", edgecolor="black", lw=0.2)
        cf_m = cf_df[cf_df["DATETIME"].dt.month == m]
        cf_m.plot(ax=axes[j, 2], color="orange", markersize=2)
        axes[j, 2].set_title(
            f"Conflict Spots ({y} - {calendar.month_name[m]})", size=20
        )


def plt_adm2_map_war(df_change):
    b2 = gpd.read_file(UK_SHP_ADM2)
    moi = [3, 5, 6, 7, 8, 10]
    # moi = [i for i in range(1, 13)]

    v = {"OBS_BAU": (-35, 35), "OBS_CHANGE": (-35, 35), "cf": (0, 300)}
    lg = {"orientation": "horizontal", "extend": "both", "shrink": 0.8}
    lbno2, lbcf = r"NO$_{2}$ col. change (%)", "Number of conflict spots"
    lbs = {"OBS_BAU": lbno2, "OBS_CHANGE": lbno2, "cf": lbcf}
    cmap = {"OBS_BAU": CMAP_NO2, "OBS_CHANGE": CMAP_NO2, "cf": CMAP_CONFLICT}
    t = ["2022-2019", "OBS_BAU", "Conflict locations"]
    y = 2022
    nrs, ncs = len(moi), 3
    w, h = 5.5 * ncs, 4 * nrs
    fig, axes = plt.subplots(nrs, ncs, figsize=(w, h), layout="constrained")
    for i, m in enumerate(moi):
        legend = False if i < (len(moi) - 1) else True

        for j, col in enumerate(["OBS_BAU", "OBS_CHANGE", "cf"]):
            ax = axes[i, j]
            vmin, vmax = v[col]
            df_change.plot(
                column=f"{col}_{m}_{y}",
                ax=ax,
                legend=legend,
                cmap=cmap[col],
                vmin=vmin,
                vmax=vmax,
                legend_kwds=dict(lg, **{"label": lbs[col]}),
            )
            b2.plot(ax=ax, facecolor="None", edgecolor="black", lw=0.05)
            ax.set_title(f"{t[j]} ({calendar.month_name[m]})", fontsize=15)


def plot_pred_true(ds, s5p_ver):
    figure, axis = plt.subplots(1, 2, figsize=(13, 5))
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

    axis[0].set_title(
        rf"a{s5p_ver}) Time series trend of OBS NO$_{2}$ and BAU NO$_{2}$"
    )
    axis[0].set_xlabel("Time")
    axis[0].set_ylabel(f"$10^{{{-6}}}$ $mol/m^2$")

    axis[1].set_title(rf"b{s5p_ver}) Scatter Plot of OBS NO$_{2}$ and BAU NO$_{2}$")
    axis[1].set_xlabel(r"OBS NO$_{2}$ ($10^{{{-6}}}$ $mol/m^2$)")
    axis[1].set_ylabel(r"BAU NO$_{2}$ ($10^{{{-6}}}$ $mol/m^2$)")

    start_x = 370 if s5p_ver == 1 else 470
    gap = 15 if s5p_ver == 1 else 20

    axis[1].annotate("n = {}".format(len(ds.test_2019[S5P_OBS_COL])), (start_x, 0))
    axis[1].annotate(
        "N = {}".format(len(ds.test_2019.groupby(["lat", "lon"]).mean())),
        (start_x, gap),
    )
    axis[1].annotate(
        "$R$ = {:.2f}".format(
            stats.pearsonr(ds.test_2019[S5P_OBS_COL], ds.test_2019[S5P_PRED_COL])[0]
        ),
        (start_x, gap * 2),
    )

    line = mlines.Line2D([0, 1], [0, 1], color="red", label="1:1 line")
    transform = axis[1].transAxes
    line.set_transform(transform)
    axis[1].add_line(line)
    axis[1].legend()


# %%
