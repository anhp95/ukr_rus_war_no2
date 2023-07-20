import matplotlib.pyplot as plt
import xarray as xr
import geopandas as gpd
import seaborn as sns
import numpy as np
import calendar
import copy

from utils import *
from const import *


def cal_change_covid(ds, mode="covid"):
    bounds, _ = get_bound_pop()
    sel_cols = [ADM2_COL]
    for p in DATE_2020.keys():
        obs_bau_m = []
        obs_bau_std = []

        obs_dif_m = []
        obs_dif_std = []

        sd20, ed20 = DATE_2020[p]
        sd19, ed19 = to_d19(sd20), to_d19(ed20)

        if mode == "war":
            sd22, ed22 = sd20.replace("2020", "2022"), ed20.replace("2020", "2022")
            sd20, ed20 = sd22, ed22

        for city in bounds[ADM2_COL].values:
            city20 = ds.ds_adm2[city].sel(time=slice(sd20, ed20))
            city19 = ds.ds_adm2[city].sel(time=slice(sd19, ed19))
            # obs_bau dif
            obs_bau_m.append(city20[OBS_BAU_COL].mean(skipna=True).item())
            obs_bau_std.append(city20[OBS_BAU_COL].mean("time").std(skipna=True).item())
            # obs dif
            obs20 = city20[S5P_OBS_COL].mean("time", skipna=True)
            obs19 = city19[S5P_OBS_COL].mean("time", skipna=True)
            obs_change = (obs20 - obs19) * 100 / obs19
            obs_dif_m.append(obs_change.mean().item())
            obs_dif_std.append(obs_change.std().item())
        cn_m = [f"{OBS_BAU_COL} {p}", f"2020-2019 {p}"]
        cn_std = [f"OBS_BAU {p}_std", f"2020-2019 {p}_std"]
        cvs = [obs_bau_m, obs_dif_m, obs_bau_std, obs_dif_std]
        sel_cols += cn_m
        for cn, cv in zip(cn_m + cn_std, cvs):
            bounds[cn] = cv

    return bounds, sel_cols


def cal_change_war_point(ds):
    conflict_df = prep_conflict_df()
    war_df = conflict_df.loc[conflict_df[EVENT_COL].isin(EVENTS)]
    b1 = gpd.read_file(UK_SHP_ADM1)
    city_dfs = {}
    for city in WAR_ADMS:
        dfs = []
        gc = b1.loc[b1[ADM1_COL] == city].geometry
        city_war = gpd.clip(war_df, gc)
        for m in range(2, 13):
            print(m)
            _, ed = monthrange(2022, m)
            sd = 24 if m == 2 else 1
            for d in range(sd, ed):
                d22, d19 = f"2022-{m}-{d}", f"2019-{m}-{d}"
                try:
                    ds22 = ds.dw_ds.sel(time=d22)
                    ds19 = ds.dw_ds.sel(time=d19)
                    ob = write_crs(ds22[OBS_BAU_COL])

                    os22, os19 = ds22[S5P_OBS_COL], ds19[S5P_OBS_COL]
                    os_change = write_crs((os22 - os19) * 100 / os19)

                    df_d = city_war.loc[city_war["DATETIME"] == d22]
                    ob_dt, os_change_dt = [], []
                    for l in range(len(df_d)):
                        g = df_d.iloc[[l]].geometry
                        try:
                            ob_dt.append(ob.rio.clip(g).mean().item())
                            os_change_dt.append(os_change.rio.clip(g).mean().item())
                        except:
                            ob_dt.append(np.nan)
                            os_change_dt.append(np.nan)
                    df_d["OBS_BAU"] = ob_dt
                    df_d["OBS_CHANGE"] = os_change_dt
                    dfs.append(df_d)
                except:
                    pass

        fdf = pd.concat(dfs, ignore_index=True)
        fdf = fdf.dropna(subset=["OBS_BAU", "OBS_CHANGE"])
        mask = (fdf["OBS_CHANGE"] < 500) & (fdf["OBS_CHANGE"] > -500)
        fdf = fdf.loc[mask]
        fdf = fdf[["OBS_BAU", "OBS_CHANGE", EVENT_COL]]
        city_dfs[city] = fdf

    # cal stats
    # df_stats = []
    # for city in city_dfs.keys():
    #     m = city_dfs[city].groupby("Event type").mean().round(1)
    #     std = city_dfs[city].groupby("Event type").std().round(1)
    #     for col in m.columns:
    #         m[col] = [f"{m} ({std})" for m, std in zip(m[col].values, std[col].values)]
    #     m = m.T
    #     m["Province"] = [city] * len(m)
    #     m["Estimate type"] = m.index
    #     df_stats.append(m)
    # stats = pd.concat(df_stats, ignore_index=True)
    # stats.to_csv("data/result_stats/war_point.csv")
    return city_dfs


def cal_change_war(ds):
    cf = get_monthly_conflict()
    list_city = cf[ADM2_COL].values
    sel_cols = [ADM2_COL]
    # extr data to plot
    for y in [2020, 2021, 2022]:
        for m in range(2, 13):
            obs_bau_m, obs_bau_std = [], []
            obs_dif_m, obs_dif_std = [], []

            sd = 24 if m == 2 else 1
            _, ed = monthrange(y, m)
            ed = 28 if ed == 29 and m == 2 else ed
            sd2x, ed2x = f"{y}-{m}-{sd}", f"{y}-{m}-{ed}"
            sd19, ed19 = f"2019-{m}-{sd}", f"2019-{m}-{ed}"

            for city in list_city:
                city_ds = ds.ds_adm2[city]

                ds22 = city_ds.sel(time=slice(sd2x, ed2x))
                ds19 = city_ds.sel(time=slice(sd19, ed19))

                obs22 = ds22[S5P_OBS_COL].mean("time", skipna=True)
                obs19 = ds19[S5P_OBS_COL].mean("time", skipna=True)
                obsbau = ds22[OBS_BAU_COL].mean("time", skipna=True)

                obsc = (obs22 - obs19) * 100 / obs19

                obs_bau_m.append(obsbau.mean().item())
                obs_bau_std.append(obsbau.std().item())

                obs_dif_m.append(obsc.mean().item())
                obs_dif_std.append(obsc.std().item())
            cn_m = [f"OBS_BAU_{m}_{y}", f"OBS_CHANGE_{m}_{y}"]
            cn_std = [f"OBS_BAU_{m}_std_{y}", f"OBS_CHANGE_{m}_std_{y}"]
            cvs = [obs_bau_m, obs_dif_m, obs_bau_std, obs_dif_std]
            sel_cols += cn_m
            for cn, cv in zip(cn_m + cn_std, cvs):
                cf[cn] = cv

    return cf, sel_cols


def tab4_decor(df_org):
    df = copy.deepcopy(df_org)
    cols = ["OBS_BAU", "OBS_CHANGE"]
    df = df.loc[df[ADM2_COL].isin(ADM2_CITIES)]
    sel_cols = [ADM2_COL]
    for col in cols:
        for y in [2020, 2021, 2022]:
            col_m, col_std = f"{col}_{y}", f"{col}_{y}_std"
            df[col_m] = df[[f"{col}_{m}_{y}" for m in range(2, 13)]].mean(axis=1)
            df[col_std] = df[[f"{col}_{m}_std_{y}" for m in range(2, 13)]].mean(axis=1)
            sel_cols = sel_cols + [col_m]

    return df.round(1), sel_cols


def tab_decor_mstd(df, cols, title):
    def add_std(x, c):
        print(x)
        std_c = c + "_std"
        m, std = round(x[c], 1), round(x[std_c], 1)
        return f"{m} ({std})"

    # do it for all cols except adm name col
    decored_df = copy.deepcopy(df)
    for c in cols[1:]:
        decored_df[c] = decored_df.apply(lambda x: add_std(x, c), axis=1)

    final = decored_df[cols].round(1)
    final.to_csv(f"./data/result_stats/{title}.csv")
    return final


def cal_change_covidtime(dsv1, dsv2):
    df_cv1, cv_sel_cols1 = cal_change_covid(dsv1)
    df_war_cvtime1, war_cvtime_sel_cols1 = cal_change_covid(dsv1, mode="war")

    df_cv2, cv_sel_cols2 = cal_change_covid(dsv2)
    df_war_cvtime2, war_cvtime_sel_cols2 = cal_change_covid(dsv2, mode="war")

    tab_decor_mstd(df_cv1, cv_sel_cols1, "covid_v1")
    tab_decor_mstd(df_cv2, cv_sel_cols2, "covid_v2")

    tab_decor_mstd(df_war_cvtime1, war_cvtime_sel_cols1, "war_covidtime_v1")
    tab_decor_mstd(df_war_cvtime2, war_cvtime_sel_cols2, "war_covidtime_v2")
