# %%
import pandas as pd
import xarray as xr
import numpy as np
import lightgbm as lgbm
import random
import joblib
import os
import math

from flaml.model import LGBMEstimator
from flaml import AutoML
from flaml import tune

from score import Score
from mypath import *
from const import *
from utils import *

# plot figures and tables used in the paper
from figs_single_alg import *
from figs_double_alg import *

from tabs import *


class GPULGBM(LGBMEstimator):
    def __init__(self, **config):
        super().__init__(device="gpu", **config)


class Deweather(object):
    rate_train = 0.8
    rate_test = 0.2

    settings = {
        "time_budget": 60 * 60 * 6.5,  # total running time in seconds 60 * 60 *10
        "metric": "rmse",  # primary metrics for regression can be chosen from: ['mae','mse','r2']
        "task": "regression",  # task type
        "seed": 7654321,  # random seed
        "custom_hp": {
            "gpu_lgbm": {
                "log_max_bin": {
                    "domain": tune.lograndint(lower=3, upper=7),
                    "init_value": 5,
                },
            }
        },
    }

    def __init__(
        self, cams_reals_nc, cams_fc_nc, era5_nc, s5p_nc, pop_nc, s5p_alg
    ) -> None:
        self.s5p_alg = s5p_alg
        self.model_path = MODEL_PATH.replace(".pkl", f"_v{s5p_alg}.pkl")

        self.cams = None
        self.era5 = None
        self.s5p = None
        self.pop = None

        self.list_geo = None
        self.train_geo = None
        self.test_geo = None

        self.train_2019 = pd.DataFrame()
        self.test_2019 = pd.DataFrame()

        self.model_config = None
        self.model = None
        self.dw_ds = None

        self.ds_adm2 = None
        self.ds_war = None
        self.ds_ppl = None

        self.load_data(cams_reals_nc, cams_fc_nc, era5_nc, s5p_nc, pop_nc)
        self.cal_rh()
        self.extract_list_geo()

        self.build_model()
        # self.eval_model()

        self.deweather()
        self.clip_to_amd_and_ppl()

    def load_data(self, cams_reals_nc, cams_fc_nc, era5_nc, s5p_nc, pop_nc):
        cams_fc = xr.open_dataset(cams_fc_nc) * 1e9
        cams_fc = cams_fc.rename(name_dict={list(cams_fc.keys())[0]: "no2"})
        self.cams = xr.concat([xr.open_dataset(cams_reals_nc), cams_fc], dim="time")
        self.cams = self.cams.rename(name_dict={list(self.cams.keys())[0]: "cams_no2"})

        self.era5 = xr.open_dataset(era5_nc)

        s5p = xr.open_dataset(s5p_nc)
        self.s5p = s5p.rename(name_dict={list(s5p.keys())[0]: S5P_OBS_COL})

        self.s5p = self.s5p.isel(band=0) if self.s5p_alg == 1 else self.s5p

        pop = xr.open_dataset(pop_nc)
        self.pop = pop.rename(name_dict={list(pop.keys())[0]: "pop"})
        self.pop = self.pop.isel(band=0)

    def cal_rh(self):
        beta = 17.625
        lamda = 243.04
        e = math.e

        dp = self.era5["d2m"] - 272.15
        t = self.era5["t2m"] - 272.15

        self.era5["relative humidity"] = 100 * (
            (e ** ((beta * dp) / (lamda + dp))) / (e ** ((beta * t) / (lamda + t)))
        )

    def reform(self, year, list_geo):
        sd = np.datetime64(f"{year}-02-01T00:00:00.000000000")
        ed = np.datetime64(f"{year}-07-31T00:00:00.000000000")

        # cams_year = self.cams.sel(time=slice(sd, ed))
        # era5_year = self.era5.sel(time=slice(sd, ed))
        s5p_year = self.s5p.sel(time=slice(sd, ed))

        s5p_dates = s5p_year.time.values

        cams_year = self.cams.sel(time=self.cams.time.isin(s5p_dates))
        era5_year = self.era5.sel(time=self.era5.time.isin(s5p_dates))

        julian_time = pd.DatetimeIndex(cams_year.time.values).to_julian_date()
        dow = pd.DataFrame(cams_year.time.values)[0].dt.dayofweek.values
        doy = pd.DataFrame(cams_year.time.values)[0].dt.dayofyear.values

        df = []
        df_pop = self.pop.to_dataframe()

        for t in range(0, len(julian_time)):
            df_cams_t = cams_year.isel(time=t).to_dataframe()[CAMS_COLS]
            df_era5_t = era5_year.isel(time=t).to_dataframe()[ERA5_COLS]
            df_s5p_t = s5p_year.isel(time=t).to_dataframe()[S5P_COLS]

            df_t = pd.concat(
                [
                    df_cams_t.loc[list_geo],
                    df_era5_t.loc[list_geo],
                    df_s5p_t.loc[list_geo],
                    df_pop.loc[list_geo],
                ],
                axis=1,
            )

            df_t["time"] = [cams_year.time.values[t]] * len(list_geo)
            df_t["dow"] = [dow[t]] * len(list_geo)
            df_t["doy"] = [doy[t]] * len(list_geo)
            df_t["lat"] = [x[0] for x in list_geo]
            df_t["lon"] = [x[1] for x in list_geo]

            df.append(df_t)

        org_data = pd.concat(df, ignore_index=True).drop(
            columns=["band", "spatial_ref"]
        )

        return org_data

    def extract_list_geo(self):
        daily_ds = self.cams.isel(time=0).to_dataframe()
        daily_ds = daily_ds.dropna()

        lat = list(daily_ds.index.get_level_values(0))
        lon = list(daily_ds.index.get_level_values(1))

        self.list_geo = list(zip(lat, lon))

    def extract_train_test(self):
        list_geo = self.list_geo.copy()
        self.test_geo = random.sample(list_geo, int(len(list_geo) * self.rate_test))

        [list_geo.remove(x) for x in self.test_geo]
        # train = random.sample(list_geo, int(len(list_geo) * self.rate_train))
        # self.train_geo = [x for x in train if x not in self.test_geo]
        self.train_geo = list_geo
        print("train/test samples: ", len(self.train_geo), len(self.test_geo))

        train_2019 = self.reform(2019, self.train_geo).dropna()
        test_2019 = self.reform(2019, self.test_geo).dropna()

        X_train = train_2019.drop(columns=[S5P_OBS_COL, "time"]).values
        X_test = test_2019.drop(columns=[S5P_OBS_COL, "time"]).values

        y_train = train_2019[S5P_OBS_COL].values
        y_test = test_2019[S5P_OBS_COL].values

        return X_train, y_train, X_test, y_test

    def build_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.model_config = joblib.load(self.model_path).get_params()
        else:
            base_cfg = {"task": "predict", "boosting_type": "gbdt", "device": "gpu"}

            train = self.reform(2019, self.list_geo).dropna()
            X = train.drop(columns=[S5P_OBS_COL, "time"])
            y = train[S5P_OBS_COL]

            automl = AutoML()
            automl.add_learner(learner_name="gpu_lgbm", learner_class=GPULGBM)
            self.settings["estimator_list"] = ["gpu_lgbm"]

            automl.fit(X, y, **self.settings)
            print("Best r2: {0:.4g}".format(1 - automl.best_loss))
            plt.barh(automl.feature_names_in_, automl.feature_importances_)

            self.mode_config = dict(base_cfg, **automl.best_config)
            self.model = lgbm.LGBMRegressor(**self.mode_config).fit(X, y)
            joblib.dump(self.model, self.model_path)

    def eval_model(self):
        X_train, y_train, X_test, y_test = self.extract_train_test()

        model = lgbm.LGBMRegressor(**self.model_config).fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        self.test_2019[S5P_PRED_COL] = y_pred
        self.train_2019[S5P_PRED_COL] = y_pred_train

        print("---- Scores on test set ----")
        Score(y_test, y_pred)
        print("---- Scores on train set ----")
        Score(y_train, y_pred_train)

    def deweather(self):
        # to df
        df_dicts = {
            2019: self.reform(2019, self.list_geo),
            2020: self.reform(2020, self.list_geo),
            2021: self.reform(2021, self.list_geo),
            2022: self.reform(2022, self.list_geo),
        }
        # norm
        cols = [S5P_PRED_COL, S5P_OBS_COL]
        xr_ds = []
        for y in df_dicts.keys():
            df_pred = df_dicts[y].drop(columns=[S5P_OBS_COL, "time"]).values
            df_dicts[y][S5P_PRED_COL] = self.model.predict(df_pred)
            df_dicts[y][cols] = df_dicts[y][cols] * 1e6

            dw_ds = df_dicts[y].set_index(["time", "lat", "lon"]).to_xarray()
            dw_ds[OBS_BAU_COL] = (
                (dw_ds[S5P_OBS_COL] - dw_ds[S5P_PRED_COL]) * 100 / dw_ds[S5P_PRED_COL]
            )
            xr_ds.append(dw_ds)

        dw_ds = xr.concat(xr_ds, dim="time")
        self.dw_ds = write_crs(dw_ds)

    def clip_to_amd_and_ppl(self):
        def clip(ds, bound, crs, list_city, adm_col):
            ds_city = {}
            for city in list_city:
                geo = bound.loc[bound[adm_col] == city].geometry
                ds_city[city] = ds.rio.clip(geo, crs)
            return ds_city

        b1 = gpd.read_file(UK_SHP_ADM1)
        b2 = gpd.read_file(UK_SHP_ADM2)

        ppl_b = gpd.read_file(UK_COAL_SHP)
        border_b = gpd.read_file(CHECKPOINT_SHP)
        border_b["geometry"] = border_b.geometry.buffer(0.2, cap_style=3)
        crs = b2.crs

        list_adm2 = b2[ADM2_COL].values
        list_ppl = ppl_b[PPL_NAME_COL].values

        # self.ds_war = clip(self.dw_ds, b1, crs, WAR_ADMS, ADM1_COL)
        # self.ds_adm2 = clip(self.dw_ds, b2, crs, list_adm2, ADM2_COL)
        # self.ds_ppl = clip(self.dw_ds, ppl_b, crs, list_ppl, PPL_NAME_COL)

        # self.ds_cp = clip(self.dw_ds, b1, crs, LIST_PROV, ADM1_COL)


def met_norm(s5p_alg):
    s5p_no2_nc = S5P_NO2_RPRO_NC if s5p_alg == 2 else S5P_NO2_GEE_NC
    return Deweather(
        CAM_REALS_NO2_NC, CAM_FC_NO2_NC, ERA5_NC, s5p_no2_nc, POP_NC, s5p_alg
    )


if __name__ == "__main__":
    dsv1 = met_norm(1)
    dsv2 = met_norm(2)

    # plt single alg
    # plt_line_ts_adm(dsv1.ds_ppl, all_ppl)
    # plt_line_ts_adm(dsv1.ds_adm2, BOR_ADMS)
    # plt_line_ts_adm(dsv1.ds_adm2, ADM2_CITIES)
    # bounds_v1, _ = cal_change_covid(dsv1)
    # bounds_v2, _ = cal_change_covid(dsv2)

    # plt_scatter_map_covid_2alg(bounds_v1, bounds_v2)
    # plt_line_ts_adm_2alg(dsv1.ds_ppl, dsv2.ds_ppl, DICT_PPLS)
    # plt_line_ts_adm_2alg(dsv1.ds_adm2, dsv2.ds_adm2, ADM2_DICT_CITIES)

    # plot checkpoint
    # plt_line_ts_adm_2alg(dsv1.ds_cp, dsv2.ds_cp, BOR_DICT_PROV)
#     plt_scatter_war(dsv1)
#     plt_scatter_war(dsv2)
# 1

# %%
