#%%
import geopandas as gpd
from datetime import datetime

from mypath import *


def preprocess_fire_count(gpd_df):
    d = gpd_df["ACQ_DATE"].values
    t = gpd_df["ACQ_TIME"].values
    dt = []
    month = []

    for di, ti in zip(d, t):

        dt_string = f"{di} {ti}00"
        dt_obj = datetime.strptime(dt_string, "%Y-%m-%d %H%M%S")
        dt.append(dt_obj)
        month.append(dt_obj.month)

    gpd_df["DATETIME"] = dt
    gpd_df["month"] = month

    return gpd_df


def shp2csv():
    gpd_2020 = preprocess_fire_count(gpd.read_file(FIRE_2020_SHP))
    gpd_2020.to_csv(FIRE_2020_SHP.replace("shp", "csv"))

    gpd_2021 = preprocess_fire_count(gpd.read_file(FIRE_2021_SHP))
    gpd_2021.to_csv(FIRE_2021_SHP.replace("shp", "csv"))

    gpd_wartime = preprocess_fire_count(gpd.read_file(FIRE_WARTIME_SHP))
    gpd_wartime.to_csv(FIRE_WARTIME_SHP.replace("shp", "csv"))


def shp2geosjon(shp_path):
    uk_shp_adm1 = gpd.read_file(shp_path)
    uk_shp_adm1.to_file(shp_path.replace("shp", "json"), driver="GeoJSON")


# %%
