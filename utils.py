# %%

import xarray as xr
import rioxarray as rioxr  # conflcict package seaborn conda remove seaborn
import cfgrib
import geopandas as gpd
import geopandas as gpd
import pandas as pd
import numpy as np
import calendar

from calendar import monthrange
from shapely.geometry import mapping
from shapely.geometry import Point

from mypath import *
from const import *


def get_bound_pop():
    city_pop_df = pd.read_csv(CITY_POP)

    gdf = gpd.read_file(UK_SHP_ADM2)

    merge_df = pd.merge(city_pop_df, gdf, on=ADM2_COL, how="inner")
    merge_df = gpd.GeoDataFrame(merge_df, crs=gdf.crs, geometry=merge_df.geometry)

    return merge_df[[ADM2_COL, "Population", "geometry"]], gdf.crs


def write_crs(ds):
    ds = ds.rio.write_crs("epsg:4326", inplace=True)
    return ds.rio.set_spatial_dims("lon", "lat", inplace=True)


def to_d19(d):
    return d.replace("2020", "2019")


def get_bound(shp_file=UK_SHP_ADM0):
    geodf = gpd.read_file(shp_file)
    geometry = geodf.geometry.apply(mapping)
    crs = geodf.crs

    return geometry, crs


def read_tif(tif_file):
    rio_ds = rioxr.open_rasterio(tif_file)
    return rio_ds.rename({"x": "lon", "y": "lat"})


def read_grib(grib_file):
    grib_ds = cfgrib.open_dataset(grib_file)
    return grib_ds.rename({"longitude": "lon", "latitude": "lat"})


def prep_s5p_ds(mode="gee"):
    nc_file = S5P_NO2_GEE_NC if mode == "gee" else S5P_NO2_RPRO_NC
    org_ds = xr.open_dataset(nc_file) * 1e6
    var_name = list(org_ds.keys())[0]
    org_ds = org_ds.rename(name_dict={var_name: S5P_OBS_COL})
    org_ds = org_ds.rio.write_crs("epsg:4326", inplace=True)
    org_ds = org_ds.rio.set_spatial_dims("lon", "lat", inplace=True)
    return org_ds


def prep_location_df(df):
    return gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["LONGITUDE"], df["LATITUDE"])
    )


def prep_fire_df():
    gdf = prep_location_df(pd.read_csv(FIRE_WARTIME_CSV))
    gdf["DATETIME"] = pd.to_datetime(gdf["DATETIME"], format="%m/%d/%Y")

    return gdf


def prep_conflict_df():
    df = pd.read_excel(CONFLICT_XLS, engine="openpyxl")
    df["DATETIME"] = pd.DatetimeIndex(df.EVENT_DATE.values)
    df["MONTH"] = df["DATETIME"].dt.month

    df = df[(df["DATETIME"] > "2022-1-1") & (df["DATETIME"] < "2023-1-1")]
    df[EVENT_COL] = df[EVENT_COL].apply(lambda x: x if x in EVENTS else "Other")
    return prep_location_df(df)


def get_nday_mean(df, nday=3):
    df["time"] = df.index
    time = df["time"].groupby(np.arange(len(df)) // nday).mean()

    df = df.groupby(np.arange(len(df)) // nday).mean()
    df["time"] = time
    return df.set_index("time")


# def get_boundary_cities():
#     bound_lv2 = gpd.read_file(UK_SHP_ADM2)
#     boundary = bound_lv2.loc[bound_lv2["ADM2_EN"].isin(LIST_BOUNDARY_CITY)]
#     return boundary


def get_monthly_conflict():
    cf_df = prep_conflict_df()

    b = gpd.read_file(UK_SHP_ADM2)
    list_city = b[ADM2_COL].values
    y = 2022

    for m in range(2, 8):
        mc = []
        sd = 24 if m == 2 else 1
        _, ed = monthrange(y, m)
        sd, ed = f"{y}-{m}-{sd}", f"{y}-{m}-{ed}"

        mask = (cf_df["DATETIME"] > sd) & (cf_df["DATETIME"] <= ed)
        m_df = cf_df.loc[mask]

        for city in list_city:
            geo = b.loc[b[ADM2_COL] == city].geometry
            mc.append(len(gpd.clip(m_df, geo)))
        b[f"cf_{m}"] = mc

    return b


# %%
