# %%
import xarray as xr
import pandas as pd
import numpy as np

from datetime import date

from mypath import *
from const import *
from utils import *


class Predictor:
    def __init__(self, filename) -> None:
        self.org_ds = None
        self.years = []
        self.file_name = filename

        self.flag_group_date = bool()
        self.flag_sel_war_time = bool()

        self.set_years()

        self.set_flag_group_date()
        self.set_flag_sel_war_time()

        self.extract_ds()
        self.clip_2_bound()
        print("clip ok")
        self.group_by_date()
        # self.select_war_time()
        self.interpolate()
        self.to_nc()

    def set_years(self):
        return

    def set_flag_group_date(self):
        # This will be defined in the ChildClass
        return

    def set_flag_sel_war_time(self):
        # This will be defined in the ChildClass
        return

    def extract_ds(self):
        # This will be defined in the ChildClass
        return

    def clip_2_bound(self):
        self.org_ds = self.org_ds.rio.write_crs("epsg:4326", inplace=True)
        self.org_ds = self.org_ds.rio.set_spatial_dims("lon", "lat", inplace=True)
        geometry, crs = get_bound()

        self.org_ds = self.org_ds.rio.clip(geometry, crs)

    def group_by_date(self):
        if self.flag_group_date:
            # self.org_ds = self.org_ds.sel(
            #     time=self.org_ds.time.dt.hour.isin([9, 10, 11, 12])
            # )
            # 13:30 pm
            self.org_ds = self.org_ds.sel(time=self.org_ds.time.dt.hour.isin([13, 14]))
            self.org_ds = self.org_ds.groupby(self.org_ds.time.dt.date).mean()

            dates = []
            for date in self.org_ds.date.values:
                m = f"0{date.month}" if date.month < 10 else date.month
                d = f"0{date.day}" if date.day < 10 else date.day
                y = date.year
                dates.append(np.datetime64(f"{y}-{m}-{d}T00:00:00.000000000"))
            self.org_ds = self.org_ds.assign_coords({"date": dates})
            self.org_ds = self.org_ds.rename({"date": "time"})

    def select_war_time(self):
        if self.flag_sel_war_time:
            dates = [f"0{i}" if i < 10 else i for i in range(1, 24)]
            list_date = [
                np.datetime64(f"{y}-02-{d}T00:00:00.000000000")
                for y in self.years
                for d in dates
            ]
            self.org_ds = self.org_ds.drop_sel(time=list_date)

    def interpolate(self):
        interp_lat = np.load(CAMS_REALS_LAT_FILE)
        interp_lon = np.load(CAMS_REALS_LON_FILE)

        self.org_ds = self.org_ds.interp(lat=interp_lat, lon=interp_lon)

    def to_nc(self):
        self.org_ds.to_netcdf(self.file_name)


class CAMS_REALS_NO2(Predictor):
    def __init__(self, filename) -> None:
        super().__init__(filename)

    def set_years(self):
        self.years = [2019]

    def set_flag_group_date(self):
        self.flag_group_date = True

    def set_flag_sel_war_time(self):
        self.flag_sel_war_time = True

    def extract_ds(self):
        list_nc_ds = []
        for nc_file in SF_REALS_NO2_2019_FILES:
            nc_ds = xr.open_dataset(nc_file)
            list_nc_ds.append(nc_ds)

        self.org_ds = xr.concat(list_nc_ds, dim="time")


class ERA5(Predictor):
    def __init__(self, filename) -> None:
        super().__init__(filename)

    def set_years(self):
        self.years = [2019, 2020, 2021, 2022]

    def set_flag_group_date(self):
        self.flag_group_date = True

    def set_flag_sel_war_time(self):
        self.flag_sel_war_time = True

    def extract_ds(self):
        list_grib_ds = []
        for grib_file in CLIMATE_FILE:
            grib_ds = read_grib(grib_file)
            list_grib_ds.append(grib_ds)
        self.org_ds = xr.concat(list_grib_ds, dim="time")


class CAMS_FC_NO2(Predictor):
    def __init__(self, filename) -> None:
        super().__init__(filename)

    def set_flag_group_date(self):
        self.flag_group_date = False

    def set_flag_sel_war_time(self):
        self.flag_sel_war_time = False

    def extract_ds(self):
        list_grib_ds = []
        for grib_file in SF_FC_NO2_2020_2022_FILES:
            # grib_ds = read_grib(grib_file).isel(step=slice(9, 13))
            # 13:30 pm
            grib_ds = read_grib(grib_file).isel(step=slice(13, 15))
            grib_ds = grib_ds.mean("step")
            list_grib_ds.append(grib_ds)

        self.org_ds = xr.concat(list_grib_ds, dim="time")
        # self.org_ds = self.org_ds.reset_index("time", drop=True)


class S5P_NO2(Predictor):
    def __init__(self, filename) -> None:
        super().__init__(filename)

    def set_flag_group_date(self):
        self.flag_group_date = False

    def set_flag_sel_war_time(self):
        self.flag_sel_war_time = False

    # extract S5P GEE dataset
    def extract_ds_s5p_gee(self):
        sd = pd.to_datetime(0, unit="s").to_julian_date()
        list_tif_ds = []
        for tif in CL_NO2_GEE_FILES:
            julian_date = float(tif.split("\\")[-1][:-4])

            tif_ds = read_tif(tif)
            tif_ds = tif_ds.expand_dims(
                time=[pd.to_datetime(julian_date - sd, unit="D")]
            )
            list_tif_ds.append(tif_ds)

        self.org_ds = xr.concat(list_tif_ds, dim="time")

    # extract S5P RPRO dataset
    def extract_ds(self):
        years = [2019, 2020, 2021, 2022]
        list_ds = []

        for y in years:
            print(y)
            list_files = glob.glob(os.path.join(NO2_S5P_RPRO_DIR, f"{y}", "*.nc"))
            for f in list_files:
                date = f.split("\\")[-1].split(".")[0]
                ds = xr.open_dataset(f).rename({"longitude": "lon", "latitude": "lat"})
                ds["time"] = [pd.to_datetime((date))]
                ds = ds.rename({"tropospheric_NO2_column_number_density": S5P_OBS_COL})
                list_ds.append(ds[S5P_OBS_COL])

        self.org_ds = xr.concat(
            list_ds,
            dim="time",
        )


class Pop(Predictor):
    def __init__(self, filename) -> None:
        super().__init__(filename)

    def set_flag_group_date(self):
        self.flag_group_date = False

    def set_flag_sel_war_time(self):
        self.flag_sel_war_time = False

    def extract_ds(self):
        self.org_ds = read_tif(POP_FILE)


# %%
if __name__ == "__main__":
    cams_reals_no2 = CAMS_REALS_NO2(CAM_REALS_NO2_NC)
    cams_fc_no2 = CAMS_FC_NO2(CAM_FC_NO2_NC)
    era5 = ERA5(ERA5_NC)
    s5p_no2 = S5P_NO2(S5P_NO2_RPRO_NC)
    # pop = Pop(POP_NC)

# %%
s5p_no2_nc = "data/preprocessed/s5p_no2.nc"
plot_s5p_no2_year(s5p_no2_nc)
# %%
