#%%

import ee

ee.Authenticate()
ee.Initialize()

#%%

import pandas as pd
from calendar import monthrange

UK_BOUND = ee.Geometry.Polygon(
    [
        22.136912,
        52.379291,
        40.227883,
        52.379291,
        40.227883,
        44.386411,
        22.136912,
        44.386411,
    ]
)
SCALE = 1113.2
FOLDER = "UKR_RUS_NO2"

YEARS = [2019, 2020, 2021, 2022]
MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

NO2_BAND = "tropospheric_NO2_column_number_density"


def export2drive(img, id):
    task = ee.batch.Export.image.toDrive(
        **{
            "image": img,
            "description": id,
            "folder": FOLDER,
            "scale": SCALE,
            "region": UK_BOUND,
        }
    )
    task.start()


def to_julian_date(year, month, day):
    ts = pd.Timestamp(year=year, month=month, day=day)
    return ts.to_julian_date()


def download_no2():
    for y in YEARS:
        for m in MONTHS:
            _, ed = monthrange(y, m)
            for dix in range(1, ed + 1):
                jd = to_julian_date(y, m, dix)
                print(jd)
                img = (
                    ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
                    .filter(ee.Filter.eq("TIME_REFERENCE_JULIAN_DAY", jd))
                    .mosaic()
                    .select(NO2_BAND)
                    .clip(UK_BOUND)
                )
                export2drive(img, str(jd))


def download_pop():
    pop = (
        ee.ImageCollection("WorldPop/GP/100m/pop")
        .filter(ee.Filter.eq("country", "UKR"))
        .filter(ee.Filter.eq("year", 2020))
        .mosaic()
    )
    export2drive(pop, "uk_pop_2020")


# %%
