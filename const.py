import matplotlib.patches as mpatches

UK_BOUND = {
    "min_lat": 44.386411,
    "max_lat": 52.379291,
    "min_lon": 22.136912,
    "max_lon": 40.227883,
}

HOUR_STR = "T00:00:00.000000000"

ADM1_COL = "ADM1_EN"
ADM2_COL = "ADM2_EN"
ADM3_COL = "ADM3_EN"
PPL_NAME_COL = "name"
CHECKPOINT_COL = "Name_ENG"

EVENT_COL = "SUB_EVENT_TYPE"
EVENTS = [
    "Attack",
    "Abduction/forced disappearance",
    "Air/drone strike",
    "Armed clash",
    "Remote explosive/landmine/IED",
    "Shelling/artillery/missile attack",
    "Other",
]

ADM2_DICT_CITIES = {
    "Kyiv": {"min": 0, "max": 100},
    "Kharkivskyi": {"min": 0, "max": 60},
    # "Odeskyi",
    "Dniprovskyi": {"min": 0, "max": 60},
    # "Donetskyi",
    "Zaporizkyi": {"min": 0, "max": 60},
    # "Lvivskyi",
    "Kryvorizkyi": {"min": 0, "max": 60},
    # "Mykolaivskyi",
    # "Sevastopol", # no data
}
ADM2_CITIES = ADM2_DICT_CITIES.keys()
WAR_ADMS = ["Dnipropetrovska", "Donetska", "Kharkivska", "Luhanska", "Zaporizka"]
BOR_DICT_PROV = {
    "Volynska": {"min": 0, "max": 100},
    "Lvivska": {"min": 0, "max": 100},
    "Zakarpatska": {"min": 0, "max": 100},
    "Chernivetska": {"min": 0, "max": 100},
    "Vinnytska": {"min": 0, "max": 100},
    "Odeska": {"min": 0, "max": 100},
}
LIST_PROV = BOR_DICT_PROV.keys()

DICT_PPLS = {
    "Zaporizhia TPP": {"min": 0, "max": 80},
    "Mironivskaya TEC": {"min": 0, "max": 100},
    "Zmiivska power station": {"min": 0, "max": 100},
    "Luhanska": {"min": 0, "max": 100},
    "Sievierodonetsk CHP power station": {"min": 0, "max": 100},
    # "Vuglegirska power station",
}
LIST_PPLS = DICT_PPLS.keys()


LIST_CPS = ["Porubne - Siret", "Luzhanka - Beregsurány", "Uzhhorod - Vyšné Nemecké"]
DICT_CPS = {cp: {"min": 0, "max": 60} for cp in LIST_CPS}

LCKDWN_SD = "2020-04-06"
LCKDWN_ED = "2020-05-10"
BF_SD = "2020-03-01"
BF_ED = "2020-03-15"

DATE_2020 = {
    "Prelockdown": (BF_SD, BF_ED),
    "Lockdown": (LCKDWN_SD, LCKDWN_ED),
}

WAR_SD = "2022-02-24"

NO2_UNIT = f"$10^{{{-6}}}$ $mol/m^2$"

S5P_OBS_COL = r"OBS_S5P"
S5P_PRED_COL = r"BAU_S5P"
OBS_BAU_COL = "OBS_BAU"
OBS_PRED_CHNAGE = r"OBS_BAU_Difference"

ERA5_COLS = ["u10", "v10", "d2m", "t2m", "blh", "z", "relative humidity"]
POP_COLS = ["pop"]
S5P_COLS = [S5P_OBS_COL]
CAMS_COLS = ["cams_no2"]

CMAP_FIRE = "Reds"
CMAP_CONFLICT = "OrRd"
CMAP_NO2 = "RdYlBu_r"
CMAP_WIND = "Spectral_r"

# level 3
ADM3_CITIES = [
    "Kyiv",
    "Kharkivska",
    "Odeska",  #
    "Dniprovska",
    "Donetska",
    "Zaporizka",
    "Lvivska",
    "Kryvorizka",
    "Mykolaivska",  #
    "Mariupolska",
    "Luhanska",  #
    "Vinnytska",  #
    "Simferopolska",
    "Makiivska",
    "Poltavska",
]

FEATURE_NAMES = [
    "Surface NO2",
    "Wind speed (U)",
    "Wind speed (V)",
    "Dewpoint temperature",
    "Temperature",
    "Boundary layer height",
    "Geopotential",
    "Relative humidity",
    "Population",
    "Day of week",
    "Day of year",
    "Latitude",
    "Longitude",
]

# notes for the data processor version
# old
# 2019 - 1.3.1
# 2020 - 1.3.2
# 2022 - 2.3.1

# rpro
# 2019 - 2.4.0
# 2020 - 2.4.0
# 2022 - 2.4.0
