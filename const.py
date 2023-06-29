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

POP_ADMS = [
    "Kyiv",
    "Kharkivskyi",
    "Odeskyi",
    "Dniprovskyi",
    "Donetskyi",
    "Zaporizkyi",
    "Lvivskyi",
    "Kryvorizkyi",
    "Mykolaivskyi",
    # "Sevastopol", # no data
]
WAR_ADMS = ["Dnipropetrovska", "Donetska", "Kharkivska", "Luhanska", "Zaporizka"]
BOR_ADMS = ["Chernivetskyi", "Uzhhorodskyi", "Berehivskyi"]
LIST_PPLS = [
    "Zaporizhia TPP",
    "Mironivskaya TEC",
    "Zmiivska power station",
    "Luhanska",
    "Sievierodonetsk CHP power station",
    "Vuglegirska power station",
    "Dobrotvir power station",
]
LCKDWN_SD = "2020-04-03"
LCKDWN_ED = "2020-06-01"
BF_SD = "2020-03-01"
BF_ED = "2020-04-02"

DATE_2020 = {
    "Before lockdown (03/01 - 04/02)": (BF_SD, BF_ED),
    "Lockdown (04/03 - 06/01)": (LCKDWN_SD, LCKDWN_ED),
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
# POP_ADMS = [
#     "Kyiv",
#     "Kharkivska",
#     "Odeska",  #
#     "Dniprovska",
#     "Donetska",
#     "Zaporizka",
#     "Lvivska",
#     "Kryvorizka",
#     "Mykolaivska",  #
#     "Mariupolska",
#     "Luhanska",  #
#     "Vinnytska",  #
#     "Simferopolska",
#     "Makiivska",
#     "Poltavska",
# ]
