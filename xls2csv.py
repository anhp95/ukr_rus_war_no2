#%%
import pandas as pd
import os

from mypath import *

df = pd.read_excel(CONFLICT_XLS, engine="openpyxl")
dates = pd.DatetimeIndex(df.EVENT_DATE.values)

df["EVENT_DATETIME"] = dates
df["MONTH"] = dates.month


start_date = pd.Timestamp("2022-1-1")

wartime_df = df[df["EVENT_DATETIME"] > start_date]


event_types = list(df.EVENT_TYPE.unique())

for event in event_types:
    event_df = wartime_df[wartime_df["EVENT_TYPE"] == event]
    event_df.to_csv(os.path.join(CONFLICT_DIR, f'{event.replace("/", "_")}.csv'))

# datetimes = [datetime.fromtimestamp(ts) for ts in df.TIMESTAMP.values]
# %%
