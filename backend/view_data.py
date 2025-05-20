# backend/view_data.py

import pandas as pd

# Read in and parse dates
df = pd.read_csv(
    "data/team_1610612747_202324_gamelog.csv",
    parse_dates=["GAME_DATE"],
    infer_datetime_format=True
)

print("Columns:")
print(list(df.columns))

print("\nFirst 5 rows:")
print(df.head())
