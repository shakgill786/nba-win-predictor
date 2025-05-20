# backend/data/merge_all_teams.py

import glob
import pandas as pd

# 1) Find all the per-team CSVs
files = glob.glob("data/*_202425_gamelog.csv")

all_dfs = []
for f in files:
    df = pd.read_csv(f, parse_dates=["GAME_DATE"])
    # extract team abbrev from filename (e.g. "LAL_1610…_202425_…")
    team = f.split("/")[-1].split("_")[0]
    df["team"] = team
    all_dfs.append(df)

# 2) Concat and save
big = pd.concat(all_dfs, ignore_index=True)
big.to_csv("data/all_teams_202425_gamelog.csv", index=False)
print("✅ Merged into data/all_teams_202425_gamelog.csv")
