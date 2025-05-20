# backend/data/prepare_all_teams.py

import pandas as pd

# 1) Load the merged logs
df = pd.read_csv(
    "data/all_teams_202425_gamelog.csv",
    parse_dates=["GAME_DATE"]
)

# 2) Sort by team & date (so rolling windows are chronological)
df = df.sort_values(["team", "GAME_DATE"])

# 3) Binary win label
df["W"] = (df["WL"] == "W").astype(int)

# 4) Rolling stats per team, via transform
df["pts_5"]     = df.groupby("team")["PTS"].transform(lambda x: x.rolling(5).mean())
df["reb_5"]     = df.groupby("team")["REB"].transform(lambda x: x.rolling(5).mean())
df["ast_5"]     = df.groupby("team")["AST"].transform(lambda x: x.rolling(5).mean())
df["win_pct_5"] = df.groupby("team")["W"].transform(lambda x: x.rolling(5).mean())

# 5) Rest days & back‐to‐back, via transform
df["days_rest"] = df.groupby("team")["GAME_DATE"].transform(lambda x: x.diff().dt.days).fillna(0)
df["back2back"] = (df["days_rest"] == 1).astype(int)

# 6) Home/Away flag
df["home"] = df["MATCHUP"].str.contains(" vs. ").astype(int)

# 7) Opponent abbreviation
df["opp"] = df["MATCHUP"].str.split().str[-1]

# 8) Drop rows where any rolling feature is NaN
df = df.dropna(subset=["pts_5","reb_5","ast_5","win_pct_5","days_rest"])

# 9) Select columns to keep
out = df[[
    "team","GAME_DATE","pts_5","reb_5","ast_5","win_pct_5",
    "days_rest","back2back","home","opp","W"
]]

# 10) Save to CSV
out.to_csv("data/all_teams_features_2025.csv", index=False)
print("✅ Saved data/all_teams_features_2025.csv")
