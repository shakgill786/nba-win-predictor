# backend/expand_features.py

import os
import pandas as pd

# Base paths
BASE        = os.path.dirname(__file__)
DATA_DIR    = os.path.join(BASE, "data")
MERGED_PATH = os.path.join(DATA_DIR, "all_teams_202425_gamelog.csv")
OUT_PATH    = os.path.join(DATA_DIR, "all_teams_features_richer_2025.csv")

# 1) Load merged logs
df = pd.read_csv(MERGED_PATH, parse_dates=["GAME_DATE"])
df = df.sort_values(["team", "GAME_DATE"])

# 2) Win label
df["W"] = (df["WL"] == "W").astype(int)

# 3) Opponent code
df["opp"] = df["MATCHUP"].str.split().str[-1]

# 4) Rolling stats per team
grp = df.groupby("team", group_keys=False)
df["pts_5"]     = grp["PTS"].transform(lambda x: x.rolling(5).mean())
df["reb_5"]     = grp["REB"].transform(lambda x: x.rolling(5).mean())
df["ast_5"]     = grp["AST"].transform(lambda x: x.rolling(5).mean())
df["win_pct_5"] = grp["W"].transform(lambda x: x.rolling(5).mean())

# 5) Opponent rolling win % (last 5 opp games)
opp_df = df[["team","GAME_DATE","W"]].rename(columns={"team":"opp","W":"opp_W"})
df = df.merge(opp_df, on=["opp","GAME_DATE"], how="left")
df["opp_win_pct_5"] = df.groupby("opp")["opp_W"].transform(lambda x: x.rolling(5).mean())

# 6) Shooting percents
df["fg_pct_5"]  = grp["FG_PCT"].transform(lambda x: x.rolling(5).mean())
df["fg3_pct_5"] = grp["FG3_PCT"].transform(lambda x: x.rolling(5).mean())
df["ft_pct_5"]  = grp["FT_PCT"].transform(lambda x: x.rolling(5).mean())

# 7) Pace proxy: must create 'poss' before grouping again
df["poss"]   = df["FGA"] + 0.4 * df["FTA"] - df["OREB"] + df["TOV"]
grp = df.groupby("team", group_keys=False)
df["pace_5"] = grp["poss"].transform(lambda x: x.rolling(5).mean())

# 8) Rest & back-to-back
df["days_rest"] = grp["GAME_DATE"].transform(lambda x: x.diff().dt.days).fillna(0)
df["back2back"] = (df["days_rest"] == 1).astype(int)

# 9) Home/Away flag
df["home"] = df["MATCHUP"].str.contains(" vs. ").astype(int)

# 10) Select & drop any NaNs
keep = [
    "team","GAME_DATE",
    "pts_5","reb_5","ast_5","win_pct_5",
    "opp_win_pct_5","fg_pct_5","fg3_pct_5","ft_pct_5","pace_5",
    "days_rest","back2back","home","opp","W"
]
out = df[keep].dropna()

# 11) Save richer features
out.to_csv(OUT_PATH, index=False)
print("✅ Saved richer features to", OUT_PATH)
