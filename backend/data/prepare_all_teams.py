# backend/data/prepare_all_teams.py

import os
import pandas as pd

# 1️⃣ Paths
HERE    = os.path.dirname(__file__)               # …/backend/data
IN_CSV  = os.path.join(HERE, "all_teams_202425_gamelog.csv")
OUT_CSV = os.path.join(HERE, "all_teams_features_richer_2025.csv")

# 2️⃣ Load & sort
df = pd.read_csv(IN_CSV, parse_dates=["GAME_DATE"])
df.sort_values(["team", "GAME_DATE"], inplace=True)

# 3️⃣ Basic flags & possessions
df["W"]    = (df["WL"] == "W").astype(int)
df["home"] = df["MATCHUP"].str.contains(" vs\. ").astype(int)
df["opp"]  = df["MATCHUP"].str.split().str[-1]
df["poss"] = df["FGA"] + df["TOV"] + 0.4*df["FTA"] - df["OREB"]

# 4️⃣ Rolling per-team (last 5 games)
grp = df.groupby("team", group_keys=False)
for col, name in [
    ("PTS","pts_5"),
    ("REB","reb_5"),
    ("AST","ast_5"),
    ("W","win_pct_5"),
    ("FG_PCT","fg_pct_5"),
    ("FG3_PCT","fg3_pct_5"),
    ("FT_PCT","ft_pct_5"),
    ("poss","pace_5"),
]:
    df[name] = grp[col].rolling(5).mean().reset_index(level=0, drop=True)

# 5️⃣ Rest days & back-to-back
df["days_rest"] = grp["GAME_DATE"].diff().dt.days.fillna(0).astype(int)
df["back2back"] = (df["days_rest"] == 1).astype(int)

# 6️⃣ Drop the first 4 games of each team (NaNs)
df = df.dropna(
    subset=[
        "pts_5","reb_5","ast_5","win_pct_5",
        "fg_pct_5","fg3_pct_5","ft_pct_5","pace_5"
    ]
)

# 7️⃣ Opponent 5-game win% via a simple merge
opp_df = df[["opp","GAME_DATE","win_pct_5"]].copy()
opp_df.rename(columns={"win_pct_5":"opp_win_pct_5"}, inplace=True)
df = df.merge(opp_df, on=["opp","GAME_DATE"], how="left")

# 8️⃣ Select final columns & save
out = df[[
    "team","GAME_DATE",
    "pts_5","reb_5","ast_5","win_pct_5","opp_win_pct_5",
    "fg_pct_5","fg3_pct_5","ft_pct_5","pace_5",
    "days_rest","back2back","home","opp","W"
]]
out.to_csv(OUT_CSV, index=False)
print(f"✅ Wrote richer features to {OUT_CSV}")
