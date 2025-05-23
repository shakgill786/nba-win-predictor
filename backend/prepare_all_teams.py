import pandas as pd

# 1. Load your merged all-teams game log
df = pd.read_csv("data/all_teams_202425_gamelog.csv",
                 parse_dates=["GAME_DATE"])

# 2. Compute per-team rolling features (you already have this)
df = df.sort_values(["team","GAME_DATE"])
df["W"] = (df["WL"]=="W").astype(int)
for col,nm in [("PTS","pts_5"),("REB","reb_5"),("AST","ast_5")]:
    df[nm] = df.groupby("team")[col].transform(lambda x: x.rolling(5).mean())
df["win_pct_5"] = df.groupby("team")["W"].transform(lambda x: x.rolling(5).mean())

# 3. Shooting splits & pace (if present in your log; adjust column names)
df["fg_pct_5"]  = df.groupby("team")["FG_PCT"].transform(lambda x: x.rolling(5).mean())
df["fg3_pct_5"] = df.groupby("team")["FG3_PCT"].transform(lambda x: x.rolling(5).mean())
df["ft_pct_5"]  = df.groupby("team")["FT_PCT"].transform(lambda x: x.rolling(5).mean())
df["pace_5"]    = df.groupby("team")["PACE"].transform(lambda x: x.rolling(5).mean())

# 4. Days rest & back2back
df["days_rest"] = df.groupby("team")["GAME_DATE"].transform(lambda x: x.diff().dt.days).fillna(0)
df["back2back"]= (df["days_rest"]==1).astype(int)

# 5. Home/Away & opponent code
df["home"] = df["MATCHUP"].str.contains(" vs. ").astype(int)
df["opp"]  = df["MATCHUP"].str.split().str[-1]

# 6. Opponent rolling win_pct:  
opp = ( df[["team","GAME_DATE","win_pct_5"]]
        .rename(columns={"team":"opp","win_pct_5":"opp_win_pct_5"}) )
df = df.merge(opp, on=["opp","GAME_DATE"], how="left")

# 7. Opponent defensive rating: load your fetched file  
def_rtg = (pd.read_csv("data/team_def_ratings_2025.csv")
             .rename(columns={"team":"opp","opp_def_rtg":"opp_def_rtg"}))
df = df.merge(def_rtg[["opp","opp_def_rtg"]], on="opp", how="left")

# 8. Drop NaNs & save  
keep = ["team","GAME_DATE","pts_5","reb_5","ast_5","win_pct_5",
        "opp_win_pct_5","fg_pct_5","fg3_pct_5","ft_pct_5","pace_5",
        "days_rest","back2back","home","opp","opp_def_rtg","W"]
df = df.dropna(subset=keep)
df[keep].to_csv("data/all_teams_features_richer_2025.csv", index=False)
print("✅ Rich features saved.")
