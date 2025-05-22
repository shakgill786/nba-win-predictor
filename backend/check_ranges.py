import pandas as pd

df = pd.read_csv("backend/data/all_teams_features_richer_2025.csv")
stats = df[[
  "pts_5","reb_5","ast_5","win_pct_5",
  "opp_win_pct_5","fg_pct_5","fg3_pct_5","ft_pct_5","pace_5"
]].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).T

print(stats)
