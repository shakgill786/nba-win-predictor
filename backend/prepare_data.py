# prepare_data.py
import pandas as pd

# 1. Load and sort by date
df = pd.read_csv(
    "data/team_1610612747_202324_gamelog.csv",
    parse_dates=["GAME_DATE"],
    infer_datetime_format=True
)
df.sort_values("GAME_DATE", inplace=True)

# 2. Make a binary win column: 1 if win, 0 if loss
df["W"] = (df["WL"] == "W").astype(int)

# 3. Compute rolling averages over the last 5 games
df["pts_5"]     = df["PTS"].rolling(5).mean()
df["reb_5"]     = df["REB"].rolling(5).mean()
df["ast_5"]     = df["AST"].rolling(5).mean()
df["win_pct_5"] = df["W"].rolling(5).mean()

# 4. Calculate rest days and back-to-back flag
df["days_rest"] = df["GAME_DATE"].diff().dt.days.fillna(0)
df["back2back"] = (df["days_rest"] == 1).astype(int)

# 5. Drop the first 4 rows (they have NaNs from rolling)
df = df.dropna(subset=["pts_5","reb_5","ast_5","win_pct_5","days_rest"])

# 6. Save your features to a new CSV
df.to_csv("data/team_features_2023.csv", index=False)
print("✅ Prepared data saved to data/team_features_2023.csv")
