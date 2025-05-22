# backend/data/prepare_richer_features.py

import os
import pandas as pd

# 1️⃣ paths to inputs + output
HERE       = os.path.dirname(__file__)
FEAT_IN    = os.path.join(HERE, "all_teams_features_2025.csv")
DEF_IN     = os.path.join(HERE, "team_def_ratings_2025.csv")
FEAT_OUT   = os.path.join(HERE, "all_teams_features_richer_2025.csv")

# 2️⃣ load
df_feat = pd.read_csv(FEAT_IN, parse_dates=["GAME_DATE"])
df_def  = pd.read_csv(DEF_IN)  # has columns ['team','opp_def_rtg']

# 3️⃣ merge on the 3-letter code
df = df_feat.merge(df_def, on="team", how="left")

# 4️⃣ fill any missing defensive-rating values
df["opp_def_rtg"].fillna(df["opp_def_rtg"].mean(), inplace=True)

# 5️⃣ write out the new richer features file
df.to_csv(FEAT_OUT, index=False)
print(f"✅ Wrote richer features to {FEAT_OUT}")
