import pandas as pd
import joblib

# 1. Load model & full feature CSV
model = joblib.load("model_nba.pkl")
df    = pd.read_csv("data/team_features_2023.csv", parse_dates=["GAME_DATE"])
df.sort_values("GAME_DATE", inplace=True)

# 2. Take the last row’s engineered features
last = df.iloc[-1]
feat = {
    "pts_5":     last["pts_5"],
    "reb_5":     last["reb_5"],
    "ast_5":     last["ast_5"],
    "win_pct_5": last["win_pct_5"],
    "days_rest": last["days_rest"],
    "back2back": last["back2back"],
}

# 3. Predict
X_next = pd.DataFrame([feat])
prob   = model.predict_proba(X_next)[0,1]
print(f"Next-game win probability: {prob:.1%}")
