import pandas as pd
import joblib
from sklearn.calibration import CalibratedClassifierCV

# 1. Load your richer features
df = pd.read_csv(
    "backend/data/all_teams_features_richer_2025.csv",
    parse_dates=["GAME_DATE"]
).sort_values("GAME_DATE")

# 2. Split off the last 20% of the data as a calibration set
split = int(len(df) * 0.8)
train, cal = df.iloc[:split], df.iloc[split:]

# 3. Define the features and target
FEATURES = [
  "pts_5","reb_5","ast_5","win_pct_5","opp_win_pct_5",
  "fg_pct_5","fg3_pct_5","ft_pct_5","pace_5",
  "days_rest","back2back","home","team","opp"
]
X_train, y_train = train[FEATURES], train["W"]
X_cal,   y_cal   = cal[FEATURES],   cal["W"]

# 4. Load your tuned XGB model
base = joblib.load("model_xgb_tuned.pkl")

# 5. Calibrate with isotonic regression
calibrated = CalibratedClassifierCV(base, cv="prefit", method="isotonic")
calibrated.fit(X_cal, y_cal)

# 6. Save the calibrated pipeline
joblib.dump(calibrated, "model_xgb_calibrated.pkl")
print("✅ Saved calibrated model to model_xgb_calibrated.pkl")
