import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, brier_score_loss, classification_report

# Load & sort
df = pd.read_csv("backend/data/all_teams_features_richer_2025.csv",
                 parse_dates=["GAME_DATE"]).sort_values("GAME_DATE")

# 90/10 split
split = int(len(df) * 0.9)
train, test = df.iloc[:split], df.iloc[split:]

# Load model
model = joblib.load("model_xgb_tuned.pkl")

# Prepare
FEATURES = [
  "pts_5","reb_5","ast_5","win_pct_5","opp_win_pct_5",
  "fg_pct_5","fg3_pct_5","ft_pct_5","pace_5",
  "days_rest","back2back","home","team","opp"
]
X_test = test[FEATURES]
y_test = test["W"]

# Predict
probs = model.predict_proba(X_test)[:,1]
preds = (probs >= 0.5).astype(int)

# Report
print("ROC AUC   :", roc_auc_score(y_test, probs))
print("Brier     :", brier_score_loss(y_test, probs))
print(classification_report(y_test, preds))
