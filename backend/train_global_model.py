# backend/train_global_model.py

import os
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# ── 1) Locate your richer features file
BASE      = os.path.dirname(__file__)
RICH_CSV  = os.path.join(BASE, "data", "all_teams_features_richer_2025.csv")

# ── 2) Load & sort
df = pd.read_csv(RICH_CSV, parse_dates=["GAME_DATE"])
df.sort_values(["team", "GAME_DATE"], inplace=True)

# ── 3) Define which columns exist in that richer file
NUMERIC     = [
    "pts_5","reb_5","ast_5","win_pct_5","opp_win_pct_5",
    "fg_pct_5","fg3_pct_5","ft_pct_5","pace_5",
    "days_rest","back2back"
]
CATEGORICAL = ["home","opp","team"]
FEATURES    = NUMERIC + CATEGORICAL
TARGET      = "W"

# ── 4) Split out X and y
X = df[FEATURES]
y = df[TARGET]

# ── 5) Build preprocessing + model pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
])
pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf",  LogisticRegression(max_iter=1000))
])

# ── 6) Train/test split (80/20 time‐aware)
split_idx     = int(len(df) * 0.8)
X_train       = X.iloc[:split_idx]
X_test        = X.iloc[split_idx:]
y_train       = y.iloc[:split_idx]
y_test        = y.iloc[split_idx:]

# ── 7) Fit & evaluate
pipeline.fit(X_train, y_train)
y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:,1]

print("Global Accuracy:", accuracy_score(y_test, y_pred))
print("Global ROC AUC :", roc_auc_score(y_test, y_proba))

# ── 8) Save your trained model
MODEL_OUT = os.path.join(BASE, "model_global_nba.pkl")
joblib.dump(pipeline, MODEL_OUT)
print(f"✅ Saved model to {MODEL_OUT}")
