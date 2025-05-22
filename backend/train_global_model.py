# backend/train_global_model.py

import os
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

# ── 1) Paths
BASE        = os.path.dirname(__file__)
CSV_PATH    = os.path.join(BASE, "data", "all_teams_features_richer_2025.csv")
MODEL_RAW   = os.path.join(BASE, "model_global_ensemble.pkl")
MODEL_CALIB = os.path.join(BASE, "model_global_calibrated.pkl")

# ── 2) Load & sort
df = pd.read_csv(CSV_PATH, parse_dates=["GAME_DATE"])
df.sort_values(["team", "GAME_DATE"], inplace=True)

# ── 3) Feature sets
NUMERIC = [
    "pts_5","reb_5","ast_5","win_pct_5","opp_win_pct_5",
    "fg_pct_5","fg3_pct_5","ft_pct_5","pace_5",
    "days_rest","back2back"
]
CATEGORICAL = ["home","opp","team"]
FEATURES    = NUMERIC + CATEGORICAL
TARGET      = "W"

# ── 4) Split X / y
X = df[FEATURES]
y = df[TARGET]

# ── 5) Preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
])

# ── 6) Build soft‐voting ensemble
clf_lr  = LogisticRegression(max_iter=1000)
clf_xgb = XGBClassifier(eval_metric="logloss", use_label_encoder=False)
ensemble = VotingClassifier(
    estimators=[("lr", clf_lr), ("xgb", clf_xgb)],
    voting="soft"
)

pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf",  ensemble),
])

# ── 7) Time‐aware train/test split (80/20)
split_idx   = int(len(df) * 0.8)
X_train     = X.iloc[:split_idx]
X_test      = X.iloc[split_idx:]
y_train     = y.iloc[:split_idx]
y_test      = y.iloc[split_idx:]

# ── 8) Fit & evaluate raw ensemble
pipeline.fit(X_train, y_train)
y_pred_raw  = pipeline.predict(X_test)
y_proba_raw = pipeline.predict_proba(X_test)[:,1]

print("📊 Raw Ensemble Metrics")
print("  Accuracy:", accuracy_score(y_test, y_pred_raw))
print("  ROC AUC :", roc_auc_score(y_test, y_proba_raw))
print("  Brier   :", brier_score_loss(y_test, y_proba_raw))

# ── 9) Save raw ensemble
joblib.dump(pipeline, MODEL_RAW)
print(f"✅ Saved raw ensemble to {MODEL_RAW}")

# ── 🔟 Calibrate probabilities on the test fold
calibrator = CalibratedClassifierCV(
    estimator=pipeline,  # <— use `estimator` here
    method="sigmoid",
    cv="prefit"
)
calibrator.fit(X_test, y_test)

y_proba_cal = calibrator.predict_proba(X_test)[:,1]
print("📊 Calibrated Metrics")
print("  ROC AUC :", roc_auc_score(y_test, y_proba_cal))
print("  Brier   :", brier_score_loss(y_test, y_proba_cal))

# ── ⓫ Save calibrated model
joblib.dump(calibrator, MODEL_CALIB)
print(f"✅ Saved calibrated model to {MODEL_CALIB}")
